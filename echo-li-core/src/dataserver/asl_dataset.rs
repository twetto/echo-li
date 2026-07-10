use echo_lie::{SE3, SO3};
use nalgebra::{Matrix4, Vector3};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::mathematical::{IMUVelocity, StampedPose};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub camera_model: Option<String>,
    pub distortion_model: Option<String>,
    pub distortion_coefficients: Option<Vec<f64>>,
}

pub struct ASLDatasetReader {
    path: PathBuf,
    pub intrinsics: Option<CameraIntrinsics>,
    pub camera_extrinsics: Option<SE3>,
    cam_lag: f64,
}

pub struct ImageIterItem {
    pub stamp: f64,
    pub image_path: PathBuf,
}

impl ASLDatasetReader {
    pub fn new(path: &str, cam_lag: f64) -> Self {
        let mut root = PathBuf::from(path);
        if root.join("mav0").exists() {
            root = root.join("mav0");
        }

        let mut reader = Self {
            path: root,
            intrinsics: None,
            camera_extrinsics: None,
            cam_lag,
        };
        reader.load_config();
        reader
    }

    fn load_config(&mut self) {
        let cam_conf = self.path.join("cam0/sensor.yaml");
        if cam_conf.exists() {
            if let Ok(f) = File::open(cam_conf) {
                let val: serde_yaml::Value = serde_yaml::from_reader(f).unwrap();
                if let Some(intr) = val.get("intrinsics") {
                    let camera_model = val
                        .get("camera_model")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let dist_model = val
                        .get("distortion_model")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let dist_coeffs = val
                        .get("distortion_coefficients")
                        .and_then(|v| v.as_sequence())
                        .map(|seq| seq.iter().filter_map(|v| v.as_f64()).collect());
                    self.intrinsics = Some(CameraIntrinsics {
                        width: val["resolution"][0].as_u64().unwrap() as usize,
                        height: val["resolution"][1].as_u64().unwrap() as usize,
                        fx: intr[0].as_f64().unwrap(),
                        fy: intr[1].as_f64().unwrap(),
                        cx: intr[2].as_f64().unwrap(),
                        cy: intr[3].as_f64().unwrap(),
                        camera_model,
                        distortion_model: dist_model,
                        distortion_coefficients: dist_coeffs,
                    });
                }
                if let Some(ext) = val.get("T_BS") {
                    let mut m = Matrix4::identity();
                    for r in 0..4 {
                        for c in 0..4 {
                            m[(r, c)] = ext["data"][r * 4 + c].as_f64().unwrap();
                        }
                    }
                    let r = SO3::from_matrix(&m.fixed_view::<3, 3>(0, 0).into_owned());
                    let t = m.fixed_view::<3, 1>(0, 3).into_owned();
                    self.camera_extrinsics = Some(SE3::new(r, t));
                }
            }
        }

        // TUM-VI ships calibration as a Kalibr camchain (dso/camchain.yaml)
        // instead of a per-camera sensor.yaml.
        if self.intrinsics.is_none() {
            self.load_camchain();
        }
    }

    /// Load calibration from a Kalibr camchain's `cam0` entry (TUM-VI).
    fn load_camchain(&mut self) {
        let camchain = match self.path.parent() {
            Some(root) => root.join("dso/camchain.yaml"),
            None => return,
        };
        if !camchain.exists() {
            return;
        }
        let Ok(f) = File::open(&camchain) else {
            return;
        };
        let Ok(val) = serde_yaml::from_reader::<_, serde_yaml::Value>(f) else {
            return;
        };
        let Some(cam0) = val.get("cam0") else {
            return;
        };
        let (Some(intr), Some(res)) = (cam0.get("intrinsics"), cam0.get("resolution")) else {
            return;
        };
        let camera_model = cam0
            .get("camera_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let distortion_model = cam0
            .get("distortion_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let distortion_coefficients = cam0
            .get("distortion_coeffs")
            .and_then(|v| v.as_sequence())
            .map(|seq| seq.iter().filter_map(|v| v.as_f64()).collect());
        self.intrinsics = Some(CameraIntrinsics {
            width: res[0].as_u64().unwrap() as usize,
            height: res[1].as_u64().unwrap() as usize,
            fx: intr[0].as_f64().unwrap(),
            fy: intr[1].as_f64().unwrap(),
            cx: intr[2].as_f64().unwrap(),
            cy: intr[3].as_f64().unwrap(),
            camera_model,
            distortion_model,
            distortion_coefficients,
        });

        // Camera offset is the camera pose in the IMU frame (matches EuRoC T_BS).
        // The camchain stores `T_cam_imu` (IMU -> camera), so invert it.
        if let Some(t) = cam0.get("T_cam_imu") {
            let mut m = Matrix4::identity();
            for r in 0..4 {
                for c in 0..4 {
                    m[(r, c)] = t[r][c].as_f64().unwrap();
                }
            }
            let rot = SO3::from_matrix(&m.fixed_view::<3, 3>(0, 0).into_owned());
            let tr = m.fixed_view::<3, 1>(0, 3).into_owned();
            self.camera_extrinsics = Some(SE3::new(rot, tr).inverse());
        }
    }

    pub fn imu_iter(&self) -> impl Iterator<Item = IMUVelocity> {
        let imu_csv = self.path.join("imu0/data.csv");
        let file = File::open(imu_csv).expect("Failed to open imu0/data.csv");
        let reader = BufReader::new(file);

        reader.lines().skip(1).filter_map(|line| {
            let l = line.ok()?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() < 7 {
                return None;
            }
            let t = parts[0].parse::<f64>().ok()? * 1e-9;
            let gyr = Vector3::new(
                parts[1].parse().ok()?,
                parts[2].parse().ok()?,
                parts[3].parse().ok()?,
            );
            let acc = Vector3::new(
                parts[4].parse().ok()?,
                parts[5].parse().ok()?,
                parts[6].parse().ok()?,
            );
            Some(IMUVelocity::new(t, gyr, acc))
        })
    }

    pub fn image_iter(&self) -> impl Iterator<Item = ImageIterItem> {
        let cam_csv = self.path.join("cam0/data.csv");
        let file = File::open(cam_csv).expect("Failed to open cam0/data.csv");
        let reader = BufReader::new(file);
        let root = self.path.join("cam0/data");
        let lag = self.cam_lag;

        reader.lines().skip(1).filter_map(move |line| {
            let l = line.ok()?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.is_empty() {
                return None;
            }
            let t = parts[0].parse::<f64>().ok()? * 1e-9;
            let fname = parts[1].trim();
            Some(ImageIterItem {
                stamp: t + lag,
                image_path: root.join(fname),
            })
        })
    }

    pub fn root_path(&self) -> &Path {
        &self.path
    }

    pub fn cam1_image_iter(&self) -> impl Iterator<Item = ImageIterItem> {
        let cam_csv = self.path.join("cam1/data.csv");
        let file = File::open(cam_csv).expect("Failed to open cam1/data.csv");
        let reader = BufReader::new(file);
        let root = self.path.join("cam1/data");
        let lag = self.cam_lag;

        reader.lines().skip(1).filter_map(move |line| {
            let l = line.ok()?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.is_empty() {
                return None;
            }
            let t = parts[0].parse::<f64>().ok()? * 1e-9;
            let fname = parts[1].trim();
            Some(ImageIterItem {
                stamp: t + lag,
                image_path: root.join(fname),
            })
        })
    }

    pub fn has_cam1(&self) -> bool {
        self.path.join("cam1/sensor.yaml").exists()
    }

    pub fn groundtruth(&self) -> Vec<StampedPose> {
        let gt_csv = self.path.join("state_groundtruth_estimate0/data.csv");
        if !gt_csv.exists() {
            return Vec::new();
        }
        let file = File::open(gt_csv).expect("Failed to open GT");
        let reader = BufReader::new(file);

        let mut poses = Vec::new();
        for line in reader.lines().skip(1) {
            let l = line.unwrap();
            let p: Vec<&str> = l.split(',').collect();
            let t = p[0].parse::<f64>().unwrap() * 1e-9;
            let pos = Vector3::new(
                p[1].parse().unwrap(),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
            );
            let q = nalgebra::Quaternion::new(
                p[4].parse().unwrap(), // w
                p[5].parse().unwrap(), // x
                p[6].parse().unwrap(), // y
                p[7].parse().unwrap(), // z
            );
            let rot = SO3::from_matrix(
                &nalgebra::UnitQuaternion::from_quaternion(q)
                    .to_rotation_matrix()
                    .into_inner(),
            );
            let pose = SE3::new(rot, pos);
            poses.push(StampedPose::new(t, pose));
        }
        poses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn loads_tumvi_camchain_when_no_sensor_yaml() {
        let dir = std::env::temp_dir().join(format!("echo_li_camchain_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("mav0")).unwrap();
        fs::create_dir_all(dir.join("dso")).unwrap();
        let camchain = "cam0:\n  \
            T_cam_imu:\n  \
            - [1.0, 0.0, 0.0, 0.1]\n  \
            - [0.0, 1.0, 0.0, 0.2]\n  \
            - [0.0, 0.0, 1.0, 0.3]\n  \
            - [0.0, 0.0, 0.0, 1.0]\n  \
            camera_model: pinhole\n  \
            distortion_coeffs: [0.0034, 0.0007, -0.0020, 0.0002]\n  \
            distortion_model: equidistant\n  \
            intrinsics: [190.97, 190.97, 254.93, 256.89]\n  \
            resolution: [512, 512]\n";
        fs::write(dir.join("dso/camchain.yaml"), camchain).unwrap();

        let reader = ASLDatasetReader::new(dir.to_str().unwrap(), 0.0);
        let intr = reader.intrinsics.expect("camchain intrinsics loaded");
        assert_eq!(intr.width, 512);
        assert_eq!(intr.camera_model.as_deref(), Some("pinhole"));
        assert_eq!(intr.distortion_model.as_deref(), Some("equidistant"));
        assert_eq!(intr.distortion_coefficients.as_ref().unwrap().len(), 4);
        assert!((intr.fx - 190.97).abs() < 1e-6);

        // camera_offset = inverse(T_cam_imu); with identity rotation, t -> -t.
        let ext = reader.camera_extrinsics.expect("camchain extrinsics loaded");
        assert!((ext.translation.x + 0.1).abs() < 1e-9);
        assert!((ext.translation.z + 0.3).abs() < 1e-9);

        let _ = fs::remove_dir_all(&dir);
    }
}
