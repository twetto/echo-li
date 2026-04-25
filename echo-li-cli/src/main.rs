use clap::Parser;
use echo_li_core::dataserver::ASLDatasetReader;
use echo_li_core::{VIOFilter, VIOFilterSettings};
use echo_li_core::mathematical::*;
use echo_li_core::initialization::{check_stationary, estimate_initial_pose};
use echo_li_core::mathematical::camera::{CameraModel, PinholeModel, RadTanModel};
use echo_li_core::config::VIOConfig;
use nalgebra::Vector2;
use rudolf_v::frontend::{Frontend, FrontendConfig};
use rudolf_v::image::Image as RudolfImage;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    dataset: String,

    #[arg(short, long)]
    config: Option<String>,

    #[arg(short, long, default_value = "Normal")]
    coord: String,

    #[arg(short = 'l', long, default_value_t = 0.0)]
    cam_lag: f64,

    /// Output directory for trajectory files (default: eqvio_output_<dataset_name>)
    #[arg(short, long)]
    output: Option<String>,

    /// Enable Rerun visualization (requires --features rerun)
    #[arg(long, default_value_t = false)]
    vis: bool,
}

fn write_trajectory(path: &std::path::Path, entries: &[(f64, VIOState)]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for (t, state) in entries {
        let pos = state.sensor.pose.translation;
        let q = state.sensor.pose.rotation.as_xyzw();
        writeln!(f, "{:.9} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            t, pos[0], pos[1], pos[2],
            q[0], q[1], q[2], q[3])?;
    }
    Ok(())
}

fn write_groundtruth(path: &std::path::Path, poses: &[StampedPose]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for sp in poses {
        let pos = sp.pose.translation;
        let q = sp.pose.rotation.as_xyzw();
        writeln!(f, "{:.9} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            sp.stamp, pos[0], pos[1], pos[2],
            q[0], q[1], q[2], q[3])?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading dataset: {}", args.dataset);

    // Load config if provided
    let vio_config = if let Some(conf_path) = &args.config {
        println!("Loading config from: {}", conf_path);
        Some(VIOConfig::from_yaml(conf_path)?)
    } else {
        None
    };

    let cam_lag = if let Some(conf) = &vio_config {
        conf.main.camera_lag
    } else {
        args.cam_lag
    };

    let reader = ASLDatasetReader::new(&args.dataset, cam_lag);
    let mut imu_it = reader.imu_iter().peekable();
    let mut image_it = reader.image_iter().peekable();

    let settings = if let Some(conf) = &vio_config {
        conf.to_filter_settings()
    } else {
        let mut s = VIOFilterSettings::default();
        s.coordinate_choice = args.coord.clone();
        s
    };

    println!("Filter: chart={}, max_landmarks={}", settings.coordinate_choice, settings.max_landmarks);

    let (cam_model, img_w, img_h): (Box<dyn CameraModel>, usize, usize) = if let Some(intr) = &reader.intrinsics {
        println!("Camera intrinsics: {}x{} fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
            intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy);
        let model: Box<dyn CameraModel> = match (intr.distortion_model.as_deref(), &intr.distortion_coefficients) {
            (Some("radial-tangential"), Some(d)) if d.len() >= 4 => {
                println!("Distortion: radial-tangential k1={:.4} k2={:.4} p1={:.6} p2={:.6}",
                    d[0], d[1], d[2], d[3]);
                Box::new(RadTanModel {
                    fx: intr.fx, fy: intr.fy, cx: intr.cx, cy: intr.cy,
                    k1: d[0], k2: d[1], p1: d[2], p2: d[3],
                })
            }
            _ => {
                println!("Distortion: none (pinhole)");
                Box::new(PinholeModel { fx: intr.fx, fy: intr.fy, cx: intr.cx, cy: intr.cy })
            }
        };
        (model, intr.width, intr.height)
    } else {
        println!("No intrinsics found, using EuRoC defaults");
        (Box::new(PinholeModel { fx: 458.65, fy: 457.3, cx: 367.2, cy: 248.3 }) as Box<dyn CameraModel>, 752, 480)
    };

    // Initialize Rudolf-V Frontend
    let mut frontend_config = FrontendConfig::default();
    if let Some(conf) = &vio_config {
        frontend_config.max_features = conf.eqf.max_features;
        frontend_config.pyramid_levels = conf.rudolf_v.max_level;
        if conf.rudolf_v.equalise_image_histogram {
            frontend_config.histeq = rudolf_v::histeq::HistEqMethod::Global;
        }
        frontend_config.cell_size = conf.rudolf_v.feature_dist as usize;
    } else {
        frontend_config.max_features = 40;
        frontend_config.cell_size = 100;
        frontend_config.histeq = rudolf_v::histeq::HistEqMethod::Global;
    }
    let tracker_max_features = frontend_config.max_features;
    let mut frontend = Frontend::new(frontend_config, img_w, img_h);
    println!("Tracker: Rudolf-V, max_features={}", tracker_max_features);

    // Initialize Rerun visualization
    #[cfg(feature = "rerun")]
    let rec: Option<rerun::RecordingStream> = if args.vis {
        match rerun::RecordingStreamBuilder::new("echo-li").spawn() {
            Ok(r) => {
                println!("Rerun viewer connected");

                // Send blueprint: camera 2D + world 3D side by side
                use rerun::blueprint::{Blueprint, Horizontal, Spatial2DView, Spatial3DView, ContainerLike};
                let blueprint = Blueprint::new(
                    Horizontal::new(vec![
                        ContainerLike::from(Spatial2DView::new("Camera")
                            .with_origin("camera")
                            .with_contents(["camera/**"])),
                        ContainerLike::from(Spatial3DView::new("World")
                            .with_origin("world")
                            .with_contents(["world/**"])),
                    ])
                );
                blueprint.send(&r, Default::default()).ok();

                Some(r)
            }
            Err(e) => {
                eprintln!("Warning: could not spawn Rerun viewer: {e}");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "rerun"))]
    if args.vis {
        eprintln!("Warning: --vis requires building with --features rerun");
    }

    let mut filter: Option<VIOFilter> = None;
    let mut initial_imu = Vec::new();
    let mut initialized = false;
    let mut states_out: Vec<(f64, VIOState)> = Vec::new();
    let mut imu_count: usize = 0;
    let mut vision_count: usize = 0;
    let t_start = std::time::Instant::now();

    #[cfg(feature = "rerun")]
    let mut trajectory_vis: Vec<nalgebra::Vector3<f64>> = Vec::new();

    println!("\nRunning filter...");

    loop {
        if let Some(next_img) = image_it.peek() {
            while let Some(imu) = imu_it.peek() {
                if imu.stamp > next_img.stamp { break; }
                let imu = imu_it.next().unwrap();
                imu_count += 1;

                if !initialized {
                    initial_imu.push(imu);
                    if initial_imu.len() >= 100 {
                        if check_stationary(&initial_imu, 100, 0.1, 0.5) {
                            let mut xi0 = VIOState::new(VIOSensorState::identity(), Vec::new());
                            xi0.sensor.pose = estimate_initial_pose(&initial_imu, 100);
                            if let Some(ext) = &reader.camera_extrinsics {
                                xi0.sensor.camera_offset = ext.clone();
                            }
                            println!("Initialized at t={:.3}", imu.stamp);
                            filter = Some(VIOFilter::new(settings.clone(), xi0));
                            // Warm up the covariance by replaying initial IMU
                            let f = filter.as_mut().unwrap();
                            for past_imu in &initial_imu {
                                f.process_imu(*past_imu);
                            }
                            initialized = true;
                        } else {
                            initial_imu.remove(0);
                        }
                    }
                } else if let Some(f) = &mut filter {
                    f.process_imu(imu);
                }
            }
        }

        if let Some(img_data) = image_it.next() {
            if let Ok(dynamic_img) = image::open(&img_data.image_path) {
                let gray_img = dynamic_img.to_luma8();

                // Clone raw pixels for Rerun before consuming into Rudolf-V
                #[cfg(feature = "rerun")]
                let gray_data = if rec.is_some() {
                    gray_img.clone().into_raw()
                } else {
                    vec![]
                };

                let rudolf_img = RudolfImage::from_vec(img_w, img_h, gray_img.into_raw());

                let (features, _stats) = frontend.process(&rudolf_img);

                if let Some(f) = &mut filter {
                    #[cfg(feature = "rerun")]
                    let tracker_points: Vec<(f32, f32)> = features.iter()
                        .map(|feat| (feat.x as f32, feat.y as f32))
                        .collect();

                    let mut feat_uvs = HashMap::new();
                    for feat in features {
                        feat_uvs.insert(feat.id, Vector2::new(feat.x, feat.y));
                    }

                    let measurement = VisionMeasurement::new(img_data.stamp, feat_uvs);
                    f.process_vision(measurement, cam_model.as_ref());
                    vision_count += 1;

                    let state = f.eqf.state_estimate();
                    #[cfg(feature = "rerun")]
                    let (feat_global, p_cam, r_cam) = echo_li_core::landmarks_to_global(&state);
                    #[cfg(not(feature = "rerun"))]
                    let (feat_global, _, _) = echo_li_core::landmarks_to_global(&state);
                    states_out.push((f.eqf.current_time, state));

                    // Rerun per-frame logging
                    #[cfg(feature = "rerun")]
                    if let Some(rec) = &rec {
                        use std::time::Duration;
                        rec.set_time("log_time", Duration::from_secs_f64(img_data.stamp));

                        // Camera image (grayscale)
                        rec.log("camera/image",
                            &rerun::Image::from_l8(gray_data, [img_w as u32, img_h as u32]),
                        ).ok();

                        // Tracked features on image
                        if !tracker_points.is_empty() {
                            rec.log("camera/image/features",
                                &rerun::Points2D::new(tracker_points)
                                    .with_colors([0xFFFF00FFu32])
                                    .with_radii([2.0f32]),
                            ).ok();
                        }

                        // Accumulate trajectory
                        trajectory_vis.push(p_cam);

                        // 3D trajectory line
                        if trajectory_vis.len() >= 2 {
                            let strip: Vec<[f32; 3]> = trajectory_vis.iter()
                                .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                                .collect();
                            rec.log("world/trajectory",
                                &rerun::LineStrips3D::new([strip])
                                    .with_colors([0x00FFFFFFu32]),
                            ).ok();
                        }

                        // 3D landmarks
                        let lm_pts: Vec<(f32, f32, f32)> = feat_global.values()
                            .map(|p| (p[0] as f32, p[1] as f32, p[2] as f32))
                            .collect();
                        if !lm_pts.is_empty() {
                            rec.log("world/landmarks",
                                &rerun::Points3D::new(lm_pts)
                                    .with_colors([0x00FF00FFu32])
                                    .with_radii([0.02f32]),
                            ).ok();
                        }

                        // Camera pose as RGB arrows (X=red, Y=green, Z=blue)
                        let origin = [p_cam[0] as f32, p_cam[1] as f32, p_cam[2] as f32];
                        let r = r_cam.as_matrix();
                        let scale = 0.1f32;
                        rec.log("world/camera_axes",
                            &rerun::Arrows3D::from_vectors([
                                [r[(0,0)] as f32 * scale, r[(1,0)] as f32 * scale, r[(2,0)] as f32 * scale],
                                [r[(0,1)] as f32 * scale, r[(1,1)] as f32 * scale, r[(2,1)] as f32 * scale],
                                [r[(0,2)] as f32 * scale, r[(1,2)] as f32 * scale, r[(2,2)] as f32 * scale],
                            ])
                            .with_origins([origin, origin, origin])
                            .with_colors([0xFF0000FFu32, 0x00FF00FFu32, 0x0000FFFFu32]),
                        ).ok();
                    }

                    if vision_count % 100 == 0 || vision_count <= 5 {
                        let pos = states_out.last().unwrap().1.sensor.pose.translation;
                        let vel = states_out.last().unwrap().1.sensor.velocity;
                        println!("  [{:4}] t={:.3}  pos=({:+.2}, {:+.2}, {:+.2})  vel=({:+.3}, {:+.3}, {:+.3})  lm={}",
                            vision_count, img_data.stamp,
                            pos[0], pos[1], pos[2],
                            vel[0], vel[1], vel[2],
                            feat_global.len());
                    }
                }
            }
        } else {
            break;
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    println!("\nProcessed {} IMU + {} vision in {:.2}s", imu_count, vision_count, elapsed);

    // Write trajectory output
    let dataset_name = PathBuf::from(&args.dataset)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let output_dir = PathBuf::from(
        args.output.unwrap_or_else(|| format!("eqvio_output_{}", dataset_name))
    );

    if !states_out.is_empty() {
        let traj_file = output_dir.join("estimated_trajectory.txt");
        write_trajectory(&traj_file, &states_out)?;
        println!("Trajectory written to {}", traj_file.display());
    }

    let gt_poses = reader.groundtruth();
    if !gt_poses.is_empty() {
        let gt_file = output_dir.join("groundtruth_trajectory.txt");
        write_groundtruth(&gt_file, &gt_poses)?;
        println!("Ground truth written to {}", gt_file.display());

        // Aligned trajectory
        if !states_out.is_empty() {
            let est_poses: Vec<(f64, echo_lie::SE3)> = states_out.iter()
                .map(|(t, s)| (*t, s.sensor.pose.clone()))
                .collect();
            let alignment = echo_li_core::alignment::align_trajectories(&est_poses, &gt_poses);
            let aligned: Vec<(f64, VIOState)> = states_out.iter()
                .map(|(t, s)| {
                    let mut s_aligned = s.clone();
                    let aligned_pose = alignment.compose(&s.sensor.pose);
                    s_aligned.sensor.pose = aligned_pose;
                    (*t, s_aligned)
                })
                .collect();
            let aligned_file = output_dir.join("aligned_trajectory.txt");
            write_trajectory(&aligned_file, &aligned)?;
            println!("Aligned trajectory written to {}", aligned_file.display());
        }
    }

    println!("Done.");
    Ok(())
}
