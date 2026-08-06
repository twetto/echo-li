use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use echo_li_core::config::VIOConfig;
use echo_li_core::initialization::estimate_initial_pose;
use echo_li_core::mathematical::camera::CameraModel;
use echo_li_core::mathematical::imu_velocity::IMUVelocity;
use echo_li_core::mathematical::vio_state::{VIOSensorState, VIOState};
use echo_li_core::mathematical::vision_measurement::VisionMeasurement;
use echo_li_core::{LandmarkDepthPrior, VIOFilter, landmarks_to_global};
use echo_lie::{SE3, SO3};
use nalgebra::{Matrix3, Matrix4, Vector2, Vector3, Vector6};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

use crate::camera::to_camera_arc;

#[pyclass(name = "VIOFilter")]
pub struct PyVIOFilter {
    filter: VIOFilter,
    camera: Arc<dyn CameraModel>,
    camera_extrinsics: Option<SE3>,
    imu_buffer: Vec<IMUVelocity>,
    initialized: bool,
    n_init_samples: usize,
    // Remembered so it survives set_initial_state, which REPLACES self.filter
    // and would otherwise silently discard the setting.
    gram_window: usize,
}

#[pymethods]
impl PyVIOFilter {
    #[new]
    #[pyo3(signature = (config_or_camera, camera=None, n_init_samples=100))]
    fn new(
        config_or_camera: &Bound<'_, PyAny>,
        camera: Option<&Bound<'_, PyAny>>,
        n_init_samples: usize,
    ) -> PyResult<Self> {
        if let Ok(path) = config_or_camera.extract::<String>() {
            let vio_config = VIOConfig::from_yaml(&path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load config: {e}"))
            })?;
            let settings = vio_config.to_filter_settings();

            let cam: Arc<dyn CameraModel> = if let Some(cam_obj) = camera {
                to_camera_arc(cam_obj)?
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "camera argument required when constructing from config path",
                ));
            };

            let xi0 = VIOState::new(VIOSensorState::identity(), vec![]);
            let filter = VIOFilter::new(settings, xi0);
            Ok(Self {
                filter,
                camera: cam,
                camera_extrinsics: None,
                imu_buffer: Vec::new(),
                initialized: false,
                n_init_samples,
                gram_window: 0,
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "First argument must be a config YAML path (str)",
            ))
        }
    }

    fn set_camera_extrinsics(&mut self, t_bs: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = t_bs.shape();
        if shape[0] != 4 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "T_BS must be a 4x4 matrix",
            ));
        }
        let data = t_bs.as_slice()?;
        let m = Matrix4::from_row_slice(data);
        self.camera_extrinsics = Some(SE3::from_matrix(&m));
        Ok(())
    }

    /// Seed the filter from a known initial state (e.g. ground truth at a mid-flight start),
    /// bypassing the stationary-start auto-initialiser. `rotation` is 3x3 body->world;
    /// `velocity` is body-frame linear velocity. Call after `set_camera_extrinsics`.
    #[pyo3(signature = (position, rotation, velocity))]
    fn set_initial_state(
        &mut self,
        position: [f64; 3],
        rotation: PyReadonlyArray2<'_, f64>,
        velocity: [f64; 3],
    ) -> PyResult<()> {
        let shape = rotation.shape();
        if shape[0] != 3 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        let r = SO3::from_matrix(&Matrix3::from_row_slice(rotation.as_slice()?));
        let pose = SE3::new(r, Vector3::new(position[0], position[1], position[2]));
        let cam_offset = self.camera_extrinsics.clone().unwrap_or_else(SE3::identity);
        let sensor = VIOSensorState {
            input_bias: Vector6::zeros(),
            pose,
            velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
            camera_offset: cam_offset,
        };
        let xi0 = VIOState::new(sensor, vec![]);
        self.filter = VIOFilter::new(self.filter.settings.clone(), xi0);
        if self.gram_window > 0 {
            self.filter.enable_gramian(self.gram_window);
        }
        self.imu_buffer.clear();
        self.initialized = true;
        Ok(())
    }

    fn process_imu(&mut self, stamp: f64, gyro: [f64; 3], accel: [f64; 3]) {
        let imu = IMUVelocity::new(
            stamp,
            Vector3::new(gyro[0], gyro[1], gyro[2]),
            Vector3::new(accel[0], accel[1], accel[2]),
        );

        if !self.initialized {
            self.imu_buffer.push(imu);
            if self.imu_buffer.len() >= self.n_init_samples {
                let pose = estimate_initial_pose(&self.imu_buffer, self.n_init_samples);
                let cam_offset = self.camera_extrinsics.clone().unwrap_or_else(SE3::identity);
                let sensor = VIOSensorState {
                    input_bias: nalgebra::Vector6::zeros(),
                    pose,
                    velocity: Vector3::zeros(),
                    camera_offset: cam_offset,
                };
                let xi0 = VIOState::new(sensor, vec![]);
                self.filter = VIOFilter::new(self.filter.settings.clone(), xi0);
        if self.gram_window > 0 {
            self.filter.enable_gramian(self.gram_window);
        }
                for buffered_imu in self.imu_buffer.drain(..) {
                    self.filter.process_imu(buffered_imu);
                }
                self.initialized = true;
            }
            return;
        }

        self.filter.process_imu(imu);
    }

    fn process_vision(&mut self, stamp: f64, feature_uvs: HashMap<u64, [f32; 2]>) {
        if !self.initialized {
            return;
        }

        let cam_coords: HashMap<u64, Vector2<f32>> = feature_uvs
            .into_iter()
            .map(|(id, uv)| (id, Vector2::new(uv[0], uv[1])))
            .collect();
        let measurement = VisionMeasurement::new(stamp, cam_coords);
        self.filter
            .process_vision(measurement, self.camera.as_ref());
    }

    /// Vision update with per-landmark range priors. Use this to seed new
    /// landmarks from a stereo triangulation: for each feature id give
    /// `(range, range_var)` where `range` is the euclidean distance from the
    /// camera origin to the 3D point (cam.undistort(uv) returns a unit-norm
    /// bearing, and the landmark is initialised as `bearing * range`) and
    /// `range_var` is its variance. The filter only consumes priors when
    /// initialising a new landmark; tracked landmarks ignore them.
    fn process_vision_with_depth_priors(
        &mut self,
        stamp: f64,
        feature_uvs: HashMap<u64, [f32; 2]>,
        depth_priors: HashMap<u64, [f64; 2]>,
    ) {
        if !self.initialized {
            return;
        }

        let cam_coords: HashMap<u64, Vector2<f32>> = feature_uvs
            .into_iter()
            .map(|(id, uv)| (id, Vector2::new(uv[0], uv[1])))
            .collect();
        let priors: HashMap<u64, LandmarkDepthPrior> = depth_priors
            .into_iter()
            .map(|(id, rv)| {
                (
                    id,
                    LandmarkDepthPrior {
                        range: rv[0],
                        range_var: rv[1],
                    },
                )
            })
            .collect();
        let measurement = VisionMeasurement::new(stamp, cam_coords);
        self.filter
            .process_vision_with_depth_priors(measurement, self.camera.as_ref(), &priors);
    }

    fn get_pose<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let state = self.filter.state_estimate();
        let pos = state.sensor.pose.translation;
        let q = state.sensor.pose.rotation.as_xyzw();

        let pos_arr =
            PyArray1::from_owned_array(py, Array1::from_vec(vec![pos[0], pos[1], pos[2]]));
        let q_arr = PyArray1::from_owned_array(py, Array1::from_vec(vec![q[0], q[1], q[2], q[3]]));
        Ok((pos_arr, q_arr))
    }

    fn get_velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let state = self.filter.state_estimate();
        let v = state.sensor.velocity;
        PyArray1::from_owned_array(py, Array1::from_vec(vec![v[0], v[1], v[2]]))
    }

    fn get_biases<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let state = self.filter.state_estimate();
        let gb = state.sensor.gyro_bias();
        let ab = state.sensor.accel_bias();
        (
            PyArray1::from_owned_array(py, Array1::from_vec(vec![gb[0], gb[1], gb[2]])),
            PyArray1::from_owned_array(py, Array1::from_vec(vec![ab[0], ab[1], ab[2]])),
        )
    }

    fn get_landmarks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let state = self.filter.state_estimate();
        let (global_lm, _, _) = landmarks_to_global(&state);
        let dict = pyo3::types::PyDict::new(py);
        for (id, pos) in &global_lm {
            let arr =
                PyArray1::from_owned_array(py, Array1::from_vec(vec![pos[0], pos[1], pos[2]]));
            dict.set_item(id, arr)?;
        }
        Ok(dict)
    }

    /// Current camera pose covariance from the EqF (J * Sigma * J^T through the camera-offset
    /// adjoint): returns (P_vv, P_ww) as two 3x3 arrays -- position and attitude covariance of
    /// the camera pose in the camera-fixed (local) frame. Feed these straight into Sparse3DFilter.update
    /// so the landmark depth covariance accounts for pose uncertainty. None if unavailable.
    fn get_camera_pose_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mat3 = |m: &Matrix3<f64>| {
            let data: Vec<f64> = (0..3)
                .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
                .collect();
            PyArray2::from_owned_array(py, Array2::from_shape_vec((3, 3), data).unwrap())
        };
        self.filter
            .sparse_camera_pose_covariances()
            .map(|(p_vv, p_ww)| (mat3(&p_vv), mat3(&p_ww)))
    }

    /// Enable observability-Gramian accumulation over `window` vision frames (0 = off).
    fn enable_gramian(&mut self, window: usize) {
        self.gram_window = window;
        self.filter.enable_gramian(window);
    }

    /// (gramian 21x21, frames, resets) or None until a full window has accumulated.
    /// Row/col 12..15 is body velocity, matching get_velocity_covariance.
    fn get_observability_gramian<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyArray2<f64>>, usize, usize)> {
        self.filter.observability_gramian().map(|(g, f, r)| {
            let mut data: Vec<f64> = Vec::with_capacity(21 * 21);
            for i in 0..21 {
                for j in 0..21 {
                    data.push(g[(i, j)]);
                }
            }
            (
                PyArray2::from_owned_array(py, Array2::from_shape_vec((21, 21), data).unwrap()),
                f,
                r,
            )
        })
    }

    /// Full 3x3 body-velocity covariance block from the EqF Riccati matrix.
    ///
    /// Body-frame velocity is the gauge-free observable (unlike global position/yaw), so this
    /// is the block to score covariance consistency (NEES) against. None if unavailable.
    fn get_velocity_covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.filter.velocity_covariance().map(|m| {
            let data: Vec<f64> = (0..3)
                .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
                .collect();
            PyArray2::from_owned_array(py, Array2::from_shape_vec((3, 3), data).unwrap())
        })
    }

    fn get_covariance_diagonal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let sigma = &self.filter.eqf.sigma;
        let n = sigma.nrows();
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            diag.push(sigma[(i, i)]);
        }
        PyArray1::from_owned_array(py, Array1::from_vec(diag))
    }

    #[getter]
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    #[getter]
    fn vision_count(&self) -> usize {
        self.filter.vision_count
    }

    fn __repr__(&self) -> String {
        format!(
            "VIOFilter(initialized={}, landmarks={}, vision_count={})",
            self.initialized,
            self.filter.eqf.xi0.camera_landmarks.len(),
            self.filter.vision_count
        )
    }
}
