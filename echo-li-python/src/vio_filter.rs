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

    /// Faithful seed from an external estimator's DYNAMIC-init state (e.g. OpenVINS
    /// DynamicInitializer output), bypassing echo-li's stationary auto-init so the two
    /// filters start from the SAME state on a mid-flight start. Differs from
    /// `set_initial_state` in two ways required for a faithful transfer:
    ///   * `velocity_world` is expressed in the WORLD frame (OV dumps v_IinG); it is
    ///     rotated into echo-li's body frame internally (`v_body = R_WB^T v_world`),
    ///     because `VIOSensorState.velocity` is the SE(3) body velocity.
    ///   * `gyro_bias` / `accel_bias` seed `input_bias = [gyro; accel]` instead of zero.
    /// `rotation` is 3x3 body->world (already in echo-li's world frame — the caller is
    /// responsible for the OV-global -> echo-world gauge alignment). Call after
    /// `set_camera_extrinsics`.
    #[pyo3(signature = (position, rotation, velocity_world, gyro_bias, accel_bias))]
    fn set_initial_state_full(
        &mut self,
        position: [f64; 3],
        rotation: PyReadonlyArray2<'_, f64>,
        velocity_world: [f64; 3],
        gyro_bias: [f64; 3],
        accel_bias: [f64; 3],
    ) -> PyResult<()> {
        let shape = rotation.shape();
        if shape[0] != 3 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        let r = SO3::from_matrix(&Matrix3::from_row_slice(rotation.as_slice()?));
        let pose = SE3::new(r.clone(), Vector3::new(position[0], position[1], position[2]));
        // World-frame velocity -> body frame (VIOSensorState.velocity is SE(3) body velocity).
        let v_world = Vector3::new(velocity_world[0], velocity_world[1], velocity_world[2]);
        let v_body = r.inverse().act(&v_world);
        let mut input_bias = Vector6::zeros();
        input_bias.fixed_rows_mut::<3>(0).copy_from(&Vector3::new(
            gyro_bias[0],
            gyro_bias[1],
            gyro_bias[2],
        ));
        input_bias.fixed_rows_mut::<3>(3).copy_from(&Vector3::new(
            accel_bias[0],
            accel_bias[1],
            accel_bias[2],
        ));
        let cam_offset = self.camera_extrinsics.clone().unwrap_or_else(SE3::identity);
        let sensor = VIOSensorState {
            input_bias,
            pose,
            velocity: v_body,
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

    /// DIAGNOSTIC ONLY (gravity-leak causal test). Overwrite the nav-state MEAN
    /// in place — pin the estimate's attitude (and optionally body velocity) to a
    /// supplied reference — WITHOUT touching the covariance or the clone window.
    ///
    /// The EqF estimate is `state_group_action(x, xi0)`:
    ///   pose      = xi0.pose ∘ x.a
    ///   v_body    = x.a.R⁻¹ · (xi0.v − x.w)
    ///   camoff    = x.a⁻¹ ∘ xi0.camoff ∘ x.b
    /// We invert these to set the desired estimate while preserving the current
    /// POSITION and CAMOFF exactly, mutating only `x.a` and `x.w`. Existing
    /// setters (`set_initial_state*`) REBUILD the filter and wipe Σ + clones, so
    /// they cannot be used for a per-frame mean reset. This is a mean-only nudge:
    /// Σ is left un-transported (small per-frame resets ⇒ negligible mismatch),
    /// which is acceptable for the causal test of whether bounding attitude tilt
    /// bounds the velocity runaway.
    fn overwrite_nav_mean(
        &mut self,
        rotation: PyReadonlyArray2<'_, f64>,
        velocity_world: [f64; 3],
        set_att: bool,
        set_vel: bool,
    ) -> PyResult<()> {
        if !self.initialized {
            return Ok(());
        }
        let shape = rotation.shape();
        if shape[0] != 3 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        // Current estimate (owned) — read before mutating x.
        let est = self.filter.state_estimate();
        let cur_pos = est.sensor.pose.translation;
        let cur_rot = est.sensor.pose.rotation.clone();
        let cur_v_body = est.sensor.velocity;
        let cur_camoff = est.sensor.camera_offset.clone();

        let r_target = if set_att {
            SO3::from_matrix(&Matrix3::from_row_slice(rotation.as_slice()?))
        } else {
            cur_rot.clone()
        };
        let pose_d = SE3::new(r_target.clone(), cur_pos);

        // Target body velocity: GT (world->body via r_target) or keep current.
        let v_body_target = if set_vel {
            let v_world = Vector3::new(velocity_world[0], velocity_world[1], velocity_world[2]);
            r_target.inverse().act(&v_world)
        } else {
            cur_v_body
        };

        // x.a = xi0.pose⁻¹ ∘ pose_d   (sets estimate.pose = pose_d)
        let new_a = self.filter.eqf.xi0.sensor.pose.inverse().compose(&pose_d);
        // x.w = xi0.v − x.a.R · v_body_target   (sets estimate.v_body = v_body_target)
        let new_w =
            self.filter.eqf.xi0.sensor.velocity - new_a.rotation.act(&v_body_target);
        // x.b = xi0.camoff⁻¹ ∘ x.a ∘ camoff_cur   (preserves estimate.camoff exactly)
        let new_b = self
            .filter
            .eqf
            .xi0
            .sensor
            .camera_offset
            .inverse()
            .compose(&new_a)
            .compose(&cur_camoff);

        self.filter.eqf.x.a = new_a;
        self.filter.eqf.x.w = new_w;
        self.filter.eqf.x.b = new_b;
        Ok(())
    }

    /// De-confound diagnostic (c92): body-velocity pseudo-measurement through the
    /// gain machinery (updates mean AND covariance), given GT world velocity. The
    /// harness converts world→body with the current attitude. `sign` (+1/−1) flips
    /// the velocity-tangent selector for empirical convergence validation.
    fn velocity_pseudo_update(
        &mut self,
        velocity_world: [f64; 3],
        sigma_v: f64,
        sign: f64,
    ) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        let est = self.filter.state_estimate();
        let r_est = est.sensor.pose.rotation.clone();
        let v_world = Vector3::new(velocity_world[0], velocity_world[1], velocity_world[2]);
        let v_gt_body = r_est.inverse().act(&v_world);
        let before = (v_gt_body - est.sensor.velocity).norm();
        self.filter.velocity_pseudo_update(v_gt_body, sigma_v, sign);
        // Return post-update residual norm so the harness can validate the sign
        // (residual MUST shrink; `before` printed alongside for the 1-frame check).
        let est2 = self.filter.state_estimate();
        let v_gt_body2 = est2.sensor.pose.rotation.inverse().act(&v_world);
        let after = (v_gt_body2 - est2.sensor.velocity).norm();
        before - after // >0 ⇒ residual shrank ⇒ correct sign
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

    /// Same as `process_vision_with_depth_priors`, but with the deferral set the CLI
    /// already uses: ids in `defer_fallback_ids` that have NO usable prior are skipped
    /// rather than born at the constant `sceneDepth`. Without this the Python path
    /// always falls back, so the guard at lib.rs is unreachable from Python.
    fn process_vision_with_depth_priors_and_deferred(
        &mut self,
        stamp: f64,
        feature_uvs: HashMap<u64, [f32; 2]>,
        depth_priors: HashMap<u64, [f64; 2]>,
        defer_fallback_ids: Vec<u64>,
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
                    LandmarkDepthPrior { range: rv[0], range_var: rv[1] },
                )
            })
            .collect();
        let deferred: std::collections::HashSet<u64> =
            defer_fallback_ids.into_iter().collect();
        let measurement = VisionMeasurement::new(stamp, cam_coords);
        self.filter.process_vision_with_depth_priors_and_deferred_fallbacks(
            measurement,
            self.camera.as_ref(),
            &priors,
            &deferred,
        );
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

    /// Stochastically clone the current camera pose into the EqF covariance window,
    /// tagged `clone_id` (flushes pending Riccati first). No-op if already live.
    fn clone_pose(&mut self, clone_id: u64, time: f64) {
        self.filter.clone_current_pose(clone_id, time);
    }

    /// Drop clone `clone_id` from the covariance window (exact block deletion).
    fn marginalize_clone(&mut self, clone_id: u64) {
        self.filter.marginalize_clone(clone_id);
    }

    /// DIAGNOSTIC: overwrite a live clone's stored world←camera pose value (4x4),
    /// leaving its covariance untouched. Lets the harness inject GT-relative clone
    /// geometry to separate an update-mechanics bug from EqVIO pose inconsistency.
    /// Returns True if the clone was live.
    fn set_clone_pose_value(&mut self, clone_id: u64, pose: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
        let shape = pose.shape();
        if shape[0] != 4 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pose must be a 4x4 matrix",
            ));
        }
        let m = Matrix4::from_row_slice(pose.as_slice()?);
        Ok(self.filter.set_clone_pose_value(clone_id, SE3::from_matrix(&m)))
    }

    /// Current (post-update) stored world←camera pose (4x4) of a live clone, or None.
    /// Companion to set_clone_pose_value: reads back echo-li's own estimate of the
    /// clone pose after any MSC correction, for the shared-track filter comparison.
    fn clone_pose_value<'py>(
        &self,
        py: Python<'py>,
        clone_id: u64,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.filter.clone_pose_value(clone_id).map(|p| {
            let m = p.as_matrix();
            let data: Vec<f64> = (0..4).flat_map(|r| (0..4).map(move |c| m[(r, c)])).collect();
            PyArray2::from_owned_array(py, Array2::from_shape_vec((4, 4), data).unwrap())
        })
    }

    /// Number of live pose clones in the covariance window.
    fn n_clones(&self) -> usize {
        self.filter.n_clones()
    }

    /// Ids of the live pose clones (block order).
    fn clone_ids(&self) -> Vec<u64> {
        self.filter.clone_ids()
    }

    /// Honest, gauge-cancelled relative-pose covariance between clone `clone_id`
    /// (the depth anchor, whose camera->world pose is `t_wc_clone`, a 4x4 array) and
    /// the current camera pose: returns (P_vv_rel, P_ww_rel) as two 3x3 arrays, a
    /// drop-in replacement for get_camera_pose_covariance when feeding Sparse3DFilter's
    /// pose-range term for a landmark anchored at that clone. None if the clone is not
    /// live or the covariance is non-finite.
    fn get_relative_pose_covariance<'py>(
        &self,
        py: Python<'py>,
        clone_id: u64,
        t_wc_clone: PyReadonlyArray2<f64>,
    ) -> Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let arr = t_wc_clone.as_array();
        if arr.shape() != [4, 4] {
            return None;
        }
        let mut m = Matrix4::<f64>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m[(r, c)] = arr[[r, c]];
            }
        }
        let t_clone = SE3::from_matrix(&m);
        let mat3 = |m: &Matrix3<f64>| {
            let data: Vec<f64> = (0..3)
                .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
                .collect();
            PyArray2::from_owned_array(py, Array2::from_shape_vec((3, 3), data).unwrap())
        };
        self.filter
            .sparse_relative_pose_covariances(clone_id, &t_clone)
            .map(|(p_vv, p_ww)| (mat3(&p_vv), mat3(&p_ww)))
    }

    /// DIAGNOSTIC: rotation-channel term-decomposition of the relative-pose cov,
    /// `(term_curr, term_clone, term_cross)` 3×3 each, with
    /// `term_curr + term_clone − term_cross == p_ww`. Localizes the lag^0.47
    /// sub-linear attitude-cov growth to a specific propagation term.
    fn get_relative_pose_cov_terms_rot<'py>(
        &self,
        py: Python<'py>,
        clone_id: u64,
        t_wc_clone: PyReadonlyArray2<f64>,
    ) -> Option<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let arr = t_wc_clone.as_array();
        if arr.shape() != [4, 4] {
            return None;
        }
        let mut m = Matrix4::<f64>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m[(r, c)] = arr[[r, c]];
            }
        }
        let t_clone = SE3::from_matrix(&m);
        let mat3 = |m: &Matrix3<f64>| {
            let data: Vec<f64> = (0..3)
                .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
                .collect();
            PyArray2::from_owned_array(py, Array2::from_shape_vec((3, 3), data).unwrap())
        };
        self.filter
            .sparse_relative_pose_cov_terms_rot(clone_id, &t_clone)
            .map(|(curr, clone, cross)| (mat3(&curr), mat3(&clone), mat3(&cross)))
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

    /// Full EqF covariance matrix (n×n). Sensor tangent order (vio_eqf.rs:861):
    /// [0:6]=input_bias (gyro[0:3], accel[3:6]), [6:9]=att, [9:12]=pos, [12:15]=vel,
    /// [15:21]=camoff; then landmarks (3 each); then clones (6 each, [rot3|trans3]).
    /// Lets Python slice cross-cov blocks e.g. Σ[gyro_bias, clone] = rows[0:3].
    fn get_full_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let sigma = &self.filter.eqf.sigma;
        let n = sigma.nrows();
        let data: Vec<f64> = (0..n).flat_map(|r| (0..n).map(move |c| sigma[(r, c)])).collect();
        PyArray2::from_owned_array(py, Array2::from_shape_vec((n, n), data).unwrap())
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
