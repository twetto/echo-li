pub mod mathematical;
pub mod dataserver;
pub mod coordinate_suite;
pub mod depth;
pub mod initialization;
pub mod alignment;
pub mod config;

#[cfg(test)]
pub mod tests;

use std::collections::{HashMap, HashSet};
use nalgebra::{DMatrix, Vector2, Vector3, Matrix3, SMatrix};
use echo_lie::SO3;

use crate::mathematical::vio_state::{VIOState, VIOSensorState, Landmark};
use crate::mathematical::vio_eqf::VIOEqF;
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vision_measurement::VisionMeasurement;
use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::coordinate_suite::invdepth::InvDepthSuite;
use crate::coordinate_suite::normal::NormalSuite;

// ---------------------------------------------------------------------------
// Settings (matches Python VIOFilterSettings)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VIOFilterSettings {
    // velocityNoise
    pub sigma_gyroscope: f64,
    pub sigma_accelerometer: f64,
    pub sigma_gyroscope_bias: f64,
    pub sigma_accelerometer_bias: f64,

    // measurementNoise
    pub sigma_bearing: f64,

    // initialVariance
    pub initial_point_variance: f64,
    pub initial_attitude_variance: f64,
    pub initial_position_variance: f64,
    pub initial_velocity_variance: f64,
    pub initial_bias_omega_variance: f64,
    pub initial_bias_accel_variance: f64,
    pub initial_camera_attitude_variance: f64,
    pub initial_camera_position_variance: f64,

    // initialValue
    pub initial_scene_depth: f64,

    // processVariance
    pub process_attitude: f64,
    pub process_position: f64,
    pub process_velocity: f64,
    pub process_bias_gyr: f64,
    pub process_bias_acc: f64,
    pub process_camera_attitude: f64,
    pub process_camera_position: f64,
    pub process_point: f64,

    // settings
    pub coordinate_choice: String,
    pub use_equivariant_output: bool,
    pub use_discrete_correction: bool,
    pub use_discrete_velocity_lift: bool,

    // Feature management
    pub max_landmarks: usize,
    pub outlier_threshold: f64,
}

impl Default for VIOFilterSettings {
    fn default() -> Self {
        Self {
            sigma_gyroscope: 0.000243153572917808,
            sigma_accelerometer: 0.012438843268295521,
            sigma_gyroscope_bias: 0.00013372703521098622,
            sigma_accelerometer_bias: 0.004462289865453429,
            sigma_bearing: 1.9297839969591413,
            initial_point_variance: 129.90415638150924,
            initial_attitude_variance: 0.13565029126052572,
            initial_position_variance: 0.1,
            initial_velocity_variance: 8.974852995731e-08,
            initial_bias_omega_variance: 97162.79515771076,
            initial_bias_accel_variance: 1.5813333765300104,
            initial_camera_attitude_variance: 0.0010228558965517584,
            initial_camera_position_variance: 0.023501400846134893,
            initial_scene_depth: 5.0,
            process_attitude: 6.025875320811407e-05,
            process_position: 9.981466095928483e-06,
            process_velocity: 0.025317333863551263,
            process_bias_gyr: 0.0,
            process_bias_acc: 0.0,
            process_camera_attitude: 5.075382174045239e-06,
            process_camera_position: 1.2188313140115635e-05,
            process_point: 0.00029845436136043135,
            coordinate_choice: "Euclidean".to_string(),
            use_equivariant_output: true,
            use_discrete_correction: false,
            use_discrete_velocity_lift: true,
            max_landmarks: 40,
            outlier_threshold: 5.0,
        }
    }
}

impl VIOFilterSettings {
    pub fn input_gain_matrix(&self) -> SMatrix<f64, 12, 12> {
        let mut q = SMatrix::<f64, 12, 12>::zeros();
        q.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * self.sigma_gyroscope.powi(2)));
        q.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * self.sigma_accelerometer.powi(2)));
        q.fixed_view_mut::<3, 3>(6, 6).copy_from(&(Matrix3::identity() * self.sigma_gyroscope_bias.powi(2)));
        q.fixed_view_mut::<3, 3>(9, 9).copy_from(&(Matrix3::identity() * self.sigma_accelerometer_bias.powi(2)));
        q
    }

    pub fn output_gain_matrix(&self, n_obs: usize) -> DMatrix<f64> {
        DMatrix::identity(2 * n_obs, 2 * n_obs) * self.sigma_bearing.powi(2)
    }

    pub fn initial_covariance(&self, n_landmarks: usize) -> DMatrix<f64> {
        let s = VIOSensorState::CDIM;
        let dim = s + 3 * n_landmarks;
        let mut sigma = DMatrix::<f64>::zeros(dim, dim);

        sigma.fixed_view_mut::<3, 3>(0, 0).copy_from(&(Matrix3::identity() * self.initial_bias_omega_variance));
        sigma.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * self.initial_bias_accel_variance));
        sigma.fixed_view_mut::<3, 3>(6, 6).copy_from(&(Matrix3::identity() * self.initial_attitude_variance));
        sigma.fixed_view_mut::<3, 3>(9, 9).copy_from(&(Matrix3::identity() * self.initial_position_variance));
        sigma.fixed_view_mut::<3, 3>(12, 12).copy_from(&(Matrix3::identity() * self.initial_velocity_variance));
        sigma.fixed_view_mut::<3, 3>(15, 15).copy_from(&(Matrix3::identity() * self.initial_camera_attitude_variance));
        sigma.fixed_view_mut::<3, 3>(18, 18).copy_from(&(Matrix3::identity() * self.initial_camera_position_variance));

        for i in 0..n_landmarks {
            let start = s + 3 * i;
            sigma.fixed_view_mut::<3, 3>(start, start).copy_from(&(Matrix3::identity() * self.initial_point_variance));
        }
        sigma
    }

    pub fn state_gain_matrix(&self, n_landmarks: usize) -> DMatrix<f64> {
        let s = VIOSensorState::CDIM;
        let dim = s + 3 * n_landmarks;
        let mut q = DMatrix::<f64>::zeros(dim, dim);

        for k in 0..3 { q[(k, k)] = self.process_bias_gyr; }
        for k in 3..6 { q[(k, k)] = self.process_bias_acc; }
        for k in 6..9 { q[(k, k)] = self.process_attitude; }
        for k in 9..12 { q[(k, k)] = self.process_position; }
        for k in 12..15 { q[(k, k)] = self.process_velocity; }
        for k in 15..18 { q[(k, k)] = self.process_camera_attitude; }
        for k in 18..21 { q[(k, k)] = self.process_camera_position; }
        for i in 0..n_landmarks {
            let start = s + 3 * i;
            for k in 0..3 { q[(start + k, start + k)] = self.process_point; }
        }
        q
    }
}

// ---------------------------------------------------------------------------
// VIOFilter (matches Python VIOFilter)
// ---------------------------------------------------------------------------

pub struct VIOFilter {
    pub settings: VIOFilterSettings,
    pub eqf: VIOEqF,
    pub suite: Box<dyn EqFCoordinateSuite>,

    pub pending_imu: Vec<IMUVelocity>,
    input_gain: SMatrix<f64, 12, 12>,
    state_gain: DMatrix<f64>,
    pub vision_count: usize,
}

impl VIOFilter {
    pub fn new(settings: VIOFilterSettings, xi0: VIOState) -> Self {
        let suite: Box<dyn EqFCoordinateSuite> = match settings.coordinate_choice.to_lowercase().as_str() {
            "euclidean" => Box::new(EuclideanSuite),
            "invdepth" => Box::new(InvDepthSuite::new()),
            _ => Box::new(NormalSuite::new()),
        };

        let input_gain = settings.input_gain_matrix();
        let n_lm = xi0.camera_landmarks.len();
        let state_gain = settings.state_gain_matrix(n_lm);
        let init_cov = settings.initial_covariance(n_lm);

        Self {
            settings,
            eqf: VIOEqF::new(xi0, &init_cov),
            suite,
            pending_imu: Vec::new(),
            input_gain,
            state_gain,
            vision_count: 0,
        }
    }

    // ------------------------------------------------------------------
    // IMU processing (matches Python process_imu)
    // ------------------------------------------------------------------

    pub fn process_imu(&mut self, imu: IMUVelocity) {
        if self.eqf.current_time < 0.0 {
            self.eqf.current_time = imu.stamp;
            self.pending_imu.push(imu);
            return;
        }

        let dt = imu.stamp - self.eqf.current_time;
        if dt <= 0.0 { return; }

        // 1. Propagate observer state (updates X)
        self.eqf.integrate_observer_state(&imu, dt, self.settings.use_discrete_velocity_lift);

        // Remove landmarks that became degenerate during propagation
        let n_before = self.eqf.x.id.len();
        self.eqf.remove_invalid_landmarks();
        if self.eqf.x.id.len() != n_before {
            self.invalidate_gain_cache();
        }

        // 2. Propagate Riccati (uses updated X)
        self.eqf.integrate_riccati_fast(
            self.suite.as_ref(), &imu, dt, &self.input_gain, &self.state_gain,
        );

        self.eqf.current_time = imu.stamp;
        self.pending_imu.push(imu);

        if self.pending_imu.len() > 200 {
            let drain_to = self.pending_imu.len() - 100;
            self.pending_imu.drain(..drain_to);
        }
    }

    fn invalidate_gain_cache(&mut self) {
        let n_lm = self.eqf.xi0.camera_landmarks.len();
        self.state_gain = self.settings.state_gain_matrix(n_lm);
    }

    // ------------------------------------------------------------------
    // Vision processing (matches Python process_vision)
    // ------------------------------------------------------------------

    pub fn process_vision(
        &mut self,
        measurement: VisionMeasurement,
        cam: &dyn CameraModel,
    ) {
        if self.eqf.current_time < 0.0 { return; }

        let current_ids: HashSet<u64> = self.eqf.x.id.iter().cloned().collect();
        let observed_ids: HashSet<u64> = measurement.cam_coordinates.keys().cloned().collect();

        // --- Remove lost landmarks ---
        let lost_ids: Vec<u64> = current_ids.difference(&observed_ids).cloned().collect();
        for id in &lost_ids {
            self.eqf.remove_landmark_by_id(*id);
        }

        // Remove invalid landmarks
        self.eqf.remove_invalid_landmarks();

        if !lost_ids.is_empty() {
            self.invalidate_gain_cache();
        }

        // --- Add new landmarks ---
        let current_ids_after: HashSet<u64> = self.eqf.x.id.iter().cloned().collect();
        let new_ids: Vec<u64> = observed_ids.difference(&current_ids_after).cloned().collect();

        let mut new_landmarks = Vec::new();
        for &id in &new_ids {
            if self.eqf.x.id.len() + new_landmarks.len() >= self.settings.max_landmarks { break; }
            let uv = measurement.cam_coordinates.get(&id).unwrap();
            let bearing = cam.undistort(&Vector2::new(uv[0] as f64, uv[1] as f64));
            let p = bearing * self.settings.initial_scene_depth;
            new_landmarks.push(Landmark { p, id });
        }

        if !new_landmarks.is_empty() {
            let n_new = new_landmarks.len();
            let mut new_cov = DMatrix::<f64>::zeros(3 * n_new, 3 * n_new);
            for i in 0..n_new {
                new_cov.fixed_view_mut::<3, 3>(3 * i, 3 * i)
                    .copy_from(&(Matrix3::identity() * self.settings.initial_point_variance));
            }
            self.eqf.add_new_landmarks(new_landmarks, &new_cov);
            self.invalidate_gain_cache();
        }

        // --- Innovation-based outlier rejection ---
        let xi_hat = self.eqf.state_estimate();
        let threshold_px = self.settings.outlier_threshold * self.settings.sigma_bearing;
        let mut outlier_ids = Vec::new();
        {
            let state_ids: HashSet<u64> = self.eqf.x.id.iter().cloned().collect();
            for &id in observed_ids.intersection(&state_ids) {
                if let Some(lm) = xi_hat.camera_landmarks.iter().find(|l| l.id == id) {
                    let y_pred = cam.project(&lm.p);
                    let uv = measurement.cam_coordinates.get(&id).unwrap();
                    let y_obs = Vector2::new(uv[0] as f64, uv[1] as f64);
                    let innov_norm = (y_obs - y_pred).norm();
                    if innov_norm > threshold_px {
                        outlier_ids.push(id);
                    }
                }
            }
        }
        if !outlier_ids.is_empty() {
            for &id in &outlier_ids {
                self.eqf.remove_landmark_by_id(id);
            }
            self.invalidate_gain_cache();
        }

        // --- Kalman update ---
        let y_ids: Vec<u64> = {
            let state_ids: HashSet<u64> = self.eqf.x.id.iter().cloned().collect();
            let mut ids: Vec<u64> = observed_ids.intersection(&state_ids).cloned().collect();
            ids.sort();
            ids
        };

        if y_ids.is_empty() {
            self.vision_count += 1;
            return;
        }

        let n_obs = y_ids.len();
        let mut y_coords = HashMap::new();
        for &id in &y_ids {
            let uv = measurement.cam_coordinates.get(&id).unwrap();
            y_coords.insert(id, Vector2::new(uv[0] as f64, uv[1] as f64));
        }

        let output_gain = self.settings.output_gain_matrix(n_obs);
        self.eqf.perform_vision_update(
            self.suite.as_ref(),
            &y_ids,
            &y_coords,
            cam,
            &output_gain,
            self.settings.use_equivariant_output,
            self.settings.use_discrete_correction,
        );

        self.vision_count += 1;
    }

    pub fn state_estimate(&self) -> VIOState {
        self.eqf.state_estimate()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

pub fn landmarks_to_global(state: &VIOState) -> (HashMap<u64, Vector3<f64>>, Vector3<f64>, SO3) {
    let r_world_imu = state.sensor.pose.rotation;
    let p_imu_world = state.sensor.pose.translation;
    let r_imu_cam = state.sensor.camera_offset.rotation;
    let p_cam_imu = state.sensor.camera_offset.translation;

    let r_world_cam = r_world_imu.compose(&r_imu_cam);
    let p_cam_world = p_imu_world + r_world_imu.act(&p_cam_imu);

    let mut global_landmarks = HashMap::new();
    for lm in &state.camera_landmarks {
        let p_world = p_cam_world + r_world_cam.act(&lm.p);
        global_landmarks.insert(lm.id, p_world);
    }

    (global_landmarks, p_cam_world, r_world_cam)
}
