pub mod alignment;
pub mod config;
pub mod coordinate_suite;
pub mod core_types;
pub mod dataserver;
pub mod depth;
pub mod initialization;
pub mod mathematical;
pub mod trajectory_metrics;

#[cfg(test)]
pub mod tests;

use echo_lie::{SE3, SO3};
use nalgebra::{DMatrix, Matrix3, SMatrix, Vector2, Vector3};
use std::collections::{HashMap, HashSet};

use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::coordinate_suite::invdepth::InvDepthSuite;
use crate::coordinate_suite::normal::NormalSuite;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_eqf::VIOEqF;
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState};
use crate::mathematical::vision_measurement::VisionMeasurement;

#[derive(Debug, Clone, Copy)]
pub struct LandmarkDepthPrior {
    pub range: f64,
    pub range_var: f64,
}

#[derive(Debug, Clone)]
struct LandmarkInitCandidate {
    id: u64,
    uv: Vector2<f32>,
    has_depth_prior: bool,
    range_var: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuBiasGroup {
    Additive,
    SemiDirect,
}

impl Default for ImuBiasGroup {
    fn default() -> Self {
        Self::Additive
    }
}

impl ImuBiasGroup {
    pub fn from_config(value: &str) -> Self {
        match value.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "semidirect" | "sdb" => Self::SemiDirect,
            _ => Self::Additive,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::SemiDirect => "semi-direct",
        }
    }
}

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
    /// Optional depth-coordinate birth variance, overriding `initial_point_variance` on the
    /// third (range/inverse-range) chart coordinate only.
    ///
    /// Landmark birth uncertainty is intrinsically ANISOTROPIC: the bearing is measured
    /// precisely by the pixel, the depth is essentially unknown. A single isotropic value
    /// cannot express that. This is what a Civera-style "ρ₀ small, σ_ρ covering ρ=0" prior
    /// needs — an uninformative depth with a tight bearing.
    pub initial_point_depth_variance: Option<f64>,
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
    pub imu_bias_group: ImuBiasGroup,
    pub use_equivariant_output: bool,
    pub use_discrete_correction: bool,
    pub use_discrete_velocity_lift: bool,

    // Feature management
    pub max_landmarks: usize,
    pub outlier_threshold: f64,

    // Riccati propagation variant (Phase 6). false = `Fast` per-sample;
    // true = `Faster` (covariance transport batched per IMU sub-frame).
    pub use_faster_riccati: bool,

    // Stereo log-inverse-range measurement channel. When true, observed
    // landmarks that carry a valid per-frame stereo range prior get an extra
    // l = -ln(range) measurement row.
    pub use_stereo_measurement: bool,
    // Chi²(1) gate on the stereo range innovation; 0 disables gating.
    pub range_gate_chi2: f64,

    // Additive MSCKF structureless vision update (OpenVINS mirror). Default OFF
    // => exact current behavior (`msc_update` is never invoked). The pose-clone
    // window and per-track observation buffer are driven by the harness; these
    // knobs parametrize the update itself.
    pub enable_msckf: bool,
    // Clone-window length (frames kept before marginalization).
    pub msckf_window: usize,
    // Minimum live observations for a track to be used (clamped to >=2).
    pub msckf_min_track: usize,
    // Multiplier on the 95% chi² innovation gate.
    pub msckf_chi2_mult: f64,
    // Pixel measurement noise for the MSC update ONLY (S = H P Hᵀ + σ²I). 0.0 =>
    // fall back to `sigma_bearing` (the base EqF feature noise), preserving the
    // pre-knob behavior. Kept separate so the structureless update can be weighted
    // independently of the in-state EqF vision update.
    pub msckf_sigma_pix: f64,
    // DIAGNOSTIC: suppress the MSC sensor(21) and/or in-state-landmark(3·n_lm)
    // mean-correction. Both true ⇒ clones only. Isolates a nav-chart regression
    // from the sceneDepth-prior landmark correction and the clone feedback.
    pub msckf_suppress_sensor: bool,
    pub msckf_suppress_landmarks: bool,

    // Delayed in-state landmark initialization (OpenVINS `StateHelper::initialize`
    // mirror). When ON, `delayed_init` births each ready track as an in-state EqF
    // landmark with a GEOMETRY-DERIVED correlated covariance (multi-view
    // triangulation + `initialize_invertible`), replacing the guessed-diagonal
    // birth in the normal `process_vision` path. Default OFF => exact current
    // behavior (`delayed_init` is never invoked; births stay diagonal).
    pub enable_delayed_init: bool,
    // Minimum live observations for a track to be delay-initialized (clamped >=2).
    pub delayed_init_min_obs: usize,
    // Multiplier on the 95% chi² gate for the delayed-init update rows.
    pub delayed_init_chi2_mult: f64,
    // Pixel measurement noise for the delayed-init augment. 0.0 => fall back to
    // `msckf_sigma_pix` if set, else `sigma_bearing`.
    pub delayed_init_sigma_pix: f64,
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
            initial_point_depth_variance: None,
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
            imu_bias_group: ImuBiasGroup::Additive,
            use_equivariant_output: true,
            use_discrete_correction: false,
            use_discrete_velocity_lift: true,
            max_landmarks: 40,
            outlier_threshold: 5.0,
            use_faster_riccati: false,
            use_stereo_measurement: false,
            range_gate_chi2: 0.0,
            enable_msckf: false,
            msckf_window: 10,
            msckf_min_track: 3,
            msckf_chi2_mult: 1.0,
            msckf_sigma_pix: 0.0,
            msckf_suppress_sensor: false,
            msckf_suppress_landmarks: false,
            enable_delayed_init: false,
            delayed_init_min_obs: 3,
            delayed_init_chi2_mult: 1.0,
            delayed_init_sigma_pix: 0.0,
        }
    }
}

impl VIOFilterSettings {
    pub fn input_gain_matrix(&self) -> SMatrix<f64, 12, 12> {
        let mut q = SMatrix::<f64, 12, 12>::zeros();
        q.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(Matrix3::identity() * self.sigma_gyroscope.powi(2)));
        q.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(Matrix3::identity() * self.sigma_accelerometer.powi(2)));
        q.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(Matrix3::identity() * self.sigma_gyroscope_bias.powi(2)));
        q.fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&(Matrix3::identity() * self.sigma_accelerometer_bias.powi(2)));
        q
    }

    pub fn output_gain_matrix(&self, n_obs: usize) -> DMatrix<f64> {
        DMatrix::identity(2 * n_obs, 2 * n_obs) * self.sigma_bearing.powi(2)
    }

    pub fn initial_covariance(&self, n_landmarks: usize) -> DMatrix<f64> {
        let s = VIOSensorState::CDIM;
        let dim = s + 3 * n_landmarks;
        let mut sigma = DMatrix::<f64>::zeros(dim, dim);

        sigma
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(Matrix3::identity() * self.initial_bias_omega_variance));
        sigma
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(Matrix3::identity() * self.initial_bias_accel_variance));
        sigma
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(Matrix3::identity() * self.initial_attitude_variance));
        sigma
            .fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&(Matrix3::identity() * self.initial_position_variance));
        sigma
            .fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&(Matrix3::identity() * self.initial_velocity_variance));
        sigma
            .fixed_view_mut::<3, 3>(15, 15)
            .copy_from(&(Matrix3::identity() * self.initial_camera_attitude_variance));
        sigma
            .fixed_view_mut::<3, 3>(18, 18)
            .copy_from(&(Matrix3::identity() * self.initial_camera_position_variance));

        for i in 0..n_landmarks {
            let start = s + 3 * i;
            let mut blk = Matrix3::identity() * self.initial_point_variance;
            if let Some(dv) = self.initial_point_depth_variance {
                blk[(2, 2)] = dv;
            }
            sigma.fixed_view_mut::<3, 3>(start, start).copy_from(&blk);
        }
        sigma
    }

    pub fn state_gain_matrix(&self, n_landmarks: usize) -> DMatrix<f64> {
        let s = VIOSensorState::CDIM;
        let dim = s + 3 * n_landmarks;
        let mut q = DMatrix::<f64>::zeros(dim, dim);

        for k in 0..3 {
            q[(k, k)] = self.process_bias_gyr;
        }
        for k in 3..6 {
            q[(k, k)] = self.process_bias_acc;
        }
        // Diagnostic sweep: ECHO_MSC_PROCATT overrides the additive attitude
        // process-noise (per-step Q[att]) to test whether a self-consistent
        // attitude-cov inflation (gain AND cov co-grow) robustly bounds the
        // tilt runaway (=> cov held too tight) or is razor-thin (=> mis-route).
        let proc_att = std::env::var("ECHO_MSC_PROCATT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(self.process_attitude);
        for k in 6..9 {
            q[(k, k)] = proc_att;
        }
        for k in 9..12 {
            q[(k, k)] = self.process_position;
        }
        for k in 12..15 {
            q[(k, k)] = self.process_velocity;
        }
        for k in 15..18 {
            q[(k, k)] = self.process_camera_attitude;
        }
        for k in 18..21 {
            q[(k, k)] = self.process_camera_position;
        }
        for i in 0..n_landmarks {
            let start = s + 3 * i;
            for k in 0..3 {
                q[(start + k, start + k)] = self.process_point;
            }
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
        let suite: Box<dyn EqFCoordinateSuite> =
            match settings.coordinate_choice.to_lowercase().as_str() {
                "euclidean" => Box::new(EuclideanSuite),
                "invdepth" => Box::new(InvDepthSuite::new()),
                _ => Box::new(NormalSuite::new()),
            };

        let input_gain = settings.input_gain_matrix();
        let n_lm = xi0.camera_landmarks.len();
        let state_gain = settings.state_gain_matrix(n_lm);
        let init_cov = settings.initial_covariance(n_lm);
        let imu_bias_group = settings.imu_bias_group;

        Self {
            settings,
            eqf: VIOEqF::new_with_bias_group(xi0, &init_cov, imu_bias_group),
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
        if dt <= 0.0 {
            return;
        }

        // DIAGNOSTIC (ECHO_IMU_LEFT_HOLD=1, default-off): integrate the interval
        // [current_time, imu.stamp] using the PREVIOUS sample's (gyr,acc) held
        // constant (left-endpoint) instead of the current sample (right-endpoint,
        // default). MSCEqF's propagator uses the left-endpoint hold. This is an
        // identity-ladder confound remover for the per-clone attitude increment
        // (see cont.48/49): echo right-hold under-rotates ~1.4% vs MSCEqF left-hold.
        // Env-gated so the shipped behaviour and covariance path are unchanged.
        let prop = if std::env::var("ECHO_IMU_LEFT_HOLD")
            .map(|s| s == "1")
            .unwrap_or(false)
        {
            match self.pending_imu.last() {
                Some(prev) => IMUVelocity {
                    stamp: imu.stamp,
                    gyr: prev.gyr,
                    acc: prev.acc,
                    gyr_bias_vel: prev.gyr_bias_vel,
                    acc_bias_vel: prev.acc_bias_vel,
                },
                None => imu,
            }
        } else {
            imu
        };

        // 1. Propagate observer state (updates X)
        self.eqf
            .integrate_observer_state(&prop, dt, self.settings.use_discrete_velocity_lift);

        // 2. Propagate covariance.
        if self.settings.use_faster_riccati {
            // `Faster` variant — accumulate this sample's transition into Φ;
            // the O(n²) transport runs once, at the next vision frame. A
            // landmark going degenerate mid-sub-frame forces an early flush so
            // the structural change lands on an up-to-date covariance.
            if self.eqf.has_degenerate_landmarks() {
                self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
                self.eqf.remove_invalid_landmarks();
                self.invalidate_gain_cache();
            }
            self.eqf
                .accumulate_transition(self.suite.as_ref(), &prop, dt);
        } else {
            // `Fast` variant — per-sample transport. Remove landmarks that
            // became degenerate during propagation, then propagate Riccati.
            let n_before = self.eqf.x.id.len();
            self.eqf.remove_invalid_landmarks();
            if self.eqf.x.id.len() != n_before {
                self.invalidate_gain_cache();
            }
            self.eqf.integrate_riccati_fast(
                self.suite.as_ref(),
                &prop,
                dt,
                &self.input_gain,
                &self.state_gain,
            );
        }

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

    pub fn process_vision(&mut self, measurement: VisionMeasurement, cam: &dyn CameraModel) {
        self.process_vision_with_depth_priors(measurement, cam, &HashMap::new());
    }

    pub fn process_vision_with_depth_priors(
        &mut self,
        measurement: VisionMeasurement,
        cam: &dyn CameraModel,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    ) {
        self.process_vision_with_depth_priors_and_deferred_fallbacks(
            measurement,
            cam,
            depth_priors,
            &HashSet::new(),
        );
    }

    pub fn process_vision_with_depth_priors_and_deferred_fallbacks(
        &mut self,
        measurement: VisionMeasurement,
        cam: &dyn CameraModel,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
        defer_fallback_ids: &HashSet<u64>,
    ) {
        if self.eqf.current_time < 0.0 {
            return;
        }

        // `Faster` variant: apply the covariance accumulated since the last
        // frame before the measurement update reads or restructures Σ — the
        // image frame is the sub-frame boundary. No-op under `Fast`.
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);

        let current_ids: HashSet<u64> = self.eqf.x.id.iter().cloned().collect();
        let observed_ids: HashSet<u64> = measurement.cam_coordinates.keys().cloned().collect();

        // --- Remove lost landmarks ---
        let mut lost_ids: Vec<u64> = current_ids.difference(&observed_ids).cloned().collect();
        lost_ids.sort_unstable();
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
        let remaining_slots = self
            .settings
            .max_landmarks
            .saturating_sub(self.eqf.x.id.len());
        let new_ids = select_new_landmark_ids(
            &measurement.cam_coordinates,
            &current_ids_after,
            depth_priors,
            defer_fallback_ids,
            self.settings.initial_scene_depth > 0.0,
            remaining_slots,
        );

        let mut new_landmarks = Vec::new();
        let mut new_covs: Vec<Matrix3<f64>> = Vec::new();
        for &id in &new_ids {
            if self.eqf.x.id.len() + new_landmarks.len() >= self.settings.max_landmarks {
                break;
            }
            let uv = measurement.cam_coordinates.get(&id).unwrap();
            let bearing = cam.undistort(&Vector2::new(uv[0] as f64, uv[1] as f64));
            let fallback_range = if bearing[2] > 1e-9 {
                self.settings.initial_scene_depth / bearing[2]
            } else {
                self.settings.initial_scene_depth
            };
            let prior = depth_priors
                .get(&id)
                .filter(|prior| valid_depth_prior(prior));
            let range = prior.map(|prior| prior.range).unwrap_or(fallback_range);
            let p = bearing * range;
            // Enable with RUST_LOG=echo_li_core=debug (needs a logger installed,
            // e.g. env_logger in the CLI). Replaces ECHO_LI_DEBUG_LANDMARK_INIT.
            log::debug!(
                "eqf landmark init id={} source={} uv=({:.2},{:.2}) bearing=({:.6},{:.6},{:.6}) range={:.6} range_var={:.6e} p=({:.6},{:.6},{:.6}) fallback_range={:.6}",
                id,
                if prior.is_some() {
                    "sparse_range"
                } else {
                    "fallback_scene_depth"
                },
                uv[0],
                uv[1],
                bearing[0],
                bearing[1],
                bearing[2],
                range,
                prior.map(|prior| prior.range_var).unwrap_or(f64::INFINITY),
                p[0],
                p[1],
                p[2],
                fallback_range,
            );
            // Birth covariance (chart coords). Non-stereo births keep the
            // isotropic default; a stereo-backed birth sets the depth chart
            // coordinate from the stereo log-range variance
            // Var(ell) = range_var / range^2, mapped through the chart's
            // d(ell)/d(eps2) = output_range_row. Charts whose depth is a single
            // coordinate (Normal, InvDepth) have the range row concentrated on
            // coord 2; Euclidean spreads it and falls back to isotropic.
            let mut cov_i = Matrix3::identity() * self.settings.initial_point_variance;
            if let Some(dv) = self.settings.initial_point_depth_variance {
                cov_i[(2, 2)] = dv;
            }
            if let Some(prior) = prior {
                let c = self.suite.output_range_row(&p);
                let var_ell = prior.range_var / (range * range);
                if c[0].abs() < 1e-9
                    && c[1].abs() < 1e-9
                    && c[2].abs() > 1e-9
                    && var_ell.is_finite()
                    && var_ell > 0.0
                {
                    cov_i[(2, 2)] = var_ell / (c[2] * c[2]);
                }
            }
            new_landmarks.push(Landmark { p, id });
            new_covs.push(cov_i);
        }

        if !new_landmarks.is_empty() {
            let n_new = new_landmarks.len();
            let mut new_cov = DMatrix::<f64>::zeros(3 * n_new, 3 * n_new);
            for i in 0..n_new {
                new_cov
                    .fixed_view_mut::<3, 3>(3 * i, 3 * i)
                    .copy_from(&new_covs[i]);
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
            let mut update_ids: Vec<_> = observed_ids.intersection(&state_ids).copied().collect();
            update_ids.sort_unstable();
            for id in update_ids {
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
        // Stereo log-inverse-range measurements reuse the per-frame stereo range
        // priors already supplied for landmark birth. THE single sign negation
        // ell = -ln(range) lives here (Rudolf-V and the patch mapper use
        // +log range; the EqF chart and this channel use log-inverse-range — see
        // the canonical sign convention here). Built by iterating the
        // sorted `y_ids` (never the prior map) to stay reproducible.
        let stereo_meas: HashMap<u64, (f64, f64)> = if self.settings.use_stereo_measurement {
            y_ids
                .iter()
                .filter_map(|&id| {
                    let prior = depth_priors.get(&id).filter(|p| valid_depth_prior(p))?;
                    if prior.range > 0.0 && prior.range_var.is_finite() && prior.range_var > 0.0 {
                        let ell = -prior.range.ln();
                        let r_ell = prior.range_var / (prior.range * prior.range);
                        Some((id, (ell, r_ell)))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            HashMap::new()
        };
        self.eqf.perform_vision_update_with_stereo(
            self.suite.as_ref(),
            &y_ids,
            &y_coords,
            cam,
            &output_gain,
            self.settings.use_equivariant_output,
            self.settings.use_discrete_correction,
            &stereo_meas,
            self.settings.range_gate_chi2,
        );

        self.vision_count += 1;
    }

    pub fn state_estimate(&self) -> VIOState {
        self.eqf.state_estimate()
    }

    pub fn sparse_camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        let state = self.eqf.state_estimate();
        let cov = sparse_camera_pose_covariance(&state, &self.eqf.sigma)?;
        if cov.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let p_ww = cov.fixed_view::<3, 3>(0, 0).into_owned();
        let p_vv = cov.fixed_view::<3, 3>(3, 3).into_owned();
        Some((p_vv, p_ww))
    }

    /// Honest, gauge-cancelled relative-pose covariance between clone `clone_id`
    /// (the depth anchor, pose value `t_wc_clone`) and the current camera pose,
    /// as `(p_vv_rel, p_ww_rel)` — a drop-in replacement for the absolute pair
    /// from [`sparse_camera_pose_covariances`] when feeding §V-D. `None` if the
    /// clone is not live or the covariance is non-finite.
    pub fn sparse_relative_pose_covariances(
        &self,
        clone_id: u64,
        t_wc_clone: &SE3,
    ) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        let state = self.eqf.state_estimate();
        let clone_start = self.eqf.clone_block_start(clone_id)?;
        let t_wc_curr = state.sensor.pose.compose(&state.sensor.camera_offset);
        let cov = sparse_relative_pose_covariance(
            &state,
            &self.eqf.sigma,
            clone_start,
            t_wc_clone,
            &t_wc_curr,
        )?;
        let p_ww = cov.fixed_view::<3, 3>(0, 0).into_owned();
        let p_vv = cov.fixed_view::<3, 3>(3, 3).into_owned();
        Some((p_vv, p_ww))
    }

    /// DIAGNOSTIC term-decomposition of [`Self::sparse_relative_pose_covariances`].
    /// Returns the ROTATION (3×3) blocks of the three additive constituents
    /// `(term_curr, term_clone, term_cross)` (see [`sparse_relative_pose_cov_terms`]);
    /// `term_curr + term_clone − term_cross == p_ww`. Lets a harness localize the
    /// `lag^0.47` sub-linear attitude-cov growth to a specific term. Read-only.
    pub fn sparse_relative_pose_cov_terms_rot(
        &self,
        clone_id: u64,
        t_wc_clone: &SE3,
    ) -> Option<(Matrix3<f64>, Matrix3<f64>, Matrix3<f64>)> {
        let state = self.eqf.state_estimate();
        let clone_start = self.eqf.clone_block_start(clone_id)?;
        let t_wc_curr = state.sensor.pose.compose(&state.sensor.camera_offset);
        let (curr, clone, cross) = sparse_relative_pose_cov_terms(
            &state,
            &self.eqf.sigma,
            clone_start,
            t_wc_clone,
            &t_wc_curr,
        )?;
        Some((
            curr.fixed_view::<3, 3>(0, 0).into_owned(),
            clone.fixed_view::<3, 3>(0, 0).into_owned(),
            cross.fixed_view::<3, 3>(0, 0).into_owned(),
        ))
    }

    /// Stochastically clone the current camera pose into the EqF covariance window,
    /// tagged `clone_id`. Flushes any pending Riccati integration first so the copied
    /// cross-covariance reflects the current sensor block. The clone is a frozen SE3
    /// pose (identity self-dynamics); its 6x6 self-block and cross-covariance to every
    /// live column come from `J Σ Jᵀ` with `J = sparse_camera_pose_jacobian`. No-op if
    /// `clone_id` is already live. Pair with [`sparse_relative_pose_covariances`] and
    /// [`marginalize_clone`].
    pub fn clone_current_pose(&mut self, clone_id: u64, time: f64) {
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
        let state = self.eqf.state_estimate();
        let mut j = sparse_camera_pose_jacobian(&state);
        // DIAGNOSTIC (ECHO_MSC_CLONE_NAVPOSE=1, default-off): drop the camera-offset
        // (camoff, cols 15:21) block from the clone birth Jacobian so the clone copies
        // ONLY the IMU-pose uncertainty (Ad_{T_ic⁻¹}·Σ_pose), NOT the separately-tracked
        // extrinsic variance. Stage-diff finding: echo's camoff-translation variance
        // spuriously grows to ≈ pos-scale and, folded in here alongside the pose, makes a
        // just-born clone's translation cov 3.5-4.5× the sensor position variance (MSCEqF's
        // single-element E clone is exactly 1.0×). This over-uncertain clone inflates
        // Σ[vel,clone] ~2× → Kalman gain ~2× → velocity/scale over-correction → divergence.
        // Mirrors MSCEqF (clones one camera-pose element; ext applied in the measurement).
        if std::env::var("ECHO_MSC_CLONE_NAVPOSE").map(|s| s == "1").unwrap_or(false) {
            j.fixed_view_mut::<6, 6>(0, 15).fill(0.0);
        }
        // DIAGNOSTIC (ECHO_MSC_JPOSE, default = current behaviour): scan the
        // pose-column (body-pose, cols 6:12) map used to build the clone cross-cov.
        // Hypothesis (cont.34): echo's sensor covariance lives in a global/left
        // perturbation (correction applied LEFT, `x ← delta·x`), but `j` uses
        // `Ad_{T_bc⁻¹}` — the RIGHT/body→right-camera adjoint — for the pose block.
        // For the ~120° MidAir extrinsic that is a large, uncompensated rotation of
        // the clone cross-cov (the leftchart flag toggles only the clone-COLUMN
        // chart via `ad_t`, never this offset adjoint inside `j`). A global-chart
        // covariance should map the pose columns by IDENTITY. Scan {adinv (current),
        // eye, ad} and read which gives the MSCEqF-like nav-att cos_dir (~+0.62).
        match std::env::var("ECHO_MSC_JPOSE").as_deref() {
            Ok("eye") => j
                .fixed_view_mut::<6, 6>(0, 6)
                .copy_from(&SMatrix::<f64, 6, 6>::identity()),
            Ok("ad") => {
                let ad = state.sensor.camera_offset.adjoint();
                j.fixed_view_mut::<6, 6>(0, 6).copy_from(&ad);
            }
            _ => {} // "adinv" / unset: keep Ad_{T_bc⁻¹} (current)
        }
        // Snapshot the world←camera pose value the covariance block linearizes
        // around, so the clone is a first-class pose the MSC update can read and
        // mean-correct (not merely a covariance mirror).
        let t_wc = state.sensor.pose.compose(&state.sensor.camera_offset);
        self.eqf.clone_pose(clone_id, time, &j, t_wc);
    }

    /// Drop clone `clone_id` from the covariance window (exact 6-dim block deletion,
    /// survivor cross-covariance preserved). No-op if the clone is not live.
    pub fn marginalize_clone(&mut self, clone_id: u64) {
        self.eqf.marginalize_clone(clone_id);
    }

    /// Additive MSCKF structureless vision update over ready tracks
    /// (`track_id -> [(clone_id, uv)]`). Flushes any pending Riccati integration
    /// first (so the clone cross-covariance is current), then delegates to
    /// [`VIOEqF::msc_update`], which triangulates each track over its observing
    /// clones, projects the feature out, gates, and corrects nav+clone poses.
    /// No-op returning 0 when MSCKF is disabled, there are no clones, or no track
    /// survives. Returns the number of accepted tracks.
    pub fn msc_update(
        &mut self,
        tracks: &HashMap<u64, Vec<(u64, Vector2<f64>)>>,
        cam: &dyn CameraModel,
    ) -> usize {
        if !self.settings.enable_msckf {
            return 0;
        }
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
        self.eqf.msc_update(
            self.suite.as_ref(),
            cam,
            tracks,
            self.settings.msckf_min_track,
            self.settings.msckf_chi2_mult,
            if self.settings.msckf_sigma_pix > 0.0 {
                self.settings.msckf_sigma_pix
            } else {
                self.settings.sigma_bearing
            },
            self.settings.use_discrete_correction,
            self.settings.msckf_suppress_sensor,
            self.settings.msckf_suppress_landmarks,
        )
    }

    /// De-confound diagnostic: body-velocity pseudo-measurement through the gain
    /// machinery (see [`VIOEqF::velocity_pseudo_update`]). Flushes pending Riccati
    /// so the clone cross-covariance is current, then applies the update.
    pub fn velocity_pseudo_update(&mut self, v_gt_body: Vector3<f64>, sigma_v: f64, sign: f64) {
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
        self.eqf
            .velocity_pseudo_update(self.suite.as_ref(), v_gt_body, sigma_v, sign);
    }

    /// c94: stash current-frame GT BODY velocity so the next vision `msc_update`
    /// emits the within-echo vision-vs-pseudo γ_v comparison. `None` disables.
    pub fn set_dbg_v_gt_body(&mut self, v: Option<Vector3<f64>>) {
        self.eqf.set_dbg_v_gt_body(v);
    }

    /// Like [`Self::msc_update`] but also returns per-track diagnostics (H1/H2
    /// localization). No-op returning `(0, empty)` when MSCKF is disabled.
    pub fn msc_update_debug(
        &mut self,
        tracks: &HashMap<u64, Vec<(u64, Vector2<f64>)>>,
        cam: &dyn CameraModel,
    ) -> (usize, Vec<crate::mathematical::vio_eqf::MscTrackDebug>) {
        if !self.settings.enable_msckf {
            return (0, Vec::new());
        }
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
        self.eqf.msc_update_debug(
            self.suite.as_ref(),
            cam,
            tracks,
            self.settings.msckf_min_track,
            self.settings.msckf_chi2_mult,
            if self.settings.msckf_sigma_pix > 0.0 {
                self.settings.msckf_sigma_pix
            } else {
                self.settings.sigma_bearing
            },
            self.settings.use_discrete_correction,
            self.settings.msckf_suppress_sensor,
            self.settings.msckf_suppress_landmarks,
        )
    }

    /// Delayed in-state landmark initialization over ready tracks
    /// (`track_id -> [(clone_id, uv)]`), the OpenVINS `StateHelper::initialize`
    /// mirror. Flushes any pending Riccati integration first (so the clone
    /// cross-covariance is current), then births each track that is not already an
    /// in-state landmark via [`VIOEqF::add_landmark_delayed`] — multi-view
    /// triangulate, geometry-derived correlated covariance augment, chi² gate, and
    /// a nav+clone update from the residual rows. No-op returning 0 when delayed
    /// init is disabled, there are no clones, or no track qualifies. Returns the
    /// number of landmarks born.
    pub fn delayed_init(
        &mut self,
        tracks: &HashMap<u64, Vec<(u64, Vector2<f64>)>>,
        cam: &dyn CameraModel,
    ) -> usize {
        if !self.settings.enable_delayed_init {
            return 0;
        }
        self.eqf.flush_riccati(&self.input_gain, &self.state_gain);
        let sigma_pix = if self.settings.delayed_init_sigma_pix > 0.0 {
            self.settings.delayed_init_sigma_pix
        } else if self.settings.msckf_sigma_pix > 0.0 {
            self.settings.msckf_sigma_pix
        } else {
            self.settings.sigma_bearing
        };
        let mut born = 0usize;
        for (tid, obs) in tracks.iter() {
            if self
                .eqf
                .add_landmark_delayed(
                    self.suite.as_ref(),
                    cam,
                    *tid,
                    obs,
                    self.settings.delayed_init_min_obs,
                    self.settings.delayed_init_chi2_mult,
                    sigma_pix,
                    self.settings.use_discrete_correction,
                )
                .is_some()
            {
                born += 1;
                // A birth grows the physical landmark block, so the cached
                // `state_gain` (sized to the old n_lm) is stale. Rebuild it now,
                // otherwise the next `flush_riccati` slices `state_gain` out of
                // bounds against the grown `xi0.dim()`.
                self.invalidate_gain_cache();
            }
        }
        born
    }

    /// Number of live pose clones in the covariance window.
    pub fn n_clones(&self) -> usize {
        self.eqf.n_clones()
    }

    /// Ids of the live pose clones (block order).
    pub fn clone_ids(&self) -> Vec<u64> {
        self.eqf.clone_ids()
    }

    /// DIAGNOSTIC: overwrite a live clone's stored pose value (covariance kept).
    pub fn set_clone_pose_value(&mut self, clone_id: u64, pose: SE3) -> bool {
        self.eqf.set_clone_pose_value(clone_id, pose)
    }

    /// Current (post-update) stored world<-camera pose of a live clone, or None.
    pub fn clone_pose_value(&self, clone_id: u64) -> Option<SE3> {
        self.eqf.clone_pose_value(clone_id)
    }

    /// Enable/disable first-estimate Jacobians (FEJ) for the MSC update. Default off.
    pub fn set_msc_fej(&mut self, on: bool) {
        self.eqf.set_msc_fej(on);
    }

    /// Whether FEJ is enabled for the MSC update.
    pub fn msc_fej(&self) -> bool {
        self.eqf.msc_fej()
    }

    /// Full 3x3 body-velocity covariance block of the EqF Riccati matrix.
    ///
    /// Body-frame velocity is the gauge-FREE observable (global position and yaw are
    /// unobservable by construction), so this is the block to score covariance
    /// consistency against. Base-state layout is
    /// `input_bias(6) | pose(6) | velocity(3) | camera_offset(6)` = 21, hence rows 12..15.
    /// Turn on observability-Gramian accumulation over a sliding window of
    /// `window` vision frames (0 disables). The Gramian measures information the
    /// filter has actually earned per direction, which -- unlike the error -- is
    /// computable from the Jacobians alone.
    pub fn enable_gramian(&mut self, window: usize) {
        self.eqf.enable_gramian(window);
    }

    /// `(gramian_21x21, frames_covered, resets_so_far)`, or None until a full
    /// window has accumulated.
    pub fn observability_gramian(&self) -> Option<(DMatrix<f64>, usize, usize)> {
        self.eqf.observability_gramian()
    }

    pub fn velocity_covariance(&self) -> Option<Matrix3<f64>> {
        let sigma = &self.eqf.sigma;
        if sigma.nrows() < 15 || sigma.ncols() < 15 {
            return None;
        }
        let cov = sigma.fixed_view::<3, 3>(12, 12).into_owned();
        if cov.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(cov)
    }
}

fn sparse_camera_pose_jacobian(state: &VIOState) -> SMatrix<f64, 6, 21> {
    let adj = state.sensor.camera_offset.inverse().adjoint();
    let mut j = SMatrix::<f64, 6, 21>::zeros();
    j.fixed_view_mut::<6, 6>(0, 6).copy_from(&adj);
    j.fixed_view_mut::<6, 6>(0, 15)
        .copy_from(&SMatrix::<f64, 6, 6>::identity());
    j
}

fn sparse_camera_pose_covariance(
    state: &VIOState,
    sigma: &DMatrix<f64>,
) -> Option<SMatrix<f64, 6, 6>> {
    if sigma.nrows() < 21 || sigma.ncols() < 21 {
        return None;
    }

    let j = sparse_camera_pose_jacobian(state);
    let mut cov = SMatrix::<f64, 6, 6>::zeros();
    for r in 0..6 {
        for c in 0..6 {
            let mut v = 0.0;
            for a in 0..21 {
                for b in 0..21 {
                    v += j[(r, a)] * sigma[(a, b)] * j[(c, b)];
                }
            }
            cov[(r, c)] = v;
        }
    }

    cov = 0.5 * (cov + cov.transpose());
    if cov.iter().all(|v| v.is_finite()) {
        Some(cov)
    } else {
        None
    }
}

/// Relative-pose covariance `Cov(T_clone⁻¹ T_curr)` in the SE3 right-perturbation
/// tangent `[ω; v]`, from the joint covariance of the clone's (stored) camera-pose
/// error `δ_c` and the current camera-pose error `δ_k`.
///
/// With both camera poses right-perturbed (`T = T̄ exp(δ^)`, camera frame — the
/// convention of `sparse_camera_pose_jacobian`),
///   `δ_rel = −A δ_c + δ_k`,  `A = Ad_{T̄_rel⁻¹}`,  `T̄_rel = T_clone⁻¹ T_curr`,
/// so `Cov_rel = A Σ_cc Aᵀ + Σ_kk − A Σ_ckᵀ_kc − Σ_kc Aᵀ` where
///   `Σ_cc` = the clone block (already in camera-pose-error coords),
///   `Σ_kk = J Σ_ss Jᵀ` (current camera-pose cov), and
///   `Σ_kc = J Σ[0:21, clone]` (current↔clone cross).
/// The common-mode (unobservable global gauge) part cancels — this is the honest,
/// gauge-cancelled relative uncertainty §V-D wants in place of the absolute pair.
fn sparse_relative_pose_covariance(
    state: &VIOState,
    sigma: &DMatrix<f64>,
    clone_start: usize,
    t_wc_clone: &SE3,
    t_wc_curr: &SE3,
) -> Option<SMatrix<f64, 6, 6>> {
    let n = sigma.nrows();
    if clone_start < 21 || sigma.ncols() != n || n < clone_start + 6 {
        return None;
    }

    let j = sparse_camera_pose_jacobian(state); // 6×21 (current)
    let sig_cc = sigma
        .fixed_view::<6, 6>(clone_start, clone_start)
        .into_owned();
    let sig_ss = sigma.fixed_view::<21, 21>(0, 0).into_owned();
    let sig_kk = j * sig_ss * j.transpose();
    let sig_s_c = sigma.view((0, clone_start), (21, 6)).into_owned(); // 21×6
    let sig_kc = j * sig_s_c; // 6×6  = Cov(δ_k, δ_c)

    let t_rel = t_wc_clone.inverse().compose(t_wc_curr);
    let a = t_rel.inverse().adjoint(); // Ad_{T_rel⁻¹}, 6×6

    let mut cov =
        a * sig_cc * a.transpose() + sig_kk - a * sig_kc.transpose() - sig_kc * a.transpose();
    cov = 0.5 * (cov + cov.transpose());
    if cov.iter().all(|v| v.is_finite()) {
        Some(cov)
    } else {
        None
    }
}

/// Per-lag DIAGNOSTIC decomposition of [`sparse_relative_pose_covariance`] into its
/// three additive constituents, so a harness can see WHICH term drives (or fails to
/// grow) the relative-pose uncertainty as clone lag increases:
///   term_curr  = `sig_kk`               (current camera-pose cov `j·Σ_ss·jᵀ`)
///   term_clone = `a·sig_cc·aᵀ`          (transported frozen clone self-cov)
///   term_cross = `a·sig_kcᵀ + sig_kc·aᵀ` (the SUBTRACTED cross term; the relative
///                cov is `term_curr + term_clone − term_cross`)
/// Each returned as a 6×6 (rot 0:3, trans 3:6) so the caller slices the channel it
/// needs. `term_curr + term_clone − term_cross == sparse_relative_pose_covariance`
/// (a built-in consistency check for the harness). Read-only; not on the ship path.
fn sparse_relative_pose_cov_terms(
    state: &VIOState,
    sigma: &DMatrix<f64>,
    clone_start: usize,
    t_wc_clone: &SE3,
    t_wc_curr: &SE3,
) -> Option<(SMatrix<f64, 6, 6>, SMatrix<f64, 6, 6>, SMatrix<f64, 6, 6>)> {
    let n = sigma.nrows();
    if clone_start < 21 || sigma.ncols() != n || n < clone_start + 6 {
        return None;
    }
    let j = sparse_camera_pose_jacobian(state);
    let sig_cc = sigma
        .fixed_view::<6, 6>(clone_start, clone_start)
        .into_owned();
    let sig_ss = sigma.fixed_view::<21, 21>(0, 0).into_owned();
    let sig_kk = j * sig_ss * j.transpose();
    let sig_s_c = sigma.view((0, clone_start), (21, 6)).into_owned();
    let sig_kc = j * sig_s_c;

    let t_rel = t_wc_clone.inverse().compose(t_wc_curr);
    let a = t_rel.inverse().adjoint();

    let term_curr = sig_kk;
    let term_clone = a * sig_cc * a.transpose();
    let term_cross = a * sig_kc.transpose() + sig_kc * a.transpose();
    if term_curr.iter().chain(term_clone.iter()).chain(term_cross.iter()).all(|v| v.is_finite()) {
        Some((term_curr, term_clone, term_cross))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn valid_depth_prior(prior: &LandmarkDepthPrior) -> bool {
    prior.range.is_finite()
        && prior.range > 0.0
        && prior.range_var.is_finite()
        && prior.range_var > 0.0
}

fn min_anchor_distance_sq(uv: &Vector2<f32>, anchors: &[Vector2<f32>]) -> f32 {
    anchors
        .iter()
        .map(|anchor| (uv - anchor).norm_squared())
        .fold(f32::INFINITY, f32::min)
}

fn append_spatially_balanced_landmarks(
    candidates: &mut Vec<LandmarkInitCandidate>,
    anchors: &mut Vec<Vector2<f32>>,
    selected: &mut Vec<u64>,
    capacity: usize,
) {
    while selected.len() < capacity && !candidates.is_empty() {
        let best_idx = candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let da = min_anchor_distance_sq(&a.uv, anchors);
                let db = min_anchor_distance_sq(&b.uv, anchors);
                da.total_cmp(&db)
                    .then_with(|| b.range_var.total_cmp(&a.range_var))
                    .then_with(|| b.id.cmp(&a.id))
            })
            .map(|(idx, _)| idx)
            .expect("candidates is non-empty");

        let candidate = candidates.swap_remove(best_idx);
        anchors.push(candidate.uv);
        selected.push(candidate.id);
    }
}

fn select_new_landmark_ids(
    cam_coordinates: &HashMap<u64, Vector2<f32>>,
    current_ids: &HashSet<u64>,
    depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    defer_fallback_ids: &HashSet<u64>,
    allow_fallback_init: bool,
    capacity: usize,
) -> Vec<u64> {
    if capacity == 0 || cam_coordinates.is_empty() {
        return Vec::new();
    }

    let mut anchors = Vec::new();
    let mut min_uv = Vector2::new(f32::INFINITY, f32::INFINITY);
    let mut max_uv = Vector2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for uv in cam_coordinates.values() {
        min_uv[0] = min_uv[0].min(uv[0]);
        min_uv[1] = min_uv[1].min(uv[1]);
        max_uv[0] = max_uv[0].max(uv[0]);
        max_uv[1] = max_uv[1].max(uv[1]);
    }

    let mut current_ids_sorted: Vec<_> = current_ids.iter().copied().collect();
    current_ids_sorted.sort_unstable();
    for id in current_ids_sorted {
        if let Some(uv) = cam_coordinates.get(&id) {
            anchors.push(*uv);
        }
    }
    if anchors.is_empty() {
        anchors.push((min_uv + max_uv) * 0.5);
    }

    let mut sparse_candidates = Vec::new();
    let mut fallback_candidates = Vec::new();
    let mut candidate_ids: Vec<_> = cam_coordinates.keys().copied().collect();
    candidate_ids.sort_unstable();
    for id in candidate_ids {
        if current_ids.contains(&id) {
            continue;
        }
        let uv = cam_coordinates[&id];

        let prior = depth_priors
            .get(&id)
            .filter(|prior| valid_depth_prior(prior));
        let candidate = LandmarkInitCandidate {
            id,
            uv,
            has_depth_prior: prior.is_some(),
            range_var: prior.map(|prior| prior.range_var).unwrap_or(f64::INFINITY),
        };
        if candidate.has_depth_prior {
            sparse_candidates.push(candidate);
        } else if defer_fallback_ids.contains(&id) {
            continue;
        } else if !allow_fallback_init {
            continue;
        } else {
            fallback_candidates.push(candidate);
        }
    }

    let mut selected = Vec::with_capacity(capacity.min(cam_coordinates.len()));
    append_spatially_balanced_landmarks(
        &mut sparse_candidates,
        &mut anchors,
        &mut selected,
        capacity,
    );
    append_spatially_balanced_landmarks(
        &mut fallback_candidates,
        &mut anchors,
        &mut selected,
        capacity,
    );

    selected
}

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

#[cfg(test)]
mod landmark_init_selection_tests {
    use super::*;

    fn uv(u: f32, v: f32) -> Vector2<f32> {
        Vector2::new(u, v)
    }

    #[test]
    fn landmark_selection_is_spatially_balanced() {
        let mut coords = HashMap::new();
        coords.insert(10, uv(50.0, 50.0));
        coords.insert(1, uv(0.0, 50.0));
        coords.insert(2, uv(50.0, 50.0));
        coords.insert(3, uv(100.0, 50.0));

        let current_ids = HashSet::from([10]);
        let selected = select_new_landmark_ids(
            &coords,
            &current_ids,
            &HashMap::new(),
            &HashSet::new(),
            true,
            2,
        );

        assert_eq!(selected, vec![1, 3]);
    }

    #[test]
    fn landmark_selection_prefers_sparse_priors_before_fallback() {
        let mut coords = HashMap::new();
        coords.insert(1, uv(48.0, 50.0));
        coords.insert(2, uv(100.0, 50.0));

        let mut priors = HashMap::new();
        priors.insert(
            1,
            LandmarkDepthPrior {
                range: 5.0,
                range_var: 1.0,
            },
        );

        let selected =
            select_new_landmark_ids(&coords, &HashSet::new(), &priors, &HashSet::new(), true, 1);

        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn landmark_selection_is_independent_of_hashmap_iteration_order() {
        let ids_and_uvs = [
            (4, uv(10.0, 10.0)),
            (2, uv(90.0, 10.0)),
            (8, uv(10.0, 90.0)),
            (6, uv(90.0, 90.0)),
            (1, uv(50.0, 50.0)),
        ];

        let mut coords_a = HashMap::new();
        for (id, uv) in ids_and_uvs {
            coords_a.insert(id, uv);
        }

        let mut coords_b = HashMap::new();
        for (id, uv) in ids_and_uvs.into_iter().rev() {
            coords_b.insert(id, uv);
        }

        let selected_a = select_new_landmark_ids(
            &coords_a,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            true,
            3,
        );
        let selected_b = select_new_landmark_ids(
            &coords_b,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            true,
            3,
        );

        assert_eq!(selected_a, selected_b);
    }

    #[test]
    fn landmark_selection_defers_sparse_tracks_without_valid_prior() {
        let mut coords = HashMap::new();
        coords.insert(1, uv(20.0, 50.0));
        coords.insert(2, uv(80.0, 50.0));

        let deferred = HashSet::from([1]);
        let selected = select_new_landmark_ids(
            &coords,
            &HashSet::new(),
            &HashMap::new(),
            &deferred,
            true,
            2,
        );

        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn landmark_selection_allows_deferred_track_with_valid_prior() {
        let mut coords = HashMap::new();
        coords.insert(1, uv(20.0, 50.0));
        coords.insert(2, uv(80.0, 50.0));

        let mut priors = HashMap::new();
        priors.insert(
            1,
            LandmarkDepthPrior {
                range: 5.0,
                range_var: 1.0,
            },
        );
        let deferred = HashSet::from([1]);
        let selected =
            select_new_landmark_ids(&coords, &HashSet::new(), &priors, &deferred, true, 1);

        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn landmark_selection_can_disable_fallback_scene_depth_init() {
        let mut coords = HashMap::new();
        coords.insert(1, uv(20.0, 50.0));
        coords.insert(2, uv(80.0, 50.0));

        let selected = select_new_landmark_ids(
            &coords,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            false,
            2,
        );

        assert!(selected.is_empty());
    }

    #[test]
    fn landmark_selection_still_uses_sparse_prior_when_fallback_disabled() {
        let mut coords = HashMap::new();
        coords.insert(1, uv(20.0, 50.0));

        let mut priors = HashMap::new();
        priors.insert(
            1,
            LandmarkDepthPrior {
                range: 5.0,
                range_var: 1.0,
            },
        );

        let selected =
            select_new_landmark_ids(&coords, &HashSet::new(), &priors, &HashSet::new(), false, 1);

        assert_eq!(selected, vec![1]);
    }
}

#[cfg(test)]
mod sparse_camera_pose_covariance_tests {
    use super::*;
    use echo_lie::{SE3, SO3};
    use nalgebra::{DMatrix, SVector, Vector6};

    fn make_state() -> VIOState {
        VIOState::new(
            VIOSensorState {
                input_bias: Vector6::zeros(),
                pose: SE3::new(
                    SO3::exp(&Vector3::new(0.13, -0.07, 0.19)),
                    Vector3::new(1.2, -0.4, 0.8),
                ),
                velocity: Vector3::new(0.3, -0.2, 0.1),
                camera_offset: SE3::new(
                    SO3::exp(&Vector3::new(-0.04, 0.08, 0.03)),
                    Vector3::new(0.12, -0.03, 0.04),
                ),
            },
            Vec::new(),
        )
    }

    fn camera_pose(state: &VIOState) -> SE3 {
        state.sensor.pose.compose(&state.sensor.camera_offset)
    }

    #[test]
    fn sparse_camera_pose_jacobian_matches_finite_difference() {
        let state = make_state();
        let base = camera_pose(&state);
        let analytic = sparse_camera_pose_jacobian(&state);
        let h = 1e-7;
        let mut numeric = SMatrix::<f64, 6, 21>::zeros();

        for col in 0..21 {
            let mut perturbed = state.clone();
            let mut dx = SVector::<f64, 6>::zeros();
            if (6..12).contains(&col) {
                dx[col - 6] = h;
                perturbed.sensor.pose = perturbed.sensor.pose.compose(&SE3::exp(&dx));
            } else if (15..21).contains(&col) {
                dx[col - 15] = h;
                perturbed.sensor.camera_offset =
                    perturbed.sensor.camera_offset.compose(&SE3::exp(&dx));
            } else {
                continue;
            }

            let right_delta = base.inverse().compose(&camera_pose(&perturbed)).log() / h;
            numeric.set_column(col, &right_delta);
        }

        let diff = analytic - numeric;
        assert!(
            diff.norm() < 1e-7,
            "camera pose covariance Jacobian mismatch: norm={:.3e}\nanalytic={:?}\nnumeric={:?}",
            diff.norm(),
            analytic,
            numeric
        );
    }

    #[test]
    fn sparse_camera_pose_covariance_is_j_sigma_jt() {
        let state = make_state();
        let mut sigma = DMatrix::<f64>::zeros(21, 21);
        for r in 0..21 {
            for c in 0..21 {
                sigma[(r, c)] = ((r + 1) as f64 * 0.07).sin() * ((c + 2) as f64 * 0.11).cos();
            }
        }
        sigma = &sigma * sigma.transpose() + DMatrix::<f64>::identity(21, 21) * 1e-6;

        let got = sparse_camera_pose_covariance(&state, &sigma).unwrap();
        let j = sparse_camera_pose_jacobian(&state);
        let mut expected = SMatrix::<f64, 6, 6>::zeros();
        for r in 0..6 {
            for c in 0..6 {
                for a in 0..21 {
                    for b in 0..21 {
                        expected[(r, c)] += j[(r, a)] * sigma[(a, b)] * j[(c, b)];
                    }
                }
            }
        }
        expected = 0.5 * (expected + expected.transpose());

        assert!((got - expected).norm() < 1e-10);
    }

    #[test]
    fn zero_motion_relative_cov_is_zero() {
        // Right after cloning the current pose (no propagation, no update), the
        // clone and the current camera pose ARE the same pose with fully
        // correlated error, so Cov(T_clone⁻¹ T_curr) must be exactly zero. This
        // exercises clone_pose's copied cross-cov, the relative formula, and the
        // perturbation convention (A = Ad_I = I here) end-to-end.
        let state = make_state();
        let n = state.dim();
        let mut sigma = DMatrix::<f64>::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                sigma[(r, c)] = ((r + 1) as f64 * 0.07).sin() * ((c + 2) as f64 * 0.11).cos();
            }
        }
        sigma = &sigma * sigma.transpose() + DMatrix::<f64>::identity(n, n) * 1e-6;

        let mut eqf = VIOEqF::new(state.clone(), &sigma);
        let j = sparse_camera_pose_jacobian(&state);
        let t_wc = camera_pose(&state); // T_clone == T_curr
        eqf.clone_pose(42, 0.0, &j, t_wc.clone());

        let clone_start = eqf.clone_block_start(42).unwrap();
        let cov =
            sparse_relative_pose_covariance(&state, &eqf.sigma, clone_start, &t_wc, &t_wc).unwrap();
        assert!(
            cov.norm() < 1e-9,
            "zero-motion relative cov not ~0: norm={:.3e}\n{cov:?}",
            cov.norm()
        );
    }

    #[test]
    fn relative_cov_jacobian_matches_finite_difference() {
        // Validate the A = Ad_{T_rel⁻¹} linearization + convention for a NONZERO
        // relative motion: the map g(δ_c, δ_k) = log(T̄_rel⁻¹ (T̄_clone e^δ_c)⁻¹
        // (T̄_curr e^δ_k)) must have Jacobian [−A | I] (6×12).
        let t_clone = SE3::new(
            SO3::exp(&Vector3::new(0.05, 0.11, -0.03)),
            Vector3::new(0.4, -0.2, 1.1),
        );
        let t_curr = SE3::new(
            SO3::exp(&Vector3::new(-0.09, 0.02, 0.14)),
            Vector3::new(0.7, 0.3, 0.9),
        );
        let t_rel = t_clone.inverse().compose(&t_curr);
        let a = t_rel.inverse().adjoint();

        let mut j_analytic = SMatrix::<f64, 6, 12>::zeros();
        j_analytic.fixed_view_mut::<6, 6>(0, 0).copy_from(&(-a));
        j_analytic
            .fixed_view_mut::<6, 6>(0, 6)
            .copy_from(&SMatrix::<f64, 6, 6>::identity());

        let h = 1e-7;
        let mut j_numeric = SMatrix::<f64, 6, 12>::zeros();
        for col in 0..12 {
            let mut d = SVector::<f64, 6>::zeros();
            d[col % 6] = h;
            let (tc, tk) = if col < 6 {
                (t_clone.compose(&SE3::exp(&d)), t_curr.clone())
            } else {
                (t_clone.clone(), t_curr.compose(&SE3::exp(&d)))
            };
            let g = t_rel.inverse().compose(&tc.inverse().compose(&tk)).log() / h;
            j_numeric.set_column(col, &g);
        }
        let diff = (j_analytic - j_numeric).amax();
        assert!(diff < 1e-6, "relative-cov Jacobian mismatch amax={diff:e}");
    }
}
