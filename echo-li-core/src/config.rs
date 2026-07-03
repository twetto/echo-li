use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

use crate::ImuBiasGroup;
use crate::depth::occupancy::{LocalOccupancySettings, OccupancyUpdateMode};
use crate::depth::patch_depth::{PatchDepthCameraMode, PatchDepthSettings, PatchDepthWarpMode};
use crate::depth::sparse_gb::{DepthParametrization, SparseVogSettings};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RudolfVConfig {
    pub equalise_image_histogram: bool,
    pub feature_dist: f64,
    pub feature_search_threshold: f64,
    #[serde(default)]
    pub fast_threshold: Option<u8>,
    /// Corner detector: "fast" (default), "harris", or "shi_tomasi".
    #[serde(default)]
    pub detector: Option<String>,
    /// Shi-Tomasi minimum-eigenvalue floor (only used when detector == shi_tomasi).
    #[serde(default)]
    pub shi_tomasi_threshold: Option<f32>,
    /// Shi-Tomasi structure-tensor block size (only used when detector == shi_tomasi).
    #[serde(default)]
    pub shi_tomasi_block_size: Option<usize>,
    #[serde(default)]
    pub lbp_policy: Option<String>,
    /// Compute the level-0 patch residual per track to feed the KLT quality
    /// term in the reservoir score (extra patch pass; off by default).
    #[serde(default)]
    pub klt_residual: bool,
    /// Run the frontend essential-matrix RANSAC geometric verification. On by
    /// default, but it degenerates under rotation-dominant motion (translation
    /// ~0 => E ill-posed) and falsely rejects good tracks. Off => let the EqF's
    /// soft outlier model handle outliers instead.
    #[serde(default = "default_true")]
    pub enable_ransac: bool,
    /// Squared-Sampson threshold (normalized coords, like ransacParams.
    /// inlierThreshold) for the pose-prior epipolar gate. Active only on
    /// frames where a relative-pose prior is supplied (Frontend::
    /// set_pose_prior); replaces RANSAC there. 0 disables.
    #[serde(default)]
    pub epipolar_gate_threshold: f64,
    /// Consensus refit of the prior gate (kept only if it increases inliers).
    #[serde(default)]
    pub epipolar_refine: bool,
    /// Skip the gate below this inter-frame baseline [m]: near-zero translation
    /// makes the prior's E pure noise (detonated MH_02's slow segments).
    #[serde(default = "default_epipolar_min_baseline")]
    pub epipolar_min_baseline: f64,
    /// Distrust the prior when it would reject more than this fraction of
    /// tracks (breaks the reject->starve->diverge feedback loop).
    #[serde(default = "default_epipolar_max_reject_frac")]
    pub epipolar_max_reject_frac: f64,
    pub max_features: usize,
    pub max_level: usize,
}

fn default_epipolar_min_baseline() -> f64 {
    1e-3
}

fn default_epipolar_max_reject_frac() -> f64 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfInitialValue {
    pub scene_depth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfInitialVariance {
    pub attitude: f64,
    pub bias_acc: f64,
    pub bias_gyr: f64,
    pub camera_attitude: f64,
    pub camera_position: f64,
    pub point: f64,
    pub position: f64,
    pub velocity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfMeasurementNoise {
    pub feature: f64,
    pub feature_outlier_abs: f64,
    pub feature_outlier_prob: f64,
    pub feature_retention: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfProcessVariance {
    pub attitude: f64,
    pub bias_acc: f64,
    pub bias_gyr: f64,
    pub camera_attitude: f64,
    pub camera_position: f64,
    pub point: f64,
    pub position: f64,
    pub velocity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfSettings {
    pub coordinate_choice: String,
    pub fast_riccati: bool,
    #[serde(default)]
    pub imu_bias_group: Option<String>,
    pub use_discrete_innovation_lift: bool,
    pub use_discrete_velocity_lift: bool,
    pub use_equivariant_output: bool,
    pub use_feature_predictions: bool,
    pub use_median_depth: bool,
    /// Riccati propagation variant: "fast" (default) = per-sample transport;
    /// "faster" = covariance transport batched across each IMU sub-frame
    /// (flushed when the next image frame arrives).
    #[serde(default)]
    pub riccati_variant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfVelocityNoise {
    pub acc: f64,
    pub acc_bias: f64,
    pub gyr: f64,
    pub gyr_bias: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EqfConfig {
    pub max_features: usize,
    pub initial_value: EqfInitialValue,
    pub initial_variance: EqfInitialVariance,
    pub measurement_noise: EqfMeasurementNoise,
    pub process_variance: EqfProcessVariance,
    pub settings: EqfSettings,
    pub velocity_noise: EqfVelocityNoise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MainConfig {
    pub write_state: bool,
    pub camera_lag: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparseVogConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_sparse_parametrization")]
    pub parametrization: String,
    #[serde(default)]
    pub max_pool_size: Option<usize>,
    #[serde(default)]
    pub min_track_length: Option<usize>,
    #[serde(default)]
    pub conv_inlier_ratio: Option<f64>,
    #[serde(default)]
    pub conv_variance_threshold: Option<f64>,
    #[serde(default)]
    pub init_depth_var: Option<f64>,
    #[serde(default)]
    pub init_invdepth_var: Option<f64>,
    #[serde(default)]
    pub sigma_pixel: Option<f64>,
    #[serde(default)]
    pub uniform_z_max: Option<f64>,
    #[serde(default)]
    pub uniform_rho_max: Option<f64>,
    #[serde(default)]
    pub uniform_d_min: Option<f64>,
    #[serde(default)]
    pub uniform_d_max: Option<f64>,
    #[serde(default)]
    pub a_init: Option<f64>,
    #[serde(default)]
    pub b_init: Option<f64>,
    #[serde(default)]
    pub ab_min: Option<f64>,
    #[serde(default)]
    pub ab_max: Option<f64>,
    #[serde(default)]
    pub min_inlier_ratio: Option<f64>,
    #[serde(default)]
    pub mahalanobis_reset_chi2: Option<f64>,
    #[serde(default)]
    pub process_depth_var: Option<f64>,
    #[serde(default)]
    pub min_parallax: Option<f64>,
    #[serde(default)]
    pub min_cos_sim: Option<f64>,
    #[serde(default)]
    pub min_depth: Option<f64>,
    #[serde(default)]
    pub max_depth: Option<f64>,
    #[serde(default)]
    pub reanchor_flow_px: Option<f64>,
    #[serde(default)]
    pub vis_min_depth: Option<f64>,
    #[serde(default)]
    pub vis_max_depth: Option<f64>,
}

fn default_true() -> bool {
    true
}

fn default_sparse_parametrization() -> String {
    "invdepth3d".to_string()
}

impl SparseVogConfig {
    pub fn to_sparse_settings(&self) -> SparseVogSettings {
        let mut settings = SparseVogSettings::default();
        settings.parametrization = match self.parametrization.to_ascii_lowercase().as_str() {
            "euclidean" => DepthParametrization::Euclidean,
            "polar" | "polar3d" => DepthParametrization::Polar,
            _ => DepthParametrization::InvDepth,
        };
        if let Some(v) = self.max_pool_size {
            settings.max_pool_size = v;
        }
        if let Some(v) = self.min_track_length {
            settings.min_track_length = v;
        }
        if let Some(v) = self.conv_inlier_ratio {
            settings.conv_inlier_ratio = v;
        }
        if let Some(v) = self.conv_variance_threshold {
            settings.conv_variance_threshold = v;
        }
        if let Some(v) = self.init_depth_var {
            settings.init_depth_var = v;
        }
        if let Some(v) = self.init_invdepth_var {
            settings.init_invdepth_var = v;
        }
        if let Some(v) = self.sigma_pixel {
            settings.sigma_pixel = v;
        }
        if let Some(v) = self.uniform_z_max {
            settings.uniform_z_max = v;
        }
        if let Some(v) = self.uniform_rho_max {
            settings.uniform_rho_max = v;
        }
        if let Some(v) = self.uniform_d_min {
            settings.uniform_d_min = v;
        }
        if let Some(v) = self.uniform_d_max {
            settings.uniform_d_max = v;
        }
        if let Some(v) = self.a_init {
            settings.a_init = v;
        }
        if let Some(v) = self.b_init {
            settings.b_init = v;
        }
        if let Some(v) = self.ab_min {
            settings.ab_min = v;
        }
        if let Some(v) = self.ab_max {
            settings.ab_max = v;
        }
        if let Some(v) = self.min_inlier_ratio {
            settings.min_inlier_ratio = v;
        }
        if let Some(v) = self.mahalanobis_reset_chi2 {
            settings.mahalanobis_reset_chi2 = v;
        }
        if let Some(v) = self.process_depth_var {
            settings.process_depth_var = v;
        }
        if let Some(v) = self.min_parallax {
            settings.min_parallax = v;
        }
        if let Some(v) = self.min_cos_sim {
            settings.min_cos_sim = v;
        }
        if let Some(v) = self.min_depth {
            settings.min_depth = v;
        }
        if let Some(v) = self.max_depth {
            settings.max_depth = v;
        }
        if let Some(v) = self.reanchor_flow_px {
            settings.reanchor_flow_px = v;
        }
        settings
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatchDepthConfig {
    #[serde(default)]
    pub camera_mode: Option<String>,
    #[serde(default)]
    pub warp_mode: Option<String>,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub patch_size: Option<usize>,
    #[serde(default)]
    pub patch_stride: Option<usize>,
    #[serde(default)]
    pub cell_size: Option<usize>,
    #[serde(default)]
    pub min_depth: Option<f64>,
    #[serde(default)]
    pub max_depth: Option<f64>,
    #[serde(default)]
    pub photo_huber_delta: Option<f64>,
    #[serde(default)]
    pub sigma_photo: Option<f64>,
    #[serde(default)]
    pub pose_angular_velocity_var: Option<f64>,
    #[serde(default)]
    pub n_gn_iters: Option<usize>,
    #[serde(default)]
    pub gn_eta_convergence_tol: Option<f64>,
    #[serde(default)]
    pub fd_eps: Option<f64>,
    #[serde(default)]
    pub lambda_seed: Option<f64>,
    #[serde(default)]
    pub seed_radius_px: Option<f64>,
    #[serde(default)]
    pub sigma_seed_floor: Option<f64>,
    #[serde(default)]
    pub n_search_candidates: Option<usize>,
    #[serde(default)]
    pub search_half_range: Option<f64>,
    #[serde(default)]
    pub min_baseline_ratio: Option<f64>,
    #[serde(default)]
    pub max_baseline_ratio: Option<f64>,
    #[serde(default)]
    pub min_photo_curvature: Option<f64>,
    #[serde(default)]
    pub max_photo_residual: Option<f64>,
    #[serde(default)]
    pub min_structure_eigen: Option<f64>,
    #[serde(default)]
    pub max_structure_condition: Option<f64>,
    #[serde(default)]
    pub n_pyramid_levels: Option<usize>,
    #[serde(default)]
    pub var_floor: Option<f64>,
    #[serde(default)]
    pub status_weight_photo: Option<f64>,
    #[serde(default)]
    pub status_weight_seed: Option<f64>,
    #[serde(default)]
    pub tiled_tile_size: Option<usize>,
    #[serde(default)]
    pub tiled_tile_overlap: Option<usize>,
    #[serde(default)]
    pub vis_min_depth: Option<f64>,
    #[serde(default)]
    pub vis_max_depth: Option<f64>,
    #[serde(default)]
    pub cov_vis_min: Option<f64>,
    #[serde(default)]
    pub cov_vis_max: Option<f64>,
}

impl PatchDepthConfig {
    pub fn to_patch_depth_settings(&self) -> PatchDepthSettings {
        let mut settings = PatchDepthSettings::default();
        if let Some(v) = &self.camera_mode {
            settings.camera_mode = match v.to_ascii_lowercase().as_str() {
                "undistorted_pinhole" | "undistorted-pinhole" | "pinhole" => {
                    PatchDepthCameraMode::UndistortedPinhole
                }
                "tiled_bearing" | "tiled-bearing" | "tiledbearing" | "tiled_pinhole"
                | "tiled-pinhole" | "tiledpinhole" => PatchDepthCameraMode::TiledBearing,
                "per_patch_bearing" | "per-patch-bearing" | "perpatchbearing" | "per_patch"
                | "per-patch" | "perpatch" => PatchDepthCameraMode::PerPatchBearing,
                _ => PatchDepthCameraMode::RawDistorted,
            };
        }
        if let Some(v) = &self.warp_mode {
            settings.warp_mode = match v.to_ascii_lowercase().as_str() {
                "fast_translation" | "fast-translation" | "fasttranslation" | "translation" => {
                    PatchDepthWarpMode::FastTranslation
                }
                _ => PatchDepthWarpMode::Exact,
            };
        }
        if let Some(v) = self.scale {
            settings.scale = v;
        }
        if let Some(v) = self.patch_size {
            settings.patch_size = v;
        }
        if let Some(v) = self.patch_stride {
            settings.patch_stride = v;
        }
        // cell_size is ignored — output is now per-pixel
        if let Some(v) = self.min_depth {
            settings.min_depth = v;
        }
        if let Some(v) = self.max_depth {
            settings.max_depth = v;
        }
        if let Some(v) = self.photo_huber_delta {
            settings.photo_huber_delta = v;
        }
        if let Some(v) = self.sigma_photo {
            settings.sigma_photo = v;
        }
        if let Some(v) = self.pose_angular_velocity_var {
            settings.pose_angular_velocity_var = v;
        }
        if let Some(v) = self.n_gn_iters {
            settings.n_gn_iters = v;
        }
        if let Some(v) = self.gn_eta_convergence_tol {
            settings.gn_eta_convergence_tol = v;
        }
        if let Some(v) = self.fd_eps {
            settings.fd_eps = v;
        }
        if let Some(v) = self.lambda_seed {
            settings.lambda_seed = v;
        }
        if let Some(v) = self.seed_radius_px {
            settings.seed_radius_px = v;
        }
        if let Some(v) = self.sigma_seed_floor {
            settings.sigma_seed_floor = v;
        }
        if let Some(v) = self.n_search_candidates {
            settings.n_search_candidates = v;
        }
        if let Some(v) = self.search_half_range {
            settings.search_half_range = v;
        }
        if let Some(v) = self.min_baseline_ratio {
            settings.min_baseline_ratio = v;
        }
        if let Some(v) = self.max_baseline_ratio {
            settings.max_baseline_ratio = v;
        }
        if let Some(v) = self.min_photo_curvature {
            settings.min_photo_curvature = v;
        }
        if let Some(v) = self.max_photo_residual {
            settings.max_photo_residual = v;
        }
        if let Some(v) = self.min_structure_eigen {
            settings.min_structure_eigen = v;
        }
        if let Some(v) = self.max_structure_condition {
            settings.max_structure_condition = v;
        }
        if let Some(v) = self.n_pyramid_levels {
            settings.n_pyramid_levels = v;
        }
        if let Some(v) = self.var_floor {
            settings.var_floor = v;
        }
        if let Some(v) = self.status_weight_photo {
            settings.status_weight_photo = v;
        }
        if let Some(v) = self.status_weight_seed {
            settings.status_weight_seed = v;
        }
        if let Some(v) = self.tiled_tile_size {
            settings.tiled_tile_size = v;
        }
        if let Some(v) = self.tiled_tile_overlap {
            settings.tiled_tile_overlap = v;
        }
        settings
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalOccupancyConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub resolution: Option<f64>,
    #[serde(default)]
    pub width_cells: Option<usize>,
    #[serde(default)]
    pub height_cells: Option<usize>,
    #[serde(default)]
    pub sample_stride: Option<usize>,
    #[serde(default)]
    pub min_range: Option<f64>,
    #[serde(default)]
    pub max_range: Option<f64>,
    #[serde(default)]
    pub max_eta_std: Option<f64>,
    #[serde(default)]
    pub log_odds_hit: Option<f32>,
    #[serde(default)]
    pub log_odds_miss: Option<f32>,
    #[serde(default)]
    pub log_odds_min: Option<f32>,
    #[serde(default)]
    pub log_odds_max: Option<f32>,
    #[serde(default)]
    pub occupied_threshold: Option<f32>,
    #[serde(default)]
    pub free_threshold: Option<f32>,
    /// "fixed" / "fixed_increment" (v0) or "uncertainty_aware" / "sigma" (v1).
    #[serde(default)]
    pub update_mode: Option<String>,
    #[serde(default)]
    pub band_k: Option<f64>,
    #[serde(default)]
    pub sigma_floor_factor: Option<f64>,
    #[serde(default)]
    pub min_confidence_weight: Option<f64>,
    #[serde(default)]
    pub min_obstacle_height: Option<f64>,
    #[serde(default)]
    pub max_obstacle_height: Option<f64>,
}

impl LocalOccupancyConfig {
    pub fn to_local_occupancy_settings(&self) -> LocalOccupancySettings {
        let mut settings = LocalOccupancySettings::default();
        settings.enabled = self.enabled;
        if let Some(v) = self.resolution {
            settings.resolution = v;
        }
        if let Some(v) = self.width_cells {
            settings.width_cells = v;
        }
        if let Some(v) = self.height_cells {
            settings.height_cells = v;
        }
        if let Some(v) = self.sample_stride {
            settings.sample_stride = v;
        }
        if let Some(v) = self.min_range {
            settings.min_range = v;
        }
        if let Some(v) = self.max_range {
            settings.max_range = v;
        }
        if let Some(v) = self.max_eta_std {
            settings.max_eta_std = v;
        }
        if let Some(v) = self.log_odds_hit {
            settings.log_odds_hit = v;
        }
        if let Some(v) = self.log_odds_miss {
            settings.log_odds_miss = v;
        }
        if let Some(v) = self.log_odds_min {
            settings.log_odds_min = v;
        }
        if let Some(v) = self.log_odds_max {
            settings.log_odds_max = v;
        }
        if let Some(v) = self.occupied_threshold {
            settings.occupied_threshold = v;
        }
        if let Some(v) = self.free_threshold {
            settings.free_threshold = v;
        }
        if let Some(v) = &self.update_mode {
            settings.update_mode = match v.as_str() {
                "fixed" | "fixed_increment" | "v0" => OccupancyUpdateMode::FixedIncrement,
                "uncertainty_aware" | "sigma" | "v1" => OccupancyUpdateMode::UncertaintyAware,
                other => {
                    log::warn!(
                        "LocalOccupancy.update_mode {other:?} unrecognised; using fixed_increment (v0)"
                    );
                    OccupancyUpdateMode::FixedIncrement
                }
            };
        }
        if let Some(v) = self.band_k {
            settings.band_k = v;
        }
        if let Some(v) = self.sigma_floor_factor {
            settings.sigma_floor_factor = v;
        }
        if let Some(v) = self.min_confidence_weight {
            settings.min_confidence_weight = v;
        }
        if let Some(v) = self.min_obstacle_height {
            settings.min_obstacle_height = v;
        }
        if let Some(v) = self.max_obstacle_height {
            settings.max_obstacle_height = v;
        }
        settings
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StereoConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub pyramid_levels: Option<usize>,
    #[serde(default)]
    pub patch_half_size: Option<usize>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub convergence_eps: Option<f64>,
    #[serde(default)]
    pub min_inv_depth: Option<f64>,
    #[serde(default)]
    pub max_inv_depth: Option<f64>,
    #[serde(default)]
    pub init_inv_depth: Option<f64>,
    #[serde(default)]
    pub max_residual: Option<f32>,
    #[serde(default)]
    pub n_search_candidates: Option<usize>,
    #[serde(default)]
    pub knn_propagation: Option<usize>,
    #[serde(default)]
    pub histeq: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIOConfig {
    #[serde(rename = "RudolfV")]
    pub rudolf_v: RudolfVConfig,
    #[serde(rename = "SparseVog", default)]
    pub sparse_vog: Option<SparseVogConfig>,
    #[serde(rename = "PatchDepth", default)]
    pub patch_depth: Option<PatchDepthConfig>,
    #[serde(rename = "LocalOccupancy", default)]
    pub local_occupancy: Option<LocalOccupancyConfig>,
    #[serde(rename = "Stereo", default)]
    pub stereo: Option<StereoConfig>,
    pub eqf: EqfConfig,
    pub main: MainConfig,
}

impl VIOConfig {
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let f = File::open(path)?;
        let config: VIOConfig = serde_yaml::from_reader(f)?;
        Ok(config)
    }

    pub fn to_filter_settings(&self) -> crate::VIOFilterSettings {
        let mut settings = crate::VIOFilterSettings::default();
        settings.coordinate_choice = self.eqf.settings.coordinate_choice.clone();
        settings.max_landmarks = self.eqf.max_features;
        settings.sigma_bearing = self.eqf.measurement_noise.feature;
        settings.initial_point_variance = self.eqf.initial_variance.point;

        // velocityNoise
        settings.sigma_gyroscope = self.eqf.velocity_noise.gyr;
        settings.sigma_accelerometer = self.eqf.velocity_noise.acc;
        settings.sigma_gyroscope_bias = self.eqf.velocity_noise.gyr_bias;
        settings.sigma_accelerometer_bias = self.eqf.velocity_noise.acc_bias;

        // initialVariance
        settings.initial_attitude_variance = self.eqf.initial_variance.attitude;
        settings.initial_position_variance = self.eqf.initial_variance.position;
        settings.initial_velocity_variance = self.eqf.initial_variance.velocity;
        settings.initial_bias_omega_variance = self.eqf.initial_variance.bias_gyr;
        settings.initial_bias_accel_variance = self.eqf.initial_variance.bias_acc;
        settings.initial_camera_attitude_variance = self.eqf.initial_variance.camera_attitude;
        settings.initial_camera_position_variance = self.eqf.initial_variance.camera_position;

        // processVariance
        settings.process_attitude = self.eqf.process_variance.attitude;
        settings.process_position = self.eqf.process_variance.position;
        settings.process_velocity = self.eqf.process_variance.velocity;
        settings.process_bias_gyr = self.eqf.process_variance.bias_gyr;
        settings.process_bias_acc = self.eqf.process_variance.bias_acc;
        settings.process_camera_attitude = self.eqf.process_variance.camera_attitude;
        settings.process_camera_position = self.eqf.process_variance.camera_position;
        settings.process_point = self.eqf.process_variance.point;

        // settings
        settings.use_equivariant_output = self.eqf.settings.use_equivariant_output;
        settings.imu_bias_group = self
            .eqf
            .settings
            .imu_bias_group
            .as_deref()
            .map(ImuBiasGroup::from_config)
            .unwrap_or_default();
        settings.use_discrete_correction = self.eqf.settings.use_discrete_innovation_lift;
        settings.use_discrete_velocity_lift = self.eqf.settings.use_discrete_velocity_lift;
        settings.use_faster_riccati = self
            .eqf
            .settings
            .riccati_variant
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("faster"))
            .unwrap_or(false);
        settings.initial_scene_depth = self.eqf.initial_value.scene_depth;

        settings
    }
}
