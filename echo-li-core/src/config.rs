use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

use crate::depth::occupancy::{LocalOccupancySettings, OccupancyUpdateMode};
use crate::depth::patch_depth::{PatchDepthCameraMode, PatchDepthSettings, PatchDepthWarpMode};
use crate::depth::sparse_gb::{DepthParametrization, SparseVogSettings};
use crate::ImuBiasGroup;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct RudolfVConfig {
    pub equalise_image_histogram: bool,
    /// Histogram equalisation method: "none", "global", or "clahe". When
    /// absent, falls back to `equaliseImageHistogram` (true => global).
    #[serde(default)]
    pub histeq: Option<String>,
    /// CLAHE tile size in pixels (used when histeq == "clahe").
    #[serde(default = "default_clahe_tile_size")]
    pub clahe_tile_size: usize,
    /// CLAHE clip limit (used when histeq == "clahe").
    #[serde(default = "default_clahe_clip_limit")]
    pub clahe_clip_limit: f32,
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

/// Civera inverse-depth band is `[d0*exp(-rel), d0*exp(+rel)]`; 1.0 spans 0.37x-2.7x.
fn default_civera_birth_rel_std() -> f64 {
    1.0
}

fn default_epipolar_min_baseline() -> f64 {
    1e-3
}

fn default_clahe_tile_size() -> usize {
    256
}

fn default_clahe_clip_limit() -> f32 {
    4.0
}

fn default_epipolar_max_reject_frac() -> f64 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct EqfInitialValue {
    pub scene_depth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct EqfInitialVariance {
    pub attitude: f64,
    pub bias_acc: f64,
    pub bias_gyr: f64,
    pub camera_attitude: f64,
    pub camera_position: f64,
    pub point: f64,
    #[serde(default)]
    pub point_depth: Option<f64>,
    pub position: f64,
    pub velocity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct EqfMeasurementNoise {
    pub feature: f64,
    pub feature_outlier_abs: f64,
    pub feature_outlier_prob: f64,
    pub feature_retention: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Enable the stereo log-inverse-range measurement channel.
    #[serde(default)]
    pub stereo_measurement: bool,
    /// Chi²(1) gate on the stereo range innovation; 0 disables.
    #[serde(default)]
    pub range_gate_chi2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct EqfVelocityNoise {
    pub acc: f64,
    pub acc_bias: f64,
    pub gyr: f64,
    pub gyr_bias: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct MainConfig {
    pub write_state: bool,
    pub camera_lag: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
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
    pub flow_age_rate_px_per_frame: Option<f64>,
    #[serde(default)]
    pub bias_walk_var: Option<f64>,
    #[serde(default)]
    pub pose_range_scale: Option<f64>,
    #[serde(default)]
    pub pose_range_coherent: Option<f64>,
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
    #[serde(default, alias = "reanchor_flow_px")]
    pub birth_min_flow_px: Option<f64>,
    #[serde(default)]
    pub pose_range_direct: Option<bool>,
    #[serde(default)]
    pub scale_sigma: Option<f64>,
    #[serde(default)]
    pub conv_range_sigma_frac: Option<f64>,
    #[serde(default)]
    pub civera_birth_depth: Option<f64>,
    #[serde(default)]
    pub civera_rho_sigma_frac: Option<f64>,
    #[serde(default)]
    pub vis_min_depth: Option<f64>,
    #[serde(default)]
    pub vis_max_depth: Option<f64>,
}

fn default_true() -> bool {
    true
}

fn default_sparse_parametrization() -> String {
    "invdepth_additive3d".to_string()
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
        if let Some(v) = self.flow_age_rate_px_per_frame {
            settings.flow_age_rate_px_per_frame = v;
        }
        if let Some(v) = self.bias_walk_var {
            settings.bias_walk_var = v;
        }
        if let Some(v) = self.pose_range_scale {
            settings.pose_range_scale = v;
        }
        if let Some(v) = self.pose_range_coherent {
            settings.pose_range_coherent = v;
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
        if let Some(v) = self.pose_range_direct {
            settings.pose_range_direct = v;
        }
        if let Some(v) = self.scale_sigma {
            settings.scale_sigma = v;
        }
        if let Some(v) = self.conv_range_sigma_frac {
            settings.conv_range_sigma_frac = v;
        }
        if let Some(v) = self.civera_birth_depth {
            settings.civera_birth_depth = v;
        }
        if let Some(v) = self.civera_rho_sigma_frac {
            settings.civera_rho_sigma_frac = v;
        }
        if let Some(v) = self.birth_min_flow_px {
            settings.birth_min_flow_px = v;
        }
        settings
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PatchDepthConfig {
    /// Run the patch-depth mapper. Two configs already set this to false and were
    /// ignored, because the key was never a field and the section's mere PRESENCE did
    /// the enabling -- so `enabled: false` switched the mapper ON. Honoured now.
    #[serde(default = "default_true")]
    pub enabled: bool,
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

/// The MSCEqF backend, as a config section.
///
/// Present and `enabled` selects it in place of the EqVIO filter. Everything here is
/// what `VIOConfig`'s other sections cannot supply: MSCEqF needs a `SystemOrigin` and a
/// `ProcessNoise`, neither of which has an EqVIO counterpart.
///
/// Three values are NOT here, because they come from elsewhere and must not be
/// duplicated: the initial attitude and position come from gravity alignment at
/// start-up, and the camera extrinsic is the same `t_bs` the rest of the system uses.
/// `ZUPT:` section. Thresholds are on the VARIABILITY of the IMU over the window, not
/// on magnitude -- a magnitude test cannot tell rest from constant-velocity motion.
/// Defaults are the values validated on the bikevid rig (window 25 samples at 122 Hz,
/// gyro RMS 0.05 rad/s, |accel| std 0.15 m/s^2, sigma_v 1e-2 m/s), where they fired on
/// the rest phases and never while riding (measured rest accel-std 0.015 vs 2.1
/// riding, a 140x margin).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct ZuptConfig {
    pub enabled: bool,
    /// Detector window, IMU samples.
    pub window: usize,
    /// Gyro RMS about its window mean, rad/s.
    pub gyro_std_threshold: f64,
    /// Std of |accel| over the window, m/s^2.
    pub acc_std_threshold: f64,
    /// Assumed std of the zero-velocity measurement, m/s.
    pub sigma_v: f64,
}

impl Default for ZuptConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window: 25,
            gyro_std_threshold: 0.05,
            acc_std_threshold: 0.15,
            sigma_v: 1.0e-2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct MsceqfConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_num_clones")]
    pub num_clones: usize,
    /// Feature noise in PIXELS. Converted to the normalized units the filter works in
    /// by dividing by fx at construction, so this reads like EqVIO's
    /// `measurementNoise.feature` rather than requiring a hand-computed 1/fx.
    #[serde(default = "default_pixel_std")]
    pub pixel_std: f64,
    #[serde(default = "default_one")]
    pub chi2_mult: f64,
    /// 0 reuses `chi2Mult`.
    #[serde(default)]
    pub slam_chi2_mult: f64,
    #[serde(default = "default_max_track")]
    pub max_track: usize,
    /// 0 means a full window span (`numClones`).
    #[serde(default)]
    pub slam_min_len: usize,
    #[serde(default)]
    pub max_slam: usize,
    #[serde(default = "default_true")]
    pub curvature: bool,
    /// Extra flush trigger for long-clone-window configurations. 0 = off; see
    /// `MsceqfSettings::update_span`.
    #[serde(default)]
    pub update_span: usize,
    /// Refuse to promote a landmark triangulated beyond this range, in metres. 0 = no
    /// cap. See `MsceqfSettings::max_promote_range`.
    #[serde(default)]
    pub max_promote_range: f64,
    /// Direct-product (static) bias chart instead of semi-direct.
    #[serde(default)]
    pub direct_product_bias: bool,
    /// Prior depth (m) for Civera in-state landmark birth; 0 = triangulated birth.
    /// See `MsceqfSettings::civera_birth_depth`.
    #[serde(default)]
    pub civera_birth_depth: f64,
    /// Inverse-depth relative std for Civera birth. See
    /// `MsceqfSettings::civera_birth_rel_std`.
    #[serde(default = "default_civera_birth_rel_std")]
    pub civera_birth_rel_std: f64,
    /// Gravity VECTOR in the world frame, m/s^2. Sign and axis are frame conventions,
    /// not preferences: NED wants +z, a z-up frame wants -z, and getting it backwards
    /// integrates gravity as phantom acceleration.
    #[serde(default = "default_gravity")]
    pub gravity: [f64; 3],
    #[serde(default)]
    pub initial_std: MsceqfInitialStd,
    #[serde(default)]
    pub process_noise: MsceqfProcessNoise,
}

/// Initial standard deviations for the MSCEqF covariance blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct MsceqfInitialStd {
    /// Dd block, in MSCEqF tangent order: attitude, velocity, position.
    pub attitude: f64,
    pub velocity: f64,
    pub position: f64,
    /// Bias block (6).
    pub bias: f64,
    /// Camera-extrinsic block (6).
    pub extrinsics: f64,
}

impl Default for MsceqfInitialStd {
    fn default() -> Self {
        Self {
            attitude: 1.0e-2,
            velocity: 1.0e-1,
            position: 1.0e-4,
            bias: 1.0e-2,
            extrinsics: 1.0e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct MsceqfProcessNoise {
    pub accel_density: f64,
    pub gyro_density: f64,
    pub accel_random_walk: f64,
    pub gyro_random_walk: f64,
    /// 1 uses `I + H*dt`; anything else uses the full matrix exponential.
    #[serde(default)]
    pub transition_order: usize,
}

impl Default for MsceqfProcessNoise {
    fn default() -> Self {
        Self {
            accel_density: 1.6798e-3,
            gyro_density: 1.2724e-3,
            accel_random_walk: 3.0e-3,
            gyro_random_walk: 1.0e-4,
            transition_order: 0,
        }
    }
}

fn default_num_clones() -> usize {
    11
}
fn default_pixel_std() -> f64 {
    1.0
}
fn default_one() -> f64 {
    1.0
}
fn default_max_track() -> usize {
    200
}
fn default_gravity() -> [f64; 3] {
    [0.0, 0.0, -9.81]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(rename = "MSCEqF", default)]
    pub msceqf: Option<MsceqfConfig>,
    /// Zero-velocity updates, applied by the BACKEND on every IMU sample the detector
    /// calls stationary. Absent or `enabled: false` leaves the backend without one --
    /// which, on a rig that starts at rest, is the single largest error source this
    /// codebase has measured (764 m of dead-reckoned drift on bikevid before a ZUPT,
    /// 0.0 m after; 81 m ATE on EuRoC V1_01 from a rest start without one).
    #[serde(rename = "ZUPT", default)]
    pub zupt: Option<ZuptConfig>,
    #[serde(rename = "Rerun", default)]
    pub rerun: RerunVisConfig,
    pub eqf: EqfConfig,
    pub main: MainConfig,
}

/// Per-entity toggles for the Rerun visualisation (`--vis`). Everything
/// defaults to on; set an entry to false in the YAML `Rerun:` section to hide
/// it. `histeq_image` shows the tracker's preprocessed (histogram-equalised)
/// image instead of the raw frame when available.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
#[serde(deny_unknown_fields)]
pub struct RerunVisConfig {
    pub image: bool,
    pub histeq_image: bool,
    pub features: bool,
    pub sparse_image: bool,
    pub patch_depth: bool,
    pub patch_depth_cov: bool,
    pub trajectory: bool,
    pub groundtruth: bool,
    pub landmarks: bool,
    pub sparse_world: bool,
    pub occupied_cells: bool,
    pub free_cells: bool,
    pub camera_axes: bool,
}

impl Default for RerunVisConfig {
    fn default() -> Self {
        Self {
            image: true,
            histeq_image: true,
            features: true,
            sparse_image: true,
            patch_depth: true,
            patch_depth_cov: true,
            trajectory: true,
            groundtruth: true,
            landmarks: true,
            sparse_world: true,
            occupied_cells: true,
            free_cells: true,
            camera_axes: true,
        }
    }
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
        settings.use_stereo_measurement = self.eqf.settings.stereo_measurement;
        settings.range_gate_chi2 = self.eqf.settings.range_gate_chi2;
        settings.max_landmarks = self.eqf.max_features;
        settings.sigma_bearing = self.eqf.measurement_noise.feature;
        settings.outlier_threshold = self.eqf.measurement_noise.feature_outlier_abs;
        settings.initial_point_variance = self.eqf.initial_variance.point;
        settings.initial_point_depth_variance = self.eqf.initial_variance.point_depth;

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
        settings.use_median_depth = self.eqf.settings.use_median_depth;

        settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midair_stereo_config_parses_and_enables_measurement() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../configs/eqvio_midair_stereo.yaml"
        );
        let cfg = VIOConfig::from_yaml(path).expect("parse eqvio_midair_stereo.yaml");
        assert!(cfg.eqf.settings.stereo_measurement);
        assert!((cfg.eqf.settings.range_gate_chi2 - 6.63).abs() < 1e-9);
        let settings = cfg.to_filter_settings();
        assert!(settings.use_stereo_measurement);
        assert!((settings.range_gate_chi2 - 6.63).abs() < 1e-9);
    }
}

#[cfg(test)]
mod config_strictness_tests {
    use super::*;

    /// Every config the repo ships must LOAD. With `deny_unknown_fields` this is no
    /// longer a formality: a key the parser does not know is now an error, so this
    /// test is what stops a config drifting away from the code that reads it.
    ///
    /// The failure it exists to prevent has happened twice. `useMedianDepth` sat in all
    /// eleven configs for months with no reader anywhere, then silently changed every
    /// baseline the day it was wired. A harness whitelist dropped `pose_range_direct`
    /// and cost a retracted result table. In both cases the config said one thing, the
    /// system did another, and nothing complained.
    #[test]
    fn every_shipped_config_loads() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("configs");
        let mut checked = 0;
        let mut failures = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("configs/ must exist") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            checked += 1;
            if let Err(e) = VIOConfig::from_yaml(path.to_str().unwrap()) {
                failures.push(format!(
                    "  {}: {e}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
        assert!(checked > 0, "no configs found in {}", dir.display());
        assert!(
            failures.is_empty(),
            "{} of {checked} shipped configs no longer load:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// An unknown key is an ERROR, not a shrug. This is the whole point of the
    /// attribute, so it is pinned rather than assumed.
    #[test]
    fn an_unknown_key_is_rejected() {
        let yaml = r#"
RudolfV:
  maxFeatures: 10
  thisKeyDoesNotExist: 42
eqf:
  maxFeatures: 10
main:
  writeState: false
"#;
        let err = serde_yaml::from_str::<VIOConfig>(yaml)
            .expect_err("an unknown key must be rejected, not ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("thisKeyDoesNotExist"),
            "the error must NAME the offending key so it can be found; got: {msg}"
        );
    }
}
