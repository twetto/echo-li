use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RudolfVConfig {
    pub equalise_image_histogram: bool,
    pub feature_dist: f64,
    pub feature_search_threshold: f64,
    pub max_features: usize,
    pub max_level: usize,
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
    pub use_discrete_innovation_lift: bool,
    pub use_discrete_velocity_lift: bool,
    pub use_equivariant_output: bool,
    pub use_feature_predictions: bool,
    pub use_median_depth: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIOConfig {
    #[serde(rename = "RudolfV")]
    pub rudolf_v: RudolfVConfig,
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
        settings.use_discrete_correction = self.eqf.settings.use_discrete_innovation_lift;
        settings.use_discrete_velocity_lift = self.eqf.settings.use_discrete_velocity_lift;
        settings.initial_scene_depth = self.eqf.initial_value.scene_depth;

        settings
    }
}
