use std::collections::HashMap;
use nalgebra::Vector2;

/// Vision measurement: timestamp and tracked feature coordinates.
#[derive(Debug, Clone)]
pub struct VisionMeasurement {
    pub stamp: f64,
    pub cam_coordinates: HashMap<u64, Vector2<f32>>,
}

impl VisionMeasurement {
    pub fn new(stamp: f64, cam_coordinates: HashMap<u64, Vector2<f32>>) -> Self {
        Self { stamp, cam_coordinates }
    }
}
