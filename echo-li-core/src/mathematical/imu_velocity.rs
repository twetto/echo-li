use nalgebra::Vector3;

/// IMU measurement: gyroscope and accelerometer data.
#[derive(Debug, Clone, Copy)]
pub struct IMUVelocity {
    pub stamp: f64,
    pub gyr: Vector3<f64>,
    pub acc: Vector3<f64>,
    pub gyr_bias_vel: Vector3<f64>,
    pub acc_bias_vel: Vector3<f64>,
}

impl IMUVelocity {
    pub fn new(stamp: f64, gyr: Vector3<f64>, acc: Vector3<f64>) -> Self {
        Self {
            stamp,
            gyr,
            acc,
            gyr_bias_vel: Vector3::zeros(),
            acc_bias_vel: Vector3::zeros(),
        }
    }
}
