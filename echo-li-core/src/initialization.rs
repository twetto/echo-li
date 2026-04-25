use nalgebra::Vector3;
use echo_lie::{SO3, SE3};
use crate::mathematical::IMUVelocity;

/// Estimate initial pose from stationary IMU accelerometer readings.
pub fn estimate_initial_pose(
    imu_readings: &[IMUVelocity],
    n_samples: usize,
) -> SE3 {
    if imu_readings.len() < 5 {
        return SE3::identity();
    }

    let n = n_samples.min(imu_readings.len());

    // Average accelerometer readings
    let mut acc_sum = Vector3::zeros();
    for i in 0..n {
        acc_sum += imu_readings[i].acc;
    }
    let acc_avg = acc_sum / (n as f64);

    // Measured gravity direction in body frame
    // When stationary, acc = R^T * [0, 0, g]
    // So g_body = acc_avg / ||acc_avg|| should map to [0, 0, 1] in world frame
    let g_body = acc_avg.normalize();
    let g_world = Vector3::new(0.0, 0.0, 1.0);

    // Find rotation: R * g_body = g_world
    let r = SO3::from_vectors(&g_body, &g_world);

    SE3::new(r, Vector3::zeros())
}

/// Check if the first n_samples IMU readings indicate a stationary platform.
pub fn check_stationary(
    imu_readings: &[IMUVelocity],
    n_samples: usize,
    gyro_std_threshold: f64,
    acc_std_threshold: f64,
) -> bool {
    if imu_readings.len() < 10 {
        return false;
    }

    let n = n_samples.min(imu_readings.len());

    // Compute gyro stats
    let mut gyro_sum = Vector3::zeros();
    let mut gyro_sq_sum = 0.0;
    
    // Compute acc norm stats
    let mut acc_norm_sum = 0.0;
    let mut acc_norm_sq_sum = 0.0;

    for i in 0..n {
        let gyr = imu_readings[i].gyr;
        gyro_sum += gyr;
        gyro_sq_sum += gyr.norm_squared();

        let acc_norm = imu_readings[i].acc.norm();
        acc_norm_sum += acc_norm;
        acc_norm_sq_sum += acc_norm * acc_norm;
    }

    let nf = n as f64;
    
    // Std dev of gyro components (approximate with average of squares)
    let gyro_mean_sq = gyro_sq_sum / nf;
    let gyro_vec_mean = gyro_sum / nf;
    let gyro_var = (gyro_mean_sq - gyro_vec_mean.norm_squared()).max(0.0);
    let gyro_std = gyro_var.sqrt();

    // Std dev of acc norm
    let acc_norm_mean = acc_norm_sum / nf;
    let acc_norm_var = (acc_norm_sq_sum / nf - acc_norm_mean * acc_norm_mean).max(0.0);
    let acc_std = acc_norm_var.sqrt();

    gyro_std < gyro_std_threshold && acc_std < acc_std_threshold
}
