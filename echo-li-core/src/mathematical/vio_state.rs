use nalgebra::{Vector3, Vector6};
use echo_lie::{SO3, SE3};

pub const GRAVITY_CONSTANT: f64 = 9.81007;

#[derive(Debug, Clone)]
pub struct StampedPose {
    pub stamp: f64,
    pub pose: SE3,
}

impl StampedPose {
    pub fn new(stamp: f64, pose: SE3) -> Self { Self { stamp, pose } }
}

#[derive(Debug, Clone)]
pub struct Landmark {
    pub p: Vector3<f64>,
    pub id: u64,
}

#[derive(Debug, Clone)]
pub struct VIOSensorState {
    pub input_bias: Vector6<f64>,
    pub pose: SE3,
    pub velocity: Vector3<f64>,
    pub camera_offset: SE3,
}

impl VIOSensorState {
    pub const CDIM: usize = 21;

    pub fn identity() -> Self {
        Self {
            input_bias: Vector6::zeros(),
            pose: SE3::identity(),
            velocity: Vector3::zeros(),
            camera_offset: SE3::identity(),
        }
    }

    pub fn gyro_bias(&self) -> Vector3<f64> {
        self.input_bias.fixed_rows::<3>(0).into_owned()
    }

    pub fn accel_bias(&self) -> Vector3<f64> {
        self.input_bias.fixed_rows::<3>(3).into_owned()
    }

    pub fn gravity_dir(&self) -> Vector3<f64> {
        self.pose.rotation.inverse().act(&Vector3::new(0.0, 0.0, 1.0))
    }
}

#[derive(Debug, Clone)]
pub struct VIOState {
    pub sensor: VIOSensorState,
    pub camera_landmarks: Vec<Landmark>,
}

impl VIOState {
    pub fn new(sensor: VIOSensorState, landmarks: Vec<Landmark>) -> Self {
        Self { sensor, camera_landmarks: landmarks }
    }

    pub fn dim(&self) -> usize {
        VIOSensorState::CDIM + 3 * self.camera_landmarks.len()
    }

    pub fn get_ids(&self) -> Vec<u64> {
        self.camera_landmarks.iter().map(|lm| lm.id).collect()
    }
}

/// Standard kinematic integration (Ground Truth for parity tests).
pub fn integrate_system_function(xi: &VIOState, imu: &crate::mathematical::imu_velocity::IMUVelocity, dt: f64) -> VIOState {
    let mut xi1 = xi.clone();
    let sensor = &xi.sensor;
    
    let v_gyr = imu.gyr - sensor.gyro_bias();
    let v_acc = imu.acc - sensor.accel_bias();
    
    // 1. Rotation
    xi1.sensor.pose.rotation = sensor.pose.rotation.compose(&SO3::exp(&(dt * v_gyr)));
    
    // 2. Position and Velocity
    let grav = Vector3::new(0.0, 0.0, -GRAVITY_CONSTANT);
    let world_acc = sensor.pose.rotation.act(&v_acc) + grav;
    
    let world_vel = sensor.pose.rotation.act(&sensor.velocity);
    let world_pos = sensor.pose.translation + dt * world_vel + 0.5 * dt * dt * world_acc;
    let world_vel_next = world_vel + dt * world_acc;
    
    xi1.sensor.pose.translation = world_pos;
    xi1.sensor.velocity = xi1.sensor.pose.rotation.inverse().act(&world_vel_next);
    
    // 3. Landmarks (constant in global frame)
    // p_cam_next = T_wc_next.inv() * p_world
    let t_wc = sensor.pose.compose(&sensor.camera_offset);
    let t_wc1 = xi1.sensor.pose.compose(&xi1.sensor.camera_offset);
    
    for (i, lm) in xi.camera_landmarks.iter().enumerate() {
        let p_world = t_wc.act(&lm.p);
        xi1.camera_landmarks[i].p = t_wc1.inverse().act(&p_world);
    }
    
    xi1
}
