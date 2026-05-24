use echo_lie::{base::LieGroup, SE23, SE3, SO3, SOT3};
use nalgebra::{Vector3, Vector4, Vector6};

use crate::mathematical::bias_group_ops::BiasGroupOps;
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState, GRAVITY_CONSTANT};
use crate::ImuBiasGroup;

/// Symmetry group element for EqVIO.
#[derive(Debug, Clone)]
pub struct VIOGroup {
    pub beta: Vector6<f64>,
    pub a: SE3,
    pub w: Vector3<f64>,
    pub b: SE3,
    pub q: Vec<SOT3>,
    pub id: Vec<u64>,
    pub imu_bias_group: ImuBiasGroup,
}

impl VIOGroup {
    pub fn identity(ids: &[u64]) -> Self {
        Self::identity_with_bias_group(ids, ImuBiasGroup::Additive)
    }

    pub fn identity_with_bias_group(ids: &[u64], imu_bias_group: ImuBiasGroup) -> Self {
        Self {
            beta: Vector6::zeros(),
            a: SE3::identity(),
            w: Vector3::zeros(),
            b: SE3::identity(),
            q: vec![SOT3::identity(); ids.len()],
            id: ids.to_vec(),
            imu_bias_group,
        }
    }

    pub fn inverse(&self) -> Self {
        let ops = BiasGroupOps::new(self.imu_bias_group);
        Self {
            beta: ops.inverse_beta(self),
            a: self.a.inverse(),
            w: -(self.a.rotation.inverse().act(&self.w)),
            b: self.b.inverse(),
            q: self.q.iter().map(|qi| qi.inverse()).collect(),
            id: self.id.clone(),
            imu_bias_group: self.imu_bias_group,
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        let ops = BiasGroupOps::new(self.imu_bias_group);
        Self {
            beta: ops.compose_beta(self, &other.beta),
            a: self.a.compose(&other.a),
            w: self.w + self.a.rotation.act(&other.w),
            b: self.b.compose(&other.b),
            q: self
                .q
                .iter()
                .zip(other.q.iter())
                .map(|(q1, q2)| q1.compose(q2))
                .collect(),
            id: self.id.clone(),
            imu_bias_group: ops.kind(),
        }
    }

    pub fn with_bias_group(mut self, imu_bias_group: ImuBiasGroup) -> Self {
        self.imu_bias_group = imu_bias_group;
        self
    }
}

/// Lie algebra element of VIOGroup.
pub struct VIOAlgebra {
    pub u_beta: Vector6<f64>,
    pub u_a: Vector6<f64>,
    pub u_b: Vector6<f64>,
    pub u_w: Vector3<f64>,
    pub w: Vec<Vector4<f64>>,
    pub id: Vec<u64>,
}

impl std::ops::Sub for VIOAlgebra {
    type Output = VIOAlgebra;
    fn sub(self, rhs: VIOAlgebra) -> VIOAlgebra {
        VIOAlgebra {
            u_beta: self.u_beta - rhs.u_beta,
            u_a: self.u_a - rhs.u_a,
            u_b: self.u_b - rhs.u_b,
            u_w: self.u_w - rhs.u_w,
            w: self
                .w
                .iter()
                .zip(rhs.w.iter())
                .map(|(a, b)| a - b)
                .collect(),
            id: self.id,
        }
    }
}

impl std::ops::Sub for &VIOAlgebra {
    type Output = VIOAlgebra;
    fn sub(self, rhs: &VIOAlgebra) -> VIOAlgebra {
        VIOAlgebra {
            u_beta: self.u_beta - rhs.u_beta,
            u_a: self.u_a - rhs.u_a,
            u_b: self.u_b - rhs.u_b,
            u_w: self.u_w - rhs.u_w,
            w: self
                .w
                .iter()
                .zip(rhs.w.iter())
                .map(|(a, b)| a - b)
                .collect(),
            id: self.id.clone(),
        }
    }
}

pub fn sensor_state_group_action(x: &VIOGroup, sensor: &VIOSensorState) -> VIOSensorState {
    let input_bias = BiasGroupOps::new(x.imu_bias_group).act_bias(x, &sensor.input_bias);
    VIOSensorState {
        input_bias,
        pose: sensor.pose.compose(&x.a),
        velocity: x.a.rotation.inverse().act(&(sensor.velocity - x.w)),
        camera_offset: x.a.inverse().compose(&sensor.camera_offset).compose(&x.b),
    }
}

pub fn state_group_action(x: &VIOGroup, state: &VIOState) -> VIOState {
    let mut new_state = VIOState {
        sensor: sensor_state_group_action(x, &state.sensor),
        camera_landmarks: Vec::with_capacity(state.camera_landmarks.len()),
    };

    for (qi, lm) in x.q.iter().zip(state.camera_landmarks.iter()) {
        new_state.camera_landmarks.push(Landmark {
            p: qi.inverse().act(&lm.p),
            id: lm.id,
        });
    }

    new_state
}

/// Continuous lift: maps (state, IMU velocity) to VIOAlgebra.
/// Reference: liftVelocity() in Python vio_group.py
pub fn lift_velocity(
    state: &VIOState,
    velocity: &crate::mathematical::imu_velocity::IMUVelocity,
) -> VIOAlgebra {
    let sensor = &state.sensor;
    let v_est_gyr = velocity.gyr - sensor.gyro_bias();
    let v_est_acc = velocity.acc - sensor.accel_bias();

    // Bias lift
    let mut u_beta = Vector6::zeros();
    u_beta
        .fixed_rows_mut::<3>(0)
        .copy_from(&velocity.gyr_bias_vel);
    u_beta
        .fixed_rows_mut::<3>(3)
        .copy_from(&velocity.acc_bias_vel);

    // SE(3) pose velocity: U_A = [omega; v] = [gyr; body_velocity]
    let mut u_a = Vector6::zeros();
    u_a.fixed_rows_mut::<3>(0).copy_from(&v_est_gyr);
    u_a.fixed_rows_mut::<3>(3).copy_from(&sensor.velocity);

    // Camera offset velocity: U_B = Ad_{T_C^{-1}} U_A
    let u_b = sensor.camera_offset.inverse().adjoint() * u_a;

    // R^3 velocity component
    let u_w = -v_est_acc + sensor.gravity_dir() * GRAVITY_CONSTANT;

    // Camera-frame velocity for landmarks
    let u_c = sensor.camera_offset.inverse().adjoint() * u_a;
    let omega_c = u_c.fixed_rows::<3>(0).into_owned();
    let v_c = u_c.fixed_rows::<3>(3).into_owned();

    // Point landmark lifts
    let mut w_vec = Vec::with_capacity(state.camera_landmarks.len());
    let mut id_vec = Vec::with_capacity(state.camera_landmarks.len());
    for lm in &state.camera_landmarks {
        let p = lm.p;
        let pp = p.norm_squared();
        let mut wi = Vector4::zeros();
        if pp > 1e-20 {
            wi.fixed_rows_mut::<3>(0)
                .copy_from(&(omega_c + p.cross(&v_c) / pp));
            wi[3] = p.dot(&v_c) / pp;
        } else {
            wi.fixed_rows_mut::<3>(0).copy_from(&omega_c);
        }
        w_vec.push(wi);
        id_vec.push(lm.id);
    }

    VIOAlgebra {
        u_beta,
        u_a,
        u_b,
        u_w,
        w: w_vec,
        id: id_vec,
    }
}

pub fn vio_exp(lam: &VIOAlgebra) -> VIOGroup {
    vio_exp_with_bias_group(lam, ImuBiasGroup::Additive)
}

pub fn vio_exp_with_bias_group(lam: &VIOAlgebra, imu_bias_group: ImuBiasGroup) -> VIOGroup {
    let mut ext_vel = nalgebra::SVector::<f64, 9>::zeros();
    ext_vel.fixed_rows_mut::<6>(0).copy_from(&lam.u_a);
    ext_vel.fixed_rows_mut::<3>(6).copy_from(&lam.u_w);
    let ext_pose = SE23::exp(&ext_vel);
    let ops = BiasGroupOps::new(imu_bias_group);

    VIOGroup {
        beta: ops.exp_beta(lam),
        a: SE3::new(ext_pose.rotation, ext_pose.position),
        w: ext_pose.velocity,
        b: SE3::exp(&lam.u_b),
        q: lam.w.iter().map(|wi| SOT3::exp(wi)).collect(),
        id: lam.id.clone(),
        imu_bias_group: ops.kind(),
    }
}

pub fn lift_velocity_discrete(
    state: &VIOState,
    velocity: &crate::mathematical::imu_velocity::IMUVelocity,
    dt: f64,
) -> VIOGroup {
    let sensor = &state.sensor;
    let v_est_gyr = velocity.gyr - sensor.gyro_bias();
    let v_est_acc = velocity.acc - sensor.accel_bias();

    let mut beta = Vector6::zeros();
    beta.fixed_rows_mut::<3>(0)
        .copy_from(&(dt * velocity.gyr_bias_vel));
    beta.fixed_rows_mut::<3>(3)
        .copy_from(&(dt * velocity.acc_bias_vel));

    // Pose: discrete integration
    let rot_change = SO3::exp(&(dt * v_est_gyr));

    let x_world = dt * (sensor.pose.rotation.act(&sensor.velocity))
        + 0.5
            * dt
            * dt
            * (sensor.pose.rotation.act(&v_est_acc) + Vector3::new(0.0, 0.0, -GRAVITY_CONSTANT));
    let pose_change_x = sensor.pose.rotation.inverse().act(&x_world);
    let a = SE3::new(rot_change, pose_change_x);

    // Camera offset
    let b = sensor
        .camera_offset
        .inverse()
        .compose(&a)
        .compose(&sensor.camera_offset);

    // Velocity change
    let body_vel_diff = v_est_acc - sensor.gravity_dir() * GRAVITY_CONSTANT;
    let w = sensor.velocity - (sensor.velocity + dt * body_vel_diff);

    // Point landmark discrete lifts
    let camera_pose_change_inv = sensor
        .camera_offset
        .inverse()
        .compose(&a.inverse())
        .compose(&sensor.camera_offset);

    let mut q_vec = Vec::with_capacity(state.camera_landmarks.len());
    let mut id_vec = Vec::with_capacity(state.camera_landmarks.len());
    for lm in &state.camera_landmarks {
        let p0 = lm.p;
        let p1 = camera_pose_change_inv.act(&p0);
        let p1_norm = p1.norm();
        if p1_norm < 1e-12 || p0.norm() < 1e-12 {
            // Degenerate: landmark at camera origin, use identity
            q_vec.push(SOT3::identity());
        } else {
            let rot = SO3::from_vectors(&(p1 / p1_norm), &(p0.normalize()));
            let scale = (p0.norm() / p1_norm).clamp(1e-8, 1e8);
            q_vec.push(SOT3::new(rot, scale));
        }
        id_vec.push(lm.id);
    }

    VIOGroup {
        beta,
        a,
        w,
        b,
        q: q_vec,
        id: id_vec,
        imu_bias_group: ImuBiasGroup::Additive,
    }
}
