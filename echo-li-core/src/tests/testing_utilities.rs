use nalgebra::{DVector, Vector3, Vector6, SVector};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use echo_lie::{SO3, SE3, SOT3, base::LieGroup};
use crate::mathematical::vio_state::{VIOState, VIOSensorState, Landmark};
use crate::mathematical::vio_group::{VIOGroup, VIOAlgebra};
use crate::ImuBiasGroup;

pub fn log_norm(x: &VIOGroup) -> f64 {
    let mut sum_sq = x.beta.norm_squared();
    sum_sq += x.a.log().norm_squared();
    sum_sq += x.w.norm_squared();
    sum_sq += x.b.log().norm_squared();
    for qi in &x.q {
        sum_sq += qi.log().norm_squared();
    }
    sum_sq.sqrt()
}

pub fn state_distance(xi1: &VIOState, xi2: &VIOState) -> f64 {
    let mut sum_sq = (xi1.sensor.input_bias - xi2.sensor.input_bias).norm_squared();
    sum_sq += xi2.sensor.pose.inverse().compose(&xi1.sensor.pose).log().norm_squared();
    sum_sq += (xi1.sensor.velocity - xi2.sensor.velocity).norm_squared();
    sum_sq += xi2.sensor.camera_offset.inverse().compose(&xi1.sensor.camera_offset).log().norm_squared();
    
    for (l1, l2) in xi1.camera_landmarks.iter().zip(xi2.camera_landmarks.iter()) {
        sum_sq += (l1.p - l2.p).norm_squared();
    }
    sum_sq.sqrt()
}

pub fn random_state_element<R: Rng>(n_landmarks: usize, rng: &mut R) -> VIOState {
    let mut landmarks = Vec::with_capacity(n_landmarks);
    for i in 0..n_landmarks {
        landmarks.push(Landmark {
            p: Vector3::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(1.0..10.0)),
            id: i as u64,
        });
    }

    VIOState {
        sensor: VIOSensorState {
            input_bias: Vector6::new(rng.random(), rng.random(), rng.random(), rng.random(), rng.random(), rng.random()),
            pose: SE3::new(SO3::exp(&Vector3::new(rng.random(), rng.random(), rng.random())), Vector3::new(rng.random(), rng.random(), rng.random())),
            velocity: Vector3::new(rng.random(), rng.random(), rng.random()),
            camera_offset: SE3::identity(),
        },
        camera_landmarks: landmarks,
    }
}

pub fn reasonable_sensor_state<R: Rng>(rng: &mut R) -> VIOSensorState {
    let normal = Normal::new(0.0, 1.0).unwrap();
    VIOSensorState {
        input_bias: Vector6::from_fn(|_, _| normal.sample(rng) * 0.1),
        pose: SE3::new(SO3::exp(&(Vector3::from_fn(|_, _| normal.sample(rng)) * 0.2)), Vector3::from_fn(|_, _| normal.sample(rng)) * 0.5),
        velocity: Vector3::from_fn(|_, _| normal.sample(rng)) * 0.3,
        camera_offset: SE3::new(SO3::identity(), Vector3::new(0.1, 0.0, 0.0)),
    }
}

pub fn reasonable_state_element<R: Rng>(n_landmarks: usize, rng: &mut R) -> VIOState {
    let normal = Normal::new(0.0, 0.5).unwrap();
    let mut landmarks = Vec::with_capacity(n_landmarks);
    for i in 0..n_landmarks {
        landmarks.push(Landmark {
            p: Vector3::new(normal.sample(rng), normal.sample(rng), 5.0 + normal.sample(rng)),
            id: i as u64,
        });
    }

    VIOState {
        sensor: reasonable_sensor_state(rng),
        camera_landmarks: landmarks,
    }
}

pub fn random_group_element<R: Rng>(n_landmarks: usize, rng: &mut R) -> VIOGroup {
    let mut q = Vec::with_capacity(n_landmarks);
    let mut id = Vec::with_capacity(n_landmarks);
    for i in 0..n_landmarks {
        q.push(SOT3::new(SO3::exp(&Vector3::new(rng.random(), rng.random(), rng.random())), rng.random_range(0.5..2.0)));
        id.push(i as u64);
    }

    VIOGroup {
        beta: Vector6::new(rng.random(), rng.random(), rng.random(), rng.random(), rng.random(), rng.random()),
        a: SE3::new(SO3::exp(&Vector3::new(rng.random(), rng.random(), rng.random())), Vector3::new(rng.random(), rng.random(), rng.random())),
        w: Vector3::new(rng.random(), rng.random(), rng.random()),
        b: SE3::new(SO3::exp(&Vector3::new(rng.random(), rng.random(), rng.random())), Vector3::new(rng.random(), rng.random(), rng.random())),
        q,
        id,
        imu_bias_group: ImuBiasGroup::Additive,
    }
}

pub fn reasonable_group_element<R: Rng>(n_landmarks: usize, rng: &mut R) -> VIOGroup {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut q = Vec::with_capacity(n_landmarks);
    let mut id = Vec::with_capacity(n_landmarks);
    for i in 0..n_landmarks {
        q.push(SOT3::new(SO3::exp(&(Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1)), (normal.sample(rng) * 0.1).exp()));
        id.push(i as u64);
    }

    VIOGroup {
        beta: Vector6::from_fn(|_, _| normal.sample(rng) * 0.1),
        a: SE3::new(SO3::exp(&(Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1)), Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1),
        w: Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1,
        b: SE3::new(SO3::exp(&(Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1)), Vector3::from_fn(|_, _| normal.sample(rng)) * 0.1),
        q,
        id,
        imu_bias_group: ImuBiasGroup::Additive,
    }
}

pub fn random_velocity_element<R: Rng>(rng: &mut R) -> crate::mathematical::imu_velocity::IMUVelocity {
    let normal = Normal::new(0.0, 1.0).unwrap();
    crate::mathematical::imu_velocity::IMUVelocity::new(
        0.0,
        Vector3::from_fn(|_, _| normal.sample(rng)) * 0.5,
        Vector3::new(0.0, 0.0, 9.81) + Vector3::from_fn(|_, _| normal.sample(rng)) * 0.5
    )
}

pub fn numerical_jacobian<F>(f: F, x0: &DVector<f64>, eps: f64) -> nalgebra::DMatrix<f64>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let y0 = f(x0);
    let rows = y0.len();
    let cols = x0.len();
    let mut jac = nalgebra::DMatrix::zeros(rows, cols);

    for j in 0..cols {
        let mut x_plus = x0.clone();
        x_plus[j] += eps;
        let y_plus = f(&x_plus);

        let mut x_minus = x0.clone();
        x_minus[j] -= eps;
        let y_minus = f(&x_minus);

        jac.set_column(j, &((y_plus - y_minus) / (2.0 * eps)));
    }
    jac
}
