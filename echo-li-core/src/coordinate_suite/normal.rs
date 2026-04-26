use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3, Matrix2x3, Matrix3x2};
use echo_lie::{SO3, SE3, SE23, SOT3, base::LieGroup};

use crate::mathematical::vio_state::{VIOState, VIOSensorState, Landmark};
use crate::mathematical::vio_group::{VIOGroup, VIOAlgebra};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::camera::CameraModel;
use crate::coordinate_suite::base_skew;
use crate::coordinate_suite::euclid::EuclideanSuite;

// ===========================================================================
// Sphere chart: normal (SO(3)-exp based)
// Port of: sphereChart_normal in VIOState.cpp
// ===========================================================================

const E3: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);

/// Normal chart on S^2: maps pole -> 0 using SO(3) exp coordinates about e3.
pub fn sphere_chart_normal(eta: &Vector3<f64>, pole: &Vector3<f64>) -> Vector2<f64> {
    let r = SO3::from_vectors(pole, &E3);
    let y = r.act(eta);
    let sin_th = base_skew(&y) * E3;
    let sin_th_norm = sin_th.norm();
    let cos_th = y.dot(&E3);
    let th = sin_th_norm.atan2(cos_th);
    let omega = if th.abs() < 1e-8 {
        sin_th
    } else {
        sin_th * (th / sin_th_norm)
    };
    Vector2::new(omega[0], omega[1])
}

/// Inverse normal chart on S^2.
pub fn sphere_chart_normal_inv(eps: &Vector2<f64>, pole: &Vector3<f64>) -> Vector3<f64> {
    let omega = Vector3::new(eps[0], eps[1], 0.0);
    let y = SO3::exp(&(-omega)).act(&E3);
    let r = SO3::from_vectors(pole, &E3);
    r.inverse().act(&y)
}

/// Jacobian of the normal chart at the pole.
pub fn sphere_chart_normal_diff0(pole: &Vector3<f64>) -> Matrix2x3<f64> {
    let r = SO3::from_vectors(pole, &E3);
    let r_mat = r.as_matrix();
    // diff = [[0, 1, 0], [-1, 0, 0]] @ R
    Matrix2x3::new(
        r_mat[(1, 0)], r_mat[(1, 1)], r_mat[(1, 2)],
        -r_mat[(0, 0)], -r_mat[(0, 1)], -r_mat[(0, 2)],
    )
}

/// Jacobian of the inverse normal chart at 0.
pub fn sphere_chart_normal_inv_diff0(pole: &Vector3<f64>) -> Matrix3x2<f64> {
    let r = SO3::from_vectors(pole, &E3);
    let r_inv = r.inverse().as_matrix();
    // diff = R^{-1} @ [[0, -1], [1, 0], [0, 0]]
    Matrix3x2::new(
        r_inv[(0, 1)], -r_inv[(0, 0)],
        r_inv[(1, 1)], -r_inv[(1, 0)],
        r_inv[(2, 1)], -r_inv[(2, 0)],
    )
}

// ===========================================================================
// Point chart: normal (stereo_normal bearing + log depth)
// ===========================================================================

/// Normal chart for a single landmark: eps = [sphere_normal_bearing(2), log(rho/rho0)].
pub fn point_chart_normal(q: &Vector3<f64>, q0: &Vector3<f64>) -> Vector3<f64> {
    let rho = 1.0 / q.norm();
    let rho0 = 1.0 / q0.norm();
    let y = q * rho;
    let y0 = q0 * rho0;
    let bearing = sphere_chart_normal(&y, &y0);
    Vector3::new(bearing[0], bearing[1], (rho / rho0).ln())
}

/// Inverse normal chart for a single landmark.
pub fn point_chart_normal_inv(eps: &Vector3<f64>, q0: &Vector3<f64>) -> Vector3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let y = sphere_chart_normal_inv(&Vector2::new(eps[0], eps[1]), &y0);
    let rho = rho0 * eps[2].exp();
    y / rho
}

// ===========================================================================
// Coordinate change: Normal <-> Euclidean (landmark slot)
// ===========================================================================

/// 3x3 Jacobian of Euclidean-to-Normal landmark coordinate change at q0.
pub fn conv_euc2normal(q0: &Vector3<f64>) -> Matrix3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let mut m = Matrix3::zeros();
    // d(sphere_normal_bearing)/dp = sphere_chart_normal_diff0(y0) * (1/||q0||) * (I - y0*y0^T)
    let diff0 = sphere_chart_normal_diff0(&y0);
    let proj = Matrix3::identity() - y0 * y0.transpose();
    let top = rho0 * diff0 * proj;
    m.fixed_view_mut::<2, 3>(0, 0).copy_from(&top);
    // d(log(rho/rho0))/dp = -(1/||q0||) * y0^T
    m.row_mut(2).copy_from(&(-rho0 * y0).transpose());
    m
}

/// 3x3 Jacobian of Normal-to-Euclidean landmark coordinate change at q0.
pub fn conv_normal2euc(q0: &Vector3<f64>) -> Matrix3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let mut m = Matrix3::zeros();
    // dp/d(eps[0:2]) = (1/rho0) * sphere_chart_normal_inv_diff0(y0)
    let inv_diff0 = sphere_chart_normal_inv_diff0(&y0);
    m.fixed_view_mut::<3, 2>(0, 0).copy_from(&(inv_diff0 / rho0));
    // dp/d(eps[2]) = -p0
    m.column_mut(2).copy_from(&(-q0));
    m
}

// ===========================================================================
// Sensor chart: normal (SE₂(3)-based pose+velocity, conjugated camera offset)
// ===========================================================================

/// Sensor chart (normal): VIOSensorState -> eps(21).
fn sensor_chart_normal(xi: &VIOSensorState, xi0: &VIOSensorState) -> nalgebra::SVector<f64, 21> {
    let a = xi0.pose.inverse().compose(&xi.pose);
    let v_xi0_world = xi0.pose.rotation.act(&xi0.velocity);
    let v_xi_world = xi.pose.rotation.act(&xi.velocity);
    let v_a = xi0.pose.rotation.inverse().act(&(v_xi_world - v_xi0_world));
    let b = xi0.camera_offset.inverse().compose(&a).compose(&xi.camera_offset);

    let mut eps = nalgebra::SVector::<f64, 21>::zeros();
    eps.fixed_rows_mut::<6>(0).copy_from(&(xi.input_bias - xi0.input_bias));
    // SE₂(3) log of (A.R, A.x, v_A)
    let se23 = SE23::new(a.rotation, a.translation, v_a);
    let se23_log = <SE23 as LieGroup>::log(&se23);
    eps.fixed_rows_mut::<9>(6).copy_from(&se23_log);
    eps.fixed_rows_mut::<6>(15).copy_from(&b.log());
    eps
}

/// Inverse sensor chart (normal): eps(21) -> VIOSensorState.
fn sensor_chart_inv_normal(eps: &nalgebra::SVector<f64, 21>, xi0: &VIOSensorState) -> VIOSensorState {
    let se23_u = eps.fixed_rows::<9>(6).into_owned();
    let x = <SE23 as LieGroup>::exp(&se23_u);
    let a = SE3::new(x.rotation, x.position);
    let v_a = x.velocity;
    let b = SE3::exp(&eps.fixed_rows::<6>(15).into_owned());

    let mut xi = VIOSensorState {
        input_bias: xi0.input_bias + eps.fixed_rows::<6>(0),
        pose: xi0.pose.compose(&a),
        velocity: Vector3::zeros(),
        camera_offset: a.inverse().compose(&xi0.camera_offset).compose(&b),
    };
    let v_xi0_world = xi0.pose.rotation.act(&xi0.velocity);
    xi.velocity = xi.pose.rotation.inverse().act(&(v_xi0_world + xi0.pose.rotation.act(&v_a)));
    xi
}

// ===========================================================================
// Coordinate differential M: normal <- euclid (analytical, sensor block only)
// ===========================================================================

/// Build only the 21x21 sensor block of M (eps_normal_sensor = M_s @ eps_euc_sensor at eps=0).
/// The full M is block-diagonal: diag(M_s, conv_euc2normal(q0_1), ..., conv_euc2normal(q0_n)),
/// so we never materialize the full matrix or invert it as a whole.
fn build_m_sensor(xi0: &VIOState) -> nalgebra::SMatrix<f64, 21, 21> {
    let mut m = nalgebra::SMatrix::<f64, 21, 21>::identity();

    // v_A(12:15) <- theta_pose(6:9): -skew(vel0)
    let vel0 = xi0.sensor.velocity;
    m[(12, 7)] = vel0[2];
    m[(12, 8)] = -vel0[1];
    m[(13, 6)] = -vel0[2];
    m[(13, 8)] = vel0[0];
    m[(14, 6)] = vel0[1];
    m[(14, 7)] = -vel0[0];

    // SE3.log(B) (15:21) <- (theta_pose, x_pose) (6:12): Ad_{Tc0^{-1}}
    let ad_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
    for i in 0..6 {
        for j in 0..6 {
            m[(15 + i, 6 + j)] = ad_tc_inv[(i, j)];
        }
    }

    m
}

pub struct NormalSuite;

impl NormalSuite {
    pub fn new() -> Self {
        Self
    }
}

impl EqFCoordinateSuite for NormalSuite {
    fn state_chart(&self, xi: &VIOState, xi0: &VIOState) -> DVector<f64> {
        let n = xi.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        let mut eps = DVector::<f64>::zeros(s + 3 * n);

        let sensor_eps = sensor_chart_normal(&xi.sensor, &xi0.sensor);
        eps.rows_mut(0, s).copy_from(&sensor_eps);

        for i in 0..n {
            let pt_eps = point_chart_normal(&xi.camera_landmarks[i].p, &xi0.camera_landmarks[i].p);
            eps.fixed_rows_mut::<3>(s + 3 * i).copy_from(&pt_eps);
        }
        eps
    }

    fn state_chart_inv(&self, eps: &DVector<f64>, xi0: &VIOState) -> VIOState {
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();

        let sensor_eps: nalgebra::SVector<f64, 21> = eps.fixed_rows::<21>(0).into();
        let sensor = sensor_chart_inv_normal(&sensor_eps, &xi0.sensor);

        let mut xi = VIOState {
            sensor,
            camera_landmarks: Vec::with_capacity(n),
        };
        for i in 0..n {
            let pt_eps = eps.fixed_rows::<3>(s + 3 * i).into_owned();
            let p = point_chart_normal_inv(&pt_eps, &xi0.camera_landmarks[i].p);
            xi.camera_landmarks.push(Landmark {
                p,
                id: xi0.camera_landmarks[i].id,
            });
        }
        xi
    }

    fn state_matrix_a(&self, x: &VIOGroup, xi0: &VIOState, imu_vel: &IMUVelocity) -> DMatrix<f64> {
        // A_normal = M @ A_euc @ M^{-1}, exploiting block-diagonal M.
        // A_euc has only sensor-sensor, landmark<-sensor, and landmark-self (diagonal) non-zero blocks,
        // so the conjugation reduces to a 21x21 sensor block transform plus per-landmark 3x21 / 3x3 transforms.
        let a_euc = EuclideanSuite.state_matrix_a(x, xi0, imu_vel);
        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        let dim = xi0.dim();

        let m_s = build_m_sensor(xi0);
        let m_s_inv = m_s.try_inverse().expect("M_s must be invertible");

        let mut a_normal = DMatrix::<f64>::zeros(dim, dim);

        // Sensor-sensor block: M_s * A_ss * M_s^{-1}
        let a_ss = a_euc.fixed_view::<21, 21>(0, 0).into_owned();
        let new_ss = m_s * a_ss * m_s_inv;
        a_normal.fixed_view_mut::<21, 21>(0, 0).copy_from(&new_ss);

        for i in 0..n {
            let q0 = xi0.camera_landmarks[i].p;
            let m_e2n = conv_euc2normal(&q0);
            let m_n2e = conv_normal2euc(&q0);

            // Landmark<-sensor row: M_e2n * A[lm_i, sensor] * M_s^{-1}
            let a_li_s = a_euc.fixed_view::<3, 21>(s + 3 * i, 0).into_owned();
            let new_li_s = m_e2n * a_li_s * m_s_inv;
            a_normal.fixed_view_mut::<3, 21>(s + 3 * i, 0).copy_from(&new_li_s);

            // Landmark<-self diagonal: M_e2n * A[lm_i, lm_i] * M_n2e
            let a_li_li = a_euc.fixed_view::<3, 3>(s + 3 * i, s + 3 * i).into_owned();
            let new_li_li = m_e2n * a_li_li * m_n2e;
            a_normal.fixed_view_mut::<3, 3>(s + 3 * i, s + 3 * i).copy_from(&new_li_li);
        }

        a_normal
    }

    fn input_matrix_b(&self, x: &VIOGroup, xi0: &VIOState) -> DMatrix<f64> {
        // B_normal = M @ B_euc, applied block-wise.
        let b_euc = EuclideanSuite.input_matrix_b(x, xi0);
        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        let dim = xi0.dim();

        let m_s = build_m_sensor(xi0);

        let mut b_normal = DMatrix::<f64>::zeros(dim, 12);

        // Sensor block: M_s * B_ss
        let b_ss = b_euc.fixed_view::<21, 12>(0, 0).into_owned();
        let new_ss = m_s * b_ss;
        b_normal.fixed_view_mut::<21, 12>(0, 0).copy_from(&new_ss);

        for i in 0..n {
            let q0 = xi0.camera_landmarks[i].p;
            let m_e2n = conv_euc2normal(&q0);

            let b_li = b_euc.fixed_view::<3, 12>(s + 3 * i, 0).into_owned();
            let new_li = m_e2n * b_li;
            b_normal.fixed_view_mut::<3, 12>(s + 3 * i, 0).copy_from(&new_li);
        }

        b_normal
    }

    fn output_matrix_ci_star(&self, q0: &Vector3<f64>, q_hat: &SOT3, cam: &dyn CameraModel, y: &Vector2<f64>) -> Matrix2x3<f64> {
        // Equivariant C* for Normal chart:
        // Average proj_jac at predicted and observed bearings, then apply
        // the rotation and chart differential.
        // Third column (log-depth) is zero since bearing measurements
        // carry no direct depth information.
        let y0 = q0.normalize();
        let y_hat_bearing = q_hat.rotation.inverse().act(&y0);
        let y_tru_bearing = cam.undistort(y);

        let avg_proj_jac = 0.5 * (cam.projection_jacobian(&y_tru_bearing)
                                 + cam.projection_jacobian(&y_hat_bearing));
        let inv_diff = sphere_chart_normal_inv_diff0(q0);
        let block_2x2 = avg_proj_jac * q_hat.rotation.as_matrix().transpose() * inv_diff;
        let mut c0i = Matrix2x3::zeros();
        c0i.fixed_view_mut::<2, 2>(0, 0).copy_from(&block_2x2);
        c0i
    }

    fn lift_innovation(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOAlgebra {
        // Continuous lift: convert normal-coord innovation to Euclidean coords block-wise,
        // then call the Euclidean lift.
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();
        let dim = xi0.dim();

        let m_s = build_m_sensor(xi0);
        let m_s_inv = m_s.try_inverse().expect("M_s must be invertible");

        let mut inn_euc = DVector::<f64>::zeros(dim);

        // Sensor part: M_s^{-1} * innovation_sensor
        let inn_s = total_innovation.fixed_rows::<21>(0).into_owned();
        let inn_euc_s = m_s_inv * inn_s;
        inn_euc.fixed_rows_mut::<21>(0).copy_from(&inn_euc_s);

        // Landmark parts: conv_normal2euc(q0_i) * innovation_lm_i
        for i in 0..n {
            let q0 = xi0.camera_landmarks[i].p;
            let m_n2e = conv_normal2euc(&q0);
            let inn_lm = total_innovation.fixed_rows::<3>(s + 3 * i).into_owned();
            let inn_euc_lm = m_n2e * inn_lm;
            inn_euc.fixed_rows_mut::<3>(s + 3 * i).copy_from(&inn_euc_lm);
        }

        EuclideanSuite.lift_innovation(&inn_euc, xi0)
    }

    fn lift_innovation_discrete(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOGroup {
        // Discrete lift: chart round-trip (normal -> state -> euclidean -> group)
        let xi = self.state_chart_inv(total_innovation, xi0);
        let inn_euc = EuclideanSuite.state_chart(&xi, xi0);
        EuclideanSuite.lift_innovation_discrete(&inn_euc, xi0)
    }
}
