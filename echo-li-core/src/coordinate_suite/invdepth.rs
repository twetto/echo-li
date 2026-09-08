use echo_lie::{SE3, SO3, SOT3};
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x2, RowVector3, Vector2, Vector3};

use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::coordinate_suite::{
    base_skew, e3_project_sphere, e3_project_sphere_diff, e3_project_sphere_inv,
    e3_project_sphere_inv_diff,
};
use crate::mathematical::camera::CameraModel;
use crate::mathematical::eqf_matrices::{EqFCoordinateSuite, RiccatiPropagationBlocks};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_group::{VIOAlgebra, VIOGroup, state_group_action};
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState};

// ===========================================================================
// Stereographic sphere chart
// Port of: sphereChart_stereo in VIOState.cpp
// ===========================================================================

/// Stereographic chart on S^2 centered at pole.
pub fn sphere_chart_stereo(eta: &Vector3<f64>, pole: &Vector3<f64>) -> Vector2<f64> {
    let e3 = Vector3::new(0.0, 0.0, 1.0);
    let r = SO3::from_vectors(&(-pole), &e3);
    let eta_rotated = r.act(eta);
    e3_project_sphere(&eta_rotated)
}

/// Inverse stereographic chart.
pub fn sphere_chart_stereo_inv(y: &Vector2<f64>, pole: &Vector3<f64>) -> Vector3<f64> {
    let e3 = Vector3::new(0.0, 0.0, 1.0);
    let eta_rotated = e3_project_sphere_inv(y);
    let r = SO3::from_vectors(&(-pole), &e3);
    r.inverse().act(&eta_rotated)
}

/// Jacobian of stereographic chart at the pole.
pub fn sphere_chart_stereo_diff0(pole: &Vector3<f64>) -> Matrix2x3<f64> {
    let e3 = Vector3::new(0.0, 0.0, 1.0);
    let r = SO3::from_vectors(&(-pole), &e3);
    let eta_rotated = r.act(pole);
    e3_project_sphere_diff(&eta_rotated) * r.as_matrix()
}

/// Jacobian of inverse stereographic chart at zero.
pub fn sphere_chart_stereo_inv_diff0(pole: &Vector3<f64>) -> Matrix3x2<f64> {
    let e3 = Vector3::new(0.0, 0.0, 1.0);
    let r = SO3::from_vectors(&(-pole), &e3);
    let zero = Vector2::zeros();
    r.inverse().as_matrix() * e3_project_sphere_inv_diff(&zero)
}

// ===========================================================================
// Coordinate change: Euclidean <-> InvDepth (landmark slot)
// ===========================================================================

/// 3x3 Jacobian of Euclidean-to-InvDepth coordinate change at q0.
pub fn conv_euc2ind(q0: &Vector3<f64>) -> Matrix3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let mut m = Matrix3::zeros();
    // d(bearing)/dp = rho0 * sphere_chart_stereo_diff0(y0) * (I - y0*y0^T)
    let diff0 = sphere_chart_stereo_diff0(&y0);
    let proj = Matrix3::identity() - y0 * y0.transpose();
    let top = rho0 * diff0 * proj;
    m.fixed_view_mut::<2, 3>(0, 0).copy_from(&top);
    // d(inv_depth)/dp = -rho0^2 * y0^T
    m.row_mut(2).copy_from(&(-rho0 * rho0 * y0).transpose());
    m
}

/// 3x3 Jacobian of InvDepth-to-Euclidean coordinate change at q0.
pub fn conv_ind2euc(q0: &Vector3<f64>) -> Matrix3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let mut m = Matrix3::zeros();
    // dp/d(bearing) = (1/rho0) * sphere_chart_stereo_inv_diff0(y0)
    let inv_diff0 = sphere_chart_stereo_inv_diff0(&y0);
    m.fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&(inv_diff0 / rho0));
    // dp/d(inv_depth) = -y0 / rho0^2
    m.column_mut(2).copy_from(&(-y0 / (rho0 * rho0)));
    m
}

// ===========================================================================
// InvDepth point chart
// ===========================================================================

/// InvDepth chart for a single landmark: eps = [bearing_stereo(2), delta_invdepth(1)].
pub fn point_chart_invdepth(q: &Vector3<f64>, q0: &Vector3<f64>) -> Vector3<f64> {
    let rho = 1.0 / q.norm();
    let rho0 = 1.0 / q0.norm();
    let y = q * rho;
    let y0 = q0 * rho0;
    let bearing = sphere_chart_stereo(&y, &y0);
    Vector3::new(bearing[0], bearing[1], rho - rho0)
}

/// Inverse InvDepth chart for a single landmark.
pub fn point_chart_invdepth_inv(eps: &Vector3<f64>, q0: &Vector3<f64>) -> Vector3<f64> {
    let rho0 = 1.0 / q0.norm();
    let y0 = q0 * rho0;
    let y = sphere_chart_stereo_inv(&Vector2::new(eps[0], eps[1]), &y0);
    let rho = eps[2] + rho0;
    let rho_clamped = if rho <= 0.0 { 1e-6 } else { rho };
    y / rho_clamped
}

// ===========================================================================
// InvDepth state chart
// ===========================================================================

pub struct InvDepthSuite;

impl InvDepthSuite {
    pub fn new() -> Self {
        Self
    }
}

impl EqFCoordinateSuite for InvDepthSuite {
    fn state_chart(&self, xi: &VIOState, xi0: &VIOState) -> DVector<f64> {
        let n = xi.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        let mut eps = DVector::<f64>::zeros(s + 3 * n);

        // Sensor state (identical to Euclidean)
        eps.fixed_rows_mut::<6>(0)
            .copy_from(&(xi.sensor.input_bias - xi0.sensor.input_bias));
        let pose_diff = xi0.sensor.pose.inverse().compose(&xi.sensor.pose);
        eps.fixed_rows_mut::<6>(6).copy_from(&pose_diff.log());
        eps.fixed_rows_mut::<3>(12)
            .copy_from(&(xi.sensor.velocity - xi0.sensor.velocity));
        let offset_diff = xi0
            .sensor
            .camera_offset
            .inverse()
            .compose(&xi.sensor.camera_offset);
        eps.fixed_rows_mut::<6>(15).copy_from(&offset_diff.log());

        // Landmarks (inverse-depth)
        for i in 0..n {
            let pt_eps =
                point_chart_invdepth(&xi.camera_landmarks[i].p, &xi0.camera_landmarks[i].p);
            eps.fixed_rows_mut::<3>(s + 3 * i).copy_from(&pt_eps);
        }
        eps
    }

    fn state_chart_inv(&self, eps: &DVector<f64>, xi0: &VIOState) -> VIOState {
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();

        // Sensor state (identical to Euclidean)
        let mut xi = xi0.clone();
        xi.sensor.input_bias = xi0.sensor.input_bias + eps.fixed_rows::<6>(0);
        xi.sensor.pose = xi0
            .sensor
            .pose
            .compose(&SE3::exp(&eps.fixed_rows::<6>(6).into_owned()));
        xi.sensor.velocity = xi0.sensor.velocity + eps.fixed_rows::<3>(12);
        xi.sensor.camera_offset = xi0
            .sensor
            .camera_offset
            .compose(&SE3::exp(&eps.fixed_rows::<6>(15).into_owned()));

        // Landmarks (inverse-depth)
        xi.camera_landmarks = Vec::with_capacity(n);
        for i in 0..n {
            let pt_eps = eps.fixed_rows::<3>(s + 3 * i).into_owned();
            let p = point_chart_invdepth_inv(&pt_eps, &xi0.camera_landmarks[i].p);
            xi.camera_landmarks.push(Landmark {
                p,
                id: xi0.camera_landmarks[i].id,
            });
        }
        xi
    }

    fn state_matrix_a(&self, x: &VIOGroup, xi0: &VIOState, imu_vel: &IMUVelocity) -> DMatrix<f64> {
        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;

        // Start from Euclidean A matrix
        let mut a0t = EuclideanSuite.state_matrix_a(x, xi0, imu_vel);

        // Transform landmark blocks by coordinate change
        let xi_hat = state_group_action(x, xi0);
        let v_est = imu_vel.gyr - xi_hat.sensor.gyro_bias();
        let mut u_i = nalgebra::SVector::<f64, 6>::zeros();
        u_i.fixed_rows_mut::<3>(0).copy_from(&v_est);
        u_i.fixed_rows_mut::<3>(3)
            .copy_from(&xi_hat.sensor.velocity);

        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let r_ahat = x.a.rotation.as_matrix();

        let ad_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        let ad_a = x.a.adjoint();
        let transformed = ad_tc_inv * ad_a * u_i;
        let common_term = x.b.inverse().adjoint() * SE3::adjoint_algebra(&transformed);

        let u_c_full = xi_hat.sensor.camera_offset.inverse().adjoint() * u_i;
        let v_c = u_c_full.fixed_rows::<3>(3).into_owned();

        for i in 0..n {
            let q0 = xi0.camera_landmarks[i].p;
            let qi = &x.q[i];
            let qhat_i = qi.rotation.as_matrix() * qi.scale;
            let m_e2i = conv_euc2ind(&q0);
            let m_i2e = conv_ind2euc(&q0);
            let qhat_p = xi_hat.camera_landmarks[i].p;

            if q0.norm() < 1e-10
                || qhat_p.norm() < 1e-10
                || !qi.scale.is_finite()
                || qi.scale.abs() < 1e-12
            {
                continue; // degenerate landmark, leave as zero block
            }

            // Bias -> Landmarks: transform the Euclidean 3x6 bias block.
            // This preserves semi-direct's Ad_B^{-1} bias-column transform.
            let block_bias = m_e2i * a0t.fixed_view::<3, 6>(s + 3 * i, 0).into_owned();
            a0t.fixed_view_mut::<3, 6>(s + 3 * i, 0)
                .copy_from(&block_bias);

            // Velocity -> Landmarks: M_e2i @ (-Qhat @ R_IC^T @ R_A^T)
            let block_vel = m_e2i * (-qhat_i * r_ic.transpose() * r_ahat.transpose());
            a0t.fixed_view_mut::<3, 3>(s + 3 * i, 12)
                .copy_from(&block_vel);

            // Camera Offset -> Landmarks: M_e2i @ temp @ common_term
            let mut temp = nalgebra::SMatrix::<f64, 3, 6>::zeros();
            temp.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(base_skew(&q0) * qi.rotation.as_matrix()));
            temp.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-qhat_i));
            let block_offset = m_e2i * temp * common_term;
            a0t.fixed_view_mut::<3, 6>(s + 3 * i, 15)
                .copy_from(&block_offset);

            // Landmark -> Landmark: M_e2i @ A_qi_euc @ M_i2e
            let qq = qhat_p.norm_squared();
            let inner = base_skew(&qhat_p) * base_skew(&v_c) - (v_c * qhat_p.transpose() * 2.0)
                + qhat_p * v_c.transpose();
            let qhat_inv = qi.rotation.as_matrix().transpose() / qi.scale;
            let a_qi_euc = -(qhat_i * inner * qhat_inv) / qq;
            let block_lm = m_e2i * a_qi_euc * m_i2e;
            a0t.fixed_view_mut::<3, 3>(s + 3 * i, s + 3 * i)
                .copy_from(&block_lm);
        }

        a0t
    }

    fn input_matrix_b(&self, x: &VIOGroup, xi0: &VIOState) -> DMatrix<f64> {
        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;

        // Start from Euclidean B matrix
        let mut bt = EuclideanSuite.input_matrix_b(x, xi0);

        // Transform landmark blocks by coordinate change
        let xi_hat = state_group_action(x, xi0);
        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let x_ic = xi_hat.sensor.camera_offset.translation;
        let term_x_ic = r_ic.transpose() * base_skew(&x_ic);

        for i in 0..n {
            let q0 = xi0.camera_landmarks[i].p;
            let qi = &x.q[i];
            if q0.norm() < 1e-10 || !qi.scale.is_finite() || qi.scale.abs() < 1e-12 {
                continue; // leave as Euclidean B block (from base call)
            }
            let qhat_i = qi.rotation.as_matrix() * qi.scale;
            let qhat_p = xi_hat.camera_landmarks[i].p;
            let m_e2i = conv_euc2ind(&q0);

            let inner = base_skew(&qhat_p) * r_ic.transpose() + term_x_ic;
            let block = m_e2i * qhat_i * inner;
            bt.fixed_view_mut::<3, 3>(s + 3 * i, 0).copy_from(&block);
        }

        bt
    }

    fn propagation_blocks(
        &self,
        x: &VIOGroup,
        xi0: &VIOState,
        imu_vel: &IMUVelocity,
    ) -> RiccatiPropagationBlocks {
        let RiccatiPropagationBlocks {
            a_ss,
            mut a_lm_s,
            mut a_lm_lm,
            b_s,
            mut b_lm,
        } = EuclideanSuite.propagation_blocks(x, xi0, imu_vel);

        let n = xi0.camera_landmarks.len();
        let xi_hat = state_group_action(x, xi0);
        let v_est = imu_vel.gyr - xi_hat.sensor.gyro_bias();
        let mut u_i = nalgebra::SVector::<f64, 6>::zeros();
        u_i.fixed_rows_mut::<3>(0).copy_from(&v_est);
        u_i.fixed_rows_mut::<3>(3)
            .copy_from(&xi_hat.sensor.velocity);

        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let r_ahat = x.a.rotation.as_matrix();

        let ad_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        let ad_a = x.a.adjoint();
        let transformed = ad_tc_inv * ad_a * u_i;
        let common_term = x.b.inverse().adjoint() * SE3::adjoint_algebra(&transformed);

        let u_c_full = xi_hat.sensor.camera_offset.inverse().adjoint() * u_i;
        let v_c = u_c_full.fixed_rows::<3>(3).into_owned();

        let x_ic = xi_hat.sensor.camera_offset.translation;
        let term_x_ic = r_ic.transpose() * base_skew(&x_ic);

        for i in 0..n {
            let row = 3 * i;
            let q0 = xi0.camera_landmarks[i].p;
            let qi = &x.q[i];

            if q0.norm() < 1e-10 || !qi.scale.is_finite() || qi.scale.abs() < 1e-12 {
                continue;
            }

            let qhat_i = qi.rotation.as_matrix() * qi.scale;
            let qhat_p = xi_hat.camera_landmarks[i].p;
            let m_e2i = conv_euc2ind(&q0);

            let inner_b = base_skew(&qhat_p) * r_ic.transpose() + term_x_ic;
            let block_b_ind = m_e2i * qhat_i * inner_b;
            b_lm.fixed_view_mut::<3, 3>(row, 0).copy_from(&block_b_ind);

            if qhat_p.norm() < 1e-10 {
                continue;
            }

            let m_i2e = conv_ind2euc(&q0);

            a_lm_s
                .fixed_view_mut::<3, 3>(row, 0)
                .copy_from(&(-block_b_ind));
            a_lm_s.fixed_view_mut::<3, 3>(row, 3).fill(0.0);

            let block_vel = m_e2i * (-qhat_i * r_ic.transpose() * r_ahat.transpose());
            a_lm_s.fixed_view_mut::<3, 3>(row, 12).copy_from(&block_vel);

            let mut temp = nalgebra::SMatrix::<f64, 3, 6>::zeros();
            temp.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(base_skew(&q0) * qi.rotation.as_matrix()));
            temp.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-qhat_i));
            let block_offset = m_e2i * temp * common_term;
            a_lm_s
                .fixed_view_mut::<3, 6>(row, 15)
                .copy_from(&block_offset);

            let qq = qhat_p.norm_squared();
            let inner = base_skew(&qhat_p) * base_skew(&v_c) - (v_c * qhat_p.transpose() * 2.0)
                + qhat_p * v_c.transpose();
            let qhat_inv = qi.rotation.as_matrix().transpose() / qi.scale;
            let a_qi_euc = -(qhat_i * inner * qhat_inv) / qq;
            a_lm_lm[i] = m_e2i * a_qi_euc * m_i2e;
        }

        RiccatiPropagationBlocks {
            a_ss,
            a_lm_s,
            a_lm_lm,
            b_s,
            b_lm,
        }
    }

    fn output_matrix_ci_star(
        &self,
        q0: &Vector3<f64>,
        q_hat: &SOT3,
        cam: &dyn CameraModel,
        y: &Vector2<f64>,
    ) -> Matrix2x3<f64> {
        // C*_invdepth = C*_euclid @ ind2euc
        let m_i2e = conv_ind2euc(q0);
        EuclideanSuite.output_matrix_ci_star(q0, q_hat, cam, y) * m_i2e
    }

    fn output_range_row(&self, q0: &Vector3<f64>) -> RowVector3<f64> {
        // C_ℓ_invdepth = C_ℓ_euclid @ ind2euc  (= [0, 0, 1/ρ0])
        (-q0.transpose() / q0.norm_squared()) * conv_ind2euc(q0)
    }

    fn lift_innovation(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOAlgebra {
        // Sensor part is identical to Euclidean.
        // Landmark part converts InvDepth perturbation to Euclidean first.
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();

        let mut delta = VIOAlgebra {
            u_beta: total_innovation.fixed_rows::<6>(0).into_owned(),
            u_a: total_innovation.fixed_rows::<6>(6).into_owned(),
            u_b: nalgebra::Vector6::zeros(),
            u_w: nalgebra::Vector3::zeros(),
            w: Vec::with_capacity(n),
            id: Vec::with_capacity(n),
        };

        let gamma_v = total_innovation.fixed_rows::<3>(12).into_owned();
        let omega_a = delta.u_a.fixed_rows::<3>(0).into_owned();
        delta.u_w = -gamma_v - base_skew(&omega_a) * xi0.sensor.velocity;

        let adj_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        delta.u_b = total_innovation.fixed_rows::<6>(15).into_owned() + adj_tc_inv * delta.u_a;

        // Point landmarks: convert InvDepth -> Euclidean, then SOT(3) lift
        for i in 0..n {
            let qi0 = xi0.camera_landmarks[i].p;
            let ind2euc = conv_ind2euc(&qi0);

            let gamma_qi_invdepth = total_innovation.fixed_rows::<3>(s + 3 * i).into_owned();
            let gamma_qi_euclid = ind2euc * gamma_qi_invdepth;

            let qq = qi0.norm_squared();
            let mut wi = nalgebra::Vector4::zeros();
            wi.fixed_rows_mut::<3>(0)
                .copy_from(&(-qi0.cross(&gamma_qi_euclid) / qq));
            wi[3] = -(qi0.dot(&gamma_qi_euclid)) / qq;

            delta.w.push(wi);
            delta.id.push(xi0.camera_landmarks[i].id);
        }

        delta
    }

    fn lift_innovation_discrete(
        &self,
        total_innovation: &DVector<f64>,
        xi0: &VIOState,
    ) -> VIOGroup {
        let xi = self.state_chart_inv(total_innovation, xi0);
        let inn_euc = EuclideanSuite.state_chart(&xi, xi0);
        EuclideanSuite.lift_innovation_discrete(&inn_euc, xi0)
    }
}
