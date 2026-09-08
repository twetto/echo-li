use echo_lie::{SE3, SO3, SOT3};
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3, SMatrix, Vector2, Vector3};

use crate::coordinate_suite::base_skew;
use crate::mathematical::bias_group_ops::BiasGroupOps;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::eqf_matrices::{EqFCoordinateSuite, RiccatiPropagationBlocks};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_group::{state_group_action, VIOAlgebra, VIOGroup};
use crate::mathematical::vio_state::{VIOSensorState, VIOState, GRAVITY_CONSTANT};
use crate::ImuBiasGroup;

pub struct EuclideanSuite;

impl EqFCoordinateSuite for EuclideanSuite {
    fn state_chart(&self, xi: &VIOState, xi0: &VIOState) -> DVector<f64> {
        let n = xi.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        let mut eps = DVector::<f64>::zeros(s + 3 * n);

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

        for i in 0..n {
            eps.fixed_rows_mut::<3>(s + 3 * i)
                .copy_from(&(xi.camera_landmarks[i].p - xi0.camera_landmarks[i].p));
        }

        eps
    }

    fn state_chart_inv(&self, eps: &DVector<f64>, xi0: &VIOState) -> VIOState {
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

        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;
        for i in 0..n {
            xi.camera_landmarks[i].p = xi0.camera_landmarks[i].p + eps.fixed_rows::<3>(s + 3 * i);
        }

        xi
    }

    fn state_matrix_a(&self, x: &VIOGroup, xi0: &VIOState, imu_vel: &IMUVelocity) -> DMatrix<f64> {
        let n = xi0.camera_landmarks.len();
        let dim = xi0.dim();
        let mut a0t = DMatrix::<f64>::zeros(dim, dim);
        let s = VIOSensorState::CDIM;

        let bt = self.input_matrix_b(x, xi0);
        for j in 0..6 {
            a0t.set_column(j, &(-bt.column(j)));
        }

        // Pose pos -> vel
        a0t.fixed_view_mut::<3, 3>(9, 12)
            .copy_from(&Matrix3::identity());
        // Vel -> Orientation (Gravity). This vel←att coupling (−g[ĝ]×) is the gravity
        // leak that builds the Σ[att,vel] correlation used to route a vision-corrected
        // velocity back into an attitude correction. It enters ONLY the covariance
        // Riccati (state_matrix_a), NOT the mean lift — so its sign/orientation sets the
        // DIRECTION of the attitude correction without touching mean physics. TRAJDUMP
        // cos_dir≈−0.10 (correction ⊥/slightly-wrong to tilt) is the symptom a wrong
        // sign/orientation here would cause. ECHO_MSC_VELATT_SIGN=-1 flips the sign to
        // test whether the routing is a sign error (gate on TRAJDUMP cos_dir>0 & ATE).
        let velatt_sign: f64 = std::env::var("ECHO_MSC_VELATT_SIGN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        a0t.fixed_view_mut::<3, 3>(12, 6)
            .copy_from(&(velatt_sign * -GRAVITY_CONSTANT * base_skew(&xi0.sensor.gravity_dir())));

        let xi_hat = state_group_action(x, xi0);
        let v_est = imu_vel.gyr - xi_hat.sensor.gyro_bias();
        let mut u_i = nalgebra::SVector::<f64, 6>::zeros();
        u_i.fixed_rows_mut::<3>(0).copy_from(&v_est);
        u_i.fixed_rows_mut::<3>(3)
            .copy_from(&xi_hat.sensor.velocity);

        let ad_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        let ad_a = x.a.adjoint();
        let transformed = ad_tc_inv * ad_a * u_i;
        // Camera Offset block: 15:21
        a0t.fixed_view_mut::<6, 6>(15, 15)
            .copy_from(&SE3::adjoint_algebra(&transformed));

        let common_term = x.b.inverse().adjoint() * SE3::adjoint_algebra(&transformed);
        let u_c_full = xi_hat.sensor.camera_offset.inverse().adjoint() * u_i;
        let v_c = u_c_full.fixed_rows::<3>(3).into_owned();

        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let r_ahat = x.a.rotation.as_matrix();
        let m_vel = r_ic.transpose() * r_ahat.transpose();

        if n > 0 {
            for i in 0..n {
                let qi = &x.q[i];

                // Guard: skip degenerate landmarks (zero-norm position or invalid scale)
                let qhat_i_pos = xi_hat.camera_landmarks[i].p;
                let qq = qhat_i_pos.norm_squared();
                if qq < 1e-20 || !qi.scale.is_finite() || qi.scale.abs() < 1e-12 {
                    continue; // degenerate landmark, leave block as zero
                }

                let qhat_i = qi.rotation.as_matrix() * qi.scale;

                // Velocity -> Landmarks
                let block_vel = -(qhat_i * m_vel);
                a0t.fixed_view_mut::<3, 3>(s + 3 * i, 12)
                    .copy_from(&block_vel);

                // Camera Offset -> Landmarks
                let q0 = xi0.camera_landmarks[i].p;
                let mut temp = nalgebra::SMatrix::<f64, 3, 6>::zeros();
                temp.fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&(base_skew(&q0) * qi.rotation.as_matrix()));
                temp.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-qhat_i));
                let block_offset = temp * common_term;
                a0t.fixed_view_mut::<3, 6>(s + 3 * i, 15)
                    .copy_from(&block_offset);

                // Landmark -> Landmark
                let skew_qhat = base_skew(&qhat_i_pos);
                let skew_vc = base_skew(&v_c);
                let inner = skew_qhat * skew_vc - (v_c * qhat_i_pos.transpose() * 2.0)
                    + qhat_i_pos * v_c.transpose();

                let qhat_inv = qi.rotation.as_matrix().transpose() / qi.scale;
                let a_qi = -(qhat_i * inner * qhat_inv) / qq;
                a0t.fixed_view_mut::<3, 3>(s + 3 * i, s + 3 * i)
                    .copy_from(&a_qi);
            }
        }

        if x.imu_bias_group == ImuBiasGroup::SemiDirect {
            let b_bias = BiasGroupOps::bias_action_matrix(x);
            let ad_b = b_bias.adjoint();
            let ad_b_inv = b_bias.inverse().adjoint();

            let bias_cols = a0t.columns(0, 6) * ad_b_inv;
            a0t.columns_mut(0, 6).copy_from(&bias_cols);

            let beta_in_bias_frame = ad_b_inv * x.beta;
            let m_beta = ad_b * SE3::adjoint_algebra(&beta_in_bias_frame);
            let mut s_bias = SMatrix::<f64, 6, 6>::identity();
            s_bias.fixed_view_mut::<3, 3>(0, 0).scale_mut(-1.0);

            a0t.rows_mut(0, 6).fill(0.0);
            let beta_beta = m_beta * s_bias;
            let beta_beta = beta_beta * ad_b_inv;
            a0t.fixed_view_mut::<6, 6>(0, 0).copy_from(&beta_beta);

            let m_g = -GRAVITY_CONSTANT * base_skew(&xi0.sensor.gravity_dir());
            let mut u_b_pose = SMatrix::<f64, 6, 3>::zeros();
            u_b_pose
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-x.a.rotation.as_matrix().transpose() * m_g));
            let beta_pose = m_beta * u_b_pose;
            a0t.fixed_view_mut::<6, 3>(0, 6).copy_from(&beta_pose);

            // MSCEqF SE23 pos<-att coupling (propagator.cpp:284), ABSENT from echo's
            // Euclidean nav block. MSCEqF: A[D+6,D] = wedge(R0Tv0) - wedge(D.p)*wedge(b_g),
            // Dd order [att,vel,pos]. echo order [att 6:9, pos 9:12, vel 12:15] -> a0t[9:12,6:9].
            // echo velocity is body-frame (= R0^T v0), gravity_dir already carries R0^T -> same chart.
            let se23_a = std::env::var("ECHO_MSC_SE23_A").ok();
            if se23_a.as_deref() == Some("1") || se23_a.as_deref() == Some("2") {
                let r0t_v0 = xi0.sensor.velocity; // body-frame origin velocity = R0^T v0
                let d_p = x.a.translation; // group position ~ X.D().p()
                let b_g = xi0.sensor.gyro_bias(); // origin gyro bias
                let pos_att = base_skew(&r0t_v0) - base_skew(&d_p) * base_skew(&b_g);
                a0t.fixed_view_mut::<3, 3>(9, 6).copy_from(&pos_att);
            }
            if se23_a.as_deref() == Some("2")
                || se23_a.as_deref() == Some("3")
                || se23_a.as_deref() == Some("4")
                || se23_a.as_deref() == Some("5")
            {
                // Full MSCEqF D-block: A[D,D] = Psi - adb0 (propagator.cpp:283). Beyond the
                // vel<-att=grav echo already has, this adds att<-att=-R_b, vel<-vel=-R_b from
                // -adb0 (adb0 = SE3::adjoint(origin bias)). Bias ~1e-3 => R_b ~= I, so -R_b ~= -I.
                // echo order: att 6:9, vel 12:15. att<-vel stays 0 (SE3 adjoint upper-right = 0).
                // ISOLATION: =3 adds ONLY these -I self-terms (skips pos<-att via the guard
                // below); =4 att<-att only; =5 vel<-vel only. Identifies which SE23 self-
                // coupling flips cos_dir>0 (the =2 breakthrough) to scope the wholesale suite.
                let neg_i = -Matrix3::identity();
                if se23_a.as_deref() != Some("5") {
                    a0t.fixed_view_mut::<3, 3>(6, 6).copy_from(&neg_i); // att<-att
                }
                if se23_a.as_deref() != Some("4") {
                    a0t.fixed_view_mut::<3, 3>(12, 12).copy_from(&neg_i); // vel<-vel
                }
            }
            // =3/4/5 skip the pos<-att graft (only the -I self-terms), so undo it here.
            if se23_a.as_deref() == Some("3")
                || se23_a.as_deref() == Some("4")
                || se23_a.as_deref() == Some("5")
            {
                a0t.fixed_view_mut::<3, 3>(9, 6).fill(0.0);
            }
            // =6: the EXACT MSCEqF -adb0 (NOT the -I approximation of =2..5). The algebra
            // adjoint ad_{[b_g,b_a]} = [[b_g^,0],[b_a^,b_g^]] in {R,v} order, so att<-att and
            // vel<-vel are -[b_g]x (tiny skew, bias~1e-3 => ~=0), and vel<-att ADDS -[b_a]x to
            // the gravity term. If =6 ~= baseline (ATE 364.8%, cos_dir -0.10) it PROVES the
            // A[D,D] block already matches MSCEqF (both ~=0) and =4's -I win is artificial
            // damping, not a MSCEqF port. echo order: att 6:9, vel 12:15.
            if se23_a.as_deref() == Some("6") {
                let b_g = xi0.sensor.gyro_bias();
                let b_a = xi0.sensor.accel_bias();
                a0t.fixed_view_mut::<3, 3>(6, 6)
                    .copy_from(&(-base_skew(&b_g))); // att<-att = -[b_g]x
                a0t.fixed_view_mut::<3, 3>(12, 12)
                    .copy_from(&(-base_skew(&b_g))); // vel<-vel = -[b_g]x
                let mut vel_att = a0t.fixed_view::<3, 3>(12, 6).into_owned();
                vel_att += -base_skew(&b_a); // vel<-att += -[b_a]x
                a0t.fixed_view_mut::<3, 3>(12, 6).copy_from(&vel_att);
            }
        }

        // MSCEqF A2 (propagator.cpp:288-289): the EQUIVARIANT bias->pose coupling is IDENTITY.
        // echo's Euclidean chart instead carries A[att,gyrobias] = -R_a (input_matrix_b:255 +
        // sign at :72, then rotated by SDB ad_b_inv :162). R_a is ORTHOGONAL => transport
        // preserves ||Sigma[nav,clone]||_F but ROTATES its direction -> exactly the c143
        // signature (magnitude matched, align rotated att 1.27x over / vel 0.77x under).
        // DISTINGUISHING TEST for the R_a-shear mechanism: flatten to MSCEqF's I and check if
        // U0 att-align moves 0.00157 -> 0.00124 (MSCEqF) at still-matched magnitude. Covariance
        // -only (A enters the Riccati, not the mean lift) so the byte-identical mean is intact.
        // echo order: att 6:9, pos 9:12, vel 12:15; bias gyro 0:3, accel 3:6. delta=(gyro,accel).
        if std::env::var("ECHO_MSC_A2_IDENT").ok().as_deref() == Some("1") {
            a0t.fixed_view_mut::<3, 3>(6, 0).copy_from(&Matrix3::identity()); // att<-gyrobias = I
            a0t.fixed_view_mut::<3, 3>(6, 3).fill(0.0); // att<-accbias = 0
            a0t.fixed_view_mut::<3, 3>(12, 0).fill(0.0); // vel<-gyrobias = 0
            a0t.fixed_view_mut::<3, 3>(12, 3)
                .copy_from(&Matrix3::identity()); // vel<-accbias = I
            a0t.fixed_view_mut::<3, 3>(9, 0)
                .copy_from(&base_skew(&x.a.translation)); // pos<-gyrobias = wedge(D.p)
            a0t.fixed_view_mut::<3, 3>(9, 3).fill(0.0); // pos<-accbias = 0
        }

        a0t
    }

    fn input_matrix_b(&self, x: &VIOGroup, xi0: &VIOState) -> DMatrix<f64> {
        let n = xi0.camera_landmarks.len();
        let dim = xi0.dim();
        let mut bt = DMatrix::<f64>::zeros(dim, 12);
        let s = VIOSensorState::CDIM;

        let xi_hat = state_group_action(x, xi0);
        let r_a = x.a.rotation.as_matrix();

        let bias_noise = BiasGroupOps::new(x.imu_bias_group).physical_bias_noise_matrix(x);
        bt.fixed_view_mut::<6, 6>(0, 6).copy_from(&bias_noise);
        bt.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_a);
        bt.fixed_view_mut::<3, 3>(9, 0)
            .copy_from(&(base_skew(&x.a.translation) * r_a));
        bt.fixed_view_mut::<3, 3>(12, 0)
            .copy_from(&(r_a * base_skew(&xi_hat.sensor.velocity)));
        bt.fixed_view_mut::<3, 3>(12, 3).copy_from(&r_a);

        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let x_ic = xi_hat.sensor.camera_offset.translation;
        let term_x_ic = r_ic.transpose() * base_skew(&x_ic);

        if n > 0 {
            for i in 0..n {
                let qi = &x.q[i];
                let qhat_i = qi.rotation.as_matrix() * qi.scale;
                let qhat_p = xi_hat.camera_landmarks[i].p;
                let inner = base_skew(&qhat_p) * r_ic.transpose() + term_x_ic;
                let block = qhat_i * inner;
                bt.fixed_view_mut::<3, 3>(s + 3 * i, 0).copy_from(&block);
            }
        }

        bt
    }

    fn propagation_blocks(
        &self,
        x: &VIOGroup,
        xi0: &VIOState,
        imu_vel: &IMUVelocity,
    ) -> RiccatiPropagationBlocks {
        let n = xi0.camera_landmarks.len();
        let s = VIOSensorState::CDIM;

        let xi_hat = state_group_action(x, xi0);
        let r_a = x.a.rotation.as_matrix();

        let mut b_s = SMatrix::<f64, 21, 12>::zeros();
        let bias_noise = BiasGroupOps::new(x.imu_bias_group).physical_bias_noise_matrix(x);
        b_s.fixed_view_mut::<6, 6>(0, 6).copy_from(&bias_noise);
        b_s.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_a);
        b_s.fixed_view_mut::<3, 3>(9, 0)
            .copy_from(&(base_skew(&x.a.translation) * r_a));
        b_s.fixed_view_mut::<3, 3>(12, 0)
            .copy_from(&(r_a * base_skew(&xi_hat.sensor.velocity)));
        b_s.fixed_view_mut::<3, 3>(12, 3).copy_from(&r_a);

        let mut a_ss = SMatrix::<f64, 21, 21>::zeros();
        for j in 0..6 {
            a_ss.column_mut(j).copy_from(&(-b_s.column(j)));
        }
        a_ss.fixed_view_mut::<3, 3>(9, 12)
            .copy_from(&Matrix3::identity());
        a_ss.fixed_view_mut::<3, 3>(12, 6)
            .copy_from(&(-GRAVITY_CONSTANT * base_skew(&xi0.sensor.gravity_dir())));

        let v_est = imu_vel.gyr - xi_hat.sensor.gyro_bias();
        let mut u_i = nalgebra::SVector::<f64, 6>::zeros();
        u_i.fixed_rows_mut::<3>(0).copy_from(&v_est);
        u_i.fixed_rows_mut::<3>(3)
            .copy_from(&xi_hat.sensor.velocity);

        let ad_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        let ad_a = x.a.adjoint();
        let transformed = ad_tc_inv * ad_a * u_i;
        a_ss.fixed_view_mut::<6, 6>(15, 15)
            .copy_from(&SE3::adjoint_algebra(&transformed));

        let common_term = x.b.inverse().adjoint() * SE3::adjoint_algebra(&transformed);
        let u_c_full = xi_hat.sensor.camera_offset.inverse().adjoint() * u_i;
        let v_c = u_c_full.fixed_rows::<3>(3).into_owned();

        let r_ic = xi_hat.sensor.camera_offset.rotation.as_matrix();
        let r_ahat = x.a.rotation.as_matrix();
        let m_vel = r_ic.transpose() * r_ahat.transpose();
        let x_ic = xi_hat.sensor.camera_offset.translation;
        let term_x_ic = r_ic.transpose() * base_skew(&x_ic);

        let mut a_lm_s = DMatrix::<f64>::zeros(3 * n, s);
        let mut a_lm_lm = Vec::with_capacity(n);
        let mut b_lm = DMatrix::<f64>::zeros(3 * n, 12);

        let semi_direct_a = if x.imu_bias_group == ImuBiasGroup::SemiDirect {
            Some(self.state_matrix_a(x, xi0, imu_vel))
        } else {
            None
        };

        for i in 0..n {
            let qi = &x.q[i];
            let row = 3 * i;
            let qhat_i = qi.rotation.as_matrix() * qi.scale;
            let qhat_p = xi_hat.camera_landmarks[i].p;
            let inner_b = base_skew(&qhat_p) * r_ic.transpose() + term_x_ic;
            let b_block = qhat_i * inner_b;
            b_lm.fixed_view_mut::<3, 3>(row, 0).copy_from(&b_block);
            a_lm_s.fixed_view_mut::<3, 3>(row, 0).copy_from(&(-b_block));

            let qq = qhat_p.norm_squared();
            if qq < 1e-20 || !qi.scale.is_finite() || qi.scale.abs() < 1e-12 {
                a_lm_lm.push(SMatrix::<f64, 3, 3>::zeros());
                continue;
            }

            let block_vel = -(qhat_i * m_vel);
            a_lm_s.fixed_view_mut::<3, 3>(row, 12).copy_from(&block_vel);

            let q0 = xi0.camera_landmarks[i].p;
            let mut temp = SMatrix::<f64, 3, 6>::zeros();
            temp.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(base_skew(&q0) * qi.rotation.as_matrix()));
            temp.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-qhat_i));
            let block_offset = temp * common_term;
            a_lm_s
                .fixed_view_mut::<3, 6>(row, 15)
                .copy_from(&block_offset);

            let skew_qhat = base_skew(&qhat_p);
            let skew_vc = base_skew(&v_c);
            let inner =
                skew_qhat * skew_vc - (v_c * qhat_p.transpose() * 2.0) + qhat_p * v_c.transpose();
            let qhat_inv = qi.rotation.as_matrix().transpose() / qi.scale;
            let a_qi = -(qhat_i * inner * qhat_inv) / qq;
            a_lm_lm.push(a_qi);
        }

        if let Some(a) = semi_direct_a {
            a_ss.copy_from(&a.fixed_view::<21, 21>(0, 0));
            for i in 0..n {
                let state_row = s + 3 * i;
                let compact_row = 3 * i;
                a_lm_s
                    .fixed_view_mut::<3, 21>(compact_row, 0)
                    .copy_from(&a.fixed_view::<3, 21>(state_row, 0));
                a_lm_lm[i].copy_from(&a.fixed_view::<3, 3>(state_row, state_row));
            }
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
        let q_hat_pos = q_hat.inverse().act(q0);
        let y_hat_bearing = q_hat_pos.normalize();

        let qq = q0.norm_squared();
        let mut m2g = nalgebra::SMatrix::<f64, 4, 3>::zeros();
        m2g.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-base_skew(q0) / qq));
        m2g.fixed_view_mut::<1, 3>(3, 0)
            .copy_from(&(-q0.transpose() / qq));

        let d_rho = |y_vec: &Vector3<f64>| {
            let mut d_rho_vec = nalgebra::Matrix3x4::<f64>::zeros();
            d_rho_vec
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&base_skew(y_vec));
            let proj_jac = cam.projection_jacobian(y_vec);
            proj_jac * d_rho_vec
        };

        let y_tru_bearing = cam.undistort(y);
        let q_inv_adj = q_hat.inverse().adjoint();

        0.5 * (d_rho(&y_tru_bearing) + d_rho(&y_hat_bearing)) * q_inv_adj * m2g
    }

    fn lift_innovation(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOAlgebra {
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();

        let mut u_b = nalgebra::Vector6::zeros();
        let u_a = total_innovation.fixed_rows::<6>(6).into_owned();
        let adj_tc_inv = xi0.sensor.camera_offset.inverse().adjoint();
        u_b.copy_from(&total_innovation.fixed_rows::<6>(15));
        u_b += adj_tc_inv * u_a;

        let mut delta = VIOAlgebra {
            u_beta: total_innovation.fixed_rows::<6>(0).into_owned(),
            u_a,
            u_b,
            u_w: nalgebra::Vector3::zeros(),
            w: Vec::with_capacity(n),
            id: Vec::with_capacity(n),
        };

        let gamma_v = total_innovation.fixed_rows::<3>(12).into_owned();
        let omega_a = delta.u_a.fixed_rows::<3>(0).into_owned();
        // DIAGNOSTIC (ECHO_MSC_SDBVEL=1, default-off): drop the `-skew(omega_a)·v0`
        // term that pre-CANCELS the semidirect attitude→velocity coupling `compose`
        // re-applies (`w ← delta.w + R_δ·v_x`, vio_group.rs:55). With it, echo's
        // velocity correction is ADDITIVE (`v ← v − gamma_v`); MSCEqF's plain
        // `SDB::exp(inn)·Dd` (updateLeft) is MULTIPLICATIVE (`v ← R_δ·v + u_w`,
        // attitude rotates velocity). Dropping it makes echo's nav correction
        // identical to MSCEqF's SDB left action in the att↔vel coupling.
        if std::env::var("ECHO_MSC_SDBVEL").as_deref() == Ok("1") {
            delta.u_w = -gamma_v;
        } else {
            delta.u_w = -gamma_v - base_skew(&omega_a) * xi0.sensor.velocity;
        }

        for i in 0..n {
            let gamma_qi = total_innovation.fixed_rows::<3>(s + 3 * i).into_owned();
            let q0 = xi0.camera_landmarks[i].p;
            let qq = q0.norm_squared();

            let mut wi = nalgebra::Vector4::zeros();
            wi.fixed_rows_mut::<3>(0)
                .copy_from(&(-q0.cross(&gamma_qi) / qq));
            wi[3] = -q0.dot(&gamma_qi) / qq;

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
        let s = VIOSensorState::CDIM;
        let n = xi0.camera_landmarks.len();

        let beta = total_innovation.fixed_rows::<6>(0).into_owned();
        let a = SE3::exp(&total_innovation.fixed_rows::<6>(6).into_owned());

        let gamma_v = total_innovation.fixed_rows::<3>(12).into_owned();
        let v0 = xi0.sensor.velocity;
        let w = v0 - a.rotation.act(&(v0 + gamma_v));

        let b = xi0
            .sensor
            .camera_offset
            .inverse()
            .compose(&a)
            .compose(&xi0.sensor.camera_offset)
            .compose(&SE3::exp(
                &total_innovation.fixed_rows::<6>(15).into_owned(),
            ));

        let mut q_vec = Vec::with_capacity(n);
        let mut id_vec = Vec::with_capacity(n);
        for i in 0..n {
            let qi = xi0.camera_landmarks[i].p;
            let gamma_qi = total_innovation.fixed_rows::<3>(s + 3 * i).into_owned();
            let qi1 = qi + gamma_qi;

            let rot = SO3::from_vectors(&(qi1.normalize()), &(qi.normalize()));
            let scale = qi.norm() / qi1.norm();
            q_vec.push(SOT3::new(rot, scale));
            id_vec.push(xi0.camera_landmarks[i].id);
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
}
