use nalgebra::{DMatrix, DVector, SMatrix, Vector2};
use std::collections::HashMap;
use echo_lie::SOT3;

use crate::mathematical::vio_state::{VIOState, VIOSensorState, Landmark};
use crate::mathematical::vio_group::{VIOGroup, state_group_action, lift_velocity, lift_velocity_discrete, vio_exp};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::camera::CameraModel;

pub struct VIOEqF {
    pub xi0: VIOState,
    pub x: VIOGroup,
    pub sigma: DMatrix<f64>,
    pub current_time: f64,
}

impl VIOEqF {
    pub fn new(xi0: VIOState, initial_covariance: &DMatrix<f64>) -> Self {
        let sigma = initial_covariance.clone();
        let x = VIOGroup::identity(&xi0.get_ids());

        Self {
            xi0,
            x,
            sigma,
            current_time: -1.0,
        }
    }

    pub fn state_estimate(&self) -> VIOState {
        state_group_action(&self.x, &self.xi0)
    }

    // ------------------------------------------------------------------
    // Observer state integration
    // ------------------------------------------------------------------

    pub fn integrate_observer_state(&mut self, imu: &IMUVelocity, dt: f64, discrete_lift: bool) {
        let lifted = if discrete_lift {
            lift_velocity_discrete(&self.state_estimate(), imu, dt)
        } else {
            let lifted_alg = lift_velocity(&self.state_estimate(), imu);
            // Scale algebra by dt, then exponentiate
            let scaled = crate::mathematical::vio_group::VIOAlgebra {
                u_beta: lifted_alg.u_beta * dt,
                u_a: lifted_alg.u_a * dt,
                u_b: lifted_alg.u_b * dt,
                u_w: lifted_alg.u_w * dt,
                w: lifted_alg.w.iter().map(|wi| wi * dt).collect(),
                id: lifted_alg.id,
            };
            vio_exp(&scaled)
        };
        self.x = self.x.compose(&lifted);
    }

    // ------------------------------------------------------------------
    // Riccati propagation (Euler)
    // ------------------------------------------------------------------

    pub fn integrate_riccati_fast<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        imu: &IMUVelocity,
        dt: f64,
        input_gain: &SMatrix<f64, 12, 12>,
        state_gain: &DMatrix<f64>,
    ) {
        let a0t = suite.state_matrix_a(&self.x, &self.xi0, imu);
        let bt = suite.input_matrix_b(&self.x, &self.xi0);
        let n = self.xi0.dim();
        let s = VIOSensorState::CDIM;
        let n_lm = (n - s) / 3;

        // F = I + A·dt has the same block-sparsity as A across all coordinate suites:
        //   F_ss (21×21), F_li_s (3×21) per landmark, F_li_li (3×3) per landmark.
        // Sensor←landmark and cross-landmark blocks are exactly zero, so F·Σ·F^T
        // reduces to a few large gemm calls on the sensor band plus per-landmark
        // 3×N updates instead of two dense n×n multiplies.
        let f_ss: SMatrix<f64, 21, 21> = SMatrix::<f64, 21, 21>::identity()
            + a0t.fixed_view::<21, 21>(0, 0) * dt;
        let mut f_lm_s = DMatrix::<f64>::zeros(3 * n_lm, s);
        let mut f_li_li: Vec<SMatrix<f64, 3, 3>> = Vec::with_capacity(n_lm);
        for i in 0..n_lm {
            let block = a0t.fixed_view::<3, 21>(s + 3 * i, 0).into_owned() * dt;
            f_lm_s.fixed_view_mut::<3, 21>(3 * i, 0).copy_from(&block);
            f_li_li.push(SMatrix::<f64, 3, 3>::identity()
                + a0t.fixed_view::<3, 3>(s + 3 * i, s + 3 * i) * dt);
        }
        let f_ss_t = f_ss.transpose();
        let f_lm_s_t = f_lm_s.transpose();

        // Q_total = dt · (B · InputGain · B^T + StateGain)
        let q_input = &bt * input_gain * bt.transpose();
        let state_gain_view = state_gain.view((0, 0), (n, n));
        let q_total = (q_input + state_gain_view) * dt;

        // ---- Step 1: M = F · Σ ----
        let mut m_buf = DMatrix::<f64>::zeros(n, n);
        {
            let sigma_active = &self.sigma;

            // M[0:s, :] = F_ss · Σ[0:s, :]
            {
                let sigma_s = sigma_active.view((0, 0), (s, n));
                let mut m_top = m_buf.view_mut((0, 0), (s, n));
                m_top.gemm(1.0, &f_ss, &sigma_s, 0.0);
            }

            if n_lm > 0 {
                // M[s:n, :] = F_lm_s · Σ[0:s, :]  (one big gemm for all landmarks)
                {
                    let sigma_s = sigma_active.view((0, 0), (s, n));
                    let mut m_lm = m_buf.view_mut((s, 0), (3 * n_lm, n));
                    m_lm.gemm(1.0, &f_lm_s, &sigma_s, 0.0);
                }
                // M[s+3i:s+3i+3, :] += F_li_li · Σ[s+3i:s+3i+3, :]
                for i in 0..n_lm {
                    let row_band = s + 3 * i;
                    let sigma_li = sigma_active.view((row_band, 0), (3, n));
                    let mut m_li = m_buf.view_mut((row_band, 0), (3, n));
                    m_li.gemm(1.0, &f_li_li[i], &sigma_li, 1.0);
                }
            }
        }

        // ---- Step 2: Σ_new = M · F^T ----
        let mut sigma_new = DMatrix::<f64>::zeros(n, n);

        // Σ_new[0:s, 0:s] = M[0:s, 0:s] · F_ss^T
        {
            let m_ss = m_buf.view((0, 0), (s, s));
            let mut sn_ss = sigma_new.view_mut((0, 0), (s, s));
            sn_ss.gemm(1.0, &m_ss, &f_ss_t, 0.0);
        }

        if n_lm > 0 {
            // Σ_new[0:s, s:n] = M[0:s, 0:s] · F_lm_s^T
            {
                let m_ss = m_buf.view((0, 0), (s, s));
                let mut sn_top_lm = sigma_new.view_mut((0, s), (s, 3 * n_lm));
                sn_top_lm.gemm(1.0, &m_ss, &f_lm_s_t, 0.0);
            }
            // Σ_new[0:s, s+3j:s+3j+3] += M[0:s, s+3j:s+3j+3] · F_lj_lj^T
            for j in 0..n_lm {
                let col_band = s + 3 * j;
                let m_s_lj = m_buf.view((0, col_band), (s, 3));
                let f_t = f_li_li[j].transpose();
                let mut sn_view = sigma_new.view_mut((0, col_band), (s, 3));
                sn_view.gemm(1.0, &m_s_lj, &f_t, 1.0);
            }

            // Σ_new[s:n, 0:s] = M[s:n, 0:s] · F_ss^T
            {
                let m_lm_s = m_buf.view((s, 0), (3 * n_lm, s));
                let mut sn_lm_s = sigma_new.view_mut((s, 0), (3 * n_lm, s));
                sn_lm_s.gemm(1.0, &m_lm_s, &f_ss_t, 0.0);
            }

            // Σ_new[s:n, s:n] = M[s:n, 0:s] · F_lm_s^T
            {
                let m_lm_s = m_buf.view((s, 0), (3 * n_lm, s));
                let mut sn_lm_lm = sigma_new.view_mut((s, s), (3 * n_lm, 3 * n_lm));
                sn_lm_lm.gemm(1.0, &m_lm_s, &f_lm_s_t, 0.0);
            }
            // Σ_new[s:n, s+3j:s+3j+3] += M[s:n, s+3j:s+3j+3] · F_lj_lj^T
            for j in 0..n_lm {
                let col_band = s + 3 * j;
                let m_lm_lj = m_buf.view((s, col_band), (3 * n_lm, 3));
                let f_t = f_li_li[j].transpose();
                let mut sn_view = sigma_new.view_mut((s, col_band), (3 * n_lm, 3));
                sn_view.gemm(1.0, &m_lm_lj, &f_t, 1.0);
            }
        }

        sigma_new += q_total;

        self.sigma = sigma_new;
        self.enforce_spd();
    }

    fn enforce_spd(&mut self) {
        let n = self.sigma.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let val = (self.sigma[(i, j)] + self.sigma[(j, i)]) * 0.5;
                self.sigma[(i, j)] = val;
                self.sigma[(j, i)] = val;
            }
            // Clamp minimum diagonal (matches Python: diagonal + 1e-12)
            self.sigma[(i, i)] += 1e-12;
        }
    }

    // ------------------------------------------------------------------
    // Vision update (standard bearing-only)
    // ------------------------------------------------------------------

    pub fn perform_vision_update<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        y_ids: &[u64],
        y_coords: &HashMap<u64, Vector2<f64>>,
        cam: &dyn CameraModel,
        output_gain: &DMatrix<f64>,
        use_equivariance: bool,
        use_discrete_correction: bool,
    ) {
        if y_ids.is_empty() { return; }

        let n_obs = y_ids.len();
        let xi_hat = self.state_estimate();

        // Innovation vector
        let mut y_tilde = DVector::<f64>::zeros(2 * n_obs);
        for (j, &id) in y_ids.iter().enumerate() {
            let lm_idx = xi_hat.camera_landmarks.iter().position(|lm| lm.id == id)
                .expect("Landmark ID must exist in state estimate");
            let q = &xi_hat.camera_landmarks[lm_idx].p;
            let y_pred = cam.project(q);
            let y_obs = y_coords.get(&id).expect("Observed ID must exist in coordinates");
            y_tilde.fixed_rows_mut::<2>(2 * j).copy_from(&(y_obs - y_pred));
        }

        // Output matrix C*
        let ct = suite.output_matrix_C(&self.xi0, &self.x, y_ids, y_coords, cam, use_equivariance);

        // Skip update if C* contains NaN (degenerate landmark)
        if !ct.iter().all(|v| v.is_finite()) {
            return;
        }

        self.perform_stacked_update(suite, &y_tilde, &ct, output_gain, use_discrete_correction);
    }

    // ------------------------------------------------------------------
    // Stacked update (Joseph form)
    // ------------------------------------------------------------------

    pub fn perform_stacked_update<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        residual: &DVector<f64>,
        c_star: &DMatrix<f64>,
        r_noise: &DMatrix<f64>,
        use_discrete_correction: bool,
    ) {
        if residual.len() == 0 { return; }

        let n = self.xi0.dim();
        let sigma_active = &self.sigma;

        // S = C * Sigma * C^T + R
        let s = c_star * sigma_active * c_star.transpose() + r_noise;

        // Skip if S is degenerate
        if !s.iter().all(|v| v.is_finite()) {
            return;
        }

        // K = Sigma * C^T * S^{-1}
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return,
        };
        let k = sigma_active * c_star.transpose() * s_inv;

        // Gamma = K * residual
        let gamma = &k * residual;

        // Skip if gain is degenerate
        if !gamma.iter().all(|v| v.is_finite()) {
            return;
        }

        // Lift to group correction
        if use_discrete_correction {
            let delta = suite.lift_innovation_discrete(
                &DVector::from_column_slice(gamma.as_slice()), &self.xi0);
            self.x = delta.compose(&self.x);
        } else {
            let delta_alg = suite.lift_innovation(
                &DVector::from_column_slice(gamma.as_slice()), &self.xi0);
            let delta = vio_exp(&delta_alg);
            self.x = delta.compose(&self.x);
        }

        // Joseph form: Σ = (I - KC) Σ (I - KC)^T + K R K^T
        let i_kc = DMatrix::<f64>::identity(n, n) - &k * c_star;
        let sigma_new = &i_kc * sigma_active * i_kc.transpose() + &k * r_noise * k.transpose();

        self.sigma = sigma_new;
        self.enforce_spd();
    }

    // ------------------------------------------------------------------
    // Landmark management
    // ------------------------------------------------------------------

    pub fn add_new_landmarks(&mut self, new_landmarks: Vec<Landmark>, new_cov: &DMatrix<f64>) {
        let n_old = self.xi0.dim();

        for lm in new_landmarks {
            self.xi0.camera_landmarks.push(lm.clone());
            self.x.q.push(SOT3::identity());
            self.x.id.push(lm.id);
        }

        let n_new = self.xi0.dim();
        let n_added = n_new - n_old;
        if n_added > 0 {
            // Augment sigma: grow from n_old×n_old to n_new×n_new
            let mut sigma_new = DMatrix::<f64>::zeros(n_new, n_new);
            sigma_new.view_mut((0, 0), (n_old, n_old)).copy_from(&self.sigma);
            let copy_size = n_added.min(new_cov.nrows());
            sigma_new.view_mut((n_old, n_old), (copy_size, copy_size))
                .copy_from(&new_cov.view((0, 0), (copy_size, copy_size)));
            self.sigma = sigma_new;
        }
    }

    pub fn remove_landmark_by_id(&mut self, lm_id: u64) {
        if let Some(idx) = self.xi0.camera_landmarks.iter().position(|lm| lm.id == lm_id) {
            let s = VIOSensorState::CDIM;
            let start = s + 3 * idx;

            // Remove from state and group
            self.xi0.camera_landmarks.remove(idx);
            self.x.q.remove(idx);
            self.x.id.remove(idx);

            let n_new = self.xi0.dim();

            // Build new smaller sigma by removing 3 rows/cols at `start`
            let mut sigma_new = DMatrix::<f64>::zeros(n_new, n_new);

            // Top-left block: [0..start, 0..start]
            if start > 0 {
                sigma_new.view_mut((0, 0), (start, start))
                    .copy_from(&self.sigma.view((0, 0), (start, start)));
            }
            // Top-right block: [0..start, start..n_new]
            let after = n_new - start;
            if start > 0 && after > 0 {
                sigma_new.view_mut((0, start), (start, after))
                    .copy_from(&self.sigma.view((0, start + 3), (start, after)));
            }
            // Bottom-left block: [start..n_new, 0..start]
            if after > 0 && start > 0 {
                sigma_new.view_mut((start, 0), (after, start))
                    .copy_from(&self.sigma.view((start + 3, 0), (after, start)));
            }
            // Bottom-right block: [start..n_new, start..n_new]
            if after > 0 {
                sigma_new.view_mut((start, start), (after, after))
                    .copy_from(&self.sigma.view((start + 3, start + 3), (after, after)));
            }

            self.sigma = sigma_new;
        }
    }

    pub fn remove_invalid_landmarks(&mut self) {
        let invalid_ids: Vec<u64> = self.x.id.iter().zip(self.x.q.iter())
            .filter(|(_, q)| !q.scale.is_finite() || q.scale <= 1e-8 || q.scale > 1e8)
            .map(|(&id, _)| id)
            .collect();
        for id in invalid_ids {
            self.remove_landmark_by_id(id);
        }
    }

    // ------------------------------------------------------------------
    // Covariance queries
    // ------------------------------------------------------------------

    pub fn get_landmark_cov_by_id(&self, lm_id: u64) -> Option<nalgebra::Matrix3<f64>> {
        let idx = self.xi0.camera_landmarks.iter().position(|lm| lm.id == lm_id)?;
        let start = VIOSensorState::CDIM + 3 * idx;
        Some(self.sigma.fixed_view::<3, 3>(start, start).into_owned())
    }
}
