use echo_lie::SOT3;
use nalgebra::{DMatrix, DVector, SMatrix, Vector2};
use std::collections::HashMap;

use crate::ImuBiasGroup;
use crate::mathematical::bias_group_ops::BiasGroupOps;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_group::{
    VIOGroup, lift_velocity, lift_velocity_discrete, state_group_action, vio_exp_with_bias_group,
};
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState};

pub struct VIOEqF {
    pub xi0: VIOState,
    pub x: VIOGroup,
    pub sigma: DMatrix<f64>,
    pub current_time: f64,
    imu_bias_group: ImuBiasGroup,
    scratch_m: DMatrix<f64>,
    scratch_sigma: DMatrix<f64>,
    // `Faster` variant accumulator — the sub-frame transition Φ in block form
    // (Phase 6, see docs/imu_optimization_plan.md). Identity when empty.
    phi_ss: SMatrix<f64, 21, 21>,
    phi_lm_s: DMatrix<f64>,
    phi_lm_s_scratch: DMatrix<f64>,
    phi_li_li: Vec<SMatrix<f64, 3, 3>>,
    accum_dt: f64,
    accum_count: usize,
    last_b_s: SMatrix<f64, 21, 12>,
    last_b_lm: DMatrix<f64>,
}

impl VIOEqF {
    pub fn new(xi0: VIOState, initial_covariance: &DMatrix<f64>) -> Self {
        Self::new_with_bias_group(xi0, initial_covariance, ImuBiasGroup::Additive)
    }

    pub fn new_with_bias_group(
        xi0: VIOState,
        initial_covariance: &DMatrix<f64>,
        imu_bias_group: ImuBiasGroup,
    ) -> Self {
        let sigma = initial_covariance.clone();
        let x = VIOGroup::identity_with_bias_group(&xi0.get_ids(), imu_bias_group);
        let n = xi0.dim();
        let n_lm = (n - VIOSensorState::CDIM) / 3;

        Self {
            xi0,
            x,
            sigma,
            current_time: -1.0,
            imu_bias_group,
            scratch_m: DMatrix::<f64>::zeros(n, n),
            scratch_sigma: DMatrix::<f64>::zeros(n, n),
            phi_ss: SMatrix::<f64, 21, 21>::identity(),
            phi_lm_s: DMatrix::<f64>::zeros(3 * n_lm, 21),
            phi_lm_s_scratch: DMatrix::<f64>::zeros(3 * n_lm, 21),
            phi_li_li: vec![SMatrix::<f64, 3, 3>::identity(); n_lm],
            accum_dt: 0.0,
            accum_count: 0,
            last_b_s: SMatrix::<f64, 21, 12>::zeros(),
            last_b_lm: DMatrix::<f64>::zeros(3 * n_lm, 12),
        }
    }

    pub fn state_estimate(&self) -> VIOState {
        state_group_action(&self.x, &self.xi0)
    }

    // ------------------------------------------------------------------
    // Observer state integration
    // ------------------------------------------------------------------

    pub fn integrate_observer_state(&mut self, imu: &IMUVelocity, dt: f64, discrete_lift: bool) {
        let state = self.state_estimate();
        let mut additive_bias_delta = nalgebra::Vector6::zeros();
        additive_bias_delta
            .fixed_rows_mut::<3>(0)
            .copy_from(&(dt * imu.gyr_bias_vel));
        additive_bias_delta
            .fixed_rows_mut::<3>(3)
            .copy_from(&(dt * imu.acc_bias_vel));

        let lifted = if discrete_lift {
            lift_velocity_discrete(&state, imu, dt)
        } else {
            let lifted_alg = lift_velocity(&state, imu);
            // Scale algebra by dt, then exponentiate
            let scaled = crate::mathematical::vio_group::VIOAlgebra {
                u_beta: lifted_alg.u_beta * dt,
                u_a: lifted_alg.u_a * dt,
                u_b: lifted_alg.u_b * dt,
                u_w: lifted_alg.u_w * dt,
                w: lifted_alg.w.iter().map(|wi| wi * dt).collect(),
                id: lifted_alg.id,
            };
            vio_exp_with_bias_group(&scaled, self.imu_bias_group)
        };
        let lifted = self.prepare_right_observer_increment(lifted, &state, additive_bias_delta);
        self.x = self.x.compose(&lifted);
    }

    fn prepare_right_observer_increment(
        &self,
        mut lifted: VIOGroup,
        state: &VIOState,
        additive_bias_delta: nalgebra::Vector6<f64>,
    ) -> VIOGroup {
        let ops = BiasGroupOps::new(self.imu_bias_group);
        lifted.beta =
            ops.observer_increment_beta(&lifted, &state.sensor.input_bias, &additive_bias_delta);
        lifted.with_bias_group(self.imu_bias_group)
    }

    // ------------------------------------------------------------------
    // Riccati propagation (Euler)
    // ------------------------------------------------------------------

    /// `Fast` variant — per-sample Euler Riccati: Σ ← F·Σ·Fᵀ + Q, F = I + A·dt.
    pub fn integrate_riccati_fast<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        imu: &IMUVelocity,
        dt: f64,
        input_gain: &SMatrix<f64, 12, 12>,
        state_gain: &DMatrix<f64>,
    ) {
        let blocks = suite.propagation_blocks(&self.x, &self.xi0, imu);
        let n = self.xi0.dim();
        let s = VIOSensorState::CDIM;
        let n_lm = (n - s) / 3;

        // F = I + A·dt has the same block-sparsity as A across all coordinate
        // suites: F_ss (21×21), F_li_s (3×21) per landmark, F_li_li (3×3) per
        // landmark; sensor←landmark and cross-landmark blocks are exactly zero.
        let f_ss: SMatrix<f64, 21, 21> = SMatrix::<f64, 21, 21>::identity() + blocks.a_ss * dt;
        let mut f_lm_s = DMatrix::<f64>::zeros(3 * n_lm, s);
        let mut f_li_li: Vec<SMatrix<f64, 3, 3>> = Vec::with_capacity(n_lm);
        for i in 0..n_lm {
            let block = blocks.a_lm_s.fixed_view::<3, 21>(3 * i, 0).into_owned() * dt;
            f_lm_s.fixed_view_mut::<3, 21>(3 * i, 0).copy_from(&block);
            f_li_li.push(SMatrix::<f64, 3, 3>::identity() + blocks.a_lm_lm[i] * dt);
        }

        // Q_total = dt · (B · InputGain · B^T + StateGain)
        let bt = dense_b(&blocks.b_s, &blocks.b_lm);
        let q_input = &bt * input_gain * bt.transpose();
        let state_gain_view = state_gain.view((0, 0), (n, n));
        let q_total = (q_input + state_gain_view) * dt;

        self.apply_transport(&f_ss, &f_lm_s, &f_li_li, &q_total);
    }

    /// Σ ← F·Σ·F^T + q_total for the block-sparse transition `F`.
    ///
    /// `F` is block lower-triangular — `F_ss` (21×21), `F_lm_s` (3·n_lm×21),
    /// per-landmark diagonal `F_li_li` (3×3) — so the transport reduces to a few
    /// large gemm calls on the sensor band plus per-landmark 3×N updates instead
    /// of two dense n×n multiplies. Shared by `integrate_riccati_fast` (`Fast`)
    /// and `flush_riccati` (`Faster`, where `F` is the composed sub-frame Φ).
    fn apply_transport(
        &mut self,
        f_ss: &SMatrix<f64, 21, 21>,
        f_lm_s: &DMatrix<f64>,
        f_li_li: &[SMatrix<f64, 3, 3>],
        q_total: &DMatrix<f64>,
    ) {
        let n = self.xi0.dim();
        let s = VIOSensorState::CDIM;
        let n_lm = (n - s) / 3;
        let f_ss_t = f_ss.transpose();
        let f_lm_s_t = f_lm_s.transpose();

        // ---- Step 1: M = F · Σ ----
        {
            let sigma_active = &self.sigma;

            // M[0:s, :] = F_ss · Σ[0:s, :]
            {
                let sigma_s = sigma_active.view((0, 0), (s, n));
                let mut m_top = self.scratch_m.view_mut((0, 0), (s, n));
                m_top.gemm(1.0, f_ss, &sigma_s, 0.0);
            }

            if n_lm > 0 {
                // M[s:n, :] = F_lm_s · Σ[0:s, :]  (one big gemm for all landmarks)
                {
                    let sigma_s = sigma_active.view((0, 0), (s, n));
                    let mut m_lm = self.scratch_m.view_mut((s, 0), (3 * n_lm, n));
                    m_lm.gemm(1.0, f_lm_s, &sigma_s, 0.0);
                }
                // M[s+3i:s+3i+3, :] += F_li_li · Σ[s+3i:s+3i+3, :]
                for i in 0..n_lm {
                    let row_band = s + 3 * i;
                    let sigma_li = sigma_active.view((row_band, 0), (3, n));
                    let mut m_li = self.scratch_m.view_mut((row_band, 0), (3, n));
                    m_li.gemm(1.0, &f_li_li[i], &sigma_li, 1.0);
                }
            }
        }

        // ---- Step 2: Σ_new = M · F^T ----

        // Σ_new[0:s, 0:s] = M[0:s, 0:s] · F_ss^T
        {
            let m_ss = self.scratch_m.view((0, 0), (s, s));
            let mut sn_ss = self.scratch_sigma.view_mut((0, 0), (s, s));
            sn_ss.gemm(1.0, &m_ss, &f_ss_t, 0.0);
        }

        if n_lm > 0 {
            // Σ_new[0:s, s:n] = M[0:s, 0:s] · F_lm_s^T
            {
                let m_ss = self.scratch_m.view((0, 0), (s, s));
                let mut sn_top_lm = self.scratch_sigma.view_mut((0, s), (s, 3 * n_lm));
                sn_top_lm.gemm(1.0, &m_ss, &f_lm_s_t, 0.0);
            }
            // Σ_new[0:s, s+3j:s+3j+3] += M[0:s, s+3j:s+3j+3] · F_lj_lj^T
            for j in 0..n_lm {
                let col_band = s + 3 * j;
                let m_s_lj = self.scratch_m.view((0, col_band), (s, 3));
                let f_t = f_li_li[j].transpose();
                let mut sn_view = self.scratch_sigma.view_mut((0, col_band), (s, 3));
                sn_view.gemm(1.0, &m_s_lj, &f_t, 1.0);
            }

            // Σ_new[s:n, 0:s] = M[s:n, 0:s] · F_ss^T
            {
                let m_lm_s = self.scratch_m.view((s, 0), (3 * n_lm, s));
                let mut sn_lm_s = self.scratch_sigma.view_mut((s, 0), (3 * n_lm, s));
                sn_lm_s.gemm(1.0, &m_lm_s, &f_ss_t, 0.0);
            }

            // Σ_new[s:n, s:n] = M[s:n, 0:s] · F_lm_s^T
            {
                let m_lm_s = self.scratch_m.view((s, 0), (3 * n_lm, s));
                let mut sn_lm_lm = self.scratch_sigma.view_mut((s, s), (3 * n_lm, 3 * n_lm));
                sn_lm_lm.gemm(1.0, &m_lm_s, &f_lm_s_t, 0.0);
            }
            // Σ_new[s:n, s+3j:s+3j+3] += M[s:n, s+3j:s+3j+3] · F_lj_lj^T
            for j in 0..n_lm {
                let col_band = s + 3 * j;
                let m_lm_lj = self.scratch_m.view((s, col_band), (3 * n_lm, 3));
                let f_t = f_li_li[j].transpose();
                let mut sn_view = self.scratch_sigma.view_mut((s, col_band), (3 * n_lm, 3));
                sn_view.gemm(1.0, &m_lm_lj, &f_t, 1.0);
            }
        }

        self.scratch_sigma += q_total;

        std::mem::swap(&mut self.sigma, &mut self.scratch_sigma);
        self.enforce_spd();
    }

    /// `Faster` variant — compose this IMU sample's transition `F = I + A·dt`
    /// into the sub-frame accumulator Φ. Cheap block products only, no O(n²)
    /// transport; `flush_riccati` applies `Φ·Σ·Φ^T` once per sub-frame.
    pub fn accumulate_transition<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        imu: &IMUVelocity,
        dt: f64,
    ) {
        let blocks = suite.propagation_blocks(&self.x, &self.xi0, imu);
        let n_lm = (self.xi0.dim() - VIOSensorState::CDIM) / 3;

        let f_ss: SMatrix<f64, 21, 21> = SMatrix::<f64, 21, 21>::identity() + blocks.a_ss * dt;

        // Φ ← F · Φ. Both are block lower-triangular and the product keeps that
        // structure, so the composition is exact and stays block-sparse:
        //   Φ_lm_s[i] ← F_lm_s[i]·Φ_ss + F_li_li[i]·Φ_lm_s[i]   (uses old Φ_ss)
        //   Φ_li_li[i] ← F_li_li[i]·Φ_li_li[i]
        //   Φ_ss      ← F_ss·Φ_ss
        for i in 0..n_lm {
            let f_lm_s_i: SMatrix<f64, 3, 21> =
                blocks.a_lm_s.fixed_view::<3, 21>(3 * i, 0).into_owned() * dt;
            let f_li_li_i: SMatrix<f64, 3, 3> =
                SMatrix::<f64, 3, 3>::identity() + blocks.a_lm_lm[i] * dt;
            let phi_lm_s_old: SMatrix<f64, 3, 21> =
                self.phi_lm_s.fixed_view::<3, 21>(3 * i, 0).into_owned();
            let new_row = f_lm_s_i * self.phi_ss + f_li_li_i * phi_lm_s_old;
            self.phi_lm_s_scratch
                .fixed_view_mut::<3, 21>(3 * i, 0)
                .copy_from(&new_row);
            self.phi_li_li[i] = f_li_li_i * self.phi_li_li[i];
        }
        self.phi_ss = f_ss * self.phi_ss;
        std::mem::swap(&mut self.phi_lm_s, &mut self.phi_lm_s_scratch);

        self.accum_dt += dt;
        self.accum_count += 1;
        self.last_b_s = blocks.b_s;
        self.last_b_lm = blocks.b_lm;
    }

    /// `Faster` variant — apply the batched sub-frame transition:
    /// Σ ← Φ·Σ·Φ^T + (Σ dt)·(B_last·InputGain·B_last^T + StateGain), then reset
    /// the accumulator. `B` is held at the sub-frame's last sample (the Phase 6
    /// process-noise approximation — see docs/imu_optimization_plan.md). No-op
    /// when nothing is accumulated, so it is safe to call unconditionally.
    pub fn flush_riccati(&mut self, input_gain: &SMatrix<f64, 12, 12>, state_gain: &DMatrix<f64>) {
        if self.accum_count == 0 {
            return;
        }
        let n = self.xi0.dim();

        // Q ≈ accum_dt · (B_last · InputGain · B_last^T + StateGain)
        let bt = dense_b(&self.last_b_s, &self.last_b_lm);
        let q_input = &bt * input_gain * bt.transpose();
        let state_gain_view = state_gain.view((0, 0), (n, n));
        let q_total = (q_input + state_gain_view) * self.accum_dt;

        let phi_ss = self.phi_ss;
        let phi_lm_s = self.phi_lm_s.clone();
        let phi_li_li = self.phi_li_li.clone();
        self.apply_transport(&phi_ss, &phi_lm_s, &phi_li_li, &q_total);
        self.reset_riccati_accumulator();
    }

    /// Reset the `Faster` accumulator to the identity transition at the current
    /// state dimension.
    fn reset_riccati_accumulator(&mut self) {
        let n_lm = (self.xi0.dim() - VIOSensorState::CDIM) / 3;
        self.phi_ss = SMatrix::<f64, 21, 21>::identity();
        if self.phi_lm_s.nrows() != 3 * n_lm {
            self.phi_lm_s = DMatrix::<f64>::zeros(3 * n_lm, 21);
            self.phi_lm_s_scratch = DMatrix::<f64>::zeros(3 * n_lm, 21);
            self.last_b_lm = DMatrix::<f64>::zeros(3 * n_lm, 12);
        } else {
            self.phi_lm_s.fill(0.0);
        }
        self.phi_li_li.clear();
        self.phi_li_li
            .resize(n_lm, SMatrix::<f64, 3, 3>::identity());
        self.accum_dt = 0.0;
        self.accum_count = 0;
    }

    /// True if any landmark group element has a degenerate scale — the same
    /// predicate `remove_invalid_landmarks` uses. Lets the batched IMU path
    /// flush before a structural change to the covariance.
    pub fn has_degenerate_landmarks(&self) -> bool {
        self.x
            .q
            .iter()
            .any(|q| !q.scale.is_finite() || q.scale <= 1e-8 || q.scale > 1e8)
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

    fn resize_scratch(&mut self) {
        let n = self.xi0.dim();
        if self.scratch_m.nrows() != n {
            self.scratch_m = DMatrix::<f64>::zeros(n, n);
            self.scratch_sigma = DMatrix::<f64>::zeros(n, n);
        }
        // Landmark add/remove is always flush-preceded, so the `Faster`
        // accumulator is empty here — resize/reset it to identity at the new n.
        self.reset_riccati_accumulator();
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
        if y_ids.is_empty() {
            return;
        }

        let n_obs = y_ids.len();
        let xi_hat = self.state_estimate();

        // Innovation vector
        let mut y_tilde = DVector::<f64>::zeros(2 * n_obs);
        for (j, &id) in y_ids.iter().enumerate() {
            let lm_idx = xi_hat
                .camera_landmarks
                .iter()
                .position(|lm| lm.id == id)
                .expect("Landmark ID must exist in state estimate");
            let q = &xi_hat.camera_landmarks[lm_idx].p;
            let y_pred = cam.project(q);
            let y_obs = y_coords
                .get(&id)
                .expect("Observed ID must exist in coordinates");
            y_tilde
                .fixed_rows_mut::<2>(2 * j)
                .copy_from(&(y_obs - y_pred));
        }

        // Output matrix C*
        let ct = suite.output_matrix_C(&self.xi0, &self.x, y_ids, y_coords, cam, use_equivariance);

        // Skip update if C* contains NaN (degenerate landmark)
        if !ct.iter().all(|v| v.is_finite()) {
            return;
        }

        self.perform_stacked_update(suite, &y_tilde, &ct, output_gain, use_discrete_correction);
    }

    /// Bearing + optional stereo log-inverse-range measurement update.
    ///
    /// `stereo_meas` maps landmark id -> `(ell_obs, r_ell)`, where
    /// `ell_obs = -ln(range_s)` (log-inverse-range; the single sign negation from
    /// Rudolf-V's `+log range` lives at the binding, see
    /// stereo_output_matrix_derivation.md §9) and `r_ell = Var(range_s)/range_s^2`.
    /// Observed ids absent from the map get the usual 2-row bearing update; ids
    /// present get an extra log-inverse-range row (Normal chart: `[0,0,+1]`).
    /// An empty map — or no observed stereo id — is byte-identical to
    /// `perform_vision_update`.
    #[allow(clippy::too_many_arguments)]
    pub fn perform_vision_update_with_stereo<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        y_ids: &[u64],
        y_coords: &HashMap<u64, Vector2<f64>>,
        cam: &dyn CameraModel,
        output_gain: &DMatrix<f64>,
        use_equivariance: bool,
        use_discrete_correction: bool,
        stereo_meas: &HashMap<u64, (f64, f64)>,
        range_gate_chi2: f64,
    ) {
        if y_ids.is_empty() {
            return;
        }
        // No observed stereo landmark -> exact bearing-only path.
        if !y_ids.iter().any(|id| stereo_meas.contains_key(id)) {
            self.perform_vision_update(
                suite,
                y_ids,
                y_coords,
                cam,
                output_gain,
                use_equivariance,
                use_discrete_correction,
            );
            return;
        }

        let n = self.xi0.dim();
        let xi_hat = self.state_estimate();
        let sigma_bearing_sq = output_gain[(0, 0)];

        // Row layout: 2 bearing rows per obs, +1 log-inv-range row for stereo obs.
        let total_rows: usize = y_ids
            .iter()
            .map(|id| if stereo_meas.contains_key(id) { 3 } else { 2 })
            .sum();

        let mut y_tilde = DVector::<f64>::zeros(total_rows);
        let mut ct = DMatrix::<f64>::zeros(total_rows, n);
        let mut r_diag = DVector::<f64>::zeros(total_rows);

        let mut row = 0usize;
        for &id in y_ids {
            let pos = self
                .xi0
                .camera_landmarks
                .iter()
                .position(|l| l.id == id)
                .expect("Landmark ID must exist in state");
            let q0 = self.xi0.camera_landmarks[pos].p;
            let q_hat = &self.x.q[pos];
            let q = &xi_hat.camera_landmarks[pos].p;
            let uv = *y_coords.get(&id).expect("Observed ID must exist");
            let col = VIOSensorState::CDIM + 3 * pos;

            // Bearing block (2x3) — same selection as output_matrix_C.
            let ci_star = if use_equivariance {
                suite.output_matrix_ci_star(&q0, q_hat, cam, &uv)
            } else {
                let p_c = q_hat.inverse().act(&q0);
                let y_hat = cam.project(&p_c);
                suite.output_matrix_ci_star(&q0, q_hat, cam, &y_hat)
            };
            ct.fixed_view_mut::<2, 3>(row, col).copy_from(&ci_star);
            y_tilde
                .fixed_rows_mut::<2>(row)
                .copy_from(&(uv - cam.project(q)));
            r_diag[row] = sigma_bearing_sq;
            r_diag[row + 1] = sigma_bearing_sq;
            row += 2;

            // Stereo log-inverse-range row (1x3): ell = -ln‖q‖. Gate on the 1-D
            // innovation (chi²(1)): S = c_ell·Σ_ll·c_ellᵀ + r_ell; drop the row
            // when (ell_obs - ell_pred)² > gate·S (heavy occlusion/depth-edge
            // tail). Bearing rows for this landmark are kept regardless.
            if let Some(&(ell_obs, r_ell)) = stereo_meas.get(&id) {
                let range_row = suite.output_range_row(&q0);
                let ell_pred = -q.norm().ln();
                let d_ell = ell_obs - ell_pred;
                let sigma_ll = self.sigma.fixed_view::<3, 3>(col, col).into_owned();
                let s = (range_row * sigma_ll).dot(&range_row) + r_ell;
                let gated = range_gate_chi2 > 0.0 && s > 0.0 && d_ell * d_ell > range_gate_chi2 * s;
                if !gated {
                    ct.fixed_view_mut::<1, 3>(row, col).copy_from(&range_row);
                    y_tilde[row] = d_ell;
                    r_diag[row] = r_ell;
                }
                // gated => leave the pre-zeroed row (skipped by perform_stacked_update).
                row += 1;
            }
        }

        if !ct.iter().all(|v| v.is_finite()) {
            return;
        }
        let r_noise = DMatrix::from_diagonal(&r_diag);
        self.perform_stacked_update(suite, &y_tilde, &ct, &r_noise, use_discrete_correction);
    }

    // ------------------------------------------------------------------
    // Sequential scalar update (sparse C, symmetric rank-1 downdate)
    // ------------------------------------------------------------------

    pub fn perform_stacked_update<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        residual: &DVector<f64>,
        c_star: &DMatrix<f64>,
        r_noise: &DMatrix<f64>,
        use_discrete_correction: bool,
    ) {
        let m = residual.len();
        if m == 0 {
            return;
        }

        let n = self.xi0.dim();
        let mut gamma = DVector::<f64>::zeros(n);

        for j in 0..m {
            let r_j = r_noise[(j, j)];

            // v = Σ · c_j, exploiting sparsity of c_j (only 3 nonzero entries per row)
            let mut v = DVector::<f64>::zeros(n);
            for col in 0..n {
                let c_jc = c_star[(j, col)];
                if c_jc != 0.0 {
                    v.axpy(c_jc, &self.sigma.column(col), 1.0);
                }
            }

            // α = c_j^T v + r_j
            let mut alpha = r_j;
            for col in 0..n {
                let c_jc = c_star[(j, col)];
                if c_jc != 0.0 {
                    alpha += c_jc * v[col];
                }
            }

            if !alpha.is_finite() || alpha.abs() < 1e-30 {
                continue;
            }

            // Sequential KF re-prediction: subtract the projection of the
            // already-accumulated γ along c_j before applying this scalar
            // update. Without this term, γ = Σ_j K_j · residual_j (an
            // un-adjusted sum) differs from the batch γ = K_batch · y_tilde
            // whenever c_j rows share state columns — which they always do
            // in VIO (u/v of the same landmark, plus the sensor-state
            // band). The error compounds across the m scalar updates and
            // is most pronounced for the Euclidean chart (R³ landmarks,
            // strongest u↔v cross-coupling on the landmark columns).
            let mut c_dot_gamma = 0.0;
            for col in 0..n {
                let c_jc = c_star[(j, col)];
                if c_jc != 0.0 {
                    c_dot_gamma += c_jc * gamma[col];
                }
            }
            let adjusted_residual = residual[j] - c_dot_gamma;

            // Σ -= v vᵀ / α (symmetric rank-1 downdate)
            let inv_alpha = 1.0 / alpha;
            self.sigma.ger(-inv_alpha, &v, &v, 1.0);

            // γ += (adjusted_residual / α) · v
            gamma.axpy(adjusted_residual * inv_alpha, &v, 1.0);
        }

        if !gamma.iter().all(|v| v.is_finite()) {
            return;
        }

        let delta = self.left_correction_increment(suite, &gamma, use_discrete_correction);
        self.x = delta.compose(&self.x);

        self.enforce_spd();
    }

    fn left_correction_increment<S: EqFCoordinateSuite + ?Sized>(
        &self,
        suite: &S,
        gamma: &DVector<f64>,
        use_discrete_correction: bool,
    ) -> VIOGroup {
        if use_discrete_correction {
            let delta = suite.lift_innovation_discrete(gamma, &self.xi0);
            let physical_bias_delta = delta.beta;
            return self.prepare_left_correction_increment(delta, &physical_bias_delta);
        }

        let delta_alg = suite.lift_innovation(gamma, &self.xi0);
        let physical_bias_delta = delta_alg.u_beta;
        let delta = vio_exp_with_bias_group(&delta_alg, self.imu_bias_group);
        self.prepare_left_correction_increment(delta, &physical_bias_delta)
    }

    fn prepare_left_correction_increment(
        &self,
        mut delta: VIOGroup,
        physical_bias_delta: &nalgebra::Vector6<f64>,
    ) -> VIOGroup {
        let ops = BiasGroupOps::new(self.imu_bias_group);
        delta.beta = ops.beta_for_physical_bias_update(
            &self.x,
            &delta,
            &self.xi0.sensor.input_bias,
            physical_bias_delta,
        );
        delta.with_bias_group(self.imu_bias_group)
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
            sigma_new
                .view_mut((0, 0), (n_old, n_old))
                .copy_from(&self.sigma);
            let copy_size = n_added.min(new_cov.nrows());
            sigma_new
                .view_mut((n_old, n_old), (copy_size, copy_size))
                .copy_from(&new_cov.view((0, 0), (copy_size, copy_size)));
            self.sigma = sigma_new;
            self.resize_scratch();
        }
    }

    pub fn remove_landmark_by_id(&mut self, lm_id: u64) {
        if let Some(idx) = self
            .xi0
            .camera_landmarks
            .iter()
            .position(|lm| lm.id == lm_id)
        {
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
                sigma_new
                    .view_mut((0, 0), (start, start))
                    .copy_from(&self.sigma.view((0, 0), (start, start)));
            }
            // Top-right block: [0..start, start..n_new]
            let after = n_new - start;
            if start > 0 && after > 0 {
                sigma_new
                    .view_mut((0, start), (start, after))
                    .copy_from(&self.sigma.view((0, start + 3), (start, after)));
            }
            // Bottom-left block: [start..n_new, 0..start]
            if after > 0 && start > 0 {
                sigma_new
                    .view_mut((start, 0), (after, start))
                    .copy_from(&self.sigma.view((start + 3, 0), (after, start)));
            }
            // Bottom-right block: [start..n_new, start..n_new]
            if after > 0 {
                sigma_new
                    .view_mut((start, start), (after, after))
                    .copy_from(&self.sigma.view((start + 3, start + 3), (after, after)));
            }

            self.sigma = sigma_new;
            self.resize_scratch();
        }
    }

    pub fn remove_invalid_landmarks(&mut self) {
        let invalid_ids: Vec<u64> = self
            .x
            .id
            .iter()
            .zip(self.x.q.iter())
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
        let idx = self
            .xi0
            .camera_landmarks
            .iter()
            .position(|lm| lm.id == lm_id)?;
        let start = VIOSensorState::CDIM + 3 * idx;
        Some(self.sigma.fixed_view::<3, 3>(start, start).into_owned())
    }
}

fn dense_b(b_s: &SMatrix<f64, 21, 12>, b_lm: &DMatrix<f64>) -> DMatrix<f64> {
    let s = VIOSensorState::CDIM;
    let n_lm = b_lm.nrows() / 3;
    let mut b = DMatrix::<f64>::zeros(s + 3 * n_lm, 12);
    b.fixed_view_mut::<21, 12>(0, 0).copy_from(b_s);
    if n_lm > 0 {
        b.view_mut((s, 0), (3 * n_lm, 12)).copy_from(b_lm);
    }
    b
}
