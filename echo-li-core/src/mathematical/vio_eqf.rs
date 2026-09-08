use echo_lie::{SE3, SOT3};
use nalgebra::{DMatrix, DVector, SMatrix, Vector2, Vector3};
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
    // (batched per IMU sub-frame instead of per sample). Identity when empty.
    phi_ss: SMatrix<f64, 21, 21>,
    phi_lm_s: DMatrix<f64>,
    phi_lm_s_scratch: DMatrix<f64>,
    phi_li_li: Vec<SMatrix<f64, 3, 3>>,
    accum_dt: f64,
    accum_count: usize,
    // --- Observability Gramian over a sliding window (0 = disabled) ---------
    // O = sum_k (C_k Psi_k)^T R_k^-1 (C_k Psi_k), restricted to the 21 sensor
    // columns, where Psi_k is the accumulated transition from the window start.
    // This measures how much information the filter has actually EARNED about
    // each sensor direction, which is computable from the Jacobians alone --
    // unlike the error itself. A direction whose reported covariance is smaller
    // than the inverse of its accumulated information is provably over-confident.
    // C has zeros in all 21 sensor columns (vision informs bearings only), so the
    // sensor information arrives entirely through Psi_lm_s, the landmark-sensor
    // coupling built during propagation.
    gram_window: usize,
    gram: DMatrix<f64>,
    gpsi_ss: SMatrix<f64, 21, 21>,
    gpsi_lm_s: DMatrix<f64>,
    gram_ids: Vec<u64>,
    gram_frames: usize,
    gram_resets: usize,
    last_b_s: SMatrix<f64, 21, 12>,
    last_b_lm: DMatrix<f64>,
    // --- Pose-clone window (stochastic cloning; MSCKF-style) ----------------
    // Frozen SE3 camera poses appended to the covariance TAIL, after the
    // landmark blocks: layout `[ sensor(21) | landmark(3)… | clone(6)… ]`. A
    // clone is a past camera pose carried WITH its cross-covariance to the live
    // state, so §V-D can score depth against the honest, gauge-cancelled
    // relative pose Cov(T_clone⁻¹ T_curr) instead of the gauge-inflated absolute
    // camera-pose covariance. Clones have identity self-dynamics (Φ_clone = I)
    // and zero process noise; their cross-covariance evolves under propagation
    // (sensor side only) and measurement updates (rides the Kalman gain for free
    // because clone columns of C are identically zero). Parallel vecs, mirroring
    // `x.id`/`x.q`. Empty ⇒ every covariance path is byte-identical to pre-clone.
    clone_ids: Vec<u64>,
    clone_times: Vec<f64>,
    clone_refcount: Vec<usize>,
    // First-class clone POSE VALUE: the world←camera SE3 snapshot taken at clone
    // time, parallel to `clone_ids`. The passive Phase-0 clone was covariance-only
    // (it borrowed the anchor pose from Sparse3D); the active MSCKF update both
    // READS these poses (multi-view triangulation) and MEAN-CORRECTS them (each
    // clone's 6-DoF row of the stacked-update γ applied as a RIGHT camera
    // perturbation `T ← T·exp(δ̂)`, the same [ω;v] tangent the clone covariance
    // block lives in). Frozen between updates (no propagation).
    clone_poses: Vec<SE3>,
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
            gram_window: 0,
            gram: DMatrix::<f64>::zeros(21, 21),
            gpsi_ss: SMatrix::<f64, 21, 21>::identity(),
            gpsi_lm_s: DMatrix::<f64>::zeros(0, 21),
            gram_ids: Vec::new(),
            gram_frames: 0,
            gram_resets: 0,
            last_b_s: SMatrix::<f64, 21, 12>::zeros(),
            last_b_lm: DMatrix::<f64>::zeros(3 * n_lm, 12),
            clone_ids: Vec::new(),
            clone_times: Vec::new(),
            clone_refcount: Vec::new(),
            clone_poses: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Covariance layout helpers  (single source of truth for the clone tail)
    // ------------------------------------------------------------------

    /// Number of pose clones currently in the window.
    #[inline]
    pub fn n_clones(&self) -> usize {
        self.clone_ids.len()
    }

    /// Ids of the live pose clones, in block order (oldest-inserted first unless a
    /// marginalization has reordered the tail).
    pub fn clone_ids(&self) -> Vec<u64> {
        self.clone_ids.clone()
    }

    /// Number of camera landmarks (the equivariant group's bearings). Clones are
    /// NOT landmarks, so this is derived from `xi0` alone and is unchanged by the
    /// clone window.
    #[inline]
    fn n_landmarks(&self) -> usize {
        (self.xi0.dim() - VIOSensorState::CDIM) / 3
    }

    /// Full covariance dimension including the clone tail: `xi0.dim() + 6·n_clones`.
    /// This — NOT `xi0.dim()` — is the size of `sigma` whenever clones are live,
    /// and is what every covariance-transport/update path must use as `n`.
    #[inline]
    fn cov_dim(&self) -> usize {
        self.xi0.dim() + 6 * self.clone_ids.len()
    }

    /// Embed a physical-block (`xi0.dim()`-square) process-noise matrix into a
    /// full `cov_dim`-square matrix with a zero clone tail. No-op (moves `q_core`
    /// through) when there are no clones.
    #[inline]
    fn pad_process_noise(&self, q_core: DMatrix<f64>) -> DMatrix<f64> {
        let n = self.cov_dim();
        if n == q_core.nrows() {
            return q_core;
        }
        let mut q = DMatrix::<f64>::zeros(n, n);
        let np = q_core.nrows();
        q.view_mut((0, 0), (np, np)).copy_from(&q_core);
        q
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
        let s = VIOSensorState::CDIM;
        let n_lm = self.n_landmarks();
        let n_phys = self.xi0.dim(); // sensor + landmarks (clone tail excluded)

        // DIAGNOSTIC (default-off): MSCEqF-exact Van-Loan discrete process noise.
        // MSCEqF (propagator.cpp:225) forms H=[[A, BWBᵀ],[0,-Aᵀ]], Hd=expm(H·dt),
        // Φ=Hd[0:n,0:n], G=Hd[0:n,n:2n], and Qd = G·Φᵀ (symmetrized). echo's
        // Q=dt·BWBᵀ keeps only the zeroth-order term and DROPS the A-coupled cross
        // terms (dt²·A·BWBᵀ+…) that DECORRELATE velocity from pose. Missing them
        // leaves the nav-vel↔clone correlation ~2× too tight ⇒ over-large
        // structureless nav gain. ECHO_MSC_VANLOAN_Q=1 tests the exact Qd on the
        // 21×21 core (msckf-only ⇒ n_lm=0; falls through to Euler when landmarks
        // are present, which this stage-diff never has).
        if n_lm == 0 && std::env::var("ECHO_MSC_VANLOAN_Q").map(|s| s == "1").unwrap_or(false) {
            let a = blocks.a_ss; // 21×21
            let bwbt: SMatrix<f64, 21, 21> = blocks.b_s * input_gain * blocks.b_s.transpose();
            let mut hmat = DMatrix::<f64>::zeros(42, 42);
            hmat.view_mut((0, 0), (21, 21)).copy_from(&a);
            hmat.view_mut((21, 21), (21, 21)).copy_from(&(-a.transpose()));
            hmat.view_mut((0, 21), (21, 21)).copy_from(&bwbt);
            let hd = echo_lie::matfn::expm(&(hmat * dt));
            let phi = hd.view((0, 0), (21, 21)).into_owned();
            let g = hd.view((0, 21), (21, 21)).into_owned();
            let mut qd = &g * phi.transpose();
            qd = 0.5 * (&qd + qd.transpose()); // symmetrize (MSCEqF selfadjointView)
            let f_ss = SMatrix::<f64, 21, 21>::from_column_slice(phi.as_slice());
            let q_total = self.pad_process_noise(qd);
            self.apply_transport(&f_ss, &DMatrix::<f64>::zeros(0, s), &[], &q_total);
            return;
        }

        // F = I + A·dt has the same block-sparsity as A across all coordinate
        // suites: F_ss (21×21), F_li_s (3×21) per landmark, F_li_li (3×3) per
        // landmark; sensor←landmark and cross-landmark blocks are exactly zero.
        let f_ss: SMatrix<f64, 21, 21> = expm_f_ss(&blocks.a_ss, dt); // ECHO_MSC_EXPM_F diagnostic
        let mut f_lm_s = DMatrix::<f64>::zeros(3 * n_lm, s);
        let mut f_li_li: Vec<SMatrix<f64, 3, 3>> = Vec::with_capacity(n_lm);
        for i in 0..n_lm {
            let block = blocks.a_lm_s.fixed_view::<3, 21>(3 * i, 0).into_owned() * dt;
            f_lm_s.fixed_view_mut::<3, 21>(3 * i, 0).copy_from(&block);
            f_li_li.push(SMatrix::<f64, 3, 3>::identity() + blocks.a_lm_lm[i] * dt);
        }

        // Q_total = dt · (B · InputGain · B^T + StateGain), on the physical block;
        // clones are frozen with zero process noise, so pad the tail with zeros.
        let bt = dense_b(&blocks.b_s, &blocks.b_lm);
        let q_input = &bt * input_gain * bt.transpose();
        let state_gain_view = state_gain.view((0, 0), (n_phys, n_phys));
        let q_core = (q_input + state_gain_view) * dt;
        let q_total = self.pad_process_noise(q_core);

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
        let n = self.cov_dim();
        let s = VIOSensorState::CDIM;
        let n_lm = self.n_landmarks();
        let cs = s + 3 * n_lm; // clone tail start (= xi0.dim())
        let nc6 = 6 * self.clone_ids.len();
        let f_ss_t = f_ss.transpose();
        let f_lm_s_t = f_lm_s.transpose();

        // Psi <- Phi Psi, with Phi = [[F_ss, 0], [F_lm_s, F_li_li]]:
        //   Psi_lm_s' = F_lm_s Psi_ss + F_li_li Psi_lm_s   (before Psi_ss is overwritten)
        //   Psi_ss'   = F_ss Psi_ss
        if self.gram_window > 0 {
            // Landmarks are born and die every vision frame, so a reset on any
            // dimension change would end the window immediately. Instead remap:
            // surviving landmarks keep their accumulated coupling, new ones start
            // at zero (a fresh landmark carries no dependence on the sensor state
            // at the window start), dead ones are dropped. The window survives.
            if self.gram_ids != self.x.id {
                let mut remap = DMatrix::<f64>::zeros(3 * n_lm, 21);
                for (i, id) in self.x.id.iter().enumerate().take(n_lm) {
                    if let Some(j) = self.gram_ids.iter().position(|o| o == id) {
                        if 3 * j + 3 <= self.gpsi_lm_s.nrows() {
                            let src = self.gpsi_lm_s.view((3 * j, 0), (3, 21)).into_owned();
                            remap.view_mut((3 * i, 0), (3, 21)).copy_from(&src);
                        }
                    }
                }
                self.gpsi_lm_s = remap;
                self.gram_ids = self.x.id.clone();
            }
            {
                // f_lm_s * gpsi_ss is Dyn x Const<21>; gpsi_lm_s is fully dynamic.
                let prod = f_lm_s * self.gpsi_ss;
                let mut lm_new =
                    DMatrix::<f64>::from_column_slice(prod.nrows(), 21, prod.as_slice());
                for (i, f_i) in f_li_li.iter().enumerate().take(n_lm) {
                    let cur = lm_new.view((3 * i, 0), (3, 21)).into_owned();
                    let blk = f_i * self.gpsi_lm_s.view((3 * i, 0), (3, 21)) + cur;
                    lm_new.view_mut((3 * i, 0), (3, 21)).copy_from(&blk);
                }
                self.gpsi_lm_s = lm_new;
                self.gpsi_ss = f_ss * self.gpsi_ss;
            }
        }

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

        // ---- Clone tail (frozen, Φ_clone = I) --------------------------------
        // Clones have identity self-dynamics and zero coupling to the sensor, so
        // F = [[F_phys, 0], [0, I]]. Then Σ_new = FΣFᵀ gives, for the clone band:
        //   Σ_new[phys, clone] = F_phys · Σ[phys, clone] = M[phys, clone]  (already
        //       computed by the n-wide Step-1 gemms — no extra multiply)
        //   Σ_new[clone, phys] = Σ_new[phys, clone]ᵀ                (symmetry)
        //   Σ_new[clone, clone] = Σ[clone, clone]                   (frozen)
        // The physical block [0:cs, 0:cs] was fully written above; here we only
        // fill the clone rows/cols that the landmark path leaves stale.
        if nc6 > 0 {
            // DIAGNOSTIC (birth-vs-propagation discriminator): when
            // ECHO_MSC_FREEZE_CLONE_XCOV=1, skip the Φ_nav application to the
            // nav↔clone cross-cov and keep the OLD (birth) value instead of the
            // transported M[0:cs, clone]. Copying old→new each step preserves the
            // cross-cov at whatever it was when the clone was born. The nav block
            // [0:cs,0:cs] still evolves normally. If the att/vel gain split stays
            // ~2.42× the skew is BORN at clone creation; if it collapses toward
            // 1.0 the skew ACCUMULATES in propagation.
            let freeze = std::env::var("ECHO_MSC_FREEZE_CLONE_XCOV")
                .map(|v| v == "1")
                .unwrap_or(false);
            let src = if freeze { &self.sigma } else { &self.scratch_m };
            // Σ_new[0:cs, clone] ← src[0:cs, clone]
            self.scratch_sigma
                .view_mut((0, cs), (cs, nc6))
                .copy_from(&src.view((0, cs), (cs, nc6)));
            // Σ_new[clone, 0:cs] ← (src[0:cs, clone])ᵀ
            let cross_t = src.view((0, cs), (cs, nc6)).transpose();
            self.scratch_sigma
                .view_mut((cs, 0), (nc6, cs))
                .copy_from(&cross_t);
            // Σ_new[clone, clone] ← Σ[clone, clone]  (unchanged; frozen)
            let clone_self = self.sigma.view((cs, cs), (nc6, nc6)).into_owned();
            self.scratch_sigma
                .view_mut((cs, cs), (nc6, nc6))
                .copy_from(&clone_self);
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

        // DIAGNOSTIC (default-off): Euler F=I+A·dt vs exact Van-Loan Φ=expm(A·dt).
        // MSCEqF discretizes the state-transition with the matrix exponential; the
        // Euler first-order form accumulates O(dt²) error that is REFRESHED away in
        // the Q-damped nav auto-cov but survives UNDAMPED in the low-process-noise
        // cross terms (Σ[vel, extrinsics/clone]). ECHO_MSC_EXPM_F=1 swaps in the
        // exact transition to test whether that discretization is the source of the
        // ~2× inflated nav↔clone cross-cov (⇒ over-large structureless nav gain).
        let f_ss: SMatrix<f64, 21, 21> = expm_f_ss(&blocks.a_ss, dt);

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
    /// process-noise approximation: Q is held constant across the sub-frame). No-op
    /// when nothing is accumulated, so it is safe to call unconditionally.
    pub fn flush_riccati(&mut self, input_gain: &SMatrix<f64, 12, 12>, state_gain: &DMatrix<f64>) {
        if self.accum_count == 0 {
            return;
        }
        let n_phys = self.xi0.dim();

        // Q ≈ accum_dt · (B_last · InputGain · B_last^T + StateGain); clones are
        // frozen (zero process noise), so build on the physical block and pad.
        let bt = dense_b(&self.last_b_s, &self.last_b_lm);
        let q_input = &bt * input_gain * bt.transpose();
        let state_gain_view = state_gain.view((0, 0), (n_phys, n_phys));
        let q_core = (q_input + state_gain_view) * self.accum_dt;
        let q_total = self.pad_process_noise(q_core);

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
        let n = self.cov_dim();
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
    /// Rudolf-V's `+log range` lives at the binding, so this channel only ever
    /// sees the canonical sign) and `r_ell = Var(range_s)/range_s^2`, the
    /// first-order propagation of range variance through `ell = -ln(range)`.
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

    /// Restart the Gramian window at the current state dimension.
    fn reset_gramian(&mut self, n_lm: usize) {
        self.gram = DMatrix::<f64>::zeros(21, 21);
        self.gpsi_ss = SMatrix::<f64, 21, 21>::identity();
        self.gpsi_lm_s = DMatrix::<f64>::zeros(3 * n_lm, 21);
        self.gram_ids = self.x.id.clone();
        self.gram_frames = 0;
        self.gram_resets += 1;
    }

    /// Enable (window > 0) or disable (0) observability-Gramian accumulation.
    pub fn enable_gramian(&mut self, window: usize) {
        self.gram_window = window;
        let n_lm = (self.xi0.dim() - VIOSensorState::CDIM) / 3;
        self.reset_gramian(n_lm);
        self.gram_resets = 0;
    }

    /// Accumulated information about the 21-dim sensor state over the current
    /// window, with the number of vision frames it covers. `None` until a full
    /// window has accumulated, so callers never see a partially-filled Gramian.
    pub fn observability_gramian(&self) -> Option<(DMatrix<f64>, usize, usize)> {
        if self.gram_window == 0 {
            return None;
        }
        // Always report, so callers can see how often the window is being cut
        // short by state-dimension changes rather than silently getting nothing.
        Some((self.gram.clone(), self.gram_frames, self.gram_resets))
    }

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

        // `n` spans the full covariance INCLUDING the clone tail: the update
        // vectors and the rank-1 downdate must run over the clone rows so a
        // landmark measurement correctly tightens each clone's cross-covariance
        // (clone columns of C are zero, so this rides the Kalman gain for free).
        // `c_star` may be physical-width (clone columns implicitly zero), so all
        // reads of it are bounded by its own width `ncc`.
        let n = self.cov_dim();
        let ncc = c_star.ncols();

        // --- FULL UPDATE DUMP (env ECHO_UPDATE_DUMP=<path>): first few updates'
        // complete internals for the byte-identical MSCEqF stage-diff. Captures the
        // covariance ENTERING the update (propagation checkpoint) here; S, gain-
        // implied correction, and per-channel norms after the downdate loop below.
        // echo-li sensor tangent order: [0:6]=input_bias, [6:9]=att, [9:12]=pos,
        // [12:15]=vel, [15:21]=camera_offset.
        let dump_path = std::env::var("ECHO_UPDATE_DUMP").ok();
        let dump_this = dump_path.is_some() && {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CNT: AtomicUsize = AtomicUsize::new(0);
            // Cap defaults to 6 (stage-diff snapshots); ECHO_UPDATE_DUMP_N raises
            // it to capture the whole per-update distribution (align_bg/bg_cu).
            let cap = std::env::var("ECHO_UPDATE_DUMP_N")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(6);
            CNT.fetch_add(1, Ordering::Relaxed) < cap
        };
        let pre_cov_diag: Vec<f64> = if dump_this {
            (0..21.min(n)).map(|i| self.sigma[(i, i)]).collect()
        } else {
            Vec::new()
        };
        let pre_sigma: Option<DMatrix<f64>> = if dump_this { Some(self.sigma.clone()) } else { None };
        // PRE-update body-frame gravity direction ĝ_b = R_pre^T·ẑ (matches MSCEqF's
        // qpre). Lets the offline tilt/yaw decomposition split gamma[6:9] (body-frame
        // attitude correction) into TILT-PLANE (⊥ĝ_b) vs YAW (∥ĝ_b) energy.
        let pre_body_g: Vec<f64> = if dump_this {
            let gb = self.state_estimate().sensor.gravity_dir();
            vec![gb[0], gb[1], gb[2]]
        } else {
            Vec::new()
        };

        // --- Observability Gramian: O += (C Psi)^T R^-1 (C Psi) ---------------
        // C's 21 sensor columns are identically zero, so C Psi = C_lm Psi_lm_s.
        let lm_cols = 3 * self.n_landmarks();
        if self.gram_window > 0 && self.gpsi_lm_s.nrows() == lm_cols {
            // (Psi is kept in step with the live landmark set by the remap in
            // apply_transport, so this guard holds except on the very first frame.)
            let s0 = VIOSensorState::CDIM;
            let c_lm = c_star.view((0, s0), (m, lm_cols));
            let cpsi = c_lm * &self.gpsi_lm_s; // m x 21
            for j in 0..m {
                let r_j = r_noise[(j, j)];
                if !(r_j > 0.0) {
                    continue;
                }
                let row = cpsi.row(j);
                let outer = row.transpose() * row / r_j;
                self.gram += outer;
            }
            self.gram_frames += 1;
            if self.gram_frames >= 2 * self.gram_window {
                self.reset_gramian(self.n_landmarks());
            }
        }

        let mut gamma = DVector::<f64>::zeros(n);

        for j in 0..m {
            let r_j = r_noise[(j, j)];

            // v = Σ · c_j, exploiting sparsity of c_j (only 3 nonzero entries per
            // row). `v` spans the full `n` (clone rows included); the column scan
            // is over c_star's own width, and each nonzero pulls the WHOLE Σ
            // column (clone entries too), so v's clone rows carry the correlation.
            let mut v = DVector::<f64>::zeros(n);
            for col in 0..ncc {
                let c_jc = c_star[(j, col)];
                if c_jc != 0.0 {
                    v.axpy(c_jc, &self.sigma.column(col), 1.0);
                }
            }

            // α = c_j^T v + r_j
            let mut alpha = r_j;
            for col in 0..ncc {
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
            for col in 0..ncc {
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

        // --- FULL UPDATE DUMP (part 2): S spectrum, gain-implied correction, and
        // per-channel nav-correction norms, written to ECHO_UPDATE_DUMP.
        if dump_this {
            if let (Some(path), Some(pre)) = (dump_path.as_ref(), pre_sigma.as_ref()) {
                use std::io::Write;
                // S = C·Σ_pre·Cᵀ + R  (m×m innovation covariance, pre-update Σ)
                let c_full = if ncc == n {
                    c_star.clone()
                } else {
                    let mut cf = DMatrix::<f64>::zeros(m, n);
                    cf.view_mut((0, 0), (m, ncc)).copy_from(c_star);
                    cf
                };
                let s_mat = &c_full * pre * c_full.transpose() + r_noise;
                let eig = s_mat.clone().symmetric_eigenvalues();
                let s_trace = s_mat.diagonal().sum();
                let (s_min, s_max) = (
                    eig.iter().cloned().fold(f64::INFINITY, f64::min),
                    eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                );
                // gain-implied full-state correction  dx = Σ_pre·Cᵀ·S⁻¹·residual
                let k_norm;
                // Per-channel gain sub-block Frobenius norms ‖K_ch‖_F where
                // K_ch = Σ[ch,clone]·Cᵀ·S⁻¹ (rows r0..r0+3 of the full gain kmat).
                // CHART- and ORDER-INVARIANT: the nav row is physical (rad / m / m·s⁻¹)
                // and S⁻¹ is in pixel space, so this is directly comparable to MSCEqF
                // regardless of clone chart or observation ordering. corr_ch = ‖K_ch·δ‖,
                // so ‖K_ch‖_F isolates the GAIN magnitude from the residual direction.
                let (mut k_att_fro, mut k_pos_fro, mut k_vel_fro) = (f64::NAN, f64::NAN, f64::NAN);
                // ‖M_clone‖_F where M = CᵀS⁻¹ restricted to the clone rows. Combined with
                // gainfro_ch=‖Σ[ch,clone]·M_clone‖ and cross_fro_ch=‖Σ[ch,clone]‖ this yields the
                // RESIDUAL-INDEPENDENT operator alignment align_op_ch = gainfro_ch/(cross_fro_ch·‖M_clone‖)
                // ∈[0,1]: how much of channel ch's cross-cov energy projects onto M's active row space.
                // gainfro_vel/gainfro_att = (align_op_vel/align_op_att)·(‖Σ[vel,clone]‖/‖Σ[att,clone]‖),
                // so align_op isolates the ORIENTATION (c79 rotation) from the cross-cov magnitude.
                let mut mclone_fro = f64::NAN;
                let mut gg_full = DVector::<f64>::zeros(n);
                let clone_start_m = n.saturating_sub(6 * self.clone_ids.len());
                let dx = if let Some(s_inv) = s_mat.clone().try_inverse() {
                    let gg = c_full.transpose() * (&s_inv * residual); // n
                    let m_full = c_full.transpose() * &s_inv; // n×m  (M = CᵀS⁻¹)
                    let kmat = pre * &m_full; // n×m  (K = Σ·M)
                    k_norm = kmat.norm();
                    let kfro = |r0: usize| -> f64 {
                        let mut s = 0.0;
                        for a in 0..3 {
                            for col in 0..m {
                                let v = kmat[(r0 + a, col)];
                                s += v * v;
                            }
                        }
                        s.sqrt()
                    };
                    k_att_fro = kfro(6);
                    k_pos_fro = kfro(9);
                    k_vel_fro = kfro(12);
                    let mut mf = 0.0;
                    for r in clone_start_m..n {
                        for col in 0..m {
                            let v = m_full[(r, col)];
                            mf += v * v;
                        }
                    }
                    mclone_fro = mf.sqrt();
                    gg_full.copy_from(&gg);
                    pre * gg
                } else {
                    k_norm = f64::NAN;
                    DVector::<f64>::zeros(n)
                };
                // ALIGNMENT DIAGNOSTIC: u = Cᵀ·S⁻¹·r has nonzero entries ONLY on the
                // clone columns (nav has no measurement Jacobian), so dx[ch] =
                // Σ[ch,clone]·u. corr_ch = ‖Σ[ch,clone]·u‖; the norms of Σ[att,clone]
                // and Σ[vel,clone] MATCH MSCEqF but corr_att does not ⇒ the excess must
                // be ALIGNMENT of the cross-cov row with u. align_ch =
                // ‖Σ[ch,clone]·u‖ / (‖Σ[ch,clone]‖_F · ‖u‖) ∈ [0,1] measures it directly.
                let clone_start = n.saturating_sub(6 * self.clone_ids.len());
                let u_clone = gg_full.rows(clone_start, n - clone_start).into_owned();
                let u_norm = u_clone.norm();
                let align_ch = |r0: usize| -> (f64, f64, f64) {
                    // Σ[ch(3),clone] · u  and the Frobenius norm of Σ[ch,clone].
                    let mut cu = [0.0f64; 3];
                    let mut fro = 0.0f64;
                    for a in 0..3 {
                        let mut acc = 0.0;
                        for (j, &uj) in u_clone.iter().enumerate() {
                            let sij = pre[(r0 + a, clone_start + j)];
                            acc += sij * uj;
                            fro += sij * sij;
                        }
                        cu[a] = acc;
                    }
                    let cun = (cu[0] * cu[0] + cu[1] * cu[1] + cu[2] * cu[2]).sqrt();
                    let fro = fro.sqrt();
                    let align = if fro > 0.0 && u_norm > 0.0 { cun / (fro * u_norm) } else { 0.0 };
                    (cun, fro, align)
                };
                let (att_cu, att_fro, att_align) = align_ch(6);
                let (pos_cu, pos_fro, pos_align) = align_ch(9);
                let (vel_cu, vel_fro, vel_align) = align_ch(12);
                // gyro-bias is sensor tangent rows 0:3; align_bg tests whether echo's
                // Σ[bg,clone] coupling row aligns with the innovation u (⇒ systematic
                // bias correction) where MSCEqF's is orthogonal. Compare vs MSCEqF align_bg.
                let (bg_cu, bg_fro, bg_align) = align_ch(0);
                let seg = |lo: usize| {
                    (dx[lo] * dx[lo] + dx[lo + 1] * dx[lo + 1] + dx[lo + 2] * dx[lo + 2]).sqrt()
                };
                let att_tr = pre[(6, 6)] + pre[(7, 7)] + pre[(8, 8)];
                let pos_tr = pre[(9, 9)] + pre[(10, 10)] + pre[(11, 11)];
                let vel_tr = pre[(12, 12)] + pre[(13, 13)] + pre[(14, 14)];
                if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(fh, "=== ECHOLI UPDATE t={:.6} ===", self.current_time);
                    let _ = writeln!(fh, "rows(dof)={m} cols(state)={n} nclones={}", self.clone_ids.len());
                    let head: Vec<f64> = (0..m.min(6)).map(|i| residual[i]).collect();
                    let _ = writeln!(fh, "residual_delta.norm={:.6e} residual_delta.head={:?}", residual.norm(), head);
                    let _ = writeln!(fh, "S.trace={s_trace:.6e} S.eig_min={s_min:.6e} S.eig_max={s_max:.6e}");
                    let _ = writeln!(fh, "K.norm={k_norm:.6e} inn(state-corr).norm={:.6e}", gamma.norm());
                    let _ = writeln!(
                        fh,
                        "corr_att.norm={:.6e} corr_pos.norm={:.6e} corr_vel.norm={:.6e} corr_bias.norm={:.6e}",
                        seg(6), seg(9), seg(12), seg(0)
                    );
                    // APPLIED per-channel correction from `gamma` (= the actual applied
                    // increment `inn`, same object MSCEqF dumps as `inn=K*delta`), NOT the
                    // batch-gain approx `dx` above. This is the apples-to-apples diff vs
                    // MSCEqF's corr_*. seg_g indexes gamma with the sensor tangent order
                    // [bias6, att3, pos3, vel3, camoff6].
                    let seg_g = |lo: usize| {
                        (gamma[lo] * gamma[lo] + gamma[lo + 1] * gamma[lo + 1] + gamma[lo + 2] * gamma[lo + 2]).sqrt()
                    };
                    let _ = writeln!(
                        fh,
                        "applied_att.norm={:.6e} applied_pos.norm={:.6e} applied_vel.norm={:.6e} applied_bias.norm={:.6e}",
                        seg_g(6), seg_g(9), seg_g(12), seg_g(0)
                    );
                    let _ = writeln!(fh, "cov_sensor_diag(bias6,att3,pos3,vel3,camoff6)={pre_cov_diag:?}");
                    let _ = writeln!(fh, "cov_att_trace={att_tr:.6e} cov_pos_trace={pos_tr:.6e} cov_vel_trace={vel_tr:.6e}");
                    let _ = writeln!(
                        fh,
                        "u_clone.norm={u_norm:.6e} align_att={att_align:.6} align_pos={pos_align:.6} align_vel={vel_align:.6} align_bg={bg_align:.6} bg_cu={bg_cu:.6e} bg_fro={bg_fro:.6e}"
                    );
                    let _ = writeln!(
                        fh,
                        "cross_dot_u(att,pos,vel)=({att_cu:.6e},{pos_cu:.6e},{vel_cu:.6e}) cross_fro(att,pos,vel)=({att_fro:.6e},{pos_fro:.6e},{vel_fro:.6e})"
                    );
                    // CHART/ORDER-INVARIANT gain magnitude per nav channel:
                    // ‖K_ch‖_F, K_ch = Σ[ch,clone]·Cᵀ·S⁻¹. Directly comparable to MSCEqF.
                    let _ = writeln!(fh, "gainfro(att,pos,vel)=({k_att_fro:.6e},{k_pos_fro:.6e},{k_vel_fro:.6e})");
                    // RESIDUAL-INDEPENDENT operator alignment (c79 rotation localizer, layout-free):
                    // align_op_ch = gainfro_ch/(cross_fro_ch·‖M_clone‖). If align_op_vel<align_op_att
                    // robustly, Σ[vel,clone] is rotated to project LESS onto M=CᵀS⁻¹'s active subspace
                    // than Σ[att,clone] (with matched cross-cov norms) ⇒ the velocity-under-gain is an
                    // ORIENTATION defect in Σ[vel,clone], not a magnitude one. MSCEqF reference: its
                    // gainfro_vel/att=17.23 with same ‖Σ‖ ratio ⇒ align_op_vel/att≈1.50 (vel MORE aligned).
                    let aop = |gfro: f64, cfro: f64| -> f64 {
                        if cfro > 0.0 && mclone_fro.is_finite() && mclone_fro > 0.0 { gfro / (cfro * mclone_fro) } else { f64::NAN }
                    };
                    let _ = writeln!(
                        fh,
                        "align_op(att,pos,vel)=({:.6},{:.6},{:.6}) mclone_fro={mclone_fro:.6e}",
                        aop(k_att_fro, att_fro), aop(k_pos_fro, pos_fro), aop(k_vel_fro, vel_fro)
                    );
                    let _ = writeln!(fh, "gamma_full.head(21)={:?}", (0..21.min(n)).map(|i| gamma[i]).collect::<Vec<_>>());
                    let _ = writeln!(fh, "pre_body_g={pre_body_g:?}");
                    // MEASUREMENT-SPACE decisive dump (ECHO_MSC_FULLDUMP=1, first update
                    // only): full stacked residual r_o (m) and full innovation S (m×m).
                    // These live in the COMMON pixel-measurement space (same for MSCEqF up
                    // to focal² and obs order), so their DIRECTION / EIGENBASIS can be
                    // compared with NO clone-basis Q (sidesteps the c58 cross-frame trap).
                    // Tests whether S's eigenVECTORS (never checked — c69 only did spectrum)
                    // re-orient u=CᵀS⁻¹r differently than MSCEqF's.
                    if std::env::var("ECHO_MSC_FULLDUMP").as_deref() == Ok("1") {
                        use std::sync::atomic::{AtomicUsize, Ordering};
                        static FCNT: AtomicUsize = AtomicUsize::new(0);
                        if FCNT.fetch_add(1, Ordering::Relaxed) < 4 {
                            let rv: Vec<f64> = (0..m).map(|i| residual[i]).collect();
                            let _ = writeln!(fh, "RESIDUAL_FULL={rv:?}");
                            let _ = writeln!(fh, "S_FULL rows={m}");
                            for r in 0..m {
                                let row: Vec<f64> = (0..m).map(|c| s_mat[(r, c)]).collect();
                                let _ = writeln!(fh, "  {row:?}");
                            }
                        }
                    }
                    // FULL u_clone (=clone rows of Cᵀ·S⁻¹·r) and the CLONE COLUMNS of C.
                    // These are the two inputs (with Σ) that set the correction routing;
                    // Σ matches MSCEqF (GRAM dir-cos ~1) so a mismatch here is the D2
                    // velocity-under-routing. Diff against MSCEqF C(H)/tript after track
                    // reorder. Gated by ECHO_MSC_DUMPC=1 to keep normal dumps small.
                    if std::env::var("ECHO_MSC_DUMPC").as_deref() == Ok("1") {
                        let uc: Vec<f64> = u_clone.iter().copied().collect();
                        let _ = writeln!(fh, "u_clone.full={uc:?}");
                        let ccols = ncc.saturating_sub(clone_start).min(ncc);
                        let _ = writeln!(fh, "C_clone rows={m} clone_cols={ccols} (col offset {clone_start} in state):");
                        for r in 0..m {
                            let row: Vec<f64> = (clone_start..ncc).map(|c| c_star[(r, c)]).collect();
                            let _ = writeln!(fh, " {row:?}");
                        }
                    }
                    // SENSOR-BLOCK velocity correlation coefficients (chart-robust when
                    // marginals match): rho(vel,X) = ||Σ[vel,X]||_F / sqrt(tr Σ_vv · tr Σ_XX),
                    // for X in {att, pos, camoff}. corr_vel is over-large with matched
                    // marginals ⇒ the vel↔pose correlation feeding the newest clone at birth
                    // is the suspect; this dumps it PRE-clone (nav-block only).
                    {
                        let blk_tr = |r0: usize, m: usize| -> f64 { (0..m).map(|k| pre[(r0 + k, r0 + k)]).sum() };
                        let cross_fro = |r0: usize, c0: usize, m: usize| -> f64 {
                            let mut s = 0.0;
                            for r in r0..r0 + 3 { for c in c0..c0 + m { s += pre[(r, c)] * pre[(r, c)]; } }
                            s.sqrt()
                        };
                        let tvel = blk_tr(12, 3);
                        let rho = |c0: usize, m: usize| -> f64 {
                            let d = (tvel * blk_tr(c0, m)).sqrt();
                            if d > 0.0 { cross_fro(12, c0, m) / d } else { 0.0 }
                        };
                        let _ = writeln!(
                            fh,
                            "sensor_rho vel_att={:.4} vel_pos={:.4} vel_camoff={:.4} | cross_fro vel_att={:.6e} vel_pos={:.6e} vel_camoff={:.6e}",
                            rho(6, 3), rho(9, 3), rho(15, 6),
                            cross_fro(12, 6, 3), cross_fro(12, 9, 3), cross_fro(12, 15, 6)
                        );
                    }
                    // ATT-CLONE COVARIANCE CONSISTENCY (ECHO_MSC_ATTCONSIST=1): the
                    // cont.15 marginal-cov paradox — echo's attitude MARGINAL cov is
                    // ~7.5× tighter than MSCEqF yet its correction is ~5× larger, and
                    // the whole correction routes through Σ[att,clone]·u. A valid joint
                    // covariance requires the Schur complement Saa − Sac·Scc⁻¹·Sca ⪰ 0,
                    // i.e. the cross-cov cannot "explain" more attitude variance than the
                    // marginal holds: tr(Sac·Scc⁻¹·Sca) ≤ tr(Saa). If explained/tr(Saa)
                    // > 1 or the Schur min-eig < 0, echo's cross-cov is INCONSISTENT with
                    // its own tight marginal (a covariance-CONSTRUCTION defect) — decisive
                    // WITHOUT needing MSCEqF's internals. If it stays ⪯, the cross-cov is
                    // valid and the defect is the DIRECTION of u (mis-pointed CᵀS⁻¹r).
                    if std::env::var("ECHO_MSC_ATTCONSIST").as_deref() == Ok("1") {
                        let ncl = n - clone_start;
                        if ncl > 0 {
                            let saa = pre.view((6, 6), (3, 3)).into_owned();
                            let scc = pre.view((clone_start, clone_start), (ncl, ncl)).into_owned();
                            let sac = pre.view((6, clone_start), (3, ncl)).into_owned();
                            let tr_saa = saa[(0, 0)] + saa[(1, 1)] + saa[(2, 2)];
                            let scc_eigs = scc.clone().symmetric_eigenvalues();
                            let scc_min = scc_eigs.iter().cloned().fold(f64::INFINITY, f64::min);
                            // regularized inverse for the explained term (Scc is PSD but
                            // may be near-singular with redundant clones)
                            let mut scc_reg = scc.clone();
                            let ridge = 1e-12 * (scc.diagonal().max()).max(1e-30);
                            for i in 0..ncl { scc_reg[(i, i)] += ridge; }
                            if let Some(scc_inv) = scc_reg.try_inverse() {
                                let explained = &sac * &scc_inv * sac.transpose(); // 3×3
                                let tr_expl = explained[(0, 0)] + explained[(1, 1)] + explained[(2, 2)];
                                let schur = &saa - &explained;
                                let schur_eigs = schur.symmetric_eigenvalues();
                                let schur_min = schur_eigs.iter().cloned().fold(f64::INFINITY, f64::min);
                                let saa_min = saa.symmetric_eigenvalues().iter().cloned().fold(f64::INFINITY, f64::min);
                                let _ = writeln!(
                                    fh,
                                    "attconsist tr_Saa={tr_saa:.6e} tr_explained={tr_expl:.6e} explained_ratio={:.4} schur_min_eig={schur_min:.6e} Saa_min_eig={saa_min:.6e} Scc_min_eig={scc_min:.6e} ncl={ncl}",
                                    if tr_saa > 0.0 { tr_expl / tr_saa } else { 0.0 }
                                );
                                // FRESH-CLONE CONFOUND CHECK: per-frame cloning copies the
                                // nav attitude exactly, so the newest clone(s) trivially span
                                // Saa (ratio→1). Recompute the explained ratio DROPPING the
                                // newest k clones (last 6k clone dims). If it stays ≈1 the
                                // attitude cov is genuinely degenerate; if it falls the ratio-1
                                // above was just the fresh-clone replica.
                                let mut ratios_ex = Vec::new();
                                for k in 1..=3usize {
                                    let keep = ncl.saturating_sub(6 * k);
                                    if keep >= 6 {
                                        let scc_k = pre.view((clone_start, clone_start), (keep, keep)).into_owned();
                                        let sac_k = pre.view((6, clone_start), (3, keep)).into_owned();
                                        let mut scc_kr = scc_k.clone();
                                        let rk = 1e-12 * (scc_k.diagonal().max()).max(1e-30);
                                        for i in 0..keep { scc_kr[(i, i)] += rk; }
                                        if let Some(inv_k) = scc_kr.try_inverse() {
                                            let ex_k = &sac_k * &inv_k * sac_k.transpose();
                                            let tr_k = ex_k[(0, 0)] + ex_k[(1, 1)] + ex_k[(2, 2)];
                                            ratios_ex.push(if tr_saa > 0.0 { tr_k / tr_saa } else { 0.0 });
                                        } else { ratios_ex.push(-1.0); }
                                    } else { ratios_ex.push(-1.0); }
                                }
                                let _ = writeln!(fh, "attconsist_dropnewest ratio_drop1={:.4} ratio_drop2={:.4} ratio_drop3={:.4}", ratios_ex[0], ratios_ex[1], ratios_ex[2]);
                            }
                        }
                    }
                    // PER-CLONE cross-cov distribution: mirrors MSCEqF's
                    // `clone[i] xatt/xpos/xvel` lines. Total cross_fro is matched at
                    // U0, but two filters can share a total yet distribute Σ[nav,clone]
                    // differently ACROSS clones — a mis-orientation along the clone
                    // dimension that the total Frobenius hides. Dump per-clone so the
                    // distribution (oldest→newest) is directly comparable.
                    {
                        let blk_fro = |r0: usize, c0: usize| -> f64 {
                            let mut s = 0.0;
                            for r in r0..r0 + 3 { for c in c0..c0 + 6 { s += pre[(r, c)] * pre[(r, c)]; } }
                            s.sqrt()
                        };
                        for (ci, cid) in self.clone_ids.iter().enumerate() {
                            let c0 = clone_start + 6 * ci;
                            if c0 + 6 <= n {
                                let _ = writeln!(
                                    fh,
                                    "clone[{ci}] id={cid} xatt={:.6e} xpos={:.6e} xvel={:.6e}",
                                    blk_fro(6, c0), blk_fro(9, c0), blk_fro(12, c0)
                                );
                            }
                        }
                    }
                    // CHART-INVARIANT clone-space Gram G_ab = Σ_c Σ[a,c]·Σ[b,c]ᵀ (3×3),
                    // nav-pairs {att,vel,pos}. Clone-chart Q cancels ⇒ invariant to
                    // echo-vs-MSCEqF clone-chart diff. Mirrors MSCEqF GRAM_* + NEWCLONE_*.
                    // echo sensor nav order: att[6:9], pos[9:12], vel[12:15].
                    {
                        let ncl_g = self.clone_ids.len();
                        let gram = |ra: usize, rb: usize| -> [f64; 9] {
                            let mut g = [0.0f64; 9];
                            for ci in 0..ncl_g {
                                let c0 = clone_start + 6 * ci;
                                if c0 + 6 > n { continue; }
                                for i in 0..3 { for j in 0..3 {
                                    let mut s = 0.0;
                                    for k in 0..6 { s += pre[(ra + i, c0 + k)] * pre[(rb + j, c0 + k)]; }
                                    g[3 * i + j] += s;
                                }}
                            }
                            g
                        };
                        let fmt = |g: [f64; 9]| format!("[{:.9e},{:.9e},{:.9e};{:.9e},{:.9e},{:.9e};{:.9e},{:.9e},{:.9e}]",
                            g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8]);
                        // cont.141 BIRTH-IDENTITY LOCALIZER: the newest clone's att error
                        // IS the live att error at birth ⇒ Σ[live_att,clone_att] MUST equal
                        // Σ[live_att,live_att] (identity copy, ρ=1, same orientation). Dump the
                        // live att marginal (3×3) and the newest clone's att marginal so the
                        // offline check compares them to NEWCLONE_att[:,0:3] (the cross block).
                        {
                            let marg = |r0: usize, c0: usize| -> [f64; 9] {
                                let mut g = [0.0f64; 9];
                                for i in 0..3 { for j in 0..3 { g[3*i+j] = pre[(r0+i, c0+j)]; } }
                                g
                            };
                            let _ = writeln!(fh, "MARG_att_att={}", fmt(marg(6, 6)));
                            if ncl_g > 0 {
                                let cc = clone_start + 6 * (ncl_g - 1);
                                if cc + 3 <= n {
                                    let _ = writeln!(fh, "NEWCLONE_marg_att={}", fmt(marg(cc, cc)));
                                    // cross block Σ[live_att, clone_att] (3×3), = NEWCLONE_att[:,0:3]
                                    let _ = writeln!(fh, "NEWCLONE_cross_att={}", fmt(marg(6, cc)));
                                }
                            }
                        }
                        let _ = writeln!(fh, "GRAM_att_att={}", fmt(gram(6,6)));
                        let _ = writeln!(fh, "GRAM_att_vel={}", fmt(gram(6,12)));
                        let _ = writeln!(fh, "GRAM_att_pos={}", fmt(gram(6,9)));
                        let _ = writeln!(fh, "GRAM_vel_vel={}", fmt(gram(12,12)));
                        let _ = writeln!(fh, "GRAM_vel_pos={}", fmt(gram(12,9)));
                        let _ = writeln!(fh, "GRAM_pos_pos={}", fmt(gram(9,9)));
                        // NEWEST clone (birth, no transport): raw 3×6 nav cross rows.
                        if ncl_g > 0 {
                            let c0 = clone_start + 6 * (ncl_g - 1);
                            let raw = |r: usize| -> String {
                                let mut v = Vec::new();
                                for i in 0..3 { for k in 0..6 { v.push(format!("{:.9e}", pre[(r + i, c0 + k)])); } }
                                format!("[{}]", v.join(","))
                            };
                            let _ = writeln!(fh, "NEWCLONE_att={}", raw(6));
                            let _ = writeln!(fh, "NEWCLONE_vel={}", raw(12));
                            let _ = writeln!(fh, "NEWCLONE_pos={}", raw(9));
                        }
                        // RAW full nav↔clone cross-cov rows over ALL clones (echo
                        // clone_ids order == u_clone.full order), + the clone id list.
                        // cont.30 Σ·u cross-substitution: with C (Jacobian) and Σ
                        // (clone_pose) BOTH global-left, echo↔MSCEqF clone bases differ
                        // only by ORDER, so permutation-aligning these against MSCEqF's
                        // SIGMA_att_clone lets us compute gamma_att four ways
                        // (Σ_e·u_e, Σ_m·u_m, Σ_m·u_e, Σ_e·u_m) and see whether the ⊥
                        // nav-att correction tracks u (input map) or Σ (routing).
                        {
                            let ids: Vec<String> =
                                self.clone_ids.iter().map(|x| x.to_string()).collect();
                            let _ = writeln!(fh, "CLONEIDS=[{}]", ids.join(","));
                            let raw_all = |r: usize| -> String {
                                let mut v = Vec::new();
                                for ci in 0..ncl_g {
                                    let c0 = clone_start + 6 * ci;
                                    if c0 + 6 > n { continue; }
                                    for i in 0..3 {
                                        for k in 0..6 {
                                            v.push(format!("{:.9e}", pre[(r + i, c0 + k)]));
                                        }
                                    }
                                }
                                format!("[{}]", v.join(","))
                            };
                            let _ = writeln!(fh, "SIGMA_att_clone_full={}", raw_all(6));
                            let _ = writeln!(fh, "SIGMA_pos_clone_full={}", raw_all(9));
                            let _ = writeln!(fh, "SIGMA_vel_clone_full={}", raw_all(12));
                            // cont.32: dump each clone's stored SE3 (world←camera) so
                            // the offline Ad_T⁻¹ test can strip clone_pose's per-clone
                            // adjoint. Row-major 3×3 rotation then 3-vec translation per
                            // clone, in clone_ids order (parallel to SIGMA_*_clone_full).
                            {
                                let mut v = Vec::new();
                                for ci in 0..ncl_g {
                                    let p = &self.clone_poses[ci];
                                    let rm = p.rotation.as_matrix();
                                    for a in 0..3 { for b in 0..3 { v.push(format!("{:.9e}", rm[(a, b)])); } }
                                    for a in 0..3 { v.push(format!("{:.9e}", p.translation[a])); }
                                }
                                let _ = writeln!(fh, "CLONE_POSES={}", format!("[{}]", v.join(",")));
                            }
                        }
                        // POST-downdate Gram on self.sigma (= post-update cov). Pairs with the
                        // GRAM_* above (pre-update, on `pre`) to split the U3 att over-tightening:
                        // POST/PRE att-Gram SV ratio at ONE update isolates the downdate's own
                        // effect (update-math); a normal per-update ratio but a drifting PRE-Gram
                        // across updates isolates propagation. vel/pos serve as matched controls.
                        {
                            let gram_post = |ra: usize, rb: usize| -> [f64; 9] {
                                let mut g = [0.0f64; 9];
                                for ci in 0..ncl_g {
                                    let c0 = clone_start + 6 * ci;
                                    if c0 + 6 > n { continue; }
                                    for i in 0..3 { for j in 0..3 {
                                        let mut s = 0.0;
                                        for k in 0..6 { s += self.sigma[(ra + i, c0 + k)] * self.sigma[(rb + j, c0 + k)]; }
                                        g[3 * i + j] += s;
                                    }}
                                }
                                g
                            };
                            let _ = writeln!(fh, "POSTGRAM_att_att={}", fmt(gram_post(6,6)));
                            let _ = writeln!(fh, "POSTGRAM_att_pos={}", fmt(gram_post(6,9)));
                            let _ = writeln!(fh, "POSTGRAM_pos_pos={}", fmt(gram_post(9,9)));
                            let _ = writeln!(fh, "POSTGRAM_vel_vel={}", fmt(gram_post(12,12)));
                        }
                    }
                    // --- QR-COMPRESSED RECOMPUTE (env ECHO_MSC_QRCOMPRESS=1) ------------
                    // MSCEqF Givens-QR-compresses its stacked system to rows=clone-cols when
                    // rows>cols; echo does NOT. This reduces echo's (H,r) over the clone
                    // columns to the SAME square form, so every quantity is read on MSCEqF's
                    // footing at EVERY frame — not just the accidentally-uncompressed U1.
                    // The point of the exercise is empirical: corr / align / u are the
                    // quantities a LOSSLESS QR preserves (u = CᵀS⁻¹δ is exactly what QR keeps
                    // invariant), so they MUST come out byte-identical to the uncompressed dump
                    // above; ONLY gainfro=‖ΣCᵀS⁻¹‖_F (an uncontracted measurement-row index)
                    // changes with the row count. Confirming that here validates both the
                    // invariance and the harness, and lands gainfro on MSCEqF's row count.
                    if std::env::var("ECHO_MSC_QRCOMPRESS").as_deref() == Ok("1") {
                        let ncl = n - clone_start; // 6*nclones
                        if m > ncl && ncl > 0 {
                            let hc = c_full.view((0, clone_start), (m, ncl)).into_owned(); // m×ncl
                            let qr = hc.qr();
                            let rmat = qr.r(); // ncl×ncl upper-triangular (m>=ncl)
                            let mut qtr = residual.clone(); // Qᵀ·δ  (m), applied in place
                            qr.q_tr_mul(&mut qtr);
                            let zc = qtr.rows(0, ncl).into_owned(); // top ncl
                            let sig2 = r_noise[(0, 0)];
                            // compressed clone-only Jacobian, ncl rows × n cols
                            let mut cfc = DMatrix::<f64>::zeros(ncl, n);
                            cfc.view_mut((0, clone_start), (ncl, ncl)).copy_from(&rmat);
                            let s_c = &cfc * pre * cfc.transpose()
                                + DMatrix::<f64>::identity(ncl, ncl) * sig2;
                            if let Some(s_inv) = s_c.clone().try_inverse() {
                                let u_c_full = cfc.transpose() * (&s_inv * &zc); // n; = CᵀS⁻¹δ
                                let dx_c = pre * &u_c_full; // full-state correction
                                let kmat_c = pre * cfc.transpose() * &s_inv; // n×ncl
                                let uc = u_c_full.rows(clone_start, ncl).into_owned();
                                let uc_norm = uc.norm();
                                let seg_c = |lo: usize| {
                                    (dx_c[lo] * dx_c[lo] + dx_c[lo + 1] * dx_c[lo + 1]
                                        + dx_c[lo + 2] * dx_c[lo + 2])
                                        .sqrt()
                                };
                                let kfro_c = |r0: usize| -> f64 {
                                    let mut s = 0.0;
                                    for a in 0..3 {
                                        for col in 0..ncl {
                                            let v = kmat_c[(r0 + a, col)];
                                            s += v * v;
                                        }
                                    }
                                    s.sqrt()
                                };
                                let align_c = |r0: usize| -> f64 {
                                    let mut cu = [0.0f64; 3];
                                    let mut fro = 0.0f64;
                                    for a in 0..3 {
                                        let mut acc = 0.0;
                                        for (j, &uj) in uc.iter().enumerate() {
                                            let sij = pre[(r0 + a, clone_start + j)];
                                            acc += sij * uj;
                                            fro += sij * sij;
                                        }
                                        cu[a] = acc;
                                    }
                                    let cun = (cu[0] * cu[0] + cu[1] * cu[1] + cu[2] * cu[2]).sqrt();
                                    let fro = fro.sqrt();
                                    if fro > 0.0 && uc_norm > 0.0 { cun / (fro * uc_norm) } else { 0.0 }
                                };
                                let _ = writeln!(
                                    fh,
                                    "COMPRESSED rows={ncl} u_clone.norm={uc_norm:.6e} corr_att={:.6e} corr_pos={:.6e} corr_vel={:.6e} align_att={:.6} align_pos={:.6} align_vel={:.6} gainfro_att={:.6e} gainfro_pos={:.6e} gainfro_vel={:.6e}",
                                    seg_c(6), seg_c(9), seg_c(12),
                                    align_c(6), align_c(9), align_c(12),
                                    kfro_c(6), kfro_c(9), kfro_c(12)
                                );
                            }
                        } else {
                            let _ = writeln!(fh, "COMPRESSED skipped rows={m}<=cols={ncl} (already minimal)");
                        }
                    }
                    // RAW 9×9 NAV COVARIANCE in canonical [att(6:9),vel(12:15),pos(9:12)]
                    // order (echo internal order is att,pos,vel — reordered here to MATCH
                    // MSCEqF's [att,vel,pos]). Off-diagonal 3×3 cross-blocks (att-vel,att-pos,
                    // vel-pos) are built by Euler propagation in echo vs Van Loan in MSCEqF;
                    // this dumps them for the clone-birth cross-cov ORIENTATION diff.
                    {
                        let ord = [6usize, 7, 8, 12, 13, 14, 9, 10, 11]; // att,vel,pos rows/cols
                        let _ = writeln!(fh, "NAVCOV9 order=[att,vel,pos]");
                        for &r in ord.iter() {
                            let row: Vec<String> = ord.iter().map(|&c| format!("{:.6e}", pre[(r, c)])).collect();
                            let _ = writeln!(fh, "  [{}]", row.join(","));
                        }
                    }
                    // PER-CLONE covariance structure by AGE (ci=0 oldest .. newest),
                    // to split birth (stochasticCloning) from transport (apply_transport):
                    // if rot/trans auto-trace and the nav<->clone cross-cov GROW with age,
                    // it's transport re-inflating the clone tail; if flat, it's birth.
                    // Clone tangent block = 6 dims at cb; auto-trace split first-3 / last-3.
                    // xvel/xpos/xatt = ||pre[nav_channel, clone_block]||_F (drives K's nav rows).
                    let cbase = self.xi0.dim();
                    for ci in 0..self.clone_ids.len() {
                        let cb = cbase + 6 * ci;
                        if cb + 6 > n { break; }
                        let tr_a: f64 = (0..3).map(|k| pre[(cb + k, cb + k)]).sum();
                        let tr_b: f64 = (3..6).map(|k| pre[(cb + k, cb + k)]).sum();
                        let xnorm = |r0: usize| -> f64 {
                            let mut s = 0.0;
                            for r in r0..r0 + 3 { for c in cb..cb + 6 { s += pre[(r, c)] * pre[(r, c)]; } }
                            s.sqrt()
                        };
                        let _ = writeln!(
                            fh,
                            "clone[{ci}] auto_tr_rot={tr_a:.6e} auto_tr_trans={tr_b:.6e} xatt={:.6e} xpos={:.6e} xvel={:.6e}",
                            xnorm(6), xnorm(9), xnorm(12)
                        );
                    }
                    // INTER-CLONE gauge: correlation between clone[0] (ref) and clone[cj]
                    // in the ROT (0:3) and TRANS (3:6) sub-blocks. ρ = ‖Σ[block_a,block_b]‖_F /
                    // √(‖Σ[a,a]‖·‖Σ[b,b]‖). A tight attitude gauge (MSCEqF, S≈noise-floor) ⇒
                    // ρ_rot→1; echo's suspected under-correlation ⇒ ρ<1 ⇒ relative-att variance
                    // inflated ⇒ S over-spread. Diagonal-block Frobenius norms as the scale.
                    let bnorm = |r0: usize, c0: usize| -> f64 {
                        let mut s = 0.0;
                        for r in 0..3 { for c in 0..3 { let v = pre[(r0 + r, c0 + c)]; s += v * v; } }
                        s.sqrt()
                    };
                    if self.clone_ids.len() >= 1 {
                        let c0 = cbase; // clone[0] reference block
                        for cj in 0..self.clone_ids.len() {
                            let cb = cbase + 6 * cj;
                            if cb + 6 > n { break; }
                            let rr0 = bnorm(c0, c0);
                            let rrj = bnorm(cb, cb);
                            let rr0j = bnorm(c0, cb);
                            let tt0 = bnorm(c0 + 3, c0 + 3);
                            let ttj = bnorm(cb + 3, cb + 3);
                            let tt0j = bnorm(c0 + 3, cb + 3);
                            let rho_rot = if rr0 > 0.0 && rrj > 0.0 { rr0j / (rr0 * rrj).sqrt() } else { 0.0 };
                            let rho_tr = if tt0 > 0.0 && ttj > 0.0 { tt0j / (tt0 * ttj).sqrt() } else { 0.0 };
                            let _ = writeln!(
                                fh,
                                "interclone[0,{cj}] rho_rot={rho_rot:.6e} rho_trans={rho_tr:.6e} xrot_fro={rr0j:.6e} xtrans_fro={tt0j:.6e}"
                            );
                        }
                    }
                }
            }
        }

        // DIAGNOSTIC (ECHO_MSC_SCALELEAK=1): the monocular MSC constraint is
        // provably scale-invariant (Jπ·q=0), so a consistent update must place
        // ZERO body-velocity correction along the velocity direction (the
        // scale channel). Project γ's velocity block [12:15] onto v̂ (leak =
        // scale injection) vs perpendicular (legitimate, observable). Prints
        // one line/fire: |along| |perp| |along/mag|. Aggregate downstream.
        if std::env::var("ECHO_MSC_SCALELEAK").as_deref() == Ok("1") {
            let vel = self.state_estimate().sensor.velocity;
            let vn = vel.norm();
            if vn > 1e-6 {
                let vhat = vel / vn;
                let gv = gamma.fixed_rows::<3>(12).into_owned();
                let along = gv.dot(&vhat);
                let perp = (gv - vhat * along).norm();
                let mag = gv.norm();
                let frac = if mag > 1e-12 { along.abs() / mag } else { 0.0 };
                eprintln!(
                    "SCALELEAK along={:.6e} perp={:.6e} frac_along={:.4} |gv|={:.6e} |v|={:.4}",
                    along, perp, frac, mag, vn
                );
            }
        }

        // CAUSAL TEST (ECHO_MSC_KILLSCALELEAK=1): project the along-v̂ component out
        // of γ's body-velocity block [12:15], i.e. subtract (γ_v·v̂)v̂. The
        // monocular MSC constraint carries no scale information, so removing the
        // scale-direction velocity correction turns the SCALELEAK measurement into
        // an intervention — if est/gt returns to ~1, the along-v̂ leak IS the
        // divergence driver; if it still runs away, attitude/position share blame.
        if std::env::var("ECHO_MSC_KILLSCALELEAK").as_deref() == Ok("1") {
            let vel = self.state_estimate().sensor.velocity;
            let vn = vel.norm();
            if vn > 1e-6 {
                let vhat = vel / vn;
                let gv = gamma.fixed_rows::<3>(12).into_owned();
                let along = gv.dot(&vhat);
                let gv_perp = gv - vhat * along;
                gamma.fixed_rows_mut::<3>(12).copy_from(&gv_perp);
            }
        }

        // POSITION-SCALE PROJECTION (ECHO_MSC_KILLPOSSCALE=1): the c121 seed is a
        // DIRECTION-CLEAN positive per-step POSITION over-length (step-ratio 1.01→1.20
        // fr1-120, cos-to-GT 0.989) — a pure scale injection by the update, which
        // KILLSCALELEAK removes from velocity only. Over one frame the position
        // displacement ≈ v·dt, so v̂ is (to first order) the scale direction of the
        // position correction too. This projects the along-v̂ component out of γ's
        // position block [9:12], i.e. subtract (γ_p·v̂)v̂. Use WITH KILLSCALELEAK to
        // remove scale from the FULL nav correction (pos+vel). TEST: if fr1-120
        // step-ratio returns to ~1.0 and full-run est/gt collapses toward MSCEqF's
        // 1.10×, the update-injected scale-gauge leak IS the divergence seed and the
        // principled fix is scale-gauge projection; if the step-ratio persists, the
        // scale direction is NOT along-v̂ (e.g. position-relative-to-anchor) and the
        // next step re-derives q from the measurement nullspace. Diagnostic only.
        if std::env::var("ECHO_MSC_KILLPOSSCALE").as_deref() == Ok("1") {
            let vel = self.state_estimate().sensor.velocity;
            let vn = vel.norm();
            if vn > 1e-6 {
                let vhat = vel / vn;
                let gp = gamma.fixed_rows::<3>(9).into_owned();
                let along = gp.dot(&vhat);
                let gp_perp = gp - vhat * along;
                gamma.fixed_rows_mut::<3>(9).copy_from(&gp_perp);
            }
        }

        // CHANNEL DECOMPOSITION (ECHO_MSC_ZERO_ATT/POS/VEL=1): zero one sensor
        // sub-block of the nav mean-correction to localize which channel carries
        // the divergence. Attitude [6:9], position [9:12], velocity [12:15].
        // Independent of KILLSCALELEAK (which removes only the scale DIRECTION of
        // velocity); ZERO_VEL removes the whole velocity correction.
        if std::env::var("ECHO_MSC_ZERO_ATT").as_deref() == Ok("1") {
            gamma.fixed_rows_mut::<3>(6).fill(0.0);
        }
        if std::env::var("ECHO_MSC_ZERO_POS").as_deref() == Ok("1") {
            gamma.fixed_rows_mut::<3>(9).fill(0.0);
        }
        if std::env::var("ECHO_MSC_ZERO_VEL").as_deref() == Ok("1") {
            gamma.fixed_rows_mut::<3>(12).fill(0.0);
        }
        // CAUSALITY TEST (ECHO_MSC_VELSCALE=f): scale the whole velocity mean-correction
        // gamma[12:15] by f every update. cont.115 measured UPD0 corr_vel ≈0.80× MSCEqF
        // but sitting on a near-total cancellation (coherence ~0.10, hypersensitive to
        // sub-5% cov diffs). This intervention tests whether that deficit is the ATE
        // driver: if boosting velocity correction (f≈1.25) collapses full-traj est/gt
        // toward MSCEqF's ~1.1, corr_vel IS causal → target vel-block cov propagation;
        // if est/gt is unchanged, corr_vel is a RED HERRING. Diagnostic only, never shipped.
        if let Ok(vs) = std::env::var("ECHO_MSC_VELSCALE") {
            if let Ok(f) = vs.parse::<f64>() {
                let gv = gamma.fixed_rows::<3>(12).into_owned();
                gamma.fixed_rows_mut::<3>(12).copy_from(&(gv * f));
            }
        }
        // GYRO-BIAS TEST (ECHO_MSC_ZERO_BG=1): zero only the gyro-bias mean-correction
        // gamma[0:3], leaving the prior/covariance intact (unlike freezing biasGyr
        // initVar→0). NOTE (cont.9,12,14): the gyro-bias channel was EXONERATED as a
        // divergence driver — echo's per-update |Δb_w| MATCHES MSCEqF (med 2.4e-4 vs
        // 2.9e-4, both low-coherence random-walk); the earlier "spurious ≈0.03 rad/s /
        // true ≈0.0005 / 365%→12.6%" story is REFUTED. The localized root is the
        // ATTITUDE correction gamma[6:9] (see ATTSCALE below and the topic-file log),
        // not this channel. Kept only as a control ablation.
        if std::env::var("ECHO_MSC_ZERO_BG").as_deref() == Ok("1") {
            gamma.fixed_rows_mut::<3>(0).fill(0.0);
        }
        // ACCEL-BIAS FREEZE (ECHO_MSC_ZERO_BA=1): zero the accel-bias mean-correction
        // gamma[3:6]. Companion to ZERO_BG. cont.7 measured echo's |b_a| runs to 0.7
        // (12x MSCEqF 0.044) — the attitude-independent velocity-PROPAGATION scale
        // driver. ZERO_BG alone worsens (residual floods b_a+vel); this tests whether
        // freezing BOTH bias channels together collapses the runaway toward GTVEL's
        // 1.9% → would prove the bias-channel UPDATE GAIN is the whole bug.
        if std::env::var("ECHO_MSC_ZERO_BA").as_deref() == Ok("1") {
            gamma.fixed_rows_mut::<3>(3).fill(0.0);
        }

        // MACRO CAUSAL TEST (ECHO_MSC_CORRSCALE=<f>): scale the whole nav
        // mean-correction (attitude[6:9], position[9:12], velocity[12:15]) by f.
        // At U0 the Euclidean-chart correction is ~0.5–0.58× MSCEqF (att 0.46, vel
        // 0.58, pos 0.51) — a ~uniform under-correction that compounds into the
        // est_len/gt_len 14.5× scale runaway. If f≈1.7 collapses the ATE toward
        // MSCEqF's, the per-update under-correction IS the divergence driver; if not,
        // the 365% is dominated by track-set drift / feedback, not gain magnitude.
        if let Ok(s) = std::env::var("ECHO_MSC_CORRSCALE") {
            if let Ok(f) = s.parse::<f64>() {
                for i in 6..15 {
                    gamma[i] *= f;
                }
            }
        }

        // ATTITUDE-CHANNEL CAUSAL TEST (ECHO_MSC_ATTSCALE=<f>): scale ONLY the
        // attitude sub-block gamma[6:9] by f (position/velocity untouched). The
        // GT-attitude-reset ladder proved that a PERFECT attitude correction
        // collapses the runaway (ATE 52%→4.3% @1500f). This asks the follow-up:
        // does boosting echo's OWN attitude correction (no GT) bound the tilt? If
        // f>1 recovers most of the GTATT benefit, the deficit is attitude-channel
        // MAGNITUDE (corr_att 0.46× under at U0); if it doesn't help / worsens, the
        // attitude correction is DIRECTIONALLY wrong (routing/align_att) and a scalar
        // boost can't fix it — needs the mis-route fix.
        // RESULT (cont.14): DIRECTIONAL. Sign-flip (f=-1) stays anti-corrective and
        // scale-down (f=0.5) just fades toward ZERO_ATT — echo's gamma[6:9] ADDS
        // gravity-tilt where MSCEqF's reduces/holds it (~45× tilt-growth gap), so
        // gamma[6:9] is ~ORTHOGONAL to the tilt-reducing direction. A scalar can't fix
        // it; the fix is in the gain routing (residual r / clone att-Jacobian C /
        // Σ[att,clone]).
        if let Ok(s) = std::env::var("ECHO_MSC_ATTSCALE") {
            if let Ok(f) = s.parse::<f64>() {
                for i in 6..9 {
                    gamma[i] *= f;
                }
            }
        }

        // Physical block [sensor | landmarks] drives the EqF innovation lift.
        let gamma_phys = if gamma.len() == self.xi0.dim() {
            gamma.clone()
        } else {
            gamma.rows(0, self.xi0.dim()).into_owned()
        };
        let delta = self.left_correction_increment(suite, &gamma_phys, use_discrete_correction);

        // FILTER-FORM PROBE. The EqF correction `x ← delta·x` is a LEFT group
        // action, so the accumulated pose translation transforms as
        // `t_x ← R_δ·t_x + t_δ` (SE3::compose): the attitude correction R_δ ROTATES
        // the whole accumulated position vector t_x (distance-from-origin), a term
        // `(R_δ−I)·t_x ≈ |ω_att|·|t_x|` that OpenVINS's ADDITIVE boxplus
        // `p ← p + δp` does NOT have and that GROWS as the drone travels.
        // ECHO_MSC_POSJUMP=1 measures it; ECHO_MSC_ADDITIVE_POS=1 removes it
        // (keeps the additive translation `t_x + t_δ`) to test causality — the
        // one-term discriminator between the EqF-lift and the OV additive form.
        let mut composed = delta.compose(&self.x);
        if std::env::var("ECHO_MSC_POSJUMP").as_deref() == Ok("1") {
            let t_x = self.x.a.translation;
            let rot_jump = delta.a.rotation.act(&t_x) - t_x; // (R_δ − I)·t_x
            let ang = delta.a.rotation.log().norm();
            eprintln!(
                "POSJUMP dist={:.4} rot_jump={:.6e} direct={:.6e} ang={:.6e}",
                t_x.norm(),
                rot_jump.norm(),
                delta.a.translation.norm(),
                ang
            );
        }
        if std::env::var("ECHO_MSC_ADDITIVE_POS").as_deref() == Ok("1") {
            composed.a.translation = self.x.a.translation + delta.a.translation;
        }
        self.x = composed;

        // Faithful OpenVINS mirror: mean-correct each stored clone pose by its own
        // 6-dof increment in the clone tail. The clone chart is the right camera-
        // frame perturbation `T ← T·exp([ω;v])` (rotation-first) matching
        // `sparse_camera_pose_jacobian` / the clone covariance block, so the sign
        // is `+γ`. This is a true no-op when no clones exist (the loop body never
        // runs) and — for a landmark/vision update whose `c_star` has zero clone
        // columns — the clone rows of γ are still filled through the cross-
        // covariance `v = Σ·c`, so a plain EqF update tightens AND recenters the
        // window exactly as OpenVINS's `EKFUpdate` corrects every `Hx_order` var.
        // ECHO_MSC_LEFTCHART=1: apply the clone increment as a LEFT/global action
        // `T ← exp(γ^)·T` (matching MSCEqF `updateLeft` and the left clone cov chart
        // set in `clone_pose`), so a global attitude correction rotates the clone
        // window and the nav pose consistently. Default is the right camera-frame
        // action `T ← T·exp(γ^)`.
        let leftchart = std::env::var("ECHO_MSC_LEFTCHART").as_deref() == Ok("1");
        let lm_end = self.xi0.dim();
        for ci in 0..self.clone_ids.len() {
            let base = lm_end + 6 * ci;
            let d = gamma.fixed_rows::<6>(base).into_owned();
            self.clone_poses[ci] = if leftchart {
                SE3::exp(&d).compose(&self.clone_poses[ci])
            } else {
                self.clone_poses[ci].compose(&SE3::exp(&d))
            };
        }

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
        // MSCEqF applies the RAW lifted bias correction as a group element
        // (`Dd.multiplyLeft(SDB::exp(inn))`, bias slot = leftJacobian·inn[bg,ba]),
        // then reads the physical bias through `phi` (= act_bias). The SDB readout
        // `B(X)^-1.adj·(b0 - delta)` transports the accumulated correction through the
        // EVOLVING attitude, so spurious per-update bias corrections mean-revert.
        //
        // echo's default (`beta_for_physical_bias_update`) instead engineers `delta.beta`
        // so that the physical bias moves by EXACTLY `physical_bias_delta` each update
        // (telescoping ad_delta·ad_current: b_new = b_old + physical_bias_delta). That
        // makes the physical bias the running SUM of corrections — inconsistent with the
        // SDB-tangent covariance. (cont.9,12,14: the once-suspected "~0.03 rad/s gyro
        // bias drives the divergence" is REFUTED — echo's per-update |Δb_w| matches
        // MSCEqF; the root is the ATTITUDE correction, not this bias readout. Kept as a
        // conventions option.) ECHO_SDB_RAW_BIAS=1 selects the MSCEqF-raw application
        // (leave delta.beta = the lifted correction).
        let raw = std::env::var("ECHO_SDB_RAW_BIAS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        if !raw {
            delta.beta = ops.beta_for_physical_bias_update(
                &self.x,
                &delta,
                &self.xi0.sensor.input_bias,
                physical_bias_delta,
            );
        }
        delta.with_bias_group(self.imu_bias_group)
    }

    // ------------------------------------------------------------------
    // Landmark management
    // ------------------------------------------------------------------

    pub fn add_new_landmarks(&mut self, new_landmarks: Vec<Landmark>, new_cov: &DMatrix<f64>) {
        // Landmark blocks live in `[s, xi0.dim())`, immediately BEFORE the clone
        // tail. New landmarks are pushed to the end of that region, so their
        // covariance rows/cols must be INSERTED at `lm_end` (= old `xi0.dim()`),
        // pushing the clone tail down — not appended at the very end (which would
        // land inside the clone block). Fresh landmarks are uncorrelated with
        // everything, including clones.
        let lm_end = self.xi0.dim(); // old physical dim (sensor + old landmarks)
        let nc6 = 6 * self.clone_ids.len();
        let old_dim = lm_end + nc6; // = old covariance dim

        for lm in new_landmarks {
            self.xi0.camera_landmarks.push(lm.clone());
            self.x.q.push(SOT3::identity());
            self.x.id.push(lm.id);
        }

        let n_added = self.xi0.dim() - lm_end;
        if n_added == 0 {
            return;
        }
        let new_dim = old_dim + n_added;

        let mut sigma_new = DMatrix::<f64>::zeros(new_dim, new_dim);
        // Sensor + old landmarks, unchanged.
        sigma_new
            .view_mut((0, 0), (lm_end, lm_end))
            .copy_from(&self.sigma.view((0, 0), (lm_end, lm_end)));
        // New landmark diagonal block (fresh; zero cross-cov elsewhere).
        let copy_size = n_added.min(new_cov.nrows());
        sigma_new
            .view_mut((lm_end, lm_end), (copy_size, copy_size))
            .copy_from(&new_cov.view((0, 0), (copy_size, copy_size)));
        // Clone tail + its cross-cov to sensor/old-landmarks, shifted down by
        // `n_added` (only present when the clone window is non-empty).
        if nc6 > 0 {
            sigma_new
                .view_mut((lm_end + n_added, lm_end + n_added), (nc6, nc6))
                .copy_from(&self.sigma.view((lm_end, lm_end), (nc6, nc6)));
            sigma_new
                .view_mut((0, lm_end + n_added), (lm_end, nc6))
                .copy_from(&self.sigma.view((0, lm_end), (lm_end, nc6)));
            sigma_new
                .view_mut((lm_end + n_added, 0), (nc6, lm_end))
                .copy_from(&self.sigma.view((lm_end, 0), (nc6, lm_end)));
        }
        self.sigma = sigma_new;
        self.resize_scratch();
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

            // Post-removal covariance dim INCLUDING the clone tail. The block
            // copies below read from `start + 3`, which transparently shifts the
            // clone tail (part of the "after" region) up by 3.
            let n_new = self.xi0.dim() + 6 * self.clone_ids.len();

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
    // Pose-clone window (stochastic cloning)
    // ------------------------------------------------------------------

    /// Append a frozen clone of the CURRENT camera pose to the covariance tail,
    /// with honest cross-covariance to the whole live state (stochastic cloning).
    ///
    /// The clone block is stored in camera-pose-error coordinates (6-DoF SE3
    /// tangent, `[ω; v]`), so with `j = sparse_camera_pose_jacobian` (6×21,
    /// mapping sensor error → camera-pose error):
    ///   Σ[clone, clone] = j Σ_ss jᵀ                 (current camera-pose cov)
    ///   Σ[clone, k]     = j Σ[0:21, k]   ∀ columns k (cross-cov, COPIED not zeroed)
    /// Unlike `add_new_landmarks` (fresh landmark ⇒ zero cross-cov), the copied
    /// cross-cov is exactly what makes the later relative-pose covariance
    /// gauge-cancel. `clone_id` must be unique among live clones. Idempotent-safe:
    /// re-cloning an existing id is a no-op (returns the existing block).
    ///
    /// `pose` is the world←camera SE3 value at clone time, stored as the clone's
    /// first-class pose (read by multi-view triangulation, mean-corrected by MSC
    /// updates). The clone covariance block lives in the RIGHT camera-pose
    /// perturbation of this stored value (`[ω; v]` tangent), the same convention
    /// `sparse_camera_pose_jacobian` (= `j`) targets.
    pub fn clone_pose(&mut self, clone_id: u64, time: f64, j: &SMatrix<f64, 6, 21>, pose: SE3) {
        if self.clone_ids.contains(&clone_id) {
            return;
        }
        let old = self.cov_dim();
        let new = old + 6;

        let mut sigma_new = DMatrix::<f64>::zeros(new, new);
        sigma_new
            .view_mut((0, 0), (old, old))
            .copy_from(&self.sigma);

        // LEFT/GLOBAL-CHART PORT (ECHO_MSC_LEFTCHART=1): the clone covariance block
        // is stored in the SAME left/global perturbation as MSCEqF (`E → exp(δ^)E`)
        // and as the nav state, instead of ECHO-LI's default right camera-frame
        // perturbation (`T·exp(δ^)`). A right tangent δ_r and left tangent δ_l of the
        // same pose T satisfy δ_l = Ad_T·δ_r, so the left-chart clone block is the
        // right-chart block pre/post-multiplied by the pose adjoint Ad_T. This makes
        // the clone window co-rotate with a global attitude correction exactly like
        // the nav pose (removing the mixed left-nav/right-clone coupling).
        let leftchart = std::env::var("ECHO_MSC_LEFTCHART").as_deref() == Ok("1");
        let ad_t = pose.adjoint(); // Ad_T of the clone camera pose (world←camera)

        // NAVLEFT (ECHO_MSC_NAVLEFT=1, requires LEFTCHART=1): the EqF covariance and
        // the LEFT nav correction (`X ← exp(γ^)·X`) put the nav BODY pose in the LEFT
        // (global) perturbation `T_body ← exp(δ^)·T_body`. But the birth Jacobian `j`
        // uses the RIGHT camera-frame convention (body block = Ad_{offset⁻¹}, per its
        // own doc). Under a LEFT nav chart the body→camera coupling is IDENTITY:
        // `exp(δ^)·T_body·T_offset = exp(δ^)·T_cam`, so a left-body perturbation IS a
        // left-camera perturbation — no Ad_T. The default path instead forms
        // `ad_t·j = Ad_{T_body}·Σ[body]`, a spurious Ad_{T_body} rotation of the
        // nav↔clone cross that preserves its NORM (⇒ cont.46 norm-match) but rotates
        // its ORIENTATION ~perpendicular (⇒ TRAJDUMP cos_dir≈−0.10 near-orthogonal,
        // invariant to every update knob per cont.53). NAVLEFT replaces j's body block
        // with Identity and drops ad_t so the stored left-clone block is Cov(δ_body_l,·)
        // directly. HYPOTHESIS UNDER TEST — gate on TRAJDUMP cos_dir>0 & full ATE.
        let navleft = leftchart
            && std::env::var("ECHO_MSC_NAVLEFT").as_deref() == Ok("1");
        let mut j_eff = *j;
        if navleft {
            // body block (rows 0:6, cols 6:12) Ad_{offset⁻¹} → Identity (left convention)
            j_eff
                .fixed_view_mut::<6, 6>(0, 6)
                .copy_from(&SMatrix::<f64, 6, 6>::identity());
        }

        // cross = Σ[clone, 0:old] = j · Σ[0:21, 0:old]   (6 × old)
        let sigma_top = self.sigma.view((0, 0), (21, old));
        let mut cross = j_eff * sigma_top;
        if leftchart && !navleft {
            cross = &ad_t * cross; // left-chart: δ_l = Ad_T·δ_r
        }
        sigma_new.view_mut((old, 0), (6, old)).copy_from(&cross);
        sigma_new
            .view_mut((0, old), (old, 6))
            .copy_from(&cross.transpose());

        // clone-self = j Σ_ss jᵀ (symmetrized)
        let sigma_ss = self.sigma.fixed_view::<21, 21>(0, 0).into_owned();
        let mut clone_self = j_eff * sigma_ss * j_eff.transpose();
        if leftchart && !navleft {
            clone_self = ad_t * clone_self * ad_t.transpose();
        }
        clone_self = 0.5 * (clone_self + clone_self.transpose());
        sigma_new
            .view_mut((old, old), (6, 6))
            .copy_from(&clone_self);

        // DIAGNOSTIC (default-off, byte-identical at 1.0): inflate this clone's
        // marginal covariance by ECHO_MSC_CLONE_INFLATE (a σ-factor f). Self-block
        // ×f², cross-cov ×f — a similarity scaling that makes the clone f× more
        // uncertain while preserving every correlation coefficient (so the gauge-
        // cancelling relative-pose structure is untouched). Forward form of the
        // validated k×-under-statement diagnostic: field clone NEES ⇒ the stored
        // clone cov under-states true clone inconsistency by ~6-7×σ (attitude-
        // driven over-confidence propagated from the EqF pose cov). Tests whether
        // the active-MSC over-tightening (K too large ⇐ S too small) is purely
        // clone-cov MAGNITUDE. Cross view spans cols 0..old (disjoint from the
        // self block at col `old`), so f and f² never compound on one entry.
        let env_f = |k: &str, d: f64| {
            std::env::var(k).ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(d)
        };
        let iso = env_f("ECHO_MSC_CLONE_INFLATE", 1.0);
        // Per-channel σ-factors on the clone [ω(0..3); v(3..6)] tangent: rotation
        // ×f_r, translation ×f_t (each falls back to the isotropic factor). Lets a
        // sweep test whether the residual is a clone-cov SHAPE issue (translation
        // over-confident at long lag while rotation is fine) vs a deeper live-nav-
        // cov-growth ceiling that birth-time inflation can't reach.
        let f_r = env_f("ECHO_MSC_CLONE_INFLATE_ROT", iso);
        let f_t = env_f("ECHO_MSC_CLONE_INFLATE_TRANS", iso);
        if f_r != 1.0 || f_t != 1.0 {
            let d = [f_r, f_r, f_r, f_t, f_t, f_t]; // per-dof σ-factor
            // cross-cov rows (clone ↔ rest): scale row i by d[i].
            let mut cr = sigma_new.view((old, 0), (6, old)).into_owned();
            for i in 0..6 {
                cr.row_mut(i).scale_mut(d[i]);
            }
            sigma_new.view_mut((old, 0), (6, old)).copy_from(&cr);
            sigma_new.view_mut((0, old), (old, 6)).copy_from(&cr.transpose());
            // self-block: entry (i,j) scales by d[i]·d[j] (σ-similarity ⇒ variance
            // ×d²; preserves correlation coefficients within the block).
            let mut cs = sigma_new.view((old, old), (6, 6)).into_owned();
            for i in 0..6 {
                for j in 0..6 {
                    cs[(i, j)] *= d[i] * d[j];
                }
            }
            sigma_new.view_mut((old, old), (6, 6)).copy_from(&cs);
        }

        // DIAGNOSTIC (default-off, byte-identical at 1.0): ECHO_MSC_CLONE_INTER_TRANS
        // (g_t) scales ONLY the clone↔clone translation coupling — the clone-tail
        // columns (>=21) of this clone's cross row by g_t on the trans dofs, and the
        // self-block trans×trans by g_t² — while LEAVING the nav↔clone cross
        // (cols 0..21, matched to MSCEqF c65-67) untouched. This isolates the
        // cont.86 finding (cross-clone xtrans_fro ~1.4x MSCEqF) from the nav↔clone
        // confound that made ECHO_MSC_CLONE_INFLATE_TRANS inconclusive (deflating it
        // guts every nav correction since nav is corrected ONLY via Σ[nav,clone]).
        // Since Σ[nav,clone], C, and S's C-part all match MSCEqF, Σ_cc (clone self +
        // clone↔clone) is the ONLY thing the persistent-E port would change — so this
        // knob is a cheap proxy for the E-port's entire gain effect. g_t≈0.71 pulls
        // echo's clone↔clone trans down to ≈MSCEqF's level.
        let g_t = env_f("ECHO_MSC_CLONE_INTER_TRANS", 1.0);
        if g_t != 1.0 {
            let base = self.xi0.dim(); // 21: nav+bias (nav↔clone cols kept at 1.0)
            if old > base {
                // clone-tail cross rows: scale this clone's TRANS rows (3..6) vs the
                // earlier-clone cols (base..old) by g_t.
                let ncols = old - base;
                let mut ct = sigma_new.view((old, base), (6, ncols)).into_owned();
                for i in 3..6 {
                    ct.row_mut(i).scale_mut(g_t);
                }
                sigma_new.view_mut((old, base), (6, ncols)).copy_from(&ct);
                sigma_new
                    .view_mut((base, old), (ncols, 6))
                    .copy_from(&ct.transpose());
            }
            // self-block trans×trans by g_t² (preserves rot and rot-trans coupling).
            let mut cs = sigma_new.view((old, old), (6, 6)).into_owned();
            for i in 3..6 {
                for j in 3..6 {
                    cs[(i, j)] *= g_t * g_t;
                }
            }
            sigma_new.view_mut((old, old), (6, 6)).copy_from(&cs);
        }

        self.sigma = sigma_new;
        self.clone_ids.push(clone_id);
        self.clone_times.push(time);
        self.clone_refcount.push(0);
        self.clone_poses.push(pose);
        self.resize_scratch();
    }

    /// Remove a clone's 6-dim block from the covariance tail (exact row/col
    /// deletion — survivor cross-covariance is preserved). No-op if absent.
    pub fn marginalize_clone(&mut self, clone_id: u64) {
        let Some(ci) = self.clone_ids.iter().position(|&id| id == clone_id) else {
            return;
        };
        let start = self.xi0.dim() + 6 * ci; // this clone's block start
        let old = self.cov_dim();
        let new = old - 6;

        let mut sigma_new = DMatrix::<f64>::zeros(new, new);
        if start > 0 {
            sigma_new
                .view_mut((0, 0), (start, start))
                .copy_from(&self.sigma.view((0, 0), (start, start)));
        }
        let after = new - start; // rows/cols after the removed block
        if start > 0 && after > 0 {
            sigma_new
                .view_mut((0, start), (start, after))
                .copy_from(&self.sigma.view((0, start + 6), (start, after)));
            sigma_new
                .view_mut((start, 0), (after, start))
                .copy_from(&self.sigma.view((start + 6, 0), (after, start)));
        }
        if after > 0 {
            sigma_new
                .view_mut((start, start), (after, after))
                .copy_from(&self.sigma.view((start + 6, start + 6), (after, after)));
        }

        self.sigma = sigma_new;
        self.clone_ids.remove(ci);
        self.clone_times.remove(ci);
        self.clone_refcount.remove(ci);
        self.clone_poses.remove(ci);
        self.resize_scratch();
    }

    /// Row/col where clone `clone_id`'s 6-dim block starts in `sigma`, if live.
    pub fn clone_block_start(&self, clone_id: u64) -> Option<usize> {
        let ci = self.clone_ids.iter().position(|&id| id == clone_id)?;
        Some(self.xi0.dim() + 6 * ci)
    }

    /// The stored world←camera pose value of clone `clone_id`, if live.
    pub fn clone_pose_value(&self, clone_id: u64) -> Option<SE3> {
        let ci = self.clone_ids.iter().position(|&id| id == clone_id)?;
        Some(self.clone_poses[ci].clone())
    }

    /// Overwrite the stored pose VALUE of clone `clone_id` (covariance untouched).
    /// DIAGNOSTIC ONLY: lets the harness inject GT-relative clone geometry to
    /// separate "the update mechanics/covariance are wrong" from "the EqVIO clone
    /// poses are geometrically inconsistent". Returns false if the clone is dead.
    pub fn set_clone_pose_value(&mut self, clone_id: u64, pose: SE3) -> bool {
        match self.clone_ids.iter().position(|&id| id == clone_id) {
            Some(ci) => {
                self.clone_poses[ci] = pose;
                true
            }
            None => false,
        }
    }

    /// Ids paired with their stored world←camera pose values, in block order.
    /// Used by the MSC update to read the clone-window camera poses for
    /// multi-view triangulation and Jacobian evaluation.
    pub fn clone_poses(&self) -> Vec<(u64, SE3)> {
        self.clone_ids
            .iter()
            .copied()
            .zip(self.clone_poses.iter().cloned())
            .collect()
    }

    /// Additive MSCKF structureless vision update (OpenVINS `UpdaterMSCKF` mirror).
    ///
    /// For each ready track (`track_id -> [(clone_id, uv)]`), using the observing
    /// clones' stored world←camera poses: multi-view triangulate, build the
    /// feature + clone-pose Jacobians, left-nullspace-project the feature out,
    /// chi²-gate the residual against the involved clones' marginal covariance,
    /// then stack every accepted track's projected constraint (embedded at its
    /// clone columns, sensor/landmark columns zero) and run ONE
    /// [`perform_stacked_update`] — which corrects the nav state through the
    /// clone cross-covariance AND mean-corrects the clone poses (faithful mirror).
    ///
    /// The Sparse3D depth bank is untouched: this only tightens the shared
    /// nav+clone state. Returns the number of accepted tracks. A true no-op
    /// (`return 0`, byte-identical covariance) when there are no clones or no
    /// track survives — so an MSCKF-off caller that never populates `tracks`
    /// leaves the filter unchanged.
    ///
    /// `min_track` is the minimum live observations per track (clamped to ≥2);
    /// DE-CONFOUND DIAGNOSTIC (default-off, driven by the harness under
    /// `ECHO_MSC_VPSEUDO=1`): a body-velocity PSEUDO-MEASUREMENT `z = v_gt_body`
    /// pushed through the SAME gain machinery as the vision update
    /// (`perform_stacked_update`, K = Σ Cᵀ S⁻¹, correction via `lift_innovation`),
    /// updating BOTH mean AND covariance — unlike `overwrite_nav_mean` teacher-
    /// forcing which resets only the mean and leaves Σ inconsistent. Isolates the
    /// c92 wall question: is the GAIN MACHINERY sound (a correct velocity residual
    /// through Σ yields a good correction ⇒ only the vision-derived DIRECTION is
    /// bad ⇒ fix clone-birth structure) or does the machinery itself rotate the
    /// correction wrong? The measurement Jacobian is the velocity-tangent selector
    /// [12:15]; to first order the lift maps `gamma_v → δv_body = −gamma_v`, so the
    /// self-consistent C is `−I` on that block. The sign is empirically validated:
    /// with the correct sign, `|v_gt − v_est|` MUST shrink after the update. The
    /// `sign` argument (+1/−1, harness env `ECHO_VPSEUDO_SIGN`) lets a one-frame
    /// convergence check flip it without a rebuild. `sigma_v` is the pseudo-meas
    /// noise σ (m/s); small ⇒ a hard pull toward GT velocity.
    pub fn velocity_pseudo_update<S: EqFCoordinateSuite + ?Sized>(
        &mut self,
        suite: &S,
        v_gt_body: Vector3<f64>,
        sigma_v: f64,
        sign: f64,
    ) {
        if self.clone_ids.is_empty() {
            return;
        }
        let v_est = self.state_estimate().sensor.velocity;
        let residual = DVector::from_column_slice((v_gt_body - v_est).as_slice());
        let s_dim = VIOSensorState::CDIM;
        let mut c = DMatrix::<f64>::zeros(3, s_dim);
        for i in 0..3 {
            c[(i, 12 + i)] = sign; // velocity tangent selector (cols 12:15)
        }
        let r = DMatrix::<f64>::identity(3, 3) * (sigma_v * sigma_v);
        self.perform_stacked_update(suite, &residual, &c, &r, false);
    }

    /// Reference-count hooks for lifecycle management (marginalize at 0).
    pub fn clone_incref(&mut self, clone_id: u64) {
        if let Some(ci) = self.clone_ids.iter().position(|&id| id == clone_id) {
            self.clone_refcount[ci] += 1;
        }
    }

    /// Decrement a clone's refcount; returns the new count (None if absent).
    pub fn clone_decref(&mut self, clone_id: u64) -> Option<usize> {
        let ci = self.clone_ids.iter().position(|&id| id == clone_id)?;
        self.clone_refcount[ci] = self.clone_refcount[ci].saturating_sub(1);
        Some(self.clone_refcount[ci])
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

/// Sensor state-transition block for one propagation sample.
///
/// Default: Euler first-order `F = I + A_ss·dt`. When `ECHO_MSC_EXPM_F=1`, the
/// exact matrix exponential `Φ = expm(A_ss·dt)` (MSCEqF's Van-Loan transition) —
/// a DIAGNOSTIC to isolate Euler-vs-Van-Loan discretization error as the source
/// of the inflated nav↔clone cross-covariance. Env is read per call (only the
/// short stage-diff runs enable it; the O(21³) expm is negligible there).
fn expm_f_ss(a_ss: &SMatrix<f64, 21, 21>, dt: f64) -> SMatrix<f64, 21, 21> {
    let a_dt = a_ss * dt;
    if std::env::var("ECHO_MSC_EXPM_F").map(|s| s == "1").unwrap_or(false) {
        let e = echo_lie::matfn::expm(&DMatrix::from_column_slice(21, 21, a_dt.as_slice()));
        SMatrix::<f64, 21, 21>::from_column_slice(e.as_slice())
    } else {
        SMatrix::<f64, 21, 21>::identity() + a_dt
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

#[cfg(test)]
mod clone_window_tests {
    use super::*;
    use echo_lie::SO3;
    use nalgebra::Vector3;

    fn make_state(n_lm: usize) -> VIOState {
        let sensor = VIOSensorState {
            input_bias: nalgebra::Vector6::zeros(),
            pose: echo_lie::SE3::new(
                SO3::exp(&Vector3::new(0.13, -0.07, 0.19)),
                Vector3::new(1.2, -0.4, 0.8),
            ),
            velocity: Vector3::new(0.3, -0.2, 0.1),
            camera_offset: echo_lie::SE3::new(
                SO3::exp(&Vector3::new(-0.04, 0.08, 0.03)),
                Vector3::new(0.12, -0.03, 0.04),
            ),
        };
        let lms = (0..n_lm)
            .map(|i| Landmark {
                p: Vector3::new(0.5 + i as f64, -0.3, 2.0 + i as f64),
                id: i as u64 + 100,
            })
            .collect();
        VIOState::new(sensor, lms)
    }

    /// Camera-pose Jacobian (6×21), identical to `lib.rs::sparse_camera_pose_jacobian`.
    fn camera_pose_jac(state: &VIOState) -> SMatrix<f64, 6, 21> {
        let adj = state.sensor.camera_offset.inverse().adjoint();
        let mut j = SMatrix::<f64, 6, 21>::zeros();
        j.fixed_view_mut::<6, 6>(0, 6).copy_from(&adj);
        j.fixed_view_mut::<6, 6>(0, 15)
            .copy_from(&SMatrix::<f64, 6, 6>::identity());
        j
    }

    /// Deterministic SPD matrix.
    fn spd(n: usize, seed: f64) -> DMatrix<f64> {
        let mut a = DMatrix::<f64>::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                a[(r, c)] = ((r as f64 + 1.0) * seed).sin() * ((c as f64 + 2.0) * (seed + 0.3)).cos();
            }
        }
        &a * a.transpose() + DMatrix::<f64>::identity(n, n) * 1e-3
    }

    #[test]
    fn clone_then_marginalize_is_identity() {
        let state = make_state(2);
        let n0 = state.dim(); // 21 + 6 = 27
        let sigma0 = spd(n0, 0.21);
        let mut eqf = VIOEqF::new(state.clone(), &sigma0);
        let j = camera_pose_jac(&state);

        eqf.clone_pose(7, 1.5, &j, state.sensor.pose.compose(&state.sensor.camera_offset));
        assert_eq!(eqf.sigma.nrows(), n0 + 6);
        assert_eq!(eqf.n_clones(), 1);

        eqf.marginalize_clone(7);
        assert_eq!(eqf.sigma.nrows(), n0);
        assert_eq!(eqf.n_clones(), 0);

        // Augment-then-marginalize is exact row/col insert+delete (no enforce_spd
        // on either path), so the survivor block returns bit-identical.
        let diff = (&eqf.sigma - &sigma0).amax();
        assert!(diff < 1e-12, "clone round-trip changed sigma: amax={diff:e}");
    }

    #[test]
    fn clone_self_block_equals_camera_pose_cov() {
        // Right after cloning, Σ[clone,clone] must equal J Σ_ss Jᵀ (the current
        // camera-pose covariance) exactly.
        let state = make_state(1);
        let n0 = state.dim();
        let sigma0 = spd(n0, 0.44);
        let mut eqf = VIOEqF::new(state.clone(), &sigma0);
        let j = camera_pose_jac(&state);
        eqf.clone_pose(1, 0.0, &j, state.sensor.pose.compose(&state.sensor.camera_offset));

        let expect = j * sigma0.fixed_view::<21, 21>(0, 0) * j.transpose();
        let got = eqf.sigma.fixed_view::<6, 6>(n0, n0).into_owned();
        assert!((got - expect).amax() < 1e-12);
    }

    #[test]
    fn apply_transport_matches_dense_block_reference_with_clone() {
        // The authoritative transport test: run apply_transport with 1 landmark
        // and 1 clone, and compare the full Σ evolution to an explicit dense
        // F_full Σ F_fullᵀ + Q, F_full = [[F_ss,0,0],[F_lm_s,F_li,0],[0,0,I₆]].
        let state = make_state(1); // dim 24
        let n0 = state.dim();
        let mut eqf = VIOEqF::new(state.clone(), &spd(n0, 0.11));
        let j = camera_pose_jac(&state);
        eqf.clone_pose(3, 0.0, &j, state.sensor.pose.compose(&state.sensor.camera_offset)); // → 30×30, scratch resized
        let n = eqf.sigma.nrows();
        assert_eq!(n, 30);

        // Independent full random SPD to stress every block (sensor/lm/clone).
        let sigma = spd(n, 0.37);
        eqf.sigma = sigma.clone();

        // Block transition.
        let mut f_ss = SMatrix::<f64, 21, 21>::identity();
        for r in 0..21 {
            for c in 0..21 {
                f_ss[(r, c)] += 0.01 * (((r * 3 + c) as f64) * 0.5).sin();
            }
        }
        let mut f_lm_s = DMatrix::<f64>::zeros(3, 21);
        for r in 0..3 {
            for c in 0..21 {
                f_lm_s[(r, c)] = 0.02 * (((r * 7 + c + 1) as f64) * 0.5).cos();
            }
        }
        let mut f_li = SMatrix::<f64, 3, 3>::identity();
        f_li[(0, 1)] = 0.03;
        f_li[(2, 0)] = -0.02;

        // Symmetric process noise on the physical block, zero clone tail.
        let mut q = DMatrix::<f64>::zeros(n, n);
        for r in 0..n0 {
            for c in 0..n0 {
                q[(r, c)] = 1e-4 * (((r + c) as f64) * 0.5).sin();
            }
        }
        q = 0.5 * (&q + q.transpose());

        // Dense reference.
        let mut ff = DMatrix::<f64>::zeros(n, n);
        ff.view_mut((0, 0), (21, 21)).copy_from(&f_ss);
        ff.view_mut((21, 0), (3, 21)).copy_from(&f_lm_s);
        ff.view_mut((21, 21), (3, 3)).copy_from(&f_li);
        ff.view_mut((24, 24), (6, 6))
            .copy_from(&DMatrix::<f64>::identity(6, 6));
        let mut reference = &ff * &sigma * ff.transpose() + &q;
        reference = 0.5 * (&reference + reference.transpose());
        for i in 0..n {
            reference[(i, i)] += 1e-12; // matches enforce_spd
        }

        eqf.apply_transport(&f_ss, &f_lm_s, &[f_li], &q);

        let diff = (&eqf.sigma - &reference).amax();
        assert!(diff < 1e-9, "transport vs dense reference amax={diff:e}");
    }

}
