use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use std::collections::HashMap;

use crate::mathematical::VisionMeasurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthParametrization {
    Euclidean,
    InvDepth,
    Polar,
}

/// Second-order measurement-update mode for the 3D IEKF.
///
/// Restores the dropped projective curvature so the reported covariance is
/// honest at weak parallax: the first-order projection Jacobian drops the
/// curvature of the inverse-depth map, which under-reports variance exactly
/// where the baseline is short and the depth is least determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecondOrderMode {
    /// First-order (iterated) EKF; `iekf_iterations` applies. Default.
    #[default]
    Off,
    /// Analytic second-order EqF. Adds closed-form innovation inflation and
    /// predicted-measurement bias correction.
    Analytic,
    // Option B (future): unscented / sigma-point EqF.
    // Unscented,
}

#[derive(Debug, Clone)]
pub struct SparseVogSettings {
    pub parametrization: DepthParametrization,
    pub max_pool_size: usize,
    pub min_track_length: usize,
    pub conv_inlier_ratio: f64,
    pub conv_variance_threshold: f64,
    pub init_depth_var: f64,
    pub init_invdepth_var: f64,
    pub sigma_pixel: f64,
    /// Optional per-feature-age pixel-noise inflation used by Sparse3D's 2D
    /// image-residual updates:
    /// `sigma_eff^2 = sigma_pixel^2 + (flow_age_rate_px_per_frame * age)^2`.
    /// The scalar 1D `SparseGBFilter` keeps its historical triangulated-depth
    /// noise model and does not consume this setting.
    pub flow_age_rate_px_per_frame: f64,
    pub uniform_z_max: f64,
    pub uniform_rho_max: f64,
    pub uniform_d_min: f64,
    pub uniform_d_max: f64,
    pub a_init: f64,
    pub b_init: f64,
    pub ab_min: f64,
    pub ab_max: f64,
    pub min_inlier_ratio: f64,
    pub mahalanobis_reset_chi2: f64,
    pub process_depth_var: f64,
    pub min_parallax: f64,
    pub min_cos_sim: f64,
    pub min_depth: f64,
    pub max_depth: f64,
    /// Minimum pixel flow a feature must accumulate since first sight before it is
    /// birthed. Two-view triangulation is severely biased toward the camera at low
    /// parallax, so a feature is held pending until it has moved this far, then
    /// triangulated and anchored in the frame where it is born. Despite the historical
    /// name this never re-anchors an existing landmark: anchors are fixed at birth.
    pub birth_min_flow_px: f64,
    /// Civera-style UNDELAYED inverse-depth birth (T-RO 2008 "Inverse Depth
    /// Parametrization for Monocular SLAM"). When > 0, a feature is birthed on its
    /// FIRST observation at depth `civera_birth_depth` with no parallax and no
    /// two-view triangulation, instead of being held pending until it has moved
    /// `birth_min_flow_px` and cleared the ray-angle gate.
    ///
    /// The point of the inverse-depth chart is that a Gaussian on rho represents
    /// "somewhere between `civera_birth_depth`/2 and infinity" honestly, which a
    /// Gaussian on depth cannot. Civera's proposed prior sets sigma_rho = rho0/2,
    /// so rho0 - 2*sigma_rho = 0 -- the 95% band reaches all the way to infinity.
    /// That reach is what makes undelayed birth safe: a distant feature is not
    /// claimed to be near, it is claimed to be unknown-but-bounded-below.
    ///
    /// This is only representable because this chart is additive in rho. On a
    /// multiplicative (log-depth) chart the band is [d0*e^-r, d0*e^+r], which can
    /// never reach rho = 0, so the same prior does NOT transfer there.
    ///
    /// 0 disables (delayed triangulated birth, the historical behaviour).
    /// Drive the range-direction process noise from the FULL anchor->current relative
    /// pose covariance evaluated at the CURRENT separation, instead of accumulating
    /// per-frame clipped increments.
    ///
    /// WHY. The shipped path adds `pose_range_scale * G_k * dSigma_k` each frame, where
    /// `G_k = r^2/b^2` is the inverse-parallax gain and `dSigma_k` a clipped covariance
    /// increment. Two things break. (i) A cumulative sum of clipped increments can only
    /// grow ~linearly in the window length, while the measured relative-pose error grows
    /// near-ballistically (p = 1.70-1.99 vs surrogate 1.00-1.01, filter_formulation
    /// Section X.1) -- the deficit is in the EXPONENT, so no scalar repairs it. (ii) The
    /// gain is applied to the increment, `G dSigma`, but the quantity that should
    /// accumulate is `d(G Sigma)` = `G dSigma + Sigma dG`; the `Sigma dG` term is dropped,
    /// and `G` changes every frame as the baseline `b` grows.
    ///
    /// Direct mode evaluates `f_k = G_k * (r^2 sig_phi^2 + sig_b^2)` from the FULL
    /// relative covariance and injects `max(0, f_k - f_{k-1})`, so the accumulated range
    /// variance tracks the true relative-pose curve including its exponent, with no free
    /// scalar. Requires a kept-pose window supplying the true anchor->current covariance
    /// (MSCEqF clones); with the absolute-marginal surrogate it is not meaningful.
    pub pose_range_direct: bool,
    /// Consider (Schmidt) treatment of the anchor->current relative pose,
    /// filter_formulation §V-F: each landmark carries its 3x6 cross-covariance with
    /// that pose error; the update uses the joint innovation covariance and the
    /// consider gain; the current-frame report adds the transported pose covariance
    /// and the cross term; re-anchoring onto the current clone uses the tracked cross
    /// term. Replaces the measurement-side pose term, the range injection
    /// (`pose_range_direct` / `pose_range_scale`) and the independence assumption at
    /// re-anchor. Needs a feed that supplies `ConsiderRel` per anchor
    /// (`CloneRelFeed::consider`). BearingInvDepthAdditive chart, bias state off.
    /// Default false. UNVALIDATED: built 2026-10-14 to test §V-F's predictions.
    pub consider_relpose: bool,
    /// Exogenous multiplicative SCALE uncertainty, as a 1-sigma fraction of range.
    /// Adds `(scale_sigma * r)^2` to the REPORTED range variance (and `(scale_sigma * d)^2`
    /// to reported depth variance). Default 0 = off.
    ///
    /// WHY IT CANNOT COME FROM THE POSE COVARIANCE. A monocular VIO's range error is
    /// dominated by a systematic BASELINE-SCALE error that is common to every landmark;
    /// measurement shows the depth error tracks it 1:1 with no parallax amplification.
    /// The pose covariance `P_vv` structurally cannot carry it -- the scale direction is
    /// the unobservable one, and its reported along-baseline variance is exactly the
    /// channel that is over-confident. So this term is sourced OUTSIDE the filter, from
    /// trajectory excitation, via [`sigma_s_from_excitation`].
    ///
    /// WHY IT IS REPORTING-ONLY, NOT PROCESS NOISE. A scale bias is common-mode and
    /// systematic: inflating the per-landmark covariance inside the recursion would only
    /// make the filter under-weight its measurements without removing the bias, and it
    /// would accumulate frame over frame. The honest treatment is to carry it into what
    /// the filter REPORTS, which is what an obstacle mapper consumes.
    pub scale_sigma: f64,
    /// RELATIVE convergence gate: a landmark is unusable when `sigma_r / r` exceeds this
    /// fraction. `0` = disabled, fall back to the absolute `conv_variance_threshold`.
    ///
    /// WHY THE ABSOLUTE FORM IS WRONG. `conv_variance_threshold` is in m^2, but depth
    /// uncertainty is multiplicative in range: a fixed 2.0 m^2 demands `sigma_r <= 1.4 m`
    /// whether the landmark is at 5 m or 500 m -- 0.3% relative at the far end, which no
    /// monocular system achieves. It also has to be re-tuned every time the covariance
    /// model changes: once the pose-driven range noise made `sigma_r` honest (3-7 m), the
    /// shipped 2.0 rejected EVERY landmark and `query_range` returned nothing at all,
    /// silently, because the NEES harness scores the raw state and never calls it.
    ///
    /// WHY 0.5 IS THE NATURAL VALUE, not a knob. At `sigma_r / r = 0.5` the 2-sigma band
    /// reaches zero inverse depth, i.e. the landmark "could be at infinity" and carries no
    /// range information. That is the same constant the Civera birth prior uses
    /// (`civera_rho_sigma_frac`), for the same reason.
    ///
    /// Evaluated on the filter's OWN variance, BEFORE `scale_sigma` is added: a
    /// trajectory-level common-mode scale uncertainty must not make a well-converged
    /// landmark look unconverged.
    pub conv_range_sigma_frac: f64,
    pub civera_birth_depth: f64,
    /// Fraction of rho0 used as the birth sigma on rho. Civera's proposed value is
    /// 0.5 (band reaches rho = 0 at 2 sigma). Only used when `civera_birth_depth` > 0.
    pub civera_rho_sigma_frac: f64,
    /// Average the perspective output Jacobian between the predicted and the
    /// measured normalized image coords in the 3D bearing update, like the EqF
    /// coordinate suite's `output_matrix_ci_star`. NOTE: experiments show this
    /// *worsens* consistency when grafted onto `Sparse3DFilter` (a plain
    /// chart-EKF) -- putting the measurement into H correlates it with the
    /// measurement noise, which the EKF covariance update assumes away, so the
    /// covariance collapses. The EqF avoids this via its equivariant error
    /// coordinates + lifted innovation; the output approximation is not a
    /// drop-in for a non-equivariant filter. Kept behind this flag (default
    /// off) for experimentation. Only consumed by `Sparse3DFilter`.
    pub use_equivariant_output: bool,
    /// Number of measurement relinearizations in the 3D update (iterated EKF).
    /// 1 = plain EKF (linearize once at the prior). >1 relinearizes the
    /// projection at the posterior to cancel the bearing-only depth bias at weak
    /// parallax (cf. ROVIO / the 1D sparse_vogiatzis iterated update). Only
    /// consumed by `Sparse3DFilter`. Ignored when `second_order_mode` is not
    /// `Off` (the second-order filter does the bias correction in closed form).
    pub iekf_iterations: usize,
    /// Second-order EqF measurement-update mode (covariance inflation + bias
    /// correction). See `SecondOrderMode`. Only consumed by `Sparse3DFilter`.
    pub second_order_mode: SecondOrderMode,
    /// Per-step radial random-walk process noise for the 3D IEKF, as a fraction
    /// of range squared added to the landmark covariance each update.
    /// The IEKF has no propagation, so without a floor the static-landmark covariance
    /// collapses below the un-modelled triangulation/range bias and NEES grows
    /// with depth. Default 0 (off). Distinct from `process_depth_var` (the 1D
    /// filter's un-scaled per-step term). Only consumed by `Sparse3DFilter`.
    /// Scales with range^2 so the injected uncertainty is a fixed *relative*
    /// depth uncertainty rather than an absolute one.
    pub range_walk_var: f64,
    /// Scale on the *pose-driven* range process noise (eq. 10-11 of the Sparse3D
    /// formulation). When > 0 and per-frame incremental pose covariances (p_vv, p_ww)
    /// are supplied to `update`, each frame injects a range (radial) process noise
    /// q = scale * (r^2/b^2)(r^2 * sig_phi^2 + sig_t^2) along the line of sight, where
    /// r is the current range, b the anchor->current baseline, and sig_.^2 the
    /// incremental relative-pose variances. This is the channel that CAN inflate the
    /// depth covariance (the measurement-noise term proj*P*proj^T is radial-blind:
    /// d(pi)/dq * r_hat = 0). Per-landmark via the inverse-parallax r/b factor.
    /// Default 0 (off). Only consumed by the `BearingInvDepthAdditive` update.
    pub pose_range_scale: f64,
    /// Coherent (bias-driven) companion to `pose_range_scale`, fixing its GROWTH RATE.
    ///
    /// `pose_range_scale` injects a near-constant increment each frame, so the variance it
    /// accumulates grows ~linearly in track age N (a random walk). A *persistent* pose bias
    /// instead makes the range error grow ~N, hence its variance ~N^2 -- a different power,
    /// which no scalar multiplier can produce. Adding `2*N*c*x` per frame telescopes to
    /// `N^2*c*x` (sum of 2n over n=1..N), so the total is `scale*x*N + coherent*x*N^2` and
    /// the effective exponent interpolates between 1 and 2 via the ratio. Here
    /// `x = (r^2/b^2)(r^2 sig_phi^2 + sig_t^2)` as for `pose_range_scale`.
    ///
    /// Default 0 (off). No constant value has been found that works across pose runs of
    /// differing quality: the reported covariance is wrong in both level and growth rate,
    /// and one constant cannot correct both.
    pub pose_range_coherent: f64,
    /// Per-frame random-walk variance (px^2) of a per-track 2D correspondence-bias
    /// state augmented onto the landmark in the `BearingInvDepthAdditive` update.
    /// Measurement model becomes `u = pi(P(s)) + b + eps`, `b_k = b_{k-1} + w_k`,
    /// `w_k ~ N(0, bias_walk_var I)`. The bias absorbs the temporally-correlated
    /// KLT/Rudolf correspondence drift so repeated same-track observations are not
    /// over-counted (exact-GT MidAir shows this drift is a random walk, not a
    /// constant offset). Default 0 (off) -> plain 3-DOF landmark EKF, exact prior
    /// behavior. Only consumed by `Sparse3DFilter`'s bearing-invdepth chart.
    /// Absorbs slow correspondence drift (the tracked pixel sliding off its
    /// landmark) as a random-walk pixel bias, so it is not misread as parallax.
    pub bias_walk_var: f64,
    /// Experimental sigma-point rotation propagation for the additive chart.
    pub rotation_unscented: bool,
}

impl Default for SparseVogSettings {
    fn default() -> Self {
        Self {
            parametrization: DepthParametrization::InvDepth,
            max_pool_size: 300,
            min_track_length: 5,
            conv_inlier_ratio: 0.7,
            conv_variance_threshold: 0.5,
            init_depth_var: 1.0,
            init_invdepth_var: 1.0,
            sigma_pixel: 0.5,
            flow_age_rate_px_per_frame: 0.0,
            uniform_z_max: 20.0,
            uniform_rho_max: 10.0,
            uniform_d_min: -5.0,
            uniform_d_max: 5.0,
            a_init: 10.0,
            b_init: 2.0,
            ab_min: 1.0,
            ab_max: 20.0,
            min_inlier_ratio: 0.5,
            mahalanobis_reset_chi2: 9.0,
            process_depth_var: 0.01,
            min_parallax: 1e-4,
            min_cos_sim: 0.95,
            min_depth: 0.1,
            max_depth: 100.0,
            birth_min_flow_px: 3.0,
            pose_range_direct: false,
            consider_relpose: false,
            scale_sigma: 0.0,
            conv_range_sigma_frac: 0.0,
            civera_birth_depth: 0.0,
            civera_rho_sigma_frac: 0.5,
            use_equivariant_output: false,
            iekf_iterations: 1,
            second_order_mode: SecondOrderMode::Off,
            range_walk_var: 0.0,
            pose_range_scale: 0.0,
            pose_range_coherent: 0.0,
            bias_walk_var: 0.0,
            rotation_unscented: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeatureState {
    pub feat_id: u64,
    pub canonical: f64,
    pub canonical_var: f64,
    pub a: f64,
    pub b: f64,
    pub track_length: usize,
}

impl FeatureState {
    pub fn inlier_ratio(&self) -> f64 {
        let ab = self.a + self.b;
        if ab <= 0.0 {
            0.0
        } else {
            self.a / ab
        }
    }
}

pub struct SparseGBFilter {
    k: Matrix3<f64>,
    settings: SparseVogSettings,
    sigma_norm_sq: f64,

    features: HashMap<u64, FeatureState>,
    prev_uvs: HashMap<u64, Vector2<f64>>,
    prev_t_wc: Option<Matrix4<f64>>,
    prev_stamp: f64,
}

impl SparseGBFilter {
    pub fn new(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        let fx = k[(0, 0)];
        let sigma_norm_sq = (settings.sigma_pixel / fx).powi(2);
        Self {
            k,
            settings,
            sigma_norm_sq,
            features: HashMap::new(),
            prev_uvs: HashMap::new(),
            prev_t_wc: None,
            prev_stamp: -1.0,
        }
    }

    pub fn update(
        &mut self,
        measurement: &VisionMeasurement,
        t_wc: &Matrix4<f64>,
        p_vv: Option<&Matrix3<f64>>,
    ) {
        let stamp = measurement.stamp;
        let curr_uvs: HashMap<u64, Vector2<f64>> = measurement
            .cam_coordinates
            .iter()
            .map(|(&id, &uv)| (id, Vector2::new(uv[0] as f64, uv[1] as f64)))
            .collect();

        if self.prev_t_wc.is_none() || self.prev_stamp < 0.0 {
            self.prev_t_wc = Some(*t_wc);
            self.prev_stamp = stamp;
            self.prev_uvs = curr_uvs;
            return;
        }

        let dt = (stamp - self.prev_stamp).max(0.0);
        let t_wc_prev = self.prev_t_wc.as_ref().unwrap();

        let t_cw_curr = t_wc.try_inverse().unwrap_or_else(Matrix4::identity);
        let t_curr_prev = t_cw_curr * t_wc_prev;
        let r = t_curr_prev.fixed_view::<3, 3>(0, 0).into_owned();
        let t = t_curr_prev.fixed_view::<3, 1>(0, 3).into_owned();

        for (&fid, &uv_curr) in &curr_uvs {
            if let Some(&uv_prev) = self.prev_uvs.get(&fid) {
                let (z_obs, drive) =
                    triangulate_pixel(&self.k, &self.settings, &uv_prev, &uv_curr, &r, &t);

                if let Some(feat) = self.features.get_mut(&fid) {
                    if feat.canonical > 0.0 {
                        predict_feature_state(
                            &self.k,
                            &self.settings,
                            feat,
                            &uv_prev,
                            &r,
                            &t,
                            p_vv,
                            dt,
                        );
                    }
                }

                if z_obs <= 0.0 {
                    continue;
                }

                if let Some(feat) = self.features.get_mut(&fid) {
                    if feat.canonical <= 0.0 {
                        feat.canonical = depth_to_canonical(&self.settings, z_obs);
                        feat.canonical_var = depth_var_to_canonical_var(
                            &self.settings,
                            z_obs,
                            self.settings.init_depth_var,
                        );
                        feat.a = self.settings.a_init;
                        feat.b = self.settings.b_init;
                    }
                    vogiatzis_update_feature_state(
                        &self.settings,
                        self.sigma_norm_sq,
                        feat,
                        z_obs,
                        drive,
                    );
                    feat.track_length += 1;
                } else if self.features.len() < self.settings.max_pool_size {
                    let mut feat = FeatureState {
                        feat_id: fid,
                        canonical: depth_to_canonical(&self.settings, z_obs),
                        canonical_var: depth_var_to_canonical_var(
                            &self.settings,
                            z_obs,
                            self.settings.init_depth_var,
                        ),
                        a: self.settings.a_init,
                        b: self.settings.b_init,
                        track_length: 1,
                    };
                    vogiatzis_update_feature_state(
                        &self.settings,
                        self.sigma_norm_sq,
                        &mut feat,
                        z_obs,
                        drive,
                    );
                    self.features.insert(fid, feat);
                }
            }
        }

        self.features.retain(|id, _| curr_uvs.contains_key(id));
        self.prev_t_wc = Some(*t_wc);
        self.prev_stamp = stamp;
        self.prev_uvs = curr_uvs;
    }

    pub fn query(&self, fid: u64) -> (f64, f64) {
        if let Some(feat) = self.features.get(&fid) {
            if feat.canonical <= 0.0 || feat.track_length < self.settings.min_track_length {
                return (-1.0, f64::INFINITY);
            }
            if feat.inlier_ratio() < self.settings.conv_inlier_ratio
                || feat.canonical_var > self.settings.conv_variance_threshold
            {
                return (-1.0, f64::INFINITY);
            }
            let depth = canonical_to_depth(&self.settings, feat.canonical);
            let depth_var = match self.settings.parametrization {
                DepthParametrization::Euclidean => feat.canonical_var,
                DepthParametrization::InvDepth => feat.canonical_var * depth.powi(4),
                DepthParametrization::Polar => feat.canonical_var * depth.powi(2),
            };
            (depth, depth_var)
        } else {
            (-1.0, f64::INFINITY)
        }
    }
}

fn triangulate_pixel(
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    uv_prev: &Vector2<f64>,
    uv_curr: &Vector2<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> (f64, f64) {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    let x_curr = (uv_curr[0] - cx) / fx;
    let y_curr = (uv_curr[1] - cy) / fy;
    let x_prev = (uv_prev[0] - cx) / fx;
    let y_prev = (uv_prev[1] - cy) / fy;

    let bearing_prev = Vector3::new(x_prev, y_prev, 1.0);
    let aligned = r * bearing_prev;
    if aligned[2] <= 1e-6 {
        return (-1.0, 0.0);
    }

    let x_prev_rect = aligned[0] / aligned[2];
    let y_prev_rect = aligned[1] / aligned[2];

    let num_x = x_prev_rect * t[2] - t[0];
    let num_y = y_prev_rect * t[2] - t[1];
    let den_x = x_prev_rect - x_curr;
    let den_y = y_prev_rect - y_curr;

    let geom_mag_sq = num_x * num_x + num_y * num_y;
    let ideal_mag = geom_mag_sq.sqrt();
    let obs_mag = (den_x * den_x + den_y * den_y).sqrt();

    if ideal_mag < settings.min_parallax || obs_mag < 1e-6 {
        return (-1.0, 0.0);
    }

    let dot = num_x * den_x + num_y * den_y;
    if dot <= 1e-6 {
        return (-1.0, 0.0);
    }

    let cos_sim = dot / (ideal_mag * obs_mag);
    if cos_sim < settings.min_cos_sim {
        return (-1.0, 0.0);
    }

    let z_curr = geom_mag_sq / dot;
    if z_curr < settings.min_depth || z_curr > settings.max_depth {
        return (-1.0, 0.0);
    }

    (z_curr, ideal_mag)
}

fn predict_feature_state(
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState,
    uv_prev: &Vector2<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    p_vv: Option<&Matrix3<f64>>,
    dt: f64,
) {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let x = (uv_prev[0] - cx) / fx;
    let y = (uv_prev[1] - cy) / fy;

    let z_old = canonical_to_depth(settings, feat.canonical);
    let p_prev = Vector3::new(x * z_old, y * z_old, z_old);
    let p_curr = r * p_prev + t;
    let z_new = p_curr[2];

    if z_new < settings.min_depth {
        feat.canonical = -1.0;
        return;
    }

    let g = r[(2, 0)] * x + r[(2, 1)] * y + r[(2, 2)];
    let j = match settings.parametrization {
        DepthParametrization::Euclidean => g,
        DepthParametrization::InvDepth => g * (z_old / z_new).powi(2),
        DepthParametrization::Polar => g * (z_old / z_new),
    };
    let mut var_new = j * j * feat.canonical_var;

    if let Some(p_vv) = p_vv {
        if dt > 0.0 {
            let b_norm_sq = x * x + y * y + 1.0;
            let num = x * x * p_vv[(0, 0)]
                + y * y * p_vv[(1, 1)]
                + p_vv[(2, 2)]
                + 2.0 * x * y * p_vv[(0, 1)]
                + 2.0 * x * p_vv[(0, 2)]
                + 2.0 * y * p_vv[(1, 2)];
            let sigma_v_along = num / b_norm_sq;
            let q_z = dt * dt * sigma_v_along;
            var_new += match settings.parametrization {
                DepthParametrization::Euclidean => q_z,
                DepthParametrization::InvDepth => q_z * (1.0 / z_new).powi(4),
                DepthParametrization::Polar => q_z / (z_new * z_new),
            };
        }
    } else {
        var_new += settings.process_depth_var * dt.max(1e-3);
    }

    feat.canonical = depth_to_canonical(settings, z_new);
    feat.canonical_var = var_new.max(1e-8);
}

fn vogiatzis_update_feature_state(
    settings: &SparseVogSettings,
    sigma_norm_sq: f64,
    feat: &mut FeatureState,
    z_obs: f64,
    drive: f64,
) {
    let obs = depth_to_canonical(settings, z_obs);
    let tau_sq = match settings.parametrization {
        DepthParametrization::Euclidean => (z_obs.powi(2) * sigma_norm_sq) / (drive * drive),
        DepthParametrization::InvDepth => sigma_norm_sq / (drive * drive),
        DepthParametrization::Polar => (z_obs.powi(2) * sigma_norm_sq) / (drive * drive),
    };

    let mu = feat.canonical;
    let sigma_sq = feat.canonical_var;
    let a = feat.a;
    let b = feat.b;

    let s_total = sigma_sq + tau_sq;
    let m_dist_sq = (obs - mu).powi(2) / s_total;

    if settings.mahalanobis_reset_chi2 > 0.0
        && (a + b) > 0.0
        && (a / (a + b)) < settings.min_inlier_ratio
        && m_dist_sq > settings.mahalanobis_reset_chi2
    {
        feat.canonical = obs;
        feat.canonical_var = tau_sq;
        feat.a = settings.a_init;
        feat.b = settings.b_init;
        return;
    }

    let m = (mu * tau_sq + obs * sigma_sq) / s_total;
    let s_sq = (sigma_sq * tau_sq) / s_total;

    let gauss_pdf = if m_dist_sq > 100.0 {
        0.0
    } else {
        (-0.5 * m_dist_sq).exp() / (2.0 * std::f64::consts::PI * s_total).sqrt()
    };
    let u_prior = match settings.parametrization {
        DepthParametrization::Euclidean => 1.0 / settings.uniform_z_max,
        DepthParametrization::InvDepth => 1.0 / settings.uniform_rho_max,
        DepthParametrization::Polar => 1.0 / (settings.uniform_d_max - settings.uniform_d_min),
    };

    let ab_sum = a + b;
    let c1 = (a / ab_sum) * gauss_pdf;
    let c2 = (b / ab_sum) * u_prior;
    let z_norm = c1 + c2;

    if z_norm < 1e-30 {
        feat.b = (b + 1.0).min(settings.ab_max);
        return;
    }

    let w1 = c1 / z_norm;
    let w2 = c2 / z_norm;

    let new_mu = w1 * m + w2 * mu;
    let e_x2 = w1 * (s_sq + m * m) + w2 * (sigma_sq + mu * mu);
    let new_sigma_sq = (e_x2 - new_mu * new_mu).max(1e-8);

    let denom1 = ab_sum + 1.0;
    let denom2 = denom1 * (ab_sum + 2.0);
    let e_pi = (w1 * (a + 1.0) + w2 * a) / denom1;
    let e_pi2 = (w1 * (a + 1.0) * (a + 2.0) + w2 * a * (a + 1.0)) / denom2;
    let v_pi = e_pi2 - e_pi * e_pi;

    let (new_a, new_b) = if v_pi < 1e-6 || e_pi <= 1e-6 || e_pi >= 1.0 - 1e-6 {
        (a + w1, b + w2)
    } else {
        let factor = (e_pi * (1.0 - e_pi) / v_pi - 1.0).max(0.5);
        (e_pi * factor, (1.0 - e_pi) * factor)
    };

    feat.canonical = new_mu;
    feat.canonical_var = new_sigma_sq;
    feat.a = new_a.clamp(settings.ab_min, settings.ab_max);
    feat.b = new_b.clamp(settings.ab_min, settings.ab_max);
}

fn depth_to_canonical(settings: &SparseVogSettings, z: f64) -> f64 {
    match settings.parametrization {
        DepthParametrization::Euclidean => z,
        DepthParametrization::InvDepth => 1.0 / z,
        DepthParametrization::Polar => z.ln(),
    }
}

fn canonical_to_depth(settings: &SparseVogSettings, c: f64) -> f64 {
    match settings.parametrization {
        DepthParametrization::Euclidean => c,
        DepthParametrization::InvDepth => 1.0 / c,
        DepthParametrization::Polar => c.exp(),
    }
}

fn depth_var_to_canonical_var(settings: &SparseVogSettings, z: f64, var_z: f64) -> f64 {
    match settings.parametrization {
        DepthParametrization::Euclidean => var_z,
        DepthParametrization::InvDepth => var_z / z.powi(4),
        DepthParametrization::Polar => var_z / z.powi(2),
    }
}
