use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use std::collections::HashMap;

use crate::mathematical::VisionMeasurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthParametrization {
    Euclidean,
    InvDepth,
    Polar,
}

/// Second-order measurement-update mode for the 3D IEKF. Restores the dropped
/// projective (perspective-division) curvature so the reported covariance is
/// honest at weak parallax — "the EqF way" of fixing the NEES overconfidence.
/// Full derivation: `ECHO-LI-notes/docs/sparse3d_secondorder_eqf_derivation.md`.
/// Only consumed by `Sparse3DFilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecondOrderMode {
    /// First-order (iterated) EKF; `iekf_iterations` applies. Default.
    #[default]
    Off,
    /// Option A — analytic second-order EqF. Adds the closed-form innovation
    /// inflation `Λ_kl = ½ tr(H_k Σ H_l Σ)` and the `½ tr(H_m Σ)` predicted-
    /// measurement bias correction. Supersedes `iekf_iterations` (the bias is
    /// removed in closed form, so iterating is redundant).
    Analytic,
    // Option B (future) — unscented / sigma-point EqF (`2·dim+1` evaluations);
    // second-order-exact by quadrature and folds in partial higher-order terms.
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
    pub reanchor_flow_px: f64,
    /// Average the perspective output Jacobian between the predicted and the
    /// measured normalized image coords in the 3D bearing update, à la the EqF
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
    /// Per-step radial (range) random-walk process noise for the 3D IEKF, as a
    /// fraction of range² added to the landmark covariance each update
    /// (`Σ += range_walk_var · ‖q_c‖² · r̂r̂ᵀ`, pulled back into the error chart).
    /// The IEKF has no propagation, so without a floor the static-landmark Σ
    /// collapses below the un-modelled triangulation/range bias and NEES grows
    /// with depth; this floor flattens it (`≈3e-10`–`1e-8` empirically). The `²`
    /// scaling matches the bias' `∝ depth` growth, so one constant calibrates
    /// every depth. Default 0 (off). Distinct from `process_depth_var` (the 1D
    /// filter's un-scaled per-step term). Only consumed by `Sparse3DFilter`.
    /// Findings: `ECHO-LI-notes/docs/sparse3d_secondorder_eqf_derivation.md` §8.
    pub range_walk_var: f64,
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
            reanchor_flow_px: 3.0,
            use_equivariant_output: false,
            iekf_iterations: 1,
            second_order_mode: SecondOrderMode::Off,
            range_walk_var: 0.0,
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
