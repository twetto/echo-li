use std::collections::{HashMap, HashSet};

use echo_lie::SOT3;
use nalgebra::{Matrix2, Matrix2x3, Matrix3, Matrix3x2, Matrix4, Vector2, Vector3, Vector4};

use crate::coordinate_suite::base_skew;
use crate::coordinate_suite::invdepth::{conv_euc2ind, conv_ind2euc, point_chart_invdepth_inv};
use crate::coordinate_suite::normal::{conv_euc2normal, conv_normal2euc, point_chart_normal_inv};
use crate::depth::sparse_gb::{SecondOrderMode, SparseVogSettings};
use crate::mathematical::vision_measurement::VisionMeasurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse3DChart {
    /// Static-anchor SOT(3) IEKF, range carried multiplicatively (log-depth).
    Polar,
    /// Same SOT(3) IEKF, inverse-depth *reporting* chart (a relabel of `Polar`;
    /// the update is identical — see the consistency docs).
    InvDepth,
    /// ρ-first: a genuinely *additive* inverse-depth EKF. State is
    /// `(alpha, beta, rho)` in the anchor camera frame (no group, no log-depth),
    /// which removes the sequential-linearisation overconfidence of `Polar`.
    InvDepthAdditive,
}

#[derive(Debug, Clone)]
struct PendingFeature {
    ref_t_wc: Matrix4<f64>,
    ref_uv: Vector2<f64>,
    ref_stamp: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureState3D {
    pub feat_id: u64,
    // Cached estimate in the CURRENT camera frame, refreshed after every update.
    // Kept for the public API / queries (depth_for_chart etc. read these).
    pub position: Vector3<f64>,
    pub covariance: Matrix3<f64>,
    // --- True IEKF state (source of truth) ---
    // Landmark fixed in its anchor camera frame; the group element x carries all
    // refinement so the error coordinates never need re-centring (no re-anchor).
    pub q0: Vector3<f64>, // anchor-frame landmark (fixed origin)  [Polar/InvDepth]
    pub x: SOT3,          // group refinement; q_hat_anchor = x^{-1} . q0  [Polar/InvDepth]
    pub sigma: Matrix3<f64>, // covariance in Euclidean error coords about q0  [Polar/InvDepth]
    // ρ-first additive inverse-depth state [InvDepthAdditive only]: s = (alpha,
    // beta, rho) = (X/Z, Y/Z, 1/Z) of the landmark in the ANCHOR camera frame,
    // with covariance inv_p in those coords. Plain additive EKF, no group / no
    // re-charting. The SOT(3) fields above are unused in this chart (and vice
    // versa). Cached `covariance` for this chart is the current-frame Euclidean
    // covariance directly (chart_to_euc_jac == I for InvDepthAdditive).
    pub inv_s: Vector3<f64>,
    pub inv_p: Matrix3<f64>,
    pub anchor_t_wc: Matrix4<f64>, // camera->world pose at the anchor (creation) frame
    // --- Gaussian-Beta inlier model + bookkeeping ---
    pub a: f64,
    pub b: f64,
    pub track_length: usize,
    pub ref_uv: Vector2<f64>,
    pub ref_stamp: f64,
    /// Last update's normalized innovation squared (NIS = the gating
    /// Mahalanobis²). Should be ~χ²(2) when the filter is consistent — the
    /// GT-free online consistency check. NaN before the first update.
    pub last_nis: f64,
}

impl FeatureState3D {
    pub fn depth_for_chart(&self, chart: Sparse3DChart) -> f64 {
        match chart {
            Sparse3DChart::Polar => self.position.norm(),
            Sparse3DChart::InvDepth | Sparse3DChart::InvDepthAdditive => self.position[2],
        }
    }

    pub fn depth_variance_for_chart(&self, chart: Sparse3DChart) -> f64 {
        match chart {
            Sparse3DChart::Polar => {
                let depth = self.position.norm();
                if depth < 1e-6 {
                    f64::INFINITY
                } else {
                    self.covariance[(2, 2)] * depth * depth
                }
            }
            Sparse3DChart::InvDepth => {
                if self.position[2] < 1e-6 {
                    f64::INFINITY
                } else {
                    let h_z =
                        Vector3::new(0.0, 0.0, 1.0).transpose() * conv_ind2euc(&self.position);
                    (h_z * self.covariance * h_z.transpose())[(0, 0)]
                }
            }
            // Cached covariance is already Euclidean (current frame) here.
            Sparse3DChart::InvDepthAdditive => {
                if self.position[2] < 1e-6 {
                    f64::INFINITY
                } else {
                    self.covariance[(2, 2)]
                }
            }
        }
    }

    pub fn range_variance_for_chart(&self, chart: Sparse3DChart) -> f64 {
        let range = self.position.norm();
        if range < 1e-6 {
            return f64::INFINITY;
        }
        let h_range = (self.position / range).transpose() * chart_to_euc_jac(chart, &self.position);
        (h_range * self.covariance * h_range.transpose())[(0, 0)]
    }

    /// Covariance in Euclidean camera-frame coordinates.
    ///
    /// The filter stores `covariance` in chart coordinates; this maps it to a
    /// 3x3 Euclidean covariance via the chart->Euclidean Jacobian, so consumers
    /// (e.g. NEES diagnostics) can score the estimate without reimplementing the
    /// chart conversion.
    pub fn covariance_euclidean(&self, chart: Sparse3DChart) -> Matrix3<f64> {
        let j = chart_to_euc_jac(chart, &self.position);
        j * self.covariance * j.transpose()
    }

    pub fn inlier_ratio(&self) -> f64 {
        let ab = self.a + self.b;
        if ab <= 0.0 { 0.0 } else { self.a / ab }
    }
}

pub struct Sparse3DFilter {
    k: Matrix3<f64>,
    chart: Sparse3DChart,
    settings: SparseVogSettings,
    sigma_norm_sq: f64,
    features: Vec<FeatureState3D>,
    feature_slots: HashMap<u64, usize>,
    pending: HashMap<u64, PendingFeature>,
    prev_uvs: HashMap<u64, Vector2<f64>>,
    prev_t_wc: Option<Matrix4<f64>>,
    prev_stamp: f64,
}

impl Sparse3DFilter {
    pub fn new(k: Matrix3<f64>, chart: Sparse3DChart, settings: SparseVogSettings) -> Self {
        let sigma_norm_sq = (settings.sigma_pixel / k[(0, 0)]).powi(2);
        Self {
            k,
            chart,
            settings,
            sigma_norm_sq,
            features: Vec::new(),
            feature_slots: HashMap::new(),
            pending: HashMap::new(),
            prev_uvs: HashMap::new(),
            prev_t_wc: None,
            prev_stamp: -1.0,
        }
    }

    pub fn polar3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::Polar, settings)
    }

    pub fn invdepth3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::InvDepth, settings)
    }

    /// ρ-first: additive inverse-depth EKF (see `Sparse3DChart::InvDepthAdditive`).
    pub fn invdepth_additive3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::InvDepthAdditive, settings)
    }

    pub fn update(
        &mut self,
        measurement: &VisionMeasurement,
        t_wc: &Matrix4<f64>,
        p_vv: Option<&Matrix3<f64>>,
        p_ww: Option<&Matrix3<f64>>,
    ) {
        let stamp = measurement.stamp;
        let curr_uvs: HashMap<u64, Vector2<f64>> = measurement
            .cam_coordinates
            .iter()
            .map(|(&id, &uv)| (id, Vector2::new(uv[0] as f64, uv[1] as f64)))
            .collect();

        let Some(prev_t_wc) = self.prev_t_wc else {
            self.prev_t_wc = Some(*t_wc);
            self.prev_stamp = stamp;
            self.prev_uvs = curr_uvs;
            return;
        };

        let dt = (stamp - self.prev_stamp).max(0.0);
        let t_cw_curr = t_wc.try_inverse().unwrap_or_else(Matrix4::identity);

        #[cfg(feature = "parallel")]
        let reset_features: Vec<_> = {
            use rayon::prelude::*;

            let chart = self.chart;
            let settings = self.settings.clone();
            let k = self.k;
            self.features
                .par_iter_mut()
                .filter_map(|feat| {
                    let fid = feat.feat_id;
                    if !self.prev_uvs.contains_key(&fid) {
                        return None;
                    }
                    let uv_curr = curr_uvs.get(&fid)?;
                    (!update_feature_3d(
                        chart, &k, &settings, feat, uv_curr, &t_cw_curr, p_vv, p_ww, dt,
                    ))
                    .then_some(fid)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let reset_features = {
            let mut reset_features = Vec::new();
            for feat in &mut self.features {
                let fid = feat.feat_id;
                if !self.prev_uvs.contains_key(&fid) {
                    continue;
                }

                let Some(uv_curr) = curr_uvs.get(&fid) else {
                    continue;
                };

                if !update_feature_3d(
                    self.chart,
                    &self.k,
                    &self.settings,
                    feat,
                    uv_curr,
                    &t_cw_curr,
                    p_vv,
                    p_ww,
                    dt,
                ) {
                    reset_features.push(fid);
                }
            }
            reset_features
        };

        let reset_set: HashSet<_> = reset_features.iter().copied().collect();
        let mut curr_ids: Vec<_> = curr_uvs.keys().copied().collect();
        curr_ids.sort_unstable();
        for fid in curr_ids {
            let &uv_curr = curr_uvs
                .get(&fid)
                .expect("sorted current ID must exist in current UV map");
            let Some(&uv_prev) = self.prev_uvs.get(&fid) else {
                continue;
            };
            if self.feature_slots.contains_key(&fid) || reset_set.contains(&fid) {
                continue;
            }

            if !self.pending.contains_key(&fid)
                && self.feature_slots.len() + self.pending.len() >= self.settings.max_pool_size
            {
                continue;
            }

            let pending = self
                .pending
                .entry(fid)
                .or_insert_with(|| PendingFeature {
                    ref_t_wc: prev_t_wc,
                    ref_uv: uv_prev,
                    ref_stamp: self.prev_stamp,
                })
                .clone();

            let t_curr_ref = t_cw_curr * pending.ref_t_wc;
            let r_ref = t_curr_ref.fixed_view::<3, 3>(0, 0).into_owned();
            let t_ref = t_curr_ref.fixed_view::<3, 1>(0, 3).into_owned();
            let (z_obs, drive) = triangulate_pixel(
                &self.k,
                &self.settings,
                &pending.ref_uv,
                &uv_curr,
                &r_ref,
                &t_ref,
            );
            if z_obs <= 0.0 || drive * self.k[(0, 0)] < self.settings.reanchor_flow_px {
                continue;
            }

            let baseline_tau_sq =
                baseline_tau(p_vv, &t_ref, (stamp - pending.ref_stamp).max(dt), dt);
            // Anchor the landmark in the CURRENT camera frame (= anchor frame);
            // the IEKF group element x (init identity) carries all later
            // refinement, so we never re-anchor. q0 is the fixed origin.
            let position = position_from_depth(&self.k, &uv_curr, z_obs);
            let covariance = init_cov_3d(
                self.chart,
                &self.settings,
                self.sigma_norm_sq,
                &position,
                z_obs,
                drive,
                baseline_tau_sq,
            );
            // IEKF covariance lives in Euclidean error coords about q0. At
            // creation x = id and the anchor frame is the current frame, so the
            // Euclidean init cov is the chart cov mapped back through the chart
            // Jacobian (chart_to_euc . cov . chart_to_euc^T).
            let j_c2e = chart_to_euc_jac(self.chart, &position);
            let sigma = j_c2e * covariance * j_c2e.transpose();
            // ρ-first additive state: anchor-frame (alpha,beta,rho) and its cov.
            // For InvDepthAdditive, `covariance` (from init_cov_3d) is already the
            // Euclidean init cov (euc_to_chart_jac == I), which is what we cache;
            // map it to (alpha,beta,rho) coords for inv_p via the euclid->invdepth
            // point Jacobian.
            let (inv_s, inv_p) = if self.chart == Sparse3DChart::InvDepthAdditive {
                let s = Vector3::new(
                    position[0] / position[2],
                    position[1] / position[2],
                    1.0 / position[2],
                );
                let j_e2i = conv_euc2ind(&position);
                (s, j_e2i * covariance * j_e2i.transpose())
            } else {
                (Vector3::zeros(), Matrix3::zeros())
            };
            let feat = FeatureState3D {
                feat_id: fid,
                position,
                covariance,
                q0: position,
                x: SOT3::identity(),
                sigma,
                inv_s,
                inv_p,
                anchor_t_wc: *t_wc,
                a: self.settings.a_init,
                b: self.settings.b_init,
                track_length: 1,
                ref_uv: uv_curr,
                ref_stamp: stamp,
                last_nis: f64::NAN,
            };
            self.insert_feature(feat);
            self.pending.remove(&fid);
        }

        for &fid in &reset_features {
            self.pending.remove(&fid);
        }

        self.remove_features(|feat| {
            reset_set.contains(&feat.feat_id) || !curr_uvs.contains_key(&feat.feat_id)
        });
        self.features.sort_by_key(|feat| feat.feat_id);
        for (slot, feat) in self.features.iter().enumerate() {
            self.feature_slots.insert(feat.feat_id, slot);
        }
        self.pending.retain(|id, _| curr_uvs.contains_key(id));
        self.prev_t_wc = Some(*t_wc);
        self.prev_stamp = stamp;
        self.prev_uvs = curr_uvs;
    }

    pub fn query(&self, fid: u64) -> (f64, f64) {
        let Some(feat) = self.feature(fid) else {
            return (-1.0, f64::INFINITY);
        };
        let depth = feat.depth_for_chart(self.chart);
        let depth_var = feat.depth_variance_for_chart(self.chart);
        if depth <= 0.0 || feat.track_length < self.settings.min_track_length {
            return (-1.0, f64::INFINITY);
        }
        if feat.inlier_ratio() < self.settings.conv_inlier_ratio
            || depth_var > self.settings.conv_variance_threshold
        {
            return (-1.0, f64::INFINITY);
        }
        (depth, depth_var)
    }

    pub fn query_range(&self, fid: u64) -> (f64, f64) {
        let Some(feat) = self.feature(fid) else {
            return (-1.0, f64::INFINITY);
        };
        let range = feat.position.norm();
        let range_var = feat.range_variance_for_chart(self.chart);
        if range <= 0.0 || feat.track_length < self.settings.min_track_length {
            return (-1.0, f64::INFINITY);
        }
        if feat.inlier_ratio() < self.settings.conv_inlier_ratio
            || range_var > self.settings.conv_variance_threshold
        {
            return (-1.0, f64::INFINITY);
        }
        (range, range_var)
    }

    pub fn feature(&self, fid: u64) -> Option<&FeatureState3D> {
        let &slot = self.feature_slots.get(&fid)?;
        self.features.get(slot)
    }

    pub fn has_track(&self, fid: u64) -> bool {
        self.feature_slots.contains_key(&fid) || self.pending.contains_key(&fid)
    }

    pub fn features_iter(&self) -> impl Iterator<Item = &FeatureState3D> {
        self.features.iter()
    }

    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    pub fn chart(&self) -> Sparse3DChart {
        self.chart
    }

    fn insert_feature(&mut self, feat: FeatureState3D) {
        let slot = self.features.len();
        self.feature_slots.insert(feat.feat_id, slot);
        self.features.push(feat);
    }

    fn remove_features(&mut self, mut should_remove: impl FnMut(&FeatureState3D) -> bool) {
        let mut i = 0;
        while i < self.features.len() {
            if should_remove(&self.features[i]) {
                let removed_id = self.features[i].feat_id;
                self.feature_slots.remove(&removed_id);
                self.features.swap_remove(i);
                if let Some(moved) = self.features.get(i) {
                    self.feature_slots.insert(moved.feat_id, i);
                }
            } else {
                i += 1;
            }
        }
    }
}

fn position_from_depth(k: &Matrix3<f64>, uv: &Vector2<f64>, depth: f64) -> Vector3<f64> {
    let x = (uv[0] - k[(0, 2)]) / k[(0, 0)];
    let y = (uv[1] - k[(1, 2)]) / k[(1, 1)];
    Vector3::new(x * depth, y * depth, depth)
}

fn chart_to_euc_jac(chart: Sparse3DChart, q: &Vector3<f64>) -> Matrix3<f64> {
    match chart {
        Sparse3DChart::Polar => conv_normal2euc(q),
        Sparse3DChart::InvDepth => conv_ind2euc(q),
        // The additive chart caches `covariance` already in Euclidean coords.
        Sparse3DChart::InvDepthAdditive => Matrix3::identity(),
    }
}

fn euc_to_chart_jac(chart: Sparse3DChart, q: &Vector3<f64>) -> Matrix3<f64> {
    match chart {
        Sparse3DChart::Polar => conv_euc2normal(q),
        Sparse3DChart::InvDepth => conv_euc2ind(q),
        Sparse3DChart::InvDepthAdditive => Matrix3::identity(),
    }
}

fn apply_chart_delta(chart: Sparse3DChart, q: &Vector3<f64>, delta: &Vector3<f64>) -> Vector3<f64> {
    match chart {
        Sparse3DChart::Polar => point_chart_normal_inv(delta, q),
        Sparse3DChart::InvDepth => point_chart_invdepth_inv(delta, q),
        // additive chart applies the delta in (alpha,beta,rho); not used via this
        // helper (the additive update is self-contained), but keep the match total.
        Sparse3DChart::InvDepthAdditive => q + delta,
    }
}

/// Per-landmark measurement update, dispatched by chart: the SOT(3) IEKF for
/// `Polar`/`InvDepth`, the additive inverse-depth EKF for `InvDepthAdditive`.
fn update_feature_3d(
    chart: Sparse3DChart,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    match chart {
        Sparse3DChart::InvDepthAdditive => {
            invdepth_additive_update_3d(k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, dt)
        }
        _ => iekf_update_3d(chart, k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, dt),
    }
}

/// ρ-first additive inverse-depth EKF update for one landmark.
///
/// State `s = (alpha, beta, rho) = (X/Z, Y/Z, 1/Z)` of the landmark in the anchor
/// camera frame, covariance `feat.inv_p` in those coords. The known relative pose
/// anchor->current is folded into the pinhole measurement; the update is a plain
/// EKF (the inverse-depth refinement group is abelian / flat — `s <- s + gamma`),
/// with the same Gaussian-Beta inlier weighting and optional process-noise floor
/// as the SOT(3) path. No log-depth, no re-charting -> no sequential-linearisation
/// overconfidence (see docs/sparse3d_invdepth_rewrite.md).
/// Anchor-frame inverse-depth chart of a 3-point: (X/Z, Y/Z, 1/Z). Involution
/// (its own inverse), so it maps both P_anchor -> s and back.
fn pa_chart(p: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(p[0] / p[2], p[1] / p[2], 1.0 / p[2])
}

/// Sigma-point (unscented) propagation of the rotation process noise `p_ww`
/// through the full `exp(-[δφ]×)·q_c -> inverse-depth-chart` map, with
/// `δφ ~ N(0, P_ww·dt²)`. Returns the added chart-space covariance and the
/// 2nd-order mean (bias) shift the first-order `[q_c]× P_ww [q_c]×ᵀ` term drops.
/// 6 symmetric points, κ=0 (n+λ=n ⇒ scale √3, weight 1/6).
fn unscented_rotation_chart(
    p_ww: &Matrix3<f64>,
    dt2: f64,
    q_c: &Vector3<f64>,
    r_ca: &Matrix3<f64>,
    t_ca_t: &Vector3<f64>,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let chol = (p_ww * dt2).cholesky()?;
    let l = chol.l();
    let scale = 3.0_f64.sqrt();
    let r_inv = r_ca.transpose();
    let s0 = pa_chart(&(r_inv * (q_c - t_ca_t))); // == nominal inv_s
    let map = |dphi: Vector3<f64>| {
        let rot = nalgebra::Rotation3::from_scaled_axis(-dphi).into_inner();
        pa_chart(&(r_inv * (rot * q_c - t_ca_t)))
    };
    let mut sp = [Vector3::<f64>::zeros(); 6];
    for i in 0..3 {
        let col: Vector3<f64> = l.column(i).into_owned() * scale;
        sp[2 * i] = map(col);
        sp[2 * i + 1] = map(-col);
    }
    let mut mean = Vector3::zeros();
    for s in &sp {
        mean += s;
    }
    mean /= 6.0;
    let mut q_chart = Matrix3::zeros();
    for s in &sp {
        let d = s - mean;
        q_chart += d * d.transpose();
    }
    q_chart /= 6.0;
    if !q_chart.iter().all(|x| x.is_finite()) || !mean.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some((q_chart, mean - s0))
}

fn invdepth_additive_update_3d(
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    let t_ca = t_cw_curr * feat.anchor_t_wc;
    let r_ca = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
    let t_ca_t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();

    // anchor-frame point P_anchor(s) and its Jacobian dP_anchor/ds.
    let pa_of = |s: &Vector3<f64>| Vector3::new(s[0] / s[2], s[1] / s[2], 1.0 / s[2]);
    let jpa_of = |s: &Vector3<f64>| {
        let (a, b, r) = (s[0], s[1], s[2]);
        let r2 = r * r;
        Matrix3::new(
            1.0 / r,
            0.0,
            -a / r2,
            0.0,
            1.0 / r,
            -b / r2,
            0.0,
            0.0,
            -1.0 / r2,
        )
    };

    if dt > 0.0 && (p_vv.is_some() || p_ww.is_some() || settings.range_walk_var > 0.0) {
        let q_c = r_ca * pa_of(&feat.inv_s) + t_ca_t;
        if q_c[2] > settings.min_depth {
            let j_g = r_ca * jpa_of(&feat.inv_s);
            if let Some(j_inv) = j_g.try_inverse() {
                let dt2 = dt * dt;
                // Linear (translation + radial floor) terms, pulled back into the
                // chart by J_inv. Both are exact (translation is additive in q_c;
                // the floor is a heuristic).
                // When pose_measurement is set, the fed-pose terms (p_vv, p_ww) are
                // folded into R below instead — only the radial floor stays here.
                let mut q_cur = Matrix3::zeros();
                if let Some(pvv) = p_vv.filter(|_| !settings.pose_measurement) {
                    q_cur += pvv * dt2;
                }
                if settings.range_walk_var > 0.0 {
                    let r_hat = q_c / q_c.norm();
                    q_cur +=
                        settings.range_walk_var * q_c.norm_squared() * (r_hat * r_hat.transpose());
                }
                let mut sigma = feat.inv_p + j_inv * q_cur * j_inv.transpose();
                // Rotation term: unscented (2nd-order: variance + mean bias) or
                // the first-order [q_c]× P_ww [q_c]×ᵀ linearisation.
                let mut inv_s_shift = Vector3::zeros();
                if let Some(pww) = p_ww.filter(|_| !settings.pose_measurement) {
                    if settings.rotation_unscented {
                        if let Some((q_chart, bias)) =
                            unscented_rotation_chart(pww, dt2, &q_c, &r_ca, &t_ca_t)
                        {
                            sigma += q_chart;
                            inv_s_shift = bias;
                        }
                    } else {
                        let qx = base_skew(&q_c);
                        sigma += j_inv * (qx * pww * qx.transpose() * dt2) * j_inv.transpose();
                    }
                }
                feat.inv_p = 0.5 * (sigma + sigma.transpose());
                feat.inv_s += inv_s_shift;
            }
        }
    }

    let mut r_meas = Matrix2::identity() * settings.sigma_pixel.powi(2);

    let q_c = r_ca * pa_of(&feat.inv_s) + t_ca_t;
    if q_c[2] < settings.min_depth {
        return false;
    }
    let (xc, yc, zc) = (q_c[0], q_c[1], q_c[2]);
    let z2 = zc * zc;
    let proj = Matrix2x3::new(fx / zc, 0.0, -fx * xc / z2, 0.0, fy / zc, -fy * yc / z2);
    // Measurement-side pose uncertainty (the "consider"/Schmidt treatment of the
    // fed pose): the pixel comes from the true pose but the filter is fed a noisy
    // pose, so pose error is a *measurement* discrepancy, folded into R so it
    // enters both S and the Joseph posterior K·R·Kᵀ. Geometrically self-scaling:
    //   translation  J_t = ∂pixel/∂ρ = -proj        -> proj·P_vv·dt²·projᵀ  (∝1/Z²)
    //   rotation     J_φ = ∂pixel/∂φ = proj·[q_c]×   -> J_φ·P_ww·dt²·J_φᵀ    (depth-indep)
    if settings.pose_measurement && dt > 0.0 {
        let dt2 = dt * dt;
        if let Some(pvv) = p_vv {
            r_meas += proj * (pvv * dt2) * proj.transpose();
        }
        if let Some(pww) = p_ww {
            let jphi = proj * base_skew(&q_c);
            r_meas += jphi * (pww * dt2) * jphi.transpose();
        }
        // Anchor-pose uncertainty: the anchor frame (T_wc_anchor) was itself set
        // from a noisy pose at init; that error is fixed for the landmark's life.
        // J_a = proj·R_ca·[-[P_anchor]× | I]  (P_anchor = anchor-frame point).
        if settings.anchor_measurement {
            let rca_proj = proj * r_ca; // = J_a translation block (2x3)
            if let Some(pvv) = p_vv {
                r_meas += rca_proj * (pvv * dt2) * rca_proj.transpose();
            }
            if let Some(pww) = p_ww {
                let j_a = rca_proj * (-base_skew(&pa_of(&feat.inv_s)));
                r_meas += j_a * (pww * dt2) * j_a.transpose();
            }
        }
    }
    let c = proj * r_ca * jpa_of(&feat.inv_s); // dh/ds (2x3)
    let s_mat = c * feat.inv_p * c.transpose() + r_meas;
    let Some(s_inv) = (s_mat + Matrix2::identity() * 1e-8).try_inverse() else {
        return true;
    };
    let det_s = s_mat.determinant();
    if det_s < 1e-30 {
        return true;
    }
    let y_pred = Vector2::new(fx * xc / zc + cx, fy * yc / zc + cy);
    let residual = uv_obs - y_pred;
    let maha_sq = (residual.transpose() * s_inv * residual)[(0, 0)];
    feat.last_nis = maha_sq;
    if settings.mahalanobis_reset_chi2 > 0.0 && maha_sq > settings.mahalanobis_reset_chi2 {
        return false;
    }
    let gain = feat.inv_p * c.transpose() * s_inv; // 3x2
    let full_delta = gain * residual;
    let i_kc = Matrix3::identity() - gain * c;
    let p_post = i_kc * feat.inv_p * i_kc.transpose() + gain * r_meas * gain.transpose();

    // Gaussian-Beta inlier weighting (mirrors iekf_update_3d).
    let gauss_pdf = (-0.5 * maha_sq).exp() / ((2.0 * std::f64::consts::PI).powi(2) * det_s).sqrt();
    let uniform_prior = 1.0 / (fx * fy * 4.0);
    let ab = feat.a + feat.b;
    if ab <= 0.0 {
        return true;
    }
    let c1 = (feat.a / ab) * gauss_pdf;
    let c2 = (feat.b / ab) * uniform_prior;
    let z_norm = c1 + c2;
    if z_norm < 1e-30 {
        feat.b = (feat.b + 1.0).min(settings.ab_max);
        return true;
    }
    let w1 = c1 / z_norm;
    let w2 = c2 / z_norm;

    let gamma = w1 * full_delta;
    let p_new = w1 * p_post + w2 * feat.inv_p + w1 * w2 * (full_delta * full_delta.transpose());
    if p_new
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|v| *v <= 0.0 || !v.is_finite())
    {
        return true;
    }

    // Additive state update (abelian / flat): s <- s + gamma. No re-charting.
    feat.inv_s += gamma;
    feat.inv_p = 0.5 * (p_new + p_new.transpose());
    feat.track_length += 1;
    update_beta(settings, feat, w1, w2);

    // Refresh cached current-frame estimate + Euclidean covariance for the API.
    let j_g = r_ca * jpa_of(&feat.inv_s); // dP_cur/ds
    feat.position = r_ca * pa_of(&feat.inv_s) + t_ca_t;
    let cov_euc = j_g * feat.inv_p * j_g.transpose();
    feat.covariance = 0.5 * (cov_euc + cov_euc.transpose());
    true
}

/// True IEKF measurement update for one landmark.
///
/// The landmark is static in its anchor frame; the group element `x` carries all
/// refinement and the covariance `sigma` lives in fixed Euclidean error coords
/// about the origin `q0` (never re-charted -> the IEKF consistency property).
/// The known relative pose anchor->current is folded into the measurement model.
fn iekf_update_3d(
    chart: Sparse3DChart,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    // Known relative pose anchor -> current camera.
    let t_ca = t_cw_curr * feat.anchor_t_wc;
    let r_ca = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
    let t_ca_t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();

    // Lift: euclid error eps(3) -> sot(3) algebra (4D = [omega(3), log-scale(1)])
    // at the origin q0. This is the same map used for the group update below, so
    // the (numerical) output Jacobian is guaranteed consistent with it.
    let q0 = feat.q0;
    let q0n2 = q0.norm_squared();
    let m2g_top = -base_skew(&q0) / q0n2;
    let m2g_bot = (-q0 / q0n2).transpose();
    let lift = |g: &Vector3<f64>| {
        let wo = m2g_top * g;
        let ws = (m2g_bot * g)[(0, 0)];
        Vector4::new(wo[0], wo[1], wo[2], ws)
    };
    let q_hat_a_of = |x: &SOT3| x.act_inverse(&q0);
    // d q_hat_a / d eps via central differences of the actual group map.
    let dq_hat_a = |x: &SOT3| {
        let h = 1e-6;
        let mut m = Matrix3::zeros();
        for j in 0..3 {
            let mut dg = Vector3::zeros();
            dg[j] = h;
            let qp = x.compose(&SOT3::exp(&lift(&dg))).act_inverse(&q0);
            let qm = x.compose(&SOT3::exp(&lift(&(-dg)))).act_inverse(&q0);
            m.set_column(j, &((qp - qm) / (2.0 * h)));
        }
        m
    };

    // Per-step process noise: the IEKF has no propagation, so without this the
    // static-landmark Σ shrinks monotonically and collapses below the
    // accumulated relative-pose / triangulation uncertainty (the source of the
    // depth-growing NEES). Two contributions, both formed in the CURRENT camera
    // frame and pulled back into the fixed-q0 error coords by J_c = R_ca·G:
    //   * p_vv·dt²            -- translation-rate (velocity) covariance,
    //   * range_walk_var·‖q_c‖²·r̂r̂ᵀ -- a depth-scaled radial (range) random-walk
    //     floor that stops Σ from going below the un-modelled range bias.
    if dt > 0.0 && (p_vv.is_some() || p_ww.is_some() || settings.range_walk_var > 0.0) {
        let p = q_hat_a_of(&feat.x);
        let q_c = r_ca * p + t_ca_t;
        if q_c[2] > settings.min_depth {
            let j_c = r_ca * dq_hat_a(&feat.x);
            if let Some(j_inv) = j_c.try_inverse() {
                let dt2 = dt * dt;
                let mut q_cur = Matrix3::zeros();
                if let Some(pvv) = p_vv {
                    q_cur += pvv * dt2;
                }
                if let Some(pww) = p_ww {
                    let qx = base_skew(&q_c);
                    q_cur += qx * pww * qx.transpose() * dt2;
                }
                if settings.range_walk_var > 0.0 {
                    let r_hat = q_c / q_c.norm();
                    q_cur +=
                        settings.range_walk_var * q_c.norm_squared() * (r_hat * r_hat.transpose());
                }
                let sigma = feat.sigma + j_inv * q_cur * j_inv.transpose();
                feat.sigma = 0.5 * (sigma + sigma.transpose());
            }
        }
    }

    let r_meas = Matrix2::identity() * settings.sigma_pixel.powi(2);

    // Each branch produces the inlier-conditioned update terms shared by the
    // Gaussian-Beta tail below: the gating Mahalanobis^2 and det(S), the full
    // (un-weighted) state correction in eps coords, and the posterior covariance
    // P assuming the measurement is an inlier.
    let (maha_sq, det_s, full_delta, p_post) = match settings.second_order_mode {
        SecondOrderMode::Analytic => {
            // Option A -- analytic second-order EqF. Restores the dropped
            // projective curvature: bias-corrects the prediction by ½tr(H_mΣ)
            // and inflates S by Λ_kl = ½tr(H_kΣH_lΣ), driving NEES -> dim at
            // weak parallax. Symbols and derivation:
            // ECHO-LI-notes/docs/sparse3d_secondorder_eqf_derivation.md (§§3-5).
            let p = q_hat_a_of(&feat.x); // q̂_a
            let q_c = r_ca * p + t_ca_t;
            if q_c[2] < settings.min_depth {
                return false;
            }
            let (xc, yc, zc) = (q_c[0], q_c[1], q_c[2]);
            let z2 = zc * zc;
            // Projection Jacobian P (2x3) and per-channel Hessians Π_u, Π_v (eqs 10-11).
            let proj = Matrix2x3::new(fx / zc, 0.0, -fx * xc / z2, 0.0, fy / zc, -fy * yc / z2);
            let pi_u = Matrix3::new(
                0.0,
                0.0,
                -fx / z2,
                0.0,
                0.0,
                0.0,
                -fx / z2,
                0.0,
                2.0 * fx * xc / (z2 * zc),
            );
            let pi_v = Matrix3::new(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -fy / z2,
                0.0,
                -fy / z2,
                2.0 * fy * yc / (z2 * zc),
            );
            // EKF Jacobian C = P R G, G the (numeric) group Jacobian -- same map
            // as the x <- x.exp update, so C is convention-consistent (eq 9 linear).
            let g = dq_hat_a(&feat.x);
            let c = proj * r_ca * g;
            let rg = r_ca * g;
            let pr = proj * r_ca; // (PR)_m weights the action-curvature term
            let pr_u = pr.row(0).transpose();
            let pr_v = pr.row(1).transpose();
            // Cholesky Σ = L Lᵀ; the columns ℓ_j whiten the directional Hessian.
            let Some(chol) = feat.sigma.cholesky() else {
                return true;
            };
            let l = chol.l();
            let mut omega = [Vector3::zeros(); 3];
            let mut alpha = [0.0f64; 3];
            let mut wxp = [Vector3::zeros(); 3]; // ω_j × p
            let mut b = [Vector3::zeros(); 3]; // R G ℓ_j
            for j in 0..3 {
                let lj = l.column(j).into_owned();
                let oj = m2g_top * lj;
                omega[j] = oj;
                alpha[j] = (m2g_bot * lj)[(0, 0)];
                wxp[j] = oj.cross(&p);
                b[j] = rg * lj;
            }
            // Whitened output Hessians H̃_u, H̃_v (symmetric 3x3, eq 9 + eq 13):
            //   (H̃_m)_{ij} = b_iᵀ Π_m b_j + (PR)_m · B(ℓ_i, ℓ_j),
            // B the polarization of the SOT(3) action's quadratic term q_a^{(2)}.
            let mut ht_u = Matrix3::zeros();
            let mut ht_v = Matrix3::zeros();
            for i in 0..3 {
                for j in i..3 {
                    let b_act = 0.5 * (omega[i].dot(&p) * omega[j] + omega[j].dot(&p) * omega[i])
                        - omega[i].dot(&omega[j]) * p
                        + alpha[i] * wxp[j]
                        + alpha[j] * wxp[i]
                        + alpha[i] * alpha[j] * p;
                    let hu = b[i].dot(&(pi_u * b[j])) + pr_u.dot(&b_act);
                    let hv = b[i].dot(&(pi_v * b[j])) + pr_v.dot(&b_act);
                    ht_u[(i, j)] = hu;
                    ht_u[(j, i)] = hu;
                    ht_v[(i, j)] = hv;
                    ht_v[(j, i)] = hv;
                }
            }
            // Bias-corrected prediction ŷ_m = h_m(0) + ½ tr(H̃_m) (eq 5).
            let y_pred = Vector2::new(fx * xc / zc + cx, fy * yc / zc + cy);
            let y_hat = y_pred + 0.5 * Vector2::new(ht_u.trace(), ht_v.trace());
            // Inflation Λ_kl = ½ tr(H̃_k H̃_l) = ½ <H̃_k, H̃_l>_F  (Gram, PSD) (eq 6).
            let lam_uv = 0.5 * ht_u.dot(&ht_v);
            let lambda = Matrix2::new(0.5 * ht_u.dot(&ht_u), lam_uv, lam_uv, 0.5 * ht_v.dot(&ht_v));
            let s = c * feat.sigma * c.transpose() + r_meas + lambda;
            let Some(s_inv) = (s + Matrix2::identity() * 1e-8).try_inverse() else {
                return true;
            };
            let det_s = s.determinant();
            if det_s < 1e-30 {
                return true;
            }
            let residual = uv_obs - y_hat;
            let maha_sq = (residual.transpose() * s_inv * residual)[(0, 0)];
            if settings.mahalanobis_reset_chi2 > 0.0 && maha_sq > settings.mahalanobis_reset_chi2 {
                return false;
            }
            // Cross-cov is Σ Cᵀ to this order, so the gain keeps the EKF shape (eq 7-8).
            let gain = feat.sigma * c.transpose() * s_inv;
            let full_delta = gain * residual;
            let p_post = feat.sigma - gain * s * gain.transpose();
            (maha_sq, det_s, full_delta, p_post)
        }
        SecondOrderMode::Off => {
            // Iterated EKF: relinearize the projection at the posterior to cancel
            // the bearing-only depth bias at weak parallax (cf. ROVIO / the 1D
            // sparse_vogiatzis update). delta is the total correction from the
            // prior x, in eps coords; full_delta/c/gain are taken from the FINAL
            // linearization; gating (maha, det_s) uses the PRIOR innovation
            // (it == 0). iekf_iterations == 1 reduces exactly to the plain EKF.
            let mut delta = Vector3::zeros();
            let mut c = Matrix2x3::<f64>::zeros();
            let mut gain = Matrix3x2::<f64>::zeros();
            let mut full_delta = Vector3::zeros();
            let mut maha_sq = 0.0;
            let mut det_s = 1.0;
            for it in 0..settings.iekf_iterations.max(1) {
                let x_it = feat.x.compose(&SOT3::exp(&lift(&delta)));
                let q_a = q_hat_a_of(&x_it);
                let q_c = r_ca * q_a + t_ca_t;
                if q_c[2] < settings.min_depth {
                    return false;
                }
                let zc = q_c[2];
                let proj = Matrix2x3::new(
                    fx / zc,
                    0.0,
                    -fx * q_c[0] / (zc * zc),
                    0.0,
                    fy / zc,
                    -fy * q_c[1] / (zc * zc),
                );
                c = proj * r_ca * dq_hat_a(&x_it);
                let s = c * feat.sigma * c.transpose() + r_meas;
                let Some(s_inv) = (s + Matrix2::identity() * 1e-8).try_inverse() else {
                    return true;
                };
                gain = feat.sigma * c.transpose() * s_inv;
                let y_pred = Vector2::new(fx * q_c[0] / zc + cx, fy * q_c[1] / zc + cy);
                let residual = uv_obs - y_pred;
                // Gauss-Newton step from the prior: delta <- K (residual + C delta).
                full_delta = gain * (residual + c * delta);
                if it == 0 {
                    det_s = s.determinant();
                    if det_s < 1e-30 {
                        return true;
                    }
                    maha_sq = (residual.transpose() * s_inv * residual)[(0, 0)];
                    if settings.mahalanobis_reset_chi2 > 0.0
                        && maha_sq > settings.mahalanobis_reset_chi2
                    {
                        return false;
                    }
                }
                delta = full_delta;
            }
            let i_kc = Matrix3::identity() - gain * c;
            let p_kalman = i_kc * feat.sigma * i_kc.transpose() + gain * r_meas * gain.transpose();
            (maha_sq, det_s, full_delta, p_kalman)
        }
    };
    feat.last_nis = maha_sq;

    // Gaussian-Beta inlier weighting.
    let gauss_pdf = (-0.5 * maha_sq).exp() / ((2.0 * std::f64::consts::PI).powi(2) * det_s).sqrt();
    let uniform_prior = 1.0 / (fx * fy * 4.0);
    let ab = feat.a + feat.b;
    if ab <= 0.0 {
        return true;
    }
    let c1 = (feat.a / ab) * gauss_pdf;
    let c2 = (feat.b / ab) * uniform_prior;
    let z_norm = c1 + c2;
    if z_norm < 1e-30 {
        feat.b = (feat.b + 1.0).min(settings.ab_max);
        return true;
    }
    let w1 = c1 / z_norm;
    let w2 = c2 / z_norm;

    let gamma = w1 * full_delta; // GB-weighted correction (from the chosen update)

    let sigma_new = w1 * p_post + w2 * feat.sigma + w1 * w2 * (full_delta * full_delta.transpose());
    if sigma_new
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|v| *v <= 0.0 || !v.is_finite())
    {
        return true;
    }

    // Group update: x <- x . exp(lift(gamma)). sigma stays in the fixed q0 error
    // frame (no re-charting) -- this is what makes it a consistent IEKF.
    feat.x = feat.x.compose(&SOT3::exp(&lift(&gamma)));
    feat.sigma = 0.5 * (sigma_new + sigma_new.transpose());
    feat.track_length += 1;
    update_beta(settings, feat, w1, w2);

    // Refresh cached current-frame estimate + chart covariance for the API.
    let q_hat_a2 = q_hat_a_of(&feat.x);
    let q_hat_c2 = r_ca * q_hat_a2 + t_ca_t;
    let j_c = r_ca * dq_hat_a(&feat.x);
    let cov_c_euc = j_c * feat.sigma * j_c.transpose();
    feat.position = q_hat_c2;
    let e2c = euc_to_chart_jac(chart, &q_hat_c2);
    let cc = e2c * cov_c_euc * e2c.transpose();
    feat.covariance = 0.5 * (cc + cc.transpose());
    true
}

fn init_cov_3d(
    chart: Sparse3DChart,
    settings: &SparseVogSettings,
    sigma_norm_sq: f64,
    position: &Vector3<f64>,
    z_obs: f64,
    drive: f64,
    baseline_tau_sq: f64,
) -> Matrix3<f64> {
    let z = z_obs.max(settings.min_depth);
    let x_over_z = position[0] / z;
    let y_over_z = position[1] / z;
    let j_bearing = nalgebra::Matrix3x2::new(z, 0.0, 0.0, z, 0.0, 0.0);
    let mut p_euc = sigma_norm_sq * (j_bearing * j_bearing.transpose());
    let var_z = if drive > 1e-12 {
        z.powi(4) * sigma_norm_sq / (drive * drive) + z * z * baseline_tau_sq
    } else {
        settings.init_depth_var * z * z
    };
    let j_depth = Vector3::new(x_over_z, y_over_z, 1.0);
    p_euc += var_z * (j_depth * j_depth.transpose());

    let m = euc_to_chart_jac(chart, position);
    let mut cov = m * p_euc * m.transpose();
    cov = 0.5 * (cov + cov.transpose());
    let eig = cov.symmetric_eigen().eigenvalues;
    let min_eig = eig[0].min(eig[1]).min(eig[2]);
    if min_eig < 1e-12 {
        cov += Matrix3::identity() * (1e-12 - min_eig);
    }
    cov
}

#[allow(dead_code)]
fn bearing_update_3d(
    chart: Sparse3DChart,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    y_observed: &Vector2<f64>,
) -> bool {
    let q = feat.position;
    if q[2].abs() < 1e-6 {
        return false;
    }
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let h = if settings.use_equivariant_output {
        // Equivariant output approximation (EqVIO). The reference defines C* in
        // EUCLIDEAN coords as the averaged DRho skew form
        // (EqFoutputMatrixCiStar_euclid); each chart's C* is then that Euclidean
        // C* mapped to the chart error coords by the chart->Euclid Jacobian
        // (invdepth uses ind2euc; polar uses conv_normal2euc). We reproduce that
        // pattern here, re-centred at the current estimate so QHat = I:
        //   DRho(b)  = proj_jac(b) . skew(b)               (2x3)
        //   C*_euc   = 0.5 (DRho(yTru) + DRho(yHat)) . (-skew(q)/|q|^2)
        //   h        = C*_euc . chart_to_euc_jac(chart, q)
        // The averaging uses the true measurement bearing yTru, reducing output
        // linearisation error to O(|eps|^3).
        let proj_jac = |b: &Vector3<f64>| {
            Matrix2x3::new(
                fx / b[2],
                0.0,
                -fx * b[0] / (b[2] * b[2]),
                0.0,
                fy / b[2],
                -fy * b[1] / (b[2] * b[2]),
            )
        };
        let d_rho = |b: &Vector3<f64>| proj_jac(b) * base_skew(b);
        let y_hat = q.normalize();
        let y_tru =
            Vector3::new((y_observed[0] - cx) / fx, (y_observed[1] - cy) / fy, 1.0).normalize();
        let c_euc = 0.5 * (d_rho(&y_tru) + d_rho(&y_hat)) * (-base_skew(&q) / q.norm_squared());
        c_euc * chart_to_euc_jac(chart, &q)
    } else {
        let h_euc = Matrix2x3::new(
            fx / q[2],
            0.0,
            -fx * q[0] / (q[2] * q[2]),
            0.0,
            fy / q[2],
            -fy * q[1] / (q[2] * q[2]),
        );
        h_euc * chart_to_euc_jac(chart, &q)
    };
    let y_pred = Vector2::new(fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy);
    let r = Matrix2::identity() * settings.sigma_pixel.powi(2);
    let s = h * feat.covariance * h.transpose() + r;
    let Some(s_inv) = (s + Matrix2::identity() * 1e-8).try_inverse() else {
        return true;
    };
    let gain = feat.covariance * h.transpose() * s_inv;
    let innovation = y_observed - y_pred;
    let maha_sq = (innovation.transpose() * s_inv * innovation)[(0, 0)];
    let det_s = s.determinant();
    if det_s < 1e-30 {
        return true;
    }
    if settings.mahalanobis_reset_chi2 > 0.0 && maha_sq > settings.mahalanobis_reset_chi2 {
        return false;
    }
    let gauss_pdf = (-0.5 * maha_sq).exp() / ((2.0 * std::f64::consts::PI).powi(2) * det_s).sqrt();
    let uniform_prior = 1.0 / (fx * fy * 4.0);
    let ab = feat.a + feat.b;
    if ab <= 0.0 {
        return true;
    }
    let c1 = (feat.a / ab) * gauss_pdf;
    let c2 = (feat.b / ab) * uniform_prior;
    let z_norm = c1 + c2;
    if z_norm < 1e-30 {
        feat.b = (feat.b + 1.0).min(settings.ab_max);
        return true;
    }
    let w1 = c1 / z_norm;
    let w2 = c2 / z_norm;
    let full_delta = gain * innovation;
    let delta = w1 * full_delta;
    let i_kh = Matrix3::identity() - gain * h;
    let p_kalman = i_kh * feat.covariance * i_kh.transpose() + gain * r * gain.transpose();
    let p_new =
        w1 * p_kalman + w2 * feat.covariance + w1 * w2 * (full_delta * full_delta.transpose());
    if p_new
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|v| *v <= 0.0 || !v.is_finite())
    {
        return true;
    }
    feat.position = apply_chart_delta(chart, &feat.position, &delta);
    feat.covariance = 0.5 * (p_new + p_new.transpose());
    update_beta(settings, feat, w1, w2);
    true
}

fn update_beta(settings: &SparseVogSettings, feat: &mut FeatureState3D, w1: f64, w2: f64) {
    let a = feat.a;
    let b = feat.b;
    let ab = a + b;
    let denom1 = ab + 1.0;
    let denom2 = denom1 * (ab + 2.0);
    let e_pi = (w1 * (a + 1.0) + w2 * a) / denom1;
    let e_pi2 = (w1 * (a + 1.0) * (a + 2.0) + w2 * a * (a + 1.0)) / denom2;
    let v_pi = e_pi2 - e_pi * e_pi;
    let (new_a, new_b) = if v_pi < 1e-6 || e_pi <= 1e-6 || e_pi >= 1.0 - 1e-6 {
        (a + w1, b + w2)
    } else {
        let factor = (e_pi * (1.0 - e_pi) / v_pi - 1.0).max(0.5);
        (e_pi * factor, (1.0 - e_pi) * factor)
    };
    feat.a = new_a.clamp(settings.ab_min, settings.ab_max);
    feat.b = new_b.clamp(settings.ab_min, settings.ab_max);
}

fn baseline_tau(p_vv: Option<&Matrix3<f64>>, t: &Vector3<f64>, dt_total: f64, dt: f64) -> f64 {
    let Some(p_vv) = p_vv else {
        return 0.0;
    };
    let norm_sq = t.norm_squared();
    if norm_sq <= 1e-16 || dt <= 0.0 {
        return 0.0;
    }
    let t_hat = t / norm_sq.sqrt();
    dt_total * dt * (t_hat.transpose() * p_vv * t_hat)[(0, 0)] / norm_sq
}

fn triangulate_pixel(
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    uv_prev: &Vector2<f64>,
    uv_curr: &Vector2<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> (f64, f64) {
    let x_curr = (uv_curr[0] - k[(0, 2)]) / k[(0, 0)];
    let y_curr = (uv_curr[1] - k[(1, 2)]) / k[(1, 1)];
    let x_prev = (uv_prev[0] - k[(0, 2)]) / k[(0, 0)];
    let y_prev = (uv_prev[1] - k[(1, 2)]) / k[(1, 1)];
    let aligned = r * Vector3::new(x_prev, y_prev, 1.0);
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};

    use super::*;

    fn settings() -> SparseVogSettings {
        SparseVogSettings {
            max_pool_size: 10,
            min_track_length: 3,
            conv_inlier_ratio: 0.5,
            conv_variance_threshold: 10.0,
            init_depth_var: 1.0,
            sigma_pixel: 0.5,
            reanchor_flow_px: 0.5,
            min_cos_sim: 0.90,
            max_depth: 100.0,
            ..Default::default()
        }
    }

    fn k() -> Matrix3<f64> {
        Matrix3::new(458.0, 0.0, 376.0, 0.0, 458.0, 240.0, 0.0, 0.0, 1.0)
    }

    fn pose(x: f64) -> Matrix4<f64> {
        let mut t = Matrix4::identity();
        t[(0, 3)] = x;
        t
    }

    fn project(k: &Matrix3<f64>, p: Vector3<f64>) -> Vector2<f32> {
        Vector2::new(
            (k[(0, 0)] * p[0] / p[2] + k[(0, 2)]) as f32,
            (k[(1, 1)] * p[1] / p[2] + k[(1, 2)]) as f32,
        )
    }

    fn update_with_point(filter: &mut Sparse3DFilter, i: usize, point_w: Vector3<f64>) {
        update_with_points(filter, i, &[(42, point_w)]);
    }

    fn update_with_points(filter: &mut Sparse3DFilter, i: usize, points: &[(u64, Vector3<f64>)]) {
        let t_wc = pose(i as f64 * 0.05);
        let mut coords = HashMap::new();
        for &(fid, point_w) in points {
            let p_c = point_w - Vector3::new(t_wc[(0, 3)], 0.0, 0.0);
            coords.insert(fid, project(&k(), p_c));
        }
        filter.update(
            &VisionMeasurement::new(i as f64 * 0.05, coords),
            &t_wc,
            None,
            None,
        );
    }

    fn feature(fid: u64) -> FeatureState3D {
        FeatureState3D {
            feat_id: fid,
            position: Vector3::new(0.0, 0.0, 3.0),
            covariance: Matrix3::identity(),
            q0: Vector3::new(0.0, 0.0, 3.0),
            x: SOT3::identity(),
            sigma: Matrix3::identity(),
            inv_s: Vector3::zeros(),
            inv_p: Matrix3::zeros(),
            anchor_t_wc: Matrix4::identity(),
            a: 1.0,
            b: 1.0,
            track_length: 3,
            ref_uv: Vector2::new(0.0, 0.0),
            ref_stamp: 0.0,
            last_nis: f64::NAN,
        }
    }

    #[test]
    fn dense_storage_updates_lookup_after_swap_remove() {
        let mut filter = Sparse3DFilter::polar3d(k(), settings());
        filter.insert_feature(feature(1));
        filter.insert_feature(feature(2));
        filter.insert_feature(feature(3));

        filter.remove_features(|feat| feat.feat_id == 1);

        assert_eq!(filter.feature_count(), 2);
        assert!(filter.feature(1).is_none());
        assert_eq!(
            filter.feature(2).expect("feature 2 should remain").feat_id,
            2
        );
        assert_eq!(
            filter.feature(3).expect("feature 3 should remain").feat_id,
            3
        );
    }

    #[test]
    fn lost_track_removal_updates_dense_lookup() {
        let mut filter = Sparse3DFilter::polar3d(k(), settings());
        let points = [
            (1, Vector3::new(1.0, 0.5, 3.0)),
            (2, Vector3::new(1.5, 0.5, 4.0)),
        ];
        for i in 0..8 {
            update_with_points(&mut filter, i, &points);
        }
        assert_eq!(filter.feature_count(), 2);

        update_with_points(&mut filter, 8, &points[1..]);

        assert_eq!(filter.feature_count(), 1);
        assert!(filter.feature(1).is_none());
        assert!(filter.feature(2).is_some());
    }

    #[test]
    fn polar3d_initializes_and_tracks() {
        let mut filter = Sparse3DFilter::polar3d(k(), settings());
        let point = Vector3::new(1.0, 0.5, 3.0);
        for i in 0..8 {
            update_with_point(&mut filter, i, point);
        }
        let feat = filter.feature(42).expect("feature should initialize");
        assert!(feat.position[2] > 2.5 && feat.position[2] < 3.5);
        let (depth, var) = filter.query(42);
        assert!(depth > 0.0, "depth should be queryable, got {depth}");
        assert!(var.is_finite());
    }

    #[test]
    fn invdepth_additive_initializes_and_tracks() {
        let mut filter = Sparse3DFilter::invdepth_additive3d(k(), settings());
        let point = Vector3::new(1.0, 0.5, 3.0);
        for i in 0..8 {
            update_with_point(&mut filter, i, point);
        }
        let feat = filter.feature(42).expect("feature should initialize");
        assert!(feat.position[2] > 2.5 && feat.position[2] < 3.5);
        assert!(feat.covariance.iter().all(|v| v.is_finite()));
        let (depth, var) = filter.query(42);
        assert!(depth > 0.0, "depth should be queryable, got {depth}");
        assert!(var.is_finite() && var > 0.0);
    }

    #[test]
    fn second_order_analytic_initializes_and_tracks() {
        let mut s = settings();
        s.second_order_mode = SecondOrderMode::Analytic;
        let mut filter = Sparse3DFilter::polar3d(k(), s);
        let point = Vector3::new(1.0, 0.5, 3.0);
        for i in 0..8 {
            update_with_point(&mut filter, i, point);
        }
        let feat = filter.feature(42).expect("feature should initialize");
        assert!(feat.position[2] > 2.5 && feat.position[2] < 3.5);
        assert!(feat.covariance.iter().all(|v| v.is_finite()));
        let (depth, var) = filter.query(42);
        assert!(depth > 0.0, "depth should be queryable, got {depth}");
        assert!(var.is_finite() && var > 0.0);
    }

    #[test]
    fn invdepth3d_initializes_and_tracks() {
        let mut filter = Sparse3DFilter::invdepth3d(k(), settings());
        let point = Vector3::new(1.0, 0.5, 3.0);
        for i in 0..8 {
            update_with_point(&mut filter, i, point);
        }
        let feat = filter.feature(42).expect("feature should initialize");
        assert!(feat.position[2] > 2.5 && feat.position[2] < 3.5);
        let (depth, var) = filter.query(42);
        assert!(depth > 0.0, "depth should be queryable, got {depth}");
        assert!(var.is_finite());
    }

    #[test]
    fn mahalanobis_reset_removes_bad_3d_feature() {
        let mut settings = settings();
        settings.mahalanobis_reset_chi2 = 1.0;
        let mut filter = Sparse3DFilter::invdepth3d(k(), settings);
        let point = Vector3::new(1.0, 0.5, 3.0);
        for i in 0..8 {
            update_with_point(&mut filter, i, point);
        }
        assert!(
            filter
                .feature(42)
                .expect("feature should initialize before reset")
                .inlier_ratio()
                > 0.5,
            "test should cover immediate hard rejection, not only low-inlier reset"
        );

        let t_wc = pose(8.0 * 0.05);
        let mut coords = HashMap::new();
        coords.insert(42, Vector2::new(10_000.0, 10_000.0));
        filter.update(
            &VisionMeasurement::new(8.0 * 0.05, coords),
            &t_wc,
            None,
            None,
        );

        assert!(
            filter.feature(42).is_none(),
            "large Mahalanobis bearing innovation should immediately remove the feature"
        );
    }
}
