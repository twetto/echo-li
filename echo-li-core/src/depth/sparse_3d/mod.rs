pub mod bearing_chart;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use echo_lie::SOT3;
use nalgebra::{
    Matrix2, Matrix2x3, Matrix2x5, Matrix2x6, Matrix3, Matrix3x2, Matrix3x6, Matrix4, Matrix5,
    Matrix6, Vector2, Vector3, Vector4,
};

use crate::coordinate_suite::base_skew;
use crate::coordinate_suite::invdepth::{conv_euc2ind, conv_ind2euc, point_chart_invdepth_inv};
use crate::coordinate_suite::normal::{conv_euc2normal, conv_normal2euc, point_chart_normal_inv};
use crate::depth::sparse_gb::{SecondOrderMode, SparseVogSettings};
use crate::mathematical::camera::CameraModel;
use crate::mathematical::vision_measurement::VisionMeasurement;

/// Minimum sine of the ray-angle between the two views before a
/// `BearingInvDepthAdditive` landmark may be triangulated at birth. This is only
/// a numerical-degeneracy floor; birth-covariance inflation (~1/parallax^2) plus
/// `conv_variance_threshold` do the quality gating.
const BEARING_MIN_PARALLAX_SIN: f64 = 0.02;

fn skew3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse3DChart {
    /// Static-anchor SOT(3) IEKF, range carried multiplicatively (log-depth).
    Polar,
    /// Same SOT(3) IEKF, inverse-depth *reporting* chart (a relabel of `Polar`;
    /// the update is identical; see the consistency docs).
    InvDepth,
    /// Rho-first: a genuinely *additive* inverse-depth EKF. State is
    /// `(alpha, beta, rho)` in the anchor camera frame (no group, no log-depth),
    /// which removes the sequential-linearisation overconfidence of `Polar`.
    /// This is the only chart that currently applies external camera pose
    /// covariance as measurement-side uncertainty.
    InvDepthAdditive,
    /// Camera-model-agnostic sibling of `InvDepthAdditive`: the same additive
    /// inverse-range EKF, but the anchor point is parameterized on a local
    /// tangent-bearing chart `P = b(eta)/rho` (see `bearing_chart`) instead of
    /// the pinhole plane, and projection goes through the shared `CameraModel`
    /// (`project`/`projection_jacobian`). Handles fisheye / wide FoV; requires a
    /// camera to be attached to the filter.
    BearingInvDepthAdditive,
}

#[derive(Debug, Clone)]
struct PendingFeature {
    ref_t_wc: Matrix4<f64>,
    ref_uv: Vector2<f64>,
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
    // Rho-first additive inverse-depth state [InvDepthAdditive only]: s = (alpha,
    // beta, rho) = (X/Z, Y/Z, 1/Z) of the landmark in the ANCHOR camera frame,
    // with covariance inv_p in those coords. Plain additive EKF, no group / no
    // re-charting. The SOT(3) fields above are unused in this chart (and vice
    // versa). Cached `covariance` for this chart is the current-frame Euclidean
    // covariance directly (chart_to_euc_jac == I for InvDepthAdditive).
    pub inv_s: Vector3<f64>,
    pub inv_p: Matrix3<f64>,
    // BearingInvDepthAdditive only: fixed anchor unit bearing `b0` and its
    // orthonormal tangent basis `U`. Here `inv_s = (eta1, eta2, rho)` and the
    // anchor point is `P = b(eta)/rho`, `b(eta) = normalize(b0 + U eta)`.
    // Zero/unused for the other charts.
    pub anchor_ray: Vector3<f64>,
    pub tangent_basis: Matrix3x2<f64>,
    pub anchor_t_wc: Matrix4<f64>, // camera->world pose at the anchor (creation) frame
    // --- Gaussian-Beta inlier model + bookkeeping ---
    pub a: f64,
    pub b: f64,
    pub track_length: usize,
    pub ref_uv: Vector2<f64>,
    pub ref_stamp: f64,
    /// Last update normalized innovation squared (NIS = gating Mahalanobis
    /// squared). Should be close to chi2(2) when the filter is consistent.
    /// NaN before the first update.
    pub last_nis: f64,
    /// Last evaluated pose-driven range variance `f = G*(r^2 sig_phi^2 + sig_b^2)`
    /// under `pose_range_direct`. Injecting `f_k - f_{k-1}` makes the ACCUMULATED range
    /// variance equal `f_k`, so it follows the true relative-pose growth curve rather
    /// than a sum of increments. Reset to 0 on re-anchor (a fresh anchor has zero
    /// relative covariance to itself).
    pose_q_prev: f64,
    /// Consider cross-covariance `Cov(delta s, delta theta^{a->k})` between the chart
    /// state and the anchor->current relative-pose error, `[phi; t]` order, in the same
    /// convention as the measurement-side pose Jacobian (`delta q_c = [q_c]_x phi + t`).
    /// Zero at birth (the anchor IS the birth frame, so the relative pose is identity
    /// with zero covariance) and after a re-anchor. Only maintained under
    /// `settings.consider_relpose`.
    pub p_stheta: Matrix3x6<f64>,

    // --- Per-track correspondence-bias state (BearingInvDepthAdditive only,
    // enabled by settings.bias_walk_var > 0). Augments the 3-DOF landmark to a
    // 5-DOF state s5 = (inv_s, bias). All zero / unused when disabled. ---
    pub bias: Vector2<f64>,   // estimated 2D pixel bias b
    pub p_sb: Matrix3x2<f64>, // cross-covariance cov(inv_s, bias)
    pub p_bb: Matrix2<f64>,   // bias covariance cov(bias, bias)
    /// Id of the EqF pose clone this landmark is anchored at (its birth frame),
    /// when the caller drives the honest clone-relative pose-covariance feed.
    /// `None` for the legacy absolute-covariance path. Used only to route the
    /// §V-D pose-range term to this feature's own anchor→current relative cov.
    pub anchor_clone_id: Option<u64>,
}

impl FeatureState3D {
    pub fn depth_for_chart(&self, chart: Sparse3DChart) -> f64 {
        match chart {
            Sparse3DChart::Polar => self.position.norm(),
            Sparse3DChart::InvDepth
            | Sparse3DChart::InvDepthAdditive
            | Sparse3DChart::BearingInvDepthAdditive => self.position[2],
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
            Sparse3DChart::InvDepthAdditive | Sparse3DChart::BearingInvDepthAdditive => {
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
        if ab <= 0.0 {
            0.0
        } else {
            self.a / ab
        }
    }
}

pub struct Sparse3DFilter {
    k: Matrix3<f64>,
    /// Shared calibrated camera, required by `BearingInvDepthAdditive` for
    /// generic projection; unused by the pinhole/K charts.
    camera: Option<Arc<dyn CameraModel>>,
    chart: Sparse3DChart,
    settings: SparseVogSettings,
    sigma_norm_sq: f64,
    features: Vec<FeatureState3D>,
    feature_slots: HashMap<u64, usize>,
    pending: HashMap<u64, PendingFeature>,
    prev_uvs: HashMap<u64, Vector2<f64>>,
    prev_t_wc: Option<Matrix4<f64>>,
    /// Per-anchor consider data for THIS frame, set by the feed before `update_with_clones`.
    consider_rel: ConsiderRelMap,
    prev_stamp: f64,
}

impl Sparse3DFilter {
    pub fn new(k: Matrix3<f64>, chart: Sparse3DChart, settings: SparseVogSettings) -> Self {
        let sigma_norm_sq = (settings.sigma_pixel / k[(0, 0)]).powi(2);
        Self {
            k,
            camera: None,
            chart,
            settings,
            sigma_norm_sq,
            features: Vec::new(),
            feature_slots: HashMap::new(),
            pending: HashMap::new(),
            prev_uvs: HashMap::new(),
            prev_t_wc: None,
            consider_rel: HashMap::new(),
            prev_stamp: -1.0,
        }
    }

    /// Attach the shared calibrated camera (required for
    /// `BearingInvDepthAdditive`).
    pub fn with_camera(mut self, camera: Arc<dyn CameraModel>) -> Self {
        self.camera = Some(camera);
        self
    }

    /// Camera-agnostic tangent-bearing inverse-range EKF
    /// (`BearingInvDepthAdditive`); needs a camera via [`Self::with_camera`].
    pub fn bearing_invdepth3d(
        k: Matrix3<f64>,
        camera: Arc<dyn CameraModel>,
        settings: SparseVogSettings,
    ) -> Self {
        Self::new(k, Sparse3DChart::BearingInvDepthAdditive, settings).with_camera(camera)
    }

    pub fn polar3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::Polar, settings)
    }

    pub fn invdepth3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::InvDepth, settings)
    }

    /// Rho-first: additive inverse-depth EKF (see `Sparse3DChart::InvDepthAdditive`).
    pub fn invdepth_additive3d(k: Matrix3<f64>, settings: SparseVogSettings) -> Self {
        Self::new(k, Sparse3DChart::InvDepthAdditive, settings)
    }

    /// Update from one camera measurement set.
    ///
    /// `p_vv` and `p_ww` are per-frame camera translation and rotation
    /// covariances. They are consumed by `InvDepthAdditive`; the legacy IEKF
    /// charts currently ignore them except for shared birth-time terms.
    pub fn update(
        &mut self,
        measurement: &VisionMeasurement,
        t_wc: &Matrix4<f64>,
        p_vv: Option<&Matrix3<f64>>,
        p_ww: Option<&Matrix3<f64>>,
    ) {
        self.update_with_clones(measurement, t_wc, p_vv, p_ww, None, None);
    }

    /// Honest clone-relative variant of [`update`]. `current_clone_id` tags any
    /// landmark born this frame with its anchor clone. `rel_cov_by_clone` maps a
    /// clone id to that anchor→current relative-pose covariance as FOUR 3×3 blocks
    /// `(full_v, full_w, inc_v, inc_w)`: the accumulated `Cov(T_clone⁻¹ T_curr)`
    /// (fed to §V-D's per-frame MEASUREMENT term) and its per-frame INCREMENT (fed
    /// to the accumulating range injection) — see [`resolve_pose_cov`]. Each
    /// landmark is then driven by its own honest gauge-cancelled relative cov
    /// instead of the shared absolute `p_vv`/`p_ww`. Passing `None` for the map is
    /// exactly [`update`] (legacy absolute path — both consumers use `p_vv`/`p_ww`).
    pub fn update_with_clones(
        &mut self,
        measurement: &VisionMeasurement,
        t_wc: &Matrix4<f64>,
        p_vv: Option<&Matrix3<f64>>,
        p_ww: Option<&Matrix3<f64>>,
        current_clone_id: Option<u64>,
        rel_cov_by_clone: Option<&CloneRelCov>,
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
            let cam = self.camera.as_deref();
            let consider_rel = &self.consider_rel;
            self.features
                .par_iter_mut()
                .filter_map(|feat| {
                    let fid = feat.feat_id;
                    if !self.prev_uvs.contains_key(&fid) {
                        return None;
                    }
                    let uv_curr = curr_uvs.get(&fid)?;
                    let (pvv_m, pww_m, pvv_i, pww_i) =
                        resolve_pose_cov(rel_cov_by_clone, feat.anchor_clone_id, p_vv, p_ww);
                    let cr = feat.anchor_clone_id.and_then(|cid| consider_rel.get(&cid));
                    (!update_feature_3d(
                        chart, &k, cam, &settings, feat, uv_curr, &t_cw_curr, pvv_m, pww_m, pvv_i,
                        pww_i, dt, cr,
                    ))
                    .then_some(fid)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let reset_features = {
            let mut reset_features = Vec::new();
            let cam = self.camera.as_deref();
            for feat in &mut self.features {
                let fid = feat.feat_id;
                if !self.prev_uvs.contains_key(&fid) {
                    continue;
                }

                let Some(uv_curr) = curr_uvs.get(&fid) else {
                    continue;
                };

                let (pvv_m, pww_m, pvv_i, pww_i) =
                    resolve_pose_cov(rel_cov_by_clone, feat.anchor_clone_id, p_vv, p_ww);
                let cr = feat
                    .anchor_clone_id
                    .and_then(|cid| self.consider_rel.get(&cid));
                if !update_feature_3d(
                    self.chart,
                    &self.k,
                    cam,
                    &self.settings,
                    feat,
                    uv_curr,
                    &t_cw_curr,
                    pvv_m,
                    pww_m,
                    pvv_i,
                    pww_i,
                    dt,
                    cr,
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
                })
                .clone();

            let t_curr_ref = t_cw_curr * pending.ref_t_wc;
            let r_ref = t_curr_ref.fixed_view::<3, 3>(0, 0).into_owned();
            let t_ref = t_curr_ref.fixed_view::<3, 1>(0, 3).into_owned();

            // Camera-agnostic bearing-chart birth: two-view ray triangulation on
            // unit bearings, anchored in the current frame at eta = 0. (The
            // rigorous birth covariance is refined in a follow-up; here it is a
            // pragmatic angular/range diagonal.)
            if self.chart == Sparse3DChart::BearingInvDepthAdditive {
                let Some(cam) = self.camera.clone() else {
                    continue;
                };
                let b0 = cam.undistort(&uv_curr);
                let focal = 0.5 * (self.k[(0, 0)] + self.k[(1, 1)]);
                let sigma_eta = self.settings.sigma_pixel / focal;
                // Two birth modes. Civera (undelayed) births on first sight from a
                // depth PRIOR; the classical path waits for parallax and triangulates.
                let (range_anchor, rho_var) = if self.settings.civera_birth_depth > 0.0 {
                    // Undelayed: no flow gate, no ray-angle gate, no triangulation.
                    // rho0 = 1/d0 with sigma_rho = frac*rho0; at frac = 0.5 the 2-sigma
                    // band spans rho in [0, 2*rho0], i.e. depth in [d0/2, infinity).
                    let d0 = self.settings.civera_birth_depth;
                    let rho0 = 1.0 / d0;
                    let s_rho = self.settings.civera_rho_sigma_frac * rho0;
                    (d0, (s_rho * s_rho).max(1e-12))
                } else {
                    if (uv_curr - pending.ref_uv).norm() < self.settings.birth_min_flow_px {
                        continue;
                    }
                    let b_prev = cam.undistort(&pending.ref_uv);
                    // Parallax gate: two-view triangulation is severely biased toward
                    // the camera at low parallax. Require enough ray-angle between the
                    // two bearings (in the current frame) before birthing.
                    let b_prev_c = (r_ref * b_prev).normalize();
                    if b0.cross(&b_prev_c).norm() < BEARING_MIN_PARALLAX_SIN {
                        continue;
                    }
                    let Some((range_anchor, _)) =
                        bearing_chart::two_ray_ranges(&b0, &b_prev, &r_ref, &t_ref)
                    else {
                        continue;
                    };
                    if !range_anchor.is_finite()
                        || range_anchor < self.settings.min_depth
                        || range_anchor > self.settings.max_depth
                    {
                        continue;
                    }
                    let rv = (self.settings.init_depth_var / range_anchor.powi(4)).max(1e-12);
                    (range_anchor, rv)
                };
                let u = bearing_chart::tangent_basis(&b0);
                let rho = 1.0 / range_anchor;
                let inv_s = Vector3::new(0.0, 0.0, rho);
                let inv_p = Matrix3::from_diagonal(&Vector3::new(
                    sigma_eta * sigma_eta,
                    sigma_eta * sigma_eta,
                    rho_var,
                ));
                let (position, j0) =
                    bearing_chart::point_and_jacobian(&b0, &u, &Vector2::zeros(), rho);
                let cov_euc = j0 * inv_p * j0.transpose();
                let cov_euc = 0.5 * (cov_euc + cov_euc.transpose());
                let feat = FeatureState3D {
                    feat_id: fid,
                    position,
                    covariance: cov_euc,
                    q0: position,
                    x: SOT3::identity(),
                    sigma: cov_euc,
                    inv_s,
                    inv_p,
                    anchor_ray: b0,
                    tangent_basis: u,
                    anchor_t_wc: *t_wc,
                    a: self.settings.a_init,
                    b: self.settings.b_init,
                    track_length: 1,
                    ref_uv: uv_curr,
                    ref_stamp: stamp,
                    last_nis: f64::NAN,
                    bias: Vector2::zeros(),
                    p_sb: Matrix3x2::zeros(),
                    p_bb: Matrix2::zeros(),
                    pose_q_prev: 0.0,
                    p_stheta: Matrix3x6::zeros(),
                    // Bearing chart anchors at the CURRENT frame (anchor_t_wc =
                    // *t_wc), so the anchor clone is this frame's clone.
                    anchor_clone_id: current_clone_id,
                };
                self.insert_feature(feat);
                self.pending.remove(&fid);
                continue;
            }

            let (z_obs, drive) = triangulate_pixel(
                &self.k,
                &self.settings,
                &pending.ref_uv,
                &uv_curr,
                &r_ref,
                &t_ref,
            );
            if z_obs <= 0.0 || drive * self.k[(0, 0)] < self.settings.birth_min_flow_px {
                continue;
            }

            let baseline_tau_sq = baseline_tau(p_vv, &t_ref);
            // Anchor the landmark in the CURRENT camera frame (= anchor frame);
            // the IEKF group element x (init identity) carries all later
            // refinement, so we never re-anchor. q0 is the fixed origin.
            let position = position_from_depth(&self.k, &uv_curr, z_obs);
            let pose_birth_cov = current_anchor_rotation_init_cov(
                &self.k,
                &pending.ref_uv,
                &uv_curr,
                &r_ref,
                &t_ref,
                &position,
                p_ww,
            );
            let covariance = init_cov_3d(
                self.chart,
                &self.settings,
                self.sigma_norm_sq,
                &position,
                z_obs,
                drive,
                baseline_tau_sq,
                &pose_birth_cov,
            );
            // IEKF covariance lives in Euclidean error coords about q0. At
            // creation x = id and the anchor frame is the current frame, so the
            // Euclidean init cov is the chart cov mapped back through the chart
            // Jacobian (chart_to_euc . cov . chart_to_euc^T).
            let j_c2e = chart_to_euc_jac(self.chart, &position);
            let sigma = j_c2e * covariance * j_c2e.transpose();
            // Rho-first additive state: anchor-frame (alpha,beta,rho) and its cov.
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
                anchor_ray: Vector3::zeros(),
                tangent_basis: Matrix3x2::zeros(),
                anchor_t_wc: *t_wc,
                a: self.settings.a_init,
                b: self.settings.b_init,
                track_length: 1,
                ref_uv: uv_curr,
                ref_stamp: stamp,
                last_nis: f64::NAN,
                bias: Vector2::zeros(),
                p_sb: Matrix3x2::zeros(),
                p_bb: Matrix2::zeros(),
                pose_q_prev: 0.0,
                p_stheta: Matrix3x6::zeros(),
                // Non-bearing charts also anchor at the current frame here.
                anchor_clone_id: current_clone_id,
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

    /// Move a landmark's anchor to `t_wc_new`, leaving the world point it represents
    /// unchanged. Returns false if the landmark is absent, the chart is not
    /// `BearingInvDepthAdditive`, or the re-expressed range leaves `[min_depth, max_depth]`.
    ///
    /// WHY THIS EXISTS: Sparse3D anchors a landmark at birth and never moves it, but the
    /// anchor->current relative-pose covariance only exists while the anchor CLONE is
    /// alive. A landmark outliving its anchor clone silently falls back to the absolute
    /// covariance in `resolve_pose_cov`. The historical workaround was to size the clone
    /// window to outlive the longest track (~120 clones), which costs O(n^3) in the filter.
    /// Re-anchoring decouples the two: the window then only has to serve the MSCKF update.
    ///
    /// MATH. With `p_old` the point in the old anchor frame and `R,t` the old->new
    /// transform, `p_new = R p_old + t`. The new state is `(0, 0, 1/|p_new|)` because the
    /// new anchor ray IS the point's own bearing, so eta is zero BY CONSTRUCTION -- its
    /// uncertainty does not vanish, it moves into the covariance. Differentiating
    /// `eta = u_new^T b` and `rho = 1/|p|` at eta = 0, where `u_new ⊥ b`:
    ///     d eta / d p = rho * u_new^T          d rho / d p = -rho^2 * b^T
    /// The landmark covariance maps through `J = (d state/d p) R (d p_old/d state_old)`,
    /// and the old->new relative-pose uncertainty enters as
    /// `[R p_old]_x P_ww [R p_old]_x^T + P_vv` pushed through `d state/d p`. Passing
    /// `rel_pose_cov = None` asserts the two anchors are perfectly known relative to each
    /// other, which they are not -- pass it whenever it is available.
    pub fn reanchor_feature(
        &mut self,
        fid: u64,
        t_wc_new: &Matrix4<f64>,
        new_clone_id: Option<u64>,
        rel_pose_cov: Option<(&Matrix3<f64>, &Matrix3<f64>)>,
    ) -> bool {
        self.reanchor_feature_full(fid, t_wc_new, new_clone_id, rel_pose_cov, None)
    }

    /// [`Self::reanchor_feature`] plus the consider-mode transfer (§V-F eq. 17): with
    /// `consider_relpose` on and `rel_full` given, the old->new pose covariance enters
    /// TOGETHER with the landmark's tracked cross-covariance,
    /// `J_s P_s J_s' + J_th S J_th' + J_s P_sth J_th' + (.)'`, instead of under the
    /// independence assumption `P_sth = 0`; the cross term is then reset for the new
    /// epoch. The radial projection used under `pose_range_direct` does not apply (no
    /// range injection runs in consider mode).
    pub fn reanchor_feature_full(
        &mut self,
        fid: u64,
        t_wc_new: &Matrix4<f64>,
        new_clone_id: Option<u64>,
        rel_pose_cov: Option<(&Matrix3<f64>, &Matrix3<f64>)>,
        rel_full: Option<&Matrix6<f64>>,
    ) -> bool {
        if self.chart != Sparse3DChart::BearingInvDepthAdditive {
            return false;
        }
        let Some(&slot) = self.feature_slots.get(&fid) else {
            return false;
        };
        let feat = &mut self.features[slot];
        let rho = feat.inv_s[2];
        if !rho.is_finite() || rho <= 0.0 {
            return false;
        }
        let eta = Vector2::new(feat.inv_s[0], feat.inv_s[1]);
        let (p_old, j_old) =
            bearing_chart::point_and_jacobian(&feat.anchor_ray, &feat.tangent_basis, &eta, rho);

        let r_old = feat.anchor_t_wc.fixed_view::<3, 3>(0, 0).into_owned();
        let t_old = feat.anchor_t_wc.fixed_view::<3, 1>(0, 3).into_owned();
        let r_new = t_wc_new.fixed_view::<3, 3>(0, 0).into_owned();
        let t_new = t_wc_new.fixed_view::<3, 1>(0, 3).into_owned();
        let r_rel = r_new.transpose() * r_old;
        let t_rel = r_new.transpose() * (t_old - t_new);
        let p_new = r_rel * p_old + t_rel;

        let range = p_new.norm();
        if !range.is_finite() || range < self.settings.min_depth || range > self.settings.max_depth
        {
            return false;
        }
        let b_new = p_new / range;
        let u_new = bearing_chart::tangent_basis(&b_new);
        let rho_new = 1.0 / range;

        let mut j_from_p = Matrix3::zeros();
        j_from_p
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(u_new.transpose() * rho_new));
        j_from_p
            .fixed_view_mut::<1, 3>(2, 0)
            .copy_from(&(-(rho_new * rho_new) * b_new.transpose()));

        let j = j_from_p * r_rel * j_old;
        let mut inv_p_new = j * feat.inv_p * j.transpose();
        if let (true, Some(sig)) = (self.settings.consider_relpose, rel_full) {
            // (17): point-in-new-frame perturbation d p_new = J_s ds + J_th dth with
            // J_th = [[p_new]_x  I] -- the SAME convention as the update's pose Jacobian
            // (`dq_c = [q_c]_x phi + t`, lever arm the point in the current frame), which
            // the sign- and lever-sensitive cross term requires; the legacy branch below
            // uses `[R p_old]_x`, which only its quadratic form is indifferent to. The
            // sum can be numerically indefinite; clip to PSD.
            let mut j_th = Matrix3x6::<f64>::zeros();
            j_th.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&base_skew(&p_new));
            j_th.fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&Matrix3::identity());
            let js = r_rel * j_old; // d p_new / d s_old
            let cross = js * feat.p_stheta * j_th.transpose();
            let cov_p = j_th * sig * j_th.transpose() + cross + cross.transpose();
            let add = j_from_p * cov_p * j_from_p.transpose();
            inv_p_new += add;
            inv_p_new = 0.5 * (inv_p_new + inv_p_new.transpose());
            let eig = inv_p_new.symmetric_eigen();
            let clipped = Matrix3::from_diagonal(&eig.eigenvalues.map(|v| v.max(1e-18)));
            inv_p_new = eig.eigenvectors * clipped * eig.eigenvectors.transpose();
            feat.p_sb = j * feat.p_sb;
            feat.inv_s = Vector3::new(0.0, 0.0, rho_new);
            feat.inv_p = 0.5 * (inv_p_new + inv_p_new.transpose());
            feat.anchor_ray = b_new;
            feat.tangent_basis = u_new;
            feat.anchor_t_wc = *t_wc_new;
            feat.anchor_clone_id = new_clone_id;
            feat.pose_q_prev = 0.0;
            feat.p_stheta = Matrix3x6::zeros();
            return true;
        }
        if let Some((pvv, pww)) = rel_pose_cov {
            let sk = skew3(&(r_rel * p_old));
            let mut cov_p = sk * pww * sk.transpose() + pvv;
            // Under `pose_range_direct` the RADIAL share of this same anchor->current
            // transition has already been injected into the range channel over the
            // epoch; adding the full 3D block here would count it twice. Project the
            // radial direction out and keep only the transverse part, which the range
            // process term never carries (it is a pure r_hat r_hat^T injection).
            if self.settings.pose_range_direct {
                let perp = Matrix3::identity() - b_new * b_new.transpose();
                cov_p = perp * cov_p * perp.transpose();
            }
            inv_p_new += j_from_p * cov_p * j_from_p.transpose();
        }
        inv_p_new = 0.5 * (inv_p_new + inv_p_new.transpose());

        // Correspondence-bias cross-covariance is in the same 3-dof landmark block.
        feat.p_sb = j * feat.p_sb;
        feat.inv_s = Vector3::new(0.0, 0.0, rho_new);
        feat.inv_p = inv_p_new;
        feat.anchor_ray = b_new;
        feat.tangent_basis = u_new;
        feat.anchor_t_wc = *t_wc_new;
        feat.anchor_clone_id = new_clone_id;
        // The landmark is now anchored HERE, so its anchor->current relative pose
        // covariance restarts at zero. Carrying the old running value would keep
        // injecting variance that the new (short) baseline does not have.
        feat.pose_q_prev = 0.0;
        true
    }

    /// Re-anchor every landmark currently anchored to `old_clone_id`. Call this
    /// immediately BEFORE marginalizing that clone, while its relative pose to the new
    /// anchor is still computable. Returns the number of landmarks moved.
    pub fn reanchor_clone(
        &mut self,
        old_clone_id: u64,
        t_wc_new: &Matrix4<f64>,
        new_clone_id: Option<u64>,
        rel_pose_cov: Option<(&Matrix3<f64>, &Matrix3<f64>)>,
    ) -> usize {
        self.reanchor_clone_full(old_clone_id, t_wc_new, new_clone_id, rel_pose_cov, None)
    }

    /// [`Self::reanchor_clone`] with, under `consider_relpose`, the FULL old->new
    /// relative covariance so the transfer can use the landmark's tracked cross term
    /// (§V-F eq. 17). `rel_full` is ignored when consider mode is off.
    pub fn reanchor_clone_full(
        &mut self,
        old_clone_id: u64,
        t_wc_new: &Matrix4<f64>,
        new_clone_id: Option<u64>,
        rel_pose_cov: Option<(&Matrix3<f64>, &Matrix3<f64>)>,
        rel_full: Option<&Matrix6<f64>>,
    ) -> usize {
        let ids: Vec<u64> = self
            .features
            .iter()
            .filter(|f| f.anchor_clone_id == Some(old_clone_id))
            .map(|f| f.feat_id)
            .collect();
        let mut n = 0;
        for fid in ids {
            if self.reanchor_feature_full(fid, t_wc_new, new_clone_id, rel_pose_cov, rel_full) {
                n += 1;
            }
        }
        n
    }

    /// Install this frame's per-anchor consider data (see [`ConsiderRel`]). Call before
    /// `update_with_clones`; the map is consumed by that call's updates and replaced
    /// wholesale next frame.
    pub fn set_consider_relpose(&mut self, map: ConsiderRelMap) {
        self.consider_rel = map;
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
            || unconverged(&self.settings, depth, depth_var)
        {
            return (-1.0, f64::INFINITY);
        }
        // Exogenous multiplicative scale uncertainty (see `scale_sigma`). Applied to what
        // the filter REPORTS, after the convergence gates, so it never changes the gates'
        // behaviour -- only the honesty of the number handed to a consumer.
        let depth_var = depth_var + (self.settings.scale_sigma * depth).powi(2);
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
            || unconverged(&self.settings, range, range_var)
        {
            return (-1.0, f64::INFINITY);
        }
        // Enforce the DECLARED operating range on the way out, not only at birth.
        // Birth rejects a triangulated range outside [min_depth, max_depth], but a
        // landmark born just inside the cap can drift far beyond it, and the far field
        // is exactly where this filter is least trustworthy: once r >> baseline the
        // triangulation is ill-conditioned, yet Eq (11)'s r^2/b^2 gain SHRINKS the
        // injected variance as the baseline grows, so a long track out there reports a
        // confident estimate (sigma/est ~0.17 measured) against a ~57% true error.
        // Reporting a range the filter would refuse to birth is an inconsistency, not a
        // capability; a consumer asking "how far is this" should get "unusable" instead.
        if range < self.settings.min_depth || range > self.settings.max_depth {
            return (-1.0, f64::INFINITY);
        }
        let range_var = range_var + (self.settings.scale_sigma * range).powi(2);
        (range, range_var)
    }

    /// The configured exogenous scale uncertainty (1-sigma fraction of range).
    pub fn scale_sigma(&self) -> f64 {
        self.settings.scale_sigma
    }

    /// Range variance the filter REPORTS for this landmark: its own estimate variance
    /// plus the exogenous `(scale_sigma * r)^2`. This is the quantity a consumer should
    /// score, not the bare chart covariance.
    pub fn reported_range_variance(&self, feat: &FeatureState3D) -> f64 {
        let r = feat.position.norm();
        feat.range_variance_for_chart(self.chart) + (self.settings.scale_sigma * r).powi(2)
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

    /// Median range (m) and median range-variance over live pool features, so
    /// `conv_variance_threshold` can be judged against what the filter actually
    /// achieves rather than against a guess.
    /// `[r_p10, r_p50, r_p90, var_p10, var_p50, var_p90, frac_short]` where
    /// `frac_short` is the fraction of the pool sitting below 0.5 m — births far
    /// below scene depth carry a huge `var_rho = init_depth_var / r^4`, so their
    /// reported range variance explodes as the range estimate later corrects.
    pub fn range_var_percentiles(&self) -> [f64; 7] {
        let mut ranges: Vec<f64> = Vec::with_capacity(self.features.len());
        let mut vars: Vec<f64> = Vec::with_capacity(self.features.len());
        for feat in &self.features {
            let r = feat.position.norm();
            if r > 0.0 && r.is_finite() {
                ranges.push(r);
                vars.push(feat.range_variance_for_chart(self.chart));
            }
        }
        if ranges.is_empty() {
            return [-1.0; 7];
        }
        ranges.sort_by(|a, b| a.total_cmp(b));
        vars.sort_by(|a, b| a.total_cmp(b));
        let pick = |v: &Vec<f64>, q: f64| v[(((v.len() - 1) as f64) * q).round() as usize];
        let short = ranges.iter().filter(|r| **r < 0.5).count() as f64 / ranges.len() as f64;
        [
            pick(&ranges, 0.1),
            pick(&ranges, 0.5),
            pick(&ranges, 0.9),
            pick(&vars, 0.1),
            pick(&vars, 0.5),
            pick(&vars, 0.9),
            short,
        ]
    }

    /// Why `query_range` yields what it does, as live counts:
    /// `[pending, in_pool, short_track, low_inlier, high_var, usable]`.
    /// The last four partition `in_pool` in the same order the gates are applied,
    /// so whichever is largest is the one actually capping seed supply.
    pub fn gate_census(&self) -> [usize; 6] {
        let mut c = [0usize; 6];
        c[0] = self.pending.len();
        c[1] = self.features.len();
        for feat in &self.features {
            if feat.position.norm() <= 0.0 || feat.track_length < self.settings.min_track_length {
                c[2] += 1;
            } else if feat.inlier_ratio() < self.settings.conv_inlier_ratio {
                c[3] += 1;
            } else if feat.range_variance_for_chart(self.chart)
                > self.settings.conv_variance_threshold
            {
                c[4] += 1;
            } else {
                c[5] += 1;
            }
        }
        c
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

/// Range-direction process variance contributed by anchor->current pose uncertainty.
///
/// Two modes, selected by `settings.pose_range_direct`:
///
/// * LEGACY (default): `pose_range_scale * G * dSigma`, the per-frame clipped increment
///   scaled by the inverse-parallax gain `G = r^2/b^2`. A cumulative sum of clipped
///   increments grows ~linearly in window length by construction, whereas the true
///   relative-pose error is near-ballistic, so the free scalar slides the curve without
///   changing its slope.
///
/// * DIRECT: evaluate `f = G * (r^2 sig_phi^2 + sig_b^2)` from the FULL relative
///   covariance at the CURRENT separation and inject `max(0, f - f_prev)`. The
///   accumulated range variance then equals `f` itself, following the true growth curve
///   including its exponent, with no free scalar. `f` is monotone-clamped because an EKF
///   process step can only add variance; a decrease (e.g. the pose filter tightening)
///   is held rather than subtracted.
///
/// `b2` is the squared anchor->current baseline, floored to bound the `1/b^2` blow-up at
/// zero parallax. Returns the variance to add to the radial channel.
/// Is this landmark's range estimate too uncertain to hand to a consumer?
///
/// Relative form (`conv_range_sigma_frac > 0`) when configured: `sigma / value > frac`.
/// Otherwise the legacy absolute form `var > conv_variance_threshold`, kept so existing
/// configs behave exactly as before. See `conv_range_sigma_frac` for why the absolute
/// form cannot survive a change to the covariance model.
fn unconverged(settings: &SparseVogSettings, value: f64, var: f64) -> bool {
    if settings.conv_range_sigma_frac > 0.0 {
        if value <= 0.0 {
            return true;
        }
        var.sqrt() / value > settings.conv_range_sigma_frac
    } else {
        var > settings.conv_variance_threshold
    }
}

/// Excitation -> multiplicative scale uncertainty, the `sigma_s(E)` envelope.
///
/// `E = std(omega_z) * std(a_y)` computed over the WHOLE trajectory on raw IMU (the same
/// excitation metric the scale-memory work uses). It is a trajectory-level property, not a
/// per-frame one, so the caller supplies it; this function only holds the fitted map.
///
/// `sigma_s(E) = max(0.02, 0.0072 * E^-0.710)`
///
/// Fitted to the ROBUST CENTRE (`median |relerr|`), not the RMS: the RMS was non-monotone
/// in `E` only because the heavy aperture-correspondence tail leaked into it, while the
/// median is monotone across the measured anchors (3.1 -> 4.2 -> 10.7 -> 44.0 % as `E`
/// falls) and fits within +-20%. The 0.02 floor keeps a well-excited trajectory from
/// claiming better-than-2% scale knowledge, which no monocular system has.
///
/// Returns a 1-sigma FRACTION of range, to be used as `sigma_range += (sigma_s * r)`.
pub fn sigma_s_from_excitation(e: f64) -> f64 {
    if !e.is_finite() || e <= 0.0 {
        return 0.02;
    }
    (0.0072 * e.powf(-0.710)).max(0.02)
}

fn pose_range_q(
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    r2: f64,
    b2: f64,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    p_vv_inc: Option<&Matrix3<f64>>,
    p_ww_inc: Option<&Matrix3<f64>>,
) -> f64 {
    let gain = r2 / b2;
    if settings.pose_range_direct {
        let (Some(pv), Some(pw)) = (p_vv, p_ww) else {
            return 0.0;
        };
        let sig_t2 = pv.trace() / 3.0;
        let sig_phi2 = pw.trace() / 3.0;
        let f = gain * (r2 * sig_phi2 + sig_t2);
        if !f.is_finite() {
            return 0.0;
        }
        let dq = (f - feat.pose_q_prev).max(0.0);
        feat.pose_q_prev = f.max(feat.pose_q_prev);
        return dq.min(4.0 * r2);
    }
    let (Some(pvi), Some(pwi)) = (p_vv_inc, p_ww_inc) else {
        return 0.0;
    };
    let sig_t2 = pvi.trace() / 3.0;
    let sig_phi2 = pwi.trace() / 3.0;
    let x = gain * (r2 * sig_phi2 + sig_t2);
    let mut pose_q = settings.pose_range_scale * x;
    if settings.pose_range_coherent > 0.0 {
        let age = feat.track_length as f64;
        pose_q += 2.0 * age * settings.pose_range_coherent * x;
    }
    pose_q.min(4.0 * r2)
}

fn chart_to_euc_jac(chart: Sparse3DChart, q: &Vector3<f64>) -> Matrix3<f64> {
    match chart {
        Sparse3DChart::Polar => conv_normal2euc(q),
        Sparse3DChart::InvDepth => conv_ind2euc(q),
        // The additive charts cache `covariance` already in Euclidean coords.
        Sparse3DChart::InvDepthAdditive | Sparse3DChart::BearingInvDepthAdditive => {
            Matrix3::identity()
        }
    }
}

fn euc_to_chart_jac(chart: Sparse3DChart, q: &Vector3<f64>) -> Matrix3<f64> {
    match chart {
        Sparse3DChart::Polar => conv_euc2normal(q),
        Sparse3DChart::InvDepth => conv_euc2ind(q),
        Sparse3DChart::InvDepthAdditive | Sparse3DChart::BearingInvDepthAdditive => {
            Matrix3::identity()
        }
    }
}

fn apply_chart_delta(chart: Sparse3DChart, q: &Vector3<f64>, delta: &Vector3<f64>) -> Vector3<f64> {
    match chart {
        Sparse3DChart::Polar => point_chart_normal_inv(delta, q),
        Sparse3DChart::InvDepth => point_chart_invdepth_inv(delta, q),
        // additive charts apply the delta in chart coords; not used via this
        // helper (the additive update is self-contained), but keep the match total.
        Sparse3DChart::InvDepthAdditive | Sparse3DChart::BearingInvDepthAdditive => q + delta,
    }
}

/// Per-clone relative-pose covariance feed for §V-D. For each live clone the map
/// carries FOUR 3×3 blocks: the accumulated `Cov(T_clone⁻¹ T_curr)` translation /
/// rotation (`full_v`, `full_w`) and the per-frame INCREMENT of that same quantity
/// (`inc_v`, `inc_w`). The two are consumed differently (see [`resolve_pose_cov`]).
pub type CloneRelCov = HashMap<u64, (Matrix3<f64>, Matrix3<f64>, Matrix3<f64>, Matrix3<f64>)>;

/// Consider-mode input for one anchor (filter_formulation §V-F): the FULL 6x6
/// anchor->current relative-pose covariance `[phi; t]` and the one-step regression
/// operator `M_k = Cov(theta_k, theta_{k-1}) Cov(theta_{k-1})^-1` that transports the
/// landmark–pose cross-covariance between frames.
#[derive(Debug, Clone, Copy)]
pub struct ConsiderRel {
    pub sigma: Matrix6<f64>,
    pub m: Matrix6<f64>,
}
pub type ConsiderRelMap = HashMap<u64, ConsiderRel>;

/// Route a landmark to the pose covariances its §V-D update should use, splitting
/// the two distinct consumers of pose uncertainty:
///   - **measurement term** `r_meas += proj·P·projᵀ` — rebuilt fresh every frame, so
///     it wants the FULL anchor→current relative uncertainty `Cov(T_clone⁻¹ T_curr)`;
///   - **range injection** `q_range ∝ Pᵥᵥ.trace()` — ACCUMULATES into `inv_p` over the
///     track, so it wants the per-frame INCREMENT (feeding the full each frame would
///     integrate to ~N× the intended total).
/// Returns `(pvv_meas, pww_meas, pvv_inc, pww_inc)`. When the landmark is anchored at
/// a live clone the four blocks come from that clone's relative cov; otherwise (legacy
/// / absolute path, or `None` map) both consumers fall back to the shared absolute
/// `(p_vv, p_ww)` — identical to the pre-clone behaviour.
#[inline]
#[allow(clippy::type_complexity)]
fn resolve_pose_cov<'a>(
    rel_cov_by_clone: Option<&'a CloneRelCov>,
    anchor_clone_id: Option<u64>,
    p_vv: Option<&'a Matrix3<f64>>,
    p_ww: Option<&'a Matrix3<f64>>,
) -> (
    Option<&'a Matrix3<f64>>,
    Option<&'a Matrix3<f64>>,
    Option<&'a Matrix3<f64>>,
    Option<&'a Matrix3<f64>>,
) {
    if let (Some(map), Some(cid)) = (rel_cov_by_clone, anchor_clone_id) {
        if let Some((fv, fw, iv, iw)) = map.get(&cid) {
            return (Some(fv), Some(fw), Some(iv), Some(iw));
        }
    }
    (p_vv, p_ww, p_vv, p_ww)
}

/// Per-landmark measurement update, dispatched by chart: the SOT(3) IEKF for
/// `Polar`/`InvDepth`, the additive inverse-depth EKF for `InvDepthAdditive`.
#[allow(clippy::too_many_arguments)]
fn update_feature_3d(
    chart: Sparse3DChart,
    k: &Matrix3<f64>,
    cam: Option<&dyn CameraModel>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    p_vv_inc: Option<&Matrix3<f64>>,
    p_ww_inc: Option<&Matrix3<f64>>,
    dt: f64,
    consider: Option<&ConsiderRel>,
) -> bool {
    match chart {
        Sparse3DChart::InvDepthAdditive => invdepth_additive_update_3d(
            k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, p_vv_inc, p_ww_inc, dt,
        ),
        Sparse3DChart::BearingInvDepthAdditive => match cam {
            Some(cam) if settings.consider_relpose && settings.bias_walk_var <= 0.0 => {
                match consider {
                    Some(cr) => bearing_consider_update_3d(
                        cam, k, settings, feat, uv_obs, t_cw_curr, cr, dt,
                    ),
                    // No consider entry for this anchor (feed not in consider mode, or the
                    // anchor is not a live clone): the landmark cannot be updated honestly.
                    // Skip the frame rather than silently run the other model.
                    None => true,
                }
            }
            Some(cam) => bearing_invdepth_additive_update_3d(
                cam, k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, p_vv_inc, p_ww_inc, dt,
            ),
            // No camera attached: cannot project. Drop the track.
            None => false,
        },
        _ => iekf_update_3d(
            chart, k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, p_vv_inc, p_ww_inc, dt,
        ),
    }
}

fn measurement_variance_px2(settings: &SparseVogSettings, feat: &FeatureState3D) -> f64 {
    let age = feat.track_length.saturating_sub(1) as f64;
    settings.sigma_pixel.powi(2) + (settings.flow_age_rate_px_per_frame * age).powi(2)
}

/// Rho-first additive inverse-depth EKF update for one landmark.
///
/// State `s = (alpha, beta, rho) = (X/Z, Y/Z, 1/Z)` of the landmark in the anchor
/// camera frame, covariance `feat.inv_p` in those coords. The known relative pose
/// anchor->current is folded into the pinhole measurement; the update is a plain
/// EKF (the inverse-depth refinement group is abelian / flat: `s <- s + gamma`),
/// with the same Gaussian-Beta inlier weighting and optional process-noise floor
/// as the SOT(3) path. No log-depth, no re-charting -> no sequential-linearisation
/// overconfidence: a multiplicative (log-depth) chart re-introduces it and
/// biases the converged range, so the additive inverse-depth chart is used.
/// Keep inverse depth inside the configured depth range.
///
/// Birth already gates `range_anchor` to `[min_depth, max_depth]`, but the
/// additive charts then update `rho` with no constraint, so it is free to walk
/// to zero (range -> infinity) or negative. That is not merely a bad estimate:
/// the reported range variance is `r^4 * var_rho`, so a drifting rho inflates it
/// by the fourth power and the feature can never pass `conv_variance_threshold`.
fn clamp_rho(settings: &SparseVogSettings, rho: f64) -> f64 {
    let lo = 1.0 / settings.max_depth.max(1e-6);
    let hi = 1.0 / settings.min_depth.max(1e-6);
    rho.clamp(lo, hi)
}

fn invdepth_additive_update_3d(
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    // This chart has no accumulating range injection; the incremental pose cov
    // (used only by the pose-range term of the bearing charts) is unused here.
    _p_vv_inc: Option<&Matrix3<f64>>,
    _p_ww_inc: Option<&Matrix3<f64>>,
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

    if dt > 0.0 && settings.range_walk_var > 0.0 {
        let q_c = r_ca * pa_of(&feat.inv_s) + t_ca_t;
        if q_c[2] > settings.min_depth {
            let j_g = r_ca * jpa_of(&feat.inv_s);
            if let Some(j_inv) = j_g.try_inverse() {
                // Radial floor, pulled back into the chart by J_inv. Fed-pose
                // covariance belongs in measurement R below.
                let mut q_cur = Matrix3::zeros();
                if settings.range_walk_var > 0.0 {
                    let r_hat = q_c / q_c.norm();
                    q_cur +=
                        settings.range_walk_var * q_c.norm_squared() * (r_hat * r_hat.transpose());
                }
                let sigma = feat.inv_p + j_inv * q_cur * j_inv.transpose();
                feat.inv_p = 0.5 * (sigma + sigma.transpose());
            }
        }
    }

    let mut r_meas = Matrix2::identity() * measurement_variance_px2(settings, feat);

    let q_c = r_ca * pa_of(&feat.inv_s) + t_ca_t;
    if q_c[2] < settings.min_depth {
        return false;
    }
    let (xc, yc, zc) = (q_c[0], q_c[1], q_c[2]);
    let z2 = zc * zc;
    let proj = Matrix2x3::new(fx / zc, 0.0, -fx * xc / z2, 0.0, fy / zc, -fy * yc / z2);
    // Measurement-side pose uncertainty: the pixel comes from the true pose but
    // the filter is fed a noisy pose. Pose error is therefore a measurement
    // discrepancy, folded into R so it enters both S and the Joseph posterior.
    // Translation scales through proj * P_vv * proj^T; rotation scales through
    // (proj * skew(q_c)) * P_ww * (proj * skew(q_c))^T.
    if dt > 0.0 {
        if let Some(pvv) = p_vv {
            r_meas += proj * pvv * proj.transpose();
        }
        if let Some(pww) = p_ww {
            let jphi = proj * base_skew(&q_c);
            r_meas += jphi * pww * jphi.transpose();
        }
        // Fixed anchor-pose uncertainty is folded into the landmark covariance
        // at feature birth. Re-adding it here as per-frame measurement noise
        // double counts the same anchor error.
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
    feat.inv_s[2] = clamp_rho(settings, feat.inv_s[2]);
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

/// Camera-agnostic additive inverse-range EKF update on the tangent-bearing
/// chart. Structurally identical to [`invdepth_additive_update_3d`]; only the
/// anchor parameterization (`P = b(eta)/rho` via [`bearing_chart`]) and the
/// projection (`cam.project` / `cam.projection_jacobian`) differ. `k` is used
/// only for the Gaussian-Beta outlier density scale.
#[allow(clippy::too_many_arguments)]
fn bearing_invdepth_additive_update_3d(
    cam: &dyn CameraModel,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    // Measurement-side (full anchor→current relative) pose cov: `r_meas`, fresh/frame.
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    // Range-injection (per-frame increment) pose cov: accumulates into `inv_p`.
    p_vv_inc: Option<&Matrix3<f64>>,
    p_ww_inc: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    if settings.bias_walk_var > 0.0 {
        return bearing_bias_update_3d(
            cam, k, settings, feat, uv_obs, t_cw_curr, p_vv, p_ww, p_vv_inc, p_ww_inc, dt,
        );
    }
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];

    let t_ca = t_cw_curr * feat.anchor_t_wc;
    let r_ca = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
    let t_ca_t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();

    // anchor-frame point P(s) and its Jacobian dP/ds on the bearing chart.
    let b0 = feat.anchor_ray;
    let u = feat.tangent_basis;
    let pj_of = |s: &Vector3<f64>| {
        let eta = Vector2::new(s[0], s[1]);
        bearing_chart::point_and_jacobian(&b0, &u, &eta, s[2])
    };

    // Range (radial) process noise: the only channel that inflates the depth covariance
    // (the measurement term proj*P*proj^T below is radial-blind, d(pi)/dq * r_hat = 0).
    // Combines the fixed range-walk floor with the pose-driven, parallax-scaled term
    // (eq. 10-11 of the formulation) when per-frame incremental pose covariances are supplied.
    let pose_range_on = if settings.pose_range_direct {
        p_vv.is_some() && p_ww.is_some()
    } else {
        settings.pose_range_scale > 0.0 && p_vv_inc.is_some() && p_ww_inc.is_some()
    };
    if dt > 0.0 && (settings.range_walk_var > 0.0 || pose_range_on) {
        let (pa, jpa) = pj_of(&feat.inv_s);
        let q_c = r_ca * pa + t_ca_t;
        if q_c[2] > settings.min_depth {
            let r2 = q_c.norm_squared();
            let mut q_range = settings.range_walk_var * r2;
            if pose_range_on {
                // baseline (anchor->current) with a 1%-parallax floor bounding 1/b^2.
                let b2 = t_ca_t.norm_squared().max(1e-4 * r2);
                q_range += pose_range_q(settings, feat, r2, b2, p_vv, p_ww, p_vv_inc, p_ww_inc);
            }
            if q_range > 0.0 {
                let j_g = r_ca * jpa;
                if let Some(j_inv) = j_g.try_inverse() {
                    let r_hat = q_c / q_c.norm();
                    let q_cur = q_range * (r_hat * r_hat.transpose());
                    let sigma = feat.inv_p + j_inv * q_cur * j_inv.transpose();
                    feat.inv_p = 0.5 * (sigma + sigma.transpose());
                }
            }
        }
    }

    let mut r_meas = Matrix2::identity() * measurement_variance_px2(settings, feat);

    let (pa, jpa) = pj_of(&feat.inv_s);
    let q_c = r_ca * pa + t_ca_t;
    if q_c[2] < settings.min_depth {
        return false;
    }
    let proj = cam.projection_jacobian(&q_c); // dpixel/dP_cam (2x3)
                                              // Measurement-side pose uncertainty (see invdepth_additive_update_3d).
    if dt > 0.0 {
        if let Some(pvv) = p_vv {
            r_meas += proj * pvv * proj.transpose();
        }
        if let Some(pww) = p_ww {
            let jphi = proj * base_skew(&q_c);
            r_meas += jphi * pww * jphi.transpose();
        }
    }
    let c = proj * r_ca * jpa; // dh/ds (2x3)
    let s_mat = c * feat.inv_p * c.transpose() + r_meas;
    let Some(s_inv) = (s_mat + Matrix2::identity() * 1e-8).try_inverse() else {
        return true;
    };
    let det_s = s_mat.determinant();
    if det_s < 1e-30 {
        return true;
    }
    let y_pred = cam.project(&q_c);
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

    feat.inv_s += gamma;
    feat.inv_s[2] = clamp_rho(settings, feat.inv_s[2]);
    feat.inv_p = 0.5 * (p_new + p_new.transpose());
    feat.track_length += 1;
    update_beta(settings, feat, w1, w2);

    let (pa_new, jpa_new) = pj_of(&feat.inv_s);
    let j_g = r_ca * jpa_new; // dP_cur/ds
    feat.position = r_ca * pa_new + t_ca_t;
    let cov_euc = j_g * feat.inv_p * j_g.transpose();
    feat.covariance = 0.5 * (cov_euc + cov_euc.transpose());
    true
}

/// Consider (Schmidt) bearing-invdepth update, filter_formulation §V-F eqs (18)-(21).
///
/// The anchor->current relative-pose error `theta` (`[phi; t]`, perturbing the point in
/// the current frame as `dq_c = [q_c]_x phi + t`, the same convention as the
/// measurement-side pose term it replaces) is a CONSIDERED variable: never estimated
/// here, but carried in the joint covariance with the landmark through
/// `feat.p_stheta`. Per frame: (21) transport `P_sth <- P_sth M'`; (19) joint innovation
/// covariance and consider gain; (20) joint-Joseph posterior and the cross-term update
/// `P_sth+ = (I - K C) P_sth - K J Sigma`, which is where a RADIAL component is born
/// (the gain's rho-row is non-zero under parallax) -- the thing the radial-blind
/// measurement term (9) cannot do and the range injection (10)-(11) imitated by hand.
/// No range injection and no measurement-side pose term run in this path. The cached
/// current-frame covariance includes the transported pose covariance and the cross term,
/// since that is the covariance of the point the consumer is handed.
#[allow(clippy::too_many_arguments)]
fn bearing_consider_update_3d(
    cam: &dyn CameraModel,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    cr: &ConsiderRel,
    dt: f64,
) -> bool {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let t_ca = t_cw_curr * feat.anchor_t_wc;
    let r_ca = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
    let t_ca_t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();
    let b0 = feat.anchor_ray;
    let u = feat.tangent_basis;
    let pj_of = |s: &Vector3<f64>| {
        let eta = Vector2::new(s[0], s[1]);
        bearing_chart::point_and_jacobian(&b0, &u, &eta, s[2])
    };
    let sigma = cr.sigma;

    // (21) transport of the cross term to this frame's relative pose.
    if dt > 0.0 {
        feat.p_stheta *= cr.m.transpose();
    }

    let (pa, jpa) = pj_of(&feat.inv_s);
    let q_c = r_ca * pa + t_ca_t;
    if q_c[2] < settings.min_depth {
        return false;
    }
    let proj = cam.projection_jacobian(&q_c); // 2x3
    let mut j_pose = Matrix2x6::<f64>::zeros(); // d pixel / d theta
    j_pose
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&(proj * base_skew(&q_c)));
    j_pose.fixed_view_mut::<2, 3>(0, 3).copy_from(&proj);
    let c = proj * r_ca * jpa; // 2x3
    let r_core = Matrix2::identity() * measurement_variance_px2(settings, feat);

    // (19)
    let p = feat.inv_p;
    let pst = feat.p_stheta;
    let cross_s = c * pst * j_pose.transpose(); // 2x2
    let s_mat = c * p * c.transpose()
        + cross_s
        + cross_s.transpose()
        + j_pose * sigma * j_pose.transpose()
        + r_core;
    let Some(s_inv) = (s_mat + Matrix2::identity() * 1e-8).try_inverse() else {
        return true;
    };
    let det_s = s_mat.determinant();
    if det_s < 1e-30 {
        return true;
    }
    let gain = (p * c.transpose() + pst * j_pose.transpose()) * s_inv; // 3x2

    let y_pred = cam.project(&q_c);
    let residual = uv_obs - y_pred;
    let maha_sq = (residual.transpose() * s_inv * residual)[(0, 0)];
    feat.last_nis = maha_sq;
    if settings.mahalanobis_reset_chi2 > 0.0 && maha_sq > settings.mahalanobis_reset_chi2 {
        return false;
    }
    let full_delta = gain * residual;

    // (20), joint Joseph form: the posterior error is L [ds; dth] + K eps with
    // L = [I - K C, -K J].
    let i_kc = Matrix3::identity() - gain * c;
    let kj = gain * j_pose; // 3x6
    let cross_post = i_kc * pst * kj.transpose();
    let p_post = i_kc * p * i_kc.transpose() - cross_post - cross_post.transpose()
        + kj * sigma * kj.transpose()
        + gain * r_core * gain.transpose();
    let pst_post = i_kc * pst - kj * sigma;

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
    let p_new = w1 * p_post + w2 * p + w1 * w2 * (full_delta * full_delta.transpose());
    if p_new
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|v| *v <= 0.0 || !v.is_finite())
    {
        return true;
    }
    // The outlier hypothesis leaves the cross term as it was; theta has zero mean
    // under both, so the mixture adds no spread to it.
    let pst_new = w1 * pst_post + w2 * pst;
    if !pst_new.iter().all(|v| v.is_finite()) {
        return true;
    }

    feat.inv_s += gamma;
    feat.inv_s[2] = clamp_rho(settings, feat.inv_s[2]);
    feat.inv_p = 0.5 * (p_new + p_new.transpose());
    feat.p_stheta = pst_new;
    feat.track_length += 1;
    update_beta(settings, feat, w1, w2);

    // Current-frame report: the point handed to a consumer is q_c = T^{ca} P(s), whose
    // error is J_s ds + J_th dth -- landmark, transported pose, and their cross term.
    let (pa_new, jpa_new) = pj_of(&feat.inv_s);
    let j_g = r_ca * jpa_new; // 3x3
    feat.position = r_ca * pa_new + t_ca_t;
    let mut j_th = Matrix3x6::<f64>::zeros();
    j_th.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&base_skew(&feat.position));
    j_th.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&Matrix3::identity());
    let cross_e = j_g * feat.p_stheta * j_th.transpose();
    let cov_euc = j_g * feat.inv_p * j_g.transpose()
        + j_th * sigma * j_th.transpose()
        + cross_e
        + cross_e.transpose();
    feat.covariance = 0.5 * (cov_euc + cov_euc.transpose());
    true
}

/// Bias-augmented bearing-invdepth update: joint 5-DOF EKF on the landmark chart
/// state `s = (eta1, eta2, rho)` plus a per-track 2D correspondence-bias `b` with a
/// random-walk process model. Measurement `u = pi(P(s)) + b + eps`, so H = [C | I2].
///
/// The bias absorbs the temporally-correlated Rudolf/KLT correspondence drift (exact-
/// GT MidAir shows it is a random walk, not a constant offset), so repeated same-track
/// observations no longer over-count and the reported landmark covariance stops
/// collapsing. Because the bias explains the drift, the landmark estimate should also
/// stay nearer truth. The reported `covariance` is the landmark MARGINAL (the 3x3
/// block of the joint posterior projected to Euclidean). Enabled by
/// `settings.bias_walk_var > 0`; see the parent function.
fn bearing_bias_update_3d(
    cam: &dyn CameraModel,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_obs: &Vector2<f64>,
    t_cw_curr: &Matrix4<f64>,
    // Measurement-side (full anchor→current relative) pose cov: `r_meas`, fresh/frame.
    p_vv: Option<&Matrix3<f64>>,
    p_ww: Option<&Matrix3<f64>>,
    // Range-injection (per-frame increment) pose cov: accumulates into `inv_p`.
    p_vv_inc: Option<&Matrix3<f64>>,
    p_ww_inc: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];

    let t_ca = t_cw_curr * feat.anchor_t_wc;
    let r_ca = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
    let t_ca_t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();
    let b0 = feat.anchor_ray;
    let u = feat.tangent_basis;
    let pj_of = |s: &Vector3<f64>| {
        let eta = Vector2::new(s[0], s[1]);
        bearing_chart::point_and_jacobian(&b0, &u, &eta, s[2])
    };

    // --- process: landmark range walk (unchanged) + bias random walk ---
    // Range (radial) process noise: the only channel that inflates the depth covariance
    // (the measurement term proj*P*proj^T below is radial-blind, d(pi)/dq * r_hat = 0).
    // Combines the fixed range-walk floor with the pose-driven, parallax-scaled term
    // (eq. 10-11 of the formulation) when per-frame incremental pose covariances are supplied.
    let pose_range_on = if settings.pose_range_direct {
        p_vv.is_some() && p_ww.is_some()
    } else {
        settings.pose_range_scale > 0.0 && p_vv_inc.is_some() && p_ww_inc.is_some()
    };
    if dt > 0.0 && (settings.range_walk_var > 0.0 || pose_range_on) {
        let (pa, jpa) = pj_of(&feat.inv_s);
        let q_c = r_ca * pa + t_ca_t;
        if q_c[2] > settings.min_depth {
            let r2 = q_c.norm_squared();
            let mut q_range = settings.range_walk_var * r2;
            if pose_range_on {
                // baseline (anchor->current) with a 1%-parallax floor bounding 1/b^2.
                let b2 = t_ca_t.norm_squared().max(1e-4 * r2);
                q_range += pose_range_q(settings, feat, r2, b2, p_vv, p_ww, p_vv_inc, p_ww_inc);
            }
            if q_range > 0.0 {
                let j_g = r_ca * jpa;
                if let Some(j_inv) = j_g.try_inverse() {
                    let r_hat = q_c / q_c.norm();
                    let q_cur = q_range * (r_hat * r_hat.transpose());
                    let sigma = feat.inv_p + j_inv * q_cur * j_inv.transpose();
                    feat.inv_p = 0.5 * (sigma + sigma.transpose());
                }
            }
        }
    }
    // Random walk on the bias: b_k = b_{k-1} + w_k, w_k ~ N(0, bias_walk_var I).
    feat.p_bb += Matrix2::identity() * settings.bias_walk_var;

    let mut r_meas = Matrix2::identity() * measurement_variance_px2(settings, feat);

    let (pa, jpa) = pj_of(&feat.inv_s);
    let q_c = r_ca * pa + t_ca_t;
    if q_c[2] < settings.min_depth {
        return false;
    }
    let proj = cam.projection_jacobian(&q_c);
    if dt > 0.0 {
        if let Some(pvv) = p_vv {
            r_meas += proj * pvv * proj.transpose();
        }
        if let Some(pww) = p_ww {
            let jphi = proj * base_skew(&q_c);
            r_meas += jphi * pww * jphi.transpose();
        }
    }
    let c = proj * r_ca * jpa; // dh/ds (2x3)

    // Assemble the joint 5x5 covariance P5 and H = [C | I2].
    let mut p5 = Matrix5::<f64>::zeros();
    p5.fixed_view_mut::<3, 3>(0, 0).copy_from(&feat.inv_p);
    p5.fixed_view_mut::<3, 2>(0, 3).copy_from(&feat.p_sb);
    p5.fixed_view_mut::<2, 3>(3, 0)
        .copy_from(&feat.p_sb.transpose());
    p5.fixed_view_mut::<2, 2>(3, 3).copy_from(&feat.p_bb);
    let mut h = Matrix2x5::<f64>::zeros();
    h.fixed_view_mut::<2, 3>(0, 0).copy_from(&c);
    h.fixed_view_mut::<2, 2>(0, 3)
        .copy_from(&Matrix2::identity());

    let s_mat = h * p5 * h.transpose() + r_meas;
    let Some(s_inv) = (s_mat + Matrix2::identity() * 1e-8).try_inverse() else {
        return true;
    };
    let det_s = s_mat.determinant();
    if det_s < 1e-30 {
        return true;
    }
    let y_pred = cam.project(&q_c);
    let residual = uv_obs - y_pred - feat.bias; // measurement now includes the bias
    let maha_sq = (residual.transpose() * s_inv * residual)[(0, 0)];
    feat.last_nis = maha_sq;
    if settings.mahalanobis_reset_chi2 > 0.0 && maha_sq > settings.mahalanobis_reset_chi2 {
        return false;
    }
    let gain = p5 * h.transpose() * s_inv; // 5x2
    let full_delta = gain * residual; // 5x1
    let i_kh = Matrix5::identity() - gain * h;
    let p_post = i_kh * p5 * i_kh.transpose() + gain * r_meas * gain.transpose();

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
    let p_new = w1 * p_post + w2 * p5 + w1 * w2 * (full_delta * full_delta.transpose());
    if p_new
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|v| *v <= 0.0 || !v.is_finite())
    {
        return true;
    }

    feat.inv_s[0] += gamma[0];
    feat.inv_s[1] += gamma[1];
    feat.inv_s[2] = clamp_rho(settings, feat.inv_s[2] + gamma[2]);
    feat.bias[0] += gamma[3];
    feat.bias[1] += gamma[4];
    let inv_p = p_new.fixed_view::<3, 3>(0, 0).into_owned();
    feat.inv_p = 0.5 * (inv_p + inv_p.transpose());
    feat.p_sb = p_new.fixed_view::<3, 2>(0, 3).into_owned();
    let p_bb = p_new.fixed_view::<2, 2>(3, 3).into_owned();
    feat.p_bb = 0.5 * (p_bb + p_bb.transpose());
    feat.track_length += 1;
    update_beta(settings, feat, w1, w2);

    let (pa_new, jpa_new) = pj_of(&feat.inv_s);
    let j_g = r_ca * jpa_new;
    feat.position = r_ca * pa_new + t_ca_t;
    // Reported covariance = landmark marginal (bias-explained, non-collapsed).
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
    _p_vv: Option<&Matrix3<f64>>,
    _p_ww: Option<&Matrix3<f64>>,
    _p_vv_inc: Option<&Matrix3<f64>>,
    _p_ww_inc: Option<&Matrix3<f64>>,
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

    // Per-step radial floor. The IEKF has no propagation, so without a floor the
    // static-landmark covariance can shrink below unmodelled range bias.
    if dt > 0.0 && settings.range_walk_var > 0.0 {
        let p = q_hat_a_of(&feat.x);
        let q_c = r_ca * p + t_ca_t;
        if q_c[2] > settings.min_depth {
            let j_c = r_ca * dq_hat_a(&feat.x);
            if let Some(j_inv) = j_c.try_inverse() {
                let mut q_cur = Matrix3::zeros();
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

    let r_meas = Matrix2::identity() * measurement_variance_px2(settings, feat);

    // Each branch produces the inlier-conditioned update terms shared by the
    // Gaussian-Beta tail below: the gating Mahalanobis^2 and det(S), the full
    // (un-weighted) state correction in eps coords, and the posterior covariance
    // P assuming the measurement is an inlier.
    let (maha_sq, det_s, full_delta, p_post) = match settings.second_order_mode {
        SecondOrderMode::Analytic => {
            // Analytic second-order EqF: bias-correct the prediction and add
            // closed-form second-order innovation inflation.
            let p = q_hat_a_of(&feat.x);
            let q_c = r_ca * p + t_ca_t;
            if q_c[2] < settings.min_depth {
                return false;
            }
            let (xc, yc, zc) = (q_c[0], q_c[1], q_c[2]);
            let z2 = zc * zc;
            // Projection Jacobian P (2x3) and per-channel Hessians.
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
            // Cholesky factor: the columns of L whiten the directional Hessian.
            let Some(chol) = feat.sigma.cholesky() else {
                return true;
            };
            let l = chol.l();
            let mut omega = [Vector3::zeros(); 3];
            let mut alpha = [0.0f64; 3];
            let mut wxp = [Vector3::zeros(); 3];
            let mut b = [Vector3::zeros(); 3];
            for j in 0..3 {
                let lj = l.column(j).into_owned();
                let oj = m2g_top * lj;
                omega[j] = oj;
                alpha[j] = (m2g_bot * lj)[(0, 0)];
                wxp[j] = oj.cross(&p);
                b[j] = rg * lj;
            }
            // Whitened output Hessians. B is the polarization of the SOT(3)
            // action quadratic term.
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
            // Bias-corrected prediction.
            let y_pred = Vector2::new(fx * xc / zc + cx, fy * yc / zc + cy);
            let y_hat = y_pred + 0.5 * Vector2::new(ht_u.trace(), ht_v.trace());
            // Second-order innovation inflation (Gram, PSD).
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
            // Cross-covariance keeps the EKF gain shape to this order.
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
    anchor_pose_cov: &Matrix3<f64>,
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
    p_euc += anchor_pose_cov;

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
    let r = Matrix2::identity() * measurement_variance_px2(settings, feat);
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

fn baseline_tau(p_vv: Option<&Matrix3<f64>>, t: &Vector3<f64>) -> f64 {
    let Some(p_vv) = p_vv else {
        return 0.0;
    };
    let norm_sq = t.norm_squared();
    if norm_sq <= 1e-16 {
        return 0.0;
    }
    let t_hat = t / norm_sq.sqrt();
    (t_hat.transpose() * p_vv * t_hat)[(0, 0)] / norm_sq
}

fn current_anchor_rotation_init_cov(
    k: &Matrix3<f64>,
    uv_ref: &Vector2<f64>,
    uv_curr: &Vector2<f64>,
    r_ref: &Matrix3<f64>,
    t_ref: &Vector3<f64>,
    position: &Vector3<f64>,
    p_ww: Option<&Matrix3<f64>>,
) -> Matrix3<f64> {
    let Some(p_ww) = p_ww else {
        return Matrix3::zeros();
    };
    let x_curr = (uv_curr[0] - k[(0, 2)]) / k[(0, 0)];
    let y_curr = (uv_curr[1] - k[(1, 2)]) / k[(1, 1)];
    let x_prev = (uv_ref[0] - k[(0, 2)]) / k[(0, 0)];
    let y_prev = (uv_ref[1] - k[(1, 2)]) / k[(1, 1)];
    let b_curr = Vector3::new(x_curr, y_curr, 1.0);
    let b_prev = Vector3::new(x_prev, y_prev, 1.0);
    let aligned = r_ref * b_prev;
    if aligned[2].abs() <= 1e-12 {
        return Matrix3::zeros();
    }
    let x_prev_rect = aligned[0] / aligned[2];
    let y_prev_rect = aligned[1] / aligned[2];
    let n = Vector2::new(
        x_prev_rect * t_ref[2] - t_ref[0],
        y_prev_rect * t_ref[2] - t_ref[1],
    );
    let d = Vector2::new(x_prev_rect - x_curr, y_prev_rect - y_curr);
    let g = n.dot(&n);
    let h = n.dot(&d);
    if h.abs() <= 1e-12 || g <= 1e-24 {
        return Matrix3::zeros();
    }

    let mut j = Matrix3::zeros();
    for axis in 0..3 {
        let da = base_skew(&aligned).column(axis).into_owned();
        let dt = base_skew(t_ref).column(axis).into_owned();
        let dx = (da[0] * aligned[2] - aligned[0] * da[2]) / (aligned[2] * aligned[2]);
        let dy = (da[1] * aligned[2] - aligned[1] * da[2]) / (aligned[2] * aligned[2]);
        let dn = Vector2::new(
            dx * t_ref[2] + x_prev_rect * dt[2] - dt[0],
            dy * t_ref[2] + y_prev_rect * dt[2] - dt[1],
        );
        let dd = Vector2::new(dx, dy);
        let dg = 2.0 * n.dot(&dn);
        let dh = dn.dot(&d) + n.dot(&dd);
        let dz = (dg * h - g * dh) / (h * h);
        let dq_est = b_curr * dz;
        let dq_true = base_skew(position).column(axis).into_owned();
        j.set_column(axis, &(dq_est - dq_true));
    }

    let cov = j * p_ww * j.transpose();
    0.5 * (cov + cov.transpose())
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
            birth_min_flow_px: 0.5,
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

    #[test]
    fn bearing_chart_matches_invdepth_for_pinhole_camera() {
        use crate::mathematical::camera::{CameraModel, PinholeModel};
        use approx::assert_relative_eq;
        use std::sync::Arc;

        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
        });
        let mut f_inv = Sparse3DFilter::invdepth_additive3d(k(), settings());
        let mut f_bear = Sparse3DFilter::bearing_invdepth3d(k(), cam, settings());

        // Camera translates along x; a fixed world point in front of it.
        let point_w = Vector3::new(0.5, 0.1, 4.0);
        let n = 40;
        for i in 0..n {
            update_with_point(&mut f_inv, i, point_w);
            update_with_point(&mut f_bear, i, point_w);
        }

        let p_inv = f_inv.feature(42).expect("invdepth track").position;
        let p_bear = f_bear.feature(42).expect("bearing track").position;

        // True current-frame point at the last processed pose. With adequate
        // parallax the bearing chart converges to the same geometry as the
        // pinhole chart and to ground truth.
        let p_true = point_w - Vector3::new((n as f64 - 1.0) * 0.05, 0.0, 0.0);
        assert_relative_eq!(p_bear, p_inv, epsilon = 0.01);
        assert_relative_eq!(p_bear, p_true, epsilon = 0.01);
    }

    /// Civera undelayed birth: a feature must be born from a STATIONARY camera,
    /// where two-view triangulation has no parallax to work with and the classical
    /// path holds the feature pending forever. Also pins the property that makes
    /// undelayed birth safe -- the 2-sigma band on rho reaches 0, i.e. infinity.
    #[test]
    fn civera_birth_needs_no_parallax_and_band_reaches_infinity() {
        use crate::mathematical::camera::{CameraModel, PinholeModel};
        use std::sync::Arc;

        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
        });
        let point_w = Vector3::new(0.5, 0.1, 4.0);

        // Feed the SAME pose twice: zero baseline, hence zero parallax.
        let stationary = |filter: &mut Sparse3DFilter| {
            for i in 0..2 {
                let t_wc = Matrix4::identity();
                let mut coords = HashMap::new();
                coords.insert(42u64, project(&k(), point_w));
                filter.update(
                    &VisionMeasurement::new(i as f64 * 0.05, coords),
                    &t_wc,
                    None,
                    None,
                );
            }
        };

        // Control: classical delayed birth cannot birth without parallax.
        let mut f_delayed = Sparse3DFilter::bearing_invdepth3d(k(), cam.clone(), settings());
        stationary(&mut f_delayed);
        assert!(
            f_delayed.feature(42).is_none(),
            "delayed triangulated birth must stay pending at zero parallax"
        );

        // Civera: births on first sight at the prior depth.
        let d0 = 5.0;
        let mut st = settings();
        st.civera_birth_depth = d0;
        st.civera_rho_sigma_frac = 0.5;
        let mut f_civera = Sparse3DFilter::bearing_invdepth3d(k(), cam, st);
        stationary(&mut f_civera);
        let feat = f_civera
            .feature(42)
            .expect("civera birth must succeed with no parallax");

        // Born at the prior depth, not at a triangulated one.
        let rho0 = 1.0 / d0;
        assert!(
            (feat.inv_s[2] - rho0).abs() < 1e-9,
            "born rho {} != prior rho {rho0}",
            feat.inv_s[2]
        );
        // sigma_rho = rho0/2 => rho0 - 2*sigma_rho = 0 => depth band reaches infinity.
        let sigma_rho = feat.inv_p[(2, 2)].sqrt();
        assert!(
            (rho0 - 2.0 * sigma_rho).abs() < 1e-9,
            "2-sigma band must reach rho=0 (infinity); rho0 {rho0} sigma {sigma_rho}"
        );
    }

    /// Re-anchoring is a change of PARAMETRIZATION: the world point must not move, and
    /// the uncertainty must be conserved (not silently dropped when eta is reset to 0).
    /// §V-F in miniature. One landmark, a camera translating along +x, the anchor->
    /// current relative-pose covariance growing with the baseline (and a translation
    /// block only, so the measurement-side term alone would be radial-blind by (9)).
    /// Checks the three things the consider recursion must do that the legacy path
    /// cannot: (i) a non-zero cross term with a RADIAL (rho) row after one update,
    /// (ii) a reported current-frame range variance that GROWS with the pose covariance
    /// while the legacy filter's stays pinned, and (iii) a re-anchor whose covariance
    /// with the tracked cross term is no larger than the independence version.
    /// Cross-epoch probe for the consider recursion (filter_formulation §V-F): the
    /// VIO's error is a COMMON-MODE scale factor on every baseline, and the covariance it
    /// hands over is honest for it (sigma_s = s). One filter keeps its birth anchor for
    /// the whole track (one epoch); the other is moved onto the current clone every
    /// `period` frames the way `CloneRelFeed` does in consider mode, which transports
    /// the old->new covariance into the landmark and RESETS the cross term. Both are
    /// scored against the TRUE range from the true camera at the last frame. Prints the
    /// two reported variances and NEES; the assertions pin only what is structural.
    #[test]
    fn consider_reanchor_epochs_are_independent_under_a_common_scale_error() {
        use crate::mathematical::camera::PinholeModel;
        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
        });
        let kmat = k();
        let point_w = Vector3::new(0.3, -0.2, 6.0);
        let (s_true, s_sig, step) = (0.05, 0.05, 0.3);
        let sigma_for = |sep: usize| {
            let mut s = Matrix6::<f64>::zeros();
            s[(3, 3)] = (s_sig * step * sep as f64).powi(2);
            s[(4, 4)] = 1e-10;
            s[(5, 5)] = 1e-10;
            s
        };
        // True camera at x = step*i; the VIO reports x = step*i*(1+s): every baseline is
        // scaled by the same factor. Pixels come from the TRUE geometry.
        let frame = |i: usize| {
            let t_true = pose(step * i as f64);
            let t_rep = pose(step * i as f64 * (1.0 + s_true));
            let q_c = t_true.try_inverse().unwrap()
                * Vector4::new(point_w[0], point_w[1], point_w[2], 1.0);
            let uv = cam.project(&Vector3::new(q_c[0], q_c[1], q_c[2]));
            let mut coords = HashMap::new();
            coords.insert(7u64, Vector2::new(uv[0] as f32, uv[1] as f32));
            (
                t_rep,
                t_true,
                VisionMeasurement::new(i as f64 * 0.04, coords),
            )
        };
        let n = 42usize;
        let run = |period: Option<usize>| -> (f64, f64) {
            let mut st = settings();
            st.min_track_length = 1;
            st.birth_min_flow_px = 0.0;
            st.pose_range_direct = false;
            st.pose_range_scale = 0.0;
            st.consider_relpose = true;
            let mut f = Sparse3DFilter::bearing_invdepth3d(kmat, cam.clone(), st);
            for i in 0..2 {
                let (t, _, m) = frame(i);
                f.update_with_clones(&m, &t, None, None, Some(i as u64), None);
            }
            assert_eq!(f.feature_count(), 1, "landmark born");
            let mut anchor = f.feature(7).unwrap().anchor_clone_id.unwrap() as usize;
            let mut true_range = 0.0;
            for i in 2..n {
                let (t, t_true, m) = frame(i);
                if let Some(p) = period {
                    if i - anchor >= p {
                        let sig = sigma_for(i - anchor);
                        assert!(f.reanchor_feature_full(7, &t, Some(i as u64), None, Some(&sig)));
                        anchor = i;
                    }
                }
                let mut cmap: ConsiderRelMap = HashMap::new();
                cmap.insert(
                    anchor as u64,
                    ConsiderRel {
                        sigma: sigma_for(i - anchor),
                        m: Matrix6::identity(),
                    },
                );
                f.set_consider_relpose(cmap);
                f.update_with_clones(&m, &t, None, None, Some(i as u64), None);
                let q = t_true.try_inverse().unwrap()
                    * Vector4::new(point_w[0], point_w[1], point_w[2], 1.0);
                true_range = Vector3::new(q[0], q[1], q[2]).norm();
            }
            let feat = f.feature(7).unwrap();
            let err = feat.position.norm() - true_range;
            let var = feat.range_variance_for_chart(f.chart());
            (err, var)
        };
        let (e1, v1) = run(None);
        let (e5, v5) = run(Some(5));
        let (e2, v2) = run(Some(2));
        eprintln!(
            "common scale error {s_true}: one epoch err {e1:.3} m sigma {:.3} m NEES {:.2}; \
             period 5 err {e5:.3} m sigma {:.3} m NEES {:.2}; \
             period 2 err {e2:.3} m sigma {:.3} m NEES {:.2}",
            v1.sqrt(),
            e1 * e1 / v1,
            v5.sqrt(),
            e5 * e5 / v5,
            v2.sqrt(),
            e2 * e2 / v2
        );
        // The estimate is the scaled reconstruction either way: the error is the same
        // scale error of the range, and no re-anchor rule can change it.
        assert!(
            (e1 - e5).abs() < 0.2 * e1.abs(),
            "errors differ: {e1} vs {e5}"
        );
        assert!(e1.abs() > 0.5 * s_true * true_range_of(&point_w, step, n - 1));
        // Structural claim under test: the one-epoch recursion carries the honest scale
        // covariance (NEES of order one); re-anchoring resets the cross term, so later
        // epochs re-measure the SAME scale error as if it were fresh, and the reported
        // variance shrinks with the number of epochs.
        assert!(
            e1 * e1 / v1 < 4.0,
            "one epoch is not honest: NEES {}",
            e1 * e1 / v1
        );
        assert!(
            v5 < 0.5 * v1,
            "period 5 did not shrink the report: {v5} vs {v1}"
        );
        assert!(v2 < v5, "period 2 should shrink it further: {v2} vs {v5}");
    }

    fn true_range_of(point_w: &Vector3<f64>, step: f64, i: usize) -> f64 {
        let q = pose(step * i as f64).try_inverse().unwrap()
            * Vector4::new(point_w[0], point_w[1], point_w[2], 1.0);
        Vector3::new(q[0], q[1], q[2]).norm()
    }

    #[test]
    fn consider_recursion_carries_pose_uncertainty_into_the_range_channel() {
        use crate::mathematical::camera::PinholeModel;
        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
        });
        let kmat = k();
        let point_w = Vector3::new(0.3, -0.2, 6.0);
        let build = |consider: bool| {
            let mut st = settings();
            st.min_track_length = 1;
            st.birth_min_flow_px = 0.0;
            st.pose_range_direct = false;
            st.pose_range_scale = 0.0;
            st.consider_relpose = consider;
            Sparse3DFilter::bearing_invdepth3d(kmat, cam.clone(), st)
        };
        let mut legacy = build(false);
        let mut consider = build(true);
        // Frames: camera at x = 0.3*i (enough parallax at 6 m for the birth gate),
        // identity attitude; landmark id 7.
        let frame = |i: usize| {
            let t = pose(0.3 * i as f64);
            let q_c =
                t.try_inverse().unwrap() * Vector4::new(point_w[0], point_w[1], point_w[2], 1.0);
            let uv = cam.project(&Vector3::new(q_c[0], q_c[1], q_c[2]));
            let mut coords = HashMap::new();
            coords.insert(7u64, Vector2::new(uv[0] as f32, uv[1] as f32));
            (t, VisionMeasurement::new(i as f64 * 0.04, coords))
        };
        // Birth needs two frames (pending -> triangulated); the anchor is frame 1.
        for i in 0..2 {
            let (t, m) = frame(i);
            legacy.update_with_clones(&m, &t, None, None, Some(i as u64), None);
            consider.update_with_clones(&m, &t, None, None, Some(i as u64), None);
        }
        assert_eq!(consider.feature_count(), 1, "landmark born");
        let anchor = consider.feature(7).unwrap().anchor_clone_id.unwrap();
        let mut ranges_legacy = Vec::new();
        let mut ranges_consider = Vec::new();
        for i in 2..12usize {
            let (t, m) = frame(i);
            // Translation-only relative covariance growing with the baseline.
            let base = 0.3 * (i as f64 - anchor as f64);
            let sig_t = (0.05 * base).powi(2);
            let mut sigma = Matrix6::<f64>::zeros();
            for d in 3..6 {
                sigma[(d, d)] = sig_t;
            }
            let mut rel: CloneRelCov = HashMap::new();
            rel.insert(
                anchor,
                (
                    Matrix3::identity() * sig_t,
                    Matrix3::zeros(),
                    Matrix3::zeros(),
                    Matrix3::zeros(),
                ),
            );
            legacy.update_with_clones(&m, &t, None, None, Some(i as u64), Some(&rel));
            let mut cmap: ConsiderRelMap = HashMap::new();
            cmap.insert(
                anchor,
                ConsiderRel {
                    sigma,
                    m: Matrix6::identity(),
                },
            );
            consider.set_consider_relpose(cmap);
            consider.update_with_clones(&m, &t, None, None, Some(i as u64), Some(&rel));
            ranges_legacy.push(
                legacy
                    .feature(7)
                    .unwrap()
                    .range_variance_for_chart(legacy.chart()),
            );
            ranges_consider.push(
                consider
                    .feature(7)
                    .unwrap()
                    .range_variance_for_chart(consider.chart()),
            );
        }
        let f = consider.feature(7).unwrap();
        // (i) cross term exists and has a radial row.
        assert!(
            f.p_stheta.iter().any(|v| v.abs() > 0.0),
            "cross term never born"
        );
        let rho_row = f.p_stheta.row(2);
        assert!(
            rho_row.iter().any(|v| v.abs() > 0.0),
            "no radial component in the cross term"
        );
        // (ii) the consider range variance grows with the pose covariance; the legacy one
        // (measurement-side term only, injection off) contracts.
        let (c0, c1) = (ranges_consider[1], *ranges_consider.last().unwrap());
        let (l0, l1) = (ranges_legacy[1], *ranges_legacy.last().unwrap());
        assert!(
            l1 <= l0,
            "legacy range variance should only contract: {l0} -> {l1}"
        );
        assert!(
            c1 > l1,
            "consider range variance must exceed the radial-blind one: {c1} vs {l1}"
        );
        assert!(
            c1 > 0.5 * c0,
            "consider range variance collapsed: {c0} -> {c1}"
        );
        // (iii) re-anchor onto the current clone is a pure re-parametrization of the
        // REPORTED current-frame covariance (landmark + transported pose + cross term):
        // the new anchor-chart covariance, mapped back to the current frame, must equal
        // what the last update cached. The sign of the cross term is NOT asserted -- it
        // is geometry-dependent (a first-version assertion that it lowers rho was wrong).
        let (t_new, _) = frame(11);
        let sigma_now = {
            let base = 0.3 * (11.0 - anchor as f64);
            let mut s = Matrix6::<f64>::zeros();
            for d in 3..6 {
                s[(d, d)] = (0.05 * base).powi(2);
            }
            s
        };
        let reported = consider.feature(7).unwrap().covariance;
        let mut indep = build(true);
        let mut ff = consider.feature(7).unwrap().clone();
        ff.p_stheta = Matrix3x6::zeros();
        indep.insert_feature(ff);
        assert!(consider.reanchor_feature_full(7, &t_new, Some(11), None, Some(&sigma_now)));
        assert!(indep.reanchor_feature_full(7, &t_new, Some(11), None, Some(&sigma_now)));
        let f = consider.feature(7).unwrap();
        let (_, j_new) = bearing_chart::point_and_jacobian(
            &f.anchor_ray,
            &f.tangent_basis,
            &Vector2::new(f.inv_s[0], f.inv_s[1]),
            f.inv_s[2],
        );
        let back = j_new * f.inv_p * j_new.transpose();
        let rel = (back - reported).abs().max() / reported.abs().max();
        assert!(
            rel < 1e-6,
            "consider re-anchor is not a re-parametrization of the report: {rel:.2e}"
        );
        let pi = indep.feature(7).unwrap().inv_p;
        assert!(
            (pi[(2, 2)] - f.inv_p[(2, 2)]).abs() > 1e-12 * pi[(2, 2)],
            "cross term made no difference"
        );
        assert!(
            consider
                .feature(7)
                .unwrap()
                .p_stheta
                .iter()
                .all(|v| *v == 0.0),
            "cross term reset for the new epoch"
        );
    }

    #[test]
    fn reanchor_preserves_the_world_point_and_conserves_uncertainty() {
        use crate::mathematical::camera::{CameraModel, PinholeModel};
        use approx::assert_relative_eq;
        use std::sync::Arc;

        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
        });
        let point_w = Vector3::new(0.5, 0.1, 4.0);
        let mut f = Sparse3DFilter::bearing_invdepth3d(k(), cam, settings());
        for i in 0..20 {
            update_with_point(&mut f, i, point_w);
        }

        let feat = f.feature(42).expect("track").clone();
        // World point implied by the CURRENT anchor parametrization.
        let (p_a, _) = bearing_chart::point_and_jacobian(
            &feat.anchor_ray,
            &feat.tangent_basis,
            &Vector2::new(feat.inv_s[0], feat.inv_s[1]),
            feat.inv_s[2],
        );
        let r_a = feat.anchor_t_wc.fixed_view::<3, 3>(0, 0).into_owned();
        let t_a = feat.anchor_t_wc.fixed_view::<3, 1>(0, 3).into_owned();
        let p_w_before = r_a * p_a + t_a;

        // Re-anchor onto a DIFFERENT pose (the one 5 steps later).
        let t_wc_new = pose(25.0 * 0.05);
        assert!(f.reanchor_feature(42, &t_wc_new, Some(7), None));

        let feat2 = f.feature(42).expect("track after reanchor");
        assert_eq!(feat2.anchor_clone_id, Some(7));
        // eta is reset to zero by construction: the new anchor ray IS the bearing.
        assert_relative_eq!(feat2.inv_s[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(feat2.inv_s[1], 0.0, epsilon = 1e-12);

        let (p_a2, _) = bearing_chart::point_and_jacobian(
            &feat2.anchor_ray,
            &feat2.tangent_basis,
            &Vector2::zeros(),
            feat2.inv_s[2],
        );
        let r_a2 = feat2.anchor_t_wc.fixed_view::<3, 3>(0, 0).into_owned();
        let t_a2 = feat2.anchor_t_wc.fixed_view::<3, 1>(0, 3).into_owned();
        let p_w_after = r_a2 * p_a2 + t_a2;

        // THE point: same world point, different parametrization.
        assert_relative_eq!(p_w_before, p_w_after, epsilon = 1e-9);

        // Uncertainty must survive the eta reset. Compare the EUCLIDEAN covariance in
        // the world frame, which is parametrization-independent; resetting eta to 0
        // without mapping its variance through would shrink this.
        let euc = |ft: &FeatureState3D| {
            let (_, j) = bearing_chart::point_and_jacobian(
                &ft.anchor_ray,
                &ft.tangent_basis,
                &Vector2::new(ft.inv_s[0], ft.inv_s[1]),
                ft.inv_s[2],
            );
            let r = ft.anchor_t_wc.fixed_view::<3, 3>(0, 0).into_owned();
            r * (j * ft.inv_p * j.transpose()) * r.transpose()
        };
        let c_before = euc(&feat);
        let c_after = euc(feat2);
        assert_relative_eq!(c_before, c_after, epsilon = 1e-9);
    }

    /// `pose_range_direct` must make the ACCUMULATED range variance equal the pose-range
    /// function evaluated at the current separation -- so it inherits the true
    /// relative-pose growth EXPONENT. The legacy increment path cannot: with a gain that
    /// changes as the baseline grows, a sum of gain-weighted increments is not the
    /// gain-weighted total (`G dS` vs `d(G S)`), and no scalar repairs a slope.
    #[test]
    fn pose_range_direct_accumulates_to_the_current_relative_covariance() {
        let r2 = 400.0; // r = 20 m
                        // Relative covariance grows BALLISTICALLY (~k^2), as measured for real VIO.
        let sig = |k: f64| Matrix3::identity() * (1e-4 * k * k);
        // Baseline grows with the window, so the inverse-parallax gain CHANGES.
        let b2 = |k: f64| (0.3 * k).powi(2).max(1e-4 * r2);

        let mut st = settings();
        st.pose_range_direct = true;
        st.pose_range_scale = 1.0; // must be IGNORED in direct mode

        let mut f_direct = feature(1);
        let mut acc_direct = 0.0;
        for k in 1..=20 {
            let kf = k as f64;
            let full = sig(kf);
            let inc = sig(kf) - sig(kf - 1.0);
            acc_direct += pose_range_q(
                &st,
                &mut f_direct,
                r2,
                b2(kf),
                Some(&full),
                Some(&full),
                Some(&inc),
                Some(&inc),
            );
        }
        // Contract: the accumulated total IS f at the final separation.
        let kf = 20.0;
        let s_end = sig(kf);
        let f_end = (r2 / b2(kf)) * (r2 * s_end.trace() / 3.0 + s_end.trace() / 3.0);
        assert!(
            (acc_direct - f_end).abs() / f_end < 1e-9,
            "direct accumulation {acc_direct} should equal f_end {f_end}"
        );

        // The legacy path, fed the same increments, lands somewhere else entirely.
        let mut st_legacy = settings();
        st_legacy.pose_range_direct = false;
        st_legacy.pose_range_scale = 1.0;
        let mut f_leg = feature(2);
        let mut acc_legacy = 0.0;
        for k in 1..=20 {
            let kf = k as f64;
            let full = sig(kf);
            let inc = sig(kf) - sig(kf - 1.0);
            acc_legacy += pose_range_q(
                &st_legacy,
                &mut f_leg,
                r2,
                b2(kf),
                Some(&full),
                Some(&full),
                Some(&inc),
                Some(&inc),
            );
        }
        assert!(
            (acc_legacy - f_end).abs() / f_end > 0.1,
            "legacy {acc_legacy} vs f_end {f_end}: the two paths must differ materially, \
             otherwise this test proves nothing"
        );

        // Direct mode must not respond to pose_range_scale at all.
        let mut st2 = st;
        st2.pose_range_scale = 37.0;
        let mut f2 = feature(3);
        let mut acc2 = 0.0;
        for k in 1..=20 {
            let kf = k as f64;
            let full = sig(kf);
            acc2 += pose_range_q(
                &st2,
                &mut f2,
                r2,
                b2(kf),
                Some(&full),
                Some(&full),
                None,
                None,
            );
        }
        assert!(
            (acc2 - acc_direct).abs() / acc_direct < 1e-12,
            "direct mode must be free of pose_range_scale"
        );
    }

    /// The scale term must (a) be multiplicative in range, (b) reach what the filter
    /// REPORTS, and (c) NOT touch the convergence gates or the filter's own estimate --
    /// a common-mode systematic bias cannot be fixed by making the recursion less
    /// confident, only by reporting it.
    #[test]
    fn scale_sigma_inflates_reported_range_quadratically_and_leaves_the_estimate_alone() {
        use crate::mathematical::camera::{CameraModel, PinholeModel};
        use std::sync::Arc;

        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
        });
        let point_w = Vector3::new(0.5, 0.1, 4.0);

        let run = |sigma_s: f64| {
            let mut st = settings();
            st.scale_sigma = sigma_s;
            st.min_track_length = 3;
            let mut f = Sparse3DFilter::bearing_invdepth3d(k(), cam.clone(), st);
            for i in 0..20 {
                update_with_point(&mut f, i, point_w);
            }
            let (r, var) = f.query_range(42);
            let p = f.feature(42).expect("track").position;
            (r, var, p)
        };

        let (r0, v0, p0) = run(0.0);
        assert!(
            r0 > 0.0,
            "baseline must converge for the test to mean anything"
        );
        let s = 0.10;
        let (r1, v1, p1) = run(s);

        // (c) the ESTIMATE is untouched -- reporting-only.
        assert!(
            (r1 - r0).abs() < 1e-12,
            "range estimate changed: {r0} vs {r1}"
        );
        assert!((p1 - p0).norm() < 1e-12, "position estimate changed");

        // (a)+(b) reported variance grows by exactly (sigma_s * r)^2.
        let expected = v0 + (s * r0).powi(2);
        assert!(
            (v1 - expected).abs() / expected < 1e-12,
            "reported var {v1} != baseline {v0} + (s*r)^2 {expected}"
        );
        // Multiplicative: doubling sigma_s quadruples the ADDED part.
        let (_, v2, _) = run(2.0 * s);
        let added1 = v1 - v0;
        let added2 = v2 - v0;
        assert!(
            (added2 / added1 - 4.0).abs() < 1e-9,
            "added variance should scale as sigma_s^2; got ratio {}",
            added2 / added1
        );
    }

    /// The fitted envelope, pinned at the anchors the fit was made on.
    #[test]
    fn sigma_s_envelope_matches_the_fitted_map_and_floors_at_two_percent() {
        // Values computed directly from the fitted map. (The recorded per-trajectory
        // numbers 0.396 / 0.131 / 0.030 / 0.039 are sigma_s OUTPUTS, not E inputs --
        // they correspond to E = 0.0035 / 0.0168 / 0.134 / 0.0926.)
        assert!((sigma_s_from_excitation(0.01) - 0.18938).abs() < 1e-4);
        assert!((sigma_s_from_excitation(0.1) - 0.036926).abs() < 1e-5);
        // A low-excitation trajectory must be told its scale is badly known.
        assert!(sigma_s_from_excitation(0.0035) > 0.35);
        // Monotone decreasing in E.
        assert!(sigma_s_from_excitation(0.0001) > sigma_s_from_excitation(0.01));
        assert!(sigma_s_from_excitation(0.01) > sigma_s_from_excitation(0.04));
        // Floor: no monocular system knows scale better than 2%.
        assert!((sigma_s_from_excitation(1e6) - 0.02).abs() < 1e-12);
        // Degenerate input falls back to the floor rather than exploding.
        assert!((sigma_s_from_excitation(0.0) - 0.02).abs() < 1e-12);
        assert!((sigma_s_from_excitation(f64::NAN) - 0.02).abs() < 1e-12);
    }

    /// Birth refuses a range outside [min_depth, max_depth]; REPORTING must refuse it
    /// too. A landmark born just inside the cap can drift far beyond it, and the far
    /// field is where the covariance model is least trustworthy -- so reporting a range
    /// the filter would not have birthed is an inconsistency, not extra capability.
    #[test]
    fn query_range_refuses_what_birth_would_have_refused() {
        use crate::mathematical::camera::{CameraModel, PinholeModel};
        use std::sync::Arc;

        let cam: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
        });
        let point_w = Vector3::new(0.5, 0.1, 4.0);
        let mut st = settings();
        st.min_track_length = 3;
        st.max_depth = 100.0;
        let mut f = Sparse3DFilter::bearing_invdepth3d(k(), cam, st);
        for i in 0..20 {
            update_with_point(&mut f, i, point_w);
        }
        let (r, _) = f.query_range(42);
        assert!(
            r > 0.0,
            "a ~4 m landmark must be reportable under a 100 m cap"
        );

        // Drive the stored range past the cap and confirm it is withheld.
        let slot = *f.feature_slots.get(&42).expect("slot");
        let p = f.features[slot].position;
        f.features[slot].position = p * (200.0 / p.norm()); // 200 m, beyond max_depth
        let (r2, v2) = f.query_range(42);
        assert!(
            r2 < 0.0 && !v2.is_finite(),
            "range beyond max_depth must report unusable, got ({r2}, {v2})"
        );

        // ... and below min_depth likewise.
        f.features[slot].position = p * (0.01 / p.norm());
        let (r3, _) = f.query_range(42);
        assert!(
            r3 < 0.0,
            "range below min_depth must report unusable, got {r3}"
        );
    }

    /// The convergence gate must be SCALE-FREE. The absolute form fails two ways that
    /// the relative form does not: it rejects a perfectly-conditioned far landmark, and
    /// it has to be re-tuned whenever the covariance model changes.
    #[test]
    fn relative_convergence_gate_is_scale_free_where_the_absolute_one_is_not() {
        let mut abs_st = settings();
        abs_st.conv_variance_threshold = 2.0; // the shipped value, in m^2
        abs_st.conv_range_sigma_frac = 0.0; // legacy path
        let mut rel_st = settings();
        rel_st.conv_range_sigma_frac = 0.5; // 2-sigma band reaches infinity

        // A landmark known to 10% relative. Equally well conditioned at every range.
        for r in [5.0_f64, 50.0, 500.0] {
            let var = (0.10 * r).powi(2);
            assert!(
                !unconverged(&rel_st, r, var),
                "relative gate must accept a 10%-known landmark at r = {r}"
            );
        }
        // The absolute gate accepts it up close and REJECTS it far away -- same quality.
        assert!(!unconverged(&abs_st, 5.0, (0.10 * 5.0f64).powi(2)));
        assert!(
            unconverged(&abs_st, 500.0, (0.10 * 500.0f64).powi(2)),
            "this is the defect: absolute gate rejects a 10%-known landmark at 500 m"
        );

        // The relative gate still rejects genuinely useless depth (2-sigma to infinity).
        assert!(unconverged(&rel_st, 50.0, (0.60 * 50.0f64).powi(2)));
        assert!(!unconverged(&rel_st, 50.0, (0.40 * 50.0f64).powi(2)));

        // Honest-covariance regression: sigma_r 3-7 m at ~50 m range is 6-14% relative,
        // which the shipped absolute 2.0 m^2 rejects outright -- the bug that silently
        // emptied query_range.
        let honest_var = 5.0_f64.powi(2);
        assert!(
            unconverged(&abs_st, 50.0, honest_var),
            "pins the old failure"
        );
        assert!(
            !unconverged(&rel_st, 50.0, honest_var),
            "relative form accepts it"
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
            anchor_ray: Vector3::zeros(),
            tangent_basis: Matrix3x2::zeros(),
            anchor_t_wc: Matrix4::identity(),
            a: 1.0,
            b: 1.0,
            track_length: 3,
            ref_uv: Vector2::new(0.0, 0.0),
            ref_stamp: 0.0,
            last_nis: f64::NAN,
            bias: Vector2::zeros(),
            p_sb: Matrix3x2::zeros(),
            p_bb: Matrix2::zeros(),
            pose_q_prev: 0.0,
            p_stheta: Matrix3x6::zeros(),
            anchor_clone_id: None,
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
