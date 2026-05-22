use std::collections::{HashMap, HashSet};

use nalgebra::{Matrix2, Matrix2x3, Matrix3, Matrix4, Vector2, Vector3};

use crate::coordinate_suite::invdepth::{conv_euc2ind, conv_ind2euc, point_chart_invdepth_inv};
use crate::coordinate_suite::normal::{conv_euc2normal, conv_normal2euc, point_chart_normal_inv};
use crate::depth::sparse_gb::SparseVogSettings;
use crate::mathematical::vision_measurement::VisionMeasurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sparse3DChart {
    Polar,
    InvDepth,
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
    pub position: Vector3<f64>,
    pub covariance: Matrix3<f64>,
    pub a: f64,
    pub b: f64,
    pub track_length: usize,
    pub ref_t_wc: Matrix4<f64>,
    pub ref_uv: Vector2<f64>,
    pub ref_stamp: f64,
}

impl FeatureState3D {
    pub fn depth_for_chart(&self, chart: Sparse3DChart) -> f64 {
        match chart {
            Sparse3DChart::Polar => self.position.norm(),
            Sparse3DChart::InvDepth => self.position[2],
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
        }
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

        let Some(prev_t_wc) = self.prev_t_wc else {
            self.prev_t_wc = Some(*t_wc);
            self.prev_stamp = stamp;
            self.prev_uvs = curr_uvs;
            return;
        };

        let dt = (stamp - self.prev_stamp).max(0.0);
        let t_cw_curr = t_wc.try_inverse().unwrap_or_else(Matrix4::identity);
        let t_curr_prev = t_cw_curr * prev_t_wc;
        let r = t_curr_prev.fixed_view::<3, 3>(0, 0).into_owned();
        let t = t_curr_prev.fixed_view::<3, 1>(0, 3).into_owned();

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
                    (!update_existing_feature_3d(
                        chart, &k, &settings, feat, uv_curr, &r, &t, p_vv, dt,
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

                if !update_existing_feature_3d(
                    self.chart,
                    &self.k,
                    &self.settings,
                    feat,
                    uv_curr,
                    &r,
                    &t,
                    p_vv,
                    dt,
                ) {
                    reset_features.push(fid);
                }
            }
            reset_features
        };

        let reset_set: HashSet<_> = reset_features.iter().copied().collect();
        for (&fid, &uv_curr) in &curr_uvs {
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
            let mut feat = FeatureState3D {
                feat_id: fid,
                position,
                covariance,
                a: self.settings.a_init,
                b: self.settings.b_init,
                track_length: 1,
                ref_t_wc: *t_wc,
                ref_uv: uv_curr,
                ref_stamp: stamp,
            };
            bearing_update_3d(self.chart, &self.k, &self.settings, &mut feat, &uv_curr);
            self.insert_feature(feat);
            self.pending.remove(&fid);
        }

        for &fid in &reset_features {
            self.pending.remove(&fid);
        }

        self.remove_features(|feat| {
            reset_set.contains(&feat.feat_id) || !curr_uvs.contains_key(&feat.feat_id)
        });
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

    pub fn feature(&self, fid: u64) -> Option<&FeatureState3D> {
        let &slot = self.feature_slots.get(&fid)?;
        self.features.get(slot)
    }

    pub fn features_iter(&self) -> impl Iterator<Item = &FeatureState3D> {
        self.features.iter()
    }

    pub fn feature_count(&self) -> usize {
        self.features.len()
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
    }
}

fn euc_to_chart_jac(chart: Sparse3DChart, q: &Vector3<f64>) -> Matrix3<f64> {
    match chart {
        Sparse3DChart::Polar => conv_euc2normal(q),
        Sparse3DChart::InvDepth => conv_euc2ind(q),
    }
}

fn apply_chart_delta(chart: Sparse3DChart, q: &Vector3<f64>, delta: &Vector3<f64>) -> Vector3<f64> {
    match chart {
        Sparse3DChart::Polar => point_chart_normal_inv(delta, q),
        Sparse3DChart::InvDepth => point_chart_invdepth_inv(delta, q),
    }
}

fn update_existing_feature_3d(
    chart: Sparse3DChart,
    k: &Matrix3<f64>,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    uv_curr: &Vector2<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    p_vv: Option<&Matrix3<f64>>,
    dt: f64,
) -> bool {
    predict_feature_3d(chart, settings, feat, r, t, p_vv, dt);
    if !bearing_update_3d(chart, k, settings, feat, uv_curr) {
        return false;
    }
    feat.track_length += 1;
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

fn predict_feature_3d(
    chart: Sparse3DChart,
    settings: &SparseVogSettings,
    feat: &mut FeatureState3D,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    p_vv: Option<&Matrix3<f64>>,
    dt: f64,
) {
    let depth = feat.depth_for_chart(chart);
    if depth < settings.min_depth {
        feat.position = Vector3::zeros();
        return;
    }
    let q_old = feat.position;
    let q_new = r * q_old + t;
    if q_new[2] < settings.min_depth {
        feat.position = Vector3::zeros();
        return;
    }

    let j = euc_to_chart_jac(chart, &q_new) * r * chart_to_euc_jac(chart, &q_old);
    let mut cov_new = j * feat.covariance * j.transpose();
    let q_euc = p_vv
        .map(|p| p * (dt * dt))
        .unwrap_or_else(|| Matrix3::identity() * settings.process_depth_var * dt.max(1e-3));
    let q_chart =
        euc_to_chart_jac(chart, &q_new) * q_euc * euc_to_chart_jac(chart, &q_new).transpose();
    cov_new += q_chart;
    feat.position = q_new;
    feat.covariance = 0.5 * (cov_new + cov_new.transpose());
}

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
    let h_euc = Matrix2x3::new(
        fx / q[2],
        0.0,
        -fx * q[0] / (q[2] * q[2]),
        0.0,
        fy / q[2],
        -fy * q[1] / (q[2] * q[2]),
    );
    let h = h_euc * chart_to_euc_jac(chart, &q);
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
        );
    }

    fn feature(fid: u64) -> FeatureState3D {
        FeatureState3D {
            feat_id: fid,
            position: Vector3::new(0.0, 0.0, 3.0),
            covariance: Matrix3::identity(),
            a: 1.0,
            b: 1.0,
            track_length: 3,
            ref_t_wc: Matrix4::identity(),
            ref_uv: Vector2::new(0.0, 0.0),
            ref_stamp: 0.0,
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
        filter.update(&VisionMeasurement::new(8.0 * 0.05, coords), &t_wc, None);

        assert!(
            filter.feature(42).is_none(),
            "large Mahalanobis bearing innovation should immediately remove the feature"
        );
    }
}
