//! The dense depth mapper the node runs (design note §7 item 2): the FlowDep-F patch
//! tracker on an equalised, undistorted, downscaled copy of the frame; its own Sparse3D
//! (pinhole bearing chart, walk variance for patch tracks) fed clone-relative like the
//! corner-track Sparse3D; the nearest+disagreement+footprint fusion and the balance-gated
//! fill of [`crate::depth::dense`]; and a per-pixel CLASS the occupancy map reads.
//!
//! What it deliberately does not do: it does not touch the pose filter. Corners stay the
//! pose filter's only features (decision of 2026-10-18); the patch tracker runs after the
//! update on the posterior pose, and it is told the corner tracker's live features so it
//! does not birth patches on them (those already feed the corner-track Sparse3D). The
//! pose covariance the patch landmarks consume is the same clone-relative one, through
//! the same `CloneRelFeed`, so the two depth filters are honest about the same pose in the
//! same way.
//!
//! Frame geometry: the tracker needs a pinhole frame, so each frame is remapped once from
//! the rig's camera model (any projection) to a pinhole at `scale` of the raw size with the
//! raw intrinsics scaled, which crops a fisheye to its central region. Everything the
//! mapper outputs is in that pinhole frame, with [`DenseDepthMapper::intrinsics`].

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::{Matrix3, Matrix4, Vector2};
use rudolf_v::histeq::{apply_histeq, HistEqMethod};
use rudolf_v::image::Image;

use crate::backend::{CameraPoseCovariance, CloneRelFeed};
use crate::core_types::{CameraIntrinsics, DepthMap};
use crate::depth::dense::{fill, fuse, DenseItem, FillParams};
use crate::depth::flowdep_f::{FlowDepF, FlowDepFSettings};
use crate::depth::patch_depth::{PatchDepthOutput, PatchStatus};
use crate::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use crate::depth::sparse_gb::SparseVogSettings;
use crate::mathematical::camera::{CameraModel, PinholeModel};
use crate::mathematical::vision_measurement::VisionMeasurement;

/// Per-pixel class of the dense frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DenseClass {
    /// No evidence; the pixel says nothing.
    Unknown = 0,
    /// Inside the footprint of a converged patch: that patch's range and σ.
    Measured = 1,
    /// Interpolated between measured patches that surround the pixel; free-space
    /// carving only at the map.
    Filled = 2,
}

/// The dense frame, row-major `height × width`, in the mapper's pinhole frame.
#[derive(Clone, Debug)]
pub struct DenseOutput {
    pub width: usize,
    pub height: usize,
    /// Log-range `ln(range)`; valid where `class != Unknown`.
    pub eta: Vec<f32>,
    /// Variance of `eta` ≈ relative range variance.
    pub eta_var: Vec<f32>,
    pub class: Vec<DenseClass>,
    pub stats: DenseStats,
}

impl DenseOutput {
    /// The frame in the patch mapper's container, for the existing publishers:
    /// `Measured → PhotoRefined`, `Filled → SeedOnly`, `Unknown → Unknown` (η NaN).
    pub fn to_patch_depth_output(&self) -> PatchDepthOutput {
        let n = self.width * self.height;
        let mut eta = Vec::with_capacity(n);
        let mut var = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        for i in 0..n {
            match self.class[i] {
                DenseClass::Unknown => {
                    eta.push(f32::NAN);
                    var.push(f32::NAN);
                    status.push(PatchStatus::Unknown);
                }
                DenseClass::Measured => {
                    eta.push(self.eta[i]);
                    var.push(self.eta_var[i]);
                    status.push(PatchStatus::PhotoRefined);
                }
                DenseClass::Filled => {
                    eta.push(self.eta[i]);
                    var.push(self.eta_var[i]);
                    status.push(PatchStatus::SeedOnly);
                }
            }
        }
        PatchDepthOutput {
            eta: DepthMap::from_vec(self.width, self.height, eta).unwrap(),
            eta_var: DepthMap::from_vec(self.width, self.height, var).unwrap(),
            status: DepthMap::from_vec(self.width, self.height, status).unwrap(),
        }
    }
}

/// Per-frame counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseStats {
    pub patches_live: usize,
    pub patches_observed: usize,
    pub landmarks_converged: usize,
    pub items_fused: usize,
    pub measured_frac: f32,
    pub filled_frac: f32,
    pub ms_equalise_remap: f32,
    pub ms_track: f32,
    pub ms_filter: f32,
    pub ms_fuse_fill: f32,
}

#[derive(Clone, Debug)]
pub struct DenseMapperSettings {
    /// Working scale of the pinhole frame relative to the raw frame.
    pub scale: f64,
    /// Equalisation of the raw frame before remapping (the corner tracker gets the same
    /// treatment inside Rudolf-V; row 5m measured raw input at a quarter of the births).
    pub histeq: HistEqMethod,
    pub tracker: FlowDepFSettings,
    /// Settings of the patch-track Sparse3D; the chart is always the bearing inverse-depth
    /// one. Copy the corner-track section and override the walk variance (0.3 for patches).
    pub sparse: SparseVogSettings,
    /// Footprint slant prior (degrees) of the fusion rule.
    pub slant_deg: f64,
    /// The fill, or `None` for Measured/Unknown only.
    pub fill: Option<FillParams>,
    /// Keep patch births away from the corner tracker's live features.
    pub exclude_corners: bool,
}

pub struct DenseDepthMapper {
    s: DenseMapperSettings,
    camera: Arc<dyn CameraModel>,
    raw_w: usize,
    raw_h: usize,
    w: usize,
    h: usize,
    intr: CameraIntrinsics,
    /// Raw-frame sample position of every pinhole pixel (NaN when outside the raw frame).
    remap: Vec<[f32; 2]>,
    und: Vec<u8>,
    tracker: FlowDepF,
    sparse: Sparse3DFilter,
    feed: CloneRelFeed,
    slant_g: f64,
    last_items: Vec<DenseItem>,
}

impl DenseDepthMapper {
    pub fn new(
        camera: Arc<dyn CameraModel>,
        raw: CameraIntrinsics,
        raw_w: usize,
        raw_h: usize,
        s: DenseMapperSettings,
    ) -> Self {
        let w = ((raw_w as f64) * s.scale).round().max(1.0) as usize;
        let h = ((raw_h as f64) * s.scale).round().max(1.0) as usize;
        let intr = CameraIntrinsics {
            fx: raw.fx * s.scale,
            fy: raw.fy * s.scale,
            cx: raw.cx * s.scale,
            cy: raw.cy * s.scale,
        };
        let mut remap = vec![[f32::NAN; 2]; w * h];
        for y in 0..h {
            for x in 0..w {
                let b = intr.normalize_pixel(&Vector2::new(x as f64, y as f64));
                let uv = camera.project(&b);
                if uv[0].is_finite()
                    && uv[1].is_finite()
                    && uv[0] >= 0.0
                    && uv[1] >= 0.0
                    && uv[0] <= (raw_w - 1) as f64
                    && uv[1] <= (raw_h - 1) as f64
                {
                    remap[y * w + x] = [uv[0] as f32, uv[1] as f32];
                }
            }
        }
        let tracker = FlowDepF::new(intr.fx, intr.fy, intr.cx, intr.cy, w, h, s.tracker.clone());
        let k = Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0);
        let pin: Arc<dyn CameraModel> = Arc::new(PinholeModel {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
        });
        let sparse =
            Sparse3DFilter::new(k, Sparse3DChart::BearingInvDepthAdditive, s.sparse.clone())
                .with_camera(pin);
        let slant_g = s.slant_deg.to_radians().tan() / intr.fx;
        // The fill's variance gradient defaults to twice the footprint gradient (rows 5k/5l).
        let mut s = s;
        if let Some(f) = s.fill.as_mut() {
            if f.g <= 0.0 {
                f.g = 2.0 * slant_g;
            }
        }
        Self {
            s,
            camera,
            raw_w,
            raw_h,
            w,
            h,
            intr,
            remap,
            und: vec![0u8; w * h],
            tracker,
            sparse,
            feed: CloneRelFeed::default(),
            slant_g,
            last_items: Vec::new(),
        }
    }

    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intr
    }

    /// The equalised, undistorted, downscaled frame the tracker last saw (`width × height`).
    pub fn last_frame(&self) -> &[u8] {
        &self.und
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// The converged patch landmarks of the last frame, as the fusion saw them (pinhole
    /// pixel, level, log-range, conditional variance, shared sigma).
    pub fn last_items(&self) -> &[DenseItem] {
        &self.last_items
    }

    /// Raw-frame pixel → pinhole-frame pixel at the working scale, `None` if behind the
    /// camera.
    pub fn raw_to_pinhole(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let b = self.camera.undistort(&Vector2::new(u, v));
        if !(b[2] > 1e-6) {
            return None;
        }
        Some((
            self.intr.fx * b[0] / b[2] + self.intr.cx,
            self.intr.fy * b[1] / b[2] + self.intr.cy,
        ))
    }

    /// One frame. `gray_raw` is the raw frame (`raw_w × raw_h`), `t_wc` the posterior camera
    /// pose of this frame, `pose_cov` the backend's pose covariance (clone-relative under
    /// MSCEqF), `corners_raw` the corner tracker's live features in raw pixels.
    pub fn process(
        &mut self,
        gray_raw: &[u8],
        stamp: f64,
        t_wc: &Matrix4<f64>,
        pose_cov: Option<&dyn CameraPoseCovariance>,
        corners_raw: &[(f64, f64)],
    ) -> DenseOutput {
        let t0 = std::time::Instant::now();
        // 1. Equalise and remap to the pinhole frame (bilinear).
        let img = Image::from_vec(self.raw_w, self.raw_h, gray_raw.to_vec());
        let eq = apply_histeq(&img, self.s.histeq.clone());
        let src = eq.as_slice();
        let (rw, rh) = (self.raw_w, self.raw_h);
        for i in 0..self.w * self.h {
            let [u, v] = self.remap[i];
            if !u.is_finite() {
                self.und[i] = 0;
                continue;
            }
            let x0 = (u.floor() as usize).min(rw - 2);
            let y0 = (v.floor() as usize).min(rh - 2);
            let fx = (u - x0 as f32).clamp(0.0, 1.0);
            let fy = (v - y0 as f32).clamp(0.0, 1.0);
            let p00 = src[y0 * rw + x0] as f32;
            let p01 = src[y0 * rw + x0 + 1] as f32;
            let p10 = src[(y0 + 1) * rw + x0] as f32;
            let p11 = src[(y0 + 1) * rw + x0 + 1] as f32;
            let val =
                (1.0 - fy) * ((1.0 - fx) * p00 + fx * p01) + fy * ((1.0 - fx) * p10 + fx * p11);
            self.und[i] = (val + 0.5) as u8;
        }
        let ms_equalise_remap = t0.elapsed().as_secs_f32() * 1e3;

        // 2. Track, with the corners excluded from births.
        let t1 = std::time::Instant::now();
        if self.s.exclude_corners {
            let ex: Vec<(f64, f64)> = corners_raw
                .iter()
                .filter_map(|&(u, v)| self.raw_to_pinhole(u, v))
                .collect();
            self.tracker.set_birth_exclusions(ex);
        }
        let obs = self.tracker.process(&self.und, t_wc);
        let ms_track = t1.elapsed().as_secs_f32() * 1e3;

        // 3. The patch-track depth filter, clone-relative.
        let t2 = std::time::Instant::now();
        let uvs: HashMap<u64, Vector2<f32>> = obs
            .iter()
            .map(|o| (o.id, Vector2::new(o.u as f32, o.v as f32)))
            .collect();
        let meas = VisionMeasurement::new(stamp, uvs);
        self.feed.feed(pose_cov, &mut self.sparse, &meas, t_wc);

        // 4. Posterior back to the tracker's search bound; converged landmarks → items.
        let lev: HashMap<u64, (f64, f64, usize)> =
            obs.iter().map(|o| (o.id, (o.u, o.v, o.level))).collect();
        let chart = self.sparse.chart();
        let mut post: HashMap<u64, (f64, f64)> = HashMap::new();
        let mut items: Vec<DenseItem> = Vec::new();
        let mut n_conv = 0usize;
        for feat in self.sparse.features_iter() {
            let r = feat.position.norm();
            if !(r.is_finite() && r > 1e-6) {
                continue;
            }
            let c = feat.covariance_euclidean(chart);
            if !c.iter().all(|v| v.is_finite()) {
                continue;
            }
            let rh = feat.position / r;
            let vc = (rh.transpose() * c * rh)[(0, 0)] / (r * r);
            if !(vc > 0.0) {
                continue;
            }
            post.insert(feat.feat_id, (r, vc.sqrt() * r));
            if self.sparse.query_range(feat.feat_id).0 <= 0.0 {
                continue; // not converged (or below the track-length gate)
            }
            n_conv += 1;
            if let Some(&(u, v, level)) = lev.get(&feat.feat_id) {
                let vr = self.sparse.reported_range_variance(feat) / (r * r);
                items.push(DenseItem {
                    px: u,
                    py: v,
                    level: level as u32,
                    eta: r.ln(),
                    var_cond: vc,
                    sigma_shared: (vr - vc).max(0.0).sqrt(),
                });
            }
        }
        self.tracker.set_posterior(&post);
        let ms_filter = t2.elapsed().as_secs_f32() * 1e3;

        // 5. Fuse, fill, classify.
        let t3 = std::time::Instant::now();
        let (w, h) = (self.w, self.h);
        let fused = fuse(&items, w, h, self.slant_g);
        let n = w * h;
        let mut eta = vec![0.0f32; n];
        let mut var = vec![0.0f32; n];
        let mut class = vec![DenseClass::Unknown; n];
        let mut n_meas = 0usize;
        for i in 0..n {
            if fused.covered[i] {
                eta[i] = fused.eta[i] as f32;
                var[i] = fused.var[i] as f32;
                class[i] = DenseClass::Measured;
                n_meas += 1;
            }
        }
        let mut n_fill = 0usize;
        if let Some(p) = &self.s.fill {
            if n_meas > 0 {
                let f = fill(&fused, p);
                for i in 0..n {
                    if f.filled[i] {
                        eta[i] = f.eta[i] as f32;
                        var[i] = f.var[i] as f32;
                        class[i] = DenseClass::Filled;
                        n_fill += 1;
                    }
                }
            }
        }
        let ms_fuse_fill = t3.elapsed().as_secs_f32() * 1e3;
        let stats = DenseStats {
            patches_live: self.tracker.n_live(),
            patches_observed: obs.len(),
            landmarks_converged: n_conv,
            items_fused: items.len(),
            measured_frac: n_meas as f32 / n as f32,
            filled_frac: n_fill as f32 / n as f32,
            ms_equalise_remap,
            ms_track,
            ms_filter,
            ms_fuse_fill,
        };
        self.last_items = items;
        DenseOutput {
            width: w,
            height: h,
            eta,
            eta_var: var,
            class,
            stats,
        }
    }
}
