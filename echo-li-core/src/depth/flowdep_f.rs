//! FlowDep-F: a semi-dense, anchor-referenced, known-pose patch track source.
//!
//! Every live patch is a landmark candidate for `Sparse3DFilter`: it is born at a pixel of an
//! ANCHOR frame (a frame whose image and camera pose are kept), carries a log-range
//! `η = ln(range)` estimate with a search bound, and on every new frame is matched along the
//! epipolar segment that the bound projects to, with the camera pose of both frames known.
//! The match is a discrete scan along the segment followed by a one-parameter Gauss–Newton
//! refinement on `η`; the matched pixel is emitted as an ordinary track observation
//! (`id → (u, v)`), so the filter downstream does not know whether a track came from a
//! corner tracker or from here.
//!
//! Why this and not frame-to-frame KLT: a 2-D tracker needs texture in two directions, so it
//! lives on corners (2–11 % of 8×8 cells on the rig's indoor frames); with the pose known the
//! match is one-dimensional and any edge with curvature along the epipolar line is usable
//! (11–25 % of cells, twice that one pyramid level down). The reference is the anchor image,
//! fixed for the patch's lifetime, so no template drift accumulates; the appearance change
//! between anchor and current view (rotation, zoom, parallax) is removed by warping each
//! reference pixel through the known relative pose at the current range estimate (every
//! pixel of the patch placed on the sphere of radius `exp(η)` around the anchor camera —
//! exact for a fronto-parallel surface, first order otherwise). A pose-free reference
//! template fails exactly because that warp is missing (zoom/parallax at the coarse level;
//! see the July 2026 template-drift study), so the warp is not optional here.
//!
//! What is deliberately NOT here: any temporal filtering of `η`. The tracker keeps a
//! per-patch estimate and bound only to size the next search segment; the filtered depth and
//! its covariance come from `Sparse3DFilter`, which can hand its posterior back through
//! [`FlowDepF::set_posterior`] to replace the tracker's own bound.
//!
//! Conventions: pinhole camera at level 0 with (fx, fy, cx, cy); pyramid level `l` halves
//! resolution and intrinsics; `t_wc` is the camera-to-world transform of the frame; pixel
//! coordinates in observations are level-0 pixels; `η = ln(range)` with the 3-D point
//! `exp(η) · unit_bearing(u, v)` in the anchor camera frame.

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use std::collections::HashMap;

/// Tunables. Defaults are the values used for the TartanAir step-3 gate.
#[derive(Debug, Clone)]
pub struct FlowDepFSettings {
    /// Pyramid levels used for births (level 0 = full resolution).
    pub levels: usize,
    /// Patch half size; the patch is `(2·half) × (2·half)` pixels at its own level.
    pub patch_half: usize,
    /// Birth grid cell size in pixels at each level: one candidate per cell at most.
    pub cell: usize,
    /// Minimum squared image gradient (grey²/px², per pixel, averaged over the patch)
    /// along the hint direction for a birth; LSD-SLAM's `minUseGrad` is 5 grey/px → 25.
    pub min_birth_curvature: f32,
    /// Minimum along-line curvature (sum over the patch of (∇I·d)² at the current frame)
    /// for an observation to count; below it the match is uninformative and is skipped.
    pub min_track_curvature: f32,
    /// Range prior at birth: uniform on [d_min, d_max] metres (η bound ±2σ covers it).
    pub d_min: f64,
    pub d_max: f64,
    /// Scan spacing along the projected segment, pixels at the patch's level.
    pub scan_step_px: f64,
    /// Cap on scan samples per patch per frame (long segments at birth are subsampled).
    pub max_scan_samples: usize,
    /// Search bound half-width in σ (LSD-SLAM uses 2).
    pub bound_sigma: f64,
    /// Floor on the η bound after an accepted match, and the per-frame growth of the bound
    /// (a crude process model while the filter's posterior is not fed back).
    pub eta_sigma_floor: f64,
    pub eta_sigma_growth: f64,
    /// Photometric noise std (grey levels) used for the along-line variance.
    pub sigma_photo: f64,
    /// Gauss–Newton iterations and convergence tolerance on |Δη|.
    pub gn_max_iters: usize,
    pub gn_tol: f64,
    /// Accept a match only if the mean absolute zero-mean residual (grey levels) is below
    /// this and at least this fraction of the patch pixels landed inside the image.
    pub max_mean_abs_residual: f32,
    pub min_valid_frac: f32,
    /// Reject a match when a second local minimum on the segment is within this ratio of
    /// the best cost (ambiguous epipolar line).
    pub ambiguity_ratio: f32,
    /// Drop a patch after this many consecutive frames without an accepted observation.
    pub max_misses: u32,
    /// Make the current frame a new anchor every this many frames, or sooner when the live
    /// patch count drops below `min_live_frac · max_patches`.
    pub keyframe_every: u32,
    pub min_live_frac: f32,
    pub max_patches: usize,
    /// No birth within this many level-0 pixels of a live patch's projection.
    pub min_birth_dist_px: f64,
    /// Re-anchor a patch when the baseline between its anchor and the current frame
    /// exceeds this fraction of its range (measured: the sphere-warp match degrades from
    /// 0.2–0.3 px at b/r < 0.1 to 0.6–0.8 px at 0.2–0.3 and is gross beyond 0.5). The
    /// patch moves to the newest keyframe whose b/r is below half of this, else the
    /// current frame becomes a keyframe. 0 disables re-anchoring.
    pub reanchor_br: f64,
    /// Withhold (do not emit) an observation whose anchor→current baseline exceeds this
    /// fraction of the matched range; the patch state still updates and re-anchors. The
    /// re-anchor rule fires only after a match, so a near patch can cross `reanchor_br` by
    /// a wide margin in one frame (measured on P000: 4 % of observations sit at b/r > 0.2
    /// and half to all of them are off the GT line by > 1 px; withholding them at 0.2 cut
    /// the converged-landmark NEES tail 9.7 → 6.8 % at equal median error). 0 disables.
    pub max_obs_br: f64,
}

impl Default for FlowDepFSettings {
    fn default() -> Self {
        Self {
            levels: 2,
            patch_half: 4,
            cell: 8,
            min_birth_curvature: 25.0,
            min_track_curvature: 25.0 * 64.0 * 0.25,
            d_min: 0.5,
            d_max: 150.0,
            scan_step_px: 1.0,
            max_scan_samples: 64,
            bound_sigma: 2.0,
            eta_sigma_floor: 0.01,
            eta_sigma_growth: 0.02,
            sigma_photo: 8.0,
            gn_max_iters: 10,
            gn_tol: 1e-4,
            max_mean_abs_residual: 20.0,
            min_valid_frac: 0.8,
            ambiguity_ratio: 0.8,
            max_misses: 5,
            keyframe_every: 10,
            min_live_frac: 0.5,
            max_patches: 3000,
            min_birth_dist_px: 4.0,
            reanchor_br: 0.15,
            max_obs_br: 0.2,
        }
    }
}

/// One pyramid level: image and its x/y gradients (central differences, grey/px).
struct Level {
    w: usize,
    h: usize,
    img: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl Level {
    fn from_u8(gray: &[u8], w: usize, h: usize) -> Self {
        let img: Vec<f32> = gray.iter().map(|&p| p as f32).collect();
        Self::with_gradients(img, w, h)
    }

    fn downsample(&self) -> Self {
        let w = self.w / 2;
        let h = self.h / 2;
        let mut img = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = 2 * y * self.w + 2 * x;
                img[y * w + x] = 0.25
                    * (self.img[i]
                        + self.img[i + 1]
                        + self.img[i + self.w]
                        + self.img[i + self.w + 1]);
            }
        }
        Self::with_gradients(img, w, h)
    }

    fn with_gradients(img: Vec<f32>, w: usize, h: usize) -> Self {
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                gx[i] = 0.5 * (img[i + 1] - img[i - 1]);
                gy[i] = 0.5 * (img[i + w] - img[i - w]);
            }
        }
        Self { w, h, img, gx, gy }
    }

    #[inline]
    fn inside(&self, u: f64, v: f64) -> bool {
        u >= 1.0 && v >= 1.0 && u < (self.w - 2) as f64 && v < (self.h - 2) as f64
    }

    /// Bilinear sample of the image only; caller guarantees `inside`.
    #[inline]
    fn sample_img(&self, u: f64, v: f64) -> f32 {
        let x0 = u.floor() as usize;
        let y0 = v.floor() as usize;
        let a = (u - x0 as f64) as f32;
        let b = (v - y0 as f64) as f32;
        let i = y0 * self.w + x0;
        (1.0 - a) * (1.0 - b) * self.img[i]
            + a * (1.0 - b) * self.img[i + 1]
            + (1.0 - a) * b * self.img[i + self.w]
            + a * b * self.img[i + self.w + 1]
    }

    /// Bilinear sample of image and gradients at (u, v); caller guarantees `inside`.
    #[inline]
    fn sample(&self, u: f64, v: f64) -> (f32, f32, f32) {
        let x0 = u.floor() as usize;
        let y0 = v.floor() as usize;
        let a = (u - x0 as f64) as f32;
        let b = (v - y0 as f64) as f32;
        let i = y0 * self.w + x0;
        let w00 = (1.0 - a) * (1.0 - b);
        let w10 = a * (1.0 - b);
        let w01 = (1.0 - a) * b;
        let w11 = a * b;
        let lerp =
            |d: &[f32]| w00 * d[i] + w10 * d[i + 1] + w01 * d[i + self.w] + w11 * d[i + self.w + 1];
        (lerp(&self.img), lerp(&self.gx), lerp(&self.gy))
    }
}

struct Pyramid {
    levels: Vec<Level>,
}

impl Pyramid {
    /// Copy with the images only (gradients dropped): what an anchor needs for a re-cut.
    fn clone_images(&self) -> Pyramid {
        Pyramid {
            levels: self
                .levels
                .iter()
                .map(|l| Level {
                    w: l.w,
                    h: l.h,
                    img: l.img.clone(),
                    gx: Vec::new(),
                    gy: Vec::new(),
                })
                .collect(),
        }
    }
}

impl Pyramid {
    fn build(gray: &[u8], w: usize, h: usize, n: usize) -> Self {
        let mut levels = vec![Level::from_u8(gray, w, h)];
        for _ in 1..n {
            let next = levels.last().unwrap().downsample();
            levels.push(next);
        }
        Self { levels }
    }
}

/// Pinhole intrinsics at one pyramid level.
#[derive(Clone, Copy, Debug)]
struct Intr {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl Intr {
    fn at_level(&self, l: usize) -> Intr {
        let s = 1.0 / (1u32 << l) as f64;
        // Pixel-centre convention: level-l pixel u_l = (u_0 + 0.5) · s − 0.5.
        Intr {
            fx: self.fx * s,
            fy: self.fy * s,
            cx: (self.cx + 0.5) * s - 0.5,
            cy: (self.cy + 0.5) * s - 0.5,
        }
    }
    #[inline]
    fn unit_bearing(&self, u: f64, v: f64) -> Vector3<f64> {
        Vector3::new((u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0).normalize()
    }
    #[inline]
    fn project(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p[2] <= 1e-6 {
            return None;
        }
        Some(Vector2::new(
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        ))
    }
    /// d(u,v)/dp at p.
    #[inline]
    fn jacobian(&self, p: &Vector3<f64>) -> nalgebra::Matrix2x3<f64> {
        let zi = 1.0 / p[2];
        nalgebra::Matrix2x3::new(
            self.fx * zi,
            0.0,
            -self.fx * p[0] * zi * zi,
            0.0,
            self.fy * zi,
            -self.fy * p[1] * zi * zi,
        )
    }
}

struct Anchor {
    t_wc: Matrix4<f64>,
    live: usize,
    frame: u32,
    /// Image pyramid of the anchor frame, kept so that patches re-anchored to it can
    /// re-cut their reference. Only anchors (keyframes) keep pixels, never other frames.
    pyr: Pyramid,
}

struct Patch {
    id: u64,
    anchor: usize,
    level: usize,
    /// Anchor pixel at the patch's level.
    u: f64,
    v: f64,
    /// Unit bearings of every patch pixel in the anchor camera, and the reference values
    /// (zero-mean), fixed at birth.
    bearings: Vec<Vector3<f64>>,
    reference: Vec<f32>,
    eta: f64,
    eta_sigma: f64,
    misses: u32,
    age: u32,
    /// Outcome of the last frame: 0 accepted, 1 residual, 2 ambiguous, 3 curvature, 4 outside.
    last: u8,
}

/// One accepted observation of a patch on the current frame.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub id: u64,
    /// Matched pixel, level-0 coordinates.
    pub u: f64,
    pub v: f64,
    /// Pyramid level the patch lives on.
    pub level: usize,
    /// Current log-range estimate and its along-line std from the Gauss–Newton Hessian.
    pub eta: f64,
    pub eta_sigma: f64,
    /// Along-line std of the pixel position (level-0 pixels) and the line direction.
    pub sigma_px: f64,
    pub dir: Vector2<f64>,
    pub mean_abs_residual: f32,
    /// Anchor pixel (level-0 coordinates) and the process() call index at which the
    /// anchor was taken, so an external truth can re-project the anchored point.
    pub anchor_u: f64,
    pub anchor_v: f64,
    pub anchor_frame: u32,
}

/// Per-frame counters, for the harness.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    pub live: usize,
    pub observed: usize,
    pub born: usize,
    pub dropped: usize,
    pub rejected_residual: usize,
    pub rejected_ambiguous: usize,
    pub rejected_curvature: usize,
    pub rejected_outside: usize,
    pub new_anchor: bool,
    pub reanchored: usize,
    /// Matches withheld by `max_obs_br` (the patch state still updated).
    pub withheld_br: usize,
}

pub struct FlowDepF {
    settings: FlowDepFSettings,
    intr: Intr,
    w: usize,
    h: usize,
    anchors: Vec<Anchor>,
    patches: Vec<Patch>,
    next_id: u64,
    frame: u32,
    frames_since_anchor: u32,
    last_t_wc: Option<Matrix4<f64>>,
    pub stats: FrameStats,
    /// Level-0 pixels no patch may be born on this frame (the corner tracker's live
    /// features: those already feed the depth filter, so a patch there is repeated work).
    /// Consumed by the next `process` call and cleared.
    birth_exclusions: Vec<(f64, f64)>,
}

impl FlowDepF {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        settings: FlowDepFSettings,
    ) -> Self {
        Self {
            settings,
            intr: Intr { fx, fy, cx, cy },
            w: width,
            h: height,
            anchors: Vec::new(),
            patches: Vec::new(),
            next_id: 1,
            frame: 0,
            frames_since_anchor: 0,
            last_t_wc: None,
            stats: FrameStats::default(),
            birth_exclusions: Vec::new(),
        }
    }

    /// Pixels (level 0) that the next frame's births must keep `min_birth_dist_px` away
    /// from, on top of the live patches: the corner tracker's current features.
    pub fn set_birth_exclusions(&mut self, pixels: Vec<(f64, f64)>) {
        self.birth_exclusions = pixels;
    }

    pub fn n_live(&self) -> usize {
        self.patches.len()
    }

    /// Replace a patch's search bound by an external posterior on its RANGE (metres).
    /// Ignored for unknown ids or non-positive values.
    pub fn set_posterior(&mut self, posteriors: &HashMap<u64, (f64, f64)>) {
        for p in &mut self.patches {
            if let Some(&(range, sigma_r)) = posteriors.get(&p.id) {
                if range > 0.0 && sigma_r > 0.0 {
                    p.eta = range.ln();
                    p.eta_sigma = (sigma_r / range).max(self.settings.eta_sigma_floor);
                }
            }
        }
    }

    /// Process one frame: match every live patch, then (maybe) make this frame an anchor
    /// and birth new patches on it. Returns the accepted observations.
    pub fn process(&mut self, gray: &[u8], t_wc: &Matrix4<f64>) -> Vec<Observation> {
        assert_eq!(gray.len(), self.w * self.h, "image size mismatch");
        let pyr = Pyramid::build(gray, self.w, self.h, self.settings.levels);
        let mut stats = FrameStats {
            live: self.patches.len(),
            ..Default::default()
        };
        let mut obs = Vec::with_capacity(self.patches.len());

        // Relative pose anchor→current for every anchor: p_c = R p_a + t.
        let t_cw = t_wc.try_inverse().expect("t_wc invertible");
        let rel: Vec<(Matrix3<f64>, Vector3<f64>)> = self
            .anchors
            .iter()
            .map(|a| {
                let t_ca = t_cw * a.t_wc;
                (
                    t_ca.fixed_view::<3, 3>(0, 0).into_owned(),
                    t_ca.fixed_view::<3, 1>(0, 3).into_owned(),
                )
            })
            .collect();

        let s = self.settings.clone();
        let intr0 = self.intr;
        // Matching reads only the patch, the pyramid and the relative pose, so the patches
        // are matched independently (in parallel with the `parallel` feature) and the
        // per-patch state is written back in one sequential pass below.
        let match_one = |p: &Patch| {
            let (r, t) = rel[p.anchor];
            let intr = intr0.at_level(p.level);
            let lvl = &pyr.levels[p.level];
            match_patch(p, lvl, &intr, &r, &t, &s)
        };
        #[cfg(feature = "parallel")]
        let results: Vec<Result<Observation, Reject>> = {
            use rayon::prelude::*;
            self.patches.par_iter().map(match_one).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let results: Vec<Result<Observation, Reject>> =
            self.patches.iter().map(match_one).collect();
        for (p, res) in self.patches.iter_mut().zip(results) {
            p.age += 1;
            let t = rel[p.anchor].1;
            match res {
                Ok(mut o) => {
                    let scp = (1u32 << p.level) as f64;
                    o.anchor_u = (p.u + 0.5) * scp - 0.5;
                    o.anchor_v = (p.v + 0.5) * scp - 0.5;
                    o.anchor_frame = self.anchors[p.anchor].frame;
                    p.last = 0;
                    p.eta = o.eta;
                    p.eta_sigma = (o.eta_sigma.max(s.eta_sigma_floor)) + s.eta_sigma_growth;
                    p.misses = 0;
                    stats.observed += 1;
                    if s.max_obs_br > 0.0 && t.norm() / o.eta.exp() > s.max_obs_br {
                        stats.withheld_br += 1;
                    } else {
                        obs.push(o);
                    }
                }
                Err(why) => {
                    p.misses += 1;
                    p.eta_sigma += s.eta_sigma_growth;
                    p.last = match why {
                        Reject::Residual => {
                            stats.rejected_residual += 1;
                            1
                        }
                        Reject::Ambiguous => {
                            stats.rejected_ambiguous += 1;
                            2
                        }
                        Reject::Curvature => {
                            stats.rejected_curvature += 1;
                            3
                        }
                        Reject::Outside => {
                            stats.rejected_outside += 1;
                            4
                        }
                    };
                }
            }
        }

        // Drop dead patches.
        let before = self.patches.len();
        let max_misses = s.max_misses;
        self.patches.retain(|p| p.misses <= max_misses);
        stats.dropped = before - self.patches.len();
        for a in &mut self.anchors {
            a.live = 0;
        }
        for p in &self.patches {
            self.anchors[p.anchor].live += 1;
        }

        // Re-anchoring: patches whose anchor→current baseline exceeds reanchor_br · range
        // move to the newest keyframe that is close enough, else the current frame becomes
        // a keyframe (with the usual births on it) and they move there.
        let mut promote_current = false;
        if s.reanchor_br > 0.0 && !self.anchors.is_empty() {
            let newest = self.anchors.len() - 1;
            let mut moved = 0usize;
            let mut need_current = Vec::new();
            for (pi, p) in self.patches.iter_mut().enumerate() {
                if p.misses > 0 {
                    continue; // only re-anchor on a frame where the patch was matched
                }
                let (_, t) = rel[p.anchor];
                let br = t.norm() / p.eta.exp();
                if br <= s.reanchor_br {
                    continue;
                }
                if newest != p.anchor {
                    let t_na = self.anchors[newest].t_wc.try_inverse().unwrap()
                        * self.anchors[p.anchor].t_wc;
                    let br_new = t_na.fixed_view::<3, 1>(0, 3).norm() / p.eta.exp();
                    if br_new <= 0.5 * s.reanchor_br {
                        if recut(p, newest, &self.anchors, &intr0, &s) {
                            moved += 1;
                            continue;
                        }
                    }
                }
                need_current.push(pi);
            }
            if !need_current.is_empty() {
                // The current frame becomes an anchor now (its pyramid is at hand); the
                // births below will reuse it as `anchor_idx`.
                let cur = self.anchors.len();
                self.anchors.push(Anchor {
                    t_wc: *t_wc,
                    live: 0,
                    frame: self.frame,
                    pyr: pyr.clone_images(),
                });
                for pi in need_current {
                    if recut(&mut self.patches[pi], cur, &self.anchors, &intr0, &s) {
                        moved += 1;
                    } else {
                        self.patches[pi].misses = s.max_misses + 1; // cannot be re-cut: drop
                    }
                }
                promote_current = true;
            }
            stats.reanchored = moved;
            let max_misses = s.max_misses;
            self.patches.retain(|p| p.misses <= max_misses);
            for a in &mut self.anchors {
                a.live = 0;
            }
            for p in &self.patches {
                self.anchors[p.anchor].live += 1;
            }
        }

        // New anchor + births.
        self.frames_since_anchor += 1;
        let need = self.anchors.is_empty()
            || promote_current
            || self.frames_since_anchor >= s.keyframe_every
            || (self.patches.len() as f32) < s.min_live_frac * s.max_patches as f32;
        if need {
            // Hint direction for the 1-D birth test: the image motion the LAST inter-frame
            // translation produced (epipolar direction at the next frame is unknown; the
            // previous one is the best available guess). None → any-direction test.
            let hint_t = self.last_t_wc.map(|prev| {
                let t_cp = t_cw * prev;
                t_cp.fixed_view::<3, 1>(0, 3).into_owned()
            });
            let born = self.birth(&pyr, t_wc, hint_t);
            stats.born = born;
            stats.new_anchor = true;
            self.frames_since_anchor = 0;
        }
        // Forget anchors without live patches (the newest is always kept).
        self.gc_anchors();

        self.birth_exclusions.clear();
        self.last_t_wc = Some(*t_wc);
        self.frame += 1;
        stats.live = self.patches.len();
        self.stats = stats;
        obs
    }

    fn gc_anchors(&mut self) {
        let n = self.anchors.len();
        if n == 0 {
            return;
        }
        let keep: Vec<bool> = self
            .anchors
            .iter()
            .enumerate()
            .map(|(i, a)| a.live > 0 || i == n - 1)
            .collect();
        if keep.iter().all(|&k| k) {
            return;
        }
        let mut remap = vec![usize::MAX; n];
        let mut j = 0;
        for i in 0..n {
            if keep[i] {
                remap[i] = j;
                j += 1;
            }
        }
        let mut i = 0;
        self.anchors.retain(|_| {
            let k = keep[i];
            i += 1;
            k
        });
        for p in &mut self.patches {
            p.anchor = remap[p.anchor];
        }
    }

    /// Make the current frame an anchor and birth patches on it, one per grid cell at most,
    /// coarsest level first (a cell already covered by a coarser birth or a live patch is
    /// skipped). Returns the number born.
    fn birth(&mut self, pyr: &Pyramid, t_wc: &Matrix4<f64>, hint_t: Option<Vector3<f64>>) -> usize {
        let s = &self.settings;
        // Reuse the anchor if the current frame was already promoted this call.
        let reuse = self.anchors.last().map_or(false, |a| a.frame == self.frame);
        let anchor_idx = if reuse {
            self.anchors.len() - 1
        } else {
            self.anchors.len()
        };
        let budget = s.max_patches.saturating_sub(self.patches.len());
        if budget == 0 {
            if !reuse {
                self.anchors.push(Anchor {
                    t_wc: *t_wc,
                    live: 0,
                    frame: self.frame,
                    pyr: pyr.clone_images(),
                });
            }
            return 0;
        }
        // Occupancy mask at level 0 from live patches' current projections (their last
        // accepted position is not stored; use the anchor pixel warped at the current η).
        let t_cw = t_wc.try_inverse().unwrap();
        let mut occupied: Vec<(f64, f64)> = Vec::with_capacity(self.patches.len());
        for p in &self.patches {
            let a = &self.anchors[p.anchor];
            let t_ca = t_cw * a.t_wc;
            let r = t_ca.fixed_view::<3, 3>(0, 0);
            let t = t_ca.fixed_view::<3, 1>(0, 3);
            let intr = self.intr.at_level(p.level);
            let x = r * (intr.unit_bearing(p.u, p.v) * p.eta.exp()) + t;
            if let Some(uv) = intr.project(&x) {
                let sc = (1u32 << p.level) as f64;
                occupied.push(((uv[0] + 0.5) * sc - 0.5, (uv[1] + 0.5) * sc - 0.5));
            }
        }
        occupied.extend(self.birth_exclusions.iter().copied());
        let min_d2 = s.min_birth_dist_px * s.min_birth_dist_px;
        let is_free = |u0: f64, v0: f64, occ: &[(f64, f64)]| {
            occ.iter()
                .all(|&(a, b)| (a - u0) * (a - u0) + (b - v0) * (b - v0) >= min_d2)
        };

        let eta0 = 0.5 * (s.d_min.ln() + s.d_max.ln());
        let sigma0 = 0.5 * (s.d_max.ln() - s.d_min.ln()) / s.bound_sigma;
        let half = s.patch_half as isize;
        let mut born = 0usize;
        let mut new_patches = Vec::new();
        for level in (0..s.levels).rev() {
            let lvl = &pyr.levels[level];
            let intr = self.intr.at_level(level);
            let sc = (1u32 << level) as f64;
            // Hint direction in this level's image: the epipolar direction of a point at the
            // prior range under the last translation — image motion ∝ J_proj · (−t) for a
            // static point when the camera moves by t (sign irrelevant for a curvature test).
            let cells_x = lvl.w / s.cell;
            let cells_y = lvl.h / s.cell;
            for cy in 0..cells_y {
                for cx in 0..cells_x {
                    if born >= budget {
                        break;
                    }
                    // Best pixel in the cell by 1-D curvature along the hint (or λ_max).
                    let mut best: Option<(f32, usize, usize)> = None;
                    let y0 = cy * s.cell;
                    let x0 = cx * s.cell;
                    for y in y0.max(half as usize + 1)..(y0 + s.cell).min(lvl.h - half as usize - 2)
                    {
                        for x in
                            x0.max(half as usize + 1)..(x0 + s.cell).min(lvl.w - half as usize - 2)
                        {
                            let (gxx, gxy, gyy) = structure_tensor(lvl, x, y, half);
                            let c = match hint_t {
                                Some(t) => {
                                    let b = intr.unit_bearing(x as f64, y as f64);
                                    let d = intr.jacobian(&(b * eta0.exp())) * (-t);
                                    let n = d.norm();
                                    if n < 1e-9 {
                                        lambda_max(gxx, gxy, gyy)
                                    } else {
                                        let (ex, ey) = (d[0] / n, d[1] / n);
                                        (gxx * ex * ex + 2.0 * gxy * ex * ey + gyy * ey * ey) as f32
                                    }
                                }
                                None => lambda_max(gxx, gxy, gyy),
                            };
                            if best.map_or(true, |(bc, _, _)| c > bc) {
                                best = Some((c, x, y));
                            }
                        }
                    }
                    let Some((c, mut x, mut y)) = best else {
                        continue;
                    };
                    if c < s.min_birth_curvature {
                        continue;
                    }
                    // Re-centre on the gradient-energy centroid of the footprint (up to twice):
                    // the depth a patch measures belongs where its gradient is, and a patch
                    // whose centre sits on a flat background next to textured foreground would
                    // otherwise report the foreground's range at a background pixel.
                    for _ in 0..2 {
                        let (dx, dy) = gradient_centroid(lvl, x, y, half);
                        if dx.abs() < 1.0 && dy.abs() < 1.0 {
                            break;
                        }
                        let nx = (x as f64 + dx).round() as isize;
                        let ny = (y as f64 + dy).round() as isize;
                        if nx < half + 1
                            || ny < half + 1
                            || nx >= lvl.w as isize - half - 2
                            || ny >= lvl.h as isize - half - 2
                        {
                            break;
                        }
                        x = nx as usize;
                        y = ny as usize;
                    }
                    let u0 = (x as f64 + 0.5) * sc - 0.5;
                    let v0 = (y as f64 + 0.5) * sc - 0.5;
                    if !is_free(u0, v0, &occupied) {
                        continue;
                    }
                    // Reference patch and bearings.
                    let n = (2 * half) as usize;
                    let mut reference = Vec::with_capacity(n * n);
                    let mut bearings = Vec::with_capacity(n * n);
                    for dy in -half..half {
                        for dx in -half..half {
                            let (uu, vv) = (x as isize + dx, y as isize + dy);
                            reference.push(lvl.img[vv as usize * lvl.w + uu as usize]);
                            bearings.push(intr.unit_bearing(uu as f64, vv as f64));
                        }
                    }
                    let mean = reference.iter().sum::<f32>() / reference.len() as f32;
                    for r in &mut reference {
                        *r -= mean;
                    }
                    new_patches.push(Patch {
                        id: self.next_id,
                        anchor: anchor_idx,
                        level,
                        u: x as f64,
                        v: y as f64,
                        bearings,
                        reference,
                        eta: eta0,
                        eta_sigma: sigma0,
                        misses: 0,
                        age: 0,
                        last: 0,
                    });
                    self.next_id += 1;
                    occupied.push((u0, v0));
                    born += 1;
                }
            }
        }
        let live = new_patches.len();
        if reuse {
            self.anchors[anchor_idx].live += live;
        } else {
            self.anchors.push(Anchor {
                t_wc: *t_wc,
                live,
                frame: self.frame,
                pyr: pyr.clone_images(),
            });
        }
        self.patches.extend(new_patches);
        born
    }

    /// Debug: zero-mean SSD cost of patch `id` against `gray` at `n` log-range samples
    /// spanning [d_min, d_max]: (η, cost per valid pixel, valid fraction, u, v at level 0).
    pub fn cost_profile(
        &self,
        id: u64,
        gray: &[u8],
        t_wc: &Matrix4<f64>,
        n: usize,
    ) -> Vec<(f64, f32, f32, f64, f64)> {
        let Some(p) = self.patches.iter().find(|p| p.id == id) else {
            return Vec::new();
        };
        let pyr = Pyramid::build(gray, self.w, self.h, self.settings.levels);
        let t_ca = t_wc.try_inverse().unwrap() * self.anchors[p.anchor].t_wc;
        let r = t_ca.fixed_view::<3, 3>(0, 0).into_owned();
        let t = t_ca.fixed_view::<3, 1>(0, 3).into_owned();
        let intr = self.intr.at_level(p.level);
        let sc = (1u32 << p.level) as f64;
        let (lo, hi) = (self.settings.d_min.ln(), self.settings.d_max.ln());
        (0..n)
            .filter_map(|i| {
                let eta = lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64;
                cost_at(p, &pyr.levels[p.level], &intr, &r, &t, eta).map(|(c, vf, uv)| {
                    (
                        eta,
                        c,
                        vf,
                        (uv[0] + 0.5) * sc - 0.5,
                        (uv[1] + 0.5) * sc - 0.5,
                    )
                })
            })
            .collect()
    }

    /// Debug view of the live patches: (id, level, anchor pixel at level 0, η, σ_η, age).
    pub fn live_patches(&self) -> Vec<(u64, usize, f64, f64, f64, f64, u32)> {
        self.patches
            .iter()
            .map(|p| {
                let sc = (1u32 << p.level) as f64;
                (
                    p.id,
                    p.level,
                    (p.u + 0.5) * sc - 0.5,
                    (p.v + 0.5) * sc - 0.5,
                    p.eta,
                    p.eta_sigma,
                    p.age,
                )
            })
            .collect()
    }

    /// Debug: (id, age, misses, last outcome code) per live patch; see `Patch::last`.
    pub fn live_outcomes(&self) -> Vec<(u64, u32, u32, u8)> {
        self.patches
            .iter()
            .map(|p| (p.id, p.age, p.misses, p.last))
            .collect()
    }
}

/// Move a patch to anchor `new_idx`: project its centre point (bearing · exp(η)) into the
/// new anchor frame, re-sample the reference patch and bearings there (bilinear, sub-pixel
/// centre), and express η in the new frame. Returns false if the point is not visible there.
fn recut(
    p: &mut Patch,
    new_idx: usize,
    anchors: &[Anchor],
    intr0: &Intr,
    s: &FlowDepFSettings,
) -> bool {
    let intr = intr0.at_level(p.level);
    let t_na = anchors[new_idx].t_wc.try_inverse().unwrap() * anchors[p.anchor].t_wc;
    let r = t_na.fixed_view::<3, 3>(0, 0);
    let t = t_na.fixed_view::<3, 1>(0, 3);
    let x_new = r * (intr.unit_bearing(p.u, p.v) * p.eta.exp()) + t;
    let Some(c) = intr.project(&x_new) else {
        return false;
    };
    let lvl = &anchors[new_idx].pyr.levels[p.level];
    let half = s.patch_half as f64;
    if !(lvl.inside(c[0] - half, c[1] - half) && lvl.inside(c[0] + half, c[1] + half)) {
        return false;
    }
    let n = 2 * s.patch_half;
    let mut reference = Vec::with_capacity(n * n);
    let mut bearings = Vec::with_capacity(n * n);
    let h = s.patch_half as isize;
    for dy in -h..h {
        for dx in -h..h {
            let (uu, vv) = (c[0] + dx as f64, c[1] + dy as f64);
            reference.push(lvl.sample_img(uu, vv));
            bearings.push(intr.unit_bearing(uu, vv));
        }
    }
    let mean = reference.iter().sum::<f32>() / reference.len() as f32;
    for v in &mut reference {
        *v -= mean;
    }
    p.anchor = new_idx;
    p.u = c[0];
    p.v = c[1];
    p.reference = reference;
    p.bearings = bearings;
    p.eta = x_new.norm().ln();
    true
}

#[inline]
fn lambda_max(gxx: f64, gxy: f64, gyy: f64) -> f32 {
    let tr = gxx + gyy;
    let d = ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt();
    (0.5 * (tr + d)) as f32
}

/// Offset (dx, dy) from (x, y) to the centroid of gradient energy |∇I|² over the
/// (2·half)² footprint; (0, 0) when the footprint has no gradient.
fn gradient_centroid(lvl: &Level, x: usize, y: usize, half: isize) -> (f64, f64) {
    let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -half..half {
        for dx in -half..half {
            let i = (y as isize + dy) as usize * lvl.w + (x as isize + dx) as usize;
            let w = (lvl.gx[i] * lvl.gx[i] + lvl.gy[i] * lvl.gy[i]) as f64;
            sw += w;
            sx += w * dx as f64;
            sy += w * dy as f64;
        }
    }
    if sw <= 0.0 {
        (0.0, 0.0)
    } else {
        (sx / sw, sy / sw)
    }
}

/// Mean structure tensor over the (2·half)² patch centred at integer (x, y).
fn structure_tensor(lvl: &Level, x: usize, y: usize, half: isize) -> (f64, f64, f64) {
    let (mut gxx, mut gxy, mut gyy) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -half..half {
        for dx in -half..half {
            let i = (y as isize + dy) as usize * lvl.w + (x as isize + dx) as usize;
            let (gx, gy) = (lvl.gx[i] as f64, lvl.gy[i] as f64);
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    let n = (4 * half * half) as f64;
    (gxx / n, gxy / n, gyy / n)
}

enum Reject {
    Residual,
    Ambiguous,
    Curvature,
    Outside,
}

/// Zero-mean SSD of the warped reference against the current image at log-range `eta`.
/// Returns (cost per valid pixel, valid fraction, projected centre).
fn cost_at(
    p: &Patch,
    lvl: &Level,
    intr: &Intr,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    eta: f64,
) -> Option<(f32, f32, Vector2<f64>)> {
    let range = eta.exp();
    let n = p.bearings.len();
    let mut vals = Vec::with_capacity(n);
    let mut valid = 0usize;
    for b in &p.bearings {
        let x = r * (b * range) + t;
        match intr.project(&x) {
            Some(uv) if lvl.inside(uv[0], uv[1]) => {
                vals.push(Some(lvl.sample(uv[0], uv[1]).0));
                valid += 1;
            }
            _ => vals.push(None),
        }
    }
    if valid == 0 {
        return None;
    }
    let mean = vals.iter().flatten().sum::<f32>() / valid as f32;
    let mut ssd = 0.0f32;
    for (v, rf) in vals.iter().zip(&p.reference) {
        if let Some(v) = v {
            let d = (v - mean) - rf;
            ssd += d * d;
        }
    }
    let centre_b = intr.unit_bearing(p.u, p.v);
    let xc = r * (centre_b * range) + t;
    let c = intr.project(&xc)?;
    Some((ssd / valid as f32, valid as f32 / n as f32, c))
}

fn match_patch(
    p: &Patch,
    lvl: &Level,
    intr: &Intr,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    s: &FlowDepFSettings,
) -> Result<Observation, Reject> {
    // 1. Segment from the bound; sample so that consecutive projections are ≤ scan_step apart.
    let mut lo = (p.eta - s.bound_sigma * p.eta_sigma).max(s.d_min.ln());
    let mut hi = (p.eta + s.bound_sigma * p.eta_sigma).min(s.d_max.ln());
    let centre_b = intr.unit_bearing(p.u, p.v);
    // Clip the interval to the ranges whose point lies in front of the current camera:
    // z_c(η) = (R b)_z e^η + t_z is monotonic in η, so the admissible set is a half-line.
    // With the camera moved forward, the near end of a wide bound is routinely behind it;
    // that must shorten the segment, not reject the patch.
    {
        let rbz = (r * centre_b)[2];
        let min_z = 0.05;
        if rbz.abs() > 1e-12 {
            let eta_star = ((min_z - t[2]) / rbz).max(1e-9).ln();
            if rbz > 0.0 {
                lo = lo.max(eta_star);
            } else {
                hi = hi.min(eta_star);
            }
        } else if t[2] <= min_z {
            return Err(Reject::Outside);
        }
        if hi <= lo {
            return Err(Reject::Outside);
        }
    }
    let proj = |eta: f64| intr.project(&(r * (centre_b * eta.exp()) + t));
    let (Some(c_lo), Some(c_hi)) = (proj(lo), proj(hi)) else {
        return Err(Reject::Outside);
    };
    let seg_len = (c_hi - c_lo).norm();
    let n_samples = ((seg_len / s.scan_step_px).ceil() as usize + 1).clamp(1, s.max_scan_samples);
    let mut best: Option<(f32, f64, Vector2<f64>)> = None;
    let mut costs: Vec<f32> = Vec::with_capacity(n_samples);
    let mut any_inside = false;
    for i in 0..n_samples {
        let eta = if n_samples == 1 {
            p.eta
        } else {
            lo + (hi - lo) * i as f64 / (n_samples - 1) as f64
        };
        match cost_at(p, lvl, intr, r, t, eta) {
            Some((c, vf, uv)) if vf >= s.min_valid_frac => {
                any_inside = true;
                costs.push(c);
                if best.map_or(true, |(bc, _, _)| c < bc) {
                    best = Some((c, eta, uv));
                }
            }
            _ => costs.push(f32::INFINITY),
        }
    }
    if !any_inside {
        return Err(Reject::Outside);
    }
    let (best_cost, mut eta, _) = best.unwrap();
    // Ambiguity: a second LOCAL minimum not adjacent to the best, within the ratio.
    if n_samples >= 3 {
        let bi = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        for i in 1..n_samples - 1 {
            if (i as isize - bi as isize).abs() <= 1 {
                continue;
            }
            if costs[i] <= costs[i - 1]
                && costs[i] <= costs[i + 1]
                && costs[i] < best_cost / s.ambiguity_ratio
            {
                return Err(Reject::Ambiguous);
            }
        }
    }

    // 2. Gauss–Newton on η: residual_i = (I_c(w_i(η)) − mean_c) − ref_i,
    //    J_i = ∇I_c(w_i) · d w_i/dη,  d x_i/dη = R b_i exp(η).
    let mut mean_abs = 0.0f32;
    let mut h_sum = 0.0f64;
    let mut last_valid_frac = 0.0f32;
    for _ in 0..s.gn_max_iters {
        let range = eta.exp();
        let mut samples: Vec<Option<(f32, f64)>> = Vec::with_capacity(p.bearings.len()); // (value, J)
        let mut valid = 0usize;
        let mut sum = 0.0f32;
        for b in &p.bearings {
            let x = r * (b * range) + t;
            match intr.project(&x) {
                Some(uv) if lvl.inside(uv[0], uv[1]) => {
                    let (val, gx, gy) = lvl.sample(uv[0], uv[1]);
                    let dxdeta = r * (b * range);
                    let duv = intr.jacobian(&x) * dxdeta;
                    let j = gx as f64 * duv[0] + gy as f64 * duv[1];
                    samples.push(Some((val, j)));
                    valid += 1;
                    sum += val;
                }
                _ => samples.push(None),
            }
        }
        last_valid_frac = valid as f32 / p.bearings.len() as f32;
        if last_valid_frac < s.min_valid_frac {
            return Err(Reject::Outside);
        }
        let mean = sum / valid as f32;
        let (mut jtj, mut jtr, mut abs_sum) = (0.0f64, 0.0f64, 0.0f32);
        for (sm, rf) in samples.iter().zip(&p.reference) {
            if let Some((val, j)) = sm {
                let res = ((val - mean) - rf) as f64;
                // Huber at 2σ_photo.
                let k = 2.0 * s.sigma_photo;
                let w = if res.abs() <= k { 1.0 } else { k / res.abs() };
                jtj += w * j * j;
                jtr += w * j * res;
                abs_sum += res.abs() as f32;
            }
        }
        mean_abs = abs_sum / valid as f32;
        h_sum = jtj;
        if jtj < s.min_track_curvature as f64 {
            return Err(Reject::Curvature);
        }
        let step = -jtr / jtj;
        eta += step;
        eta = eta.clamp(s.d_min.ln(), s.d_max.ln());
        if step.abs() < s.gn_tol {
            break;
        }
    }
    if mean_abs > s.max_mean_abs_residual || last_valid_frac < s.min_valid_frac {
        return Err(Reject::Residual);
    }
    // 3. Output: centre projection at the refined η, along-line variance from the Hessian.
    let range = eta.exp();
    let xc = r * (centre_b * range) + t;
    let Some(uv) = intr.project(&xc) else {
        return Err(Reject::Outside);
    };
    let duv = intr.jacobian(&xc) * (r * (centre_b * range));
    let dn = duv.norm();
    let eta_sigma = (s.sigma_photo * s.sigma_photo / h_sum).sqrt();
    let sc = (1u32 << p.level) as f64;
    Ok(Observation {
        id: p.id,
        u: (uv[0] + 0.5) * sc - 0.5,
        v: (uv[1] + 0.5) * sc - 0.5,
        level: p.level,
        eta,
        eta_sigma,
        sigma_px: eta_sigma * dn * sc,
        dir: if dn > 1e-12 {
            duv / dn
        } else {
            Vector2::new(1.0, 0.0)
        },
        mean_abs_residual: mean_abs,
        anchor_u: 0.0,
        anchor_v: 0.0,
        anchor_frame: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fronto-parallel textured plane at 5 m seen by a camera translating sideways: every
    /// born patch must be matched at the right range once the baseline is non-trivial.
    #[test]
    fn plane_sideways_recovers_range() {
        let (w, h) = (160usize, 120usize);
        let (fx, fy, cx, cy) = (120.0, 120.0, 79.5, 59.5);
        let z = 5.0;
        // Texture: smooth random field, sampled by projecting the plane into each view.
        let tex = |x: f64, y: f64| -> f32 {
            let v = (x * 1.7).sin() * 40.0
                + (y * 2.3).cos() * 40.0
                + ((x * 0.9 + y * 1.1) * 3.0).sin() * 30.0;
            (128.0 + v) as f32
        };
        let render = |t_wc: &Matrix4<f64>| -> Vec<u8> {
            let t_cw = t_wc.try_inverse().unwrap();
            let r = t_cw.fixed_view::<3, 3>(0, 0);
            let tt = t_cw.fixed_view::<3, 1>(0, 3);
            let mut img = vec![0u8; w * h];
            // world point on plane z=5 (world == first camera frame): invert the ray.
            let rwc = t_wc.fixed_view::<3, 3>(0, 0);
            let twc = t_wc.fixed_view::<3, 1>(0, 3);
            let _ = (r, tt);
            for y in 0..h {
                for x in 0..w {
                    let d = Vector3::new((x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0);
                    let dw = rwc * d;
                    let s = (z - twc[2]) / dw[2];
                    let pw = twc + dw * s;
                    img[y * w + x] = tex(pw[0], pw[1]).clamp(0.0, 255.0) as u8;
                }
            }
            img
        };
        let mut settings = FlowDepFSettings::default();
        settings.levels = 1;
        settings.keyframe_every = 100;
        settings.d_min = 1.0;
        settings.d_max = 30.0;
        let mut f = FlowDepF::new(fx, fy, cx, cy, w, h, settings);
        let mut last = Vec::new();
        for k in 0..6 {
            let mut t_wc = Matrix4::identity();
            t_wc[(0, 3)] = 0.08 * k as f64; // 8 cm per frame sideways
            let img = render(&t_wc);
            last = f.process(&img, &t_wc);
        }
        assert!(f.n_live() > 50, "born {}", f.n_live());
        assert!(last.len() > 40, "observed {}", last.len());
        // η is log-RANGE of the patch centre; on the plane z = 5 the true range of the pixel
        // the patch was born at is z · |ray|. Use the anchor pixel (frame 0, identity pose).
        let live = f.live_patches();
        let errs: Vec<f64> = last
            .iter()
            .map(|o| {
                let (_, _, u0, v0, _, _, _) = live.iter().find(|q| q.0 == o.id).copied().unwrap();
                let truth = z * ((u0 - cx) / fx).hypot((v0 - cy) / fy).hypot(1.0);
                (o.eta.exp() - truth).abs() / truth
            })
            .collect();
        let mut sorted = errs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        assert!(
            med < 0.03,
            "median range error {med:.3} (frame 5, baseline 0.4 m at 5 m)"
        );
    }
}
