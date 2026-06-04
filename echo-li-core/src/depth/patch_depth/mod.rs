use std::sync::Arc;

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rudolf_v::image::Image;
use rudolf_v::pyramid::{Pyramid, PyramidScratch};

use crate::core_types::{CameraIntrinsics, DepthMap};
use crate::depth::sparse_3d::Sparse3DFilter;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::vision_measurement::VisionMeasurement;

mod fuse;
mod image_ops;
mod seeds;
#[cfg(target_arch = "x86_64")]
mod simd;
#[cfg(target_arch = "aarch64")]
mod simd_neon;

use fuse::PatchGrid;
#[cfg(not(feature = "parallel"))]
use fuse::densify_pixels;
#[cfg(feature = "parallel")]
use fuse::{densify_pixels_parallel, patch_centers};
use image_ops::{
    bilerp_ptr, bilinear_patch_footprint, build_bilinear_valid_pyramid, build_pinhole_to_raw_lut,
    build_pyramid_from_u8, dyadic_scale_offset, empty_pyramid, gradients, mask_row_valid,
    sample_bilinear_valid, sample_bilinear_valid_with_grad, sample_nearest, sample_valid_nearest,
    scaled_intrinsics, undistort_level_specs,
};
use seeds::{SeedGrid, median_seed_depth, nearby_seed_weights, scale_seeds};
#[cfg(target_arch = "x86_64")]
use simd::{
    PerPatchAffineGeom, fast_translation_accum_avx2_if_available,
    per_patch_affine_accum_avx2_if_available,
};
#[cfg(target_arch = "aarch64")]
use simd_neon::fast_translation_accum_neon_if_available;

/// Default Gauss-Newton convergence tolerance on the log-range step `|Δη|`. Below
/// this the depth update is sub-0.1% of range; further iterations are wasted leaf
/// evaluations. Overridable via `PatchDepthSettings::gn_eta_convergence_tol`
/// (config key `gn_eta_convergence_tol`); used to early-exit the GN loop.
const GN_ETA_CONVERGENCE_TOL: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchDepthCameraMode {
    RawDistorted,
    UndistortedPinhole,
    TiledBearing,
    /// Per-patch local tangent rectification: wide-FoV, seamless replacement for
    /// `TiledBearing`. Works in the raw image (like `RawDistorted`) but rectifies
    /// each patch into its own tangent frame for the FastTranslation solve.
    PerPatchBearing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchDepthWarpMode {
    Exact,
    FastTranslation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchDepthSeedCoordinates {
    RawDistorted,
    UndistortedPinhole,
}

#[derive(Debug, Clone)]
pub struct PatchDepthSettings {
    pub camera_mode: PatchDepthCameraMode,
    pub warp_mode: PatchDepthWarpMode,
    pub scale: f64,
    pub patch_size: usize,
    pub patch_stride: usize,
    pub min_depth: f64,
    pub max_depth: f64,
    pub photo_huber_delta: f64,
    pub sigma_photo: f64,
    pub n_gn_iters: usize,
    /// Gauss-Newton early-exit tolerance on `|Δη|` (per-patch-bearing solve).
    pub gn_eta_convergence_tol: f64,
    pub fd_eps: f64,
    pub lambda_seed: f64,
    pub seed_radius_px: f64,
    pub sigma_seed_floor: f64,
    pub n_search_candidates: usize,
    pub search_half_range: f64,
    pub min_baseline_ratio: f64,
    pub max_baseline_ratio: f64,
    pub min_photo_curvature: f64,
    pub max_photo_residual: f64,
    pub min_structure_eigen: f64,
    pub max_structure_condition: f64,
    pub n_pyramid_levels: usize,
    pub var_floor: f64,
    pub status_weight_photo: f64,
    pub status_weight_seed: f64,
    pub tiled_tile_size: usize,
    pub tiled_tile_overlap: usize,
}

impl Default for PatchDepthSettings {
    fn default() -> Self {
        Self {
            camera_mode: PatchDepthCameraMode::RawDistorted,
            warp_mode: PatchDepthWarpMode::Exact,
            scale: 1.0,
            patch_size: 8,
            patch_stride: 4,
            max_depth: 20.0,
            min_depth: 0.1,
            photo_huber_delta: 5.0,
            sigma_photo: 5.0,
            n_gn_iters: 5,
            gn_eta_convergence_tol: GN_ETA_CONVERGENCE_TOL,
            fd_eps: 1e-3,
            lambda_seed: 1.0,
            seed_radius_px: 32.0,
            sigma_seed_floor: 0.01,
            n_search_candidates: 5,
            search_half_range: 0.5,
            min_baseline_ratio: 0.005,
            max_baseline_ratio: 0.3,
            min_photo_curvature: 1e-6,
            max_photo_residual: 20.0,
            min_structure_eigen: 0.0,
            max_structure_condition: 0.0,
            n_pyramid_levels: 1,
            var_floor: 1e-6,
            status_weight_photo: 1.0,
            status_weight_seed: 0.6,
            tiled_tile_size: 96,
            tiled_tile_overlap: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PatchStatus {
    Unknown = 0,
    SeedOnly = 1,
    PhotoRefined = 2,
    Rejected = 3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchDepthOutput {
    /// Per-pixel log-range `η = ln(range)`, where `range` is the Euclidean
    /// camera-to-point distance. `NaN` marks pixels with no estimate. Convert at
    /// the consumer: `range = exp(η)`, 3D point `= exp(η) · unit_b[u,v]` (from the
    /// bearing LUT), z-depth `= exp(η) · unit_b.z`. Range is projection-agnostic
    /// and stays valid at wide FoV where z-depth degenerates.
    pub eta: DepthMap<f32>,
    /// Per-pixel `var(η)` ≈ relative range variance (`σ_range/range ≈ σ_η`).
    /// This is the reciprocal of an accumulated *confidence weight* that bakes in
    /// `status_weight`, fusion `photo_w`/`tile_w`, and a `var_floor` clamp — so its
    /// absolute scale is **not** calibrated; use it only as a relative gate.
    pub eta_var: DepthMap<f32>,
    pub status: DepthMap<PatchStatus>,
}

#[derive(Debug, Clone)]
pub struct FrameProducts {
    pub frame_id: u64,
    pub stamp: f64,
    pub gray: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub pose_t_wc: Matrix4<f64>,
}

#[derive(Debug, Clone)]
pub struct SparseDepthPrior {
    pub uv: Vector2<f64>,
    /// Log-range η = ln(range). Conversion from sensor output happens in
    /// `seeds_from_sparse_filter`; the GN solvers consume this directly.
    pub eta: f64,
    /// Variance of η (= var_rho / rho²).
    pub eta_var: f64,
}

#[derive(Debug, Clone)]
struct DepthKeyframe {
    frame: Arc<FrameProducts>,
    ref_pyramid: Arc<Vec<Image<f32>>>,
    bilinear_valid_pyramid: Option<Arc<Vec<Image<f32>>>>,
    grad_x_pyramid: Vec<Image<f32>>,
    grad_y_pyramid: Vec<Image<f32>>,
}

struct DepthFrameProducts {
    frame: Arc<FrameProducts>,
    pyramid: Arc<Vec<Image<f32>>>,
    valid_pyramid: Option<Arc<Vec<Image<f32>>>>,
}

#[derive(Debug, Clone)]
pub(super) struct RelativePose {
    pub(super) r: Matrix3<f64>,
    pub(super) t: Vector3<f64>,
}

impl RelativePose {
    fn from_matrix(t_ref_curr: &Matrix4<f64>) -> Self {
        Self {
            r: t_ref_curr.fixed_view::<3, 3>(0, 0).into_owned(),
            t: t_ref_curr.fixed_view::<3, 1>(0, 3).into_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScaledIntrinsics {
    pub(super) scale_from_original: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PatchEstimate {
    /// Log-range η = ln(range) of the 3D point at the patch centre.
    pub(super) eta: f64,
    pub(super) status: PatchStatus,
    pub(super) inv_var_w: f64,
}

impl PatchEstimate {
    pub(super) fn unknown() -> Self {
        Self {
            eta: 0.0,
            status: PatchStatus::Unknown,
            inv_var_w: 0.0,
        }
    }

    fn rejected(eta: f64) -> Self {
        Self {
            eta,
            status: PatchStatus::Rejected,
            inv_var_w: 0.0,
        }
    }

    fn photo_refined(eta: f64, eta_var: f64, settings: &PatchDepthSettings) -> Self {
        Self {
            eta,
            status: PatchStatus::PhotoRefined,
            inv_var_w: settings.status_weight_photo / eta_var.max(settings.var_floor),
        }
    }

    fn seed_only(eta: f64, eta_var: f64, settings: &PatchDepthSettings) -> Self {
        Self {
            eta,
            status: PatchStatus::SeedOnly,
            inv_var_w: settings.status_weight_seed / eta_var.max(settings.var_floor),
        }
    }

    #[inline(always)]
    pub(super) fn inv_var_weight_f32(self) -> Option<f32> {
        (self.inv_var_w > 0.0).then_some(self.inv_var_w as f32)
    }
}

#[derive(Debug, Clone, Copy)]
struct BilinearPatchFootprint {
    x: usize,
    y: usize,
    weights: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct TranslatedPatchFootprint {
    ref_fp: BilinearPatchFootprint,
    curr_x0: isize,
    curr_y0: isize,
    side: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct PatchAccum {
    grad: f64,
    hess: f64,
    sum_abs_res: f64,
    n_valid: usize,
}

#[derive(Debug, Clone, Copy)]
struct TiledPatchResult {
    tile_idx: usize,
    global_u: usize,
    global_v: usize,
    local_u: f64,
    local_v: f64,
    estimate: PatchEstimate,
}

#[derive(Debug, Clone, Copy)]
struct UndistortSample {
    idx00: usize,
    idx10: usize,
    idx01: usize,
    idx11: usize,
    weights: [f32; 4],
}

/// Undistort LUT for one pyramid level: maps each undistorted-pinhole pixel to
/// a bilinear footprint in that level's raw (distorted) image.
#[derive(Debug, Clone)]
struct UndistortLut {
    width: usize,
    height: usize,
    src_width: usize,
    src_height: usize,
    idx00: Vec<u32>,
    idx10: Vec<u32>,
    idx01: Vec<u32>,
    idx11: Vec<u32>,
    w00: Vec<f32>,
    w10: Vec<f32>,
    w01: Vec<f32>,
    w11: Vec<f32>,
    valid: Vec<u8>,
}

impl UndistortLut {
    fn from_options(lut: Vec<Option<UndistortSample>>, width: usize, height: usize) -> Self {
        Self::from_options_with_source(lut, width, height, width, height)
    }

    fn from_options_with_source(
        lut: Vec<Option<UndistortSample>>,
        width: usize,
        height: usize,
        src_width: usize,
        src_height: usize,
    ) -> Self {
        let n = lut.len();
        debug_assert_eq!(n, width * height);
        let mut out = Self {
            width,
            height,
            src_width,
            src_height,
            idx00: vec![0; n],
            idx10: vec![0; n],
            idx01: vec![0; n],
            idx11: vec![0; n],
            w00: vec![0.0; n],
            w10: vec![0.0; n],
            w01: vec![0.0; n],
            w11: vec![0.0; n],
            valid: vec![0; n],
        };
        for (i, sample) in lut.into_iter().enumerate() {
            if let Some(s) = sample {
                out.idx00[i] = s.idx00 as u32;
                out.idx10[i] = s.idx10 as u32;
                out.idx01[i] = s.idx01 as u32;
                out.idx11[i] = s.idx11 as u32;
                out.w00[i] = s.weights[0];
                out.w10[i] = s.weights[1];
                out.w01[i] = s.weights[2];
                out.w11[i] = s.weights[3];
                out.valid[i] = 1;
            }
        }
        out
    }

    /// Undistort one raw pyramid level into a pinhole `Image<f32>`. Invalid
    /// pixels (raw footprint out of bounds) are left at zero.
    fn undistort_level(&self, raw: &Image<f32>) -> Image<f32> {
        debug_assert_eq!(raw.width(), self.src_width);
        debug_assert_eq!(raw.height(), self.src_height);
        debug_assert_eq!(raw.stride(), raw.width());
        let src = raw.as_slice();
        let n = self.width * self.height;
        let mut dst = vec![0.0f32; n];
        for i in 0..n {
            if self.valid[i] == 0 {
                continue;
            }
            let i00 = src[self.idx00[i] as usize];
            let i10 = src[self.idx10[i] as usize];
            let i01 = src[self.idx01[i] as usize];
            let i11 = src[self.idx11[i] as usize];
            dst[i] = self.w00[i] * i00 + self.w10[i] * i10 + self.w01[i] * i01 + self.w11[i] * i11;
        }
        Image::from_vec(self.width, self.height, dst)
    }

    /// Static validity mask for this level (1.0 where the pinhole pixel has a
    /// fully in-bounds raw footprint). Frame-invariant — the raw image has no
    /// holes and the undistort geometry is fixed.
    fn valid_image(&self) -> Image<f32> {
        let data = self
            .valid
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect();
        Image::from_vec(self.width, self.height, data)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingTile {
    level: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    center_u: f64,
    center_v: f64,
    center_bearing: Vector3<f64>,
    tangent_u: Vector3<f64>,
    tangent_v: Vector3<f64>,
    focal: f64,
    lut: UndistortLut,
}

#[allow(dead_code)]
impl TiledBearingTile {
    fn contains_patch(&self, cu: f64, cv: f64, half: usize) -> bool {
        let u_min = cu - half as f64;
        let v_min = cv - half as f64;
        let u_max = cu + half as f64;
        let v_max = cv + half as f64;
        u_min >= self.x0 as f64
            && v_min >= self.y0 as f64
            && u_max < (self.x0 + self.width) as f64
            && v_max < (self.y0 + self.height) as f64
    }

    fn bearing_at_level_pixel(&self, u: f64, v: f64) -> Vector3<f64> {
        let x = (u - self.center_u) / self.focal;
        let y = (v - self.center_v) / self.focal;
        (self.center_bearing + self.tangent_u * x + self.tangent_v * y).normalize()
    }

    fn project_to_level_pixel(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        let z = p.dot(&self.center_bearing);
        if z <= 1e-9 {
            return None;
        }
        let x = p.dot(&self.tangent_u) / z;
        let y = p.dot(&self.tangent_v) / z;
        Some(Vector2::new(
            self.center_u + self.focal * x,
            self.center_v + self.focal * y,
        ))
    }

    fn project_to_local_pixel(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        self.project_to_level_pixel(p)
            .map(|uv| self.global_to_local(uv))
    }

    fn projection_jacobian(&self, p: &Vector3<f64>) -> Option<nalgebra::Matrix2x3<f64>> {
        let z = p.dot(&self.center_bearing);
        if z <= 1e-9 {
            return None;
        }
        let x = p.dot(&self.tangent_u);
        let y = p.dot(&self.tangent_v);
        let z2 = z * z;
        let row_u = (self.tangent_u.transpose() * z - self.center_bearing.transpose() * x)
            * (self.focal / z2);
        let row_v = (self.tangent_v.transpose() * z - self.center_bearing.transpose() * y)
            * (self.focal / z2);
        Some(nalgebra::Matrix2x3::from_rows(&[row_u, row_v]))
    }

    fn warp_local_pixel(
        &self,
        u_local: f64,
        v_local: f64,
        rho: f64,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
        if rho <= 0.0 {
            return None;
        }
        let u = self.x0 as f64 + u_local;
        let v = self.y0 as f64 + v_local;
        let bearing = self.bearing_at_level_pixel(u, v);
        let x_curr = bearing / rho;
        let x_ref = rel_pose.r * x_curr + rel_pose.t;
        let uv_ref = self.project_to_local_pixel(&x_ref)?;
        Some((uv_ref[0], uv_ref[1], x_ref, bearing))
    }

    fn contains_point(&self, u: f64, v: f64) -> bool {
        u >= self.x0 as f64
            && v >= self.y0 as f64
            && u < (self.x0 + self.width) as f64
            && v < (self.y0 + self.height) as f64
    }

    fn global_to_local(&self, uv: Vector2<f64>) -> Vector2<f64> {
        Vector2::new(uv[0] - self.x0 as f64, uv[1] - self.y0 as f64)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingLevel {
    level: usize,
    width: usize,
    height: usize,
    tiles: Vec<TiledBearingTile>,
}

#[allow(dead_code)]
impl TiledBearingLevel {
    fn owning_tile_for_patch(&self, cu: f64, cv: f64, half: usize) -> Option<usize> {
        self.tiles
            .iter()
            .position(|tile| tile.contains_patch(cu, cv, half))
            .or_else(|| {
                self.tiles
                    .iter()
                    .position(|tile| tile.contains_point(cu, cv))
            })
    }

    fn patch_centers_in_tile(
        &self,
        tile_idx: usize,
        settings: &PatchDepthSettings,
    ) -> Vec<(usize, usize)> {
        let Some(tile) = self.tiles.get(tile_idx) else {
            return Vec::new();
        };
        let half = settings.patch_size / 2;
        let mut centers = Vec::new();
        for v in (half..self.height.saturating_sub(half)).step_by(settings.patch_stride) {
            for u in (half..self.width.saturating_sub(half)).step_by(settings.patch_stride) {
                if tile.contains_patch(u as f64, v as f64, half) {
                    centers.push((u, v));
                }
            }
        }
        centers
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingImageTile {
    tile: TiledBearingTile,
    image: Image<f32>,
    valid: Image<f32>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingFrameLevel {
    level: usize,
    width: usize,
    height: usize,
    tiles: Vec<TiledBearingImageTile>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingFrameProducts {
    frame: Arc<FrameProducts>,
    raw_pyramid: Arc<Vec<Image<f32>>>,
    levels: Vec<TiledBearingFrameLevel>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingKeyframeTile {
    tile: TiledBearingTile,
    ref_image: Image<f32>,
    bilinear_valid: Image<f32>,
    grad_x: Image<f32>,
    grad_y: Image<f32>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingKeyframeLevel {
    level: usize,
    width: usize,
    height: usize,
    tiles: Vec<TiledBearingKeyframeTile>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TiledBearingKeyframe {
    frame: Arc<FrameProducts>,
    levels: Vec<TiledBearingKeyframeLevel>,
}

/// Level-0 image context for per-pixel photometric weighting during tiled fusion.
/// All images are tile-local crops; coordinates passed alongside are tile-local.
struct TiledPhotoContext<'a> {
    curr_img: &'a Image<f32>,
    curr_valid: &'a Image<f32>,
    ref_tile: &'a TiledBearingTile,
    ref_img: &'a Image<f32>,
    ref_valid: &'a Image<f32>,
    rel_pose: &'a RelativePose,
}

#[allow(dead_code)]
fn build_tiled_bearing_levels(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Vec<TiledBearingLevel> {
    let specs = undistort_level_specs(width, height, scale, levels);
    specs
        .iter()
        .enumerate()
        .map(|(level, spec)| {
            let tiles = build_tiled_bearing_tiles_for_level(
                camera,
                intrinsics,
                spec,
                level,
                tile_size,
                tile_overlap,
            );
            TiledBearingLevel {
                level,
                width: spec.lw,
                height: spec.lh,
                tiles,
            }
        })
        .collect()
}

#[allow(dead_code)]
fn build_tiled_bearing_frame_levels(
    layout: &[TiledBearingLevel],
    raw_pyramid: &[Image<f32>],
) -> Vec<TiledBearingFrameLevel> {
    layout
        .iter()
        .zip(raw_pyramid)
        .map(|(level, raw)| {
            let tiles = level
                .tiles
                .iter()
                .map(|tile| TiledBearingImageTile {
                    tile: tile.clone(),
                    image: tile.lut.undistort_level(raw),
                    valid: tile.lut.valid_image(),
                })
                .collect();
            TiledBearingFrameLevel {
                level: level.level,
                width: level.width,
                height: level.height,
                tiles,
            }
        })
        .collect()
}

#[allow(dead_code)]
fn assign_tiled_bearing_seeds(
    level: &TiledBearingLevel,
    seeds: &[SparseDepthPrior],
    scale_from_original: f64,
    patch_half: usize,
) -> Vec<Vec<SparseDepthPrior>> {
    let mut by_tile = vec![Vec::new(); level.tiles.len()];
    for seed in seeds {
        let scaled_uv = seed.uv * scale_from_original;
        for (tile_idx, tile) in level.tiles.iter().enumerate() {
            if !tile.contains_patch(scaled_uv[0], scaled_uv[1], patch_half)
                && !tile.contains_point(scaled_uv[0], scaled_uv[1])
            {
                continue;
            }
            by_tile[tile_idx].push(SparseDepthPrior {
                uv: tile.global_to_local(scaled_uv),
                eta: seed.eta,
                eta_var: seed.eta_var,
            });
        }
    }
    by_tile
}

#[allow(dead_code)]
fn bilinear_valid_image_from_mask(mask: &Image<f32>) -> Image<f32> {
    build_bilinear_valid_pyramid(std::slice::from_ref(mask))
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            Image::from_vec(
                mask.width(),
                mask.height(),
                vec![0.0; mask.width() * mask.height()],
            )
        })
}

#[allow(dead_code)]
fn build_tiled_bearing_tiles_for_level(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Vec<TiledBearingTile> {
    let tile_size = tile_size.max(2);
    let tile_overlap = tile_overlap.min(tile_size.saturating_sub(1));
    let stride = (tile_size - tile_overlap).max(1);
    let mut tiles = Vec::new();
    let mut y0 = 0;
    loop {
        let h = tile_size.min(spec.lh - y0);
        let mut x0 = 0;
        loop {
            let w = tile_size.min(spec.lw - x0);
            tiles.push(build_tiled_bearing_tile(
                camera, intrinsics, spec, level, x0, y0, w, h,
            ));
            if x0 + w >= spec.lw {
                break;
            }
            x0 = (x0 + stride).min(spec.lw - 1);
        }
        if y0 + h >= spec.lh {
            break;
        }
        y0 = (y0 + stride).min(spec.lh - 1);
    }
    tiles
}

#[allow(dead_code)]
fn build_tiled_bearing_tile(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> TiledBearingTile {
    let center_u = x0 as f64 + 0.5 * (width.saturating_sub(1)) as f64;
    let center_v = y0 as f64 + 0.5 * (height.saturating_sub(1)) as f64;
    let raw_center_u = center_u / spec.level_scale;
    let raw_center_v = center_v / spec.level_scale;
    let center_bearing = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v))
        .normalize();
    let (tangent_u, tangent_v) = image_axis_tangent_basis(
        camera,
        &center_bearing,
        raw_center_u,
        raw_center_v,
        1.0 / spec.level_scale,
    );
    let focal = 0.5 * (intrinsics.fx + intrinsics.fy) * spec.level_scale;
    let lut = build_tiled_bearing_lut(
        camera,
        spec,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        focal,
        &center_bearing,
        &tangent_u,
        &tangent_v,
    );
    TiledBearingTile {
        level,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        center_bearing,
        tangent_u,
        tangent_v,
        focal,
        lut,
    }
}

fn build_tiled_bearing_tile_geometry_from_center(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    center_bearing: Vector3<f64>,
) -> TiledBearingTile {
    let center_u = x0 as f64 + 0.5 * (width.saturating_sub(1)) as f64;
    let center_v = y0 as f64 + 0.5 * (height.saturating_sub(1)) as f64;
    let center_bearing = center_bearing.normalize();
    let (tangent_u, tangent_v) = projection_jacobian_tangent_basis(camera, &center_bearing);
    let focal = 0.5 * (intrinsics.fx + intrinsics.fy) * spec.level_scale;
    TiledBearingTile {
        level,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        center_bearing,
        tangent_u,
        tangent_v,
        focal,
        lut: UndistortLut {
            width,
            height,
            src_width: spec.lw,
            src_height: spec.lh,
            idx00: Vec::new(),
            idx10: Vec::new(),
            idx01: Vec::new(),
            idx11: Vec::new(),
            w00: Vec::new(),
            w10: Vec::new(),
            w01: Vec::new(),
            w11: Vec::new(),
            valid: Vec::new(),
        },
    }
}

#[inline(always)]
fn sample_bilinear_raw_nomask_slice(
    data: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    u: f64,
    v: f64,
) -> Option<f32> {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return None;
    }
    let x = u as usize;
    let y = v as usize;
    let dx = (u - x as f64) as f32;
    let dy = (v - y as f64) as f32;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    unsafe {
        let r0 = data.as_ptr().add(y * stride);
        let r1 = data.as_ptr().add((y + 1) * stride);
        Some(w00 * *r0.add(x) + w10 * *r0.add(x + 1) + w01 * *r1.add(x) + w11 * *r1.add(x + 1))
    }
}

#[inline(always)]
fn sample_bilinear_raw_nomask_with_grad_slice(
    img: &[f32],
    grad_x: &[f32],
    grad_y: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    u: f64,
    v: f64,
) -> Option<(f32, f32, f32)> {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return None;
    }
    let x = u as usize;
    let y = v as usize;
    let dx = (u - x as f64) as f32;
    let dy = (v - y as f64) as f32;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    unsafe {
        let i0 = img.as_ptr().add(y * stride);
        let i1 = img.as_ptr().add((y + 1) * stride);
        let gx0 = grad_x.as_ptr().add(y * stride);
        let gx1 = grad_x.as_ptr().add((y + 1) * stride);
        let gy0 = grad_y.as_ptr().add(y * stride);
        let gy1 = grad_y.as_ptr().add((y + 1) * stride);
        let sample = |r0: *const f32, r1: *const f32| {
            w00 * *r0.add(x) + w10 * *r0.add(x + 1) + w01 * *r1.add(x) + w11 * *r1.add(x + 1)
        };
        Some((sample(i0, i1), sample(gx0, gx1), sample(gy0, gy1)))
    }
}

#[derive(Debug, Clone, Copy)]
struct PerPatchAffineMap {
    raw_center: Vector2<f64>,
    raw_du: Vector2<f64>,
    raw_dv: Vector2<f64>,
    center_u: f64,
    center_v: f64,
}

impl PerPatchAffineMap {
    fn from_tile(
        camera: &dyn CameraModel,
        spec: &image_ops::UndistortLevelSpec,
        tile: &TiledBearingTile,
    ) -> Option<Self> {
        let center_raw = camera.project_ray(&tile.center_bearing)?;
        let proj_j = camera.projection_jacobian(&tile.center_bearing);
        let level_per_tangent = spec.level_scale / tile.focal;
        Some(Self {
            raw_center: Vector2::new(
                center_raw[0] * spec.level_scale + spec.raw_offset,
                center_raw[1] * spec.level_scale + spec.raw_offset,
            ),
            raw_du: proj_j * tile.tangent_u * level_per_tangent,
            raw_dv: proj_j * tile.tangent_v * level_per_tangent,
            center_u: tile.center_u,
            center_v: tile.center_v,
        })
    }
}

#[allow(dead_code)]
fn build_tiled_bearing_lut(
    camera: &dyn CameraModel,
    spec: &image_ops::UndistortLevelSpec,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    center_u: f64,
    center_v: f64,
    focal: f64,
    center_bearing: &Vector3<f64>,
    tangent_u: &Vector3<f64>,
    tangent_v: &Vector3<f64>,
) -> UndistortLut {
    let mut samples = Vec::with_capacity(width * height);
    for v in y0..y0 + height {
        for u in x0..x0 + width {
            let x = (u as f64 - center_u) / focal;
            let y = (v as f64 - center_v) / focal;
            let bearing = (center_bearing + tangent_u * x + tangent_v * y).normalize();
            let raw_uv = camera.project(&bearing);
            let raw_u = raw_uv[0] * spec.level_scale + spec.raw_offset;
            let raw_v = raw_uv[1] * spec.level_scale + spec.raw_offset;
            samples.push(image_ops::undistort_sample(raw_u, raw_v, spec.lw, spec.lh));
        }
    }
    UndistortLut::from_options_with_source(samples, width, height, spec.lw, spec.lh)
}

#[allow(dead_code)]
fn image_axis_tangent_basis(
    camera: &dyn CameraModel,
    center_bearing: &Vector3<f64>,
    raw_center_u: f64,
    raw_center_v: f64,
    raw_step: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let b = center_bearing.normalize();
    let step = raw_step.max(1e-3);
    let bu_plus = camera
        .undistort(&Vector2::new(raw_center_u + step, raw_center_v))
        .normalize();
    let bu_minus = camera
        .undistort(&Vector2::new(raw_center_u - step, raw_center_v))
        .normalize();
    let bv_plus = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v + step))
        .normalize();
    let bv_minus = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v - step))
        .normalize();

    let du = project_to_tangent(&(bu_plus - bu_minus), &b);
    let mut tangent_u = normalize_or_fallback(du, image_axis_fallback_u(&b));
    tangent_u = project_to_tangent(&tangent_u, &b).normalize();

    let dv = project_to_tangent(&(bv_plus - bv_minus), &b);
    let dv_orthogonal = project_to_tangent(&(dv - tangent_u * dv.dot(&tangent_u)), &b);
    let mut tangent_v = normalize_or_fallback(dv_orthogonal, b.cross(&tangent_u));
    tangent_v = project_to_tangent(&tangent_v, &b).normalize();

    if tangent_v.dot(&dv) < 0.0 {
        tangent_v = -tangent_v;
    }
    (tangent_u, tangent_v)
}

fn projection_jacobian_tangent_basis(
    camera: &dyn CameraModel,
    center_bearing: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let b = center_bearing.normalize();
    let j = camera.projection_jacobian(&b);
    let du = project_to_tangent(&Vector3::new(j[(0, 0)], j[(0, 1)], j[(0, 2)]), &b);
    let mut tangent_u = normalize_or_fallback(du, image_axis_fallback_u(&b));
    tangent_u = project_to_tangent(&tangent_u, &b).normalize();

    let dv = project_to_tangent(&Vector3::new(j[(1, 0)], j[(1, 1)], j[(1, 2)]), &b);
    let dv_orthogonal = project_to_tangent(&(dv - tangent_u * dv.dot(&tangent_u)), &b);
    let mut tangent_v = normalize_or_fallback(dv_orthogonal, b.cross(&tangent_u));
    tangent_v = project_to_tangent(&tangent_v, &b).normalize();

    if tangent_v.dot(&dv) < 0.0 {
        tangent_v = -tangent_v;
    }
    (tangent_u, tangent_v)
}

fn project_to_tangent(v: &Vector3<f64>, b: &Vector3<f64>) -> Vector3<f64> {
    v - b * v.dot(b)
}

fn normalize_or_fallback(v: Vector3<f64>, fallback: Vector3<f64>) -> Vector3<f64> {
    if v.norm_squared() > 1e-18 {
        v.normalize()
    } else {
        fallback.normalize()
    }
}

fn image_axis_fallback_u(b: &Vector3<f64>) -> Vector3<f64> {
    let x_axis = Vector3::new(1.0, 0.0, 0.0);
    let projected = project_to_tangent(&x_axis, b);
    if projected.norm_squared() > 1e-18 {
        projected
    } else {
        project_to_tangent(&Vector3::new(0.0, 1.0, 0.0), b)
    }
}

fn smoothstep01(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Tiled warp at log-range η.  Tile bearings are already unit, so
/// `x_curr = unit_b · exp(η)` directly.  Returns `(u_local_ref, v_local_ref, x_ref, unit_b)`.
fn warp_tiled_local_pixel_eta(
    curr_tile: &TiledBearingTile,
    ref_tile: &TiledBearingTile,
    u_local: f64,
    v_local: f64,
    eta: f64,
    rel_pose: &RelativePose,
) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
    let u = curr_tile.x0 as f64 + u_local;
    let v = curr_tile.y0 as f64 + v_local;
    let unit_b = curr_tile.bearing_at_level_pixel(u, v);
    let x_curr = unit_b * eta.exp();
    let x_ref = rel_pose.r * x_curr + rel_pose.t;
    let uv_ref = ref_tile.project_to_local_pixel(&x_ref)?;
    Some((uv_ref[0], uv_ref[1], x_ref, unit_b))
}

pub struct PatchDepthMapper {
    camera: Arc<dyn CameraModel>,
    pub(super) camera_mode: PatchDepthCameraMode,
    pub(super) intrinsics: CameraIntrinsics,
    width: usize,
    height: usize,
    settings: PatchDepthSettings,
    bearing_lut: Vec<Vector3<f64>>,
    /// One undistort LUT per pyramid level (`UndistortedPinhole` mode only).
    undistort_luts: Option<Vec<UndistortLut>>,
    /// Frame-invariant per-level validity mask, derived from `undistort_luts`.
    valid_pyramid: Option<Arc<Vec<Image<f32>>>>,
    keyframes: Vec<DepthKeyframe>,
    pyramid_work: Pyramid,
    pyramid_scratch: PyramidScratch,
    tiled_keyframes: Vec<TiledBearingKeyframe>,
    tiled_bearing_levels: Option<Arc<Vec<TiledBearingLevel>>>,
    stereo_tiled_bearing_levels: Option<Arc<Vec<TiledBearingLevel>>>,
    /// Stereo: LUTs that rectify cam1 raw pixels into cam0's pinhole projection.
    stereo_undistort_luts: Option<Vec<UndistortLut>>,
    stereo_valid_pyramid: Option<Arc<Vec<Image<f32>>>>,
    stereo_t_c1_c0: Option<Matrix4<f64>>,
}

// ---------------------------------------------------------------------------
// Log-range depth conversion helpers.
//
// Internally the GN solver optimises η = ln(range), where range is the
// Euclidean distance from the camera to the 3D point.  All warp paths use
// unit bearings: x_curr = unit_b · exp(η).
//
// Seed conversion (rho_to_eta) uses bearing_norm = ‖b_non_unit‖ = 1/unit_b.z:
//   pinhole (rho = 1/z, b = [bx,by,1]): bearing_norm = ‖b‖, η = ln(‖b‖/ρ) = ln(range)
//   tiled   (rho = 1/range, unit b):     bearing_norm = 1,   η = ln(1/ρ)   = ln(range)
//
// This avoids the 1/ρ² singularity in the Jacobian near large depths.
// ---------------------------------------------------------------------------

/// Converts inverse depth ρ to log-range η = ln(range).
///
/// `range_per_z` encodes the rho semantics:
/// - Inverse-z prior (`rho = 1/z`, pinhole bearing `b = [bx,by,1]`):
///   pass `range_per_z = norm(b) = 1/unit_b.z`. Then `eta = ln(norm/rho) = ln(range)`.
/// - Inverse-range prior (`rho = 1/range`, unit bearing):
///   pass `range_per_z = 1.0`. Then `eta = ln(1/rho) = ln(range)`.
///
/// Do NOT pass `unit_b.norm()` (= 1.0) when the prior is inverse-z — that
/// conflates the two cases and silently drops the `1/unit_b.z` factor.
/// For omnidirectional models where `unit_b.z <= 0`, inverse-z priors are
/// undefined; use range priors (`query_range`) directly.
pub(crate) fn rho_to_eta(rho: f64, range_per_z: f64) -> f64 {
    (range_per_z / rho).ln()
}

/// Propagates inverse-depth variance to log-range variance via the delta method.
/// Holds for both inverse-z and inverse-range: dη/dρ = −1/ρ → var(η) ≈ var(ρ)/ρ².
pub(crate) fn rho_var_to_eta_var(rho: f64, rho_var: f64) -> f64 {
    rho_var / (rho * rho)
}

// Eigenvalue test for a 2×2 symmetric structure tensor [[gxx, gxy], [gxy, gyy]].
// Returns true if the patch has sufficient texture for reliable depth estimation.
// Inputs must be already normalized (divided by pixel count).
//
// When `epipolar` is Some, the min_eigen check uses the directional curvature
// ê^T S ê rather than λ_min. An edge perpendicular to the epipolar line has
// ê^T S ê ≈ λ_max even though λ_min ≈ 0, so it correctly passes: that edge
// provides all the depth information the photometric solver needs. The condition
// number check is skipped when the epipolar direction is known, because only the
// epipolar direction matters for the solve. When epipolar is None (degenerate
// camera motion or unknown direction), both the λ_min and condition checks are
// applied as a conservative fallback.
pub(crate) fn structure_tensor_passes(
    gxx: f64,
    gxy: f64,
    gyy: f64,
    min_eigen: f64,
    max_condition: f64,
    epipolar: Option<(f64, f64)>,
) -> bool {
    let trace = gxx + gyy;
    let discr = ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt();
    let lambda_min = 0.5 * (trace - discr);
    let lambda_max = 0.5 * (trace + discr);

    if let Some((ex, ey)) = epipolar {
        let h_epi = gxx * ex * ex + 2.0 * gxy * ex * ey + gyy * ey * ey;
        if min_eigen > 0.0 && h_epi < min_eigen {
            return false;
        }
    } else {
        if min_eigen > 0.0 && lambda_min < min_eigen {
            return false;
        }
        if max_condition > 0.0 {
            if lambda_min <= 1e-12 {
                return false;
            }
            if lambda_max / lambda_min > max_condition {
                return false;
            }
        }
    }
    true
}

/// Build the working-scale image pyramid from raw `u8` gray, reusing Rudolf-V's
/// `Pyramid`/scratch buffers. Dyadic scales take Rudolf-V's `build_reuse` fast
/// path and slice out the kept levels; non-dyadic scales fall back to
/// `build_pyramid_from_u8`.
pub(super) fn build_working_pyramid(
    pyramid_work: &mut Pyramid,
    pyramid_scratch: &mut PyramidScratch,
    gray: &[u8],
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
) -> Vec<Image<f32>> {
    if let Some(offset) = dyadic_scale_offset(scale) {
        let src = Image::from_vec(width, height, gray.to_vec());
        pyramid_work.build_reuse(&src, offset + levels, pyramid_scratch);
        return pyramid_work.levels[offset..offset + levels].to_vec();
    }
    build_pyramid_from_u8(gray, width, height, scale, levels)
}

impl PatchDepthMapper {
    pub fn new(
        camera: Arc<dyn CameraModel>,
        intrinsics: CameraIntrinsics,
        width: usize,
        height: usize,
        settings: PatchDepthSettings,
    ) -> anyhow::Result<Self> {
        let camera_mode = settings.camera_mode;
        Self::new_with_mode(camera, intrinsics, width, height, settings, camera_mode)
    }

    pub fn new_undistorted_pinhole(
        raw_camera: Arc<dyn CameraModel>,
        intrinsics: CameraIntrinsics,
        width: usize,
        height: usize,
        settings: PatchDepthSettings,
    ) -> anyhow::Result<Self> {
        let mut settings = settings;
        settings.camera_mode = PatchDepthCameraMode::UndistortedPinhole;
        Self::new_with_mode(
            raw_camera,
            intrinsics,
            width,
            height,
            settings,
            PatchDepthCameraMode::UndistortedPinhole,
        )
    }

    pub fn camera_mode(&self) -> PatchDepthCameraMode {
        self.camera_mode
    }

    pub fn expected_seed_coordinates(&self) -> PatchDepthSeedCoordinates {
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                PatchDepthSeedCoordinates::RawDistorted
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                PatchDepthSeedCoordinates::UndistortedPinhole
            }
        }
    }

    fn new_with_mode(
        camera: Arc<dyn CameraModel>,
        intrinsics: CameraIntrinsics,
        width: usize,
        height: usize,
        settings: PatchDepthSettings,
        camera_mode: PatchDepthCameraMode,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "image size must be non-zero");
        anyhow::ensure!(
            settings.scale > 0.0 && settings.scale <= 1.0,
            "scale must be in (0, 1]"
        );
        anyhow::ensure!(
            settings.n_pyramid_levels > 0,
            "n_pyramid_levels must be positive"
        );
        anyhow::ensure!(
            camera_mode != PatchDepthCameraMode::TiledBearing
                || settings.warp_mode == PatchDepthWarpMode::FastTranslation,
            "PatchDepth camera_mode=tiled_bearing requires warp_mode=fast_translation"
        );
        anyhow::ensure!(
            camera_mode != PatchDepthCameraMode::TiledBearing
                || settings.tiled_tile_overlap > settings.patch_size,
            "PatchDepth camera_mode=tiled_bearing requires tiled_tile_overlap > patch_size for overlap blending"
        );

        let bearing_lut = match camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                let mut lut = Vec::with_capacity(width * height);
                for v in 0..height {
                    for u in 0..width {
                        let uv = Vector2::new(u as f64, v as f64);
                        let bearing = camera.undistort(&uv);
                        if bearing[2].abs() > 1e-12 {
                            lut.push(bearing / bearing[2]);
                        } else {
                            lut.push(Vector3::new(0.0, 0.0, 1.0));
                        }
                    }
                }
                lut
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                Vec::new()
            }
        };

        let undistort_luts = match camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => None,
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let specs =
                    undistort_level_specs(width, height, settings.scale, settings.n_pyramid_levels);
                let luts = specs
                    .iter()
                    .map(|spec| {
                        UndistortLut::from_options(
                            build_pinhole_to_raw_lut(camera.as_ref(), &intrinsics, spec),
                            spec.lw,
                            spec.lh,
                        )
                    })
                    .collect();
                Some(luts)
            }
        };
        // The raw image has no holes and the undistort geometry is fixed, so
        // each level's validity mask is the same every frame — build it once.
        let valid_pyramid = undistort_luts.as_ref().map(|luts: &Vec<UndistortLut>| {
            Arc::new(
                luts.iter()
                    .map(UndistortLut::valid_image)
                    .collect::<Vec<_>>(),
            )
        });
        let tiled_bearing_levels = if camera_mode == PatchDepthCameraMode::TiledBearing {
            Some(Arc::new(build_tiled_bearing_levels(
                camera.as_ref(),
                &intrinsics,
                width,
                height,
                settings.scale,
                settings.n_pyramid_levels,
                settings.tiled_tile_size,
                settings.tiled_tile_overlap,
            )))
        } else {
            None
        };

        Ok(Self {
            camera,
            camera_mode,
            intrinsics,
            width,
            height,
            settings,
            bearing_lut,
            undistort_luts,
            valid_pyramid,
            keyframes: Vec::new(),
            pyramid_work: empty_pyramid(),
            pyramid_scratch: PyramidScratch::new(width, height, 1.0),
            tiled_keyframes: Vec::new(),
            tiled_bearing_levels,
            stereo_tiled_bearing_levels: None,
            stereo_undistort_luts: None,
            stereo_valid_pyramid: None,
            stereo_t_c1_c0: None,
        })
    }

    pub fn update(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        seed_coordinates: PatchDepthSeedCoordinates,
        frame: FrameProducts,
    ) -> Option<PatchDepthOutput> {
        if seed_coordinates != self.expected_seed_coordinates() {
            return None;
        }
        let seeds = self.gather_seeds(sparse_filter, measurement);
        self.update_with_priors(frame, &seeds, None, 0.0)
    }

    pub fn update_with_priors(
        &mut self,
        frame: FrameProducts,
        seeds: &[SparseDepthPrior],
        p_vv: Option<&Matrix3<f64>>,
        dt: f64,
    ) -> Option<PatchDepthOutput> {
        if frame.width != self.width
            || frame.height != self.height
            || frame.gray.len() != self.width * self.height
        {
            return None;
        }
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            return self.update_with_priors_tiled_bearing(frame, seeds, p_vv, dt);
        }

        let depth_frame = self.depth_frame_products(frame)?;
        let median_depth = median_seed_depth(seeds).unwrap_or(self.settings.max_depth);
        let selected = self.select_keyframe(&depth_frame.frame.pose_t_wc, median_depth);
        self.manage_keyframes(&depth_frame, median_depth);
        let (ref_keyframe, t_ref_curr) = selected?;
        let sigma_warp_sq =
            compute_sigma_warp_sq(&self.intrinsics, &t_ref_curr, p_vv, dt, median_depth);
        Some(self.solve(
            &depth_frame,
            &ref_keyframe,
            &t_ref_curr,
            seeds,
            sigma_warp_sq,
        ))
    }

    fn update_with_priors_tiled_bearing(
        &mut self,
        frame: FrameProducts,
        seeds: &[SparseDepthPrior],
        p_vv: Option<&Matrix3<f64>>,
        dt: f64,
    ) -> Option<PatchDepthOutput> {
        let depth_frame = self.tiled_bearing_frame_products(frame)?;
        let median_depth = median_seed_depth(seeds).unwrap_or(self.settings.max_depth);
        let selected = self.select_tiled_keyframe(&depth_frame.frame.pose_t_wc, median_depth);
        self.manage_tiled_keyframes(&depth_frame, median_depth);
        let (ref_keyframe, t_ref_curr) = selected?;
        let sigma_warp_sq =
            compute_sigma_warp_sq(&self.intrinsics, &t_ref_curr, p_vv, dt, median_depth);
        Some(self.solve_tiled_bearing(
            &depth_frame,
            &ref_keyframe,
            &t_ref_curr,
            seeds,
            sigma_warp_sq,
        ))
    }

    /// Set up stereo reference frame support. Builds rectification LUTs that
    /// map cam0-pinhole pixels to cam1 raw pixels, so cam1 images can be used
    /// as reference frames with the existing warp/project pipeline.
    ///
    /// `cam1_model` must use the same resolution as cam0.
    pub fn init_stereo_ref(&mut self, cam1_model: &dyn CameraModel, t_c1_c0: Matrix4<f64>) {
        self.stereo_t_c1_c0 = Some(t_c1_c0);
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            self.stereo_tiled_bearing_levels = Some(Arc::new(build_tiled_bearing_levels(
                cam1_model,
                &self.intrinsics,
                self.width,
                self.height,
                self.settings.scale,
                self.settings.n_pyramid_levels,
                self.settings.tiled_tile_size,
                self.settings.tiled_tile_overlap,
            )));
            return;
        }
        if self.camera_mode != PatchDepthCameraMode::UndistortedPinhole {
            self.stereo_t_c1_c0 = None;
            return;
        }
        let specs = undistort_level_specs(
            self.width,
            self.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let luts: Vec<UndistortLut> = specs
            .iter()
            .map(|spec| {
                UndistortLut::from_options(
                    build_pinhole_to_raw_lut(cam1_model, &self.intrinsics, spec),
                    spec.lw,
                    spec.lh,
                )
            })
            .collect();
        let valid = Arc::new(
            luts.iter()
                .map(UndistortLut::valid_image)
                .collect::<Vec<_>>(),
        );
        self.stereo_undistort_luts = Some(luts);
        self.stereo_valid_pyramid = Some(valid);
    }

    pub fn has_stereo_ref(&self) -> bool {
        self.stereo_undistort_luts.is_some() || self.stereo_tiled_bearing_levels.is_some()
    }

    /// Use cam1 as the reference frame instead of the motion-based keyframe pool.
    pub fn update_with_stereo_ref(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        seed_coordinates: PatchDepthSeedCoordinates,
        frame: FrameProducts,
        cam1_gray: &[u8],
        cam1_width: usize,
        cam1_height: usize,
    ) -> Option<PatchDepthOutput> {
        if seed_coordinates != self.expected_seed_coordinates() {
            return None;
        }
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            return self.update_with_stereo_ref_tiled_bearing(
                sparse_filter,
                measurement,
                frame,
                cam1_gray,
                cam1_width,
                cam1_height,
            );
        }
        let stereo_luts = self.stereo_undistort_luts.as_ref()?;
        let t_c1_c0 = self.stereo_t_c1_c0?;

        let raw_cam1_pyr = build_pyramid_from_u8(
            cam1_gray,
            cam1_width,
            cam1_height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let rectified_pyr: Vec<Image<f32>> = raw_cam1_pyr
            .iter()
            .zip(stereo_luts)
            .map(|(raw, lut)| lut.undistort_level(raw))
            .collect();
        let ref_valid_pyr = self.stereo_valid_pyramid.clone();
        let bv_pyr = ref_valid_pyr
            .as_ref()
            .map(|p| build_bilinear_valid_pyramid(p));
        let mut grad_x_pyr = Vec::with_capacity(self.settings.n_pyramid_levels);
        let mut grad_y_pyr = Vec::with_capacity(self.settings.n_pyramid_levels);
        for img in &rectified_pyr {
            let (gx, gy) = gradients(img);
            grad_x_pyr.push(gx);
            grad_y_pyr.push(gy);
        }
        let ref_keyframe = DepthKeyframe {
            frame: Arc::new(FrameProducts {
                frame_id: 0,
                stamp: frame.stamp,
                gray: Vec::new(),
                width: cam1_width,
                height: cam1_height,
                pose_t_wc: Matrix4::identity(),
            }),
            ref_pyramid: Arc::new(rectified_pyr),
            bilinear_valid_pyramid: bv_pyr.map(Arc::new),
            grad_x_pyramid: grad_x_pyr,
            grad_y_pyramid: grad_y_pyr,
        };

        let seeds = self.gather_seeds(sparse_filter, measurement);
        let depth_frame = self.depth_frame_products(frame)?;
        let median_depth = median_seed_depth(&seeds).unwrap_or(self.settings.max_depth);
        self.manage_keyframes(&depth_frame, median_depth);

        Some(self.solve(&depth_frame, &ref_keyframe, &t_c1_c0, &seeds, 0.0))
    }

    fn update_with_stereo_ref_tiled_bearing(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        frame: FrameProducts,
        cam1_gray: &[u8],
        cam1_width: usize,
        cam1_height: usize,
    ) -> Option<PatchDepthOutput> {
        if cam1_width != self.width
            || cam1_height != self.height
            || cam1_gray.len() != self.width * self.height
        {
            return None;
        }
        let stereo_layout = Arc::clone(self.stereo_tiled_bearing_levels.as_ref()?);
        let t_c1_c0 = self.stereo_t_c1_c0?;
        let seeds = self.gather_seeds(sparse_filter, measurement);
        let depth_frame = self.tiled_bearing_frame_products(frame)?;

        let raw_cam1_pyramid = Arc::new(build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            cam1_gray,
            cam1_width,
            cam1_height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        ));
        let cam1_levels =
            build_tiled_bearing_frame_levels(stereo_layout.as_ref(), raw_cam1_pyramid.as_ref());
        let cam1_frame_products = TiledBearingFrameProducts {
            frame: Arc::new(FrameProducts {
                frame_id: 0,
                stamp: depth_frame.frame.stamp,
                gray: Vec::new(),
                width: cam1_width,
                height: cam1_height,
                pose_t_wc: Matrix4::identity(),
            }),
            raw_pyramid: raw_cam1_pyramid,
            levels: cam1_levels,
        };
        let ref_keyframe = self.make_tiled_bearing_keyframe(&cam1_frame_products);
        let median_depth = median_seed_depth(&seeds).unwrap_or(self.settings.max_depth);
        self.manage_tiled_keyframes(&depth_frame, median_depth);

        Some(self.solve_tiled_bearing(&depth_frame, &ref_keyframe, &t_c1_c0, &seeds, 0.0))
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    fn depth_frame_products(&mut self, frame: FrameProducts) -> Option<DepthFrameProducts> {
        // Build the pyramid on the raw image, then undistort each kept level.
        // Undistorting the small downsampled levels instead of the full-res
        // frame is ~16x less work; the anti-alias blur happens in raw space.
        let raw_pyramid = build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let (pyramid, valid_pyramid) = match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                (raw_pyramid, None)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let luts = self.undistort_luts.as_ref()?;
                let undistorted: Vec<Image<f32>> = raw_pyramid
                    .iter()
                    .zip(luts)
                    .map(|(raw, lut)| lut.undistort_level(raw))
                    .collect();
                (undistorted, self.valid_pyramid.clone())
            }
        };
        Some(DepthFrameProducts {
            frame: Arc::new(frame),
            pyramid: Arc::new(pyramid),
            valid_pyramid,
        })
    }

    #[allow(dead_code)]
    fn tiled_bearing_frame_products(
        &mut self,
        frame: FrameProducts,
    ) -> Option<TiledBearingFrameProducts> {
        let layout = Arc::clone(self.tiled_bearing_levels.as_ref()?);
        let raw_pyramid = Arc::new(build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        ));
        let levels = build_tiled_bearing_frame_levels(layout.as_ref(), raw_pyramid.as_ref());
        Some(TiledBearingFrameProducts {
            frame: Arc::new(frame),
            raw_pyramid,
            levels,
        })
    }

    #[allow(dead_code)]
    fn make_tiled_bearing_keyframe(
        &self,
        frame_products: &TiledBearingFrameProducts,
    ) -> TiledBearingKeyframe {
        let levels = frame_products
            .levels
            .iter()
            .map(|level| {
                let tiles = level
                    .tiles
                    .iter()
                    .map(|tile| {
                        let (grad_x, grad_y) = gradients(&tile.image);
                        TiledBearingKeyframeTile {
                            tile: tile.tile.clone(),
                            ref_image: tile.image.clone(),
                            bilinear_valid: bilinear_valid_image_from_mask(&tile.valid),
                            grad_x,
                            grad_y,
                        }
                    })
                    .collect();
                TiledBearingKeyframeLevel {
                    level: level.level,
                    width: level.width,
                    height: level.height,
                    tiles,
                }
            })
            .collect();
        TiledBearingKeyframe {
            frame: Arc::clone(&frame_products.frame),
            levels,
        }
    }

    fn select_tiled_keyframe(
        &self,
        t_wc: &Matrix4<f64>,
        median_depth: f64,
    ) -> Option<(TiledBearingKeyframe, Matrix4<f64>)> {
        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let max_bl = self.settings.max_baseline_ratio * median_depth;
        let mut best: Option<(TiledBearingKeyframe, Matrix4<f64>, f64)> = None;

        for keyframe in &self.tiled_keyframes {
            let t_ref_curr = keyframe
                .frame
                .pose_t_wc
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
                * t_wc;
            let baseline = t_ref_curr.fixed_view::<3, 1>(0, 3).norm();
            if baseline >= min_bl
                && baseline <= max_bl
                && best.as_ref().map(|(_, _, b)| baseline > *b).unwrap_or(true)
            {
                best = Some((keyframe.clone(), t_ref_curr, baseline));
            }
        }

        best.map(|(kf, t, _)| (kf, t))
    }

    fn manage_tiled_keyframes(
        &mut self,
        depth_frame: &TiledBearingFrameProducts,
        median_depth: f64,
    ) {
        if self.tiled_keyframes.len() < 2 {
            self.tiled_keyframes
                .push(self.make_tiled_bearing_keyframe(depth_frame));
            return;
        }

        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let newest = &self.tiled_keyframes[self.tiled_keyframes.len() - 1];
        let t_new_curr = newest
            .frame
            .pose_t_wc
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
            * depth_frame.frame.pose_t_wc;
        let baseline = t_new_curr.fixed_view::<3, 1>(0, 3).norm();
        if baseline >= min_bl {
            self.tiled_keyframes.remove(0);
            self.tiled_keyframes
                .push(self.make_tiled_bearing_keyframe(depth_frame));
        }
    }

    fn gather_seeds(
        &self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
    ) -> Vec<SparseDepthPrior> {
        let intr_orig = ScaledIntrinsics {
            scale_from_original: 1.0,
        };
        let is_tiled = self.camera_mode == PatchDepthCameraMode::TiledBearing;
        let mut fids: Vec<u64> = measurement.cam_coordinates.keys().copied().collect();
        fids.sort_unstable();
        let mut seeds = Vec::new();
        for fid in fids {
            let uv_f32 = &measurement.cam_coordinates[&fid];
            let (depth, depth_var) = if is_tiled {
                sparse_filter.query_range(fid)
            } else {
                sparse_filter.query(fid)
            };
            if depth <= 0.0
                || depth < self.settings.min_depth
                || depth > self.settings.max_depth
                || !depth_var.is_finite()
            {
                continue;
            }
            let uv = Vector2::new(uv_f32[0] as f64, uv_f32[1] as f64);
            // range_per_z = ‖bearing‖; for tiled seeds (rho = 1/range) this is 1.0.
            let range_per_z = if is_tiled {
                1.0
            } else {
                self.bearing_for_scaled_pixel(uv[0], uv[1], &intr_orig)
                    .map(|b| b.norm())
                    .unwrap_or(1.0)
            };
            let rho = 1.0 / depth;
            let rho_var = depth_var / depth.powi(4);
            let eta = rho_to_eta(rho, range_per_z);
            let eta_var = rho_var_to_eta_var(rho, rho_var);
            if eta.is_finite() && eta_var.is_finite() && eta_var > 0.0 {
                seeds.push(SparseDepthPrior { uv, eta, eta_var });
            }
        }
        seeds
    }

    fn select_keyframe(
        &self,
        t_wc: &Matrix4<f64>,
        median_depth: f64,
    ) -> Option<(DepthKeyframe, Matrix4<f64>)> {
        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let max_bl = self.settings.max_baseline_ratio * median_depth;
        let mut best: Option<(DepthKeyframe, Matrix4<f64>, f64)> = None;

        for keyframe in &self.keyframes {
            let t_ref_curr = keyframe
                .frame
                .pose_t_wc
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
                * t_wc;
            let baseline = t_ref_curr.fixed_view::<3, 1>(0, 3).norm();
            if baseline >= min_bl
                && baseline <= max_bl
                && best.as_ref().map(|(_, _, b)| baseline > *b).unwrap_or(true)
            {
                best = Some((keyframe.clone(), t_ref_curr, baseline));
            }
        }

        best.map(|(kf, t, _)| (kf, t))
    }

    fn manage_keyframes(&mut self, depth_frame: &DepthFrameProducts, median_depth: f64) {
        if self.keyframes.len() < 2 {
            self.keyframes.push(self.make_keyframe(depth_frame));
            return;
        }

        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let newest = &self.keyframes[self.keyframes.len() - 1];
        let t_new_curr = newest
            .frame
            .pose_t_wc
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
            * depth_frame.frame.pose_t_wc;
        let baseline = t_new_curr.fixed_view::<3, 1>(0, 3).norm();
        if baseline >= min_bl {
            self.keyframes.remove(0);
            self.keyframes.push(self.make_keyframe(depth_frame));
        }
    }

    fn make_keyframe(&self, depth_frame: &DepthFrameProducts) -> DepthKeyframe {
        let bilinear_valid_pyramid = depth_frame
            .valid_pyramid
            .as_ref()
            .map(|pyramid| build_bilinear_valid_pyramid(pyramid));
        let mut grad_x_pyramid = Vec::with_capacity(self.settings.n_pyramid_levels);
        let mut grad_y_pyramid = Vec::with_capacity(self.settings.n_pyramid_levels);
        for img in depth_frame.pyramid.iter() {
            let (gx, gy) = gradients(img);
            grad_x_pyramid.push(gx);
            grad_y_pyramid.push(gy);
        }
        DepthKeyframe {
            frame: Arc::clone(&depth_frame.frame),
            ref_pyramid: Arc::clone(&depth_frame.pyramid),
            bilinear_valid_pyramid: bilinear_valid_pyramid.map(Arc::new),
            grad_x_pyramid,
            grad_y_pyramid,
        }
    }

    fn solve(
        &self,
        depth_frame: &DepthFrameProducts,
        ref_keyframe: &DepthKeyframe,
        t_ref_curr: &Matrix4<f64>,
        seeds: &[SparseDepthPrior],
        sigma_warp_sq: f64,
    ) -> PatchDepthOutput {
        let curr_pyramid = depth_frame.pyramid.as_ref();
        let curr_valid_pyramid = depth_frame.valid_pyramid.as_ref().map(|p| p.as_slice());
        let width = curr_pyramid[0].width();
        let height = curr_pyramid[0].height();
        let scaled_intrinsics =
            scaled_intrinsics(self.settings.scale, self.settings.n_pyramid_levels);
        let scaled_seeds = scale_seeds(seeds, self.settings.scale);
        let seed_grid = SeedGrid::new(
            &scaled_seeds,
            self.settings.seed_radius_px * self.settings.scale,
            width,
            height,
        );

        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let ref_img = &ref_keyframe.ref_pyramid[0];
        let ref_valid: Option<&Image<f32>> =
            ref_keyframe.bilinear_valid_pyramid.as_ref().map(|p| &p[0]);

        let curr_img = &curr_pyramid[0];
        let curr_valid = curr_valid_pyramid.map(|p| &p[0]);

        #[cfg(feature = "parallel")]
        {
            let patch_centers_list = patch_centers(width, height, &self.settings);
            let mut grid = PatchGrid::new(width, height, &self.settings);
            // Target ~8 chunks per thread; degrades gracefully when patches < threads.
            let min_len = (patch_centers_list.len() / (rayon::current_num_threads() * 8)).max(1);
            let estimates: Vec<_> = patch_centers_list
                .par_iter()
                .with_min_len(min_len)
                .map(|&(u, v)| {
                    let estimate = self.solve_one_patch_dispatch(
                        u as f64,
                        v as f64,
                        &scaled_seeds,
                        &seed_grid,
                        curr_pyramid,
                        curr_valid_pyramid,
                        ref_keyframe,
                        &scaled_intrinsics,
                        t_ref_curr,
                        &rel_pose,
                        sigma_warp_sq,
                    );
                    (u, v, estimate)
                })
                .collect();
            for (u, v, estimate) in estimates {
                grid.set(u, v, estimate);
            }
            densify_pixels_parallel(
                &grid,
                width,
                height,
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                self,
                &scaled_intrinsics[0],
                &rel_pose,
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut grid = PatchGrid::new(width, height, &self.settings);
            let half = self.settings.patch_size / 2;
            for v in (half..height.saturating_sub(half)).step_by(self.settings.patch_stride) {
                for u in (half..width.saturating_sub(half)).step_by(self.settings.patch_stride) {
                    let estimate = self.solve_one_patch_dispatch(
                        u as f64,
                        v as f64,
                        &scaled_seeds,
                        &seed_grid,
                        curr_pyramid,
                        curr_valid_pyramid,
                        ref_keyframe,
                        &scaled_intrinsics,
                        t_ref_curr,
                        &rel_pose,
                        sigma_warp_sq,
                    );
                    grid.set(u, v, estimate);
                }
            }
            densify_pixels(
                &grid,
                width,
                height,
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                self,
                &scaled_intrinsics[0],
                &rel_pose,
            )
        }
    }

    fn solve_tiled_bearing(
        &self,
        depth_frame: &TiledBearingFrameProducts,
        ref_keyframe: &TiledBearingKeyframe,
        t_ref_curr: &Matrix4<f64>,
        seeds: &[SparseDepthPrior],
        sigma_warp_sq: f64,
    ) -> PatchDepthOutput {
        let curr_level = &depth_frame.levels[0];
        let ref_level = &ref_keyframe.levels[0];
        let width = curr_level.width;
        let height = curr_level.height;
        let n = width * height;
        let mut eta_acc = vec![0.0f32; n];
        let mut w_acc = vec![0.0f32; n];
        let mut status = vec![PatchStatus::Unknown; n];
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let half = self.settings.patch_size / 2;
        let scaled_seeds = scale_seeds(seeds, self.settings.scale);
        let layout = self
            .tiled_bearing_levels
            .as_ref()
            .and_then(|levels| levels.first())
            .expect("tiled bearing layout must exist");
        let tile_seeds = assign_tiled_bearing_seeds(layout, &scaled_seeds, 1.0, half);

        #[cfg(feature = "parallel")]
        let patch_results: Vec<TiledPatchResult> = {
            let min_len = (curr_level.tiles.len() / (rayon::current_num_threads() * 8)).max(1);
            (0..curr_level.tiles.len())
                .into_par_iter()
                .with_min_len(min_len)
                .map(|tile_idx| {
                    self.solve_tiled_bearing_tile_patches(
                        tile_idx,
                        curr_level,
                        ref_level,
                        layout,
                        &tile_seeds,
                        depth_frame,
                        ref_keyframe,
                        &rel_pose,
                        sigma_warp_sq,
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let patch_results: Vec<TiledPatchResult> = (0..curr_level.tiles.len())
            .flat_map(|tile_idx| {
                self.solve_tiled_bearing_tile_patches(
                    tile_idx,
                    curr_level,
                    ref_level,
                    layout,
                    &tile_seeds,
                    depth_frame,
                    ref_keyframe,
                    &rel_pose,
                    sigma_warp_sq,
                )
            })
            .collect();

        for result in patch_results {
            let Some(curr_tile) = curr_level.tiles.get(result.tile_idx) else {
                continue;
            };
            if !curr_tile
                .tile
                .contains_point(result.global_u as f64, result.global_v as f64)
            {
                continue;
            }
            let photo = ref_level
                .tiles
                .get(result.tile_idx)
                .map(|ref_tile| TiledPhotoContext {
                    curr_img: &curr_tile.image,
                    curr_valid: &curr_tile.valid,
                    ref_tile: &ref_tile.tile,
                    ref_img: &ref_tile.ref_image,
                    ref_valid: &ref_tile.bilinear_valid,
                    rel_pose: &rel_pose,
                });
            self.accumulate_tiled_patch(
                &mut eta_acc,
                &mut w_acc,
                &mut status,
                width,
                height,
                &curr_tile.tile,
                result.local_u,
                result.local_v,
                result.estimate,
                photo.as_ref(),
            );
        }

        let mut eta = vec![f32::NAN; n];
        let mut eta_var = vec![f32::INFINITY; n];
        for idx in 0..n {
            if w_acc[idx] > 0.0 {
                eta[idx] = eta_acc[idx] / w_acc[idx];
                eta_var[idx] = 1.0 / w_acc[idx];
            }
        }
        PatchDepthOutput {
            eta: DepthMap::from_vec(width, height, eta).expect("eta size"),
            eta_var: DepthMap::from_vec(width, height, eta_var).expect("eta_var size"),
            status: DepthMap::from_vec(width, height, status).expect("status size"),
        }
    }

    fn solve_tiled_bearing_tile_patches(
        &self,
        tile_idx: usize,
        curr_level: &TiledBearingFrameLevel,
        ref_level: &TiledBearingKeyframeLevel,
        layout: &TiledBearingLevel,
        tile_seeds: &[Vec<SparseDepthPrior>],
        depth_frame: &TiledBearingFrameProducts,
        ref_keyframe: &TiledBearingKeyframe,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> Vec<TiledPatchResult> {
        let Some(curr_tile) = curr_level.tiles.get(tile_idx) else {
            return Vec::new();
        };
        if ref_level.tiles.get(tile_idx).is_none() {
            return Vec::new();
        }
        let Some(local_seeds) = tile_seeds.get(tile_idx) else {
            return Vec::new();
        };
        let seed_grid = SeedGrid::new(
            local_seeds,
            self.settings.seed_radius_px * self.settings.scale,
            curr_tile.tile.width,
            curr_tile.tile.height,
        );

        layout
            .patch_centers_in_tile(tile_idx, &self.settings)
            .into_iter()
            .map(|(global_u, global_v)| {
                let local = curr_tile
                    .tile
                    .global_to_local(Vector2::new(global_u as f64, global_v as f64));
                let estimate = self.solve_one_tiled_bearing_patch_level(
                    global_u as f64,
                    global_v as f64,
                    local[0],
                    local[1],
                    local_seeds,
                    &seed_grid,
                    depth_frame,
                    ref_keyframe,
                    rel_pose,
                    sigma_warp_sq,
                );
                TiledPatchResult {
                    tile_idx,
                    global_u,
                    global_v,
                    local_u: local[0],
                    local_v: local[1],
                    estimate,
                }
            })
            .collect()
    }

    fn solve_one_tiled_bearing_patch_level(
        &self,
        global_cu: f64,
        global_cv: f64,
        local_cu: f64,
        local_cv: f64,
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        depth_frame: &TiledBearingFrameProducts,
        ref_keyframe: &TiledBearingKeyframe,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> PatchEstimate {
        let nearby = nearby_seed_weights(local_cu, local_cv, seeds, seed_grid, &self.settings);
        if nearby.is_empty() {
            return PatchEstimate::unknown();
        }

        // Seeds already in η space. eta = ln(range), precision = 1/eta_var.
        let eta_min = self.settings.min_depth.ln();
        let eta_max = self.settings.max_depth.ln();

        let mut seed_eta_sum = 0.0;
        let mut seed_weight_total = 0.0;
        let mut seed_precision_sum_eta = 0.0;
        for item in nearby.iter() {
            let wp = item.w_spatial * item.precision;
            seed_eta_sum += wp * seeds[item.idx].eta;
            seed_weight_total += wp;
            seed_precision_sum_eta += wp;
        }
        let eta_init = (seed_eta_sum / seed_weight_total).clamp(eta_min, eta_max);
        let mut eta = eta_init;
        let mut final_residual = 1e10;
        let mut final_curvature = 0.0;

        for _ in 0..self.settings.n_gn_iters {
            let (grad_photo, hess_photo, mean_res, valid) = self
                .patch_residual_jacobian_fast_translation_tiled(
                    global_cu,
                    global_cv,
                    eta,
                    depth_frame,
                    ref_keyframe,
                    rel_pose,
                    sigma_warp_sq,
                );
            if valid == 0 {
                break;
            }

            let mut grad_seed = 0.0;
            let mut hess_seed = 0.0;
            for item in nearby.iter() {
                let wp = self.settings.lambda_seed * item.w_spatial * item.precision;
                grad_seed += wp * (eta - seeds[item.idx].eta);
                hess_seed += wp;
            }

            let hess_total = hess_photo + hess_seed;
            if hess_total < 1e-12 {
                break;
            }
            eta = (eta - (grad_photo + grad_seed) / hess_total).clamp(eta_min, eta_max);
            final_residual = mean_res;
            final_curvature = hess_photo;
        }
        let min_curvature = self.settings.min_photo_curvature
            * (self.settings.patch_size * self.settings.patch_size) as f64;
        if final_curvature >= min_curvature && final_residual <= self.settings.max_photo_residual {
            let eta_var = 1.0
                / (final_curvature + seed_precision_sum_eta * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::photo_refined(eta, eta_var, &self.settings)
        } else if final_residual > self.settings.max_photo_residual
            && final_curvature >= min_curvature
        {
            PatchEstimate::rejected(eta)
        } else {
            let eta_var = 1.0 / (seed_precision_sum_eta * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::seed_only(eta_init, eta_var, &self.settings)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_tiled_patch(
        &self,
        eta_acc: &mut [f32],
        w_acc: &mut [f32],
        status: &mut [PatchStatus],
        width: usize,
        height: usize,
        tile: &TiledBearingTile,
        cu: f64,
        cv: f64,
        estimate: PatchEstimate,
        photo: Option<&TiledPhotoContext>,
    ) {
        let Some(inv_var_w) = estimate.inv_var_weight_f32() else {
            return;
        };
        // Per-pixel photometric weighting only applies to photo-refined patches; seed-only
        // and the geometry-only test path fall back to a flat weight of 1.0.
        let photo_ctx = (estimate.status == PatchStatus::PhotoRefined)
            .then_some(photo)
            .flatten();
        let half = self.settings.patch_size / 2;
        for dy in -(half as isize)..half as isize {
            for dx in -(half as isize)..half as isize {
                let local_u = cu + dx as f64;
                let local_v = cv + dy as f64;
                let global_u = tile.x0 as isize + local_u.round() as isize;
                let global_v = tile.y0 as isize + local_v.round() as isize;
                if global_u < 0
                    || global_v < 0
                    || global_u >= width as isize
                    || global_v >= height as isize
                {
                    continue;
                }
                let tile_w = self.tiled_blend_weight(tile, local_u, local_v, width, height);
                if tile_w <= 0.0 {
                    continue;
                }
                let photo_w = match photo_ctx {
                    Some(ctx) => {
                        self.tiled_pixel_photo_weight(tile, local_u, local_v, estimate.eta, ctx)
                    }
                    None => 1.0,
                };
                let idx = global_v as usize * width + global_u as usize;
                // eta = ln(range) is tile-frame-independent (range is purely radial), so
                // overlapping tiles fuse their eta directly — a geometric mean of range.
                // The eta -> 3D/z conversion is deferred to the consumer (vis / mapping).
                let w = inv_var_w * tile_w * photo_w;
                eta_acc[idx] += w * estimate.eta as f32;
                w_acc[idx] += w;
                if (estimate.status as u8) > (status[idx] as u8) {
                    status[idx] = estimate.status;
                }
            }
        }
    }

    /// Per-pixel photometric agreement weight for a tiled patch: warps the current
    /// tile-local pixel into the reference tile at the patch's η and down-weights pixels
    /// whose reprojected intensity disagrees, mirroring the pinhole densify path.
    fn tiled_pixel_photo_weight(
        &self,
        curr_tile: &TiledBearingTile,
        local_u: f64,
        local_v: f64,
        eta: f64,
        ctx: &TiledPhotoContext,
    ) -> f32 {
        let cx = local_u.round();
        let cy = local_v.round();
        if cx < 0.0
            || cy < 0.0
            || cx >= ctx.curr_img.width() as f64
            || cy >= ctx.curr_img.height() as f64
        {
            return 1.0;
        }
        let (cx, cy) = (cx as usize, cy as usize);
        if ctx.curr_valid.as_slice()[cy * ctx.curr_valid.width() + cx] < 0.5 {
            return 1.0;
        }
        let i_curr = ctx.curr_img.as_slice()[cy * ctx.curr_img.width() + cx];
        let Some((u_ref, v_ref, _, _)) = warp_tiled_local_pixel_eta(
            curr_tile,
            ctx.ref_tile,
            local_u,
            local_v,
            eta,
            ctx.rel_pose,
        ) else {
            return 1.0;
        };
        let Some(i_ref) = sample_bilinear_valid(ctx.ref_img, Some(ctx.ref_valid), u_ref, v_ref)
        else {
            return 1.0;
        };
        let residual = (i_ref - i_curr).abs();
        1.0 / residual.max(1.0)
    }

    fn tiled_blend_weight(
        &self,
        tile: &TiledBearingTile,
        local_u: f64,
        local_v: f64,
        width: usize,
        height: usize,
    ) -> f32 {
        let margin = self
            .settings
            .tiled_tile_overlap
            .saturating_sub(self.settings.patch_size)
            .max(1) as f64;
        let right = tile.width.saturating_sub(1) as f64;
        let bottom = tile.height.saturating_sub(1) as f64;

        let mut wx = 1.0;
        if tile.x0 > 0 {
            wx *= smoothstep01(local_u / margin);
        }
        if tile.x0 + tile.width < width {
            wx *= smoothstep01((right - local_u) / margin);
        }

        let mut wy = 1.0;
        if tile.y0 > 0 {
            wy *= smoothstep01(local_v / margin);
        }
        if tile.y0 + tile.height < height {
            wy *= smoothstep01((bottom - local_v) / margin);
        }

        (wx * wy) as f32
    }

    /// Pick the per-patch solver for the (non-tiled) `solve` path: `PerPatchBearing`
    /// uses per-patch tangent rectification; all others use `solve_one_patch`.
    #[allow(clippy::too_many_arguments)]
    fn solve_one_patch_dispatch(
        &self,
        cu: f64,
        cv: f64,
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> PatchEstimate {
        if self.camera_mode == PatchDepthCameraMode::PerPatchBearing {
            self.solve_one_per_patch_bearing(
                cu,
                cv,
                seeds,
                seed_grid,
                curr_pyramid,
                curr_valid_pyramid,
                ref_keyframe,
                intrinsics_by_level,
                rel_pose,
                sigma_warp_sq,
            )
        } else {
            self.solve_one_patch(
                cu,
                cv,
                seeds,
                seed_grid,
                curr_pyramid,
                curr_valid_pyramid,
                ref_keyframe,
                intrinsics_by_level,
                t_ref_curr,
                sigma_warp_sq,
            )
        }
    }

    fn solve_one_patch(
        &self,
        cu: f64,
        cv: f64,
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
        sigma_warp_sq: f64,
    ) -> PatchEstimate {
        let nearby = nearby_seed_weights(cu, cv, seeds, seed_grid, &self.settings);
        if nearby.is_empty() {
            return PatchEstimate::unknown();
        }

        // η bounds: range = z * bearing_norm, so eta = ln(range) = ln(z * bearing_norm).
        let range_per_z = match self
            .bearing_for_scaled_pixel(cu, cv, &intrinsics_by_level[0])
            .map(|b| b.norm())
        {
            Some(n) => n,
            None => return PatchEstimate::unknown(),
        };
        let eta_min = (self.settings.min_depth * range_per_z).ln();
        let eta_max = (self.settings.max_depth * range_per_z).ln();

        // Seeds already in η space (converted in gather_seeds). precision = 1/eta_var.
        let mut seed_eta_sum = 0.0;
        let mut seed_weight_total = 0.0;
        let mut seed_precision_sum = 0.0;
        for item in nearby.iter() {
            let wp = item.w_spatial * item.precision;
            seed_eta_sum += wp * seeds[item.idx].eta;
            seed_weight_total += wp;
            seed_precision_sum += wp;
        }
        let eta_init = (seed_eta_sum / seed_weight_total).clamp(eta_min, eta_max);

        if !self.patch_has_enough_structure(
            cu,
            cv,
            eta_init,
            ref_keyframe,
            &intrinsics_by_level[0],
            &RelativePose::from_matrix(t_ref_curr),
        ) {
            return PatchEstimate::rejected(eta_init);
        }
        let mut eta = self.search_initial_eta(
            cu,
            cv,
            eta_init,
            eta_min,
            eta_max,
            curr_pyramid,
            curr_valid_pyramid,
            ref_keyframe,
            intrinsics_by_level,
            t_ref_curr,
        );
        let mut final_residual = 1e10;
        let mut final_curvature = 0.0;
        let use_fast_translation = self.settings.warp_mode == PatchDepthWarpMode::FastTranslation
            && self.camera_mode == PatchDepthCameraMode::UndistortedPinhole;

        for _ in 0..self.settings.n_gn_iters {
            let (grad_photo, hess_photo, mean_res, valid) = if use_fast_translation {
                self.patch_residual_jacobian_fast_translation(
                    cu,
                    cv,
                    eta,
                    curr_pyramid,
                    curr_valid_pyramid,
                    ref_keyframe,
                    intrinsics_by_level,
                    t_ref_curr,
                    sigma_warp_sq,
                )
            } else {
                self.patch_residual_jacobian(
                    cu,
                    cv,
                    eta,
                    curr_pyramid,
                    curr_valid_pyramid,
                    ref_keyframe,
                    intrinsics_by_level,
                    t_ref_curr,
                    sigma_warp_sq,
                )
            };
            if valid == 0 {
                break;
            }

            let mut grad_seed = 0.0;
            let mut hess_seed = 0.0;
            for item in nearby.iter() {
                let wp = self.settings.lambda_seed * item.w_spatial * item.precision;
                grad_seed += wp * (eta - seeds[item.idx].eta);
                hess_seed += wp;
            }

            let hess_total = hess_photo + hess_seed;
            if hess_total < 1e-12 {
                break;
            }
            eta = (eta - (grad_photo + grad_seed) / hess_total).clamp(eta_min, eta_max);
            final_residual = mean_res;
            final_curvature = hess_photo;
        }

        let min_curvature = self.settings.min_photo_curvature
            * (self.settings.patch_size * self.settings.patch_size) as f64;
        if final_curvature >= min_curvature && final_residual <= self.settings.max_photo_residual {
            let eta_var =
                1.0 / (final_curvature + seed_precision_sum * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::photo_refined(eta, eta_var, &self.settings)
        } else if final_residual > self.settings.max_photo_residual
            && final_curvature >= min_curvature
        {
            PatchEstimate::rejected(eta)
        } else {
            let eta_var = 1.0 / (seed_precision_sum * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::seed_only(eta_init, eta_var, &self.settings)
        }
    }

    /// Per-patch target-anchored solve: approximate the local raw projection by a
    /// patch-local affine chart, then solve the translated patch directly in raw
    /// images. This keeps PPB seamless for wide FoV without materializing a
    /// rectified tile image per patch.
    #[allow(clippy::too_many_arguments)]
    fn solve_one_per_patch_bearing(
        &self,
        cu: f64,
        cv: f64,
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        curr_pyramid: &[Image<f32>],
        _curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> PatchEstimate {
        let nearby = nearby_seed_weights(cu, cv, seeds, seed_grid, &self.settings);
        if nearby.is_empty() {
            return PatchEstimate::unknown();
        }

        // η bounds + seed init (range = z * ‖bearing‖), identical to solve_one_patch.
        let Some(center_bearing) = self.bearing_for_scaled_pixel(cu, cv, &intrinsics_by_level[0])
        else {
            return PatchEstimate::unknown();
        };
        let range_per_z = center_bearing.norm();
        let unit_center_bearing = center_bearing / range_per_z;
        let eta_min = (self.settings.min_depth * range_per_z).ln();
        let eta_max = (self.settings.max_depth * range_per_z).ln();
        let mut seed_eta_sum = 0.0;
        let mut seed_weight_total = 0.0;
        let mut seed_precision_sum = 0.0;
        for item in nearby.iter() {
            let wp = item.w_spatial * item.precision;
            seed_eta_sum += wp * seeds[item.idx].eta;
            seed_weight_total += wp;
            seed_precision_sum += wp;
        }
        let eta_init = (seed_eta_sum / seed_weight_total).clamp(eta_min, eta_max);
        let seed_var = 1.0 / (seed_precision_sum * self.settings.lambda_seed).max(1e-12);

        // One per-patch tangent tile, centred on the patch and shared by the
        // current and keyframe rectifications. Current and reference MUST use the
        // same tangent basis — FastTranslation's translated square assumes the two
        // patches differ only by a translation; differing tile centres would
        // rotate them relative to each other (by the warp angle) and break it.
        // The tile must be big enough to contain the warp displacement, like a
        // TiledBearing tile, so we size it from `tiled_tile_size`.
        let specs = undistort_level_specs(self.width, self.height, self.settings.scale, 1);
        let spec = &specs[0];
        let (lw, lh) = (spec.lw, spec.lh);
        // Size the per-patch tile to just contain the warp displacement at the
        // seed depth, plus a margin for the bilinear footprint and GN drift —
        // not a fixed `tiled_tile_size`. The tile build is O(side²) and is by far
        // the dominant cost (per-pixel camera.project in the LUT), so for the
        // common small-baseline case this is a large win; large displacements
        // clamp back up to `tiled_tile_size`, matching the old behaviour.
        let disp = {
            let x_ref = rel_pose.r * (unit_center_bearing * eta_init.exp()) + rel_pose.t;
            if x_ref[2] > 1e-6 {
                let (u_ref, v_ref) =
                    self.project_scaled(&x_ref, intrinsics_by_level[0].scale_from_original);
                (u_ref - cu).hypot(v_ref - cv)
            } else {
                0.0
            }
        };
        let margin = self.settings.patch_size as f64;
        let need = (self.settings.patch_size as f64 + 2.0 * (disp + margin)).ceil() as usize;
        let max_side = self
            .settings
            .tiled_tile_size
            .max(2 * self.settings.patch_size);
        let tile_side = need.clamp(2 * self.settings.patch_size, max_side);
        if tile_side >= lw || tile_side >= lh {
            return PatchEstimate::seed_only(eta_init, seed_var, &self.settings);
        }
        let x0 =
            (cu.round() as i64 - (tile_side as i64) / 2).clamp(0, (lw - tile_side) as i64) as usize;
        let y0 =
            (cv.round() as i64 - (tile_side as i64) / 2).clamp(0, (lh - tile_side) as i64) as usize;
        let tile_center_u = x0 as f64 + 0.5 * (tile_side.saturating_sub(1)) as f64;
        let tile_center_v = y0 as f64 + 0.5 * (tile_side.saturating_sub(1)) as f64;
        let tile_center_bearing = self
            .bearing_for_scaled_pixel(tile_center_u, tile_center_v, &intrinsics_by_level[0])
            .unwrap_or(unit_center_bearing);
        let tile = build_tiled_bearing_tile_geometry_from_center(
            self.camera.as_ref(),
            &self.intrinsics,
            spec,
            0,
            x0,
            y0,
            tile_side,
            tile_side,
            tile_center_bearing,
        );
        let Some(affine) = PerPatchAffineMap::from_tile(self.camera.as_ref(), spec, &tile) else {
            return PatchEstimate::seed_only(eta_init, seed_var, &self.settings);
        };

        let local_cu = cu - x0 as f64;
        let local_cv = cv - y0 as f64;

        let mut eta = eta_init;
        let mut final_residual = 1e10;
        let mut final_curvature = 0.0;
        let mut observed = false;
        for _ in 0..self.settings.n_gn_iters {
            let (grad_photo, hess_photo, sum_abs_res, valid_n) = self
                .patch_residual_jacobian_per_patch_bearing_affine(
                    &tile,
                    &affine,
                    local_cu,
                    local_cv,
                    eta,
                    &curr_pyramid[0],
                    &ref_keyframe.ref_pyramid[0],
                    &ref_keyframe.grad_x_pyramid[0],
                    &ref_keyframe.grad_y_pyramid[0],
                    rel_pose,
                    sigma_warp_sq,
                );
            if valid_n == 0 {
                break;
            }
            observed = true;
            let mut grad_seed = 0.0;
            let mut hess_seed = 0.0;
            for item in nearby.iter() {
                let wp = self.settings.lambda_seed * item.w_spatial * item.precision;
                grad_seed += wp * (eta - seeds[item.idx].eta);
                hess_seed += wp;
            }
            let hess_total = hess_photo + hess_seed;
            if hess_total < 1e-12 {
                break;
            }
            let eta_next = (eta - (grad_photo + grad_seed) / hess_total).clamp(eta_min, eta_max);
            let delta = (eta_next - eta).abs();
            eta = eta_next;
            // The tiled leaf returns the *summed* abs residual; the gate (and the
            // UP path's wrapper) work in per-pixel mean. Divide so the
            // `<= max_photo_residual` test matches UndistortedPinhole's scale.
            final_residual = sum_abs_res / valid_n.max(1) as f64;
            final_curvature = hess_photo;
            // η = ln(range); the default sub-1e-3 step is <0.1% range — below
            // noise. The photo leaf is the dominant per-patch cost, so stopping
            // once the GN step converges (often by iter 2-3) skips redundant leaf
            // calls without changing the refined depth or the photo/seed gate.
            if delta < self.settings.gn_eta_convergence_tol {
                break;
            }
        }

        let min_curvature = self.settings.min_photo_curvature
            * (self.settings.patch_size * self.settings.patch_size) as f64;

        if observed
            && final_curvature >= min_curvature
            && final_residual <= self.settings.max_photo_residual
        {
            let eta_var =
                1.0 / (final_curvature + seed_precision_sum * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::photo_refined(eta, eta_var, &self.settings)
        } else {
            // Warp left the tile, too little curvature, or residual too high:
            // fall back to the (trustworthy, converged) seed so coverage stays
            // dense rather than dropping the patch.
            PatchEstimate::seed_only(eta_init, seed_var, &self.settings)
        }
    }

    fn patch_has_enough_structure(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        ref_keyframe: &DepthKeyframe,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> bool {
        if self.settings.min_structure_eigen <= 0.0 && self.settings.max_structure_condition <= 0.0
        {
            return true;
        }

        let Some((u_ref_center, v_ref_center, x_ref, _unit_b)) =
            self.warp_with_eta(cu, cv, eta, intr, rel_pose)
        else {
            return false;
        };

        // Epipolar direction: dx_ref/dη = x_ref − t (the log-range identity).
        // Project to image space via J_proj.  Scale and sign cancel on normalisation;
        // None signals degenerate motion (pure forward) and falls back to λ_min.
        let epipolar = {
            let q = x_ref - rel_pose.t;
            let eu = self.intrinsics.fx * (q[0] * x_ref[2] - x_ref[0] * q[2]);
            let ev = self.intrinsics.fy * (q[1] * x_ref[2] - x_ref[1] * q[2]);
            let norm = (eu * eu + ev * ev).sqrt();
            if norm > 1e-12 {
                Some((eu / norm, ev / norm))
            } else {
                None
            }
        };

        let half = self.settings.patch_size / 2;
        let side = half * 2;
        let Some(fp) = bilinear_patch_footprint(
            &ref_keyframe.grad_x_pyramid[0],
            u_ref_center,
            v_ref_center,
            half,
            side,
        ) else {
            return false;
        };

        let mut gxx = 0.0;
        let mut gxy = 0.0;
        let mut gyy = 0.0;
        let mut n = 0usize;
        unsafe {
            for ly in 0..side {
                let gx_row0 = ref_keyframe.grad_x_pyramid[0].row_ptr(fp.y + ly);
                let gx_row1 = ref_keyframe.grad_x_pyramid[0].row_ptr(fp.y + ly + 1);
                let gy_row0 = ref_keyframe.grad_y_pyramid[0].row_ptr(fp.y + ly);
                let gy_row1 = ref_keyframe.grad_y_pyramid[0].row_ptr(fp.y + ly + 1);
                for lx in 0..side {
                    let ix = fp.x + lx;
                    let gx = bilerp_ptr(gx_row0, gx_row1, ix, fp.weights) as f64;
                    let gy = bilerp_ptr(gy_row0, gy_row1, ix, fp.weights) as f64;
                    gxx += gx * gx;
                    gxy += gx * gy;
                    gyy += gy * gy;
                    n += 1;
                }
            }
        }
        if n == 0 {
            return false;
        }
        let inv_n = 1.0 / n as f64;
        gxx *= inv_n;
        gxy *= inv_n;
        gyy *= inv_n;
        structure_tensor_passes(
            gxx,
            gxy,
            gyy,
            self.settings.min_structure_eigen,
            self.settings.max_structure_condition,
            epipolar,
        )
    }

    fn search_initial_eta(
        &self,
        cu: f64,
        cv: f64,
        eta_init: f64,
        eta_min: f64,
        eta_max: f64,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
    ) -> f64 {
        let n = self.settings.n_search_candidates.max(1);
        if n == 1 {
            return eta_init;
        }
        let half_range = self.settings.search_half_range.max(0.0);
        let lo = (eta_init - half_range).clamp(eta_min, eta_max);
        let hi = (eta_init + half_range).clamp(eta_min, eta_max);
        let mut best_eta = eta_init;
        let mut best_cost = f64::INFINITY;
        let use_fast_translation = self.settings.warp_mode == PatchDepthWarpMode::FastTranslation
            && self.camera_mode == PatchDepthCameraMode::UndistortedPinhole;
        for i in 0..n {
            let a = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let eta = lo + (hi - lo) * a;
            let (cost, valid) = if use_fast_translation {
                self.patch_cost_fast_translation(
                    cu,
                    cv,
                    eta,
                    &curr_pyramid[0],
                    curr_valid_pyramid.map(|p| &p[0]),
                    &ref_keyframe.ref_pyramid[0],
                    ref_keyframe.bilinear_valid_pyramid.as_ref().map(|p| &p[0]),
                    &intrinsics_by_level[0],
                    t_ref_curr,
                )
            } else {
                self.patch_cost(
                    cu,
                    cv,
                    eta,
                    &curr_pyramid[0],
                    curr_valid_pyramid.map(|p| &p[0]),
                    &ref_keyframe.ref_pyramid[0],
                    ref_keyframe.bilinear_valid_pyramid.as_ref().map(|p| &p[0]),
                    &intrinsics_by_level[0],
                    t_ref_curr,
                )
            };
            if valid > 0 && cost < best_cost {
                best_cost = cost;
                best_eta = eta;
            }
        }
        best_eta
    }

    fn patch_cost(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        intr: &ScaledIntrinsics,
        t_ref_curr: &Matrix4<f64>,
    ) -> (f64, usize) {
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let half = self.settings.patch_size / 2;
        let mut cost = 0.0;
        let mut valid = 0;
        for dy in -(half as isize)..half as isize {
            for dx in -(half as isize)..half as isize {
                let pu = cu + dx as f64;
                let pv = cv + dy as f64;
                let Some(i_curr) = sample_nearest(curr_img, pu, pv) else {
                    continue;
                };
                if !sample_valid_nearest(curr_valid, pu, pv) {
                    continue;
                }
                let Some((u_ref, v_ref, _, _)) = self.warp_with_eta(pu, pv, eta, intr, &rel_pose)
                else {
                    continue;
                };
                let Some(i_ref) = sample_bilinear_valid(ref_img, ref_valid, u_ref, v_ref) else {
                    continue;
                };
                let r = i_ref as f64 - i_curr as f64;
                let ar = r.abs();
                cost += if ar <= self.settings.photo_huber_delta {
                    0.5 * r * r
                } else {
                    self.settings.photo_huber_delta * (ar - 0.5 * self.settings.photo_huber_delta)
                };
                valid += 1;
            }
        }
        (cost, valid)
    }

    fn patch_cost_fast_translation(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        intr: &ScaledIntrinsics,
        t_ref_curr: &Matrix4<f64>,
    ) -> (f64, usize) {
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let Some((u_ref_center, v_ref_center, _, _)) =
            self.warp_with_eta(cu, cv, eta, intr, &rel_pose)
        else {
            return (0.0, 0);
        };

        let Some(patch) =
            self.translated_patch_footprint(cu, cv, ref_img, u_ref_center, v_ref_center)
        else {
            return (0.0, 0);
        };
        let mut cost = 0.0;
        let mut valid = 0;

        for ly in 0..patch.side {
            let cy = patch.curr_y0 + ly as isize;
            if cy < 0 || cy >= curr_img.height() as isize {
                continue;
            }
            unsafe {
                let curr_row = curr_img.row_ptr(cy as usize);
                let curr_mask_row = curr_valid.map(|mask| mask.row_ptr(cy as usize));
                let ref_row0 = ref_img.row_ptr(patch.ref_fp.y + ly);
                let ref_row1 = ref_img.row_ptr(patch.ref_fp.y + ly + 1);
                let ref_mask_row = ref_valid.map(|mask| mask.row_ptr(patch.ref_fp.y + ly));
                for lx in 0..patch.side {
                    let cx = patch.curr_x0 + lx as isize;
                    if cx < 0 || cx >= curr_img.width() as isize {
                        continue;
                    }
                    if !mask_row_valid(curr_mask_row, cx as usize) {
                        continue;
                    }
                    if !mask_row_valid(ref_mask_row, patch.ref_fp.x + lx) {
                        continue;
                    }

                    let i_curr = *curr_row.add(cx as usize);
                    let i_ref = bilerp_ptr(
                        ref_row0,
                        ref_row1,
                        patch.ref_fp.x + lx,
                        patch.ref_fp.weights,
                    );
                    let r = i_ref as f64 - i_curr as f64;
                    cost += huber_cost(r, self.settings.photo_huber_delta);
                    valid += 1;
                }
            }
        }
        (cost, valid)
    }

    fn patch_residual_jacobian(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;

        for level in 0..curr_pyramid.len() {
            let scale = 1.0 / (1usize << level) as f64;
            let (g, h, sar, nv) = self.patch_residual_jacobian_level(
                cu * scale,
                cv * scale,
                eta,
                &curr_pyramid[level],
                curr_valid_pyramid.map(|p| &p[level]),
                &ref_keyframe.ref_pyramid[level],
                ref_keyframe
                    .bilinear_valid_pyramid
                    .as_ref()
                    .map(|p| &p[level]),
                &ref_keyframe.grad_x_pyramid[level],
                &ref_keyframe.grad_y_pyramid[level],
                &intrinsics_by_level[level],
                &rel_pose,
                sigma_warp_sq,
            );
            grad += g;
            hess += h;
            sum_abs_res += sar;
            n_valid += nv;
        }

        (grad, hess, sum_abs_res / n_valid.max(1) as f64, n_valid)
    }

    fn patch_residual_jacobian_fast_translation(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;

        for level in 0..curr_pyramid.len() {
            let scale = 1.0 / (1usize << level) as f64;
            let (g, h, sar, nv) = self.patch_residual_jacobian_fast_translation_level(
                cu * scale,
                cv * scale,
                eta,
                &curr_pyramid[level],
                curr_valid_pyramid.map(|p| &p[level]),
                &ref_keyframe.ref_pyramid[level],
                ref_keyframe
                    .bilinear_valid_pyramid
                    .as_ref()
                    .map(|p| &p[level]),
                &ref_keyframe.grad_x_pyramid[level],
                &ref_keyframe.grad_y_pyramid[level],
                &intrinsics_by_level[level],
                &rel_pose,
                sigma_warp_sq,
            );
            grad += g;
            hess += h;
            sum_abs_res += sar;
            n_valid += nv;
        }

        (grad, hess, sum_abs_res / n_valid.max(1) as f64, n_valid)
    }

    fn patch_residual_jacobian_fast_translation_level(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        ref_grad_x: &Image<f32>,
        ref_grad_y: &Image<f32>,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let Some((u_ref_center, v_ref_center, x_ref_center, _unit_b)) =
            self.warp_with_eta(cu, cv, eta, intr, rel_pose)
        else {
            return (0.0, 0.0, 0.0, 0);
        };

        let dx_ref_deta = x_ref_center - rel_pose.t;
        let du_dxref = self.projection_jacobian(&x_ref_center) * dx_ref_deta;
        let du_deta = intr.scale_from_original * du_dxref[0];
        let dv_deta = intr.scale_from_original * du_dxref[1];

        let Some(patch) =
            self.translated_patch_footprint(cu, cv, ref_img, u_ref_center, v_ref_center)
        else {
            return (0.0, 0.0, 0.0, 0);
        };
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;
        let sigma_photo_sq = self.settings.sigma_photo * self.settings.sigma_photo;
        let constant_inv_sigma_photo_sq =
            (sigma_warp_sq <= 1e-18).then_some(1.0 / sigma_photo_sq.max(1e-12));

        #[cfg(target_arch = "x86_64")]
        if let Some(inv_sigma_photo_sq) = constant_inv_sigma_photo_sq {
            if let Some(accum) = fast_translation_accum_avx2_if_available(
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                ref_grad_x,
                ref_grad_y,
                patch,
                du_deta as f32,
                dv_deta as f32,
                inv_sigma_photo_sq as f32,
                self.settings.photo_huber_delta as f32,
            ) {
                return (accum.grad, accum.hess, accum.sum_abs_res, accum.n_valid);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(inv_sigma_photo_sq) = constant_inv_sigma_photo_sq {
            if let Some(accum) = fast_translation_accum_neon_if_available(
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                ref_grad_x,
                ref_grad_y,
                patch,
                du_deta as f32,
                dv_deta as f32,
                inv_sigma_photo_sq as f32,
                self.settings.photo_huber_delta as f32,
            ) {
                return (accum.grad, accum.hess, accum.sum_abs_res, accum.n_valid);
            }
        }

        for ly in 0..patch.side {
            let cy = patch.curr_y0 + ly as isize;
            if cy < 0 || cy >= curr_img.height() as isize {
                continue;
            }
            unsafe {
                let curr_row = curr_img.row_ptr(cy as usize);
                let curr_mask_row = curr_valid.map(|mask| mask.row_ptr(cy as usize));
                let ref_row0 = ref_img.row_ptr(patch.ref_fp.y + ly);
                let ref_row1 = ref_img.row_ptr(patch.ref_fp.y + ly + 1);
                let gx_row0 = ref_grad_x.row_ptr(patch.ref_fp.y + ly);
                let gx_row1 = ref_grad_x.row_ptr(patch.ref_fp.y + ly + 1);
                let gy_row0 = ref_grad_y.row_ptr(patch.ref_fp.y + ly);
                let gy_row1 = ref_grad_y.row_ptr(patch.ref_fp.y + ly + 1);
                let ref_mask_row = ref_valid.map(|mask| mask.row_ptr(patch.ref_fp.y + ly));
                for lx in 0..patch.side {
                    let cx = patch.curr_x0 + lx as isize;
                    if cx < 0 || cx >= curr_img.width() as isize {
                        continue;
                    }
                    if !mask_row_valid(curr_mask_row, cx as usize) {
                        continue;
                    }
                    if !mask_row_valid(ref_mask_row, patch.ref_fp.x + lx) {
                        continue;
                    }

                    let ix = patch.ref_fp.x + lx;
                    let i_curr = *curr_row.add(cx as usize);
                    let i_ref = bilerp_ptr(ref_row0, ref_row1, ix, patch.ref_fp.weights);
                    let gx = bilerp_ptr(gx_row0, gx_row1, ix, patch.ref_fp.weights);
                    let gy = bilerp_ptr(gy_row0, gy_row1, ix, patch.ref_fp.weights);

                    let jac = gx as f64 * du_deta + gy as f64 * dv_deta;
                    let residual = i_ref as f64 - i_curr as f64;
                    let ar = residual.abs();
                    let inv_sigma_eff_sq = photo_inv_sigma_eff_sq(
                        gx,
                        gy,
                        sigma_photo_sq,
                        sigma_warp_sq,
                        constant_inv_sigma_photo_sq,
                    );
                    let weight = huber_weight_from_abs_res(
                        ar,
                        self.settings.photo_huber_delta,
                        inv_sigma_eff_sq,
                    );
                    grad += weight * jac * residual;
                    hess += weight * jac * jac;
                    sum_abs_res += ar;
                    n_valid += 1;
                }
            }
        }

        (grad, hess, sum_abs_res, n_valid)
    }

    fn patch_residual_jacobian_fast_translation_tiled(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        depth_frame: &TiledBearingFrameProducts,
        ref_keyframe: &TiledBearingKeyframe,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let Some(layouts) = self.tiled_bearing_levels.as_ref() else {
            return (0.0, 0.0, 0.0, 0);
        };
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;
        let half = self.settings.patch_size / 2;

        for (level, layout) in layouts.iter().enumerate() {
            let scale = 1.0 / (1usize << level) as f64;
            let cu_l = cu * scale;
            let cv_l = cv * scale;
            let Some(tile_idx) = layout.owning_tile_for_patch(cu_l, cv_l, half) else {
                continue;
            };
            let Some(curr_level) = depth_frame.levels.get(level) else {
                continue;
            };
            let Some(ref_level) = ref_keyframe.levels.get(level) else {
                continue;
            };
            let Some(curr_tile) = curr_level.tiles.get(tile_idx) else {
                continue;
            };
            let Some(ref_tile) = ref_level.tiles.get(tile_idx) else {
                continue;
            };
            let local = curr_tile.tile.global_to_local(Vector2::new(cu_l, cv_l));
            let (g, h, sar, nv) = self.patch_residual_jacobian_fast_translation_tiled_level(
                &curr_tile.tile,
                &ref_tile.tile,
                local[0],
                local[1],
                eta,
                &curr_tile.image,
                Some(&curr_tile.valid),
                &ref_tile.ref_image,
                Some(&ref_tile.bilinear_valid),
                &ref_tile.grad_x,
                &ref_tile.grad_y,
                rel_pose,
                sigma_warp_sq,
            );
            grad += g;
            hess += h;
            sum_abs_res += sar;
            n_valid += nv;
        }

        (grad, hess, sum_abs_res / n_valid.max(1) as f64, n_valid)
    }

    #[allow(dead_code)]
    fn patch_residual_jacobian_fast_translation_tiled_level(
        &self,
        curr_tile: &TiledBearingTile,
        ref_tile: &TiledBearingTile,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        ref_grad_x: &Image<f32>,
        ref_grad_y: &Image<f32>,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let Some((u_ref_center, v_ref_center, x_ref_center, _unit_b)) =
            warp_tiled_local_pixel_eta(curr_tile, ref_tile, cu, cv, eta, rel_pose)
        else {
            return (0.0, 0.0, 0.0, 0);
        };

        let dx_ref_deta = x_ref_center - rel_pose.t;
        let Some(du_dxref) = ref_tile.projection_jacobian(&x_ref_center) else {
            return (0.0, 0.0, 0.0, 0);
        };
        let duv_deta = du_dxref * dx_ref_deta;
        let du_deta = duv_deta[0];
        let dv_deta = duv_deta[1];

        let Some(patch) =
            self.translated_patch_footprint(cu, cv, ref_img, u_ref_center, v_ref_center)
        else {
            return (0.0, 0.0, 0.0, 0);
        };

        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;
        let sigma_photo_sq = self.settings.sigma_photo * self.settings.sigma_photo;
        let constant_inv_sigma_photo_sq =
            (sigma_warp_sq <= 1e-18).then_some(1.0 / sigma_photo_sq.max(1e-12));

        for ly in 0..patch.side {
            let cy = patch.curr_y0 + ly as isize;
            if cy < 0 || cy >= curr_img.height() as isize {
                continue;
            }
            unsafe {
                let curr_row = curr_img.row_ptr(cy as usize);
                let curr_mask_row = curr_valid.map(|mask| mask.row_ptr(cy as usize));
                let ref_row0 = ref_img.row_ptr(patch.ref_fp.y + ly);
                let ref_row1 = ref_img.row_ptr(patch.ref_fp.y + ly + 1);
                let gx_row0 = ref_grad_x.row_ptr(patch.ref_fp.y + ly);
                let gx_row1 = ref_grad_x.row_ptr(patch.ref_fp.y + ly + 1);
                let gy_row0 = ref_grad_y.row_ptr(patch.ref_fp.y + ly);
                let gy_row1 = ref_grad_y.row_ptr(patch.ref_fp.y + ly + 1);
                let ref_mask_row = ref_valid.map(|mask| mask.row_ptr(patch.ref_fp.y + ly));
                for lx in 0..patch.side {
                    let cx = patch.curr_x0 + lx as isize;
                    if cx < 0 || cx >= curr_img.width() as isize {
                        continue;
                    }
                    if !mask_row_valid(curr_mask_row, cx as usize) {
                        continue;
                    }
                    if !mask_row_valid(ref_mask_row, patch.ref_fp.x + lx) {
                        continue;
                    }

                    let ix = patch.ref_fp.x + lx;
                    let i_curr = *curr_row.add(cx as usize);
                    let i_ref = bilerp_ptr(ref_row0, ref_row1, ix, patch.ref_fp.weights);
                    let gx = bilerp_ptr(gx_row0, gx_row1, ix, patch.ref_fp.weights);
                    let gy = bilerp_ptr(gy_row0, gy_row1, ix, patch.ref_fp.weights);

                    let jac = gx as f64 * du_deta + gy as f64 * dv_deta;
                    let residual = i_ref as f64 - i_curr as f64;
                    let ar = residual.abs();
                    let inv_sigma_eff_sq = photo_inv_sigma_eff_sq(
                        gx,
                        gy,
                        sigma_photo_sq,
                        sigma_warp_sq,
                        constant_inv_sigma_photo_sq,
                    );
                    let weight = huber_weight_from_abs_res(
                        ar,
                        self.settings.photo_huber_delta,
                        inv_sigma_eff_sq,
                    );
                    grad += weight * jac * residual;
                    hess += weight * jac * jac;
                    sum_abs_res += ar;
                    n_valid += 1;
                }
            }
        }

        (grad, hess, sum_abs_res, n_valid)
    }

    fn patch_residual_jacobian_per_patch_bearing_affine(
        &self,
        tile: &TiledBearingTile,
        affine: &PerPatchAffineMap,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        ref_img: &Image<f32>,
        ref_grad_x: &Image<f32>,
        ref_grad_y: &Image<f32>,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let Some((u_ref_center, v_ref_center, x_ref_center, _unit_b)) =
            warp_tiled_local_pixel_eta(tile, tile, cu, cv, eta, rel_pose)
        else {
            return (0.0, 0.0, 0.0, 0);
        };

        let dx_ref_deta = x_ref_center - rel_pose.t;
        let Some(du_dxref) = tile.projection_jacobian(&x_ref_center) else {
            return (0.0, 0.0, 0.0, 0);
        };
        let duv_deta = du_dxref * dx_ref_deta;
        let du_deta = duv_deta[0];
        let dv_deta = duv_deta[1];

        let half = self.settings.patch_size / 2;
        let side = self.settings.patch_size;
        let sigma_photo_sq = self.settings.sigma_photo * self.settings.sigma_photo;
        let constant_inv_sigma_photo_sq =
            (sigma_warp_sq <= 1e-18).then_some(1.0 / sigma_photo_sq.max(1e-12));
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;

        let curr_data = curr_img.as_slice();
        let ref_data = ref_img.as_slice();
        let ref_gx_data = ref_grad_x.as_slice();
        let ref_gy_data = ref_grad_y.as_slice();
        let width = curr_img.width();
        let height = curr_img.height();
        let stride = curr_img.stride();
        debug_assert_eq!(ref_img.width(), width);
        debug_assert_eq!(ref_img.height(), height);
        debug_assert_eq!(ref_img.stride(), stride);
        debug_assert_eq!(ref_grad_x.width(), width);
        debug_assert_eq!(ref_grad_x.height(), height);
        debug_assert_eq!(ref_grad_x.stride(), stride);
        debug_assert_eq!(ref_grad_y.width(), width);
        debug_assert_eq!(ref_grad_y.height(), height);
        debug_assert_eq!(ref_grad_y.stride(), stride);

        let raw_du_x = affine.raw_du[0];
        let raw_du_y = affine.raw_du[1];
        let raw_dv_x = affine.raw_dv[0];
        let raw_dv_y = affine.raw_dv[1];
        let curr_base_u = tile.x0 as f64 + cu - half as f64 - affine.center_u;
        let curr_base_v = tile.y0 as f64 + cv - half as f64 - affine.center_v;
        let ref_base_u = tile.x0 as f64 + u_ref_center - half as f64 - affine.center_u;
        let ref_base_v = tile.y0 as f64 + v_ref_center - half as f64 - affine.center_v;

        // SIMD leaf: constant photometric weight only (sigma_warp_sq == 0), the
        // same precondition as FastTranslation's AVX2 path.
        #[cfg(target_arch = "x86_64")]
        if let Some(inv_sigma_photo_sq) = constant_inv_sigma_photo_sq {
            // The reference Jacobian raw_gx·cgx + raw_gy·cgy folds the gradient
            // rotation (raw→tangent) and the tangent→η chain into two scalars.
            let cgx = raw_du_x * du_deta + raw_dv_x * dv_deta;
            let cgy = raw_du_y * du_deta + raw_dv_y * dv_deta;
            if let Some(accum) = per_patch_affine_accum_avx2_if_available(
                curr_img,
                ref_img,
                ref_grad_x,
                ref_grad_y,
                side,
                PerPatchAffineGeom {
                    raw_center: [affine.raw_center[0] as f32, affine.raw_center[1] as f32],
                    raw_du: [raw_du_x as f32, raw_du_y as f32],
                    raw_dv: [raw_dv_x as f32, raw_dv_y as f32],
                    curr_base_u: curr_base_u as f32,
                    curr_base_v: curr_base_v as f32,
                    ref_base_u: ref_base_u as f32,
                    ref_base_v: ref_base_v as f32,
                    cgx: cgx as f32,
                    cgy: cgy as f32,
                },
                inv_sigma_photo_sq as f32,
                self.settings.photo_huber_delta as f32,
            ) {
                return (accum.grad, accum.hess, accum.sum_abs_res, accum.n_valid);
            }
        }

        for ly in 0..side {
            let curr_du = curr_base_u;
            let curr_dv = curr_base_v + ly as f64;
            let ref_du = ref_base_u;
            let ref_dv = ref_base_v + ly as f64;
            let mut curr_raw_u = affine.raw_center[0] + raw_du_x * curr_du + raw_dv_x * curr_dv;
            let mut curr_raw_v = affine.raw_center[1] + raw_du_y * curr_du + raw_dv_y * curr_dv;
            let mut ref_raw_u = affine.raw_center[0] + raw_du_x * ref_du + raw_dv_x * ref_dv;
            let mut ref_raw_v = affine.raw_center[1] + raw_du_y * ref_du + raw_dv_y * ref_dv;
            for _ in 0..side {
                let Some(i_curr) = sample_bilinear_raw_nomask_slice(
                    curr_data, width, height, stride, curr_raw_u, curr_raw_v,
                ) else {
                    curr_raw_u += raw_du_x;
                    curr_raw_v += raw_du_y;
                    ref_raw_u += raw_du_x;
                    ref_raw_v += raw_du_y;
                    continue;
                };
                let Some((i_ref, raw_gx, raw_gy)) = sample_bilinear_raw_nomask_with_grad_slice(
                    ref_data,
                    ref_gx_data,
                    ref_gy_data,
                    width,
                    height,
                    stride,
                    ref_raw_u,
                    ref_raw_v,
                ) else {
                    curr_raw_u += raw_du_x;
                    curr_raw_v += raw_du_y;
                    ref_raw_u += raw_du_x;
                    ref_raw_v += raw_du_y;
                    continue;
                };
                let gx = (raw_gx as f64 * raw_du_x + raw_gy as f64 * raw_du_y) as f32;
                let gy = (raw_gx as f64 * raw_dv_x + raw_gy as f64 * raw_dv_y) as f32;
                let jac = gx as f64 * du_deta + gy as f64 * dv_deta;
                let residual = i_ref as f64 - i_curr as f64;
                let ar = residual.abs();
                let inv_sigma_eff_sq = photo_inv_sigma_eff_sq(
                    gx,
                    gy,
                    sigma_photo_sq,
                    sigma_warp_sq,
                    constant_inv_sigma_photo_sq,
                );
                let weight = huber_weight_from_abs_res(
                    ar,
                    self.settings.photo_huber_delta,
                    inv_sigma_eff_sq,
                );
                grad += weight * jac * residual;
                hess += weight * jac * jac;
                sum_abs_res += ar;
                n_valid += 1;
                curr_raw_u += raw_du_x;
                curr_raw_v += raw_du_y;
                ref_raw_u += raw_du_x;
                ref_raw_v += raw_du_y;
            }
        }

        (grad, hess, sum_abs_res, n_valid)
    }

    fn translated_patch_footprint(
        &self,
        cu: f64,
        cv: f64,
        ref_img: &Image<f32>,
        u_ref_center: f64,
        v_ref_center: f64,
    ) -> Option<TranslatedPatchFootprint> {
        let half = self.settings.patch_size / 2;
        let side = half * 2;
        let ref_fp = bilinear_patch_footprint(ref_img, u_ref_center, v_ref_center, half, side)?;
        Some(TranslatedPatchFootprint {
            ref_fp,
            curr_x0: (cu - half as f64) as isize,
            curr_y0: (cv - half as f64) as isize,
            side,
        })
    }

    fn patch_residual_jacobian_level(
        &self,
        cu: f64,
        cv: f64,
        eta: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        ref_grad_x: &Image<f32>,
        ref_grad_y: &Image<f32>,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
    ) -> (f64, f64, f64, usize) {
        let half = self.settings.patch_size / 2;
        let mut grad = 0.0;
        let mut hess = 0.0;
        let mut sum_abs_res = 0.0;
        let mut n_valid = 0;
        let sigma_photo_sq = self.settings.sigma_photo * self.settings.sigma_photo;

        for dy in -(half as isize)..half as isize {
            for dx in -(half as isize)..half as isize {
                let pu = cu + dx as f64;
                let pv = cv + dy as f64;
                let Some(i_curr) = sample_nearest(curr_img, pu, pv) else {
                    continue;
                };
                if !sample_valid_nearest(curr_valid, pu, pv) {
                    continue;
                }
                let Some((u_ref, v_ref, x_ref, _unit_b)) =
                    self.warp_with_eta(pu, pv, eta, intr, rel_pose)
                else {
                    continue;
                };
                let Some((i_ref, gx, gy)) = sample_bilinear_valid_with_grad(
                    ref_img, ref_valid, ref_grad_x, ref_grad_y, u_ref, v_ref,
                ) else {
                    continue;
                };

                let dx_ref_deta = x_ref - rel_pose.t;
                let du_dxref = self.projection_jacobian(&x_ref) * dx_ref_deta;
                let du_deta = intr.scale_from_original * du_dxref[0];
                let dv_deta = intr.scale_from_original * du_dxref[1];
                let jac = gx as f64 * du_deta + gy as f64 * dv_deta;

                let residual = i_ref as f64 - i_curr as f64;
                let ar = residual.abs();
                let grad_i_sq = gx as f64 * gx as f64 + gy as f64 * gy as f64;
                let sigma_eff_sq = sigma_photo_sq + grad_i_sq * sigma_warp_sq;
                let inv_sigma_eff_sq = 1.0 / sigma_eff_sq.max(1e-12);
                let weight = if ar <= self.settings.photo_huber_delta {
                    inv_sigma_eff_sq
                } else {
                    inv_sigma_eff_sq * self.settings.photo_huber_delta / ar
                };
                grad += weight * jac * residual;
                hess += weight * jac * jac;
                sum_abs_res += ar;
                n_valid += 1;
            }
        }

        (grad, hess, sum_abs_res, n_valid)
    }

    pub(super) fn warp_scaled_pixel(
        &self,
        u: f64,
        v: f64,
        rho: f64,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
        if rho <= 0.0 {
            return None;
        }
        let bearing = self.bearing_for_scaled_pixel(u, v, intr)?;
        let x_curr = bearing / rho;
        let x_ref = rel_pose.r * x_curr + rel_pose.t;
        if x_ref[2] <= 1e-6 {
            return None;
        }
        let (u_ref, v_ref) = self.project_scaled(&x_ref, intr.scale_from_original);
        Some((u_ref, v_ref, x_ref, bearing))
    }

    /// Warp a pixel at log-range η = ln(range).  All modes normalise the bearing to unit
    /// length before computing `x_curr = unit_b · exp(η)`, so the return bearing is unit.
    /// Returns `(u_ref, v_ref, x_ref, unit_b)`.
    pub(super) fn warp_with_eta(
        &self,
        u: f64,
        v: f64,
        eta: f64,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
        let bearing = self.bearing_for_scaled_pixel(u, v, intr)?;
        let unit_b = bearing.normalize();
        let x_curr = unit_b * eta.exp();
        let x_ref = rel_pose.r * x_curr + rel_pose.t;
        if x_ref[2] <= 1e-6 {
            return None;
        }
        let (u_ref, v_ref) = self.project_scaled(&x_ref, intr.scale_from_original);
        Some((u_ref, v_ref, x_ref, unit_b))
    }

    pub(super) fn bearing_for_scaled_pixel(
        &self,
        u: f64,
        v: f64,
        intr: &ScaledIntrinsics,
    ) -> Option<Vector3<f64>> {
        let original_u = u / intr.scale_from_original;
        let original_v = v / intr.scale_from_original;
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                self.bearing_at_original(original_u, original_v)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                if original_u < 0.0
                    || original_v < 0.0
                    || original_u >= (self.width - 1) as f64
                    || original_v >= (self.height - 1) as f64
                {
                    return None;
                }
                Some(Vector3::new(
                    (original_u - self.intrinsics.cx) / self.intrinsics.fx,
                    (original_v - self.intrinsics.cy) / self.intrinsics.fy,
                    1.0,
                ))
            }
        }
    }

    pub(super) fn project_scaled(&self, p: &Vector3<f64>, scale_from_original: f64) -> (f64, f64) {
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                let uv = self.camera.project(p);
                (uv[0] * scale_from_original, uv[1] * scale_from_original)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let z_inv = 1.0 / p[2];
                (
                    (self.intrinsics.fx * p[0] * z_inv + self.intrinsics.cx) * scale_from_original,
                    (self.intrinsics.fy * p[1] * z_inv + self.intrinsics.cy) * scale_from_original,
                )
            }
        }
    }

    fn projection_jacobian(&self, p: &Vector3<f64>) -> nalgebra::Matrix2x3<f64> {
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                self.camera.projection_jacobian(p)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let z_inv = 1.0 / p[2];
                let z_inv2 = z_inv * z_inv;
                nalgebra::Matrix2x3::new(
                    self.intrinsics.fx * z_inv,
                    0.0,
                    -self.intrinsics.fx * p[0] * z_inv2,
                    0.0,
                    self.intrinsics.fy * z_inv,
                    -self.intrinsics.fy * p[1] * z_inv2,
                )
            }
        }
    }

    fn bearing_at_original(&self, u: f64, v: f64) -> Option<Vector3<f64>> {
        if u < 0.0 || v < 0.0 || u >= (self.width - 1) as f64 || v >= (self.height - 1) as f64 {
            return None;
        }
        let ix = u as usize;
        let iy = v as usize;
        let dx = u - ix as f64;
        let dy = v - iy as f64;
        let b00 = self.bearing_lut[iy * self.width + ix];
        let b10 = self.bearing_lut[iy * self.width + ix + 1];
        let b01 = self.bearing_lut[(iy + 1) * self.width + ix];
        let b11 = self.bearing_lut[(iy + 1) * self.width + ix + 1];
        Some(
            b00 * ((1.0 - dx) * (1.0 - dy))
                + b10 * (dx * (1.0 - dy))
                + b01 * ((1.0 - dx) * dy)
                + b11 * (dx * dy),
        )
    }
}

fn compute_sigma_warp_sq(
    intrinsics: &CameraIntrinsics,
    t_ref_curr: &Matrix4<f64>,
    p_vv: Option<&Matrix3<f64>>,
    dt: f64,
    median_depth: f64,
) -> f64 {
    let Some(p_vv) = p_vv else {
        return 0.0;
    };
    if dt <= 0.0 {
        return 0.0;
    }
    let t = t_ref_curr.fixed_view::<3, 1>(0, 3).into_owned();
    let norm = t.norm();
    if norm < 1e-8 {
        return 0.0;
    }
    let t_hat = t / norm;
    let var_t_mag = dt * dt * (t_hat.transpose() * p_vv * t_hat)[(0, 0)];
    let f = 0.5 * (intrinsics.fx + intrinsics.fy);
    (f / median_depth).powi(2) * var_t_mag
}

#[inline(always)]
fn huber_cost(residual: f64, delta: f64) -> f64 {
    let ar = residual.abs();
    if ar <= delta {
        0.5 * residual * residual
    } else {
        delta * (ar - 0.5 * delta)
    }
}

#[inline(always)]
fn huber_weight_from_abs_res(abs_residual: f64, delta: f64, inv_sigma_eff_sq: f64) -> f64 {
    if abs_residual <= delta {
        inv_sigma_eff_sq
    } else {
        inv_sigma_eff_sq * delta / abs_residual
    }
}

#[inline(always)]
fn photo_inv_sigma_eff_sq(
    gx: f32,
    gy: f32,
    sigma_photo_sq: f64,
    sigma_warp_sq: f64,
    constant_inv_sigma_photo_sq: Option<f64>,
) -> f64 {
    if let Some(inv) = constant_inv_sigma_photo_sq {
        inv
    } else {
        let grad_i_sq = gx as f64 * gx as f64 + gy as f64 * gy as f64;
        let sigma_eff_sq = sigma_photo_sq + grad_i_sq * sigma_warp_sq;
        1.0 / sigma_eff_sq.max(1e-12)
    }
}

#[cfg(test)]
mod tests;
