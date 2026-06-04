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
mod geometry;
mod image_ops;
mod mapper;
mod residual;
mod seeds;
#[cfg(target_arch = "x86_64")]
mod simd;
#[cfg(target_arch = "aarch64")]
mod simd_neon;
mod solve;
mod tiled_bearing;

// Free functions live in the topical submodules above; re-export so this module
// and the other submodules (`use super::*`) resolve them by bare name.
pub(in crate::depth::patch_depth) use geometry::*;
pub(in crate::depth::patch_depth) use tiled_bearing::*;

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
