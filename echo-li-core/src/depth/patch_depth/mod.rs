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

#[cfg(not(feature = "parallel"))]
use fuse::densify_pixels;
use fuse::PatchGrid;
#[cfg(feature = "parallel")]
use fuse::{densify_pixels_parallel, patch_centers};
use image_ops::{
    bilerp_ptr, bilinear_patch_footprint, build_bilinear_valid_pyramid, build_pinhole_to_raw_lut,
    build_pyramid_from_u8, dyadic_scale_offset, empty_pyramid, gradients, mask_row_valid,
    sample_bilinear_valid, sample_bilinear_valid_with_grad, sample_nearest, sample_valid_nearest,
    scaled_intrinsics, undistort_level_specs,
};
use seeds::{median_seed_depth, nearby_seed_weights, scale_seeds, SeedGrid};
#[cfg(target_arch = "x86_64")]
use simd::fast_translation_accum_avx2_if_available;
#[cfg(target_arch = "aarch64")]
use simd_neon::fast_translation_accum_neon_if_available;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchDepthCameraMode {
    RawDistorted,
    UndistortedPinhole,
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
    pub depth: DepthMap<f32>,
    pub variance: DepthMap<f32>,
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
    pub rho: f64,
    pub rho_var: f64,
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
    pub(super) rho: f64,
    #[allow(dead_code)]
    pub(super) var: f64,
    pub(super) status: PatchStatus,
    pub(super) inv_var_w: f64,
}

impl PatchEstimate {
    pub(super) fn unknown() -> Self {
        Self {
            rho: 0.0,
            var: 1e10,
            status: PatchStatus::Unknown,
            inv_var_w: 0.0,
        }
    }

    fn rejected(rho: f64) -> Self {
        Self {
            rho,
            var: 1e10,
            status: PatchStatus::Rejected,
            inv_var_w: 0.0,
        }
    }

    fn photo_refined(rho: f64, var: f64, settings: &PatchDepthSettings) -> Self {
        Self {
            rho,
            var,
            status: PatchStatus::PhotoRefined,
            inv_var_w: settings.status_weight_photo / var.max(settings.var_floor),
        }
    }

    fn seed_only(rho: f64, var: f64, settings: &PatchDepthSettings) -> Self {
        Self {
            rho,
            var,
            status: PatchStatus::SeedOnly,
            inv_var_w: settings.status_weight_seed / var.max(settings.var_floor),
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
struct UndistortSample {
    idx00: usize,
    idx10: usize,
    idx01: usize,
    idx11: usize,
    weights: [f32; 4],
}

/// Undistort LUT for one pyramid level: maps each undistorted-pinhole pixel to
/// a bilinear footprint in that level's raw (distorted) image.
struct UndistortLut {
    width: usize,
    height: usize,
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
        let n = lut.len();
        debug_assert_eq!(n, width * height);
        let mut out = Self {
            width,
            height,
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
        debug_assert_eq!(raw.width(), self.width);
        debug_assert_eq!(raw.height(), self.height);
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
            PatchDepthCameraMode::RawDistorted => PatchDepthSeedCoordinates::RawDistorted,
            PatchDepthCameraMode::UndistortedPinhole => {
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

        let mut bearing_lut = Vec::with_capacity(width * height);
        for v in 0..height {
            for u in 0..width {
                let uv = Vector2::new(u as f64, v as f64);
                let bearing = camera.undistort(&uv);
                if bearing[2].abs() > 1e-12 {
                    bearing_lut.push(bearing / bearing[2]);
                } else {
                    bearing_lut.push(Vector3::new(0.0, 0.0, 1.0));
                }
            }
        }

        let undistort_luts = match camera_mode {
            PatchDepthCameraMode::RawDistorted => None,
            PatchDepthCameraMode::UndistortedPinhole => {
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

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    fn depth_frame_products(&mut self, frame: FrameProducts) -> Option<DepthFrameProducts> {
        // Build the pyramid on the raw image, then undistort each kept level.
        // Undistorting the small downsampled levels instead of the full-res
        // frame is ~16x less work; the anti-alias blur happens in raw space.
        let raw_pyramid = self.build_depth_pyramid_from_u8(
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let (pyramid, valid_pyramid) = match self.camera_mode {
            PatchDepthCameraMode::RawDistorted => (raw_pyramid, None),
            PatchDepthCameraMode::UndistortedPinhole => {
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

    fn build_depth_pyramid_from_u8(
        &mut self,
        gray: &[u8],
        width: usize,
        height: usize,
        scale: f64,
        levels: usize,
    ) -> Vec<Image<f32>> {
        if let Some(offset) = dyadic_scale_offset(scale) {
            let src = Image::from_vec(width, height, gray.to_vec());
            self.pyramid_work
                .build_reuse(&src, offset + levels, &mut self.pyramid_scratch);
            return self.pyramid_work.levels[offset..offset + levels].to_vec();
        }

        build_pyramid_from_u8(gray, width, height, scale, levels)
    }

    fn gather_seeds(
        &self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
    ) -> Vec<SparseDepthPrior> {
        let mut seeds = Vec::new();
        for (&fid, uv_f32) in &measurement.cam_coordinates {
            let (z, z_var) = sparse_filter.query(fid);
            if z <= 0.0
                || z < self.settings.min_depth
                || z > self.settings.max_depth
                || !z_var.is_finite()
            {
                continue;
            }
            let rho = 1.0 / z;
            let rho_var = z_var / z.powi(4);
            if rho.is_finite() && rho_var.is_finite() && rho_var > 0.0 {
                seeds.push(SparseDepthPrior {
                    uv: Vector2::new(uv_f32[0] as f64, uv_f32[1] as f64),
                    rho,
                    rho_var,
                });
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
                    let estimate = self.solve_one_patch(
                        u as f64,
                        v as f64,
                        &scaled_seeds,
                        &seed_grid,
                        curr_pyramid,
                        curr_valid_pyramid,
                        ref_keyframe,
                        &scaled_intrinsics,
                        t_ref_curr,
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
                    let estimate = self.solve_one_patch(
                        u as f64,
                        v as f64,
                        &scaled_seeds,
                        &seed_grid,
                        curr_pyramid,
                        curr_valid_pyramid,
                        ref_keyframe,
                        &scaled_intrinsics,
                        t_ref_curr,
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

        let mut seed_rho_init = 0.0;
        let mut seed_weight_total = 0.0;
        let mut seed_precision_sum = 0.0;
        for item in nearby.iter() {
            let weighted_precision = item.w_spatial * item.precision;
            seed_rho_init += weighted_precision * seeds[item.idx].rho;
            seed_weight_total += weighted_precision;
            seed_precision_sum += weighted_precision;
        }

        let rho_min = 1.0 / self.settings.max_depth;
        let rho_max = 1.0 / self.settings.min_depth;
        let rho_init = (seed_rho_init / seed_weight_total).clamp(rho_min, rho_max);
        if !self.patch_has_enough_structure(
            cu,
            cv,
            rho_init,
            ref_keyframe,
            &intrinsics_by_level[0],
            &RelativePose::from_matrix(t_ref_curr),
        ) {
            return PatchEstimate::rejected(rho_init);
        }
        let mut rho = self.search_initial_rho(
            cu,
            cv,
            rho_init,
            rho_min,
            rho_max,
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
                    rho,
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
                    rho,
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
                grad_seed += wp * (rho - seeds[item.idx].rho);
                hess_seed += wp;
            }

            let hess_total = hess_photo + hess_seed;
            if hess_total < 1e-12 {
                break;
            }
            rho = (rho - (grad_photo + grad_seed) / hess_total).clamp(rho_min, rho_max);
            final_residual = mean_res;
            final_curvature = hess_photo;
        }

        let min_curvature = self.settings.min_photo_curvature
            * (self.settings.patch_size * self.settings.patch_size) as f64;
        if final_curvature >= min_curvature && final_residual <= self.settings.max_photo_residual {
            let hess_total = final_curvature + seed_precision_sum * self.settings.lambda_seed;
            let var = 1.0 / hess_total.max(1e-12);
            PatchEstimate::photo_refined(rho, var, &self.settings)
        } else if final_residual > self.settings.max_photo_residual
            && final_curvature >= min_curvature
        {
            PatchEstimate::rejected(rho)
        } else {
            let var = 1.0 / (seed_precision_sum * self.settings.lambda_seed).max(1e-12);
            PatchEstimate::seed_only(rho_init, var, &self.settings)
        }
    }

    fn patch_has_enough_structure(
        &self,
        cu: f64,
        cv: f64,
        rho: f64,
        ref_keyframe: &DepthKeyframe,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> bool {
        if self.settings.min_structure_eigen <= 0.0 && self.settings.max_structure_condition <= 0.0
        {
            return true;
        }

        let Some((u_ref_center, v_ref_center, _, _)) =
            self.warp_scaled_pixel(cu, cv, rho, intr, rel_pose)
        else {
            return false;
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

        let trace = gxx + gyy;
        let discr = ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt();
        let lambda_min = 0.5 * (trace - discr);
        let lambda_max = 0.5 * (trace + discr);

        if self.settings.min_structure_eigen > 0.0 && lambda_min < self.settings.min_structure_eigen
        {
            return false;
        }
        if self.settings.max_structure_condition > 0.0 {
            if lambda_min <= 1e-12 {
                return false;
            }
            if lambda_max / lambda_min > self.settings.max_structure_condition {
                return false;
            }
        }
        true
    }

    fn search_initial_rho(
        &self,
        cu: f64,
        cv: f64,
        rho_init: f64,
        rho_min: f64,
        rho_max: f64,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        ref_keyframe: &DepthKeyframe,
        intrinsics_by_level: &[ScaledIntrinsics],
        t_ref_curr: &Matrix4<f64>,
    ) -> f64 {
        let n = self.settings.n_search_candidates.max(1);
        if n == 1 {
            return rho_init;
        }
        let half_range = self.settings.search_half_range.max(0.0);
        let lo = (rho_init * (1.0 - half_range)).clamp(rho_min, rho_max);
        let hi = (rho_init * (1.0 + half_range)).clamp(rho_min, rho_max);
        let mut best_rho = rho_init;
        let mut best_cost = f64::INFINITY;
        let use_fast_translation = self.settings.warp_mode == PatchDepthWarpMode::FastTranslation
            && self.camera_mode == PatchDepthCameraMode::UndistortedPinhole;
        for i in 0..n {
            let a = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let rho = lo + (hi - lo) * a;
            let (cost, valid) = if use_fast_translation {
                self.patch_cost_fast_translation(
                    cu,
                    cv,
                    rho,
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
                    rho,
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
                best_rho = rho;
            }
        }
        best_rho
    }

    fn patch_cost(
        &self,
        cu: f64,
        cv: f64,
        rho: f64,
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
                let Some((u_ref, v_ref, _, _)) =
                    self.warp_scaled_pixel(pu, pv, rho, intr, &rel_pose)
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
        rho: f64,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        intr: &ScaledIntrinsics,
        t_ref_curr: &Matrix4<f64>,
    ) -> (f64, usize) {
        let rel_pose = RelativePose::from_matrix(t_ref_curr);
        let Some((u_ref_center, v_ref_center, _, _)) =
            self.warp_scaled_pixel(cu, cv, rho, intr, &rel_pose)
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
        rho: f64,
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
                rho,
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
        rho: f64,
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
                rho,
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
        rho: f64,
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
        let Some((u_ref_center, v_ref_center, x_ref_center, bearing_center)) =
            self.warp_scaled_pixel(cu, cv, rho, intr, rel_pose)
        else {
            return (0.0, 0.0, 0.0, 0);
        };

        let dx_ref_drho = rel_pose.r * (-bearing_center / (rho * rho));
        let du_dxref = self.projection_jacobian(&x_ref_center) * dx_ref_drho;
        let du_drho = intr.scale_from_original * du_dxref[0];
        let dv_drho = intr.scale_from_original * du_dxref[1];

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
                du_drho as f32,
                dv_drho as f32,
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
                du_drho as f32,
                dv_drho as f32,
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

                    let jac = gx as f64 * du_drho + gy as f64 * dv_drho;
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
        rho: f64,
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
        let dx_curr_drho_scale = -1.0 / (rho * rho);
        let r_dx_curr_drho = rel_pose.r * dx_curr_drho_scale;

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
                let Some((u_ref, v_ref, x_ref, bearing)) =
                    self.warp_scaled_pixel(pu, pv, rho, intr, rel_pose)
                else {
                    continue;
                };
                let Some((i_ref, gx, gy)) = sample_bilinear_valid_with_grad(
                    ref_img, ref_valid, ref_grad_x, ref_grad_y, u_ref, v_ref,
                ) else {
                    continue;
                };

                let dx_ref_drho = r_dx_curr_drho * bearing;
                let du_dxref = self.projection_jacobian(&x_ref) * dx_ref_drho;
                let du_drho = intr.scale_from_original * du_dxref[0];
                let dv_drho = intr.scale_from_original * du_dxref[1];
                let jac = gx as f64 * du_drho + gy as f64 * dv_drho;

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

    pub(super) fn bearing_for_scaled_pixel(
        &self,
        u: f64,
        v: f64,
        intr: &ScaledIntrinsics,
    ) -> Option<Vector3<f64>> {
        let original_u = u / intr.scale_from_original;
        let original_v = v / intr.scale_from_original;
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted => self.bearing_at_original(original_u, original_v),
            PatchDepthCameraMode::UndistortedPinhole => {
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
            PatchDepthCameraMode::RawDistorted => {
                let uv = self.camera.project(p);
                (uv[0] * scale_from_original, uv[1] * scale_from_original)
            }
            PatchDepthCameraMode::UndistortedPinhole => {
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
            PatchDepthCameraMode::RawDistorted => self.camera.projection_jacobian(p),
            PatchDepthCameraMode::UndistortedPinhole => {
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
