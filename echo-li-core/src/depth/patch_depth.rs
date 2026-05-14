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
    pub cell_size: usize,
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
            cell_size: 8,
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
    pub depth_cells: DepthMap<f32>,
    pub variance_cells: DepthMap<f32>,
    pub status_cells: DepthMap<PatchStatus>,
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
struct RelativePose {
    r: Matrix3<f64>,
    t: Vector3<f64>,
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
struct ScaledIntrinsics {
    scale_from_original: f64,
}

#[derive(Debug, Clone)]
struct SeedGrid {
    ids: Vec<usize>,
    starts: Vec<usize>,
    cols: usize,
    rows: usize,
    cell_size: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct NearbySeed {
    idx: usize,
    w_spatial: f64,
    precision: f64,
}

#[derive(Debug, Clone)]
struct NearbySeeds {
    len: usize,
    items: [NearbySeed; NearbySeeds::MAX],
}

impl NearbySeeds {
    const MAX: usize = 64;

    fn new() -> Self {
        Self {
            len: 0,
            items: [NearbySeed::default(); Self::MAX],
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, item: NearbySeed) -> bool {
        if self.len >= Self::MAX {
            return false;
        }
        self.items[self.len] = item;
        self.len += 1;
        true
    }

    fn iter(&self) -> impl Iterator<Item = &NearbySeed> {
        self.items[..self.len].iter()
    }
}

#[derive(Debug, Clone, Copy)]
struct PatchEstimate {
    rho: f64,
    var: f64,
    status: PatchStatus,
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

impl UndistortSample {
    #[inline(always)]
    fn sample_u8(self, gray: &[u8]) -> u8 {
        let i00 = gray[self.idx00] as f32;
        let i10 = gray[self.idx10] as f32;
        let i01 = gray[self.idx01] as f32;
        let i11 = gray[self.idx11] as f32;
        (self.weights[0] * i00
            + self.weights[1] * i10
            + self.weights[2] * i01
            + self.weights[3] * i11)
            .round() as u8
    }
}

pub struct PatchDepthMapper {
    camera: Arc<dyn CameraModel>,
    camera_mode: PatchDepthCameraMode,
    intrinsics: CameraIntrinsics,
    width: usize,
    height: usize,
    settings: PatchDepthSettings,
    bearing_lut: Vec<Vector3<f64>>,
    undistort_lut: Option<Vec<Option<UndistortSample>>>,
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
            settings.patch_stride == settings.cell_size / 2,
            "patch_stride must equal cell_size / 2"
        );
        anyhow::ensure!(
            settings.patch_size >= settings.cell_size,
            "patch_size must be >= cell_size"
        );
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

        let undistort_lut = match camera_mode {
            PatchDepthCameraMode::RawDistorted => None,
            PatchDepthCameraMode::UndistortedPinhole => Some(build_pinhole_to_raw_lut(
                camera.as_ref(),
                &intrinsics,
                width,
                height,
            )),
        };

        Ok(Self {
            camera,
            camera_mode,
            intrinsics,
            width,
            height,
            settings,
            bearing_lut,
            undistort_lut,
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
        let (frame, valid_mask) = match self.camera_mode {
            PatchDepthCameraMode::RawDistorted => (frame, None),
            PatchDepthCameraMode::UndistortedPinhole => {
                let lut = self.undistort_lut.as_ref()?;
                let mut gray = vec![0u8; self.width * self.height];
                let mut valid_mask = vec![0u8; self.width * self.height];
                for (idx, sample) in lut.iter().enumerate() {
                    if let Some(sample) = sample {
                        gray[idx] = sample.sample_u8(&frame.gray);
                        valid_mask[idx] = 1;
                    }
                }
                (FrameProducts { gray, ..frame }, Some(valid_mask))
            }
        };
        let pyramid = Arc::new(self.build_depth_pyramid_from_u8(
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        ));
        let valid_pyramid = valid_mask.map(|mask| {
            Arc::new(build_mask_pyramid(
                &mask,
                frame.width,
                frame.height,
                self.settings.scale,
                self.settings.n_pyramid_levels,
            ))
        });
        Some(DepthFrameProducts {
            frame: Arc::new(frame),
            pyramid,
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

        #[cfg(feature = "parallel")]
        {
            let patch_centers = patch_centers(width, height, &self.settings);
            let patches: Vec<_> = patch_centers
                .par_iter()
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
                    (u as f64, v as f64, estimate)
                })
                .collect();
            fuse_cells(&patches, width, height, &self.settings)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut fuse = FuseAccumulator::new(width, height, &self.settings);
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
                    fuse.add(u as f64, v as f64, estimate, &self.settings);
                }
            }
            fuse.finish()
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
            return PatchEstimate {
                rho: 0.0,
                var: 1e10,
                status: PatchStatus::Unknown,
            };
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
            PatchEstimate {
                rho,
                var: 1.0 / hess_total.max(1e-12),
                status: PatchStatus::PhotoRefined,
            }
        } else if final_residual > self.settings.max_photo_residual
            && final_curvature >= min_curvature
        {
            PatchEstimate {
                rho,
                var: 1e10,
                status: PatchStatus::Rejected,
            }
        } else {
            PatchEstimate {
                rho: rho_init,
                var: 1.0 / (seed_precision_sum * self.settings.lambda_seed).max(1e-12),
                status: PatchStatus::SeedOnly,
            }
        }
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

    fn warp_scaled_pixel(
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

    fn bearing_for_scaled_pixel(
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

    fn project_scaled(&self, p: &Vector3<f64>, scale_from_original: f64) -> (f64, f64) {
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

impl SeedGrid {
    fn new(seeds: &[SparseDepthPrior], cell_size: f64, width: usize, height: usize) -> Self {
        let cell_size = cell_size.max(1.0);
        let cols = ((width as f64) / cell_size).ceil().max(1.0) as usize;
        let rows = ((height as f64) / cell_size).ceil().max(1.0) as usize;
        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); cols * rows];
        for (idx, seed) in seeds.iter().enumerate() {
            let ci = ((seed.uv[0] / cell_size) as usize).min(cols - 1);
            let cj = ((seed.uv[1] / cell_size) as usize).min(rows - 1);
            bins[cj * cols + ci].push(idx);
        }
        let mut ids = Vec::with_capacity(seeds.len());
        let mut starts = Vec::with_capacity(cols * rows + 1);
        starts.push(0);
        for bin in bins {
            ids.extend(bin);
            starts.push(ids.len());
        }
        Self {
            ids,
            starts,
            cols,
            rows,
            cell_size,
        }
    }
}

fn median_seed_depth(seeds: &[SparseDepthPrior]) -> Option<f64> {
    if seeds.is_empty() {
        return None;
    }
    let mut depths: Vec<f64> = seeds
        .iter()
        .filter_map(|seed| (seed.rho > 0.0).then_some(1.0 / seed.rho))
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.total_cmp(b));
    Some(depths[depths.len() / 2])
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

fn sample_nearest(img: &Image<f32>, u: f64, v: f64) -> Option<f32> {
    let x = u as isize;
    let y = v as isize;
    if x < 0 || y < 0 || x >= img.width() as isize || y >= img.height() as isize {
        return None;
    }
    // SAFETY: bounds were checked above.
    Some(unsafe { img.get_unchecked(x as usize, y as usize) })
}

fn sample_bilinear_valid(
    img: &Image<f32>,
    mask: Option<&Image<f32>>,
    u: f64,
    v: f64,
) -> Option<f32> {
    let (x, y, weights) = bilinear_footprint(img, u, v)?;
    if !valid_bilinear_footprint(mask, x, y) {
        return None;
    }
    Some(unsafe { bilinear_unchecked(img, x, y, weights) })
}

fn sample_bilinear_valid_with_grad(
    img: &Image<f32>,
    mask: Option<&Image<f32>>,
    grad_x: &Image<f32>,
    grad_y: &Image<f32>,
    u: f64,
    v: f64,
) -> Option<(f32, f32, f32)> {
    let (x, y, weights) = bilinear_footprint(img, u, v)?;
    if !valid_bilinear_footprint(mask, x, y) {
        return None;
    }
    // SAFETY: all three pyramids are built from the same image dimensions, and
    // the bilinear footprint was checked against the reference image above.
    unsafe {
        Some((
            bilinear_unchecked(img, x, y, weights),
            bilinear_unchecked(grad_x, x, y, weights),
            bilinear_unchecked(grad_y, x, y, weights),
        ))
    }
}

fn bilinear_footprint(img: &Image<f32>, u: f64, v: f64) -> Option<(usize, usize, [f32; 4])> {
    if u < 0.0 || v < 0.0 || u >= (img.width() - 1) as f64 || v >= (img.height() - 1) as f64 {
        return None;
    }
    let x = u as usize;
    let y = v as usize;
    let dx = (u - x as f64) as f32;
    let dy = (v - y as f64) as f32;
    let one_minus_dx = 1.0 - dx;
    let one_minus_dy = 1.0 - dy;
    Some((
        x,
        y,
        [
            one_minus_dx * one_minus_dy,
            dx * one_minus_dy,
            one_minus_dx * dy,
            dx * dy,
        ],
    ))
}

fn bilinear_patch_footprint(
    img: &Image<f32>,
    u_center: f64,
    v_center: f64,
    half: usize,
    patch_size: usize,
) -> Option<BilinearPatchFootprint> {
    let u0 = u_center - half as f64;
    let v0 = v_center - half as f64;
    if u0 < 0.0 || v0 < 0.0 {
        return None;
    }

    let x = u0 as usize;
    let y = v0 as usize;
    if x + patch_size >= img.width() || y + patch_size >= img.height() {
        return None;
    }

    let dx = (u0 - x as f64) as f32;
    let dy = (v0 - y as f64) as f32;
    let one_minus_dx = 1.0 - dx;
    let one_minus_dy = 1.0 - dy;
    Some(BilinearPatchFootprint {
        x,
        y,
        weights: [
            one_minus_dx * one_minus_dy,
            dx * one_minus_dy,
            one_minus_dx * dy,
            dx * dy,
        ],
    })
}

fn valid_bilinear_footprint(mask: Option<&Image<f32>>, x: usize, y: usize) -> bool {
    let Some(mask) = mask else {
        return true;
    };
    // SAFETY: callers check the bilinear image footprint before passing x/y.
    unsafe { mask.get_unchecked(x, y) > 0.5 }
}

#[inline(always)]
unsafe fn mask_row_valid(row: Option<*const f32>, x: usize) -> bool {
    match row {
        Some(row) => unsafe { *row.add(x) > 0.5 },
        None => true,
    }
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

#[cfg(target_arch = "x86_64")]
fn fast_translation_accum_avx2_if_available(
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    ref_grad_x: &Image<f32>,
    ref_grad_y: &Image<f32>,
    patch: TranslatedPatchFootprint,
    du_drho: f32,
    dv_drho: f32,
    inv_sigma_photo_sq: f32,
    huber_delta: f32,
) -> Option<PatchAccum> {
    if patch.side != 4 && patch.side != 8 && patch.side != 16 {
        return None;
    }
    if patch.curr_x0 < 0
        || patch.curr_y0 < 0
        || patch.curr_x0 as usize + patch.side > curr_img.width()
        || patch.curr_y0 as usize + patch.side > curr_img.height()
    {
        return None;
    }
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return None;
    }

    // SAFETY: AVX2/FMA support is checked above. The translated footprint and
    // current image bounds are validated before entering the SIMD routine.
    unsafe {
        if patch.side == 4 {
            fast_translation_accum_avx2_4x4(
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                ref_grad_x,
                ref_grad_y,
                patch,
                du_drho,
                dv_drho,
                inv_sigma_photo_sq,
                huber_delta,
            )
        } else {
            fast_translation_accum_avx2(
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                ref_grad_x,
                ref_grad_y,
                patch,
                du_drho,
                dv_drho,
                inv_sigma_photo_sq,
                huber_delta,
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_translation_accum_avx2(
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    ref_grad_x: &Image<f32>,
    ref_grad_y: &Image<f32>,
    patch: TranslatedPatchFootprint,
    du_drho: f32,
    dv_drho: f32,
    inv_sigma_photo_sq: f32,
    huber_delta: f32,
) -> Option<PatchAccum> {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_andnot_ps, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_div_ps,
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_sub_ps, _CMP_LE_OQ,
    };

    let weights = patch.ref_fp.weights;
    let vw00 = _mm256_set1_ps(weights[0]);
    let vw10 = _mm256_set1_ps(weights[1]);
    let vw01 = _mm256_set1_ps(weights[2]);
    let vw11 = _mm256_set1_ps(weights[3]);
    let v_du = _mm256_set1_ps(du_drho);
    let v_dv = _mm256_set1_ps(dv_drho);
    let v_inv_sigma = _mm256_set1_ps(inv_sigma_photo_sq);
    let v_delta = _mm256_set1_ps(huber_delta);
    let v_delta_inv_sigma = _mm256_set1_ps(huber_delta * inv_sigma_photo_sq);
    let v_abs_mask = _mm256_set1_ps(-0.0);

    let mut sum_grad = _mm256_setzero_ps();
    let mut sum_hess = _mm256_setzero_ps();
    let mut sum_abs = _mm256_setzero_ps();
    let mut n_valid = 0usize;

    unsafe {
        for ly in 0..patch.side {
            let cy = patch.curr_y0 as usize + ly;
            let curr_row = curr_img.row_ptr(cy);
            let curr_mask_row = curr_valid.map(|mask| mask.row_ptr(cy));
            let ref_row0 = ref_img.row_ptr(patch.ref_fp.y + ly);
            let ref_row1 = ref_img.row_ptr(patch.ref_fp.y + ly + 1);
            let gx_row0 = ref_grad_x.row_ptr(patch.ref_fp.y + ly);
            let gx_row1 = ref_grad_x.row_ptr(patch.ref_fp.y + ly + 1);
            let gy_row0 = ref_grad_y.row_ptr(patch.ref_fp.y + ly);
            let gy_row1 = ref_grad_y.row_ptr(patch.ref_fp.y + ly + 1);
            let ref_mask_row = ref_valid.map(|mask| mask.row_ptr(patch.ref_fp.y + ly));

            for lx in (0..patch.side).step_by(8) {
                let cx = patch.curr_x0 as usize + lx;
                let ix = patch.ref_fp.x + lx;
                if !mask_chunk8_valid(curr_mask_row, cx) || !mask_chunk8_valid(ref_mask_row, ix) {
                    return None;
                }

                let curr = _mm256_loadu_ps(curr_row.add(cx));
                let i_ref = bilerp8_ptr(ref_row0, ref_row1, ix, vw00, vw10, vw01, vw11);
                let gx = bilerp8_ptr(gx_row0, gx_row1, ix, vw00, vw10, vw01, vw11);
                let gy = bilerp8_ptr(gy_row0, gy_row1, ix, vw00, vw10, vw01, vw11);

                let jac = _mm256_fmadd_ps(gy, v_dv, _mm256_mul_ps(gx, v_du));
                let residual = _mm256_sub_ps(i_ref, curr);
                let abs_res = _mm256_andnot_ps(v_abs_mask, residual);
                let huber_mask = _mm256_cmp_ps(abs_res, v_delta, _CMP_LE_OQ);
                let robust = _mm256_blendv_ps(
                    _mm256_div_ps(v_delta_inv_sigma, abs_res),
                    v_inv_sigma,
                    huber_mask,
                );

                sum_grad = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, residual), sum_grad);
                sum_hess = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, jac), sum_hess);
                sum_abs = _mm256_add_ps(sum_abs, abs_res);
                n_valid += 8;
            }
        }

        Some(PatchAccum {
            grad: hsum256(sum_grad) as f64,
            hess: hsum256(sum_hess) as f64,
            sum_abs_res: hsum256(sum_abs) as f64,
            n_valid,
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_translation_accum_avx2_4x4(
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    ref_grad_x: &Image<f32>,
    ref_grad_y: &Image<f32>,
    patch: TranslatedPatchFootprint,
    du_drho: f32,
    dv_drho: f32,
    inv_sigma_photo_sq: f32,
    huber_delta: f32,
) -> Option<PatchAccum> {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_andnot_ps, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_div_ps,
        _mm256_fmadd_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sub_ps,
        _CMP_LE_OQ,
    };

    debug_assert_eq!(patch.side, 4);

    let weights = patch.ref_fp.weights;
    let vw00 = _mm256_set1_ps(weights[0]);
    let vw10 = _mm256_set1_ps(weights[1]);
    let vw01 = _mm256_set1_ps(weights[2]);
    let vw11 = _mm256_set1_ps(weights[3]);
    let v_du = _mm256_set1_ps(du_drho);
    let v_dv = _mm256_set1_ps(dv_drho);
    let v_inv_sigma = _mm256_set1_ps(inv_sigma_photo_sq);
    let v_delta = _mm256_set1_ps(huber_delta);
    let v_delta_inv_sigma = _mm256_set1_ps(huber_delta * inv_sigma_photo_sq);
    let v_abs_mask = _mm256_set1_ps(-0.0);

    let mut sum_grad = _mm256_setzero_ps();
    let mut sum_hess = _mm256_setzero_ps();
    let mut sum_abs = _mm256_setzero_ps();
    let mut n_valid = 0usize;

    unsafe {
        for ly in (0..4).step_by(2) {
            let cy0 = patch.curr_y0 as usize + ly;
            let cy1 = cy0 + 1;
            let cx = patch.curr_x0 as usize;
            let ix = patch.ref_fp.x;
            let ry0 = patch.ref_fp.y + ly;
            let ry1 = ry0 + 1;

            let curr_mask_row0 = curr_valid.map(|mask| mask.row_ptr(cy0));
            let curr_mask_row1 = curr_valid.map(|mask| mask.row_ptr(cy1));
            let ref_mask_row0 = ref_valid.map(|mask| mask.row_ptr(ry0));
            let ref_mask_row1 = ref_valid.map(|mask| mask.row_ptr(ry1));
            if !mask_chunk4_valid(curr_mask_row0, cx)
                || !mask_chunk4_valid(curr_mask_row1, cx)
                || !mask_chunk4_valid(ref_mask_row0, ix)
                || !mask_chunk4_valid(ref_mask_row1, ix)
            {
                return None;
            }

            let curr = load4x2_ptr(curr_img.row_ptr(cy0), curr_img.row_ptr(cy1), cx, cx);
            let i_ref = bilerp4x2_ptr(
                ref_img.row_ptr(ry0),
                ref_img.row_ptr(ry1),
                ref_img.row_ptr(ry1),
                ref_img.row_ptr(ry1 + 1),
                ix,
                ix,
                vw00,
                vw10,
                vw01,
                vw11,
            );
            let gx = bilerp4x2_ptr(
                ref_grad_x.row_ptr(ry0),
                ref_grad_x.row_ptr(ry1),
                ref_grad_x.row_ptr(ry1),
                ref_grad_x.row_ptr(ry1 + 1),
                ix,
                ix,
                vw00,
                vw10,
                vw01,
                vw11,
            );
            let gy = bilerp4x2_ptr(
                ref_grad_y.row_ptr(ry0),
                ref_grad_y.row_ptr(ry1),
                ref_grad_y.row_ptr(ry1),
                ref_grad_y.row_ptr(ry1 + 1),
                ix,
                ix,
                vw00,
                vw10,
                vw01,
                vw11,
            );

            let jac = _mm256_fmadd_ps(gy, v_dv, _mm256_mul_ps(gx, v_du));
            let residual = _mm256_sub_ps(i_ref, curr);
            let abs_res = _mm256_andnot_ps(v_abs_mask, residual);
            let huber_mask = _mm256_cmp_ps(abs_res, v_delta, _CMP_LE_OQ);
            let robust = _mm256_blendv_ps(
                _mm256_div_ps(v_delta_inv_sigma, abs_res),
                v_inv_sigma,
                huber_mask,
            );

            sum_grad = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, residual), sum_grad);
            sum_hess = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, jac), sum_hess);
            sum_abs = _mm256_add_ps(sum_abs, abs_res);
            n_valid += 8;
        }

        Some(PatchAccum {
            grad: hsum256(sum_grad) as f64,
            hess: hsum256(sum_hess) as f64,
            sum_abs_res: hsum256(sum_abs) as f64,
            n_valid,
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilerp8_ptr(
    r0: *const f32,
    r1: *const f32,
    x: usize,
    vw00: std::arch::x86_64::__m256,
    vw10: std::arch::x86_64::__m256,
    vw01: std::arch::x86_64::__m256,
    vw11: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps};
    unsafe {
        let p00 = _mm256_loadu_ps(r0.add(x));
        let p10 = _mm256_loadu_ps(r0.add(x + 1));
        let p01 = _mm256_loadu_ps(r1.add(x));
        let p11 = _mm256_loadu_ps(r1.add(x + 1));
        let acc = _mm256_mul_ps(vw00, p00);
        let acc = _mm256_fmadd_ps(vw10, p10, acc);
        let acc = _mm256_fmadd_ps(vw01, p01, acc);
        _mm256_fmadd_ps(vw11, p11, acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load4x2_ptr(
    row0: *const f32,
    row1: *const f32,
    x0: usize,
    x1: usize,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_castps128_ps256, _mm256_insertf128_ps, _mm_loadu_ps};
    unsafe {
        let lo = _mm_loadu_ps(row0.add(x0));
        let hi = _mm_loadu_ps(row1.add(x1));
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilerp4x2_ptr(
    row00: *const f32,
    row01: *const f32,
    row10: *const f32,
    row11: *const f32,
    x0: usize,
    x1: usize,
    vw00: std::arch::x86_64::__m256,
    vw10: std::arch::x86_64::__m256,
    vw01: std::arch::x86_64::__m256,
    vw11: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_mul_ps};
    unsafe {
        let p00 = load4x2_ptr(row00, row10, x0, x1);
        let p10 = load4x2_ptr(row00, row10, x0 + 1, x1 + 1);
        let p01 = load4x2_ptr(row01, row11, x0, x1);
        let p11 = load4x2_ptr(row01, row11, x0 + 1, x1 + 1);
        let acc = _mm256_mul_ps(vw00, p00);
        let acc = _mm256_fmadd_ps(vw10, p10, acc);
        let acc = _mm256_fmadd_ps(vw01, p01, acc);
        _mm256_fmadd_ps(vw11, p11, acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mask_chunk8_valid(row: Option<*const f32>, x: usize) -> bool {
    let Some(row) = row else {
        return true;
    };
    unsafe {
        for i in 0..8 {
            if *row.add(x + i) <= 0.5 {
                return false;
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mask_chunk4_valid(row: Option<*const f32>, x: usize) -> bool {
    let Some(row) = row else {
        return true;
    };
    unsafe {
        for i in 0..4 {
            if *row.add(x + i) <= 0.5 {
                return false;
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    let mut tmp = [0.0f32; 8];
    unsafe {
        std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), v);
    }
    tmp.iter().sum()
}

unsafe fn bilinear_unchecked(img: &Image<f32>, x: usize, y: usize, weights: [f32; 4]) -> f32 {
    unsafe {
        let p00 = img.get_unchecked(x, y);
        let p10 = img.get_unchecked(x + 1, y);
        let p01 = img.get_unchecked(x, y + 1);
        let p11 = img.get_unchecked(x + 1, y + 1);
        weights[0] * p00 + weights[1] * p10 + weights[2] * p01 + weights[3] * p11
    }
}

#[inline(always)]
unsafe fn bilerp_ptr(r0: *const f32, r1: *const f32, x: usize, weights: [f32; 4]) -> f32 {
    unsafe {
        let p00 = *r0.add(x);
        let p10 = *r0.add(x + 1);
        let p01 = *r1.add(x);
        let p11 = *r1.add(x + 1);
        weights[0] * p00 + weights[1] * p10 + weights[2] * p01 + weights[3] * p11
    }
}

fn sample_valid_nearest(mask: Option<&Image<f32>>, u: f64, v: f64) -> bool {
    let Some(mask) = mask else {
        return true;
    };
    let x = u as isize;
    let y = v as isize;
    if x < 0 || y < 0 || x >= mask.width() as isize || y >= mask.height() as isize {
        return false;
    }
    // SAFETY: bounds were checked above.
    unsafe { mask.get_unchecked(x as usize, y as usize) > 0.5 }
}

fn scaled_image_from_u8(gray: &[u8], width: usize, height: usize, scale: f64) -> Image<f32> {
    if (scale - 1.0).abs() < f64::EPSILON {
        return Image::from_vec(width, height, gray.iter().map(|&v| v as f32).collect());
    }
    let out_w = ((width as f64) * scale + 0.5).floor().max(1.0) as usize;
    let out_h = ((height as f64) * scale + 0.5).floor().max(1.0) as usize;
    let mut data = vec![0.0f32; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = ((x as f64 + 0.5) / scale - 0.5).clamp(0.0, (width - 1) as f64);
            let src_y = ((y as f64 + 0.5) / scale - 0.5).clamp(0.0, (height - 1) as f64);
            data[y * out_w + x] = sample_u8_bilinear(gray, width, height, src_x, src_y);
        }
    }
    Image::from_vec(out_w, out_h, data)
}

fn scaled_mask_from_u8(mask: &[u8], width: usize, height: usize, scale: f64) -> Image<f32> {
    if (scale - 1.0).abs() < f64::EPSILON {
        return Image::from_vec(
            width,
            height,
            mask.iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect(),
        );
    }
    let out_w = ((width as f64) * scale + 0.5).floor().max(1.0) as usize;
    let out_h = ((height as f64) * scale + 0.5).floor().max(1.0) as usize;
    let mut data = vec![0.0f32; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = ((x as f64 + 0.5) / scale - 0.5).clamp(0.0, (width - 1) as f64);
            let src_y = ((y as f64 + 0.5) / scale - 0.5).clamp(0.0, (height - 1) as f64);
            if mask_bilinear_footprint_valid(mask, width, height, src_x, src_y) {
                data[y * out_w + x] = 1.0;
            }
        }
    }
    Image::from_vec(out_w, out_h, data)
}

fn mask_bilinear_footprint_valid(mask: &[u8], width: usize, height: usize, u: f64, v: f64) -> bool {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return false;
    }
    let ix = u.floor() as usize;
    let iy = v.floor() as usize;
    mask[iy * width + ix] != 0
        && mask[iy * width + ix + 1] != 0
        && mask[(iy + 1) * width + ix] != 0
        && mask[(iy + 1) * width + ix + 1] != 0
}

fn build_mask_pyramid(
    mask: &[u8],
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
) -> Vec<Image<f32>> {
    if let Some(offset) = dyadic_scale_offset(scale) {
        return build_dyadic_mask_pyramid(mask, width, height, offset, levels);
    }

    let mut out = Vec::with_capacity(levels);
    out.push(scaled_mask_from_u8(mask, width, height, scale));
    for level in 1..levels {
        let prev = &out[level - 1];
        let next_w = (prev.width() / 2).max(1);
        let next_h = (prev.height() / 2).max(1);
        let mut next = vec![0.0f32; next_w * next_h];
        for y in 0..next_h {
            for x in 0..next_w {
                if mask_pyrdown_footprint_valid(prev, x * 2, y * 2) {
                    next[y * next_w + x] = 1.0;
                }
            }
        }
        out.push(Image::from_vec(next_w, next_h, next));
    }
    out
}

fn build_dyadic_mask_pyramid(
    mask: &[u8],
    width: usize,
    height: usize,
    offset: usize,
    levels: usize,
) -> Vec<Image<f32>> {
    let total_levels = offset + levels;
    let mut full = Vec::with_capacity(total_levels);
    full.push(Image::from_vec(
        width,
        height,
        mask.iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect(),
    ));
    for level in 1..total_levels {
        let prev = &full[level - 1];
        let next_w = (prev.width() / 2).max(1);
        let next_h = (prev.height() / 2).max(1);
        let mut next = vec![0.0f32; next_w * next_h];
        for y in 0..next_h {
            for x in 0..next_w {
                if mask_pyrdown_footprint_valid(prev, x * 2, y * 2) {
                    next[y * next_w + x] = 1.0;
                }
            }
        }
        full.push(Image::from_vec(next_w, next_h, next));
    }
    full.into_iter().skip(offset).collect()
}

fn build_bilinear_valid_pyramid(valid_pyramid: &[Image<f32>]) -> Vec<Image<f32>> {
    valid_pyramid.iter().map(bilinear_valid_image).collect()
}

fn bilinear_valid_image(mask: &Image<f32>) -> Image<f32> {
    let width = mask.width();
    let height = mask.height();
    let mut data = vec![0.0f32; width * height];
    if width < 2 || height < 2 {
        return Image::from_vec(width, height, data);
    }
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            // SAFETY: x/y are constrained so the full bilinear footprint is in bounds.
            let valid = unsafe {
                mask.get_unchecked(x, y) > 0.5
                    && mask.get_unchecked(x + 1, y) > 0.5
                    && mask.get_unchecked(x, y + 1) > 0.5
                    && mask.get_unchecked(x + 1, y + 1) > 0.5
            };
            if valid {
                data[y * width + x] = 1.0;
            }
        }
    }
    Image::from_vec(width, height, data)
}

fn mask_pyrdown_footprint_valid(mask: &Image<f32>, cx: usize, cy: usize) -> bool {
    let cx = cx as isize;
    let cy = cy as isize;
    for dy in -2..=2 {
        for dx in -2..=2 {
            let x = cx + dx;
            let y = cy + dy;
            if x < 0
                || y < 0
                || x >= mask.width() as isize
                || y >= mask.height() as isize
                || mask.get(x as usize, y as usize) <= 0.5
            {
                return false;
            }
        }
    }
    true
}

fn build_pinhole_to_raw_lut(
    raw_camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    width: usize,
    height: usize,
) -> Vec<Option<UndistortSample>> {
    let mut lut = Vec::with_capacity(width * height);
    for v in 0..height {
        for u in 0..width {
            let x = (u as f64 - intrinsics.cx) / intrinsics.fx;
            let y = (v as f64 - intrinsics.cy) / intrinsics.fy;
            let raw_uv = raw_camera.project(&Vector3::new(x, y, 1.0));
            lut.push(undistort_sample(raw_uv[0], raw_uv[1], width, height));
        }
    }
    lut
}

fn undistort_sample(u: f64, v: f64, width: usize, height: usize) -> Option<UndistortSample> {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return None;
    }
    let ix = u.floor() as usize;
    let iy = v.floor() as usize;
    let dx = (u - ix as f64) as f32;
    let dy = (v - iy as f64) as f32;
    let one_minus_dx = 1.0 - dx;
    let one_minus_dy = 1.0 - dy;
    Some(UndistortSample {
        idx00: iy * width + ix,
        idx10: iy * width + ix + 1,
        idx01: (iy + 1) * width + ix,
        idx11: (iy + 1) * width + ix + 1,
        weights: [
            one_minus_dx * one_minus_dy,
            dx * one_minus_dy,
            one_minus_dx * dy,
            dx * dy,
        ],
    })
}

fn sample_u8_bilinear(gray: &[u8], width: usize, height: usize, u: f64, v: f64) -> f32 {
    let ix = u.floor() as usize;
    let iy = v.floor() as usize;
    let ix1 = (ix + 1).min(width - 1);
    let iy1 = (iy + 1).min(height - 1);
    let dx = u - ix as f64;
    let dy = v - iy as f64;
    let i00 = gray[iy * width + ix] as f64;
    let i10 = gray[iy * width + ix1] as f64;
    let i01 = gray[iy1 * width + ix] as f64;
    let i11 = gray[iy1 * width + ix1] as f64;
    ((1.0 - dx) * (1.0 - dy) * i00 + dx * (1.0 - dy) * i10 + (1.0 - dx) * dy * i01 + dx * dy * i11)
        as f32
}

fn build_pyramid_from_u8(
    gray: &[u8],
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
) -> Vec<Image<f32>> {
    let base = scaled_image_from_u8(gray, width, height, scale);
    Pyramid::build(&base, levels, 1.0).levels
}

fn empty_pyramid() -> Pyramid {
    Pyramid {
        levels: Vec::new(),
        u8_levels: Vec::new(),
        padded_levels: Vec::new(),
        pad_border: 0,
    }
}

fn dyadic_scale_offset(scale: f64) -> Option<usize> {
    let mut dyadic = 1.0;
    for offset in 0..=8 {
        if (scale - dyadic).abs() <= 1e-12 {
            return Some(offset);
        }
        dyadic *= 0.5;
    }
    None
}

fn gradients(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    let mut gx = vec![0.0f32; img.width() * img.height()];
    let mut gy = vec![0.0f32; img.width() * img.height()];
    if img.width() >= 3 {
        for y in 0..img.height() {
            for x in 1..img.width() - 1 {
                gx[y * img.width() + x] = 0.5 * (img.get(x + 1, y) - img.get(x - 1, y));
            }
        }
    }
    if img.height() >= 3 {
        for y in 1..img.height() - 1 {
            for x in 0..img.width() {
                gy[y * img.width() + x] = 0.5 * (img.get(x, y + 1) - img.get(x, y - 1));
            }
        }
    }
    (
        Image::from_vec(img.width(), img.height(), gx),
        Image::from_vec(img.width(), img.height(), gy),
    )
}

fn scaled_intrinsics(base_scale: f64, levels: usize) -> Vec<ScaledIntrinsics> {
    (0..levels)
        .map(|level| {
            let level_scale = base_scale / (1usize << level) as f64;
            ScaledIntrinsics {
                scale_from_original: level_scale,
            }
        })
        .collect()
}

fn scale_seeds(seeds: &[SparseDepthPrior], scale: f64) -> Vec<SparseDepthPrior> {
    seeds
        .iter()
        .map(|seed| SparseDepthPrior {
            uv: seed.uv * scale,
            rho: seed.rho,
            rho_var: seed.rho_var,
        })
        .collect()
}

fn nearby_seed_weights(
    cu: f64,
    cv: f64,
    seeds: &[SparseDepthPrior],
    seed_grid: &SeedGrid,
    settings: &PatchDepthSettings,
) -> NearbySeeds {
    let radius = seed_grid.cell_size;
    let radius_sq = radius * radius;
    let ci_min = ((cu - radius) / seed_grid.cell_size).floor().max(0.0) as usize;
    let cj_min = ((cv - radius) / seed_grid.cell_size).floor().max(0.0) as usize;
    let ci_max = ((cu + radius) / seed_grid.cell_size)
        .floor()
        .min((seed_grid.cols - 1) as f64) as usize;
    let cj_max = ((cv + radius) / seed_grid.cell_size)
        .floor()
        .min((seed_grid.rows - 1) as f64) as usize;
    let mut out = NearbySeeds::new();
    for cj in cj_min..=cj_max {
        for ci in ci_min..=ci_max {
            let cell_idx = cj * seed_grid.cols + ci;
            for k in seed_grid.starts[cell_idx]..seed_grid.starts[cell_idx + 1] {
                let idx = seed_grid.ids[k];
                let du = seeds[idx].uv[0] - cu;
                let dv = seeds[idx].uv[1] - cv;
                let dist_sq = du * du + dv * dv;
                if dist_sq > radius_sq {
                    continue;
                }
                let dist = dist_sq.sqrt();
                let w_spatial = 1.0 - dist / radius;
                let var_capped = seeds[idx]
                    .rho_var
                    .max(settings.sigma_seed_floor * settings.sigma_seed_floor);
                if !out.push(NearbySeed {
                    idx,
                    w_spatial,
                    precision: 1.0 / var_capped,
                }) {
                    return out;
                }
            }
        }
    }
    out
}

#[cfg(feature = "parallel")]
fn patch_centers(
    width: usize,
    height: usize,
    settings: &PatchDepthSettings,
) -> Vec<(usize, usize)> {
    let half = settings.patch_size / 2;
    let u_count = width
        .saturating_sub(half)
        .saturating_sub(half)
        .saturating_add(settings.patch_stride - 1)
        / settings.patch_stride;
    let v_count = height
        .saturating_sub(half)
        .saturating_sub(half)
        .saturating_add(settings.patch_stride - 1)
        / settings.patch_stride;
    let mut centers = Vec::with_capacity(u_count * v_count);
    for v in (half..height.saturating_sub(half)).step_by(settings.patch_stride) {
        for u in (half..width.saturating_sub(half)).step_by(settings.patch_stride) {
            centers.push((u, v));
        }
    }
    centers
}

struct FuseAccumulator {
    n_cells_u: usize,
    n_cells_v: usize,
    rho_acc: Vec<f64>,
    w_acc: Vec<f64>,
    status: Vec<PatchStatus>,
}

impl FuseAccumulator {
    fn new(width: usize, height: usize, settings: &PatchDepthSettings) -> Self {
        let n_cells_u = (width / settings.cell_size).max(1);
        let n_cells_v = (height / settings.cell_size).max(1);
        let n = n_cells_u * n_cells_v;
        Self {
            n_cells_u,
            n_cells_v,
            rho_acc: vec![0.0; n],
            w_acc: vec![0.0; n],
            status: vec![PatchStatus::Unknown; n],
        }
    }

    fn add(&mut self, cu: f64, cv: f64, estimate: PatchEstimate, settings: &PatchDepthSettings) {
        if estimate.status == PatchStatus::Unknown || estimate.status == PatchStatus::Rejected {
            return;
        }
        let ci = (cu as usize / settings.cell_size).min(self.n_cells_u - 1);
        let cj = (cv as usize / settings.cell_size).min(self.n_cells_v - 1);
        let idx = cj * self.n_cells_u + ci;
        let status_weight = match estimate.status {
            PatchStatus::PhotoRefined => settings.status_weight_photo,
            PatchStatus::SeedOnly => settings.status_weight_seed,
            _ => 0.0,
        };
        let w = status_weight / estimate.var.max(settings.var_floor);
        self.rho_acc[idx] += w * estimate.rho;
        self.w_acc[idx] += w;
        if (estimate.status as u8) > (self.status[idx] as u8) {
            self.status[idx] = estimate.status;
        }
    }

    fn finish(self) -> PatchDepthOutput {
        let mut depth = vec![f32::NAN; self.rho_acc.len()];
        let mut variance = vec![f32::INFINITY; self.rho_acc.len()];
        for i in 0..self.rho_acc.len() {
            if self.w_acc[i] > 0.0 {
                depth[i] = (1.0 / (self.rho_acc[i] / self.w_acc[i])) as f32;
                variance[i] = (1.0 / self.w_acc[i]) as f32;
            }
        }

        PatchDepthOutput {
            depth_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, depth)
                .expect("depth cell size"),
            variance_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, variance)
                .expect("variance cell size"),
            status_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, self.status)
                .expect("status cell size"),
        }
    }
}

#[cfg(feature = "parallel")]
fn fuse_cells(
    patches: &[(f64, f64, PatchEstimate)],
    width: usize,
    height: usize,
    settings: &PatchDepthSettings,
) -> PatchDepthOutput {
    let mut fuse = FuseAccumulator::new(width, height, settings);
    for &(cu, cv, estimate) in patches {
        fuse.add(cu, cv, estimate, settings);
    }
    fuse.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematical::camera::PinholeModel;

    fn camera() -> (Arc<dyn CameraModel>, CameraIntrinsics) {
        let intr = CameraIntrinsics::new(40.0, 40.0, 16.0, 16.0);
        (
            Arc::new(PinholeModel {
                fx: intr.fx,
                fy: intr.fy,
                cx: intr.cx,
                cy: intr.cy,
            }),
            intr,
        )
    }

    fn frame(id: u64, tx: f64, gray: Vec<u8>) -> FrameProducts {
        let mut pose = Matrix4::identity();
        pose[(0, 3)] = tx;
        FrameProducts {
            frame_id: id,
            stamp: id as f64,
            gray,
            width: 32,
            height: 32,
            pose_t_wc: pose,
        }
    }

    fn textured_image() -> Vec<u8> {
        let mut data = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                data[y * 32 + x] = ((x * 5 + y * 3) % 255) as u8;
            }
        }
        data
    }

    #[test]
    fn keyframe_buffer_uses_baseline_gate() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings::default();
        let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
        let seeds = vec![SparseDepthPrior {
            uv: Vector2::new(16.0, 16.0),
            rho: 0.5,
            rho_var: 0.01,
        }];

        let img = textured_image();
        assert!(mapper
            .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
            .is_none());
        assert_eq!(mapper.keyframe_count(), 1);
        assert!(mapper
            .update_with_priors(frame(1, 0.001, img.clone()), &seeds, None, 0.0)
            .is_none());
        assert_eq!(mapper.keyframe_count(), 2);
        assert!(mapper
            .update_with_priors(frame(2, 0.02, img), &seeds, None, 0.0)
            .is_some());
        assert_eq!(mapper.keyframe_count(), 2);
    }

    #[test]
    fn seed_priors_produce_cell_depths() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings {
            min_photo_curvature: 1e12,
            ..PatchDepthSettings::default()
        };
        let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
        let seeds = vec![SparseDepthPrior {
            uv: Vector2::new(16.0, 16.0),
            rho: 0.5,
            rho_var: 0.01,
        }];
        let img = vec![120u8; 32 * 32];

        assert!(mapper
            .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
            .is_none());
        let out = mapper
            .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
            .unwrap();
        assert!(out
            .depth_cells
            .data
            .iter()
            .any(|z| z.is_finite() && (*z - 2.0).abs() < 0.1));
        assert!(out.status_cells.data.contains(&PatchStatus::SeedOnly));
    }

    #[test]
    fn bearing_lut_handles_distorted_path_for_photometric_solve() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings {
            min_photo_curvature: 0.0,
            max_photo_residual: 255.0,
            ..PatchDepthSettings::default()
        };
        let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
        let seeds = vec![SparseDepthPrior {
            uv: Vector2::new(16.0, 16.0),
            rho: 0.5,
            rho_var: 0.01,
        }];
        let img = textured_image();

        assert!(mapper
            .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
            .is_none());
        let out = mapper
            .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
            .unwrap();
        assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
    }

    #[test]
    fn fast_translation_refines_undistorted_pinhole_patches() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings {
            camera_mode: PatchDepthCameraMode::UndistortedPinhole,
            warp_mode: PatchDepthWarpMode::FastTranslation,
            min_photo_curvature: 0.0,
            max_photo_residual: 255.0,
            ..PatchDepthSettings::default()
        };
        let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
        let seeds = vec![SparseDepthPrior {
            uv: Vector2::new(16.0, 16.0),
            rho: 0.5,
            rho_var: 0.01,
        }];
        let img = textured_image();

        assert!(mapper
            .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
            .is_none());
        let out = mapper
            .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
            .unwrap();
        assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
    }

    #[test]
    fn fast_translation_refines_undistorted_pinhole_4x4_patches() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings {
            camera_mode: PatchDepthCameraMode::UndistortedPinhole,
            warp_mode: PatchDepthWarpMode::FastTranslation,
            patch_size: 4,
            patch_stride: 2,
            cell_size: 4,
            min_photo_curvature: 0.0,
            max_photo_residual: 255.0,
            ..PatchDepthSettings::default()
        };
        let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
        let seeds = vec![SparseDepthPrior {
            uv: Vector2::new(16.0, 16.0),
            rho: 0.5,
            rho_var: 0.01,
        }];
        let img = textured_image();

        assert!(mapper
            .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
            .is_none());
        let out = mapper
            .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
            .unwrap();
        assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
    }

    #[test]
    fn raw_bearing_lut_interpolates_fractional_pixels() {
        let (camera, intr) = camera();
        let settings = PatchDepthSettings::default();
        let mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();

        let bearing = mapper.bearing_at_original(16.5, 16.25).unwrap();
        assert!((bearing[0] - 0.5 / 40.0).abs() < 1e-12);
        assert!((bearing[1] - 0.25 / 40.0).abs() < 1e-12);
        assert!((bearing[2] - 1.0).abs() < 1e-12);
    }
}
