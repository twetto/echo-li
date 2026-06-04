use super::*;

impl PatchDepthMapper {
    pub(super) fn solve(
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

    pub(super) fn solve_tiled_bearing(
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

    pub(super) fn solve_tiled_bearing_tile_patches(
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

    pub(super) fn solve_one_tiled_bearing_patch_level(
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
    pub(super) fn accumulate_tiled_patch(
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
    pub(super) fn tiled_pixel_photo_weight(
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

    pub(super) fn tiled_blend_weight(
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
    pub(super) fn solve_one_patch_dispatch(
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

    pub(super) fn solve_one_patch(
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
    pub(super) fn solve_one_per_patch_bearing(
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

        // Structure gate (parity with solve_one_patch): reject patches whose
        // keyframe gradient structure tensor is too weak / ill-conditioned along
        // the epipolar direction before paying for the GN refinement. No-op when
        // min_structure_eigen and max_structure_condition are both ≤ 0.
        if !self.patch_has_enough_structure(
            cu,
            cv,
            eta_init,
            ref_keyframe,
            &intrinsics_by_level[0],
            rel_pose,
        ) {
            return PatchEstimate::rejected(eta_init);
        }

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

    pub(super) fn patch_has_enough_structure(
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

    pub(super) fn search_initial_eta(
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
}
