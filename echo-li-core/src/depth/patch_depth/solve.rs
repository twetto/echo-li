use super::*;

/// `ln(max_depth * range_per_z)`: the far bound of eta at a bearing.
fn eta_max_for(s: &PatchDepthSettings, range_per_z: f64) -> f64 {
    (s.max_depth * range_per_z).ln()
}

/// Outcome of the candidate search over eta.
struct CandidateSearch {
    best_eta: f64,
    median_cost: Option<f64>,
    /// Second-best local minimum cost over the best; +inf when there is none.
    distinct_ratio: f64,
    /// The best candidate is bracketed by worse neighbours on both sides. A
    /// minimum at either end of the search is a cost curve still falling
    /// (a patch sliding into flat background, or the true depth outside the
    /// range), not a depth the image picked.
    best_interior: bool,
    /// The best candidate was the far end (largest eta) of the search.
    best_at_far: bool,
}

impl PatchDepthMapper {
    /// Solve every patch against the reference candidates (`refs[0]` is the
    /// global choice; the per-patch-bearing solver may pick another per patch)
    /// and densify. `scene_depth` sizes the per-patch parallax window.
    pub(super) fn solve(
        &self,
        depth_frame: &DepthFrameProducts,
        refs: &[RefCandidate],
        seeds: &[SparseDepthPrior],
        scene_depth: f64,
        prev_photo: &mut Option<PrevPhoto>,
    ) -> PatchDepthOutput {
        let curr_pyramid = depth_frame.pyramid.as_ref();
        let curr_valid_pyramid = depth_frame.valid_pyramid.as_ref().map(|p| p.as_slice());
        let width = curr_pyramid[0].width();
        let height = curr_pyramid[0].height();
        let scaled_intrinsics =
            scaled_intrinsics(self.settings.scale, self.settings.n_pyramid_levels);
        let scaled_seeds = scale_seeds(seeds, self.settings.scale);
        let seed_radius = self.seed_reach_px();
        let stride = self.settings.patch_stride.max(1) as f64;
        let ref_keyframe = refs[0].kf;
        let rel_pose = refs[0].rel.clone();

        // Coarse-to-fine, as in solve_tiled_bearing: the coarsest level is
        // solved from the sparse seeds and each finer level from the level
        // above, so a patch's prior is a validated neighbour a few pixels away
        // rather than a landmark far across the image. The constructor limits
        // this to the modes whose per-patch solver evaluates one level.
        let n_levels = if self.settings.coarse_to_fine {
            curr_pyramid
                .len()
                .min(ref_keyframe.ref_pyramid.len())
                .min(scaled_intrinsics.len())
                .max(1)
        } else {
            1
        };
        let mut grid = PatchGrid::new(width, height, &self.settings);
        let mut inherited: Vec<SparseDepthPrior> = Vec::new();
        for lvl in (0..n_levels).rev() {
            let (w_l, h_l) = (curr_pyramid[lvl].width(), curr_pyramid[lvl].height());
            let level_scale = 1.0 / (1usize << lvl) as f64;
            let (level_seeds, radius) = if n_levels == 1 {
                (scaled_seeds.clone(), seed_radius)
            } else {
                let coarsest = lvl + 1 == n_levels;
                let mut priors = std::mem::take(&mut inherited);
                if coarsest || self.settings.c2f_seed_all_levels {
                    priors.extend(scaled_seeds.iter().map(|seed| SparseDepthPrior {
                        uv: seed.uv * level_scale,
                        eta: seed.eta,
                        eta_var: seed.eta_var,
                    }));
                }
                let radius = if coarsest {
                    (seed_radius * level_scale).max(stride)
                } else {
                    self.settings.c2f_prior_radius_px.max(stride)
                };
                (priors, radius)
            };
            let seed_grid = SeedGrid::new(&level_seeds, radius, w_l, h_l);
            let centers = patch_centers(w_l, h_l, &self.settings);
            let level_refs: Vec<RefCandidate> =
                refs.iter().map(|r| r.at_pixel_scale(level_scale)).collect();
            let estimates = self.solve_level_estimates(
                &centers,
                &level_seeds,
                &seed_grid,
                curr_pyramid,
                curr_valid_pyramid,
                &level_refs,
                &scaled_intrinsics,
                lvl,
                scene_depth,
            );
            if lvl == 0 {
                let estimates = self.temporal_gate_photo_only(
                    estimates,
                    &grid,
                    &depth_frame.frame.pose_t_wc,
                    prev_photo,
                    &scaled_intrinsics[0],
                );
                for (u, v, estimate) in estimates {
                    grid.set(u, v, estimate);
                }
            } else {
                inherited = self.children_priors_from(estimates);
            }
        }

        let curr_img = &curr_pyramid[0];
        let curr_valid = curr_valid_pyramid.map(|p| &p[0]);
        let ref_img = &ref_keyframe.ref_pyramid[0];
        let ref_valid: Option<&Image<f32>> =
            ref_keyframe.bilinear_valid_pyramid.as_ref().map(|p| &p[0]);
        let densify = |g: &PatchGrid| {
            self.densify_grid(
                g,
                width,
                height,
                curr_img,
                curr_valid,
                ref_img,
                ref_valid,
                &scaled_intrinsics[0],
                &rel_pose,
            )
        };
        if !self.settings.photo_confidence_weighting {
            return densify(&grid);
        }
        // Image-confirmed patches first; seed-only patches fill only the pixels
        // they leave empty. The same rule as the tiled path, done as a merge of
        // two passes so the densify kernels stay untouched.
        let mut out = densify(&grid.retain(|p| p.status.is_photometric()));
        let fill = densify(&grid.retain(|p| p.status == PatchStatus::SeedOnly));
        for idx in 0..out.eta.data.len() {
            if !out.eta.data[idx].is_finite() && fill.eta.data[idx].is_finite() {
                out.eta.data[idx] = fill.eta.data[idx];
                out.eta_var.data[idx] = fill.eta_var.data[idx];
                out.status.data[idx] = fill.status.data[idx];
            }
        }
        out
    }

    /// Photo-only patches must agree with the previous frame's photo-only
    /// estimate at the same bearing (rotation-compensated, translation ignored
    /// over one frame) within `photo_temporal_tol`; the fresh estimates replace
    /// the remembered map whether or not they passed.
    fn temporal_gate_photo_only(
        &self,
        estimates: Vec<(usize, usize, PatchEstimate)>,
        grid: &PatchGrid,
        t_wc: &Matrix4<f64>,
        prev: &mut Option<PrevPhoto>,
        intr: &ScaledIntrinsics,
    ) -> Vec<(usize, usize, PatchEstimate)> {
        let tol = self.settings.photo_temporal_tol;
        if tol <= 0.0 {
            return estimates;
        }
        let (n_u, n_v, half, stride) = (grid.n_u, grid.n_v, grid.half, grid.stride);
        let r_curr: Matrix3<f64> = t_wc.fixed_view::<3, 3>(0, 0).into_owned();
        let gate = prev
            .as_ref()
            .filter(|p| p.n_u == n_u && p.n_v == n_v)
            .map(|p| (p, p.r_wc.transpose() * r_curr));
        let mut fresh = vec![f32::NAN; n_u * n_v];
        let out = estimates
            .into_iter()
            .map(|(u, v, mut e)| {
                if e.status != PatchStatus::PhotoOnly {
                    return (u, v, e);
                }
                let iu = (u - half) / stride;
                let iv = (v - half) / stride;
                if iu < n_u && iv < n_v {
                    fresh[iv * n_u + iu] = e.eta as f32;
                }
                let consistent = gate.as_ref().is_some_and(|(p, r_pc)| {
                    let Some(b) = self.bearing_for_scaled_pixel(u as f64, v as f64, intr) else {
                        return false;
                    };
                    let Some(uv) = self.camera.project_ray(&(r_pc * b)) else {
                        return false;
                    };
                    let ju = ((uv[0] * self.settings.scale - half as f64) / stride as f64).round();
                    let jv = ((uv[1] * self.settings.scale - half as f64) / stride as f64).round();
                    if !(ju >= 0.0 && jv >= 0.0 && ju < n_u as f64 && jv < n_v as f64) {
                        return false;
                    }
                    let prev_eta = p.etas[jv as usize * n_u + ju as usize];
                    prev_eta.is_finite() && (prev_eta as f64 - e.eta).abs() <= tol
                });
                if !consistent {
                    self.funnel.hit(Funnel::Temporal);
                    e = PatchEstimate::rejected(e.eta);
                }
                (u, v, e)
            })
            .collect();
        *prev = Some(PrevPhoto {
            r_wc: r_curr,
            etas: fresh,
            n_u,
            n_v,
        });
        out
    }

    /// Candidate search over eta around `eta_init`: the best-cost start for the
    /// refinement, and the median candidate cost for the contrast test. `cost`
    /// returns the mean absolute residual at an eta, or None if nothing was valid.
    fn search_candidates(
        &self,
        eta_init: f64,
        eta_min: f64,
        eta_max: f64,
        hypotheses: &[f64],
        mut cost: impl FnMut(f64) -> Option<f64>,
    ) -> (f64, Option<f64>) {
        let n = self.settings.n_search_candidates.max(1);
        let half = self.settings.search_half_range.max(0.0);
        if n < 2 || half <= 0.0 {
            return (eta_init, None);
        }
        let lo = (eta_init - half).clamp(eta_min, eta_max);
        let hi = (eta_init + half).clamp(eta_min, eta_max);
        let mut candidates: Vec<f64> = (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect();
        if self.settings.photo_search_parent_hypotheses {
            for &h in hypotheses {
                let h = h.clamp(eta_min, eta_max);
                if candidates.iter().all(|c| (c - h).abs() > 0.02) {
                    candidates.push(h);
                }
            }
        }
        let mut costs = Vec::with_capacity(candidates.len());
        let mut best = (eta_init, f64::INFINITY);
        for eta in candidates {
            if let Some(c) = cost(eta) {
                costs.push(c);
                if c < best.1 {
                    best = (eta, c);
                }
            }
        }
        if costs.len() < 3 {
            return (eta_init, None);
        }
        costs.sort_by(|a, b| a.total_cmp(b));
        (best.0, Some(costs[costs.len() / 2]))
    }

    /// The contrast veto: with a median candidate cost in hand, a refined
    /// residual that is not clearly below it means the cost curve was flat and
    /// the image did not decide the depth.
    fn fails_contrast(&self, final_residual: f64, median_cost: Option<f64>) -> bool {
        let ratio = self.settings.photo_contrast_max_ratio;
        match median_cost {
            Some(median) if ratio < 1.0 => median <= 1e-9 || final_residual > ratio * median,
            _ => false,
        }
    }

    /// Every patch centre of one pyramid level through the per-patch solver.
    #[allow(clippy::too_many_arguments)]
    fn solve_level_estimates(
        &self,
        centers: &[(usize, usize)],
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        curr_pyramid: &[Image<f32>],
        curr_valid_pyramid: Option<&[Image<f32>]>,
        refs: &[RefCandidate],
        intrinsics_by_level: &[ScaledIntrinsics],
        level: usize,
        scene_depth: f64,
    ) -> Vec<(usize, usize, PatchEstimate)> {
        let solve_at = |&(u, v): &(usize, usize)| {
            (
                u,
                v,
                self.solve_one_patch_dispatch(
                    u as f64,
                    v as f64,
                    seeds,
                    seed_grid,
                    curr_pyramid,
                    curr_valid_pyramid,
                    refs,
                    intrinsics_by_level,
                    level,
                    scene_depth,
                ),
            )
        };
        #[cfg(feature = "parallel")]
        {
            // Target ~8 chunks per thread; degrades gracefully when patches < threads.
            let min_len = (centers.len() / (rayon::current_num_threads() * 8)).max(1);
            centers
                .par_iter()
                .with_min_len(min_len)
                .map(solve_at)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            centers.iter().map(solve_at).collect()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn densify_grid(
        &self,
        grid: &PatchGrid,
        width: usize,
        height: usize,
        curr_img: &Image<f32>,
        curr_valid: Option<&Image<f32>>,
        ref_img: &Image<f32>,
        ref_valid: Option<&Image<f32>>,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> PatchDepthOutput {
        #[cfg(feature = "parallel")]
        {
            densify_pixels_parallel(
                grid, width, height, curr_img, curr_valid, ref_img, ref_valid, self, intr, rel_pose,
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            densify_pixels(
                grid, width, height, curr_img, curr_valid, ref_img, ref_valid, self, intr, rel_pose,
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
        let stride = self.settings.patch_stride.max(1) as f64;
        let scaled_seeds = scale_seeds(seeds, self.settings.scale);
        let layouts = self
            .tiled_bearing_levels
            .as_ref()
            .expect("tiled bearing layout must exist");
        let seed_radius = self.seed_reach_px();

        // Coarse-to-fine: the coarsest level is solved from the sparse seeds and
        // each finer level from the level above. A patch's prior is then a
        // photometrically validated neighbour a few pixels away rather than a
        // landmark far across the image, so the seed radius no longer trades
        // coverage against leakage: at the coarsest level a small radius already
        // reaches far in level-0 pixels, and below it the prior travels with the
        // solution.
        let n_levels = if self.settings.coarse_to_fine {
            depth_frame
                .levels
                .len()
                .min(ref_keyframe.levels.len())
                .min(layouts.len())
                .max(1)
        } else {
            1
        };
        let mut inherited: Vec<SparseDepthPrior> = Vec::new();
        let mut level0_results: Vec<TiledPatchResult> = Vec::new();
        for lvl in (0..n_levels).rev() {
            let layout = &layouts[lvl];
            let curr = &depth_frame.levels[lvl];
            let refl = &ref_keyframe.levels[lvl];
            let level_scale = 1.0 / (1usize << lvl) as f64;
            let (level_seeds, radius, level_arg): (Vec<SparseDepthPrior>, f64, Option<usize>) =
                if !self.settings.coarse_to_fine {
                    (scaled_seeds.clone(), seed_radius, None)
                } else {
                    let coarsest = lvl + 1 == n_levels;
                    let mut priors = std::mem::take(&mut inherited);
                    if coarsest || self.settings.c2f_seed_all_levels {
                        priors.extend(scaled_seeds.iter().map(|seed| SparseDepthPrior {
                            uv: seed.uv * level_scale,
                            eta: seed.eta,
                            eta_var: seed.eta_var,
                        }));
                    }
                    let radius = if coarsest {
                        (seed_radius * level_scale).max(stride)
                    } else {
                        self.settings.c2f_prior_radius_px.max(stride)
                    };
                    (priors, radius, Some(lvl))
                };
            let tile_seeds = assign_tiled_bearing_seeds(layout, &level_seeds, 1.0, half);
            let results = self.solve_tiled_level(
                curr,
                refl,
                layout,
                &tile_seeds,
                depth_frame,
                ref_keyframe,
                &rel_pose,
                sigma_warp_sq,
                radius,
                level_arg,
            );
            if lvl == 0 {
                level0_results = results;
            } else {
                inherited = self.children_priors_from(
                    results.iter().map(|r| (r.global_u, r.global_v, r.estimate)),
                );
            }
        }

        if self.settings.photo_confidence_weighting {
            // Image-confirmed patches first; seed-only patches fill only what
            // they leave empty, instead of outvoting them with prior confidence.
            for (only, fill_only) in [
                (PatchStatus::PhotoRefined, false),
                (PatchStatus::SeedOnly, true),
            ] {
                self.accumulate_tiled_results(
                    &level0_results,
                    Some(only),
                    fill_only,
                    curr_level,
                    ref_level,
                    &rel_pose,
                    width,
                    height,
                    &mut eta_acc,
                    &mut w_acc,
                    &mut status,
                );
            }
        } else {
            self.accumulate_tiled_results(
                &level0_results,
                None,
                false,
                curr_level,
                ref_level,
                &rel_pose,
                width,
                height,
                &mut eta_acc,
                &mut w_acc,
                &mut status,
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

    /// Every patch of one pyramid level, tile by tile.
    #[allow(clippy::too_many_arguments)]
    fn solve_tiled_level(
        &self,
        curr_level: &TiledBearingFrameLevel,
        ref_level: &TiledBearingKeyframeLevel,
        layout: &TiledBearingLevel,
        tile_seeds: &[Vec<SparseDepthPrior>],
        depth_frame: &TiledBearingFrameProducts,
        ref_keyframe: &TiledBearingKeyframe,
        rel_pose: &RelativePose,
        sigma_warp_sq: f64,
        prior_radius_px: f64,
        level: Option<usize>,
    ) -> Vec<TiledPatchResult> {
        #[cfg(feature = "parallel")]
        {
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
                        tile_seeds,
                        depth_frame,
                        ref_keyframe,
                        rel_pose,
                        sigma_warp_sq,
                        prior_radius_px,
                        level,
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..curr_level.tiles.len())
                .flat_map(|tile_idx| {
                    self.solve_tiled_bearing_tile_patches(
                        tile_idx,
                        curr_level,
                        ref_level,
                        layout,
                        tile_seeds,
                        depth_frame,
                        ref_keyframe,
                        rel_pose,
                        sigma_warp_sq,
                        prior_radius_px,
                        level,
                    )
                })
                .collect()
        }
    }

    /// Priors for the next finer level, from this level's solution. A dyadic
    /// pyramid with a centred pyrdown maps a level-l pixel to 2x at level l-1.
    /// A parent hands down its posterior (seed anchor plus whatever the image
    /// added), inflated by `c2f_var_inflation` per level; seed-only parents
    /// therefore carry the seeds' reach down the pyramid where the image had
    /// nothing to say, and refined parents carry a basin the image confirmed.
    fn children_priors_from(
        &self,
        parents: impl IntoIterator<Item = (usize, usize, PatchEstimate)>,
    ) -> Vec<SparseDepthPrior> {
        let s = &self.settings;
        parents
            .into_iter()
            .filter_map(|(u, v, e)| {
                if !e.eta.is_finite() || !(e.inv_var_w > 0.0) {
                    return None;
                }
                // The parent's posterior variance, recovered from its fusion weight.
                let posterior_var = match e.status {
                    PatchStatus::PhotoRefined | PatchStatus::PhotoOnly
                        if e.confidence >= s.c2f_min_confidence && e.photo_precision > 0.0 =>
                    {
                        s.status_weight_photo / e.inv_var_w
                    }
                    PatchStatus::SeedOnly if s.c2f_propagate_seed_only => {
                        s.status_weight_seed / e.inv_var_w
                    }
                    _ => return None,
                };
                let eta_var = s.c2f_var_inflation * posterior_var;
                Some(SparseDepthPrior {
                    uv: Vector2::new(u as f64 * 2.0, v as f64 * 2.0),
                    eta: e.eta,
                    eta_var: eta_var.max(s.var_floor),
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_tiled_results(
        &self,
        results: &[TiledPatchResult],
        only: Option<PatchStatus>,
        fill_only: bool,
        curr_level: &TiledBearingFrameLevel,
        ref_level: &TiledBearingKeyframeLevel,
        rel_pose: &RelativePose,
        width: usize,
        height: usize,
        eta_acc: &mut [f32],
        w_acc: &mut [f32],
        status: &mut [PatchStatus],
    ) {
        for result in results {
            if only.is_some_and(|wanted| result.estimate.status != wanted) {
                continue;
            }
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
                    rel_pose,
                });
            self.accumulate_tiled_patch(
                eta_acc,
                w_acc,
                status,
                width,
                height,
                &curr_tile.tile,
                result.local_u,
                result.local_v,
                result.estimate,
                photo.as_ref(),
                fill_only,
            );
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
        prior_radius_px: f64,
        level: Option<usize>,
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
            prior_radius_px,
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
                    level,
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
        level: Option<usize>,
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
        if !(seed_weight_total > 0.0) {
            return PatchEstimate::unknown();
        }
        let eta_init = (seed_eta_sum / seed_weight_total).clamp(eta_min, eta_max);
        let hypotheses: Vec<f64> = nearby.iter().map(|it| seeds[it.idx].eta).collect();
        let (mut eta, median_cost) = if self.settings.photo_search {
            self.search_candidates(eta_init, eta_min, eta_max, &hypotheses, |eta_c| {
                let (_, _, mean_res, valid) = self.patch_residual_jacobian_fast_translation_tiled(
                    global_cu,
                    global_cv,
                    eta_c,
                    depth_frame,
                    ref_keyframe,
                    rel_pose,
                    sigma_warp_sq,
                    level,
                );
                (valid > 0).then_some(mean_res)
            })
        } else {
            (eta_init, None)
        };
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
                    level,
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
        if final_curvature >= min_curvature
            && final_residual <= self.settings.max_photo_residual
            && !self.fails_contrast(final_residual, median_cost)
        {
            PatchEstimate::photo_refined_with_information(
                eta,
                final_curvature,
                seed_precision_sum_eta * self.settings.lambda_seed,
                &self.settings,
            )
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
        fill_only: bool,
    ) {
        // With photometric-confidence weighting a refined patch counts by what
        // the image established, `hess_photo`, not by a posterior the prior may
        // have tightened; a patch that only sat on its prior then weighs nothing
        // against a neighbour the image confirmed.
        let base_w = if self.settings.photo_confidence_weighting
            && estimate.status == PatchStatus::PhotoRefined
        {
            let w = self.settings.status_weight_photo * estimate.photo_precision;
            if !(w > 0.0) {
                return;
            }
            w as f32
        } else {
            let Some(w) = estimate.inv_var_weight_f32() else {
                return;
            };
            w
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
                if fill_only && w_acc[idx] > 0.0 {
                    continue;
                }
                // eta = ln(range) is tile-frame-independent (range is purely radial), so
                // overlapping tiles fuse their eta directly — a geometric mean of range.
                // The eta -> 3D/z conversion is deferred to the consumer (vis / mapping).
                let w = base_w * tile_w * photo_w;
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
        refs: &[RefCandidate],
        intrinsics_by_level: &[ScaledIntrinsics],
        level: usize,
        scene_depth: f64,
    ) -> PatchEstimate {
        if self.camera_mode == PatchDepthCameraMode::PerPatchBearing {
            self.solve_one_per_patch_bearing(
                cu,
                cv,
                seeds,
                seed_grid,
                curr_pyramid,
                curr_valid_pyramid,
                refs,
                intrinsics_by_level,
                level,
                scene_depth,
            )
        } else {
            // The stacked-pyramid solvers only know level 0 and one reference;
            // the constructor keeps coarse-to-fine off these modes.
            debug_assert_eq!(level, 0);
            self.solve_one_patch(
                cu,
                cv,
                seeds,
                seed_grid,
                curr_pyramid,
                curr_valid_pyramid,
                refs[0].kf,
                intrinsics_by_level,
                &refs[0].t_ref_curr,
                refs[0].unc.scalar_sq,
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
        if !(seed_weight_total > 0.0) {
            return PatchEstimate::unknown();
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

    /// Reach of a landmark prior in working-scale pixels: `seed_radius_px`,
    /// never more than one patch. A landmark further away than the patch it
    /// would seed is on some other surface as often as not, and a prior
    /// carried that far is what painted "seed blobs" over untextured
    /// neighbourhoods.
    pub(super) fn seed_reach_px(&self) -> f64 {
        let cap = self.settings.patch_size as f64 * self.settings.seed_reach_max_patches.max(0.0);
        (self.settings.seed_radius_px * self.settings.scale)
            .min(cap)
            .max(1.0)
    }

    /// Search candidates between `eta_lo` and `eta_hi`, in decreasing eta
    /// (increasing inverse range, i.e. disparity order). With `search_step_px`
    /// set they are `search_step_px` pixels of disparity apart, `f_b` being the
    /// disparity per unit inverse range at this patch; otherwise the legacy
    /// `n_search_candidates` uniform in eta.
    fn candidate_etas(&self, eta_lo: f64, eta_hi: f64, f_b: f64) -> Vec<f64> {
        let s = &self.settings;
        if !(eta_hi > eta_lo) {
            return vec![eta_lo];
        }
        let n_min = s.n_search_candidates.max(2);
        if s.search_step_px > 0.0 && f_b > 1e-9 {
            let rho_lo = (-eta_hi).exp();
            let rho_hi = (-eta_lo).exp();
            let step = s.search_step_px / f_b;
            let n = (((rho_hi - rho_lo) / step).ceil() as usize + 1)
                .clamp(n_min, s.max_search_candidates.max(n_min));
            return (0..n)
                .map(|i| {
                    let rho = rho_lo + (rho_hi - rho_lo) * i as f64 / (n - 1) as f64;
                    (-rho.ln()).clamp(eta_lo, eta_hi)
                })
                .collect();
        }
        (0..n_min)
            .map(|i| eta_hi - (eta_hi - eta_lo) * i as f64 / (n_min - 1) as f64)
            .collect()
    }

    /// Evaluate the candidate costs: the best start for the refinement, the
    /// median cost for the contrast veto, and how clearly the best basin beats
    /// the runner-up (second-best local minimum at least two samples away).
    fn evaluate_candidates(
        &self,
        eta_init: f64,
        mut candidates: Vec<f64>,
        hypotheses: &[f64],
        eta_min: f64,
        eta_max: f64,
        mut cost: impl FnMut(f64) -> Option<f64>,
    ) -> CandidateSearch {
        if self.settings.photo_search_parent_hypotheses {
            for &h in hypotheses {
                let h = h.clamp(eta_min, eta_max);
                if candidates.iter().all(|c| (c - h).abs() > 0.02) {
                    candidates.push(h);
                }
            }
        }
        candidates.sort_by(|a, b| b.total_cmp(a));
        let valid: Vec<(f64, f64)> = candidates
            .iter()
            .filter_map(|&eta| cost(eta).map(|c| (eta, c)))
            .collect();
        if valid.len() < 3 {
            return CandidateSearch {
                best_eta: eta_init,
                median_cost: None,
                distinct_ratio: f64::INFINITY,
                best_interior: false,
                best_at_far: false,
            };
        }
        let (bi, &(best_eta, best_cost)) = valid
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .expect("non-empty");
        let mut sorted: Vec<f64> = valid.iter().map(|v| v.1).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        let mut second = f64::INFINITY;
        for i in 0..valid.len() {
            if (i as isize - bi as isize).abs() < 2 {
                continue;
            }
            let c = valid[i].1;
            let left = if i > 0 { valid[i - 1].1 } else { f64::INFINITY };
            let right = valid.get(i + 1).map_or(f64::INFINITY, |v| v.1);
            if c <= left && c <= right && c < second {
                second = c;
            }
        }
        let distinct_ratio = if best_cost > 1e-9 {
            second / best_cost
        } else if second <= 1e-9 {
            1.0
        } else {
            f64::INFINITY
        };
        CandidateSearch {
            best_eta,
            median_cost: Some(median),
            distinct_ratio,
            best_interior: bi > 0 && bi + 1 < valid.len(),
            best_at_far: bi == 0,
        }
    }

    /// Direction of the warp in the tangent chart, `d(u,v)/d eta` at `eta`.
    fn epipolar_direction(
        &self,
        tile: &TiledBearingTile,
        local_cu: f64,
        local_cv: f64,
        eta: f64,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64)> {
        let (_, _, x_ref, _) =
            warp_tiled_local_pixel_eta(tile, tile, local_cu, local_cv, eta, rel_pose)?;
        let j = tile.projection_jacobian(&x_ref)?;
        let d = j * (x_ref - rel_pose.t);
        (d[0].is_finite() && d[1].is_finite()).then_some((d[0], d[1]))
    }

    /// Mean squared derivative of the *current* image over the patch along the
    /// epipolar direction, in raw pixels. Independent of any depth prior: a
    /// patch without gradient along the line it would move on cannot be
    /// resolved by the image, whatever it is initialised from.
    #[allow(clippy::too_many_arguments)]
    fn current_patch_epipolar_gradient(
        &self,
        tile: &TiledBearingTile,
        affine: &PerPatchAffineMap,
        local_cu: f64,
        local_cv: f64,
        du_deta: f64,
        dv_deta: f64,
        curr_img: &Image<f32>,
    ) -> Option<f64> {
        let e_raw = affine.raw_du * du_deta + affine.raw_dv * dv_deta;
        let norm = e_raw.norm();
        if norm < 1e-12 {
            return None;
        }
        let e = e_raw / norm;
        let half = self.settings.patch_size / 2;
        let side = self.settings.patch_size;
        let base_u = tile.x0 as f64 + local_cu - half as f64 - affine.center_u;
        let base_v = tile.y0 as f64 + local_cv - half as f64 - affine.center_v;
        let data = curr_img.as_slice();
        let (w, h, st) = (curr_img.width(), curr_img.height(), curr_img.stride());
        let mut acc = 0.0;
        let mut n = 0usize;
        for ly in 0..side {
            for lx in 0..side {
                let du = base_u + lx as f64;
                let dv = base_v + ly as f64;
                let p = affine.raw_center + affine.raw_du * du + affine.raw_dv * dv;
                let (Some(a), Some(b)) = (
                    sample_bilinear_raw_nomask_slice(data, w, h, st, p[0] + e[0], p[1] + e[1]),
                    sample_bilinear_raw_nomask_slice(data, w, h, st, p[0] - e[0], p[1] - e[1]),
                ) else {
                    continue;
                };
                let g = 0.5 * (a - b) as f64;
                acc += g * g;
                n += 1;
            }
        }
        (n > 0).then(|| acc / n as f64)
    }

    /// Per-patch target-anchored solve: approximate the local raw projection by a
    /// patch-local affine chart, then solve the translated patch directly in raw
    /// images. This keeps PPB seamless for wide FoV without materializing a
    /// rectified tile image per patch.
    ///
    /// A patch with a landmark within one patch of it starts from that prior
    /// and keeps it in the refinement (`PhotoRefined` / `SeedOnly`). A patch
    /// without one is, with `photo_only`, solved from the image alone: the
    /// candidate search runs from the range at which the parallax falls below
    /// one pixel inwards to `search_max_disp_px`, and the result is `PhotoOnly`
    /// with a photometric-only variance. Either way a patch without gradient
    /// along its epipolar direction fails, quietly, as `Rejected`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_one_per_patch_bearing(
        &self,
        cu: f64,
        cv: f64,
        seeds: &[SparseDepthPrior],
        seed_grid: &SeedGrid,
        curr_pyramid: &[Image<f32>],
        _curr_valid_pyramid: Option<&[Image<f32>]>,
        refs: &[RefCandidate],
        intrinsics_by_level: &[ScaledIntrinsics],
        level: usize,
        scene_depth: f64,
    ) -> PatchEstimate {
        let s = &self.settings;
        let nearby = nearby_seed_weights(cu, cv, seeds, seed_grid, s);
        let seeded = !nearby.is_empty();
        if !seeded && !s.photo_only {
            return PatchEstimate::unknown();
        }

        // η bounds + seed init (range = z * ‖bearing‖), identical to solve_one_patch.
        let Some(center_bearing) =
            self.bearing_for_scaled_pixel(cu, cv, &intrinsics_by_level[level])
        else {
            return PatchEstimate::unknown();
        };
        let range_per_z = center_bearing.norm();
        let unit_center_bearing = center_bearing / range_per_z;

        // Reference choice per patch: the candidate with the most baseline
        // perpendicular to this bearing (within the parallax window at the
        // scene depth), and for verification a second one whose perpendicular
        // baseline differs by at least 35 %.
        let f_px = 0.5 * (self.intrinsics.fx + self.intrinsics.fy) * s.scale;
        let metres_per_px = scene_depth / f_px.max(1e-9);
        let perps: Vec<f64> = refs
            .iter()
            .map(|r| {
                let t = r.baseline_in_current();
                (t - unit_center_bearing * t.dot(&unit_center_bearing)).norm()
            })
            .collect();
        let eligible = |perp: f64| {
            let par = perp / metres_per_px;
            (s.min_parallax_px <= 0.0 || par >= s.min_parallax_px)
                && (s.max_parallax_px <= 0.0 || par <= s.max_parallax_px)
        };
        let mut best_a: Option<(usize, f64)> = None;
        for (k, &perp) in perps.iter().enumerate() {
            if eligible(perp) && best_a.is_none_or(|(_, p)| perp > p) {
                best_a = Some((k, perp));
            }
        }
        let ref_a = match best_a {
            Some((k, _)) => k,
            None if seeded => 0,
            None => {
                self.funnel.hit(Funnel::Seedless);
                self.funnel.hit(Funnel::NoParallax);
                return PatchEstimate::rejected(eta_max_for(s, range_per_z));
            }
        };
        let verify: Option<(&DepthKeyframe, &RelativePose)> = if !seeded && s.photo_verify_tol > 0.0
        {
            let perp_a = perps[ref_a];
            let mut best_b: Option<(usize, f64)> = None;
            for (k, &perp) in perps.iter().enumerate() {
                if k != ref_a
                    && eligible(perp)
                    && (perp - perp_a).abs() >= 0.35 * perp_a
                    && best_b.is_none_or(|(_, p)| perp > p)
                {
                    best_b = Some((k, perp));
                }
            }
            best_b.map(|(k, _)| (refs[k].kf, &refs[k].rel))
        } else {
            None
        };
        let ref_keyframe = refs[ref_a].kf;
        let rel_pose = &refs[ref_a].rel;
        let warp_uncertainty = &refs[ref_a].unc;
        let eta_min = (s.min_depth * range_per_z).ln();
        let eta_max = (s.max_depth * range_per_z).ln();
        let mut seed_eta_sum = 0.0;
        let mut seed_precision_sum = 0.0;
        for item in nearby.iter() {
            let wp = item.w_spatial * item.precision;
            seed_eta_sum += wp * seeds[item.idx].eta;
            seed_precision_sum += wp;
        }
        if seeded && !(seed_precision_sum > 0.0) {
            return PatchEstimate::unknown();
        }
        let seed_var = 1.0 / (seed_precision_sum * s.lambda_seed).max(1e-12);
        // Seedless patches start at the far end; the search walks inwards.
        let eta_init = if seeded {
            (seed_eta_sum / seed_precision_sum).clamp(eta_min, eta_max)
        } else {
            eta_max
        };
        let fallback = |eta: f64| {
            if seeded {
                PatchEstimate::seed_only(eta, seed_var, s)
            } else {
                PatchEstimate::rejected(eta)
            }
        };

        // Legacy structure gate on the reference keyframe (pinhole parity);
        // only the condition-number half is left to it, since the epipolar
        // gradient gate below owns `min_structure_eigen` on this path.
        if level == 0
            && s.max_structure_condition > 0.0
            && !self.patch_has_enough_structure(
                cu,
                cv,
                eta_init,
                ref_keyframe,
                &intrinsics_by_level[0],
                rel_pose,
            )
        {
            return PatchEstimate::rejected(eta_init);
        }

        // One per-patch tangent tile, centred on the patch and shared by the
        // current and keyframe rectifications. Current and reference MUST use the
        // same tangent basis — FastTranslation's translated square assumes the two
        // patches differ only by a translation; differing tile centres would
        // rotate them relative to each other (by the warp angle) and break it.
        // The tile carries no LUT on this path, so its side only bounds the
        // chart; a seeded patch sizes it to the warp at the prior, a seedless
        // one takes the full `tiled_tile_size` for its search.
        let specs = undistort_level_specs(self.width, self.height, s.scale, level + 1);
        let spec = &specs[level];
        let (lw, lh) = (spec.lw, spec.lh);
        let max_side = s.tiled_tile_size.max(2 * s.patch_size);
        let tile_side = if seeded {
            let disp = {
                let x_ref = rel_pose.r * (unit_center_bearing * eta_init.exp()) + rel_pose.t;
                if x_ref[2] > 1e-6 {
                    let (u_ref, v_ref) =
                        self.project_scaled(&x_ref, intrinsics_by_level[level].scale_from_original);
                    (u_ref - cu).hypot(v_ref - cv)
                } else {
                    0.0
                }
            };
            let margin = s.patch_size as f64;
            let need = (s.patch_size as f64 + 2.0 * (disp + margin)).ceil() as usize;
            need.clamp(2 * s.patch_size, max_side)
        } else {
            max_side
        };
        if tile_side >= lw || tile_side >= lh {
            return fallback(eta_init);
        }
        let x0 =
            (cu.round() as i64 - (tile_side as i64) / 2).clamp(0, (lw - tile_side) as i64) as usize;
        let y0 =
            (cv.round() as i64 - (tile_side as i64) / 2).clamp(0, (lh - tile_side) as i64) as usize;
        let tile_center_u = x0 as f64 + 0.5 * (tile_side.saturating_sub(1)) as f64;
        let tile_center_v = y0 as f64 + 0.5 * (tile_side.saturating_sub(1)) as f64;
        let tile_center_bearing = self
            .bearing_for_scaled_pixel(tile_center_u, tile_center_v, &intrinsics_by_level[level])
            .unwrap_or(unit_center_bearing);
        let tile = build_tiled_bearing_tile_geometry_from_center(
            self.camera.as_ref(),
            &self.intrinsics,
            spec,
            level,
            x0,
            y0,
            tile_side,
            tile_side,
            tile_center_bearing,
        );
        let Some(affine) = PerPatchAffineMap::from_tile(self.camera.as_ref(), spec, &tile) else {
            return fallback(eta_init);
        };
        let local_cu = cu - x0 as f64;
        let local_cv = cv - y0 as f64;

        // Epipolar geometry at a reference depth: the warp direction in the
        // chart and `f_b`, the disparity per unit inverse range.
        let eta_ref = if seeded {
            eta_init
        } else {
            0.5 * (eta_min + eta_max)
        };
        let Some((du_deta, dv_deta)) =
            self.epipolar_direction(&tile, local_cu, local_cv, eta_ref, rel_pose)
        else {
            return fallback(eta_init);
        };
        let f_b = du_deta.hypot(dv_deta) * eta_ref.exp();

        // Gradient gate along the epipolar direction, on the current image.
        if s.min_structure_eigen > 0.0 {
            let g2 = self.current_patch_epipolar_gradient(
                &tile,
                &affine,
                local_cu,
                local_cv,
                du_deta,
                dv_deta,
                &curr_pyramid[level],
            );
            if !g2.is_some_and(|g2| g2 >= s.min_structure_eigen) {
                if !seeded {
                    self.funnel.hit(Funnel::Seedless);
                    self.funnel.hit(Funnel::NoGradient);
                }
                return PatchEstimate::rejected(eta_init);
            }
        }

        if !seeded {
            self.funnel.hit(Funnel::Seedless);
            // Image-only: solve against the reference, then reproduce it against
            // a second baseline before believing it.
            let Some((eta, hess)) = self.photo_only_estimate(
                &tile,
                &affine,
                local_cu,
                local_cv,
                eta_min,
                eta_max,
                ref_keyframe,
                rel_pose,
                curr_pyramid,
                level,
                warp_uncertainty,
                true,
            ) else {
                return PatchEstimate::rejected(eta_init);
            };
            if s.photo_verify_tol > 0.0 {
                let Some((kf_b, rel_b)) = verify else {
                    self.funnel.hit(Funnel::NoVerifyRef);
                    return PatchEstimate::rejected(eta);
                };
                let Some((eta_b, _)) = self.photo_only_estimate(
                    &tile,
                    &affine,
                    local_cu,
                    local_cv,
                    eta_min,
                    eta_max,
                    kf_b,
                    rel_b,
                    curr_pyramid,
                    level,
                    warp_uncertainty,
                    false,
                ) else {
                    self.funnel.hit(Funnel::VerifyFailed);
                    return PatchEstimate::rejected(eta);
                };
                if (eta_b - eta).abs() > s.photo_verify_tol {
                    self.funnel.hit(Funnel::VerifyDisagree);
                    return PatchEstimate::rejected(eta);
                }
            }
            self.funnel.hit(Funnel::Reported);
            return PatchEstimate::photo_only(eta, hess, s);
        }

        // Seeded search window: around the prior, at least `search_half_range`
        // and two prior sigmas.
        let (eta_lo, eta_hi) = {
            let half = s.search_half_range.max(0.0).max(2.0 * seed_var.sqrt());
            (
                (eta_init - half).clamp(eta_min, eta_max),
                (eta_init + half).clamp(eta_min, eta_max),
            )
        };

        let hypotheses: Vec<f64> = nearby.iter().map(|it| seeds[it.idx].eta).collect();
        let cost_at = |eta_c: f64| {
            let (_, _, sum_abs_res, valid_n) = self
                .patch_residual_jacobian_per_patch_bearing_affine(
                    &tile,
                    &affine,
                    local_cu,
                    local_cv,
                    eta_c,
                    &curr_pyramid[level],
                    &ref_keyframe.ref_pyramid[level],
                    &ref_keyframe.grad_x_pyramid[level],
                    &ref_keyframe.grad_y_pyramid[level],
                    rel_pose,
                    warp_uncertainty,
                );
            (valid_n > 0).then(|| sum_abs_res / valid_n as f64)
        };
        let search = if s.photo_search {
            let candidates = self.candidate_etas(eta_lo, eta_hi, f_b);
            Some(self.evaluate_candidates(
                eta_init,
                candidates,
                &hypotheses,
                eta_min,
                eta_max,
                cost_at,
            ))
        } else {
            None
        };
        let (mut eta, median_cost) = match &search {
            Some(c) => (c.best_eta, c.median_cost),
            None => (eta_init, None),
        };

        let mut final_residual = 1e10;
        let mut final_curvature = 0.0;
        let mut observed = false;
        for _ in 0..s.n_gn_iters {
            let (grad_photo, hess_photo, sum_abs_res, valid_n) = self
                .patch_residual_jacobian_per_patch_bearing_affine(
                    &tile,
                    &affine,
                    local_cu,
                    local_cv,
                    eta,
                    &curr_pyramid[level],
                    &ref_keyframe.ref_pyramid[level],
                    &ref_keyframe.grad_x_pyramid[level],
                    &ref_keyframe.grad_y_pyramid[level],
                    rel_pose,
                    warp_uncertainty,
                );
            if valid_n == 0 {
                break;
            }
            observed = true;
            let mut grad_seed = 0.0;
            let mut hess_seed = 0.0;
            for item in nearby.iter() {
                let wp = s.lambda_seed * item.w_spatial * item.precision;
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
            if delta < s.gn_eta_convergence_tol {
                break;
            }
        }

        let min_curvature = s.min_photo_curvature * (s.patch_size * s.patch_size) as f64;
        let accepted = observed
            && final_curvature >= min_curvature
            && final_residual <= s.max_photo_residual
            && !self.fails_contrast(final_residual, median_cost);
        if accepted {
            PatchEstimate::photo_refined_with_information(
                eta,
                final_curvature,
                seed_precision_sum * s.lambda_seed,
                s,
            )
        } else {
            // Warp left the tile, too little curvature, or residual too high:
            // fall back to the (trustworthy, converged) seed so coverage stays
            // dense rather than dropping the patch.
            PatchEstimate::seed_only(eta_init, seed_var, s)
        }
    }

    /// Image-only depth of one patch against one reference: candidate search
    /// from the range where the parallax is one pixel ("zero inverse depth" in
    /// log range) in to `search_max_disp_px`, then Gauss-Newton without any
    /// prior. None when the cost curve has no bracketed minimum or the
    /// refinement fails the residual gate; `strict` adds the distinctness and
    /// contrast tests (the primary solve), which a verification solve skips
    /// since agreement with the primary is its test. Returns (eta, hess_photo).
    #[allow(clippy::too_many_arguments)]
    fn photo_only_estimate(
        &self,
        tile: &TiledBearingTile,
        affine: &PerPatchAffineMap,
        local_cu: f64,
        local_cv: f64,
        eta_min: f64,
        eta_max: f64,
        ref_keyframe: &DepthKeyframe,
        rel_pose: &RelativePose,
        curr_pyramid: &[Image<f32>],
        level: usize,
        warp_uncertainty: &WarpUncertainty,
        strict: bool,
    ) -> Option<(f64, f64)> {
        let s = &self.settings;
        let eta_ref = 0.5 * (eta_min + eta_max);
        let (du_deta, dv_deta) =
            self.epipolar_direction(tile, local_cu, local_cv, eta_ref, rel_pose)?;
        let f_b = du_deta.hypot(dv_deta) * eta_ref.exp();
        if !(f_b > 1e-9) {
            return None;
        }
        let eta_far = f_b.ln().clamp(eta_min, eta_max);
        let rho_near = (-eta_min).exp().min(s.search_max_disp_px.max(1.0) / f_b);
        let eta_near = (-rho_near.ln()).clamp(eta_min, eta_far);
        let leaf = |eta_c: f64| {
            self.patch_residual_jacobian_per_patch_bearing_affine(
                tile,
                affine,
                local_cu,
                local_cv,
                eta_c,
                &curr_pyramid[level],
                &ref_keyframe.ref_pyramid[level],
                &ref_keyframe.grad_x_pyramid[level],
                &ref_keyframe.grad_y_pyramid[level],
                rel_pose,
                warp_uncertainty,
            )
        };
        let candidates = self.candidate_etas(eta_near, eta_far, f_b);
        let search =
            self.evaluate_candidates(eta_far, candidates, &[], eta_min, eta_max, |eta_c| {
                let (_, _, sum_abs_res, valid_n) = leaf(eta_c);
                (valid_n > 0).then(|| sum_abs_res / valid_n as f64)
            });
        let Some(median_cost) = search.median_cost else {
            if strict {
                self.funnel.hit(Funnel::NoMinimum);
            }
            return None;
        };
        if !search.best_interior {
            if strict {
                self.funnel.hit(if search.best_at_far {
                    Funnel::MinAtFar
                } else {
                    Funnel::MinAtNear
                });
            }
            return None;
        }
        if strict && search.distinct_ratio < s.photo_distinct_min_ratio {
            self.funnel.hit(Funnel::NotDistinct);
            return None;
        }
        let mut eta = search.best_eta;
        let mut final_residual = 1e10;
        let mut final_curvature = 0.0;
        let mut observed = false;
        for _ in 0..s.n_gn_iters {
            let (grad_photo, hess_photo, sum_abs_res, valid_n) = leaf(eta);
            if valid_n == 0 {
                break;
            }
            observed = true;
            if hess_photo < 1e-12 {
                break;
            }
            let eta_next = (eta - grad_photo / hess_photo).clamp(eta_min, eta_max);
            let delta = (eta_next - eta).abs();
            eta = eta_next;
            final_residual = sum_abs_res / valid_n.max(1) as f64;
            final_curvature = hess_photo;
            if delta < s.gn_eta_convergence_tol {
                break;
            }
        }
        let min_curvature = s.min_photo_curvature * (s.patch_size * s.patch_size) as f64;
        let ok = observed
            && final_curvature >= min_curvature
            && final_residual <= s.max_photo_residual
            && !(strict && self.fails_contrast(final_residual, Some(median_cost)));
        if !ok && strict {
            self.funnel.hit(Funnel::RefineFailed);
        }
        ok.then_some((eta, final_curvature))
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
