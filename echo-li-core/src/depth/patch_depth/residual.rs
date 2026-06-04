use super::*;

impl PatchDepthMapper {
    pub(super) fn patch_cost(
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

    pub(super) fn patch_cost_fast_translation(
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

    pub(super) fn patch_residual_jacobian(
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

    pub(super) fn patch_residual_jacobian_fast_translation(
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

    pub(super) fn patch_residual_jacobian_fast_translation_level(
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

    pub(super) fn patch_residual_jacobian_fast_translation_tiled(
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
    pub(super) fn patch_residual_jacobian_fast_translation_tiled_level(
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

    pub(super) fn patch_residual_jacobian_per_patch_bearing_affine(
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

    pub(super) fn translated_patch_footprint(
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

    pub(super) fn patch_residual_jacobian_level(
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
}
