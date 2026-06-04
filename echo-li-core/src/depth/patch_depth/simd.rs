use super::{PatchAccum, TranslatedPatchFootprint};
use rudolf_v::image::Image;

pub(super) fn fast_translation_accum_avx2_if_available(
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

/// Geometry for the per-patch-bearing affine photometric leaf. Each output pixel
/// `(lx, ly)` maps to raw image coordinates by the affine chart
/// `raw = raw_center + raw_du·(base_u + lx) + raw_dv·(base_v + ly)`, evaluated
/// separately for the current and reference images (they share the affine basis
/// but have different `base`). The reference Jacobian collapses to
/// `jac = raw_gx·cgx + raw_gy·cgy` (see `solve_one_per_patch_bearing`).
#[derive(Debug, Clone, Copy)]
pub(super) struct PerPatchAffineGeom {
    pub raw_center: [f32; 2],
    pub raw_du: [f32; 2],
    pub raw_dv: [f32; 2],
    pub curr_base_u: f32,
    pub curr_base_v: f32,
    pub ref_base_u: f32,
    pub ref_base_v: f32,
    pub cgx: f32,
    pub cgy: f32,
}

/// SIMD path for the per-patch affine leaf. Only valid for constant photometric
/// weight (`sigma_warp_sq == 0`, mirroring the FastTranslation SIMD precondition)
/// and `side` a multiple of 8. Returns `None` to fall back to the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(super) fn per_patch_affine_accum_avx2_if_available(
    curr_img: &Image<f32>,
    ref_img: &Image<f32>,
    ref_grad_x: &Image<f32>,
    ref_grad_y: &Image<f32>,
    side: usize,
    geom: PerPatchAffineGeom,
    inv_sigma_photo_sq: f32,
    huber_delta: f32,
) -> Option<PatchAccum> {
    if side % 8 != 0 || side == 0 {
        return None;
    }
    let width = curr_img.width();
    let height = curr_img.height();
    if width < 2 || height < 2 {
        return None;
    }
    let stride = curr_img.stride();
    if ref_img.width() != width
        || ref_img.height() != height
        || ref_img.stride() != stride
        || ref_grad_x.stride() != stride
        || ref_grad_y.stride() != stride
    {
        return None;
    }
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return None;
    }
    // SAFETY: AVX2/FMA verified above; the gather indices are clamped into
    // `[0, (height-2)*stride + (width-2)]` so all four bilinear corners stay in
    // bounds for any (even out-of-frame, later masked) lane.
    unsafe {
        Some(per_patch_affine_accum_avx2(
            curr_img.as_slice(),
            ref_img.as_slice(),
            ref_grad_x.as_slice(),
            ref_grad_y.as_slice(),
            width,
            height,
            stride,
            side,
            geom,
            inv_sigma_photo_sq,
            huber_delta,
        ))
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn per_patch_affine_accum_avx2(
    curr: &[f32],
    ref_i: &[f32],
    ref_gx: &[f32],
    ref_gy: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    side: usize,
    geom: PerPatchAffineGeom,
    inv_sigma_photo_sq: f32,
    huber_delta: f32,
) -> PatchAccum {
    use std::arch::x86_64::{
        _CMP_LE_OQ, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_blendv_ps,
        _mm256_cmp_ps, _mm256_div_ps, _mm256_fmadd_ps, _mm256_movemask_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_setr_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };

    unsafe {
        let v_lane = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let v_zero = _mm256_setzero_ps();
        let v_one = _mm256_set1_ps(1.0);
        let v_wm1 = _mm256_set1_ps((width - 1) as f32);
        let v_hm1 = _mm256_set1_ps((height - 1) as f32);
        let v_xmax = _mm256_set1_ps((width - 2) as f32);
        let v_ymax = _mm256_set1_ps((height - 2) as f32);
        let v_inv_sigma = _mm256_set1_ps(inv_sigma_photo_sq);
        let v_delta = _mm256_set1_ps(huber_delta);
        let v_delta_inv_sigma = _mm256_set1_ps(huber_delta * inv_sigma_photo_sq);
        let v_abs_mask = _mm256_set1_ps(-0.0);
        let v_cgx = _mm256_set1_ps(geom.cgx);
        let v_cgy = _mm256_set1_ps(geom.cgy);
        let v_rdu_u = _mm256_set1_ps(geom.raw_du[0]);
        let v_rdu_v = _mm256_set1_ps(geom.raw_du[1]);

        let curr_ptr = curr.as_ptr();
        let ref_ptr = ref_i.as_ptr();
        let gx_ptr = ref_gx.as_ptr();
        let gy_ptr = ref_gy.as_ptr();

        let mut sum_grad = _mm256_setzero_ps();
        let mut sum_hess = _mm256_setzero_ps();
        let mut sum_abs = _mm256_setzero_ps();
        let mut n_valid = 0usize;

        for ly in 0..side {
            let lyf = ly as f32;
            // Row-0 (lane 0) raw coordinates for current and reference, then the
            // per-lane ramp adds `raw_du · (lane + lx_chunk)`.
            let curr_u0 = geom.raw_center[0]
                + geom.raw_du[0] * geom.curr_base_u
                + geom.raw_dv[0] * (geom.curr_base_v + lyf);
            let curr_v0 = geom.raw_center[1]
                + geom.raw_du[1] * geom.curr_base_u
                + geom.raw_dv[1] * (geom.curr_base_v + lyf);
            let ref_u0 = geom.raw_center[0]
                + geom.raw_du[0] * geom.ref_base_u
                + geom.raw_dv[0] * (geom.ref_base_v + lyf);
            let ref_v0 = geom.raw_center[1]
                + geom.raw_du[1] * geom.ref_base_u
                + geom.raw_dv[1] * (geom.ref_base_v + lyf);

            for lx in (0..side).step_by(8) {
                let v_lx = _mm256_add_ps(v_lane, _mm256_set1_ps(lx as f32));
                let curr_u = _mm256_fmadd_ps(v_rdu_u, v_lx, _mm256_set1_ps(curr_u0));
                let curr_v = _mm256_fmadd_ps(v_rdu_v, v_lx, _mm256_set1_ps(curr_v0));
                let ref_u = _mm256_fmadd_ps(v_rdu_u, v_lx, _mm256_set1_ps(ref_u0));
                let ref_v = _mm256_fmadd_ps(v_rdu_v, v_lx, _mm256_set1_ps(ref_v0));

                let curr_valid = bounds_mask8(curr_u, curr_v, v_zero, v_wm1, v_hm1);
                let ref_valid = bounds_mask8(ref_u, ref_v, v_zero, v_wm1, v_hm1);
                let valid = _mm256_and_ps(curr_valid, ref_valid);
                if _mm256_movemask_ps(valid) == 0 {
                    continue;
                }

                let i_curr = bilinear_gather8(
                    curr_ptr, curr_u, curr_v, stride, v_zero, v_one, v_xmax, v_ymax,
                );
                let i_ref =
                    bilinear_gather8(ref_ptr, ref_u, ref_v, stride, v_zero, v_one, v_xmax, v_ymax);
                let raw_gx =
                    bilinear_gather8(gx_ptr, ref_u, ref_v, stride, v_zero, v_one, v_xmax, v_ymax);
                let raw_gy =
                    bilinear_gather8(gy_ptr, ref_u, ref_v, stride, v_zero, v_one, v_xmax, v_ymax);

                let jac = _mm256_fmadd_ps(raw_gy, v_cgy, _mm256_mul_ps(raw_gx, v_cgx));
                let residual = _mm256_sub_ps(i_ref, i_curr);
                let abs_res = _mm256_andnot_ps(v_abs_mask, residual);
                let huber_mask = _mm256_cmp_ps(abs_res, v_delta, _CMP_LE_OQ);
                let robust = _mm256_blendv_ps(
                    _mm256_div_ps(v_delta_inv_sigma, abs_res),
                    v_inv_sigma,
                    huber_mask,
                );
                // Zero out the contribution of out-of-bounds lanes.
                let valid_f = _mm256_and_ps(valid, v_one);
                let robust = _mm256_mul_ps(robust, valid_f);
                let abs_res = _mm256_mul_ps(abs_res, valid_f);

                sum_grad = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, residual), sum_grad);
                sum_hess = _mm256_fmadd_ps(robust, _mm256_mul_ps(jac, jac), sum_hess);
                sum_abs = _mm256_add_ps(sum_abs, abs_res);
                n_valid += (_mm256_movemask_ps(valid) as u32).count_ones() as usize;
            }
        }

        PatchAccum {
            grad: hsum256(sum_grad) as f64,
            hess: hsum256(sum_hess) as f64,
            sum_abs_res: hsum256(sum_abs) as f64,
            n_valid,
        }
    }
}

/// `(u >= 0) & (v >= 0) & (u < width-1) & (v < height-1)` as a float lane mask.
#[target_feature(enable = "avx2,fma")]
unsafe fn bounds_mask8(
    u: std::arch::x86_64::__m256,
    v: std::arch::x86_64::__m256,
    v_zero: std::arch::x86_64::__m256,
    v_wm1: std::arch::x86_64::__m256,
    v_hm1: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_CMP_GE_OQ, _CMP_LT_OQ, _mm256_and_ps, _mm256_cmp_ps};
    unsafe {
        let mu = _mm256_and_ps(
            _mm256_cmp_ps(u, v_zero, _CMP_GE_OQ),
            _mm256_cmp_ps(u, v_wm1, _CMP_LT_OQ),
        );
        let mv = _mm256_and_ps(
            _mm256_cmp_ps(v, v_zero, _CMP_GE_OQ),
            _mm256_cmp_ps(v, v_hm1, _CMP_LT_OQ),
        );
        _mm256_and_ps(mu, mv)
    }
}

/// Bilinear sample of 8 lanes via gather. Indices are clamped to keep all four
/// corners in bounds; out-of-frame lanes must be masked out by the caller.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn bilinear_gather8(
    base: *const f32,
    u: std::arch::x86_64::__m256,
    v: std::arch::x86_64::__m256,
    stride: usize,
    v_zero: std::arch::x86_64::__m256,
    v_one: std::arch::x86_64::__m256,
    v_xmax: std::arch::x86_64::__m256,
    v_ymax: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_cvttps_epi32, _mm256_floor_ps, _mm256_fmadd_ps,
        _mm256_i32gather_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_mullo_epi32,
        _mm256_set1_epi32, _mm256_sub_ps,
    };
    unsafe {
        let xf = _mm256_floor_ps(u);
        let yf = _mm256_floor_ps(v);
        let dx = _mm256_sub_ps(u, xf);
        let dy = _mm256_sub_ps(v, yf);
        // Clamp the integer corner into range; for valid lanes this is a no-op.
        let xc = _mm256_min_ps(_mm256_max_ps(xf, v_zero), v_xmax);
        let yc = _mm256_min_ps(_mm256_max_ps(yf, v_zero), v_ymax);
        let xi = _mm256_cvttps_epi32(xc);
        let yi = _mm256_cvttps_epi32(yc);
        let v_stride = _mm256_set1_epi32(stride as i32);
        let v_one_i = _mm256_set1_epi32(1);
        let idx0 = _mm256_add_epi32(_mm256_mullo_epi32(yi, v_stride), xi);
        let idx1 = _mm256_add_epi32(idx0, v_stride);
        let p00 = _mm256_i32gather_ps::<4>(base, idx0);
        let p10 = _mm256_i32gather_ps::<4>(base, _mm256_add_epi32(idx0, v_one_i));
        let p01 = _mm256_i32gather_ps::<4>(base, idx1);
        let p11 = _mm256_i32gather_ps::<4>(base, _mm256_add_epi32(idx1, v_one_i));

        let one_dx = _mm256_sub_ps(v_one, dx);
        let one_dy = _mm256_sub_ps(v_one, dy);
        let w00 = _mm256_mul_ps(one_dx, one_dy);
        let w10 = _mm256_mul_ps(dx, one_dy);
        let w01 = _mm256_mul_ps(one_dx, dy);
        let w11 = _mm256_mul_ps(dx, dy);
        let acc = _mm256_mul_ps(w00, p00);
        let acc = _mm256_fmadd_ps(w10, p10, acc);
        let acc = _mm256_fmadd_ps(w01, p01, acc);
        _mm256_fmadd_ps(w11, p11, acc)
    }
}

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
        _CMP_LE_OQ, _mm256_add_ps, _mm256_andnot_ps, _mm256_blendv_ps, _mm256_cmp_ps,
        _mm256_div_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_sub_ps,
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
        _CMP_LE_OQ, _mm256_add_ps, _mm256_andnot_ps, _mm256_blendv_ps, _mm256_cmp_ps,
        _mm256_div_ps, _mm256_fmadd_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_sub_ps,
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

#[target_feature(enable = "avx2")]
unsafe fn load4x2_ptr(
    row0: *const f32,
    row1: *const f32,
    x0: usize,
    x1: usize,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm_loadu_ps, _mm256_castps128_ps256, _mm256_insertf128_ps};
    unsafe {
        let lo = _mm_loadu_ps(row0.add(x0));
        let hi = _mm_loadu_ps(row1.add(x1));
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }
}

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

#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    let mut tmp = [0.0f32; 8];
    unsafe {
        std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), v);
    }
    tmp.iter().sum()
}
