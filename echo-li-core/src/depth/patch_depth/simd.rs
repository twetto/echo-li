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
    use std::arch::x86_64::{_mm256_castps128_ps256, _mm256_insertf128_ps, _mm_loadu_ps};
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
