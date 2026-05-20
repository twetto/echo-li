use super::{PatchAccum, TranslatedPatchFootprint};
use rudolf_v::image::Image;
use std::arch::aarch64::{
    float32x4_t, vabsq_f32, vaddq_f32, vaddvq_f32, vbslq_f32, vcleq_f32, vdivq_f32, vdupq_n_f32,
    vfmaq_f32, vld1q_f32, vmulq_f32, vsubq_f32,
};

pub(super) fn fast_translation_accum_neon_if_available(
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

    // SAFETY: NEON is part of the aarch64 base ISA, and the translated footprint
    // and current image bounds are validated above.
    unsafe {
        fast_translation_accum_neon(
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

#[target_feature(enable = "neon")]
unsafe fn fast_translation_accum_neon(
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
    let weights = patch.ref_fp.weights;
    let vw00 = vdupq_n_f32(weights[0]);
    let vw10 = vdupq_n_f32(weights[1]);
    let vw01 = vdupq_n_f32(weights[2]);
    let vw11 = vdupq_n_f32(weights[3]);
    let v_du = vdupq_n_f32(du_drho);
    let v_dv = vdupq_n_f32(dv_drho);
    let v_inv_sigma = vdupq_n_f32(inv_sigma_photo_sq);
    let v_delta = vdupq_n_f32(huber_delta);
    let v_delta_inv_sigma = vdupq_n_f32(huber_delta * inv_sigma_photo_sq);

    let mut sum_grad = vdupq_n_f32(0.0);
    let mut sum_hess = vdupq_n_f32(0.0);
    let mut sum_abs = vdupq_n_f32(0.0);
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

            for lx in (0..patch.side).step_by(4) {
                let cx = patch.curr_x0 as usize + lx;
                let ix = patch.ref_fp.x + lx;
                if !mask_chunk4_valid(curr_mask_row, cx) || !mask_chunk4_valid(ref_mask_row, ix) {
                    return None;
                }

                let curr = vld1q_f32(curr_row.add(cx));
                let i_ref = bilerp4_ptr(ref_row0, ref_row1, ix, vw00, vw10, vw01, vw11);
                let gx = bilerp4_ptr(gx_row0, gx_row1, ix, vw00, vw10, vw01, vw11);
                let gy = bilerp4_ptr(gy_row0, gy_row1, ix, vw00, vw10, vw01, vw11);

                let jac = vfmaq_f32(vmulq_f32(gx, v_du), gy, v_dv);
                let residual = vsubq_f32(i_ref, curr);
                let abs_res = vabsq_f32(residual);
                let huber_mask = vcleq_f32(abs_res, v_delta);
                // bsl is `mask ? b : a`: when |r| <= delta take v_inv_sigma, else delta/|r| * inv_sigma.
                let robust = vbslq_f32(
                    huber_mask,
                    v_inv_sigma,
                    vdivq_f32(v_delta_inv_sigma, abs_res),
                );

                sum_grad = vfmaq_f32(sum_grad, robust, vmulq_f32(jac, residual));
                sum_hess = vfmaq_f32(sum_hess, robust, vmulq_f32(jac, jac));
                sum_abs = vaddq_f32(sum_abs, abs_res);
                n_valid += 4;
            }
        }

        Some(PatchAccum {
            grad: vaddvq_f32(sum_grad) as f64,
            hess: vaddvq_f32(sum_hess) as f64,
            sum_abs_res: vaddvq_f32(sum_abs) as f64,
            n_valid,
        })
    }
}

#[target_feature(enable = "neon")]
unsafe fn bilerp4_ptr(
    r0: *const f32,
    r1: *const f32,
    x: usize,
    vw00: float32x4_t,
    vw10: float32x4_t,
    vw01: float32x4_t,
    vw11: float32x4_t,
) -> float32x4_t {
    unsafe {
        let p00 = vld1q_f32(r0.add(x));
        let p10 = vld1q_f32(r0.add(x + 1));
        let p01 = vld1q_f32(r1.add(x));
        let p11 = vld1q_f32(r1.add(x + 1));
        let acc = vmulq_f32(vw00, p00);
        let acc = vfmaq_f32(acc, vw10, p10);
        let acc = vfmaq_f32(acc, vw01, p01);
        vfmaq_f32(acc, vw11, p11)
    }
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
