use super::image_ops::sample_bilinear_valid;
use super::PatchDepthCameraMode;
use super::{
    PatchDepthMapper, PatchDepthOutput, PatchDepthSettings, PatchEstimate, PatchStatus,
    RelativePose, ScaledIntrinsics,
};
use crate::core_types::DepthMap;
use rudolf_v::image::Image;

#[cfg(feature = "parallel")]
pub(super) fn patch_centers(
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

pub(super) struct PatchGrid {
    patches: Vec<PatchEstimate>,
    pub(super) n_u: usize,
    pub(super) n_v: usize,
    half: usize,
    stride: usize,
}

impl PatchGrid {
    pub(super) fn new(width: usize, height: usize, settings: &PatchDepthSettings) -> Self {
        let half = settings.patch_size / 2;
        let stride = settings.patch_stride;
        let n_u = if width > 2 * half {
            (width - 2 * half + stride - 1) / stride
        } else {
            0
        };
        let n_v = if height > 2 * half {
            (height - 2 * half + stride - 1) / stride
        } else {
            0
        };
        let unknown = PatchEstimate::unknown();
        Self {
            patches: vec![unknown; n_u * n_v],
            n_u,
            n_v,
            half,
            stride,
        }
    }

    pub(super) fn set(&mut self, cu: usize, cv: usize, estimate: PatchEstimate) {
        let iu = (cu - self.half) / self.stride;
        let iv = (cv - self.half) / self.stride;
        if iu < self.n_u && iv < self.n_v {
            self.patches[iv * self.n_u + iu] = estimate;
        }
    }

    #[inline(always)]
    fn get(&self, iu: usize, iv: usize) -> &PatchEstimate {
        &self.patches[iv * self.n_u + iu]
    }

    #[inline(always)]
    fn overlap_u(&self, px: usize) -> (usize, usize) {
        let half = self.half;
        let stride = self.stride;
        let start = if px + 1 >= 2 * half {
            (px + 1 - 2 * half) / stride + 1
        } else {
            0
        };
        let end = (px / stride).min(self.n_u.saturating_sub(1));
        (start, end)
    }

    #[inline(always)]
    fn overlap_v(&self, py: usize) -> (usize, usize) {
        let half = self.half;
        let stride = self.stride;
        let start = if py + 1 >= 2 * half {
            (py + 1 - 2 * half) / stride + 1
        } else {
            0
        };
        let end = (py / stride).min(self.n_v.saturating_sub(1));
        (start, end)
    }
}

#[cfg(not(feature = "parallel"))]
pub(super) fn densify_pixels(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    if mapper.camera_mode == PatchDepthCameraMode::UndistortedPinhole {
        return densify_pixels_pinhole(
            grid, width, height, curr_img, curr_valid, ref_img, ref_valid, mapper, intr, rel_pose,
        );
    }
    densify_pixels_generic(
        grid, width, height, curr_img, curr_valid, ref_img, ref_valid, mapper, intr, rel_pose,
    )
}

#[cfg(not(feature = "parallel"))]
fn densify_pixels_generic(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    let n = width * height;
    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    let mut status = vec![PatchStatus::Unknown; n];

    for py in 0..height {
        let (iv_start, iv_end) = grid.overlap_v(py);
        if iv_start > iv_end {
            continue;
        }
        for px in 0..width {
            let (iu_start, iu_end) = grid.overlap_u(px);
            if iu_start > iu_end {
                continue;
            }
            let mut eta_acc = 0.0_f32;
            let mut w_acc = 0.0_f32;
            let mut best_status = PatchStatus::Unknown;
            for iv in iv_start..=iv_end {
                for iu in iu_start..=iu_end {
                    let patch = grid.get(iu, iv);
                    let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                        continue;
                    };
                    let photo_w = if patch.status == PatchStatus::PhotoRefined {
                        let cu = iu * grid.stride + grid.half;
                        let cv = iv * grid.stride + grid.half;
                        let range_per_z = mapper
                            .bearing_for_scaled_pixel(cu as f64, cv as f64, intr)
                            .map(|b| b.norm())
                            .unwrap_or(1.0);
                        let rho_center = range_per_z * (-patch.eta).exp();
                        compute_photo_weight(
                            px, py, rho_center, curr_img, curr_valid, ref_img, ref_valid, mapper,
                            intr, rel_pose,
                        )
                    } else {
                        1.0_f32
                    };
                    let w = inv_var_w * photo_w;
                    eta_acc += w * patch.eta as f32;
                    w_acc += w;
                    if (patch.status as u8) > (best_status as u8) {
                        best_status = patch.status;
                    }
                }
            }
            if w_acc > 0.0 {
                let idx = py * width + px;
                let unit_b_z = mapper.unit_b_z_for_pixel(px as f64, py as f64, intr) as f32;
                depth[idx] = (eta_acc / w_acc).exp() * unit_b_z;
                variance[idx] = 1.0 / w_acc;
                status[idx] = best_status;
            }
        }
    }
    PatchDepthOutput {
        depth: DepthMap::from_vec(width, height, depth).expect("depth size"),
        variance: DepthMap::from_vec(width, height, variance).expect("variance size"),
        status: DepthMap::from_vec(width, height, status).expect("status size"),
    }
}

struct RowWarpCoeffs {
    // x_ref[i] = a[i] * px + b[i], for i in 0..3
    a: [f32; 3],
    b: [f32; 3],
    // projection constants
    fx_s: f32,
    fy_s: f32,
    cx_s: f32,
    cy_s: f32,
}

impl RowWarpCoeffs {
    fn new(
        py: usize,
        z: f64,
        mapper: &PatchDepthMapper,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> Self {
        let s = intr.scale_from_original;
        let fx = mapper.intrinsics.fx;
        let fy = mapper.intrinsics.fy;
        let cx = mapper.intrinsics.cx;
        let cy = mapper.intrinsics.cy;

        // bearing = [(px/s - cx)/fx, (py/s - cy)/fy, 1.0]
        // x_curr = bearing * z  (since rho = 1/z, bearing/rho = bearing*z)
        // x_ref = R * x_curr + t
        let kx = 1.0 / (s * fx);
        let bx = -cx / fx;
        let by_val = (py as f64 / s - cy) / fy;

        let r = &rel_pose.r;
        let t = &rel_pose.t;
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        for j in 0..3 {
            a[j] = (r[(j, 0)] * kx * z) as f32;
            b[j] = (r[(j, 0)] * bx * z + r[(j, 1)] * by_val * z + r[(j, 2)] * z + t[j]) as f32;
        }

        Self {
            a,
            b,
            fx_s: (fx * s) as f32,
            fy_s: (fy * s) as f32,
            cx_s: (cx * s) as f32,
            cy_s: (cy * s) as f32,
        }
    }

    #[inline(always)]
    fn warp(&self, px: f32) -> (f32, f32, f32) {
        let xr0 = self.a[0] * px + self.b[0];
        let xr1 = self.a[1] * px + self.b[1];
        let xr2 = self.a[2] * px + self.b[2];
        let z_inv = 1.0 / xr2;
        let u_ref = self.fx_s * xr0 * z_inv + self.cx_s;
        let v_ref = self.fy_s * xr1 * z_inv + self.cy_s;
        (u_ref, v_ref, xr2)
    }
}

#[cfg(not(feature = "parallel"))]
fn densify_pixels_pinhole(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    let n = width * height;
    let mut eta_buf = vec![0.0_f32; n];
    let mut w_buf = vec![0.0_f32; n];
    let mut status_buf = vec![PatchStatus::Unknown; n];

    let ref_w = ref_img.width();
    let ref_h = ref_img.height();
    let s = intr.scale_from_original;
    let fx = mapper.intrinsics.fx as f32;
    let fy = mapper.intrinsics.fy as f32;
    let cx = mapper.intrinsics.cx as f32;
    let cy = mapper.intrinsics.cy as f32;

    for iv in 0..grid.n_v {
        for iu in 0..grid.n_u {
            let patch = grid.get(iu, iv);
            let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                continue;
            };

            let patch_eta_f32 = patch.eta as f32;
            let weighted_eta = inv_var_w * patch_eta_f32;

            let patch_size = 2 * grid.half;
            let py_start = iv * grid.stride;
            let py_end = (py_start + patch_size).min(height);
            let px_start = iu * grid.stride;
            let px_end = (px_start + patch_size).min(width);

            // Patch centre z-depth for RowWarpCoeffs (photo weighting warp).
            let cu = (iu * grid.stride + grid.half) as f64;
            let cv = (iv * grid.stride + grid.half) as f64;
            let bx_c = (cu / s - mapper.intrinsics.cx) / mapper.intrinsics.fx;
            let by_c = (cv / s - mapper.intrinsics.cy) / mapper.intrinsics.fy;
            let range_per_z_center = (bx_c * bx_c + by_c * by_c + 1.0).sqrt();
            let z_center = patch.eta.exp() / range_per_z_center;

            if patch.status == PatchStatus::PhotoRefined {
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;

                for py in py_start..py_end {
                    let coeffs = RowWarpCoeffs::new(py, z_center, mapper, intr, rel_pose);
                    let curr_row = &curr_img.as_slice()[py * curr_img.width()..];
                    let valid_row = curr_valid.map(|v| &v.as_slice()[py * v.width()..]);
                    let row_off = py * width;

                    let mut px = px_start;

                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        while px + 8 <= px_end {
                            unsafe {
                                densify_row_avx2(
                                    &coeffs,
                                    px,
                                    curr_row,
                                    valid_row,
                                    ref_img,
                                    ref_valid,
                                    ref_w,
                                    ref_h,
                                    inv_var_w,
                                    patch_eta_f32,
                                    &mut eta_buf[row_off..],
                                    &mut w_buf[row_off..],
                                    &mut status_buf[row_off..],
                                    PatchStatus::PhotoRefined,
                                );
                            }
                            px += 8;
                        }
                    }

                    #[cfg(target_arch = "aarch64")]
                    while px + 4 <= px_end {
                        unsafe {
                            densify_row_neon(
                                &coeffs,
                                px,
                                curr_row,
                                valid_row,
                                ref_img,
                                ref_valid,
                                ref_w,
                                ref_h,
                                inv_var_w,
                                patch_eta_f32,
                                &mut eta_buf[row_off..],
                                &mut w_buf[row_off..],
                                &mut status_buf[row_off..],
                                PatchStatus::PhotoRefined,
                            );
                        }
                        px += 4;
                    }

                    // Scalar tail
                    for px in px..px_end {
                        let (u_ref, v_ref, z_ref) = coeffs.warp(px as f32);
                        let photo_w = if z_ref > 1e-6
                            && u_ref >= 0.0
                            && v_ref >= 0.0
                            && u_ref < (ref_w - 1) as f32
                            && v_ref < (ref_h - 1) as f32
                        {
                            let i_curr = curr_row[px];
                            if let Some(vr) = valid_row {
                                if vr[px] < 0.5 {
                                    1.0_f32
                                } else {
                                    photo_weight_inline(
                                        i_curr,
                                        ref_img,
                                        ref_valid,
                                        u_ref as f64,
                                        v_ref as f64,
                                    )
                                }
                            } else {
                                photo_weight_inline(
                                    i_curr,
                                    ref_img,
                                    ref_valid,
                                    u_ref as f64,
                                    v_ref as f64,
                                )
                            }
                        } else {
                            1.0_f32
                        };
                        let w = inv_var_w * photo_w;
                        let idx = row_off + px;
                        eta_buf[idx] += w * patch_eta_f32;
                        w_buf[idx] += w;
                        if (patch.status as u8) > (status_buf[idx] as u8) {
                            status_buf[idx] = patch.status;
                        }
                    }
                }
            } else {
                // SeedOnly: photo_w = 1.0, just accumulate
                for py in py_start..py_end {
                    let row_off = py * width;
                    for px in px_start..px_end {
                        let idx = row_off + px;
                        eta_buf[idx] += weighted_eta;
                        w_buf[idx] += inv_var_w;
                        if (patch.status as u8) > (status_buf[idx] as u8) {
                            status_buf[idx] = patch.status;
                        }
                    }
                }
            }
        }
    }

    // Convert accumulated eta mean → z-depth via per-pixel bearing direction.
    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    for py in 0..height {
        for px in 0..width {
            let i = py * width + px;
            if w_buf[i] > 0.0 {
                let bx = (px as f32 / s as f32 - cx) / fx;
                let by = (py as f32 / s as f32 - cy) / fy;
                let unit_b_z = 1.0 / (bx * bx + by * by + 1.0).sqrt();
                depth[i] = (eta_buf[i] / w_buf[i]).exp() * unit_b_z;
                variance[i] = 1.0 / w_buf[i];
            }
        }
    }

    PatchDepthOutput {
        depth: DepthMap::from_vec(width, height, depth).expect("depth size"),
        variance: DepthMap::from_vec(width, height, variance).expect("variance size"),
        status: DepthMap::from_vec(width, height, status_buf).expect("status size"),
    }
}

#[inline(always)]
fn photo_weight_inline(
    i_curr: f32,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    u_ref: f64,
    v_ref: f64,
) -> f32 {
    let Some(i_ref) = sample_bilinear_valid(ref_img, ref_valid, u_ref, v_ref) else {
        return 1.0;
    };
    let residual = (i_ref - i_curr).abs();
    1.0 / residual.max(1.0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn densify_row_avx2(
    coeffs: &RowWarpCoeffs,
    px_start: usize,
    curr_row: &[f32],
    valid_row: Option<&[f32]>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    ref_w: usize,
    ref_h: usize,
    inv_var_w: f32,
    patch_eta: f32,
    eta_buf: &mut [f32],
    w_buf: &mut [f32],
    status_buf: &mut [PatchStatus],
    patch_status: PatchStatus,
) {
    unsafe {
        use std::arch::x86_64::*;

        let ref_slice = ref_img.as_slice();
        let ref_valid_slice = ref_valid.map(|v| v.as_slice());
        let ref_w_i32 = ref_w as i32;

        let px_base = _mm256_set1_ps(px_start as f32);
        let px_offsets = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let px_vec = _mm256_add_ps(px_base, px_offsets);

        let a0 = _mm256_set1_ps(coeffs.a[0]);
        let b0 = _mm256_set1_ps(coeffs.b[0]);
        let a1 = _mm256_set1_ps(coeffs.a[1]);
        let b1 = _mm256_set1_ps(coeffs.b[1]);
        let a2 = _mm256_set1_ps(coeffs.a[2]);
        let b2 = _mm256_set1_ps(coeffs.b[2]);

        let xr0 = _mm256_fmadd_ps(a0, px_vec, b0);
        let xr1 = _mm256_fmadd_ps(a1, px_vec, b1);
        let xr2 = _mm256_fmadd_ps(a2, px_vec, b2);

        let rcp = _mm256_rcp_ps(xr2);
        let two = _mm256_set1_ps(2.0);
        let z_inv = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(xr2, rcp, two));

        let fx_s = _mm256_set1_ps(coeffs.fx_s);
        let fy_s = _mm256_set1_ps(coeffs.fy_s);
        let cx_s = _mm256_set1_ps(coeffs.cx_s);
        let cy_s = _mm256_set1_ps(coeffs.cy_s);
        let u_ref = _mm256_fmadd_ps(fx_s, _mm256_mul_ps(xr0, z_inv), cx_s);
        let v_ref = _mm256_fmadd_ps(fy_s, _mm256_mul_ps(xr1, z_inv), cy_s);

        let eps = _mm256_set1_ps(1e-6);
        let zero = _mm256_setzero_ps();
        let max_u = _mm256_set1_ps((ref_w - 1) as f32);
        let max_v = _mm256_set1_ps((ref_h - 1) as f32);

        let valid_mask = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GT_OQ>(xr2, eps),
                _mm256_cmp_ps::<_CMP_GE_OQ>(u_ref, zero),
            ),
            _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_LT_OQ>(u_ref, max_u),
                _mm256_and_ps(
                    _mm256_cmp_ps::<_CMP_GE_OQ>(v_ref, zero),
                    _mm256_cmp_ps::<_CMP_LT_OQ>(v_ref, max_v),
                ),
            ),
        );

        let valid_mask = if let Some(vr) = valid_row {
            let curr_valid_vec = _mm256_loadu_ps(vr.as_ptr().add(px_start));
            let half_vec = _mm256_set1_ps(0.5);
            _mm256_and_ps(
                valid_mask,
                _mm256_cmp_ps::<_CMP_GE_OQ>(curr_valid_vec, half_vec),
            )
        } else {
            valid_mask
        };

        let ix = _mm256_floor_ps(u_ref);
        let iy = _mm256_floor_ps(v_ref);
        let dx = _mm256_sub_ps(u_ref, ix);
        let dy = _mm256_sub_ps(v_ref, iy);
        let one = _mm256_set1_ps(1.0);
        let one_minus_dx = _mm256_sub_ps(one, dx);
        let one_minus_dy = _mm256_sub_ps(one, dy);

        let ix_i32 = _mm256_cvttps_epi32(ix);
        let iy_i32 = _mm256_cvttps_epi32(iy);
        let ref_w_vec = _mm256_set1_epi32(ref_w_i32);
        let idx00 = _mm256_add_epi32(_mm256_mullo_epi32(iy_i32, ref_w_vec), ix_i32);
        let idx10 = _mm256_add_epi32(idx00, _mm256_set1_epi32(1));
        let idx01 = _mm256_add_epi32(idx00, ref_w_vec);
        let idx11 = _mm256_add_epi32(idx01, _mm256_set1_epi32(1));

        // AVX2 gather is unconditional. Keep invalid lanes in-bounds; their
        // contribution is selected back to photo_w=1.0 below.
        let zero_i32 = _mm256_setzero_si256();
        let valid_idx_mask = _mm256_castps_si256(valid_mask);
        let idx00_s = _mm256_blendv_epi8(zero_i32, idx00, valid_idx_mask);
        let idx10_s = _mm256_blendv_epi8(zero_i32, idx10, valid_idx_mask);
        let idx01_s = _mm256_blendv_epi8(zero_i32, idx01, valid_idx_mask);
        let idx11_s = _mm256_blendv_epi8(zero_i32, idx11, valid_idx_mask);

        let p00 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx00_s);
        let p10 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx10_s);
        let p01 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx01_s);
        let p11 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx11_s);

        let valid_mask = if let Some(rv) = ref_valid_slice {
            let half_vec = _mm256_set1_ps(0.5);
            let rv00 = _mm256_i32gather_ps::<4>(rv.as_ptr(), idx00_s);
            let rv_valid = _mm256_cmp_ps::<_CMP_GT_OQ>(rv00, half_vec);
            _mm256_and_ps(valid_mask, rv_valid)
        } else {
            valid_mask
        };

        let w00 = _mm256_mul_ps(one_minus_dx, one_minus_dy);
        let w10 = _mm256_mul_ps(dx, one_minus_dy);
        let w01 = _mm256_mul_ps(one_minus_dx, dy);
        let w11 = _mm256_mul_ps(dx, dy);
        let mut i_ref = _mm256_mul_ps(w00, p00);
        i_ref = _mm256_fmadd_ps(w10, p10, i_ref);
        i_ref = _mm256_fmadd_ps(w01, p01, i_ref);
        i_ref = _mm256_fmadd_ps(w11, p11, i_ref);

        let i_curr = _mm256_loadu_ps(curr_row.as_ptr().add(px_start));

        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF_u32 as i32));
        let residual = _mm256_and_ps(_mm256_sub_ps(i_ref, i_curr), abs_mask);
        let residual_clamped = _mm256_max_ps(residual, one);
        let photo_w_valid = _mm256_rcp_ps(residual_clamped);

        let photo_w = _mm256_blendv_ps(one, photo_w_valid, valid_mask);

        let inv_var_w_vec = _mm256_set1_ps(inv_var_w);
        let patch_eta_vec = _mm256_set1_ps(patch_eta);
        let w = _mm256_mul_ps(inv_var_w_vec, photo_w);
        let w_eta = _mm256_mul_ps(w, patch_eta_vec);

        let eta_ptr = eta_buf.as_mut_ptr().add(px_start);
        let w_ptr = w_buf.as_mut_ptr().add(px_start);
        let eta_old = _mm256_loadu_ps(eta_ptr);
        let w_old = _mm256_loadu_ps(w_ptr);
        _mm256_storeu_ps(eta_ptr, _mm256_add_ps(eta_old, w_eta));
        _mm256_storeu_ps(w_ptr, _mm256_add_ps(w_old, w));

        let status_ptr = &mut status_buf[px_start..px_start + 8];
        for s in status_ptr.iter_mut() {
            if (patch_status as u8) > (*s as u8) {
                *s = patch_status;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn densify_row_neon(
    coeffs: &RowWarpCoeffs,
    px_start: usize,
    curr_row: &[f32],
    valid_row: Option<&[f32]>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    ref_w: usize,
    ref_h: usize,
    inv_var_w: f32,
    patch_eta: f32,
    eta_buf: &mut [f32],
    w_buf: &mut [f32],
    status_buf: &mut [PatchStatus],
    patch_status: PatchStatus,
) {
    use std::arch::aarch64::*;

    unsafe {
        let ref_slice = ref_img.as_slice();
        let ref_valid_slice = ref_valid.map(|v| v.as_slice());

        let px_offsets: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let px_vec = vaddq_f32(vdupq_n_f32(px_start as f32), vld1q_f32(px_offsets.as_ptr()));

        let xr0 = vfmaq_f32(vdupq_n_f32(coeffs.b[0]), vdupq_n_f32(coeffs.a[0]), px_vec);
        let xr1 = vfmaq_f32(vdupq_n_f32(coeffs.b[1]), vdupq_n_f32(coeffs.a[1]), px_vec);
        let xr2 = vfmaq_f32(vdupq_n_f32(coeffs.b[2]), vdupq_n_f32(coeffs.a[2]), px_vec);

        // 1/xr2 via reciprocal estimate + one Newton step (matches AVX2 path).
        let rcp = vrecpeq_f32(xr2);
        let z_inv = vmulq_f32(rcp, vrecpsq_f32(xr2, rcp));

        let fx_s = vdupq_n_f32(coeffs.fx_s);
        let fy_s = vdupq_n_f32(coeffs.fy_s);
        let cx_s = vdupq_n_f32(coeffs.cx_s);
        let cy_s = vdupq_n_f32(coeffs.cy_s);
        let u_ref = vfmaq_f32(cx_s, fx_s, vmulq_f32(xr0, z_inv));
        let v_ref = vfmaq_f32(cy_s, fy_s, vmulq_f32(xr1, z_inv));

        let eps = vdupq_n_f32(1e-6);
        let zero = vdupq_n_f32(0.0);
        let max_u = vdupq_n_f32((ref_w - 1) as f32);
        let max_v = vdupq_n_f32((ref_h - 1) as f32);

        let valid_mask = vandq_u32(
            vandq_u32(vcgtq_f32(xr2, eps), vcgeq_f32(u_ref, zero)),
            vandq_u32(
                vcltq_f32(u_ref, max_u),
                vandq_u32(vcgeq_f32(v_ref, zero), vcltq_f32(v_ref, max_v)),
            ),
        );

        let valid_mask = if let Some(vr) = valid_row {
            let curr_valid_vec = vld1q_f32(vr.as_ptr().add(px_start));
            vandq_u32(valid_mask, vcgeq_f32(curr_valid_vec, vdupq_n_f32(0.5)))
        } else {
            valid_mask
        };

        let ix = vrndmq_f32(u_ref); // round toward -inf
        let iy = vrndmq_f32(v_ref);
        let dx = vsubq_f32(u_ref, ix);
        let dy = vsubq_f32(v_ref, iy);
        let one = vdupq_n_f32(1.0);
        let one_minus_dx = vsubq_f32(one, dx);
        let one_minus_dy = vsubq_f32(one, dy);

        let ix_i32 = vcvtq_s32_f32(ix);
        let iy_i32 = vcvtq_s32_f32(iy);
        let ref_w_vec = vdupq_n_s32(ref_w as i32);
        let idx00_v = vaddq_s32(vmulq_s32(iy_i32, ref_w_vec), ix_i32);
        let idx10_v = vaddq_s32(idx00_v, vdupq_n_s32(1));
        let idx01_v = vaddq_s32(idx00_v, ref_w_vec);
        let idx11_v = vaddq_s32(idx01_v, vdupq_n_s32(1));

        // NEON has no masked gather, so substitute index 0 on invalid lanes to keep
        // the unconditional gather in-bounds; their photo_w gets selected back to 1.0 below.
        let zero_i32 = vdupq_n_s32(0);
        let idx00_s = vbslq_s32(valid_mask, idx00_v, zero_i32);
        let idx10_s = vbslq_s32(valid_mask, idx10_v, zero_i32);
        let idx01_s = vbslq_s32(valid_mask, idx01_v, zero_i32);
        let idx11_s = vbslq_s32(valid_mask, idx11_v, zero_i32);

        let p00 = gather4_f32(ref_slice, idx00_s);
        let p10 = gather4_f32(ref_slice, idx10_s);
        let p01 = gather4_f32(ref_slice, idx01_s);
        let p11 = gather4_f32(ref_slice, idx11_s);

        let valid_mask = if let Some(rv) = ref_valid_slice {
            let rv00 = gather4_f32(rv, idx00_s);
            vandq_u32(valid_mask, vcgtq_f32(rv00, vdupq_n_f32(0.5)))
        } else {
            valid_mask
        };

        let w00v = vmulq_f32(one_minus_dx, one_minus_dy);
        let w10v = vmulq_f32(dx, one_minus_dy);
        let w01v = vmulq_f32(one_minus_dx, dy);
        let w11v = vmulq_f32(dx, dy);
        let mut i_ref = vmulq_f32(w00v, p00);
        i_ref = vfmaq_f32(i_ref, w10v, p10);
        i_ref = vfmaq_f32(i_ref, w01v, p01);
        i_ref = vfmaq_f32(i_ref, w11v, p11);

        let i_curr = vld1q_f32(curr_row.as_ptr().add(px_start));
        let residual = vabsq_f32(vsubq_f32(i_ref, i_curr));
        let residual_clamped = vmaxq_f32(residual, one);
        let photo_w_valid = vrecpeq_f32(residual_clamped); // ~12-bit, matches AVX2 `_mm256_rcp_ps`

        let photo_w = vbslq_f32(valid_mask, photo_w_valid, one);

        let w = vmulq_f32(vdupq_n_f32(inv_var_w), photo_w);
        let w_eta = vmulq_f32(w, vdupq_n_f32(patch_eta));

        let eta_ptr = eta_buf.as_mut_ptr().add(px_start);
        let w_ptr = w_buf.as_mut_ptr().add(px_start);
        let eta_old = vld1q_f32(eta_ptr);
        let w_old = vld1q_f32(w_ptr);
        vst1q_f32(eta_ptr, vaddq_f32(eta_old, w_eta));
        vst1q_f32(w_ptr, vaddq_f32(w_old, w));

        let status_ptr = &mut status_buf[px_start..px_start + 4];
        for s in status_ptr.iter_mut() {
            if (patch_status as u8) > (*s as u8) {
                *s = patch_status;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gather4_f32(
    slice: &[f32],
    idx: std::arch::aarch64::int32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let base = slice.as_ptr();
        let i0 = vgetq_lane_s32::<0>(idx) as usize;
        let i1 = vgetq_lane_s32::<1>(idx) as usize;
        let i2 = vgetq_lane_s32::<2>(idx) as usize;
        let i3 = vgetq_lane_s32::<3>(idx) as usize;
        let v = vdupq_n_f32(0.0);
        let v = vld1q_lane_f32::<0>(base.add(i0), v);
        let v = vld1q_lane_f32::<1>(base.add(i1), v);
        let v = vld1q_lane_f32::<2>(base.add(i2), v);
        vld1q_lane_f32::<3>(base.add(i3), v)
    }
}

#[cfg(feature = "parallel")]
pub(super) fn densify_pixels_parallel(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    if mapper.camera_mode == PatchDepthCameraMode::UndistortedPinhole {
        return densify_pixels_pinhole_parallel(
            grid, width, height, curr_img, curr_valid, ref_img, ref_valid, mapper, intr, rel_pose,
        );
    }
    densify_pixels_generic_parallel(
        grid, width, height, curr_img, curr_valid, ref_img, ref_valid, mapper, intr, rel_pose,
    )
}

#[cfg(feature = "parallel")]
fn densify_pixels_pinhole_parallel(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    use rayon::prelude::*;

    let n = width * height;
    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    let mut status = vec![PatchStatus::Unknown; n];

    let ref_w = ref_img.width();
    let ref_h = ref_img.height();
    let curr_w = curr_img.width();
    let curr_valid_w = curr_valid.map(|v| v.width());

    #[cfg(target_arch = "x86_64")]
    let use_avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let patch_size = 2 * grid.half;
    let min_len = (height / (rayon::current_num_threads() * 8)).max(1);

    depth
        .par_chunks_mut(width)
        .zip(variance.par_chunks_mut(width))
        .zip(status.par_chunks_mut(width))
        .enumerate()
        .with_min_len(min_len)
        .for_each(|(py, ((d_row, v_row), s_row))| {
            let (iv_start, iv_end) = grid.overlap_v(py);
            if iv_start > iv_end {
                return;
            }

            let mut eta_row = vec![0.0_f32; width];
            let mut w_row = vec![0.0_f32; width];
            let mut status_row = vec![PatchStatus::Unknown; width];

            let curr_row = &curr_img.as_slice()[py * curr_w..py * curr_w + width];
            let valid_row = curr_valid.map(|v| {
                let vw = curr_valid_w.unwrap();
                &v.as_slice()[py * vw..py * vw + width]
            });

            for iv in iv_start..=iv_end {
                for iu in 0..grid.n_u {
                    let patch = grid.get(iu, iv);
                    let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                        continue;
                    };

                    let patch_eta_f32 = patch.eta as f32;
                    let weighted_eta = inv_var_w * patch_eta_f32;

                    let px_start = iu * grid.stride;
                    let px_end = (px_start + patch_size).min(width);

                    if patch.status == PatchStatus::PhotoRefined {
                        let cu = (iu * grid.stride + grid.half) as f64;
                        let cv = (iv * grid.stride + grid.half) as f64;
                        let bx_c = (cu / intr.scale_from_original - mapper.intrinsics.cx)
                            / mapper.intrinsics.fx;
                        let by_c = (cv / intr.scale_from_original - mapper.intrinsics.cy)
                            / mapper.intrinsics.fy;
                        let range_per_z_c = (bx_c * bx_c + by_c * by_c + 1.0).sqrt();
                        let z_center = patch.eta.exp() / range_per_z_c;
                        let coeffs = RowWarpCoeffs::new(py, z_center, mapper, intr, rel_pose);
                        let mut px = px_start;

                        #[cfg(target_arch = "x86_64")]
                        if use_avx2 {
                            while px + 8 <= px_end {
                                unsafe {
                                    densify_row_avx2(
                                        &coeffs,
                                        px,
                                        curr_row,
                                        valid_row,
                                        ref_img,
                                        ref_valid,
                                        ref_w,
                                        ref_h,
                                        inv_var_w,
                                        patch_eta_f32,
                                        &mut eta_row,
                                        &mut w_row,
                                        &mut status_row,
                                        PatchStatus::PhotoRefined,
                                    );
                                }
                                px += 8;
                            }
                        }

                        #[cfg(target_arch = "aarch64")]
                        while px + 4 <= px_end {
                            unsafe {
                                densify_row_neon(
                                    &coeffs,
                                    px,
                                    curr_row,
                                    valid_row,
                                    ref_img,
                                    ref_valid,
                                    ref_w,
                                    ref_h,
                                    inv_var_w,
                                    patch_eta_f32,
                                    &mut eta_row,
                                    &mut w_row,
                                    &mut status_row,
                                    PatchStatus::PhotoRefined,
                                );
                            }
                            px += 4;
                        }

                        for px in px..px_end {
                            let (u_ref, v_ref, z_ref) = coeffs.warp(px as f32);
                            let photo_w = if z_ref > 1e-6
                                && u_ref >= 0.0
                                && v_ref >= 0.0
                                && u_ref < (ref_w - 1) as f32
                                && v_ref < (ref_h - 1) as f32
                            {
                                let i_curr = curr_row[px];
                                let vr_block = valid_row.map(|vr| vr[px] < 0.5).unwrap_or(false);
                                if vr_block {
                                    1.0_f32
                                } else {
                                    photo_weight_inline(
                                        i_curr,
                                        ref_img,
                                        ref_valid,
                                        u_ref as f64,
                                        v_ref as f64,
                                    )
                                }
                            } else {
                                1.0_f32
                            };
                            let w = inv_var_w * photo_w;
                            eta_row[px] += w * patch_eta_f32;
                            w_row[px] += w;
                            if (patch.status as u8) > (status_row[px] as u8) {
                                status_row[px] = patch.status;
                            }
                        }
                    } else {
                        for px in px_start..px_end {
                            eta_row[px] += weighted_eta;
                            w_row[px] += inv_var_w;
                            if (patch.status as u8) > (status_row[px] as u8) {
                                status_row[px] = patch.status;
                            }
                        }
                    }
                }
            }

            let s = intr.scale_from_original as f32;
            let fx = mapper.intrinsics.fx as f32;
            let fy = mapper.intrinsics.fy as f32;
            let cx = mapper.intrinsics.cx as f32;
            let cy = mapper.intrinsics.cy as f32;
            for px in 0..width {
                if w_row[px] > 0.0 {
                    let bx = (px as f32 / s - cx) / fx;
                    let by = (py as f32 / s - cy) / fy;
                    let unit_b_z = 1.0 / (bx * bx + by * by + 1.0).sqrt();
                    d_row[px] = (eta_row[px] / w_row[px]).exp() * unit_b_z;
                    v_row[px] = 1.0 / w_row[px];
                    s_row[px] = status_row[px];
                }
            }
        });

    PatchDepthOutput {
        depth: DepthMap::from_vec(width, height, depth).expect("depth size"),
        variance: DepthMap::from_vec(width, height, variance).expect("variance size"),
        status: DepthMap::from_vec(width, height, status).expect("status size"),
    }
}

#[cfg(feature = "parallel")]
fn densify_pixels_generic_parallel(
    grid: &PatchGrid,
    width: usize,
    height: usize,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> PatchDepthOutput {
    use rayon::prelude::*;

    let n = width * height;
    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    let mut status = vec![PatchStatus::Unknown; n];

    let min_len = (height / (rayon::current_num_threads() * 8)).max(1);
    depth
        .par_chunks_mut(width)
        .zip(variance.par_chunks_mut(width))
        .zip(status.par_chunks_mut(width))
        .enumerate()
        .with_min_len(min_len)
        .for_each(|(py, ((d_row, v_row), s_row))| {
            let (iv_start, iv_end) = grid.overlap_v(py);
            if iv_start > iv_end {
                return;
            }

            for px in 0..width {
                let (iu_start, iu_end) = grid.overlap_u(px);
                if iu_start > iu_end {
                    continue;
                }

                let mut eta_acc = 0.0_f32;
                let mut w_acc = 0.0_f32;
                let mut best_status = PatchStatus::Unknown;

                for iv in iv_start..=iv_end {
                    for iu in iu_start..=iu_end {
                        let patch = grid.get(iu, iv);
                        let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                            continue;
                        };

                        let photo_w = if patch.status == PatchStatus::PhotoRefined {
                            let cu = iu * grid.stride + grid.half;
                            let cv = iv * grid.stride + grid.half;
                            let range_per_z = mapper
                                .bearing_for_scaled_pixel(cu as f64, cv as f64, intr)
                                .map(|b| b.norm())
                                .unwrap_or(1.0);
                            let rho_center = range_per_z * (-patch.eta).exp();
                            compute_photo_weight(
                                px, py, rho_center, curr_img, curr_valid, ref_img, ref_valid,
                                mapper, intr, rel_pose,
                            )
                        } else {
                            1.0_f32
                        };

                        let w = inv_var_w * photo_w;
                        eta_acc += w * patch.eta as f32;
                        w_acc += w;

                        if (patch.status as u8) > (best_status as u8) {
                            best_status = patch.status;
                        }
                    }
                }

                if w_acc > 0.0 {
                    let unit_b_z = mapper.unit_b_z_for_pixel(px as f64, py as f64, intr) as f32;
                    d_row[px] = (eta_acc / w_acc).exp() * unit_b_z;
                    v_row[px] = 1.0 / w_acc;
                    s_row[px] = best_status;
                }
            }
        });

    PatchDepthOutput {
        depth: DepthMap::from_vec(width, height, depth).expect("depth size"),
        variance: DepthMap::from_vec(width, height, variance).expect("variance size"),
        status: DepthMap::from_vec(width, height, status).expect("status size"),
    }
}

#[inline]
fn compute_photo_weight(
    px: usize,
    py: usize,
    rho: f64,
    curr_img: &Image<f32>,
    curr_valid: Option<&Image<f32>>,
    ref_img: &Image<f32>,
    ref_valid: Option<&Image<f32>>,
    mapper: &PatchDepthMapper,
    intr: &ScaledIntrinsics,
    rel_pose: &RelativePose,
) -> f32 {
    let Some((u_ref, v_ref, _, _)) =
        mapper.warp_scaled_pixel(px as f64, py as f64, rho, intr, rel_pose)
    else {
        return 1.0;
    };

    let i_curr = if px < curr_img.width() && py < curr_img.height() {
        curr_img.as_slice()[py * curr_img.width() + px]
    } else {
        return 1.0;
    };

    if let Some(valid) = curr_valid {
        if valid.as_slice()[py * valid.width() + px] < 0.5 {
            return 1.0;
        }
    }

    let Some(i_ref) = sample_bilinear_valid(ref_img, ref_valid, u_ref, v_ref) else {
        return 1.0;
    };

    let residual = (i_ref - i_curr).abs();
    1.0 / residual.max(1.0)
}
