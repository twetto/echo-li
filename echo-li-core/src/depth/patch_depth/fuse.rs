use super::image_ops::sample_bilinear_valid;
#[cfg(not(feature = "parallel"))]
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
            let mut rho_acc = 0.0_f32;
            let mut w_acc = 0.0_f32;
            let mut best_status = PatchStatus::Unknown;
            for iv in iv_start..=iv_end {
                for iu in iu_start..=iu_end {
                    let patch = grid.get(iu, iv);
                    let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                        continue;
                    };
                    let photo_w = if patch.status == PatchStatus::PhotoRefined {
                        compute_photo_weight(
                            px, py, patch.rho, curr_img, curr_valid, ref_img, ref_valid, mapper,
                            intr, rel_pose,
                        )
                    } else {
                        1.0_f32
                    };
                    let w = inv_var_w * photo_w;
                    rho_acc += w * patch.rho as f32;
                    w_acc += w;
                    if (patch.status as u8) > (best_status as u8) {
                        best_status = patch.status;
                    }
                }
            }
            if w_acc > 0.0 {
                let idx = py * width + px;
                depth[idx] = w_acc / rho_acc;
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

#[cfg(not(feature = "parallel"))]
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

#[cfg(not(feature = "parallel"))]
impl RowWarpCoeffs {
    fn new(
        py: usize,
        rho: f64,
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
        // x_curr = bearing / rho
        // x_ref = R * x_curr + t
        //
        // bearing_x = px * (1/(s*fx)) + (-cx/fx)
        // bearing_y = py * (1/(s*fy)) + (-cy/fy)  [constant for row]
        // bearing_z = 1.0
        let inv_rho = 1.0 / rho;
        let kx = 1.0 / (s * fx);
        let bx = -cx / fx;
        let by_val = (py as f64 / s - cy) / fy;

        // x_curr = [bearing_x / rho, bearing_y / rho, 1/rho]
        // x_ref[j] = sum_i R[j][i] * x_curr[i] + t[j]
        //          = R[j][0]*(px*kx + bx)/rho + R[j][1]*by_val/rho + R[j][2]/rho + t[j]
        //          = px * (R[j][0]*kx/rho) + (R[j][0]*bx/rho + R[j][1]*by_val/rho + R[j][2]/rho + t[j])
        let r = &rel_pose.r;
        let t = &rel_pose.t;
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        for j in 0..3 {
            a[j] = (r[(j, 0)] * kx * inv_rho) as f32;
            b[j] = (r[(j, 0)] * bx * inv_rho
                + r[(j, 1)] * by_val * inv_rho
                + r[(j, 2)] * inv_rho
                + t[j]) as f32;
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
    let mut rho_buf = vec![0.0_f32; n];
    let mut w_buf = vec![0.0_f32; n];
    let mut status_buf = vec![PatchStatus::Unknown; n];

    let ref_w = ref_img.width();
    let ref_h = ref_img.height();

    for iv in 0..grid.n_v {
        for iu in 0..grid.n_u {
            let patch = grid.get(iu, iv);
            let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                continue;
            };

            let patch_rho_f32 = patch.rho as f32;
            let weighted_rho = inv_var_w * patch_rho_f32;

            let patch_size = 2 * grid.half;
            let py_start = iv * grid.stride;
            let py_end = (py_start + patch_size).min(height);
            let px_start = iu * grid.stride;
            let px_end = (px_start + patch_size).min(width);

            if patch.status == PatchStatus::PhotoRefined {
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;

                for py in py_start..py_end {
                    let coeffs = RowWarpCoeffs::new(py, patch.rho, mapper, intr, rel_pose);
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
                                    patch_rho_f32,
                                    &mut rho_buf[row_off..],
                                    &mut w_buf[row_off..],
                                    &mut status_buf[row_off..],
                                    PatchStatus::PhotoRefined,
                                );
                            }
                            px += 8;
                        }
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
                        rho_buf[idx] += w * patch_rho_f32;
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
                        rho_buf[idx] += weighted_rho;
                        w_buf[idx] += inv_var_w;
                        if (patch.status as u8) > (status_buf[idx] as u8) {
                            status_buf[idx] = patch.status;
                        }
                    }
                }
            }
        }
    }

    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    for i in 0..n {
        if w_buf[i] > 0.0 {
            depth[i] = w_buf[i] / rho_buf[i];
            variance[i] = 1.0 / w_buf[i];
        }
    }

    PatchDepthOutput {
        depth: DepthMap::from_vec(width, height, depth).expect("depth size"),
        variance: DepthMap::from_vec(width, height, variance).expect("variance size"),
        status: DepthMap::from_vec(width, height, status_buf).expect("status size"),
    }
}

#[inline(always)]
#[cfg(not(feature = "parallel"))]
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

#[cfg(all(not(feature = "parallel"), target_arch = "x86_64"))]
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
    patch_rho: f32,
    rho_buf: &mut [f32],
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

        let p00 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx00);
        let p10 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx10);
        let p01 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx01);
        let p11 = _mm256_i32gather_ps::<4>(ref_slice.as_ptr(), idx11);

        let valid_mask = if let Some(rv) = ref_valid_slice {
            let half_vec = _mm256_set1_ps(0.5);
            let rv00 = _mm256_i32gather_ps::<4>(rv.as_ptr(), idx00);
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
        let patch_rho_vec = _mm256_set1_ps(patch_rho);
        let w = _mm256_mul_ps(inv_var_w_vec, photo_w);
        let w_rho = _mm256_mul_ps(w, patch_rho_vec);

        let rho_ptr = rho_buf.as_mut_ptr().add(px_start);
        let w_ptr = w_buf.as_mut_ptr().add(px_start);
        let rho_old = _mm256_loadu_ps(rho_ptr);
        let w_old = _mm256_loadu_ps(w_ptr);
        _mm256_storeu_ps(rho_ptr, _mm256_add_ps(rho_old, w_rho));
        _mm256_storeu_ps(w_ptr, _mm256_add_ps(w_old, w));

        let status_ptr = &mut status_buf[px_start..px_start + 8];
        for s in status_ptr.iter_mut() {
            if (patch_status as u8) > (*s as u8) {
                *s = patch_status;
            }
        }
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
    use rayon::prelude::*;

    let n = width * height;
    let mut depth = vec![f32::NAN; n];
    let mut variance = vec![f32::INFINITY; n];
    let mut status = vec![PatchStatus::Unknown; n];

    let row_chunks: Vec<_> = (0..height).collect();
    let depth_chunks: Vec<&mut [f32]> = depth.chunks_mut(width).collect();
    let var_chunks: Vec<&mut [f32]> = variance.chunks_mut(width).collect();
    let status_chunks: Vec<&mut [PatchStatus]> = status.chunks_mut(width).collect();

    // Combine into tuples for parallel iteration
    let mut rows: Vec<(usize, &mut [f32], &mut [f32], &mut [PatchStatus])> = row_chunks
        .into_iter()
        .zip(depth_chunks)
        .zip(var_chunks)
        .zip(status_chunks)
        .map(|(((py, d), v), s)| (py, d, v, s))
        .collect();

    rows.par_iter_mut().for_each(|(py, d_row, v_row, s_row)| {
        let (iv_start, iv_end) = grid.overlap_v(*py);
        if iv_start > iv_end {
            return;
        }

        for px in 0..width {
            let (iu_start, iu_end) = grid.overlap_u(px);
            if iu_start > iu_end {
                continue;
            }

            let mut rho_acc = 0.0_f32;
            let mut w_acc = 0.0_f32;
            let mut best_status = PatchStatus::Unknown;

            for iv in iv_start..=iv_end {
                for iu in iu_start..=iu_end {
                    let patch = grid.get(iu, iv);
                    let Some(inv_var_w) = patch.inv_var_weight_f32() else {
                        continue;
                    };

                    let photo_w = if patch.status == PatchStatus::PhotoRefined {
                        compute_photo_weight(
                            px, *py, patch.rho, curr_img, curr_valid, ref_img, ref_valid, mapper,
                            intr, rel_pose,
                        )
                    } else {
                        1.0_f32
                    };

                    let w = inv_var_w * photo_w;
                    rho_acc += w * patch.rho as f32;
                    w_acc += w;

                    if (patch.status as u8) > (best_status as u8) {
                        best_status = patch.status;
                    }
                }
            }

            if w_acc > 0.0 {
                d_row[px] = w_acc / rho_acc;
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
