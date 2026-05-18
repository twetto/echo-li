use super::image_ops::sample_bilinear_valid;
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
        let unknown = PatchEstimate {
            rho: 0.0,
            var: 1e10,
            status: PatchStatus::Unknown,
        };
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
    settings: &PatchDepthSettings,
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

            let mut rho_acc = 0.0_f64;
            let mut w_acc = 0.0_f64;
            let mut best_status = PatchStatus::Unknown;

            for iv in iv_start..=iv_end {
                for iu in iu_start..=iu_end {
                    let patch = grid.get(iu, iv);
                    if patch.status == PatchStatus::Unknown || patch.status == PatchStatus::Rejected
                    {
                        continue;
                    }

                    let status_weight = match patch.status {
                        PatchStatus::PhotoRefined => settings.status_weight_photo,
                        PatchStatus::SeedOnly => settings.status_weight_seed,
                        _ => 0.0,
                    };
                    let inv_var_w = status_weight / patch.var.max(settings.var_floor);

                    let photo_w = if patch.status == PatchStatus::PhotoRefined {
                        compute_photo_weight(
                            px, py, patch.rho, curr_img, curr_valid, ref_img, ref_valid, mapper,
                            intr, rel_pose,
                        )
                    } else {
                        1.0
                    };

                    let w = inv_var_w * photo_w;
                    rho_acc += w * patch.rho;
                    w_acc += w;

                    if (patch.status as u8) > (best_status as u8) {
                        best_status = patch.status;
                    }
                }
            }

            if w_acc > 0.0 {
                let idx = py * width + px;
                depth[idx] = (1.0 / (rho_acc / w_acc)) as f32;
                variance[idx] = (1.0 / w_acc) as f32;
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
    settings: &PatchDepthSettings,
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

            let mut rho_acc = 0.0_f64;
            let mut w_acc = 0.0_f64;
            let mut best_status = PatchStatus::Unknown;

            for iv in iv_start..=iv_end {
                for iu in iu_start..=iu_end {
                    let patch = grid.get(iu, iv);
                    if patch.status == PatchStatus::Unknown || patch.status == PatchStatus::Rejected
                    {
                        continue;
                    }

                    let status_weight = match patch.status {
                        PatchStatus::PhotoRefined => settings.status_weight_photo,
                        PatchStatus::SeedOnly => settings.status_weight_seed,
                        _ => 0.0,
                    };
                    let inv_var_w = status_weight / patch.var.max(settings.var_floor);

                    let photo_w = if patch.status == PatchStatus::PhotoRefined {
                        compute_photo_weight(
                            px, *py, patch.rho, curr_img, curr_valid, ref_img, ref_valid, mapper,
                            intr, rel_pose,
                        )
                    } else {
                        1.0
                    };

                    let w = inv_var_w * photo_w;
                    rho_acc += w * patch.rho;
                    w_acc += w;

                    if (patch.status as u8) > (best_status as u8) {
                        best_status = patch.status;
                    }
                }
            }

            if w_acc > 0.0 {
                d_row[px] = (1.0 / (rho_acc / w_acc)) as f32;
                v_row[px] = (1.0 / w_acc) as f32;
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
) -> f64 {
    let Some((u_ref, v_ref, _, _)) =
        mapper.warp_scaled_pixel(px as f64, py as f64, rho, intr, rel_pose)
    else {
        return 1.0;
    };

    let i_curr = if px < curr_img.width() && py < curr_img.height() {
        curr_img.as_slice()[py * curr_img.width() + px] as f64
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

    let residual = (i_ref as f64 - i_curr).abs();
    1.0 / residual.max(1.0)
}
