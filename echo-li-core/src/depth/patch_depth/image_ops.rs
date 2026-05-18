use nalgebra::Vector3;
use rudolf_v::image::Image;
use rudolf_v::pyramid::Pyramid;

use super::{BilinearPatchFootprint, ScaledIntrinsics, UndistortSample};
use crate::core_types::CameraIntrinsics;
use crate::mathematical::camera::CameraModel;

pub(super) fn sample_nearest(img: &Image<f32>, u: f64, v: f64) -> Option<f32> {
    let x = u as isize;
    let y = v as isize;
    if x < 0 || y < 0 || x >= img.width() as isize || y >= img.height() as isize {
        return None;
    }
    // SAFETY: bounds were checked above.
    Some(unsafe { img.get_unchecked(x as usize, y as usize) })
}

pub(super) fn sample_bilinear_valid(
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

pub(super) fn sample_bilinear_valid_with_grad(
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

pub(super) fn bilinear_patch_footprint(
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
pub(super) unsafe fn mask_row_valid(row: Option<*const f32>, x: usize) -> bool {
    match row {
        Some(row) => unsafe { *row.add(x) > 0.5 },
        None => true,
    }
}

pub(super) unsafe fn bilinear_unchecked(
    img: &Image<f32>,
    x: usize,
    y: usize,
    weights: [f32; 4],
) -> f32 {
    unsafe {
        let p00 = img.get_unchecked(x, y);
        let p10 = img.get_unchecked(x + 1, y);
        let p01 = img.get_unchecked(x, y + 1);
        let p11 = img.get_unchecked(x + 1, y + 1);
        weights[0] * p00 + weights[1] * p10 + weights[2] * p01 + weights[3] * p11
    }
}

#[inline(always)]
pub(super) unsafe fn bilerp_ptr(
    r0: *const f32,
    r1: *const f32,
    x: usize,
    weights: [f32; 4],
) -> f32 {
    unsafe {
        let p00 = *r0.add(x);
        let p10 = *r0.add(x + 1);
        let p01 = *r1.add(x);
        let p11 = *r1.add(x + 1);
        weights[0] * p00 + weights[1] * p10 + weights[2] * p01 + weights[3] * p11
    }
}

pub(super) fn sample_valid_nearest(mask: Option<&Image<f32>>, u: f64, v: f64) -> bool {
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

pub(super) fn build_mask_pyramid(
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
        let eroded = mask_pyrdown_erode(prev);
        let (next, next_w, next_h) = mask_pyrdown_from_eroded(&eroded, prev.width(), prev.height());
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
        let eroded = mask_pyrdown_erode(prev);
        let (next, next_w, next_h) = mask_pyrdown_from_eroded(&eroded, prev.width(), prev.height());
        full.push(Image::from_vec(next_w, next_h, next));
    }
    full.into_iter().skip(offset).collect()
}

pub(super) fn build_bilinear_valid_pyramid(valid_pyramid: &[Image<f32>]) -> Vec<Image<f32>> {
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

fn mask_pyrdown_erode(mask: &Image<f32>) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { mask_pyrdown_erode_avx2(mask) };
        }
    }
    mask_pyrdown_erode_scalar(mask)
}

fn mask_pyrdown_erode_scalar(mask: &Image<f32>) -> Vec<u8> {
    let w = mask.width();
    let h = mask.height();
    let src = mask.as_slice();

    let mut bin = vec![0u8; w * h];
    for (i, &v) in src.iter().enumerate() {
        bin[i] = (v > 0.5) as u8;
    }

    let mut h_eroded = vec![0u8; w * h];
    if w >= 5 {
        for y in 0..h {
            let row = &bin[y * w..(y + 1) * w];
            let out_row = &mut h_eroded[y * w..(y + 1) * w];
            for x in 2..w - 2 {
                out_row[x] = row[x - 2] & row[x - 1] & row[x] & row[x + 1] & row[x + 2];
            }
        }
    }

    let mut eroded = vec![0u8; w * h];
    if h >= 5 {
        for y in 2..h - 2 {
            let r0 = &h_eroded[(y - 2) * w..(y - 1) * w];
            let r1 = &h_eroded[(y - 1) * w..y * w];
            let r2 = &h_eroded[y * w..(y + 1) * w];
            let r3 = &h_eroded[(y + 1) * w..(y + 2) * w];
            let r4 = &h_eroded[(y + 2) * w..(y + 3) * w];
            let out_row = &mut eroded[y * w..(y + 1) * w];
            for x in 0..w {
                out_row[x] = r0[x] & r1[x] & r2[x] & r3[x] & r4[x];
            }
        }
    }

    eroded
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mask_pyrdown_erode_avx2(mask: &Image<f32>) -> Vec<u8> {
    unsafe {
        use std::arch::x86_64::*;

        let w = mask.width();
        let h = mask.height();
        let src = mask.as_slice();

        let mut bin = vec![0u8; w * h];
        let threshold = _mm256_set1_ps(0.5);
        let n = w * h;
        let mut i = 0;
        while i + 32 <= n {
            let m0 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_loadu_ps(src.as_ptr().add(i)), threshold);
            let m1 =
                _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_loadu_ps(src.as_ptr().add(i + 8)), threshold);
            let m2 =
                _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_loadu_ps(src.as_ptr().add(i + 16)), threshold);
            let m3 =
                _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_loadu_ps(src.as_ptr().add(i + 24)), threshold);
            let b0 = _mm256_movemask_ps(m0) as u32;
            let b1 = _mm256_movemask_ps(m1) as u32;
            let b2 = _mm256_movemask_ps(m2) as u32;
            let b3 = _mm256_movemask_ps(m3) as u32;
            for bit in 0..8u32 {
                *bin.get_unchecked_mut(i + bit as usize) = ((b0 >> bit) & 1) as u8;
                *bin.get_unchecked_mut(i + 8 + bit as usize) = ((b1 >> bit) & 1) as u8;
                *bin.get_unchecked_mut(i + 16 + bit as usize) = ((b2 >> bit) & 1) as u8;
                *bin.get_unchecked_mut(i + 24 + bit as usize) = ((b3 >> bit) & 1) as u8;
            }
            i += 32;
        }
        for j in i..n {
            bin[j] = (*src.get_unchecked(j) > 0.5) as u8;
        }

        let mut h_eroded = vec![0u8; w * h];
        if w >= 5 {
            for y in 0..h {
                let row_ptr = bin.as_ptr().add(y * w);
                let out_ptr = h_eroded.as_mut_ptr().add(y * w);
                let interior = w - 4;
                let mut x = 0usize;
                while x + 32 <= interior {
                    let a0 = _mm256_loadu_si256(row_ptr.add(x) as *const __m256i);
                    let a1 = _mm256_loadu_si256(row_ptr.add(x + 1) as *const __m256i);
                    let a2 = _mm256_loadu_si256(row_ptr.add(x + 2) as *const __m256i);
                    let a3 = _mm256_loadu_si256(row_ptr.add(x + 3) as *const __m256i);
                    let a4 = _mm256_loadu_si256(row_ptr.add(x + 4) as *const __m256i);
                    let result = _mm256_and_si256(
                        _mm256_and_si256(a0, a1),
                        _mm256_and_si256(_mm256_and_si256(a2, a3), a4),
                    );
                    _mm256_storeu_si256(out_ptr.add(x + 2) as *mut __m256i, result);
                    x += 32;
                }
                for xx in x..interior {
                    *out_ptr.add(xx + 2) = *row_ptr.add(xx)
                        & *row_ptr.add(xx + 1)
                        & *row_ptr.add(xx + 2)
                        & *row_ptr.add(xx + 3)
                        & *row_ptr.add(xx + 4);
                }
            }
        }

        let mut eroded = vec![0u8; w * h];
        if h >= 5 {
            for y in 2..h - 2 {
                let r0 = h_eroded.as_ptr().add((y - 2) * w);
                let r1 = h_eroded.as_ptr().add((y - 1) * w);
                let r2 = h_eroded.as_ptr().add(y * w);
                let r3 = h_eroded.as_ptr().add((y + 1) * w);
                let r4 = h_eroded.as_ptr().add((y + 2) * w);
                let out = eroded.as_mut_ptr().add(y * w);
                let mut x = 0usize;
                while x + 32 <= w {
                    let v0 = _mm256_loadu_si256(r0.add(x) as *const __m256i);
                    let v1 = _mm256_loadu_si256(r1.add(x) as *const __m256i);
                    let v2 = _mm256_loadu_si256(r2.add(x) as *const __m256i);
                    let v3 = _mm256_loadu_si256(r3.add(x) as *const __m256i);
                    let v4 = _mm256_loadu_si256(r4.add(x) as *const __m256i);
                    let result = _mm256_and_si256(
                        _mm256_and_si256(v0, v1),
                        _mm256_and_si256(_mm256_and_si256(v2, v3), v4),
                    );
                    _mm256_storeu_si256(out.add(x) as *mut __m256i, result);
                    x += 32;
                }
                for xx in x..w {
                    *out.add(xx) =
                        *r0.add(xx) & *r1.add(xx) & *r2.add(xx) & *r3.add(xx) & *r4.add(xx);
                }
            }
        }

        eroded
    }
}

fn mask_pyrdown_from_eroded(
    eroded: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    let next_w = (width / 2).max(1);
    let next_h = (height / 2).max(1);
    let mut out = vec![0.0f32; next_w * next_h];
    for y in 0..next_h {
        let sy = y * 2;
        if sy >= height {
            break;
        }
        for x in 0..next_w {
            let sx = x * 2;
            if sx >= width {
                break;
            }
            if eroded[sy * width + sx] != 0 {
                out[y * next_w + x] = 1.0;
            }
        }
    }
    (out, next_w, next_h)
}

pub(super) fn build_pinhole_to_raw_lut(
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

pub(super) fn build_pyramid_from_u8(
    gray: &[u8],
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
) -> Vec<Image<f32>> {
    let base = scaled_image_from_u8(gray, width, height, scale);
    Pyramid::build(&base, levels, 1.0).levels
}

pub(super) fn empty_pyramid() -> Pyramid {
    Pyramid {
        levels: Vec::new(),
        u8_levels: Vec::new(),
        padded_levels: Vec::new(),
        pad_border: 0,
    }
}

pub(super) fn dyadic_scale_offset(scale: f64) -> Option<usize> {
    let mut dyadic = 1.0;
    for offset in 0..=8 {
        if (scale - dyadic).abs() <= 1e-12 {
            return Some(offset);
        }
        dyadic *= 0.5;
    }
    None
}

pub(super) fn gradients(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
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

pub(super) fn scaled_intrinsics(base_scale: f64, levels: usize) -> Vec<ScaledIntrinsics> {
    (0..levels)
        .map(|level| {
            let level_scale = base_scale / (1usize << level) as f64;
            ScaledIntrinsics {
                scale_from_original: level_scale,
            }
        })
        .collect()
}
