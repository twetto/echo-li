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

/// Geometry of one undistort LUT level: the pinhole (output) dimensions and
/// the affine map from full-resolution raw coordinates into this level's raw
/// grid (`raw_level = raw_full * level_scale + raw_offset`).
pub(super) struct UndistortLevelSpec {
    pub(super) lw: usize,
    pub(super) lh: usize,
    /// Pinhole scale of this level relative to the full-resolution image.
    pub(super) level_scale: f64,
    /// Additive offset applied after scaling raw coords into the level grid.
    pub(super) raw_offset: f64,
}

/// Per-level geometry for the undistort LUTs, mirroring how the depth pyramid
/// derives each kept level from the raw image.
pub(super) fn undistort_level_specs(
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
) -> Vec<UndistortLevelSpec> {
    let mut specs = Vec::with_capacity(levels);
    if let Some(offset) = dyadic_scale_offset(scale) {
        // Dyadic: each level is the raw image floor-halved `offset + l` times.
        // The [1,4,6,4,1] pyrdown is centered, so raw_full * level_scale maps
        // exactly with no half-pixel offset.
        for l in 0..levels {
            let shift = offset + l;
            specs.push(UndistortLevelSpec {
                lw: (width >> shift).max(1),
                lh: (height >> shift).max(1),
                level_scale: scale / (1usize << l) as f64,
                raw_offset: 0.0,
            });
        }
    } else {
        // Non-dyadic: the base is a pixel-center-aligned resample
        // (src = (dst + 0.5) / scale - 0.5), then centered halving above it.
        let bw = ((width as f64) * scale + 0.5).floor().max(1.0) as usize;
        let bh = ((height as f64) * scale + 0.5).floor().max(1.0) as usize;
        for l in 0..levels {
            let div = (1usize << l) as f64;
            specs.push(UndistortLevelSpec {
                lw: (bw >> l).max(1),
                lh: (bh >> l).max(1),
                level_scale: scale / div,
                raw_offset: (0.5 * scale - 0.5) / div,
            });
        }
    }
    specs
}

/// Build the undistort LUT for one pyramid level: each pinhole pixel maps to a
/// bilinear footprint in that level's raw (distorted) image.
pub(super) fn build_pinhole_to_raw_lut(
    raw_camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &UndistortLevelSpec,
) -> Vec<Option<UndistortSample>> {
    // Pinhole intrinsics scaled to this level.
    let cx = intrinsics.cx * spec.level_scale;
    let cy = intrinsics.cy * spec.level_scale;
    let fx = intrinsics.fx * spec.level_scale;
    let fy = intrinsics.fy * spec.level_scale;
    let mut lut = Vec::with_capacity(spec.lw * spec.lh);
    for v in 0..spec.lh {
        for u in 0..spec.lw {
            let x = (u as f64 - cx) / fx;
            let y = (v as f64 - cy) / fy;
            let raw_uv = raw_camera.project(&Vector3::new(x, y, 1.0));
            // `raw_uv` is full-resolution raw coords; map into this level.
            let raw_u = raw_uv[0] * spec.level_scale + spec.raw_offset;
            let raw_v = raw_uv[1] * spec.level_scale + spec.raw_offset;
            lut.push(undistort_sample(raw_u, raw_v, spec.lw, spec.lh));
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
