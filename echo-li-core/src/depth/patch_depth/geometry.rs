use super::*;

impl PatchDepthMapper {
    pub(super) fn warp_scaled_pixel(
        &self,
        u: f64,
        v: f64,
        rho: f64,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
        if rho <= 0.0 {
            return None;
        }
        let bearing = self.bearing_for_scaled_pixel(u, v, intr)?;
        let x_curr = bearing / rho;
        let x_ref = rel_pose.r * x_curr + rel_pose.t;
        if x_ref[2] <= 1e-6 {
            return None;
        }
        let (u_ref, v_ref) = self.project_scaled(&x_ref, intr.scale_from_original);
        Some((u_ref, v_ref, x_ref, bearing))
    }

    /// Warp a pixel at log-range η = ln(range).  All modes normalise the bearing to unit
    /// length before computing `x_curr = unit_b · exp(η)`, so the return bearing is unit.
    /// Returns `(u_ref, v_ref, x_ref, unit_b)`.
    pub(super) fn warp_with_eta(
        &self,
        u: f64,
        v: f64,
        eta: f64,
        intr: &ScaledIntrinsics,
        rel_pose: &RelativePose,
    ) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
        let bearing = self.bearing_for_scaled_pixel(u, v, intr)?;
        let unit_b = bearing.normalize();
        let x_curr = unit_b * eta.exp();
        let x_ref = rel_pose.r * x_curr + rel_pose.t;
        if x_ref[2] <= 1e-6 {
            return None;
        }
        let (u_ref, v_ref) = self.project_scaled(&x_ref, intr.scale_from_original);
        Some((u_ref, v_ref, x_ref, unit_b))
    }

    pub(super) fn bearing_for_scaled_pixel(
        &self,
        u: f64,
        v: f64,
        intr: &ScaledIntrinsics,
    ) -> Option<Vector3<f64>> {
        let original_u = u / intr.scale_from_original;
        let original_v = v / intr.scale_from_original;
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                self.bearing_at_original(original_u, original_v)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                if original_u < 0.0
                    || original_v < 0.0
                    || original_u >= (self.width - 1) as f64
                    || original_v >= (self.height - 1) as f64
                {
                    return None;
                }
                Some(Vector3::new(
                    (original_u - self.intrinsics.cx) / self.intrinsics.fx,
                    (original_v - self.intrinsics.cy) / self.intrinsics.fy,
                    1.0,
                ))
            }
        }
    }

    pub(super) fn project_scaled(&self, p: &Vector3<f64>, scale_from_original: f64) -> (f64, f64) {
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                let uv = self.camera.project(p);
                (uv[0] * scale_from_original, uv[1] * scale_from_original)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let z_inv = 1.0 / p[2];
                (
                    (self.intrinsics.fx * p[0] * z_inv + self.intrinsics.cx) * scale_from_original,
                    (self.intrinsics.fy * p[1] * z_inv + self.intrinsics.cy) * scale_from_original,
                )
            }
        }
    }

    pub(super) fn projection_jacobian(&self, p: &Vector3<f64>) -> nalgebra::Matrix2x3<f64> {
        match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                self.camera.projection_jacobian(p)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let z_inv = 1.0 / p[2];
                let z_inv2 = z_inv * z_inv;
                nalgebra::Matrix2x3::new(
                    self.intrinsics.fx * z_inv,
                    0.0,
                    -self.intrinsics.fx * p[0] * z_inv2,
                    0.0,
                    self.intrinsics.fy * z_inv,
                    -self.intrinsics.fy * p[1] * z_inv2,
                )
            }
        }
    }

    pub(super) fn bearing_at_original(&self, u: f64, v: f64) -> Option<Vector3<f64>> {
        if u < 0.0 || v < 0.0 || u >= (self.width - 1) as f64 || v >= (self.height - 1) as f64 {
            return None;
        }
        let ix = u as usize;
        let iy = v as usize;
        let dx = u - ix as f64;
        let dy = v - iy as f64;
        let b00 = self.bearing_lut[iy * self.width + ix];
        let b10 = self.bearing_lut[iy * self.width + ix + 1];
        let b01 = self.bearing_lut[(iy + 1) * self.width + ix];
        let b11 = self.bearing_lut[(iy + 1) * self.width + ix + 1];
        Some(
            b00 * ((1.0 - dx) * (1.0 - dy))
                + b10 * (dx * (1.0 - dy))
                + b01 * ((1.0 - dx) * dy)
                + b11 * (dx * dy),
        )
    }
}

#[allow(dead_code)]
pub(super) fn build_tiled_bearing_tile(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> TiledBearingTile {
    let center_u = x0 as f64 + 0.5 * (width.saturating_sub(1)) as f64;
    let center_v = y0 as f64 + 0.5 * (height.saturating_sub(1)) as f64;
    let raw_center_u = center_u / spec.level_scale;
    let raw_center_v = center_v / spec.level_scale;
    let center_bearing = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v))
        .normalize();
    let (tangent_u, tangent_v) = image_axis_tangent_basis(
        camera,
        &center_bearing,
        raw_center_u,
        raw_center_v,
        1.0 / spec.level_scale,
    );
    let focal = 0.5 * (intrinsics.fx + intrinsics.fy) * spec.level_scale;
    let lut = build_tiled_bearing_lut(
        camera,
        spec,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        focal,
        &center_bearing,
        &tangent_u,
        &tangent_v,
    );
    TiledBearingTile {
        level,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        center_bearing,
        tangent_u,
        tangent_v,
        focal,
        lut,
    }
}

pub(super) fn build_tiled_bearing_tile_geometry_from_center(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    center_bearing: Vector3<f64>,
) -> TiledBearingTile {
    let center_u = x0 as f64 + 0.5 * (width.saturating_sub(1)) as f64;
    let center_v = y0 as f64 + 0.5 * (height.saturating_sub(1)) as f64;
    let center_bearing = center_bearing.normalize();
    let (tangent_u, tangent_v) = projection_jacobian_tangent_basis(camera, &center_bearing);
    let focal = 0.5 * (intrinsics.fx + intrinsics.fy) * spec.level_scale;
    TiledBearingTile {
        level,
        x0,
        y0,
        width,
        height,
        center_u,
        center_v,
        center_bearing,
        tangent_u,
        tangent_v,
        focal,
        lut: UndistortLut {
            width,
            height,
            src_width: spec.lw,
            src_height: spec.lh,
            idx00: Vec::new(),
            idx10: Vec::new(),
            idx01: Vec::new(),
            idx11: Vec::new(),
            w00: Vec::new(),
            w10: Vec::new(),
            w01: Vec::new(),
            w11: Vec::new(),
            valid: Vec::new(),
        },
    }
}

#[inline(always)]
pub(super) fn sample_bilinear_raw_nomask_slice(
    data: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    u: f64,
    v: f64,
) -> Option<f32> {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return None;
    }
    let x = u as usize;
    let y = v as usize;
    let dx = (u - x as f64) as f32;
    let dy = (v - y as f64) as f32;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    unsafe {
        let r0 = data.as_ptr().add(y * stride);
        let r1 = data.as_ptr().add((y + 1) * stride);
        Some(w00 * *r0.add(x) + w10 * *r0.add(x + 1) + w01 * *r1.add(x) + w11 * *r1.add(x + 1))
    }
}

#[inline(always)]
pub(super) fn sample_bilinear_raw_nomask_with_grad_slice(
    img: &[f32],
    grad_x: &[f32],
    grad_y: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    u: f64,
    v: f64,
) -> Option<(f32, f32, f32)> {
    if u < 0.0 || v < 0.0 || u >= (width - 1) as f64 || v >= (height - 1) as f64 {
        return None;
    }
    let x = u as usize;
    let y = v as usize;
    let dx = (u - x as f64) as f32;
    let dy = (v - y as f64) as f32;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    unsafe {
        let i0 = img.as_ptr().add(y * stride);
        let i1 = img.as_ptr().add((y + 1) * stride);
        let gx0 = grad_x.as_ptr().add(y * stride);
        let gx1 = grad_x.as_ptr().add((y + 1) * stride);
        let gy0 = grad_y.as_ptr().add(y * stride);
        let gy1 = grad_y.as_ptr().add((y + 1) * stride);
        let sample = |r0: *const f32, r1: *const f32| {
            w00 * *r0.add(x) + w10 * *r0.add(x + 1) + w01 * *r1.add(x) + w11 * *r1.add(x + 1)
        };
        Some((sample(i0, i1), sample(gx0, gx1), sample(gy0, gy1)))
    }
}
#[allow(dead_code)]
pub(super) fn build_tiled_bearing_lut(
    camera: &dyn CameraModel,
    spec: &image_ops::UndistortLevelSpec,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    center_u: f64,
    center_v: f64,
    focal: f64,
    center_bearing: &Vector3<f64>,
    tangent_u: &Vector3<f64>,
    tangent_v: &Vector3<f64>,
) -> UndistortLut {
    let mut samples = Vec::with_capacity(width * height);
    for v in y0..y0 + height {
        for u in x0..x0 + width {
            let x = (u as f64 - center_u) / focal;
            let y = (v as f64 - center_v) / focal;
            let bearing = (center_bearing + tangent_u * x + tangent_v * y).normalize();
            let raw_uv = camera.project(&bearing);
            let raw_u = raw_uv[0] * spec.level_scale + spec.raw_offset;
            let raw_v = raw_uv[1] * spec.level_scale + spec.raw_offset;
            samples.push(image_ops::undistort_sample(raw_u, raw_v, spec.lw, spec.lh));
        }
    }
    UndistortLut::from_options_with_source(samples, width, height, spec.lw, spec.lh)
}

#[allow(dead_code)]
pub(super) fn image_axis_tangent_basis(
    camera: &dyn CameraModel,
    center_bearing: &Vector3<f64>,
    raw_center_u: f64,
    raw_center_v: f64,
    raw_step: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let b = center_bearing.normalize();
    let step = raw_step.max(1e-3);
    let bu_plus = camera
        .undistort(&Vector2::new(raw_center_u + step, raw_center_v))
        .normalize();
    let bu_minus = camera
        .undistort(&Vector2::new(raw_center_u - step, raw_center_v))
        .normalize();
    let bv_plus = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v + step))
        .normalize();
    let bv_minus = camera
        .undistort(&Vector2::new(raw_center_u, raw_center_v - step))
        .normalize();

    let du = project_to_tangent(&(bu_plus - bu_minus), &b);
    let mut tangent_u = normalize_or_fallback(du, image_axis_fallback_u(&b));
    tangent_u = project_to_tangent(&tangent_u, &b).normalize();

    let dv = project_to_tangent(&(bv_plus - bv_minus), &b);
    let dv_orthogonal = project_to_tangent(&(dv - tangent_u * dv.dot(&tangent_u)), &b);
    let mut tangent_v = normalize_or_fallback(dv_orthogonal, b.cross(&tangent_u));
    tangent_v = project_to_tangent(&tangent_v, &b).normalize();

    if tangent_v.dot(&dv) < 0.0 {
        tangent_v = -tangent_v;
    }
    (tangent_u, tangent_v)
}

pub(super) fn projection_jacobian_tangent_basis(
    camera: &dyn CameraModel,
    center_bearing: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let b = center_bearing.normalize();
    let j = camera.projection_jacobian(&b);
    let du = project_to_tangent(&Vector3::new(j[(0, 0)], j[(0, 1)], j[(0, 2)]), &b);
    let mut tangent_u = normalize_or_fallback(du, image_axis_fallback_u(&b));
    tangent_u = project_to_tangent(&tangent_u, &b).normalize();

    let dv = project_to_tangent(&Vector3::new(j[(1, 0)], j[(1, 1)], j[(1, 2)]), &b);
    let dv_orthogonal = project_to_tangent(&(dv - tangent_u * dv.dot(&tangent_u)), &b);
    let mut tangent_v = normalize_or_fallback(dv_orthogonal, b.cross(&tangent_u));
    tangent_v = project_to_tangent(&tangent_v, &b).normalize();

    if tangent_v.dot(&dv) < 0.0 {
        tangent_v = -tangent_v;
    }
    (tangent_u, tangent_v)
}

pub(super) fn project_to_tangent(v: &Vector3<f64>, b: &Vector3<f64>) -> Vector3<f64> {
    v - b * v.dot(b)
}

pub(super) fn normalize_or_fallback(v: Vector3<f64>, fallback: Vector3<f64>) -> Vector3<f64> {
    if v.norm_squared() > 1e-18 {
        v.normalize()
    } else {
        fallback.normalize()
    }
}

pub(super) fn image_axis_fallback_u(b: &Vector3<f64>) -> Vector3<f64> {
    let x_axis = Vector3::new(1.0, 0.0, 0.0);
    let projected = project_to_tangent(&x_axis, b);
    if projected.norm_squared() > 1e-18 {
        projected
    } else {
        project_to_tangent(&Vector3::new(0.0, 1.0, 0.0), b)
    }
}

pub(super) fn smoothstep01(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Tiled warp at log-range η.  Tile bearings are already unit, so
/// `x_curr = unit_b · exp(η)` directly.  Returns `(u_local_ref, v_local_ref, x_ref, unit_b)`.
pub(super) fn warp_tiled_local_pixel_eta(
    curr_tile: &TiledBearingTile,
    ref_tile: &TiledBearingTile,
    u_local: f64,
    v_local: f64,
    eta: f64,
    rel_pose: &RelativePose,
) -> Option<(f64, f64, Vector3<f64>, Vector3<f64>)> {
    let u = curr_tile.x0 as f64 + u_local;
    let v = curr_tile.y0 as f64 + v_local;
    let unit_b = curr_tile.bearing_at_level_pixel(u, v);
    let x_curr = unit_b * eta.exp();
    let x_ref = rel_pose.r * x_curr + rel_pose.t;
    let uv_ref = ref_tile.project_to_local_pixel(&x_ref)?;
    Some((uv_ref[0], uv_ref[1], x_ref, unit_b))
}
