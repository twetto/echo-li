use camera_geometry::{CameraModel as BearingCameraModel, CameraProjection, Pixel};
use nalgebra::{Matrix2x3, Vector2, Vector3};

/// ECHO-LI's ergonomic, infallible camera interface.
///
/// This is a thin adapter over `camera_geometry::CameraProjection`, the single
/// projection authority (pinhole / rad-tan / equidistant fisheye, and future
/// models such as double-sphere / EUCM as they are added upstream). The
/// infallible signatures are kept for the existing EqF / Sparse3D / patch-depth
/// call sites; degenerate projections (behind camera, invalid intrinsics) fall
/// back to a benign default instead of returning `Option`.
///
/// The projection taxonomy lives entirely in `camera_geometry` — ECHO-LI does
/// not re-encode model families here.
pub trait CameraModel: Send + Sync {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64>;
    fn project_ray(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        (p[2] > 1e-6).then(|| self.project(p))
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64>;
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64>;
}

// --- forwarding helpers over camera_geometry::CameraProjection ---------------

fn project_via(proj: &CameraProjection, p: &Vector3<f64>) -> Vector2<f64> {
    BearingCameraModel::project(proj, *p)
        .map(|px| px.0)
        .unwrap_or_else(Vector2::zeros)
}

fn undistort_via(proj: &CameraProjection, uv: &Vector2<f64>) -> Vector3<f64> {
    BearingCameraModel::unproject(proj, Pixel::new(uv[0], uv[1]))
        .map(|b| b.vector())
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0))
}

fn projection_jacobian_via(proj: &CameraProjection, p: &Vector3<f64>) -> Matrix2x3<f64> {
    BearingCameraModel::project_jacobian(proj, *p).unwrap_or_else(Matrix2x3::zeros)
}

/// Bridge the ECHO-LI infallible interface onto the calibrated projection model.
/// A `CameraProjection` is the production camera: pass `Arc<CameraProjection>` as
/// `Arc<dyn CameraModel>`.
impl CameraModel for CameraProjection {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        project_via(self, p)
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        undistort_via(self, uv)
    }
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        projection_jacobian_via(self, p)
    }
}

/// Convenience pinhole camera (fx, fy, cx, cy) for tests and simple call sites;
/// backed by the same `camera_geometry` authority.
pub struct PinholeModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl PinholeModel {
    pub fn projection(&self) -> CameraProjection {
        CameraProjection::pinhole([self.fx, self.fy, self.cx, self.cy], [0, 0])
    }
}

impl CameraModel for PinholeModel {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        project_via(&self.projection(), p)
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        undistort_via(&self.projection(), uv)
    }
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        projection_jacobian_via(&self.projection(), p)
    }
}

/// Convenience radial-tangential camera for tests and simple call sites; backed
/// by the same `camera_geometry` authority.
pub struct RadTanModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
}

impl RadTanModel {
    pub fn projection(&self) -> CameraProjection {
        CameraProjection::pinhole_radtan(
            [self.fx, self.fy, self.cx, self.cy],
            [self.k1, self.k2, self.p1, self.p2],
            [0, 0],
        )
    }
}

impl CameraModel for RadTanModel {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        project_via(&self.projection(), p)
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        undistort_via(&self.projection(), uv)
    }
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        projection_jacobian_via(&self.projection(), p)
    }
}

#[cfg(test)]
mod equivalence_tests {
    //! Prove the camera-geometry-backed path is numerically equivalent to the
    //! old `rudolf_v::camera::CameraIntrinsics` math on the pinhole/rad-tan path,
    //! so any EuRoC ATE change from the migration is attributable to the Rudolf-V
    //! frontend version bump, not this refactor.
    use super::*;
    use rudolf_v::camera::CameraIntrinsics as RudolfIntrinsics;

    fn euroc_rudolf() -> RudolfIntrinsics {
        RudolfIntrinsics {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            resolution: [752, 480],
            distortion: vec![-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
            model: rudolf_v::camera::DistortionModel::RadTan,
        }
    }

    fn euroc_projection() -> CameraProjection {
        CameraProjection::pinhole_radtan(
            [458.654, 457.296, 367.215, 248.375],
            [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
            [752, 480],
        )
    }

    // Replicate the OLD echo-li `impl CameraModel for RudolfIntrinsics`.
    fn old_undistort(cam: &RudolfIntrinsics, uv: &Vector2<f64>) -> Vector3<f64> {
        let (x, y) = cam.normalize_undistorted(uv[0], uv[1]);
        Vector3::new(x, y, 1.0).normalize()
    }
    fn old_project(cam: &RudolfIntrinsics, p: &Vector3<f64>) -> Vector2<f64> {
        let (u, v) = cam.project_point([p[0], p[1], p[2]]).unwrap();
        Vector2::new(u, v)
    }
    fn old_jac(cam: &RudolfIntrinsics, p: &Vector3<f64>) -> Matrix2x3<f64> {
        let j = cam.projection_jacobian([p[0], p[1], p[2]]);
        Matrix2x3::new(j[0][0], j[0][1], j[0][2], j[1][0], j[1][1], j[1][2])
    }

    #[test]
    fn radtan_camera_math_matches_old_rudolf() {
        let rud = euroc_rudolf();
        let proj = euroc_projection();

        let mut max_undist = 0.0f64;
        let mut max_proj = 0.0f64;
        let mut max_jac = 0.0f64;

        // Grid of pixels across the image.
        for py in (0..=480).step_by(48) {
            for px in (0..=752).step_by(47) {
                let uv = Vector2::new(px as f64, py as f64);
                let b_old = old_undistort(&rud, &uv);
                let b_new = CameraModel::undistort(&proj, &uv);
                max_undist = max_undist.max((b_old - b_new).amax());

                // Project a point along the (undistorted) bearing at a few depths.
                for depth in [0.5, 1.0, 3.0, 8.0] {
                    let p = b_new * depth;
                    if p[2] <= 1e-3 {
                        continue;
                    }
                    max_proj = max_proj
                        .max((old_project(&rud, &p) - CameraModel::project(&proj, &p)).amax());
                    max_jac = max_jac.max(
                        (old_jac(&rud, &p) - CameraModel::projection_jacobian(&proj, &p)).amax(),
                    );
                }
            }
        }

        println!("max diff: undistort={max_undist:.3e} project={max_proj:.3e} jac={max_jac:.3e}");
        assert!(max_undist < 1e-9, "undistort diff {max_undist:.3e}");
        assert!(max_proj < 1e-7, "project diff {max_proj:.3e}");
        assert!(max_jac < 1e-7, "jacobian diff {max_jac:.3e}");
    }
}
