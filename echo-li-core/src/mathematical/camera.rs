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
