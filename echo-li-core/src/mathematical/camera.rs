use nalgebra::{Matrix2x3, Vector2, Vector3};
use rudolf_v::camera::CameraIntrinsics as RudolfIntrinsics;

pub trait CameraModel: Send + Sync {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64>;
    fn project_ray(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        (p[2] > 1e-6).then(|| self.project(p))
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64>;
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64>;
}

impl CameraModel for RudolfIntrinsics {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        let (u, v) = self.project_point([p[0], p[1], p[2]]).unwrap_or((0.0, 0.0));
        Vector2::new(u, v)
    }

    fn project_ray(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p[2] <= 1e-6 {
            return None;
        }
        let z_inv = 1.0 / p[2];
        let x = p[0] * z_inv;
        let y = p[1] * z_inv;
        let (xd, yd) = self.distort_normalized(x, y);
        let (u, v) = self.denormalize(xd, yd);
        Some(Vector2::new(u, v))
    }

    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        let (x, y) = self.normalize_undistorted(uv[0], uv[1]);
        Vector3::new(x, y, 1.0).normalize()
    }

    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        let j = self.projection_jacobian([p[0], p[1], p[2]]);
        Matrix2x3::new(j[0][0], j[0][1], j[0][2], j[1][0], j[1][1], j[1][2])
    }
}

pub struct PinholeModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl CameraModel for PinholeModel {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        Vector2::new(
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        )
    }
    fn project_ray(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p[2] <= 1e-6 {
            return None;
        }
        Some(Vector2::new(
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        ))
    }
    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        Vector3::new(
            (uv[0] - self.cx) / self.fx,
            (uv[1] - self.cy) / self.fy,
            1.0,
        )
        .normalize()
    }
    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        let z_inv = 1.0 / p[2];
        let z_inv2 = z_inv * z_inv;
        Matrix2x3::new(
            self.fx * z_inv,
            0.0,
            -self.fx * p[0] * z_inv2,
            0.0,
            self.fy * z_inv,
            -self.fy * p[1] * z_inv2,
        )
    }
}

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
    fn distort_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx, y * radial + dy)
    }
}

impl CameraModel for RadTanModel {
    fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        let x = p[0] / p[2];
        let y = p[1] / p[2];
        let (xd, yd) = self.distort_normalized(x, y);
        Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy)
    }

    fn project_ray(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p[2] <= 1e-6 {
            return None;
        }
        let x = p[0] / p[2];
        let y = p[1] / p[2];
        let (xd, yd) = self.distort_normalized(x, y);
        Some(Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy))
    }

    fn undistort(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        let x0 = (uv[0] - self.cx) / self.fx;
        let y0 = (uv[1] - self.cy) / self.fy;
        let mut x = x0;
        let mut y = y0;
        for _ in 0..10 {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4;
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (x0 - dx) / radial;
            y = (y0 - dy) / radial;
        }
        Vector3::new(x, y, 1.0).normalize()
    }

    fn projection_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        let z_inv = 1.0 / p[2];
        let x = p[0] * z_inv;
        let y = p[1] * z_inv;
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4;
        let d_radial_dr2 = self.k1 + 2.0 * self.k2 * r2;

        let dxd_dx = radial + 2.0 * x * x * d_radial_dr2 + 2.0 * self.p1 * y + 6.0 * self.p2 * x;
        let dxd_dy = 2.0 * x * y * d_radial_dr2 + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
        let dyd_dx = 2.0 * x * y * d_radial_dr2 + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
        let dyd_dy = radial + 2.0 * y * y * d_radial_dr2 + 6.0 * self.p1 * y + 2.0 * self.p2 * x;

        Matrix2x3::new(
            self.fx * dxd_dx * z_inv,
            self.fx * dxd_dy * z_inv,
            self.fx * (-dxd_dx * x * z_inv - dxd_dy * y * z_inv),
            self.fy * dyd_dx * z_inv,
            self.fy * dyd_dy * z_inv,
            self.fy * (-dyd_dx * x * z_inv - dyd_dy * y * z_inv),
        )
    }
}
