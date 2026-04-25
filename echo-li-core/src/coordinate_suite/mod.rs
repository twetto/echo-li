use nalgebra::{Matrix3x2, Matrix2x3, Vector2, Vector3};

pub mod euclid;
pub mod invdepth;
pub mod normal;

pub fn base_skew(v: &Vector3<f64>) -> nalgebra::Matrix3<f64> {
    echo_lie::base::skew(v)
}

/// Stereographic projection: S^2 -> R^2.
/// Maps the sphere to a plane from the north pole (0,0,1).
pub fn e3_project_sphere(p: &Vector3<f64>) -> Vector2<f64> {
    let p_norm = p.normalize();
    let d = 1.0 - p_norm[2];
    if d.abs() < 1e-12 {
        Vector2::new(0.0, 0.0)
    } else {
        Vector2::new(p_norm[0] / d, p_norm[1] / d)
    }
}

/// Jacobian of stereographic projection: R^3 -> R^2.
pub fn e3_project_sphere_diff(p: &Vector3<f64>) -> Matrix2x3<f64> {
    let p_norm = p.normalize();
    let px = p_norm[0];
    let py = p_norm[1];
    let pz = p_norm[2];
    let d = 1.0 - pz;
    if d.abs() < 1e-12 {
        Matrix2x3::zeros()
    } else {
        let mut diff = Matrix2x3::zeros();
        diff[(0, 0)] = 1.0 / d;
        diff[(0, 2)] = px / (d * d);
        diff[(1, 1)] = 1.0 / d;
        diff[(1, 2)] = py / (d * d);
        diff
    }
}

/// Inverse stereographic projection: R^2 -> S^2.
pub fn e3_project_sphere_inv(y: &Vector2<f64>) -> Vector3<f64> {
    let sq_norm = y.norm_squared();
    let d = sq_norm + 1.0;
    Vector3::new(2.0 * y[0] / d, 2.0 * y[1] / d, (sq_norm - 1.0) / d)
}

/// Jacobian of inverse stereographic projection: R^2 -> R^3.
pub fn e3_project_sphere_inv_diff(y: &Vector2<f64>) -> Matrix3x2<f64> {
    let sq_norm = y.norm_squared();
    let d = sq_norm + 1.0;
    let mut diff = Matrix3x2::zeros();
    diff[(0, 0)] = d - 2.0 * y[0] * y[0];
    diff[(0, 1)] = -2.0 * y[0] * y[1];
    diff[(1, 0)] = -2.0 * y[0] * y[1];
    diff[(1, 1)] = d - 2.0 * y[1] * y[1];
    diff[(2, 0)] = 2.0 * y[0];
    diff[(2, 1)] = 2.0 * y[1];
    (2.0 / (d * d)) * diff
}
