//! Local tangent-bearing chart for Sparse3D wide-FOV landmarks.
//!
//! State `(eta1, eta2, rho)`: a 2D perturbation `eta` in an orthonormal tangent
//! basis `U` at the fixed anchor unit bearing `b0`, plus additive inverse range
//! `rho` along that bearing. The anchor-frame point is
//!
//! ```text
//! P = b(eta) / rho,   b(eta) = normalize(b0 + U eta).
//! ```
//!
//! This is the projection-model-agnostic analogue of the pinhole
//! `(alpha, beta, rho) = (X/Z, Y/Z, 1/Z)` chart: for a pinhole camera and a
//! central bearing `b0 = normalize(alpha, beta, 1)` the two represent the same
//! anchor point, but the tangent chart stays well-conditioned for any camera
//! model (fisheye, wide FoV). See `sparse3d_bearing_camera_plan.md`.

use nalgebra::{Matrix3, Matrix3x2, Vector2, Vector3};

/// Orthonormal tangent basis `U = [e1 e2]` at unit bearing `b0`
/// (`U^T U = I`, `U^T b0 = 0`). Matches `camera_geometry`'s seed convention so
/// charts line up with the shared camera's tangent frame.
pub fn tangent_basis(b0: &Vector3<f64>) -> Matrix3x2<f64> {
    let seed = if b0.z.abs() < 0.9 {
        Vector3::z()
    } else {
        Vector3::x()
    };
    let e1 = seed.cross(b0).normalize();
    let e2 = b0.cross(&e1);
    Matrix3x2::from_columns(&[e1, e2])
}

/// Unit bearing from the chart coordinate: `normalize(b0 + U eta)`.
pub fn bearing(b0: &Vector3<f64>, u: &Matrix3x2<f64>, eta: &Vector2<f64>) -> Vector3<f64> {
    (b0 + u * eta).normalize()
}

/// Anchor-frame point `P = b(eta) / rho` and its 3x3 Jacobian
/// `dP / d(eta1, eta2, rho)`.
///
/// With `v = b0 + U eta`, `n = |v|`, `b = v / n`:
/// `db/deta = (1/n)(I - b b^T) U`, `dP/deta = (1/rho) db/deta`,
/// `dP/drho = -b / rho^2`.
pub fn point_and_jacobian(
    b0: &Vector3<f64>,
    u: &Matrix3x2<f64>,
    eta: &Vector2<f64>,
    rho: f64,
) -> (Vector3<f64>, Matrix3<f64>) {
    let v = b0 + u * eta;
    let n = v.norm();
    let b = v / n;
    let p = b / rho;

    let db_deta = (Matrix3::identity() - b * b.transpose()) * u / n; // 3x2
    let dp_deta = db_deta / rho; // 3x2
    let dp_drho = -b / (rho * rho); // 3x1

    let mut j = Matrix3::zeros();
    j.fixed_columns_mut::<2>(0).copy_from(&dp_deta);
    j.set_column(2, &dp_drho);
    (p, j)
}

/// Two-view ray triangulation.
///
/// `b_anchor` is the unit bearing in the anchor frame; `b_other` is the unit
/// bearing in the other view. `(r, t)` map a point from the other frame into the
/// anchor frame: `p_anchor = r * p_other + t`. Returns the least-squares ranges
/// `(range_anchor, range_other)` along each bearing (the closest-approach
/// midpoint solution), or `None` if the rays are near-parallel.
///
/// Model: `range_anchor * b_anchor - range_other * (r * b_other) = t`.
pub fn two_ray_ranges(
    b_anchor: &Vector3<f64>,
    b_other: &Vector3<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> Option<(f64, f64)> {
    let b = b_anchor;
    let a = r * b_other; // other bearing expressed in the anchor frame
    let bb = b.dot(b);
    let ba = b.dot(&a);
    let aa = a.dot(&a);
    // det of [[bb, -ba], [ba, -aa]] is -(bb*aa - ba^2).
    let d = bb * aa - ba * ba;
    if d.abs() < 1e-12 {
        return None;
    }
    let bt = b.dot(t);
    let at = a.dot(t);
    let range_anchor = (aa * bt - ba * at) / d;
    let range_other = (ba * bt - bb * at) / d;
    Some((range_anchor, range_other))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2, Vector2};

    fn unit(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z).normalize()
    }

    #[test]
    fn tangent_basis_is_orthonormal_and_perpendicular() {
        for b0 in [
            unit(0.0, 0.0, 1.0),
            unit(0.3, -0.2, 1.0),
            unit(0.9, 0.1, 0.05), // near the z<0.9 branch switch
            unit(-0.4, 0.8, -0.3),
        ] {
            let u = tangent_basis(&b0);
            let g = u.transpose() * u;
            assert_relative_eq!(g, Matrix2::identity(), epsilon = 1e-12);
            let perp = u.transpose() * b0;
            assert_relative_eq!(perp.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn bearing_at_zero_is_b0() {
        let b0 = unit(0.2, -0.5, 1.0);
        let u = tangent_basis(&b0);
        let b = bearing(&b0, &u, &Vector2::zeros());
        assert_relative_eq!(b, b0, epsilon = 1e-12);
    }

    #[test]
    fn point_jacobian_matches_finite_difference() {
        let b0 = unit(0.3, -0.2, 1.0);
        let u = tangent_basis(&b0);
        let eta = Vector2::new(0.12, -0.07);
        let rho = 0.7;
        let (_, j) = point_and_jacobian(&b0, &u, &eta, rho);

        let eps = 1e-7;
        // eta columns
        for c in 0..2 {
            let mut ep = eta;
            let mut em = eta;
            ep[c] += eps;
            em[c] -= eps;
            let pp = point_and_jacobian(&b0, &u, &ep, rho).0;
            let pm = point_and_jacobian(&b0, &u, &em, rho).0;
            let num = (pp - pm) / (2.0 * eps);
            assert_relative_eq!(j.column(c).into_owned(), num, epsilon = 1e-5);
        }
        // rho column
        let pp = point_and_jacobian(&b0, &u, &eta, rho + eps).0;
        let pm = point_and_jacobian(&b0, &u, &eta, rho - eps).0;
        let num = (pp - pm) / (2.0 * eps);
        assert_relative_eq!(j.column(2).into_owned(), num, epsilon = 1e-5);
    }

    #[test]
    fn two_ray_recovers_known_point() {
        // Known point in the anchor frame.
        let p_anchor = Vector3::new(0.4, -0.3, 2.0);
        let b_anchor = p_anchor.normalize();

        // Other frame related by (r, t): p_anchor = r p_other + t.
        let r = SOT3_rot(0.05, -0.03, 0.02);
        let t = Vector3::new(0.15, 0.02, -0.05);
        let p_other = r.transpose() * (p_anchor - t);
        let b_other = p_other.normalize();

        let (range_anchor, range_other) = two_ray_ranges(&b_anchor, &b_other, &r, &t).unwrap();
        assert_relative_eq!(range_anchor, p_anchor.norm(), epsilon = 1e-9);
        assert_relative_eq!(range_other, p_other.norm(), epsilon = 1e-9);
        assert_relative_eq!(range_anchor * b_anchor, p_anchor, epsilon = 1e-9);
    }

    #[test]
    fn pinhole_point_matches_chart_roundtrip() {
        // A pinhole (alpha,beta,rho)=(X/Z,Y/Z,1/Z) point and the tangent-bearing
        // (b0=normalize(alpha,beta,1), rho=1/range, eta=0) chart represent the
        // same anchor point.
        let p = Vector3::new(0.35, -0.15, 2.5);
        let b0 = p.normalize();
        let u = tangent_basis(&b0);
        let rho = 1.0 / p.norm();
        let (p_chart, _) = point_and_jacobian(&b0, &u, &Vector2::zeros(), rho);
        assert_relative_eq!(p_chart, p, epsilon = 1e-12);
    }

    // Small helper: rotation from an axis-angle-ish (rx,ry,rz) via nalgebra.
    fn SOT3_rot(rx: f64, ry: f64, rz: f64) -> Matrix3<f64> {
        let axis = Vector3::new(rx, ry, rz);
        let angle = axis.norm();
        if angle < 1e-12 {
            return Matrix3::identity();
        }
        let k = axis / angle;
        let kx = crate::coordinate_suite::base_skew(&k);
        Matrix3::identity() + angle.sin() * kx + (1.0 - angle.cos()) * (kx * kx)
    }
}
