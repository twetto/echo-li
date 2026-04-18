use nalgebra::{Matrix3, UnitQuaternion, Quaternion, Vector3, Vector4, U3};
use rand::Rng;

use crate::base::{skew, vex, LieGroup};

/// SO(3) — Special Orthogonal Group in 3D (rotations).
///
/// Internal representation: `nalgebra::UnitQuaternion<f64>` with [x, y, z, w] convention.
#[derive(Debug, Clone, Copy)]
pub struct SO3 {
    pub q: UnitQuaternion<f64>,
}

// -- Lie algebra dimension --
impl SO3 {
    pub const CDIM: usize = 3;
}

// -- Constructors --
impl SO3 {
    /// Identity rotation.
    #[inline]
    pub fn identity() -> Self {
        Self {
            q: UnitQuaternion::identity(),
        }
    }

    /// From a unit quaternion (assumed already normalized).
    #[inline]
    pub fn from_quaternion(q: UnitQuaternion<f64>) -> Self {
        Self { q }
    }

    /// From raw quaternion components `[x, y, z, w]`, will be normalized.
    pub fn from_xyzw(x: f64, y: f64, z: f64, w: f64) -> Self {
        let q = Quaternion::new(w, x, y, z); // nalgebra stores as [w, x, y, z]
        Self {
            q: UnitQuaternion::new_normalize(q),
        }
    }

    /// From a 3×3 rotation matrix.
    pub fn from_matrix(m: &Matrix3<f64>) -> Self {
        Self {
            q: UnitQuaternion::from_matrix(m),
        }
    }

    /// Uniform random rotation (Haar measure) via Shoemake's method.
    pub fn random() -> Self {
        let mut rng = rand::rng();
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let u3: f64 = rng.random();
        let s1 = (1.0 - u1).sqrt();
        let s2 = u1.sqrt();
        let a1 = std::f64::consts::TAU * u2;
        let a2 = std::f64::consts::TAU * u3;
        let q = Quaternion::new(s2 * a2.cos(), s1 * a1.sin(), s1 * a1.cos(), s2 * a2.sin());
        Self {
            q: UnitQuaternion::new_normalize(q),
        }
    }

    /// Rotation that maps `origin` direction to `dest` direction.
    pub fn from_vectors(origin: &Vector3<f64>, dest: &Vector3<f64>) -> Self {
        let o = origin.normalize();
        let d = dest.normalize();
        let cos_a = o.dot(&d).clamp(-1.0, 1.0);

        if cos_a > 1.0 - 1e-10 {
            return Self::identity();
        }

        if cos_a < -1.0 + 1e-10 {
            // 180° rotation around any perpendicular axis
            let perp = if o[0].abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let axis = o.cross(&perp).normalize();
            let q = Quaternion::new(0.0, axis[0], axis[1], axis[2]);
            return Self {
                q: UnitQuaternion::new_normalize(q),
            };
        }

        let c = o.cross(&d);
        let q = Quaternion::new(1.0 + cos_a, c[0], c[1], c[2]);
        Self {
            q: UnitQuaternion::new_normalize(q),
        }
    }
}

// -- Lie algebra maps --
impl SO3 {
    /// wedge: R³ → so(3) (skew-symmetric matrix).
    #[inline]
    pub fn wedge(v: &Vector3<f64>) -> Matrix3<f64> {
        skew(v)
    }

    /// vee: so(3) → R³.
    #[inline]
    pub fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
        vex(m)
    }

    /// Lie-algebra adjoint ad_ω = [ω]×.
    #[inline]
    pub fn adjoint_algebra(omega: &Vector3<f64>) -> Matrix3<f64> {
        skew(omega)
    }
}

// -- LieGroup Trait Implementation --
impl LieGroup for SO3 {
    type D = U3;
    type Tangent = Vector3<f64>;
    type Adjoint = Matrix3<f64>;

    #[inline]
    fn identity() -> Self {
        Self::identity()
    }

    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
    }

    #[inline]
    fn compose(&self, other: &Self) -> Self {
        self.compose(other)
    }

    #[inline]
    fn exp(v: &Self::Tangent) -> Self {
        Self::exp(v)
    }

    #[inline]
    fn log(&self) -> Self::Tangent {
        self.log()
    }

    #[inline]
    fn adjoint(&self) -> Self::Adjoint {
        self.adjoint()
    }

    #[inline]
    fn act(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.act(p)
    }
}

// -- Exponential / Logarithm --
impl SO3 {
    /// Exponential map so(3) → SO(3).
    ///
    /// Converts rotation vector to quaternion:
    /// `q = [sin(θ/2) · axis, cos(θ/2)]`
    pub fn exp(w: &Vector3<f64>) -> Self {
        let theta_sq = w.norm_squared();
        if theta_sq < 1e-20 {
            let q = Quaternion::new(1.0, w[0] * 0.5, w[1] * 0.5, w[2] * 0.5);
            return Self {
                q: UnitQuaternion::new_normalize(q),
            };
        }
        let theta = theta_sq.sqrt();
        let half = theta * 0.5;
        let s = half.sin() / theta;
        let q = Quaternion::new(half.cos(), w[0] * s, w[1] * s, w[2] * s);
        Self {
            q: UnitQuaternion::new_normalize(q),
        }
    }

    /// Logarithm map SO(3) → so(3).
    ///
    /// `w = 2 · atan2(‖q_xyz‖, q_w) · q_xyz / ‖q_xyz‖`
    pub fn log(&self) -> Vector3<f64> {
        let q = self.q.as_ref();
        // nalgebra Quaternion: q.w, q.i, q.j, q.k
        let xyz = Vector3::new(q.i, q.j, q.k);
        let sin_half = xyz.norm();

        if sin_half < 1e-10 {
            xyz * 2.0
        } else {
            let coeff = 2.0 * sin_half.atan2(q.w) / sin_half;
            xyz * coeff
        }
    }

    /// Left Jacobian of SO(3).
    pub fn left_jacobian(w: &Vector3<f64>) -> Matrix3<f64> {
        let angle = w.norm();
        if angle < 1e-6 {
            return Matrix3::identity() + 0.5 * skew(w);
        }
        let ax = w / angle;
        let s = angle.sin() / angle;
        let c = angle.cos();

        s * Matrix3::identity() + ((1.0 - c) / angle) * skew(&ax) + (1.0 - s) * ax * ax.transpose()
    }

    /// Right Jacobian of SO(3).
    pub fn right_jacobian(w: &Vector3<f64>) -> Matrix3<f64> {
        Self::left_jacobian(&-w)
    }
}

// -- Group operations --
impl SO3 {
    /// Group multiplication.
    #[inline]
    pub fn compose(&self, other: &SO3) -> SO3 {
        SO3 {
            q: self.q * other.q,
        }
    }

    /// Inverse rotation.
    #[inline]
    pub fn inverse(&self) -> SO3 {
        SO3 {
            q: self.q.inverse(),
        }
    }

    /// Rotate a point: R · p.
    #[inline]
    pub fn act(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.q * point
    }

    /// Inverse-rotate a point: R^T · p.
    #[inline]
    pub fn act_inverse(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.q.inverse() * point
    }

    /// 3×3 rotation matrix.
    #[inline]
    pub fn as_matrix(&self) -> Matrix3<f64> {
        *self.q.to_rotation_matrix().matrix()
    }

    /// Quaternion as [x, y, z, w].
    #[inline]
    pub fn as_xyzw(&self) -> Vector4<f64> {
        let q = self.q.as_ref();
        Vector4::new(q.i, q.j, q.k, q.w)
    }

    /// Lie-group adjoint Ad_R = R (rotation matrix).
    #[inline]
    pub fn adjoint(&self) -> Matrix3<f64> {
        self.as_matrix()
    }
}

impl std::ops::Mul for SO3 {
    type Output = SO3;
    #[inline]
    fn mul(self, rhs: SO3) -> SO3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SO3> for SO3 {
    type Output = SO3;
    #[inline]
    fn mul(self, rhs: &SO3) -> SO3 {
        self.compose(rhs)
    }
}

impl std::ops::Mul<&SO3> for &SO3 {
    type Output = SO3;
    #[inline]
    fn mul(self, rhs: &SO3) -> SO3 {
        self.compose(rhs)
    }
}

impl std::ops::Mul<Vector3<f64>> for &SO3 {
    type Output = Vector3<f64>;
    #[inline]
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.act(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn identity() {
        let id = SO3::identity();
        assert_abs_diff_eq!(id.as_matrix(), Matrix3::identity(), epsilon = 1e-15);
    }

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..100 {
            let w = Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            );
            let r = SO3::exp(&w);
            let w2 = r.log();
            let r2 = SO3::exp(&w2);
            assert_abs_diff_eq!(r.as_matrix(), r2.as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_gives_identity() {
        for _ in 0..100 {
            let r = SO3::random();
            let id = r.compose(&r.inverse());
            assert_abs_diff_eq!(id.as_matrix(), Matrix3::identity(), epsilon = 1e-10);
        }
    }

    #[test]
    fn from_vectors_maps_correctly() {
        for _ in 0..100 {
            let v: Vector3<f64> = Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            )
            .normalize();
            let w: Vector3<f64> = Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            )
            .normalize();
            let r = SO3::from_vectors(&v, &w);
            let w2 = r.act(&v);
            assert_abs_diff_eq!(w, w2, epsilon = 1e-10);
        }
    }
}
