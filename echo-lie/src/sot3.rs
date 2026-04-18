use nalgebra::{Matrix3, Matrix4, Vector3, Vector4, U4};

use crate::base::{skew, LieGroup};
use crate::so3::SO3;

/// SOT(3) — SO(3) × R⁺ (rotation with isotropic scale).
///
/// Lie algebra dimension: 4 = [ω(3), α(1)].
#[derive(Debug, Clone)]
pub struct SOT3 {
    pub rotation: SO3,
    /// Positive scale factor.
    pub scale: f64,
}

impl SOT3 {
    pub const CDIM: usize = 4;
}

// -- Constructors --
impl SOT3 {
    pub fn new(rotation: SO3, scale: f64) -> Self {
        Self { rotation, scale }
    }

    pub fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            scale: 1.0,
        }
    }

    pub fn random() -> Self {
        Self {
            rotation: SO3::random(),
            scale: (rand::random::<f64>() - 0.5).exp(),
        }
    }

    pub fn from_matrix(m: &Matrix4<f64>) -> Self {
        let a = m[(3, 3)];
        let mut r = m.fixed_view::<3, 3>(0, 0).into_owned();
        r /= a;
        Self {
            rotation: SO3::from_matrix(&r),
            scale: a,
        }
    }
}

// -- Lie algebra maps --
impl SOT3 {
    /// wedge: R⁴ → sot(3).  u = [ω; α].
    pub fn wedge(u: &Vector4<f64>) -> Matrix4<f64> {
        let mut m = Matrix4::zeros();
        let omega = Vector3::new(u[0], u[1], u[2]);
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&omega));
        m[(3, 3)] = u[3];
        m
    }

    /// vee: sot(3) → R⁴.
    pub fn vee(m: &Matrix4<f64>) -> Vector4<f64> {
        let omega = crate::base::vex(&m.fixed_view::<3, 3>(0, 0).into_owned());
        Vector4::new(omega[0], omega[1], omega[2], m[(3, 3)])
    }

    /// Lie-algebra adjoint ad_u.
    pub fn adjoint_algebra(u: &Vector4<f64>) -> Matrix4<f64> {
        let mut ad = Matrix4::zeros();
        let omega = Vector3::new(u[0], u[1], u[2]);
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&omega));
        ad
    }
}

// -- LieGroup Trait Implementation --
impl LieGroup for SOT3 {
    type D = U4;
    type Tangent = Vector4<f64>;
    type Adjoint = Matrix4<f64>;

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
impl SOT3 {
    /// Exponential map sot(3) → SOT(3). u = [ω; α].
    pub fn exp(u: &Vector4<f64>) -> Self {
        let w = Vector3::new(u[0], u[1], u[2]);
        Self {
            rotation: SO3::exp(&w),
            scale: u[3].exp(),
        }
    }

    /// Logarithm map SOT(3) → sot(3).
    pub fn log(&self) -> Vector4<f64> {
        let w = self.rotation.log();
        Vector4::new(w[0], w[1], w[2], self.scale.ln())
    }
}

// -- Group operations --
impl SOT3 {
    pub fn compose(&self, other: &SOT3) -> SOT3 {
        SOT3 {
            rotation: self.rotation.compose(&other.rotation),
            scale: self.scale * other.scale,
        }
    }

    pub fn inverse(&self) -> SOT3 {
        SOT3 {
            rotation: self.rotation.inverse(),
            scale: 1.0 / self.scale,
        }
    }

    /// Action on a point: a · (R · p).
    #[inline]
    pub fn act(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.scale * self.rotation.act(point)
    }

    /// Inverse action: (1/a) · (R^T · p).
    #[inline]
    pub fn act_inverse(&self, point: &Vector3<f64>) -> Vector3<f64> {
        (1.0 / self.scale) * self.rotation.act_inverse(point)
    }

    /// 4×4 matrix representation.
    pub fn as_matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.as_matrix());
        m[(3, 3)] = self.scale;
        m
    }

    /// 3×3 scaled rotation matrix: a · R.
    pub fn as_matrix3(&self) -> Matrix3<f64> {
        self.scale * self.rotation.as_matrix()
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> Matrix4<f64> {
        let mut ad = Matrix4::identity();
        ad.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.as_matrix());
        ad
    }
}

impl std::ops::Mul for SOT3 {
    type Output = SOT3;
    fn mul(self, rhs: SOT3) -> SOT3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SOT3> for &SOT3 {
    type Output = SOT3;
    fn mul(self, rhs: &SOT3) -> SOT3 {
        self.compose(rhs)
    }
}

impl std::ops::Mul<Vector3<f64>> for &SOT3 {
    type Output = Vector3<f64>;
    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.act(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..100 {
            let mut u = Vector4::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            );
            u[0] *= 0.5;
            u[1] *= 0.5;
            u[2] *= 0.5;
            let s = SOT3::exp(&u);
            let u2 = s.log();
            let s2 = SOT3::exp(&u2);
            assert_abs_diff_eq!(s.as_matrix(), s2.as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_gives_identity() {
        for _ in 0..100 {
            let s = SOT3::random();
            let id = s.compose(&s.inverse());
            assert_abs_diff_eq!(id.as_matrix(), Matrix4::identity(), epsilon = 1e-10);
        }
    }
}
