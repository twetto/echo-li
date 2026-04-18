use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6, U6};

use crate::base::{skew, vex, LieGroup};
use crate::so3::SO3;

/// SE(3) — Special Euclidean Group in 3D (rotation + translation).
#[derive(Debug, Clone)]
pub struct SE3 {
    pub rotation: SO3,
    pub translation: Vector3<f64>,
}

impl SE3 {
    pub const CDIM: usize = 6;
}

// -- Constructors --
impl SE3 {
    pub fn new(rotation: SO3, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn from_matrix(m: &Matrix4<f64>) -> Self {
        let r = m.fixed_view::<3, 3>(0, 0).into_owned();
        let t = m.fixed_view::<3, 1>(0, 3).into_owned();
        Self {
            rotation: SO3::from_matrix(&r),
            translation: t,
        }
    }

    pub fn random() -> Self {
        Self {
            rotation: SO3::random(),
            translation: Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ),
        }
    }
}

// -- Lie algebra maps --
impl SE3 {
    /// wedge: R⁶ → se(3).  u = [ω; v].
    pub fn wedge(u: &Vector6<f64>) -> Matrix4<f64> {
        let mut m = Matrix4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&skew(&u.fixed_rows::<3>(0).into_owned()));
        m.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&u.fixed_rows::<3>(3));
        m
    }

    /// vee: se(3) → R⁶.
    pub fn vee(m: &Matrix4<f64>) -> Vector6<f64> {
        let omega = vex(&m.fixed_view::<3, 3>(0, 0).into_owned());
        let v = m.fixed_view::<3, 1>(0, 3).into_owned();
        Vector6::new(omega[0], omega[1], omega[2], v[0], v[1], v[2])
    }

    /// Lie-algebra adjoint ad_u.
    pub fn adjoint_algebra(u: &Vector6<f64>) -> Matrix6<f64> {
        let omega = u.fixed_rows::<3>(0).into_owned();
        let v = u.fixed_rows::<3>(3).into_owned();
        let omega_skew = skew(&omega);
        let v_skew = skew(&v);
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&omega_skew);
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&omega_skew);
        ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&v_skew);
        ad
    }
}

// -- LieGroup Trait Implementation --
impl LieGroup for SE3 {
    type D = U6;
    type Tangent = Vector6<f64>;
    type Adjoint = Matrix6<f64>;

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
impl SE3 {
    /// Exponential map se(3) → SE(3). u = [ω; v].
    pub fn exp(u: &Vector6<f64>) -> Self {
        let w = u.fixed_rows::<3>(0).into_owned();
        let v = u.fixed_rows::<3>(3).into_owned();
        let th = w.norm();

        let (a, b, c) = if th.abs() > 1e-12 {
            let s = th.sin();
            let co = th.cos();
            (s / th, (1.0 - co) / (th * th), (1.0 - s / th) / (th * th))
        } else {
            (1.0, 0.5, 1.0 / 6.0)
        };

        let wx = skew(&w);
        let wx2 = wx * wx;
        let rot_m = Matrix3::identity() + a * wx + b * wx2;
        let v_mat = Matrix3::identity() + b * wx + c * wx2;

        Self {
            rotation: SO3::from_matrix(&rot_m),
            translation: v_mat * v,
        }
    }

    /// Logarithm map SE(3) → se(3).
    pub fn log(&self) -> Vector6<f64> {
        let omega = self.rotation.log();
        let omega_skew = skew(&omega);
        let theta = omega.norm();

        let coeff = if theta.abs() > 1e-6 {
            1.0 / (theta * theta) * (1.0 - (theta * theta.sin()) / (2.0 * (1.0 - theta.cos())))
        } else {
            1.0 / 12.0
        };

        let v_inv = Matrix3::identity() - 0.5 * omega_skew + coeff * omega_skew * omega_skew;
        let v = v_inv * self.translation;

        Vector6::new(omega[0], omega[1], omega[2], v[0], v[1], v[2])
    }

    /// Q matrix for left Jacobian computation.
    pub fn left_jacobian_q(w: &Vector3<f64>, v: &Vector3<f64>) -> Matrix3<f64> {
        let p = skew(w);
        let r = skew(v);
        let ang = w.norm();
        let s = ang.sin();
        let c = ang.cos();

        let ang_p2 = ang * ang;
        let ang_p3 = ang_p2 * ang;
        let ang_p4 = ang_p3 * ang;
        let ang_p5 = ang_p4 * ang;

        let c1 = (ang - s) / ang_p3;
        let c2 = (0.5 * ang_p2 + c - 1.0) / ang_p4;
        let c3 = (ang * (1.0 + 0.5 * c) - 1.5 * s) / ang_p5;

        let m1 = p * r + r * p + p * r * p;
        let m2 = p * p * r + r * p * p - 3.0 * p * r * p;
        let m3 = p * r * p * p + p * p * r * p;

        0.5 * r + c1 * m1 + c2 * m2 + c3 * m3
    }

    /// Left Jacobian of SE(3).
    pub fn left_jacobian(u: &Vector6<f64>) -> Matrix6<f64> {
        let w = u.fixed_rows::<3>(0).into_owned();
        let v = u.fixed_rows::<3>(3).into_owned();
        let ang = w.norm();

        if ang < 1e-6 {
            return Matrix6::identity() + 0.5 * Self::adjoint_algebra(u);
        }

        let jl_so3 = SO3::left_jacobian(&w);
        let mut j = Matrix6::zeros();
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(&jl_so3);
        j.fixed_view_mut::<3, 3>(3, 3).copy_from(&jl_so3);
        j.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&Self::left_jacobian_q(&w, &v));
        j
    }

    /// Right Jacobian of SE(3).
    pub fn right_jacobian(u: &Vector6<f64>) -> Matrix6<f64> {
        Self::left_jacobian(&-u)
    }
}

// -- Group operations --
impl SE3 {
    pub fn compose(&self, other: &SE3) -> SE3 {
        SE3 {
            rotation: self.rotation.compose(&other.rotation),
            translation: self.translation + self.rotation.act(&other.translation),
        }
    }

    pub fn inverse(&self) -> SE3 {
        let r_inv = self.rotation.inverse();
        SE3 {
            translation: r_inv.act(&(-self.translation)),
            rotation: r_inv,
        }
    }

    /// Action on a point: R · p + t.
    #[inline]
    pub fn act(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.act(point) + self.translation
    }

    pub fn as_matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.as_matrix());
        m.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        m
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> Matrix6<f64> {
        let r = self.rotation.as_matrix();
        let tx_r = skew(&self.translation) * r;
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&tx_r);
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad
    }
}

impl std::ops::Mul for SE3 {
    type Output = SE3;
    fn mul(self, rhs: SE3) -> SE3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SE3> for &SE3 {
    type Output = SE3;
    fn mul(self, rhs: &SE3) -> SE3 {
        self.compose(rhs)
    }
}

impl std::ops::Mul<Vector3<f64>> for &SE3 {
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
    fn identity_is_neutral() {
        let id = SE3::identity();
        assert_abs_diff_eq!(id.as_matrix(), Matrix4::identity(), epsilon = 1e-15);
    }

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..100 {
            let mut u = Vector6::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            );
            u.fixed_rows_mut::<3>(0).scale_mut(0.5); // keep rotation small
            let p = SE3::exp(&u);
            let u2 = p.log();
            let p2 = SE3::exp(&u2);
            assert_abs_diff_eq!(p.as_matrix(), p2.as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_gives_identity() {
        for _ in 0..100 {
            let p = SE3::random();
            let id = p.compose(&p.inverse());
            assert_abs_diff_eq!(id.as_matrix(), Matrix4::identity(), epsilon = 1e-10);
        }
    }
}
