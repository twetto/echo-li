use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3, SVector, U9};

use crate::base::{skew, vex, LieGroup};
use crate::se3::SE3;
use crate::so3::SO3;

/// SE23 — Special Euclidean Group with 2 translations (Extended Pose).
///
/// Specifically SO(3) × R³ × R³ (rotation, position, velocity).
/// Lie algebra dimension: 9.
#[derive(Debug, Clone)]
pub struct SE23 {
    pub rotation: SO3,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
}

impl SE23 {
    pub const CDIM: usize = 9;
}

impl SE23 {
    pub fn new(rotation: SO3, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            rotation,
            position,
            velocity,
        }
    }

    pub fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
        }
    }

    pub fn random() -> Self {
        Self {
            rotation: SO3::random(),
            position: Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ),
            velocity: Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ),
        }
    }
}

impl LieGroup for SE23 {
    type D = U9;
    type Tangent = SVector<f64, 9>;
    type Adjoint = SMatrix<f64, 9, 9>;

    fn identity() -> Self {
        Self::identity()
    }

    fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        Self {
            rotation: r_inv,
            position: r_inv.act(&-self.position),
            velocity: r_inv.act(&-self.velocity),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation.compose(&other.rotation),
            position: self.rotation.act(&other.position) + self.position,
            velocity: self.rotation.act(&other.velocity) + self.velocity,
        }
    }

    fn exp(u: &Self::Tangent) -> Self {
        let w = u.fixed_rows::<3>(0).into_owned();
        let v1 = u.fixed_rows::<3>(3).into_owned();
        let v2 = u.fixed_rows::<3>(6).into_owned();

        let th = w.norm();
        let (b, c) = if th.abs() > 1e-12 {
            let s = th.sin();
            let co = th.cos();
            ((1.0 - co) / (th * th), (1.0 - s / th) / (th * th))
        } else {
            (0.5, 1.0 / 6.0)
        };

        let w_skew = skew(&w);
        let w_skew2 = w_skew * w_skew;
        let v_mat = Matrix3::identity() + w_skew * b + w_skew2 * c;

        Self {
            rotation: SO3::exp(&w),
            position: v_mat * v1,
            velocity: v_mat * v2,
        }
    }

    fn log(&self) -> Self::Tangent {
        let w = self.rotation.log();
        let th = w.norm();
        let w_skew = skew(&w);

        let inv_v_mat = if th.abs() > 1e-12 {
            let a = th.sin() / th;
            let b = (1.0 - th.cos()) / (th * th);
            Matrix3::identity() - w_skew * 0.5 + w_skew * w_skew * (1.0 - a / (2.0 * b)) / (th * th)
        } else {
            Matrix3::identity() - w_skew * 0.5 + w_skew * w_skew * (1.0 / 12.0)
        };

        let v1 = inv_v_mat * self.position;
        let v2 = inv_v_mat * self.velocity;

        let mut u = SVector::<f64, 9>::zeros();
        u.fixed_rows_mut::<3>(0).copy_from(&w);
        u.fixed_rows_mut::<3>(3).copy_from(&v1);
        u.fixed_rows_mut::<3>(6).copy_from(&v2);
        u
    }

    fn adjoint(&self) -> Self::Adjoint {
        let mut ad = SMatrix::<f64, 9, 9>::zeros();
        let r = self.rotation.as_matrix();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);

        let p_skew = skew(&self.position);
        let v_skew = skew(&self.velocity);

        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(6, 6).copy_from(&r);

        ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r));
        ad.fixed_view_mut::<3, 3>(6, 0).copy_from(&(v_skew * r));
        ad
    }

    fn act(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.act(p) + self.position
    }
}

/// SEn(3) — Extended SE(3) with n translation components.
///
/// Represents rotation R ∈ SO(3) with n translation vectors x₁, …, xₙ ∈ R³.
/// Lie algebra dimension: 3 + 3n.
#[derive(Debug, Clone)]
pub struct SEn3 {
    pub n: usize,
    pub rotation: SO3,
    pub translations: Vec<Vector3<f64>>,
}

impl SEn3 {
    #[inline]
    pub fn cdim(&self) -> usize {
        3 + 3 * self.n
    }
}

// -- Constructors --
impl SEn3 {
    pub fn new(n: usize, rotation: SO3, translations: Vec<Vector3<f64>>) -> Self {
        assert_eq!(translations.len(), n);
        Self {
            n,
            rotation,
            translations,
        }
    }

    pub fn identity(n: usize) -> Self {
        Self {
            n,
            rotation: SO3::identity(),
            translations: vec![Vector3::zeros(); n],
        }
    }

    pub fn random(n: usize) -> Self {
        Self {
            n,
            rotation: SO3::random(),
            translations: (0..n)
                .map(|_| {
                    Vector3::new(
                        rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5,
                    )
                })
                .collect(),
        }
    }

    pub fn from_matrix(n: usize, m: &DMatrix<f64>) -> Self {
        let r = m.fixed_view::<3, 3>(0, 0).into_owned();
        let translations = (0..n)
            .map(|i| m.fixed_view::<3, 1>(0, 3 + i).into_owned())
            .collect();
        Self {
            n,
            rotation: SO3::from_matrix(&r),
            translations,
        }
    }
}

// -- Lie algebra maps --
impl SEn3 {
    /// wedge: R^{3+3n} → matrix in se_n(3).
    pub fn wedge(n: usize, u: &DVector<f64>) -> DMatrix<f64> {
        let dim = 3 + n;
        let mut m = DMatrix::zeros(dim, dim);
        let omega = Vector3::new(u[0], u[1], u[2]);
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&omega));
        for i in 0..n {
            let vi = Vector3::new(u[3 + 3 * i], u[3 + 3 * i + 1], u[3 + 3 * i + 2]);
            m.fixed_view_mut::<3, 1>(0, 3 + i).copy_from(&vi);
        }
        m
    }

    /// vee: matrix → R^{3+3n}.
    pub fn vee(n: usize, m: &DMatrix<f64>) -> DVector<f64> {
        let cdim = 3 + 3 * n;
        let mut u = DVector::zeros(cdim);
        let omega = vex(&m.fixed_view::<3, 3>(0, 0).into_owned());
        u[0] = omega[0];
        u[1] = omega[1];
        u[2] = omega[2];
        for i in 0..n {
            let vi = m.fixed_view::<3, 1>(0, 3 + i).into_owned();
            u[3 + 3 * i] = vi[0];
            u[3 + 3 * i + 1] = vi[1];
            u[3 + 3 * i + 2] = vi[2];
        }
        u
    }

    /// Lie-algebra adjoint.
    pub fn adjoint_algebra(n: usize, u: &DVector<f64>) -> DMatrix<f64> {
        let dim = 3 + 3 * n;
        let mut ad = DMatrix::zeros(dim, dim);
        let omega = Vector3::new(u[0], u[1], u[2]);
        let omega_skew = skew(&omega);
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&omega_skew);
        for i in 0..n {
            let vi = Vector3::new(u[3 + 3 * i], u[3 + 3 * i + 1], u[3 + 3 * i + 2]);
            ad.view_mut((3 + 3 * i, 3 + 3 * i), (3, 3))
                .copy_from(&omega_skew);
            ad.view_mut((3 + 3 * i, 0), (3, 3)).copy_from(&skew(&vi));
        }
        ad
    }
}

// -- Exponential / Logarithm --
impl SEn3 {
    /// Exponential map se_n(3) → SE_n(3).
    pub fn exp(n: usize, u: &DVector<f64>) -> Self {
        let w = Vector3::new(u[0], u[1], u[2]);
        let th = w.norm();

        let (a, b, c) = if th.abs() >= 1e-12 {
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

        let translations = (0..n)
            .map(|i| {
                let vi = Vector3::new(u[3 + 3 * i], u[3 + 3 * i + 1], u[3 + 3 * i + 2]);
                v_mat * vi
            })
            .collect();

        Self {
            n,
            rotation: SO3::from_matrix(&rot_m),
            translations,
        }
    }

    /// Logarithm map SE_n(3) → se_n(3).
    pub fn log(&self) -> DVector<f64> {
        let omega = self.rotation.log();
        let omega_skew = skew(&omega);
        let theta = omega.norm();

        let coeff = if theta.abs() > 1e-8 {
            1.0 / (theta * theta)
                * (1.0 - (theta * theta.sin()) / (2.0 * (1.0 - theta.cos())))
        } else {
            1.0 / 12.0
        };

        let v_inv =
            Matrix3::identity() - 0.5 * omega_skew + coeff * omega_skew * omega_skew;

        let cdim = 3 + 3 * self.n;
        let mut u = DVector::zeros(cdim);
        u[0] = omega[0];
        u[1] = omega[1];
        u[2] = omega[2];
        for i in 0..self.n {
            let vi = v_inv * self.translations[i];
            u[3 + 3 * i] = vi[0];
            u[3 + 3 * i + 1] = vi[1];
            u[3 + 3 * i + 2] = vi[2];
        }
        u
    }

    /// Left Jacobian of SE_n(3).
    pub fn left_jacobian(n: usize, u: &DVector<f64>) -> DMatrix<f64> {
        let w = Vector3::new(u[0], u[1], u[2]);
        let ang = w.norm();
        let dim = 3 + 3 * n;

        if ang < 1e-6 {
            return DMatrix::identity(dim, dim) + 0.5 * Self::adjoint_algebra(n, u);
        }

        let jl_so3 = SO3::left_jacobian(&w);
        let mut j = DMatrix::zeros(dim, dim);
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(&jl_so3);
        for i in 0..n {
            let vi = Vector3::new(u[3 + 3 * i], u[3 + 3 * i + 1], u[3 + 3 * i + 2]);
            j.view_mut((3 + 3 * i, 0), (3, 3))
                .copy_from(&SE3::left_jacobian_q(&w, &vi));
            j.view_mut((3 + 3 * i, 3 + 3 * i), (3, 3))
                .copy_from(&jl_so3);
        }
        j
    }

    /// Right Jacobian of SE_n(3).
    pub fn right_jacobian(n: usize, u: &DVector<f64>) -> DMatrix<f64> {
        Self::left_jacobian(n, &-u)
    }
}

// -- Group operations --
impl SEn3 {
    pub fn compose(&self, other: &SEn3) -> SEn3 {
        assert_eq!(self.n, other.n);
        let translations = (0..self.n)
            .map(|i| self.rotation.act(&other.translations[i]) + self.translations[i])
            .collect();
        SEn3 {
            n: self.n,
            rotation: self.rotation.compose(&other.rotation),
            translations,
        }
    }

    pub fn inverse(&self) -> SEn3 {
        let r_inv = self.rotation.inverse();
        let translations = self
            .translations
            .iter()
            .map(|xi| r_inv.act(&-xi))
            .collect();
        SEn3 {
            n: self.n,
            rotation: r_inv,
            translations,
        }
    }

    pub fn as_matrix(&self) -> DMatrix<f64> {
        let dim = 3 + self.n;
        let mut m = DMatrix::identity(dim, dim);
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.as_matrix());
        for i in 0..self.n {
            m.fixed_view_mut::<3, 1>(0, 3 + i)
                .copy_from(&self.translations[i]);
        }
        m
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> DMatrix<f64> {
        let dim = 3 + 3 * self.n;
        let mut ad = DMatrix::zeros(dim, dim);
        let r = self.rotation.as_matrix();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        for i in 0..self.n {
            let tx_r = skew(&self.translations[i]) * r;
            ad.view_mut((3 + 3 * i, 0), (3, 3)).copy_from(&tx_r);
            ad.view_mut((3 + 3 * i, 3 + 3 * i), (3, 3)).copy_from(&r);
        }
        ad
    }
}

impl std::ops::Mul for SEn3 {
    type Output = SEn3;
    fn mul(self, rhs: SEn3) -> SEn3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SEn3> for &SEn3 {
    type Output = SEn3;
    fn mul(self, rhs: &SEn3) -> SEn3 {
        self.compose(rhs)
    }
}

/// Convenience alias: SE₂(3).
pub fn se23_identity() -> SEn3 {
    SEn3::identity(2)
}

pub fn se23_random() -> SEn3 {
    SEn3::random(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn se23_exp_log_roundtrip() {
        for _ in 0..100 {
            let mut u = DVector::from_fn(9, |_, _| rand::random::<f64>() - 0.5);
            // Smaller rotation
            u[0] *= 0.5;
            u[1] *= 0.5;
            u[2] *= 0.5;
            let p = SEn3::exp(2, &u);
            let u2 = p.log();
            let p2 = SEn3::exp(2, &u2);
            assert_abs_diff_eq!(p.as_matrix(), p2.as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn se23_inverse_gives_identity() {
        for _ in 0..100 {
            let p = SEn3::random(2);
            let id = p.compose(&p.inverse());
            assert_abs_diff_eq!(
                id.as_matrix(),
                DMatrix::identity(5, 5),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn se23_associativity() {
        for _ in 0..100 {
            let a = SEn3::random(2);
            let b = SEn3::random(2);
            let c = SEn3::random(2);
            let ab_c = a.compose(&b).compose(&c);
            let a_bc = a.compose(&b.compose(&c));
            assert_abs_diff_eq!(ab_c.as_matrix(), a_bc.as_matrix(), epsilon = 1e-10);
        }
    }
}
