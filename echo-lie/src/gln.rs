use nalgebra::DMatrix;

use crate::matfn;

/// GL(n) — General Linear Group (invertible n×n matrices).
///
/// Lie algebra dimension: n².
#[derive(Debug, Clone)]
pub struct GLn {
    pub n: usize,
    pub matrix: DMatrix<f64>,
}

impl GLn {
    #[inline]
    pub fn cdim(n: usize) -> usize {
        n * n
    }
}

// -- Constructors --
impl GLn {
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            matrix: DMatrix::identity(n, n),
        }
    }

    pub fn from_matrix(n: usize, m: &DMatrix<f64>) -> Self {
        Self {
            n,
            matrix: m.clone(),
        }
    }

    pub fn random(n: usize) -> Self {
        loop {
            let m = DMatrix::from_fn(n, n, |_, _| rand::random::<f64>() - 0.5);
            let d = m.determinant();
            if d.abs() > 1e-10 {
                let scale = d.abs().powf(1.0 / n as f64) * d.signum();
                return Self {
                    n,
                    matrix: m / scale,
                };
            }
        }
    }
}

// -- Lie algebra maps --
impl GLn {
    /// wedge: R^{n²} → n×n matrix (row-major).
    pub fn wedge(n: usize, u: &nalgebra::DVector<f64>) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = u[n * i + j];
            }
        }
        m
    }

    /// vee: n×n matrix → R^{n²} (row-major).
    pub fn vee(n: usize, m: &DMatrix<f64>) -> nalgebra::DVector<f64> {
        let mut u = nalgebra::DVector::zeros(n * n);
        for i in 0..n {
            for j in 0..n {
                u[n * i + j] = m[(i, j)];
            }
        }
        u
    }

    /// Lie-algebra adjoint.
    pub fn adjoint_algebra(n: usize, u: &nalgebra::DVector<f64>) -> DMatrix<f64> {
        let cdim = Self::cdim(n);
        let u_wedge = Self::wedge(n, u);
        let mut ad = DMatrix::zeros(cdim, cdim);
        for i in 0..cdim {
            let mut ei = nalgebra::DVector::zeros(cdim);
            ei[i] = 1.0;
            let ei_wedge = Self::wedge(n, &ei);
            let bracket = &u_wedge * &ei_wedge - &ei_wedge * &u_wedge;
            ad.set_column(i, &Self::vee(n, &bracket));
        }
        ad
    }
}

// -- Exponential / Logarithm --
impl GLn {
    pub fn exp(n: usize, u: &nalgebra::DVector<f64>) -> Self {
        Self {
            n,
            matrix: matfn::expm(&Self::wedge(n, u)),
        }
    }

    pub fn log(&self) -> nalgebra::DVector<f64> {
        Self::vee(self.n, &matfn::logm(&self.matrix))
    }
}

// -- Group operations --
impl GLn {
    pub fn compose(&self, other: &GLn) -> GLn {
        assert_eq!(self.n, other.n);
        GLn {
            n: self.n,
            matrix: &self.matrix * &other.matrix,
        }
    }

    pub fn inverse(&self) -> GLn {
        GLn {
            n: self.n,
            matrix: self
                .matrix
                .clone()
                .try_inverse()
                .expect("GLn matrix must be invertible"),
        }
    }

    pub fn act(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        &self.matrix * point
    }

    pub fn act_inverse(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        self.matrix
            .clone()
            .lu()
            .solve(point)
            .expect("GLn must be invertible")
    }

    pub fn as_matrix(&self) -> DMatrix<f64> {
        self.matrix.clone()
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> DMatrix<f64> {
        let cdim = Self::cdim(self.n);
        let a_inv = self.matrix.clone().try_inverse().unwrap();
        let mut ad = DMatrix::zeros(cdim, cdim);
        for i in 0..cdim {
            let mut ei = nalgebra::DVector::zeros(cdim);
            ei[i] = 1.0;
            let col = Self::vee(
                self.n,
                &(&self.matrix * Self::wedge(self.n, &ei) * &a_inv),
            );
            ad.set_column(i, &col);
        }
        ad
    }
}

impl std::ops::Mul for GLn {
    type Output = GLn;
    fn mul(self, rhs: GLn) -> GLn {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&GLn> for &GLn {
    type Output = GLn;
    fn mul(self, rhs: &GLn) -> GLn {
        self.compose(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn inverse_gives_identity() {
        for _ in 0..20 {
            let x = GLn::random(5);
            let id = x.compose(&x.inverse());
            assert_abs_diff_eq!(id.matrix, DMatrix::identity(5, 5), epsilon = 1e-8);
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..20 {
            let cdim = GLn::cdim(3);
            let v = nalgebra::DVector::from_fn(cdim, |_, _| rand::random::<f64>() - 0.5);
            let x = GLn::exp(3, &v);
            let v2 = x.log();
            assert_abs_diff_eq!(v, v2, epsilon = 1e-6);
        }
    }
}
