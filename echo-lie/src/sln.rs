use nalgebra::DMatrix;

use crate::matfn;

/// SL(n) — Special Linear Group (determinant = 1).
///
/// Lie algebra dimension: n² − 1.
#[derive(Debug, Clone)]
pub struct SLn {
    pub n: usize,
    pub matrix: DMatrix<f64>,
}

impl SLn {
    #[inline]
    pub fn cdim(n: usize) -> usize {
        n * n - 1
    }
}

// -- Constructors --
impl SLn {
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
                // Normalize to det = 1
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
impl SLn {
    /// wedge: R^{n²-1} → n×n traceless matrix.
    pub fn wedge(n: usize, u: &nalgebra::DVector<f64>) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(n, n);
        m[(n - 1, n - 1)] = 0.0;
        for i in 0..n {
            for j in 0..n {
                let idx = n * i + j;
                if idx < n * n - 1 {
                    m[(i, j)] = u[idx];
                    if i == j {
                        m[(n - 1, n - 1)] -= u[idx];
                    }
                }
            }
        }
        m
    }

    /// vee: n×n traceless matrix → R^{n²-1}.
    pub fn vee(n: usize, m: &DMatrix<f64>) -> nalgebra::DVector<f64> {
        let cdim = Self::cdim(n);
        let mut u = nalgebra::DVector::zeros(cdim);
        for i in 0..n {
            for j in 0..n {
                let idx = n * i + j;
                if idx < n * n - 1 {
                    u[idx] = m[(i, j)];
                }
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
impl SLn {
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
impl SLn {
    pub fn compose(&self, other: &SLn) -> SLn {
        assert_eq!(self.n, other.n);
        SLn {
            n: self.n,
            matrix: &self.matrix * &other.matrix,
        }
    }

    pub fn inverse(&self) -> SLn {
        SLn {
            n: self.n,
            matrix: self
                .matrix
                .clone()
                .try_inverse()
                .expect("SLn matrix must be invertible"),
        }
    }

    pub fn act(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        &self.matrix * point
    }

    pub fn act_inverse(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        self.matrix.clone().lu().solve(point).expect("SLn must be invertible")
    }

    pub fn as_matrix(&self) -> DMatrix<f64> {
        self.matrix.clone()
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> DMatrix<f64> {
        let cdim = Self::cdim(self.n);
        let h_inv = self.matrix.clone().try_inverse().unwrap();
        let mut ad = DMatrix::zeros(cdim, cdim);
        for i in 0..cdim {
            let mut ei = nalgebra::DVector::zeros(cdim);
            ei[i] = 1.0;
            let col = Self::vee(
                self.n,
                &(&self.matrix * Self::wedge(self.n, &ei) * &h_inv),
            );
            ad.set_column(i, &col);
        }
        ad
    }
}

impl std::ops::Mul for SLn {
    type Output = SLn;
    fn mul(self, rhs: SLn) -> SLn {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SLn> for &SLn {
    type Output = SLn;
    fn mul(self, rhs: &SLn) -> SLn {
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
            let x = SLn::random(3);
            let id = x.compose(&x.inverse());
            assert_abs_diff_eq!(id.matrix, DMatrix::identity(3, 3), epsilon = 1e-8);
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..20 {
            let cdim = SLn::cdim(3);
            let v = nalgebra::DVector::from_fn(cdim, |_, _| rand::random::<f64>() - 0.5);
            let x = SLn::exp(3, &v);
            let v2 = x.log();
            assert_abs_diff_eq!(v, v2, epsilon = 1e-6);
        }
    }
}
