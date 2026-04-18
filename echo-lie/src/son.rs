use nalgebra::DMatrix;

use crate::matfn;

/// SO(n) — Special Orthogonal Group in n dimensions.
///
/// Lie algebra dimension: n(n−1)/2.
#[derive(Debug, Clone)]
pub struct SOn {
    pub n: usize,
    pub matrix: DMatrix<f64>,
}

impl SOn {
    #[inline]
    pub fn cdim(n: usize) -> usize {
        n * (n - 1) / 2
    }
}

// -- Constructors --
impl SOn {
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            matrix: DMatrix::identity(n, n),
        }
    }

    /// Project a matrix to nearest orthogonal via SVD.
    pub fn from_matrix(n: usize, m: &DMatrix<f64>) -> Self {
        let svd = m.clone().svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        Self {
            n,
            matrix: u * vt,
        }
    }

    pub fn random(n: usize) -> Self {
        let m = DMatrix::from_fn(n, n, |_, _| rand::random::<f64>() - 0.5);
        Self::from_matrix(n, &m)
    }
}

// -- Lie algebra maps --
impl SOn {
    /// wedge: R^{n(n-1)/2} → n×n skew-symmetric.
    pub fn wedge(n: usize, v: &nalgebra::DVector<f64>) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(n, n);
        let cdim = Self::cdim(n);
        let mut i = 0usize;
        let mut j = 0usize;
        for k in 0..cdim {
            j += 1;
            if j >= n {
                i += 1;
                j = i + 1;
            }
            m[(i, j)] = v[k];
            m[(j, i)] = -v[k];
        }
        m
    }

    /// vee: n×n skew-symmetric → R^{n(n-1)/2}.
    pub fn vee(n: usize, m: &DMatrix<f64>) -> nalgebra::DVector<f64> {
        let cdim = Self::cdim(n);
        let mut v = nalgebra::DVector::zeros(cdim);
        let mut i = 0usize;
        let mut j = 0usize;
        for k in 0..cdim {
            j += 1;
            if j >= n {
                i += 1;
                j = i + 1;
            }
            v[k] = m[(i, j)];
        }
        v
    }

    /// Lie-algebra adjoint: ad_v(u) = [wedge(v), wedge(u)].
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
impl SOn {
    pub fn exp(n: usize, u: &nalgebra::DVector<f64>) -> Self {
        let m = matfn::expm(&Self::wedge(n, u));
        Self::from_matrix(n, &m)
    }

    pub fn log(&self) -> nalgebra::DVector<f64> {
        Self::vee(self.n, &matfn::logm(&self.matrix))
    }
}

// -- Group operations --
impl SOn {
    pub fn compose(&self, other: &SOn) -> SOn {
        assert_eq!(self.n, other.n);
        SOn {
            n: self.n,
            matrix: &self.matrix * &other.matrix,
        }
    }

    pub fn inverse(&self) -> SOn {
        SOn {
            n: self.n,
            matrix: self.matrix.transpose(),
        }
    }

    pub fn act(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        &self.matrix * point
    }

    pub fn act_inverse(&self, point: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        self.matrix.transpose() * point
    }

    pub fn as_matrix(&self) -> DMatrix<f64> {
        self.matrix.clone()
    }

    /// Lie-group adjoint Ad.
    pub fn adjoint(&self) -> DMatrix<f64> {
        let cdim = Self::cdim(self.n);
        let rt = self.matrix.transpose();
        let mut ad = DMatrix::zeros(cdim, cdim);
        for i in 0..cdim {
            let mut ei = nalgebra::DVector::zeros(cdim);
            ei[i] = 1.0;
            let col = Self::vee(
                self.n,
                &(&self.matrix * Self::wedge(self.n, &ei) * &rt),
            );
            ad.set_column(i, &col);
        }
        ad
    }
}

impl std::ops::Mul for SOn {
    type Output = SOn;
    fn mul(self, rhs: SOn) -> SOn {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SOn> for &SOn {
    type Output = SOn;
    fn mul(self, rhs: &SOn) -> SOn {
        self.compose(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn identity_matrix() {
        let id = SOn::identity(4);
        assert_abs_diff_eq!(id.matrix, DMatrix::identity(4, 4), epsilon = 1e-15);
    }

    #[test]
    fn inverse_gives_identity() {
        for _ in 0..20 {
            let x = SOn::random(4);
            let id = x.compose(&x.inverse());
            assert_abs_diff_eq!(id.matrix, DMatrix::identity(4, 4), epsilon = 1e-10);
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        for _ in 0..20 {
            let cdim = SOn::cdim(4);
            let v = nalgebra::DVector::from_fn(cdim, |_, _| 0.5 * (rand::random::<f64>() - 0.5));
            let x = SOn::exp(4, &v);
            let v2 = x.log();
            let x2 = SOn::exp(4, &v2);
            assert_abs_diff_eq!(x.matrix, x2.matrix, epsilon = 1e-6);
        }
    }
}
