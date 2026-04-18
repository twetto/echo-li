use nalgebra::{Matrix3, Vector3, Dim, DefaultAllocator};
use nalgebra::allocator::Allocator;

/// Convert 3-vector to 3x3 skew-symmetric matrix.
#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1], //
        v[2], 0.0, -v[0], //
        -v[1], v[0], 0.0,
    )
}

/// Convert 3x3 skew-symmetric matrix to 3-vector.
#[inline]
pub fn vex(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

/// Core trait for Lie Groups.
///
/// This trait provides a unified interface for groups used in VIO,
/// allowing the filter to be generic over the state representation.
pub trait LieGroup: Sized + Clone + std::fmt::Debug 
where 
    DefaultAllocator: Allocator<Self::D> + Allocator<Self::D, Self::D>
{
    /// Dimension of the Lie Algebra (tangent space).
    type D: Dim;
    
    /// The associated Lie Algebra type.
    type Tangent;
    
    /// The type for the Adjoint matrix.
    type Adjoint;

    /// Identity element of the group.
    fn identity() -> Self;
    
    /// Group inverse.
    fn inverse(&self) -> Self;
    
    /// Group composition: self * other.
    fn compose(&self, other: &Self) -> Self;
    
    /// Exponential map: Lie Algebra → Lie Group.
    fn exp(v: &Self::Tangent) -> Self;
    
    /// Logarithm map: Lie Group → Lie Algebra.
    fn log(&self) -> Self::Tangent;
    
    /// Adjoint representation Ad_X: Tangent → Tangent.
    fn adjoint(&self) -> Self::Adjoint;
    
    /// Group action on a 3D Euclidean point: X * p.
    fn act(&self, p: &Vector3<f64>) -> Vector3<f64>;
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn skew_vex_roundtrip() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let m = skew(&v);
        let v2 = vex(&m);
        assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
    }

    #[test]
    fn skew_is_antisymmetric() {
        let v = Vector3::new(0.5, -1.3, 2.7);
        let m = skew(&v);
        assert_abs_diff_eq!(m, -m.transpose(), epsilon = 1e-15);
    }
}
