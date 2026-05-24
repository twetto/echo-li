use nalgebra::{SVector, Vector6};

use crate::base::LieGroup;
use crate::se3::SE3;
use crate::sen3::SE23;

/// Semi-direct bias group for inertial navigation bias symmetries.
///
/// This group stores `(D, delta)` with `D in SE_2(3)` and `delta in R6`.
/// `echo-lie::SE23` stores its two translation slots as `(position, velocity)`;
/// the bias action uses the velocity slot, matching the MSCEqF `B(D)` subgroup.
#[derive(Debug, Clone)]
pub struct SemiDirectBias {
    pub d: SE23,
    pub delta: Vector6<f64>,
}

impl SemiDirectBias {
    pub const CDIM: usize = 15;

    pub fn new(d: SE23, delta: Vector6<f64>) -> Self {
        Self { d, delta }
    }

    pub fn identity() -> Self {
        Self {
            d: SE23::identity(),
            delta: Vector6::zeros(),
        }
    }

    pub fn random() -> Self {
        Self {
            d: SE23::random(),
            delta: Vector6::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ),
        }
    }

    pub fn b(&self) -> SE3 {
        SE3::new(self.d.rotation.clone(), self.d.velocity)
    }

    pub fn c(&self) -> SE3 {
        SE3::new(self.d.rotation.clone(), self.d.position)
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            d: self.d.compose(&other.d),
            delta: self.delta + self.b().adjoint() * other.delta,
        }
    }

    pub fn inverse(&self) -> Self {
        let b_inv_ad = self
            .b()
            .adjoint()
            .try_inverse()
            .expect("SE3 adjoint should be invertible");
        Self {
            d: self.d.inverse(),
            delta: -(b_inv_ad * self.delta),
        }
    }

    pub fn exp(u: &SVector<f64, 15>) -> Self {
        let d_tangent = u.fixed_rows::<9>(0).into_owned();
        let delta_tangent = u.fixed_rows::<6>(9).into_owned();
        let b_tangent = Self::b_tangent_from_se23_tangent(&d_tangent);
        Self {
            d: SE23::exp(&d_tangent),
            delta: SE3::left_jacobian(&b_tangent) * delta_tangent,
        }
    }

    pub fn log(&self) -> SVector<f64, 15> {
        let d_tangent = self.d.log();
        let b_tangent = Self::b_tangent_from_se23_tangent(&d_tangent);
        let delta_tangent = SE3::inv_left_jacobian(&b_tangent) * self.delta;

        let mut u = SVector::<f64, 15>::zeros();
        u.fixed_rows_mut::<9>(0).copy_from(&d_tangent);
        u.fixed_rows_mut::<6>(9).copy_from(&delta_tangent);
        u
    }

    fn b_tangent_from_se23_tangent(u: &SVector<f64, 9>) -> Vector6<f64> {
        let mut b_tangent = Vector6::zeros();
        b_tangent
            .fixed_rows_mut::<3>(0)
            .copy_from(&u.fixed_rows::<3>(0));
        b_tangent
            .fixed_rows_mut::<3>(3)
            .copy_from(&u.fixed_rows::<3>(6));
        b_tangent
    }
}

impl std::ops::Mul for SemiDirectBias {
    type Output = SemiDirectBias;
    fn mul(self, rhs: SemiDirectBias) -> SemiDirectBias {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&SemiDirectBias> for &SemiDirectBias {
    type Output = SemiDirectBias;
    fn mul(self, rhs: &SemiDirectBias) -> SemiDirectBias {
        self.compose(rhs)
    }
}
