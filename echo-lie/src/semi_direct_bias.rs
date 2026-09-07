use nalgebra::{SMatrix, SVector, Vector6};

use crate::base::{LieGroup, skew};
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

    /// Lie-algebra adjoint `ad_u` of the SDB group, in the tangent order this
    /// crate's `exp`/`log` use: `[ D (se23 = att, pos, vel) (9) ; delta (bias) (6) ]`.
    ///
    /// Derived from the SDB bracket `[(A,a),(B,b)] = ([A,B]_se23,
    /// ad^se3_{β(A)} b − ad^se3_{β(B)} a)`, where `β(A) = [A_att; A_vel]` is the
    /// `B(D)=SE3(R,v)` sub-tangent that the bias factor is acted on by. Blocks:
    /// - `(D,D)` = `ad_se23(A)`;
    /// - `(δ,δ)` = `ad_se3(β(A))`;
    /// - `(δ,D)` = `ad_se3(a) · P_β` (antisymmetry: `−ad_{β(B)} a = +ad_a β(B)`),
    ///   `P_β` selecting the att/vel columns of the se23 tangent.
    ///
    /// This mirrors MSCEqF's SDB curvature blocks (symmetry.cpp:141-148) in the
    /// `Dd = SE23 ⋉ bias` group. Validated in `tests/test_groups.rs`: the
    /// diagonal blocks equal the tested component matrix adjoints
    /// (`SEn3::adjoint_algebra`, `SE3::adjoint_algebra`) exactly, and the full
    /// matrix — coupling block included — satisfies the Jacobi identity
    /// `ad_{[u,v]} = [ad_u, ad_v]` (an FD-free representation check; a group-chart
    /// finite difference is ill-conditioned because SO3 `log` near identity
    /// swamps the translation rows).
    pub fn adjoint_algebra(u: &SVector<f64, 15>) -> SMatrix<f64, 15, 15> {
        let att = u.fixed_rows::<3>(0).into_owned();
        let pos = u.fixed_rows::<3>(3).into_owned();
        let vel = u.fixed_rows::<3>(6).into_owned();
        let delta = u.fixed_rows::<6>(9).into_owned();

        let mut ad = SMatrix::<f64, 15, 15>::zeros();
        let att_sk = skew(&att);

        // (D,D) = ad_se23(A): [ω]× on each of the three 3-blocks' diagonal, and
        // [pos]×, [vel]× in the (pos,att) / (vel,att) off-diagonals.
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&att_sk);
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&att_sk);
        ad.fixed_view_mut::<3, 3>(6, 6).copy_from(&att_sk);
        ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&skew(&pos));
        ad.fixed_view_mut::<3, 3>(6, 0).copy_from(&skew(&vel));

        // β(A) = [att; vel] (the B(D) SE3 tangent).
        let mut beta = Vector6::zeros();
        beta.fixed_rows_mut::<3>(0).copy_from(&att);
        beta.fixed_rows_mut::<3>(3).copy_from(&vel);

        // (δ,δ) = ad_se3(β(A)).
        ad.fixed_view_mut::<6, 6>(9, 9)
            .copy_from(&SE3::adjoint_algebra(&beta));

        // (δ,D) = ad_se3(a) · P_β: the se3-adjoint of the bias tangent, its
        // ω-columns scattered onto the att block (D cols 0:3) and its v-columns
        // onto the vel block (D cols 6:9); the pos block (D cols 3:6) stays zero.
        let ad_delta = SE3::adjoint_algebra(&delta);
        ad.fixed_view_mut::<6, 3>(9, 0)
            .copy_from(&ad_delta.fixed_view::<6, 3>(0, 0));
        ad.fixed_view_mut::<6, 3>(9, 6)
            .copy_from(&ad_delta.fixed_view::<6, 3>(0, 3));

        ad
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
