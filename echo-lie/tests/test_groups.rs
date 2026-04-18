//! Integration tests mirroring liepp-python's test_groups.py.
//!
//! Tests group axioms for all Lie groups: associativity, identity, inverse,
//! wedge/vee, exp/log, algebra adjoint, group adjoint, matrix product.

use approx::assert_abs_diff_eq;
use echo_lie::*;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Matrix6, Vector3, Vector4, Vector6};

const N_TRIALS: usize = 100;

// =====================================================================
// SO(3)
// =====================================================================
mod so3_tests {
    use super::*;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let v = Vector3::new(rand::random(), rand::random(), rand::random());
            let m = SO3::wedge(&v);
            let v2 = SO3::vee(&m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let v = 0.5 * Vector3::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5);
            let x = SO3::exp(&v);
            let v2 = x.log();
            assert_abs_diff_eq!(x.as_matrix(), SO3::exp(&v2).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SO3::random();
            let b = SO3::random();
            let c = SO3::random();
            let ab_c = a.compose(&b).compose(&c);
            let a_bc = a.compose(&b.compose(&c));
            assert_abs_diff_eq!(ab_c.as_matrix(), a_bc.as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn identity_element() {
        for _ in 0..N_TRIALS {
            let x = SO3::random();
            let id = SO3::identity();
            assert_abs_diff_eq!(x.as_matrix(), x.compose(&id).as_matrix(), epsilon = 1e-10);
            assert_abs_diff_eq!(x.as_matrix(), id.compose(&x).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SO3::random();
            let xi = x.inverse();
            assert_abs_diff_eq!(x.compose(&xi).as_matrix(), Matrix3::identity(), epsilon = 1e-10);
            assert_abs_diff_eq!(xi.compose(&x).as_matrix(), Matrix3::identity(), epsilon = 1e-10);
        }
    }

    #[test]
    fn matrix_product() {
        for _ in 0..N_TRIALS {
            let a = SO3::random();
            let b = SO3::random();
            let z1 = a.as_matrix() * b.as_matrix();
            let z2 = a.compose(&b).as_matrix();
            assert_abs_diff_eq!(z1, z2, epsilon = 1e-10);
        }
    }

    #[test]
    fn matrix_inverse() {
        for _ in 0..N_TRIALS {
            let x = SO3::random();
            let xi1 = x.inverse().as_matrix();
            let xi2 = x.as_matrix().try_inverse().unwrap();
            assert_abs_diff_eq!(xi1, xi2, epsilon = 1e-10);
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SO3::random();
            let u = Vector3::new(rand::random(), rand::random(), rand::random());
            let ad_xu1 = SO3::wedge(&(x.adjoint() * u));
            let ad_xu2 = x.as_matrix() * SO3::wedge(&u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let v = Vector3::new(rand::random(), rand::random(), rand::random());
            let u = Vector3::new(rand::random(), rand::random(), rand::random());
            let ad_vu1 = SO3::wedge(&(SO3::adjoint_algebra(&v) * u));
            let ad_vu2 = SO3::wedge(&v) * SO3::wedge(&u) - SO3::wedge(&u) * SO3::wedge(&v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn left_jacobian() {
        for _ in 0..N_TRIALS {
            let u = Vector3::new(rand::random(), rand::random(), rand::random());
            let jl = SO3::left_jacobian(&u);
            let a = SO3::exp(&u).adjoint();
            let b = Matrix3::identity() + SO3::adjoint_algebra(&u) * jl;
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn from_vectors() {
        for _ in 0..N_TRIALS {
            let v: Vector3<f64> = Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ).normalize();
            let w: Vector3<f64> = Vector3::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            ).normalize();
            let r = SO3::from_vectors(&v, &w);
            assert_abs_diff_eq!(w, r.act(&v), epsilon = 1e-10);
        }
    }
}

// =====================================================================
// SE(3)
// =====================================================================
mod se3_tests {
    use super::*;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let v = Vector6::new(
                rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(),
            );
            let m = SE3::wedge(&v);
            let v2 = SE3::vee(&m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let mut v = Vector6::new(
                rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5,
            );
            v.fixed_rows_mut::<3>(0).scale_mut(0.5);
            let x = SE3::exp(&v);
            let v2 = x.log();
            assert_abs_diff_eq!(x.as_matrix(), SE3::exp(&v2).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SE3::random();
            let b = SE3::random();
            let c = SE3::random();
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn identity_element() {
        for _ in 0..N_TRIALS {
            let x = SE3::random();
            let id = SE3::identity();
            assert_abs_diff_eq!(x.as_matrix(), x.compose(&id).as_matrix(), epsilon = 1e-10);
            assert_abs_diff_eq!(x.as_matrix(), id.compose(&x).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SE3::random();
            assert_abs_diff_eq!(x.compose(&x.inverse()).as_matrix(), Matrix4::identity(), epsilon = 1e-10);
            assert_abs_diff_eq!(x.inverse().compose(&x).as_matrix(), Matrix4::identity(), epsilon = 1e-10);
        }
    }

    #[test]
    fn matrix_product() {
        for _ in 0..N_TRIALS {
            let a = SE3::random();
            let b = SE3::random();
            assert_abs_diff_eq!(
                a.as_matrix() * b.as_matrix(),
                a.compose(&b).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SE3::random();
            let u = Vector6::new(
                rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(),
            );
            let ad_xu1 = SE3::wedge(&(x.adjoint() * u));
            let ad_xu2 = x.as_matrix() * SE3::wedge(&u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let v = Vector6::new(
                rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(),
            );
            let u = Vector6::new(
                rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(),
            );
            let ad_vu1 = SE3::wedge(&(SE3::adjoint_algebra(&v) * u));
            let ad_vu2 = SE3::wedge(&v) * SE3::wedge(&u) - SE3::wedge(&u) * SE3::wedge(&v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn left_jacobian() {
        for _ in 0..N_TRIALS {
            let u = Vector6::new(
                rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(),
            );
            let jl = SE3::left_jacobian(&u);
            let a = SE3::exp(&u).adjoint();
            let b = Matrix6::identity() + SE3::adjoint_algebra(&u) * jl;
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }
}

// =====================================================================
// SOT(3)
// =====================================================================
mod sot3_tests {
    use super::*;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let v = Vector4::new(rand::random(), rand::random(), rand::random(), rand::random());
            let m = SOT3::wedge(&v);
            let v2 = SOT3::vee(&m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let mut v = Vector4::new(
                rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5,
            );
            v[0] *= 0.5;
            v[1] *= 0.5;
            v[2] *= 0.5;
            let x = SOT3::exp(&v);
            let v2 = x.log();
            assert_abs_diff_eq!(x.as_matrix(), SOT3::exp(&v2).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SOT3::random();
            let b = SOT3::random();
            let c = SOT3::random();
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn identity_element() {
        for _ in 0..N_TRIALS {
            let x = SOT3::random();
            let id = SOT3::identity();
            assert_abs_diff_eq!(x.as_matrix(), x.compose(&id).as_matrix(), epsilon = 1e-10);
            assert_abs_diff_eq!(x.as_matrix(), id.compose(&x).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SOT3::random();
            assert_abs_diff_eq!(x.compose(&x.inverse()).as_matrix(), Matrix4::identity(), epsilon = 1e-10);
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SOT3::random();
            let u = Vector4::new(rand::random(), rand::random(), rand::random(), rand::random());
            let ad_xu1 = SOT3::wedge(&(x.adjoint() * u));
            let ad_xu2 = x.as_matrix() * SOT3::wedge(&u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let v = Vector4::new(rand::random(), rand::random(), rand::random(), rand::random());
            let u = Vector4::new(rand::random(), rand::random(), rand::random(), rand::random());
            let ad_vu1 = SOT3::wedge(&(SOT3::adjoint_algebra(&v) * u));
            let ad_vu2 = SOT3::wedge(&v) * SOT3::wedge(&u) - SOT3::wedge(&u) * SOT3::wedge(&v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }
}

// =====================================================================
// SEn(3) — tested with n = 2
// =====================================================================
mod sen3_tests {
    use super::*;

    const N: usize = 2;

    fn rand_algebra() -> DVector<f64> {
        let mut u = DVector::from_fn(9, |_, _| rand::random::<f64>() - 0.5);
        u[0] *= 0.5;
        u[1] *= 0.5;
        u[2] *= 0.5;
        u
    }

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let v = DVector::from_fn(9, |_, _| rand::random::<f64>());
            let m = SEn3::wedge(N, &v);
            let v2 = SEn3::vee(N, &m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let u = rand_algebra();
            let x = SEn3::exp(N, &u);
            let u2 = x.log();
            assert_abs_diff_eq!(x.as_matrix(), SEn3::exp(N, &u2).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SEn3::random(N);
            let b = SEn3::random(N);
            let c = SEn3::random(N);
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn identity_element() {
        for _ in 0..N_TRIALS {
            let x = SEn3::random(N);
            let id = SEn3::identity(N);
            assert_abs_diff_eq!(x.as_matrix(), x.compose(&id).as_matrix(), epsilon = 1e-10);
            assert_abs_diff_eq!(x.as_matrix(), id.compose(&x).as_matrix(), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SEn3::random(N);
            assert_abs_diff_eq!(
                x.compose(&x.inverse()).as_matrix(),
                DMatrix::identity(5, 5),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SEn3::random(N);
            let u = DVector::from_fn(9, |_, _| rand::random::<f64>());
            let ad_xu1 = SEn3::wedge(N, &(x.adjoint() * &u));
            let ad_xu2 =
                x.as_matrix() * SEn3::wedge(N, &u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let v = DVector::from_fn(9, |_, _| rand::random::<f64>());
            let u = DVector::from_fn(9, |_, _| rand::random::<f64>());
            let ad_vu1 = SEn3::wedge(N, &(SEn3::adjoint_algebra(N, &v) * &u));
            let ad_vu2 =
                SEn3::wedge(N, &v) * SEn3::wedge(N, &u) - SEn3::wedge(N, &u) * SEn3::wedge(N, &v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn left_jacobian() {
        for _ in 0..N_TRIALS {
            let u = rand_algebra();
            let jl = SEn3::left_jacobian(N, &u);
            let a = SEn3::exp(N, &u).adjoint();
            let b = DMatrix::identity(9, 9) + SEn3::adjoint_algebra(N, &u) * jl;
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }
}

// =====================================================================
// SO(n) — tested with n = 4
// =====================================================================
mod son_tests {
    use super::*;

    const DIM: usize = 4;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let cdim = SOn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let m = SOn::wedge(DIM, &v);
            let v2 = SOn::vee(DIM, &m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let cdim = SOn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| 0.5 * (rand::random::<f64>() - 0.5));
            let x = SOn::exp(DIM, &v);
            let v2 = x.log();
            let x2 = SOn::exp(DIM, &v2);
            assert_abs_diff_eq!(x.as_matrix(), x2.as_matrix(), epsilon = 1e-6);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SOn::random(DIM);
            let b = SOn::random(DIM);
            let c = SOn::random(DIM);
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn identity_element() {
        let id = SOn::identity(DIM);
        assert_abs_diff_eq!(id.as_matrix(), DMatrix::identity(DIM, DIM), epsilon = 1e-15);
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SOn::random(DIM);
            assert_abs_diff_eq!(
                x.compose(&x.inverse()).as_matrix(),
                DMatrix::identity(DIM, DIM),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn matrix_inverse() {
        for _ in 0..N_TRIALS {
            let x = SOn::random(DIM);
            let xi1 = x.inverse().as_matrix();
            let xi2 = x.as_matrix().try_inverse().unwrap();
            assert_abs_diff_eq!(xi1, xi2, epsilon = 1e-10);
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SOn::random(DIM);
            let cdim = SOn::cdim(DIM);
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_xu1 = SOn::wedge(DIM, &(x.adjoint() * &u));
            let ad_xu2 = x.as_matrix() * SOn::wedge(DIM, &u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-10);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let cdim = SOn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_vu1 = SOn::wedge(DIM, &(SOn::adjoint_algebra(DIM, &v) * &u));
            let ad_vu2 = SOn::wedge(DIM, &v) * SOn::wedge(DIM, &u)
                - SOn::wedge(DIM, &u) * SOn::wedge(DIM, &v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }
}

// =====================================================================
// SL(n) — tested with n = 3
// =====================================================================
mod sln_tests {
    use super::*;

    const DIM: usize = 3;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let cdim = SLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let m = SLn::wedge(DIM, &v);
            let v2 = SLn::vee(DIM, &m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let cdim = SLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>() - 0.5);
            let x = SLn::exp(DIM, &v);
            let v2 = x.log();
            assert_abs_diff_eq!(v, v2, epsilon = 1e-6);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = SLn::random(DIM);
            let b = SLn::random(DIM);
            let c = SLn::random(DIM);
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = SLn::random(DIM);
            assert_abs_diff_eq!(
                x.compose(&x.inverse()).as_matrix(),
                DMatrix::identity(DIM, DIM),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = SLn::random(DIM);
            let cdim = SLn::cdim(DIM);
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_xu1 = SLn::wedge(DIM, &(x.adjoint() * &u));
            let ad_xu2 = x.as_matrix() * SLn::wedge(DIM, &u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-8);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let cdim = SLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_vu1 = SLn::wedge(DIM, &(SLn::adjoint_algebra(DIM, &v) * &u));
            let ad_vu2 = SLn::wedge(DIM, &v) * SLn::wedge(DIM, &u)
                - SLn::wedge(DIM, &u) * SLn::wedge(DIM, &v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }
}

// =====================================================================
// GL(n) — tested with n = 5
// =====================================================================
mod gln_tests {
    use super::*;

    const DIM: usize = 5;

    #[test]
    fn wedge_vee() {
        for _ in 0..N_TRIALS {
            let cdim = GLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let m = GLn::wedge(DIM, &v);
            let v2 = GLn::vee(DIM, &m);
            assert_abs_diff_eq!(v, v2, epsilon = 1e-15);
        }
    }

    #[test]
    fn exp_log() {
        for _ in 0..N_TRIALS {
            let cdim = GLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>() - 0.5);
            let x = GLn::exp(DIM, &v);
            let v2 = x.log();
            assert_abs_diff_eq!(v, v2, epsilon = 1e-6);
        }
    }

    #[test]
    fn associativity() {
        for _ in 0..N_TRIALS {
            let a = GLn::random(DIM);
            let b = GLn::random(DIM);
            let c = GLn::random(DIM);
            assert_abs_diff_eq!(
                a.compose(&b).compose(&c).as_matrix(),
                a.compose(&b.compose(&c)).as_matrix(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn inverse() {
        for _ in 0..N_TRIALS {
            let x = GLn::random(DIM);
            assert_abs_diff_eq!(
                x.compose(&x.inverse()).as_matrix(),
                DMatrix::identity(DIM, DIM),
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn group_adjoint() {
        for _ in 0..N_TRIALS {
            let x = GLn::random(DIM);
            let cdim = GLn::cdim(DIM);
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_xu1 = GLn::wedge(DIM, &(x.adjoint() * &u));
            let ad_xu2 = x.as_matrix() * GLn::wedge(DIM, &u) * x.inverse().as_matrix();
            assert_abs_diff_eq!(ad_xu1, ad_xu2, epsilon = 1e-8);
        }
    }

    #[test]
    fn algebra_adjoint() {
        for _ in 0..N_TRIALS {
            let cdim = GLn::cdim(DIM);
            let v = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let u = DVector::from_fn(cdim, |_, _| rand::random::<f64>());
            let ad_vu1 = GLn::wedge(DIM, &(GLn::adjoint_algebra(DIM, &v) * &u));
            let ad_vu2 = GLn::wedge(DIM, &v) * GLn::wedge(DIM, &u)
                - GLn::wedge(DIM, &u) * GLn::wedge(DIM, &v);
            assert_abs_diff_eq!(ad_vu1, ad_vu2, epsilon = 1e-10);
        }
    }
}
