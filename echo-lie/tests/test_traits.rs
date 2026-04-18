use approx::assert_abs_diff_eq;
use echo_lie::base::LieGroup;
use echo_lie::{SO3, SE3, SOT3, SE23};
use nalgebra::Vector3;

macro_rules! test_lie_group_axioms {
    ($name:ident, $group:ty, $eps:expr) => {
        #[test]
        fn $name() {
            let trials = 100;
            let eps = $eps;
            let id = <$group as LieGroup>::identity();
            
            // Tangent zero
            let v_zero = <$group as LieGroup>::Tangent::zeros();
            assert_abs_diff_eq!(<$group as LieGroup>::exp(&v_zero).log(), v_zero, epsilon = eps);

            for _ in 0..trials {
                // Random displacement
                let mut v = <$group as LieGroup>::Tangent::zeros();
                for i in 0..v.len() {
                    v[i] = (rand::random::<f64>() - 0.5) * 0.1;
                }
                let x = <$group as LieGroup>::exp(&v);
                
                // Identity composition
                let x_id = x.compose(&id);
                let id_x = id.compose(&x);
                assert_abs_diff_eq!(x.log(), x_id.log(), epsilon = eps);
                assert_abs_diff_eq!(x.log(), id_x.log(), epsilon = eps);
                
                // Inverse
                let x_inv = x.inverse();
                let should_be_id_1 = x.compose(&x_inv);
                let should_be_id_2 = x_inv.compose(&x);
                assert_abs_diff_eq!(should_be_id_1.log(), v_zero, epsilon = eps);
                assert_abs_diff_eq!(should_be_id_2.log(), v_zero, epsilon = eps);
                
                // Exp/Log Roundtrip
                let x_log = x.log();
                assert_abs_diff_eq!(v, x_log, epsilon = eps);
                
                // Adjoint Property: exp(Ad_X * v) = X * exp(v) * X.inv()
                let mut v2 = <$group as LieGroup>::Tangent::zeros();
                for i in 0..v2.len() {
                    v2[i] = (rand::random::<f64>() - 0.5) * 0.1;
                }
                
                let lhs = <$group as LieGroup>::exp(&(x.adjoint() * &v2));
                let rhs = x.compose(&<$group as LieGroup>::exp(&v2)).compose(&x.inverse());
                assert_abs_diff_eq!(lhs.log(), rhs.log(), epsilon = eps);

                // Action
                let p = Vector3::new(1.0, 2.0, 3.0);
                assert_abs_diff_eq!(id.act(&p), p, epsilon = eps);
            }
        }
    };
}

test_lie_group_axioms!(test_so3_traits, SO3, 1e-10);
test_lie_group_axioms!(test_se3_traits, SE3, 1e-10);
test_lie_group_axioms!(test_sot3_traits, SOT3, 1e-10);
test_lie_group_axioms!(test_se23_traits, SE23, 1e-10);
