pub mod test_source_text;
pub mod test_vio_filter;
pub mod testing_utilities;

use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::coordinate_suite::invdepth::InvDepthSuite;
use crate::coordinate_suite::normal::NormalSuite;
use crate::depth::sparse_gb::{SparseGBFilter, SparseVogSettings};
use crate::mathematical::camera::CameraModel;
use crate::mathematical::*;
use crate::tests::testing_utilities::*;
use approx::assert_abs_diff_eq;
use echo_lie::SO3;
use nalgebra::{DVector, Matrix3, Matrix4, Vector2, Vector3};
use rand::Rng;
use std::collections::HashMap;

const NEAR_ZERO: f64 = 1e-9;
const TEST_REPS: usize = 5;

fn assert_rotation_eq(r1: &SO3, r2: &SO3, eps: f64) {
    let m1 = r1.as_matrix();
    let m2 = r2.as_matrix();
    assert_abs_diff_eq!(m1, m2, epsilon = eps);
}

// ---------------------------------------------------------------------------
// 1. VIOGroup Basic Operations (test_vio_group.py)
// ---------------------------------------------------------------------------

#[test]
fn test_vio_group_inverse_left() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let x = random_group_element(ids.len(), &mut rng);
        let result = x.inverse().compose(&x);
        assert!(log_norm(&result) < NEAR_ZERO);
    }
}

#[test]
fn test_vio_group_associativity() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let x1 = random_group_element(ids.len(), &mut rng);
        let x2 = random_group_element(ids.len(), &mut rng);
        let x3 = random_group_element(ids.len(), &mut rng);

        let lhs = x1.compose(&x2).compose(&x3);
        let rhs = x1.compose(&x2.compose(&x3));

        assert!(log_norm(&lhs.inverse().compose(&rhs)) < NEAR_ZERO);
        assert!(log_norm(&rhs.inverse().compose(&lhs)) < NEAR_ZERO);
    }
}

#[test]
fn test_identity_is_neutral() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let x = random_group_element(ids.len(), &mut rng);
        let i = VIOGroup::identity(&ids);

        assert!(log_norm(&i) < NEAR_ZERO);
        assert!(log_norm(&i.compose(&x).compose(&x.inverse())) < NEAR_ZERO);
        assert!(log_norm(&x.inverse().compose(&x.compose(&i))) < NEAR_ZERO);
    }
}

// ---------------------------------------------------------------------------
// 2. VIOGroup Actions (test_vio_group_actions.py)
// ---------------------------------------------------------------------------

#[test]
fn test_state_action_identity() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let xi0 = random_state_element(ids.len(), &mut rng);
        let i = VIOGroup::identity(&ids);
        let xi0_id = state_group_action(&i, &xi0);
        assert!(state_distance(&xi0_id, &xi0) < NEAR_ZERO);
    }
}

#[test]
fn test_state_action_compatibility() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let x1 = random_group_element(ids.len(), &mut rng);
        let x2 = random_group_element(ids.len(), &mut rng);
        let xi0 = random_state_element(ids.len(), &mut rng);

        let xi1 = state_group_action(&x2, &state_group_action(&x1, &xi0));
        let xi2 = state_group_action(&x1.compose(&x2), &xi0);

        assert!(state_distance(&xi1, &xi2) < NEAR_ZERO);
    }
}

// ---------------------------------------------------------------------------
// 3. EqF Matrices (test_eqf_matrices.py)
// ---------------------------------------------------------------------------

/// Helper: project all visible landmarks to pixel coordinates (measure_system_state equivalent)
fn measure_system_state(xi: &VIOState, cam: &dyn CameraModel) -> HashMap<u64, Vector2<f64>> {
    let mut coords = HashMap::new();
    for lm in &xi.camera_landmarks {
        if lm.p[2] > 0.01 {
            coords.insert(lm.id, cam.project(&lm.p));
        }
    }
    coords
}

macro_rules! test_a0t {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            let mut rng = rand::rng();
            let suite = $suite;
            for _ in 0..TEST_REPS {
                let xi0 = reasonable_state_element(5, &mut rng);
                let x_hat = reasonable_group_element(5, &mut rng);
                let vel = random_velocity_element(&mut rng);

                let a_analytical = suite.state_matrix_a(&x_hat, &xi0, &vel);

                let a0 = |eps: &DVector<f64>| {
                    let xi_hat = state_group_action(&x_hat, &xi0);
                    let xi_e = suite.state_chart_inv(eps, &xi0);
                    let xi = state_group_action(&x_hat, &xi_e);

                    let lambda_tilde = &lift_velocity(&xi, &vel) - &lift_velocity(&xi_hat, &vel);
                    let xi_hat_next = state_group_action(&vio_exp(&lambda_tilde), &xi_hat);
                    let xi_e_next = state_group_action(&x_hat.inverse(), &xi_hat_next);

                    suite.state_chart(&xi_e_next, &xi0)
                };

                let dim = xi0.dim();
                let zero = DVector::zeros(dim);
                assert!(a0(&zero).norm() < 1e-8, "a0(0) should be zero");

                let a_numerical = numerical_jacobian(a0, &zero, 1e-6);

                let diff = (&a_analytical - &a_numerical).norm();
                assert!(
                    diff < 1e-4 * (dim as f64),
                    "A0t Jacobian mismatch: ||A - A_num|| = {:.2e}",
                    diff
                );
            }
        }
    };
}

test_a0t!(test_a0t_finite_difference, EuclideanSuite);
test_a0t!(test_a0t_finite_difference_normal, NormalSuite::new());
test_a0t!(test_a0t_finite_difference_invdepth, InvDepthSuite::new());

macro_rules! test_bt {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            let mut rng = rand::rng();
            let suite = $suite;
            for _ in 0..TEST_REPS {
                let xi0 = reasonable_state_element(5, &mut rng);
                let x_hat = reasonable_group_element(5, &mut rng);
                let vel = random_velocity_element(&mut rng);

                let b_analytical = suite.input_matrix_b(&x_hat, &xi0);

                let b0 = |nu: &DVector<f64>| {
                    let xi_hat = state_group_action(&x_hat, &xi0);
                    let mut vel_perturbed = vel.clone();
                    vel_perturbed.gyr += nu.fixed_rows::<3>(0);
                    vel_perturbed.acc += nu.fixed_rows::<3>(3);

                    let mut lambda_tilde =
                        &lift_velocity(&xi_hat, &vel_perturbed) - &lift_velocity(&xi_hat, &vel);
                    lambda_tilde.u_beta += nu.fixed_rows::<6>(6);

                    let xi_hat_next = state_group_action(&vio_exp(&lambda_tilde), &xi_hat);
                    let xi_e_next = state_group_action(&x_hat.inverse(), &xi_hat_next);

                    suite.state_chart(&xi_e_next, &xi0)
                };

                let dim = xi0.dim();
                let zero_nu = DVector::zeros(12);
                assert!(b0(&zero_nu).norm() < 1e-8, "b0(0) should be zero");

                let b_numerical = numerical_jacobian(b0, &zero_nu, 1e-6);

                let diff = (&b_analytical - &b_numerical).norm();
                assert!(
                    diff < 1e-4 * (dim as f64),
                    "Bt Jacobian mismatch: ||B - B_num|| = {:.2e}",
                    diff
                );
            }
        }
    };
}

test_bt!(test_bt_finite_difference, EuclideanSuite);
test_bt!(test_bt_finite_difference_normal, NormalSuite::new());
test_bt!(test_bt_finite_difference_invdepth, InvDepthSuite::new());

macro_rules! test_ct {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            use crate::mathematical::camera::PinholeModel;

            let mut rng = rand::rng();
            let suite = $suite;
            let cam = PinholeModel {
                fx: 458.654,
                fy: 457.296,
                cx: 367.215,
                cy: 248.375,
            };

            for _ in 0..TEST_REPS {
                let xi0 = reasonable_state_element(5, &mut rng);
                let x_hat = reasonable_group_element(5, &mut rng);

                let xi_hat = state_group_action(&x_hat, &xi0);
                let y_hat = measure_system_state(&xi_hat, &cam);
                let mut y_ids: Vec<u64> = y_hat.keys().cloned().collect();
                y_ids.sort();
                if y_ids.is_empty() {
                    continue;
                }

                let n_obs = y_ids.len();

                #[allow(non_snake_case)]
                let Ct_star = suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, true);

                // Finite difference validation
                let ct = |eps: &DVector<f64>| {
                    let xi_e = suite.state_chart_inv(eps, &xi0);
                    let xi = state_group_action(&x_hat, &xi_e);
                    let y = measure_system_state(&xi, &cam);

                    let mut y_tilde = DVector::<f64>::zeros(2 * n_obs);
                    for (j, &id) in y_ids.iter().enumerate() {
                        if let (Some(y_obs), Some(y_pred)) = (y.get(&id), y_hat.get(&id)) {
                            y_tilde[2 * j] = y_obs[0] - y_pred[0];
                            y_tilde[2 * j + 1] = y_obs[1] - y_pred[1];
                        }
                    }
                    y_tilde
                };

                let dim = xi0.dim();
                let zero = DVector::zeros(dim);
                assert!(ct(&zero).norm() < 1e-8, "ct(0) should be zero");

                let step = (f32::EPSILON as f64).cbrt();
                let ct_numerical = numerical_jacobian(ct, &zero, step);

                let diff = (&Ct_star - &ct_numerical).norm();
                assert!(
                    diff < 1e-3,
                    "Ct Jacobian mismatch: ||Ct - Ct_numerical|| = {:.2e}",
                    diff
                );
            }
        }
    };
}

test_ct!(test_ct_finite_difference, EuclideanSuite);
// Note: Ct finite difference tests for Normal/InvDepth are intentionally omitted.
// The equivariant C* in those charts is an approximation (not exact Jacobian of
// h(X ⊲ chart_inv(eps))), so it doesn't match finite differences. The Python
// reference also only tests Ct finite difference for the Euclidean suite.

// Equivariant == non-equivariant at predicted measurement (Euclidean only,
// since Normal/InvDepth C*_i is chart-specific and doesn't equal the
// non-equivariant Jacobian in those coordinates)
#[test]
fn test_ct_equivariant_at_prediction() {
    use crate::mathematical::camera::PinholeModel;

    let mut rng = rand::rng();
    let suite = EuclideanSuite;
    let cam = PinholeModel {
        fx: 458.654,
        fy: 457.296,
        cx: 367.215,
        cy: 248.375,
    };

    for _ in 0..TEST_REPS {
        let xi0 = reasonable_state_element(5, &mut rng);
        let x_hat = reasonable_group_element(5, &mut rng);

        let xi_hat = state_group_action(&x_hat, &xi0);
        let y_hat = measure_system_state(&xi_hat, &cam);
        let mut y_ids: Vec<u64> = y_hat.keys().cloned().collect();
        y_ids.sort();
        if y_ids.is_empty() {
            continue;
        }

        #[allow(non_snake_case)]
        let Ct_star = suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, true);
        #[allow(non_snake_case)]
        let Ct_noneq = suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, false);

        let ct_diff = (&Ct_star - &Ct_noneq).norm();
        assert!(
            ct_diff < 1e-8,
            "C*_t and Ct should be equal at predicted measurement, diff={:.2e}",
            ct_diff
        );
    }
}

// Verify equivariant C* reduces to non-equivariant at predicted measurement
// for Normal and InvDepth charts (Euclidean already tested above).
#[test]
fn test_ct_equivariant_at_prediction_normal_invdepth() {
    use crate::mathematical::camera::PinholeModel;

    let mut rng = rand::rng();
    let cam = PinholeModel {
        fx: 458.654,
        fy: 457.296,
        cx: 367.215,
        cy: 248.375,
    };

    for (name, suite) in [
        (
            "Normal",
            Box::new(NormalSuite::new()) as Box<dyn EqFCoordinateSuite>,
        ),
        (
            "InvDepth",
            Box::new(InvDepthSuite::new()) as Box<dyn EqFCoordinateSuite>,
        ),
    ] {
        for _ in 0..TEST_REPS {
            let xi0 = reasonable_state_element(5, &mut rng);
            let x_hat = reasonable_group_element(5, &mut rng);

            let xi_hat = state_group_action(&x_hat, &xi0);
            let y_hat = measure_system_state(&xi_hat, &cam);
            let mut y_ids: Vec<u64> = y_hat.keys().cloned().collect();
            y_ids.sort();
            if y_ids.is_empty() {
                continue;
            }

            #[allow(non_snake_case)]
            let C_eq = suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, true);
            #[allow(non_snake_case)]
            let C_noneq = suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, false);

            let diff = (&C_eq - &C_noneq).norm();
            assert!(
                diff < 1e-8,
                "{}: C* and C should be equal at predicted measurement, diff={:.2e}",
                name,
                diff
            );
        }
    }
}

// Verify: for InvDepth C*, averaging-then-mapping == mapping-then-averaging
// Since ind2euc is a constant matrix (given q0), linearity guarantees
// 0.5*(A+B)*M == 0.5*(A*M + B*M). This test confirms it numerically.
#[test]
fn test_invdepth_cstar_avg_map_commutativity() {
    use crate::coordinate_suite::invdepth::conv_ind2euc;
    use crate::mathematical::camera::PinholeModel;

    let mut rng = rand::rng();
    let euclid_suite = EuclideanSuite;
    let invdepth_suite = InvDepthSuite::new();
    let cam = PinholeModel {
        fx: 458.654,
        fy: 457.296,
        cx: 367.215,
        cy: 248.375,
    };

    for _ in 0..TEST_REPS {
        let xi0 = reasonable_state_element(5, &mut rng);
        let x_hat = reasonable_group_element(5, &mut rng);

        let xi_hat = state_group_action(&x_hat, &xi0);
        let y_hat = measure_system_state(&xi_hat, &cam);
        let mut y_ids: Vec<u64> = y_hat.keys().cloned().collect();
        y_ids.sort();
        if y_ids.is_empty() {
            continue;
        }

        // Method 1 (current): C*_invdepth via invdepth suite
        // Internally does: C*_euclid(avg of y_tru, y_hat) * ind2euc
        #[allow(non_snake_case)]
        let C_current = invdepth_suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, true);

        // Method 2 (proposed): map each C_euclid term to invdepth, then average
        // For each landmark: 0.5 * (C_euclid(y_tru) * ind2euc + C_euclid(y_hat) * ind2euc)
        // Build this by getting the non-equivariant (y_hat only) and equivariant (averaged)
        // Euclidean C*, converting each to invdepth, then averaging.
        //
        // Actually, since C*_euclid = 0.5*(D_rho(y_tru) + D_rho(y_hat)) * rest,
        // and the "rest" includes Adj(Q^-1)*m2g which is constant,
        // mapping then averaging means:
        //   0.5 * (D_rho(y_tru) * Adj(Q^-1) * m2g * ind2euc + D_rho(y_hat) * Adj(Q^-1) * m2g * ind2euc)
        // = 0.5 * (D_rho(y_tru) + D_rho(y_hat)) * Adj(Q^-1) * m2g * ind2euc
        // = C*_euclid * ind2euc  (= current method)
        //
        // So they're algebraically identical. Let's verify by computing both
        // the full equivariant C and the per-landmark mapped-then-averaged version.

        // Per-landmark check: compare ci_star results
        for &id in &y_ids {
            let lm_idx = xi0
                .camera_landmarks
                .iter()
                .position(|l| l.id == id)
                .unwrap();
            let q0 = xi0.camera_landmarks[lm_idx].p;
            let qi = &x_hat.q[lm_idx];

            // Current: avg in euclid, then map
            let ci_avg_then_map =
                invdepth_suite.output_matrix_ci_star(&q0, qi, &cam, y_hat.get(&id).unwrap());

            // Alternative: map each D_rho term, then average
            // = C_euclid(y_tru) * ind2euc  averaged with  C_euclid(y_hat) * ind2euc
            // Since output_matrix_ci_star with equivariant=true already averages,
            // we verify by computing: euclid_ci_star * ind2euc
            let ci_euclid =
                euclid_suite.output_matrix_ci_star(&q0, qi, &cam, y_hat.get(&id).unwrap());
            let ind2euc = conv_ind2euc(&q0);
            let ci_map_then_avg = ci_euclid * ind2euc;

            let diff = (ci_avg_then_map - ci_map_then_avg).norm();
            assert!(
                diff < 1e-12,
                "avg->map vs map->avg should be identical, diff={:.2e}",
                diff
            );
        }

        // Also verify the full stacked C matrix matches
        #[allow(non_snake_case)]
        let C_euclid = euclid_suite.output_matrix_C(&xi0, &x_hat, &y_ids, &y_hat, &cam, true);

        // Convert Euclidean C* to InvDepth by right-multiplying landmark blocks by ind2euc
        let s = VIOSensorState::CDIM;
        let n_obs = y_ids.len();
        let dim = xi0.dim();
        let mut c_manual = C_euclid.clone();
        for (j, &id) in y_ids.iter().enumerate() {
            let _ = j; // used implicitly via the loop
            let lm_idx = xi0
                .camera_landmarks
                .iter()
                .position(|l| l.id == id)
                .unwrap();
            let q0 = xi0.camera_landmarks[lm_idx].p;
            let ind2euc = conv_ind2euc(&q0);

            let col_start = s + 3 * lm_idx;
            let block = C_euclid.view((0, col_start), (2 * n_obs, 3)).into_owned();
            let mapped = block * ind2euc;
            c_manual
                .view_mut((0, col_start), (2 * n_obs, 3))
                .copy_from(&mapped);
        }

        let full_diff = (&C_current - &c_manual).norm();
        assert!(
            full_diff < 1e-10,
            "Full C* invdepth: avg->map vs map->avg should match, diff={:.2e}",
            full_diff
        );
    }
}

// Verify: Normal C*_i (custom formula) == C*_euc * conv_normal2euc
// If they differ, the custom formula is wrong and should be replaced.
#[test]
fn test_normal_cstar_vs_euc_composed() {
    use crate::coordinate_suite::normal::conv_normal2euc;
    use crate::mathematical::camera::PinholeModel;

    let mut rng = rand::rng();
    let euclid_suite = EuclideanSuite;
    let normal_suite = NormalSuite::new();
    let cam = PinholeModel {
        fx: 458.654,
        fy: 457.296,
        cx: 367.215,
        cy: 248.375,
    };

    for _ in 0..TEST_REPS {
        let xi0 = reasonable_state_element(5, &mut rng);
        let x_hat = reasonable_group_element(5, &mut rng);

        let xi_hat = state_group_action(&x_hat, &xi0);
        let y_hat = measure_system_state(&xi_hat, &cam);
        let mut y_ids: Vec<u64> = y_hat.keys().cloned().collect();
        y_ids.sort();
        if y_ids.is_empty() {
            continue;
        }

        for &id in &y_ids {
            let lm_idx = xi0
                .camera_landmarks
                .iter()
                .position(|l| l.id == id)
                .unwrap();
            let q0 = xi0.camera_landmarks[lm_idx].p;
            let qi = &x_hat.q[lm_idx];
            let y_obs = y_hat.get(&id).unwrap();

            // Method A: Normal's custom ci_star
            let ci_normal = normal_suite.output_matrix_ci_star(&q0, qi, &cam, y_obs);

            // Method B: C*_euc * conv_normal2euc
            let ci_euc = euclid_suite.output_matrix_ci_star(&q0, qi, &cam, y_obs);
            let n2e = conv_normal2euc(&q0);
            let ci_composed = ci_euc * n2e;

            let diff = (ci_normal - ci_composed).norm();
            let scale = ci_composed.norm().max(1e-10);
            assert!(
                diff / scale < 1e-6,
                "Normal C*_i differs from C*_euc * normal2euc: diff={:.2e}, rel={:.2e}",
                diff,
                diff / scale
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. VIO Lift Tests (test_vio_lift.py)
// ---------------------------------------------------------------------------

#[test]
fn test_discrete_lift_exact() {
    let mut rng = rand::rng();
    let ids = vec![0, 1, 2, 3, 4];
    for _ in 0..TEST_REPS {
        let xi0 = random_state_element(ids.len(), &mut rng);
        let vel = random_velocity_element(&mut rng);
        let dt = 0.1;

        // Ground truth: Kinematic integration
        let xi_gt = integrate_system_function(&xi0, &vel, dt);

        // Discrete lift + action
        let lambda = lift_velocity_discrete(&xi0, &vel, dt);
        let xi_lifted = state_group_action(&lambda, &xi0);

        assert!(state_distance(&xi_gt, &xi_lifted) < 1e-10);
    }
}

macro_rules! test_innovation_lift {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            let mut rng = rand::rng();
            let suite = $suite;
            for _ in 0..TEST_REPS {
                let xi0 = reasonable_state_element(5, &mut rng);
                let dim = xi0.dim();
                let eps = DVector::from_fn(dim, |_, _| rng.random_range(-0.01..0.01));

                // Discrete lift to group then action
                let lambda = suite.lift_innovation_discrete(&eps, &xi0);
                let xi1 = state_group_action(&lambda, &xi0);

                // Chart back should recover eps
                let eps_back = suite.state_chart(&xi1, &xi0);
                assert_abs_diff_eq!(eps, eps_back, epsilon = 1e-8);
            }
        }
    };
}

test_innovation_lift!(test_innovation_lift_roundtrip, EuclideanSuite);
test_innovation_lift!(test_innovation_lift_roundtrip_normal, NormalSuite::new());
test_innovation_lift!(
    test_innovation_lift_roundtrip_invdepth,
    InvDepthSuite::new()
);

// ---------------------------------------------------------------------------
// 5. Coordinate Chart Axioms (test_coordinate_charts.py)
// ---------------------------------------------------------------------------

macro_rules! test_vio_chart {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            let mut rng = rand::rng();
            let suite = $suite;
            let n_landmarks = 5;

            for _ in 0..TEST_REPS {
                let xi0 = random_state_element(n_landmarks, &mut rng);
                let xi1 = random_state_element(n_landmarks, &mut rng);

                let eps = suite.state_chart(&xi1, &xi0);
                let xi2 = suite.state_chart_inv(&eps, &xi0);

                assert_abs_diff_eq!(xi1.sensor.input_bias, xi2.sensor.input_bias, epsilon = 1e-8);
                assert_rotation_eq(&xi1.sensor.pose.rotation, &xi2.sensor.pose.rotation, 1e-8);
                assert_abs_diff_eq!(
                    xi1.sensor.pose.translation,
                    xi2.sensor.pose.translation,
                    epsilon = 1e-8
                );
                assert_abs_diff_eq!(xi1.sensor.velocity, xi2.sensor.velocity, epsilon = 1e-8);
                assert_rotation_eq(
                    &xi1.sensor.camera_offset.rotation,
                    &xi2.sensor.camera_offset.rotation,
                    1e-8,
                );
                assert_abs_diff_eq!(
                    xi1.sensor.camera_offset.translation,
                    xi2.sensor.camera_offset.translation,
                    epsilon = 1e-8
                );

                for i in 0..n_landmarks {
                    assert_abs_diff_eq!(
                        xi1.camera_landmarks[i].p,
                        xi2.camera_landmarks[i].p,
                        epsilon = 1e-8
                    );
                }

                let eps_zero = suite.state_chart(&xi0, &xi0);
                assert_abs_diff_eq!(eps_zero.norm(), 0.0, epsilon = 1e-10);

                let xi_zero = suite.state_chart_inv(&DVector::zeros(xi0.dim()), &xi0);
                assert_rotation_eq(
                    &xi0.sensor.pose.rotation,
                    &xi_zero.sensor.pose.rotation,
                    1e-10,
                );
                assert_abs_diff_eq!(
                    xi0.sensor.pose.translation,
                    xi_zero.sensor.pose.translation,
                    epsilon = 1e-10
                );
            }
        }
    };
}

test_vio_chart!(test_euclidean_chart_axioms, EuclideanSuite);
test_vio_chart!(test_normal_chart_axioms, NormalSuite::new());
test_vio_chart!(test_invdepth_chart_axioms, InvDepthSuite::new());

// ---------------------------------------------------------------------------
// 6. Sparse Depth Convergence (test_polar3d_convergence.py)
// ---------------------------------------------------------------------------

#[test]
fn test_depth_filter_convergence() {
    let mut rng = rand::rng();
    let fx = 458.0;
    let k = Matrix3::new(fx, 0.0, 376.0, 0.0, fx, 240.0, 0.0, 0.0, 1.0);
    let mut filt = SparseGBFilter::new(k, SparseVogSettings::default());
    let fid = 42;
    let p_world = Vector3::new(1.0, 0.5, 3.0);
    let baseline = 0.05;

    for i in 0..30 {
        let cam_pos = Vector3::new(i as f64 * baseline, 0.0, 0.0);
        let mut t_wc = Matrix4::identity();
        t_wc.fixed_view_mut::<3, 1>(0, 3).copy_from(&cam_pos);

        let p_cam = p_world - cam_pos;
        let uv = Vector2::new(
            fx * p_cam[0] / p_cam[2] + 376.0 + rng.random_range(-0.1..0.1),
            fx * p_cam[1] / p_cam[2] + 240.0 + rng.random_range(-0.1..0.1),
        );

        let mut cam_coords = HashMap::new();
        cam_coords.insert(fid as u64, Vector2::new(uv[0] as f32, uv[1] as f32));
        let meas = VisionMeasurement::new(i as f64 * 0.05, cam_coords);

        filt.update(&meas, &t_wc, None);
    }

    let (depth, var) = filt.query(fid);
    assert!(depth > 0.0);
    assert!((depth - 3.0).abs() < 0.2);
    assert!(var < 0.1);
}
