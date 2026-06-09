use approx::assert_abs_diff_eq;
use echo_lie::{SE3, SO3};
use nalgebra::{DMatrix, DVector, SMatrix, Vector2, Vector3, Vector6};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::time::Instant;

use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::coordinate_suite::invdepth::InvDepthSuite;
use crate::coordinate_suite::normal::NormalSuite;
use crate::mathematical::bias_group_ops::BiasGroupOps;
use crate::mathematical::camera::{CameraModel, PinholeModel};
use crate::mathematical::eqf_matrices::EqFCoordinateSuite;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_eqf::VIOEqF;
use crate::mathematical::vio_group::{
    lift_velocity, state_group_action, vio_exp_with_bias_group, VIOGroup,
};
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState, GRAVITY_CONSTANT};
use crate::mathematical::vision_measurement::VisionMeasurement;
use crate::tests::testing_utilities::*;
use crate::{ImuBiasGroup, LandmarkDepthPrior, VIOFilter, VIOFilterSettings};

fn make_xi0_with_landmarks(n: usize) -> VIOState {
    let mut landmarks = Vec::new();
    for i in 0..n {
        landmarks.push(Landmark {
            p: Vector3::new(
                (i as f64 - n as f64 / 2.0) * 0.5,
                ((i * 7 + 3) % 5) as f64 * 0.3,
                3.0 + (i as f64) * 0.2,
            ),
            id: i as u64,
        });
    }
    VIOState {
        sensor: VIOSensorState {
            input_bias: Vector6::zeros(),
            pose: SE3::identity(),
            velocity: Vector3::zeros(),
            camera_offset: SE3::new(SO3::identity(), Vector3::new(0.05, 0.0, 0.0)),
        },
        camera_landmarks: landmarks,
    }
}

fn make_pinhole() -> PinholeModel {
    PinholeModel {
        fx: 458.0,
        fy: 458.0,
        cx: 376.0,
        cy: 240.0,
    }
}

fn stationary_imu(stamp: f64) -> IMUVelocity {
    IMUVelocity::new(
        stamp,
        Vector3::zeros(),
        Vector3::new(0.0, 0.0, GRAVITY_CONSTANT),
    )
}

// ---------------------------------------------------------------------------
// 1. VIOEqF Construction
// ---------------------------------------------------------------------------

#[test]
fn test_eqf_new_identity_state() {
    let xi0 = make_xi0_with_landmarks(5);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.1;
    let eqf = VIOEqF::new(xi0.clone(), &init_cov);

    // State estimate should equal xi0 when X = identity
    let est = eqf.state_estimate();
    assert_abs_diff_eq!(
        est.sensor.pose.translation,
        xi0.sensor.pose.translation,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(est.sensor.velocity, xi0.sensor.velocity, epsilon = 1e-12);
    for (e, o) in est.camera_landmarks.iter().zip(xi0.camera_landmarks.iter()) {
        assert_abs_diff_eq!(e.p, o.p, epsilon = 1e-12);
    }

    // Covariance should match what was given
    let n = xi0.dim();
    for i in 0..n {
        assert_abs_diff_eq!(eqf.sigma[(i, i)], 0.1, epsilon = 1e-12);
    }
}

#[test]
fn test_eqf_new_get_ids() {
    let xi0 = make_xi0_with_landmarks(3);
    assert_eq!(xi0.get_ids(), vec![0, 1, 2]);
}

#[test]
fn test_filter_new_wires_semi_direct_bias_group() {
    let mut settings = VIOFilterSettings::default();
    settings.imu_bias_group = ImuBiasGroup::SemiDirect;
    let xi0 = make_xi0_with_landmarks(0);
    let filter = VIOFilter::new(settings, xi0);

    assert_eq!(filter.eqf.x.imu_bias_group, ImuBiasGroup::SemiDirect);
}

#[test]
fn test_vio_group_semi_direct_bias_composition() {
    let mut x = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    x.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    x.a = SE3::new(
        SO3::exp(&Vector3::new(0.02, -0.03, 0.01)),
        Vector3::new(0.4, -0.2, 0.1),
    );
    x.w = Vector3::new(0.2, -0.1, 0.3);

    let mut y = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    y.beta = Vector6::new(-0.05, 0.04, 0.02, -0.2, 0.15, -0.03);
    y.a = SE3::new(
        SO3::exp(&Vector3::new(-0.01, 0.04, 0.02)),
        Vector3::new(-0.3, 0.1, 0.2),
    );
    y.w = Vector3::new(-0.1, 0.05, 0.2);

    let composed = x.compose(&y);
    let expected_beta = x.beta + SE3::new(x.a.rotation.clone(), x.w).adjoint() * y.beta;

    assert_abs_diff_eq!(composed.beta, expected_beta, epsilon = 1e-12);
    assert_abs_diff_eq!(
        composed.a.translation,
        x.a.compose(&y.a).translation,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(composed.w, x.w + x.a.rotation.act(&y.w), epsilon = 1e-12);
}

#[test]
fn test_semi_direct_bias_action_matches_group_composition() {
    let ops = BiasGroupOps::new(ImuBiasGroup::SemiDirect);
    let mut x = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    x.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    x.a = SE3::new(
        SO3::exp(&Vector3::new(0.2, -0.1, 0.05)),
        Vector3::new(0.4, -0.2, 0.1),
    );
    x.w = Vector3::new(0.2, -0.1, 0.3);

    let mut y = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    y.beta = Vector6::new(-0.05, 0.04, 0.02, -0.2, 0.15, -0.03);
    y.a = SE3::new(
        SO3::exp(&Vector3::new(-0.08, 0.12, 0.04)),
        Vector3::new(-0.3, 0.1, 0.2),
    );
    y.w = Vector3::new(-0.1, 0.05, 0.2);

    let bias_origin = Vector6::new(0.02, -0.01, 0.03, 0.1, -0.05, 0.04);
    let sequential = ops.act_bias(&y, &ops.act_bias(&x, &bias_origin));
    let composed = ops.act_bias(&x.compose(&y), &bias_origin);
    assert_abs_diff_eq!(composed, sequential, epsilon = 1e-12);

    let identity = x.compose(&x.inverse());
    assert_abs_diff_eq!(identity.beta, Vector6::zeros(), epsilon = 1e-12);
    assert_abs_diff_eq!(
        identity.a.log(),
        nalgebra::Vector6::zeros(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(identity.w, Vector3::zeros(), epsilon = 1e-12);
}

#[test]
fn test_semi_direct_physical_bias_beta_jacobian() {
    let ops = BiasGroupOps::new(ImuBiasGroup::SemiDirect);
    let mut x = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    x.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    x.a = SE3::new(
        SO3::exp(&Vector3::new(0.25, -0.15, 0.08)),
        Vector3::new(0.4, -0.2, 0.1),
    );
    x.w = Vector3::new(0.2, -0.1, 0.3);
    let bias_origin = Vector6::new(0.02, -0.01, 0.03, 0.1, -0.05, 0.04);

    let eps = 1e-7;
    let mut numerical = SMatrix::<f64, 6, 6>::zeros();
    for col in 0..6 {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp.beta[col] += eps;
        xm.beta[col] -= eps;
        let fp = ops.act_bias(&xp, &bias_origin);
        let fm = ops.act_bias(&xm, &bias_origin);
        numerical.set_column(col, &((fp - fm) / (2.0 * eps)));
    }

    let expected = BiasGroupOps::bias_action_matrix(&x).inverse().adjoint();
    assert_abs_diff_eq!(numerical, expected, epsilon = 1e-8);
}

#[test]
fn test_semi_direct_beta_for_physical_bias_update() {
    let ops = BiasGroupOps::new(ImuBiasGroup::SemiDirect);
    let mut current = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    current.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    current.a = SE3::new(
        SO3::exp(&Vector3::new(0.12, -0.05, 0.03)),
        Vector3::new(0.2, -0.1, 0.05),
    );
    current.w = Vector3::new(0.08, -0.03, 0.1);

    let mut delta = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::SemiDirect);
    delta.a = SE3::new(
        SO3::exp(&Vector3::new(-0.02, 0.04, 0.01)),
        Vector3::new(-0.03, 0.02, 0.01),
    );
    delta.w = Vector3::new(0.01, -0.02, 0.03);

    let bias_origin = Vector6::new(0.02, -0.01, 0.03, 0.1, -0.05, 0.04);
    let physical_delta = Vector6::new(0.004, -0.002, 0.003, -0.01, 0.005, -0.006);
    let before = ops.act_bias(&current, &bias_origin);
    delta.beta = ops.beta_for_physical_bias_update(&current, &delta, &bias_origin, &physical_delta);
    let after = ops.act_bias(&delta.compose(&current), &bias_origin);

    assert_abs_diff_eq!(after, before + physical_delta, epsilon = 1e-12);
}

#[test]
fn test_semi_direct_bias_process_noise_uses_action_matrix() {
    let suite = EuclideanSuite;
    let xi0 = make_xi0_with_landmarks(0);
    let imu = IMUVelocity::new(
        0.0,
        Vector3::new(0.1, -0.2, 0.3),
        Vector3::new(0.4, -0.1, 9.7),
    );

    let mut additive = VIOGroup::identity_with_bias_group(&[], ImuBiasGroup::Additive);
    additive.a = SE3::new(
        SO3::exp(&Vector3::new(0.12, -0.05, 0.03)),
        Vector3::new(0.2, -0.1, 0.05),
    );
    additive.w = Vector3::new(0.08, -0.03, 0.1);
    let additive_blocks = suite.propagation_blocks(&additive, &xi0, &imu);
    assert_abs_diff_eq!(
        additive_blocks.b_s.fixed_view::<6, 6>(0, 6).into_owned(),
        SMatrix::<f64, 6, 6>::identity(),
        epsilon = 1e-12
    );

    let mut semi_direct = additive.with_bias_group(ImuBiasGroup::SemiDirect);
    semi_direct.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    let semi_direct_blocks = suite.propagation_blocks(&semi_direct, &xi0, &imu);
    let expected = BiasGroupOps::bias_action_matrix(&semi_direct).adjoint();
    assert_abs_diff_eq!(
        semi_direct_blocks.b_s.fixed_view::<6, 6>(0, 6).into_owned(),
        expected,
        epsilon = 1e-12
    );
}

fn nontrivial_semi_direct_group(ids: &[u64]) -> VIOGroup {
    let mut x = VIOGroup::identity_with_bias_group(ids, ImuBiasGroup::SemiDirect);
    x.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    x.a = SE3::new(
        SO3::exp(&Vector3::new(0.12, -0.05, 0.03)),
        Vector3::new(0.2, -0.1, 0.05),
    );
    x.w = Vector3::new(0.08, -0.03, 0.1);
    x.b = SE3::new(
        SO3::exp(&Vector3::new(-0.03, 0.02, 0.01)),
        Vector3::new(0.04, -0.02, 0.03),
    );
    x.q = ids
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let k = i as f64 + 1.0;
            echo_lie::SOT3::new(
                SO3::exp(&Vector3::new(0.01 * k, -0.005 * k, 0.003 * k)),
                1.0 + 0.01 * k,
            )
        })
        .collect();
    x
}

fn assert_semi_direct_a0t_matches_finite_difference<S: EqFCoordinateSuite>(
    suite: &S,
    name: &str,
) {
    let xi0 = make_xi0_with_landmarks(2);
    let mut x_hat = VIOGroup::identity_with_bias_group(&xi0.get_ids(), ImuBiasGroup::SemiDirect);
    x_hat.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);
    x_hat.a = SE3::new(
        SO3::exp(&Vector3::new(0.12, -0.05, 0.03)),
        Vector3::new(0.2, -0.1, 0.05),
    );
    x_hat.w = Vector3::new(0.08, -0.03, 0.1);

    let imu = IMUVelocity::new(
        0.0,
        Vector3::new(0.1, -0.2, 0.3),
        Vector3::new(0.4, -0.1, 9.7),
    );
    let a_analytical = suite.state_matrix_a(&x_hat, &xi0, &imu);

    let a0 = |eps: &DVector<f64>| {
        let xi_hat = state_group_action(&x_hat, &xi0);
        let xi_e = suite.state_chart_inv(eps, &xi0);
        let xi = state_group_action(&x_hat, &xi_e);

        let lambda_tilde = &lift_velocity(&xi, &imu) - &lift_velocity(&xi_hat, &imu);
        let delta = vio_exp_with_bias_group(&lambda_tilde, ImuBiasGroup::SemiDirect);
        let xi_hat_next = state_group_action(&delta, &xi_hat);
        let xi_e_next = state_group_action(&x_hat.inverse(), &xi_hat_next);

        suite.state_chart(&xi_e_next, &xi0)
    };

    let zero = DVector::zeros(xi0.dim());
    assert!(a0(&zero).norm() < 1e-8, "a0(0) should be zero");

    let a_numerical = numerical_jacobian(a0, &zero, 1e-6);
    let diff = (&a_analytical - &a_numerical).norm();
    assert!(
        diff < 1e-4 * (xi0.dim() as f64),
        "{name} semi-direct A0t Jacobian mismatch: ||A - A_num|| = {:.2e}",
        diff
    );
}

#[test]
fn test_semi_direct_a0t_matches_finite_difference() {
    assert_semi_direct_a0t_matches_finite_difference(&EuclideanSuite, "euclidean");
}

#[test]
fn test_semi_direct_a0t_matches_finite_difference_normal() {
    assert_semi_direct_a0t_matches_finite_difference(&NormalSuite::new(), "normal");
}

#[test]
fn test_semi_direct_a0t_matches_finite_difference_invdepth() {
    assert_semi_direct_a0t_matches_finite_difference(&InvDepthSuite::new(), "invdepth");
}

#[test]
fn test_semi_direct_stacked_update_preserves_physical_bias_increment() {
    let suite = EuclideanSuite;
    let mut xi0 = make_xi0_with_landmarks(0);
    xi0.sensor.input_bias = Vector6::new(0.02, -0.01, 0.03, 0.1, -0.05, 0.04);

    for use_discrete_correction in [false, true] {
        let mut eqf = VIOEqF::new_with_bias_group(
            xi0.clone(),
            &DMatrix::<f64>::identity(VIOSensorState::CDIM, VIOSensorState::CDIM),
            ImuBiasGroup::SemiDirect,
        );
        eqf.x.a = SE3::new(
            SO3::exp(&Vector3::new(0.12, -0.05, 0.03)),
            Vector3::new(0.2, -0.1, 0.05),
        );
        eqf.x.w = Vector3::new(0.08, -0.03, 0.1);
        eqf.x.beta = Vector6::new(0.1, -0.2, 0.05, 0.3, -0.1, 0.2);

        let before = eqf.state_estimate().sensor.input_bias;

        let physical_bias_delta = Vector6::new(0.004, -0.002, 0.003, -0.01, 0.005, -0.006);
        let mut residual = DVector::<f64>::zeros(VIOSensorState::CDIM);
        residual
            .fixed_rows_mut::<6>(0)
            .copy_from(&physical_bias_delta);
        residual
            .fixed_rows_mut::<6>(6)
            .copy_from(&Vector6::new(0.02, -0.01, 0.03, -0.04, 0.02, 0.01));

        eqf.perform_stacked_update(
            &suite,
            &residual,
            &DMatrix::<f64>::identity(VIOSensorState::CDIM, VIOSensorState::CDIM),
            &DMatrix::<f64>::zeros(VIOSensorState::CDIM, VIOSensorState::CDIM),
            use_discrete_correction,
        );

        let after = eqf.state_estimate().sensor.input_bias;
        assert_abs_diff_eq!(after, before + physical_bias_delta, epsilon = 1e-12);
    }
}

#[test]
fn test_semi_direct_observer_integration_matches_kinematics() {
    let mut rng = rand::rng();
    let xi0 = reasonable_state_element(3, &mut rng);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.01;
    let mut eqf = VIOEqF::new_with_bias_group(xi0.clone(), &init_cov, ImuBiasGroup::SemiDirect);

    let imu = random_velocity_element(&mut rng);
    let dt = 0.01;

    eqf.integrate_observer_state(&imu, dt, true);
    let est = eqf.state_estimate();
    let gt = crate::mathematical::vio_state::integrate_system_function(&xi0, &imu, dt);

    assert!(
        state_distance(&est, &gt) < 1e-10,
        "Semi-direct observer integration diverged from kinematics: dist={}",
        state_distance(&est, &gt)
    );
}

#[test]
fn test_semi_direct_observer_integration_matches_kinematics_from_nonidentity_group() {
    let mut rng = rand::rng();
    let xi0 = reasonable_state_element(3, &mut rng);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.01;
    let mut eqf = VIOEqF::new_with_bias_group(xi0.clone(), &init_cov, ImuBiasGroup::SemiDirect);
    eqf.x = nontrivial_semi_direct_group(&xi0.get_ids());

    let before = eqf.state_estimate();
    let imu = random_velocity_element(&mut rng);
    let dt = 0.01;

    eqf.integrate_observer_state(&imu, dt, true);
    let est = eqf.state_estimate();
    let gt = crate::mathematical::vio_state::integrate_system_function(&before, &imu, dt);

    assert!(
        state_distance(&est, &gt) < 1e-10,
        "Semi-direct observer integration from nonidentity group diverged: dist={}",
        state_distance(&est, &gt)
    );
}

fn assert_semi_direct_fast_and_faster_riccati_match_zero_noise<S: EqFCoordinateSuite>(
    suite: &S,
    name: &str,
) {
    let mut rng = rand::rng();
    let xi0 = reasonable_state_element(2, &mut rng);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.01;
    let mut fast =
        VIOEqF::new_with_bias_group(xi0.clone(), &init_cov, ImuBiasGroup::SemiDirect);
    let mut faster = VIOEqF::new_with_bias_group(xi0.clone(), &init_cov, ImuBiasGroup::SemiDirect);
    fast.x = nontrivial_semi_direct_group(&xi0.get_ids());
    faster.x = fast.x.clone();

    let zero_input = SMatrix::<f64, 12, 12>::zeros();
    let zero_state = DMatrix::<f64>::zeros(xi0.dim(), xi0.dim());
    let samples = [
        IMUVelocity::new(
            0.0,
            Vector3::new(0.11, -0.05, 0.03),
            Vector3::new(0.2, -0.1, 9.7),
        ),
        IMUVelocity::new(
            0.01,
            Vector3::new(0.10, -0.04, 0.02),
            Vector3::new(0.1, -0.2, 9.8),
        ),
        IMUVelocity::new(
            0.02,
            Vector3::new(0.09, -0.03, 0.01),
            Vector3::new(0.0, -0.1, 9.75),
        ),
    ];

    for imu in &samples {
        let dt = 0.005;
        fast.integrate_riccati_fast(suite, imu, dt, &zero_input, &zero_state);
        faster.accumulate_transition(suite, imu, dt);
    }
    faster.flush_riccati(&zero_input, &zero_state);

    let diff = (&fast.sigma - &faster.sigma).norm();
    assert!(
        diff < 1e-10,
        "{name} semi-direct fast/faster Riccati mismatch with zero process noise: ||diff||={diff:.2e}"
    );
}

#[test]
fn test_semi_direct_fast_and_faster_riccati_match_zero_noise() {
    assert_semi_direct_fast_and_faster_riccati_match_zero_noise(&EuclideanSuite, "euclidean");
}

#[test]
fn test_semi_direct_fast_and_faster_riccati_match_zero_noise_normal() {
    assert_semi_direct_fast_and_faster_riccati_match_zero_noise(&NormalSuite::new(), "normal");
}

#[test]
fn test_semi_direct_fast_and_faster_riccati_match_zero_noise_invdepth() {
    assert_semi_direct_fast_and_faster_riccati_match_zero_noise(&InvDepthSuite::new(), "invdepth");
}

#[test]
fn test_semi_direct_bias_update_runs_for_all_charts() {
    let cam = make_pinhole();

    for coordinate_choice in ["Euclidean", "InvDepth", "Normal"] {
        for use_discrete_correction in [false, true] {
            let mut settings = VIOFilterSettings::default();
            settings.coordinate_choice = coordinate_choice.to_string();
            settings.imu_bias_group = ImuBiasGroup::SemiDirect;
            settings.use_discrete_correction = use_discrete_correction;

            let xi0 = make_xi0_with_landmarks(3);
            let mut filter = VIOFilter::new(settings, xi0.clone());

            filter.process_imu(stationary_imu(0.0));
            filter.process_imu(stationary_imu(0.005));

            let mut coords = HashMap::new();
            for lm in &xi0.camera_landmarks {
                let proj = cam.project(&lm.p) + Vector2::new(0.4, -0.25);
                coords.insert(lm.id, Vector2::new(proj[0] as f32, proj[1] as f32));
            }
            let measurement = VisionMeasurement::new(0.005, coords);
            filter.process_vision(measurement, &cam);

            let est = filter.state_estimate();
            assert!(
                est.sensor.input_bias.iter().all(|v| v.is_finite()),
                "non-finite bias for chart={coordinate_choice}, discrete={use_discrete_correction}"
            );
            assert!(
                est.sensor.pose.translation.iter().all(|v| v.is_finite()),
                "non-finite position for chart={coordinate_choice}, discrete={use_discrete_correction}"
            );
            assert!(
                filter.eqf.sigma.iter().all(|v| v.is_finite()),
                "non-finite covariance for chart={coordinate_choice}, discrete={use_discrete_correction}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Observer State Integration
// ---------------------------------------------------------------------------

#[test]
fn test_observer_integration_stationary() {
    let xi0 = make_xi0_with_landmarks(3);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.01;
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    // Stationary IMU (acc = gravity, no rotation) => state should barely change
    let imu = stationary_imu(0.0);
    let dt = 0.005;
    eqf.integrate_observer_state(&imu, dt, true);

    let est = eqf.state_estimate();
    // Position should stay near zero (no velocity, no net acceleration)
    assert!(
        est.sensor.pose.translation.norm() < 1e-6,
        "Position drifted: {:?}",
        est.sensor.pose.translation
    );
    assert!(
        est.sensor.velocity.norm() < 1e-6,
        "Velocity drifted: {:?}",
        est.sensor.velocity
    );
}

#[test]
fn test_observer_integration_matches_kinematics() {
    let mut rng = rand::rng();
    let xi0 = reasonable_state_element(3, &mut rng);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.01;
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let imu = random_velocity_element(&mut rng);
    let dt = 0.01;

    // Observer integration: X = X * Lambda(xi_hat, u, dt)
    eqf.integrate_observer_state(&imu, dt, true);
    let est = eqf.state_estimate();

    // Ground truth kinematic integration
    let gt = crate::mathematical::vio_state::integrate_system_function(&xi0, &imu, dt);

    // They should match since X starts at identity, so xi_hat = xi0
    assert!(
        state_distance(&est, &gt) < 1e-10,
        "Observer integration diverged from kinematics: dist={}",
        state_distance(&est, &gt)
    );
}

// ---------------------------------------------------------------------------
// 3. Riccati Propagation
// ---------------------------------------------------------------------------

#[test]
fn test_riccati_covariance_grows() {
    let xi0 = make_xi0_with_landmarks(3);
    let settings = VIOFilterSettings::default();
    let n = xi0.dim();
    let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let suite = EuclideanSuite;
    let imu = stationary_imu(0.0);
    let dt = 0.005;
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());

    let diag_before: Vec<f64> = (0..n).map(|i| eqf.sigma[(i, i)]).collect();

    eqf.integrate_riccati_fast(&suite, &imu, dt, &input_gain, &state_gain);

    // Covariance diagonal should generally grow (process noise adds uncertainty)
    let diag_after: Vec<f64> = (0..n).map(|i| eqf.sigma[(i, i)]).collect();
    let grew_count = diag_before
        .iter()
        .zip(diag_after.iter())
        .filter(|(b, a)| *a > *b)
        .count();
    assert!(
        grew_count > n / 2,
        "Expected majority of diag entries to grow, but only {}/{} did",
        grew_count,
        n
    );
}

#[test]
fn test_riccati_preserves_symmetry() {
    let xi0 = make_xi0_with_landmarks(5);
    let settings = VIOFilterSettings::default();
    let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let suite = EuclideanSuite;
    let imu = stationary_imu(0.0);
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());

    // Propagate multiple steps
    for _ in 0..20 {
        eqf.integrate_riccati_fast(&suite, &imu, 0.005, &input_gain, &state_gain);
    }

    let n = xi0.dim();
    for i in 0..n {
        for j in 0..n {
            assert_abs_diff_eq!(eqf.sigma[(i, j)], eqf.sigma[(j, i)], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_riccati_positive_diagonal() {
    let xi0 = make_xi0_with_landmarks(5);
    let settings = VIOFilterSettings::default();
    let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let suite = EuclideanSuite;
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());

    // Use stationary IMU to avoid landmarks going behind camera
    let imu = stationary_imu(0.0);
    for step in 0..50 {
        eqf.integrate_observer_state(&imu, 0.005, true);
        eqf.integrate_riccati_fast(&suite, &imu, 0.005, &input_gain, &state_gain);

        let n = xi0.dim();
        for i in 0..n {
            assert!(
                eqf.sigma[(i, i)] > 0.0,
                "Negative diagonal at step={}, index={}: {}",
                step,
                i,
                eqf.sigma[(i, i)]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Vision Update
// ---------------------------------------------------------------------------

#[test]
fn test_vision_update_reduces_uncertainty() {
    let xi0 = make_xi0_with_landmarks(3);
    let settings = VIOFilterSettings::default();
    let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let suite = EuclideanSuite;
    let cam = make_pinhole();

    // Propagate covariance a bit first
    let imu = stationary_imu(0.0);
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());
    for _ in 0..10 {
        eqf.integrate_observer_state(&imu, 0.005, true);
        eqf.integrate_riccati_fast(&suite, &imu, 0.005, &input_gain, &state_gain);
    }

    let n = xi0.dim();
    let diag_before: Vec<f64> = (0..n).map(|i| eqf.sigma[(i, i)]).collect();

    // Generate perfect observations of all landmarks
    let est = eqf.state_estimate();
    let y_ids: Vec<u64> = est.camera_landmarks.iter().map(|lm| lm.id).collect();
    let mut y_coords = HashMap::new();
    for lm in &est.camera_landmarks {
        let proj = cam.project(&lm.p);
        y_coords.insert(lm.id, proj);
    }

    let output_gain = settings.output_gain_matrix(y_ids.len());
    eqf.perform_vision_update(&suite, &y_ids, &y_coords, &cam, &output_gain, true, false);

    // Landmark covariance should decrease
    let diag_after: Vec<f64> = (0..n).map(|i| eqf.sigma[(i, i)]).collect();
    let s = VIOSensorState::CDIM;
    for i in s..n {
        assert!(
            diag_after[i] <= diag_before[i] + 1e-12,
            "Landmark cov increased at idx {}: {} -> {}",
            i,
            diag_before[i],
            diag_after[i]
        );
    }
}

#[test]
fn test_vision_update_with_perfect_obs_no_state_jump() {
    let xi0 = make_xi0_with_landmarks(3);
    let settings = VIOFilterSettings::default();
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.001;
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    let suite = EuclideanSuite;
    let cam = make_pinhole();

    // Perfect observation from identity state => innovation = 0 => no state change
    let est = eqf.state_estimate();
    let y_ids: Vec<u64> = est.camera_landmarks.iter().map(|lm| lm.id).collect();
    let mut y_coords = HashMap::new();
    for lm in &est.camera_landmarks {
        y_coords.insert(lm.id, cam.project(&lm.p));
    }

    let output_gain = settings.output_gain_matrix(y_ids.len());
    eqf.perform_vision_update(&suite, &y_ids, &y_coords, &cam, &output_gain, true, false);

    let est_after = eqf.state_estimate();
    assert!(
        state_distance(&est, &est_after) < 1e-6,
        "State jumped on perfect observation: dist={}",
        state_distance(&est, &est_after)
    );
}

// ---------------------------------------------------------------------------
// 5. Landmark Management
// ---------------------------------------------------------------------------

#[test]
fn test_add_landmarks() {
    let xi0 = make_xi0_with_landmarks(2);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim()) * 0.1;
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    assert_eq!(eqf.xi0.camera_landmarks.len(), 2);
    assert_eq!(eqf.xi0.dim(), 21 + 6);

    let new_lms = vec![
        Landmark {
            p: Vector3::new(1.0, 2.0, 5.0),
            id: 100,
        },
        Landmark {
            p: Vector3::new(-1.0, 0.5, 3.0),
            id: 101,
        },
    ];
    let new_cov = DMatrix::<f64>::identity(6, 6) * 0.5;
    eqf.add_new_landmarks(new_lms, &new_cov);

    assert_eq!(eqf.xi0.camera_landmarks.len(), 4);
    assert_eq!(eqf.xi0.dim(), 21 + 12);
    assert_eq!(eqf.x.q.len(), 4);
    assert_eq!(eqf.x.id, vec![0, 1, 100, 101]);

    // New landmark covariance block
    let start = 21 + 6;
    assert_abs_diff_eq!(eqf.sigma[(start, start)], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(eqf.sigma[(start + 3, start + 3)], 0.5, epsilon = 1e-12);

    // Old block untouched
    assert_abs_diff_eq!(eqf.sigma[(0, 0)], 0.1, epsilon = 1e-12);
    assert_abs_diff_eq!(eqf.sigma[(21, 21)], 0.1, epsilon = 1e-12);
}

#[test]
fn test_remove_landmark_by_id() {
    let xi0 = make_xi0_with_landmarks(3);
    let n = xi0.dim();
    let mut init_cov = DMatrix::<f64>::identity(n, n);
    // Tag each landmark's diagonal block with unique values for verification
    for i in 0..3 {
        let start = 21 + 3 * i;
        for k in 0..3 {
            init_cov[(start + k, start + k)] = (i + 1) as f64;
        }
    }
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

    // Remove middle landmark (id=1)
    eqf.remove_landmark_by_id(1);
    assert_eq!(eqf.xi0.camera_landmarks.len(), 2);
    assert_eq!(eqf.x.id, vec![0, 2]);

    // Landmark 0 covariance should still be 1.0
    assert_abs_diff_eq!(eqf.sigma[(21, 21)], 1.0, epsilon = 1e-10);
    // Landmark 2 covariance (now at position 1) should be 3.0
    assert_abs_diff_eq!(eqf.sigma[(24, 24)], 3.0, epsilon = 1e-10);
}

#[test]
fn test_remove_nonexistent_landmark() {
    let xi0 = make_xi0_with_landmarks(2);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim());
    let mut eqf = VIOEqF::new(xi0, &init_cov);

    // Should be no-op
    eqf.remove_landmark_by_id(999);
    assert_eq!(eqf.xi0.camera_landmarks.len(), 2);
}

#[test]
fn test_remove_invalid_landmarks() {
    let xi0 = make_xi0_with_landmarks(3);
    let init_cov = DMatrix::<f64>::identity(xi0.dim(), xi0.dim());
    let mut eqf = VIOEqF::new(xi0, &init_cov);

    // Corrupt one landmark's scale
    eqf.x.q[1].scale = 1e-10; // too small

    eqf.remove_invalid_landmarks();
    assert_eq!(eqf.xi0.camera_landmarks.len(), 2);
    assert_eq!(eqf.x.id, vec![0, 2]);
}

#[test]
fn test_get_landmark_cov() {
    let xi0 = make_xi0_with_landmarks(2);
    let n = xi0.dim();
    let mut init_cov = DMatrix::<f64>::identity(n, n);
    let start = 21;
    for k in 0..3 {
        init_cov[(start + k, start + k)] = 42.0;
    }
    let eqf = VIOEqF::new(xi0, &init_cov);

    let cov = eqf.get_landmark_cov_by_id(0).unwrap();
    assert_abs_diff_eq!(cov[(0, 0)], 42.0, epsilon = 1e-10);

    assert!(eqf.get_landmark_cov_by_id(999).is_none());
}

// ---------------------------------------------------------------------------
// 6. VIOFilter IMU Processing
// ---------------------------------------------------------------------------

#[test]
fn test_filter_first_imu_sets_time() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    assert!(filter.eqf.current_time < 0.0);
    filter.process_imu(stationary_imu(1.0));
    assert_abs_diff_eq!(filter.eqf.current_time, 1.0, epsilon = 1e-12);
}

#[test]
fn test_filter_imu_time_monotonic() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    filter.process_imu(stationary_imu(1.0));
    filter.process_imu(stationary_imu(1.005));
    assert_abs_diff_eq!(filter.eqf.current_time, 1.005, epsilon = 1e-12);

    // Non-positive dt should be skipped
    filter.process_imu(stationary_imu(1.005));
    filter.process_imu(stationary_imu(1.003));
    assert_abs_diff_eq!(filter.eqf.current_time, 1.005, epsilon = 1e-12);
}

#[test]
fn test_filter_imu_stationary_stable() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(3);
    let mut filter = VIOFilter::new(settings, xi0);

    // Process 1 second of stationary IMU at 200Hz
    for i in 0..200 {
        let t = i as f64 * 0.005;
        filter.process_imu(stationary_imu(t));
    }

    let est = filter.state_estimate();
    assert!(
        est.sensor.pose.translation.norm() < 1e-4,
        "Stationary IMU caused position drift: {:?}",
        est.sensor.pose.translation
    );
    assert!(
        est.sensor.velocity.norm() < 1e-4,
        "Stationary IMU caused velocity drift: {:?}",
        est.sensor.velocity
    );
}

#[test]
fn test_filter_pending_imu_buffer_bounded() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    for i in 0..300 {
        filter.process_imu(stationary_imu(i as f64 * 0.005));
    }

    // Buffer drains when > 200 entries, keeping last 100
    // With 300 samples: first drain at 201 → 101, then grows to 200 (not > 200)
    assert!(
        filter.pending_imu.len() <= 200,
        "IMU buffer not bounded: len={}",
        filter.pending_imu.len()
    );
}

// ---------------------------------------------------------------------------
// 7. VIOFilter Vision Processing
// ---------------------------------------------------------------------------

#[test]
fn test_filter_vision_before_imu_is_noop() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    let meas = VisionMeasurement::new(0.0, HashMap::new());
    filter.process_vision(meas, &cam);
    assert_eq!(filter.vision_count, 0);
}

#[test]
fn test_filter_vision_adds_landmarks() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    // Vision with new features
    let mut coords = HashMap::new();
    coords.insert(10u64, Vector2::new(400.0f32, 250.0f32));
    coords.insert(11u64, Vector2::new(350.0f32, 200.0f32));
    let meas = VisionMeasurement::new(0.005, coords);
    filter.process_vision(meas, &cam);

    assert_eq!(filter.eqf.xi0.camera_landmarks.len(), 2);
    assert_eq!(filter.vision_count, 1);
}

#[test]
fn test_filter_vision_initializes_new_landmark_from_range_prior() {
    let mut settings = VIOFilterSettings::default();
    settings.initial_scene_depth = 3.0;
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    let mut coords = HashMap::new();
    coords.insert(42u64, Vector2::new(400.0f32, 250.0f32));
    let meas = VisionMeasurement::new(0.005, coords);

    let mut priors = HashMap::new();
    priors.insert(
        42u64,
        LandmarkDepthPrior {
            range: 7.0,
            range_var: 0.1,
        },
    );

    filter.process_vision_with_depth_priors(meas, &cam, &priors);

    let landmark = filter
        .eqf
        .xi0
        .camera_landmarks
        .iter()
        .find(|landmark| landmark.id == 42)
        .unwrap();
    assert_abs_diff_eq!(landmark.p.norm(), 7.0, epsilon = 1e-9);
}

#[test]
fn test_filter_vision_scene_depth_zero_disables_fallback_landmark_init() {
    let mut settings = VIOFilterSettings::default();
    settings.initial_scene_depth = 0.0;
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    let mut coords = HashMap::new();
    coords.insert(42u64, Vector2::new(400.0f32, 250.0f32));
    let meas = VisionMeasurement::new(0.005, coords);

    filter.process_vision(meas, &cam);

    assert!(filter.eqf.xi0.camera_landmarks.is_empty());
}

#[test]
fn test_filter_vision_removes_lost_landmarks() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(3);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    // Only observe landmark 1 (lose 0 and 2)
    let est = filter.state_estimate();
    let lm1 = &est.camera_landmarks[1];
    let proj = cam.project(&lm1.p);
    let mut coords = HashMap::new();
    coords.insert(1u64, Vector2::new(proj[0] as f32, proj[1] as f32));
    let meas = VisionMeasurement::new(0.005, coords);
    filter.process_vision(meas, &cam);

    // Only landmark 1 should remain
    assert_eq!(filter.eqf.xi0.camera_landmarks.len(), 1);
    assert_eq!(filter.eqf.x.id, vec![1]);
}

#[test]
fn test_filter_max_landmarks_respected() {
    let mut settings = VIOFilterSettings::default();
    settings.max_landmarks = 5;
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0);

    let cam = make_pinhole();
    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    // Try to add 10 features at once
    let mut coords = HashMap::new();
    for i in 0..10 {
        coords.insert(i as u64, Vector2::new(300.0f32 + i as f32 * 20.0, 200.0f32));
    }
    let meas = VisionMeasurement::new(0.005, coords);
    filter.process_vision(meas, &cam);

    assert!(
        filter.eqf.xi0.camera_landmarks.len() <= 5,
        "Exceeded max landmarks: {}",
        filter.eqf.xi0.camera_landmarks.len()
    );
}

// ---------------------------------------------------------------------------
// 8. Full Pipeline Integration
// ---------------------------------------------------------------------------

#[test]
fn test_filter_imu_then_vision_convergence() {
    // Simulate a simple scenario: stationary camera observing fixed landmarks
    // Start with no landmarks (they get added via vision)
    let settings = VIOFilterSettings::default();
    let cam = make_pinhole();
    let xi0 = make_xi0_with_landmarks(0);
    let mut filter = VIOFilter::new(settings, xi0.clone());

    // Ground truth landmarks in camera frame (used for generating observations)
    let gt_landmarks: Vec<(u64, Vector3<f64>)> = (0..5)
        .map(|i| {
            (
                i as u64,
                Vector3::new(
                    (i as f64 - 2.0) * 0.5,
                    ((i * 7 + 3) % 5) as f64 * 0.3,
                    3.0 + (i as f64) * 0.2,
                ),
            )
        })
        .collect();

    let dt_imu = 0.005;
    let mut t = 0.0;

    for _ in 0..5 {
        // 10 IMU samples per vision frame
        for _ in 0..10 {
            filter.process_imu(stationary_imu(t));
            t += dt_imu;
        }

        // Observe all ground truth landmarks
        let mut coords = HashMap::new();
        for (id, p) in &gt_landmarks {
            let proj = cam.project(p);
            coords.insert(*id, Vector2::new(proj[0] as f32, proj[1] as f32));
        }
        let meas = VisionMeasurement::new(t, coords);
        filter.process_vision(meas, &cam);
    }

    let final_est = filter.state_estimate();
    // Position should stay near zero (stationary)
    assert!(
        final_est.sensor.pose.translation.norm() < 1.0,
        "Position diverged: {:?}",
        final_est.sensor.pose.translation
    );
    // Velocity should stay near zero
    assert!(
        final_est.sensor.velocity.norm() < 1.0,
        "Velocity diverged: {:?}",
        final_est.sensor.velocity
    );
}

#[test]
fn test_filter_state_gain_invalidation() {
    let settings = VIOFilterSettings::default();
    let xi0 = make_xi0_with_landmarks(3);
    let mut filter = VIOFilter::new(settings, xi0.clone());
    let cam = make_pinhole();

    filter.process_imu(stationary_imu(0.0));
    filter.process_imu(stationary_imu(0.005));

    let gain_rows_before = filter.state_gain.nrows();
    assert_eq!(gain_rows_before, 21 + 9);

    // Vision that drops all existing landmarks and adds 2 new ones
    let mut coords = HashMap::new();
    coords.insert(100u64, Vector2::new(400.0f32, 250.0f32));
    coords.insert(101u64, Vector2::new(350.0f32, 200.0f32));
    let meas = VisionMeasurement::new(0.005, coords);
    filter.process_vision(meas, &cam);

    // State gain should have been recomputed for new landmark count
    let gain_rows_after = filter.state_gain.nrows();
    assert_eq!(gain_rows_after, 21 + 6); // 2 new landmarks
}

// ---------------------------------------------------------------------------
// 9. Vision Update Variants
// ---------------------------------------------------------------------------

#[test]
fn test_vision_update_discrete_vs_continuous() {
    // Both lift types should produce finite, reasonable results
    let xi0 = make_xi0_with_landmarks(3);
    let settings = VIOFilterSettings::default();
    let cam = make_pinhole();
    let suite = EuclideanSuite;

    for use_discrete in [false, true] {
        let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
        let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);

        // Add some covariance growth
        let imu = stationary_imu(0.0);
        let input_gain = settings.input_gain_matrix();
        let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());
        for _ in 0..10 {
            eqf.integrate_observer_state(&imu, 0.005, true);
            eqf.integrate_riccati_fast(&suite, &imu, 0.005, &input_gain, &state_gain);
        }

        // Slightly perturbed observations
        let est = eqf.state_estimate();
        let y_ids: Vec<u64> = est.camera_landmarks.iter().map(|lm| lm.id).collect();
        let mut y_coords = HashMap::new();
        for lm in &est.camera_landmarks {
            let proj = cam.project(&lm.p);
            y_coords.insert(lm.id, proj + Vector2::new(0.5, -0.3));
        }

        let output_gain = settings.output_gain_matrix(y_ids.len());
        eqf.perform_vision_update(
            &suite,
            &y_ids,
            &y_coords,
            &cam,
            &output_gain,
            true,
            use_discrete,
        );

        let est_after = eqf.state_estimate();
        assert!(
            est_after.sensor.pose.translation.norm() < 100.0,
            "Diverged with use_discrete={}: pos={:?}",
            use_discrete,
            est_after.sensor.pose.translation
        );
        assert!(
            est_after.sensor.velocity.norm() < 100.0,
            "Diverged with use_discrete={}: vel={:?}",
            use_discrete,
            est_after.sensor.velocity
        );
    }
}

#[test]
fn test_empty_vision_update_is_noop() {
    let xi0 = make_xi0_with_landmarks(3);
    let settings = VIOFilterSettings::default();
    let init_cov = settings.initial_covariance(xi0.camera_landmarks.len());
    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);
    let suite = EuclideanSuite;
    let cam = make_pinhole();

    let est_before = eqf.state_estimate();

    let output_gain = settings.output_gain_matrix(0);
    eqf.perform_vision_update(
        &suite,
        &[],
        &HashMap::new(),
        &cam,
        &output_gain,
        true,
        false,
    );

    let est_after = eqf.state_estimate();
    assert!(state_distance(&est_before, &est_after) < 1e-15);
}

// ---------------------------------------------------------------------------
// 10. Settings / Gain Matrices
// ---------------------------------------------------------------------------

#[test]
fn test_settings_input_gain_symmetric_positive() {
    let settings = VIOFilterSettings::default();
    let q = settings.input_gain_matrix();

    for i in 0..12 {
        assert!(q[(i, i)] >= 0.0, "Negative diagonal in input gain at {}", i);
        for j in 0..12 {
            assert_abs_diff_eq!(q[(i, j)], q[(j, i)], epsilon = 1e-15);
        }
    }
}

#[test]
fn test_settings_state_gain_grows_with_landmarks() {
    let settings = VIOFilterSettings::default();
    let q0 = settings.state_gain_matrix(0);
    let q5 = settings.state_gain_matrix(5);

    assert_eq!(q0.nrows(), 21);
    assert_eq!(q5.nrows(), 21 + 15);

    // Landmark diagonal entries should use process_point
    for i in 0..5 {
        let start = 21 + 3 * i;
        for k in 0..3 {
            assert_abs_diff_eq!(
                q5[(start + k, start + k)],
                settings.process_point,
                epsilon = 1e-15
            );
        }
    }
}

#[test]
fn test_settings_initial_covariance_structure() {
    let settings = VIOFilterSettings::default();
    let cov = settings.initial_covariance(3);

    assert_eq!(cov.nrows(), 21 + 9);

    // Bias gyro block
    for k in 0..3 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_bias_omega_variance,
            epsilon = 1e-15
        );
    }
    // Bias accel block
    for k in 3..6 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_bias_accel_variance,
            epsilon = 1e-15
        );
    }
    // Attitude block
    for k in 6..9 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_attitude_variance,
            epsilon = 1e-15
        );
    }
    // Position block
    for k in 9..12 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_position_variance,
            epsilon = 1e-15
        );
    }
    // Velocity block
    for k in 12..15 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_velocity_variance,
            epsilon = 1e-15
        );
    }
    // Camera attitude block
    for k in 15..18 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_camera_attitude_variance,
            epsilon = 1e-15
        );
    }
    // Camera position block
    for k in 18..21 {
        assert_abs_diff_eq!(
            cov[(k, k)],
            settings.initial_camera_position_variance,
            epsilon = 1e-15
        );
    }
    // Landmark blocks
    for i in 0..3 {
        let start = 21 + 3 * i;
        for k in 0..3 {
            assert_abs_diff_eq!(
                cov[(start + k, start + k)],
                settings.initial_point_variance,
                epsilon = 1e-15
            );
        }
    }
}

#[test]
fn test_settings_output_gain_diagonal() {
    let settings = VIOFilterSettings::default();
    let r = settings.output_gain_matrix(4);
    assert_eq!(r.nrows(), 8);
    for i in 0..8 {
        assert_abs_diff_eq!(r[(i, i)], settings.sigma_bearing.powi(2), epsilon = 1e-15);
    }
}

// ---------------------------------------------------------------------------
// `Faster` Riccati variant (Phase 6)
// ---------------------------------------------------------------------------

/// A single-sample `Faster` flush must reproduce one `Fast` step bit-for-bit:
/// with one accumulated sample Φ = F and Q is built from that sample's B, so
/// `accumulate_transition` + `flush_riccati` == `integrate_riccati_fast`.
/// This is the regression guard for the accumulate/flush machinery.
#[test]
fn test_faster_single_sample_matches_fast() {
    let xi0 = make_xi0_with_landmarks(5);
    let settings = VIOFilterSettings::default();
    let n = xi0.dim();
    let init_cov = DMatrix::<f64>::identity(n, n) * 0.3;
    let suite = EuclideanSuite;
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());
    let imu = IMUVelocity::new(
        0.0,
        Vector3::new(0.13, -0.07, 0.21),
        Vector3::new(0.4, -0.25, GRAVITY_CONSTANT + 0.15),
    );
    let dt = 0.005;

    let mut eqf_fast = VIOEqF::new(xi0.clone(), &init_cov);
    eqf_fast.integrate_riccati_fast(&suite, &imu, dt, &input_gain, &state_gain);

    let mut eqf_faster = VIOEqF::new(xi0.clone(), &init_cov);
    eqf_faster.accumulate_transition(&suite, &imu, dt);
    eqf_faster.flush_riccati(&input_gain, &state_gain);

    assert_eq!(eqf_fast.sigma.nrows(), eqf_faster.sigma.nrows());
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                eqf_fast.sigma[(i, j)],
                eqf_faster.sigma[(i, j)],
                "Fast vs Faster mismatch at ({i}, {j})"
            );
        }
    }
}

/// `flush_riccati` with nothing accumulated is a no-op (safe to call under the
/// `Fast` variant, where `process_vision` flushes unconditionally).
#[test]
fn test_faster_flush_empty_is_noop() {
    let xi0 = make_xi0_with_landmarks(4);
    let settings = VIOFilterSettings::default();
    let n = xi0.dim();
    let init_cov = DMatrix::<f64>::identity(n, n) * 0.2;
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());

    let mut eqf = VIOEqF::new(xi0.clone(), &init_cov);
    eqf.flush_riccati(&input_gain, &state_gain);
    for i in 0..n {
        for j in 0..n {
            assert_eq!(eqf.sigma[(i, j)], init_cov[(i, j)]);
        }
    }
}

/// A multi-sample `Faster` flush composes the transitions across the sub-frame:
/// it stays symmetric, finite, and positive on the diagonal — and differs from
/// the per-sample `Fast` path (the bounded process-noise approximation).
#[test]
fn test_faster_multi_sample_well_formed() {
    let xi0 = make_xi0_with_landmarks(5);
    let settings = VIOFilterSettings::default();
    let n = xi0.dim();
    let init_cov = DMatrix::<f64>::identity(n, n) * 0.3;
    let suite = EuclideanSuite;
    let input_gain = settings.input_gain_matrix();
    let state_gain = settings.state_gain_matrix(xi0.camera_landmarks.len());
    let imu = IMUVelocity::new(
        0.0,
        Vector3::new(0.13, -0.07, 0.21),
        Vector3::new(0.4, -0.25, GRAVITY_CONSTANT + 0.15),
    );
    let dt = 0.005;

    let mut eqf_fast = VIOEqF::new(xi0.clone(), &init_cov);
    let mut eqf_faster = VIOEqF::new(xi0.clone(), &init_cov);
    for _ in 0..10 {
        eqf_fast.integrate_riccati_fast(&suite, &imu, dt, &input_gain, &state_gain);
        eqf_faster.accumulate_transition(&suite, &imu, dt);
    }
    eqf_faster.flush_riccati(&input_gain, &state_gain);

    let mut max_dev = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = eqf_faster.sigma[(i, j)];
            assert!(v.is_finite(), "non-finite at ({i}, {j})");
            assert_abs_diff_eq!(v, eqf_faster.sigma[(j, i)], epsilon = 1e-12);
            max_dev = max_dev.max((v - eqf_fast.sigma[(i, j)]).abs());
        }
        assert!(
            eqf_faster.sigma[(i, i)] > 0.0,
            "non-positive diagonal at {i}"
        );
    }
    // Faster is a distinct, more-approximate variant — it must differ from Fast.
    assert!(
        max_dev > 0.0,
        "Faster should not be identical to Fast over 10 samples"
    );
}

#[test]
#[ignore = "single-trial covariance-growth trace; run with --ignored --nocapture"]
fn diag_eqvio_far_landmark_cov_trace() {
    // One trial, NormalSuite, depth 640. Print per-step covariance health to see
    // whether the SPD failure is a gradual propagation blow-up or a sudden vision
    // event, and whether it originates in the sensor block (shared, coordinate-
    // independent) or the landmark block.
    let depth = 640.0_f64;
    let n_landmarks = 40;
    let dt = 0.2_f64;
    let sigma_px = 0.5_f64;
    let n_steps = 45;
    // sensor-block init covariance (diagonal), tunable to test whether the
    // zero-init harness accelerates the offset-gauge drift; SENSOR_INIT_VAR=0
    // reproduces the original harness.
    let sensor_init_var: f64 = std::env::var("SENSOR_INIT_VAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    eprintln!("SENSOR_INIT_VAR = {sensor_init_var:.1e}");
    let suite = NormalSuite::new();
    let cam = make_pinhole();
    let s = VIOSensorState::CDIM;
    let mut rng = StdRng::seed_from_u64(0xE0F0_0000_u64 ^ depth.to_bits());
    let pixel_noise = Normal::new(0.0, sigma_px).unwrap();

    let mut landmarks = Vec::with_capacity(n_landmarks);
    for i in 0..n_landmarks {
        let col = (i % 10) as f64;
        let row = (i / 10) as f64;
        let x_frac = 0.055 + 0.010 * col;
        let y_frac = -0.045 + 0.030 * row;
        landmarks.push(Landmark {
            p: Vector3::new(x_frac * depth, y_frac * depth, depth),
            id: i as u64,
        });
    }
    let truth0 = VIOState::new(
        VIOSensorState {
            input_bias: Vector6::zeros(),
            pose: SE3::identity(),
            velocity: Vector3::new(1.0, 0.0, 0.0),
            camera_offset: SE3::identity(),
        },
        landmarks,
    );
    let init_std = Vector3::<f64>::new(0.0005, 0.0005, 0.002);
    let init_noise = [
        Normal::new(0.0, init_std[0]).unwrap(),
        Normal::new(0.0, init_std[1]).unwrap(),
        Normal::new(0.0, init_std[2]).unwrap(),
    ];
    let mut init_eps = DVector::<f64>::zeros(truth0.dim());
    for i in 0..n_landmarks {
        init_eps[s + 3 * i] = init_noise[0].sample(&mut rng);
        init_eps[s + 3 * i + 1] = init_noise[1].sample(&mut rng);
        init_eps[s + 3 * i + 2] = init_noise[2].sample(&mut rng);
    }
    let xi0_est = suite.state_chart_inv(&init_eps, &truth0);
    let mut init_cov = DMatrix::<f64>::zeros(truth0.dim(), truth0.dim());
    for i in 0..s {
        init_cov[(i, i)] = sensor_init_var;
    }
    for i in 0..n_landmarks {
        for k in 0..3 {
            init_cov[(s + 3 * i + k, s + 3 * i + k)] = init_std[k].powi(2);
        }
    }
    let mut eqf = VIOEqF::new(xi0_est, &init_cov);
    let settings = VIOFilterSettings::default();
    let input_gain = SMatrix::<f64, 12, 12>::zeros();
    let state_gain = settings.state_gain_matrix(truth0.camera_landmarks.len());
    let output_gain = DMatrix::<f64>::identity(2 * n_landmarks, 2 * n_landmarks) * sigma_px.powi(2);
    let imu = IMUVelocity::new(0.0, Vector3::zeros(), Vector3::new(0.0, 0.0, GRAVITY_CONSTANT));
    let mut truth = truth0;

    let block_stats = |sigma: &DMatrix<f64>, lo: usize, hi: usize| -> (f64, f64, usize) {
        let mut maxd = f64::MIN;
        let mut argmax = lo;
        for i in lo..hi {
            if sigma[(i, i)] > maxd {
                maxd = sigma[(i, i)];
                argmax = i;
            }
        }
        let mut mind = f64::MAX;
        for i in lo..hi {
            mind = mind.min(sigma[(i, i)]);
        }
        (maxd, mind, argmax)
    };
    let min_eig = |sigma: &DMatrix<f64>| -> f64 {
        let sym = (sigma + sigma.transpose()) * 0.5;
        sym.symmetric_eigenvalues()
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min)
    };

    println!("step phase | sensor[max_diag min_diag] lm[max_diag argmax] min_eig");
    for step in 0..n_steps {
        truth = crate::mathematical::vio_state::integrate_system_function(&truth, &imu, dt);
        eqf.integrate_observer_state(&imu, dt, true);
        // pre-propagation diagnostics: A-block magnitudes and Σ condition
        let blocks = suite.propagation_blocks(&eqf.x, &eqf.xi0, &imu);
        let max_a_ss = blocks.a_ss.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let max_a_lmlm = blocks
            .a_lm_lm
            .iter()
            .flat_map(|m| m.iter())
            .fold(0.0_f64, |a, &v| a.max(v.abs()));
        let max_a_lms = blocks.a_lm_s.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let est = eqf.state_estimate();
        let min_lm_z = est
            .camera_landmarks
            .iter()
            .map(|l| l.p[2])
            .fold(f64::MAX, f64::min);
        let min_lm_norm = est
            .camera_landmarks
            .iter()
            .map(|l| l.p.norm())
            .fold(f64::MAX, f64::min);
        let vel = est.sensor.velocity.norm();
        let max_scale = eqf
            .x
            .q
            .iter()
            .map(|q| q.scale.abs())
            .fold(0.0_f64, f64::max);
        let min_scale = eqf
            .x
            .q
            .iter()
            .map(|q| q.scale.abs())
            .fold(f64::MAX, f64::min);
        let adj_b = eqf.x.b.adjoint().norm();
        let me_pre = min_eig(&eqf.sigma);
        let cond = block_stats(&eqf.sigma, 0, eqf.sigma.nrows()).0 / me_pre.abs().max(1e-300);
        eqf.integrate_riccati_fast(&suite, &imu, dt, &input_gain, &state_gain);
        let (lmax, _, larg) = block_stats(&eqf.sigma, s, eqf.sigma.nrows());
        let me_prop = if eqf.sigma.iter().all(|v| v.is_finite()) {
            min_eig(&eqf.sigma)
        } else {
            f64::NAN
        };
        println!(
            "{step:3} prop  | est[min_z={min_lm_z:.2e} |v|={vel:.2e}] grp[scale {min_scale:.2e}..{max_scale:.2e} adjB={adj_b:.1e}] maxA[lmlm={max_a_lmlm:.1e} lms={max_a_lms:.1e}] -> lm[{lmax:.2e}] min_eig={me_prop:.3e}",
        );
        let _ = (max_a_ss, me_pre, min_lm_norm, larg, cond);

        let mut y_ids = Vec::with_capacity(n_landmarks);
        let mut y_coords = HashMap::with_capacity(n_landmarks);
        for lm in &truth.camera_landmarks {
            if lm.p[2] <= 1e-6 {
                continue;
            }
            let mut uv = cam.project(&lm.p);
            uv[0] += pixel_noise.sample(&mut rng);
            uv[1] += pixel_noise.sample(&mut rng);
            y_ids.push(lm.id);
            y_coords.insert(lm.id, uv);
        }
        eqf.perform_vision_update(&suite, &y_ids, &y_coords, &cam, &output_gain, true, false);
        let (smax, smin, _) = block_stats(&eqf.sigma, 0, s);
        let (lmax, _, larg) = block_stats(&eqf.sigma, s, eqf.sigma.nrows());
        let me_vis = if eqf.sigma.iter().all(|v| v.is_finite()) {
            min_eig(&eqf.sigma)
        } else {
            f64::NAN
        };
        println!(
            "{step:3} vis   | sensor[{smax:.2e} {smin:.2e}] lm[{lmax:.2e} #{}] min_eig={me_vis:.3e}",
            (larg - s) / 3
        );
        if !eqf.sigma.iter().all(|v| v.is_finite()) {
            println!("  -> nonfinite covariance at step {step}, stopping");
            break;
        }
    }
}

#[test]
#[ignore = "diagnostic Monte Carlo; run with --ignored --nocapture"]
fn diag_eqvio_far_landmark_nees_depth_sweep() {
    let depths = [320.0, 640.0, 1280.0];
    let n_mc = 8;
    let n_steps = 120;
    let n_landmarks = 40;
    let dt = 0.2;
    let sigma_px = 0.5;

    // Coordinate-suite comparison. The init covariance is matched *physically*
    // across suites (bearing ~0.5 mrad, range ~0.2 % of depth), because the same
    // raw numbers mean wildly different things in each chart's third (depth)
    // coordinate: log-range (Normal) is δr/r, inverse-depth (InvDepth) is
    // δρ = ρ·δr/r = (δr/r)/depth, and Euclidean is δr (and bearing·depth in x,y).
    // Without this matching the comparison would be confounded by the init alone.
    let bear = 0.0005_f64; // physical bearing init std [rad]
    let frange = 0.002_f64; // physical fractional-range init std
    let suites: [&str; 3] = ["normal", "invdepth", "euclid"];

    println!("EqVIO coordinate-suite far-landmark SPD/NEES diagnostic");
    println!("known constant lateral motion, 40 in-state landmarks, noisy pixels");
    println!("suite, variant, depth_m, valid_trials, finite_landmarks, mean_nees, median_nees, mean_range_rel_err, nonfinite_sigma, nan_sigma, inf_sigma, prop_fail, vision_fail, min_fail_step, median_fail_step, non_spd_sigma, spd_prop_fail, spd_vision_fail, min_spd_step, median_spd_step, min_diag_at_spd_fail, max_asym_at_spd_fail, nonpos_alpha, min_alpha, min_alpha_at_spd_fail, nonfinite_chart, singular_cov, no_finite_landmarks");

    let total_rows = suites.len() * 2 * depths.len();
    let sweep_start = Instant::now();
    let mut row = 0usize;
    for suite_name in suites {
        for use_faster in [false, true] {
            let variant = if use_faster { "faster" } else { "fast" };
            for depth in depths {
                row += 1;
                // physically-matched per-suite init std in that chart's coords
                let init_std = match suite_name {
                    "normal" => Vector3::new(bear, bear, frange),
                    "invdepth" => Vector3::new(bear, bear, frange / depth),
                    "euclid" => Vector3::new(bear * depth, bear * depth, frange * depth),
                    _ => unreachable!(),
                };
                let suite: Box<dyn EqFCoordinateSuite> = match suite_name {
                    "normal" => Box::new(NormalSuite::new()),
                    "invdepth" => Box::new(InvDepthSuite::new()),
                    "euclid" => Box::new(EuclideanSuite),
                    _ => unreachable!(),
                };
                let elapsed = sweep_start.elapsed().as_secs_f64();
                let eta = elapsed / (row - 1).max(1) as f64 * (total_rows - row + 1) as f64;
                eprintln!(
                    "=== row {row}/{total_rows}: suite={suite_name} variant={variant} depth={depth:.0} | overall elapsed={} eta~{} ===",
                    format_seconds(elapsed),
                    format_seconds(if row == 1 { 0.0 } else { eta })
                );
                let progress_label = format!("{suite_name} {variant} depth={depth:.0}");
                let stats = eqvio_far_landmark_nees_for_depth(
                    suite.as_ref(),
                    init_std,
                    depth,
                    n_mc,
                    n_steps,
                    n_landmarks,
                    dt,
                    sigma_px,
                    use_faster,
                    &progress_label,
                );
                println!(
                    "{suite_name}, {variant}, {depth:.0}, {}/{}, {}, {:.3}, {:.3}, {:.4e}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.4e}, {:.4e}, {}, {:.4e}, {:.4e}, {}, {}, {}",
                    stats.valid_trials,
                stats.total_trials,
                stats.finite_landmarks,
                stats.mean_nees,
                stats.median_nees,
                stats.mean_range_rel_err,
                stats.nonfinite_sigma_trials,
                stats.nan_sigma_trials,
                stats.inf_sigma_trials,
                stats.propagation_sigma_failures,
                stats.vision_sigma_failures,
                stats.min_sigma_failure_step
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                stats.median_sigma_failure_step
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                stats.non_spd_sigma_trials,
                stats.propagation_spd_failures,
                stats.vision_spd_failures,
                stats.min_spd_failure_step
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                stats.median_spd_failure_step
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                stats.min_diag_at_first_spd_failure,
                stats.max_asym_at_first_spd_failure,
                stats.nonpositive_alpha_trials,
                stats.min_alpha,
                stats.min_alpha_at_first_spd_failure,
                stats.nonfinite_chart_trials,
                stats.singular_cov_trials,
                stats.no_finite_landmark_trials
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EqvioFarLandmarkStats {
    valid_trials: usize,
    total_trials: usize,
    finite_landmarks: usize,
    mean_nees: f64,
    median_nees: f64,
    mean_range_rel_err: f64,
    nonfinite_sigma_trials: usize,
    nan_sigma_trials: usize,
    inf_sigma_trials: usize,
    propagation_sigma_failures: usize,
    vision_sigma_failures: usize,
    min_sigma_failure_step: Option<usize>,
    median_sigma_failure_step: Option<usize>,
    non_spd_sigma_trials: usize,
    propagation_spd_failures: usize,
    vision_spd_failures: usize,
    min_spd_failure_step: Option<usize>,
    median_spd_failure_step: Option<usize>,
    min_diag_at_first_spd_failure: f64,
    max_asym_at_first_spd_failure: f64,
    nonpositive_alpha_trials: usize,
    min_alpha: f64,
    min_alpha_at_first_spd_failure: f64,
    nonfinite_chart_trials: usize,
    singular_cov_trials: usize,
    no_finite_landmark_trials: usize,
}

#[allow(clippy::too_many_arguments)]
fn eqvio_far_landmark_nees_for_depth(
    suite: &dyn EqFCoordinateSuite,
    init_std: Vector3<f64>,
    depth: f64,
    n_mc: usize,
    n_steps: usize,
    n_landmarks: usize,
    dt: f64,
    sigma_px: f64,
    use_faster_riccati: bool,
    progress_label: &str,
) -> EqvioFarLandmarkStats {
    let cam = make_pinhole();
    let s = VIOSensorState::CDIM;
    let mut rng = StdRng::seed_from_u64(0xE0F0_0000_u64 ^ depth.to_bits());
    let pixel_noise = Normal::new(0.0, sigma_px).unwrap();
    // physically-matched per-suite init (see caller); std per chart coordinate
    let init_noise = [
        Normal::new(0.0, init_std[0].max(1e-12)).unwrap(),
        Normal::new(0.0, init_std[1].max(1e-12)).unwrap(),
        Normal::new(0.0, init_std[2].max(1e-12)).unwrap(),
    ];
    let init_lm_cov = Vector3::new(
        init_std[0].powi(2),
        init_std[1].powi(2),
        init_std[2].powi(2),
    );
    let imu = IMUVelocity::new(
        0.0,
        Vector3::zeros(),
        Vector3::new(0.0, 0.0, GRAVITY_CONSTANT),
    );
    let settings = VIOFilterSettings::default();
    let input_gain = SMatrix::<f64, 12, 12>::zeros();
    let output_gain = DMatrix::<f64>::identity(2 * n_landmarks, 2 * n_landmarks) * sigma_px.powi(2);

    let mut nees = Vec::with_capacity(n_mc);
    let mut range_rel_err = Vec::with_capacity(n_mc);
    let mut valid_trials = 0;
    let mut nonfinite_sigma_trials = 0;
    let mut nonfinite_chart_trials = 0;
    let mut singular_cov_trials = 0;
    let mut no_finite_landmark_trials = 0;
    let mut propagation_sigma_failures = 0;
    let mut vision_sigma_failures = 0;
    let mut nan_sigma_trials = 0;
    let mut inf_sigma_trials = 0;
    let mut sigma_failure_steps = Vec::new();
    let mut non_spd_sigma_trials = 0;
    let mut propagation_spd_failures = 0;
    let mut vision_spd_failures = 0;
    let mut spd_failure_steps = Vec::new();
    let mut min_diag_at_spd_failures = Vec::new();
    let mut max_asym_at_spd_failures = Vec::new();
    let mut nonpositive_alpha_trials = 0;
    let mut min_alphas = Vec::new();
    let mut min_alphas_at_spd_failures = Vec::new();
    let row_start = Instant::now();
    for trial_idx in 0..n_mc {
        let mut landmarks = Vec::with_capacity(n_landmarks);
        for i in 0..n_landmarks {
            let col = (i % 10) as f64;
            let row = (i / 10) as f64;
            let x_frac = 0.055 + 0.010 * col;
            let y_frac = -0.045 + 0.030 * row;
            landmarks.push(Landmark {
                p: Vector3::new(x_frac * depth, y_frac * depth, depth),
                id: i as u64,
            });
        }
        let truth0 = VIOState::new(
            VIOSensorState {
                input_bias: Vector6::zeros(),
                pose: SE3::identity(),
                velocity: Vector3::new(1.0, 0.0, 0.0),
                camera_offset: SE3::identity(),
            },
            landmarks,
        );

        let mut init_eps = DVector::<f64>::zeros(truth0.dim());
        for i in 0..n_landmarks {
            init_eps[s + 3 * i] = init_noise[0].sample(&mut rng);
            init_eps[s + 3 * i + 1] = init_noise[1].sample(&mut rng);
            init_eps[s + 3 * i + 2] = init_noise[2].sample(&mut rng);
        }
        let xi0_est = suite.state_chart_inv(&init_eps, &truth0);

        let mut init_cov = DMatrix::<f64>::zeros(truth0.dim(), truth0.dim());
        for i in 0..n_landmarks {
            for k in 0..3 {
                init_cov[(s + 3 * i + k, s + 3 * i + k)] = init_lm_cov[k];
            }
        }
        let mut eqf = VIOEqF::new(xi0_est, &init_cov);
        let state_gain = settings.state_gain_matrix(truth0.camera_landmarks.len());
        let mut truth = truth0;
        let mut sigma_failure: Option<(usize, &'static str, &'static str)> = None;
        let mut spd_failure: Option<(usize, &'static str, SigmaSpdFailure)> = None;
        let mut alpha_failure = false;

        for step in 0..n_steps {
            truth = crate::mathematical::vio_state::integrate_system_function(&truth, &imu, dt);
            eqf.integrate_observer_state(&imu, dt, true);
            if use_faster_riccati {
                eqf.accumulate_transition(suite, &imu, dt);
                eqf.flush_riccati(&input_gain, &state_gain);
            } else {
                eqf.integrate_riccati_fast(suite, &imu, dt, &input_gain, &state_gain);
            }
            if let Some(kind) = sigma_nonfinite_kind(&eqf.sigma) {
                sigma_failure = Some((step + 1, "propagation", kind));
                break;
            }
            if spd_failure.is_none() {
                if let Some(health) = sigma_spd_failure(&eqf.sigma) {
                    spd_failure = Some((step + 1, "propagation", health));
                }
            }

            let mut y_ids = Vec::with_capacity(n_landmarks);
            let mut y_coords = HashMap::with_capacity(n_landmarks);
            for lm in &truth.camera_landmarks {
                if lm.p[2] <= 1e-6 {
                    continue;
                }
                let mut uv = cam.project(&lm.p);
                uv[0] += pixel_noise.sample(&mut rng);
                uv[1] += pixel_noise.sample(&mut rng);
                y_ids.push(lm.id);
                y_coords.insert(lm.id, uv);
            }
            let ct =
                suite.output_matrix_C(&eqf.xi0, &eqf.x, &y_ids, &y_coords, &cam, true);
            let alpha_health = scalar_update_alpha_health(&ct, &output_gain, &eqf.sigma);
            if let Some(alpha) = alpha_health.min_alpha {
                min_alphas.push(alpha);
                if spd_failure.is_none() && alpha_health.first_nonpositive_row.is_some() {
                    alpha_failure = true;
                }
            }
            eqf.perform_vision_update(suite, &y_ids, &y_coords, &cam, &output_gain, true, false);
            if let Some(kind) = sigma_nonfinite_kind(&eqf.sigma) {
                sigma_failure = Some((step + 1, "vision", kind));
                break;
            }
            if spd_failure.is_none() {
                if let Some(health) = sigma_spd_failure(&eqf.sigma) {
                    spd_failure = Some((step + 1, "vision", health));
                    if let Some(alpha) = alpha_health.min_alpha {
                        min_alphas_at_spd_failures.push(alpha);
                    }
                }
            }
        }

        if alpha_failure {
            nonpositive_alpha_trials += 1;
        }
        if let Some((step, phase, health)) = spd_failure {
            non_spd_sigma_trials += 1;
            spd_failure_steps.push(step);
            min_diag_at_spd_failures.push(health.min_diag);
            max_asym_at_spd_failures.push(health.max_asym);
            match phase {
                "propagation" => propagation_spd_failures += 1,
                "vision" => vision_spd_failures += 1,
                _ => {}
            }
        }
        if let Some((step, phase, kind)) = sigma_failure {
            nonfinite_sigma_trials += 1;
            sigma_failure_steps.push(step);
            match kind {
                "nan" => nan_sigma_trials += 1,
                "inf" => inf_sigma_trials += 1,
                _ => {}
            }
            match phase {
                "propagation" => propagation_sigma_failures += 1,
                "vision" => vision_sigma_failures += 1,
                _ => {}
            }
            print_diag_progress(progress_label, trial_idx + 1, n_mc, row_start);
            continue;
        }
        let err_state = state_group_action(&eqf.x.inverse(), &truth);
        let eps = suite.state_chart(&err_state, &eqf.xi0);
        if !eps.iter().all(|v| v.is_finite()) {
            nonfinite_chart_trials += 1;
            print_diag_progress(progress_label, trial_idx + 1, n_mc, row_start);
            continue;
        }
        let est = eqf.state_estimate();
        let mut trial_finite_landmarks = 0;
        let mut trial_singular_cov = false;
        for (i, lm_true) in truth.camera_landmarks.iter().enumerate() {
            let e_lm = eps.fixed_rows::<3>(s + 3 * i).into_owned();
            let Some(p_lm) = eqf.get_landmark_cov_by_id(lm_true.id) else {
                continue;
            };
            if !p_lm.iter().all(|v| v.is_finite()) || !e_lm.iter().all(|v| v.is_finite()) {
                continue;
            }
            let Some(p_lm_inv) =
                (p_lm + SMatrix::<f64, 3, 3>::identity() * 1e-10).try_inverse()
            else {
                trial_singular_cov = true;
                continue;
            };
            let trial_nees = (e_lm.transpose() * p_lm_inv * e_lm)[(0, 0)];
            if trial_nees.is_finite() {
                nees.push(trial_nees);
                trial_finite_landmarks += 1;
            }

            let true_range = lm_true.p.norm();
            let est_range = est.camera_landmarks[i].p.norm();
            let rel_err = (est_range - true_range).abs() / true_range.max(1e-12);
            if rel_err.is_finite() {
                range_rel_err.push(rel_err);
            }
        }
        if trial_finite_landmarks == 0 {
            if trial_singular_cov {
                singular_cov_trials += 1;
            } else {
                no_finite_landmark_trials += 1;
            }
        } else {
            valid_trials += 1;
        }
        print_diag_progress(progress_label, trial_idx + 1, n_mc, row_start);
    }

    if nees.is_empty() || range_rel_err.is_empty() {
        return EqvioFarLandmarkStats {
            valid_trials,
            total_trials: n_mc,
            finite_landmarks: nees.len(),
            mean_nees: f64::NAN,
            median_nees: f64::NAN,
            mean_range_rel_err: f64::NAN,
            nonfinite_sigma_trials,
            nan_sigma_trials,
            inf_sigma_trials,
            propagation_sigma_failures,
            vision_sigma_failures,
            min_sigma_failure_step: min_step(&sigma_failure_steps),
            median_sigma_failure_step: median_step(&mut sigma_failure_steps),
            non_spd_sigma_trials,
            propagation_spd_failures,
            vision_spd_failures,
            min_spd_failure_step: min_step(&spd_failure_steps),
            median_spd_failure_step: median_step(&mut spd_failure_steps),
            min_diag_at_first_spd_failure: min_finite(&min_diag_at_spd_failures),
            max_asym_at_first_spd_failure: max_finite(&max_asym_at_spd_failures),
            nonpositive_alpha_trials,
            min_alpha: min_finite(&min_alphas),
            min_alpha_at_first_spd_failure: min_finite(&min_alphas_at_spd_failures),
            nonfinite_chart_trials,
            singular_cov_trials,
            no_finite_landmark_trials,
        };
    }
    nees.sort_by(|a, b| a.total_cmp(b));
    let mean_nees = nees.iter().sum::<f64>() / nees.len() as f64;
    let median_nees = nees[nees.len() / 2];
    let mean_range_rel_err = range_rel_err.iter().sum::<f64>() / range_rel_err.len() as f64;
    EqvioFarLandmarkStats {
        valid_trials,
        total_trials: n_mc,
        finite_landmarks: nees.len(),
        mean_nees,
        median_nees,
        mean_range_rel_err,
        nonfinite_sigma_trials,
        nan_sigma_trials,
        inf_sigma_trials,
        propagation_sigma_failures,
        vision_sigma_failures,
        min_sigma_failure_step: min_step(&sigma_failure_steps),
        median_sigma_failure_step: median_step(&mut sigma_failure_steps),
        non_spd_sigma_trials,
        propagation_spd_failures,
        vision_spd_failures,
        min_spd_failure_step: min_step(&spd_failure_steps),
        median_spd_failure_step: median_step(&mut spd_failure_steps),
        min_diag_at_first_spd_failure: min_finite(&min_diag_at_spd_failures),
        max_asym_at_first_spd_failure: max_finite(&max_asym_at_spd_failures),
        nonpositive_alpha_trials,
        min_alpha: min_finite(&min_alphas),
        min_alpha_at_first_spd_failure: min_finite(&min_alphas_at_spd_failures),
        nonfinite_chart_trials,
        singular_cov_trials,
        no_finite_landmark_trials,
    }
}

fn sigma_nonfinite_kind(sigma: &DMatrix<f64>) -> Option<&'static str> {
    if sigma.iter().any(|v| v.is_nan()) {
        Some("nan")
    } else if sigma.iter().any(|v| v.is_infinite()) {
        Some("inf")
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct SigmaSpdFailure {
    min_diag: f64,
    max_asym: f64,
}

#[derive(Debug, Clone, Copy)]
struct ScalarUpdateAlphaHealth {
    min_alpha: Option<f64>,
    first_nonpositive_row: Option<usize>,
}

fn scalar_update_alpha_health(
    c_star: &DMatrix<f64>,
    r_noise: &DMatrix<f64>,
    sigma: &DMatrix<f64>,
) -> ScalarUpdateAlphaHealth {
    let n = sigma.nrows();
    let mut sigma_seq = sigma.clone();
    let mut min_alpha = None;
    let mut first_nonpositive_row = None;

    for j in 0..c_star.nrows() {
        let r_j = r_noise[(j, j)];
        let mut v = DVector::<f64>::zeros(n);
        for col in 0..n {
            let c_jc = c_star[(j, col)];
            if c_jc != 0.0 {
                v.axpy(c_jc, &sigma_seq.column(col), 1.0);
            }
        }

        let mut alpha = r_j;
        for col in 0..n {
            let c_jc = c_star[(j, col)];
            if c_jc != 0.0 {
                alpha += c_jc * v[col];
            }
        }

        if alpha.is_finite() {
            min_alpha = Some(min_alpha.map_or(alpha, |current: f64| current.min(alpha)));
        }
        if first_nonpositive_row.is_none() && (!alpha.is_finite() || alpha <= 0.0) {
            first_nonpositive_row = Some(j);
            break;
        }
        if alpha.abs() < 1e-30 {
            continue;
        }
        sigma_seq.ger(-1.0 / alpha, &v, &v, 1.0);
    }

    ScalarUpdateAlphaHealth {
        min_alpha,
        first_nonpositive_row,
    }
}

fn sigma_spd_failure(sigma: &DMatrix<f64>) -> Option<SigmaSpdFailure> {
    if sigma_nonfinite_kind(sigma).is_some() {
        return None;
    }

    let mut min_diag = f64::INFINITY;
    let mut max_asym = 0.0_f64;
    for i in 0..sigma.nrows() {
        min_diag = min_diag.min(sigma[(i, i)]);
        for j in (i + 1)..sigma.ncols() {
            max_asym = max_asym.max((sigma[(i, j)] - sigma[(j, i)]).abs());
        }
    }

    let sym_sigma = (sigma.clone() + sigma.transpose()) * 0.5;
    if sym_sigma.cholesky().is_some() {
        None
    } else {
        Some(SigmaSpdFailure { min_diag, max_asym })
    }
}

fn min_step(steps: &[usize]) -> Option<usize> {
    steps.iter().copied().min()
}

fn median_step(steps: &mut [usize]) -> Option<usize> {
    if steps.is_empty() {
        return None;
    }
    steps.sort_unstable();
    Some(steps[steps.len() / 2])
}

fn min_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::min)
        .unwrap_or(f64::NAN)
}

fn max_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(f64::NAN)
}

fn print_diag_progress(label: &str, done: usize, total: usize, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let per_trial = elapsed / done.max(1) as f64;
    let eta = per_trial * total.saturating_sub(done) as f64;
    eprintln!(
        "progress: {label} trial {done}/{total} elapsed={} eta={}",
        format_seconds(elapsed),
        format_seconds(eta)
    );
}

fn format_seconds(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    let minutes = seconds / 60;
    let seconds = seconds % 60;
    format!("{minutes}m{seconds:02}s")
}
