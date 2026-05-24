use approx::assert_abs_diff_eq;
use echo_lie::{SE3, SO3};
use nalgebra::{DMatrix, Vector2, Vector3, Vector6};
use std::collections::HashMap;

use crate::coordinate_suite::euclid::EuclideanSuite;
use crate::mathematical::camera::{CameraModel, PinholeModel};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_eqf::VIOEqF;
use crate::mathematical::vio_state::{Landmark, VIOSensorState, VIOState, GRAVITY_CONSTANT};
use crate::mathematical::vision_measurement::VisionMeasurement;
use crate::tests::testing_utilities::*;
use crate::{LandmarkDepthPrior, VIOFilter, VIOFilterSettings};

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
