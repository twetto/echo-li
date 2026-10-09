use crate::mathematical::IMUVelocity;
use echo_lie::{SE3, SO3};
use nalgebra::Vector3;

/// Estimate initial pose from stationary IMU accelerometer readings.
pub fn estimate_initial_pose(imu_readings: &[IMUVelocity], n_samples: usize) -> SE3 {
    if imu_readings.len() < 5 {
        return SE3::identity();
    }

    let n = n_samples.min(imu_readings.len());

    // Average accelerometer readings
    let mut acc_sum = Vector3::zeros();
    for i in 0..n {
        acc_sum += imu_readings[i].acc;
    }
    let acc_avg = acc_sum / (n as f64);

    // Measured gravity direction in body frame
    // When stationary, acc = R^T * [0, 0, g]
    // So g_body = acc_avg / ||acc_avg|| should map to [0, 0, 1] in world frame
    let g_body = acc_avg.normalize();
    let g_world = Vector3::new(0.0, 0.0, 1.0);

    // Find rotation: R * g_body = g_world
    let r = SO3::from_vectors(&g_body, &g_world);

    SE3::new(r, Vector3::zeros())
}

/// Check if the first n_samples IMU readings indicate a stationary platform.
pub fn check_stationary(
    imu_readings: &[IMUVelocity],
    n_samples: usize,
    gyro_std_threshold: f64,
    acc_std_threshold: f64,
) -> bool {
    if imu_readings.len() < 10 {
        return false;
    }

    let n = n_samples.min(imu_readings.len());

    // Compute gyro stats
    let mut gyro_sum = Vector3::zeros();
    let mut gyro_sq_sum = 0.0;

    // Compute acc norm stats
    let mut acc_norm_sum = 0.0;
    let mut acc_norm_sq_sum = 0.0;

    for i in 0..n {
        let gyr = imu_readings[i].gyr;
        gyro_sum += gyr;
        gyro_sq_sum += gyr.norm_squared();

        let acc_norm = imu_readings[i].acc.norm();
        acc_norm_sum += acc_norm;
        acc_norm_sq_sum += acc_norm * acc_norm;
    }

    let nf = n as f64;

    // Std dev of gyro components (approximate with average of squares)
    let gyro_mean_sq = gyro_sq_sum / nf;
    let gyro_vec_mean = gyro_sum / nf;
    let gyro_var = (gyro_mean_sq - gyro_vec_mean.norm_squared()).max(0.0);
    let gyro_std = gyro_var.sqrt();

    // Std dev of acc norm
    let acc_norm_mean = acc_norm_sum / nf;
    let acc_norm_var = (acc_norm_sq_sum / nf - acc_norm_mean * acc_norm_mean).max(0.0);
    let acc_std = acc_norm_var.sqrt();

    gyro_std < gyro_std_threshold && acc_std < acc_std_threshold
}

// ===========================================================================
// Static (at-rest) initializer — port of OpenVINS `ov_init::StaticInitializer`.
// ===========================================================================

/// Seed produced by [`static_initialize`]: a full IMU state plus the diagonal
/// standard deviations to put on it.
#[derive(Debug, Clone)]
pub struct StaticInit {
    /// Timestamp the seed refers to (end of the still window).
    pub stamp: f64,
    /// Body->world rotation, gravity-aligned. Yaw is a free gauge (unobservable
    /// from gravity) and is fixed arbitrarily.
    pub r_wb: SO3,
    /// Position and velocity are ZERO by assertion: the platform is at rest.
    pub p: Vector3<f64>,
    pub v: Vector3<f64>,
    /// Gyro bias = mean angular rate while still.
    pub bg: Vector3<f64>,
    /// Accel bias = mean specific force minus the gravity it should have read.
    pub ba: Vector3<f64>,
    /// 1-sigma for [attitude, position, velocity], OpenVINS' static values.
    pub std_att: f64,
    pub std_pos: f64,
    pub std_vel: f64,
}

/// Detect an at-rest window and seed the state from it.
///
/// Port of OpenVINS `StaticInitializer::initialize`. Splits the newest
/// `window_time` seconds of IMU into two half-windows and compares their accel
/// sample standard deviations:
///
/// * `wait_for_jerk = true`  — require the NEWER half to be excited and the OLDER
///   half still, i.e. motion has just begun. The seed is built from the OLDER
///   (still) half, so `v = 0` is true at that instant. This is what OpenVINS uses
///   by default, and it is why it initializes AT REST rather than skipping ahead
///   to motion onset and then asserting `v = 0` anyway.
/// * `wait_for_jerk = false` — require BOTH halves still.
///
/// Returns `None` when the excitation pattern does not match, so the caller keeps
/// buffering. `v = 0` here is a genuine measurement (the platform is stationary),
/// not an assumption, which is what earns the tight `std_vel`.
///
/// LIMITATION (measured, see `check_stationary`): the accel-variance test is blind
/// to CONSTANT motion — a platform in steady cruise, or turning at a constant rate,
/// has near-zero accel variance and reads as "still". No IMU-only test can do
/// better; confirming rest needs vision (zero disparity). Use this only where a
/// genuine stationary start is plausible.
pub fn static_initialize(
    imu: &[IMUVelocity],
    window_time: f64,
    imu_thresh: f64,
    gravity_mag: f64,
    wait_for_jerk: bool,
) -> Option<StaticInit> {
    if imu.len() < 4 || window_time <= 0.0 || gravity_mag <= 0.0 {
        return None;
    }
    let newest = imu.last()?.stamp;
    // newer half (jerk candidate) and older half (the still window we seed from)
    let mut w_1to0: Vec<&IMUVelocity> = Vec::new();
    let mut w_2to1: Vec<&IMUVelocity> = Vec::new();
    for d in imu {
        if d.stamp > newest - 0.5 * window_time && d.stamp <= newest {
            w_1to0.push(d);
        } else if d.stamp > newest - window_time && d.stamp <= newest - 0.5 * window_time {
            w_2to1.push(d);
        }
    }
    if w_1to0.len() < 2 || w_2to1.len() < 2 {
        return None;
    }

    let accel_std = |w: &[&IMUVelocity]| -> (Vector3<f64>, f64) {
        let n = w.len() as f64;
        let avg = w.iter().fold(Vector3::zeros(), |s, d| s + d.acc) / n;
        let var = w.iter().map(|d| (d.acc - avg).norm_squared()).sum::<f64>() / (n - 1.0);
        (avg, var.sqrt())
    };
    let (_a_avg_1to0, a_std_1to0) = accel_std(&w_1to0);
    let (a_avg_2to1, a_std_2to1) = accel_std(&w_2to1);

    if wait_for_jerk {
        // Need excitation NOW and stillness BEFORE it.
        if a_std_1to0 < imu_thresh || a_std_2to1 > imu_thresh {
            return None;
        }
    } else if a_std_1to0 > imu_thresh || a_std_2to1 > imu_thresh {
        return None;
    }

    let n2 = w_2to1.len() as f64;
    let w_avg_2to1 = w_2to1.iter().fold(Vector3::zeros(), |s, d| s + d.gyr) / n2;
    if a_avg_2to1.norm() < 1e-9 {
        return None;
    }

    // At rest the accelerometer reads specific force f = -R^T g_vec, so the measured
    // direction maps to "up" in the world. Yaw about that axis is unobservable.
    let up = a_avg_2to1 / a_avg_2to1.norm();
    let r_wb = SO3::from_vectors(&up, &Vector3::new(0.0, 0.0, 1.0));
    // ba = measured - what gravity alone should have produced, in body coords.
    let g_body = r_wb.inverse().act(&Vector3::new(0.0, 0.0, gravity_mag));
    let ba = a_avg_2to1 - g_body;

    Some(StaticInit {
        stamp: w_2to1.last()?.stamp,
        r_wb,
        p: Vector3::zeros(),
        v: Vector3::zeros(),
        bg: w_avg_2to1,
        ba,
        // OpenVINS' static seed covariance (StaticInitializer.cpp).
        std_att: 0.02,
        std_pos: 0.05,
        std_vel: 0.01,
    })
}

#[cfg(test)]
mod static_init_tests {
    use super::*;

    /// Stationary IMU in a world where gravity is -z: at rest the accelerometer
    /// reads the specific force +z*g, rotated into the body frame.
    fn still_stream(r_wb: &SO3, g: f64, n: usize, dt: f64, noise: f64) -> Vec<IMUVelocity> {
        let f_world = Vector3::new(0.0, 0.0, g);
        (0..n)
            .map(|i| {
                let j = (i % 3) as f64 - 1.0; // deterministic tiny dither
                IMUVelocity {
                    stamp: i as f64 * dt,
                    gyr: Vector3::new(0.0, 0.0, 0.0),
                    acc: r_wb.inverse().act(&f_world) + Vector3::repeat(j * noise),
                    gyr_bias_vel: Vector3::zeros(),
                    acc_bias_vel: Vector3::zeros(),
                }
            })
            .collect()
    }

    #[test]
    fn static_init_recovers_gravity_and_asserts_zero_velocity() {
        let g = 9.81;
        let truth = SO3::exp(&Vector3::new(0.12, -0.07, 0.9)); // includes yaw
        let imu = still_stream(&truth, g, 200, 0.005, 1e-4);
        let out = static_initialize(&imu, 1.0, 0.1, g, false).expect("still stream must init");

        // TILT is observable; yaw is not, so compare the gravity direction only.
        let u = Vector3::new(0.0, 0.0, 1.0);
        let tilt = (out.r_wb.inverse().act(&u))
            .dot(&truth.inverse().act(&u))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(tilt < 0.1, "tilt error {tilt} deg");
        assert_eq!(out.p, Vector3::zeros());
        assert_eq!(out.v, Vector3::zeros());
        assert!(out.ba.norm() < 1e-2, "accel bias {:?}", out.ba);
        assert!(out.bg.norm() < 1e-9);
        assert_eq!(out.std_vel, 0.01);
    }

    #[test]
    fn static_init_waits_for_jerk_and_seeds_from_the_still_half() {
        let g = 9.81;
        let truth = SO3::identity();
        let mut imu = still_stream(&truth, g, 200, 0.005, 1e-4);
        // Excite only the NEWER half: motion has just begun.
        let n = imu.len();
        for d in imu.iter_mut().skip(n / 2) {
            d.acc += Vector3::new(1.5, 0.0, 0.0);
            d.stamp += 0.0;
        }
        for (i, d) in imu.iter_mut().skip(n / 2).enumerate() {
            d.acc[0] += if i % 2 == 0 { 0.8 } else { -0.8 }; // variance, not offset
        }
        let out = static_initialize(&imu, 1.0, 0.1, g, true);
        assert!(out.is_some(), "jerk after stillness must initialize");

        // A fully still stream has no jerk, so wait_for_jerk must REFUSE it.
        let still = still_stream(&truth, g, 200, 0.005, 1e-4);
        assert!(static_initialize(&still, 1.0, 0.1, g, true).is_none());
        // ...while the non-jerk mode accepts it.
        assert!(static_initialize(&still, 1.0, 0.1, g, false).is_some());
    }

    #[test]
    fn static_init_is_blind_to_constant_motion() {
        // DOCUMENTED LIMITATION: steady cruise has zero accel variance and reads as
        // "still". This test pins the failure so nobody trusts the detector alone.
        let g = 9.81;
        let imu = still_stream(&SO3::identity(), g, 200, 0.005, 1e-4);
        let out = static_initialize(&imu, 1.0, 0.1, g, false);
        assert!(
            out.is_some(),
            "an IMU-only test cannot distinguish rest from constant velocity"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared filter start-up
// ---------------------------------------------------------------------------

/// Everything needed to start a filter from the first IMU samples.
///
/// Returned by [`StationaryInit`] once enough samples have arrived. `replay` is the
/// buffer that was consumed to produce `state`: those samples have NOT been given to any
/// filter yet, and dropping them loses the only evidence of bias the filter has before
/// its first image.
pub struct FilterStart {
    /// Gravity-aligned initial state: zero position, zero velocity, zero bias, and the
    /// caller's camera offset.
    pub state: crate::mathematical::vio_state::VIOState,
    /// The buffered samples, oldest first. Feed them to the filter, or splice them back
    /// onto the front of an event stream -- but do exactly one of the two.
    pub replay: Vec<IMUVelocity>,
    /// Whether those samples looked stationary. Diagnostic only: alignment happens
    /// either way, because a moving start aligned from an accel reading is within
    /// ~10-15 deg of gravity -- inside the filter's initial attitude sigma -- whereas
    /// falling back to identity can be 180 deg off and diverges immediately as gravity
    /// integrates into phantom acceleration.
    pub stationary: bool,
    pub roll_deg: f64,
    pub pitch_deg: f64,
}

/// Collects the first IMU samples and turns them into a filter's initial state.
///
/// This exists because the three wrappers each grew their own copy and drifted: before
/// this type the CLI averaged ONE sample while the pyo3 bindings and the ROS 2 node
/// averaged a hundred, and only the CLI reported stationarity at all. Nobody chose that;
/// it is what three copies of a policy turn into. The policy lives here now, and
/// `n_samples` is the only knob.
pub struct StationaryInit {
    n_samples: usize,
    camera_offset: SE3,
    buffer: Vec<IMUVelocity>,
    done: bool,
}

impl StationaryInit {
    /// Fewer readings than this and `estimate_initial_pose` short-circuits to identity.
    pub const MIN_SAMPLES: usize = 5;

    /// `n_samples` is clamped to at least [`Self::MIN_SAMPLES`]. Below that
    /// [`estimate_initial_pose`] returns IDENTITY rather than an alignment, and an
    /// identity start can be 180 deg from gravity, which diverges immediately as gravity
    /// integrates into phantom acceleration. Silently handing back the one attitude the
    /// rest of this module exists to avoid is not an acceptable answer to "I only have
    /// three samples", so the floor is enforced here instead.
    pub fn new(n_samples: usize, camera_offset: SE3) -> Self {
        Self {
            n_samples: n_samples.max(Self::MIN_SAMPLES),
            camera_offset,
            buffer: Vec::new(),
            done: false,
        }
    }

    /// Feed one sample. Returns the start-up data on the call that completes the buffer,
    /// and `None` every other time -- including after it has fired once, so a caller
    /// that keeps pushing cannot re-initialise the filter mid-run.
    pub fn push(&mut self, imu: IMUVelocity) -> Option<FilterStart> {
        if self.done {
            return None;
        }
        self.buffer.push(imu);
        if self.buffer.len() < self.n_samples {
            return None;
        }
        self.done = true;
        Some(self.finish())
    }

    /// Eager form: hand over an already-collected buffer. Used where the samples are
    /// read up front rather than arriving one at a time.
    pub fn from_samples(
        n_samples: usize,
        camera_offset: SE3,
        samples: Vec<IMUVelocity>,
    ) -> FilterStart {
        let mut me = Self::new(n_samples, camera_offset);
        me.buffer = samples;
        // The floor applies to what is actually averaged, which is bounded by the
        // buffer: asking for 100 from a 6-sample buffer averages 6, and that is fine,
        // but a 3-sample buffer would silently align to identity.
        me.n_samples = me.n_samples.min(me.buffer.len().max(Self::MIN_SAMPLES));
        me.done = true;
        me.finish()
    }

    /// Set the camera offset after construction, for callers that learn their
    /// extrinsics later than they build the filter. No effect once initialisation has
    /// fired -- by then the offset is already baked into the filter's state.
    pub fn set_camera_offset(&mut self, camera_offset: SE3) {
        self.camera_offset = camera_offset;
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    fn finish(&mut self) -> FilterStart {
        let pose = estimate_initial_pose(&self.buffer, self.n_samples);
        let stationary = check_stationary(&self.buffer, self.n_samples, 0.1, 0.5);
        let r = pose.rotation.as_matrix();
        let pitch_deg = (-r[(2, 0)]).clamp(-1.0, 1.0).asin().to_degrees();
        let roll_deg = r[(2, 1)].atan2(r[(2, 2)]).to_degrees();

        let sensor = crate::mathematical::vio_state::VIOSensorState {
            input_bias: nalgebra::Vector6::zeros(),
            pose,
            velocity: Vector3::zeros(),
            camera_offset: self.camera_offset.clone(),
        };
        FilterStart {
            state: crate::mathematical::vio_state::VIOState::new(sensor, Vec::new()),
            replay: std::mem::take(&mut self.buffer),
            stationary,
            roll_deg,
            pitch_deg,
        }
    }
}

/// Build a filter at `start` and feed it the buffered samples.
///
/// For callers that replay into the filter directly. A caller that instead splices
/// `start.replay` back onto its own event stream must NOT use this, or the samples are
/// integrated twice.
pub fn start_filter(settings: crate::VIOFilterSettings, start: FilterStart) -> crate::VIOFilter {
    let mut filter = crate::VIOFilter::new(settings, start.state);
    for imu in start.replay {
        filter.process_imu(imu);
    }
    filter
}

#[cfg(test)]
mod stationary_init_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    fn imu(stamp: f64, acc: Vector3<f64>) -> IMUVelocity {
        IMUVelocity::new(stamp, Vector3::zeros(), acc)
    }

    /// A level, stationary rig: measured gravity along +z maps to world +z, so the
    /// initial attitude is identity and roll/pitch are zero.
    #[test]
    fn level_rest_gives_identity_attitude() {
        let mut init = StationaryInit::new(10, SE3::identity());
        let mut out = None;
        for k in 0..10 {
            out = init.push(imu(k as f64 * 0.01, Vector3::new(0.0, 0.0, 9.81)));
        }
        let start = out.expect("should fire on the 10th sample");
        assert!(start.stationary);
        assert!(start.roll_deg.abs() < 1e-9, "roll {}", start.roll_deg);
        assert!(start.pitch_deg.abs() < 1e-9, "pitch {}", start.pitch_deg);
        assert_eq!(
            start.replay.len(),
            10,
            "every buffered sample must come back"
        );
    }

    /// Fires exactly once. A caller that keeps pushing must not be able to re-initialise
    /// the filter mid-run.
    #[test]
    fn fires_once_and_only_once() {
        let n = 6;
        let mut init = StationaryInit::new(n, SE3::identity());
        for k in 0..n - 1 {
            assert!(
                init.push(imu(k as f64 * 0.01, Vector3::new(0.0, 0.0, 9.81)))
                    .is_none(),
                "fired early at sample {k}"
            );
        }
        assert!(
            init.push(imu(0.05, Vector3::new(0.0, 0.0, 9.81))).is_some(),
            "did not fire on the nth sample"
        );
        assert!(init.is_done());
        for k in n..n + 7 {
            assert!(
                init.push(imu(k as f64 * 0.01, Vector3::new(0.0, 0.0, 9.81)))
                    .is_none(),
                "re-initialised after completing"
            );
        }
    }

    /// The floor is enforced: asking for 3 must not yield an identity alignment, which
    /// is what `estimate_initial_pose` hands back below 5 readings.
    #[test]
    fn below_the_floor_still_aligns() {
        let mut init = StationaryInit::new(3, SE3::identity());
        let mut out = None;
        for k in 0..StationaryInit::MIN_SAMPLES {
            out = init.push(imu(k as f64 * 0.01, Vector3::new(9.81, 0.0, 0.0)));
        }
        let start = out.expect("must fire at the clamped floor, not at 3");
        let g_world = start
            .state
            .sensor
            .pose
            .rotation
            .act(&Vector3::new(9.81, 0.0, 0.0));
        assert!(
            (g_world - Vector3::new(0.0, 0.0, 9.81)).norm() < 1e-9,
            "clamping failed; aligned to identity, got {g_world:?}"
        );
    }

    /// Tilted rest: gravity read along the body x axis means the rig is pitched 90 deg,
    /// and alignment must rotate measured gravity onto world +z regardless.
    #[test]
    fn tilted_rest_is_aligned_not_rejected() {
        let mut init = StationaryInit::new(5, SE3::identity());
        let mut out = None;
        for k in 0..5 {
            out = init.push(imu(k as f64 * 0.01, Vector3::new(9.81, 0.0, 0.0)));
        }
        let start = out.unwrap();
        let g_world = start
            .state
            .sensor
            .pose
            .rotation
            .act(&Vector3::new(9.81, 0.0, 0.0));
        assert!(
            (g_world - Vector3::new(0.0, 0.0, 9.81)).norm() < 1e-9,
            "measured gravity must rotate onto world +z, got {g_world:?}"
        );
    }

    /// Averaging must actually average: the eager and incremental paths agree, and
    /// noise across samples is reduced rather than the first sample being taken.
    #[test]
    fn eager_and_incremental_paths_agree() {
        let accs = [
            Vector3::new(0.30, 0.0, 9.81),
            Vector3::new(-0.30, 0.0, 9.81),
            Vector3::new(0.15, 0.0, 9.81),
            Vector3::new(-0.15, 0.0, 9.81),
            Vector3::new(0.00, 0.0, 9.81),
            Vector3::new(0.00, 0.0, 9.81),
        ];
        let samples: Vec<_> = accs
            .iter()
            .enumerate()
            .map(|(k, a)| imu(k as f64 * 0.01, *a))
            .collect();

        let mut init = StationaryInit::new(6, SE3::identity());
        let mut inc = None;
        for s in samples.clone() {
            inc = init.push(s);
        }
        let inc = inc.unwrap();
        let eager = StationaryInit::from_samples(6, SE3::identity(), samples);

        assert_abs_diff_eq!(
            inc.state.sensor.pose.rotation.as_matrix(),
            eager.state.sensor.pose.rotation.as_matrix(),
            epsilon = 0.0
        );
        // The x-components cancel, so the average is level even though no single
        // sample is. Taking only the first would have tilted by atan(0.30/9.81).
        assert!(
            inc.pitch_deg.abs() < 1e-9,
            "averaging should cancel the x wobble, pitch {}",
            inc.pitch_deg
        );
    }
}
