//! The seam between a pose backend and everything that drives one.
//!
//! WHY THIS EXISTS. Three wrappers drive a filter today -- `echo-li-cli`,
//! `echo-li-ros2`'s node, and the pyo3 bindings -- and all three already call the same
//! surface: feed IMU, feed a frame, read the state. What they cannot do is choose a
//! DIFFERENT backend, because [`crate::VIOFilter`] hard-wires the EqVIO filter and its
//! coordinate suite. The consequence was measured, not hypothetical: driving the MSCEqF
//! filter from a diagnostic meant hand-rolling an adapter in Python
//! (`midair_e2e_depth_nees.py::MSCEqFAdapter`), which shipped five silent bugs -- a
//! gravity sign, a quaternion ordering invisible to the harness's own error metric, a
//! conflated update-span and anchor-window, body-vs-world velocity, and borrowed IMU
//! noise that understated the gyro by 20x.
//!
//! WHY A TRAIT IS ENOUGH. The two backends differ in ORDERING, but only INSIDE the
//! vision step: EqVIO keeps landmarks internally and consumes one frame of
//! observations, whereas an MSCKF must clone the pose first, buffer observations across
//! frames, promote window-spanning tracks, update, then marginalize. That is eight
//! internal steps against one -- but it is still a single call from the caller, which is
//! exactly what the Python adapter demonstrated by keeping the scoring path
//! byte-identical across both backends. So the seam sits where the wrappers already
//! call, and no driving loop has to change.
//!
//! WHAT A SECOND IMPLEMENTOR OWES. The MSCKF track buffer, the in-state landmark set,
//! the clone-id counter and the windowing policy are BACKEND state, not caller state.
//! They live in a test harness today; an implementor must bring them inside.

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};

use crate::LandmarkDepthPrior;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vision_measurement::VisionMeasurement;
use echo_lie::SE3;

/// What every VIO backend has by definition: where the body is, how fast it is moving,
/// and what it currently believes about its IMU biases.
///
/// This exists so the seam does NOT traffic in [`crate::mathematical::vio_state::VIOState`].
/// That type is EqVIO's own state -- a sensor block plus camera-frame landmarks in
/// EqVIO's parametrization -- and making it the interface currency would force every
/// other backend to project itself into EqVIO's shape on every read. An MSCKF would have
/// to invent a `camera_landmarks` vector it does not keep that way and silently drop its
/// pose clones, which are not an EqVIO concept at all. The projection would be lossy in
/// one direction and fictional in the other.
#[derive(Clone, Debug)]
pub struct NavEstimate {
    /// Body pose in the world frame.
    pub pose: SE3,
    /// Linear velocity in the BODY frame, matching `VIOSensorState::velocity`. Mixing
    /// this up with the world-frame velocity is one of the five bugs the Python MSCEqF
    /// adapter shipped, and it does not show up in a position-only error metric.
    pub velocity: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
}

/// A pose estimator that can be driven by IMU and image measurements.
///
/// Deliberately small. Anything only SOME backends can answer lives in a capability
/// trait below, so a backend advertises what it actually has instead of returning a
/// placeholder, and a consumer names the capability it needs in its own bounds:
///
/// ```ignore
/// fn drive<B: VioBackend + CameraPoseCovariance>(b: &mut B) { ... }
/// ```
pub trait VioBackend {
    /// Integrate one IMU sample. Implementors buffer or propagate as their design
    /// requires; callers make no assumption about when the work happens.
    fn process_imu(&mut self, imu: IMUVelocity);

    /// Consume one frame of feature observations.
    ///
    /// Whatever a backend must do internally -- cloning, buffering across frames,
    /// promoting tracks, marginalizing -- happens here, so the caller's loop is
    /// identical whichever backend is in use. The camera model is NOT an argument: a
    /// backend is constructed against one camera and does not change it mid-run, and
    /// both implementors need it in places other than this call.
    ///
    /// `depth_priors` is empty for most callers. It is a per-frame argument rather than
    /// backend state because the alternative -- a setter called before each frame -- is
    /// order-dependent hidden state, and order-dependent hidden state is precisely what
    /// produced the silent bugs this seam exists to prevent. Only a source with its OWN
    /// metric scale belongs here: stereo with a known baseline, a range sensor, or
    /// accel-vs-gravity. Feeding back a depth derived from this filter's own poses
    /// closes a loop on the unobservable scale direction and has been measured to
    /// compound rather than converge.
    fn process_vision(
        &mut self,
        measurement: VisionMeasurement,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    );

    /// Current navigation estimate.
    fn nav_estimate(&self) -> NavEstimate;
}

/// Backends that hold a map of landmarks and can report it in the world frame.
pub trait LandmarkMap {
    fn landmarks_world(&self) -> HashMap<u64, Vector3<f64>>;
}

/// Backends that can report camera-pose uncertainty -- the input the Sparse3D depth
/// filter consumes.
///
/// Separate from [`VioBackend`] because "I cannot tell you" and "my uncertainty is zero"
/// are different answers, and a consumer that treats the first as the second will report
/// a confident depth it has no basis for.
pub trait CameraPoseCovariance {
    /// `(P_vv, P_ww)`: position and attitude covariance of the camera pose, in the
    /// camera-fixed frame. `None` means unavailable, which a consumer must treat as
    /// missing information, NOT as certainty.
    fn camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)>;
}

/// The EqVIO filter plus the camera it was built against.
///
/// A wrapper rather than `impl VioBackend for VIOFilter` because the trait no longer
/// passes a camera per call, and because this is the same shape the MSCEqF backend must
/// take: that one additionally owns the track buffer, in-state landmark set, clone-id
/// counter and windowing policy that currently live in a Python test harness.
pub struct EqvioBackend {
    filter: crate::VIOFilter,
    cam: Arc<dyn CameraModel>,
}

impl EqvioBackend {
    pub fn new(filter: crate::VIOFilter, cam: Arc<dyn CameraModel>) -> Self {
        Self { filter, cam }
    }

    /// Borrow the concrete filter. Composition, not encapsulation: a consumer that needs
    /// something EqVIO-specific -- pose clones, the covariance block, the gramian --
    /// reaches for the real type instead of growing the shared trait to fit one caller.
    pub fn filter(&self) -> &crate::VIOFilter {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut crate::VIOFilter {
        &mut self.filter
    }

    pub fn into_filter(self) -> crate::VIOFilter {
        self.filter
    }
}

impl VioBackend for EqvioBackend {
    fn process_imu(&mut self, imu: IMUVelocity) {
        self.filter.process_imu(imu);
    }

    fn process_vision(
        &mut self,
        measurement: VisionMeasurement,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    ) {
        self.filter
            .process_vision_with_depth_priors(measurement, self.cam.as_ref(), depth_priors);
    }

    fn nav_estimate(&self) -> NavEstimate {
        let state = self.filter.eqf.state_estimate();
        NavEstimate {
            pose: state.sensor.pose.clone(),
            velocity: state.sensor.velocity,
            gyro_bias: state.sensor.gyro_bias(),
            accel_bias: state.sensor.accel_bias(),
        }
    }
}

impl LandmarkMap for EqvioBackend {
    fn landmarks_world(&self) -> HashMap<u64, Vector3<f64>> {
        let state = self.filter.eqf.state_estimate();
        crate::landmarks_to_global(&state).0
    }
}

impl CameraPoseCovariance for EqvioBackend {
    fn camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        self.filter.sparse_camera_pose_covariances()
    }
}
