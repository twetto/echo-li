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

use nalgebra::Matrix3;

use crate::LandmarkDepthPrior;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_state::VIOState;
use crate::mathematical::vision_measurement::VisionMeasurement;

/// A pose estimator that can be driven by IMU and image measurements.
pub trait VioBackend {
    /// Integrate one IMU sample. Implementors buffer or propagate as their design
    /// requires; callers make no assumption about when the work happens.
    fn process_imu(&mut self, imu: IMUVelocity);

    /// Consume one frame of feature observations.
    ///
    /// This is the full form. Whatever a backend must do internally -- cloning,
    /// buffering across frames, promoting tracks, marginalizing -- happens here, so the
    /// caller's loop is identical whichever backend is in use.
    fn process_vision_full(
        &mut self,
        measurement: VisionMeasurement,
        cam: &dyn CameraModel,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    );

    /// Current state estimate.
    fn state_estimate(&self) -> VIOState;

    /// `(P_vv, P_ww)` for the camera pose, when the backend can supply it -- the input
    /// the Sparse3D depth filter consumes. `None` means "not available", which a
    /// consumer must treat as missing information rather than as zero uncertainty.
    fn camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        None
    }

    /// Convenience: a frame with no depth priors.
    fn process_vision(&mut self, measurement: VisionMeasurement, cam: &dyn CameraModel) {
        self.process_vision_full(measurement, cam, &HashMap::new());
    }
}

impl VioBackend for crate::VIOFilter {
    fn process_imu(&mut self, imu: IMUVelocity) {
        crate::VIOFilter::process_imu(self, imu);
    }

    fn process_vision_full(
        &mut self,
        measurement: VisionMeasurement,
        cam: &dyn CameraModel,
        depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    ) {
        crate::VIOFilter::process_vision_with_depth_priors(
            self,
            measurement,
            cam,
            depth_priors,
        );
    }

    fn state_estimate(&self) -> VIOState {
        self.eqf.state_estimate()
    }

    fn camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        crate::VIOFilter::sparse_camera_pose_covariances(self)
    }
}
