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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use nalgebra::{Matrix3, Vector2, Vector3};

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

// ---------------------------------------------------------------------------
// MSCEqF backend
// ---------------------------------------------------------------------------

/// Windowing and update policy for [`MsceqfBackend`].
#[derive(Clone, Debug)]
pub struct MsceqfSettings {
    /// Pose clones kept in the window. Older ones marginalize.
    pub num_clones: usize,
    /// Measurement noise in NORMALIZED image units (the backend normalizes pixels
    /// before they reach the filter, so this is not a pixel count).
    pub pixel_std: f64,
    /// Chi-squared gate multiplier.
    pub chi2_mult: f64,
    /// Hard cap on observations buffered for one structureless track.
    pub max_track: usize,
    /// Observations after which a track is flushed into an MSC update.
    ///
    /// Independent of `num_clones` ON PURPOSE. Tying the update span to the clone
    /// window breaks one or the other: a short window starves the anchor supply, while
    /// flushing only at marginalization means almost no update fires and the filter
    /// runs near open-loop.
    pub update_span: usize,
    /// Maximum in-state SLAM landmarks. 0 = structureless only, which is the WEAKEST
    /// configuration of this filter.
    pub max_slam: usize,
}

impl Default for MsceqfSettings {
    fn default() -> Self {
        Self {
            num_clones: 11,
            pixel_std: 1.0,
            chi2_mult: 1.0,
            max_track: 200,
            update_span: 11,
            max_slam: 0,
        }
    }
}

/// The MSCEqF filter plus the track bookkeeping that turns its primitives into one
/// frame step.
///
/// [`crate::mathematical::msceqf_filter::MSCEqFFilter`] is a primitives API --
/// `stochastic_clone`, `birth_landmark`, `msc_update`, `landmark_stream_update`,
/// `reanchor_landmark`, `marginalize_clone` -- with no notion of a frame. The policy
/// that sequences them lived in a Python test harness; this type brings it inside, which
/// is what the seam's doc comment means by "what a second implementor owes".
///
/// Two id spaces meet here and must not be confused. `clone_ids` and `landmark_ids` map
/// caller-facing ids to POSITIONS in the filter's window and landmark list, and those
/// positions shift as things marginalize. Buffered observations therefore store a
/// `clone_id`, never a position.
pub struct MsceqfBackend {
    filter: crate::mathematical::msceqf_filter::MSCEqFFilter,
    /// Real camera, used to turn pixels into normalized bearings.
    cam: Arc<dyn CameraModel>,
    /// Identity projection handed to the filter, which consumes already-normalized
    /// coordinates. `pixel_std` is in those same units.
    z1: crate::mathematical::camera::PinholeModel,
    settings: MsceqfSettings,
    /// Frame id of each live clone, parallel to the filter's clone window (ascending).
    clone_ids: Vec<u64>,
    /// Track id of each in-state landmark, parallel to the filter's landmark list.
    landmark_ids: Vec<u64>,
    /// Structureless observation buffer: track id -> [(clone_id, u_n, v_n)]. A BTreeMap
    /// rather than a HashMap because the iteration order feeds the stacking order of
    /// the update, and a randomized walk makes the filter non-deterministic run to run.
    buf: BTreeMap<u64, Vec<(u64, f64, f64)>>,
    /// Tracks currently held as in-state landmarks.
    slam: BTreeSet<u64>,
    /// Frame counter, used as the clone id.
    k: u64,
    last_stamp: Option<f64>,
}

impl MsceqfBackend {
    pub fn new(
        filter: crate::mathematical::msceqf_filter::MSCEqFFilter,
        cam: Arc<dyn CameraModel>,
        settings: MsceqfSettings,
    ) -> Self {
        Self {
            filter,
            cam,
            z1: crate::mathematical::camera::PinholeModel {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            },
            settings,
            clone_ids: Vec::new(),
            landmark_ids: Vec::new(),
            buf: BTreeMap::new(),
            slam: BTreeSet::new(),
            k: 0,
            last_stamp: None,
        }
    }

    pub fn filter(&self) -> &crate::mathematical::msceqf_filter::MSCEqFFilter {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut crate::mathematical::msceqf_filter::MSCEqFFilter {
        &mut self.filter
    }

    pub fn n_clones(&self) -> usize {
        self.clone_ids.len()
    }

    pub fn n_slam(&self) -> usize {
        self.slam.len()
    }

    /// Pixel -> normalized image coordinates, through the REAL camera model, so lens
    /// distortion is removed. The Python harness used a bare inverse-K, which is only
    /// equivalent on an undistorted pinhole.
    fn normalize(&self, uv: &Vector2<f32>) -> Option<(f64, f64)> {
        let b = self
            .cam
            .undistort(&Vector2::new(uv[0] as f64, uv[1] as f64));
        if b[2].abs() < 1e-9 || !b[2].is_finite() {
            return None;
        }
        Some((b[0] / b[2], b[1] / b[2]))
    }

    /// Keep only observations whose clone is still live, translating id -> position.
    fn obs_on_live_clones(
        &self,
        obs: &[(u64, f64, f64)],
    ) -> Vec<crate::mathematical::msceqf_filter::MscTrackObs> {
        obs.iter()
            .filter_map(|&(cid, un, vn)| {
                self.clone_ids.iter().position(|&c| c == cid).map(|pos| {
                    crate::mathematical::msceqf_filter::MscTrackObs {
                        clone: pos,
                        uvn: Vector2::new(un, vn),
                        stereo_range: None,
                    }
                })
            })
            .collect()
    }
}

impl VioBackend for MsceqfBackend {
    /// The filter propagates on an INTERVAL, so the first sample only establishes a
    /// timestamp. Non-positive and implausibly large gaps are dropped rather than
    /// integrated: a backwards stamp would propagate the covariance backwards, and a
    /// long gap after a dropped frame integrates a stale angular rate over a window
    /// where it was never constant.
    fn process_imu(&mut self, imu: IMUVelocity) {
        if let Some(prev) = self.last_stamp {
            let dt = imu.stamp - prev;
            if dt > 0.0 && dt < 0.1 {
                let u = crate::mathematical::msceqf_filter::Imu {
                    ang: imu.gyr,
                    acc: imu.acc,
                };
                self.filter.propagate_step(&u, dt);
            }
        }
        self.last_stamp = Some(imu.stamp);
    }

    /// One frame, in the order the stages must run.
    ///
    /// The ordering is the substance of this method, and every step depends on the one
    /// before it:
    ///
    /// 1. clone FIRST, so this frame's observations have a pose to attach to;
    /// 2. buffer observations, excluding tracks already held as in-state landmarks;
    /// 3. promote window-spanning tracks to in-state, re-anchoring each to the current
    ///    clone so a fresh landmark never depends on a clone about to die;
    /// 4. stream-update the in-state landmarks still visible, and marginalize the rest
    ///    -- an in-state landmark with no observation contributes nothing and holds
    ///    state;
    /// 5. flush structureless tracks into one MSC update;
    /// 6. re-anchor any in-state landmark still anchored to the clone about to go;
    /// 7. marginalize clones down to the window and drop the observations that
    ///    referenced them.
    ///
    /// `depth_priors` is accepted but currently unused by this backend: MSCEqF births
    /// take a range prior per landmark, and wiring that through is only meaningful once
    /// a metric source (stereo, range sensor) is driving this path.
    fn process_vision(
        &mut self,
        measurement: VisionMeasurement,
        _depth_priors: &HashMap<u64, LandmarkDepthPrior>,
    ) {
        let k = self.k;
        self.k += 1;

        // (1) Clone BEFORE the update.
        self.filter.stochastic_clone(measurement.stamp);
        self.clone_ids.push(k);

        // (2) Buffer this frame's observations. Sorted so the buffer's iteration order,
        // which feeds the update's stacking order, does not depend on the caller's map.
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut ids: Vec<u64> = measurement.cam_coordinates.keys().copied().collect();
        ids.sort_unstable();
        let mut normalized: BTreeMap<u64, (f64, f64)> = BTreeMap::new();
        for id in ids {
            let uv = measurement.cam_coordinates[&id];
            let Some((un, vn)) = self.normalize(&uv) else {
                continue;
            };
            normalized.insert(id, (un, vn));
            seen.insert(id);
            if !self.slam.contains(&id) {
                self.buf.entry(id).or_default().push((k, un, vn));
            }
        }

        // The clone that is about to be marginalized, if the window is now over budget.
        let marg = if self.clone_ids.len() > self.settings.num_clones {
            self.clone_ids.first().copied()
        } else {
            None
        };

        // (3) Promote window-spanning tracks to in-state landmarks, longest first.
        if self.settings.max_slam > 0 && self.slam.len() < self.settings.max_slam {
            let mut cand: Vec<u64> = self
                .buf
                .iter()
                .filter(|(t, o)| seen.contains(t) && o.len() >= self.settings.update_span)
                .map(|(t, _)| *t)
                .collect();
            // Longest track first; track id breaks ties so the order is total.
            cand.sort_by(|a, b| {
                self.buf[b]
                    .len()
                    .cmp(&self.buf[a].len())
                    .then_with(|| a.cmp(b))
            });
            let room = self.settings.max_slam - self.slam.len();
            for tid in cand.into_iter().take(room) {
                let obs = self.obs_on_live_clones(&self.buf[&tid]);
                if obs.len() < 2 || self.landmark_ids.contains(&tid) {
                    continue;
                }
                let track = crate::mathematical::msceqf_filter::MscTrack { obs };
                let born = self.filter.birth_landmark(
                    &track,
                    &self.z1,
                    self.settings.pixel_std,
                    self.settings.chi2_mult,
                    None,
                );
                if let Some(j) = born {
                    debug_assert_eq!(j, self.landmark_ids.len(), "birth appends at the tail");
                    self.landmark_ids.push(tid);
                    // Re-anchor to the CURRENT clone: the triangulation anchored on the
                    // oldest observing clone, which is the one nearest marginalization.
                    if let Some(pos) = self.clone_ids.iter().position(|&c| c == k) {
                        self.filter.reanchor_landmark(j, pos);
                    }
                    self.slam.insert(tid);
                    self.buf.remove(&tid);
                }
            }
        }

        // (4) Stream-update visible in-state landmarks; marginalize the rest.
        if !self.slam.is_empty() {
            let mut updates: Vec<crate::mathematical::msceqf_filter::LmStreamUpdate> = Vec::new();
            let mut lost: Vec<u64> = Vec::new();
            for &tid in &self.slam {
                if let Some(&(un, vn)) = normalized.get(&tid) {
                    if let Some(j) = self.landmark_ids.iter().position(|&t| t == tid) {
                        let obs = self.obs_on_live_clones(&[(k, un, vn)]);
                        if !obs.is_empty() {
                            updates.push(crate::mathematical::msceqf_filter::LmStreamUpdate {
                                j,
                                obs,
                            });
                        }
                    }
                } else {
                    lost.push(tid);
                }
            }
            if !updates.is_empty() {
                self.filter.landmark_stream_update(
                    &updates,
                    &self.z1,
                    self.settings.pixel_std,
                    self.settings.chi2_mult,
                );
            }
            // Drop lost landmarks AFTER the update: marginalizing first would shift the
            // indices the update was built against.
            for tid in lost {
                if let Some(j) = self.landmark_ids.iter().position(|&t| t == tid) {
                    self.filter.marginalize_landmark(j);
                    self.landmark_ids.remove(j);
                }
                self.slam.remove(&tid);
            }
        }

        // (5) Flush structureless tracks into one MSC update.
        let mut ready: Vec<crate::mathematical::msceqf_filter::MscTrack> = Vec::new();
        let mut done: Vec<u64> = Vec::new();
        for (&tid, obs) in &self.buf {
            let lost = !seen.contains(&tid);
            let too_long = obs.len() >= self.settings.max_track;
            let spanned = obs.len() >= self.settings.update_span;
            let touches_marg = marg.is_some_and(|m| obs.iter().any(|o| o.0 == m));
            if lost || too_long || spanned || touches_marg {
                if obs.len() >= 2 {
                    let live = self.obs_on_live_clones(obs);
                    if live.len() >= 2 {
                        ready.push(crate::mathematical::msceqf_filter::MscTrack { obs: live });
                    }
                }
                done.push(tid);
            }
        }
        if !ready.is_empty() {
            self.filter.msc_update(
                &ready,
                &self.z1,
                self.settings.pixel_std,
                self.settings.chi2_mult,
                false,
            );
        }
        for tid in done {
            self.buf.remove(&tid);
        }

        // (6) Rescue in-state landmarks anchored to the dying clone.
        if let Some(m) = marg {
            if !self.slam.is_empty() {
                if let Some(&newest) = self.clone_ids.last() {
                    let newest_pos = self.clone_ids.len() - 1;
                    if newest != m {
                        for j in 0..self.landmark_ids.len() {
                            let stamp = self.filter.x.landmarks[j].anchor;
                            let anchored_to_marg = self
                                .filter
                                .x
                                .clones
                                .iter()
                                .position(|c| c.stamp == stamp)
                                .and_then(|p| self.clone_ids.get(p).copied())
                                == Some(m);
                            if anchored_to_marg {
                                self.filter.reanchor_landmark(j, newest_pos);
                            }
                        }
                    }
                }
            }
        }

        // (7) Marginalize down to the window, pruning observations that referenced the
        // clones that went.
        while self.clone_ids.len() > self.settings.num_clones {
            let old = self.clone_ids[0];
            self.filter.marginalize_clone(0);
            self.clone_ids.remove(0);
            self.buf.retain(|_, obs| {
                obs.retain(|o| o.0 != old);
                !obs.is_empty()
            });
        }
    }

    fn nav_estimate(&self) -> NavEstimate {
        let nav = self.filter.phi();
        let rotation = nav.t.rotation;
        // MSCEqF carries velocity in the WORLD frame (its SE_2(3) state is seeded with
        // an unrotated world velocity), while NavEstimate -- like EqVIO's
        // VIOSensorState -- is BODY frame. Rotating here is not optional: handing the
        // world vector back unchanged is a silent error that a position-only metric
        // never shows, and it is one of the five the Python adapter shipped.
        let velocity = rotation.inverse().act(&nav.t.velocity);
        NavEstimate {
            pose: SE3::new(rotation, nav.t.position),
            velocity,
            gyro_bias: Vector3::new(nav.b[0], nav.b[1], nav.b[2]),
            accel_bias: Vector3::new(nav.b[3], nav.b[4], nav.b[5]),
        }
    }
}

impl LandmarkMap for MsceqfBackend {
    fn landmarks_world(&self) -> HashMap<u64, Vector3<f64>> {
        let mut out = HashMap::new();
        for (j, &tid) in self.landmark_ids.iter().enumerate() {
            let lm = &self.filter.x.landmarks[j];
            if let Some(anchor) = self.filter.x.clones.iter().find(|c| c.stamp == lm.anchor) {
                out.insert(tid, anchor.pose.act(&lm.point()));
            }
        }
        out
    }
}

#[cfg(test)]
mod msceqf_backend_tests {
    use super::*;
    use crate::mathematical::msceqf_filter::{MSCEqFFilter, ProcessNoise, SystemOrigin};
    use echo_lie::{SE23, SO3};
    use nalgebra::{SMatrix, Vector6};

    /// Constant world velocity along +x. The filter is seeded with it, and the
    /// synthetic observations are generated from the SAME motion, so the geometry the
    /// filter is asked to explain is the geometry that produced the pixels. Without
    /// that, every clone sits at one pose, triangulation is degenerate, and the chi2
    /// gate silently refuses every birth -- which is how the first version of these
    /// tests passed while exercising nothing.
    const VX: f64 = 2.0;

    fn make_backend(settings: MsceqfSettings) -> MsceqfBackend {
        let origin = SystemOrigin::new(
            SE23::new(
                SO3::identity(),
                Vector3::zeros(),
                Vector3::new(VX, 0.0, 0.0),
            ),
            Vector6::zeros(),
            SE3::identity(),
            Vector3::new(0.0, 0.0, -9.81),
        );
        let d_init = SMatrix::<f64, 9, 9>::identity() * 1e-4;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 1e-4;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let noise = ProcessNoise {
            angular_velocity_std: 1e-4,
            acceleration_std: 1e-3,
            angular_velocity_bias_std: 1e-6,
            acceleration_bias_std: 1e-5,
            state_transition_order: 1,
        };
        let filter = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, noise);
        MsceqfBackend::new(filter, test_cam(), settings)
    }

    fn test_cam() -> Arc<dyn CameraModel> {
        Arc::new(crate::mathematical::camera::PinholeModel {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
        })
    }

    /// World points spread in depth so the grid has real structure, not one plane.
    fn world_points(n: u64) -> Vec<Vector3<f64>> {
        (0..n)
            .map(|i| {
                let col = (i % 5) as f64;
                let row = (i / 5) as f64;
                Vector3::new(
                    -1.5 + col * 0.75,
                    -0.8 + row * 0.5,
                    4.0 + ((i * 7) % 5) as f64 * 0.9,
                )
            })
            .collect()
    }

    /// Project the world grid through the TRUE camera pose at time `t`: identity
    /// attitude, position (VX*t, 0, 0). Features behind the camera are dropped.
    fn frame_at(t: f64, ids: &[u64]) -> VisionMeasurement {
        let cam = test_cam();
        let p_cam = Vector3::new(VX * t, 0.0, 0.0);
        let pts = world_points(40);
        let mut coords = HashMap::new();
        for &i in ids {
            let p = pts[i as usize] - p_cam;
            if p[2] < 0.5 {
                continue;
            }
            let uv = cam.project(&p);
            if uv[0] > 0.0 && uv[0] < 640.0 && uv[1] > 0.0 && uv[1] < 480.0 {
                coords.insert(i, Vector2::new(uv[0] as f32, uv[1] as f32));
            }
        }
        VisionMeasurement::new(t, coords)
    }

    fn imu(stamp: f64) -> IMUVelocity {
        // Constant velocity under gravity -g*z: specific force is +9.81 along body z.
        IMUVelocity::new(stamp, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81))
    }

    /// The clone window is BOUNDED. An MSCKF whose window grows without limit is the
    /// failure this backend's step (7) exists to prevent, and it costs quadratic
    /// covariance work before it is noticed.
    #[test]
    fn clone_window_stays_bounded() {
        let mut b = make_backend(MsceqfSettings {
            num_clones: 6,
            ..Default::default()
        });
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..40u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
            assert!(
                b.n_clones() <= 6,
                "clone window grew to {} at frame {k}",
                b.n_clones()
            );
        }
        assert_eq!(b.n_clones(), 6, "window should be full, not starved");
    }

    /// Buffered observations must never reference a clone that has been marginalized.
    /// They are keyed by clone ID precisely because positions shift underneath them,
    /// and a stale id silently drops an observation at best.
    #[test]
    fn buffer_never_references_a_dead_clone() {
        let mut b = make_backend(MsceqfSettings {
            num_clones: 5,
            update_span: 100, // keep tracks in the buffer rather than flushing them
            max_track: 100,
            ..Default::default()
        });
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..30u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
            for (tid, obs) in &b.buf {
                for o in obs {
                    assert!(
                        b.clone_ids.contains(&o.0),
                        "track {tid} holds dead clone {} at frame {k}; live {:?}",
                        o.0,
                        b.clone_ids
                    );
                }
            }
        }
    }

    /// `landmark_ids` is parallel to the filter's landmark list. If these ever drift,
    /// every later update, re-anchor and marginalize addresses the WRONG landmark --
    /// and the state stays plausible while meaning nothing.
    #[test]
    fn landmark_ids_stay_parallel_to_the_filter() {
        let mut b = make_backend(MsceqfSettings {
            num_clones: 8,
            update_span: 8,
            max_slam: 6,
            ..Default::default()
        });
        let mut born_max = 0usize;
        let mut slam_before_drop = 0usize;
        for k in 0..45u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            // Drop a couple of tracks midway so landmarks are lost and marginalized.
            // Drop a few tracks midway so landmarks are lost and marginalized.
            let ids: Vec<u64> = if k > 25 {
                (0..14).collect()
            } else {
                (0..20).collect()
            };
            b.process_vision(frame_at(t, &ids), &HashMap::new());
            assert_eq!(
                b.landmark_ids.len(),
                b.filter().n_landmarks(),
                "id table and filter disagree at frame {k}"
            );
            assert!(b.n_slam() <= 6, "exceeded max_slam at frame {k}");
            born_max = born_max.max(b.n_slam());
            if k == 25 {
                slam_before_drop = b.n_slam();
            }
        }
        // Non-vacuity: a run in which nothing was ever promoted would satisfy every
        // assertion above while exercising none of the code they are about.
        assert!(
            born_max >= 3,
            "only {born_max} landmark(s) ever promoted; too few to exercise the paths"
        );
        // Dropping tracks at k>25 must actually cost in-state landmarks, so the
        // marginalize-on-loss branch is exercised and not merely compiled.
        assert!(
            b.n_slam() < slam_before_drop,
            "losing tracks did not marginalize any landmark ({slam_before_drop} -> {})",
            b.n_slam()
        );
    }

    /// `NavEstimate.velocity` is BODY frame; MSCEqF carries world. Rotating back is
    /// invisible to a position-only error metric, which is how it shipped unnoticed
    /// once already -- so it is pinned here rather than trusted.
    #[test]
    fn nav_estimate_velocity_is_body_frame() {
        let v_world = Vector3::new(3.0, -1.0, 0.5);
        let r = SO3::exp(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        let origin = SystemOrigin::new(
            SE23::new(r, Vector3::zeros(), v_world),
            Vector6::zeros(),
            SE3::identity(),
            Vector3::new(0.0, 0.0, -9.81),
        );
        let d_init = SMatrix::<f64, 9, 9>::identity() * 1e-6;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let noise = ProcessNoise {
            angular_velocity_std: 1e-6,
            acceleration_std: 1e-6,
            angular_velocity_bias_std: 1e-8,
            acceleration_bias_std: 1e-8,
            state_transition_order: 1,
        };
        let filter = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, noise);
        let b = MsceqfBackend::new(filter, test_cam(), MsceqfSettings::default());

        let nav = b.nav_estimate();
        let expected_body = nav.pose.rotation.inverse().act(&v_world);
        assert!(
            (nav.velocity - expected_body).norm() < 1e-12,
            "velocity not rotated into body: got {:?}, expected {:?}",
            nav.velocity,
            expected_body
        );
        // And it is genuinely a different vector, so the test cannot pass by accident
        // on an identity rotation.
        assert!(
            (nav.velocity - v_world).norm() > 1.0,
            "rotation was identity; the test proves nothing"
        );
    }
}
