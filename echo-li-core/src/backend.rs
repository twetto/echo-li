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

    /// Order-independent fingerprint of the backend's own state, for determinism
    /// traces: two runs of one build over one input must produce the same sequence.
    ///
    /// Deliberately NOT a structured state dump. What a backend should hash is its own
    /// state in its own representation; the only contract is that the value is stable
    /// for identical runs and changes when the state does. Comparing fingerprints
    /// ACROSS backends is meaningless and is not the point.
    fn state_fingerprint(&self) -> u64;

    /// Assert "the body is not moving" as a measurement. The caller decides WHEN,
    /// from a [`crate::initialization::ZeroVelocityDetector`]; the backend decides
    /// whether the state agrees enough to accept it. Returns false if unsupported or
    /// gated out.
    ///
    /// Default is "unsupported", not "did nothing": a backend that silently ignored
    /// this would leave a caller believing the rig was being held at rest when it was
    /// free-running.
    fn zero_velocity_update(&mut self, _sigma_v: f64) -> bool {
        false
    }

    /// Pose of the CAMERA in the world frame.
    ///
    /// Not derivable from [`NavEstimate`]: that carries the body pose, and the
    /// body-to-camera transform is part of the backend's state -- both implementors
    /// estimate it online rather than holding it fixed. Consumers that reproject into
    /// the image, or feed a depth filter, need this one and not the body pose.
    fn camera_pose(&self) -> SE3;
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

/// What a full wrapper needs behind a trait object.
///
/// `Box<dyn VioBackend>` cannot reach the capability traits -- a trait object carries
/// exactly one vtable -- so a wrapper that stores one needs a single trait naming
/// everything it requires. Landmarks are required, because every wrapper publishes or
/// renders them. Camera-pose covariance stays OPTIONAL and is reached through
/// [`Self::pose_covariance`], which returns `None` for a backend that cannot supply it:
/// collapsing that into a required method would force an implementor to invent a
/// covariance, and "I cannot tell you" must not become "my uncertainty is zero".
pub trait VioBackendFull: VioBackend + LandmarkMap {
    fn pose_covariance(&self) -> Option<&dyn CameraPoseCovariance> {
        None
    }
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

    fn camera_pose(&self) -> SE3 {
        let state = self.filter.eqf.state_estimate();
        state.sensor.pose.compose(&state.sensor.camera_offset)
    }

    /// EqVIO already had the machinery, as a GT-velocity injection diagnostic: a ZUPT
    /// is that measurement with v = 0.
    fn zero_velocity_update(&mut self, sigma_v: f64) -> bool {
        self.filter
            .velocity_pseudo_update(Vector3::zeros(), sigma_v, 1.0);
        true
    }

    /// Byte-compatible with the hash the CLI computed inline before this seam existed,
    /// so determinism traces recorded either side of the change stay comparable:
    /// attitude as xyzw, then position, velocity, bias, then each landmark's id and its
    /// CAMERA-frame point, in state order.
    fn state_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let state = self.filter.eqf.state_estimate();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in state.sensor.pose.rotation.as_xyzw().iter() {
            v.to_bits().hash(&mut h);
        }
        for v in state.sensor.pose.translation.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in state.sensor.velocity.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in state.sensor.input_bias.iter() {
            v.to_bits().hash(&mut h);
        }
        for lm in &state.camera_landmarks {
            lm.id.hash(&mut h);
            for v in lm.p.iter() {
                v.to_bits().hash(&mut h);
            }
        }
        h.finish()
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

impl VioBackendFull for EqvioBackend {
    fn pose_covariance(&self) -> Option<&dyn CameraPoseCovariance> {
        Some(self)
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
///
/// Defaults mirror the reference MidAir ATE harness, which is the configuration the
/// published figures were measured under.
#[derive(Clone, Debug)]
pub struct MsceqfSettings {
    /// Pose clones kept in the window. Older ones marginalize.
    pub num_clones: usize,
    /// Measurement noise in NORMALIZED image units. The backend normalizes pixels
    /// before they reach the filter, so this is not a pixel count: the reference uses
    /// `1.0 / fx`.
    pub pixel_std: f64,
    /// Chi-squared gate multiplier for the structureless update.
    pub chi2_mult: f64,
    /// Chi-squared gate for SLAM birth and streaming update. `None` reuses
    /// `chi2_mult`.
    pub slam_chi2_mult: Option<f64>,
    /// Hard cap on observations buffered for one structureless track.
    pub max_track: usize,
    /// Minimum track length to promote to an in-state landmark. `None` means
    /// `num_clones`, i.e. a full-window span.
    pub slam_min_len: Option<usize>,
    /// Maximum in-state SLAM landmarks. 0 = structureless only, which is the WEAKEST
    /// configuration of this filter.
    pub max_slam: usize,
    /// Second-order curvature term in the structureless update.
    pub curvature: bool,
    /// Refuse to promote a landmark whose triangulated range exceeds this (metres).
    /// 0 = no cap.
    ///
    /// Promotion picks the longest window-spanning tracks, and on a forward-moving
    /// vehicle those are the DISTANT features -- near ones leave the frame quickly. The
    /// budget then fills with points that carry attitude but not translation: measured
    /// on the rig's ride segment, in-state landmarks sat at a median 138 m against
    /// EqVIO's 9 m, which at fx=227 is 1.6 px of disparity per metre travelled. A
    /// landmark that cannot see the camera move cannot hold the scale.
    pub max_promote_range: f64,

    /// EXTRA flush trigger: consume a track once it reaches this many observations,
    /// even if it is neither lost nor touching the clone about to marginalize. 0 =
    /// off, which is the normal configuration.
    ///
    /// This exists for one case. The depth work runs a LONG clone window so Sparse3D
    /// has anchors, and with a long window `touches_marg` almost never fires, so
    /// almost no update fires and the filter runs near open-loop. Such a
    /// configuration sets this to decouple the update span from the anchor window.
    /// Turning it on with an ordinary 11-clone window makes updates fire far earlier
    /// than the reference and is NOT the validated behaviour.
    pub update_span: usize,
}

impl Default for MsceqfSettings {
    fn default() -> Self {
        Self {
            num_clones: 11,
            pixel_std: 1.0,
            chi2_mult: 1.0,
            slam_chi2_mult: None,
            max_track: 200,
            slam_min_len: None,
            max_slam: 0,
            curvature: true,
            update_span: 0,
            max_promote_range: 0.0,
        }
    }
}

impl MsceqfSettings {
    fn slam_chi2(&self) -> f64 {
        self.slam_chi2_mult.unwrap_or(self.chi2_mult)
    }

    fn min_promote_len(&self) -> usize {
        self.slam_min_len.unwrap_or(self.num_clones)
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
    /// Running totals: tracks offered to the structureless update, and tracks the
    /// filter actually accepted. A backend that is silently accepting nothing looks
    /// exactly like one that is running well until the trajectory diverges.
    msc_offered: u64,
    msc_accepted: u64,
    last_stamp: Option<f64>,
}

impl MsceqfBackend {
    /// Build from an `MSCEqF:` config section and the state gravity alignment produced.
    ///
    /// `start` supplies the attitude, position and velocity; `fx` converts the config's
    /// PIXEL noise into the normalized units the filter works in, because this backend
    /// normalizes observations through `cam` before they reach the filter.
    ///
    /// Note the initial bias is zero, matching how the EqVIO path starts: gravity
    /// alignment at rest cannot observe accelerometer bias -- the bias is parallel to
    /// measured gravity by construction -- so seeding a non-zero one would be inventing
    /// information.
    pub fn from_config(
        cfg: &crate::config::MsceqfConfig,
        cam: Arc<dyn CameraModel>,
        fx: f64,
        start: &crate::initialization::FilterStart,
    ) -> Self {
        use crate::mathematical::msceqf_filter::{MSCEqFFilter, ProcessNoise, SystemOrigin};
        use nalgebra::{SMatrix, Vector6};

        let sensor = &start.state.sensor;
        let origin = SystemOrigin::new(
            echo_lie::SE23::new(
                sensor.pose.rotation,
                sensor.pose.translation,
                sensor.velocity,
            ),
            Vector6::zeros(),
            sensor.camera_offset.clone(),
            Vector3::new(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]),
        );

        let sd = &cfg.initial_std;
        // Dd block in MSCEqF tangent order: attitude, velocity, position.
        let d_std = [
            sd.attitude,
            sd.attitude,
            sd.attitude,
            sd.velocity,
            sd.velocity,
            sd.velocity,
            sd.position,
            sd.position,
            sd.position,
        ];
        let d_init =
            SMatrix::<f64, 9, 9>::from_fn(|i, j| if i == j { d_std[i] * d_std[i] } else { 0.0 });
        let delta_init =
            SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { sd.bias * sd.bias } else { 0.0 });
        let e_init = SMatrix::<f64, 6, 6>::from_fn(|i, j| {
            if i == j {
                sd.extrinsics * sd.extrinsics
            } else {
                0.0
            }
        });

        let pn = &cfg.process_noise;
        let filter = MSCEqFFilter::new(
            origin,
            &d_init,
            &delta_init,
            &e_init,
            ProcessNoise {
                angular_velocity_std: pn.gyro_density,
                acceleration_std: pn.accel_density,
                angular_velocity_bias_std: pn.gyro_random_walk,
                acceleration_bias_std: pn.accel_random_walk,
                state_transition_order: pn.transition_order,
            },
        );

        let settings = MsceqfSettings {
            num_clones: cfg.num_clones,
            // Config carries pixels; the filter wants normalized units.
            pixel_std: cfg.pixel_std / fx,
            chi2_mult: cfg.chi2_mult,
            slam_chi2_mult: (cfg.slam_chi2_mult > 0.0).then_some(cfg.slam_chi2_mult),
            max_track: cfg.max_track,
            slam_min_len: (cfg.slam_min_len > 0).then_some(cfg.slam_min_len),
            max_slam: cfg.max_slam,
            curvature: cfg.curvature,
            update_span: cfg.update_span,
            max_promote_range: cfg.max_promote_range,
        };
        Self::new(filter, cam, settings)
    }

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
            msc_offered: 0,
            msc_accepted: 0,
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

    /// Track ids of the in-state landmarks, parallel to the filter's landmark list.
    pub fn landmark_ids_snapshot(&self) -> Vec<u64> {
        self.landmark_ids.clone()
    }

    /// `(tracks offered, tracks accepted)` by the structureless update so far.
    pub fn msc_update_counts(&self) -> (u64, u64) {
        (self.msc_offered, self.msc_accepted)
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
    /// 3. promote tracks that span the window to in-state, re-anchoring each onto the
    ///    current clone so a fresh landmark never depends on a clone about to die;
    /// 4. flush structureless tracks into one MSC update;
    /// 5. stream-update the in-state landmarks still visible, and marginalize the rest
    ///    -- an in-state landmark with no observation contributes nothing and holds
    ///    state;
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
        //
        // Gated on a clone being about to marginalize, and on the track actually
        // REACHING that clone: a track that spans the full window is the one whose
        // geometry is about to be thrown away, and is therefore the one worth keeping
        // in the state. Promoting merely long tracks instead would spend the budget on
        // features that the structureless update can still see.
        if self.settings.max_slam > 0 && marg.is_some() && self.slam.len() < self.settings.max_slam
        {
            let m = marg.unwrap();
            let min_len = self.settings.min_promote_len();
            let mut cand: Vec<u64> = self
                .buf
                .iter()
                .filter(|(t, o)| {
                    seen.contains(t) && o.len() >= min_len && o.iter().any(|x| x.0 == m)
                })
                .map(|(t, _)| *t)
                .collect();
            // Longest track first, track id breaking ties, so the order is TOTAL.
            //
            // The Python reference sorted by length alone. Python's sort is stable, so
            // its ties fell in dict INSERTION order -- and a track that is flushed and
            // later re-observed is re-inserted at the end, so that order is neither id
            // order nor arrival order, and is not reproducible from the data alone.
            // Which equally-long track wins the last SLAM slot is arbitrary either way;
            // a total order at least makes it the same arbitrary choice every run.
            // Verified: with the reference's tie-break replaced by this one, the two
            // implementations agree to 0.000000 m over a full 7866-frame trajectory.
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
                    self.settings.slam_chi2(),
                    None,
                );
                if let Some(j) = born {
                    // Reject a landmark that was born too far to inform translation.
                    if self.settings.max_promote_range > 0.0 {
                        let r = self.filter.x.landmarks[j].point().norm();
                        if !(r.is_finite() && r <= self.settings.max_promote_range) {
                            self.filter.marginalize_landmark(j);
                            continue;
                        }
                    }
                    debug_assert_eq!(j, self.landmark_ids.len(), "birth appends at the tail");
                    self.landmark_ids.push(tid);
                    // Birth anchors on the OLDEST observing clone, which here is the
                    // one about to marginalize. Move the anchor onto the newest clone
                    // before the slide, or the landmark loses its frame of reference.
                    if let Some(pos) = self.clone_ids.iter().position(|&c| c == k) {
                        self.filter.reanchor_landmark(j, pos);
                    }
                    self.slam.insert(tid);
                    self.buf.remove(&tid);
                }
            }
        }

        // (4) Flush structureless tracks into one MSC update.
        //
        // A track's measurements are consumed exactly once, when it is lost, at its
        // length cap, or about to lose its oldest observation to marginalization.
        // Dropping a track at the marginalized clone without consuming it would
        // discard most of the geometric constraint and let scale drift.
        let mut ready: Vec<crate::mathematical::msceqf_filter::MscTrack> = Vec::new();
        let mut done: Vec<u64> = Vec::new();
        for (&tid, obs) in &self.buf {
            let lost = !seen.contains(&tid);
            let too_long = obs.len() >= self.settings.max_track;
            let touches_marg = marg.is_some_and(|m| obs.iter().any(|o| o.0 == m));
            let spanned = self.settings.update_span > 0 && obs.len() >= self.settings.update_span;
            if lost || too_long || touches_marg || spanned {
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
            self.msc_offered += ready.len() as u64;
            self.msc_accepted += self.filter.msc_update(
                &ready,
                &self.z1,
                self.settings.pixel_std,
                self.settings.chi2_mult,
                self.settings.curvature,
            ) as u64;
        }
        for tid in done {
            self.buf.remove(&tid);
        }

        // (5) Stream-update visible in-state landmarks; marginalize the rest. Each
        // in-state landmark contributes ONLY its fresh bearing, so no measurement is
        // reused. A landmark whose track ended is evicted: it can no longer contribute
        // and it holds state.
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
                    self.settings.slam_chi2(),
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

    fn camera_pose(&self) -> SE3 {
        self.filter.camera_pose()
    }

    fn zero_velocity_update(&mut self, sigma_v: f64) -> bool {
        self.filter.zero_velocity_update(sigma_v, 1.0)
    }

    /// Nav state plus the in-state landmarks in their own anchor frame. Track ids come
    /// from `landmark_ids`, which is kept parallel to the filter's landmark list.
    fn state_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let nav = self.filter.phi();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in nav.t.rotation.as_xyzw().iter() {
            v.to_bits().hash(&mut h);
        }
        for v in nav.t.position.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in nav.t.velocity.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in nav.b.iter() {
            v.to_bits().hash(&mut h);
        }
        for (j, &tid) in self.landmark_ids.iter().enumerate() {
            tid.hash(&mut h);
            for v in self.filter.x.landmarks[j].point().iter() {
                v.to_bits().hash(&mut h);
            }
        }
        h.finish()
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

/// Landmarks yes, camera-pose covariance not yet: MSCEqF carries covariance on the
/// group algebra over its clone window, and projecting that onto a single camera pose
/// is a derivation this backend does not do. Reporting `None` is the honest answer
/// until it does.
impl VioBackendFull for MsceqfBackend {}

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
            slam_min_len: Some(6),
            max_slam: 6,
            ..Default::default()
        });
        let mut born_max = 0usize;
        let mut evicted = 0usize;
        let mut prev_ids: BTreeSet<u64> = BTreeSet::new();
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
            // An id that was in-state and is no longer is an eviction. Counting the
            // set size instead would miss it, because promotions refill the budget.
            let now: BTreeSet<u64> = b.landmark_ids.iter().copied().collect();
            evicted += prev_ids.difference(&now).count();
            prev_ids = now;
        }
        // Non-vacuity: a run in which nothing was ever promoted would satisfy every
        // assertion above while exercising none of the code they are about.
        assert!(
            born_max >= 3,
            "only {born_max} landmark(s) ever promoted; too few to exercise the paths"
        );
        // Dropping tracks must actually evict in-state landmarks, so the
        // marginalize-on-loss branch is exercised and not merely compiled.
        assert!(
            evicted > 0,
            "no landmark was ever evicted; the marginalize-on-loss branch never ran"
        );
    }

    /// The config path must build the SAME filter as direct construction. Every field
    /// here is a chance to transpose something silently: the Dd block's tangent order
    /// (attitude, velocity, position), the pixel-to-normalized conversion, the gravity
    /// sign, and the noise mapping (gyro density -> angular_velocity_std, accel random
    /// walk -> acceleration_bias_std). A misplaced one still runs, and still produces a
    /// plausible trajectory.
    #[test]
    fn config_path_matches_direct_construction() {
        use crate::config::{MsceqfConfig, MsceqfInitialStd, MsceqfProcessNoise};
        use crate::initialization::StationaryInit;

        let fx = 400.0;
        let cfg = MsceqfConfig {
            enabled: true,
            num_clones: 7,
            pixel_std: 1.5, // PIXELS; the filter must see 1.5/fx
            chi2_mult: 1.3,
            slam_chi2_mult: 0.0,
            max_track: 123,
            slam_min_len: 0,
            max_slam: 4,
            curvature: true,
            update_span: 0,
            gravity: [0.0, 0.0, -9.81],
            initial_std: MsceqfInitialStd {
                attitude: 2.0e-2,
                velocity: 3.0e-1,
                position: 4.0e-4,
                bias: 5.0e-2,
                extrinsics: 6.0e-4,
            },
            process_noise: MsceqfProcessNoise {
                accel_density: 1.6798e-3,
                gyro_density: 1.2724e-3,
                accel_random_walk: 3.0e-3,
                gyro_random_walk: 1.0e-4,
                transition_order: 0,
            },
        };

        // Gravity-align from a level rest, as the wrappers do, then set the velocity
        // the synthetic imagery actually implies. StationaryInit zeroes velocity, which
        // is right for a real rest start -- but leaving it zero here makes the frames
        // describe a 2 m/s camera while the filter believes it is stationary, so every
        // update is chi2-rejected and the comparison below becomes vacuous.
        let samples: Vec<IMUVelocity> = (0..10)
            .map(|k| {
                IMUVelocity::new(
                    k as f64 * 0.01,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, 9.81),
                )
            })
            .collect();
        let mut start = StationaryInit::from_samples(10, SE3::identity(), samples);
        start.state.sensor.velocity = Vector3::new(VX, 0.0, 0.0);

        let mut from_cfg = MsceqfBackend::from_config(&cfg, test_cam(), fx, &start);

        // The same thing, spelled out.
        let sd = &cfg.initial_std;
        let d = [
            sd.attitude,
            sd.attitude,
            sd.attitude,
            sd.velocity,
            sd.velocity,
            sd.velocity,
            sd.position,
            sd.position,
            sd.position,
        ];
        let origin = SystemOrigin::new(
            SE23::new(
                start.state.sensor.pose.rotation,
                start.state.sensor.pose.translation,
                start.state.sensor.velocity,
            ),
            Vector6::zeros(),
            start.state.sensor.camera_offset.clone(),
            Vector3::new(0.0, 0.0, -9.81),
        );
        let filter = MSCEqFFilter::new(
            origin,
            &SMatrix::<f64, 9, 9>::from_fn(|i, j| if i == j { d[i] * d[i] } else { 0.0 }),
            &SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { 5.0e-2 * 5.0e-2 } else { 0.0 }),
            &SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { 6.0e-4 * 6.0e-4 } else { 0.0 }),
            ProcessNoise {
                angular_velocity_std: 1.2724e-3,
                acceleration_std: 1.6798e-3,
                angular_velocity_bias_std: 1.0e-4,
                acceleration_bias_std: 3.0e-3,
                state_transition_order: 0,
            },
        );
        let mut direct = MsceqfBackend::new(
            filter,
            test_cam(),
            MsceqfSettings {
                num_clones: 7,
                pixel_std: 1.5 / fx,
                chi2_mult: 1.3,
                slam_chi2_mult: None,
                max_track: 123,
                slam_min_len: None,
                max_slam: 4,
                curvature: true,
                update_span: 0,
            },
        );

        assert_eq!(
            from_cfg.filter().cov.shape(),
            direct.filter().cov.shape(),
            "covariance dimensions differ"
        );
        assert!(
            (from_cfg.filter().cov.clone() - direct.filter().cov.clone()).amax() < 1e-18,
            "initial covariance differs: the config mapping transposed something"
        );

        // Drive both and compare the fingerprint, so settings (not just covariance)
        // are covered: pixel_std, chi2 and the window all act during the run.
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..30u64 {
            let t = k as f64 * 0.05;
            from_cfg.process_imu(imu(t));
            direct.process_imu(imu(t));
            let frame = frame_at(t, &ids);
            from_cfg.process_vision(frame.clone(), &HashMap::new());
            direct.process_vision(frame, &HashMap::new());
            assert_eq!(
                from_cfg.state_fingerprint(),
                direct.state_fingerprint(),
                "diverged at frame {k}"
            );
        }
        // NON-VACUITY. The loop above only proves equivalence if the settings it is
        // comparing actually BITE. A control with a deliberately wrong pixel noise must
        // diverge; if it does not, this scenario never exercises the update and the
        // equivalence above is empty. (Found the hard way: mutating away the
        // pixel-to-normalized conversion left the original version of this test green.)
        let mut perturbed = MsceqfBackend::from_config(
            &MsceqfConfig {
                pixel_std: cfg.pixel_std * 100.0,
                ..cfg.clone()
            },
            test_cam(),
            fx,
            &start,
        );
        for k in 0..30u64 {
            let t = k as f64 * 0.05;
            perturbed.process_imu(imu(t));
            perturbed.process_vision(frame_at(t, &ids), &HashMap::new());
        }
        assert_ne!(
            perturbed.state_fingerprint(),
            direct.state_fingerprint(),
            "a 100x pixel noise changed nothing: this scenario never exercises the \
             update, so the equivalence check above proves nothing"
        );
        assert!(from_cfg.n_clones() > 1, "nothing ran");
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
