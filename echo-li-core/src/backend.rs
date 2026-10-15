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

use crate::mathematical::camera::CameraModel;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vision_measurement::VisionMeasurement;
use crate::LandmarkDepthPrior;
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

    /// Gauge-cancelled anchor->current relative-pose covariance `(P_vv, P_ww)` between
    /// the clone `clone_id` and the current camera pose. This, not the absolute pair,
    /// is what the Sparse3D depth NEES was calibrated on: the cross term between the two
    /// clones cancels the gauge they share, which neither marginal can. `t_wc_clone` is
    /// the anchor's pose as the consumer stored it; an MSCKF backend, whose anchor IS a
    /// live clone, may ignore it. `None` when the clone is not live.
    fn relative_pose_covariances(
        &self,
        _clone_id: u64,
        _t_wc_clone: &SE3,
    ) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        None
    }

    /// The FULL 6x6 anchor->current relative covariance, tangent order
    /// `[rot(0:3), trans(3:6)]`, for consumers that need the rotation-translation
    /// cross block. `None` when unsupported or the clone is not live.
    fn relative_pose_covariance_full(&self, _clone_id: u64) -> Option<nalgebra::Matrix6<f64>> {
        None
    }

    /// Joint 12x12 covariance of the two relative poses `anchor->prev` (block 0) and
    /// `anchor->current` (block 1), both live clones. What the depth filter's consider
    /// recursion needs to regress this frame's relative-pose error on last frame's.
    /// `None` when unsupported or a clone is not live.
    fn relative_pose_joint_covariance(
        &self,
        _clone_id: u64,
        _prev_clone_id: u64,
    ) -> Option<nalgebra::SMatrix<f64, 12, 12>> {
        None
    }
    /// Ids of the clones a consumer may anchor on right now, oldest first.
    fn live_clone_ids(&self) -> Vec<u64> {
        Vec::new()
    }

    /// The clone that IS the current camera pose, if the backend manages its own
    /// window (an MSCKF does; EqVIO's clones are the caller's).
    fn current_clone_id(&self) -> Option<u64> {
        None
    }

    /// The clone the backend evicted during its last frame step, with the evidence a
    /// depth filter needs to move landmarks anchored on it: captured BEFORE the
    /// marginalization, while the old->new relative pose was still computable.
    fn last_clone_eviction(&self) -> Option<&CloneEviction> {
        None
    }
}

/// Evidence of a clone eviction, for re-anchoring landmarks before their anchor dies.
#[derive(Debug, Clone)]
pub struct CloneEviction {
    pub old_clone_id: u64,
    pub new_clone_id: u64,
    /// The new anchor's camera pose (world <- camera).
    pub t_wc_new: SE3,
    /// Old->new relative-pose covariance `(P_vv, P_ww)`, gauge-cancelled.
    pub p_vv: Matrix3<f64>,
    pub p_ww: Matrix3<f64>,
}

/// Nearest positive-semidefinite matrix: clip negative eigenvalues of the symmetrized
/// input to zero. The per-frame INCREMENT of a relative-pose covariance,
/// `Cov_rel(k -> t) - Cov_rel(k -> t-1)`, can be momentarily non-PSD after a measurement
/// update; the accumulating range injection must only ever add variance.
pub fn psd_clip(m: &Matrix3<f64>) -> Matrix3<f64> {
    let sym = (m + m.transpose()) * 0.5;
    let eig = nalgebra::SymmetricEigen::new(sym);
    let mut d = Matrix3::zeros();
    for i in 0..3 {
        d[(i, i)] = eig.eigenvalues[i].max(0.0);
    }
    eig.eigenvectors * d * eig.eigenvectors.transpose()
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
/// `Send + Sync` is part of the contract: the ROS 2 node drives a backend from its
/// executor threads and the Python binding has to release the GIL around a frame step,
/// so a backend that cannot cross a thread is not usable by two of the three wrappers.
pub trait VioBackendFull: VioBackend + LandmarkMap + Send + Sync {
    fn pose_covariance(&self) -> Option<&dyn CameraPoseCovariance> {
        None
    }

    /// Downcast hook, so a caller that needs something one backend alone provides --
    /// EqVIO's pose clones, its full covariance, its gramian -- can reach the concrete
    /// type behind the trait object instead of the shared trait growing a method for
    /// one caller. A caller that downcasts to the wrong backend gets `None`, which is
    /// the honest answer: the capability is not there.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// IMU samples on which an installed [`ZuptPolicy`] was accepted; 0 without one.
    /// On the seam because every wrapper should be able to report it: a run whose ZUPT
    /// never fired on a rest start is the single most expensive silent failure this
    /// codebase has measured.
    fn zupt_fired(&self) -> u64 {
        0
    }
}

/// Zero-velocity policy a backend runs INSIDE `process_imu`: the detector decides
/// stationarity from IMU variability, the backend then asserts v = 0 through its own
/// `zero_velocity_update`. Owned by the backend rather than the wrapper so the CLI, the
/// ROS 2 node and the Python binding cannot each forget it -- before this, the only
/// ZUPT in use anywhere was a Python reimplementation inside a diagnostic harness, and
/// no production wrapper ran one.
pub struct ZuptPolicy {
    detector: crate::initialization::ZeroVelocityDetector,
    sigma_v: f64,
    /// Samples on which the update was offered and accepted. Exposed because a policy
    /// that silently never fires is indistinguishable from one that works, until the
    /// rest phase has drifted hundreds of metres.
    fired: u64,
}

impl ZuptPolicy {
    pub fn new(
        window: usize,
        gyro_std_threshold: f64,
        acc_std_threshold: f64,
        sigma_v: f64,
    ) -> Self {
        Self {
            detector: crate::initialization::ZeroVelocityDetector::new(
                window,
                gyro_std_threshold,
                acc_std_threshold,
            ),
            sigma_v,
            fired: 0,
        }
    }

    /// `None` when the section is absent or disabled.
    pub fn from_config(cfg: Option<&crate::config::ZuptConfig>) -> Option<Self> {
        let c = cfg.filter(|c| c.enabled)?;
        Some(Self::new(
            c.window,
            c.gyro_std_threshold,
            c.acc_std_threshold,
            c.sigma_v,
        ))
    }

    /// Feed one sample; `Some(sigma_v)` when the backend should assert zero velocity now.
    fn observe(&mut self, imu: &IMUVelocity) -> Option<f64> {
        self.detector.push(imu.clone()).then_some(self.sigma_v)
    }

    pub fn fired(&self) -> u64 {
        self.fired
    }
}

/// Feeds a Sparse3D depth filter from a backend's pose covariance the way the depth
/// NEES harness did when it calibrated the filter: clone-RELATIVE, with re-anchoring.
///
/// Owned by the consumer (node, CLI) and called once per frame. If the backend manages
/// its own clone window (`current_clone_id()` is `Some`), each landmark is driven by
/// the gauge-cancelled anchor->current covariance of its own anchor clone, the
/// per-frame INCREMENT of that covariance is PSD-clipped for the accumulating range
/// injection, and every frame the landmarks sitting on the OLDEST live clone are moved
/// onto the PREVIOUS frame's clone -- the pose this feed registered for it a frame ago
/// -- with the oldest->current relative covariance. If the backend does not manage
/// clones (EqVIO), this is exactly the legacy absolute `update`. Shared because the
/// first version of this logic lived in a Python harness and nowhere else.
///
/// The re-anchor rule is the harness's, reproduced to the frame, and not the obvious
/// alternative (move landmarks when the backend evicts their clone, onto the newest
/// clone, with the pose the filter holds for it). The two were measured on all 30
/// Kite_training trajectories with the same backend: the harness rule is honest
/// (tail <= the Student-t reference) on 26/30, the eviction rule on 22/30, with the
/// >chi2_99 tail systematically heavier under eviction. Why one frame and one clone of
/// difference does that is NOT established; what is established is which rule the
/// filter formulation was validated with, so that is the one that ships.
pub struct CloneRelFeed {
    prev: HashMap<u64, (Matrix3<f64>, Matrix3<f64>)>,
    /// Camera pose this feed passed to the depth filter for each live clone -- the
    /// anchor pose Sparse3D holds for it, which is what a landmark must be moved
    /// ONTO (not the filter's later-corrected clone pose).
    poses: HashMap<u64, nalgebra::Matrix4<f64>>,
    /// DIAGNOSTIC sweep knob, `None` in production. `Some(a)`: instead of the harness
    /// rule, move every landmark whose anchor clone is at least `a` frames old onto
    /// the previous frame's clone (same target, same covariance convention). Varies
    /// the re-anchor PERIOD, which the harness rule fixes at window-1, so the two
    /// measured rules' 26/30-vs-22/30 gap can be attributed or not to how often a
    /// landmark is re-anchored.
    pub reanchor_age: Option<u64>,
    /// `false` in production (since 2026-10-15). `true` re-anchors WITH the
    /// old->current relative-pose covariance added to the landmark; under
    /// `pose_range_direct` the depth filter projects the radial direction out of it,
    /// so what is added is the TRANSVERSE remainder of the independence form
    /// `R P_a R' + J Sigma J'`. Dropped because (i) under the error channel the
    /// pose-covariance audit measures -- a baseline-scale error common to landmark and
    /// translation, p_n = (1+s)(R p_a + t) -- the exact moved covariance is
    /// sigma_s^2 p_n p_n', purely radial, so the transverse block is an artefact of
    /// assuming the landmark and pose errors independent, not a second copy of
    /// anything; and (ii) the next bearing update is taken from the new anchor, where
    /// the chart's transverse coordinates are observed directly at pixel level, so the
    /// block is contracted away by that one update whatever the channel, leaving only
    /// leakage into the range chart through the chart Jacobian and the gate. Measured
    /// on 30 Kite_training trajectories at the shipped re-anchor rule: adding it costs
    /// range accuracy (6.6 vs 5.7 %), one honest trajectory (26 vs 27/30) and inflates
    /// reported sigma_r (5.15 vs 4.40 m).
    pub reanchor_transport_cov: bool,
    /// Consider (Schmidt) mode, filter_formulation §V-F: per live anchor the feed also
    /// hands the depth filter the FULL anchor->current relative covariance and the
    /// one-step regression operator `M_k` of that relative-pose error, and re-anchors
    /// onto the CURRENT clone with the full covariance, so the landmark's tracked
    /// landmark–pose cross-covariance is exactly the term the transfer needs. Must
    /// match `SparseVogSettings::consider_relpose` on the filter being fed.
    pub consider: bool,
    /// DIAGNOSTIC for consider mode: 0 = the VIO's regression operator (production),
    /// 1 = identity (pure persistence of the cross term), 2 = zero (no persistence).
    /// Brackets how much of the consider result depends on `M_k` being right.
    pub consider_m_mode: u8,
    /// DIAGNOSTIC for consider mode: 0 = the VIO's full 6x6 anchor->current
    /// covariance (production); 1 = its translation block replaced by `tr/3 * I` and
    /// the rotation–translation cross blocks zeroed; 2 = both diagonal blocks
    /// isotropised, cross blocks zeroed; 3 = translation block only (rotation and cross
    /// blocks zeroed). Tests whether the legacy path's honesty on
    /// long tracks comes from `tr(Sigma_v)/3` smearing an over-conservative
    /// perpendicular channel into the range term: if the consider recursion fed the
    /// isotropised covariance recovers legacy's long-track coverage, the remaining
    /// deficit is the VIO's along-baseline covariance, not the depth filter.
    pub consider_sigma_mode: u8,
}

/// Apply [`CloneRelFeed::consider_sigma_mode`] to a `[rot(0:3), trans(3:6)]` 6x6.
fn isotropise_rel_cov(sigma: &nalgebra::Matrix6<f64>, mode: u8) -> nalgebra::Matrix6<f64> {
    if mode == 0 {
        return *sigma;
    }
    let mut out = nalgebra::Matrix6::<f64>::zeros();
    let ww = sigma.fixed_view::<3, 3>(0, 0).into_owned();
    let vv = sigma.fixed_view::<3, 3>(3, 3).into_owned();
    let iso = |b: &Matrix3<f64>| Matrix3::<f64>::identity() * (b.trace() / 3.0);
    match mode {
        1 => {
            out.fixed_view_mut::<3, 3>(0, 0).copy_from(&ww);
            out.fixed_view_mut::<3, 3>(3, 3).copy_from(&iso(&vv));
        }
        2 => {
            out.fixed_view_mut::<3, 3>(0, 0).copy_from(&iso(&ww));
            out.fixed_view_mut::<3, 3>(3, 3).copy_from(&iso(&vv));
        }
        _ => {
            out.fixed_view_mut::<3, 3>(3, 3).copy_from(&vv);
        }
    }
    out
}

impl Default for CloneRelFeed {
    fn default() -> Self {
        Self {
            prev: HashMap::new(),
            poses: HashMap::new(),
            reanchor_age: None,
            reanchor_transport_cov: false,
            consider: false,
            consider_m_mode: 0,
            consider_sigma_mode: 0,
        }
    }
}

/// What one [`CloneRelFeed::feed`] did, so a wrapper can log it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedReport {
    /// `true` when the clone-relative path ran; `false` for the legacy absolute path.
    pub clone_relative: bool,
    /// Landmarks moved to a new anchor this frame.
    pub reanchored: usize,
    /// Clones for which a relative covariance was supplied.
    pub n_rel: usize,
    /// Whether an absolute pair was available at all.
    pub had_absolute: bool,
    /// Anchors given a consider (full covariance + regression operator) entry.
    pub n_consider: usize,
}

impl CloneRelFeed {
    pub fn feed(
        &mut self,
        cov: Option<&dyn CameraPoseCovariance>,
        sparse: &mut crate::depth::sparse_3d::Sparse3DFilter,
        measurement: &VisionMeasurement,
        t_wc: &nalgebra::Matrix4<f64>,
    ) -> FeedReport {
        let mut report = FeedReport::default();
        let absolute = cov.and_then(|c| c.camera_pose_covariances());
        report.had_absolute = absolute.is_some();
        let (p_vv, p_ww) = match absolute.as_ref() {
            Some((v, w)) => (Some(v), Some(w)),
            None => (None, None),
        };
        let Some(c) = cov.filter(|c| c.current_clone_id().is_some()) else {
            sparse.update(measurement, t_wc, p_vv, p_ww);
            return report;
        };
        report.clone_relative = true;

        let live = c.live_clone_ids();
        self.poses.retain(|id, _| live.contains(id));
        // Re-anchor BEFORE the update: landmarks on the oldest live clone move onto
        // the newest clone this feed has REGISTERED (the previous frame's), with the
        // oldest->current relative covariance. Both clones are live, so the covariance
        // exists; the current clone is not registered yet, so it is never the target.
        let registered: Vec<u64> = live
            .iter()
            .copied()
            .filter(|id| self.poses.contains_key(id))
            .collect();
        if registered.len() >= 2 {
            // Consider mode moves onto the CURRENT clone (its pose is this frame's
            // `t_wc`), the harness rule onto the previous frame's registered clone.
            let (new, t_new) = match (self.consider, c.current_clone_id()) {
                (true, Some(cur)) => (cur, *t_wc),
                _ => {
                    let n = registered[registered.len() - 1];
                    (n, self.poses[&n])
                }
            };
            let olds: Vec<u64> = match (self.reanchor_age, c.current_clone_id()) {
                (Some(age), Some(cur)) => registered[..registered.len() - 1]
                    .iter()
                    .copied()
                    .filter(|id| cur.saturating_sub(*id) >= age)
                    .collect(),
                _ => vec![registered[0]],
            };
            for old in olds {
                if old == new {
                    continue;
                }
                let cov = if self.reanchor_transport_cov {
                    c.relative_pose_covariances(old, &SE3::identity())
                } else {
                    None
                };
                let full = if self.consider {
                    c.relative_pose_covariance_full(old)
                        .map(|s| isotropise_rel_cov(&s, self.consider_sigma_mode))
                } else {
                    None
                };
                report.reanchored += sparse.reanchor_clone_full(
                    old,
                    &t_new,
                    Some(new),
                    cov.as_ref().map(|(v, w)| (v, w)),
                    full.as_ref(),
                );
            }
        }
        let prev_frame_clone = c
            .current_clone_id()
            .and_then(|cur| live.iter().rev().find(|&&id| id != cur).copied());
        if let Some(cur) = c.current_clone_id() {
            self.poses.insert(cur, *t_wc);
        }
        if self.consider {
            // Per live anchor: the full anchor->current covariance and the regression
            // operator M = Cov(a->cur, a->prev) Cov(a->prev)^-1 (zero when the previous
            // relative covariance is singular, i.e. the anchor IS the previous clone;
            // a landmark there carries no cross term yet, so M is never used).
            let mut map: crate::depth::sparse_3d::ConsiderRelMap = HashMap::new();
            for &id in &c.live_clone_ids() {
                let sigma = if Some(id) == c.current_clone_id() {
                    Some(nalgebra::Matrix6::<f64>::zeros())
                } else {
                    c.relative_pose_covariance_full(id)
                };
                let Some(sigma) = sigma else { continue };
                let sigma = isotropise_rel_cov(&sigma, self.consider_sigma_mode);
                let m = prev_frame_clone
                    .filter(|&p| p != id)
                    .and_then(|p| c.relative_pose_joint_covariance(id, p))
                    .and_then(|j| {
                        let s_prev = j.fixed_view::<6, 6>(0, 0).into_owned();
                        let cross = j.fixed_view::<6, 6>(6, 0).into_owned();
                        s_prev.try_inverse().map(|inv| cross * inv)
                    })
                    .filter(|m| m.iter().all(|v| v.is_finite()))
                    .unwrap_or_else(nalgebra::Matrix6::<f64>::zeros);
                let m = match self.consider_m_mode {
                    1 => nalgebra::Matrix6::<f64>::identity(),
                    2 => nalgebra::Matrix6::<f64>::zeros(),
                    _ => m,
                };
                map.insert(id, crate::depth::sparse_3d::ConsiderRel { sigma, m });
            }
            report.n_consider = map.len();
            sparse.set_consider_relpose(map);
        }

        self.prev.retain(|id, _| live.contains(id));
        let mut rel: crate::depth::sparse_3d::CloneRelCov = HashMap::new();
        for id in live {
            let Some((rv, rw)) = c.relative_pose_covariances(id, &SE3::identity()) else {
                continue;
            };
            let (dv, dw) = match self.prev.get(&id) {
                None => (Matrix3::zeros(), Matrix3::zeros()),
                Some((pv, pw)) => (psd_clip(&(rv - pv)), psd_clip(&(rw - pw))),
            };
            self.prev.insert(id, (rv, rw));
            rel.insert(id, (rv, rw, dv, dw));
        }
        report.n_rel = rel.len();
        sparse.update_with_clones(
            measurement,
            t_wc,
            p_vv,
            p_ww,
            c.current_clone_id(),
            Some(&rel),
        );
        report
    }
}

/// Which backend [`backend_from_config`] built. Logged by every wrapper so a run's
/// provenance says which filter produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Eqvio,
    Msceqf,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Eqvio => "EqVIO",
            BackendKind::Msceqf => "MSCEqF",
        }
    }
}

/// THE one place a backend is chosen and built from configuration.
///
/// An `MSCEqF:` section with `enabled: true` selects MSCEqF; anything else is EqVIO.
/// The CLI, the ROS 2 node and the Python binding all call this, so there is exactly
/// one reading of the config and one construction path -- the ROS 2 node used to carry
/// this logic privately, which left the CLI unable to select MSCEqF at all and the
/// Python binding re-typing twenty constructor arguments by hand.
///
/// `start` is the output of [`crate::initialization::StationaryInit`]: both backends
/// seed from its gravity-aligned state. `fx` converts the MSCEqF section's PIXEL noise
/// into the normalized units that backend works in.
pub fn backend_from_config(
    msceqf: Option<&crate::config::MsceqfConfig>,
    zupt: Option<&crate::config::ZuptConfig>,
    eqvio: &crate::VIOFilterSettings,
    cam: Arc<dyn CameraModel>,
    fx: f64,
    start: &crate::initialization::FilterStart,
) -> (Box<dyn VioBackendFull>, BackendKind) {
    match msceqf.filter(|c| c.enabled) {
        Some(cfg) => {
            let mut b = MsceqfBackend::from_config(cfg, cam, fx, start);
            b.set_zupt(ZuptPolicy::from_config(zupt));
            (Box::new(b), BackendKind::Msceqf)
        }
        None => {
            let mut b = EqvioBackend::new(
                crate::VIOFilter::new(eqvio.clone(), start.state.clone()),
                cam,
            );
            b.set_zupt(ZuptPolicy::from_config(zupt));
            (Box::new(b), BackendKind::Eqvio)
        }
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
    zupt: Option<ZuptPolicy>,
}

impl EqvioBackend {
    pub fn new(filter: crate::VIOFilter, cam: Arc<dyn CameraModel>) -> Self {
        Self {
            filter,
            cam,
            zupt: None,
        }
    }

    /// Install (or remove) the zero-velocity policy. See [`ZuptPolicy`].
    pub fn set_zupt(&mut self, zupt: Option<ZuptPolicy>) {
        self.zupt = zupt;
    }

    /// Samples on which a ZUPT was accepted; 0 when no policy is installed.
    pub fn zupt_fired(&self) -> u64 {
        self.zupt.as_ref().map_or(0, |z| z.fired())
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
        self.filter.process_imu(imu.clone());
        // Propagate first, then assert rest: the detector reads the raw sample, the
        // update acts on the state that sample produced.
        if let Some(sigma) = self.zupt.as_mut().and_then(|z| z.observe(&imu)) {
            if self.zero_velocity_update(sigma) {
                if let Some(z) = self.zupt.as_mut() {
                    z.fired += 1;
                }
            }
        }
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn zupt_fired(&self) -> u64 {
        EqvioBackend::zupt_fired(self)
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
    fn relative_pose_covariances(
        &self,
        clone_id: u64,
        t_wc_clone: &SE3,
    ) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        self.filter
            .sparse_relative_pose_covariances(clone_id, t_wc_clone)
    }
    fn live_clone_ids(&self) -> Vec<u64> {
        self.filter.clone_ids()
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
    /// Use a DIRECT-PRODUCT (static) bias chart instead of the semi-direct one.
    /// See `MSCEqFFilter::new_direct_product_bias`.
    pub direct_product_bias: bool,
    /// Birth in-state landmarks from a DEPTH PRIOR instead of a triangulation, when
    /// positive: the prior depth in metres, with `civera_birth_rel_std` the inverse-depth
    /// relative std. Zero keeps the triangulated birth.
    ///
    /// Triangulated birth can only admit tracks whose geometry triangulates, and the
    /// condition-number gate needs ~0.57 deg of bearing spread, so on an outdoor ride
    /// with a 0.19 s clone window that is the NEAR scene only -- measured in-state range
    /// 0.56-12.9 m, median 3.8 m. Which part of the scene is allowed to constrain the
    /// filter is then a selection effect rather than a choice. EqVIO, on the same bag and
    /// the same tracks, admits every track against a fixed `sceneDepth` prior and holds
    /// scale to 1.04x where this backend reaches 0.90x, which is what makes the admission
    /// rule worth testing rather than assuming.
    pub civera_birth_depth: f64,
    /// Inverse-depth relative std for `civera_birth_depth`. The band is
    /// `[d0*exp(-rel), d0*exp(+rel)]`, so 1.0 spans 0.37x to 2.7x the prior. Wide priors
    /// are known to be catastrophic in this chart, so this is not a free parameter.
    pub civera_birth_rel_std: f64,

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
            direct_product_bias: false,
            civera_birth_depth: 0.0,
            civera_birth_rel_std: 1.0,
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
    /// The most recent IMU reading, held so a frame can be propagated to its OWN
    /// stamp before cloning. See [`Self::propagate_to`].
    last_imu: Option<crate::mathematical::msceqf_filter::Imu>,
    /// Propagate to the frame stamp before cloning instead of cloning whatever the
    /// last IMU sample left behind. See [`Self::propagate_to`].
    clone_at_frame_stamp: bool,
    zupt: Option<ZuptPolicy>,
    /// See [`CameraPoseCovariance::last_clone_eviction`]. Reset every frame step.
    last_eviction: Option<CloneEviction>,
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
        use nalgebra::SMatrix;

        let sensor = &start.state.sensor;
        // The origin bias is whatever StationaryInit seeded -- the gyro bias measured
        // over a stationary start, zero otherwise. A first version passed zeros here,
        // which silently discarded that seed on this path alone: the EqVIO backend takes
        // it through `start.state`, and the bikevid harness seeded it by hand (762 m of
        // IMU-only drift -> 37 m on that rig, whose gyro bias is 2.5 deg/s), so the
        // ROS 2 node -- the one consumer of this constructor -- was the only caller
        // running MSCEqF without it.
        // `VIOSensorState.velocity` is the SE(3) BODY velocity; `SE23` carries WORLD
        // velocity. A first version passed it through unrotated, which was invisible at
        // the rest starts every wrapper uses (v = 0) and wrong for any seeded moving
        // start, e.g. the depth-NEES harness's GT seed.
        let v_world = sensor.pose.rotation.act(&sensor.velocity);
        let origin = SystemOrigin::new(
            echo_lie::SE23::new(sensor.pose.rotation, sensor.pose.translation, v_world),
            sensor.input_bias,
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
        let mk = if cfg.direct_product_bias {
            MSCEqFFilter::new_direct_product_bias
        } else {
            MSCEqFFilter::new
        };
        let filter = mk(
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
            direct_product_bias: cfg.direct_product_bias,
            civera_birth_depth: cfg.civera_birth_depth,
            civera_birth_rel_std: cfg.civera_birth_rel_std,
        };
        let mut b = Self::new(filter, cam, settings);
        b.set_clone_at_frame_stamp(cfg.clone_at_frame_stamp);
        b
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
            last_imu: None,
            clone_at_frame_stamp: false,
            zupt: None,
            last_eviction: None,
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

    /// Feature ids the backend currently holds: buffered structureless tracks plus
    /// in-state landmarks, ascending. What a front-end budget should prefer to keep,
    /// the way EqVIO's `landmarks_world()` keys are -- `landmarks_world()` here lists
    /// only the in-state ones, which is empty in structureless mode.
    pub fn tracked_ids(&self) -> Vec<u64> {
        self.buf.keys().chain(self.slam.iter()).copied().collect()
    }

    /// Cumulative per-reason drop counts from the structureless update:
    /// `[too_few_obs, stale_clone, triangulation_failed, jacobian_failed, chi2_rejected]`.
    pub fn msc_reject_counts(&self) -> [u64; 5] {
        self.filter.msc_reject
    }

    /// Take the recorded per-update `(|dv|, sigma_v, mahalanobis)` rows, emptying the
    /// buffer. See `MSCEqFFilter::msc_vel_corr`.
    pub fn take_msc_vel_corr(&mut self) -> Vec<[f64; 3]> {
        std::mem::take(&mut self.filter.msc_vel_corr)
    }

    /// Take the per-update correction-split rows, emptying the buffer. See
    /// `MSCEqFFilter::msc_corr_split`.
    pub fn take_msc_corr_split(&mut self) -> Vec<[f64; 12]> {
        std::mem::take(&mut self.filter.msc_corr_split)
    }

    /// Take the recorded per-track `chi2 / gate` ratios, emptying the buffer.
    pub fn take_msc_chi2_ratios(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.filter.msc_chi2_ratio)
    }

    /// Install (or remove) the zero-velocity policy. See [`ZuptPolicy`].
    pub fn set_zupt(&mut self, zupt: Option<ZuptPolicy>) {
        self.zupt = zupt;
    }

    /// Samples on which a ZUPT was accepted; 0 when no policy is installed.
    pub fn zupt_fired(&self) -> u64 {
        self.zupt.as_ref().map_or(0, |z| z.fired())
    }

    /// Clone at the frame's own stamp rather than at the last IMU sample. Off by
    /// default so existing traces stay comparable. See [`Self::propagate_to`].
    pub fn set_clone_at_frame_stamp(&mut self, on: bool) {
        self.clone_at_frame_stamp = on;
    }

    /// Advance the filter from the last IMU sample to `stamp`, holding the last IMU
    /// reading constant over the sub-sample interval.
    ///
    /// The interval is at most one IMU period (8.2 ms on this rig), so the zero-order
    /// hold costs `|a| dt^2 / 2` -- order 3e-5 m at 1 m/s^2 -- while the lag it removes
    /// is `|v| dt`, order 1e-2 m at riding speed. Linear interpolation between the
    /// bracketing samples would be better still, but a streaming backend has not seen
    /// the next sample yet, and would have to hold the frame back to get it.
    ///
    /// One inconsistency is deliberate: [`Self::process_imu`] integrates each sample's
    /// reading BACKWARDS over the interval ending at its own stamp, whereas this holds
    /// the PREVIOUS reading forwards, because the next sample has not arrived. The two
    /// differ by `jerk * dt^2`, below the term this removes by several orders.
    fn propagate_to(&mut self, stamp: f64) {
        if !self.clone_at_frame_stamp {
            return;
        }
        let (Some(prev), Some(u)) = (self.last_stamp, self.last_imu) else {
            return;
        };
        let dt = stamp - prev;
        if dt > 0.0 && dt < 0.1 {
            self.filter.propagate_step(&u, dt);
            self.last_stamp = Some(stamp);
        }
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
        self.last_imu = Some(crate::mathematical::msceqf_filter::Imu {
            ang: imu.gyr,
            acc: imu.acc,
        });
        // Propagate first, then assert rest (see EqvioBackend::process_imu).
        if let Some(sigma) = self.zupt.as_mut().and_then(|z| z.observe(&imu)) {
            if self.zero_velocity_update(sigma) {
                if let Some(z) = self.zupt.as_mut() {
                    z.fired += 1;
                }
            }
        }
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
        self.last_eviction = None;

        // (0) Bring the state to THIS FRAME'S stamp before cloning. Without it the
        // clone holds wherever the last IMU sample left the state, which on a
        // 122 Hz IMU / 62 Hz camera rig is a median 3.7 ms behind the image -- a
        // SPEED-PROPORTIONAL position lag (1.8 cm at 4.7 m/s, against a 7.5 cm
        // frame-to-frame baseline). Because the lag resets with the IMU/frame phase it
        // is per-clone jitter, not a common offset, so it corrupts the very relative
        // geometry a structureless multi-view update is made of, and nothing in the
        // clone covariance accounts for it.
        self.propagate_to(measurement.stamp);

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
                let born = if self.settings.civera_birth_depth > 0.0 {
                    self.filter.birth_landmark_civera(
                        &track,
                        self.settings.pixel_std,
                        self.settings.civera_birth_depth,
                        self.settings.civera_birth_rel_std,
                    )
                } else {
                    self.filter.birth_landmark(
                        &track,
                        &self.z1,
                        self.settings.pixel_std,
                        self.settings.slam_chi2(),
                        None,
                    )
                };
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
            // Evidence for re-anchoring, taken while the old->new relative covariance
            // still exists: a depth filter anchored on this clone has to be moved onto
            // the newest one, and it can only do that honestly with this covariance.
            let newest_pos = self.clone_ids.len() - 1;
            if newest_pos > 0 {
                if let Some((p_vv, p_ww)) = self.filter.relative_pose_covariance(0, newest_pos) {
                    self.last_eviction = Some(CloneEviction {
                        old_clone_id: old,
                        new_clone_id: self.clone_ids[newest_pos],
                        t_wc_new: self.filter.clone_world_pose(newest_pos),
                        p_vv,
                        p_ww,
                    });
                }
            }
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
impl CameraPoseCovariance for MsceqfBackend {
    /// The newest clone's own 6x6 marginal. In an MSCKF the current camera pose IS a
    /// clone -- `phi()`'s camera pose is invariant to the navigation direction, so there
    /// is no core-state Jacobian route -- and this is exactly what the Sparse3D depth
    /// NEES harness fed in place of EqVIO's absolute pair when it calibrated 10/30
    /// Kite trajectories. Chart: the clone's left (global) SE(3) tangent,
    /// `[rot(0:3), trans(3:6)]`, returned as `(P_vv, P_ww)`.
    fn camera_pose_covariances(&self) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        let pos = self.clone_ids.len().checked_sub(1)?;
        let ci = self.filter.clone_idx(pos);
        if ci + 6 > self.filter.dim() {
            return None;
        }
        let blk = |r0: usize| {
            self.filter
                .cov
                .fixed_view::<3, 3>(ci + r0, ci + r0)
                .into_owned()
        };
        let (p_ww, p_vv) = (blk(0), blk(3));
        (p_vv.iter().chain(p_ww.iter()).all(|v| v.is_finite())).then_some((p_vv, p_ww))
    }
    fn relative_pose_covariances(
        &self,
        clone_id: u64,
        _t_wc_clone: &SE3,
    ) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
        let apos = self.clone_ids.iter().position(|&c| c == clone_id)?;
        let cpos = self.clone_ids.len().checked_sub(1)?;
        self.filter.relative_pose_covariance(apos, cpos)
    }
    fn relative_pose_covariance_full(&self, clone_id: u64) -> Option<nalgebra::Matrix6<f64>> {
        let apos = self.clone_ids.iter().position(|&c| c == clone_id)?;
        let cpos = self.clone_ids.len().checked_sub(1)?;
        self.filter.relative_pose_covariance_full(apos, cpos)
    }
    fn relative_pose_joint_covariance(
        &self,
        clone_id: u64,
        prev_clone_id: u64,
    ) -> Option<nalgebra::SMatrix<f64, 12, 12>> {
        let apos = self.clone_ids.iter().position(|&c| c == clone_id)?;
        let ppos = self.clone_ids.iter().position(|&c| c == prev_clone_id)?;
        let cpos = self.clone_ids.len().checked_sub(1)?;
        self.filter.relative_pose_joint_covariance(apos, ppos, cpos)
    }
    fn live_clone_ids(&self) -> Vec<u64> {
        self.clone_ids.clone()
    }
    fn current_clone_id(&self) -> Option<u64> {
        self.clone_ids.last().copied()
    }
    fn last_clone_eviction(&self) -> Option<&CloneEviction> {
        self.last_eviction.as_ref()
    }
}

impl VioBackendFull for MsceqfBackend {
    fn pose_covariance(&self) -> Option<&dyn CameraPoseCovariance> {
        Some(self)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn zupt_fired(&self) -> u64 {
        MsceqfBackend::zupt_fired(self)
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

    /// `backend_from_config` is the only chooser, so it must agree with direct construction
    /// on BOTH branches, and the branch must actually follow the section: an EqVIO config
    /// yields an EqVIO fingerprint, an enabled `MSCEqF:` section yields MSCEqF's.
    #[test]
    fn backend_from_config_selects_by_section_and_matches_direct_construction() {
        use crate::config::{MsceqfConfig, MsceqfInitialStd, MsceqfProcessNoise};
        use crate::initialization::StationaryInit;
        let samples: Vec<IMUVelocity> = (0..10)
            .map(|i| {
                IMUVelocity::new(
                    i as f64 * 0.01,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, 9.81),
                )
            })
            .collect();
        let start = StationaryInit::from_samples(10, SE3::identity(), samples);
        let eqvio = crate::VIOFilterSettings::default();
        let fx = 400.0;

        // No section: EqVIO, identical to building it by hand.
        let (built, kind) = backend_from_config(None, None, &eqvio, test_cam(), fx, &start);
        assert_eq!(kind, BackendKind::Eqvio);
        let direct = EqvioBackend::new(
            crate::VIOFilter::new(eqvio.clone(), start.state.clone()),
            test_cam(),
        );
        assert_eq!(built.state_fingerprint(), direct.state_fingerprint());
        assert!(built.as_any().downcast_ref::<EqvioBackend>().is_some());
        assert!(built.as_any().downcast_ref::<MsceqfBackend>().is_none());

        // Enabled section: MSCEqF, identical to from_config.
        let cfg = MsceqfConfig {
            enabled: true,
            num_clones: 7,
            pixel_std: 1.5,
            chi2_mult: 1.3,
            slam_chi2_mult: 0.0,
            max_track: 123,
            slam_min_len: 0,
            max_slam: 4,
            curvature: true,
            update_span: 0,
            max_promote_range: 0.0,
            direct_product_bias: false,
            clone_at_frame_stamp: false,
            civera_birth_depth: 0.0,
            civera_birth_rel_std: 1.0,
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
        let (built, kind) = backend_from_config(Some(&cfg), None, &eqvio, test_cam(), fx, &start);
        assert_eq!(kind, BackendKind::Msceqf);
        let direct = MsceqfBackend::from_config(&cfg, test_cam(), fx, &start);
        assert_eq!(built.state_fingerprint(), direct.state_fingerprint());
        assert!(built.as_any().downcast_ref::<MsceqfBackend>().is_some());

        // A DISABLED section must not select MSCEqF.
        let off = MsceqfConfig {
            enabled: false,
            ..cfg
        };
        let (_, kind) = backend_from_config(Some(&off), None, &eqvio, test_cam(), fx, &start);
        assert_eq!(kind, BackendKind::Eqvio);
    }

    /// A backend at rest with no vision must HOLD velocity when the ZUPT policy is
    /// installed, and the policy must actually fire. The control arm has to drift, or
    /// the assertion is vacuous: a filter that never moves passes either way.
    #[test]
    fn zupt_policy_holds_velocity_at_rest_and_fires() {
        use crate::config::ZuptConfig;
        let cfg = ZuptConfig {
            enabled: true,
            ..ZuptConfig::default()
        };
        // A TRUE rest start (zero velocity) with an accel-magnitude error of 0.5 m/s^2
        // -- the error that dead-reckons into velocity on a real rig -- and gyro at zero.
        // The start must be at rest: a ZUPT is a chi2-gated update, so it holds rest, it
        // does not rescue a large pre-existing velocity error (a first version of this
        // test started at 2 m/s and the gate correctly refused every update).
        let run = |with_zupt: bool| -> (f64, u64) {
            let origin = SystemOrigin::new(
                SE23::new(SO3::identity(), Vector3::zeros(), Vector3::zeros()),
                Vector6::zeros(),
                SE3::identity(),
                Vector3::new(0.0, 0.0, -9.81),
            );
            // Everything at the RIG operating point, where the policy was validated
            // (bikevid: 764 m of rest drift -> 0.0 m): initial stds attitude 1e-2,
            // velocity 1e-1, position 1e-4, bias 1e-2; accel bias random walk 4.1e-3;
            // and the rig's own 0.14 m/s^2 gravity-magnitude error. Two earlier
            // versions of this test taught why the operating point matters. With
            // velocity std 1e-2 the first ZUPT opportunity (sample 25) is already 6
            // sigma and the gate refuses every update forever. With the bias random
            // walk frozen at 1e-5 against a 0.5 m/s^2 error, the gate accepted 23
            // updates then stalled: each accepted ZUPT shrinks P_vv, the gain falls, the
            // unmodelled acceleration re-accumulates faster than the update removes it,
            // and the innovation walks out of the gate. A ZUPT holds rest only while the
            // bias state can absorb the acceleration it keeps seeing.
            let d_std = [1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4];
            let filter = MSCEqFFilter::new(
                origin,
                &SMatrix::<f64, 9, 9>::from_fn(
                    |i, j| if i == j { d_std[i] * d_std[i] } else { 0.0 },
                ),
                &(SMatrix::<f64, 6, 6>::identity() * 1e-4),
                &(SMatrix::<f64, 6, 6>::identity() * 1e-6),
                ProcessNoise {
                    angular_velocity_std: 1.7e-4,
                    acceleration_std: 4.2e-3,
                    angular_velocity_bias_std: 1.2e-4,
                    acceleration_bias_std: 4.1e-3,
                    state_transition_order: 0,
                },
            );
            let mut b = MsceqfBackend::new(filter, test_cam(), MsceqfSettings::default());
            if with_zupt {
                b.set_zupt(ZuptPolicy::from_config(Some(&cfg)));
            }
            for k in 0..1200u64 {
                let t = k as f64 * (1.0 / 122.0);
                b.process_imu(IMUVelocity::new(
                    t,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, 9.81 + 0.14),
                ));
            }
            (b.nav_estimate().velocity.norm(), b.zupt_fired())
        };
        let (v_off, fired_off) = run(false);
        let (v_on, fired_on) = run(true);
        assert_eq!(fired_off, 0);
        assert!(
            fired_on > 100,
            "policy should fire on most rest samples, fired {fired_on}"
        );
        assert!(
            v_on < 0.05,
            "with ZUPT the velocity must be held near zero, got {v_on:.4} m/s"
        );
        // 0.14 m/s^2 over ~9.8 s dead-reckons to ~1.4 m/s without a ZUPT.
        assert!(
            v_off > 0.5,
            "control arm must drift or the test proves nothing, got {v_off:.4} m/s"
        );
    }

    /// The absolute camera-pose covariance IS the newest clone's block, the relative
    /// covariance of the current clone to itself is zero, and an older anchor's is not
    /// -- the last being the non-vacuity control.
    #[test]
    fn msceqf_pose_covariances_are_the_clone_window_relations() {
        let mut b = make_backend(MsceqfSettings {
            num_clones: 6,
            ..Default::default()
        });
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..12u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
        }
        let (p_vv, p_ww) = b.camera_pose_covariances().expect("a clone exists");
        let cur = b.current_clone_id().expect("current clone");
        let ci = b.filter.clone_idx(b.clone_ids.len() - 1);
        assert_eq!(p_ww, b.filter.cov.fixed_view::<3, 3>(ci, ci).into_owned());
        assert_eq!(
            p_vv,
            b.filter.cov.fixed_view::<3, 3>(ci + 3, ci + 3).into_owned()
        );
        assert!(
            p_vv.trace() > 0.0 && p_ww.trace() > 0.0,
            "block must carry variance"
        );
        let (rv, rw) = b.relative_pose_covariances(cur, &SE3::identity()).unwrap();
        assert!(
            rv.amax() < 1e-12 && rw.amax() < 1e-12,
            "self-relative must vanish"
        );
        let oldest = b.live_clone_ids()[0];
        let (ov, ow) = b
            .relative_pose_covariances(oldest, &SE3::identity())
            .unwrap();
        assert!(
            ov.trace() > 1e-12 || ow.trace() > 1e-12,
            "anchor-relative must not vanish"
        );
        assert_eq!(b.live_clone_ids().len(), 6);
    }

    /// The eviction evidence is captured BEFORE the clone dies: the old id is gone from
    /// the live set afterwards, the new id is the current clone, and the covariance is
    /// finite. Without it a depth filter re-anchors blind.
    #[test]
    fn msceqf_eviction_is_recorded_before_the_clone_dies() {
        // A NON-TRIVIAL origin, or the world-pose check below is vacuous: with T0 = I
        // and S0 = I the clone's group element and its world pose coincide, which is
        // exactly the special case under which the bug this guards was invisible.
        let mut b = make_backend_with_origin(
            MsceqfSettings {
                num_clones: 4,
                ..Default::default()
            },
            Vector3::new(3.0, -2.0, 1.0),
            SE3::new(
                SO3::exp(&Vector3::new(0.0, 0.0, 0.3)),
                Vector3::new(0.1, 0.0, 0.0),
            ),
        );
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..3u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
            assert!(b.last_clone_eviction().is_none(), "window not full yet");
        }
        for k in 3..8u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
            if b.clone_ids.len() == 4 && k >= 4 {
                let ev = b
                    .last_clone_eviction()
                    .expect("an eviction every full frame");
                assert!(!b.live_clone_ids().contains(&ev.old_clone_id));
                assert_eq!(Some(ev.new_clone_id), b.current_clone_id());
                assert!(ev.p_vv.iter().chain(ev.p_ww.iter()).all(|v| v.is_finite()));
                // The newest clone IS the current camera pose, in the WORLD frame. The
                // first version handed out the clone's group element `E_k` instead,
                // which drifted from the camera pose by the whole trajectory (28 m by
                // frame 39 on Kite t0) and sent every re-anchored landmark somewhere
                // else; nothing here caught it because nothing compared the two.
                let cam = b.camera_pose().as_matrix();
                let err = (ev.t_wc_new.as_matrix() - cam).abs().max();
                assert!(
                    err < 1e-9,
                    "eviction pose is not the world camera pose: {err:.3e}"
                );
            }
        }
    }

    /// The joint 12x12 of two relative poses must agree block-wise with the two 6x6
    /// relative covariances it generalizes, and its cross block must be the diagonal
    /// block when prev == current (Cov(x, x) = Var(x)).
    #[test]
    fn relative_pose_joint_covariance_is_consistent_with_the_pairwise_one() {
        let mut b = make_backend(MsceqfSettings {
            num_clones: 6,
            ..Default::default()
        });
        let ids: Vec<u64> = (0..20).collect();
        for k in 0..6u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            b.process_vision(frame_at(t, &ids), &HashMap::new());
        }
        let live = b.live_clone_ids();
        assert!(live.len() >= 3);
        let (a, p, c) = (live[0], live[live.len() - 2], live[live.len() - 1]);
        let j = b.relative_pose_joint_covariance(a, p).expect("joint");
        let ap = b
            .filter
            .relative_pose_covariance_full(0, live.len() - 2)
            .expect("a->p");
        let ac = b.relative_pose_covariance_full(a).expect("a->c");
        let err_pp = (j.fixed_view::<6, 6>(0, 0) - ap).abs().max();
        let err_cc = (j.fixed_view::<6, 6>(6, 6) - ac).abs().max();
        assert!(
            err_pp < 1e-12 && err_cc < 1e-12,
            "diagonal blocks differ: {err_pp:.2e} {err_cc:.2e}"
        );
        let jcc = b
            .relative_pose_joint_covariance(a, c)
            .expect("joint with prev == cur");
        let err_x = (jcc.fixed_view::<6, 6>(6, 0) - ac).abs().max();
        assert!(
            err_x < 1e-12,
            "cross block with prev == cur must be the variance: {err_x:.2e}"
        );
        // Non-trivial cross block for a genuinely previous clone.
        assert!(j.fixed_view::<6, 6>(6, 0).abs().max() > 0.0);
    }

    /// Consider mode of the feed: every live anchor gets a full covariance and a finite
    /// regression operator; the current clone's covariance is zero.
    #[test]
    fn clone_rel_feed_consider_mode_serves_every_live_anchor() {
        use crate::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
        use crate::depth::sparse_gb::SparseVogSettings;
        let k = Matrix3::new(400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0);
        let mut sparse = Sparse3DFilter::new(
            k,
            Sparse3DChart::BearingInvDepthAdditive,
            SparseVogSettings {
                min_track_length: 1,
                birth_min_flow_px: 0.0,
                consider_relpose: true,
                ..Default::default()
            },
        )
        .with_camera(test_cam());
        let mut b = make_backend(MsceqfSettings {
            num_clones: 5,
            ..Default::default()
        });
        let mut feed = CloneRelFeed {
            consider: true,
            ..Default::default()
        };
        let ids: Vec<u64> = (0..20).collect();
        let mut any_cross = false;
        for k in 0..10u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            let frame = frame_at(t, &ids);
            b.process_vision(frame.clone(), &HashMap::new());
            let t_wc = b.nav_estimate().pose.as_matrix();
            let r = feed.feed(b.pose_covariance(), &mut sparse, &frame, &t_wc);
            assert!(r.clone_relative);
            assert_eq!(
                r.n_consider,
                b.live_clone_ids().len(),
                "one consider entry per live clone"
            );
            for f in sparse.features_iter() {
                any_cross |= f.p_stheta.iter().any(|v| v.abs() > 0.0);
                assert!(f.p_stheta.iter().all(|v| v.is_finite()));
                assert!(f.covariance.iter().all(|v| v.is_finite()));
            }
        }
        assert!(sparse.feature_count() > 0);
        assert!(any_cross, "no landmark ever acquired a cross term");
    }

    /// `from_config` must hand MSCEqF a WORLD velocity: seed a BODY velocity under a
    /// 90-degree yaw and read the body velocity back through the seam.
    #[test]
    fn from_config_keeps_the_body_velocity_through_a_rotated_seed() {
        use crate::config::{MsceqfConfig, MsceqfInitialStd, MsceqfProcessNoise};
        use crate::initialization::FilterStart;
        use crate::mathematical::vio_state::{VIOSensorState, VIOState};
        let yaw = SO3::exp(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        let sensor = VIOSensorState {
            input_bias: Vector6::zeros(),
            pose: SE3::new(yaw, Vector3::zeros()),
            velocity: Vector3::new(1.0, 0.0, 0.0), // BODY
            camera_offset: SE3::identity(),
        };
        let start = FilterStart {
            state: VIOState::new(sensor, vec![]),
            replay: Vec::new(),
            stationary: false,
            roll_deg: 0.0,
            pitch_deg: 0.0,
            gyro_bias: Vector3::zeros(),
        };
        let cfg = MsceqfConfig {
            enabled: true,
            num_clones: 7,
            pixel_std: 1.0,
            chi2_mult: 1.0,
            slam_chi2_mult: 0.0,
            max_track: 100,
            slam_min_len: 0,
            max_slam: 0,
            curvature: true,
            update_span: 0,
            max_promote_range: 0.0,
            direct_product_bias: false,
            clone_at_frame_stamp: false,
            civera_birth_depth: 0.0,
            civera_birth_rel_std: 1.0,
            gravity: [0.0, 0.0, -9.81],
            initial_std: MsceqfInitialStd {
                attitude: 1e-2,
                velocity: 1e-1,
                position: 1e-4,
                bias: 1e-2,
                extrinsics: 1e-4,
            },
            process_noise: MsceqfProcessNoise {
                accel_density: 1e-3,
                gyro_density: 1e-4,
                accel_random_walk: 1e-5,
                gyro_random_walk: 1e-6,
                transition_order: 0,
            },
        };
        let b = MsceqfBackend::from_config(&cfg, test_cam(), 400.0, &start);
        let v_body = b.nav_estimate().velocity;
        assert!(
            (v_body - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
            "body velocity must round-trip, got {v_body:?}"
        );
    }

    /// The flag must actually move the clone, by the amount the lag implies. A flag
    /// that silently does nothing looks exactly like a mechanism that does not matter.
    #[test]
    fn clone_at_frame_stamp_advances_the_state_by_the_frame_lag() {
        let lag = 0.004; // s -- this rig's median clone lag is 3.7 ms
        let run = |on: bool| {
            let mut b = make_backend(MsceqfSettings::default());
            b.set_clone_at_frame_stamp(on);
            // Two IMU samples establish dt and leave the state at t = 0.008.
            for (i, t) in [0.0_f64, 0.008].iter().enumerate() {
                let _ = i;
                b.process_imu(IMUVelocity::new(
                    *t,
                    Vector3::zeros(),
                    Vector3::new(0.0, 0.0, 9.81),
                ));
            }
            // Frame stamped one lag later than the last IMU sample.
            b.process_vision(
                VisionMeasurement::new(0.008 + lag, Default::default()),
                &HashMap::new(),
            );
            b.camera_pose().translation
        };
        let off = run(false);
        let on = run(true);
        // The origin carries vx = VX, so the lag is worth VX * lag along x.
        let moved = (on - off).norm();
        assert!(
            (moved - VX * lag).abs() < 1e-6,
            "expected the clone to advance by v*lag = {:.6} m, got {moved:.6} m",
            VX * lag
        );
    }

    /// The shared feed must do under MSCEqF what the depth NEES harness did when it
    /// calibrated: clone-relative covariances for every live clone, and landmarks moved
    /// off an evicted anchor before it dies. Under EqVIO it must be the legacy absolute
    /// update, untouched. Each assertion is one thing the feed can silently stop doing.
    #[test]
    fn clone_rel_feed_is_clone_relative_under_msceqf_and_legacy_under_eqvio() {
        use crate::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
        use crate::depth::sparse_gb::SparseVogSettings;
        let k = Matrix3::new(400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0);
        let mut sparse = Sparse3DFilter::new(
            k,
            Sparse3DChart::BearingInvDepthAdditive,
            SparseVogSettings {
                min_track_length: 1,
                birth_min_flow_px: 0.0,
                ..Default::default()
            },
        )
        .with_camera(test_cam());
        let mut b = make_backend(MsceqfSettings {
            num_clones: 4,
            ..Default::default()
        });
        let mut feed = CloneRelFeed::default();
        let ids: Vec<u64> = (0..20).collect();
        let (mut saw_rel, mut reanchored) = (false, 0usize);
        for k in 0..10u64 {
            let t = k as f64 * 0.05;
            b.process_imu(imu(t));
            let frame = frame_at(t, &ids);
            b.process_vision(frame.clone(), &HashMap::new());
            let t_wc = b.nav_estimate().pose.as_matrix();
            let r = feed.feed(b.pose_covariance(), &mut sparse, &frame, &t_wc);
            assert!(
                r.clone_relative,
                "MSCEqF manages clones: clone-relative path"
            );
            assert!(r.had_absolute, "newest clone marginal is always available");
            assert_eq!(
                r.n_rel,
                b.live_clone_ids().len(),
                "one relative covariance per live clone"
            );
            saw_rel |= r.n_rel > 1;
            reanchored += r.reanchored;
            let live = b.live_clone_ids();
            for f in sparse.features_iter() {
                let Some(a) = f.anchor_clone_id else { continue };
                // No landmark may point at a clone the backend no longer holds ...
                assert!(
                    live.contains(&a),
                    "feature {} anchored on dead clone {a}",
                    f.feat_id
                );
                // ... and, the harness rule: once two registered clones exist, nothing
                // is left on the oldest live clone after the feed (it was moved onto
                // the previous frame's clone before the update; births anchor on the
                // current one).
                if k >= 2 {
                    assert_ne!(
                        a, live[0],
                        "feature {} still on the oldest clone",
                        f.feat_id
                    );
                }
            }
        }
        assert!(
            saw_rel,
            "relative covariances for more than the current clone"
        );
        assert!(
            sparse.feature_count() > 0,
            "the depth filter actually holds features"
        );
        assert!(
            reanchored > 0,
            "the harness rule re-anchors every frame from frame 2 on"
        );

        // EqVIO: no clone management -> the legacy absolute update, no re-anchoring.
        let settings = crate::VIOFilterSettings::default();
        let state = crate::mathematical::vio_state::VIOState::new(
            crate::mathematical::vio_state::VIOSensorState {
                input_bias: Vector6::zeros(),
                pose: SE3::identity(),
                velocity: Vector3::new(VX, 0.0, 0.0),
                camera_offset: SE3::identity(),
            },
            vec![],
        );
        let mut e = EqvioBackend::new(crate::VIOFilter::new(settings, state), test_cam());
        let mut sparse_e = Sparse3DFilter::new(
            k,
            Sparse3DChart::BearingInvDepthAdditive,
            SparseVogSettings {
                min_track_length: 1,
                birth_min_flow_px: 0.0,
                ..Default::default()
            },
        )
        .with_camera(test_cam());
        let mut feed_e = CloneRelFeed::default();
        for k in 0..6u64 {
            let t = k as f64 * 0.05;
            e.process_imu(imu(t));
            let frame = frame_at(t, &ids);
            e.process_vision(frame.clone(), &HashMap::new());
            let t_wc = e.nav_estimate().pose.as_matrix();
            let r = feed_e.feed(e.pose_covariance(), &mut sparse_e, &frame, &t_wc);
            assert!(
                !r.clone_relative,
                "EqVIO does not manage clones: legacy path"
            );
            assert_eq!(r.n_rel, 0);
            assert_eq!(r.reanchored, 0);
        }
        assert!(sparse_e
            .features_iter()
            .all(|f| f.anchor_clone_id.is_none()));
    }

    fn make_backend(settings: MsceqfSettings) -> MsceqfBackend {
        make_backend_with_origin(settings, Vector3::zeros(), SE3::identity())
    }

    fn make_backend_with_origin(
        settings: MsceqfSettings,
        p0: Vector3<f64>,
        s0: SE3,
    ) -> MsceqfBackend {
        let origin = SystemOrigin::new(
            SE23::new(SO3::identity(), p0, Vector3::new(VX, 0.0, 0.0)),
            Vector6::zeros(),
            s0,
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

    /// Civera birth must admit tracks the triangulated path REFUSES, or the setting is
    /// a no-op dressed as a choice.
    ///
    /// A near-stationary camera gives almost no bearing spread, which is exactly what the
    /// condition-number gate inside `triangulate` rejects, and it is also the regime that
    /// dominates this rig's outdoor ride. Civera needs no parallax because the depth
    /// comes from the prior, so the two paths must disagree here. Both arms see
    /// byte-identical imagery and differ only in the birth rule.
    #[test]
    fn civera_birth_admits_what_triangulated_birth_refuses() {
        // Camera essentially still: 30 frames x 0.05 s x 1 mm/s = 1.5 mm of baseline.
        fn still_frame(t: f64, ids: &[u64]) -> VisionMeasurement {
            let cam = test_cam();
            let p_cam = Vector3::new(1.0e-3 * t, 0.0, 0.0);
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
        // Zero initial velocity, so the filter is not propagating away from imagery that
        // says it is still -- the arms must differ on the BIRTH RULE, nothing else.
        let born = |civera: f64| -> usize {
            let origin = SystemOrigin::new(
                SE23::new(SO3::identity(), Vector3::zeros(), Vector3::zeros()),
                Vector6::zeros(),
                SE3::identity(),
                Vector3::new(0.0, 0.0, -9.81),
            );
            let filter = MSCEqFFilter::new(
                origin,
                &(SMatrix::<f64, 9, 9>::identity() * 1e-4),
                &(SMatrix::<f64, 6, 6>::identity() * 1e-4),
                &(SMatrix::<f64, 6, 6>::identity() * 1e-6),
                ProcessNoise {
                    angular_velocity_std: 1e-4,
                    acceleration_std: 1e-3,
                    angular_velocity_bias_std: 1e-6,
                    acceleration_bias_std: 1e-5,
                    state_transition_order: 1,
                },
            );
            let mut b = MsceqfBackend::new(
                filter,
                test_cam(),
                MsceqfSettings {
                    num_clones: 6,
                    slam_min_len: Some(4),
                    max_slam: 8,
                    civera_birth_depth: civera,
                    ..Default::default()
                },
            );
            let ids: Vec<u64> = (0..20).collect();
            for k in 0..30u64 {
                let t = k as f64 * 0.05;
                b.process_imu(imu(t));
                b.process_vision(still_frame(t, &ids), &HashMap::new());
            }
            b.n_slam()
        };
        let tri = born(0.0);
        let civ = born(6.0); // world_points sit a few metres out, so 6 m is a sane prior
        assert_eq!(
            tri, 0,
            "triangulated birth should refuse a 1.5 mm baseline, admitted {tri}"
        );
        assert!(
            civ > 0,
            "Civera birth should admit tracks without parallax, admitted {civ}"
        );
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
            max_promote_range: 0.0,
            direct_product_bias: false,
            clone_at_frame_stamp: false,
            civera_birth_depth: 0.0,
            civera_birth_rel_std: 1.0,
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
                max_promote_range: 0.0,
                direct_product_bias: false,
                civera_birth_depth: 0.0,
                civera_birth_rel_std: 1.0,
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
