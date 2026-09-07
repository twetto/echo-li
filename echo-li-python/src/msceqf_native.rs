//! Python binding for the MSCEqF-native filter
//! (`echo_li_core::mathematical::msceqf_filter`).
//!
//! This is the PARALLEL, MSCEqF-native symmetry-group filter — covariance on the
//! group algebra (Dd = SE_2(3)⋉bias, E = SE3, clones = SE3) — exposed SEPARATELY
//! from `VIOFilter` (the EqVIO EqF). It exists so the MidAir harness can drive the
//! faithful MSCEqF port and measure ATE against the C++ reference (0.6% on VO_test
//! t2) WITHOUT touching the EqVIO covariance machinery.
//!
//! Clones are addressed from Python by an opaque `clone_id` (the harness's frame
//! index). The binding keeps a `clone_ids` vector parallel to the filter's clone
//! window (ascending birth order) and translates `clone_id -> position` for the
//! update/marginalize calls.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

use echo_li_core::mathematical::camera::PinholeModel;
use echo_li_core::mathematical::msceqf_filter::{
    Imu, LmStreamUpdate, LmUpdate, MSCEqFFilter, MscTrack, MscTrackObs, ProcessNoise, SystemOrigin,
};
use echo_lie::{SE3, SE23, SO3};
use nalgebra::{Matrix3, Matrix4, SMatrix, Vector2, Vector3, Vector6};

fn mat3(a: PyReadonlyArray2<'_, f64>) -> Matrix3<f64> {
    let v = a.as_array();
    Matrix3::from_fn(|i, j| v[[i, j]])
}
fn mat4(a: PyReadonlyArray2<'_, f64>) -> Matrix4<f64> {
    let v = a.as_array();
    Matrix4::from_fn(|i, j| v[[i, j]])
}
fn vec3(a: &PyReadonlyArray1<'_, f64>) -> Vector3<f64> {
    let v = a.as_array();
    Vector3::new(v[0], v[1], v[2])
}
fn vec6(a: &PyReadonlyArray1<'_, f64>) -> Vector6<f64> {
    let v = a.as_array();
    Vector6::new(v[0], v[1], v[2], v[3], v[4], v[5])
}

#[pyclass(name = "MSCEqFNativeFilter")]
pub struct PyMSCEqFNativeFilter {
    filter: MSCEqFFilter,
    /// Frame-id of each live clone, parallel to `filter.x.clones` (ascending).
    clone_ids: Vec<u64>,
    /// Track-id of each in-state landmark, parallel to `filter.x.landmarks`
    /// (birth order). Gives the harness a STABLE handle: clone marginalization
    /// shifts landmark cov columns but never their order, so a landmark keeps its
    /// index (hence its track-id slot) until it is itself marginalized.
    landmark_ids: Vec<u64>,
    /// Normalized (Z1) projection model: the harness pre-normalizes pixels, so
    /// `fx=fy=1, cx=cy=0` and `pixel_std` is in normalized units.
    cam: PinholeModel,
    max_clones: usize,
}

impl PyMSCEqFNativeFilter {
    /// Translate `[(clone_id, u_n, v_n)]` into `MscTrackObs` on live clones.
    /// (Private — kept out of `#[pymethods]` so pyo3 does not try to export it.)
    fn obs_on_live_clones(&self, obs_list: &[(u64, f64, f64)]) -> Vec<MscTrackObs> {
        let mut obs: Vec<MscTrackObs> = Vec::new();
        for &(cid, un, vn) in obs_list {
            if let Some(pos) = self.clone_ids.iter().position(|&c| c == cid) {
                obs.push(MscTrackObs {
                    clone: pos,
                    uvn: Vector2::new(un, vn),
                });
            }
        }
        obs
    }
}

#[pymethods]
impl PyMSCEqFNativeFilter {
    /// Construct at a GT-seeded given origin.
    ///
    /// * `r0` (3x3) body->world rotation, `p0`/`v0` (3) world position/velocity,
    ///   `b0` (6) IMU bias [gyro;accel] — the fixed origin `xi0`.
    /// * `s0` (4x4) camera extrinsics `T_imu_cam` (cam->body).
    /// * `g` (3) world gravity (MidAir NED = [0,0,+9.81]).
    /// * `d_std` (9) SE_2(3) std in MSCEqF order [att,vel,pos]; `delta_std` (6) bias
    ///   std; `e_std` (6) extrinsics std — initial-covariance diagonals.
    /// * noise: accel/gyro density + random walk; `transition_order` (1 => I+H·dt,
    ///   else matrix-exp); `num_clones` = sliding-window size.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (r0, p0, v0, b0, s0, g, d_std, delta_std, e_std,
        accel_density, gyro_density, accel_rw, gyro_rw, transition_order, num_clones))]
    fn new(
        r0: PyReadonlyArray2<'_, f64>,
        p0: PyReadonlyArray1<'_, f64>,
        v0: PyReadonlyArray1<'_, f64>,
        b0: PyReadonlyArray1<'_, f64>,
        s0: PyReadonlyArray2<'_, f64>,
        g: PyReadonlyArray1<'_, f64>,
        d_std: PyReadonlyArray1<'_, f64>,
        delta_std: PyReadonlyArray1<'_, f64>,
        e_std: PyReadonlyArray1<'_, f64>,
        accel_density: f64,
        gyro_density: f64,
        accel_rw: f64,
        gyro_rw: f64,
        transition_order: usize,
        num_clones: usize,
    ) -> Self {
        let r0m = mat3(r0);
        let t0 = SE23::new(SO3::from_matrix(&r0m), vec3(&p0), vec3(&v0));
        let s0m = mat4(s0);
        let s0e = SE3::from_matrix(&s0m);
        let origin = SystemOrigin::new(t0, vec6(&b0), s0e, vec3(&g));

        let ds = d_std.as_array();
        let d_init = SMatrix::<f64, 9, 9>::from_fn(|i, j| if i == j { ds[i] * ds[i] } else { 0.0 });
        let dl = delta_std.as_array();
        let delta_init =
            SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { dl[i] * dl[i] } else { 0.0 });
        let es = e_std.as_array();
        let e_init = SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { es[i] * es[i] } else { 0.0 });

        let noise = ProcessNoise {
            angular_velocity_std: gyro_density,
            acceleration_std: accel_density,
            angular_velocity_bias_std: gyro_rw,
            acceleration_bias_std: accel_rw,
            state_transition_order: transition_order,
        };
        let filter = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, noise);
        Self {
            filter,
            clone_ids: Vec::new(),
            landmark_ids: Vec::new(),
            cam: PinholeModel {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            },
            max_clones: num_clones,
        }
    }

    /// Propagate one IMU step (covariance then mean). `gyro`/`accel` are body-frame.
    fn process_imu(
        &mut self,
        gyro: PyReadonlyArray1<'_, f64>,
        accel: PyReadonlyArray1<'_, f64>,
        dt: f64,
    ) {
        let u = Imu {
            ang: vec3(&gyro),
            acc: vec3(&accel),
        };
        self.filter.propagate_step(&u, dt);
    }

    /// Stochastically clone the current camera element, tagged `clone_id`.
    fn clone_pose(&mut self, clone_id: u64, stamp: f64) {
        self.filter.stochastic_clone(stamp);
        self.clone_ids.push(clone_id);
    }

    /// Marginalize clone `clone_id` (exact block deletion). Returns True if live.
    fn marginalize_clone(&mut self, clone_id: u64) -> bool {
        if let Some(pos) = self.clone_ids.iter().position(|&c| c == clone_id) {
            self.filter.marginalize_clone(pos);
            self.clone_ids.remove(pos);
            true
        } else {
            false
        }
    }

    /// Marginalize the oldest clone; returns its `clone_id` or None if empty.
    fn marginalize_oldest(&mut self) -> Option<u64> {
        if self.clone_ids.is_empty() {
            return None;
        }
        let id = self.clone_ids[0];
        self.filter.marginalize_clone(0);
        self.clone_ids.remove(0);
        Some(id)
    }

    fn n_clones(&self) -> usize {
        self.filter.x.clones.len()
    }
    fn clone_ids(&self) -> Vec<u64> {
        self.clone_ids.clone()
    }
    #[getter]
    fn max_clones(&self) -> usize {
        self.max_clones
    }

    /// Structureless MSC update. `tracks` maps `track_id -> [(clone_id, u_n, v_n)]`
    /// with NORMALIZED image coordinates. Observations referencing a non-live clone
    /// are dropped; tracks with <2 live observations are skipped. Returns the number
    /// of accepted tracks.
    #[pyo3(signature = (tracks, pixel_std, chi2_mult=1.0, curvature=true))]
    fn msc_update(
        &mut self,
        tracks: HashMap<u64, Vec<(u64, f64, f64)>>,
        pixel_std: f64,
        chi2_mult: f64,
        curvature: bool,
    ) -> usize {
        // clone_id -> window position.
        let pos_of: HashMap<u64, usize> = self
            .clone_ids
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        let mut msc_tracks: Vec<MscTrack> = Vec::new();
        for obs_list in tracks.values() {
            let mut obs: Vec<MscTrackObs> = Vec::new();
            for &(cid, un, vn) in obs_list {
                if let Some(&pos) = pos_of.get(&cid) {
                    obs.push(MscTrackObs {
                        clone: pos,
                        uvn: Vector2::new(un, vn),
                    });
                }
            }
            if obs.len() >= 2 {
                msc_tracks.push(MscTrack { obs });
            }
        }
        if msc_tracks.is_empty() {
            return 0;
        }
        self.filter
            .msc_update(&msc_tracks, &self.cam, pixel_std, chi2_mult, curvature)
    }

    // ---- In-state (SLAM) landmarks --------------------------------------------

    /// Promote a structureless track to an in-state landmark. `obs` is
    /// `[(clone_id, u_n, v_n)]` (normalized). Triangulates in the oldest observing
    /// clone's frame, χ²-gates the geometry, and augments the state with a
    /// correlated SOT3 landmark. Returns True if the landmark was born (tagged by
    /// `track_id` for later `landmark_update`/`reanchor`/`marginalize`).
    #[pyo3(signature = (track_id, obs, pixel_std, chi2_mult=1.0))]
    fn birth_landmark(
        &mut self,
        track_id: u64,
        obs: Vec<(u64, f64, f64)>,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> bool {
        if self.landmark_ids.contains(&track_id) {
            return false; // already in-state
        }
        let obs = self.obs_on_live_clones(&obs);
        if obs.len() < 2 {
            return false;
        }
        let track = MscTrack { obs };
        match self
            .filter
            .birth_landmark(&track, &self.cam, pixel_std, chi2_mult)
        {
            Some(j) => {
                debug_assert_eq!(j, self.landmark_ids.len(), "birth appends at tail");
                self.landmark_ids.push(track_id);
                true
            }
            None => false,
        }
    }

    /// EKF update of in-state landmarks. `updates` maps `track_id ->
    /// [(clone_id, u_n, v_n)]`. Each landmark is updated jointly with its anchor
    /// clone (no nullspace projection — the feature is a state). Observations on
    /// dead clones are dropped; a landmark whose anchor is not among the live
    /// observations, or with <2 live observations, is skipped. Returns the number
    /// of accepted landmark updates.
    #[pyo3(signature = (updates, pixel_std, chi2_mult=1.0))]
    fn landmark_update(
        &mut self,
        updates: HashMap<u64, Vec<(u64, f64, f64)>>,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> usize {
        let mut lm_updates: Vec<LmUpdate> = Vec::new();
        for (tid, obs_list) in &updates {
            let j = match self.landmark_ids.iter().position(|&t| t == *tid) {
                Some(j) => j,
                None => continue,
            };
            let obs = self.obs_on_live_clones(obs_list);
            if obs.len() < 2 {
                continue;
            }
            lm_updates.push(LmUpdate { j, obs });
        }
        if lm_updates.is_empty() {
            return 0;
        }
        self.filter
            .landmark_update(&lm_updates, &self.cam, pixel_std, chi2_mult)
    }

    /// Streaming (per-frame) SLAM update: update each in-state landmark with only
    /// its NEW observations, the anchor decoupled (never re-observed) so no
    /// measurement is used twice. `updates` maps `track_id -> [(clone_id, u_n, v_n)]`
    /// where the obs are the landmark's fresh observations at NON-anchor clones
    /// (anchor obs, if present, are ignored). Returns the accepted count. This is the
    /// update the driver calls every frame for persistent features; `landmark_update`
    /// (batch, anchor re-observed) is only for one-shot re-triangulation.
    #[pyo3(signature = (updates, pixel_std, chi2_mult=1.0))]
    fn landmark_stream_update(
        &mut self,
        updates: HashMap<u64, Vec<(u64, f64, f64)>>,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> usize {
        let mut lm_updates: Vec<LmStreamUpdate> = Vec::new();
        for (tid, obs_list) in &updates {
            let j = match self.landmark_ids.iter().position(|&t| t == *tid) {
                Some(j) => j,
                None => continue,
            };
            let obs = self.obs_on_live_clones(obs_list);
            if obs.is_empty() {
                continue;
            }
            lm_updates.push(LmStreamUpdate { j, obs });
        }
        if lm_updates.is_empty() {
            return 0;
        }
        self.filter
            .landmark_stream_update(&lm_updates, &self.cam, pixel_std, chi2_mult)
    }

    /// Move landmark `track_id`'s anchor to clone `new_clone_id` (covariance-
    /// consistent change of variables; the world point is invariant). Call this
    /// before the current anchor clone marginalizes. Returns True on success.
    fn reanchor_landmark(&mut self, track_id: u64, new_clone_id: u64) -> bool {
        let j = match self.landmark_ids.iter().position(|&t| t == track_id) {
            Some(j) => j,
            None => return false,
        };
        let pos = match self.clone_ids.iter().position(|&c| c == new_clone_id) {
            Some(p) => p,
            None => return false,
        };
        self.filter.reanchor_landmark(j, pos)
    }

    /// Marginalize (drop) landmark `track_id` from the state. Returns True if live.
    fn marginalize_landmark(&mut self, track_id: u64) -> bool {
        if let Some(j) = self.landmark_ids.iter().position(|&t| t == track_id) {
            self.filter.marginalize_landmark(j);
            self.landmark_ids.remove(j);
            true
        } else {
            false
        }
    }

    fn n_landmarks(&self) -> usize {
        self.filter.n_landmarks()
    }
    fn landmark_ids(&self) -> Vec<u64> {
        self.landmark_ids.clone()
    }

    /// Clone-id of landmark `track_id`'s current anchor, or None if the track is not
    /// in-state or its anchor clone is no longer live. The driver uses this to
    /// reanchor a landmark BEFORE its anchor clone marginalizes.
    fn landmark_anchor(&self, track_id: u64) -> Option<u64> {
        let j = self.landmark_ids.iter().position(|&t| t == track_id)?;
        let stamp = self.filter.x.landmarks[j].anchor;
        let pos = self.filter.x.clones.iter().position(|c| c.stamp == stamp)?;
        self.clone_ids.get(pos).copied()
    }

    /// World-frame point estimate of landmark `track_id` (anchor_pose · q·origin),
    /// or None if the track is not in-state or its anchor is not live.
    fn landmark_world<'py>(
        &self,
        py: Python<'py>,
        track_id: u64,
    ) -> Option<Bound<'py, PyArray1<f64>>> {
        let j = self.landmark_ids.iter().position(|&t| t == track_id)?;
        let lm = &self.filter.x.landmarks[j];
        let anchor = self.filter.x.clones.iter().find(|c| c.stamp == lm.anchor)?;
        let w = anchor.pose.act(&lm.point());
        Some(PyArray1::from_slice(py, w.as_slice()))
    }

    /// The 3x3 covariance block (chart tangent) of landmark `track_id`, or None if
    /// the track is not in-state.
    fn landmark_cov<'py>(
        &self,
        py: Python<'py>,
        track_id: u64,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        let j = self.landmark_ids.iter().position(|&t| t == track_id)?;
        let li = self.filter.landmark_idx(j);
        let mut m = Array2::<f64>::zeros((3, 3));
        for r in 0..3 {
            for c in 0..3 {
                m[[r, c]] = self.filter.cov[(li + r, li + c)];
            }
        }
        Some(PyArray2::from_array(py, &m))
    }

    /// Current nav (body-in-world) state via the `phi` action: returns
    /// `(position[3], rotation[3x3], velocity[3])`.
    fn nav_body_pose<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let nav = self.filter.phi();
        let p = nav.t.position;
        let r = nav.t.rotation.as_matrix();
        let v = nav.t.velocity;
        let pos = PyArray1::from_slice(py, p.as_slice());
        let mut rm = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                rm[[i, j]] = r[(i, j)];
            }
        }
        let vel = PyArray1::from_slice(py, v.as_slice());
        (pos, PyArray2::from_array(py, &rm), vel)
    }

    /// Current IMU bias estimate [gyro(3); accel(3)].
    fn bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let nav = self.filter.phi();
        PyArray1::from_slice(py, nav.b.as_slice())
    }

    /// Total covariance dimension (`21 + 6*n_clones`).
    fn cov_dim(&self) -> usize {
        self.filter.dim()
    }
}
