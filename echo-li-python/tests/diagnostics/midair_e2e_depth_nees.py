"""End-to-end depth NEES: real VIO + real frontend + Sparse3D → score against GT depth.

Runs the FULL pipeline: Rudolf-V tracker, EqVIO, Sparse3D with pose-driven range
process noise. Depth NEES scored against MidAir GT depth map at the tracked pixel.

Architecture mirrors midair_vio_sparse3d_prior_ab.py (event-based IMU/camera
interleaving, gyro frame repair, external Frontend tracker).

Evidence script for filter_formulation.md "End-to-end depth NEES" subsection.

To reproduce the VO_test table (≈3 min per run):

  PY=echo-li-python/venv/bin/python
  S=echo-li-python/tests/diagnostics/midair_e2e_depth_nees.py
  CFG=configs/diagnostics_midair_e2e_depth_nees.yaml
  ROOT=~/18TB/datasets/dataset_MidAir/MidAir

  # Baseline — prs=0, over-confident
  $PY $S --root $ROOT --traj 0 --config $CFG --pose-range-scale 0 --min-track 20
  $PY $S --root $ROOT --traj 2 --config $CFG --pose-range-scale 0 --min-track 20
  # Recommended — prs=0.003, honest
  $PY $S --root $ROOT --traj 0 --config $CFG --pose-range-scale 0.003 --min-track 20
  $PY $S --root $ROOT --traj 2 --config $CFG --pose-range-scale 0.003 --min-track 20

Expected output (configs/diagnostics_midair_e2e_depth_nees.yaml, noise-density fix):

  | traj | prs   | NEES med | %%>χ²₉₅ | %%>χ²₉₉ | σ_r (m) | err med |
  |------|-------|----------|---------|---------|---------|---------|
  | 0    | 0.003 | 0.356    | 17.2%%  | 14.2%%  | 51      | −28%%   |
  | 2    | 0.003 | 0.184    | 6.9%%   | 5.1%%   | 19      | −10%%   |
  | t(2.6) ref |  | 0.609   | 15.9%%  | 9.5%%   |         |         |

For Kite_training cross-trajectory validation (≈90 min), see
midair_measure_imu_noise.py and midair_e2e_depth_nees_batch.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md
import echo_li

# Keys forwarded from the YAML SparseVog section to the Sparse3D constructor.
# NOTE: a key absent from this list is silently DROPPED from the YAML -- the filter
# then runs with its default and the run looks superficially fine. Any new
# SparseVogSettings field must be added here AND verified to arrive (see the
# "Sparse3D range knobs" banner below, which echoes what actually took effect).
SPARSE_KEYS = [
    "max_pool_size", "min_track_length", "conv_variance_threshold",
    "civera_birth_depth", "civera_rho_sigma_frac",
    "init_depth_var", "init_invdepth_var", "sigma_pixel", "flow_age_rate_px_per_frame",
    "bias_walk_var", "uniform_z_max", "uniform_rho_max", "uniform_d_min", "uniform_d_max",
    "a_init", "b_init", "ab_min", "ab_max", "min_inlier_ratio",
    "mahalanobis_reset_chi2", "process_depth_var", "range_walk_var", "pose_range_scale",
    "pose_range_coherent", "pose_range_direct", "scale_sigma", "conv_range_sigma_frac",
    "min_parallax", "min_cos_sim", "min_depth", "max_depth", "birth_min_flow_px",
    "use_equivariant_output", "iekf_iterations", "rotation_unscented",
]
R_MARGIN = 4  # pixels from image border to ignore when reading GT depth


def _grid_select(all_uvs, existing, budget, W, H, n_cols=5, n_rows=5):
    """Select up to *budget* feature IDs spread evenly across a grid.

    Each cell gets at most ceil(budget / n_cells) features.  Within a cell,
    existing landmarks (already in the EqF state) are kept first, then new
    ones are added.  The overall result is truncated to *budget*.
    """
    import math
    n_cells = n_cols * n_rows
    per_cell = math.ceil(budget / n_cells)
    cw, ch = W / n_cols, H / n_rows

    # Bin features into grid cells.
    cells = [[] for _ in range(n_cells)]
    for fid, (u, v) in all_uvs.items():
        ci = min(int(u / cw), n_cols - 1)
        ri = min(int(v / ch), n_rows - 1)
        cells[ri * n_cols + ci].append(fid)

    selected = []
    for cell in cells:
        # Sort: existing first, then new.
        cell.sort(key=lambda fid: (0 if fid in existing else 1, fid))
        selected.extend(cell[:per_cell])

    # If total exceeds budget (because many cells are sparse and a few dense),
    # prefer existing landmarks globally.
    if len(selected) > budget:
        selected.sort(key=lambda fid: (0 if fid in existing else 1, fid))
        selected = selected[:budget]
    return selected


def psd_clip(M):
    """Nearest PSD matrix (clip negative eigenvalues to 0). Used for the per-frame
    INCREMENTAL relative-pose covariance Cov_rel(k→t) − Cov_rel(k→t−1), which a
    measurement update can make momentarily non-PSD."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.clip(w, 0.0, None)) @ V.T


class MSCEqFAdapter:
    """Presents EqVIO's surface, backed by the fork-b native MSCEqF filter.

    Why an adapter rather than a branch in the main loop: the loop's EqVIO calls are
    spread over ~100 lines, and the two filters differ in ORDERING, not just naming.
    EqVIO keeps landmarks internally and is fed per-frame uvs; the native filter is an
    MSCKF that must CLONE first, then update from multi-observation tracks. Hiding that
    behind the same method names keeps the scoring path byte-identical between pose
    sources, which is the whole point of the comparison.

    The relative covariance it exposes is the clone-to-clone anchor->current one. There
    is no core-state Jacobian route here: phi()'s camera pose is invariant to the whole
    navigation direction (D cancels between t = T0.D and s = C(D)^-1.S0.E), so the
    current camera pose IS a clone and both ends live in the clone tail.
    """

    def __init__(self, R0, p0, v0, S0, grav, K, num_clones, pixel_std,
                 acc_d, gyr_d, acc_rw, gyr_rw, chi2_mult=1.0, max_track=200,
                 update_span=11, max_slam=0):
        from echo_li import MSCEqFNativeFilter
        d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
        self.f = MSCEqFNativeFilter(
            np.ascontiguousarray(R0), np.ascontiguousarray(p0),
            np.ascontiguousarray(v0), np.zeros(6), np.ascontiguousarray(S0),
            np.ascontiguousarray(grav), d_std, np.full(6, 1e-2), np.full(6, 1e-4),
            acc_d, gyr_d, acc_rw, gyr_rw, 0, num_clones)
        self.K_inv = np.linalg.inv(np.asarray(K, float))
        self.num_clones = num_clones
        self.pixel_std = pixel_std
        self.chi2_mult = chi2_mult
        self.max_track = max_track
        # The MSCKF UPDATE span and the Sparse3D ANCHOR window are independent. Tying
        # them together breaks one or the other: an 11-clone tail starves Sparse3D of
        # anchors (feats 25->13->3), while flushing tracks only when they reach a
        # 120-clone marginalization means almost no update fires and the filter runs
        # near open-loop (navErr 0.2 -> 2.1 m). Keep a LONG tail for anchors, flush
        # tracks for update after `update_span` observations.
        self.update_span = max(2, int(update_span))
        # In-state SLAM landmarks. Structureless-only is the WEAKEST configuration of
        # this filter -- measured earlier: enabling them roughly halves ATE (t1 0.983 ->
        # 0.459), and gravity-init structureless diverges outright where +landmarks does
        # not. The 7 no-output trajectories all show pose divergence (navErr 2.9-8.2 m),
        # so running them structureless is testing the filter at its most fragile.
        self.max_slam = int(max_slam)
        self.slam = set()
        self.buf = {}
        self._last_stamp = None
        self._k = -1
        self._ext = np.asarray(S0, float)

    # --- IMU -------------------------------------------------------------
    def process_imu(self, stamp, gyr, acc):
        if self._last_stamp is not None:
            dt = stamp - self._last_stamp
            if 0.0 < dt < 0.1:
                self.f.process_imu(np.asarray(gyr, float), np.asarray(acc, float), dt)
        self._last_stamp = stamp

    # --- vision ----------------------------------------------------------
    def _norm(self, uv):
        v = self.K_inv @ np.array([uv[0], uv[1], 1.0])
        return float(v[0]), float(v[1])

    def process_vision(self, stamp, uvs):
        self._k += 1
        k = self._k
        self.f.clone_pose(k, stamp)                      # clone BEFORE update
        cur = set()
        for fid, uv in uvs.items():
            un, vn = self._norm(uv)
            if int(fid) not in self.slam:
                self.buf.setdefault(int(fid), []).append((k, un, vn))
            cur.add(int(fid))
        marg = self.f.clone_ids()[0] if self.f.n_clones() > self.num_clones else None

        # (0) promote window-spanning tracks to in-state landmarks, then stream-update
        #     the live ones. Mirrors midair_msceqf_native_stereo_ate.py.
        if self.max_slam > 0 and len(self.slam) < self.max_slam:
            cand = [t for t, o in self.buf.items()
                    if t in cur and len(o) >= self.update_span]
            cand.sort(key=lambda t: len(self.buf[t]), reverse=True)
            for tid in cand[: self.max_slam - len(self.slam)]:
                if self.f.birth_landmark(tid, self.buf[tid], self.pixel_std,
                                         self.chi2_mult, None):
                    self.f.reanchor_landmark(tid, k)
                    self.slam.add(tid)
                    self.buf.pop(tid, None)
        if self.slam:
            supd = {}
            for tid in list(self.slam):
                if tid in cur:
                    un, vn = self._norm(uvs[tid])
                    supd[tid] = [(k, un, vn)]
                else:
                    self.f.marginalize_landmark(tid)
                    self.slam.discard(tid)
            if supd:
                self.f.landmark_stream_update(supd, self.pixel_std, self.chi2_mult)

        ready, done = {}, []
        for tid, obs in self.buf.items():
            lost = tid not in cur
            toolong = len(obs) >= self.max_track
            spanned = len(obs) >= self.update_span
            touches = marg is not None and any(o[0] == marg for o in obs)
            if lost or toolong or touches or spanned:
                if len(obs) >= 2:
                    ready[tid] = obs
                done.append(tid)
        if ready:
            self.f.msc_update(ready, self.pixel_std, self.chi2_mult, False)
        for tid in done:
            self.buf.pop(tid, None)
        if self.slam and marg is not None:
            newest = self.f.clone_ids()[-1]
            for tid in list(self.slam):
                if self.f.landmark_anchor(tid) == marg:
                    self.f.reanchor_landmark(tid, newest)
        while self.f.n_clones() > self.num_clones:
            old = self.f.marginalize_oldest()
            for tid in list(self.buf):
                b = [o for o in self.buf[tid] if o[0] != old]
                if b:
                    self.buf[tid] = b
                else:
                    self.buf.pop(tid, None)

    # --- EqVIO-shaped accessors -----------------------------------------
    def get_landmarks(self):
        return {int(t): None for t in self.buf}

    def get_pose(self):
        # vio_body_pose() feeds this straight to scipy Rot.from_quat, which wants
        # XYZW. Returning WXYZ scrambles the rotation -- and it is INVISIBLE in the
        # harness's navErr, which reads only [:3, 3]. Return scipy order unchanged.
        p, R, _v = self.f.nav_body_pose()
        q = Rot.from_matrix(np.asarray(R, float)).as_quat()   # xyzw, as scipy wants
        return np.asarray(p, float), q

    def get_camera_pose_covariance(self):
        ids = self.clone_ids()
        if not ids:
            return None
        return self.f.clone_pose_covariance(ids[-1])

    def clone_pose(self, k, stamp):
        pass                   # already cloned in process_vision (MSCKF ordering)

    def marginalize_clone(self, cid):
        return False           # window is managed internally

    def clone_ids(self):
        return [int(c) for c in self.f.clone_ids()]

    def n_clones(self):
        return int(self.f.n_clones())

    def get_relative_pose_covariance(self, cid, _t_wc_clone):
        ids = self.clone_ids()
        if not ids:
            return None
        return self.f.relative_pose_covariance(int(cid), ids[-1])

    def relative_pose_covariance_full(self, cid, _cur_id):
        """Full 6x6 anchor->current relative-pose covariance, [rot, trans] order.

        Mirrors get_relative_pose_covariance (which returns only the two diagonal
        blocks); the caller here needs the dropped rotation-translation cross-block.
        """
        ids = self.clone_ids()
        if not ids:
            return None
        return self.f.relative_pose_covariance_full(int(cid), ids[-1])

    def set_camera_extrinsics(self, ext):
        pass

    def set_initial_state(self, *a, **kw):
        pass


class BackendAdapter:
    """Presents EqVIO's surface, backed by the SHIPPING path: `echo_li.VIOFilter` with an
    `MSCEqF:` config section, i.e. `backend_from_config` plus the `CameraPoseCovariance`
    seam that the CLI and the ROS 2 node consume. Built to the same construction as
    MSCEqFAdapter (initial stds, IMU noise, window, update span, curvature off, gravity),
    so a difference between `--pose-source msceqf` and `--pose-source backend` is a
    difference in the shipped wiring, not in the filter.

    Two surface differences are real and deliberate:
      * the backend owns clone timing (clone-before-update inside process_vision), so
        `clone_pose()` / `marginalize_clone()` are no-ops exactly as in MSCEqFAdapter;
      * `get_landmarks()` returns the backend's buffered track ids, which is what
        MSCEqFAdapter returned, so the front-end budget prefers the same features.
    """

    def __init__(self, cfg, cam, f, num_clones, pixel_std_px, max_slam,
                 acc_d, gyr_d, acc_rw, gyr_rw, update_span=11, max_track=200):
        import echo_li
        import tempfile
        full = _deep_copy(cfg)
        full["MSCEqF"] = {
            "enabled": True,
            "numClones": int(num_clones),
            "pixelStd": float(pixel_std_px),          # pixels; /fx inside
            "chi2Mult": 1.0,
            "maxTrack": int(max_track),
            "maxSlam": int(max_slam),
            "curvature": False,                       # MSCEqFAdapter passes False
            "updateSpan": int(update_span),
            "cloneAtFrameStamp": False,
            # NWU harness: z is UP, gravity ACCELERATION is -z (see MSCEqFAdapter).
            "gravity": [0.0, 0.0, -9.81],
            "initialStd": {"attitude": 1e-2, "velocity": 1e-1, "position": 1e-4,
                           "bias": 1e-2, "extrinsics": 1e-4},
            "processNoise": {"accelDensity": float(acc_d), "gyroDensity": float(gyr_d),
                             "accelRandomWalk": float(acc_rw),
                             "gyroRandomWalk": float(gyr_rw), "transitionOrder": 0},
        }
        full.pop("ZUPT", None)                        # the adapter has no ZUPT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(full, tf, sort_keys=False)
            self.config_path = tf.name
        self.f = echo_li.VIOFilter(self.config_path, cam, fx=float(f))
        assert self.f.backend_kind == "MSCEqF", self.f.backend_kind

    def process_imu(self, stamp, gyr, acc):
        self.f.process_imu(stamp, list(gyr), list(acc))

    def process_vision(self, stamp, uvs):
        self.f.process_vision(stamp, uvs)

    def get_landmarks(self):
        return {int(t): None for t in self.f.msc_tracked_ids()}

    def get_pose(self):
        return self.f.get_pose()

    def get_camera_pose_covariance(self):
        return self.f.get_camera_pose_covariance()

    def clone_pose(self, k, stamp):
        pass                   # the backend clones inside process_vision

    def marginalize_clone(self, cid):
        return False           # window is managed internally

    def clone_ids(self):
        return [int(c) for c in self.f.clone_ids()]

    def n_clones(self):
        return int(self.f.n_clones())

    def get_relative_pose_covariance(self, cid, t_wc_clone):
        return self.f.get_relative_pose_covariance(int(cid), np.ascontiguousarray(t_wc_clone))

    def relative_pose_covariance_full(self, cid, _cur_id):
        return self.f.relative_pose_covariance_full(int(cid))

    def last_clone_eviction(self):
        """(old_id, new_id, T_wc_new, P_vv, P_ww) captured BEFORE the backend dropped
        the clone this frame, or None. The CLI/ROS 2 feed re-anchors from this."""
        return self.f.last_clone_eviction()

    def set_camera_extrinsics(self, ext):
        self.f.set_camera_extrinsics(np.ascontiguousarray(ext))

    def set_initial_state(self, p, R, v_body):
        self.f.set_initial_state(p, np.ascontiguousarray(R), v_body)


def vio_body_pose(vio):
    """SE(3) pose of the VIO body frame in the world frame."""
    pos, quat = vio.get_pose()
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
    T[:3, 3] = np.asarray(pos)
    return T


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end depth NEES test for Sparse3D on MidAir.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to MidAir root")
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="image scale factor (0.5 = 512x256)")
    ap.add_argument("--config", required=True,
                    help="YAML config for VIO + Sparse3D + frontend")
    ap.add_argument("--eqf-max-obs", type=int, default=40,
                    help="cap on features sent to EqF (must match config maxFeatures)")
    ap.add_argument("--min-track", type=int, default=5,
                    help="minimum Sparse3D track length before scoring")
    ap.add_argument("--pvv-scale", type=float, default=1.0,
                    help="multiply P_vv by this before feeding to Sparse3D "
                         "(excitation gate γ²; 1 = no inflation)")
    ap.add_argument("--pose-range-scale", type=float, default=None,
                    help="override SparseVog.pose_range_scale (must be >0 for P_vv "
                         "to reach the range process noise; None = use config)")
    ap.add_argument("--pose-range-coherent", type=float, default=None,
                    help="override SparseVog.pose_range_coherent")
    ap.add_argument("--range-walk-var", type=float, default=None,
                    help="override SparseVog.range_walk_var (fixed radial floor)")
    # Principled per-trajectory overrides (bias priors, IMU noise).
    # These match the MidAir traj_0 measured noise; they override the YAML.
    ap.add_argument("--bias-acc", type=float, default=None,
                    help="override eqf.initialVariance.biasAcc")
    ap.add_argument("--bias-gyr", type=float, default=None,
                    help="override eqf.initialVariance.biasGyr")
    ap.add_argument("--vel-acc", type=float, default=None,
                    help="override eqf.velocityNoise.acc")
    ap.add_argument("--vel-gyr", type=float, default=None,
                    help="override eqf.velocityNoise.gyr")
    ap.add_argument("--sigma-pixel", type=float, default=None,
                    help="override SparseVog.sigma_pixel")
    ap.add_argument("--scene-depth", type=float, default=None,
                    help="override eqf.initialValue.sceneDepth")
    ap.add_argument("--eqf-max-depth", type=float, default=500.0,
                    help="reject features deeper than this from EqF input "
                         "(sky features at infinity cause phantom parallax)")
    ap.add_argument("--eqf-selection", choices=("grid", "existing-first"),
                    default="grid",
                    help="feature selection when --eqf-max-obs is exceeded")
    ap.add_argument("--save-npz", default="",
                    help="save raw results + provenance to this .npz path")
    ap.add_argument("--dump-traj", default="",
                    help="save per-frame estimated and GT positions (NWU) to .npz for "
                         "trajectory plotting")
    ap.add_argument("--msceqf-max-slam", type=int, default=0,
                    help="in-state SLAM landmarks for the native pose source. 0 = "
                         "structureless only, which is this filter's WEAKEST mode "
                         "(landmarks roughly halve ATE). The 7 no-output trajectories "
                         "all diverge in pose, so this is the first thing to try.")
    ap.add_argument("--pose-source", choices=("eqvio", "msceqf", "backend"), default="eqvio",
                    help="which filter supplies poses AND the clone-relative covariance. "
                         "'msceqf' routes through the fork-b native filter via "
                         "MSCEqFAdapter; its relative cov is clone-to-clone (no "
                         "core-state Jacobian exists there). 'backend' is the SHIPPING "
                         "wiring: VIOFilter with an MSCEqF: section, covariances through "
                         "the CameraPoseCovariance seam (BackendAdapter). The scoring "
                         "path is identical either way, so the three are directly "
                         "comparable; msceqf vs backend is the match test for the seam.")
    ap.add_argument("--reanchor-eviction", action="store_true",
                    help="with --pose-source backend --reanchor: re-anchor from the "
                         "backend's last_clone_eviction() record (old->newest, cov "
                         "captured before the eviction), which is what CloneRelFeed "
                         "does in the CLI/ROS 2 node, instead of the harness's "
                         "oldest-live->newest-registered rule one frame earlier")
    ap.add_argument("--clone-relative", action="store_true",
                    help="feed Sparse3D's §V-D pose-range term the HONEST, "
                         "gauge-cancelled anchor→current relative-pose covariance "
                         "from an EqF pose-clone window (instead of the absolute "
                         "per-frame cov); default off = exact absolute-cov behaviour.")
    ap.add_argument("--clone-window", type=int, default=120,
                    help="rolling clone-window length (frames); clones older than "
                         "this are marginalized. Must exceed the longest scored "
                         "track so its anchor clone survives.")
    ap.add_argument("--sigma-s", type=float, default=0.0,
                    help="exogenous multiplicative scale-uncertainty (1-sigma "
                         "fraction). Injects the VALIDATED honest scale term "
                         "var_r += (sigma_s * r)^2 into the scored radial "
                         "variance, sourced from excitation E, NOT p_vv "
                         "(sparse3d-depth-error-is-pose-scale-1to1). Compose with "
                         "--clone-relative (attitude channel) to test Verif #3. "
                         "sigma_s(E)=max(0.02, 0.0072*E^-0.710); Kite t0 -> 0.0395.")
    ap.add_argument("--reanchor-cov-scale", type=float, default=1.0,
                    help="scale the re-anchor covariance (sanity: does this path matter?)")
    ap.add_argument("--reanchor-nocov", action="store_true",
                    help="omit the relative-pose covariance at re-anchor (test for "
                         "double-counting against pose_range_direct)")
    ap.add_argument("--xcov-audit", action="store_true",
                    help="measure the discarded rotation-translation cross-block and "
                         "its effect on the combined range variance")
    ap.add_argument("--scale-sigma-auto", action="store_true",
                    help="compute E = std(w_z)*std(a_y) on this trajectory's raw IMU and "
                         "set settings.scale_sigma = max(0.02, 0.0072*E^-0.710)")
    ap.add_argument("--score-reported", action="store_true",
                    help="score the filter's OWN reported range variance "
                         "(range_var_reported = estimate var + (scale_sigma*r)^2) "
                         "instead of reconstructing it from covariance_euclidean")
    ap.add_argument("--pose-cov-audit", action="store_true",
                    help="score the REPORTED anchor->current translation covariance "
                         "against the ACTUAL error vs GT (3-dof NEES)")
    ap.add_argument("--nees-space", choices=("range", "invdepth"), default="range",
                    help="range = metric range vs the delta-method push-forward "
                         "(current default); invdepth = the anchor chart where the "
                         "EKF Gaussian actually lives")
    ap.add_argument("--reanchor", action="store_true",
                    help="re-anchor Sparse3D landmarks onto the newest clone before "
                         "their anchor clone is marginalized (lets --clone-window shrink)")
    ap.add_argument("--anchor-census", action="store_true",
                    help="report how many DISTINCT anchor clones are live, and the "
                         "oldest anchor's age -- the true clone-window requirement")
    ap.add_argument("--gt-anchored", action="store_true",
                    help="score against the reprojected GT BIRTH landmark instead of "
                         "GT depth at the current (drifted) tracked pixel")
    ap.add_argument("--converged-only", action="store_true",
                    help="score only features passing query_range's convergence gates")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    # --- Dataset ---
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    ext = md.RT_BC.copy()
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])

    # --- Config: load YAML, apply CLI overrides ---
    cfg_raw = yaml.safe_load(open(args.config)) or {}
    # VIO overrides (modify the dict before passing to VIOFilter via a temp file
    # if needed — but VIOFilter takes a path, so we write a merged copy).
    cfg = _deep_copy(cfg_raw)
    if args.bias_acc is not None:
        cfg.setdefault("eqf", {}).setdefault("initialVariance", {})["biasAcc"] = args.bias_acc
    if args.bias_gyr is not None:
        cfg.setdefault("eqf", {}).setdefault("initialVariance", {})["biasGyr"] = args.bias_gyr
    # --vel-acc / --vel-gyr are per-sample σ (from midair_measure_imu_noise.py).
    # EqF expects continuous-time noise density: σ_density = σ_sample · √dt.
    # MidAir IMU runs at 100 Hz → √dt = 0.1.
    imu_sqrt_dt = 0.1
    if args.vel_acc is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["acc"] = args.vel_acc * imu_sqrt_dt
    if args.vel_gyr is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["gyr"] = args.vel_gyr * imu_sqrt_dt
    if args.scene_depth is not None:
        cfg.setdefault("eqf", {}).setdefault("initialValue", {})["sceneDepth"] = args.scene_depth
    # Write merged config to a temp file for VIOFilter (which takes a path).
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        merged_config_path = tf.name

    # Sparse3D settings from config + CLI overrides
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    settings = {k: sparse_cfg[k] for k in SPARSE_KEYS if k in sparse_cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings.setdefault("min_track_length", 1)
    if args.pose_range_scale is not None:
        settings["pose_range_scale"] = args.pose_range_scale
    if args.pose_range_coherent is not None:
        settings["pose_range_coherent"] = args.pose_range_coherent
    if args.range_walk_var is not None:
        settings["range_walk_var"] = args.range_walk_var
    if args.sigma_pixel is not None:
        settings["sigma_pixel"] = args.sigma_pixel

    # Print effective range-critical settings for provenance.
    eff_prs = settings.get("pose_range_scale", 0.0)
    eff_prc = settings.get("pose_range_coherent", 0.0)
    eff_rwv = settings.get("range_walk_var", 0.0)
    eff_pdv = settings.get("process_depth_var", 0.0)
    print(f"Config: {args.config}")
    print(f"Sparse3D range knobs: direct={settings.get('pose_range_direct', False)}, pose_range_scale={eff_prs}, "
          f"pose_range_coherent={eff_prc}, range_walk_var={eff_rwv}, "
          f"process_depth_var={eff_pdv}")
    print(f"PVV scale (γ²): {args.pvv_scale}")
    if eff_prs == 0.0 and args.pvv_scale != 1.0:
        print("WARNING: pvv_scale has no effect when pose_range_scale=0 "
              "(P_vv enters measurement R only, which is radial-blind; "
              "Section V-D of filter_formulation.md).")

    # --- Frontend (Rudolf-V tracker) ---
    fcfg = echo_li.FrontendConfig.from_yaml(merged_config_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)

    # --- VIO ---
    # GT seed, computed ONCE so both pose sources start from the same state.
    gt0 = ds.pose(args.start)
    v0 = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][args.start * 4])
    R0 = t_ned_to_nwu @ gt0[:3, :3]
    p0_w = t_ned_to_nwu @ gt0[:3, 3]
    v0_w = t_ned_to_nwu @ v0
    v0_body = R0.T @ v0_w        # EqVIO wants BODY velocity; the native filter WORLD.

    _vn = dict(cfg.get("eqf", {}).get("velocityNoise", {}))
    if args.pose_source == "msceqf":
        print(f"  IMU noise from config: acc={_vn.get('acc')} gyr={_vn.get('gyr')} "
              f"accBias={_vn.get('accBias')} gyrBias={_vn.get('gyrBias')}")
        vio = MSCEqFAdapter(
            R0=R0, p0=p0_w, v0=v0_w, S0=ext,
            # This harness works in NWU (t_ned_to_nwu), so z is UP and gravity is -z.
            # The MidAir ATE harness uses NED (+9.81); copying that sign here makes the
            # filter integrate 2g of phantom acceleration.
            grav=np.array([0.0, 0.0, -9.81]),
            K=np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]),
            # Sparse3D anchors features to clones and needs them ALIVE for the whole
            # track; an 11-frame MSCKF window silently starves it (feats 25->13->3).
            # Match the harness's Sparse3D anchor window instead.
            num_clones=max(args.clone_window, 11), max_slam=args.msceqf_max_slam,
            pixel_std=(args.sigma_pixel or 1.0) / max(f, 1e-9),
            # Take IMU noise from the SAME config EqVIO uses. Hardcoding EuRoC-ish
            # values here understated gyro density 20x and gyro bias RW 7x, which makes
            # the filter over-trust the gyro -- and MidAir's IMU noise is randomised
            # PER TRAJECTORY (4-8x spread), so a borrowed constant cannot be right.
            acc_d=_vn.get("acc", 3.28e-3), gyr_d=_vn.get("gyr", 3.94e-3),
            acc_rw=_vn.get("accBias", 4.462e-3), gyr_rw=_vn.get("gyrBias", 1.337e-4))
        print("pose source: fork-b native MSCEqF (clone-to-clone relative cov); "
              f"seed |v0_world|={np.linalg.norm(v0_w):.2f} m/s")
    elif args.pose_source == "backend":
        print(f"  IMU noise from config: acc={_vn.get('acc')} gyr={_vn.get('gyr')} "
              f"accBias={_vn.get('accBias')} gyrBias={_vn.get('gyrBias')}")
        vio = BackendAdapter(
            cfg, cam, f,
            num_clones=max(args.clone_window, 11), max_slam=args.msceqf_max_slam,
            pixel_std_px=(args.sigma_pixel or 1.0),
            acc_d=_vn.get("acc", 3.28e-3), gyr_d=_vn.get("gyr", 3.94e-3),
            acc_rw=_vn.get("accBias", 4.462e-3), gyr_rw=_vn.get("gyrBias", 1.337e-4))
        print("pose source: shipping backend (VIOFilter + MSCEqF: section, "
              f"CameraPoseCovariance seam); config {vio.config_path}; "
              f"seed |v0_world|={np.linalg.norm(v0_w):.2f} m/s; "
              f"reanchor rule={'eviction' if args.reanchor_eviction else 'harness'}")
    else:
        vio = echo_li.VIOFilter(merged_config_path, cam)
    if args.reanchor_eviction and args.pose_source != "backend":
        sys.exit("--reanchor-eviction needs --pose-source backend")
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    vio.set_initial_state(p0_w.tolist(), np.ascontiguousarray(R0), v0_body.tolist())

    # --- Sparse3D ---
    if args.scale_sigma_auto:
        # sigma_s(E) with E = std(omega_z)*std(a_y) over the WHOLE trajectory on RAW IMU
        # -- the same excitation metric the scale-memory work uses (no frame transform).
        # Computed here rather than looked up so the term is operational, not a manual knob.
        _imu = ds.db[ds.traj]["imu"]
        _g = np.asarray(_imu["gyroscope"], float)
        _a = np.asarray(_imu["accelerometer"], float)
        _E = float(np.std(_g[:, 2]) * np.std(_a[:, 1]))
        _ss = max(0.02, 0.0072 * _E ** (-0.710)) if _E > 0 else 0.02
        settings["scale_sigma"] = _ss
        print(f"scale_sigma AUTO: E = {_E:.5f}  ->  sigma_s = {_ss:.4f}")

    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)

    # --- Build interleaved event list (IMU @ 100Hz, camera @ 25Hz) ---
    nimg = min(args.frames, ds.n - args.start)
    imu_data = ds.db[ds.traj]["imu"]
    accel = imu_data["accelerometer"][:]
    gyro_raw = imu_data["gyroscope"][:]
    imu0 = args.start * 4
    imu1 = min(len(accel), (args.start + nimg) * 4 + 4)
    events = [(i / 100.0, "imu", (gyro_raw[i].tolist(), accel[i].tolist()))
              for i in range(imu0, imu1)]
    for n in range(nimg):
        k = args.start + n
        events.append((k / 25.0, "cam", k))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    # --- Main loop ---
    rows = []
    t0 = time.time()
    n_frames = 0
    traj_est, traj_gt, traj_k = [], [], []
    # Clone-window bookkeeping (clone-relative pose covariance for Sparse3D §V-D).
    _dbg_reanchor = [0, 0, 0]   # [cov=None, cov set, dbg prints]
    n_reanchored = [0]    # landmarks moved to a new anchor (--reanchor)
    anchor_census = []    # (k, n_clones, n_feats, n_distinct_anchors, oldest_anchor_age)
    xcov_audit = []       # (|corr|, v_full/v_quadrature, baseline m)
    pose_audit = []       # (age, 3dof NEES, |err| m, sigma m, NEES_along, NEES_perp)
    srr = []              # sigma_rho / rho -- delta-method validity at the operating point
    gt_anchor = {}        # fid -> GT world point of the BIRTH landmark (--gt-anchored)
    clone_pose = {}       # clone_id (= frame k) -> T_wc (4x4) anchor pose
    clone_rel_prev = {}   # clone_id -> (P_vv, P_ww) accumulated Cov_rel(k→t−1)
    if args.clone_relative:
        print(f"Clone-relative pose cov ON  (window={args.clone_window} frames)")

    for stamp, et, data in events:
        if et == "imu":
            gyr, acc = data
            # Gyro frame repair: MidAir spatial → body using GT attitude.
            # Same as prior_ab.py lines 572-576 (--gyro-frame repaired_gt).
            gi = min(int(round(stamp * 100.0)), len(ds.att) - 1)
            q = ds.att[gi]
            R_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            gyr_body = (R_gt.T @ np.asarray(gyr)).tolist()
            vio.process_imu(stamp, gyr_body, acc)
            continue

        # Camera event.
        k = data
        gray = ds.image(k)

        # Track features with Rudolf-V.
        feats, _stats = tracker.process(gray)
        all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}

        # Cap observations for EqF with spatial distribution.
        # Grid-based selection: divide frame into cells, pick at most
        # ceil(budget/n_cells) per cell, preferring existing landmarks.
        existing = {int(x) for x in vio.get_landmarks().keys()}
        obs_ids = sorted(all_uvs.keys())
        if args.eqf_max_obs > 0 and len(obs_ids) > args.eqf_max_obs:
            if args.eqf_selection == "grid":
                obs_ids = _grid_select(all_uvs, existing, args.eqf_max_obs, W, H)
            else:
                keep_exist = [fid for fid in obs_ids if fid in existing]
                keep_new = [fid for fid in obs_ids if fid not in existing]
                obs_ids = (keep_exist + keep_new)[:args.eqf_max_obs]
        vio_uvs = {fid: all_uvs[fid] for fid in obs_ids}
        vio.process_vision(stamp, vio_uvs)

        # Sparse3D update with ALL tracked features.
        T_wc = vio_body_pose(vio) @ ext
        if args.dump_traj:
            traj_k.append(k)
            traj_est.append(vio_body_pose(vio)[:3, 3].copy())
            traj_gt.append((t_ned_to_nwu @ ds.pose(k)[:3, 3]).copy())
        pcov = vio.get_camera_pose_covariance()
        if pcov is not None:
            pvv = np.asarray(pcov[0], float) * args.pvv_scale
            pww = np.asarray(pcov[1], float)
            pvv_arg, pww_arg = pvv.tolist(), pww.tolist()
        else:
            pvv_arg = pww_arg = None

        # --- Clone-relative pose covariance ---
        # Clone THIS frame's pose (= anchor for any landmark born now), roll the
        # window, then feed Sparse3D the honest per-clone INCREMENTAL relative-pose
        # covariance so each landmark's §V-D term uses its own anchor→current cov.
        clone_id_arg = None
        rel_cov_arg = None
        if args.clone_relative:
            # Re-anchor BEFORE cloning this frame. The MSCEqF filter keeps its own
            # fixed-size clone window and evicts the oldest clone inside clone_pose(),
            # so a clone read AFTER that call may already be gone -- in which case
            # relative_pose_covariance returns None and the landmark is re-anchored as
            # if the two anchors were perfectly known relative to each other. Move the
            # oldest clone's landmarks onto the newest LIVE clone while both still exist.
            if args.reanchor and args.reanchor_eviction:
                # The shipping rule (CloneRelFeed): the backend evicted a clone inside
                # this frame's process_vision and recorded, BEFORE dropping it, the
                # old->newest relative covariance and the newest pose.
                _ev = vio.last_clone_eviction()
                if _ev is not None:
                    _old, _new, _T, _pv, _pw = _ev
                    if os.environ.get("DBG_EV"):
                        _dT = np.abs(np.asarray(_T, float) - T_wc).max()
                        _na = sum(1 for fd in sparse.get_features().values()
                                  if fd["anchor_clone_id"] == _old)
                        print(f"  DBG_EV k={k} old={_old} new={_new} |T_ev-T_wc|max={_dT:.2e} "
                              f"live={vio.clone_ids()[0]}..{vio.clone_ids()[-1]} "
                              f"n_on_old={_na}", file=sys.stderr)
                    _cov = (np.asarray(_pv, float).tolist(), np.asarray(_pw, float).tolist())
                    if args.reanchor_cov_scale != 1.0:
                        _cov = ((np.asarray(_cov[0]) * args.reanchor_cov_scale).tolist(),
                                (np.asarray(_cov[1]) * args.reanchor_cov_scale).tolist())
                    if args.reanchor_nocov:
                        _cov = None
                    _dbg_reanchor[0 if _cov is None else 1] += 1
                    n_reanchored[0] += sparse.reanchor_clone(
                        int(_old), np.asarray(_T, float).tolist(), int(_new), _cov)
                    clone_rel_prev.pop(int(_old), None)
            elif args.reanchor:
                # The filter's clone list and the harness's pose book can disagree
                # (the filter may hold clones the harness never registered), so only
                # use ids present in BOTH -- we need the pose to re-anchor onto.
                _live = [int(x) for x in vio.clone_ids() if int(x) in clone_pose]
                if len(_live) >= 2 and len(vio.clone_ids()) >= vio.n_clones():
                    _old, _new = _live[0], _live[-1]
                    _rc = vio.get_relative_pose_covariance(
                        _old, np.ascontiguousarray(clone_pose[_new]))
                    _cov = None
                    if _rc is not None:
                        _cov = (np.asarray(_rc[0], float).tolist(),
                                np.asarray(_rc[1], float).tolist())
                    if _cov is not None and args.reanchor_cov_scale != 1.0:
                        _cov = ((np.asarray(_cov[0]) * args.reanchor_cov_scale).tolist(),
                                (np.asarray(_cov[1]) * args.reanchor_cov_scale).tolist())
                    if args.reanchor_nocov:
                        _cov = None
                    _dbg_reanchor[0 if _cov is None else 1] += 1
                    n_reanchored[0] += sparse.reanchor_clone(
                        _old, clone_pose[_new].tolist(), _new, _cov)

            vio.clone_pose(int(k), stamp)
            clone_pose[int(k)] = T_wc.copy()

            for cid in list(clone_pose):
                if cid < k - args.clone_window:
                    # Re-anchoring already happened above, while the clone was still
                    # live in the filter's own window.
                    vio.marginalize_clone(int(cid))
                    clone_pose.pop(cid, None)
                    clone_rel_prev.pop(cid, None)
            rel_cov_arg = {}
            for cid in vio.clone_ids():
                cid = int(cid)
                Tc = clone_pose.get(cid)
                if Tc is None:
                    continue
                rc = vio.get_relative_pose_covariance(cid, np.ascontiguousarray(Tc))
                if rc is None:
                    continue
                if args.xcov_audit:
                    # Eq (11) adds the rotation and baseline range-contributions IN
                    # QUADRATURE, i.e. assumes Sigma_{phi,t} = 0. That block is computed
                    # and then dropped by relative_pose_covariance. Measure (a) its
                    # correlation strength and (b) what it does to the combined range
                    # variance along the directions that actually matter:
                    #   s_t = (r/b) * t_hat     (baseline-length error scales r)
                    #   s_phi = (r^2/b) * n_hat (parallax-angle error, n perp to the
                    #                            anchor-current epipolar plane)
                    _C = vio.relative_pose_covariance_full(int(cid), int(k))
                    if _C is not None:
                        _C = np.asarray(_C, float)
                        _W = _C[:3, :3]; _T = _C[3:, 3:]; _X = _C[:3, 3:]
                        _dw = np.sqrt(np.trace(_W) / 3.0)
                        _dt = np.sqrt(np.trace(_T) / 3.0)
                        _rho = (np.abs(_X).max() / max(_dw * _dt, 1e-18))
                        _tb = (np.linalg.inv(Tc) @ T_wc)[:3, 3]
                        _b = np.linalg.norm(_tb)
                        if _b > 1e-6:
                            _that = _tb / _b
                            _r = 20.0          # representative scene range, scale-free below
                            _st = (_r / _b) * _that
                            # rotation axis that most changes parallax: perpendicular to
                            # the baseline, pick the dominant eigenvector of W restricted
                            # to that plane -- approximate with any unit vector perp to t.
                            _tmp = np.array([0.0, 0.0, 1.0])
                            if abs(_that @ _tmp) > 0.9:
                                _tmp = np.array([1.0, 0.0, 0.0])
                            _n = np.cross(_that, _tmp); _n /= np.linalg.norm(_n)
                            _sp = (_r * _r / _b) * _n
                            _s = np.concatenate([_sp, _st])
                            _v_full = float(_s @ _C @ _s)
                            _v_quad = float(_sp @ _W @ _sp + _st @ _T @ _st)
                            if _v_quad > 0 and np.isfinite(_v_full):
                                xcov_audit.append((_rho, _v_full / _v_quad, _b))

                if args.pose_cov_audit:
                    # Does the REPORTED anchor->current translation covariance match the
                    # ACTUAL error against GT? This is upstream of Sparse3D entirely: if
                    # P_vv is N x too small here, the depth filter is being fed a wrong
                    # number and no amount of work inside Sparse3D can fix it.
                    _tf = np.linalg.inv(Tc) @ T_wc               # filter anchor->cur
                    _gc = ds.pose(cid) @ ext
                    _tg = np.linalg.inv(_gc) @ (ds.pose(k) @ ext)  # GT anchor->cur
                    _e = (np.linalg.inv(_tg) @ _tf)[:3, 3]       # translation discrepancy
                    _P = np.asarray(rc[0], float)
                    try:
                        _n = float(_e @ np.linalg.solve(_P, _e))
                        # DIRECTIONAL split. The recorded fork-b defect is a SCALE-channel
                        # over-confidence ALONG the velocity/baseline direction, shared
                        # with the reference MSCEqF C++ (i.e. intrinsic to SE_2(3), not a
                        # windowing problem). If the relative-pose over-confidence is
                        # along-baseline it is that same defect; if it is isotropic it is
                        # something else.
                        _bv = _tf[:3, 3]
                        _bn = float(np.linalg.norm(_bv))
                        _na = _np_ = np.nan
                        if _bn > 1e-6:
                            _u = _bv / _bn
                            _ea = float(_u @ _e)
                            _va = float(_u @ _P @ _u)
                            if _va > 0:
                                _na = _ea * _ea / _va
                            # one perpendicular direction
                            _tmp = np.array([0.0, 0.0, 1.0])
                            if abs(_u @ _tmp) > 0.9:
                                _tmp = np.array([1.0, 0.0, 0.0])
                            _w = np.cross(_u, _tmp); _w /= np.linalg.norm(_w)
                            _ep = float(_w @ _e); _vp = float(_w @ _P @ _w)
                            if _vp > 0:
                                _np_ = _ep * _ep / _vp
                        if np.isfinite(_n):
                            pose_audit.append((k - cid, _n, float(np.linalg.norm(_e)),
                                               float(np.sqrt(np.trace(_P) / 3.0)),
                                               _na, _np_))
                    except np.linalg.LinAlgError:
                        pass
                pvv_rel = np.asarray(rc[0], float) * args.pvv_scale
                pww_rel = np.asarray(rc[1], float)
                prev = clone_rel_prev.get(cid)
                if prev is None:
                    dv = np.zeros((3, 3))
                    dw = np.zeros((3, 3))
                else:
                    dv = psd_clip(pvv_rel - prev[0])
                    dw = psd_clip(pww_rel - prev[1])
                clone_rel_prev[cid] = (pvv_rel, pww_rel)
                # (full_v, full_w, inc_v, inc_w): full → §V-D measurement term,
                # increment → accumulating range injection.
                rel_cov_arg[cid] = (pvv_rel.tolist(), pww_rel.tolist(),
                                    dv.tolist(), dw.tolist())
            clone_id_arg = int(k)

        if os.environ.get("DBG_COV") and n_frames % 40 == 0:
            tv = np.trace(np.asarray(pvv_arg)) if pvv_arg is not None else float("nan")
            tw = np.trace(np.asarray(pww_arg)) if pww_arg is not None else float("nan")
            nrel = len(rel_cov_arg) if rel_cov_arg else 0
            rv = (np.trace(np.asarray(list(rel_cov_arg.values())[0][0]))
                  if nrel else float("nan"))
            print(f"  DBG k={k} n_uvs={len(all_uvs)} tr(pvv)={tv:.3e} tr(pww)={tw:.3e} "
                  f"n_rel={nrel} tr(rel_vv[0])={rv:.3e}", file=sys.stderr)
        sparse.update(stamp, all_uvs, T_wc.tolist(), pvv_arg, pww_arg,
                      clone_id_arg, rel_cov_arg)

        if args.anchor_census and n_frames % 25 == 0:
            _sf = sparse.get_features()
            _anch = [fd["anchor_clone_id"] for fd in _sf.values()
                     if fd["anchor_clone_id"] is not None]
            _d = sorted(set(int(a) for a in _anch))
            if _d:
                # Orphans: landmarks whose anchor clone the pose source no longer holds.
                # They get NO relative covariance this frame and fall back to the
                # absolute pair -- the "bigger error bars flatter NEES" trap.
                _livec = {int(c) for c in vio.clone_ids()} if args.clone_relative else set()
                _orph = sum(1 for a in _anch if int(a) not in _livec)
                anchor_census.append((k, vio.n_clones() if args.clone_relative else 0,
                                      len(_sf), len(_d), k - min(_d), _orph))

        if args.clone_relative and n_frames % 40 == 0:
            sf = sparse.get_features()
            tls = [int(fd["track_length"]) for fd in sf.values()]
            maxtl = max(tls) if tls else 0
            n_ge = sum(t >= args.min_track for t in tls)
            # Nav position error vs GT (NWU world), to localize the nav pose the
            # clone-relative covariance is read against.
            gt_k = ds.pose(k)
            p_gt = t_ned_to_nwu @ gt_k[:3, 3]
            p_est = vio_body_pose(vio)[:3, 3]
            nav_err = float(np.linalg.norm(p_est - p_gt))
            print(f"  [clone k={k}] n_clones={vio.n_clones()} feats={len(sf)} "
                  f"maxtl={maxtl} n(tl>={args.min_track})={n_ge} "
                  f"navErr={nav_err:.3f}m", file=sys.stderr)

        # Score Sparse3D features against GT depth.
        depth_map = ds.depth(k)
        for fid, fd in sparse.get_features().items():
            tl = int(fd["track_length"])
            if tl < args.min_track:
                continue
            # get_features() applies NO convergence gates. --converged-only restricts
            # scoring to features query_range would actually hand to a consumer, which
            # matters once undelayed birth fills the pool with un-converged landmarks.
            if args.converged_only and sparse.query_range(int(fid))[0] <= 0.0:
                continue
            est_cam = np.asarray(fd["position"], float)
            cov_cam = np.asarray(fd["covariance_euclidean"], float)
            re = float(np.linalg.norm(est_cam))
            if re <= 1e-6 or not np.isfinite(cov_cam).all():
                continue
            rhat = est_cam / re
            var_r = float(rhat @ cov_cam @ rhat)
            if var_r <= 0:
                continue
            # Honest exogenous scale channel (validated, sourced from E not p_vv):
            # var_r_scored = var_r_filter + (sigma_s * r)^2. Composes with the
            # clone-relative attitude cov to test plan Verification #3.
            if args.sigma_s > 0.0:
                # Legacy scoring-time injection. Now that the term lives in the filter
                # (settings.scale_sigma), prefer --scale-sigma so the SHIPPED quantity is
                # what gets scored; this path stays for reproducing the 2026-08 numbers.
                var_r += (args.sigma_s * re) ** 2
            if args.score_reported:
                # Score what the filter reports, including its own scale term.
                _rv = float(fd["range_var_reported"])
                if _rv > 0:
                    var_r = _rv

            # Read GT depth at the tracked pixel.
            if fid in all_uvs:
                px, py = all_uvs[fid]
            elif est_cam[2] > 0.1:
                px = f * est_cam[0] / est_cam[2] + cx
                py = f * est_cam[1] / est_cam[2] + cy
            else:
                continue
            gx, gy = int(round(px)), int(round(py))
            if not (R_MARGIN < gx < W - R_MARGIN and R_MARGIN < gy < H - R_MARGIN):
                continue
            gt_range = float(depth_map[gy, gx])
            if not (0.5 < gt_range < md.SKY):
                continue

            # --gt-anchored: score against the BIRTH landmark, not the current pixel.
            # The default reads GT depth wherever the tracked pixel now sits, so as
            # correspondence drifts the GT target slides with it while the filter's
            # estimate stays anchored to the point it was born on. That mismatch grows
            # with track length and would masquerade as a filter defect. Here the birth
            # landmark is fixed in the world from GT depth + GT pose at first sight and
            # its TRUE current range is recomputed from GT pose each frame.
            if args.gt_anchored:
                gt_cam = ds.pose(k) @ ext          # GT T_wc (body pose * cam extrinsic)
                if fid not in gt_anchor:
                    b = np.array([(px - cx) / f, (py - cy) / f, 1.0])
                    b /= np.linalg.norm(b)          # MidAir depth is RADIAL range
                    gt_anchor[fid] = gt_cam[:3, :3] @ (gt_range * b) + gt_cam[:3, 3]
                    continue
                gt_range = float(np.linalg.norm(
                    gt_cam[:3, :3].T @ (gt_anchor[fid] - gt_cam[:3, 3])))
                if not (0.5 < gt_range < md.SKY):
                    continue

            # sigma_rho/rho governs whether the delta-method push-forward used for
            # `covariance_euclidean` is even valid: r = 1/rho is only approximately
            # Gaussian when this is << 1. Collected always, it is the precondition
            # check for scoring in RANGE at all.
            _isv = np.asarray(fd["inv_s"], float)
            _ipv = np.asarray(fd["inv_p"], float)
            _rho_e = float(_isv[2])
            _vrho = float(_ipv[2, 2])
            if _rho_e > 0.0 and _vrho > 0.0:
                srr.append(np.sqrt(_vrho) / _rho_e)

            # Both spaces are computed from the SAME observation so one run reports
            # both: metric range (what a mapper consumes) and the anchor chart (where
            # the EKF Gaussian lives, hence the self-consistent consistency test).
            nees_rho = np.nan
            if True:
                # Score in the ANCHOR CHART, where the EKF's Gaussian actually lives.
                # The filter's claim is "inverse range along my anchor ray is rho +- s";
                # test THAT, not its linearized image in metric range.
                if _rho_e <= 0.0 or _vrho <= 0.0:
                    pass
                else:
                # Stay in the FILTER's frames. Going via the GT world pose would put
                # accumulated global pose drift (metres) into rho_gt and measure nav
                # error, not depth error -- the range test is drift-free precisely
                # because both sides live in the current camera frame. So build the GT
                # point in the CURRENT camera frame (no pose at all) and carry it back
                # through the filter's own anchor->current transform, which spans only
                # the clone window.
                    b_c = np.array([(px - cx) / f, (py - cy) / f, 1.0])
                    b_c /= np.linalg.norm(b_c)
                    q_cur_gt = gt_range * b_c
                    A = np.asarray(fd["anchor_t_wc"], float)
                    t_ac = np.linalg.inv(A) @ T_wc          # cur -> anchor
                    p_a = t_ac[:3, :3] @ q_cur_gt + t_ac[:3, 3]
                    r_a = float(np.linalg.norm(p_a))
                    if 0.5 < r_a < md.SKY:
                        rho_gt = 1.0 / r_a
                        nees_rho = (_rho_e - rho_gt) ** 2 / _vrho

            nees_r = (re - gt_range) ** 2 / var_r
            if args.nees_space == "invdepth":
                if not np.isfinite(nees_rho):
                    continue
                rows.append((k, tl, _rho_e - rho_gt, rho_gt, nees_rho,
                             np.sqrt(_vrho), _rho_e, nees_r))
            else:
                rows.append((k, tl, re - gt_range, gt_range, nees_r,
                             np.sqrt(var_r), re, nees_rho))

        n_frames += 1
        if not args.no_progress and n_frames % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_frames}/{nimg}] {n_frames / elapsed:.0f} fps, "
                  f"{len(rows)} scored obs")

    # --- Clean up temp file ---
    os.unlink(merged_config_path)

    # --- Report ---
    A = np.array(rows, float) if rows else np.empty((0, 7))
    if len(A) == 0:
        print("No scored observations!")
        return

    _k, tl, derr, rng, nees, sig, r_est, nees_alt = A.T
    relerr = derr / rng

    if xcov_audit:
        _x = np.array(xcov_audit, dtype=float)
        print(f"\n  CROSS-BLOCK AUDIT (Sigma_phi,t -- dropped by Eq 11's quadrature):")
        print(f"    |corr| max-element: median {np.median(_x[:, 0]):.3f}  "
              f"p90 {np.percentile(_x[:, 0], 90):.3f}")
        _r = _x[:, 1]
        print(f"    v_full / v_quadrature: median {np.median(_r):.3f}  "
              f"p10 {np.percentile(_r, 10):.3f}  p90 {np.percentile(_r, 90):.3f}")
        print(f"    => cross-term {'REDUCES' if np.median(_r) < 1 else 'RAISES'} the "
              f"combined range variance by x{1 / np.median(_r) if np.median(_r) < 1 else np.median(_r):.2f}")
        print(f"    baseline median {np.median(_x[:, 2]):.2f} m")
    if pose_audit:
        _pa = np.array(pose_audit, dtype=float)
        _n = _pa[:, 1]
        print(f"\n  POSE-COV AUDIT: reported anchor->cur translation cov vs GT error")
        print(f"    3-dof NEES median {np.median(_n):.2f}  mean {np.mean(_n):.1f}   "
              f"(ideal median 2.366, mean 3.0)")
        print(f"    => variance under-scale x{np.median(_n) / 2.366:.1f}  "
              f"(sigma x{np.sqrt(np.median(_n) / 2.366):.1f})")
        _al = _pa[:, 4][np.isfinite(_pa[:, 4])]
        _pe = _pa[:, 5][np.isfinite(_pa[:, 5])]
        if len(_al) and len(_pe):
            print(f"    1-DOF along-baseline NEES median {np.median(_al):.2f}  "
                  f"PERP median {np.median(_pe):.2f}   (ideal 0.455 each)")
            print(f"    => along/perp ratio {np.median(_al) / max(np.median(_pe), 1e-9):.1f}"
                  f"  (>>1 = the known SCALE-channel defect; ~1 = isotropic)")
        print(f"    |err| median {np.median(_pa[:, 2]):.4f} m   "
              f"reported sigma median {np.median(_pa[:, 3]):.4f} m   "
              f"ratio {np.median(_pa[:, 2]) / max(np.median(_pa[:, 3]), 1e-9):.1f}")
    if srr:
        _sr = np.array(srr)
        print(f"\n  sigma_rho/rho (delta-method validity; <<1 required to score in RANGE):")
        print(f"    median {np.median(_sr):.3f}  p90 {np.percentile(_sr, 90):.3f}  "
              f"p99 {np.percentile(_sr, 99):.3f}  %>0.3 {100 * (_sr > 0.3).mean():.1f}%")
    if n_reanchored[0]:
        print(f"  reanchor cov: {_dbg_reanchor[1]} calls WITH cov, "
              f"{_dbg_reanchor[0]} without (scale={args.reanchor_cov_scale})")
        print(f"\n  re-anchored {n_reanchored[0]} landmark-events "
              f"(clone window {args.clone_window})")
    if anchor_census:
        _c = np.array(anchor_census, dtype=float)
        print(f"\n  ANCHOR CENSUS -- distinct anchor clones actually live:")
        print(f"    distinct anchors : median {np.median(_c[:, 3]):.0f}  "
              f"p90 {np.percentile(_c[:, 3], 90):.0f}  max {_c[:, 3].max():.0f}")
        print(f"    oldest anchor age: median {np.median(_c[:, 4]):.0f}  "
              f"p90 {np.percentile(_c[:, 4], 90):.0f}  max {_c[:, 4].max():.0f} frames")
        print(f"    live features    : median {np.median(_c[:, 2]):.0f}")
        print(f"    orphan anchors   : median {np.median(_c[:, 5]):.0f}  "
              f"p90 {np.percentile(_c[:, 5], 90):.0f}  max {_c[:, 5].max():.0f}  "
              f"(landmarks whose anchor clone is not live)")
        print(f"    clones ALLOCATED : {_c[:, 1].max():.0f}")

    print(f"\n{'=' * 72}")
    print(f"END-TO-END depth NEES  traj {args.traj}  "
          f"prs={eff_prs}  γ²={args.pvv_scale}")
    print(f"{'=' * 72}")
    print(f"  {len(A)} scored range obs  ({nimg} frames, {len(A) / nimg:.0f} obs/frame)")
    _alt = nees_alt[np.isfinite(nees_alt)]
    if len(_alt):
        _other = "invdepth" if args.nees_space == "range" else "range"
        print(f"  ALT[{_other}]  NEES median / mean: {np.median(_alt):.3f} / "
              f"{np.mean(_alt):.1f}   %>3.84 {100 * (_alt > 3.84).mean():.1f}%")
    print(f"  range NEES-1D  median / mean: {np.median(nees):.3f} / {np.mean(nees):.1f}"
          f"   (ideal median 0.455, mean 1)")
    print(f"  %>3.84 / %>6.63:  {100 * np.mean(nees > 3.84):.1f}% / "
          f"{100 * np.mean(nees > 6.63):.1f}%   (ideal 5% / 1%)")
    print(f"  signed range err:  median {100 * np.median(relerr):+.1f}%  "
          f"mean {100 * np.mean(relerr):+.1f}%")
    print(f"  abs range err:     median {100 * np.median(np.abs(relerr)):.1f}%  "
          f"p90 {100 * np.percentile(np.abs(relerr), 90):.1f}%")
    # In invdepth space `sig` is sigma_rho (1/m), not metres -- label it correctly
    # instead of printing a metre value that is really an inverse-depth sigma.
    if args.nees_space == "invdepth":
        print(f"  reported σ_rho:    median {np.median(sig):.5f} 1/m "
              f"(= σ_r {np.median(sig / np.maximum(r_est, 1e-9) ** 2):.2f} m at r̂)")
    else:
        print(f"  reported σ_r:      median {np.median(sig):.1f} m")
    print()
    print("  Track-length breakdown:")
    for lo, hi in [(5, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if m.sum() >= 10:
            label = f"∞" if hi > 1e8 else str(int(hi))
            print(f"    tl {lo:>3}–{label:<3}: "
                  f"n={int(m.sum()):5d}  NEES med {np.median(nees[m]):7.3f}  "
                  f"absErr% {100 * np.median(np.abs(relerr[m])):5.1f}  "
                  f"σ_r {np.median(sig[m]):6.1f}")

    if args.dump_traj and traj_est:
        np.savez(args.dump_traj, k=np.array(traj_k), est=np.array(traj_est),
                 gt=np.array(traj_gt))
        print(f"trajectory -> {args.dump_traj}  ({len(traj_est)} frames)")
    if args.save_npz:
        provenance = {
            "config_file": str(Path(args.config).resolve()),
            "sparse3d_settings": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                  for k, v in settings.items()},
            "cli": vars(args),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        np.savez(args.save_npz,
                 cols=np.array(["k", "tl", "derr", "rng", "nees", "sig", "r_est",
                                "nees_alt"]),
                 data=A,
                 provenance=json.dumps(provenance))
        print(f"\nSaved → {args.save_npz}")


def _deep_copy(d):
    """Deep-copy a nested dict/list structure (no numpy)."""
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_copy(v) for v in d]
    return d


if __name__ == "__main__":
    main()
