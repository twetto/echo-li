"""End-to-end depth NEES for Sparse3D on TartanAir V2.

Same pipeline as midair_e2e_depth_nees.py: Rudolf-V tracker → EqVIO → Sparse3D
with pose-driven range process noise.  Depth NEES scored against GT depth map.

TartanAir V2 differences from MidAir:
  - Gyro is already body-frame (no spatial→body repair needed).
  - Depth is float32 encoded as RGBA PNG, PLANAR z (not range): verified 2026-10-17 by
    two-frame GT consistency (planar z agrees to 0.1 % median, range disagrees 13-17 %).
    The scorer converts it to range along the pixel ray before comparing. Every number
    this harness printed before 2026-10-17 scored range against z: +12-17 % biased.
  - Camera 640×640, f=320, cx=cy=320, pinhole.
  - IMU noise uniform across trajectories: σ_a≈0.047, σ_g≈0.063.
  - Pose convention: NED world, FRD body (same R_bc as MidAir).

Usage:
  PY=echo-li-python/venv/bin/python
  S=echo-li-python/tests/diagnostics/tartanair_e2e_depth_nees.py
  CFG=configs/diagnostics_tartanair_e2e_depth_nees.yaml
  ROOT=~/18TB/datasets/tartanair_v2/OldScandinavia/Data_easy

  # Baseline — prs=0
  $PY $S --root $ROOT --traj P000 --config $CFG --pose-range-scale 0 --min-track 20
  # Recommended — prs=0.003
  $PY $S --root $ROOT --traj P000 --config $CFG --pose-range-scale 0.003 --min-track 20
  # Known-pose arm (FlowDep-F step 3): GT camera pose, zero pose covariance, so the
  # score isolates the track source + filter from the VIO.
  $PY $S --root $ROOT --traj P000 --config $CFG --pose-source gt --dump-tracks klt.npz
  $PY $S --root $ROOT --traj P000 --config $CFG --pose-source gt --tracks fdf.npz
  # FlowDep-F step-3 arm as run on 2026-10-17 (pool cap lifted, filter posterior fed back
  # as the search bound, converged-only scoring, GT-ambiguity split in the summary):
  $PY $S --root $ROOT --traj P000 --config $CFG --start 610 --frames 110 --pose-source gt \
      --track-source fdf --converged-only --sparse max_pool_size=4000 --fdf-posterior

Track file format (--tracks / --dump-tracks): npz with int32 `frame` (dataset frame
index k), uint64 `id`, float32 `u`, `v` in the SCALED image, one row per observation.
Any track source that writes this (Rudolf-V, FlowDep-F, OpenVINS) is scored the same way.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li

# Body←Camera extrinsic:  CV camera (x-right, y-down, z-fwd) → FRD body (x-fwd, y-right, z-down).
# Same as MidAir — both datasets use FRD body frame.
RT_BC = np.array([[0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], float)

SKY = 500.0  # range cutoff for "sky / too far" depth

SPARSE_KEYS = [
    "max_pool_size", "min_track_length", "conv_variance_threshold",
    "init_depth_var", "init_invdepth_var", "sigma_pixel", "flow_age_rate_px_per_frame",
    "bias_walk_var", "bias_init_var", "uniform_z_max", "uniform_rho_max", "uniform_d_min", "uniform_d_max",
    "a_init", "b_init", "ab_min", "ab_max", "min_inlier_ratio",
    "mahalanobis_reset_chi2", "process_depth_var", "range_walk_var", "pose_range_scale",
    "pose_range_coherent", "conv_range_sigma_frac",
    "min_parallax", "min_cos_sim", "min_depth", "max_depth", "birth_min_flow_px",
    "use_equivariant_output", "iekf_iterations", "rotation_unscented",
]
R_MARGIN = 4


# ---------------------------------------------------------------------------
# TartanAir V2 dataset helper
# ---------------------------------------------------------------------------

class TartanAir:
    """Minimal reader for one TartanAir V2 trajectory (e.g. OldScandinavia/Data_easy/P000)."""

    def __init__(self, root, traj, scale=0.5):
        """
        Parameters
        ----------
        root : str
            Path to the scene/difficulty level, e.g.
            ``~/18TB/datasets/tartanair_v2/OldScandinavia/Data_easy``
        traj : str
            Trajectory name, e.g. ``P000``
        scale : float
            Image scale factor (0.5 → 320×320).
        """
        self.dir = Path(root) / traj
        if not self.dir.exists():
            raise FileNotFoundError(f"Trajectory not found: {self.dir}")
        self.scale = scale

        # Camera pose at 10 Hz: [tx ty tz qx qy qz qw] — NED world, FRD body.
        self.pose_raw = np.loadtxt(self.dir / "pose_lcam_front.txt")
        self.n_cam = len(self.pose_raw)

        # IMU at 100 Hz
        imu = self.dir / "imu"
        self.acc = np.load(imu / "acc.npy")        # specific force, body FRD
        self.gyro = np.load(imu / "gyro.npy")       # angular velocity, body frame
        self.imu_t = np.load(imu / "imu_time.npy")
        self.cam_t = np.loadtxt(imu / "cam_time.txt")
        self.vel_global = np.load(imu / "vel_global.npy")

        # Images
        self._img_dir = self.dir / "image_lcam_front"
        self._depth_dir = self.dir / "depth_lcam_front"

        # Slerp for IMU-rate orientation (used for vel init only)
        quats = self.pose_raw[:, 3:]  # xyzw
        self._slerp = Slerp(self.cam_t[:self.n_cam], Rot.from_quat(quats))

    # --- Pose / velocity ---

    def pose(self, k):
        """4×4 T_wb (body→world) at camera frame k.  NED convention."""
        p = self.pose_raw[k]
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(p[3:]).as_matrix()
        T[:3, 3] = p[:3]
        return T

    def velocity_world(self, k):
        """World-frame velocity at camera frame k (NED).

        Uses vel_global at the IMU sample closest to camera time k.
        """
        t_cam = self.cam_t[k]
        idx = np.argmin(np.abs(self.imu_t - t_cam))
        return self.vel_global[idx]

    # --- Images ---

    def image(self, k):
        """Grayscale image at camera frame k, scaled."""
        fname = sorted(self._img_dir.glob("*.png"))[k]
        img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        return img

    def depth(self, k):
        """Planar-z depth map at camera frame k (float32, full resolution)."""
        fname = sorted(self._depth_dir.glob("*.png"))[k]
        raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)  # (H, W, 4) uint8
        return raw.view(np.float32)[:, :, 0]


def intrinsics(W, H):
    """Pinhole intrinsics for TartanAir V2 at resolution W×H.

    Full-res: 640×640, f=320, cx=cy=320.
    At scale s: f=320*s, cx=cy=320*s.
    """
    # TartanAir V2: 90° FoV → f = W/2
    f = W / 2.0
    cx = W / 2.0
    cy = H / 2.0
    return f, cx, cy


# ---------------------------------------------------------------------------

def vio_body_pose(vio):
    """SE(3) pose of the VIO body frame in the world frame."""
    pos, quat = vio.get_pose()
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
    T[:3, 3] = np.asarray(pos)
    return T


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end depth NEES for Sparse3D on TartanAir V2.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="path to scene/difficulty (e.g. .../OldScandinavia/Data_easy)")
    ap.add_argument("--traj", default="P000",
                    help="trajectory name (e.g. P000)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="image scale (0.5 → 320×320)")
    ap.add_argument("--config", required=True,
                    help="YAML config for VIO + Sparse3D + frontend")
    ap.add_argument("--eqf-max-obs", type=int, default=40)
    ap.add_argument("--min-track", type=int, default=5)
    ap.add_argument("--pvv-scale", type=float, default=1.0)
    ap.add_argument("--pose-range-scale", type=float, default=None)
    ap.add_argument("--pose-range-coherent", type=float, default=None)
    ap.add_argument("--range-walk-var", type=float, default=None)
    ap.add_argument("--scene-depth", type=float, default=None,
                    help="override eqf.initialValue.sceneDepth")
    ap.add_argument("--pose-source", choices=("eqvio", "gt"), default="eqvio",
                    help="eqvio: EqVIO pose + its covariance (the original pipeline); "
                         "gt: dataset camera pose with ZERO pose covariance (known-pose arm)")
    ap.add_argument("--track-source", choices=("rudolf", "fdf"), default="rudolf",
                    help="rudolf: the FAST+KLT front end; fdf: FlowDep-F anchor-referenced "
                         "epipolar patch tracker driven by the --pose-source pose (settings "
                         "from the config's FlowDepF section plus --fdf KEY=VAL)")
    ap.add_argument("--sparse", action="append", default=[], metavar="KEY=VAL",
                    help="SparseVog setting override, e.g. --sparse max_pool_size=4000")
    ap.add_argument("--fdf", action="append", default=[], metavar="KEY=VAL",
                    help="FlowDepF setting override, e.g. --fdf levels=2 --fdf keyframe_every=5")
    ap.add_argument("--fdf-dump", default="",
                    help="npz of every FlowDep-F observation: frame, id, u, v, level, eta, "
                         "eta_sigma, sigma_px, dir_x, dir_y, residual")
    ap.add_argument("--fdf-aniso-r", type=float, default=0.0, metavar="SIGMA_ACROSS_PX",
                    help="step 4: per-observation 2x2 R for FlowDep-F tracks: along the epipolar "
                         "line the tracker's Gauss-Newton sigma (floored by --fdf-along-floor), "
                         "across it this value (px). 0 = isotropic sigma_pixel (default)")
    ap.add_argument("--fdf-along-floor", type=float, default=0.2,
                    help="floor (px) on the along-line sigma used with --fdf-aniso-r")
    ap.add_argument("--fdf-along-residual", default="", metavar="A,B",
                    help="with --fdf-aniso-r: along sigma = A * exp(residual / B) px instead of the "
                         "Hessian sigma (empirical: on P000 the median |along error| vs the mean abs "
                         "photometric residual is 0.16 px at 2, 0.3 at 10, 1.1 at 22, 1.9 at 28; "
                         "A=0.21,B=10.6 fits sigma = 1.48 * median)")
    ap.add_argument("--fdf-posterior", action="store_true",
                    help="feed Sparse3D's converged range posteriors back as FlowDep-F search bounds")
    ap.add_argument("--tracks", default="",
                    help="npz of external per-frame tracks (frame, id, u, v) replacing the "
                         "Rudolf-V tracker as the Sparse3D track source")
    ap.add_argument("--dump-tracks", default="",
                    help="write the track source actually used, in the --tracks format")
    ap.add_argument("--converged-only", action="store_true",
                    help="score only features passing query_range's convergence gates "
                         "(what a consumer would receive), as midair_e2e_depth_nees.py does")
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    # --- Dataset ---
    ds = TartanAir(args.root, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = intrinsics(W, H)
    f_full, cx_full, cy_full = intrinsics(int(round(W / ds.scale)), int(round(H / ds.scale)))
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    ext = RT_BC.copy()
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])

    # --- Config ---
    cfg_raw = yaml.safe_load(open(args.config)) or {}
    cfg = _deep_copy(cfg_raw)
    if args.scene_depth is not None:
        cfg.setdefault("eqf", {}).setdefault("initialValue", {})["sceneDepth"] = args.scene_depth

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        merged_config_path = tf.name

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
    for kv in args.sparse:
        k, v = kv.split("=", 1)
        if k not in SPARSE_KEYS:
            sys.exit(f"--sparse: unknown key {k}")
        cur = sparse_cfg.get(k)
        settings[k] = type(cur)(v) if isinstance(cur, (int, float, bool)) and not isinstance(cur, bool) else (
            v.lower() == "true" if isinstance(cur, bool) else float(v))
        print(f"SparseVog override {k} = {settings[k]!r}")

    eff_prs = settings.get("pose_range_scale", 0.0)
    eff_prc = settings.get("pose_range_coherent", 0.0)
    eff_rwv = settings.get("range_walk_var", 0.0)
    eff_pdv = settings.get("process_depth_var", 0.0)
    print(f"Config: {args.config}")
    print(f"Sparse3D range knobs: pose_range_scale={eff_prs}, "
          f"pose_range_coherent={eff_prc}, range_walk_var={eff_rwv}, "
          f"process_depth_var={eff_pdv}")

    # --- Frontend (Rudolf-V tracker) ---
    fcfg = echo_li.FrontendConfig.from_yaml(merged_config_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)

    # --- VIO ---
    vio = echo_li.VIOFilter(merged_config_path, cam)
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))

    # Initial state from GT.  TartanAir gyro is body-frame, so the VIO can
    # auto-initialise, but seeding from GT avoids the startup transient.
    gt0 = ds.pose(args.start)
    v0_ned = ds.velocity_world(args.start)
    R0 = t_ned_to_nwu @ gt0[:3, :3]       # body(FRD) → world(NWU)
    v0_body = R0.T @ (t_ned_to_nwu @ v0_ned)
    vio.set_initial_state(
        (t_ned_to_nwu @ gt0[:3, 3]).tolist(),
        np.ascontiguousarray(R0),
        v0_body.tolist(),
    )

    # --- Sparse3D ---
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)

    # --- Build interleaved event list (IMU @ 100 Hz, camera @ 10 Hz) ---
    nimg = min(args.frames, ds.n_cam - args.start)
    cam_dt = 0.1   # 10 Hz
    imu_dt = 0.01  # 100 Hz

    # IMU sample range covering the camera frames
    t_start = ds.cam_t[args.start]
    t_end = ds.cam_t[min(args.start + nimg, ds.n_cam - 1)]
    imu_mask = (ds.imu_t >= t_start - imu_dt) & (ds.imu_t <= t_end + imu_dt)
    imu_indices = np.where(imu_mask)[0]

    events = []
    for i in imu_indices:
        events.append((float(ds.imu_t[i]), "imu", i))
    for n in range(nimg):
        k = args.start + n
        events.append((float(ds.cam_t[k]), "cam", k))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    fdf = None
    fdf_acc = {}
    fdf_dump_rows = []
    if args.track_source == "fdf":
        fdf_settings = {k: float(v) for k, v in (cfg.get("FlowDepF", {}) or {}).items()}
        for kv in args.fdf:
            k, v = kv.split("=", 1)
            fdf_settings[k] = float(v)
        fdf = echo_li.FlowDepF(f, f, cx, cy, W, H, fdf_settings or None)
        print(f"track source: FlowDep-F  settings {fdf_settings}  posterior feedback {args.fdf_posterior}")
    ext_tracks = None
    if args.tracks:
        z = np.load(args.tracks)
        ext_tracks = {}
        for fr, fid, u, v in zip(z["frame"], z["id"], z["u"], z["v"]):
            ext_tracks.setdefault(int(fr), {})[int(fid)] = (float(u), float(v))
        print(f"external tracks: {args.tracks}  {len(z['id'])} obs over {len(ext_tracks)} frames")
    dump = {"frame": [], "id": [], "u": [], "v": []}
    T_nwu = np.eye(4)
    T_nwu[:3, :3] = t_ned_to_nwu
    if args.pose_source == "gt":
        print("pose source: dataset GT camera pose, pose covariance 0 (known-pose arm)")

    # --- Main loop ---
    rows = []
    t0 = time.time()
    n_frames = 0

    for stamp, et, data in events:
        if et == "imu":
            i = data
            # TartanAir gyro is already body-frame — no repair needed.
            gyr = ds.gyro[i].tolist()
            acc = ds.acc[i].tolist()
            vio.process_imu(stamp, gyr, acc)
            continue

        # Camera event.
        k = data
        gray = ds.image(k)

        # Track source: Rudolf-V, the external file, or FlowDep-F.
        if ext_tracks is not None:
            all_uvs = ext_tracks.get(k, {})
        elif fdf is not None:
            # Pose for the tracker: the same source Sparse3D will get; for eqvio that is
            # the IMU-propagated pose before this frame's update.
            T_track = T_nwu @ ds.pose(k) @ ext if args.pose_source == "gt" else vio_body_pose(vio) @ ext
            all_uvs = fdf.process(np.ascontiguousarray(gray), T_track.tolist())
            fdf_level = {o[0]: o[3] for o in fdf.last_observations()}
            if args.fdf_aniso_r > 0:
                if args.fdf_along_residual:
                    _A, _B = (float(x) for x in args.fdf_along_residual.split(","))
                    _along = lambda o: _A * np.exp(o[9] / _B)
                else:
                    _along = lambda o: o[6]
                sparse.set_observation_noise({o[0]: (max(_along(o), args.fdf_along_floor), o[7], o[8], args.fdf_aniso_r)
                                              for o in fdf.last_observations()})
            if args.fdf_dump:
                for o in fdf.last_observations():
                    fdf_dump_rows.append((k,) + tuple(o[:10]) + tuple(o[10]))
            for kk, vv in fdf.stats().items():
                fdf_acc[kk] = fdf_acc.get(kk, 0.0) + vv
        else:
            feats, _stats = tracker.process(gray)
            all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
        if args.dump_tracks:
            for fid, (u, v) in all_uvs.items():
                dump["frame"].append(k); dump["id"].append(fid); dump["u"].append(u); dump["v"].append(v)

        # Cap observations for EqF.
        existing = {int(x) for x in vio.get_landmarks().keys()}
        obs_ids = sorted(all_uvs.keys())
        if args.eqf_max_obs > 0 and len(obs_ids) > args.eqf_max_obs:
            keep_exist = [fid for fid in obs_ids if fid in existing]
            keep_new = [fid for fid in obs_ids if fid not in existing]
            obs_ids = (keep_exist + keep_new)[:args.eqf_max_obs]
        vio_uvs = {fid: all_uvs[fid] for fid in obs_ids}
        vio.process_vision(stamp, vio_uvs)

        # Sparse3D update.
        if args.pose_source == "gt":
            # Same world frame the VIO is initialised in (NWU), camera pose from the
            # dataset; the VIO above still runs so the two arms see identical events.
            T_wc = T_nwu @ ds.pose(k) @ ext
            pcov = None
        else:
            T_wc = vio_body_pose(vio) @ ext
            pcov = vio.get_camera_pose_covariance()
        if pcov is not None:
            pvv = np.asarray(pcov[0], float) * args.pvv_scale
            pww = np.asarray(pcov[1], float)
            pvv_arg, pww_arg = pvv.tolist(), pww.tolist()
        else:
            pvv_arg = pww_arg = None
        sparse.update(stamp, all_uvs, T_wc.tolist(), pvv_arg, pww_arg)
        if fdf is not None and args.fdf_posterior:
            # Every tracked feature's current range posterior (converged or not): the
            # tracker's search bound must stay as wide as the filter is unsure.
            post = {}
            for fid, fd in sparse.get_features().items():
                est = np.asarray(fd["position"], float)
                cov = np.asarray(fd["covariance_euclidean"], float)
                rr = float(np.linalg.norm(est))
                if rr <= 1e-6 or not np.isfinite(cov).all():
                    continue
                rh = est / rr
                vr = float(rh @ cov @ rh)
                if vr > 0:
                    post[int(fid)] = (rr, float(np.sqrt(vr)))
            if post:
                fdf.set_posterior(post)

        # Score against GT depth.
        depth_map = ds.depth(k)
        for fid, fd in sparse.get_features().items():
            tl = int(fd["track_length"])
            if tl < args.min_track:
                continue
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

            # GT range at the tracked pixel.
            if fid in all_uvs:
                px, py = all_uvs[fid]
            elif est_cam[2] > 0.1:
                px = f * est_cam[0] / est_cam[2] + cx
                py = f * est_cam[1] / est_cam[2] + cy
            else:
                continue
            # Pixel in SCALED image; depth map is FULL resolution.
            gx_full = int(round(px / ds.scale))
            gy_full = int(round(py / ds.scale))
            full_H, full_W = depth_map.shape[:2]
            if not (R_MARGIN < gx_full < full_W - R_MARGIN and
                    R_MARGIN < gy_full < full_H - R_MARGIN):
                continue
            # Stored value is planar z; the estimate is range along the ray.
            gt_z = float(depth_map[gy_full, gx_full])
            ray_norm = float(np.sqrt(((gx_full - cx_full) / f_full) ** 2 +
                                     ((gy_full - cy_full) / f_full) ** 2 + 1.0))
            gt_range = gt_z * ray_norm
            if not (0.5 < gt_range < SKY):
                continue

            # GT ambiguity at the scored pixel: relative spread of GT range over the 5x5
            # full-res neighbourhood. A pixel on a leaf/sky boundary has no single truth;
            # the tail is reported with and without such pixels, for every track source.
            win = depth_map[gy_full - 2:gy_full + 3, gx_full - 2:gx_full + 3]
            win = win[(win > 0.5) & (win < SKY)]
            gt_spread = float((win.max() - win.min()) / np.median(win)) if win.size else 9.0
            derr = re - gt_range
            nees = derr * derr / var_r
            rows.append((k, tl, derr, gt_range, nees, np.sqrt(var_r), re, int(fid),
                         fdf_level.get(int(fid), -1) if fdf is not None else -1, gt_spread))

        n_frames += 1
        if not args.no_progress and n_frames % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_frames}/{nimg}] {n_frames / elapsed:.0f} fps, "
                  f"{len(rows)} scored obs")

    # --- Clean up ---
    import os
    os.unlink(merged_config_path)

    # --- Report ---
    A = np.array(rows, float) if rows else np.empty((0, 10))
    if len(A) == 0:
        print("No scored observations!")
        return

    if fdf is not None and args.fdf_dump:
        D = np.array(fdf_dump_rows, float)
        np.savez(args.fdf_dump, data=D, cols=np.array(["frame", "id", "u", "v", "level", "eta", "eta_sigma",
                                                        "sigma_px", "dir_x", "dir_y", "residual",
                                                        "anchor_u", "anchor_v", "anchor_frame"]))
        print(f"FlowDep-F observations written: {args.fdf_dump} ({len(D)} rows)")
    if fdf is not None and n_frames:
        print("FlowDep-F per-frame means: " + "  ".join(f"{k} {v / n_frames:.1f}" for k, v in sorted(fdf_acc.items())))
    if args.dump_tracks:
        np.savez(args.dump_tracks, frame=np.array(dump["frame"], np.int32),
                 id=np.array(dump["id"], np.uint64), u=np.array(dump["u"], np.float32),
                 v=np.array(dump["v"], np.float32))
        print(f"tracks written: {args.dump_tracks}  ({len(dump['id'])} obs)")
    _k, tl, derr, rng, nees, sig, r_est = A.T[:7]
    relerr = derr / rng

    print(f"\n{'=' * 72}")
    print(f"END-TO-END depth NEES  traj {args.traj}  "
          f"prs={eff_prs}  γ²={args.pvv_scale}")
    print(f"{'=' * 72}")
    print(f"  {len(A)} scored range obs  ({nimg} frames, {len(A) / nimg:.0f} obs/frame)")
    print(f"  range NEES-1D  median / mean: {np.median(nees):.3f} / {np.mean(nees):.1f}"
          f"   (ideal median 0.455, mean 1)")
    amb = A[:, 9] > 0.2
    rel_all = np.abs(derr) / rng
    for name, m in (("GT unambiguous (5x5 spread <= 20%)", ~amb), ("GT ambiguous", amb)):
        if m.sum():
            print(f"  {name}: n {m.sum()} ({100 * m.mean():.0f}%)  abs err median {100 * np.median(rel_all[m]):.1f}%  "
                  f"p90 {100 * np.percentile(rel_all[m], 90):.1f}%  NEES median {np.median(nees[m]):.3f}  "
                  f">3.84 {100 * np.mean(nees[m] > 3.84):.1f}%  >6.63 {100 * np.mean(nees[m] > 6.63):.1f}%")
    print(f"  %>3.84 / %>6.63:  {100 * np.mean(nees > 3.84):.1f}% / "
          f"{100 * np.mean(nees > 6.63):.1f}%   (ideal 5% / 1%)")
    print(f"  signed range err:  median {100 * np.median(relerr):+.1f}%  "
          f"mean {100 * np.mean(relerr):+.1f}%")
    print(f"  abs range err:     median {100 * np.median(np.abs(relerr)):.1f}%  "
          f"p90 {100 * np.percentile(np.abs(relerr), 90):.1f}%")
    print(f"  reported σ_r:      median {np.median(sig):.1f} m")
    print()
    print("  Track-length breakdown:")
    for lo, hi in [(5, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if m.sum() >= 10:
            label = "∞" if hi > 1e8 else str(int(hi))
            print(f"    tl {lo:>3}–{label:<3}: "
                  f"n={int(m.sum()):5d}  NEES med {np.median(nees[m]):7.3f}  "
                  f"absErr% {100 * np.median(np.abs(relerr[m])):5.1f}  "
                  f"σ_r {np.median(sig[m]):6.1f}")

    if args.save_npz:
        provenance = {
            "config_file": str(Path(args.config).resolve()),
            "sparse3d_settings": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                  for k, v in settings.items()},
            "cli": vars(args),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        np.savez(args.save_npz,
                 cols=np.array(["k", "tl", "derr", "rng", "nees", "sig", "r_est", "fid", "level", "gt_spread"]),
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
