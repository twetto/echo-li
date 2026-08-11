"""End-to-end depth NEES for Sparse3D on TartanAir V2.

Same pipeline as midair_e2e_depth_nees.py: Rudolf-V tracker → EqVIO → Sparse3D
with pose-driven range process noise.  Depth NEES scored against GT depth map.

TartanAir V2 differences from MidAir:
  - Gyro is already body-frame (no spatial→body repair needed).
  - Depth is float32 encoded as RGBA PNG (euclidean range).
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
    "bias_walk_var", "uniform_z_max", "uniform_rho_max", "uniform_d_min", "uniform_d_max",
    "a_init", "b_init", "ab_min", "ab_max", "min_inlier_ratio",
    "mahalanobis_reset_chi2", "process_depth_var", "range_walk_var", "pose_range_scale",
    "pose_range_coherent",
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
        """Euclidean range map at camera frame k (float32)."""
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
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    # --- Dataset ---
    ds = TartanAir(args.root, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = intrinsics(W, H)
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

        # Track features with Rudolf-V.
        feats, _stats = tracker.process(gray)
        all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}

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
        T_wc = vio_body_pose(vio) @ ext
        pcov = vio.get_camera_pose_covariance()
        if pcov is not None:
            pvv = np.asarray(pcov[0], float) * args.pvv_scale
            pww = np.asarray(pcov[1], float)
            pvv_arg, pww_arg = pvv.tolist(), pww.tolist()
        else:
            pvv_arg = pww_arg = None
        sparse.update(stamp, all_uvs, T_wc.tolist(), pvv_arg, pww_arg)

        # Score against GT depth.
        depth_map = ds.depth(k)
        for fid, fd in sparse.get_features().items():
            tl = int(fd["track_length"])
            if tl < args.min_track:
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
            gt_range = float(depth_map[gy_full, gx_full])
            if not (0.5 < gt_range < SKY):
                continue

            derr = re - gt_range
            nees = derr * derr / var_r
            rows.append((k, tl, derr, gt_range, nees, np.sqrt(var_r), re))

        n_frames += 1
        if not args.no_progress and n_frames % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_frames}/{nimg}] {n_frames / elapsed:.0f} fps, "
                  f"{len(rows)} scored obs")

    # --- Clean up ---
    import os
    os.unlink(merged_config_path)

    # --- Report ---
    A = np.array(rows, float) if rows else np.empty((0, 7))
    if len(A) == 0:
        print("No scored observations!")
        return

    _k, tl, derr, rng, nees, sig, r_est = A.T
    relerr = derr / rng

    print(f"\n{'=' * 72}")
    print(f"END-TO-END depth NEES  traj {args.traj}  "
          f"prs={eff_prs}  γ²={args.pvv_scale}")
    print(f"{'=' * 72}")
    print(f"  {len(A)} scored range obs  ({nimg} frames, {len(A) / nimg:.0f} obs/frame)")
    print(f"  range NEES-1D  median / mean: {np.median(nees):.3f} / {np.mean(nees):.1f}"
          f"   (ideal median 0.455, mean 1)")
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
                 cols=np.array(["k", "tl", "derr", "rng", "nees", "sig", "r_est"]),
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
