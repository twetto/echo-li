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

Expected output (configs/diagnostics_midair_e2e_depth_nees.yaml):

  | traj | prs   | NEES med | %%>χ²₉₅ | %%>χ²₉₉ | σ_r (m) | err med |
  |------|-------|----------|---------|---------|---------|---------|
  | 0    | 0     | 30.7     | 82.4%%  | 74.8%%  | 3.1     | −40%%   |
  | 0    | 0.003 | 0.114    | 5.8%%   | 4.2%%   | 97      | −10%%   |
  | 2    | 0     | 3.58     | 49.2%%  | 42.7%%  | 3.1     | −21%%   |
  | 2    | 0.003 | 0.230    | 15.3%%  | 11.3%%  | 29      | −11%%   |
  | t(2.6) ref |  | 0.609   | 15.9%%  | 9.5%%   |         |         |

For Kite_training cross-trajectory validation (≈90 min), see
midair_measure_imu_noise.py and midair_e2e_depth_nees_batch.py.
"""
import argparse
import json
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
R_MARGIN = 4  # pixels from image border to ignore when reading GT depth


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
    ap.add_argument("--save-npz", default="",
                    help="save raw results + provenance to this .npz path")
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
    if args.vel_acc is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["acc"] = args.vel_acc
    if args.vel_gyr is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["gyr"] = args.vel_gyr
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
    print(f"Sparse3D range knobs: pose_range_scale={eff_prs}, "
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
    vio = echo_li.VIOFilter(merged_config_path, cam)
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    gt0 = ds.pose(args.start)
    v0 = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][args.start * 4])
    R0 = t_ned_to_nwu @ gt0[:3, :3]
    v0_body = R0.T @ (t_ned_to_nwu @ v0)
    vio.set_initial_state(
        (t_ned_to_nwu @ gt0[:3, 3]).tolist(),
        np.ascontiguousarray(R0),
        v0_body.tolist(),
    )

    # --- Sparse3D ---
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

        # Cap observations for EqF (must match config maxFeatures to avoid churn).
        existing = {int(x) for x in vio.get_landmarks().keys()}
        obs_ids = sorted(all_uvs.keys())
        if args.eqf_max_obs > 0 and len(obs_ids) > args.eqf_max_obs:
            keep_exist = [fid for fid in obs_ids if fid in existing]
            keep_new = [fid for fid in obs_ids if fid not in existing]
            obs_ids = (keep_exist + keep_new)[:args.eqf_max_obs]
        vio_uvs = {fid: all_uvs[fid] for fid in obs_ids}
        vio.process_vision(stamp, vio_uvs)

        # Sparse3D update with ALL tracked features.
        T_wc = vio_body_pose(vio) @ ext
        pcov = vio.get_camera_pose_covariance()
        if pcov is not None:
            pvv = np.asarray(pcov[0], float) * args.pvv_scale
            pww = np.asarray(pcov[1], float)
            pvv_arg, pww_arg = pvv.tolist(), pww.tolist()
        else:
            pvv_arg = pww_arg = None
        sparse.update(stamp, all_uvs, T_wc.tolist(), pvv_arg, pww_arg)

        # Score Sparse3D features against GT depth.
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

            derr = re - gt_range
            nees = derr * derr / var_r
            rows.append((k, tl, derr, gt_range, nees, np.sqrt(var_r), re))

        n_frames += 1
        if not args.no_progress and n_frames % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_frames}/{nimg}] {n_frames / elapsed:.0f} fps, "
                  f"{len(rows)} scored obs")

    # --- Clean up temp file ---
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
            label = f"∞" if hi > 1e8 else str(int(hi))
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
