"""Live VIO trajectory video for TartanAir V2.

Runs Rudolf-V → EqVIO on a TartanAir trajectory, producing an MP4 with:
  left  — camera image with tracked features
  right — top-down trajectory: GT (blue) vs estimate (red, colored by error)

Usage:
  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/tartanair_vio_video.py \
      --root ~/18TB/datasets/tartanair_v2/OldScandinavia/Data_easy \
      --traj P000 --config configs/diagnostics_tartanair_e2e_depth_nees.yaml \
      --out /tmp/tartanair_P000.mp4 --frames 600
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li

# Re-use TartanAir dataset class from e2e script.
from tartanair_e2e_depth_nees import TartanAir, intrinsics, RT_BC, _deep_copy


def main():
    ap = argparse.ArgumentParser(
        description="Live VIO trajectory video for TartanAir V2.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--traj", default="P000")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene-depth", type=float, default=None)
    ap.add_argument("--eqf-max-obs", type=int, default=40)
    ap.add_argument("--out", default="/tmp/tartanair_vio.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="only render every N-th camera frame to video")
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
    cfg = _deep_copy(yaml.safe_load(open(args.config)) or {})
    if args.scene_depth is not None:
        cfg.setdefault("eqf", {}).setdefault("initialValue", {})["sceneDepth"] = args.scene_depth
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        merged = tf.name

    # --- Frontend ---
    fcfg = echo_li.FrontendConfig.from_yaml(merged)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)

    # --- VIO ---
    vio = echo_li.VIOFilter(merged, cam)
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    gt0 = ds.pose(args.start)
    v0_ned = ds.velocity_world(args.start)
    R0_nwu = t_ned_to_nwu @ gt0[:3, :3]
    v0_body = R0_nwu.T @ (t_ned_to_nwu @ v0_ned)
    vio.set_initial_state(
        (t_ned_to_nwu @ gt0[:3, 3]).tolist(),
        np.ascontiguousarray(R0_nwu),
        v0_body.tolist(),
    )

    # --- Collect GT trajectory (NWU world, for plotting) ---
    nimg = min(args.frames, ds.n_cam - args.start)
    gt_poses = []
    for k in range(args.start, args.start + nimg):
        p = ds.pose(k)
        gt_poses.append(t_ned_to_nwu @ p[:3, 3])
    gt_poses = np.array(gt_poses)

    # Determine plot bounds from full GT.
    xy_gt = gt_poses[:, :2]
    pad = 0.08 * (xy_gt.max(0) - xy_gt.min(0) + 1e-3)
    lo = xy_gt.min(0) - pad
    hi = xy_gt.max(0) + pad

    # --- Build event list ---
    imu_dt = 0.01
    t_start = ds.cam_t[args.start]
    t_end = ds.cam_t[min(args.start + nimg - 1, ds.n_cam - 1)]
    imu_mask = (ds.imu_t >= t_start - imu_dt) & (ds.imu_t <= t_end + imu_dt)
    imu_indices = np.where(imu_mask)[0]

    events = []
    for i in imu_indices:
        events.append((float(ds.imu_t[i]), "imu", i))
    for n in range(nimg):
        k = args.start + n
        events.append((float(ds.cam_t[k]), "cam", k))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    # --- Video setup ---
    fig, (axi, axt) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.06, wspace=0.1)
    fig.canvas.draw()

    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

    est_poses = []
    n_cam = 0
    t0_wall = time.time()

    for stamp, et, data in events:
        if et == "imu":
            gyr = ds.gyro[data].tolist()
            acc = ds.acc[data].tolist()
            vio.process_imu(stamp, gyr, acc)
            continue

        k = data
        gray = ds.image(k)
        feats, _ = tracker.process(gray)
        all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}

        existing = {int(x) for x in vio.get_landmarks().keys()}
        obs_ids = sorted(all_uvs.keys())
        if args.eqf_max_obs > 0 and len(obs_ids) > args.eqf_max_obs:
            keep_exist = [fid for fid in obs_ids if fid in existing]
            keep_new = [fid for fid in obs_ids if fid not in existing]
            obs_ids = (keep_exist + keep_new)[:args.eqf_max_obs]
        vio_uvs = {fid: all_uvs[fid] for fid in obs_ids}
        vio.process_vision(stamp, vio_uvs)

        pos_est, _ = vio.get_pose()
        est_poses.append(np.asarray(pos_est))
        n_cam += 1

        if n_cam % args.stride != 0:
            continue

        # --- Render frame ---
        axi.clear()
        axt.clear()

        # Left: preprocessed image (histeq etc.) with features
        pp = tracker.preprocessed_image()
        disp = pp if pp is not None else gray
        vis = cv2.cvtColor(np.asarray(disp), cv2.COLOR_GRAY2RGB)
        for fid, (u, v) in all_uvs.items():
            color = (0, 255, 0) if fid in existing else (255, 100, 0)
            cv2.circle(vis, (int(u), int(v)), 2, color, -1)
        axi.imshow(vis)
        axi.set_title(f"{args.traj} frame {k}  ({len(all_uvs)} feats)", fontsize=10)
        axi.axis("off")

        # Right: top-down trajectory
        gt_so_far = gt_poses[:n_cam]
        est_arr = np.array(est_poses)

        axt.plot(gt_so_far[:, 0], gt_so_far[:, 1], "b-", lw=1.5, label="GT", alpha=0.6)
        axt.plot(gt_so_far[-1, 0], gt_so_far[-1, 1], "bo", ms=5)

        if len(est_arr) > 1:
            # Color by position error
            errs = np.linalg.norm(est_arr - gt_so_far[:len(est_arr)], axis=1)
            emax = max(np.percentile(errs, 95), 0.5)
            for j in range(1, len(est_arr)):
                c = plt.cm.RdYlGn_r(min(errs[j] / emax, 1.0))
                axt.plot(est_arr[j-1:j+1, 0], est_arr[j-1:j+1, 1], "-", color=c, lw=1.5)
            axt.plot(est_arr[-1, 0], est_arr[-1, 1], "rs", ms=5)

        # Also plot faded full GT for reference
        axt.plot(gt_poses[:, 0], gt_poses[:, 1], "b-", lw=0.3, alpha=0.2)

        axt.set_xlim(lo[0], hi[0])
        axt.set_ylim(lo[1], hi[1])
        axt.set_aspect("equal")
        err_now = np.linalg.norm(est_arr[-1] - gt_so_far[len(est_arr) - 1]) if len(est_arr) > 0 else 0
        traj_len = np.sum(np.linalg.norm(np.diff(gt_so_far, axis=0), axis=1)) if len(gt_so_far) > 1 else 1e-6
        ate_pct = 100 * err_now / max(traj_len, 1e-6)
        axt.set_title(f"err={err_now:.2f}m  ATE={ate_pct:.1f}%  ({len(all_uvs)} feat)", fontsize=10)
        axt.legend(loc="upper right", fontsize=8)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        writer.write(frame)

        if n_cam % 50 == 0:
            elapsed = time.time() - t0_wall
            print(f"  [{n_cam}/{nimg}] {n_cam/elapsed:.0f} fps  err={err_now:.2f}m  ATE={ate_pct:.1f}%")

    writer.release()
    plt.close(fig)
    import os
    os.unlink(merged)

    # Final stats
    est_arr = np.array(est_poses)
    errs = np.linalg.norm(est_arr - gt_poses[:len(est_arr)], axis=1)
    traj_len = np.sum(np.linalg.norm(np.diff(gt_poses[:len(est_arr)], axis=0), axis=1))
    print(f"\n{args.traj}: {nimg} frames, final err={errs[-1]:.2f}m, "
          f"ATE={100*errs[-1]/traj_len:.1f}%, max err={errs.max():.2f}m")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
