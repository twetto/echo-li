"""Diagnostic video for the Mid-Air EqVIO run: left = tracker frame (scene + tracked
features), right = top-down trajectory (GT vs SE3-aligned estimate, estimate colored by
per-frame error) with the current error line. Lets us see WHERE the VIO error/scale-drift
comes from (fast rotation? texture-poor? altitude change?). Uses a saved vio_*.npz
(est,gt,R,t,k) and re-tracks the frames for the overlay.

  PY=echo-li-python/venv/bin/python
  $PY midair_vio_video.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --npz vio_traj2_long.npz --out vio_traj2.mp4 --stride 2
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_midair.yaml"))
    ap.add_argument("--out", default="vio_traj2.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    d = np.load(args.npz)
    est, gt, R, t, K = d["est"], d["gt"], d["R"], d["t"], d["k"].astype(int)
    aligned = (R @ est.T).T + t                      # estimate in GT (NED) frame
    err = np.linalg.norm(aligned - gt, axis=1)
    emax = float(np.percentile(err, 98))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)

    xy = np.vstack([gt[:, :2], aligned[:, :2]])
    pad = 0.06 * (xy.max(0) - xy.min(0) + 1e-6)
    lo, hi = xy.min(0) - pad, xy.max(0) + pad

    fig, (axi, axt) = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.06, wspace=0.12)
    writer = None
    idxs = list(range(0, len(K), args.stride))
    for n, i in enumerate(idxs):
        kf = int(K[i])
        img = np.asarray(ds.image(kf))
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.01 else 1), 0, 255).astype(np.uint8)
        feats, stats = tracker.process(img)

        axi.clear(); axi.imshow(img, cmap="gray", vmin=0, vmax=255); axi.axis("off")
        if feats:
            axi.scatter([fd["x"] for fd in feats], [fd["y"] for fd in feats],
                        s=6, facecolors="none", edgecolors="#39ff9a", linewidths=0.7)
        axi.set_title(f"frame {kf}   tracked {len(feats)}", fontsize=10)

        axt.clear()
        axt.plot(gt[:, 0], gt[:, 1], color="0.82", lw=1.0, zorder=1)                 # full GT ref
        axt.plot(gt[:i + 1, 0], gt[:i + 1, 1], color="#2e7d46", lw=1.8, zorder=2, label="GT")
        axt.scatter(aligned[:i + 1, 0], aligned[:i + 1, 1], c=err[:i + 1], cmap="inferno",
                    s=5, vmin=0, vmax=emax, zorder=3)
        axt.plot([gt[i, 0], aligned[i, 0]], [gt[i, 1], aligned[i, 1]], color="red", lw=1.2, zorder=4)
        axt.scatter([gt[i, 0]], [gt[i, 1]], c="#2e7d46", s=45, marker="x", zorder=5)
        axt.scatter([aligned[i, 0]], [aligned[i, 1]], c="red", s=30, zorder=5, label="estimate")
        axt.set_xlim(lo[0], hi[0]); axt.set_ylim(lo[1], hi[1]); axt.set_aspect("equal")
        axt.set_title(f"top-down (North-East, m)   t={kf/25:4.1f}s   err {err[i]:5.1f} m "
                      f"({100*err[i]/(np.linalg.norm(gt[i])+1e-6):.0f}% of |pos|)", fontsize=10)
        axt.tick_params(labelsize=7); axt.grid(alpha=0.25)
        if n == 0:
            axt.legend(loc="upper right", fontsize=8)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)
        if n % 100 == 0:
            print(f"  [{n}/{len(idxs)}] frame {kf} err {err[i]:.1f}m tracked {len(feats)}")
    writer.release()
    print(f"wrote {args.out}  ({len(idxs)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
