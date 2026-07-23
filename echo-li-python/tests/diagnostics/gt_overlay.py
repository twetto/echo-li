"""Independent GT sanity check: reproject the Leica point cloud with the GT pose and
overlay it on the (undistorted) image. If the projected cloud lands ON the image
structures, the GT pose + extrinsics + depth + time-sync are consistent; if it is
shifted, the GT is off -- and by how much / which direction, visibly.

This uses NO tracker and NO feature -- only the raw GT pipeline (Vicon/estimate pose,
T_BS, Leica cloud), so it judges the ground truth itself.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/gt_overlay.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_01_easy [--frames 200 900 1600 2400]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, load_cloud  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default="gt_overlay")
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--sub", type=int, default=6, help="plot every Nth in-view cloud point")
    ap.add_argument("--zoom", type=int, default=0, help="if >0, zoom each overlay into a "
                    "high-gradient window of this half-size (px), upscaled")
    ap.add_argument("--dt", type=float, default=0.0, help="time offset added to image ts [s]")
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = dcoef[:4]

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))

    def cam_pose(t):
        m = np.eye(4)
        m[:3, :3] = slerp(t).as_matrix()
        m[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return m @ t_bs

    cloud = load_cloud(root / "pointcloud0" / "data.ply").astype(np.float64)  # world frame

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    fr_idx = args.frames or list(np.linspace(len(frames) // 10, len(frames) - 5, 4).astype(int))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(fr_idx)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 6.6))
    if n == 1:
        ax = ax.reshape(2, 1)
    for c, fi in enumerate(fr_idx):
        t, p = frames[fi]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        und = cv2.undistort(img, K, D)                      # pinhole overlay
        t_cw = np.linalg.inv(cam_pose(t + args.dt))
        cc = cloud @ t_cw[:3, :3].T + t_cw[:3, 3]           # camera frame
        z = cc[:, 2]
        m = z > 0.1
        u = fx * cc[m, 0] / z[m] + cx
        v = fy * cc[m, 1] / z[m] + cy
        zz = z[m]
        inv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, zz = u[inv][::args.sub], v[inv][::args.sub], zz[inv][::args.sub]
        # optional zoom window: centre on the strongest local gradient (a sharp edge)
        if args.zoom > 0:
            g = np.abs(cv2.Sobel(und, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(und, cv2.CV_32F, 0, 1, 3))
            gb = cv2.boxFilter(g, -1, (2 * args.zoom + 1, 2 * args.zoom + 1))
            gb[:args.zoom] = 0; gb[-args.zoom:] = 0; gb[:, :args.zoom] = 0; gb[:, -args.zoom:] = 0
            yc, xc = np.unravel_index(np.argmax(gb), gb.shape)
            x0, x1, y0, y1 = xc - args.zoom, xc + args.zoom, yc - args.zoom, yc + args.zoom
        else:
            x0, x1, y0, y1 = 0, w, 0, h
        for rr, title in [(0, f"frame {fi}: image"), (1, "image + reprojected Leica cloud")]:
            a = ax[rr, c]
            a.imshow(und, cmap="gray", vmin=0, vmax=255)
            a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=9)
            if rr == 1:
                a.scatter(u, v, c=zz, s=(6 if args.zoom else 1.2), cmap="jet", alpha=0.6,
                          vmin=np.percentile(zz, 2), vmax=np.percentile(zz, 98))
            a.set_xlim(x0, x1); a.set_ylim(y1, y0)
    fig.suptitle(f"GT check: Leica cloud reprojected with GT pose over the image  "
                 f"(dt={args.dt*1000:.0f} ms).  Aligned = GT good; shifted = GT off.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out + ".png", dpi=130)
    print(f"saved {args.out}.png  frames={fr_idx}")


if __name__ == "__main__":
    main()
