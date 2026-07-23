"""(b) Coverage / false-negatives (SPARSE proxy). The residual obstacle-avoidance risk after the
depth is honest is a MISSED thin obstacle falling in a gap between sparse landmarks. Measure the
actual Rudolf landmark density: over frames, nearest-landmark distance across the image, and the
implied minimum detectable obstacle size (physical) at scene depth. This motivates the dense patch
mapper (the real fix) which we can't measure here -> flagged as proxy.

  PY=echo-li-python/venv/bin/python
  $PY midair_coverage.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir --cond sunny --traj 2 \
      --frames 120 --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "diagnostics_midair_sparse3d.yaml"))
    ap.add_argument("--grid", type=int, default=16, help="query-grid stride (px)")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.frames, ds.n)
    gx, gy = np.meshgrid(np.arange(0, W, args.grid), np.arange(0, H, args.grid))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    nn_all, nfeat, depths = [], [], []
    for i in range(last):
        img = ds.image(i); depth = ds.depth(i)
        feats, _ = tracker.process(img)
        pts = []
        for fd in feats:
            x, y = float(fd["x"]), float(fd["y"])
            if 0 <= x < W and 0 <= y < H:
                d = float(depth[min(int(round(y)), H - 1), min(int(round(x)), W - 1)])
                if 1.0 < d < md.SKY:
                    pts.append((x, y))
        if len(pts) < 8:
            continue
        pts = np.array(pts)
        d_near, _ = cKDTree(pts).query(grid, k=1)   # nearest landmark px-dist for each grid point
        nn_all.append(d_near)
        nfeat.append(len(pts))
        dm = depth[(depth > 1.0) & (depth < md.SKY)]
        depths.append(np.median(dm))

    nn = np.concatenate(nn_all)
    nfeat = np.array(nfeat); med_depth = float(np.median(depths))
    print(f"\n=== {args.cond}/traj{ds.traj}: {last} frames, {W}x{H}, ~{int(np.median(nfeat))} landmarks/frame ===")
    print(f"nearest-landmark distance across image (px):  median {np.median(nn):.1f}  "
          f"p90 {np.percentile(nn,90):.1f}  p95 {np.percentile(nn,95):.1f}  max {nn.max():.0f}")
    for D in [8, 16, 24, 32]:
        print(f"  image area > {D:2d} px from any landmark (a <{D}px obstacle could fall in a gap): "
              f"{100*np.mean(nn > D):5.1f}%")
    print(f"\nscene median depth ~ {med_depth:.1f} m, f = {f:.0f} px (work res {W}x{H})")
    for D in [16, 32]:
        s = D * med_depth / f
        print(f"  a gap of {D} px at {med_depth:.0f} m = a real obstacle ~{s:.2f} m wide could be MISSED by sparse")
    print("\nSPARSE proxy only: sparse landmarks leave px-scale gaps -> thin obstacles missed.")
    print("Real fix = DENSE patch mapper (per-pixel coverage), not measured here -> open item.")


if __name__ == "__main__":
    main()
