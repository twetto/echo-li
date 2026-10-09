"""Measure KLT correspondence drift against MidAir ground truth.

A tracked pixel is supposed to follow ONE world point. It does not: the template
slides. This script measures how far, using GT depth + GT pose to reproject each
track's birth landmark into every later frame and comparing to where the tracker
actually put it.

Why it matters: Sparse3D's `bias_walk_var` models this drift as a per-track random
walk on the measurement, `b_k = b_{k-1} + w_k`, `w_k ~ N(0, bias_walk_var I)` px^2.
That setting is a physical quantity, not a tuning knob -- this script measures it, so
a value that improves NEES can be checked against the drift that actually occurs
instead of being fitted.

Random-walk signature: drift std grows as sqrt(age). A CONSTANT offset (e.g. a
calibration error) would instead be flat in age. The two are distinguished here by
fitting std^2 vs age and reporting both the slope and how well the sqrt law holds.

Run:
  PY=.venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_correspondence_drift.py \
      --root ~/18TB/datasets/dataset_MidAir/MidAir --traj 2 --frames 600
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import midair_drift as md  # noqa: E402

import echo_li  # noqa: E402

R_MARGIN = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="must match the NEES harness (default 0.5) for the measured "
                         "px^2 to be comparable to its bias_walk_var")
    ap.add_argument("--config", default="configs/diagnostics_midair_e2e_depth_nees.yaml")
    ap.add_argument("--save-npz", default="")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)

    cfg = args.config
    if not Path(cfg).is_absolute():
        cfg = str(Path(__file__).resolve().parents[3] / cfg)
    fcfg = echo_li.FrontendConfig.from_yaml(cfg)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)

    # GT poses. `ds.pose(k)` is T_wb at camera frame k; the camera extrinsic is the
    # same cam->body S0 the NEES harness uses, so T_wc = T_wb * S0.
    S0 = np.array([[0., 0., 1., 0.],
                   [1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.]])

    def T_wc(k):
        return ds.pose(k) @ S0

    # birth[tid] = (world point, birth frame). drift rows: (age, dx, dy)
    birth = {}
    rows = []
    nimg = min(args.frames, ds.n)
    for k in range(nimg):
        gray = np.asarray(ds.image(k))
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        feats, _ = tracker.process(gray)
        depth = ds.depth(k)
        Twc = T_wc(k)
        Rcw = Twc[:3, :3].T
        tcw = -Rcw @ Twc[:3, 3]
        for fd in feats:
            tid = int(fd["id"])
            u, v = float(fd["x"]), float(fd["y"])
            gx, gy = int(round(u)), int(round(v))
            if not (R_MARGIN < gx < W - R_MARGIN and R_MARGIN < gy < H - R_MARGIN):
                continue
            if tid not in birth:
                # Seed the landmark from GT at first sight. MidAir depth is RADIAL ray
                # distance, so the world point is r * unit_bearing, not r along z.
                r = float(depth[gy, gx])
                if not (0.5 < r < md.SKY):
                    continue
                b = np.array([(u - cx) / f, (v - cy) / f, 1.0])
                b /= np.linalg.norm(b)
                p_c = r * b
                birth[tid] = (Twc[:3, :3] @ p_c + Twc[:3, 3], k)
                continue
            p_w, k0 = birth[tid]
            p_c = Rcw @ p_w + tcw
            if p_c[2] < 0.5:
                continue
            up = f * p_c[0] / p_c[2] + cx
            vp = f * p_c[1] / p_c[2] + cy
            rows.append((k - k0, u - up, v - vp))

    d = np.array(rows)
    if len(d) == 0:
        print("no drift samples")
        return
    age, dx, dy = d[:, 0], d[:, 1], d[:, 2]
    mag = np.hypot(dx, dy)
    print(f"traj {args.traj}  scale {args.scale}  {nimg} frames  "
          f"{len(birth)} tracks  {len(d)} drift samples")
    print(f"\ndrift magnitude (px): median {np.median(mag):.3f}  "
          f"p90 {np.percentile(mag, 90):.3f}  p99 {np.percentile(mag, 99):.3f}")
    print(f"\n{'age':>6} {'n':>7} {'std_dx':>8} {'std_dy':>8} {'var/age':>9} {'med|d|':>8}")
    bins = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 100000)]
    slopes = []
    for lo, hi in bins:
        m = (age >= lo) & (age < hi)
        if m.sum() < 30:
            continue
        sx, sy = dx[m].std(), dy[m].std()
        a = age[m].mean()
        # Random walk: var = bias_walk_var * age  =>  var/age is CONSTANT in age.
        vpa = 0.5 * (sx ** 2 + sy ** 2) / a
        slopes.append(vpa)
        lab = f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
        print(f"{lab:>6} {m.sum():7d} {sx:8.3f} {sy:8.3f} {vpa:9.5f} "
              f"{np.median(mag[m]):8.3f}")
    if slopes:
        s = np.array(slopes)
        print(f"\nvar/age across bins: median {np.median(s):.5f} px^2/frame  "
              f"spread {s.min():.5f}-{s.max():.5f}")
        print("  (constant across bins => random walk; rising/falling => not)")
        print(f"\n==> measured bias_walk_var ~ {np.median(s):.5f} px^2/frame")
    # Mean drift: a random walk is ZERO-MEAN; a systematic slide is not.
    print(f"\nmean drift dx {dx.mean():+.4f} dy {dy.mean():+.4f} px "
          f"(zero-mean => random walk, non-zero => systematic slide)")
    if args.save_npz:
        np.savez(args.save_npz, age=age, dx=dx, dy=dy)
        print(f"wrote {args.save_npz}")


if __name__ == "__main__":
    main()
