"""Collector for the anisotropic-vs-isotropic bias calibration test. Per drifting
obs, decompose the cumulative drift vector into (along-weak, perpendicular) in the
L0 structure-tensor eigenbasis and save. Analysis (separate) fits isotropic vs
anisotropic (+ heavy-tailed) bias covariance and checks chi2(2) calibration + transfer.

  PY=echo-li-python/venv/bin/python
  $PY midair_aniso_calib.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml \
      --save-npz /tmp/aniso_sunny.npz
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend, structure_tensor_fields, eig_sym2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "diagnostics_midair_sparse3d.yaml"))
    ap.add_argument("--min-age", type=int, default=5)
    ap.add_argument("--save-npz", required=True)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(getattr(fcfg, "klt_window", 7))
    last = min(args.start + args.frames, ds.n)

    Xw = {}; born = {}
    rows = []   # b_along, b_perp, mag, lambda_min
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep).astype(np.float32) if prep is not None else img.astype(np.float32)
        sxx, sxy, syy = structure_tensor_fields(prep, r)
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (2 <= x < W - 2 and 2 <= y < H - 2):
                continue
            if fid not in born:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xw[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy); born[fid] = i
                continue
            if i - born[fid] < args.min_age:
                continue
            gp, _rng, _z = md.project_world(Xw[fid], T_wb, f, cx, cy)
            if gp is None:
                continue
            d = np.array([x - gp[0], y - gp[1]]); mag = float(np.hypot(*d))
            if mag < 0.3:
                continue
            xi, yi = int(round(x)), int(round(y))
            lmin, lmax, ux, uy, vx, vy = eig_sym2(float(sxx[yi, xi]), float(sxy[yi, xi]), float(syy[yi, xi]))
            b_along = abs(d[0] * ux + d[1] * uy)   # weak eigenvector (aperture axis)
            b_perp = abs(d[0] * vx + d[1] * vy)    # strong eigenvector
            rows.append((b_along, b_perp, mag, lmin))
        if (i - args.start) % 100 == 0:
            print(f"  [{args.cond} {i-args.start}/{last-args.start}] obs={len(rows)}")

    A = np.array(rows, float)
    np.savez(args.save_npz, rows=A, cols=np.array(["b_along", "b_perp", "mag", "lambda_min"]))
    print(f"saved {args.save_npz}: {len(A)} obs  "
          f"median b_along={np.median(A[:,0]):.3f} b_perp={np.median(A[:,1]):.3f} "
          f"(anisotropy ratio {np.median(A[:,0])/max(np.median(A[:,1]),1e-9):.2f})")


if __name__ == "__main__":
    main()
