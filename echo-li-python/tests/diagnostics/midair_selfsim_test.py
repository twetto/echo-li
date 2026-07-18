"""Repetitive-texture vs pure-aperture test. For each drifting track, measure patch
SELF-SIMILARITY: NCC of the patch against the local image region (matchTemplate);
the peak at center is the self-match (=1), a strong SECONDARY peak away from center
means a repeated/self-similar pattern (mislock risk) rather than a single edge valley.

If high-drift tracks have higher secondary NCC than low-drift ones (AUC >> the raw
texture cues), and the drift points TOWARD the secondary peak, the drift is (partly)
the REPETITIVE-TEXTURE / aliasing problem, not just continuous aperture slide.

  PY=echo-li-python/venv/bin/python
  $PY midair_selfsim_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend, structure_tensor_fields, eig_sym2  # noqa: E402


def auc(sig, lab):
    g = np.isfinite(sig); s, y = sig[g], lab[g].astype(float); npos, nneg = y.sum(), (1 - y).sum()
    if npos < 5 or nneg < 5:
        return np.nan
    rr = rankdata(s); a = (rr[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return max(a, 1 - a)


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
    ap.add_argument("--search", type=int, default=10, help="secondary-peak search radius beyond the patch")
    ap.add_argument("--exclude", type=int, default=3, help="central-peak exclusion radius (px)")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(getattr(fcfg, "klt_window", 7))
    S = r + args.search
    last = min(args.start + args.frames, ds.n)
    print(f"self-similarity: patch r={r}, search +/-{args.search}px, exclude central {args.exclude}px")

    Xw = {}; born = {}
    rows = []   # drift_mag, sec_ncc, angle(drift, dir_to_secondary), lambda_min
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep).astype(np.float32) if prep is not None else img.astype(np.float32)
        sxx, sxy, syy = structure_tensor_fields(prep, r)
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (S + 1 <= x < W - S - 1 and S + 1 <= y < H - S - 1):
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
            if mag < 0.5:
                continue
            xi, yi = int(round(x)), int(round(y))
            patch = prep[yi - r:yi + r + 1, xi - r:xi + r + 1]
            search = prep[yi - S:yi + S + 1, xi - S:xi + S + 1]
            if patch.shape != (2 * r + 1, 2 * r + 1) or search.shape != (2 * S + 1, 2 * S + 1):
                continue
            ncc = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)  # (2*search+1)^2, center=self
            cc = ncc.shape[0] // 2
            yy, xx = np.mgrid[0:ncc.shape[0], 0:ncc.shape[1]]
            mask = (np.hypot(xx - cc, yy - cc) > args.exclude)
            sec = float(ncc[mask].max())
            k = np.argmax(np.where(mask, ncc, -1))
            sy, sx = np.unravel_index(k, ncc.shape)
            sdir = np.array([sx - cc, sy - cc], float)
            ang = np.degrees(np.arccos(np.clip(abs(d @ sdir) / (mag * (np.hypot(*sdir) + 1e-9) + 1e-9), 0, 1)))
            lmin, lmax, *_ = eig_sym2(float(sxx[yi, xi]), float(sxy[yi, xi]), float(syy[yi, xi]))
            rows.append((mag, sec, ang, lmin))
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] obs={len(rows)}")

    A = np.array(rows, float)
    mag, sec, ang = A[:, 0], A[:, 1], A[:, 2]
    print(f"\n=== {args.cond}/traj{args.traj}: {len(A)} drifting obs ===")
    print("repetitive-texture hypothesis: high-drift => HIGH secondary-NCC + drift ALONG dir-to-secondary")
    print(f"\n  {'drift |d|':>12} {'n':>7} {'sec_NCC med':>11} {'>0.7':>7} {'ang(drift,2ndpk)':>17}")
    for lo, hi in [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]:
        m = (mag >= lo) & (mag < hi)
        if m.sum() > 20:
            print(f"  {lo:4.1f}-{hi:<6.0f} {int(m.sum()):7d} {np.median(sec[m]):11.3f} "
                  f"{100*np.mean(sec[m]>0.7):6.1f}% {np.median(ang[m]):16.1f}")
    hi = (mag >= np.quantile(mag, 0.75)).astype(float)
    print(f"\n  AUC predict high-drift (top25% |d|):  secondary-NCC={auc(sec, hi):.3f}   "
          f"raw 1/lambda_min={auc(-A[:,3], hi):.3f}")
    rho, _ = spearmanr(sec, mag)
    print(f"  spearman(secondary-NCC, drift |d|) = {rho:+.3f}")
    print(f"  median angle(drift, dir-to-secondary-peak) = {np.median(ang):.1f}deg  (0=drift toward it, 90=random~45)")


if __name__ == "__main__":
    main()
