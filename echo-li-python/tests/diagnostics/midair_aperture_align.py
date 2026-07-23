"""Quantitative aperture confirmation: does the correspondence-drift direction align
with the CURRENT-frame structure-tensor WEAK eigenvector? If yes (angle << 45deg random,
and smaller for larger drift), the aperture cause is confirmed AND the drift DIRECTION is
observable per-frame -> an anisotropic bias along the weak eigendirection is justified.

Per obs (birth-anchored): d = cumulative drift vector (tracked - GT reproj); w = weak
eigenvector of the structure tensor at the tracked pixel on the current preprocessed image.
Reports acute angle(d, w) in [0,90] stratified by |d|.

  PY=echo-li-python/venv/bin/python
  $PY midair_aperture_align.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import (  # noqa: E402
    make_frontend, structure_tensor_fields, eig_sym2)


def acute_angle(d, w):
    """Acute angle (deg) between drift vector d and undirected eigen-axis w."""
    dn = d / (np.hypot(*d) + 1e-12)
    c = abs(dn[0] * w[0] + dn[1] * w[1])
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


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
    ap.add_argument("--border", type=int, default=16)
    ap.add_argument("--min-age", type=int, default=5)
    args = ap.parse_args()

    import cv2
    from scipy.stats import rankdata, spearmanr
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(getattr(fcfg, "klt_window", 7))
    LV = int(getattr(fcfg, "pyramid_levels", 3))
    last = min(args.start + args.frames, ds.n)
    ks = (2 * r + 1, 2 * r + 1)
    print(f"predicted basin ~ 2^lv * sqrt(Var / lambda_min) per level; {LV} levels, window r={r}")

    def st_var(im):  # normalized (mean) structure tensor + local variance
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, 3)
        sxx = cv2.boxFilter(gx * gx, -1, ks); sxy = cv2.boxFilter(gx * gy, -1, ks); syy = cv2.boxFilter(gy * gy, -1, ks)
        mI = cv2.boxFilter(im, -1, ks); v = np.maximum(cv2.boxFilter(im * im, -1, ks) - mI * mI, 1e-6)
        return sxx, sxy, syy, v

    Xw = {}; born = {}
    rows = []   # mag, age, [angle_lv..], angle_best, [basin_lv..], commit_lv_angle, pred_basin, lmin_L0
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep).astype(np.float32) if prep is not None else img.astype(np.float32)
        pyr = [prep]
        for _ in range(1, LV):
            pyr.append(cv2.pyrDown(pyr[-1]))
        stv = [st_var(p) for p in pyr]
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (args.border <= x < W - args.border and args.border <= y < H - args.border):
                continue
            if fid not in born:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xw[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy); born[fid] = i
                continue
            if i - born[fid] < args.min_age:
                continue
            gp, rng, _z = md.project_world(Xw[fid], T_wb, f, cx, cy)
            if gp is None:
                continue
            d = np.array([x - gp[0], y - gp[1]]); mag = float(np.hypot(*d))
            if mag < 0.5:
                continue
            angs, basins, lmin0 = [], [], np.nan
            for lv in range(LV):
                s = 0.5 ** lv
                sxx, sxy, syy, var = stv[lv]
                hh, ww = sxx.shape; xi, yi = int(round(x * s)), int(round(y * s))
                if not (0 <= xi < ww and 0 <= yi < hh):
                    angs.append(np.nan); basins.append(np.nan); continue
                lmin, lmax, ux, uy, vx, vy = eig_sym2(float(sxx[yi, xi]), float(sxy[yi, xi]), float(syy[yi, xi]))
                angs.append(acute_angle(d, (ux, uy)))
                basins.append((2.0 ** lv) * np.sqrt(float(var[yi, xi]) / max(lmin, 1e-9)))
                if lv == 0:
                    lmin0 = lmin
            commit_lv = int(np.nanargmax(basins))
            rows.append([mag, i - born[fid]] + angs + [np.nanmin(angs)]
                        + basins + [angs[commit_lv], float(np.nanmax(basins)), lmin0, commit_lv])
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] obs={len(rows)}")

    A = np.array(rows, float)
    mag = A[:, 0]
    c_ang = list(range(2, 2 + LV)); c_best = 2 + LV
    c_commit_ang = 2 + LV + 1 + LV; c_pred = c_commit_ang + 1; c_lmin0 = c_pred + 1; c_clv = c_lmin0 + 1
    print(f"\n=== {args.cond}/traj{args.traj}: {len(A)} drifting obs ===")
    print("median angle(drift, weak eigvec): per-level vs oracle-best vs BASIN-SELECTED (non-oracle)")
    print(f"  {'drift |d|':>12} {'n':>7} " + "".join(f"{'L'+str(l):>7}" for l in range(LV))
          + f" {'oracle':>7} {'basin-sel':>9}")
    for lo, hi in [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]:
        m = (mag >= lo) & (mag < hi)
        if m.sum() > 20:
            vals = [np.nanmedian(A[m, c]) for c in c_ang] + [np.nanmedian(A[m, c_best]), np.nanmedian(A[m, c_commit_ang])]
            print(f"  {lo:4.1f}-{hi:<6.0f} {int(m.sum()):7d} " + "".join(f"{v:7.1f}" for v in vals))
    vals = [np.nanmedian(A[:, c]) for c in c_ang] + [np.nanmedian(A[:, c_best]), np.nanmedian(A[:, c_commit_ang])]
    print(f"  {'ALL':>12} {len(A):7d} " + "".join(f"{v:7.1f}" for v in vals) + "   (random=45)")

    # level agreement: does basin-selected commit level match the oracle best-aligning level?
    clv = A[:, c_clv].astype(int)
    olv = np.array([int(np.nanargmin(A[k, c_ang])) for k in range(len(A))])
    print(f"\n  commit-level (max basin): " + ", ".join(f"L{l}={100*np.mean(clv==l):.0f}%" for l in range(LV))
          + f"  |  agrees w/ oracle best-level: {100*np.mean(clv==olv):.0f}%")

    # magnitude: does predicted basin beat raw lambda_min at predicting high drift?
    def auc(sig, lab):
        g = np.isfinite(sig); s, y = sig[g], lab[g].astype(float); npos, nneg = y.sum(), (1-y).sum()
        if npos < 5 or nneg < 5: return np.nan
        rr = rankdata(s); a = (rr[y==1].sum()-npos*(npos+1)/2)/(npos*nneg); return max(a, 1-a)
    hi = (mag >= np.quantile(mag, 0.75)).astype(float)
    print(f"\n  predict high-drift (top 25% |d|):  AUC(predicted basin)={auc(A[:,c_pred],hi):.3f}   "
          f"AUC(raw 1/lambda_min L0)={auc(-A[:,c_lmin0],hi):.3f}")
    rho, _ = spearmanr(A[:, c_pred], mag)
    print(f"  spearman(predicted basin, drift |d|) = {rho:+.3f}")


if __name__ == "__main__":
    main()
