"""Is the POSE-induced depth-NEES heavy tail OBSERVABLE (predictable from geometry we
compute anyway), or unpredictable like the correspondence-drift tail? Hypothesis: the tail
= near-degenerate (low-parallax) triangulations, so it should correlate with parallax angle
and the triangulation info conditioning (lambda_min of A). If a high-parallax subset is
calibrated (%>chi2_95 ~ 5) and the low-parallax subset carries the tail -> OBSERVABLE, gate-
able, NOT doomed. If NEES is uncorrelated with both -> doomed again.

Reuses the pose-cov test machinery; realistic coherent (rw-imu) pose error, per-obs (Rust)
propagated covariance (the one whose tail was 31% > chi2_95).

  PY=echo-li-python/venv/bin/python
  $PY midair_posetail_observability.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 160 --window 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_posecov_nees_test import (se3_exp, proj_jac, pose_jac, cov6,  # noqa: E402
                                       range_of, range_grad, perturb_seq, triangulate)


def cam_center_world(T_wb):
    c_body = -md.RT_BC[:3, :3].T @ md.RT_BC[:3, 3]
    return T_wb[:3, :3] @ c_body + T_wb[:3, 3]


def parallax_deg(X, poses):
    rays = []
    for T in poses:
        v = X - cam_center_world(T); nv = np.linalg.norm(v)
        if nv > 1e-6:
            rays.append(v / nv)
    if len(rays) < 2:
        return 0.0
    R = np.array(rays); G = np.clip(R @ R.T, -1, 1)
    return np.rad2deg(np.arccos(G.min()))   # max pairwise ray angle


def auc(score, label):
    label = np.asarray(label, bool)
    if label.sum() == 0 or label.sum() == len(label):
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = ranks[label].sum()
    return (pos - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--grid-step", type=int, default=18)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--rot-deg", type=float, default=0.12)
    ap.add_argument("--tr", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    g0 = ds.image(args.start); H, W = g0.shape
    f, cx, cy = md.intrinsics(W, H)
    depth0 = ds.depth(args.start); pose0 = ds.pose(args.start)
    last = min(args.start + args.frames, ds.n)
    T_gt = [ds.pose(i) for i in range(last)]
    K = args.window
    rng = np.random.default_rng(args.seed)

    lms = []
    for y in range(10, H - 10, args.grid_step):
        for x in range(10, W - 10, args.grid_step):
            d0 = float(depth0[y, x])
            if not (1.0 < d0 < md.SKY):
                continue
            Xg = md.backproject_world((float(x), float(y)), d0, pose0, f, cx, cy)
            fr = []
            for i in range(args.start, last):
                gp = md.project_world(Xg, T_gt[i], f, cx, cy)[0]
                if gp is not None and 2 <= gp[0] < W - 2 and 2 <= gp[1] < H - 2:
                    fr.append(i)
                    if len(fr) >= K:
                        break
            if len(fr) >= K:
                lms.append((Xg, fr))

    nees, para, lmin, relz = [], [], [], []
    for Xg, fr in lms:
        n = fr[-1] - fr[0] + 1
        e_dep = perturb_seq(n, "rw", args.rot_deg, args.tr, rng)
        Tset = {fi: T_gt[fi] @ se3_exp(e_dep[fi - fr[0]]) for fi in fr}
        us = [md.project_world(Xg, T_gt[fi], f, cx, cy)[0] + rng.normal(0, args.sigma, 2) for fi in fr]
        Xe, _A = triangulate(us, [Tset[fi] for fi in fr], Xg, f, cx, cy, args.sigma)
        if Xe is None:
            continue
        fl = fr[-1]
        r_est = range_of(Xe, Tset[fl], f, cx, cy); r_true = range_of(Xg, T_gt[fl], f, cx, cy)
        gr = range_grad(Xe, Tset[fl], f, cx, cy)
        if r_est is None or r_true is None or r_true <= 0 or gr is None:
            continue
        # per-obs (Rust) propagated covariance
        A_pp = np.zeros((3, 3)); ok = True
        for fi in fr:
            _gp, Hk = proj_jac(Xe, Tset[fi], f, cx, cy)
            Jp = pose_jac(Xe, Tset[fi], f, cx, cy)
            if Hk is None or Jp is None:
                ok = False; break
            Rk = args.sigma ** 2 * np.eye(2) + Jp @ cov6("rw", fi - fr[0], args.rot_deg, args.tr) @ Jp.T
            A_pp += Hk.T @ np.linalg.inv(Rk) @ Hk
        if not ok:
            continue
        try:
            var_pp = float(gr @ np.linalg.inv(A_pp) @ gr)
            lm = float(np.linalg.eigvalsh(A_pp)[0])
        except np.linalg.LinAlgError:
            continue
        if var_pp <= 0:
            continue
        nees.append((r_est - r_true) ** 2 / var_pp)
        para.append(parallax_deg(Xe, [Tset[fi] for fi in fr]))
        lmin.append(lm)
        relz.append((r_est - r_true) / r_true)

    nees = np.array(nees); para = np.array(para); lmin = np.array(lmin); relz = np.array(relz)
    tail = nees > 3.841
    print(f"\n=== {args.cond}/traj{ds.traj}: {len(nees)} landmarks, rw-imu pose, per-obs cov ===")
    print(f"depth NEES: median {np.median(nees):.2f}  mean {np.mean(nees):.2f}  "
          f"%>3.84 {100*tail.mean():.1f} (ideal 5)  %>6.63 {100*np.mean(nees>6.635):.1f}")
    print(f"\nIS THE TAIL OBSERVABLE?  (tail = NEES > chi2_95 = 3.841, {tail.sum()} of {len(nees)})")
    print(f"  spearman(NEES, parallax_deg)      = {spearmanr(nees, para)[0]:+.3f}  (<0 = low parallax -> tail)")
    print(f"  spearman(NEES, lambda_min(A))     = {spearmanr(nees, lmin)[0]:+.3f}  (<0 = ill-cond -> tail)")
    print(f"  AUC 1/parallax  predicts tail     = {auc(-para, tail):.3f}  (vs corr-drift cues ~0.5-0.66)")
    print(f"  AUC 1/lambda_min predicts tail    = {auc(-lmin, tail):.3f}")
    print(f"  AUC reported-sigma predicts tail  = {auc(-lmin, tail):.3f}  (sigma_depth ~ 1/sqrt(lmin))")
    print("\n  parallax terciles: does gating low-parallax remove the tail?")
    lo, hi = np.percentile(para, [33, 67])
    for tag, m in [("low  para (<p33)", para < lo), ("mid  para", (para >= lo) & (para < hi)),
                   ("high para (>p67)", para >= hi)]:
        if m.sum() > 5:
            print(f"    {tag:>18}: n={m.sum():4d}  median parallax {np.median(para[m]):5.2f}deg  "
                  f"NEES med {np.median(nees[m]):6.2f}  %>3.84 {100*np.mean(nees[m]>3.841):5.1f}  "
                  f"dangerous(relz>0) {100*np.mean(relz[m]>0):4.0f}%")
    print("\n  high-parallax subset ~5% tail + strong AUC => OBSERVABLE, gate-able (NOT doomed).")


if __name__ == "__main__":
    main()
