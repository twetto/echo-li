"""Is parallax-scaled variance inflation GOOD ENOUGH vs the exact joint covariance?
Held-out test: calibrate an inflation factor on a TRAIN split (against GT error, as you would
once per deployment), apply on a TEST split, compare depth-NEES coverage of:
  per-obs      : no inflation (the Rust term as-is)          -> tail ~31% last time
  global-infl  : single scalar f (parallax-INdependent)      -> isolates whether alpha matters
  parallax-infl: f(alpha), binned                            -> the cheap fix under test
  MC-joint     : exact correlated covariance (trimmed var)   -> the expensive reference
Ideal chi2(1): median 0.455 | mean 1 | %>3.84 = 5 | %>6.63 = 1.

If parallax-infl coverage ~ joint => cheap fix is good enough. If it only fixes the median but
the tail stays >> joint => parallax is too lossy a summary; you need the joint (or heavy-tail).

  PY=echo-li-python/venv/bin/python
  $PY midair_parallax_inflation_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 170 --window 12 --mc 30
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_posecov_nees_test import (se3_exp, proj_jac, pose_jac, cov6,  # noqa: E402
                                       range_of, range_grad, perturb_seq, triangulate)


def cam_center_world(T):
    return T[:3, :3] @ (-md.RT_BC[:3, :3].T @ md.RT_BC[:3, 3]) + T[:3, 3]


def parallax_deg(X, poses):
    rays = []
    for T in poses:
        v = X - cam_center_world(T); nv = np.linalg.norm(v)
        if nv > 1e-6:
            rays.append(v / nv)
    if len(rays) < 2:
        return 0.0
    R = np.array(rays)
    return np.rad2deg(np.arccos(np.clip(R @ R.T, -1, 1).min()))


def cov_table(tag, nees):
    a = np.asarray(nees)
    print(f"  {tag:>14}: {np.median(a):8.3f} | {np.mean(a):8.2f} | "
          f"{100*np.mean(a > 3.841):5.1f} | {100*np.mean(a > 6.635):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=170)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--grid-step", type=int, default=18)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--rot-deg", type=float, default=0.12)
    ap.add_argument("--tr", type=float, default=0.025)
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--bins", type=int, default=5)
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

    def mc_pose_var(Xg, fr, us, n, correlated):
        rr = []
        for _ in range(args.mc):
            e = perturb_seq(n, "rw", args.rot_deg, args.tr, rng, correlated=correlated)
            Tm = {fi: T_gt[fi] @ se3_exp(e[fi - fr[0]]) for fi in fr}
            Xm, _ = triangulate(us, [Tm[fi] for fi in fr], Xg, f, cx, cy, args.sigma)
            if Xm is not None:
                rv = range_of(Xm, Tm[fr[-1]], f, cx, cy)
                if rv is not None:
                    rr.append(rv)
        if len(rr) < 10:
            return np.nan
        rr = np.array(rr); med = np.median(rr); mad = np.median(np.abs(rr - med)) + 1e-9
        rr = rr[np.abs(rr - med) < 8 * mad]           # trim degenerate wrong-pose triangulations
        return np.var(rr) if len(rr) > 8 else np.nan

    err2, vperobs, vjoint, para = [], [], [], []
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
        # v_pix: pixels-only estimator variance (analytic)
        A_pix = np.zeros((3, 3)); ok = True
        for fi in fr:
            _gp, Hk = proj_jac(Xe, Tset[fi], f, cx, cy)
            if Hk is None:
                ok = False; break
            A_pix += Hk.T @ Hk / args.sigma ** 2
        if not ok:
            continue
        try:
            v_pix = float(gr @ np.linalg.inv(A_pix) @ gr)
        except np.linalg.LinAlgError:
            continue
        # pose-induced variance, fixed measurements: per-obs (independent) vs joint (correlated)
        v_pose_indep = mc_pose_var(Xg, fr, us, n, correlated=False)
        v_pose_joint = mc_pose_var(Xg, fr, us, n, correlated=True)
        if not np.isfinite(v_pose_indep) or not np.isfinite(v_pose_joint) or v_pix <= 0:
            continue
        err2.append((r_est - r_true) ** 2)
        vperobs.append(v_pix + v_pose_indep)      # what per-obs propagation reports
        vjoint.append(v_pix + v_pose_joint)       # exact joint
        para.append(parallax_deg(Xe, [Tset[fi] for fi in fr]))

    err2 = np.array(err2); vperobs = np.array(vperobs); vjoint = np.array(vjoint); para = np.array(para)
    n = len(err2)
    idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]

    # calibrate on TRAIN: INFLATION-ONLY (never shrink reported variance -> never make a landmark
    # look more certain), TAIL-calibrated (put the 95th pct of NEES at chi2_95=3.841 -> 5% coverage).
    r_tr = err2[tr] / vperobs[tr]
    f_glob = max(np.percentile(r_tr, 95) / 3.841, 1.0)
    edges = np.quantile(para[tr], np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    fbin = []
    for b in range(args.bins):
        m = (para[tr] >= edges[b]) & (para[tr] < edges[b + 1])
        fbin.append(max(np.percentile(r_tr[m], 95) / 3.841, 1.0) if m.sum() > 5 else 1.0)
    fbin = np.array(fbin)

    def f_para(a):
        b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, args.bins - 1)
        return fbin[b]

    print(f"\n=== {args.cond}/traj{ds.traj}: {n} landmarks ({len(tr)} train / {len(te)} test), rw-imu ===")
    print(f"global inflation f = {f_glob:.2f}   parallax-bin f(alpha) = {np.round(fbin, 2)}  "
          f"(bins by parallax quantile)")
    print("\nTEST-split depth NEES coverage   [median | mean | %>3.84 | %>6.63]  ideal 0.455 | 1 | 5 | 1")
    cov_table("per-obs", err2[te] / vperobs[te])
    cov_table("global-infl", err2[te] / (vperobs[te] * f_glob))
    cov_table("parallax-infl", err2[te] / (vperobs[te] * np.array([f_para(a) for a in para[te]])))
    cov_table("MC-joint", err2[te] / vjoint[te])
    print("\nparallax-infl ~ MC-joint => cheap fix good enough. parallax-infl tail >> joint => too lossy.")
    print("parallax-infl ~ global-infl => the alpha-dependence didn't help (just needed more variance).")


if __name__ == "__main__":
    main()
