"""(a) Safety under ESTIMATED poses, the number that matters: after the honest (joint) depth
covariance, does the one-sided LOWER confidence bound stay conservative?

Two error sources push opposite ways: correspondence drift collapses depth NEARER (safe);
triangulation-under-noise OVER-estimates depth for LOW-PARALLAX landmarks (dangerous, r=1/rho
blows up). So the raw point estimate has a dangerous farther-tail. But those are exactly the
high-variance landmarks, so the honest lower bound D_lo = r_est - k*sqrt(v_joint) should still
be <= r_true. Metric: P(D_lo > r_true) = residual dangerous rate AFTER honest uncertainty.

Exact correspondences + rw-imu pose (WLS estimate, analytic joint variance). k=2 (one-sided ~97.7%).

  PY=echo-li-python/venv/bin/python
  $PY midair_safety_estpose.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir --cond sunny --traj 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_posecov_nees_test import (se3_exp, proj_jac, pose_jac, cov6,  # noqa: E402
                                       range_of, range_grad, perturb_seq)
from midair_perobs_reconcile import triangulate_wls  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--grid-step", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--rot-deg", type=float, default=0.12)
    ap.add_argument("--tr", type=float, default=0.025)
    ap.add_argument("--k", type=float, default=2.0, help="one-sided sigma multiplier for lower bound")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    g0 = ds.image(0); H, W = g0.shape
    f, cx, cy = md.intrinsics(W, H)
    depth0 = ds.depth(0); pose0 = ds.pose(0)
    last = min(args.frames, ds.n)
    T_gt = [ds.pose(i) for i in range(last)]
    K = args.window
    rng = np.random.default_rng(args.seed)
    sr = np.deg2rad(args.rot_deg)
    base = np.diag([sr * sr] * 3 + [args.tr * args.tr] * 3)

    lms = []
    for y in range(10, H - 10, args.grid_step):
        for x in range(10, W - 10, args.grid_step):
            d0 = float(depth0[y, x])
            if not (1.0 < d0 < md.SKY):
                continue
            Xg = md.backproject_world((float(x), float(y)), d0, pose0, f, cx, cy)
            fr = []
            for i in range(last):
                gp = md.project_world(Xg, T_gt[i], f, cx, cy)[0]
                if gp is not None and 2 <= gp[0] < W - 2 and 2 <= gp[1] < H - 2:
                    fr.append(i)
                    if len(fr) >= K:
                        break
            if len(fr) >= K:
                lms.append((Xg, fr))

    relz, dlo_danger, raw_danger, rep_danger = [], [], [], []
    for Xg, fr in lms:
        n = fr[-1] - fr[0] + 1
        e = perturb_seq(n, "rw", args.rot_deg, args.tr, rng)
        Tset = {fi: T_gt[fi] @ se3_exp(e[fi - fr[0]]) for fi in fr}
        us = [md.project_world(Xg, T_gt[fi], f, cx, cy)[0] + rng.normal(0, args.sigma, 2) for fi in fr]
        Xw, A_pp = triangulate_wls(us, fr, Xg, Tset, args.rot_deg, args.tr, args.sigma, f, cx, cy)
        if Xw is None or A_pp is None:
            continue
        fl = fr[-1]
        r_e = range_of(Xw, Tset[fl], f, cx, cy); r_t = range_of(Xg, T_gt[fl], f, cx, cy)
        gr = range_grad(Xw, Tset[fl], f, cx, cy)
        if r_e is None or r_t is None or r_t <= 0 or gr is None:
            continue
        RiHs, gks, gaps, ok = [], [], [], True
        for fi in fr:
            gp, Hk = proj_jac(Xw, Tset[fi], f, cx, cy)
            Jp = pose_jac(Xw, Tset[fi], f, cx, cy)
            if Hk is None or Jp is None:
                ok = False; break
            Rk = args.sigma ** 2 * np.eye(2) + Jp @ cov6("rw", fi - fr[0], args.rot_deg, args.tr) @ Jp.T
            Ri = np.linalg.inv(Rk)
            RiHs.append(Ri @ Hk); gks.append(Hk.T @ Ri @ Jp); gaps.append(fi - fr[0])
        if not ok:
            continue
        try:
            Ainv = np.linalg.inv(A_pp)
        except np.linalg.LinAlgError:
            continue
        M = args.sigma ** 2 * sum(rh.T @ rh for rh in RiHs)
        for i in range(len(gks)):
            for j in range(len(gks)):
                M = M + min(gaps[i], gaps[j]) * (gks[i] @ base @ gks[j].T)
        v_rep = float(gr @ Ainv @ gr)
        v_true = float(gr @ Ainv @ M @ Ainv @ gr)
        if v_true <= 0 or v_rep <= 0:
            continue
        relz.append((r_e - r_t) / r_t)
        raw_danger.append(r_e > 1.2 * r_t)                                  # raw est >20% farther
        rep_danger.append(r_e - args.k * np.sqrt(v_rep) > r_t)              # lower bnd w/ REPORTED (overconf) cov
        dlo_danger.append(r_e - args.k * np.sqrt(v_true) > r_t)             # lower bnd w/ HONEST joint cov

    z = np.array(relz)
    print(f"\n=== {args.cond}/traj{ds.traj}: {len(z)} landmarks, rw-imu est-pose, k={args.k} ===")
    print(f"RAW point estimate:   median relz {100*np.median(z):+.2f}%   dangerous(r_est>r_true) {100*np.mean(z>0):.0f}%")
    print(f"  est >20% farther (dangerous, unflagged risk if trusted): {100*np.mean(raw_danger):.1f}%")
    print(f"\nAfter one-sided LOWER bound D_lo = r_est - {args.k}*sigma:")
    print(f"  with REPORTED (per-obs, overconfident) cov:  P(D_lo > r_true) = {100*np.mean(rep_danger):.2f}%  (still risky)")
    print(f"  with HONEST (joint) cov:                     P(D_lo > r_true) = {100*np.mean(dlo_danger):.2f}%  (the safety number)")
    print("\nlow P(D_lo>r_true) with honest cov => the dangerous over-estimates are the high-variance")
    print("landmarks; the honest covariance flags them and the lower bound stays conservative.")


if __name__ == "__main__":
    main()
