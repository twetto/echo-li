"""Inflation test against the RUST-FAITHFUL baseline (corrects the retracted parallax test,
which used a non-shipping plain-LS covariance). Shipping estimator = WLS with per-obs
R_k = sigma^2 I + J_k C_kk J_k^T; reported covariance = A_pp^-1 = (Sum H^T R_k^-1 H)^-1.

Exact joint reference is ANALYTIC (no MC): the WLS estimate error is
  dX = A_pp^-1 Sum_k H_k^T R_k^-1 eps_k ,  eps_k = pixel n_k  -  J_k dtheta_k
True cov = A_pp^-1 [ Sum_k H_k^T R_k^-1 (sigma^2 I) R_k^-1 H_k                       (pixel, indep)
                   + Sum_ij H_i^T R_i^-1 J_i C_ij J_j^T R_j^-1 H_j ] A_pp^-1          (pose, CORRELATED)
with C_ij = min(gap_i,gap_j) * Sigma_step for the random-walk pose error. Reported drops the
off-diagonal C_ij (i!=j) -> overconfident. Project to range for a 1-DOF depth NEES.

Held-out: calibrate INFLATION-ONLY, TAIL-calibrated f on TRAIN, evaluate coverage on TEST for
  per-obs (A_pp, ships) | global-infl | parallax-infl | exact-joint (analytic).
Answers, against the REAL baseline: how big is the fix, and does parallax earn its keep.

  PY=echo-li-python/venv/bin/python
  $PY midair_inflation_correct.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 220 --window 12 --grid-step 14
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_posecov_nees_test import (se3_exp, proj_jac, pose_jac, cov6,  # noqa: E402
                                       range_of, range_grad, perturb_seq, triangulate)
from midair_perobs_reconcile import triangulate_wls  # noqa: E402


def cov_row(tag, nees):
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
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--grid-step", type=int, default=14)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--rot-deg", type=float, default=0.12)
    ap.add_argument("--tr", type=float, default=0.025)
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
    RT_BC = md.RT_BC
    sr = np.deg2rad(args.rot_deg)
    base = np.diag([sr * sr] * 3 + [args.tr * args.tr] * 3)   # Sigma_step (6x6)

    def cam_center(T):
        return T[:3, :3] @ (-RT_BC[:3, :3].T @ RT_BC[:3, 3]) + T[:3, 3]

    def parallax(X, poses):
        rays = []
        for T in poses:
            v = X - cam_center(T); nv = np.linalg.norm(v)
            if nv > 1e-6:
                rays.append(v / nv)
        R = np.array(rays)
        return np.rad2deg(np.arccos(np.clip(R @ R.T, -1, 1).min())) if len(rays) > 1 else 0.0

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

    err2, v_rep, v_true, para = [], [], [], []
    for Xg, fr in lms:
        n = fr[-1] - fr[0] + 1
        e_dep = perturb_seq(n, "rw", args.rot_deg, args.tr, rng)
        Tset = {fi: T_gt[fi] @ se3_exp(e_dep[fi - fr[0]]) for fi in fr}
        us = [md.project_world(Xg, T_gt[fi], f, cx, cy)[0] + rng.normal(0, args.sigma, 2) for fi in fr]
        Xw, A_pp = triangulate_wls(us, fr, Xg, Tset, args.rot_deg, args.tr, args.sigma, f, cx, cy)
        if Xw is None or A_pp is None:
            continue
        fl = fr[-1]
        r_w = range_of(Xw, Tset[fl], f, cx, cy); r_true = range_of(Xg, T_gt[fl], f, cx, cy)
        gr = range_grad(Xw, Tset[fl], f, cx, cy)
        if r_w is None or r_true is None or r_true <= 0 or gr is None:
            continue
        Hs, RiHs, gks, gaps, ok = [], [], [], [], True
        for fi in fr:
            gp, Hk = proj_jac(Xw, Tset[fi], f, cx, cy)
            Jp = pose_jac(Xw, Tset[fi], f, cx, cy)
            if Hk is None or Jp is None:
                ok = False; break
            Rk = args.sigma ** 2 * np.eye(2) + Jp @ cov6("rw", fi - fr[0], args.rot_deg, args.tr) @ Jp.T
            Ri = np.linalg.inv(Rk)
            Hs.append(Hk); RiHs.append(Ri @ Hk); gks.append(Hk.T @ Ri @ Jp); gaps.append(fi - fr[0])
        if not ok:
            continue
        try:
            Ainv = np.linalg.inv(A_pp)
        except np.linalg.LinAlgError:
            continue
        vr = float(gr @ Ainv @ gr)
        # M = pixel (independent) + pose (correlated) inner covariance of Sum H^T R^-1 eps
        M = args.sigma ** 2 * sum(rh.T @ rh for rh in RiHs)          # pixel term
        for i in range(len(gks)):
            for j in range(len(gks)):
                M = M + min(gaps[i], gaps[j]) * (gks[i] @ base @ gks[j].T)
        vt = float(gr @ Ainv @ M @ Ainv @ gr)
        if vr <= 0 or vt <= 0:
            continue
        err2.append((r_w - r_true) ** 2); v_rep.append(vr); v_true.append(vt)
        para.append(parallax(Xw, [Tset[fi] for fi in fr]))

    err2 = np.array(err2); v_rep = np.array(v_rep); v_true = np.array(v_true); para = np.array(para)
    n = len(err2)
    idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]

    r_tr = err2[tr] / v_rep[tr]                          # per-obs NEES on train
    f_glob = max(np.percentile(r_tr, 95) / 3.841, 1.0)
    edges = np.quantile(para[tr], np.linspace(0, 1, args.bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    fbin = np.array([max(np.percentile(r_tr[(para[tr] >= edges[b]) & (para[tr] < edges[b + 1])], 95) / 3.841, 1.0)
                     if ((para[tr] >= edges[b]) & (para[tr] < edges[b + 1])).sum() > 5 else 1.0
                     for b in range(args.bins)])
    fpar = fbin[np.clip(np.searchsorted(edges, para[te], side="right") - 1, 0, args.bins - 1)]

    print(f"\n=== {args.cond}/traj{ds.traj}: {n} lm ({len(tr)} tr/{len(te)} te), rw-imu, RUST-FAITHFUL (WLS+A_pp) ===")
    print(f"global f = {f_glob:.2f}   parallax f(alpha) = {np.round(fbin, 2)}")
    print("median analytic-joint / reported variance = %.2f  (>1 => reported is overconfident)"
          % np.median(v_true / v_rep))
    print("\nTEST coverage  [median | mean | %>3.84 | %>6.63]   ideal 0.455 | 1 | 5 | 1")
    cov_row("per-obs(ships)", err2[te] / v_rep[te])
    cov_row("global-infl", err2[te] / (v_rep[te] * f_glob))
    cov_row("parallax-infl", err2[te] / (v_rep[te] * fpar))
    cov_row("exact-joint", err2[te] / v_true[te])
    print("\nper-obs ~33% confirms Rust baseline. joint ~5% = exact fix. Compare global vs parallax to joint.")


if __name__ == "__main__":
    main()
