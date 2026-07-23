"""Does propagating POSE COVARIANCE make the windowed depth honest (NEES calibrated),
and is the per-observation INDEPENDENT propagation (what the Rust r_meas += proj*Pvv*proj^T
term does) enough for COHERENT pose drift -- or does that need the JOINT window covariance?

Setup isolates pose from tracker drift: seed grid landmarks (GT depth), synthesize EXACT
reprojections + pixel noise (sigma), so the ONLY estimation error beyond pixels is POSE.
Perturb poses two structures: iid (independent per frame) vs rw (random-walk = IMU-like
coherent drift). Triangulate; obstacle-relevant 1-DOF depth (range) NEES ~ chi2(1) (ideal
median 0.455, mean 1) under three covariances:
  pixels-only   : ignores pose (the current behaviour if Pvv/Pww not passed)
  per-obs-indep : pose cov added per observation INDEPENDENTLY (Monte-Carlo w/ frames
                  resampled independently from their marginal) = the Rust per-obs term
  MC-joint      : full joint pose covariance (Monte-Carlo resampling the ACTUAL correlated
                  pose distribution) = the correct propagation

Reading: pixels-only >> ideal (pose ignored -> overconfident). MC-joint ~ ideal => pose error
is coverable VARIANCE, honest depth achievable. If per-obs-indep ~ MC-joint for iid but
per-obs-indep still overconfident for rw => the per-observation term is INSUFFICIENT for
coherent drift; the joint (cross-frame) covariance is needed (same lesson as the bias state).

  PY=echo-li-python/venv/bin/python
  $PY midair_posecov_nees_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 200 --window 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402


def se3_exp(xi):
    R = cv2.Rodrigues(np.asarray(xi[:3], float))[0]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xi[3:6]
    return T


def proj_jac(X, T, f, cx, cy, eps=1e-4):
    gp, _r, z = md.project_world(X, T, f, cx, cy)
    if gp is None or z <= 0.1:
        return None, None
    J = np.zeros((2, 3))
    for d in range(3):
        dv = np.zeros(3); dv[d] = eps
        a = md.project_world(X + dv, T, f, cx, cy)[0]; b = md.project_world(X - dv, T, f, cx, cy)[0]
        if a is None or b is None:
            return gp, None
        J[:, d] = (a - b) / (2 * eps)
    return gp, J


def triangulate(us, Ts, X0, f, cx, cy, sigma, iters=15):
    X = X0.astype(float).copy(); s2 = sigma * sigma; A = np.zeros((3, 3))
    for _ in range(iters):
        A = np.zeros((3, 3)); b = np.zeros(3); used = 0
        for u, T in zip(us, Ts):
            gp, J = proj_jac(X, T, f, cx, cy)
            if gp is None or J is None:
                continue
            r = u - gp
            A += (J.T @ J) / s2; b += (J.T @ r) / s2; used += 1
        if used < 2:
            return None, None
        try:
            X = X + np.linalg.solve(A + 1e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None, None
    return X, A


def range_of(X, T, f, cx, cy):
    return md.project_world(X, T, f, cx, cy)[1]


def pose_jac(X, T, f, cx, cy, eps=1e-4):
    """d(pixel)/d(body-frame pose perturbation), 2x6 (numerical, frame-safe)."""
    J = np.zeros((2, 6))
    for j in range(6):
        d = np.zeros(6); d[j] = eps
        ap = md.project_world(X, T @ se3_exp(d), f, cx, cy)[0]
        am = md.project_world(X, T @ se3_exp(-d), f, cx, cy)[0]
        if ap is None or am is None:
            return None
        J[:, j] = (ap - am) / (2 * eps)
    return J


def cov6(mode, gap, sr_deg, st):
    """Per-observation 6-DOF pose covariance. iid: constant marginal. rw: marginal grows
    with the frame gap from window start (independent per-obs = ignores cross-frame corr)."""
    sr = np.deg2rad(sr_deg)
    base = np.diag([sr * sr] * 3 + [st * st] * 3)
    return base * max(gap, 1e-6) if mode == "rw" else base


def range_grad(X, T, f, cx, cy, eps=1e-3):
    g = np.zeros(3)
    for d in range(3):
        dv = np.zeros(3); dv[d] = eps
        rp = range_of(X + dv, T, f, cx, cy); rm = range_of(X - dv, T, f, cx, cy)
        if rp is None or rm is None:
            return None
        g[d] = (rp - rm) / (2 * eps)
    return g


def win_errs(win_fr, Xg, gen, tri, f, cx, cy, sigma, rng):
    """Measurements from the TRUE poses `gen` (+pixel noise); triangulate with the
    ESTIMATED poses `tri`. The gen/tri mismatch is the pose error. Returns (X, A_pix, us)."""
    us = []
    for fi in win_fr:
        gp = md.project_world(Xg, gen[fi], f, cx, cy)[0]
        us.append(gp + rng.normal(0, sigma, 2))
    X, A = triangulate(us, [tri[fi] for fi in win_fr], Xg, f, cx, cy, sigma)
    return X, A, us


def perturb_seq(n, mode, s_rot_deg, s_tr, rng, correlated=True, ref=0):
    """Per-frame body-frame pose error e_i (referenced to frame `ref`=0).
    rw+correlated: shared random walk. rw+independent: each frame drawn from its
    marginal N(0,|i-ref|*Sigma). iid: independent N(0,Sigma) either way."""
    sr = np.deg2rad(s_rot_deg); sig = np.array([sr, sr, sr, s_tr, s_tr, s_tr])
    out = [np.zeros(6) for _ in range(n)]
    if mode == "iid":
        for i in range(n):
            out[i] = rng.normal(0, sig)
    else:  # rw
        if correlated:
            e = np.zeros(6)
            for i in range(n):
                e = e + rng.normal(0, sig)
                out[i] = e.copy()
        else:
            for i in range(n):
                out[i] = rng.normal(0, sig * np.sqrt(max(i, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--grid-step", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--mc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-npz", default="", help="dump per-landmark NEES arrays for QQ/coverage")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    g0 = ds.image(args.start); H, W = g0.shape
    f, cx, cy = md.intrinsics(W, H)
    depth0 = ds.depth(args.start); pose0 = ds.pose(args.start)
    last = min(args.start + args.frames, ds.n)
    T_gt = [ds.pose(i) for i in range(last)]
    K = args.window

    # seed grid landmarks; each keeps its first K in-FOV frames (indices into T_gt)
    lms = []  # (Xgt, [frame indices])
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
    print(f"Mid-Air {args.cond}/traj{ds.traj}: {len(lms)} grid landmarks w/ {K}-frame windows, "
          f"sigma={args.sigma}px, MC={args.mc}")

    modes = [("exact", None, 0, 0), ("iid", "iid", 0.40, 0.060),
             ("rw-imu", "rw", 0.12, 0.025), ("rw-large", "rw", 0.30, 0.060)]
    rng = np.random.default_rng(args.seed)
    dump = {}
    # chi2(1) reference: median 0.455, mean 1.0, P(>3.841)=5%, P(>6.635)=1%
    print("\n1-DOF depth NEES ~ chi2(1) if honest.  ideal: median 0.455 | mean 1.0 | %>3.84 = 5 | %>6.63 = 1")
    for name, mode, sr, st in modes:
        nees_pix, nees_ind, nees_joint, relrot = [], [], [], []
        for Xg, fr in lms:
            n = fr[-1] - fr[0] + 1
            # deployment poses: one correlated realization over the whole span, indexed by frame
            e_dep = perturb_seq(n, mode, sr, st, rng) if mode else [np.zeros(6)] * n
            Tset = {fi: T_gt[fi] @ se3_exp(e_dep[fi - fr[0]]) for fi in fr} if mode else \
                   {fi: T_gt[fi] for fi in fr}
            Xe, A, _us = win_errs(fr, Xg, T_gt, Tset, f, cx, cy, args.sigma, rng)
            if Xe is None or A is None:
                continue
            fl = fr[-1]
            r_est = range_of(Xe, Tset[fl], f, cx, cy); r_true = range_of(Xg, T_gt[fl], f, cx, cy)
            if r_est is None or r_true is None or r_true <= 0:
                continue
            err = r_est - r_true
            gr = range_grad(Xe, Tset[fl], f, cx, cy)
            if gr is None:
                continue
            try:
                cov_pix = np.linalg.inv(A + 1e-9 * np.eye(3))
            except np.linalg.LinAlgError:
                continue
            var_pix = float(gr @ cov_pix @ gr)
            if var_pix <= 0:
                continue
            if mode is None:
                nees_pix.append(err * err / var_pix)
                nees_ind.append(err * err / var_pix)
                nees_joint.append(err * err / var_pix)
                relrot.append(0.0)
                continue
            # (1) analytic per-observation propagation (the Rust r_meas += J*C*J^T term):
            #     R_k = sigma^2 I + Jpose_k C_k Jpose_k^T, C_k added INDEPENDENTLY per obs.
            A_pp = np.zeros((3, 3)); ok = True
            for u, fi in zip(_us, fr):
                _gp, Hk = proj_jac(Xe, Tset[fi], f, cx, cy)
                Jp = pose_jac(Xe, Tset[fi], f, cx, cy)
                if Hk is None or Jp is None:
                    ok = False; break
                Ck = cov6(mode, fi - fr[0], sr, st)
                Rk = args.sigma ** 2 * np.eye(2) + Jp @ Ck @ Jp.T
                A_pp += Hk.T @ np.linalg.inv(Rk) @ Hk
            if not ok:
                continue
            try:
                var_pp = float(gr @ np.linalg.inv(A_pp) @ gr)
            except np.linalg.LinAlgError:
                continue
            # (2) robust MC-joint (correlated resampling = full cross-frame cov; IQR var,
            #     outlier-safe against near-degenerate wrong-pose triangulations):
            rr = []
            for _ in range(args.mc):
                e = perturb_seq(n, mode, sr, st, rng, correlated=True)
                Tm = {fi: T_gt[fi] @ se3_exp(e[fi - fr[0]]) for fi in fr}
                Xm, _ = triangulate(_us, [Tm[fi] for fi in fr], Xg, f, cx, cy, args.sigma)
                if Xm is not None:
                    rv = range_of(Xm, Tm[fr[-1]], f, cx, cy)
                    if rv is not None:
                        rr.append(rv)
            if len(rr) < 8:
                continue
            q1, q3 = np.percentile(rr, [25, 75]); v_joint = ((q3 - q1) / 1.349) ** 2
            if var_pp <= 0 or v_joint <= 0:
                continue
            nees_pix.append(err * err / var_pix)
            nees_ind.append(err * err / var_pp)                 # per-obs analytic (Rust)
            nees_joint.append(err * err / (var_pix + v_joint))  # correlated truth
            # relative-pose error over window (first->last), deployment realization
            eR = np.linalg.inv(Tset[fl]) @ Tset[fr[0]] @ np.linalg.inv(
                np.linalg.inv(T_gt[fl]) @ T_gt[fr[0]])
            relrot.append(np.rad2deg(np.linalg.norm(cv2.Rodrigues(eR[:3, :3])[0])))
        if len(nees_pix) < 20:
            print(f"{name:>13}  too few"); continue
        rr = np.median(relrot) if relrot else 0.0
        print(f"\n{name} (rel {rr:.2f}deg, n={len(nees_pix)})   [median | mean | %>3.84 | %>6.63]")
        for lab, arr in [("pixels-only", nees_pix), ("per-obs(Rust)", nees_ind), ("MC-joint", nees_joint)]:
            a = np.array(arr)
            print(f"    {lab:>13}: {np.median(a):8.3f} | {np.mean(a):9.2f} | "
                  f"{100*np.mean(a > 3.841):5.1f} | {100*np.mean(a > 6.635):5.1f}")
            dump[f"{name}__{lab}"] = a
    if args.save_npz:
        np.savez(args.save_npz, **dump)
        print(f"\nsaved NEES arrays -> {args.save_npz}")
    print("\nWhole-distribution read: median AND mean AND tail (%>3.84 should be ~5) must all match chi2(1);")
    print("a right median with a fat %>3.84 = calibrated centre, heavy tail (the project's recurring pattern).")


if __name__ == "__main__":
    main()
