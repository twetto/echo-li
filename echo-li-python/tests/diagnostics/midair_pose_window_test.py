"""Estimated-pose robustness of the per-landmark windowed smoother. All prior smoother
results used EXACT GT poses; pose uncertainty is the standing TODO. Here we re-triangulate
each landmark's window with PERTURBED poses and measure the obstacle-relevant metric:
relative DEPTH (range-from-camera) error, which cancels global frame drift and is bitten
only by RELATIVE pose error within the window.

Two pose-error structures (this is the IMU point):
  imu-like  = random-walk pose error (accurate RELATIVE over a short window, drifts absolute)
  iid       = independent per-frame pose error (bad relative geometry, no inertial anchor)
If imu-like degrades the windowed depth far less than iid, the smoother needs good LOCAL
pose (which IMU provides), not global accuracy.

SCOPE: fixed (estimated) poses -> measures how pose error degrades windowed depth. It does
NOT test "IMU breaks the correspondence confound" (that needs FREE poses in a joint BA); the
per-landmark confound (landmark absorbs the 73% rigid-shift drift) is pose-accuracy-invariant.

  PY=echo-li-python/venv/bin/python
  $PY midair_pose_window_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --window 15
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend  # noqa: E402


def se3_exp(xi):
    R = cv2.Rodrigues(np.asarray(xi[:3], float))[0]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xi[3:6]
    return T


def perturb(T_gt, mode, s_rot_deg, s_tr, rng):
    """Right (body-frame) perturbation T_est[i] = T_gt[i] @ Exp(e_i); rw accumulates e."""
    sr = np.deg2rad(s_rot_deg)
    sig = np.array([sr, sr, sr, s_tr, s_tr, s_tr])
    out = []; e = np.zeros(6)
    for T in T_gt:
        if mode == "rw":
            e = e + rng.normal(0, sig)
            d = e
        else:
            d = rng.normal(0, sig)
        out.append(T @ se3_exp(d))
    return out


def proj_jac(X, T, f, cx, cy, eps=1e-4):
    gp, _r, z = md.project_world(X, T, f, cx, cy)
    if gp is None or z <= 0.1:
        return None, None
    J = np.zeros((2, 3))
    for d in range(3):
        dv = np.zeros(3); dv[d] = eps
        a, _, za = md.project_world(X + dv, T, f, cx, cy)
        b, _, zb = md.project_world(X - dv, T, f, cx, cy)
        if a is None or b is None:
            return gp, None
        J[:, d] = (a - b) / (2 * eps)
    return gp, J


def triangulate(win, Tlist, X0, f, cx, cy, sigma, nu=2.6, iters=20):
    X = X0.astype(float).copy(); s2 = sigma * sigma; A = np.zeros((3, 3)); used = 0
    for _ in range(iters):
        A = np.zeros((3, 3)); b = np.zeros(3); used = 0
        for fi, u in win:
            gp, J = proj_jac(X, Tlist[fi], f, cx, cy)
            if gp is None or J is None:
                continue
            r = u - gp
            w = (nu + 2.0) / (nu + (r @ r) / s2)
            A += w * (J.T @ J) / s2; b += w * (J.T @ r) / s2; used += 1
        if used < 2:
            return None
        try:
            X = X + np.linalg.solve(A + 1e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
    # post-fit unweighted chi2 (nullspace inconsistency), dof = 2*used - 3
    ss = 0.0; nfit = 0
    for fi, u in win:
        gp, _r, z = md.project_world(X, Tlist[fi], f, cx, cy)
        if gp is not None and z > 0.1:
            ss += ((u - gp) @ (u - gp)) / s2; nfit += 1
    return X, ss / max(2 * nfit - 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--config", default=str(repo / "configs" / "diagnostics_midair_sparse3d.yaml"))
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.42)
    ap.add_argument("--clean-px", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.start + args.frames, ds.n)

    T_gt = [None] * last
    Xgt = {}; obs = {}
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i); T_gt[i] = T
        feats, _ = tracker.process(img)
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (2 <= x < W - 2 and 2 <= y < H - 2):
                continue
            if fid not in Xgt:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xgt[fid] = md.backproject_world((x, y), d0, T, f, cx, cy); obs[fid] = []
                continue
            obs[fid].append((i, np.array([x, y])))
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] landmarks={len(Xgt)}")

    K = args.window
    modes = [("exact", None, 0, 0), ("imu-like", "rw", 0.02, 0.005),
             ("imu-worse", "rw", 0.05, 0.010), ("poor-iid", "iid", 0.10, 0.020),
             ("bad-iid", "iid", 0.30, 0.050)]
    rng = np.random.default_rng(args.seed)

    print(f"\n=== {args.cond}/traj{ds.traj}: windowed robust triangulation, K={K}, "
          f"clean-core drift<{args.clean_px}px ===")
    print("obstacle metric = |range_est - range_true|/range_true at latest window frame\n")
    print(f"{'pose mode':>11} {'relpose rot/tr':>16} {'depth-err% med/p90':>20} {'nullchi2':>9} {'n':>6}")
    for name, mode, sr, st in modes:
        Test = list(T_gt) if mode is None else perturb(T_gt, mode, sr, st, rng)
        derr, chi2s, relrot, reltr = [], [], [], []
        for fid, ol in obs.items():
            if len(ol) < K:
                continue
            win = ol[-K:]
            Xg = Xgt[fid]
            dr = np.median([np.hypot(*(u - md.project_world(Xg, T_gt[fi], f, cx, cy)[0]))
                            for fi, u in win if md.project_world(Xg, T_gt[fi], f, cx, cy)[0] is not None])
            if not np.isfinite(dr) or dr > args.clean_px:
                continue
            out = triangulate(win, Test, Xg, f, cx, cy, args.sigma)
            if out is None:
                continue
            Xe, chi2 = out
            fi_last = win[-1][0]
            _p, rng_e, _z = md.project_world(Xe, Test[fi_last], f, cx, cy)
            _p2, rng_t, _z2 = md.project_world(Xg, T_gt[fi_last], f, cx, cy)
            if rng_e is None or rng_t is None or rng_t <= 0:
                continue
            derr.append(abs(rng_e - rng_t) / rng_t); chi2s.append(chi2)
            # injected relative-pose error over the window (first->last)
            fi0 = win[0][0]
            rel_gt = np.linalg.inv(T_gt[fi_last]) @ T_gt[fi0]
            rel_e = np.linalg.inv(Test[fi_last]) @ Test[fi0]
            eR = np.linalg.inv(rel_gt) @ rel_e
            relrot.append(np.rad2deg(np.linalg.norm(cv2.Rodrigues(eR[:3, :3])[0])))
            reltr.append(np.linalg.norm(eR[:3, 3]))
        if len(derr) < 20:
            print(f"{name:>11}  too few"); continue
        print(f"{name:>11} {np.median(relrot):5.3f}d/{np.median(reltr):5.3f}m "
              f"  {100*np.median(derr):7.2f}/{100*np.percentile(derr,90):6.2f}     "
              f"{np.median(chi2s):8.2f} {len(derr):6d}")
    print("\nimu-like ~ exact => smoother robust to SLOW pose drift (needs good LOCAL pose = IMU).")
    print("nullchi2 rising with pose error => pose inconsistency CONTAMINATES the drift-observability signal.")


if __name__ == "__main__":
    main()
