"""Per-landmark fixed-lag SMOOTHER prototype: does a WINDOWED reprojection constraint
make the correspondence drift observable (and robustly rejectable), where the sequential
per-landmark EKF could not? (faithful conduit -> innovation blind; multi-view residual
saw 27%). This tests the "clever cheap middle": batch a landmark over TIME (still O(N),
decoupled per-landmark), fixed GT poses, and measure how the observable-drift signal +
robust-triangulation accuracy scale with WINDOW LENGTH K.

At the LS optimum X*, the post-fit residual is the left-nullspace component of the landmark
Jacobian H_f (orthogonal to its column space): Sum||r_j(X*)||^2/sigma^2 ~ chi2(2K-3) under
the no-drift null. It is the drift the landmark CANNOT absorb as a rigid 3D shift.
  reduced chi2 ~ 1  => absorbed (blind, like the sequential filter)
  reduced chi2 > 1  => drift OBSERVABLE with the windowed constraint

Robust IRLS Student-t (nu=2.6) weight w_j=(nu+2)/(nu+||r_j||^2/sigma^2) down-weights the
geometrically-inconsistent views; covariance from the robust information A=Sum w_j Hj^T Hj/s^2.

Prediction (to be tested, not assumed): coherent bias within a SHORT window looks like a
rigid shift -> blind; observability RISES with K (viewpoint span). If even full-track robust
only recovers ~27% and modest accuracy -> the confounded majority is the irreducible floor.

  PY=echo-li-python/venv/bin/python
  $PY midair_window_smoother_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend  # noqa: E402


def proj(X, T_wb, f, cx, cy):
    gp, _r, z = md.project_world(X, T_wb, f, cx, cy)
    return gp if (gp is not None and z > 0.1) else None


def proj_jac(X, T_wb, f, cx, cy, eps=1e-4):
    gp = proj(X, T_wb, f, cx, cy)
    if gp is None:
        return None, None
    J = np.zeros((2, 3))
    for d in range(3):
        dv = np.zeros(3); dv[d] = eps
        gpp = proj(X + dv, T_wb, f, cx, cy); gpm = proj(X - dv, T_wb, f, cx, cy)
        if gpp is None or gpm is None:
            return gp, None
        J[:, d] = (gpp - gpm) / (2 * eps)
    return gp, J


def triangulate(obs, X0, f, cx, cy, sigma, nu=None, iters=20):
    """(Robust) Gauss-Newton triangulation. nu=None -> plain LS; else Student-t IRLS.
    Returns (X, A, used) where A = Sum w_j Hj^T Hj / sigma^2 is the (robust) info matrix."""
    X = X0.astype(float).copy()
    s2 = sigma * sigma
    A = np.zeros((3, 3))
    used = 0
    for _ in range(iters):
        A = np.zeros((3, 3)); b = np.zeros(3); used = 0
        for T_wb, u in obs:
            gp, J = proj_jac(X, T_wb, f, cx, cy)
            if gp is None or J is None:
                continue
            r = u - gp
            w = 1.0 if nu is None else (nu + 2.0) / (nu + (r @ r) / s2)
            A += w * (J.T @ J) / s2
            b += w * (J.T @ r) / s2
            used += 1
        if used < 2:
            return None
        try:
            dX = np.linalg.solve(A + 1e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
        X += dX
        if np.linalg.norm(dX) < 1e-5:
            break
    return X, A, used


def postfit_chi2(obs, X, f, cx, cy, sigma):
    """Unweighted post-fit residual = nullspace inconsistency. Returns (chi2, dof, n)."""
    ss = 0.0; n = 0
    for T_wb, u in obs:
        gp = proj(X, T_wb, f, cx, cy)
        if gp is None:
            continue
        r = u - gp
        ss += (r @ r) / (sigma * sigma); n += 1
    dof = max(2 * n - 3, 1)
    return ss, dof, n


def raw_drift(obs, Xgt, f, cx, cy):
    d = []
    for T_wb, u in obs:
        gp = proj(Xgt, T_wb, f, cx, cy)
        if gp is not None:
            d.append(np.hypot(*(u - gp)))
    return np.median(d) if d else np.nan


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
    ap.add_argument("--sigma", type=float, default=0.42, help="white-core pixel sigma at work res")
    ap.add_argument("--nu", type=float, default=2.6, help="Student-t dof for robust IRLS")
    ap.add_argument("--windows", default="3,5,8,15,0", help="K values; 0 = full track")
    ap.add_argument("--clean-px", type=float, default=3.0, help="raw-drift px cap for clean-core subset")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.start + args.frames, ds.n)

    # ---- one tracker pass: collect per-landmark obs + birth GT landmark ----
    Xgt = {}; obs = {}
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (2 <= x < W - 2 and 2 <= y < H - 2):
                continue
            if fid not in Xgt:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xgt[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy); obs[fid] = []
                continue
            obs[fid].append((T_wb.copy(), np.array([x, y])))
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] landmarks={len(Xgt)}")

    ks = [int(v) for v in args.windows.split(",")]
    print(f"\n=== {args.cond}/traj{args.traj}: {len(obs)} landmarks, sigma={args.sigma}px, nu={args.nu} ===")
    print("sequential bias-EKF baseline (camera-range abs-3D, comparable): ~1.77%; is windowing better?\n")
    hdr = (f"{'K':>5} {'n_lmk':>6} {'redchi2':>8} {'rho':>6} {'rel3_LS%':>9} {'rel3_rob%':>10} "
           f"{'depth_rob%':>11} {'NEES':>8}   (rel3,depth normalized by CAMERA RANGE)")
    print("--- CLEAN-CORE (median raw-drift < {:.0f}px) ---".format(args.clean_px))
    print(hdr)

    def run_subset(clean):
        for K in ks:
            redchi, drifts, rel_ls, rel_rob, depth_rob, nees = [], [], [], [], [], []
            for fid, ol in obs.items():
                need = K if K > 0 else 2
                if len(ol) < max(need, 4):
                    continue
                win = ol[-K:] if K > 0 else ol
                Xg = Xgt[fid]
                dr = raw_drift(win, Xg, f, cx, cy)
                if not np.isfinite(dr):
                    continue
                if clean and dr > args.clean_px:
                    continue
                if (not clean) and dr <= args.clean_px:
                    continue
                out_ls = triangulate(win, Xg, f, cx, cy, args.sigma, nu=None)
                if out_ls is None:
                    continue
                X_ls, _A_ls, used = out_ls
                chi2, dof, n = postfit_chi2(win, X_ls, f, cx, cy, args.sigma)
                out_rob = triangulate(win, X_ls, f, cx, cy, args.sigma, nu=args.nu)
                if out_rob is None:
                    continue
                X_rob, A_rob, _ = out_rob
                # Obstacle-relevant metric: normalize by RANGE-from-camera at the last window
                # frame, NOT distance-from-world-origin (which deflates the error ~4x here and
                # made an apples-to-oranges comparison vs the sequential EKF's camera-range 1.77%).
                T_last = win[-1][0]
                rng_t = md.project_world(Xg, T_last, f, cx, cy)[1]
                rng_l = md.project_world(X_ls, T_last, f, cx, cy)[1]
                rng_r = md.project_world(X_rob, T_last, f, cx, cy)[1]
                if rng_t is None or rng_l is None or rng_r is None or rng_t <= 0:
                    continue
                redchi.append(chi2 / dof); drifts.append(dr)
                rel_ls.append(np.linalg.norm(X_ls - Xg) / rng_t)   # camera-range 3D err
                rel_rob.append(np.linalg.norm(X_rob - Xg) / rng_t)
                depth_rob.append(abs(rng_r - rng_t) / rng_t)        # radial (obstacle depth) err
                e = X_rob - Xg
                nees.append(float(e @ A_rob @ e))
            if len(redchi) < 10:
                print(f"{('full' if K==0 else K):>5} {len(redchi):>6}  (too few)")
                continue
            rho = spearmanr(redchi, drifts)[0] if len(redchi) > 5 else np.nan
            print(f"{('full' if K==0 else K):>5} {len(redchi):>6} {np.median(redchi):>8.2f} "
                  f"{rho:>6.2f} {100*np.median(rel_ls):>9.2f} {100*np.median(rel_rob):>10.2f} "
                  f"{100*np.median(depth_rob):>11.2f} {np.median(nees):>8.1f}")

    run_subset(clean=True)
    print("\n--- GROSS (median raw-drift >= {:.0f}px: occlusion/swaps) ---".format(args.clean_px))
    print(hdr)
    run_subset(clean=False)
    print("\nredchi2 ~1 => drift absorbed (blind);  >1 => observable with K-view constraint")
    print("relerr_rob < relerr_LS => robust rejection helps;  NEES_rob ~3 => honest windowed cov")


if __name__ == "__main__":
    main()
