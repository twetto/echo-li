"""Does MORE CONSTRAINT make the drift visible? The sequential monocular update
absorbs each 2D measurement into the 3D landmark (faithful conduit -> innovation
~0.1px). But a static landmark over-determined by MANY poses can only absorb the
drift that looks like a rigid 3D shift; the rest becomes a multi-view reprojection
residual that robust methods CAN act on. Rotation should make the aperture drift
non-absorbable (the edge/ambiguous direction rotates frame-to-frame).

Test: batch-triangulate each landmark from its (drifted) observations + GT poses,
then compare the batch reprojection residual to the raw drift-vs-truth. Ratio ~1 =>
drift is VISIBLE with multi-view constraint (blocker broken); ratio ~0 => absorbed.

  PY=echo-li-python/venv/bin/python
  $PY midair_multiview_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
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


def triangulate(obs, X0, f, cx, cy, iters=10):
    """Gauss-Newton batch triangulation minimizing sum||proj_k(X)-u_k||^2."""
    X = X0.astype(float).copy()
    eps = 1e-4
    for _ in range(iters):
        H = np.zeros((3, 3)); b = np.zeros(3); used = 0
        for T_wb, u in obs:
            gp = proj(X, T_wb, f, cx, cy)
            if gp is None:
                continue
            J = np.zeros((2, 3))
            for d in range(3):
                dv = np.zeros(3); dv[d] = eps
                gpp = proj(X + dv, T_wb, f, cx, cy); gpm = proj(X - dv, T_wb, f, cx, cy)
                if gpp is None or gpm is None:
                    J = None; break
                J[:, d] = (gpp - gpm) / (2 * eps)
            if J is None:
                continue
            r = u - gp
            H += J.T @ J; b += J.T @ r; used += 1
        if used < 2:
            return None
        try:
            dX = np.linalg.solve(H + 1e-6 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
        X += dX
        if np.linalg.norm(dX) < 1e-4:
            break
    return X


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
    ap.add_argument("--min-obs", type=int, default=12)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.start + args.frames, ds.n)

    Xgt = {}; born = {}; obs = {}
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (2 <= x < W - 2 and 2 <= y < H - 2):
                continue
            if fid not in born:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xgt[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy); born[fid] = i
                    obs[fid] = []
                continue
            obs[fid].append((T_wb.copy(), np.array([x, y])))
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] landmarks={len(born)}")

    raw, batch, err_gt, err_batch, drift_of, resid_of = [], [], [], [], [], []
    for fid, ol in obs.items():
        if len(ol) < args.min_obs:
            continue
        Xb = triangulate(ol, Xgt[fid], f, cx, cy)
        if Xb is None:
            continue
        rr, br = [], []
        for T_wb, u in ol:
            g_gt = proj(Xgt[fid], T_wb, f, cx, cy); g_b = proj(Xb, T_wb, f, cx, cy)
            if g_gt is None or g_b is None:
                continue
            rr.append(np.hypot(*(u - g_gt)))   # raw drift vs truth
            br.append(np.hypot(*(u - g_b)))    # multi-view residual vs batch fit
            drift_of.append(rr[-1]); resid_of.append(br[-1])
        if rr:
            raw.append(np.median(rr)); batch.append(np.median(br))
            # camera-range normalization (not distance-from-origin, which deflates ~4x)
            rng_t = md.project_world(Xgt[fid], ol[-1][0], f, cx, cy)[1]
            if rng_t and rng_t > 0:
                err_gt.append(np.linalg.norm(Xb - Xgt[fid]) / rng_t)

    raw = np.array(raw); batch = np.array(batch)
    print(f"\n=== {args.cond}/traj{args.traj}: {len(raw)} landmarks (>= {args.min_obs} obs) ===")
    print(f"sequential innovation (filter vs frontend) ~ 0.10 px  (drift invisible there)")
    print(f"raw drift vs truth:        median {np.median(raw):.3f} px")
    print(f"multi-view batch residual: median {np.median(batch):.3f} px")
    print(f"  ratio batch_resid / raw_drift = {np.median(batch)/max(np.median(raw),1e-9):.2f}  "
          f"(~1 = drift VISIBLE with multi-view; ~0 = absorbed)")
    rho, _ = spearmanr(resid_of, drift_of)
    print(f"  per-obs spearman(batch residual, raw drift) = {rho:+.3f}  (>0 = residual reveals drift)")
    print(f"  batch landmark error vs GT: median {100*np.median(err_gt):.2f}%  "
          f"(camera-range normalized; does over-determination pull it toward truth?)")


if __name__ == "__main__":
    main()
