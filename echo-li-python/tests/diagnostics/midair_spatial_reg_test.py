"""DIS-like SPARSE spatial regularization for the aperture drift. The along-edge
(weak structure-tensor) direction is unconstrained by a feature's own patch, but a
same-surface neighbor constrains it. Test: replace each feature's WEAK-direction flow
component with the (depth-gated) neighbor-smoothed flow, keep the STRONG (data-
constrained) component; does the per-frame drift increment shrink?

Decisive question: is the drift a LOCAL anomaly (deviates from neighbors -> regularizable)
or a SHARED field (neighbors drift together -> smoothing does nothing)?

  PY=echo-li-python/venv/bin/python
  $PY midair_spatial_reg_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend, structure_tensor_fields, eig_sym2  # noqa: E402


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
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--depth-gate", type=float, default=0.1, help="rel depth diff to couple neighbors")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(getattr(fcfg, "klt_window", 7))
    last = min(args.start + args.frames, ds.n)

    Xw = {}; born = {}; prev_pos = {}
    rows = []  # raw_incr, reg_incr, lambda_min, drift_mag, n_same_depth_neighbors
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep).astype(np.float32) if prep is not None else img.astype(np.float32)
        sxx, sxy, syy = structure_tensor_fields(prep, r)

        cur = {}
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (2 <= x < W - 2 and 2 <= y < H - 2):
                continue
            cur[fid] = np.array([x, y])
            if fid not in born:
                gx, gy = int(round(x)), int(round(y)); d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xw[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy); born[fid] = i

        ids = [j for j in cur if j in prev_pos]
        if len(ids) > args.knn + 1:
            P = np.array([cur[j] for j in ids])
            flow = np.array([cur[j] - prev_pos[j] for j in ids])
            dep = np.array([float(depth[int(round(min(max(cur[j][1], 0), H - 1))),
                                         int(round(min(max(cur[j][0], 0), W - 1)))]) for j in ids])
            tree = cKDTree(P)
            for a, j in enumerate(ids):
                if j not in Xw:
                    continue
                gp, _rng, _z = md.project_world(Xw[j], T_wb, f, cx, cy)
                if gp is None:
                    continue
                true_flow = gp - prev_pos[j]
                xi, yi = int(round(cur[j][0])), int(round(cur[j][1]))
                lmin, lmax, ux, uy, vx, vy = eig_sym2(float(sxx[yi, xi]), float(sxy[yi, xi]), float(syy[yi, xi]))
                # depth-gated k-NN -> fit a local AFFINE flow field v = A[x,y,1]
                # (accounts for the smoothly-varying true flow; constant-average would blur it)
                d_, nb = tree.query(P[a], k=min(args.knn + 1, len(ids)))
                pn = []; fn = []
                for b in nb[1:]:
                    if abs(dep[b] - dep[a]) / max(dep[a], 1e-3) < args.depth_gate:
                        pn.append(P[b]); fn.append(flow[b])
                if len(fn) < 4:
                    continue
                Xmat = np.column_stack([np.array(pn), np.ones(len(pn))])
                AT, *_ = np.linalg.lstsq(Xmat, np.array(fn), rcond=None)   # 3x2
                v_smooth = np.array([cur[j][0], cur[j][1], 1.0]) @ AT      # predicted flow at feature
                v_raw = flow[a]
                # anisotropic: keep strong (data) dir of raw, take weak (aperture) dir from neighbors
                strong = np.array([vx, vy]); weak = np.array([ux, uy])
                v_reg = (v_raw @ strong) * strong + (v_smooth @ weak) * weak
                raw_incr = np.hypot(*(v_raw - true_flow))
                reg_incr = np.hypot(*(v_reg - true_flow))
                drift = np.hypot(*(cur[j] - gp))
                rows.append((raw_incr, reg_incr, lmin, drift, len(fn)))
        prev_pos = cur
        if (i - args.start) % 100 == 0:
            print(f"  [{i-args.start}/{last-args.start}] obs={len(rows)}")

    A = np.array(rows, float)
    raw, reg, lmin, drift = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    print(f"\n=== {args.cond}/traj{args.traj}: {len(A)} feature-frames (depth-gated knn={args.knn}) ===")
    print("does anisotropic neighbor-regularization reduce the per-frame drift increment?")
    print(f"\n  {'group':>22} {'n':>7} {'raw incr':>9} {'reg incr':>9} {'reg/raw':>8}")
    def row(tag, m):
        if m.sum() > 30:
            print(f"  {tag:>22} {int(m.sum()):7d} {np.median(raw[m]):9.3f} {np.median(reg[m]):9.3f} "
                  f"{np.median(reg[m])/max(np.median(raw[m]),1e-9):8.2f}")
    row("ALL", np.ones(len(A), bool))
    lo_l = lmin < np.percentile(lmin, 33)
    row("edge-like (low lmin)", lo_l)
    row("corner-like (high lmin)", lmin > np.percentile(lmin, 67))
    row("high-drift (>1px)", drift > 1.0)
    row("high-drift edge", (drift > 1.0) & lo_l)
    print("\n  reg/raw < 1 => spatial regularization REDUCES the drift (local anomaly, regularizable)")
    print("  reg/raw ~ 1 => drift is a SHARED field (neighbors drift together, smoothing can't help)")


if __name__ == "__main__":
    main()
