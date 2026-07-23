"""Is the per-track correspondence-drift rate (the physical random-walk `q`)
PREDICTABLE from observable cues, without GT? If yes, an adaptive q(cue) is a
physical, transferable model; if no, q is an irreducible per-track random effect
that can only be conservatively bounded. This is the AUC cue-ranking test
(cf. tail_cue_analysis.py) applied to MidAir exact GT, with a CONTINUOUS target
binarized to "high-drift track".

Per obs (prev->cur) it measures the two-frame residual vs exact GT
    err_k = u_rudolf,k - project_exact(X_{prev}, T_k)
and records observable cues. Per track it forms:
    target  drift_rate = || mean_k err_k ||   (coherent per-frame drift, px/frame)
    cues    (median over the track, all observable WITHOUT GT):
            lambda_min, condition (aperture), flow_mag, omega (motion), range,
            plus any numeric Rudolf per-feature fields (klt_quality, lbp_distance...)
Then AUC of each cue for the label "drift_rate in top quartile", sign-adjusted.

  PY=echo-li-python/venv/bin/python
  $PY midair_drift_cue_auc.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml \
      --save-npz /tmp/cue_sunny.npz
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata

import echo_li

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import (  # noqa: E402
    structure_tensor_fields, eig_sym2, project_world_fast, make_frontend)

# cue -> expected sign of correlation with drift (for readable sign-adjusted AUC)
CUES = ["lambda_min", "condition", "flow_mag", "omega", "range"]


def auc(signal, label):
    """Rank AUC of `signal` for the binary `label`; sign-adjusted to >=0.5."""
    g = np.isfinite(signal)
    s, y = signal[g], label[g].astype(float)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos < 5 or nneg < 5:
        return np.nan, 0
    r = rankdata(s)
    a = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return max(a, 1 - a), int(npos)


def run_slice(args, cond, traj):
    ds = md.MidAir(args.root, args.subset, cond, traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(args.window or getattr(fcfg, "klt_window", 7))
    last = min(args.start + args.frames, ds.n)

    # per-track accumulators
    err_sum = {}       # fid -> [sum ex, sum ey, n]
    cue_acc = {}       # fid -> {cue: [values]}
    extra_keys = set()
    prev = None
    t0 = time.time()
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i)
        T_wb = ds.pose(i); T_bw = np.linalg.inv(T_wb)
        feats, _ = tracker.process(img)
        cur = {}
        fd_extra = {}
        for fd in feats:
            fid = int(fd["id"])
            cur[fid] = np.array([float(fd["x"]), float(fd["y"])])
            ex = {k: float(v) for k, v in fd.items()
                  if k not in ("id", "x", "y") and isinstance(v, (int, float))
                  and np.isfinite(float(v))}
            fd_extra[fid] = ex
            extra_keys.update(ex.keys())
        # tracker's own per-track quality metadata (klt_quality/lbp_distance/reservoir_score)
        for m in tracker.track_meta():
            fid = int(m["id"])
            if fid not in fd_extra:
                continue
            for k in ("klt_quality", "lbp_distance", "reservoir_score"):
                v = m.get(k)
                if v is not None and np.isfinite(float(v)):
                    fd_extra[fid][k] = float(v)
                    extra_keys.add(k)
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep) if prep is not None else cv2.equalizeHist(img)
        omega = float(ds.omega_mag(i))

        if prev is not None:
            prev_cur, prev_depth, prev_T_wb, prev_prep = prev
            sxx, sxy, syy = structure_tensor_fields(prev_prep, r)
            for fid, p0 in prev_cur.items():
                p1 = cur.get(fid)
                if p1 is None:
                    continue
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                if not (args.border <= p0[0] < W - args.border
                        and args.border <= p0[1] < H - args.border
                        and args.border <= p1[0] < W - args.border
                        and args.border <= p1[1] < H - args.border):
                    continue
                d0 = float(prev_depth[y0, x0])
                if not (1.0 < d0 < md.SKY):
                    continue
                Xw = md.backproject_world((float(p0[0]), float(p0[1])), d0, prev_T_wb, f, cx, cy)
                gp, rng, _z = project_world_fast(Xw, T_bw, f, cx, cy)
                if gp is None:
                    continue
                err = p1 - gp
                lmin, lmax, *_ = eig_sym2(float(sxx[y0, x0]), float(sxy[y0, x0]), float(syy[y0, x0]))
                flow = float(np.hypot(*(p1 - p0)))
                s = err_sum.setdefault(fid, np.zeros(3))
                s[0] += err[0]; s[1] += err[1]; s[2] += 1
                ca = cue_acc.setdefault(fid, {c: [] for c in CUES})
                ca["lambda_min"].append(lmin); ca["condition"].append(lmax / max(lmin, 1e-9))
                ca["flow_mag"].append(flow); ca["omega"].append(omega); ca["range"].append(rng)
                for k, v in fd_extra.get(fid, {}).items():
                    ca.setdefault(k, []).append(v)
        prev = (cur, depth, T_wb, prep)
        if (i - args.start) % 100 == 0:
            print(f"  [{cond}/{traj} {i-args.start:4d}/{last-args.start}] tracks={len(cur):4d} "
                  f"{(i-args.start+1)/max(time.time()-t0,1e-9):.1f} fps")

    all_keys = CUES + sorted(extra_keys)
    rows = []
    for fid, s in err_sum.items():
        n = int(s[2])
        if n < args.min_obs:
            continue
        drift_rate = float(np.hypot(s[0] / n, s[1] / n))   # coherent per-frame drift
        ca = cue_acc[fid]
        cvals = [np.median(ca[k]) if ca.get(k) else np.nan for k in all_keys]
        rows.append([fid, n, drift_rate] + cvals)
    return np.array(rows, float), all_keys


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
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--top-frac", type=float, default=0.25, help="top-quantile drift = 'high-drift' label")
    ap.add_argument("--save-npz", default="")
    args = ap.parse_args()

    A, keys = run_slice(args, args.cond, args.traj)
    if args.save_npz:
        np.savez(args.save_npz, rows=A, cols=np.array(["fid", "n", "drift_rate"] + keys))
        print(f"saved {args.save_npz}")
    dr = A[:, 2]
    thr = np.quantile(dr, 1 - args.top_frac)
    label = (dr >= thr).astype(float)
    print(f"\n=== {args.cond}/traj{args.traj}: {len(A)} tracks (>= {args.min_obs} obs), "
          f"drift_rate med={np.median(dr):.3f} p90={np.percentile(dr,90):.3f} px/frame ===")
    print(f"label 'high-drift' = top {args.top_frac:.0%} (drift_rate >= {thr:.3f} px/frame)")
    print(f"\n  {'cue':>14} {'AUC':>6}   (0.5=useless, 1=perfect discriminator)")
    res = []
    for j, k in enumerate(keys):
        a, npos = auc(A[:, 3 + j], label)
        res.append((k, a))
    for k, a in sorted(res, key=lambda z: -(z[1] if np.isfinite(z[1]) else 0)):
        print(f"  {k:>14} {a:6.3f}")


if __name__ == "__main__":
    main()
