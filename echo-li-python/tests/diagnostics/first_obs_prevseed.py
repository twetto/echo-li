"""Test the first_obs_iters.py prediction: first-obs drift is self-inflicted by seeding
the reference-template search from its OWN drifted history; the SSD objective minimum
stays at GT, so re-seeding each frame from the fresh previous-frame tracker (in-basin)
should let Gauss-Newton converge back to truth and stop the drift.

Three self/re-seeded trackers on Mid-Air exact GT, same births/anchors:
  prev    : previous-frame template, init = own prev position     (baseline good)
  first   : fixed birth reference,   init = own drifted history    (baseline bad)
  pseed   : fixed birth reference,   init = the fresh PREV tracker (the test)

pseed is NOT self-propagated -- every frame it is re-initialised at pos_prev and refined
by the birth reference. Prediction: beta_pseed stays small/flat with age (like prev),
NOT runaway (like first); and on tracks where first has ALREADY lost, pseed recovers
truth (it ignores first's drifted state).

  PY=echo-li-python/venv/bin/python
  $PY first_obs_prevseed.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 0 --scale 0.5 --frames 300
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photometric_klt_ab as pk  # noqa: E402
import midair_drift as md  # noqa: E402

LV, R, ITERS = md.LV, md.R, md.ITERS
OFFX = np.repeat(np.arange(-R, R + 1, dtype=np.float32), 2 * R + 1)
OFFY = np.tile(np.arange(-R, R + 1, dtype=np.float32), 2 * R + 1)
LAM = 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--redetect", type=int, default=250)
    ap.add_argument("--out", default="midair_prevseed")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    W, H = ds.image(0).shape[1], ds.image(0).shape[0]
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}; born = {}; first_patch = {}; pos_prev = {}; pos_first = {}
    nid = 0
    rows = []      # age, beta_prev, beta_first, beta_pseed, first_lost
    prev = None
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        if prev is not None:
            ppyr, pgx, pgy = prev
            # previous-frame KLT (self, drop on invalid)
            idp = list(pos_prev)
            if idp:
                u, v, _ = pk.klt_track(ppyr, cur_pyr, cgx, cgy,
                                       np.array([pos_prev[j] for j in idp]), R, ITERS, "ssd")
                for k, j in enumerate(idp):
                    if v[k]:
                        pos_prev[j] = u[k]
                    else:
                        pos_prev.pop(j, None)
            # first-observation KLT (self, drop on invalid)
            idf = list(pos_first)
            if idf:
                tref = np.array([first_patch[j] for j in idf])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_first[j] for j in idf]), R, ITERS,
                                       "blendB", "ssd", LAM, tref)
                for k, j in enumerate(idf):
                    if v[k]:
                        pos_first[j] = u[k]
                    else:
                        pos_first.pop(j, None)
            # prev-SEEDED reference KLT (re-init at fresh pos_prev, refine by birth reference;
            # NOT stored -- recomputed every frame)
            pseed = {}
            ids = [j for j in pos_prev if j in first_patch]
            if ids:
                tref = np.array([first_patch[j] for j in ids])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_prev[j] for j in ids]), R, ITERS,
                                       "blendB", "ssd", LAM, tref)
                for k, j in enumerate(ids):
                    if v[k]:
                        pseed[j] = u[k]
            # exact GT + the three betas
            for j in list(born):
                gp, rng, _z = md.project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                if not inb or (j not in pos_prev and j not in pos_first):
                    for d in (Xw, born, first_patch, pos_prev, pos_first):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                bp = np.hypot(*(pos_prev[j] - gp)) if j in pos_prev else np.nan
                bf = np.hypot(*(pos_first[j] - gp)) if j in pos_first else np.nan
                bs = np.hypot(*(pseed[j] - gp)) if j in pseed else np.nan
                rows.append((age, bp, bf, bs, float(bf > 3) if np.isfinite(bf) else np.nan))
        # births
        if len(pos_prev) < args.redetect:
            mask = np.uint8((depth < md.SKY) & (depth > 1.0)) * 255
            for j in pos_prev:
                p = pos_prev[j]; cv2.circle(mask, (int(p[0]), int(p[1])), R + 2, 0, -1)
            corners = cv2.goodFeaturesToTrack(img, args.max_tracks - len(pos_prev),
                                              0.01, 2 * R + 3, mask=mask)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    x, y = float(c[0]), float(c[1]); gx, gy = int(round(x)), int(round(y))
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < md.SKY):
                        continue
                    Xw[nid] = md.backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = np.stack(
                        [pk.sample(cur_pyr[lv], np.array([x * 0.5 ** lv]),
                                   np.array([y * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    nid += 1
        prev = (cur_pyr, cgx, cgy)
        if i % 100 == 0:
            print(f"  [{i}/{min(args.frames, ds.n)}] active={len(pos_prev)} obs={len(rows)}")

    A = np.array(rows, float)
    np.savez(f"{args.out}_{args.cond}.npz", rows=A,
             cols=np.array("age beta_prev beta_first beta_pseed first_lost".split()))
    print(f"\n=== prev-seeded reference test, {args.cond} ({len(A)} obs) ===")
    print(f"  {'age':>8} {'n':>7} {'prev':>8} {'first':>8} {'PSEED':>8}   (median |beta| px)")
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            print(f"  {lo:3d}-{hi:<4d} {int(m.sum()):7d} "
                  f"{np.nanmedian(A[m,1]):8.3f} {np.nanmedian(A[m,2]):8.3f} {np.nanmedian(A[m,3]):8.3f}")
    # does pseed RECOVER tracks where first has already lost (>3px)?
    lost = np.isfinite(A[:, 2]) & (A[:, 2] > 3) & np.isfinite(A[:, 3])
    if lost.sum() > 20:
        print(f"\n  on first-obs-LOST obs (beta_first>3px, n={int(lost.sum())}): "
              f"median beta_first {np.nanmedian(A[lost,2]):.2f}  ->  beta_PSEED {np.nanmedian(A[lost,3]):.2f}px")
        print(f"  pseed recovers to <1px on {100*np.mean(A[lost,3] < 1):.0f}% of them, "
              f"<2px on {100*np.mean(A[lost,3] < 2):.0f}%")
    # tail: p90 by age
    print(f"\n  p90 |beta| (the drift tail):")
    print(f"  {'age':>8} {'prev':>8} {'first':>8} {'PSEED':>8}")
    for lo, hi in [(20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            print(f"  {lo:3d}-{hi:<4d} {np.nanpercentile(A[m,1],90):8.2f} "
                  f"{np.nanpercentile(A[m,2],90):8.2f} {np.nanpercentile(A[m,3],90):8.2f}")


if __name__ == "__main__":
    main()
