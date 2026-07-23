"""Where is the fixed-reference SSD minimum, level by level? (coarse-scale staleness test)

first_obs_prevseed.py showed re-seeding the init from the fresh prev tracker barely helps:
prev-seeded reference ~= self-propagated first-obs, both far worse than previous-frame. So
the failure is the fixed reference's OBJECTIVE, not the init. Hypothesis: the fine-level
(L0) SSD min stays near GT (first_obs_iters), but the COARSE pyramid levels -- where the
coarse-to-fine solve commits -- have their min DISPLACED by the appearance/scale change, so
the solve is pulled off GT regardless of init.

This measures, for failing tracks, the SSD-landscape argmin at EACH level (L2/L1/L0, in L0
px vs GT) for:
  reference : the fixed BIRTH patch      (stale)
  fresh     : the PREVIOUS-FRAME patch    (control -- should stay on GT at all levels)

Confirmation = reference L2 argmin displaced (several px) while L0 near GT, AND the fresh
template near GT at every level.

  PY=echo-li-python/venv/bin/python
  $PY first_obs_perlevel.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --scale 0.5 --frames 300 --n-victims 8
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
GRID = np.arange(-6.0, 6.01, 0.5)                 # L0-px search grid around GT


def perlevel_argmin(pyr, tref_levels, gt):
    """For each pyramid level, SSD(reference_level, current@c) over an L0-px grid around GT.
    Returns list of (argmin_offset_L0px (2,), rms_at_argmin, rms_at_gt)."""
    out = []
    for lv in range(LV):
        s = 0.5 ** lv
        tl = tref_levels[lv]
        best = None; rms_gt = None
        for dy in GRID:
            for dx in GRID:
                c = ((gt[0] + dx) * s, (gt[1] + dy) * s)
                Iw = pk.sample(pyr[lv], np.array([c[0]]), np.array([c[1]]), OFFX, OFFY)[0]
                rms = float(np.sqrt(np.mean((Iw - tl) ** 2)))
                if dx == 0 and dy == 0:
                    rms_gt = rms
                if best is None or rms < best[1]:
                    best = (np.array([dx, dy]), rms)
        out.append((best[0], best[1], rms_gt))
    return out


def sample_template(pyr, pos):
    """Build a fresh template pyramid (LV, P) by sampling `pyr` at `pos` per level."""
    return np.stack([pk.sample(pyr[lv], np.array([pos[0] * 0.5 ** lv]),
                               np.array([pos[1] * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])


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
    ap.add_argument("--capture-ages", default="40,80")
    ap.add_argument("--first-thr", type=float, default=5.0)
    ap.add_argument("--prev-thr", type=float, default=1.5)
    ap.add_argument("--n-victims", type=int, default=8)
    args = ap.parse_args()
    cap_ages = set(int(a) for a in args.capture_ages.split(","))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    W, H = ds.image(0).shape[1], ds.image(0).shape[0]
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}; born = {}; first_patch = {}; pos_prev = {}; pos_first = {}
    nid = 0; victims = []; prev = None
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        init_prev = {j: pos_prev[j].copy() for j in pos_prev}     # pre-update prev (fresh template src)
        if prev is not None:
            ppyr, pgx, pgy = prev
            idp = list(pos_prev)
            if idp:
                u, v, _ = pk.klt_track(ppyr, cur_pyr, cgx, cgy,
                                       np.array([pos_prev[j] for j in idp]), R, ITERS, "ssd")
                for k, j in enumerate(idp):
                    if v[k]:
                        pos_prev[j] = u[k]
                    else:
                        pos_prev.pop(j, None)
            idf = list(pos_first)
            if idf:
                tref = np.array([first_patch[j] for j in idf])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_first[j] for j in idf]), R, ITERS,
                                       "blendB", "ssd", 1e6, tref)
                for k, j in enumerate(idf):
                    if v[k]:
                        pos_first[j] = u[k]
                    else:
                        pos_first.pop(j, None)
            for j in list(born):
                gp, rng, _z = md.project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                if not inb or (j not in pos_prev and j not in pos_first):
                    for d in (Xw, born, first_patch, pos_prev, pos_first):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                if (len(victims) < args.n_victims and age in cap_ages
                        and j in pos_first and j in pos_prev and j in init_prev):
                    bf = np.hypot(*(pos_first[j] - gp)); bp = np.hypot(*(pos_prev[j] - gp))
                    if bf > args.first_thr and bp < args.prev_thr:
                        ref_pl = perlevel_argmin(cur_pyr, first_patch[j], gp)
                        fresh_tref = sample_template(ppyr, init_prev[j])   # fresh prev-frame patch
                        fresh_pl = perlevel_argmin(cur_pyr, fresh_tref, gp)
                        victims.append(dict(id=j, frame=i, age=age, bf=bf, bp=bp,
                                            ref=ref_pl, fresh=fresh_pl))
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
                    first_patch[nid] = sample_template(cur_pyr, np.array([x, y]))
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    nid += 1
        prev = (cur_pyr, cgx, cgy)
        if len(victims) >= args.n_victims:
            break

    print(f"\ncaptured {len(victims)} first-obs-loses tracks\n")
    print(f"per-level SSD-argmin displacement from GT (L0 px):  "
          f"L2=coarsest ... L0=finest  [radius sampled +/-6px]")
    print(f"  {'id':>5} {'age':>4} {'1st_beta':>8} | "
          f"{'ref_L2':>7} {'ref_L1':>7} {'ref_L0':>7} | {'fresh_L2':>8} {'fresh_L1':>8} {'fresh_L0':>8}")
    agg = {"ref": [[], [], []], "fresh": [[], [], []]}
    for v in victims:
        rd = [np.hypot(*v["ref"][lv][0]) for lv in range(LV)]     # lv index 0=L0..2=L2
        fd = [np.hypot(*v["fresh"][lv][0]) for lv in range(LV)]
        # print coarse->fine (L2,L1,L0)
        print(f"  {v['id']:>5} {v['age']:>4} {v['bf']:>8.1f} | "
              f"{rd[2]:>7.2f} {rd[1]:>7.2f} {rd[0]:>7.2f} | "
              f"{fd[2]:>8.2f} {fd[1]:>8.2f} {fd[0]:>8.2f}")
        for lv in range(LV):
            agg["ref"][lv].append(rd[lv]); agg["fresh"][lv].append(fd[lv])
    print(f"\n  {'MEDIAN':>10} | "
          f"{np.median(agg['ref'][2]):>7.2f} {np.median(agg['ref'][1]):>7.2f} {np.median(agg['ref'][0]):>7.2f} | "
          f"{np.median(agg['fresh'][2]):>8.2f} {np.median(agg['fresh'][1]):>8.2f} {np.median(agg['fresh'][0]):>8.2f}")
    print("\n  reading: if ref_L2 >> ref_L0 (coarse displaced, fine on GT) AND fresh_* all ~0,")
    print("  the fixed reference is stale at COARSE scale -> coarse-to-fine commits off GT,")
    print("  no init fixes it; a fresh co-moving template stays on GT at every level.")


if __name__ == "__main__":
    main()
