"""End-to-end test of the coarse-fresh / fine-fixed HYBRID tracker (no pose, no warp).

first_obs_perlevel.py: the fresh previous-frame template is on GT at EVERY pyramid level;
the fixed birth reference is on GT only at the FINE level (coarse displaced ~3.5px).
first_obs_warptest.py: that coarse displacement is mostly geometric, and the fresh template
supplies the correct coarse content for free.

Hybrid idea: run coarse pyramid levels (L2,L1) against the FRESH previous-frame template
(correct basin) and let the fine level (L0) include the FIXED birth reference (long-term
identity pressure, its L0 min is often near GT). Combines previous-frame's basin with
first-obs's anti-drift, with no pose / warp / back-end.

Self-propagating trackers, same births/anchors, exact-GT beta vs age:
  prev   : previous-frame template, all levels          (drifts slowly ~2px@80)
  first  : fixed birth reference, all levels            (runs away ~12px@80)
  Hλ     : prev template on L2/L1, blend(prev, λ*birth-ref) on L0

λ=inf is the original hard switch. Finite λ tests whether gentler reference pressure helps
without letting the stale reference objective dominate.

  PY=echo-li-python/venv/bin/python
  $PY first_obs_hybrid.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --scale 0.5 --frames 300
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
_off = np.arange(-R, R + 1, dtype=np.float32)
OFFX = np.repeat(_off, len(_off)); OFFY = np.tile(_off, len(_off))


def klt_hybrid(ppyr, cur_pyr, cgx, cgy, p0, tref, r, iters, fine_ref_level=0, fine_lam=np.inf):
    """Batched per-level-template translation KLT. Coarse levels (lv > fine_ref_level) use
    the FRESH previous-frame patch (sampled from ppyr at p0); the fine level(s) blend in
    the stored birth reference tref[:,lv,:]. Forward-additive, current-frame gradients."""
    n = len(p0)
    u = p0.copy().astype(np.float64)
    valid = np.ones(n, bool)
    for lv in reversed(range(LV)):
        s = 0.5 ** lv
        Pl, Cl, Gx, Gy = ppyr[lv], cur_pyr[lv], cgx[lv], cgy[lv]
        c0x = p0[:, 0] * s; c0y = p0[:, 1] * s
        Tprev = pk.sample(Pl, c0x, c0y, OFFX, OFFY)          # fresh prev-frame template
        if lv > fine_ref_level or fine_lam <= 0.0:
            T = Tprev
        elif np.isfinite(fine_lam):
            T = (Tprev + fine_lam * tref[:, lv, :]) / (1.0 + fine_lam)
        else:
            T = tref[:, lv, :]                                # hard fixed birth reference
        ux = u[:, 0] * s; uy = u[:, 1] * s
        for _ in range(iters):
            Iw = pk.sample(Cl, ux, uy, OFFX, OFFY)
            jx = pk.sample(Gx, ux, uy, OFFX, OFFY)
            jy = pk.sample(Gy, ux, uy, OFFX, OFFY)
            res = Iw - T
            Hxx = np.sum(jx * jx, 1); Hxy = np.sum(jx * jy, 1); Hyy = np.sum(jy * jy, 1)
            bx = -np.sum(jx * res, 1); by = -np.sum(jy * res, 1)
            reg = 1e-3 * (Hxx + Hyy + 1e-6); Hxx = Hxx + reg; Hyy = Hyy + reg
            det = Hxx * Hyy - Hxy * Hxy
            ok = np.abs(det) > 1e-6
            dx = np.where(ok, (Hyy * bx - Hxy * by) / np.where(ok, det, 1), 0.0)
            dy = np.where(ok, (Hxx * by - Hxy * bx) / np.where(ok, det, 1), 0.0)
            step = np.hypot(dx, dy); scl = np.where(step > 1.0, 1.0 / np.maximum(step, 1e-12), 1.0)
            ux = ux + dx * scl; uy = uy + dy * scl
        u[:, 0] = ux / s; u[:, 1] = uy / s
    hh, ww = ppyr[0].shape
    valid &= (u[:, 0] > r) & (u[:, 0] < ww - r) & (u[:, 1] > r) & (u[:, 1] < hh - r)
    return u, valid


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
    ap.add_argument("--fine-lams", default="0.05,0.10,0.25,0.50,1.00,inf",
                    help="comma-separated reference weights used at fine levels; inf=hard ref")
    ap.add_argument("--out", default="midair_hybrid")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    fine_lams = []
    for tok in args.fine_lams.split(","):
        tok = tok.strip().lower()
        fine_lams.append(np.inf if tok in ("inf", "infinity") else float(tok))
    labels = [f"H{lam:g}" if np.isfinite(lam) else "Hinf" for lam in fine_lams]

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    W, H = ds.image(0).shape[1], ds.image(0).shape[0]
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}; born = {}; fp = {}; pos_prev = {}; pos_first = {}
    pos_hyb = {label: {} for label in labels}
    nid = 0; rows = []; prev = None
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        if prev is not None:
            ppyr, pgx, pgy = prev
            modes = [("prev", pos_prev, "ssd", None),
                     ("first", pos_first, "blendB", None)]
            modes += [(label, pos_hyb[label], "hybrid", lam)
                      for label, lam in zip(labels, fine_lams)]
            for name, pos, kind, lam in modes:
                ids = list(pos)
                if not ids:
                    continue
                p0 = np.array([pos[j] for j in ids])
                if kind == "ssd":
                    u, v, _ = pk.klt_track(ppyr, cur_pyr, cgx, cgy, p0, R, ITERS, "ssd")
                elif kind == "blendB":
                    tref = np.array([fp[j] for j in ids])
                    u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy, p0, R, ITERS,
                                           "blendB", "ssd", 1e6, tref)
                else:
                    tref = np.array([fp[j] for j in ids])
                    u, v = klt_hybrid(ppyr, cur_pyr, cgx, cgy, p0, tref, R, ITERS,
                                      fine_lam=lam)
                for k, j in enumerate(ids):
                    if v[k]:
                        pos[j] = u[k]
                    else:
                        pos.pop(j, None)
            for j in list(born):
                gp, rng, _z = md.project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                alive_h = any(j in pos_hyb[label] for label in labels)
                if not inb or (j not in pos_prev and j not in pos_first and not alive_h):
                    for d in (Xw, born, fp, pos_prev, pos_first):
                        d.pop(j, None)
                    for d in pos_hyb.values():
                        d.pop(j, None)
                    continue
                age = i - born[j]
                row = [age,
                       np.hypot(*(pos_prev[j] - gp)) if j in pos_prev else np.nan,
                       np.hypot(*(pos_first[j] - gp)) if j in pos_first else np.nan]
                row += [np.hypot(*(pos_hyb[label][j] - gp)) if j in pos_hyb[label] else np.nan
                        for label in labels]
                rows.append(tuple(row))
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
                    fp[nid] = np.stack([pk.sample(cur_pyr[lv], np.array([x * 0.5 ** lv]),
                                                  np.array([y * 0.5 ** lv]), OFFX, OFFY)[0]
                                        for lv in range(LV)])
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    for d in pos_hyb.values():
                        d[nid] = np.array([x, y])
                    nid += 1
        prev = (cur_pyr, cgx, cgy)
        if i % 100 == 0:
            print(f"  [{i}] active={len(pos_prev)} obs={len(rows)}")

    A = np.array(rows, float)
    if not args.no_save:
        try:
            np.savez(f"{args.out}_{args.cond}.npz", rows=A,
                     cols=np.array(["age", "prev", "first", *labels]))
        except OSError as exc:
            print(f"\nWARNING: could not save {args.out}_{args.cond}.npz: {exc}")
    print(f"\n=== coarse-fresh/fine-fixed HYBRID, {args.cond} ({len(A)} obs) ===")
    header = f"  {'age':>8} {'n':>7} {'prev':>8} {'first':>8}" + "".join(f" {x:>8}" for x in labels)
    print(header + "   (median |beta| px)")
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            vals = [np.nanmedian(A[m, col]) for col in range(1, A.shape[1])]
            print(f"  {lo:3d}-{hi:<4d} {int(m.sum()):7d} "
                  + " ".join(f"{v:8.3f}" for v in vals))
    print(f"\n  p90 |beta| (tail):")
    print(f"  {'age':>8} {'prev':>8} {'first':>8}" + "".join(f" {x:>8}" for x in labels))
    for lo, hi in [(20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            vals = [np.nanpercentile(A[m, col], 90) for col in range(1, A.shape[1])]
            print(f"  {lo:3d}-{hi:<4d} " + " ".join(f"{v:8.2f}" for v in vals))

    finite_prev = np.isfinite(A[:, 1])
    print("\n  delta vs previous-frame median (negative would beat prev):")
    for label, col in zip(labels, range(3, A.shape[1])):
        paired = finite_prev & np.isfinite(A[:, col])
        if paired.sum() > 20:
            print(f"    {label:>8}: median {np.nanmedian(A[paired, col] - A[paired, 1]):+.3f}px, "
                  f"worse {100*np.mean(A[paired, col] > A[paired, 1]):.1f}% "
                  f"(n={int(paired.sum())})")


if __name__ == "__main__":
    main()
