"""Does DIS optical flow (dense + variational spatial regularization) reduce the TEMPORAL
drift KLT accumulates -- and does the answer depend on the CASE? Separate the strata the
pooled test conflated: NORMAL (well-conditioned corner) vs WEAK-MIN-EIG (edge/aperture) vs
REPETITIVE (secondary self-similarity peak) vs OCCLUDED. DIS's variational smoothness is
*designed* for the weak-eig case, so that stratum is the real test; goodFeaturesToTrack
rejects those points, so we GRID-seed to get the full structure-tensor range.

Exact-GT MidAir: seed a grid (valid GT depth), classify each seed by structure-tensor
eigenvalues (cv2.cornerEigenValsAndVecs) and repetitiveness (masked NCC secondary peak),
track chained frame-to-frame three ways, drift vs the GT reprojection, stratified:
  LK | DIS-norefine (iters 0) | DIS-refine (iters 5)   (refine-vs-norefine isolates the spatial reg)

  PY=echo-li-python/venv/bin/python
  $PY midair_dis_drift_test.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 250 --grid-step 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402


def to_u8(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    if a.dtype in (np.float32, np.float64):
        a = a * (255.0 if a.max() <= 1.0 + 1e-6 else 1.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def sample_flow(flow, x, y):
    H, W = flow.shape[:2]
    x = min(max(x, 0.0), W - 1.001); y = min(max(y, 0.0), H - 1.001)
    x0 = int(x); y0 = int(y); ax = x - x0; ay = y - y0
    f00 = flow[y0, x0]; f01 = flow[y0, x0 + 1]; f10 = flow[y0 + 1, x0]; f11 = flow[y0 + 1, x0 + 1]
    return (1 - ay) * ((1 - ax) * f00 + ax * f01) + ay * ((1 - ax) * f10 + ax * f11)


def make_dis(iters):
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.setVariationalRefinementIterations(iters)
    return d


def repetitiveness(gray_f, x, y, half=5, search=12):
    """Masked NCC secondary peak: template-match an 11x11 patch in a +/-search window,
    zero a disk around the central (self) peak, return the max remaining correlation.
    ~1 => strongly repetitive/self-similar."""
    xi, yi = int(round(x)), int(round(y))
    templ = gray_f[yi - half:yi + half + 1, xi - half:xi + half + 1]
    win = gray_f[yi - half - search:yi + half + search + 1, xi - half - search:xi + half + search + 1]
    if templ.shape != (2 * half + 1, 2 * half + 1) or win.shape[0] < templ.shape[0] + 2:
        return np.nan
    res = cv2.matchTemplate(win, templ, cv2.TM_CCOEFF_NORMED)  # (2*search+1)^2
    cv2.circle(res, (search, search), 3, -1.0, -1)  # blank the self peak
    return float(res.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--grid-step", type=int, default=12)
    ap.add_argument("--occ-tol", type=float, default=0.04)
    ap.add_argument("--repet-thr", type=float, default=0.7, help="secondary-NCC repetitive threshold")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    g0 = to_u8(ds.image(args.start)); H, W = g0.shape
    g0f = g0.astype(np.float32)
    f, cx, cy = md.intrinsics(W, H)
    depth0 = ds.depth(args.start); pose0 = ds.pose(args.start)
    last = min(args.start + args.frames, ds.n)

    # structure-tensor eigenvalues per pixel: ch0 = lmax, ch1 = lmin (lmax >= lmin)
    eig = cv2.cornerEigenValsAndVecs(g0, 7, 3)
    lmax_f = eig[:, :, 0]; lmin_f = eig[:, :, 1]

    m = 5 + 12 + 1  # NCC margin
    seeds = []; Xw = []; lmn = []; lmx = []; rep = []
    for y in range(m, H - m, args.grid_step):
        for x in range(m, W - m, args.grid_step):
            d0 = float(depth0[y, x])
            if not (1.0 < d0 < md.SKY):
                continue
            seeds.append([float(x), float(y)])
            Xw.append(md.backproject_world((float(x), float(y)), d0, pose0, f, cx, cy))
            lmn.append(float(lmin_f[y, x])); lmx.append(float(lmax_f[y, x]))
            rep.append(repetitiveness(g0f, x, y))
    seeds = np.array(seeds, np.float32); n = len(seeds)
    lmn = np.array(lmn); lmx = np.array(lmx); rep = np.array(rep)

    # classify seeds (exclude untrackable flat: lmax below 20th pct)
    flat = lmx < np.percentile(lmx, 20)
    lmn_lo, lmn_hi = np.percentile(lmn[~flat], 33), np.percentile(lmn[~flat], 67)
    weak = (~flat) & (lmn < lmn_lo)                 # edge / aperture
    strong = (~flat) & (lmn > lmn_hi)               # corner
    repet = (~flat) & (rep > args.repet_thr)
    print(f"Mid-Air {args.cond}/traj{ds.traj} {W}x{H}: {n} grid seeds "
          f"(flat {flat.mean()*100:.0f}%, weak-eig {weak.mean()*100:.0f}%, "
          f"strong {strong.mean()*100:.0f}%, repetitive {repet.mean()*100:.0f}%)")

    p_lk = seeds.reshape(-1, 1, 2).copy(); p_dr = seeds.copy(); p_dn = seeds.copy()
    alive = (~flat).copy()
    dis_r = make_dis(5); dis_n = make_dis(0); prev = g0
    lk_par = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    rows = []  # seed_idx, age, dlk, ddr, ddn, occ
    for i in range(args.start + 1, last):
        cur = to_u8(ds.image(i)); posei = ds.pose(i); depthi = ds.depth(i)
        p_lk_new, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p_lk, None, **lk_par)
        flow_r = dis_r.calc(prev, cur, None); flow_n = dis_n.calc(prev, cur, None)
        age = i - args.start
        for j in range(n):
            if not alive[j]:
                continue
            gp, _r, z = md.project_world(Xw[j], posei, f, cx, cy)
            if gp is None or not (1 <= gp[0] < W - 1 and 1 <= gp[1] < H - 1) or z <= 0.1:
                alive[j] = False; continue
            p_dr[j] += sample_flow(flow_r, p_dr[j, 0], p_dr[j, 1])
            p_dn[j] += sample_flow(flow_n, p_dn[j, 0], p_dn[j, 1])
            gx, gy = int(round(gp[0])), int(round(gp[1]))
            occ = float(depthi[gy, gx]) < z * (1 - args.occ_tol)
            d_lk = np.hypot(*(p_lk_new[j, 0] - gp)) if st[j, 0] else np.nan
            rows.append((j, age, d_lk, np.hypot(*(p_dr[j] - gp)), np.hypot(*(p_dn[j] - gp)), occ))
        prev = cur; p_lk = p_lk_new

    A = np.array(rows, float)
    sidx = A[:, 0].astype(int); dlk, ddr, ddn, occ = A[:, 2], A[:, 3], A[:, 4], A[:, 5] == 1
    print(f"\n{len(A)} point-frames.  drift vs exact GT, median/p90 px  [LK | DIS-refine | DIS-norefine]:")

    def report(mask, tag):
        mm = mask & np.isfinite(dlk)
        if mm.sum() < 50:
            print(f"  {tag:>22} (n={int(mm.sum())}): too few"); return
        print(f"  {tag:>22} (n={int(mm.sum()):6d}):  "
              f"LK {np.nanmedian(dlk[mm]):6.3f}/{np.nanpercentile(dlk[mm],90):6.2f}   "
              f"refine {np.median(ddr[mm]):6.3f}/{np.percentile(ddr[mm],90):6.2f}   "
              f"norefine {np.median(ddn[mm]):6.3f}/{np.percentile(ddn[mm],90):6.2f}")

    weak_r = weak[sidx]; strong_r = strong[sidx]; repet_r = repet[sidx]
    report(strong_r & ~repet_r & ~occ, "NORMAL (corner)")
    report(weak_r & ~occ, "WEAK-EIG (aperture)")
    report(repet_r & ~occ, "REPETITIVE")
    report(occ, "OCCLUDED")
    report(~occ, "all non-occluded (pooled)")
    print("\nWithin WEAK-EIG/REPETITIVE: refine<LK => DIS spatial reg fixes the drift case; else it doesn't.")


if __name__ == "__main__":
    main()
