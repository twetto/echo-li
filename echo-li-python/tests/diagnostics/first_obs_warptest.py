"""Oracle-homography warp test: is the fixed-reference's coarse-scale staleness GEOMETRIC
(scale/perspective -> warp-fixable, variant E viable) or PHOTOMETRIC (lighting/non-Lambertian
-> unfixable, E dead)?

first_obs_perlevel.py showed the stale birth reference's SSD min is displaced ~3.5px at the
COARSE pyramid level (on GT at fine). This warps the birth patch by the EXACT plane-induced
homography (Mid-Air exact pose + depth) to how it should look at the current viewpoint, and
re-measures the per-level SSD argmin. Compared three ways, stratified by patch planarity
(depth variance), so non-planar/foliage failures don't masquerade as photometric:

  unwarped ref : fixed birth patch                          (expect coarse displaced)
  WARPED  ref  : birth patch warped by oracle homography    (geometric -> snaps to GT?)
  fresh        : previous-frame patch                        (control, on GT all levels)

  H maps birth pixel -> current pixel; H@birth_center must land on GT (printed self-check).

  PY=echo-li-python/venv/bin/python
  $PY first_obs_warptest.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --scale 0.5 --frames 300 --n-victims 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as LA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photometric_klt_ab as pk  # noqa: E402
import midair_drift as md  # noqa: E402

LV, R, ITERS = md.LV, md.R, md.ITERS
OFFX = np.repeat(np.arange(-R, R + 1, dtype=np.float32), 2 * R + 1)
OFFY = np.tile(np.arange(-R, R + 1, dtype=np.float32), 2 * R + 1)
GRID = np.arange(-6.0, 6.01, 0.5)


def perlevel_argmin(pyr, tref_levels, gt):
    out = []
    for lv in range(LV):
        s = 0.5 ** lv
        tl = tref_levels[lv]
        best = None
        for dy in GRID:
            for dx in GRID:
                Iw = pk.sample(pyr[lv], np.array([(gt[0] + dx) * s]),
                               np.array([(gt[1] + dy) * s]), OFFX, OFFY)[0]
                rms = float(np.sqrt(np.mean((Iw - tl) ** 2)))
                if best is None or rms < best[1]:
                    best = (np.hypot(dx, dy), rms)
        out.append(best[0])
    return out


def sample_template(pyr, pos):
    return np.stack([pk.sample(pyr[lv], np.array([pos[0] * 0.5 ** lv]),
                               np.array([pos[1] * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])


def cam2world(T_wb):
    return T_wb @ md.RT_BC                      # cam -> body -> world


def plane_at(depth, u, v, f, cx, cy, win=9):
    """Fit a local plane in the (birth) camera frame from range depth. Returns (n, d, var)
    with plane n^T X = d, X in camera coords, d>0 in front."""
    us, vs = np.meshgrid(np.arange(u - win, u + win + 1), np.arange(v - win, v + win + 1))
    us = us.ravel(); vs = vs.ravel()
    ok = (us >= 0) & (us < depth.shape[1]) & (vs >= 0) & (vs < depth.shape[0])
    us, vs = us[ok], vs[ok]
    rng = depth[vs, us]
    m = rng < md.SKY
    us, vs, rng = us[m], vs[m], rng[m]
    ray = np.stack([(us - cx) / f, (vs - cy) / f, np.ones_like(us, float)], 1)
    Z = rng / LA.norm(ray, axis=1)
    X = ray * Z[:, None]                          # (N,3) camera points
    c = X.mean(0)
    _, _, Vt = LA.svd(X - c)
    n = Vt[-1]
    d = n @ c
    if d < 0:
        n, d = -n, -d
    return n, d, float(np.std(rng))


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
    ap.add_argument("--n-victims", type=int, default=12)
    args = ap.parse_args()
    cap_ages = set(int(a) for a in args.capture_ages.split(","))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    W, H = ds.image(0).shape[1], ds.image(0).shape[0]
    f, cx, cy = md.intrinsics(W, H)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = LA.inv(K)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}; born = {}; first_patch = {}; pos_prev = {}; pos_first = {}
    bimg = {}; bpose = {}; bpix = {}; bplane = {}       # birth image/pose/pixel/plane
    nid = 0; victims = []; prev = None
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        init_prev = {j: pos_prev[j].copy() for j in pos_prev}
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
                    for d in (Xw, born, first_patch, pos_prev, pos_first,
                              bimg, bpose, bpix, bplane):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                if (len(victims) < args.n_victims and age in cap_ages
                        and j in pos_first and j in pos_prev and j in init_prev and j in bimg):
                    bf = np.hypot(*(pos_first[j] - gp)); bp = np.hypot(*(pos_prev[j] - gp))
                    if not (bf > args.first_thr and bp < args.prev_thr):
                        continue
                    n_b, d_b, dvar = bplane[j]
                    # relative birth-cam -> current-cam
                    G = LA.inv(cam2world(T)) @ cam2world(bpose[j])
                    Rcb, tcb = G[:3, :3], G[:3, 3]
                    Hh = K @ (Rcb + np.outer(tcb, n_b) / d_b) @ Kinv     # birth px -> current px
                    # self-check: birth centre must map to current GT
                    pc = Hh @ np.array([bpix[j][0], bpix[j][1], 1.0]); pc = pc / pc[2]
                    hval = float(np.hypot(pc[0] - gp[0], pc[1] - gp[1]))
                    warped = cv2.warpPerspective(bimg[j], Hh, (W, H), flags=cv2.INTER_LINEAR)
                    wpyr, _, _ = pk.build_pyramid(warped, LV, histeq=False)
                    warped_tref = sample_template(wpyr, gp)
                    fresh_tref = sample_template(ppyr, init_prev[j])
                    # validity gates (codex result-14): border zero-pad + occlusion/visibility
                    vmask = cv2.warpPerspective(np.ones_like(bimg[j]), Hh, (W, H),
                                                flags=cv2.INTER_NEAREST).astype(np.float32)
                    cext = 2 ** (LV - 1)                           # coarse (L2) patch footprint in L0 px
                    vp = pk.sample(vmask, np.array([gp[0]]), np.array([gp[1]]),
                                   OFFX * cext, OFFY * cext)[0]
                    vfrac = float((vp > 0.5).mean())               # real-content fraction over coarse footprint
                    gx, gy = int(round(gp[0])), int(round(gp[1]))
                    occ = int(depth[gy, gx] < 0.95 * rng)          # current scene nearer -> disoccluded
                    victims.append(dict(
                        id=j, age=age, bf=bf, dvar=dvar, hval=hval, vfrac=vfrac, occ=occ,
                        unwarp=perlevel_argmin(cur_pyr, first_patch[j], gp),
                        warp=perlevel_argmin(cur_pyr, warped_tref, gp),
                        fresh=perlevel_argmin(cur_pyr, fresh_tref, gp)))
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
                    n_b, d_b, dvar = plane_at(depth, gx, gy, f, cx, cy)
                    Xw[nid] = md.backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = sample_template(cur_pyr, np.array([x, y]))
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    bimg[nid] = img.copy(); bpose[nid] = T.copy()
                    bpix[nid] = (x, y); bplane[nid] = (n_b, d_b, dvar)
                    nid += 1
        prev = (cur_pyr, cgx, cgy)
        if len(victims) >= args.n_victims:
            break

    print(f"\ncaptured {len(victims)} first-obs-loses tracks   "
          f"(median homography self-check err {np.median([v['hval'] for v in victims]):.2f}px "
          f"-- should be ~0 if pose/H correct)\n")
    print(f"per-level SSD-argmin displacement from GT (L0 px), coarse L2 -> fine L0:")
    print(f"  {'id':>5} {'age':>4} {'dvar':>6} {'vfrac':>5} {'occ':>3} {'valid':>5} | "
          f"{'u_L2':>5} {'u_L0':>5} | {'W_L2':>5} {'W_L1':>5} {'W_L0':>5} | {'f_L2':>5} {'f_L0':>5}")

    def is_valid(v):                                    # codex result-14 gates
        return v["vfrac"] >= 0.9 and v["occ"] == 0
    for v in sorted(victims, key=lambda z: z["dvar"]):
        print(f"  {v['id']:>5} {v['age']:>4} {v['dvar']:>6.1f} {v['vfrac']:>5.2f} {v['occ']:>3} "
              f"{('Y' if is_valid(v) else 'n'):>5} | "
              f"{v['unwarp'][2]:>5.1f} {v['unwarp'][0]:>5.1f} | "
              f"{v['warp'][2]:>5.1f} {v['warp'][1]:>5.1f} {v['warp'][0]:>5.1f} | "
              f"{v['fresh'][2]:>5.1f} {v['fresh'][0]:>5.1f}")

    def med(sel, key, lv):
        xs = [v[key][lv] for v in victims if sel(v)]
        return np.median(xs) if xs else float("nan")
    nvalid = sum(is_valid(v) for v in victims)
    for lbl, sel in [("ALL", lambda v: True),
                     (f"VALID warp (vfrac>=.9 & !occ, n={nvalid})", is_valid),
                     ("INVALID (border/occlusion)", lambda v: not is_valid(v))]:
        print(f"\n  MEDIAN [{lbl}]  unwarp L2/L0 = {med(sel,'unwarp',2):.2f}/{med(sel,'unwarp',0):.2f}   "
              f"WARP L2/L1/L0 = {med(sel,'warp',2):.2f}/{med(sel,'warp',1):.2f}/{med(sel,'warp',0):.2f}   "
              f"fresh L2/L0 = {med(sel,'fresh',2):.2f}/{med(sel,'fresh',0):.2f}")
    print("\n  reading (codex r14): the aggregate warp number is contaminated by border zero-pad")
    print("  (vfrac<0.9) and disocclusion (occ=1). On VALID warped patches, if WARP_L2 -> ~fresh_L2")
    print("  the coarse staleness is GEOMETRIC and a GATED reference-warp is viable; invalid patches")
    print("  must be rejected, not optimized through.")


if __name__ == "__main__":
    main()
