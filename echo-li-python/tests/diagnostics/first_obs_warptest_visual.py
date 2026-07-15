"""Visual version of first_obs_warptest.py.

Captures the same Mid-Air first-observation failure cases, then shows whether an exact
birth->current homography warp fixes the shifted coarse-level first-reference objective.

Outputs:
  <out>_landscapes.png  rows=victims, columns=unwarped/warped/fresh at L2/L1/L0
  <out>_patches.png     context + per-level source/pyramid patches for visual checking
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
OFF = np.arange(-R, R + 1, dtype=np.float32)
OFFX = np.repeat(OFF, len(OFF))
OFFY = np.tile(OFF, len(OFF))
GRID_STEP = 0.5


def sample_template(pyr, pos):
    return np.stack([pk.sample(pyr[lv], np.array([pos[0] * 0.5 ** lv]),
                               np.array([pos[1] * 0.5 ** lv]), OFFX, OFFY)[0]
                     for lv in range(LV)])


def cam2world(T_wb):
    return T_wb @ md.RT_BC


def plane_at(depth, u, v, f, cx, cy, win=9):
    us, vs = np.meshgrid(np.arange(u - win, u + win + 1), np.arange(v - win, v + win + 1))
    us = us.ravel()
    vs = vs.ravel()
    ok = (us >= 0) & (us < depth.shape[1]) & (vs >= 0) & (vs < depth.shape[0])
    us, vs = us[ok], vs[ok]
    rng = depth[vs, us]
    m = rng < md.SKY
    us, vs, rng = us[m], vs[m], rng[m]
    ray = np.stack([(us - cx) / f, (vs - cy) / f, np.ones_like(us, float)], 1)
    z = rng / LA.norm(ray, axis=1)
    X = ray * z[:, None]
    c = X.mean(0)
    _, _, Vt = LA.svd(X - c)
    n = Vt[-1]
    d = n @ c
    if d < 0:
        n, d = -n, -d
    return n, d, float(np.std(rng))


def make_grid(gt, *pts, base_half=8.0, max_half=48.0, pad=4.0):
    offs = [np.zeros(2)]
    offs.extend(np.asarray(p, float) - gt for p in pts)
    half = min(max(base_half, float(np.ceil(np.max(np.abs(np.array(offs))) + pad))), max_half)
    return np.arange(-half, half + 0.5 * GRID_STEP, GRID_STEP)


def perlevel_landscape(cur_pyr, tref, gt, lv, grid):
    s = 0.5 ** lv
    dx, dy = np.meshgrid(grid, grid)
    cx = ((gt[0] + dx) * s).ravel()
    cy = ((gt[1] + dy) * s).ravel()
    Iw = pk.sample(cur_pyr[lv], cx, cy, OFFX, OFFY)
    return np.mean((Iw - tref[lv][None, :]) ** 2, axis=1).reshape(len(grid), len(grid))


def argmin_offset(land, grid):
    iy, ix = np.unravel_index(np.argmin(land), land.shape)
    return np.array([grid[ix], grid[iy]], float)


def norm_patch(p):
    if p.ndim == 1:
        q = p.reshape(2 * R + 1, 2 * R + 1).T
    else:
        q = p
    lo, hi = np.percentile(q, [2, 98])
    if hi <= lo:
        return np.zeros_like(q)
    return np.clip((q - lo) / (hi - lo), 0, 1)


def source_level_patch(img, pos, lv):
    scale = 2 ** lv
    side = (2 * R + 1) * scale
    crop = cv2.getRectSubPix(img, (side, side), (float(pos[0]), float(pos[1])))
    return cv2.resize(crop, (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_AREA).astype(np.float32)


def context_crop(img, center, boxes, size):
    half = size // 2
    cx, cy = float(center[0]), float(center[1])
    crop = cv2.getRectSubPix(img, (size, size), (cx, cy))
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for name, p, rad in boxes:
        qx = int(round(p[0] - cx + half))
        qy = int(round(p[1] - cy + half))
        x0, y0 = qx - rad, qy - rad
        x1, y1 = qx + rad, qy + rad
        if x1 >= 0 and y1 >= 0 and x0 < size and y0 < size:
            cv2.rectangle(vis,
                          (int(np.clip(x0, 0, size - 1)), int(np.clip(y0, 0, size - 1))),
                          (int(np.clip(x1, 0, size - 1)), int(np.clip(y1, 0, size - 1))),
                          (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, name, (int(np.clip(x0, 0, size - 1)),
                                    int(np.clip(y0 - 2, 8, size - 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


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
    ap.add_argument("--grid-half", type=float, default=8.0)
    ap.add_argument("--max-grid-half", type=float, default=48.0)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--n-victims", type=int, default=12)
    ap.add_argument("--out", default="first_obs_warptest")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cap_ages = set(int(a) for a in args.capture_ages.split(","))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    H, W = ds.image(0).shape
    f, cx, cy = md.intrinsics(W, H)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = LA.inv(K)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}
    born = {}
    first_patch = {}
    pos_prev = {}
    pos_first = {}
    bimg = {}
    bpose = {}
    bpix = {}
    bplane = {}
    nid = 0
    victims = []
    prev = None

    for i in range(min(args.frames, ds.n)):
        img = ds.image(i)
        depth = ds.depth(i)
        T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        init_prev = {j: pos_prev[j].copy() for j in pos_prev}
        if prev is not None:
            ppyr, _, _, prev_img = prev
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
                gp, _, _ = md.project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                if not inb or j not in pos_prev or j not in pos_first or j not in init_prev:
                    for d in (Xw, born, first_patch, pos_prev, pos_first, bimg, bpose, bpix, bplane):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                bf = np.hypot(*(pos_first[j] - gp))
                bp = np.hypot(*(pos_prev[j] - gp))
                if len(victims) >= args.n_victims or age not in cap_ages or bf <= args.first_thr or bp >= args.prev_thr:
                    continue
                n_b, d_b, dvar = bplane[j]
                G = LA.inv(cam2world(T)) @ cam2world(bpose[j])
                Rcb, tcb = G[:3, :3], G[:3, 3]
                Hh = K @ (Rcb + np.outer(tcb, n_b) / d_b) @ Kinv
                pc = Hh @ np.array([bpix[j][0], bpix[j][1], 1.0])
                pc = pc / pc[2]
                hcheck = float(np.hypot(pc[0] - gp[0], pc[1] - gp[1]))
                warped = cv2.warpPerspective(bimg[j], Hh, (W, H), flags=cv2.INTER_LINEAR)
                wpyr, _, _ = pk.build_pyramid(warped, LV, histeq=False)
                warped_tref = sample_template(wpyr, gp)
                fresh_tref = sample_template(ppyr, init_prev[j])
                grid = make_grid(gp, pos_prev[j], pos_first[j],
                                 base_half=args.grid_half, max_half=args.max_grid_half)
                lands = {
                    "unwarp": [perlevel_landscape(cur_pyr, first_patch[j], gp, lv, grid) for lv in range(LV)],
                    "warp": [perlevel_landscape(cur_pyr, warped_tref, gp, lv, grid) for lv in range(LV)],
                    "fresh": [perlevel_landscape(cur_pyr, fresh_tref, gp, lv, grid) for lv in range(LV)],
                }
                victims.append(dict(
                    id=j, frame=i, age=age, gt=gp.copy(), first=pos_first[j].copy(),
                    prev=pos_prev[j].copy(), init=init_prev[j].copy(), bf=bf, bp=bp,
                    dvar=dvar, hcheck=hcheck, grid=grid, lands=lands,
                    first_patch=first_patch[j], warped_tref=warped_tref, fresh_tref=fresh_tref,
                    bimg=bimg[j].copy(), bpix=np.array(bpix[j], float),
                    prev_img=prev_img.copy(), cur_img=img.copy(),
                    src_first=[source_level_patch(bimg[j], bpix[j], lv) for lv in range(LV)],
                    src_warp=[source_level_patch(warped, gp, lv) for lv in range(LV)],
                    src_fresh=[source_level_patch(prev_img, init_prev[j], lv) for lv in range(LV)],
                    src_cur=[source_level_patch(img, gp, lv) for lv in range(LV)],
                    warped_img=warped.copy()))
        if len(pos_prev) < args.redetect:
            mask = np.uint8((depth < md.SKY) & (depth > 1.0)) * 255
            for j in pos_prev:
                p = pos_prev[j]
                cv2.circle(mask, (int(p[0]), int(p[1])), R + 2, 0, -1)
            corners = cv2.goodFeaturesToTrack(img, args.max_tracks - len(pos_prev),
                                              0.01, 2 * R + 3, mask=mask)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    x, y = float(c[0]), float(c[1])
                    gx, gy = int(round(x)), int(round(y))
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < md.SKY):
                        continue
                    n_b, d_b, dvar = plane_at(depth, gx, gy, f, cx, cy)
                    Xw[nid] = md.backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = sample_template(cur_pyr, np.array([x, y]))
                    pos_prev[nid] = np.array([x, y])
                    pos_first[nid] = np.array([x, y])
                    bimg[nid] = img.copy()
                    bpose[nid] = T.copy()
                    bpix[nid] = np.array([x, y])
                    bplane[nid] = (n_b, d_b, dvar)
                    nid += 1
        prev = (cur_pyr, cgx, cgy, img.copy())
        if len(victims) >= args.n_victims:
            break
        if i % 100 == 0:
            print(f"  [{i}] active={len(pos_prev)} victims={len(victims)}")

    report(victims, args.out, args.context, save=not args.no_save)


def report(victims, out, context, save=True):
    print(f"\ncaptured {len(victims)} visual warptest victims")
    for v in victims:
        vals = []
        for name in ("unwarp", "warp", "fresh"):
            vals.append("/".join(f"{LA.norm(argmin_offset(v['lands'][name][lv], v['grid'])):.1f}"
                                 for lv in (2, 1, 0)))
        print(f"id {v['id']} age {v['age']} dvar {v['dvar']:.1f} "
              f"prev {v['bp']:.1f} first {v['bf']:.1f} Hcheck {v['hcheck']:.2f} | "
              f"L2/L1/L0 unwarp {vals[0]} warp {vals[1]} fresh {vals[2]}")
    if not save or not victims:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("unwarp", 2), ("warp", 2), ("fresh", 2),
            ("unwarp", 1), ("warp", 1), ("fresh", 1),
            ("unwarp", 0), ("warp", 0), ("fresh", 0)]
    fig, ax = plt.subplots(len(victims), len(cols), figsize=(2.6 * len(cols), 2.7 * len(victims)), squeeze=False)
    for r, v in enumerate(victims):
        grid = v["grid"]
        ext = [grid[0], grid[-1], grid[-1], grid[0]]
        for c, (name, lv) in enumerate(cols):
            a = ax[r, c]
            land = v["lands"][name][lv]
            a.imshow(np.sqrt(land), extent=ext, origin="upper", cmap="viridis")
            amin = argmin_offset(land, grid)
            a.plot(0, 0, "+", color="lime", ms=9, mew=1.7)
            a.plot(*amin, "s", mfc="none", mec="white", ms=7, mew=1.2)
            a.set_title(f"{name} L{lv}\nmin {LA.norm(amin):.1f}px", fontsize=8)
            a.tick_params(labelsize=6)
            if c == 0:
                a.set_ylabel(f"id {v['id']} dvar {v['dvar']:.1f}", fontsize=7)
    fig.suptitle("Oracle homography warptest landscapes: unwarped first vs warped first vs fresh prev", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out}_landscapes.png", dpi=150)

    rows = 4 * len(victims)
    fig2, ax2 = plt.subplots(rows, 5, figsize=(10.5, 1.9 * rows), squeeze=False)
    for r, v in enumerate(victims):
        base = 4 * r
        rad = R
        ctxs = [
            context_crop(v["bimg"], v["bpix"], [("first", v["bpix"], rad)], context),
            context_crop(v["warped_img"], v["gt"], [("warp", v["gt"], rad)], context),
            context_crop(v["cur_img"], v["gt"], [("GT", v["gt"], rad), ("first", v["first"], rad)], context),
            context_crop(v["prev_img"], v["init"], [("fresh", v["init"], rad)], context),
        ]
        for c in range(5):
            a = ax2[base, c]
            if c < 4:
                a.imshow(ctxs[c])
                a.set_title(["birth context", "warped birth context",
                             "current context", "previous context"][c], fontsize=8)
            else:
                a.axis("off")
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                a.set_ylabel(f"id {v['id']} age {v['age']}\ncontext", fontsize=8)
        for rr, lv in enumerate((2, 1, 0), start=1):
            imgs = [
                norm_patch(v["first_patch"][lv]),
                norm_patch(v["warped_tref"][lv]),
                norm_patch(v["fresh_tref"][lv]),
                norm_patch(v["src_warp"][lv]),
                norm_patch(v["src_cur"][lv]),
            ]
            titles = [f"unwarp first L{lv}", f"warp first L{lv}", f"fresh prev L{lv}",
                      f"src warp L{lv}", f"current@GT L{lv}"]
            for c, im in enumerate(imgs):
                a = ax2[base + rr, c]
                a.imshow(im, cmap="gray", vmin=0, vmax=1)
                a.set_title(titles[c], fontsize=8)
                a.set_xticks([])
                a.set_yticks([])
                if c == 0:
                    a.set_ylabel(f"L{lv}", fontsize=8)
    fig2.suptitle("Oracle homography warptest patches", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.savefig(f"{out}_patches.png", dpi=150)
    print(f"\nsaved {out}_landscapes.png")
    print(f"saved {out}_patches.png")


if __name__ == "__main__":
    main()
