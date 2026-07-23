"""Visual Mid-Air diagnostic: why similar first/prev patches do not converge alike.

This captures tracks where first-observation tracking has lost but previous-frame tracking
is still close, then tests the convergence mechanism directly:

  full-pyramid first-ref solve from prev init:  L2 -> L1 -> L0
  fine-only first-ref solve from prev init:     L0 only
  full-pyramid first-ref solve from GT init:    L2 -> L1 -> L0

The key visual check is whether the L0 first-reference landscape looks reasonable while
the coarse L2/L1 first-reference landscapes pull the optimizer into the wrong basin before
L0 can refine. If fine-only succeeds but full-pyramid fails, the issue is not patch
similarity at L0; it is the stale coarse-to-fine objective.

Example:

  PY=echo-li-python/venv/bin/python
  $PY first_obs_convergence_visual.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --scale 0.5 --frames 300 --n-victims 4 --out first_obs_convergence
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


def make_grid(gt, *pts, base_half=8.0, max_half=48.0, pad=4.0):
    offs = [np.zeros(2)]
    offs.extend(np.asarray(p, float) - gt for p in pts)
    need = float(np.ceil(np.max(np.abs(np.array(offs))) + pad))
    half = min(max(base_half, need), max_half)
    return np.arange(-half, half + 0.5 * GRID_STEP, GRID_STEP)


def perlevel_landscape(cur_pyr, tref_levels, gt, lv, grid):
    s = 0.5 ** lv
    dx, dy = np.meshgrid(grid, grid)
    cx = ((gt[0] + dx) * s).ravel()
    cy = ((gt[1] + dy) * s).ravel()
    Iw = pk.sample(cur_pyr[lv], cx, cy, OFFX, OFFY)
    return np.mean((Iw - tref_levels[lv][None, :]) ** 2, axis=1).reshape(len(grid), len(grid))


def argmin_offset(land, grid):
    iy, ix = np.unravel_index(np.argmin(land), land.shape)
    return np.array([grid[ix], grid[iy]], float)


def trace_first_ref(cur_pyr, cgx, cgy, p0, tref, levels):
    """Trace pure first-reference translation KLT. Positions are stored in L0 pixels."""
    u = p0.astype(np.float64).copy()
    log = []
    for lv in levels:
        s = 0.5 ** lv
        Cl, Gx, Gy = cur_pyr[lv], cgx[lv], cgy[lv]
        ux, uy = u[0] * s, u[1] * s
        T = tref[lv][None, :]
        for it in range(ITERS):
            Iw = pk.sample(Cl, np.array([ux]), np.array([uy]), OFFX, OFFY)
            jx = pk.sample(Gx, np.array([ux]), np.array([uy]), OFFX, OFFY)
            jy = pk.sample(Gy, np.array([ux]), np.array([uy]), OFFX, OFFY)
            res = Iw - T
            Hxx = float(np.sum(jx * jx))
            Hxy = float(np.sum(jx * jy))
            Hyy = float(np.sum(jy * jy))
            bx = -float(np.sum(jx * res))
            by = -float(np.sum(jy * res))
            reg = 1e-3 * (Hxx + Hyy + 1e-6)
            Hxx += reg
            Hyy += reg
            det = Hxx * Hyy - Hxy * Hxy
            dx = (Hyy * bx - Hxy * by) / det if abs(det) > 1e-6 else 0.0
            dy = (Hxx * by - Hxy * bx) / det if abs(det) > 1e-6 else 0.0
            step = np.hypot(dx, dy)
            scl = 1.0 / step if step > 1.0 else 1.0
            ux += dx * scl
            uy += dy * scl
            tr = Hxx + Hyy
            disc = max(tr * tr - 4 * det, 0.0)
            lam_min = 0.5 * (tr - np.sqrt(disc))
            lam_max = 0.5 * (tr + np.sqrt(disc))
            u_abs = np.array([ux / s, uy / s])
            log.append(dict(lv=lv, it=it, u=u_abs.copy(),
                            rms=float(np.sqrt(np.mean(res ** 2))),
                            step=float(step * scl / s),
                            cond=float(lam_max / max(lam_min, 1e-9))))
        u = np.array([ux / s, uy / s])
    return u, log


def context_crop(img, center, boxes, size):
    half = size // 2
    cx, cy = float(center[0]), float(center[1])
    crop = cv2.getRectSubPix(img, (size, size), (cx, cy))
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        if len(box) == 2:
            name, p = box
            rad = R
        else:
            name, p, rad = box
        qx = int(round(p[0] - cx + half))
        qy = int(round(p[1] - cy + half))
        x0, y0 = qx - rad, qy - rad
        x1, y1 = qx + rad, qy + rad
        if x1 >= 0 and y1 >= 0 and x0 < size and y0 < size:
            cv2.rectangle(vis,
                          (int(np.clip(x0, 0, size - 1)), int(np.clip(y0, 0, size - 1))),
                          (int(np.clip(x1, 0, size - 1)), int(np.clip(y1, 0, size - 1))),
                          (255, 255, 255), 1, cv2.LINE_AA)
            tx = int(np.clip(x0, 0, size - 1))
            ty = int(np.clip(y0 - 2, 8, size - 1))
            cv2.putText(vis, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def source_level_patch(img, pos, lv):
    """Crop the original-image footprint for pyramid level `lv`, then resize to patch display size."""
    scale = 2 ** lv
    side = (2 * R + 1) * scale
    crop = cv2.getRectSubPix(img, (side, side), (float(pos[0]), float(pos[1])))
    small = cv2.resize(crop, (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def norm_patch(p):
    if p.ndim == 1:
        # pk.sample flattens patches x-major: OFFX repeats each x while OFFY tiles y.
        # Image display is y-major, so transpose after reshape for visual comparison.
        q = p.reshape(2 * R + 1, 2 * R + 1).T
    else:
        q = p
    lo, hi = np.percentile(q, [2, 98])
    if hi <= lo:
        return np.zeros_like(q)
    return np.clip((q - lo) / (hi - lo), 0, 1)


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
    ap.add_argument("--context", type=int, default=64,
                    help="context crop side in px; default is about 4x the 15px KLT patch")
    ap.add_argument("--n-victims", type=int, default=4)
    ap.add_argument("--out", default="first_obs_convergence")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cap_ages = set(int(a) for a in args.capture_ages.split(","))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    H, W = ds.image(0).shape
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}
    born = {}
    first_patch = {}
    pos_prev = {}
    pos_first = {}
    bimg = {}
    bpix = {}
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
                    for d in (Xw, born, first_patch, pos_prev, pos_first, bimg, bpix):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                bf = np.hypot(*(pos_first[j] - gp))
                bp = np.hypot(*(pos_prev[j] - gp))
                if len(victims) < args.n_victims and age in cap_ages and bf > args.first_thr and bp < args.prev_thr:
                    fresh_tref = sample_template(ppyr, init_prev[j])
                    full, full_log = trace_first_ref(cur_pyr, cgx, cgy, init_prev[j],
                                                     first_patch[j], [2, 1, 0])
                    fine, fine_log = trace_first_ref(cur_pyr, cgx, cgy, init_prev[j],
                                                     first_patch[j], [0])
                    gtfull, gtfull_log = trace_first_ref(cur_pyr, cgx, cgy, gp,
                                                         first_patch[j], [2, 1, 0])
                    grid = make_grid(gp, pos_prev[j], pos_first[j], full, fine,
                                     base_half=args.grid_half, max_half=args.max_grid_half)
                    first_lands = [perlevel_landscape(cur_pyr, first_patch[j], gp, lv, grid)
                                   for lv in range(LV)]
                    fresh_lands = [perlevel_landscape(cur_pyr, fresh_tref, gp, lv, grid)
                                   for lv in range(LV)]
                    birth_context = context_crop(bimg[j], bpix[j],
                                                 [("first", np.array(bpix[j], float))],
                                                 args.context)
                    prev_context = context_crop(prev_img, init_prev[j],
                                                [("prev", init_prev[j])],
                                                args.context)
                    cur_context = context_crop(img, gp,
                                               [("GT", gp), ("prev", init_prev[j]),
                                                ("first", pos_first[j])],
                                               args.context)
                    victims.append(dict(
                        id=j, frame=i, age=age, gt=gp.copy(), init=init_prev[j].copy(),
                        prev=pos_prev[j].copy(), first=pos_first[j].copy(),
                        full=full, fine=fine, gtfull=gtfull,
                        bf=bf, bp=bp, grid=grid, first_lands=first_lands,
                        fresh_lands=fresh_lands, full_log=full_log,
                        fine_log=fine_log, gtfull_log=gtfull_log,
                        first_patch=first_patch[j], fresh_tref=fresh_tref,
                        birth_context=birth_context, prev_context=prev_context,
                        context=cur_context,
                        birth_img=bimg[j].copy(), prev_img=prev_img.copy(), cur_img=img.copy(),
                        birth_pix=bpix[j].copy(),
                        src_first=[source_level_patch(bimg[j], bpix[j], lv) for lv in range(LV)],
                        src_prev=[source_level_patch(prev_img, init_prev[j], lv) for lv in range(LV)],
                        src_gt=[source_level_patch(img, gp, lv) for lv in range(LV)],
                        src_curprev=[source_level_patch(img, pos_prev[j], lv) for lv in range(LV)],
                        cur_gt=[pk.sample(cur_pyr[lv], np.array([gp[0] * 0.5 ** lv]),
                                          np.array([gp[1] * 0.5 ** lv]), OFFX, OFFY)[0]
                                for lv in range(LV)],
                        cur_prev=[pk.sample(cur_pyr[lv], np.array([pos_prev[j][0] * 0.5 ** lv]),
                                            np.array([pos_prev[j][1] * 0.5 ** lv]), OFFX, OFFY)[0]
                                  for lv in range(LV)]))
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
                    Xw[nid] = md.backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = sample_template(cur_pyr, np.array([x, y]))
                    pos_prev[nid] = np.array([x, y])
                    pos_first[nid] = np.array([x, y])
                    bimg[nid] = img.copy()
                    bpix[nid] = np.array([x, y])
                    nid += 1
        prev = (cur_pyr, cgx, cgy, img.copy())
        if len(victims) >= args.n_victims:
            break
        if i % 100 == 0:
            print(f"  [{i}] active={len(pos_prev)} victims={len(victims)}")

    print(f"\ncaptured {len(victims)} convergence victims")
    report(victims, args.out, save=not args.no_save)


def plot_trace(ax, log, gt, lv=None, color="white", label=None, marker="o"):
    pts = np.array([e["u"] - gt for e in log if lv is None or e["lv"] == lv])
    if len(pts):
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=1.0, alpha=0.9)
        ax.plot(pts[-1, 0], pts[-1, 1], marker, color=color, ms=4, label=label)


def report(victims, out, save=True):
    if not victims:
        return
    for v in victims:
        print(f"id {v['id']} frame {v['frame']} age {v['age']} "
              f"prev {v['bp']:.2f}px first {v['bf']:.2f}px | "
              f"full-from-prev {LA.norm(v['full'] - v['gt']):.2f}px "
              f"fine-only {LA.norm(v['fine'] - v['gt']):.2f}px "
              f"full-from-GT {LA.norm(v['gtfull'] - v['gt']):.2f}px")
        first_mins = [LA.norm(argmin_offset(v["first_lands"][lv], v["grid"])) for lv in (2, 1, 0)]
        fresh_mins = [LA.norm(argmin_offset(v["fresh_lands"][lv], v["grid"])) for lv in (2, 1, 0)]
        print(f"  first-ref objective minima L2/L1/L0: "
              f"{first_mins[0]:.2f}, {first_mins[1]:.2f}, {first_mins[2]:.2f}px from GT")
        print(f"  fresh-prev objective minima  L2/L1/L0: "
              f"{fresh_mins[0]:.2f}, {fresh_mins[1]:.2f}, {fresh_mins[2]:.2f}px from GT")
    if not save:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(victims), 6, figsize=(18, 3.1 * len(victims)), squeeze=False)
    for r, v in enumerate(victims):
        grid = v["grid"]
        ext = [grid[0], grid[-1], grid[-1], grid[0]]
        prev_off = v["prev"] - v["gt"]
        first_off = v["first"] - v["gt"]
        init_off = v["init"] - v["gt"]
        for c, (kind, lv) in enumerate([
            ("first", 2), ("first", 1), ("first", 0),
            ("fresh", 2), ("fresh", 1), ("fresh", 0),
        ]):
            land = v[f"{kind}_lands"][lv]
            a = ax[r, c]
            a.imshow(np.sqrt(land), extent=ext, origin="upper", cmap="viridis")
            amin = argmin_offset(land, grid)
            a.plot(0, 0, "+", color="lime", ms=11, mew=2, label="GT")
            a.plot(*amin, "s", mfc="none", mec="white", ms=8, mew=1.4, label="min")
            a.plot(*init_off, "x", color="cyan", ms=7, mew=1.4, label="init/prev-in")
            a.plot(*prev_off, "x", color="orange", ms=7, mew=1.4, label="prev out")
            a.plot(*first_off, "o", mfc="none", mec="red", ms=7, mew=1.4, label="first out")
            if kind == "first":
                plot_trace(a, v["full_log"], v["gt"], lv=lv, color="magenta",
                           label="full pyramid")
                if lv == 0:
                    plot_trace(a, v["fine_log"], v["gt"], lv=0, color="deepskyblue",
                               label="L0 only", marker="^")
                    plot_trace(a, v["gtfull_log"], v["gt"], lv=0, color="white",
                               label="full from GT", marker="d")
            a.set_title(f"{kind} template L{lv}\nmin {LA.norm(amin):.1f}px", fontsize=8)
            if c == 0:
                a.set_ylabel(f"id {v['id']} age {v['age']}\ndy from GT px", fontsize=7)
            a.set_xlabel("dx from GT px", fontsize=7)
            a.tick_params(labelsize=6)
            if r == 0 and c == 5:
                a.legend(fontsize=5, loc="upper right")
    fig.suptitle("Why similar first/prev patches diverge: per-level objectives and GN traces", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}_perlevel.png", dpi=150)

    rows = 7 * len(victims)
    fig2, ax2 = plt.subplots(rows, 8, figsize=(16.5, 1.9 * rows), squeeze=False)
    for r, v in enumerate(victims):
        base = 7 * r
        context_imgs = [v["birth_context"], v["prev_context"], v["context"]]
        context_titles = [
            "birth context\nfirst square",
            "previous context\nprev square",
            "current context\nGT / prev / first squares",
        ]
        for c in range(8):
            ax = ax2[base, c]
            if c < 3:
                ax.imshow(context_imgs[c])
                ax.set_title(context_titles[c], fontsize=8)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"id {v['id']} age {v['age']}\ncontext", fontsize=8)

        for k, lv in enumerate((2, 1, 0)):
            ctx_row = base + 1 + 2 * k
            patch_row = ctx_row + 1
            rad = 0.5 * (2 * R + 1) * (2 ** lv)
            level_context_imgs = [
                context_crop(v["birth_img"], v["birth_pix"], [("first", v["birth_pix"], rad)], 64),
                context_crop(v["prev_img"], v["init"], [("prev", v["init"], rad)], 64),
                context_crop(v["cur_img"], v["gt"], [("GT", v["gt"], rad),
                                                     ("prev", v["init"], rad),
                                                     ("first", v["first"], rad)], 64),
            ]
            level_context_titles = [
                f"birth context L{lv}\nfirst footprint",
                f"previous context L{lv}\nprev footprint",
                f"current context L{lv}\nGT / prev / first footprints",
            ]
            for c in range(8):
                ax = ax2[ctx_row, c]
                if c < 3:
                    ax.imshow(level_context_imgs[c])
                    ax.set_title(level_context_titles[c], fontsize=8)
                else:
                    ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"L{lv} context", fontsize=8)

            imgs = [
                norm_patch(v["src_first"][lv]),
                norm_patch(v["first_patch"][lv]),
                norm_patch(v["src_prev"][lv]),
                norm_patch(v["fresh_tref"][lv]),
                norm_patch(v["src_gt"][lv]),
                norm_patch(v["cur_gt"][lv]),
                norm_patch(v["src_curprev"][lv]),
                norm_patch(v["cur_prev"][lv]),
            ]
            titles = [
                f"src first L{lv}", f"pyr first L{lv}",
                f"src prev L{lv}", f"pyr prev L{lv}",
                f"src GT L{lv}", f"pyr GT L{lv}",
                f"src curprev L{lv}", f"pyr curprev L{lv}",
            ]
            for c, im in enumerate(imgs):
                ax = ax2[patch_row, c]
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
                ax.set_title(titles[c], fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"L{lv} patches", fontsize=8)
    fig2.suptitle("Spatial context with KLT patch squares only, plus patch appearances", fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(f"{out}_patches.png", dpi=150)

    print(f"\nsaved {out}_perlevel.png")
    print(f"saved {out}_patches.png")


if __name__ == "__main__":
    main()
