"""Visual Mid-Air check for the "joint inherits drift" hypothesis.

This is not an aggregate score. It captures tracks where the first-observation tracker
loses while the previous-frame tracker is still close, then draws the finest-level SSD
objective landscapes around exact GT for several objectives:

  prev          fresh previous-frame template vs current image
  first         fixed birth template vs current image
  joint         E = E_prev + lambda * E_first, one shared current center
  corr-first    birth template center stays fixed; appearance is corrected by the
                fresh previous-frame residual measured at the prev tracker output
  oracle-warp   birth template warped by exact Mid-Air pose/depth local plane

What to look for:

  If joint's minimum lies between prev and first, it is a weighted compromise and
  inherits both terms' biases.

  If corr-first/oracle-warp move the first-reference minimum toward GT while the first
  reference center remains fixed, then the promising route is "use prev to correct the
  first-reference comparison" rather than "jointly optimize one center against both".

  If corr-first stays near prev's offset or first's stale minimum, then the correction is
  also inheriting the wrong thing.

Example:

  PY=echo-li-python/venv/bin/python
  $PY first_obs_hypothesis_visual.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --scale 0.5 --frames 300 --n-victims 6 --out first_obs_hypothesis
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


def make_grid(gt, prev, first, base_half, max_half, pad=4.0):
    pts = np.array([np.zeros(2), prev - gt, first - gt], float)
    need = float(np.ceil(np.max(np.abs(pts)) + pad))
    half = min(max(base_half, need), max_half)
    return np.arange(-half, half + 0.5 * GRID_STEP, GRID_STEP)


def ssd_landscape(cur, tref, gt, grid):
    dx, dy = np.meshgrid(grid, grid)
    cx = (gt[0] + dx).ravel()
    cy = (gt[1] + dy).ravel()
    Iw = pk.sample(cur, cx, cy, OFFX, OFFY)
    return np.mean((Iw - tref[None, :]) ** 2, axis=1).reshape(len(grid), len(grid))


def argmin_offset(land, grid):
    iy, ix = np.unravel_index(np.argmin(land), land.shape)
    return np.array([grid[ix], grid[iy]], float)


def row_limits(v):
    grid = v["grid"]
    pts = [np.zeros(2)]
    pts.extend(argmin_offset(v["lands"][name], grid) for name in ("prev", "first", "joint", "corr", "warp"))
    pts.append(v["prev"] - v["gt"])
    pts.append(v["first"] - v["gt"])
    pts = np.array(pts, float)
    lo = np.minimum(pts.min(axis=0) - 2.0, grid[0])
    hi = np.maximum(pts.max(axis=0) + 2.0, grid[-1])
    span = np.maximum(hi - lo, 1.0)
    pad = np.maximum(0.0, span.max() - span) * 0.5
    lo -= pad
    hi += pad
    return lo[0], hi[0], lo[1], hi[1]


def context_crop(img, center, points, size):
    half = size // 2
    cx, cy = float(center[0]), float(center[1])
    crop = cv2.getRectSubPix(img, (size, size), (cx, cy))
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    colors = {"GT": (0, 255, 0), "prev": (0, 190, 255), "first": (0, 0, 255), "birth": (255, 80, 0)}
    markers = {"GT": cv2.MARKER_CROSS, "prev": cv2.MARKER_TILTED_CROSS,
               "birth": cv2.MARKER_DIAMOND}
    for name, p in points.items():
        qx = int(round(p[0] - cx + half))
        qy = int(round(p[1] - cy + half))
        if -20 <= qx < size + 20 and -20 <= qy < size + 20:
            qx = int(np.clip(qx, 0, size - 1))
            qy = int(np.clip(qy, 0, size - 1))
            if name == "first":
                cv2.circle(vis, (qx, qy), 6, colors[name], 1, cv2.LINE_AA)
            else:
                cv2.drawMarker(vis, (qx, qy), colors[name], markers[name], 12, 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


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
    return n, d


def warped_birth_template(bimg, bpose, bplane, bpix, cur_pose, gt, W, H, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = LA.inv(K)
    n_b, d_b = bplane
    G = LA.inv(cam2world(cur_pose)) @ cam2world(bpose)
    Rcb, tcb = G[:3, :3], G[:3, 3]
    Hh = K @ (Rcb + np.outer(tcb, n_b) / d_b) @ Kinv
    pc = Hh @ np.array([bpix[0], bpix[1], 1.0])
    pc = pc / pc[2]
    hcheck = float(np.hypot(pc[0] - gt[0], pc[1] - gt[1]))
    warped = cv2.warpPerspective(bimg, Hh, (W, H), flags=cv2.INTER_LINEAR)
    wpyr, _, _ = pk.build_pyramid(warped, LV, histeq=False)
    return sample_template(wpyr, gt)[0], hcheck


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
    ap.add_argument("--joint-lam", type=float, default=0.1)
    ap.add_argument("--grid-half", type=float, default=8.0,
                    help="minimum half-width, in GT-relative pixels, for objective landscapes")
    ap.add_argument("--max-grid-half", type=float, default=48.0,
                    help="maximum adaptive half-width; raise this if first markers still clip")
    ap.add_argument("--context", type=int, default=128,
                    help="large context crop size in pixels for the patch figure")
    ap.add_argument("--max-context", type=int, default=384,
                    help="maximum adaptive current-frame context crop size")
    ap.add_argument("--n-victims", type=int, default=6)
    ap.add_argument("--out", default="first_obs_hypothesis")
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
                    for d in (Xw, born, first_patch, pos_prev, pos_first,
                              bimg, bpose, bpix, bplane):
                        d.pop(j, None)
                    continue

                age = i - born[j]
                bf = np.hypot(*(pos_first[j] - gp))
                bp = np.hypot(*(pos_prev[j] - gp))
                if len(victims) < args.n_victims and age in cap_ages and bf > args.first_thr and bp < args.prev_thr:
                    Tprev = pk.sample(ppyr[0], np.array([init_prev[j][0]]), np.array([init_prev[j][1]]),
                                      OFFX, OFFY)[0]
                    Tfirst = first_patch[j][0]
                    Iprev_out = pk.sample(cur_pyr[0], np.array([pos_prev[j][0]]), np.array([pos_prev[j][1]]),
                                          OFFX, OFFY)[0]
                    Tcorr = Tfirst + (Iprev_out - Tprev)
                    Twarp, hcheck = warped_birth_template(bimg[j], bpose[j], bplane[j], bpix[j],
                                                          T, gp, W, H, f, cx, cy)

                    grid = make_grid(gp, pos_prev[j], pos_first[j],
                                     args.grid_half, args.max_grid_half)
                    clipped = max(np.max(np.abs(pos_prev[j] - gp)),
                                  np.max(np.abs(pos_first[j] - gp))) > abs(grid[0])
                    Lprev = ssd_landscape(cur_pyr[0], Tprev, gp, grid)
                    Lfirst = ssd_landscape(cur_pyr[0], Tfirst, gp, grid)
                    Ljoint = Lprev + args.joint_lam * Lfirst
                    Lcorr = ssd_landscape(cur_pyr[0], Tcorr, gp, grid)
                    Lwarp = ssd_landscape(cur_pyr[0], Twarp, gp, grid)
                    current_span = int(2 * (np.ceil(max(np.max(np.abs(pos_prev[j] - gp)),
                                                       np.max(np.abs(pos_first[j] - gp)))) + 32))
                    current_context = min(max(args.context, current_span), args.max_context)
                    cur_ctx = context_crop(img, gp, {
                        "GT": gp, "prev": pos_prev[j], "first": pos_first[j],
                    }, current_context)
                    prev_ctx = context_crop(prev_img, init_prev[j], {
                        "prev": init_prev[j],
                    }, args.context)
                    birth_ctx = context_crop(bimg[j], bpix[j], {
                        "birth": np.array(bpix[j], float),
                    }, args.context)
                    victims.append(dict(
                        id=j, frame=i, age=age, gt=gp.copy(), prev=pos_prev[j].copy(),
                        first=pos_first[j].copy(), bf=bf, bp=bp, hcheck=hcheck,
                        grid=grid, clipped=clipped,
                        templates=dict(prev=Tprev, first=Tfirst, corr=Tcorr, warp=Twarp,
                                       cur_gt=pk.sample(cur_pyr[0], np.array([gp[0]]), np.array([gp[1]]),
                                                        OFFX, OFFY)[0],
                                       cur_prev=Iprev_out),
                        contexts=dict(current=cur_ctx, previous=prev_ctx, birth=birth_ctx),
                        lands=dict(prev=Lprev, first=Lfirst, joint=Ljoint, corr=Lcorr, warp=Lwarp)))

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
                    bpose[nid] = T.copy()
                    bpix[nid] = (x, y)
                    bplane[nid] = plane_at(depth, gx, gy, f, cx, cy)
                    nid += 1

        prev = (cur_pyr, cgx, cgy, img.copy())
        if len(victims) >= args.n_victims:
            break
        if i % 100 == 0:
            print(f"  [{i}] active={len(pos_prev)} victims={len(victims)}")

    print(f"\ncaptured {len(victims)} visual victims")
    report(victims, args.out, args.joint_lam)


def report(victims, out, joint_lam):
    if not victims:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["prev", "first", "joint", "corr", "warp"]
    titles = {
        "prev": "prev objective",
        "first": "first objective",
        "joint": f"joint prev + {joint_lam:g} first",
        "corr": "first + prev residual correction",
        "warp": "oracle warped first",
    }
    n = len(victims)
    fig, ax = plt.subplots(n, len(names), figsize=(3.2 * len(names), 3.0 * n), squeeze=False)
    for r, v in enumerate(victims):
        grid = v["grid"]
        xmin, xmax, ymin, ymax = row_limits(v)
        print(f"id {v['id']} frame {v['frame']} age {v['age']} "
              f"prev {v['bp']:.2f}px first {v['bf']:.2f}px "
              f"grid +/-{abs(grid[0]):.0f}px"
              f"{' CLIPPED' if v['clipped'] else ''} homography-check {v['hcheck']:.2f}px")
        for c, name in enumerate(names):
            land = v["lands"][name]
            a = ax[r, c]
            a.imshow(np.sqrt(land), extent=[grid[0], grid[-1], grid[-1], grid[0]],
                     origin="upper", cmap="viridis")
            amin = argmin_offset(land, grid)
            prev_off = v["prev"] - v["gt"]
            first_off = v["first"] - v["gt"]
            rect = plt.Rectangle((grid[0], grid[0]), grid[-1] - grid[0], grid[-1] - grid[0],
                                 fill=False, edgecolor="white", linewidth=0.8, linestyle=":")
            a.add_patch(rect)
            a.plot(0, 0, "+", color="lime", ms=11, mew=2, label="GT")
            a.plot(*amin, "s", mfc="none", mec="white", ms=9, mew=1.5, label="min")
            a.plot(*prev_off, "x", color="orange", ms=8, mew=1.8, label="prev")
            a.plot(*first_off, "o", mfc="none", mec="red", ms=8, mew=1.5, label="first")
            a.set_xlim(xmin, xmax)
            a.set_ylim(ymax, ymin)
            a.set_title(f"{titles[name]}\nmin {LA.norm(amin):.1f}px from GT", fontsize=8)
            a.set_xlabel("dx from GT px", fontsize=7)
            if c == 0:
                a.set_ylabel(f"id {v['id']} age {v['age']}\ndy from GT px", fontsize=7)
            a.tick_params(labelsize=6)
            if r == 0 and c == len(names) - 1:
                a.legend(fontsize=6, loc="upper right")
    fig.suptitle("Mid-Air visual hypothesis check: objective minima around exact GT", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}_landscapes.png", dpi=150)

    fig2, ax2 = plt.subplots(n, 9, figsize=(16.2, 2.1 * n), squeeze=False)
    context_names = ["birth", "previous", "current"]
    context_titles = ["birth context", "previous context", "current context"]
    patch_names = ["first", "prev", "cur_gt", "cur_prev", "corr", "warp"]
    patch_titles = ["birth first", "prev patch", "current@GT", "current@prev",
                    "corrected first", "oracle warped first"]
    for r, v in enumerate(victims):
        for c, name in enumerate(context_names):
            ax2[r, c].imshow(v["contexts"][name])
            ax2[r, c].set_title(context_titles[c], fontsize=7)
            ax2[r, c].set_xticks([])
            ax2[r, c].set_yticks([])
            if c == 0:
                ax2[r, c].set_ylabel(f"id {v['id']}\nage {v['age']}", fontsize=7)
        for k, name in enumerate(patch_names):
            c = k + len(context_names)
            ax2[r, c].imshow(norm_patch(v["templates"][name]), cmap="gray", vmin=0, vmax=1)
            ax2[r, c].set_title(patch_titles[k], fontsize=7)
            ax2[r, c].set_xticks([])
            ax2[r, c].set_yticks([])
    fig2.suptitle("Large spatial context plus template/patch appearance", fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(f"{out}_patches.png", dpi=150)

    print(f"\nsaved {out}_landscapes.png")
    print(f"saved {out}_patches.png")


if __name__ == "__main__":
    main()
