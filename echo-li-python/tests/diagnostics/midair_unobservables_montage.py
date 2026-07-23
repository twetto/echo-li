"""Curated montage of the common MEASUREMENT unobservables in monocular tracking: aperture
(drift slides along an edge, the along-edge component is unobservable from the patch),
occlusion (the tracker follows the occluder, not the true point), and repetitive texture
(the tracker locks onto a self-similar secondary match). Real MidAir crops with the birth-
anchored ground-truth point (green) vs. the tracker (red), the drift vector, and — for the
aperture case — the weak structure-tensor eigenvector (the slide axis).

  PY=echo-li-python/venv/bin/python
  $PY midair_unobservables_montage.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 220 --out figs/f_unobservables.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend  # noqa: E402

GREEN = (60, 200, 70); RED = (235, 70, 40); YEL = (245, 210, 40)


def eig_weakvec(gray, x, y, r=7):
    xi, yi = int(round(x)), int(round(y)); H, W = gray.shape
    x0, x1 = max(xi - r, 0), min(xi + r + 1, W); y0, y1 = max(yi - r, 0), min(yi + r + 1, H)
    p = gray[y0:y1, x0:x1].astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    Jxx, Jyy, Jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    tr, det = Jxx + Jyy, Jxx * Jyy - Jxy * Jxy
    disc = max(tr * tr - 4 * det, 0) ** 0.5
    lmax, lmin = (tr + disc) / 2, (tr - disc) / 2
    # weak eigenvector = eigvec of lmin (the low-gradient / along-edge direction)
    wv = np.array([Jxy, lmin - Jxx]) if abs(Jxy) > 1e-6 else np.array([1.0, 0.0])
    n = np.linalg.norm(wv); wv = wv / n if n > 1e-9 else np.array([1.0, 0.0])
    return lmin, lmax, wv


def selfsim(gray, x, y, half=5, search=13):
    xi, yi = int(round(x)), int(round(y)); H, W = gray.shape
    if not (half + search <= xi < W - half - search and half + search <= yi < H - half - search):
        return 0.0
    t = gray[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float32)
    win = gray[yi - half - search:yi + half + search + 1, xi - half - search:xi + half + search + 1].astype(np.float32)
    res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
    cv2.circle(res, (search, search), 3, -1.0, -1)
    return float(res.max())


def draw(gray, x, y, gx, gy, wv=None):
    """Crop around the GT/tracked pair, upsample, annotate. Returns an RGB tile."""
    cxp, cyp = (x + gx) / 2, (y + gy) / 2
    r = int(max(16, np.hypot(x - gx, y - gy) * 0.7 + 10)); UP = 7
    H, W = gray.shape
    xi, yi = int(round(cxp)), int(round(cyp))
    x0, y0 = max(xi - r, 0), max(yi - r, 0); x1, y1 = min(xi + r, W), min(yi + r, H)
    crop = gray[y0:y1, x0:x1]
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    crop = cv2.resize(crop, ((x1 - x0) * UP, (y1 - y0) * UP), interpolation=cv2.INTER_NEAREST)
    def P(px, py):
        return int((px - x0) * UP), int((py - y0) * UP)
    if wv is not None:
        a = np.array(P(x, y)); d = (wv * 26 * UP / 7).astype(int)
        cv2.line(crop, tuple(a - d), tuple(a + d), YEL, 2, cv2.LINE_AA)   # slide axis
    cv2.line(crop, P(gx, gy), P(x, y), RED, 2, cv2.LINE_AA)               # drift
    gxx, gyy = P(gx, gy)
    cv2.rectangle(crop, (gxx - 8, gyy - 8), (gxx + 8, gyy + 8), GREEN, 2)  # GT
    cv2.circle(crop, P(x, y), 6, RED, -1, cv2.LINE_AA)                     # tracker
    return crop


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "diagnostics_midair_sparse3d.yaml"))
    ap.add_argument("--out", default="figs/f_unobservables.png")
    ap.add_argument("--min-age", type=int, default=15)
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.frames, ds.n)

    imgs = {}; Xw = {}; born = {}; last_obs = {}; prev_xy = {}
    for i in range(last):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i); imgs[i] = img
        for fd in tracker.process(img)[0]:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (6 <= x < W - 6 and 6 <= y < H - 6):
                continue
            fl = float(np.hypot(x - prev_xy[fid][0], y - prev_xy[fid][1])) if fid in prev_xy else 0.0
            prev_xy[fid] = (x, y)
            if fid not in born:
                gxi, gyi = int(round(x)), int(round(y)); d0 = float(depth[gyi, gxi])
                if 1.0 < d0 < md.SKY:
                    Xw[fid] = md.backproject_world((x, y), d0, T, f, cx, cy); born[fid] = i
                continue
            if fid not in Xw:
                continue
            gp, rng, z = md.project_world(Xw[fid], T, f, cx, cy)
            if gp is None or z <= 0.1:
                continue
            gxr = int(round(min(max(gp[0], 0), W - 1))); gyr = int(round(min(max(gp[1], 0), H - 1)))
            occ = float(depth[gyr, gxr]) < z * 0.96
            # local depth roughness at the true point: small = flat ground (grass); large = depth edge
            dp = depth[max(gyr - 4, 0):gyr + 5, max(gxr - 4, 0):gxr + 5]
            dpv = dp[(dp > 0.5) & (dp < md.SKY)]
            dsmooth = float(np.std(dpv) / (np.median(dpv) + 1e-6)) if dpv.size > 5 else 9.9
            drift = float(np.hypot(x - gp[0], y - gp[1]))
            last_obs[fid] = (i, x, y, float(gp[0]), float(gp[1]), occ, drift, i - born[fid], fl, dsmooth)

    # keep only in-frame, in-range drift (exclude runaway tracks whose true point left the FOV)
    cand = [(fid, *v) for fid, v in last_obs.items()
            if v[7] >= args.min_age and 1.5 < v[6] < 45.0
            and 8 <= v[3] < W - 8 and 8 <= v[4] < H - 8]
    # attach patch descriptors
    rows = []
    for fid, i, x, y, gx, gy, occ, drift, age, fl, dsmooth in cand:
        g = imgs[i]
        lmin, lmax, wv = eig_weakvec(g, x, y)
        ss = selfsim(g, x, y)
        dv = np.array([x - gx, y - gy]); dv = dv / (np.linalg.norm(dv) + 1e-9)
        align = abs(float(dv @ wv))                      # |cos| between drift and weak eigvec
        edge = lmax / (lmin + 1e-6)
        rows.append(dict(fid=fid, i=i, x=x, y=y, gx=gx, gy=gy, occ=occ, drift=drift, age=age,
                         flow=fl, dsmooth=dsmooth, lmin=lmin, edge=edge, ss=ss, align=align, wv=wv))

    def pick(pred, key, rev=True):
        c = [r for r in rows if pred(r)]
        return sorted(c, key=key, reverse=rev)[0] if c else None

    aperture = pick(lambda r: (not r["occ"]) and 3 < r["drift"] < 22 and r["edge"] > 6 and r["align"] > 0.8,
                    lambda r: r["align"] * min(r["drift"], 15))
    occl = pick(lambda r: r["occ"] and 4 < r["drift"] < 40, lambda r: min(r["drift"], 25))
    # genuine repetitive failure = fast-moving ground texture (bottom of frame, large flow) that is
    # self-similar, isotropic (not an edge), and on FLAT depth (small roughness -> no depth-edge/occlusion
    # confound). Universal across weathers; the near ground at the frame bottom has the largest flow.
    flow_hi = np.percentile([r["flow"] for r in rows], 60) if rows else 0.0
    repet = pick(lambda r: (not r["occ"]) and r["y"] > 0.5 * H and r["flow"] > flow_hi
                 and r["ss"] > 0.70 and r["edge"] < 4.0 and r["dsmooth"] < 0.06 and 3 < r["drift"] < 25,
                 lambda r: r["ss"] * r["flow"])
    picks = [("Aperture — drift slides along the edge\n(along-edge component unobservable)", aperture, True),
             ("Occlusion — tracker follows the occluder\n(true point hidden)", occl, False),
             ("Repetitive texture — locks to a\nself-similar secondary match", repet, False)]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, (title, r, show_wv) in zip(axes, picks):
        if r is None:
            ax.text(0.5, 0.5, "no example found", ha="center"); ax.axis("off"); continue
        tile = draw(imgs[r["i"]], r["x"], r["y"], r["gx"], r["gy"], r["wv"] if show_wv else None)
        ax.imshow(tile); ax.set_title(title, fontsize=8.6)
        ax.set_xlabel(f"drift {r['drift']:.1f} px   (age {r['age']})", fontsize=7.6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = args.out if Path(args.out).is_absolute() else str(Path.cwd() / args.out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("wrote", out)
    for name, r, _ in picks:
        print(f"  {name.splitlines()[0]:<45} {'FOUND fid=%d drift=%.1f' % (r['fid'], r['drift']) if r else 'MISSING'}")


if __name__ == "__main__":
    main()
