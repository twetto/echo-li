"""Where does a first-observation KLT tracker actually LAND vs the true feature?

The staleness montage (first_obs_visual.py) was criticised -- rightly -- for using
NCC, which penalises the feature simply moving within the patch. This shows the honest
mechanism instead: run a self-propagating first-observation KLT and a previous-frame
KLT, and mark where each converges on a large context crop centred on the GT-true
feature. If the feature is plainly still there (it is) yet the first-obs window slides
off it, that is the failure -- an algorithmic limit of local translation alignment
against a stale template, not the feature disappearing.

Dots on each crop (crop is centred on GT-true, so green sits at centre):
  green  = GT-true feature centre
  red    = first-observation KLT landing
  yellow = previous-frame KLT landing

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/first_obs_landing.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_01_easy [--patch 124 --out first_obs_land]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
import photometric_klt_ab as pk  # noqa: E402  (build_pyramid, klt_track, sample)
import echo_li  # noqa: E402

AGES = [0, 5, 10, 20, 40, 80, 120]
LV, R, ITERS = 3, 7, 12


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="first_obs_land")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-tracks", type=int, default=6)
    ap.add_argument("--patch", type=int, default=124)
    ap.add_argument("--fb-thr", type=float, default=999.0,
                    help="forward-backward gate [px]: drop a track when its round-trip "
                         "into the previous frame exceeds this (999 = ungated)")
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = dcoef[:4]

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))

    def cam_pose(t):
        m = np.eye(4)
        m[:3, :3] = slerp(t).as_matrix()
        m[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return m @ t_bs

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    off = np.arange(-R, R + 1, dtype=np.float32)
    OFFX = np.repeat(off, len(off)); OFFY = np.tile(off, len(off))
    ps = args.patch; hp = ps // 2 + 1

    anchors = {}          # fid -> (X_world, birth_i)
    first_patch = {}      # fid -> (LV, P) first-observation template patches
    pos_first = {}        # fid -> [x, y] self-propagated first-obs KLT position
    pos_prev = {}         # fid -> [x, y] self-propagated previous-frame KLT position
    rec = {}              # fid -> {age: (gp, pfirst|None, pprev|None, crop)}
    prev = None

    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        eq = cv2.equalizeHist(img)
        feats, _ = tracker.process(img)
        raw = {int(f["id"]): np.array([float(f["x"]), float(f["y"])]) for f in feats}
        alive = set(raw)
        if feats:
            und = cv2.undistortPoints(
                np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2),
                K, D, P=K).reshape(-1, 2)
        else:
            und = np.zeros((0, 2))
        und_by = {int(f["id"]): uv for f, uv in zip(feats, und)}
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=True)

        if prev is not None:
            ppyr, pgx, pgy = prev
            # previous-frame KLT (template = prev frame at pos_prev, init pos_prev)
            def fb_gate(p0, u, v):
                if args.fb_thr < 900:
                    ub, vb, _ = pk.klt_track(cur_pyr, ppyr, pgx, pgy, u, R, ITERS, "ssd")
                    fb = np.hypot(*(ub - p0).T)
                    v = v & vb & (fb <= args.fb_thr)
                return v
            fp = [f for f in pos_prev if f in alive]
            if fp:
                p0 = np.array([pos_prev[f] for f in fp])
                u, v, _ = pk.klt_track(ppyr, cur_pyr, pgx, pgy, p0, R, ITERS, "ssd")
                v = fb_gate(p0, u, v)
                for k, f in enumerate(fp):
                    if v[k]:
                        pos_prev[f] = u[k]
                    else:
                        pos_prev.pop(f, None)
            # first-observation KLT (template ~ stored first patch via blendB lam huge)
            ff = [f for f in pos_first if f in alive]
            if ff:
                tref = np.array([first_patch[f] for f in ff])
                p0 = np.array([pos_first[f] for f in ff])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy, p0, R, ITERS,
                                       "blendB", "ssd", 1e6, tref)
                v = fb_gate(p0, u, v)
                for k, f in enumerate(ff):
                    if v[k]:
                        pos_first[f] = u[k]
                    else:
                        pos_first.pop(f, None)

        t_cw = np.linalg.inv(cam_pose(t))
        for fid in alive:
            if fid not in anchors:
                continue
            pc = t_cw[:3, :3] @ anchors[fid][0] + t_cw[:3, 3]
            if pc[2] < 0.1:
                continue
            gp = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].ravel()
            age = i - anchors[fid][1]
            if age in AGES and hp <= gp[0] < w - hp and hp <= gp[1] < h - hp:
                crop = cv2.getRectSubPix(eq, (ps, ps), (float(gp[0]), float(gp[1])))
                rec.setdefault(fid, {})[age] = (
                    gp.copy(), pos_first.get(fid, None), pos_prev.get(fid, None), crop)

        # births
        for fid in alive:
            if fid in anchors:
                continue
            uv = und_by[fid]
            d0 = zbuf_lookup(zbufs[i], [uv], w, h)[0]
            if not np.isfinite(d0):
                continue
            pc = np.array([(uv[0] - cx) / fx * d0, (uv[1] - cy) / fy * d0, d0])
            t_wc = cam_pose(t)
            anchors[fid] = (t_wc[:3, :3] @ pc + t_wc[:3, 3], i)
            bx, by = raw[fid]
            first_patch[fid] = np.stack(
                [pk.sample(cur_pyr[lv], np.array([bx * 0.5 ** lv]), np.array([by * 0.5 ** lv]),
                           OFFX, OFFY)[0] for lv in range(LV)])
            pos_first[fid] = raw[fid].copy()
            pos_prev[fid] = raw[fid].copy()
        for fid in [f for f in anchors if f not in alive]:
            del anchors[fid]
            first_patch.pop(fid, None); pos_first.pop(fid, None); pos_prev.pop(fid, None)
        prev = (cur_pyr, cgx, cgy)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(anchors)} "
                  f"long={sum(1 for v in rec.values() if 80 in v)}")

    # select long tracks where the first-obs KLT drifted MOST off the true centre
    cand = []
    for fid, rm in rec.items():
        old = max((a for a in rm if a > 0), default=0)
        if old >= 80 and rm[old][1] is not None:
            gp, pf, _, _ = rm[old]
            cand.append((-np.hypot(*(pf - gp)), fid))
    cand.sort()
    sel = [fid for _, fid in cand[: args.n_tracks]]
    print(f"selected {len(sel)} tracks with the largest first-obs drift")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = AGES
    fig, ax = plt.subplots(len(sel), len(cols), figsize=(1.7 * len(cols), 1.8 * len(sel)))
    if len(sel) == 1:
        ax = ax[None, :]
    for r, fid in enumerate(sel):
        rm = rec[fid]
        for c, age in enumerate(cols):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            if age not in rm:
                a.axis("off"); continue
            gp, pf, pp, crop = rm[age]
            a.imshow(crop, cmap="gray", vmin=0, vmax=255)
            cen = ps / 2
            a.add_patch(plt.Rectangle((cen - R, cen - R), 2 * R + 1, 2 * R + 1,
                                      fill=False, ec="0.4", lw=0.8))
            a.plot(cen, cen, "+", color="lime", ms=9, mew=2)                 # GT true
            if pp is not None:
                a.plot(pp[0] - gp[0] + cen, pp[1] - gp[1] + cen, "x", color="gold", ms=7, mew=2)
            dp = np.hypot(*(pp - gp)) if pp is not None else np.nan
            if pf is not None:
                d = np.hypot(*(pf - gp))
                a.plot(pf[0] - gp[0] + cen, pf[1] - gp[1] + cen, "o", mfc="none",
                       mec="red", ms=9, mew=2)
                a.set_title(f"n={age}  1st {d:.0f} / prev {dp:.0f} px", fontsize=7)
            else:
                a.set_title(f"n={age}  1st LOST / prev {dp:.0f}", fontsize=7, color="red")
            if c == 0:
                a.set_ylabel(f"id {fid}", fontsize=8)
    fig.suptitle("Where the trackers LAND (crop centred on GT-true).  green + = truth   "
                 "red o = first-obs KLT   gold x = previous-frame KLT", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out + "_montage.png", dpi=130)
    print(f"saved {args.out}_montage.png")

    # ---- UNBIASED aggregate: first-obs vs previous-frame offset from GT vs age ----
    # over ALL anchored tracks (not the cherry-picked worst). Report survivor median
    # AND survival rate (first-obs 'LOST' tracks are excluded from the median, so the
    # median UNDER-states first-obs badness -- the survival rate carries the rest).
    ages_e = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]
    print(f"\n{'age':>8} | {'nseen':>6} {'1st_alive%':>10} {'1st_off':>8} {'prev_off':>9}")
    xs, yf, yp = [], [], []
    for lo, hi in ages_e:
        fo, po, seen, alive1 = [], [], 0, 0
        for rm in rec.values():
            for age, (gp, pf, pp, _) in rm.items():
                if age < lo or age >= hi:
                    continue
                seen += 1
                if pf is not None:
                    alive1 += 1; fo.append(np.hypot(*(pf - gp)))
                if pp is not None:
                    po.append(np.hypot(*(pp - gp)))
        if seen > 15:
            mf = np.median(fo) if fo else np.nan
            mp = np.median(po) if po else np.nan
            print(f"{lo:3d}-{hi:<4d} | {seen:6d} {100*alive1/seen:9.0f}% {mf:8.2f} {mp:9.2f}")
            xs.append(0.5 * (lo + hi)); yf.append(mf); yp.append(mp)

    import matplotlib.pyplot as plt
    fig2, a2 = plt.subplots(figsize=(7, 4.5))
    a2.plot(xs, yf, "o-", color="red", label="first-obs KLT (survivors)")
    a2.plot(xs, yp, "s-", color="goldenrod", label="previous-frame KLT")
    a2.set_xlabel("track age [frames]"); a2.set_ylabel("median |landing - GT| [px]")
    a2.set_title("first-obs vs previous-frame KLT offset from truth (unbiased, all tracks)")
    a2.grid(alpha=0.3); a2.legend()
    fig2.tight_layout(); fig2.savefig(args.out + "_offsets.png", dpi=130)
    print(f"saved {args.out}_offsets.png")


if __name__ == "__main__":
    main()
