"""Human-inspection diagnostic: WHAT are the high-drift tracks? Anchor each track's
true 3D point at birth (exact MidAir depth+pose), reproject it into every frame
(GT point), and overlay the Rudolf-tracked point; the line between them is the live
cumulative drift. Color-code by per-track drift rank so we can visually identify the
cause of the clean-inlier heavy tail (repetitive texture? slant? specular?).

Two passes: (1) track + record per-frame (GT point, tracked point, occlusion) and
per-track cumulative drift; (2) rank tracks, re-read frames, draw overlays -> video;
plus a zoomed montage (birth/mid/late crops) of the worst CLEAN (non-occluded) tracks.

  colors:  green = normal    RED = clean top-drifter    CYAN = occluded top-drifter

  PY=echo-li-python/venv/bin/python
  $PY midair_drift_inspect.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 300 --config configs/diagnostics_midair_sparse3d.yaml \
      --out /tmp/drift_inspect
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import make_frontend  # noqa: E402

GREEN = (90, 200, 90)
RED = (40, 40, 235)
CYAN = (220, 220, 40)
GRAY = (150, 150, 150)


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "diagnostics_midair_sparse3d.yaml"))
    ap.add_argument("--border", type=int, default=16)
    ap.add_argument("--min-obs", type=int, default=10)
    ap.add_argument("--top-frac", type=float, default=0.20)
    ap.add_argument("--montage-n", type=int, default=15)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--out", default="midair_drift_inspect")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, _ = make_frontend(args.config, f, cx, cy, W, H)
    last = min(args.start + args.frames, ds.n)
    print(f"MidAir {args.subset}/{args.cond}/{ds.traj}  {W}x{H} f={f:.1f}  frames {args.start}..{last}")

    # ---- pass 1: track, anchor at birth, record per-frame overlay data + per-track drift ----
    Xw = {}; born = {}
    per_frame = {}          # i -> list of (fid, p1(x,y), gp(x,y), occ)
    cum = {}                # fid -> (last_drift_px, n_obs, sum_drift, max_drift)
    for i in range(args.start, last):
        img = ds.image(i); depth = ds.depth(i); T_wb = ds.pose(i)
        feats, _ = tracker.process(img)
        recs = []
        for fd in feats:
            fid = int(fd["id"]); x, y = float(fd["x"]), float(fd["y"])
            if not (args.border <= x < W - args.border and args.border <= y < H - args.border):
                continue
            if fid not in born:
                gx, gy = int(round(x)), int(round(y))
                d0 = float(depth[gy, gx])
                if 1.0 < d0 < md.SKY:
                    Xw[fid] = md.backproject_world((x, y), d0, T_wb, f, cx, cy)
                    born[fid] = i
                continue
            gp, rng, _z = md.project_world(Xw[fid], T_wb, f, cx, cy)
            if gp is None:
                continue
            gxp, gyp = int(round(gp[0])), int(round(gp[1]))
            occ = 0
            if 0 <= gxp < W and 0 <= gyp < H:
                map_rng = float(depth[gyp, gxp])
                occ = int(1.0 < map_rng < rng - max(0.05 * rng, 0.5))
            drift = float(np.hypot(x - gp[0], y - gp[1]))
            recs.append((fid, (x, y), (float(gp[0]), float(gp[1])), occ))
            c = cum.get(fid, [0.0, 0, 0.0, 0.0])
            cum[fid] = [drift, c[1] + 1, c[2] + drift, max(c[3], drift)]
        per_frame[i] = recs
        if (i - args.start) % 100 == 0:
            print(f"  pass1 [{i-args.start}/{last-args.start}] tracks={len(feats)}")

    # ---- rank tracks by drift rate (mean cumulative drift / obs), split by occlusion ----
    stats = {}
    occ_life = {}
    for i, recs in per_frame.items():
        for fid, _p, _g, occ in recs:
            occ_life[fid] = occ_life.get(fid, [0, 0]); occ_life[fid][0] += occ; occ_life[fid][1] += 1
    for fid, (last_d, n, sum_d, max_d) in cum.items():
        if n >= args.min_obs:
            stats[fid] = (sum_d / n, max_d, n, occ_life[fid][0] / occ_life[fid][1])
    rates = np.array([v[0] for v in stats.values()])
    thr = np.quantile(rates, 1 - args.top_frac)
    top = {fid for fid, v in stats.items() if v[0] >= thr}
    top_clean = sorted([fid for fid in top if stats[fid][3] < 0.2],
                       key=lambda j: -stats[j][0])
    top_occ = {fid for fid in top if stats[fid][3] >= 0.2}
    print(f"\n{len(stats)} tracks (>= {args.min_obs} obs).  drift-rate median={np.median(rates):.2f} "
          f"p90={np.percentile(rates,90):.2f}px.  top {args.top_frac:.0%} (>= {thr:.2f}px): "
          f"{len(top)}  ({len(top_occ)} occluded, {len(top_clean)} clean)")

    # ---- pass 2: render video ----
    vpath = f"{args.out}_{args.cond}.mp4"
    vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for i in range(args.start, last):
        vis = cv2.cvtColor(ds.image(i), cv2.COLOR_GRAY2BGR)
        for fid, (x, y), (gx, gy), occ in per_frame.get(i, []):
            is_top = fid in top
            col = (CYAN if fid in top_occ else RED) if is_top else GREEN
            th = 2 if is_top else 1
            cv2.line(vis, (int(gx), int(gy)), (int(x), int(y)), col, th, cv2.LINE_AA)
            cv2.rectangle(vis, (int(gx) - 3, int(gy) - 3), (int(gx) + 3, int(gy) + 3), col, th)  # GT
            cv2.circle(vis, (int(x), int(y)), 2 if is_top else 1, col, -1, cv2.LINE_AA)          # tracker
        cv2.rectangle(vis, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(vis, f"frame {i}   square=GT  dot=tracker  line=drift", (6, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(vis, "RED=clean top-drifter  CYAN=occluded top-drifter  green=normal", (6, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(vis)
    vw.release()
    print(f"saved video: {vpath}")

    # ---- montage: worst CLEAN top-drifters, crops at birth / mid / late ----
    R = 34; UP = 3; cols = 3
    tiles = []
    for fid in top_clean[:args.montage_n]:
        frs = sorted(i for i, recs in per_frame.items() if any(r[0] == fid for r in recs))
        if len(frs) < 3:
            continue
        picks = [frs[0], frs[len(frs) // 2], frs[-1]]
        strip = []
        for i in picks:
            rec = next(r for r in per_frame[i] if r[0] == fid)
            _, (x, y), (gx, gy), occ = rec
            img = cv2.cvtColor(ds.image(i), cv2.COLOR_GRAY2BGR)
            cxi, cyi = int(round(x)), int(round(y))
            x0, y0 = max(0, cxi - R), max(0, cyi - R)
            crop = img[y0:y0 + 2 * R, x0:x0 + 2 * R]
            crop = cv2.resize(crop, (2 * R * UP, 2 * R * UP), interpolation=cv2.INTER_NEAREST)
            gxl, gyl = int((gx - x0) * UP), int((gy - y0) * UP)
            xl, yl = int((x - x0) * UP), int((y - y0) * UP)
            cv2.line(crop, (gxl, gyl), (xl, yl), RED, 1, cv2.LINE_AA)
            cv2.rectangle(crop, (gxl - 4, gyl - 4), (gxl + 4, gyl + 4), (60, 220, 60), 2)   # GT green
            cv2.circle(crop, (xl, yl), 3, RED, -1, cv2.LINE_AA)                              # tracker red
            cv2.putText(crop, f"age{i-born[fid]} d{stats[fid][0]:.1f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
            strip.append(crop)
        tiles.append(np.hstack(strip))
    if tiles:
        wmax = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 2, 2, 2, wmax - t.shape[1] + 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
        rows = [np.vstack(tiles[k:k + 1]) for k in range(len(tiles))]
        # arrange into `cols` columns of stacked strips
        colw = [rows[k::cols] for k in range(cols)]
        hmax = max(sum(t.shape[0] for t in c) for c in colw)
        packed = []
        for c in colw:
            col_img = np.vstack(c) if c else np.zeros((hmax, wmax + 4, 3), np.uint8)
            if col_img.shape[0] < hmax:
                col_img = cv2.copyMakeBorder(col_img, 0, hmax - col_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            packed.append(col_img)
        montage = np.hstack(packed)
        mpath = f"{args.out}_{args.cond}_montage.png"
        cv2.imwrite(mpath, montage)
        print(f"saved montage: {mpath}  (worst {len(tiles)} clean top-drifters, birth/mid/late)")


if __name__ == "__main__":
    main()
