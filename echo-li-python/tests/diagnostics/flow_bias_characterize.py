"""Characterize the tracker flow bias beta: its SHAPE and its TRIGGERS.

The depth-NEES inconsistency traced to a correlated flow bias beta (see
sparse3d_inconsistency_decomposition.md). This measures beta directly instead of
inferring it. Ground truth: each track is anchored at its FIRST frame to the 3D
scene point (GT Leica depth + Vicon pose); at every later frame the perfect-tracker
pixel is that point reprojected with the GT pose. beta(age) = tracker_pixel -
reprojected_pixel is the accumulated drift (beta(0)=0). NB the CUMULATIVE residual
is used on purpose: the frame-to-frame residual cancels a persistent offset.

Analyses:
  1. Temporal shape  -- is beta a constant offset, a drift, or random? (spaghetti +
     mean|beta| vs age + within-track direction autocorrelation)
  2. Direction shape -- beta relative to the local image gradient (aperture problem
     predicts beta perpendicular to grad) and radial/tangential.
  3. Triggers        -- |beta| vs {ang. rate w, grad magnitude, age, depth, radius}.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/flow_bias_characterize.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--out flow_bias.png]
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
from flow_gt_eval import quat_ang_rate  # noqa: E402
import echo_li  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="flow_bias.png")
    ap.add_argument("--min-age", type=int, default=5)
    ap.add_argument("--anchor-frames", type=int, default=1,
                    help="average the GT-depth anchor 3D point over the first N frames")
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
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    def cam_pose(t):
        m = np.eye(4)
        m[:3, :3] = slerp(t).as_matrix()
        m[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return m @ t_bs

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    anchors = {}   # fid -> (X_world, frame0)
    anchor_buf = {}  # fid -> list of early backprojected world points (averaging)
    per_track = {}  # fid -> list of (age, bx, by, gx, gy, wmag, radius, depth)
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        if not feats:
            continue
        gximg = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gyimg = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        t_wc = cam_pose(t)
        t_cw = np.linalg.inv(t_wc)
        wmag = float(np.interp(t, gt_t, gt_w))
        und = cv2.undistortPoints(
            np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2),
            K, D, P=K).reshape(-1, 2)
        for f, (uu, vv) in zip(feats, und):
            fid = int(f["id"])
            if fid not in anchors:
                d0 = zbuf_lookup(zbufs[i], [(uu, vv)], w, h)[0]
                if np.isfinite(d0):
                    pc = np.array([(uu - cx) / fx * d0, (vv - cy) / fy * d0, d0])
                    buf = anchor_buf.setdefault(fid, [])
                    buf.append(t_wc[:3, :3] @ pc + t_wc[:3, 3])
                    if len(buf) >= args.anchor_frames:
                        anchors[fid] = (np.mean(buf, axis=0), i)
                        per_track[fid] = []
                        del anchor_buf[fid]
                continue
            X, i0 = anchors[fid]
            pc1 = t_cw[:3, :3] @ X + t_cw[:3, 3]
            if pc1[2] < 0.1:
                continue
            proj = cv2.projectPoints(pc1.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                     K, D)[0].ravel()
            rawx, rawy = float(f["x"]), float(f["y"])
            bx, by = rawx - proj[0], rawy - proj[1]
            ix, iy = int(round(rawx)), int(round(rawy))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            per_track[fid].append((i - i0, bx, by, gximg[iy, ix], gyimg[iy, ix],
                                   wmag, np.hypot(rawx - cx, rawy - cy), pc1[2]))
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(per_track)}")

    # flatten
    rows = [r for v in per_track.values() for r in v if r[0] >= 1]
    A = np.array(rows)  # age, bx, by, gx, gy, w, radius, depth
    age, bx, by = A[:, 0], A[:, 1], A[:, 2]
    gx, gy, wm, rad, dep = A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7]
    bmag = np.hypot(bx, by)
    gmag = np.hypot(gx, gy) + 1e-9
    # angle of beta relative to gradient (0 = along grad, 90 = perpendicular)
    cosang = np.abs((bx * gx + by * gy) / (bmag + 1e-9) / gmag)
    ang_deg = np.degrees(np.arccos(np.clip(cosang, 0, 1)))

    print(f"\n=== flow bias beta ({len(A)} obs, {len(per_track)} tracks) ===")
    print(f"|beta| median {np.median(bmag):.3f} px  p90 {np.percentile(bmag,90):.2f}  "
          f"p99 {np.percentile(bmag,99):.2f}")
    # temporal: within-track direction autocorrelation + persistent vs drift
    persist, autocorr, driftslope = [], [], []
    for v in per_track.values():
        if len(v) < args.min_age:
            continue
        b = np.array([(r[1], r[2]) for r in v])
        a = np.array([r[0] for r in v])
        mag = np.hypot(b[:, 0], b[:, 1])
        persist.append(np.hypot(*b.mean(0)) / (mag.mean() + 1e-9))  # |mean|/mean| ~1 => fixed dir
        u = b / (mag[:, None] + 1e-9)
        autocorr.append(np.mean(np.sum(u[1:] * u[:-1], axis=1)))     # consecutive dir corr
        driftslope.append(np.polyfit(a, mag, 1)[0])
    persist = np.array(persist); autocorr = np.array(autocorr); driftslope = np.array(driftslope)
    print(f"temporal (tracks>= {args.min_age} obs, n={len(persist)}):")
    print(f"  direction persistence |mean b|/mean|b|:  median {np.median(persist):.2f}  "
          f"(1=fixed direction, 0=random)")
    print(f"  consecutive direction autocorr:          median {np.median(autocorr):.2f}")
    print(f"  |beta| drift slope [px/frame]:           median {np.median(driftslope):+.3f}")
    print(f"direction vs gradient: median angle {np.median(ang_deg):.0f} deg  "
          f"(90=perp/aperture, 0=along)   frac>60deg {np.mean(ang_deg>60)*100:.0f}%")

    def binned(x, y, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            out.append((0.5 * (lo + hi), np.median(y[m]) if m.sum() > 20 else np.nan, int(m.sum())))
        return np.array(out)

    print("triggers (median |beta| px):")
    for name, x, edges in [
        ("ang rate w", wm, np.percentile(wm, [0, 20, 40, 60, 80, 95, 100])),
        ("grad mag", gmag, np.percentile(gmag, [0, 20, 40, 60, 80, 95, 100])),
        ("track age", age, np.array([1, 5, 10, 20, 40, 80, 1e9])),
        ("depth", dep, np.percentile(dep, [0, 20, 40, 60, 80, 100])),
        ("radius", rad, np.percentile(rad, [0, 20, 40, 60, 80, 100])),
    ]:
        b = binned(x, bmag, edges)
        s = "  ".join(f"{c:.0f}:{m:.2f}" for c, m, n in b if np.isfinite(m))
        print(f"  {name:>10}: {s}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # (0,0) spaghetti of |beta| vs age for a sample of long tracks
    long = [v for v in per_track.values() if len(v) >= 15][:40]
    for v in long:
        a = [r[0] for r in v]
        m = [np.hypot(r[1], r[2]) for r in v]
        ax[0, 0].plot(a, m, alpha=0.4, lw=0.8)
    ax[0, 0].set_title("|beta| vs age (40 long tracks)")
    ax[0, 0].set_xlabel("track age [frames]"); ax[0, 0].set_ylabel("|beta| [px]")
    # (0,1) mean |beta| vs age binned
    be = binned(age, bmag, np.array([1, 5, 10, 20, 40, 80, 160, 1e9]))
    ax[0, 1].plot(be[:, 0], be[:, 1], "o-")
    ax[0, 1].set_title("median |beta| vs age"); ax[0, 1].set_xlabel("age [frames]")
    ax[0, 1].set_ylabel("median |beta| [px]")
    # (0,2) angle rel gradient hist
    ax[0, 2].hist(ang_deg, bins=45)
    ax[0, 2].axvline(90, color="r", ls="--")
    ax[0, 2].set_title("beta angle vs image gradient\n(90=perp=aperture)")
    ax[0, 2].set_xlabel("angle [deg]")
    # (1,0) |beta| vs w
    bw = binned(wm, bmag, np.percentile(wm, np.linspace(0, 100, 9)))
    ax[1, 0].plot(bw[:, 0], bw[:, 1], "o-")
    ax[1, 0].set_title("median |beta| vs angular rate"); ax[1, 0].set_xlabel("|w| [rad/s]")
    ax[1, 0].set_ylabel("median |beta| [px]")
    # (1,1) |beta| vs grad mag
    bg = binned(gmag, bmag, np.percentile(gmag, np.linspace(0, 100, 9)))
    ax[1, 1].plot(bg[:, 0], bg[:, 1], "o-")
    ax[1, 1].set_title("median |beta| vs grad magnitude"); ax[1, 1].set_xlabel("|grad I|")
    # (1,2) beta direction scatter (grad-parallel vs grad-perp)
    gp = (bx * gx + by * gy) / gmag                       # parallel to grad
    gperp = (bx * (-gy) + by * gx) / gmag                 # perpendicular
    ax[1, 2].scatter(np.clip(gp, -5, 5), np.clip(gperp, -5, 5), s=2, alpha=0.1)
    ax[1, 2].set_title("beta: grad-parallel vs grad-perp")
    ax[1, 2].set_xlabel("along grad [px]"); ax[1, 2].set_ylabel("perp grad [px]")
    ax[1, 2].set_aspect("equal")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
