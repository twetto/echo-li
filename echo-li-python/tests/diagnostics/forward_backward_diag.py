"""Forward-backward KLT gate diagnostic on EuRoC GT flow.

The production tracker supplies the forward correspondence. This script tracks the
reported current feature position backward into the previous image with OpenCV LK,
then compares the round-trip error against EuRoC GT-flow error. The question is
whether a WGSL-feasible forward/backward gate is predictive enough to reject bad
correspondences without throwing away most good tracks.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_gt_eval import quat_ang_rate  # noqa: E402
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
import echo_li  # noqa: E402


THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def load_frames(root):
    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        return [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]


def gt_flow_error(common, prev_und, cur_und, zbuf, t_wc0, t_cw1, fx, fy, cx, cy, w, h):
    p0 = [prev_und[fid] for fid in common]
    d0 = zbuf_lookup(zbuf, p0, w, h)
    out = {}
    for fid, (u0, v0), d in zip(common, p0, d0):
        if not np.isfinite(d):
            continue
        pc0 = np.array([(u0 - cx) / fx * d, (v0 - cy) / fy * d, d])
        xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
        pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
        if pc1[2] < 0.1:
            continue
        ugt = fx * pc1[0] / pc1[2] + cx
        vgt = fy * pc1[1] / pc1[2] + cy
        u1, v1 = cur_und[fid]
        out[fid] = float(np.hypot(u1 - ugt, v1 - vgt))
    return out


def summarize(rows):
    out = []
    e = np.array([r["gt_err"] for r in rows], float)
    fb = np.array([r["fb_err"] for r in rows], float)
    outlier = e > 3.0
    finite = np.isfinite(e) & np.isfinite(fb)
    e = e[finite]
    fb = fb[finite]
    outlier = outlier[finite]
    total = len(e)
    if total == 0:
        return out

    out.append({
        "threshold": "none",
        "kept_frac": 1.0,
        "kept_count": total,
        "median_gt_err": float(np.median(e)),
        "p90_gt_err": float(np.percentile(e, 90)),
        "out_3px": float(np.mean(outlier)),
        "rejected_outlier_recall": 0.0,
        "reject_precision": 0.0,
    })

    for th in THRESHOLDS:
        keep = fb <= th
        reject = ~keep
        kept_count = int(np.sum(keep))
        if kept_count == 0:
            continue
        rejected_outliers = np.sum(reject & outlier)
        all_outliers = np.sum(outlier)
        out.append({
            "threshold": th,
            "kept_frac": float(np.mean(keep)),
            "kept_count": kept_count,
            "median_gt_err": float(np.median(e[keep])),
            "p90_gt_err": float(np.percentile(e[keep], 90)),
            "out_3px": float(np.mean(outlier[keep])),
            "rejected_outlier_recall": float(rejected_outliers / max(all_outliers, 1)),
            "reject_precision": float(rejected_outliers / max(np.sum(reject), 1)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--window-radius", type=int, default=15)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--out", default="forward_backward_diag.csv")
    ap.add_argument("--summary", default="forward_backward_summary.csv")
    ap.add_argument("--plot", default="forward_backward_diag.png")
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

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])

    def cam_pose(t):
        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(t).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return t_wb @ t_bs

    frames = [(t, p) for t, p in load_frames(root) if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    if hasattr(fcfg, "klt_template_policy"):
        fcfg.klt_template_policy = "previous"
    if hasattr(fcfg, "klt_warp"):
        fcfg.klt_warp = "translation"
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    lk_win = (2 * args.window_radius + 1, 2 * args.window_radius + 1)
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    prev = None
    rows = []
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        feats, _ = tracker.process(img)
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        raw = {int(f["id"]): (float(f["x"]), float(f["y"])) for f in feats}
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = (
            cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
            if len(feats)
            else np.zeros((0, 2))
        )
        cur_und = {int(f["id"]): uv for f, uv in zip(feats, und)}

        if prev is not None and i % args.every == 0 and raw:
            i0, t0, prev_img, prev_raw, prev_und = prev
            common = [fid for fid in raw if fid in prev_raw]
            if common:
                cur_pts = np.array([raw[fid] for fid in common], np.float32).reshape(-1, 1, 2)
                prev_pts = np.array([prev_raw[fid] for fid in common], np.float32).reshape(-1, 1, 2)
                back_pts, status, lk_err = cv2.calcOpticalFlowPyrLK(
                    img,
                    prev_img,
                    cur_pts,
                    None,
                    winSize=lk_win,
                    maxLevel=args.levels,
                    criteria=lk_criteria,
                )
                gt_err = gt_flow_error(
                    common,
                    prev_und,
                    cur_und,
                    zbufs[i0],
                    cam_pose(t0),
                    np.linalg.inv(cam_pose(t)),
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                )
                wmag = float(np.interp(t, gt_t, gt_w))
                for j, fid in enumerate(common):
                    if fid not in gt_err:
                        continue
                    ok = bool(status[j, 0]) if status is not None else False
                    if ok:
                        fb = float(np.linalg.norm(back_pts[j, 0] - prev_pts[j, 0]))
                        berr = float(lk_err[j, 0]) if lk_err is not None else np.nan
                    else:
                        fb = np.inf
                        berr = np.inf
                    m = meta.get(fid, {})
                    rows.append({
                        "frame": i,
                        "id": fid,
                        "age": float(m.get("age", np.nan)),
                        "fb_err": fb,
                        "back_lk_err": berr,
                        "gt_err": gt_err[fid],
                        "wmag": wmag,
                    })

        prev = (i, t, img, raw, cur_und)
        if i % 400 == 0:
            fps = i / max(time.time() - tstart, 1e-9)
            print(f"  [{i}/{len(frames)}] rows={len(rows)} {fps:.0f}fps")

    fields = ["frame", "id", "age", "fb_err", "back_lk_err", "gt_err", "wmag"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")

    summary = summarize(rows)
    sfields = [
        "threshold",
        "kept_frac",
        "kept_count",
        "median_gt_err",
        "p90_gt_err",
        "out_3px",
        "rejected_outlier_recall",
        "reject_precision",
    ]
    with open(args.summary, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sfields)
        wr.writeheader()
        wr.writerows(summary)
    print(f"saved {args.summary}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.array([r["gt_err"] for r in rows], float)
    fb = np.array([r["fb_err"] for r in rows], float)
    finite = np.isfinite(e) & np.isfinite(fb)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(np.clip(fb[finite], 0, 10), np.clip(e[finite], 0, 20), s=2, alpha=0.12)
    ax[0].set_xlabel("forward-backward round-trip error [px]")
    ax[0].set_ylabel("GT-flow error [px]")
    ax[0].grid(alpha=0.3)
    sm = [r for r in summary if r["threshold"] != "none"]
    xs = [float(r["threshold"]) for r in sm]
    ax[1].plot(xs, [100 * float(r["kept_frac"]) for r in sm], marker="o", label="kept")
    ax[1].plot(xs, [100 * float(r["out_3px"]) for r in sm], marker="o", label="kept >3px")
    ax[1].plot(xs, [100 * float(r["rejected_outlier_recall"]) for r in sm], marker="o", label="outlier recall")
    ax[1].set_xlabel("FB gate threshold [px]")
    ax[1].set_ylabel("%")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
