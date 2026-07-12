"""Paired previous-frame translation vs affine KLT diagnostics.

This joins two independent Rudolf-V frontend runs by track id and frame, then
logs whether affine's extra warp freedom buys photometric quality or only adds
center jitter. Ground-truth flow uses the same EuRoC Leica/Vicon depth machinery
as reference_klt_ab.py.
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


def load_frames(root):
    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        return [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]


def make_tracker(config, w, h, fx, fy, cx, cy, dcoef, klt_warp):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.klt_template_policy = "previous"
    fcfg.klt_warp = klt_warp
    fcfg.klt_reference_warp = "translation"
    fcfg.klt_residual = True
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    return echo_li.Frontend(fcfg, w, h)


def tracker_step(tracker, img, K, dcoef):
    feats, _ = tracker.process(img)
    meta = {int(m["id"]): m for m in tracker.track_meta()}
    raw = {int(f["id"]): np.array([float(f["x"]), float(f["y"])]) for f in feats}
    if feats:
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
    else:
        und = np.zeros((0, 2))
    und_by_id = {int(f["id"]): uv for f, uv in zip(feats, und)}
    return raw, und_by_id, meta


def classify_f_inliers(prev_raw, raw):
    common = [fid for fid in raw if fid in prev_raw]
    flags = {}
    if len(common) < 8:
        return flags
    p0 = np.array([prev_raw[fid] for fid in common], np.float32)
    p1 = np.array([raw[fid] for fid in common], np.float32)
    _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return {fid: True for fid in common}
    return {fid: bool(inlier) for fid, inlier in zip(common, mask.ravel())}


def bilerp_gray(img, x, y):
    h, w = img.shape
    x = float(np.clip(x, 0.0, w - 1.0))
    y = float(np.clip(y, 0.0, h - 1.0))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    return (
        (1.0 - fx) * (1.0 - fy) * img[y0, x0]
        + fx * (1.0 - fy) * img[y0, x1]
        + (1.0 - fx) * fy * img[y1, x0]
        + fx * fy * img[y1, x1]
    )


def affine_hessian_stats(img, x, y, radius):
    h = np.zeros((6, 6), np.float64)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            px = x + ox
            py = y + oy
            gx = 0.5 * (bilerp_gray(img, px + 1.0, py) - bilerp_gray(img, px - 1.0, py))
            gy = 0.5 * (bilerp_gray(img, px, py + 1.0) - bilerp_gray(img, px, py - 1.0))
            j = np.array([gx * ox, gx * oy, gy * ox, gy * oy, gx, gy], np.float64)
            h += np.outer(j, j)
    eig = np.linalg.eigvalsh(h)
    eig_max = float(np.max(eig))
    eig_min = float(np.min(eig))
    cond = eig_max / eig_min if eig_min > 1e-12 else np.inf
    return eig_min, eig_max, cond


def gt_project_error(fid, prev_state, cur_und, zbufs, cam_pose, fx, fy, cx, cy, w, h, t):
    i0, t0, prev_und = prev_state
    if fid not in prev_und or fid not in cur_und:
        return np.nan
    d = zbuf_lookup(zbufs[i0], [prev_und[fid]], w, h)[0]
    if not np.isfinite(d):
        return np.nan
    u0, v0 = prev_und[fid]
    pc0 = np.array([(u0 - cx) / fx * d, (v0 - cy) / fy * d, d])
    t_wc0 = cam_pose(t0)
    xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
    t_cw1 = np.linalg.inv(cam_pose(t))
    pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
    if pc1[2] < 0.1:
        return np.nan
    ugt = fx * pc1[0] / pc1[2] + cx
    vgt = fy * pc1[1] / pc1[2] + cy
    u1, v1 = cur_und[fid]
    return float(np.hypot(u1 - ugt, v1 - vgt))


def summarize(rows):
    if not rows:
        print("no paired rows")
        return
    arr = {k: np.array([r[k] for r in rows], float) for k in [
        "center_delta", "quality_delta", "err_translation", "err_affine",
        "age_translation", "affine_h_min", "affine_h_cond",
    ]}
    finite = np.isfinite(arr["err_translation"]) & np.isfinite(arr["err_affine"])
    print(f"\npaired rows: {len(rows)}  finite GT pairs: {int(np.sum(finite))}")
    if np.any(finite):
        de = arr["err_affine"][finite] - arr["err_translation"][finite]
        print(
            "affine-translation GT error delta: "
            f"median {np.median(de):.3f}px  p90 {np.percentile(de, 90):.3f}px  "
            f"frac worse {100*np.mean(de > 0):.1f}%"
        )
        print(
            "center disagreement: "
            f"median {np.median(arr['center_delta'][finite]):.3f}px  "
            f"p90 {np.percentile(arr['center_delta'][finite], 90):.3f}px"
        )
        print(
            "quality delta affine-translation: "
            f"median {np.nanmedian(arr['quality_delta'][finite]):.4f}  "
            f"p90 {np.nanpercentile(arr['quality_delta'][finite], 90):.4f}"
        )
        improved_quality = arr["quality_delta"][finite] > 0
        worsened_error = de > 0
        if np.any(improved_quality):
            print(
                "when affine quality improves: "
                f"count {int(np.sum(improved_quality))}, "
                f"frac GT worse {100*np.mean(worsened_error[improved_quality]):.1f}%"
            )
        bad_cond = arr["affine_h_cond"][finite] > np.nanpercentile(arr["affine_h_cond"][finite], 75)
        if np.any(bad_cond):
            print(
                "top-quartile affine Hessian condition: "
                f"median err delta {np.nanmedian(de[bad_cond]):.3f}px, "
                f"frac GT worse {100*np.mean(worsened_error[bad_cond]):.1f}%"
            )


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-pair-delta", type=float, default=20.0)
    ap.add_argument("--out", default="paired_affine_klt_diag.csv")
    ap.add_argument("--plot", default="paired_affine_klt_diag.png")
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
    hessian_radius = echo_li.FrontendConfig.from_yaml(args.config).klt_window

    trackers = {
        "translation": make_tracker(args.config, w, h, fx, fy, cx, cy, dcoef, "translation"),
        "affine": make_tracker(args.config, w, h, fx, fy, cx, cy, dcoef, "affine"),
    }
    prev = {k: None for k in trackers}
    prev_img = None
    rows = []
    tstart = time.time()

    for i, (t, path) in enumerate(frames):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        cur = {}
        for name, tracker in trackers.items():
            raw, und, meta = tracker_step(tracker, img, K, dcoef)
            f_flags = classify_f_inliers(prev[name][2] if prev[name] else {}, raw)
            cur[name] = (raw, und, meta, f_flags)

        if i % args.every == 0 and prev["translation"] is not None:
            common = (
                set(cur["translation"][0])
                & set(cur["affine"][0])
                & set(prev["translation"][1])
                & set(prev["affine"][1])
            )
            wmag = float(np.interp(t, gt_t, gt_w))
            for fid in common:
                raw_t, und_t, meta_t, flags_t = cur["translation"]
                raw_a, und_a, meta_a, flags_a = cur["affine"]
                err_t = gt_project_error(
                    fid, (prev["translation"][0], prev["translation"][3], prev["translation"][1]),
                    und_t, zbufs, cam_pose, fx, fy, cx, cy, w, h, t
                )
                err_a = gt_project_error(
                    fid, (prev["affine"][0], prev["affine"][3], prev["affine"][1]),
                    und_a, zbufs, cam_pose, fx, fy, cx, cy, w, h, t
                )
                center_delta = float(np.linalg.norm(raw_a[fid] - raw_t[fid]))
                if center_delta > args.max_pair_delta:
                    continue
                q_t = float(meta_t.get(fid, {}).get("klt_quality", np.nan))
                q_a = float(meta_a.get(fid, {}).get("klt_quality", np.nan))
                h_min, h_max, h_cond = affine_hessian_stats(
                    prev_img,
                    prev["translation"][2][fid][0],
                    prev["translation"][2][fid][1],
                    hessian_radius,
                )
                rows.append({
                    "frame": i,
                    "time": t,
                    "id": fid,
                    "age_translation": float(meta_t.get(fid, {}).get("age", np.nan)),
                    "age_affine": float(meta_a.get(fid, {}).get("age", np.nan)),
                    "center_delta": center_delta,
                    "quality_translation": q_t,
                    "quality_affine": q_a,
                    "quality_delta": q_a - q_t,
                    "err_translation": err_t,
                    "err_affine": err_a,
                    "err_delta": err_a - err_t if np.isfinite(err_t) and np.isfinite(err_a) else np.nan,
                    "f_inlier_translation": int(flags_t.get(fid, True)),
                    "f_inlier_affine": int(flags_a.get(fid, True)),
                    "affine_h_min": h_min,
                    "affine_h_max": h_max,
                    "affine_h_cond": h_cond,
                    "wmag": wmag,
                })

        for name in trackers:
            raw, und, _, _ = cur[name]
            prev[name] = (i, und, raw, t)
        prev_img = img

        if i % 400 == 0:
            fps = i / max(time.time() - tstart, 1e-9)
            print(f"  frame {i}/{len(frames)} paired_rows={len(rows)} {fps:.0f}fps")

    fieldnames = [
        "frame", "time", "id", "age_translation", "age_affine", "center_delta",
        "quality_translation", "quality_affine", "quality_delta",
        "err_translation", "err_affine", "err_delta",
        "f_inlier_translation", "f_inlier_affine",
        "affine_h_min", "affine_h_max", "affine_h_cond", "wmag",
    ]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")
    summarize(rows)

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        center = np.array([r["center_delta"] for r in rows], float)
        ed = np.array([r["err_delta"] for r in rows], float)
        qd = np.array([r["quality_delta"] for r in rows], float)
        cond = np.array([r["affine_h_cond"] for r in rows], float)
        age = np.array([r["age_translation"] for r in rows], float)
        finite = np.isfinite(ed)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].hist(np.clip(center[finite], 0, 5), bins=np.linspace(0, 5, 80))
        ax[0].set_xlabel("affine vs translation center delta [px]")
        ax[1].scatter(center[finite], np.clip(ed[finite], -2, 5), s=3, alpha=0.15)
        ax[1].set_xlabel("center delta [px]")
        ax[1].set_ylabel("GT error delta affine-translation [px]")
        ax[2].scatter(age[finite], np.clip(qd[finite], -1, 1), s=3, alpha=0.15)
        ax[2].set_xlabel("track age [frames]")
        ax[2].set_ylabel("quality delta affine-translation")
        ax[3].scatter(np.log10(cond[finite]), np.clip(ed[finite], -2, 5), s=3, alpha=0.15)
        ax[3].set_xlabel("log10 affine Hessian condition")
        ax[3].set_ylabel("GT error delta [px]")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
