"""Estimate true local affine patch motion from EuRoC depth + Vicon pose.

This tests whether adjacent-frame KLT patches actually need non-translation
geometry. For each feature, sample a small pattern around the previous-frame
pixel, backproject each sample with Leica depth, transform by GT pose, reproject
into the current frame, and fit:

    current_uv = reference_center + [tx, ty] + A * reference_offset

If A is usually close to identity, affine KLT mostly estimates noise.
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


def make_tracker(config, w, h, fx, fy, cx, cy, dcoef):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.klt_template_policy = "previous"
    fcfg.klt_warp = "translation"
    fcfg.klt_reference_warp = "translation"
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    return echo_li.Frontend(fcfg, w, h)


def tracker_step(tracker, img, K, dcoef):
    feats, _ = tracker.process(img)
    if feats:
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
    else:
        und = np.zeros((0, 2))
    return {int(f["id"]): uv for f, uv in zip(feats, und)}


def project_points(points_uv, zbuf, t_wc0, t_cw1, fx, fy, cx, cy, w, h):
    depths = zbuf_lookup(zbuf, points_uv, w, h)
    projected = []
    for (u, v), d in zip(points_uv, depths):
        if not np.isfinite(d):
            projected.append(None)
            continue
        pc0 = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d])
        xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
        pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
        if pc1[2] < 0.1:
            projected.append(None)
            continue
        projected.append(np.array([fx * pc1[0] / pc1[2] + cx, fy * pc1[1] / pc1[2] + cy]))
    return projected


def fit_local_affine(center, offsets, projected):
    valid_offsets = []
    valid_targets = []
    for off, uv1 in zip(offsets, projected):
        if uv1 is None:
            continue
        valid_offsets.append(off)
        valid_targets.append(uv1 - center)
    if len(valid_offsets) < 4:
        return None

    x = np.asarray(valid_offsets, np.float64)
    y = np.asarray(valid_targets, np.float64)
    design = np.column_stack([x[:, 0], x[:, 1], np.ones(len(x))])
    px, *_ = np.linalg.lstsq(design, y[:, 0], rcond=None)
    py, *_ = np.linalg.lstsq(design, y[:, 1], rcond=None)
    a = np.array([[px[0], px[1]], [py[0], py[1]]])
    t = np.array([px[2], py[2]])
    residual = y - design @ np.vstack([px, py]).T
    return a, t, float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))), len(valid_offsets)


def summarize(rows):
    if not rows:
        print("no rows")
        return
    keys = ["a_minus_i_fro", "shear_abs", "scale_abs", "rot_abs", "translation_norm", "affine_residual"]
    arr = {k: np.array([r[k] for r in rows], float) for k in keys}
    print(f"rows: {len(rows)}")
    for k in keys:
        print(
            f"{k:18s} median={np.median(arr[k]):.5f} "
            f"p90={np.percentile(arr[k], 90):.5f} p99={np.percentile(arr[k], 99):.5f}"
        )
    ratio = arr["a_minus_i_fro"] / np.maximum(arr["translation_norm"], 1e-6)
    print(
        "A-I / |t|          "
        f"median={np.median(ratio):.5f} p90={np.percentile(ratio, 90):.5f}"
    )


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--radius", type=float, default=21.0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="local_affine_gt_diag.csv")
    ap.add_argument("--plot", default="local_affine_gt_diag.png")
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

    r = args.radius
    offsets = [
        np.array([0.0, 0.0]),
        np.array([r, 0.0]),
        np.array([-r, 0.0]),
        np.array([0.0, r]),
        np.array([0.0, -r]),
        np.array([r, r]),
        np.array([r, -r]),
        np.array([-r, r]),
        np.array([-r, -r]),
    ]

    tracker = make_tracker(args.config, w, h, fx, fy, cx, cy, dcoef)
    prev = None
    rows = []
    tstart = time.time()

    for i, (t, path) in enumerate(frames):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cur = tracker_step(tracker, img, K, dcoef)

        if prev is not None and i % args.every == 0:
            i0, t0, prev_uv = prev
            common = [fid for fid in cur if fid in prev_uv]
            t_wc0 = cam_pose(t0)
            t_cw1 = np.linalg.inv(cam_pose(t))
            wmag = float(np.interp(t, gt_t, gt_w))
            for fid in common:
                center = prev_uv[fid]
                samples = [center + off for off in offsets]
                projected = project_points(samples, zbufs[i0], t_wc0, t_cw1, fx, fy, cx, cy, w, h)
                fit = fit_local_affine(center, offsets, projected)
                if fit is None:
                    continue
                a, affine_t, residual, n_valid = fit
                a_minus_i = a - np.eye(2)
                rows.append({
                    "frame": i,
                    "id": fid,
                    "radius": r,
                    "a00": a[0, 0],
                    "a01": a[0, 1],
                    "a10": a[1, 0],
                    "a11": a[1, 1],
                    "tx": affine_t[0],
                    "ty": affine_t[1],
                    "translation_norm": float(np.linalg.norm(affine_t)),
                    "a_minus_i_fro": float(np.linalg.norm(a_minus_i, ord="fro")),
                    "shear_abs": float(np.hypot(a[0, 1], a[1, 0])),
                    "scale_abs": float(np.hypot(a[0, 0] - 1.0, a[1, 1] - 1.0)),
                    "rot_abs": float(abs(0.5 * (a[1, 0] - a[0, 1]))),
                    "affine_residual": residual,
                    "valid_points": n_valid,
                    "wmag": wmag,
                })

        prev = (i, t, cur)
        if i % 400 == 0:
            fps = i / max(time.time() - tstart, 1e-9)
            print(f"  frame {i}/{len(frames)} rows={len(rows)} {fps:.0f}fps")

    fields = [
        "frame", "id", "radius", "a00", "a01", "a10", "a11", "tx", "ty",
        "translation_norm", "a_minus_i_fro", "shear_abs", "scale_abs", "rot_abs",
        "affine_residual", "valid_points", "wmag",
    ]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")
    summarize(rows)

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        aerr = np.array([r["a_minus_i_fro"] for r in rows], float)
        trans = np.array([r["translation_norm"] for r in rows], float)
        shear = np.array([r["shear_abs"] for r in rows], float)
        wmag = np.array([r["wmag"] for r in rows], float)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].hist(np.clip(aerr, 0, 0.2), bins=np.linspace(0, 0.2, 80))
        ax[0].set_xlabel("||A_gt - I||_F (clip 0.2)")
        ax[1].scatter(trans, aerr, s=2, alpha=0.12)
        ax[1].set_xlabel("GT translation |t| [px]")
        ax[1].set_ylabel("||A_gt - I||_F")
        ax[2].scatter(wmag, shear, s=2, alpha=0.12)
        ax[2].set_xlabel("|w|")
        ax[2].set_ylabel("GT shear/rotation terms")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
