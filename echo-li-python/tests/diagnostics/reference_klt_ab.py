"""A/B test Rudolf-V previous-frame KLT vs first-observation reference KLT.

Uses the same EuRoC Leica/Vicon optical-flow ground truth as flow_gt_eval.py.
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


def score_policy(args, name, policy, klt_warp, reference_warp, frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h):
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.klt_template_policy = policy
    fcfg.klt_warp = klt_warp
    fcfg.klt_reference_warp = reference_warp
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    prev = None
    errs = []
    ages = []
    werrs = []
    tstart = time.time()

    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        feats, _ = tracker.process(img)
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = (
            cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
            if len(feats)
            else np.zeros((0, 2))
        )
        cur = {int(f["id"]): uv for f, uv in zip(feats, und)}

        if prev is not None and i % args.every == 0 and cur:
            i0, t0, uv0 = prev
            common = [fid for fid in cur if fid in uv0]
            if common:
                t_wc0 = cam_pose(t0)
                p0 = [uv0[fid] for fid in common]
                d0 = zbuf_lookup(zbufs[i0], p0, w, h)
                t_cw1 = np.linalg.inv(cam_pose(t))
                wmag = float(np.interp(t, gt_t, gt_w))
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
                    u1, v1 = cur[fid]
                    errs.append(np.hypot(u1 - ugt, v1 - vgt))
                    ages.append(meta.get(fid, {}).get("age", np.nan))
                    werrs.append(wmag)

        prev = (i, t, cur)
        if i % 400 == 0:
            print(f"  {name:>18} [{i}/{len(frames)}] scored={len(errs)} "
                  f"{i / max(time.time() - tstart, 1e-9):.0f}fps")

    return np.asarray(errs), np.asarray(ages), np.asarray(werrs)


def summarize(name, e, ages, wm):
    inl = e < 3.0
    hi = wm > np.percentile(wm, 75) if len(wm) else np.array([], dtype=bool)
    sig = np.sqrt(np.mean(e[inl] ** 2) / 2.0) if np.any(inl) else np.nan
    print(f"\n=== {name} ({len(e)} feature-pairs) ===")
    print(f"flow err: median {np.median(e):.3f} px  p90 {np.percentile(e, 90):.3f}  "
          f"p99 {np.percentile(e, 99):.2f}")
    print(f"outliers (>3 px): {100*np.mean(~inl):.1f}%   (>1 px: {100*np.mean(e > 1):.1f}%)")
    print(f"effective sigma_pixel (inlier RMS/sqrt2): {sig:.3f} px")
    print(f"age: median {np.nanmedian(ages):.1f}  p90 {np.nanpercentile(ages, 90):.1f}")
    if len(wm):
        print(f"by rotation: hi-|w| median {np.median(e[hi]):.3f} px, outliers "
              f"{100*np.mean(e[hi] >= 3):.1f}%   calm {np.median(e[~hi]):.3f} px, "
              f"{100*np.mean(e[~hi] >= 3):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="reference_klt_ab.png")
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

    prev_e, prev_age, prev_w = score_policy(
        args, "previous+translation", "previous", "translation", "translation",
        frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
    )
    prev_aff_e, prev_aff_age, prev_aff_w = score_policy(
        args, "previous+affine", "previous", "affine", "translation",
        frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
    )
    ref_e, ref_age, ref_w = score_policy(
        args, "first+translation", "first_observation", "translation", "translation",
        frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
    )
    aff_e, aff_age, aff_w = score_policy(
        args, "first+affine", "first_observation", "translation", "affine",
        frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
    )

    summarize("previous-frame translation", prev_e, prev_age, prev_w)
    summarize("previous-frame affine", prev_aff_e, prev_aff_age, prev_aff_w)
    summarize("first-observation translation", ref_e, ref_age, ref_w)
    summarize("first-observation affine", aff_e, aff_age, aff_w)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 5, 100)
    ax[0].hist(np.clip(prev_e, 0, 5), bins=bins, alpha=0.50, label="previous translation")
    ax[0].hist(np.clip(prev_aff_e, 0, 5), bins=bins, alpha=0.42, label="previous affine")
    ax[0].hist(np.clip(ref_e, 0, 5), bins=bins, alpha=0.45, label="first translation")
    ax[0].hist(np.clip(aff_e, 0, 5), bins=bins, alpha=0.45, label="first affine")
    ax[0].set_xlabel("flow error [px] (clip 5)")
    ax[0].legend()
    ax[1].scatter(prev_age, np.clip(prev_e, 0, 10), s=3, alpha=0.12, label="previous translation")
    ax[1].scatter(prev_aff_age, np.clip(prev_aff_e, 0, 10), s=3, alpha=0.12, label="previous affine")
    ax[1].scatter(ref_age, np.clip(ref_e, 0, 10), s=3, alpha=0.12, label="first translation")
    ax[1].scatter(aff_age, np.clip(aff_e, 0, 10), s=3, alpha=0.12, label="first affine")
    ax[1].set_xlabel("track age [frames]")
    ax[1].set_ylabel("flow error [px] (clip 10)")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
