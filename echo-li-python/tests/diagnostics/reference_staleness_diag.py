"""Diagnose first-observation reference-template staleness by track age."""

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


MODES = [
    ("previous_translation", "previous", "translation", "translation"),
    ("previous_affine", "previous", "affine", "translation"),
    ("first_translation", "first_observation", "translation", "translation"),
    ("first_affine", "first_observation", "translation", "affine"),
]

AGE_BINS = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 10**9)]


def load_frames(root):
    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        return [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]


def score_mode(args, name, policy, klt_warp, reference_warp, frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h):
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.klt_template_policy = policy
    fcfg.klt_warp = klt_warp
    fcfg.klt_reference_warp = reference_warp
    fcfg.klt_residual = True
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    prev = None
    rows = []
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
                    m = meta.get(fid, {})
                    rows.append({
                        "mode": name,
                        "frame": i,
                        "id": fid,
                        "age": float(m.get("age", np.nan)),
                        "klt_quality": float(m.get("klt_quality", np.nan)),
                        "err": float(np.hypot(u1 - ugt, v1 - vgt)),
                        "wmag": wmag,
                    })

        prev = (i, t, cur)
        if i % 400 == 0:
            fps = i / max(time.time() - tstart, 1e-9)
            print(f"  {name:>20} [{i}/{len(frames)}] rows={len(rows)} {fps:.0f}fps")

    return rows


def age_bin_label(age):
    for lo, hi in AGE_BINS:
        if lo <= age < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "nan"


def summarize(rows):
    out = []
    for mode in sorted(set(r["mode"] for r in rows)):
        mr = [r for r in rows if r["mode"] == mode and np.isfinite(r["err"]) and np.isfinite(r["age"])]
        for lo, hi in AGE_BINS:
            br = [r for r in mr if lo <= r["age"] < hi]
            if not br:
                continue
            err = np.array([r["err"] for r in br], float)
            q = np.array([r["klt_quality"] for r in br], float)
            out.append({
                "mode": mode,
                "age_bin": f"{lo}-{hi if hi < 10**9 else 'inf'}",
                "count": len(br),
                "err_median": float(np.median(err)),
                "err_p90": float(np.percentile(err, 90)),
                "out_3px": float(np.mean(err > 3.0)),
                "quality_median": float(np.nanmedian(q)),
                "quality_p10": float(np.nanpercentile(q, 10)),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="reference_staleness_diag.csv")
    ap.add_argument("--summary", default="reference_staleness_summary.csv")
    ap.add_argument("--plot", default="reference_staleness_diag.png")
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

    rows = []
    for mode in MODES:
        rows.extend(score_mode(
            args, *mode, frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
        ))

    fields = ["mode", "frame", "id", "age", "klt_quality", "err", "wmag"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.out}")

    summary = summarize(rows)
    sfields = ["mode", "age_bin", "count", "err_median", "err_p90", "out_3px", "quality_median", "quality_p10"]
    with open(args.summary, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sfields)
        wr.writeheader()
        wr.writerows(summary)
    print(f"saved {args.summary}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{lo}-{hi if hi < 10**9 else 'inf'}" for lo, hi in AGE_BINS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mode, *_ in MODES:
        sm = [r for r in summary if r["mode"] == mode]
        by = {r["age_bin"]: r for r in sm}
        xs = [i for i, label in enumerate(labels) if label in by]
        ax[0].plot(xs, [by[labels[i]]["err_median"] for i in xs], marker="o", label=mode)
        ax[1].plot(xs, [by[labels[i]]["quality_median"] for i in xs], marker="o", label=mode)
    for a in ax:
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, rotation=30, ha="right")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("median GT-flow error [px]")
    ax[1].set_ylabel("median KLT quality")
    ax[0].set_xlabel("track age bin [frames]")
    ax[1].set_xlabel("track age bin [frames]")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
