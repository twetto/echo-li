"""Sweep KLT window size across the four translation/affine template modes."""

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


def load_frames(root):
    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        return [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]


def score_mode(args, name, policy, klt_warp, reference_warp, window, frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h):
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.klt_template_policy = policy
    fcfg.klt_warp = klt_warp
    fcfg.klt_reference_warp = reference_warp
    fcfg.klt_window = window
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    prev = None
    errs = []
    ages = []
    werrs = []
    tstart = time.time()
    timing = {
        "read": 0.0,
        "process": 0.0,
        "convert": 0.0,
        "score": 0.0,
    }
    processed = 0
    scored_frames = 0

    for i, (t, p) in enumerate(frames):
        t_read = time.perf_counter()
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        timing["read"] += time.perf_counter() - t_read
        if img is None:
            continue

        t_process = time.perf_counter()
        feats, _ = tracker.process(img)
        timing["process"] += time.perf_counter() - t_process
        processed += 1

        t_convert = time.perf_counter()
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = (
            cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
            if len(feats)
            else np.zeros((0, 2))
        )
        cur = {int(f["id"]): uv for f, uv in zip(feats, und)}
        timing["convert"] += time.perf_counter() - t_convert

        if prev is not None and i % args.every == 0 and cur:
            t_score = time.perf_counter()
            scored_frames += 1
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
            timing["score"] += time.perf_counter() - t_score

        prev = (i, t, cur)
        if i % 400 == 0:
            fps = i / max(time.time() - tstart, 1e-9)
            print(f"  window={window:2d} {name:>20} [{i}/{len(frames)}] scored={len(errs)} {fps:.0f}fps")

    e = np.asarray(errs)
    age = np.asarray(ages)
    wm = np.asarray(werrs)
    inl = e < 3.0 if len(e) else np.array([], dtype=bool)
    hi = wm > np.percentile(wm, 75) if len(wm) else np.array([], dtype=bool)
    return {
        "window": window,
        "mode": name,
        "pairs": len(e),
        "median": float(np.median(e)) if len(e) else np.nan,
        "p90": float(np.percentile(e, 90)) if len(e) else np.nan,
        "p99": float(np.percentile(e, 99)) if len(e) else np.nan,
        "out_1px": float(np.mean(e > 1.0)) if len(e) else np.nan,
        "out_3px": float(np.mean(~inl)) if len(e) else np.nan,
        "sigma": float(np.sqrt(np.mean(e[inl] ** 2) / 2.0)) if np.any(inl) else np.nan,
        "age_median": float(np.nanmedian(age)) if len(age) else np.nan,
        "age_p90": float(np.nanpercentile(age, 90)) if len(age) else np.nan,
        "hi_w_median": float(np.median(e[hi])) if np.any(hi) else np.nan,
        "calm_median": float(np.median(e[~hi])) if len(e) and np.any(~hi) else np.nan,
        "seconds_read": timing["read"],
        "seconds_process": timing["process"],
        "seconds_convert": timing["convert"],
        "seconds_score": timing["score"],
        "seconds_total": time.time() - tstart,
        "processed_frames": processed,
        "scored_frames": scored_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--windows", default="7,10,15,21,31")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="klt_window_sweep.csv")
    ap.add_argument("--plot", default="klt_window_sweep.png")
    ap.add_argument("--timing", action="store_true")
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

    windows = [int(v) for v in args.windows.split(",") if v.strip()]
    fields = [
        "window", "mode", "pairs", "median", "p90", "p99",
        "out_1px", "out_3px", "sigma", "age_median", "age_p90",
        "hi_w_median", "calm_median",
        "seconds_read", "seconds_process", "seconds_convert", "seconds_score",
        "seconds_total", "processed_frames", "scored_frames",
    ]
    rows = []
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                row["window"] = int(row["window"])
                for key in fields:
                    if key not in ("mode", "window"):
                        row[key] = float(row[key])
                rows.append(row)
                done.add((row["window"], row["mode"]))
    if not out_path.exists():
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for window in windows:
        for mode in MODES:
            if (window, mode[0]) in done:
                print(f"window={window:2d} {mode[0]:>20}: already done")
                continue
            row = score_mode(
                args, *mode, window,
                frames, zbufs, cam_pose, gt_t, gt_w, K, dcoef, fx, fy, cx, cy, w, h
            )
            rows.append(row)
            with open(out_path, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writerow(row)
            print(
                f"window={window:2d} {mode[0]:>20}: "
                f"median={row['median']:.3f}px p90={row['p90']:.3f}px "
                f">3px={100*row['out_3px']:.1f}% pairs={row['pairs']}"
            )
            if args.timing:
                print(
                    f"  timing: total={row['seconds_total']:.2f}s "
                    f"read={row['seconds_read']:.2f}s "
                    f"process={row['seconds_process']:.2f}s "
                    f"convert={row['seconds_convert']:.2f}s "
                    f"score={row['seconds_score']:.2f}s "
                    f"frames={int(row['processed_frames'])}"
                )

    print(f"saved {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mode_name, *_ in MODES:
        mr = [r for r in rows if r["mode"] == mode_name]
        xs = [r["window"] for r in mr]
        ax[0].plot(xs, [r["median"] for r in mr], marker="o", label=mode_name)
        ax[1].plot(xs, [100 * r["out_3px"] for r in mr], marker="o", label=mode_name)
    ax[0].set_xlabel("KLT window radius")
    ax[0].set_ylabel("median GT-flow error [px]")
    ax[1].set_xlabel("KLT window radius")
    ax[1].set_ylabel(">3px outliers [%]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
