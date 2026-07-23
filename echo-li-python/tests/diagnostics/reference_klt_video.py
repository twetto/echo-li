"""Render side-by-side videos for KLT template/warp policies.

Panels:
  previous-frame translation KLT
  previous-frame affine KLT
  first-observation translation KLT
  first-observation affine KLT

Colors are based on a frame-to-frame fundamental-matrix RANSAC proxy:
  green  geometric inlier
  red    geometric outlier
  yellow new track
"""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml

import echo_li


def load_csv(path):
    with open(path) as f:
        return np.array([r for r in csv.reader(f) if r and not r[0].startswith("#")], dtype=float)


def quat_ang_rate(t, quat):
    def qmul(a, b):
        aw, ax, ay, az = a.T
        bw, bx, by, bz = b.T
        return np.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            axis=1,
        )

    qc = quat.copy()
    qc[:, 1:] *= -1
    dq = qmul(qc[:-1], quat[1:])
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    w = 2 * np.arccos(np.clip(np.abs(dq[:, 0]), -1, 1)) / np.diff(t)
    return np.concatenate([w, w[-1:]])


def make_tracker(config, w, h, fx, fy, cx, cy, dcoef, policy, klt_warp, reference_warp):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.klt_template_policy = policy
    fcfg.klt_warp = klt_warp
    fcfg.klt_reference_warp = reference_warp
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef if dcoef else [])
    return echo_li.Frontend(fcfg, w, h)


def tracker_step(tracker, img):
    feats, _ = tracker.process(img)
    meta = {int(m["id"]): int(m["age"]) for m in tracker.track_meta()}
    pos = {int(f["id"]): (float(f["x"]), float(f["y"])) for f in feats}
    age = {fid: meta.get(fid, 1) for fid in pos}
    return pos, age


def classify(prev_pos, pos):
    common = [fid for fid in pos if fid in prev_pos]
    status = {fid: "new" for fid in pos}
    outfrac = 0.0
    if len(common) >= 8:
        p0 = np.array([prev_pos[fid] for fid in common], np.float32)
        p1 = np.array([pos[fid] for fid in common], np.float32)
        _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.0, 0.99)
        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(common), bool)
        for fid, inlier in zip(common, inliers):
            status[fid] = "in" if inlier else "out"
        outfrac = 1.0 - float(np.mean(inliers))
    return status, outfrac


def draw_panel(gray, title, pos, age, prev_pos, trails, t_rel, wmag):
    status, outfrac = classify(prev_pos, pos)
    for fid in list(trails):
        if fid not in pos:
            del trails[fid]
    for fid, xy in pos.items():
        trails[fid].append(xy)

    colors = {
        "in": (0, 210, 0),
        "out": (0, 0, 255),
        "new": (0, 220, 220),
    }
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for fid, xy in pos.items():
        st = status.get(fid, "new")
        trail = trails[fid]
        if len(trail) >= 2:
            cv2.polylines(vis, [np.array(trail, np.int32)], False, colors[st], 1, cv2.LINE_AA)
        x, y = xy
        radius = 3 if st == "out" else 2
        cv2.circle(vis, (int(x), int(y)), radius, colors[st], -1, cv2.LINE_AA)

    ages = np.array(list(age.values()), dtype=float)
    med_age = float(np.median(ages)) if len(ages) else 0.0
    out_count = sum(1 for st in status.values() if st == "out")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"t={t_rel:5.1f}s |w|={wmag:.2f} tracks={len(pos)} med_age={med_age:.0f}",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"F-outliers={out_count} ({100*outfrac:.0f}%)",
        (8, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 180, 255),
        1,
        cv2.LINE_AA,
    )
    return vis, outfrac


def open_writer(path, w, h, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if writer.isOpened():
        return writer, path
    avi = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer, avi


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="reference_klt_video.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = cfg.get("distortion_coefficients", [])

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])
    t0 = gt_t[0]

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    panels = [
        ("previous + translation", "previous", "translation", "translation"),
        ("previous + affine", "previous", "affine", "translation"),
        ("first + translation", "first_observation", "translation", "translation"),
        ("first + affine", "first_observation", "translation", "affine"),
    ]
    trackers = [
        make_tracker(args.config, w, h, fx, fy, cx, cy, dcoef, policy, klt_warp, reference_warp)
        for _, policy, klt_warp, reference_warp in panels
    ]
    prev = [dict() for _ in panels]
    trails = [defaultdict(lambda: deque(maxlen=10)) for _ in panels]
    out_hist = [[] for _ in panels]

    out_path = Path(args.out)
    writer, final_path = open_writer(out_path, w * len(panels), h, args.fps)
    for idx, (t, path) in enumerate(frames):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        t_rel = t - t0
        wmag = float(np.interp(t, gt_t, gt_w))
        rendered = []
        for panel_idx, (title, _, _, _) in enumerate(panels):
            pos, age = tracker_step(trackers[panel_idx], img)
            panel, outfrac = draw_panel(
                img, title, pos, age, prev[panel_idx], trails[panel_idx], t_rel, wmag
            )
            rendered.append(panel)
            prev[panel_idx] = pos
            out_hist[panel_idx].append(outfrac)
        writer.write(np.hstack(rendered))
        if idx % 400 == 0:
            msg = " ".join(
                f"{panels[i][0]}={100*np.mean(out_hist[i]):.0f}%"
                for i in range(len(panels))
                if out_hist[i]
            )
            print(f"  frame {idx}/{len(frames)} {msg}")
    writer.release()

    print(f"saved {final_path}")
    for i, (title, _, _, _) in enumerate(panels):
        print(f"  {title:20s} mean F-outlier fraction {100*np.mean(out_hist[i]):.1f}%")


if __name__ == "__main__":
    main()
