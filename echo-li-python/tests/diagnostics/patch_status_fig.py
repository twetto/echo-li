"""Patch-depth coverage video: scrub and cherry-pick a frame for the paper figure.

Drives the real pipeline on a EuRoC Vicon-room sequence like `real_depth_eval.py`
(frontend tracker + `Sparse3DFilter` fed **GT poses**), then hands each frame to
`PatchDepthMapper`. Every frame with a mapper output is composited into a 3-panel
video frame:

  1. coverage labels: measured / guessed / unknown / rejected  (operational envelope)
  2. depth:      range = exp(eta)          (fixed colour scale)
  3. confidence: sqrt(var(eta)) ~ sigma_r/r (fixed colour scale)

HUD shows the frame index so you can pick one and extract it to PNG, e.g.
  ffmpeg -i patch_status.mp4 -vf "select=eq(n\\,137)" -vframes 1 frame137.png

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/patch_status_fig.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
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
from real_depth_eval import load_csv  # noqa: E402
import echo_li  # noqa: E402

# PatchStatus enum (echo-li-core patch_depth::PatchStatus, repr(u8)).
UNKNOWN, SEED_ONLY, PHOTO_REFINED, REJECTED = 0, 1, 2, 3
# BGR overlay colours for the coverage panel.
STATUS_BGR = {PHOTO_REFINED: (0, 180, 0),      # measured  -> green
              SEED_ONLY: (10, 190, 250),        # guessed   -> amber
              REJECTED: (0, 0, 220)}            # rejected  -> red


def seeds_from_filter(live, min_track, coord_key, max_depth):
    """Live tracks -> patch-mapper priors (u, v, eta=ln range, var eta).

    `coord_key` selects the seed pixel domain the mapper expects: "uv_raw" for
    raw_distorted / per_patch_bearing, "uv_pinhole" for the undistorted modes.
    """
    seeds = []
    for fd in live.values():
        if fd["track_length"] < min_track:
            continue
        x = np.asarray(fd["position"], float)
        r = float(np.linalg.norm(x))
        if not (0.1 <= r <= max_depth):
            continue
        u_dir = x / r
        var_r = float(u_dir @ np.asarray(fd["covariance_euclidean"], float) @ u_dir)
        if not np.isfinite(var_r) or var_r <= 0:
            continue
        u, v = fd[coord_key]
        seeds.append((float(u), float(v), float(np.log(r)), float(var_r / (r * r))))
    return seeds


def coverage_panel(undist_gray, status, alpha=0.6):
    base = cv2.cvtColor(undist_gray, cv2.COLOR_GRAY2BGR)
    vis = base.copy()
    for s, col in STATUS_BGR.items():
        m = status == s
        if m.any():
            vis[m] = ((1 - alpha) * base[m] + alpha * np.array(col)).astype(np.uint8)
    return vis


def scalar_panel(field, valid, vmin, vmax, flip=False):
    """Colour a scalar field with COLORMAP_JET, matching the CLI rerun views.

    `flip=True` reproduces the CLI's depth mapping (near=red, far=blue); direct
    (flip=False) gives the confidence mapping (confident=blue, uncertain=red).
    """
    norm = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    if flip:
        norm = 1.0 - norm
    u8 = (norm * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    col[~valid] = (0, 0, 0)
    return col


def label(img, text, org, scale=0.5, col=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="patch_status.mp4")
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--depth-min", type=float, default=0.5, help="depth colour scale min (m)")
    ap.add_argument("--depth-max", type=float, default=5.0, help="depth colour scale max (m)")
    ap.add_argument("--relstd-max", type=float, default=0.5, help="confidence colour scale max")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence")
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
    undist_map = cv2.initUndistortRectifyMap(K, dcoef[:4], None, K, (w, h), cv2.CV_32FC1)

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)
    filt = echo_li.Sparse3DFilter.invdepth_additive3d(
        fx, fy, cx, cy, sigma_pixel=0.5, min_track_length=1)
    cam = echo_li.RadTanCamera(fx, fy, cx, cy, *dcoef[:4].tolist())
    mapper = echo_li.PatchDepthMapper(cam, fx, fy, cx, cy, w, h, config=args.config)
    # Seed/background domain follows the mapper's camera_mode (from the YAML).
    coord_key = "uv_raw" if mapper.seed_coordinates == "raw" else "uv_pinhole"
    print(f"patch mapper seed domain: {mapper.seed_coordinates}  "
          f"(config {Path(args.config).name})")

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames:
        frames = frames[:args.max_frames]

    vw = None
    t0 = gt_t[0]
    tstart = time.time()
    written = 0
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        if not feats:
            continue
        px = np.array([[fd["x"], fd["y"]] for fd in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
        uvs = {int(fd["id"]): (float(u), float(v)) for fd, (u, v) in zip(feats, und)}
        raw_uv = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}

        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(t).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        t_wc = t_wb @ t_bs
        filt.update(t, uvs, t_wc.tolist(), None, None)

        fdict = filt.get_features()
        live = {}
        for fid, fd in fdict.items():
            if fid in uvs:
                fd["uv_pinhole"] = uvs[fid]
                fd["uv_raw"] = raw_uv[fid]
                live[fid] = fd
        seeds = seeds_from_filter(live, args.min_track, coord_key, args.depth_max)
        out = mapper.update(t, i, img, t_wc.tolist(), seeds)
        if out is None:
            continue

        status = np.asarray(out["status"])
        eta = np.asarray(out["eta"], float)
        eta_var = np.asarray(out["eta_var"], float)
        # Coverage counts on the native (pre-upscale) patch grid.
        n_ph = int((status == PHOTO_REFINED).sum())
        n_sd = int((status == SEED_ONLY).sum())
        n_rj = int((status == REJECTED).sum())
        n_un = int((status == UNKNOWN).sum())
        # The mapper grid is scale*full-res; upscale (nearest, keeps the patch
        # blocks crisp and honest) onto the full-res photo for a legible figure.
        if status.shape != (h, w):
            status = cv2.resize(status, (w, h), interpolation=cv2.INTER_NEAREST)
            eta = cv2.resize(eta, (w, h), interpolation=cv2.INTER_NEAREST)
            eta_var = cv2.resize(eta_var, (w, h), interpolation=cv2.INTER_NEAREST)
        dh, dw = h, w
        # Background matches the mapper's domain: raw modes render on the raw
        # (histeq) frame, pinhole modes on the undistorted frame.
        if coord_key == "uv_raw":
            bg = cv2.equalizeHist(img)
        else:
            bg = cv2.equalizeHist(cv2.remap(img, *undist_map, cv2.INTER_LINEAR))

        rng = np.exp(eta)
        valid_d = np.isfinite(rng)
        p_cov = coverage_panel(bg, status)
        # Match the CLI rerun colourmaps: JET, depth flipped (near=red, far=blue).
        p_depth = scalar_panel(np.where(valid_d, rng, 0), valid_d, args.depth_min, args.depth_max,
                               flip=True)
        rel_std = np.sqrt(np.where(eta_var > 0, eta_var, np.nan))
        valid_c = np.isfinite(rel_std)
        p_conf = scalar_panel(np.where(valid_c, rel_std, 0), valid_c, 0.0, args.relstd_max)

        label(p_cov, "coverage: measured/guessed/rejected/unknown", (6, dh - 10))
        label(p_depth, f"depth  range=e^eta  [{args.depth_min:.1f}, {args.depth_max:.0f}] m",
              (6, dh - 10))
        label(p_conf, f"confidence  sqrt(var eta)  [0, {args.relstd_max:.1f}]", (6, dh - 10))
        composite = np.hstack([p_cov, p_depth, p_conf])
        label(composite, f"frame {i}  t={t-t0:5.1f}s  seeds={len(seeds)}  "
              f"measured={n_ph} guessed={n_sd} rejected={n_rj} unknown={n_un}",
              (8, 22), scale=0.55, col=(0, 255, 255))

        if vw is None:
            vh, vwid = composite.shape[:2]
            vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (vwid, vh))
        vw.write(composite)
        written += 1
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] written={written} seeds={len(seeds)} "
                  f"cover={n_ph+n_sd} {i/max(time.time()-tstart,1e-9):.0f}fps")

    if vw is not None:
        vw.release()
        print(f"saved {args.out}  ({written} frames)")
    else:
        print("no mapper output produced — check seeds / baseline")


if __name__ == "__main__":
    main()
