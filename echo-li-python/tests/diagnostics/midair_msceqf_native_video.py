#!/usr/bin/env python3
"""Diagnostic video for the MSCEqF-native run WITH in-state SLAM landmarks.

Re-runs the exact validated recipe from midair_msceqf_native_ate.py (structureless
MSCKF update + promoted in-state SLAM landmarks, VO_test traj2 → 0.24% ATE) and,
per camera frame, records the tracked features, the estimated camera pose, and which
tracks are in-state SLAM landmarks. Then renders:

  LEFT  = the scene image with the tracked features overlaid:
            · NORMAL (structureless) features → hollow circles
            · SLAM   (in-state)      features → solid dots
          both colored by inverse-JET (jet_r) on a NORMALIZED metric range.
  RIGHT = top-down trajectory (GT vs SE3-aligned estimate, estimate colored by
          per-frame position error), with the current error line.

Metric range is computed by ONE consistent pipeline for every feature: multi-view
midpoint triangulation of the track's bearings against the estimated camera
trajectory (the filter's own poses). Using a single range definition keeps the color
scale comparable across both marker classes — the SLAM/normal distinction is carried
by marker shape alone, exactly as requested. Freshly-born / low-parallax tracks that
cannot be triangulated are drawn gray.

  PY=echo-li-python/.venv/bin/python
  $PY tests/diagnostics/midair_msceqf_native_video.py \
      --max-slam 100 --out msceqf_native_slam_traj2.mp4 --stride 2
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402
from echo_li import MSCEqFNativeFilter  # noqa: E402

DEF_ROOT = "/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir"
TRACKS = Path(__file__).resolve().parents[3] / \
    "artifacts/msceqf_midair/stagediff/msceqf_own_tracks.txt"

DT = 0.01          # 100 Hz IMU
FX = CX = 512.0    # MidAir 90deg @1024, no distortion (track-file normalization)


def R_wb(q):
    """Rotation body->world from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_tracks(nimg, tracks_path, t0, imu_per_img):
    """frame k -> {track_id: (un, vn)} normalized Z1 coords."""
    by_frame = defaultdict(dict)
    with open(tracks_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            ts = float(p[0]); tid = int(p[2]); u = float(p[3]); v = float(p[4])
            k = int(round((ts - t0) / (DT * imu_per_img)))
            if k < 0 or k >= nimg:
                continue
            by_frame[k][tid] = ((u - CX) / FX, (v - CX) / FX)
    return by_frame


def build_rudolf_tracks(root, subset, cond, traj, nimg, cfg_path, track_scale,
                        lbp_reject, max_features=0):
    """Run echo-li's own Rudolf-V front-end live over frames 0..nimg-1.

    Returns frame k -> {track_id: (un, vn)} normalized Z1 coords, using the SAME
    normalization convention as the MSCEqF track dump ((u-CX)/FX at 1024px, 90deg).
    Because normalized bearings are resolution-independent, tracking may run at any
    `track_scale`; f/cx/cy are taken from that resolution.
    """
    ds = md.MidAir(root, subset, cond, traj, track_scale)
    im0 = ds.image(0); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)      # f=W/2, cx=W/2, cy=H/2
    fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    if not lbp_reject:
        fcfg.lbp_verification = False    # user: no LBP rejection
    if max_features > 0:
        fcfg.max_features = max_features
    tracker = echo_li.Frontend(fcfg, W, H)
    by_frame = defaultdict(dict)
    for k in range(nimg):
        gray = np.asarray(ds.image(k))
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        feats, _stats = tracker.process(gray)
        for fd in feats:
            tid = int(fd["id"]); x = float(fd["x"]); y = float(fd["y"])
            by_frame[k][tid] = ((x - cx) / f, (y - cy) / f)
        if k % 500 == 0:
            print(f"  rudolf-V tracked frame {k}/{nimg}: {len(feats)} feats")
    return by_frame


def align_umeyama(est, gt):
    """SE3 (no-scale) alignment of est to gt; returns (aligned, R, t)."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    H = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_g - R @ mu_e
    return (R @ est.T).T + t, R, t


def triangulate(origins, dirs, min_parallax=1e-3):
    """Multi-view midpoint: X minimizing sum ||(I - dd^T)(X - o)||^2.

    Returns (X, ok). ok=False if <2 rays or the ray bundle is near-degenerate
    (no parallax) or the normal system is ill-conditioned.
    """
    if len(origins) < 2:
        return None, False
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    # Parallax proxy: smallest eigenvalue of A grows with angular spread.
    w = np.linalg.eigvalsh(A)
    if w[0] < min_parallax:
        return None, False
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, False
    return X, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEF_ROOT)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5, help="display image scale")
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--imu-per-img", type=int, default=4)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-track-len", type=int, default=200)
    ap.add_argument("--curvature", type=int, default=1)
    ap.add_argument("--chi2-mult", type=float, default=1.0)
    ap.add_argument("--max-slam", type=int, default=100)
    ap.add_argument("--slam-min-len", type=int, default=0)
    ap.add_argument("--slam-chi2-mult", type=float, default=0.0)
    ap.add_argument("--acc-density", type=float, default=1.6798e-3)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-density", type=float, default=1.2724e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.0e-4)
    ap.add_argument("--tracks", default=str(TRACKS))
    ap.add_argument("--tracks-t0", type=float, default=0.0)
    ap.add_argument("--frontend", choices=["msceqf", "rudolf"], default="msceqf",
                    help="msceqf = replay MSCEqF's own KLT dump (front-end isolated); "
                         "rudolf = run echo-li's Rudolf-V tracker live")
    ap.add_argument("--frontend-config", default="configs/eqvio_midair.yaml")
    ap.add_argument("--track-scale", type=float, default=1.0,
                    help="image scale for the live Rudolf-V tracker (default native)")
    ap.add_argument("--lbp-reject", action="store_true",
                    help="keep Rudolf-V LBP rejection on (default: OFF)")
    ap.add_argument("--max-features", type=int, default=0,
                    help="override Rudolf-V max_features (0 => use config; OV parity=200)")
    ap.add_argument("--renderer", choices=["cv2", "mpl"], default="cv2",
                    help="cv2 = fast incremental compositing; mpl = matplotlib (slow)")
    # Rendering / colour scale
    ap.add_argument("--out", default="msceqf_native_slam_traj2.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--tri-window", type=int, default=40,
                    help="max recent observations per track used for triangulation")
    ap.add_argument("--diag", action="store_true",
                    help="print per-frame feature/triangulation stats and exit "
                         "(no video): explains why some frames colour few features")
    args = ap.parse_args()
    IPI = args.imu_per_img
    tmr = {}
    t0c = time.perf_counter()

    base = f"{args.root}/{args.subset}/{args.cond}"
    tr = f"trajectory_{args.traj:04d}"
    g = h5py.File(f"{base}/sensor_records.hdf5", "r")[tr]
    acc = np.asarray(g["imu"]["accelerometer"], float)
    gyr = np.asarray(g["imu"]["gyroscope"], float)
    att = np.asarray(g["groundtruth"]["attitude"], float)   # wxyz body->world
    pos = np.asarray(g["groundtruth"]["position"], float)
    vel = np.asarray(g["groundtruth"]["velocity"], float)
    m = min(len(gyr), len(att))
    Rwb = [R_wb(att[i]) for i in range(m)]
    gyr_body = np.array([Rwb[i].T @ gyr[i] for i in range(m)])
    acc = acc[:m]
    nimg_all = m // IPI
    nimg = nimg_all if args.nframes <= 0 else min(args.nframes, nimg_all)
    if args.frontend == "rudolf":
        cfg = args.frontend_config
        if not Path(cfg).is_absolute():
            cfg = str(Path(__file__).resolve().parents[3] / cfg)
        print(f"front-end: Rudolf-V live @ scale {args.track_scale} "
              f"(LBP rejection {'ON' if args.lbp_reject else 'OFF'})")
        by_frame = build_rudolf_tracks(args.root, args.subset, args.cond, args.traj,
                                       nimg, cfg, args.track_scale, args.lbp_reject,
                                       args.max_features)
    else:
        print("front-end: MSCEqF own KLT dump (replayed)")
        by_frame = load_tracks(nimg, args.tracks, args.tracks_t0, IPI)
    tmr["front-end"] = time.perf_counter() - t0c

    # --- GT-seeded given-origin init (frame 0), same as the ATE harness ---
    R0 = Rwb[0]; p0 = pos[0].copy(); v0 = vel[0].copy(); b0 = np.zeros(6)
    S0 = np.array([[0., 0., 1., 0.],
                   [1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.]])          # T_imu_cam (cam->body)
    R_bc = S0[:3, :3]; t_bc = S0[:3, 3]
    grav = np.array([0., 0., 9.81])
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
    delta_std = np.full(6, 1e-2); e_std = np.full(6, 1e-4)

    filt = MSCEqFNativeFilter(
        R0, p0, v0, b0, S0, grav, d_std, delta_std, e_std,
        args.acc_density, args.gyr_density, args.acc_rw, args.gyr_rw,
        0, args.num_clones)

    pixel_std = 1.0 / FX
    slam_min_len = args.slam_min_len if args.slam_min_len > 0 else args.num_clones
    slam_chi2 = args.slam_chi2_mult if args.slam_chi2_mult > 0 else args.chi2_mult
    buf = {}
    slam = set()
    est_p, gt_p = [], []

    # Per-frame recordings for rendering.
    cam_R = {}      # k -> R_wc (world<-camera)
    cam_p = {}      # k -> camera centre in world
    slam_at = {}    # k -> frozenset of tids in-state at end of frame k

    def record_pose(k):
        p_wb, R, _v = filt.nav_body_pose()
        cam_p[k] = np.asarray(p_wb, float) + np.asarray(R, float) @ t_bc
        cam_R[k] = np.asarray(R, float) @ R_bc

    def add_frame(k):
        filt.clone_pose(k, k * DT * IPI)
        for tid, (un, vn) in by_frame.get(k, {}).items():
            if tid in slam:
                continue
            buf.setdefault(tid, []).append((k, un, vn))

    add_frame(0)
    record_pose(0)
    slam_at[0] = frozenset()
    est_p.append(filt.nav_body_pose()[0].copy()); gt_p.append(pos[0].copy())

    for k in range(1, nimg):
        for i in range((k - 1) * IPI, k * IPI):
            filt.process_imu(gyr_body[i], acc[i], DT)
        add_frame(k)

        cur = set(by_frame.get(k, {}).keys())
        marg = filt.clone_ids()[0] if filt.n_clones() > args.num_clones else None

        # (0) Promote long window-spanning tracks about to marginalize to in-state.
        if args.max_slam > 0 and marg is not None and len(slam) < args.max_slam:
            cand = [tid for tid, obs in buf.items()
                    if tid in cur and len(obs) >= slam_min_len
                    and any(fk == marg for (fk, _, _) in obs)]
            cand.sort(key=lambda t: len(buf[t]), reverse=True)
            for tid in cand[: args.max_slam - len(slam)]:
                obs = [(fk, un, vn) for (fk, un, vn) in buf[tid]]
                if filt.birth_landmark(tid, obs, pixel_std, slam_chi2):
                    filt.reanchor_landmark(tid, k)
                    slam.add(tid)
                    del buf[tid]

        # (1) Structureless MSCKF update on terminating / maxed / marg-touching tracks.
        ready, done = {}, []
        for tid, obs in buf.items():
            lost = tid not in cur
            toolong = len(obs) >= args.max_track_len
            touches_marg = marg is not None and any(fk == marg for (fk, _, _) in obs)
            if lost or toolong or touches_marg:
                if len(obs) >= 2:
                    ready[tid] = [(fk, un, vn) for (fk, un, vn) in obs]
                done.append(tid)
        if ready:
            filt.msc_update(ready, pixel_std, args.chi2_mult, bool(args.curvature))
        for tid in done:
            del buf[tid]

        # (2) Streaming SLAM update: fresh bearing per in-state landmark; evict lost.
        if slam:
            supd = {}
            for tid in list(slam):
                if tid in cur:
                    un, vn = by_frame[k][tid]
                    supd[tid] = [(k, un, vn)]
                else:
                    filt.marginalize_landmark(tid)
                    slam.discard(tid)
            if supd:
                filt.landmark_stream_update(supd, pixel_std, slam_chi2)

        # (3) Reanchor surviving landmarks off the clone about to marginalize.
        if slam and marg is not None:
            newest = filt.clone_ids()[-1]
            for tid in list(slam):
                if filt.landmark_anchor(tid) == marg:
                    filt.reanchor_landmark(tid, newest)

        # (4) Slide the window.
        while filt.n_clones() > args.num_clones:
            old = filt.marginalize_oldest()
            for tid in list(buf):
                b = [(fk, un, vn) for (fk, un, vn) in buf[tid] if fk != old]
                if b:
                    buf[tid] = b
                else:
                    del buf[tid]

        record_pose(k)
        slam_at[k] = frozenset(slam)
        est_p.append(filt.nav_body_pose()[0].copy()); gt_p.append(pos[k * IPI].copy())

    est_p = np.array(est_p); gt_p = np.array(gt_p)
    aligned, _R, _t = align_umeyama(est_p, gt_p)
    err = np.linalg.norm(aligned - gt_p, axis=1)
    path = np.linalg.norm(np.diff(gt_p, axis=0), axis=1).sum()
    tmr["filter"] = time.perf_counter() - t0c - sum(tmr.values())
    print(f"run done: frames={nimg}  ATE={100*np.sqrt((err**2).mean())/path:.3f}%  "
          f"final SLAM landmarks recorded")

    # --- Triangulate every observed feature's metric range per frame -------------
    # Build a per-track rolling observation list keyed by frame (normalized bearings),
    # then triangulate against the recorded camera trajectory. obs_hist accumulates on
    # EVERY frame (history must be complete), but the expensive triangulation only runs
    # on frames we will actually render (diag mode wants all of them).
    want = set(range(nimg)) if args.diag else set(range(0, nimg, args.stride))
    obs_hist = defaultdict(list)   # tid -> [(k, un, vn)]  (append-only, capped later)
    range_at = {}                  # k -> {tid: range_m or nan}
    for k in range(nimg):
        for tid, (un, vn) in by_frame.get(k, {}).items():
            obs_hist[tid].append((k, un, vn))
        if k not in want:
            continue
        rr = {}
        for tid in by_frame.get(k, {}):
            obs = [o for o in obs_hist[tid] if o[0] <= k][-args.tri_window:]
            origins, dirs = [], []
            for (kk, un, vn) in obs:
                if kk not in cam_R:
                    continue
                d = cam_R[kk] @ np.array([un, vn, 1.0])
                d /= (np.linalg.norm(d) + 1e-12)
                origins.append(cam_p[kk]); dirs.append(d)
            X, ok = triangulate(origins, dirs)
            if not ok:
                rr[tid] = np.nan
                continue
            # Reject points behind the current camera.
            depth = float((cam_R[k].T @ (X - cam_p[k]))[2])
            r = float(np.linalg.norm(X - cam_p[k]))
            rr[tid] = r if (depth > 0.1 and r < 1000.0) else np.nan
        range_at[k] = rr
    tmr["triangulate"] = time.perf_counter() - t0c - sum(tmr.values())

    if args.diag:
        # Per-frame: visible tracks, how many have >=2 window views (triangulable in
        # principle), how many actually got a finite range (parallax passed), and the
        # inter-frame GT translation (parallax proxy). Answers: are low-colour frames
        # tracker churn (few multi-view tracks) or low-parallax (multi-view but no
        # baseline)? And confirms filter health is decoupled from colour.
        print(f"{'k':>5} {'t(s)':>5} {'vis':>4} {'mv≥2':>5} {'col':>4} "
              f"{'col%':>5} {'dGT(m)':>7} {'errATE(m)':>8}")
        rows_d = []
        for k in range(nimg):
            vis = list(by_frame.get(k, {}).keys())
            mv = sum(1 for t in vis
                     if len([o for o in obs_hist[t] if o[0] <= k][-args.tri_window:]) >= 2)
            col = sum(1 for t in vis if np.isfinite(range_at.get(k, {}).get(t, np.nan)))
            dgt = float(np.linalg.norm(gt_p[k] - gt_p[k - 1])) if k > 0 else 0.0
            rows_d.append((k, len(vis), mv, col, dgt, err[k]))
        rows_d = np.array(rows_d, float)
        # Show the 15 lowest-colour frames (col% among frames with >=5 visible).
        vis_ok = rows_d[rows_d[:, 1] >= 5]
        colpct = vis_ok[:, 3] / np.maximum(vis_ok[:, 1], 1)
        order = np.argsort(colpct)[:15]
        for idx in sorted(order, key=lambda i: vis_ok[i, 0]):
            k, nv, mv, col, dgt, e = vis_ok[idx]
            print(f"{int(k):5d} {k/25:5.1f} {int(nv):4d} {int(mv):5d} {int(col):4d} "
                  f"{100*col/max(nv,1):4.0f}% {dgt:7.3f} {e:8.3f}")
        # Correlate colour-fraction with GT translation (parallax) across all frames.
        cf = rows_d[:, 3] / np.maximum(rows_d[:, 1], 1)
        good = rows_d[:, 1] >= 5
        if good.sum() > 10:
            cc = np.corrcoef(rows_d[good, 4], cf[good])[0, 1]
            print(f"\ncorr(colour-fraction, inter-frame GT translation) = {cc:+.3f}  "
                  f"(low-parallax frames colour fewer)")
            print(f"median colour-fraction = {np.median(cf[good]):.2f}   "
                  f"frames with <10% coloured: {int((cf[good] < 0.1).sum())} / "
                  f"{int(good.sum())}")
            print(f"ATE over run stayed {100*np.sqrt((err**2).mean())/path:.3f}% "
                  f"regardless — colour is a triangulation-viz signal, not filter health")
        return

    all_r = np.array([r for rr in range_at.values() for r in rr.values()
                      if np.isfinite(r)])
    if len(all_r) == 0:
        print("no triangulable features — aborting"); return
    cmap = matplotlib.colormaps["jet_r"]
    # Colour is normalized PER FRAME to the depth spread of the features in view
    # (5-95 pct), so the near/far structure of the current scene uses the full
    # colormap instead of being crushed by the far-tail of the whole run. A global
    # 5-95 pct is kept only as a fallback for frames with <4 triangulated features.
    g_lo, g_hi = (float(x) for x in np.percentile(all_r, [5, 95]))
    print(f"metric-range colour: PER-FRAME 5-95 pct of in-view features "
          f"(jet_r; global fallback [{g_lo:.1f}, {g_hi:.1f}] m)")

    # --- Render ------------------------------------------------------------------
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0); H, W = im0.shape
    fI, cxI, cyI = md.intrinsics(W, H)

    def to_px(un, vn):
        return fI * un + cxI, fI * vn + cyI

    # Top-down map convention: East → horizontal (right), North → vertical (up),
    # so a physical left turn reads as a left turn (NED position is [N, E, D]).
    E = np.concatenate([gt_p[:, 1], aligned[:, 1]])
    N = np.concatenate([gt_p[:, 0], aligned[:, 0]])
    epad = 0.06 * (E.max() - E.min() + 1e-6)
    npad = 0.06 * (N.max() - N.min() + 1e-6)
    e_lo, e_hi = E.min() - epad, E.max() + epad
    n_lo, n_hi = N.min() - npad, N.max() + npad
    emax = float(np.percentile(err, 98))

    idxs = list(range(0, nimg, args.stride))

    def frame_clim(k):
        """Per-frame colour scale from the depths actually in view (5-95 pct)."""
        rr = range_at.get(k, {})
        finite = np.array([rr[t] for t in by_frame.get(k, {})
                           if np.isfinite(rr.get(t, np.nan))])
        if len(finite) >= 4:
            vlo, vhi = (float(x) for x in np.percentile(finite, [5, 95]))
            if vhi - vlo < 1.0:
                vlo, vhi = float(finite.min()), float(finite.min()) + 1.0
        else:
            vlo, vhi = g_lo, g_hi
        return float(vlo), float(vhi), rr

    writer = None
    if args.renderer == "cv2":
        # Fast cv2 compositing. Left panel: image + circle overlays + colorbar strip.
        # Right panel: a PERSISTENT trajectory canvas onto which the revealed GT line and
        # the error-coloured estimate points are drawn INCREMENTALLY (O(stride)/frame),
        # so render cost no longer grows with trajectory length and no matplotlib
        # rasterization happens in the hot loop.
        jetr = (matplotlib.colormaps["jet_r"](np.linspace(0, 1, 256))[:, :3]
                * 255).astype(np.uint8)[:, ::-1]      # RGB->BGR LUT
        infl = (matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]
                * 255).astype(np.uint8)[:, ::-1]
        CBW, RW, mrg = 80, W, 36

        def txt(im, s, org, sc=0.42, col=(255, 255, 255)):
            cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

        def clut(tab, t):
            return tuple(int(c) for c in tab[int(np.clip(t, 0.0, 1.0) * 255)])

        # East->x(right), North->y(up), equal aspect (same metres/pixel on both axes).
        Espan = max(e_hi - e_lo, 1e-6); Nspan = max(n_hi - n_lo, 1e-6)
        smap = min((RW - 2 * mrg) / Espan, (H - 2 * mrg) / Nspan)
        xoff = mrg + ((RW - 2 * mrg) - smap * Espan) / 2
        yoff = mrg + ((H - 2 * mrg) - smap * Nspan) / 2

        def P(Ev, Nv):
            return (int(round(xoff + (Ev - e_lo) * smap)),
                    int(round(H - (yoff + (Nv - n_lo) * smap))))

        # Static trajectory backdrop: grid + faint full GT reference line.
        traj = np.full((H, RW, 3), 30, np.uint8)
        for gx in np.linspace(e_lo, e_hi, 5):
            x, _ = P(gx, n_lo); cv2.line(traj, (x, 0), (x, H), (55, 55, 55), 1)
        for gy in np.linspace(n_lo, n_hi, 5):
            _, y = P(e_lo, gy); cv2.line(traj, (0, y), (RW, y), (55, 55, 55), 1)
        gpts = np.array([P(gt_p[i, 1], gt_p[i, 0]) for i in range(len(gt_p))], np.int32)
        cv2.polylines(traj, [gpts], False, (120, 120, 120), 1, cv2.LINE_AA)
        cbar_grad = jetr[(np.linspace(1.0, 0.0, H) * 255).astype(int)]   # top=far

        last_j = -1
        for n, k in enumerate(idxs):
            img = np.asarray(ds.image(k))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            left = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vlo, vhi, rr = frame_clim(k)
            sset = slam_at.get(k, frozenset())
            nn = ns = 0
            for tid, (un, vn) in by_frame.get(k, {}).items():
                uf, vf = to_px(un, vn); u = int(round(uf)); v = int(round(vf))
                r = rr.get(tid, np.nan)
                color = (140, 140, 140) if not np.isfinite(r) else \
                    clut(jetr, (r - vlo) / (vhi - vlo + 1e-9))
                if tid in sset:
                    cv2.circle(left, (u, v), 3, color, -1, cv2.LINE_AA)
                    cv2.circle(left, (u, v), 3, (0, 0, 0), 1, cv2.LINE_AA); ns += 1
                else:
                    cv2.circle(left, (u, v), 4, color, 1, cv2.LINE_AA); nn += 1
            txt(left, f"frame {k}  norm(o) {nn}  SLAM(*) {ns}  "
                      f"depth {vlo:.0f}-{vhi:.0f}m", (6, 18))
            cbar = np.full((H, CBW, 3), 30, np.uint8)
            cbar[:, 8:30] = cbar_grad[:, None, :]
            cv2.rectangle(cbar, (8, 0), (29, H - 1), (255, 255, 255), 1)
            txt(cbar, f"{vhi:.0f}", (34, 16)); txt(cbar, f"{vlo:.0f}", (34, H - 8))
            txt(cbar, "m", (34, H // 2))
            left_full = np.hstack([left, cbar])

            # Extend the persistent trail up to the current frame, then overlay markers.
            j = min(k, len(gt_p) - 1)
            for jj in range(last_j + 1, j + 1):
                if jj > 0:
                    cv2.line(traj, P(gt_p[jj - 1, 1], gt_p[jj - 1, 0]),
                             P(gt_p[jj, 1], gt_p[jj, 0]), (70, 125, 46), 2, cv2.LINE_AA)
                cv2.circle(traj, P(aligned[jj, 1], aligned[jj, 0]), 2,
                           clut(infl, err[jj] / (emax + 1e-9)), -1, cv2.LINE_AA)
            last_j = j
            right = traj.copy()
            pg = P(gt_p[j, 1], gt_p[j, 0]); pe = P(aligned[j, 1], aligned[j, 0])
            cv2.line(right, pg, pe, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(right, (pg[0] - 6, pg[1] - 6), (pg[0] + 6, pg[1] + 6),
                     (70, 180, 70), 2, cv2.LINE_AA)
            cv2.line(right, (pg[0] - 6, pg[1] + 6), (pg[0] + 6, pg[1] - 6),
                     (70, 180, 70), 2, cv2.LINE_AA)
            cv2.circle(right, pe, 4, (0, 0, 255), -1, cv2.LINE_AA)
            txt(right, f"top-down  t={k/25:4.1f}s  err {err[j]:5.1f}m", (6, 18))

            frame = np.hstack([left_full, right])
            if writer is None:
                hh, ww = frame.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (ww, hh))
            writer.write(frame)
            if n % 100 == 0:
                print(f"  [{n}/{len(idxs)}] frame {k}  err {err[j]:.1f}m  "
                      f"normal {nn} slam {ns}")
    else:
        fig, (axi, axt) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.06, wspace=0.14)
        sm = cm.ScalarMappable(norm=Normalize(g_lo, g_hi), cmap=cmap)
        cbar = fig.colorbar(sm, ax=axi, fraction=0.046, pad=0.02)
        cbar.set_label("metric range (m, per-frame)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        for n, k in enumerate(idxs):
            img = np.asarray(ds.image(k))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            vlo, vhi, rr = frame_clim(k)
            sset = slam_at.get(k, frozenset())
            fnorm = Normalize(vlo, vhi)
            sm.set_clim(vlo, vhi)

            axi.clear()
            axi.imshow(img, cmap="gray", vmin=0, vmax=255); axi.axis("off")
            hx, hy, hc = [], [], []      # normal (hollow)
            sx, sy, sc = [], [], []      # slam (solid)
            for tid, (un, vn) in by_frame.get(k, {}).items():
                u, v = to_px(un, vn)
                r = rr.get(tid, np.nan)
                col = (0.55, 0.55, 0.55, 0.9) if not np.isfinite(r) else cmap(fnorm(r))
                if tid in sset:
                    sx.append(u); sy.append(v); sc.append(col)
                else:
                    hx.append(u); hy.append(v); hc.append(col)
            if hx:
                axi.scatter(hx, hy, s=16, facecolors="none", edgecolors=hc, linewidths=0.9)
            if sx:
                axi.scatter(sx, sy, s=18, c=sc, edgecolors="k", linewidths=0.3)
            axi.set_xlim(-0.5, W - 0.5); axi.set_ylim(H - 0.5, -0.5)
            axi.set_title(f"frame {k}   normal(○) {len(hx)}   SLAM(●) {len(sx)}   "
                          f"depth {vlo:.0f}-{vhi:.0f} m", fontsize=10)

            axt.clear()
            j = min(k, len(gt_p) - 1)
            axt.plot(gt_p[:, 1], gt_p[:, 0], color="0.82", lw=1.0, zorder=1)
            axt.plot(gt_p[:j + 1, 1], gt_p[:j + 1, 0], color="#2e7d46", lw=1.8, zorder=2, label="GT")
            axt.scatter(aligned[:j + 1, 1], aligned[:j + 1, 0], c=err[:j + 1], cmap="inferno",
                        s=5, vmin=0, vmax=emax, zorder=3)
            axt.plot([gt_p[j, 1], aligned[j, 1]], [gt_p[j, 0], aligned[j, 0]],
                     color="red", lw=1.2, zorder=4)
            axt.scatter([gt_p[j, 1]], [gt_p[j, 0]], c="#2e7d46", s=45, marker="x", zorder=5)
            axt.scatter([aligned[j, 1]], [aligned[j, 0]], c="red", s=30, zorder=5, label="estimate")
            axt.set_xlim(e_lo, e_hi); axt.set_ylim(n_lo, n_hi); axt.set_aspect("equal")
            axt.set_xlabel("East (m)", fontsize=8); axt.set_ylabel("North (m)", fontsize=8)
            axt.set_title(f"top-down  t={k/25:4.1f}s  err {err[j]:5.1f} m", fontsize=10)
            axt.tick_params(labelsize=7); axt.grid(alpha=0.25)
            if n == 0:
                axt.legend(loc="upper right", fontsize=8)

            fig.canvas.draw()
            frame = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3],
                                 cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
            writer.write(frame)
            if n % 100 == 0:
                print(f"  [{n}/{len(idxs)}] frame {k}  err {err[j]:.1f}m  "
                      f"normal {len(hx)} slam {len(sx)}")
    writer.release()
    tmr["render"] = time.perf_counter() - t0c - sum(tmr.values())
    print(f"wrote {args.out}  ({len(idxs)} frames @ {args.fps} fps)")
    tot = time.perf_counter() - t0c
    print("phase timing:  " + "  ".join(
        f"{k} {v:.1f}s ({100*v/tot:.0f}%)" for k, v in tmr.items())
        + f"  |  total {tot:.1f}s")


if __name__ == "__main__":
    main()
