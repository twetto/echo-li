"""FlowDep-F + Sparse3D on EuRoC Vicon rooms against the Leica scan, with the audits the
2026-07 harness never had.

The 2026-07 EuRoC depth evaluator was quarantined ("Vicon/Leica/timing errors contaminated the
measured residuals") without establishing whether the flaw was in the dataset or in our
projection. This script runs the same tracker + filter that is validated on TartanAir and
MidAir at a known pose, and scores against a z-buffer of `mav0/pointcloud0/data.ply` projected
through the fused Vicon+IMU ground truth, plus three audits that separate our side from theirs:

  1. GT self-consistency: the z-buffer of frame A back-projected to the world and re-projected
     into frame B against B's own z-buffer. Pose/scan misalignment or a frame-convention slip
     shows here with no tracker involved.
  2. Time offset: the whole run repeated with the GT pose sampled at t + delta; a dataset timing
     error puts the depth-error minimum away from delta = 0.
  3. Scale/offset: FlowDep-F range against GT range as a regression (slope != 1 is a scale
     error in the scan or the poses; an intercept is a frame offset).

Frames: cam0 (752x480, radtan) undistorted to a pinhole with the same K, then scaled. The
pointcloud is cached as .npy on first use. Range is radial (|P_c|), like MidAir.

Usage:
  .venv/bin/python echo-li-python/tests/diagnostics/euroc_fdf_depth.py \
      --root ~/18TB/datasets/EuRoC/vicon_room1/V1_01_easy/V1_01_easy --start 300 --frames 300 \
      --config configs/diagnostics_midair_e2e_depth_nees.yaml --cache $SCRATCH/euroc
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
import fdf_densify_study as dz  # noqa: E402
from tartanair_e2e_depth_nees import SPARSE_KEYS  # noqa: E402

R_MARGIN = 4


def load_gt(root):
    d = np.loadtxt(root / "mav0/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    t = d[:, 0] * 1e-9
    p = d[:, 1:4]
    q = d[:, [5, 6, 7, 4]]          # EuRoC stores w,x,y,z; scipy wants x,y,z,w
    return t, p, Rot.from_quat(q)


def load_cam(root, cam="cam0"):
    y = yaml.safe_load(open(root / f"mav0/{cam}/sensor.yaml"))
    T_BS = np.array(y["T_BS"]["data"], float).reshape(4, 4)
    fu, fv, cu, cv_ = y["intrinsics"]
    D = np.array(y["distortion_coefficients"], float)
    W, H = y["resolution"]
    rows = [ln.strip().split(",") for ln in open(root / f"mav0/{cam}/data.csv") if ln.strip() and not ln.startswith("#")]
    ts = np.array([int(r[0]) for r in rows], np.int64) * 1e-9
    names = [r[1].strip() for r in rows]          # keep the exact filename: ns timestamps do not survive float64
    return T_BS, (fu, fv, cu, cv_), D, (W, H), ts, names


def load_cloud(root, cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{root.name}_cloud.npy"
    if not f.exists():
        t0 = time.time()
        P = np.loadtxt(root / "mav0/pointcloud0/data.ply", skiprows=12, usecols=(0, 1, 2), dtype=np.float32)
        np.save(f, P)
        print(f"cloud parsed: {P.shape[0]} points in {time.time() - t0:.0f} s -> {f}")
    return np.load(f)


def stereo_setup(root, W0, H0):
    y0 = yaml.safe_load(open(root / "mav0/cam0/sensor.yaml")); y1 = yaml.safe_load(open(root / "mav0/cam1/sensor.yaml"))
    K0 = np.array([[y0["intrinsics"][0], 0, y0["intrinsics"][2]], [0, y0["intrinsics"][1], y0["intrinsics"][3]], [0, 0, 1]])
    K1 = np.array([[y1["intrinsics"][0], 0, y1["intrinsics"][2]], [0, y1["intrinsics"][1], y1["intrinsics"][3]], [0, 0, 1]])
    D0 = np.array(y0["distortion_coefficients"], float); D1 = np.array(y1["distortion_coefficients"], float)
    T0 = np.array(y0["T_BS"]["data"], float).reshape(4, 4); T1 = np.array(y1["T_BS"]["data"], float).reshape(4, 4)
    T_10 = np.linalg.inv(T1) @ T0                 # cam0 -> cam1
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, D0, K1, D1, (W0, H0), np.ascontiguousarray(T_10[:3, :3]), np.ascontiguousarray(T_10[:3, 3]).reshape(3, 1), flags=0, alpha=0)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (W0, H0), cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (W0, H0), cv2.CV_32FC1)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7, P1=8 * 49, P2=32 * 49,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return m0, m1, sgbm, R1, P1, Q


def stereo_cloud_cam0(raw0, raw1, st):
    """SGBM disparity -> 3-D points in the ORIGINAL cam0 frame (rectification undone)."""
    m0, m1, sgbm, R1, P1, Q = st
    r0 = cv2.remap(raw0, m0[0], m0[1], cv2.INTER_LINEAR); r1 = cv2.remap(raw1, m1[0], m1[1], cv2.INTER_LINEAR)
    disp = sgbm.compute(r0, r1).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    ok = (disp > 1.0) & np.isfinite(pts).all(-1) & (pts[..., 2] > 0.3) & (pts[..., 2] < 30)
    P = pts[ok].reshape(-1, 3) @ R1              # rectified cam0 -> original cam0 (P_orig = R1^T P_rect)
    return P


def zbuffer(P_w, T_wc, f, cx, cy, W, H, splat=1):
    """Radial range z-buffer of the cloud in camera (T_wc), pinhole (f, cx, cy) at WxH.
    Min range per pixel; `splat` dilates each point to a (2s+1)^2 block to close pinholes."""
    T_cw = np.linalg.inv(T_wc)
    Pc = P_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = Pc[:, 2]
    m = z > 0.1
    Pc = Pc[m]; z = z[m]
    u = f * Pc[:, 0] / z + cx; v = f * Pc[:, 1] / z + cy
    r = np.linalg.norm(Pc, axis=1)
    img = np.full((H, W), np.inf, np.float32)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            ui = np.round(u).astype(int) + dx; vi = np.round(v).astype(int) + dy
            ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            np.minimum.at(img.ravel(), vi[ok] * W + ui[ok], r[ok].astype(np.float32))
    img[~np.isfinite(img)] = np.nan
    return img


def zbuf_median(img, k):
    if k <= 1:
        return img
    filled = np.where(np.isfinite(img), img, 0).astype(np.float32)
    cnt = np.isfinite(img).astype(np.float32)
    med = cv2.medianBlur(filled, k)
    ok = cv2.blur(cnt, (k, k)) > 0.5          # majority of the window valid
    out = np.where(ok & np.isfinite(img), med, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--splat", type=int, default=1)
    ap.add_argument("--zbuf-median", type=int, default=0,
                    help="apply a kxk median to the Leica z-buffer after the min-splat (0 = off): removes isolated "
                         "far points leaking through gaps in the scanned surfaces and near clutter")
    ap.add_argument("--stereo", action="store_true",
                    help="third depth source: SGBM stereo from cam0+cam1 at full resolution, back-projected to a cam0 "
                         "point cloud and z-buffered into the working frame like the Leica scan; scores FlowDep-F vs "
                         "stereo and stereo vs Leica on the same pixels")
    ap.add_argument("--time-offset-ms", type=float, default=0.0, help="sample the GT pose at t + this")
    ap.add_argument("--min-track", type=int, default=5)
    ap.add_argument("--sparse", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--fdf", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--consistency-pairs", type=int, default=20, help="frame pairs (k, k+5) for audit 1")
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--track-source", choices=("fdf", "rudolf"), default="fdf",
                    help="rudolf: the FAST+KLT front end on the same undistorted frames feeding the same Sparse3D "
                         "(the control for whether a bias is the patch tracker's or common to filter/pose/scene)")
    args = ap.parse_args()
    root = Path(args.root)

    gt_t, gt_p, gt_R = load_gt(root)
    T_BS, (fu, fv, cu, cv_), D, (W0, H0), cam_t, cam_names = load_cam(root)
    K0 = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], float)
    s = args.scale
    W, H = int(round(W0 * s)), int(round(H0 * s))
    f = float(0.5 * (fu + fv) * s); cx = float(cu * s); cy = float(cv_ * s)
    K1 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    map1, map2 = cv2.initUndistortRectifyMap(K0, D, np.eye(3), K1, (W, H), cv2.CV_32FC1)
    slerp = Slerp(gt_t, gt_R)

    def pose_at(t):
        t = float(np.clip(t, gt_t[0], gt_t[-1]))
        T = np.eye(4)
        T[:3, :3] = slerp([t]).as_matrix()[0]
        T[:3, 3] = np.array([np.interp(t, gt_t, gt_p[:, i]) for i in range(3)])
        return T @ T_BS                     # T_WS @ T_SC = T_WC

    P_w = load_cloud(root, args.cache)
    print(f"EuRoC {root.name}: {len(cam_t)} frames, GT {gt_t[0]:.2f}..{gt_t[-1]:.2f} s, cloud {len(P_w)} pts, "
          f"working {W}x{H} f {f:.1f}, time offset {args.time_offset_ms:+.1f} ms")

    cfg = yaml.safe_load(open(args.config)) or {}
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    settings = {k: sparse_cfg[k] for k in SPARSE_KEYS if k in sparse_cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings["max_pool_size"] = 4000
    settings["min_track_length"] = args.min_track
    for kv in args.sparse:
        k, v = kv.split("=", 1)
        settings[k] = type(sparse_cfg.get(k, 0.0))(v) if not isinstance(sparse_cfg.get(k), bool) else v.lower() == "true"
    fdf_settings = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    fdf = echo_li.FlowDepF(f, f, cx, cy, W, H, fdf_settings or None) if args.track_source == "fdf" else None
    klt = None
    if args.track_source == "rudolf":
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(cfg, tf); cfg_path = tf.name
        fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
        fcfg.set_camera(f, f, cx, cy, W, H, [])
        klt = echo_li.Frontend(fcfg, W, H)
        print("track source: Rudolf-V (FAST+KLT) on the undistorted pinhole frame")
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)
    dz.SLANT_G[0] = np.tan(np.radians(35.0)) / f

    # --- Audit 1: GT self-consistency (no tracker) ---
    cons = []
    ks = np.linspace(args.start, args.start + args.frames - 6, args.consistency_pairs).astype(int)
    for k in ks:
        Ta = pose_at(cam_t[k] + args.time_offset_ms * 1e-3); Tb = pose_at(cam_t[k + 5] + args.time_offset_ms * 1e-3)
        za = zbuffer(P_w, Ta, f, cx, cy, W, H, args.splat); zb = zbuffer(P_w, Tb, f, cx, cy, W, H, args.splat)
        yy, xx = np.mgrid[0:H, 0:W]
        ray = np.stack([(xx - cx) / f, (yy - cy) / f, np.ones_like(xx, float)], -1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        ok = np.isfinite(za)
        Pa = ray[ok] * za[ok][:, None]
        Pw = Pa @ Ta[:3, :3].T + Ta[:3, 3]
        Tbi = np.linalg.inv(Tb); Pb = Pw @ Tbi[:3, :3].T + Tbi[:3, 3]
        zpos = Pb[:, 2] > 0.1
        ub = np.round(f * Pb[zpos, 0] / Pb[zpos, 2] + cx).astype(int); vb = np.round(f * Pb[zpos, 1] / Pb[zpos, 2] + cy).astype(int)
        rb = np.linalg.norm(Pb[zpos], axis=1)
        inb = (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
        zb_at = zb[vb[inb], ub[inb]]
        good = np.isfinite(zb_at)
        rel = (rb[inb][good] - zb_at[good]) / zb_at[good]
        cons.append(rel)
    cons = np.concatenate(cons)
    print(f"AUDIT 1 GT self-consistency ({len(ks)} pairs, k -> k+5): |rel| median {100 * np.median(np.abs(cons)):.2f}%  "
          f"p90 {100 * np.percentile(np.abs(cons), 90):.2f}%  signed median {100 * np.median(cons):+.2f}%  (occlusion changes make a one-sided tail; the median is the misalignment)")

    st = stereo_setup(root, W0, H0) if args.stereo else None
    T_cw_of = lambda T: np.linalg.inv(T)  # noqa: E731
    # --- Tracker + filter at GT pose ---
    rows = []; dense = []; dense_cov = []; n_conv = []; stereo_rows = []; sl_rows = []
    t0 = time.time()
    for i in range(args.frames):
        k = args.start + i
        raw = cv2.imread(str(root / "mav0/cam0/data" / cam_names[k]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        gray = np.ascontiguousarray(cv2.remap(raw, map1, map2, cv2.INTER_LINEAR))
        T_wc = pose_at(cam_t[k] + args.time_offset_ms * 1e-3)
        if fdf is not None:
            uv = fdf.process(gray, T_wc.tolist())
        else:
            feats, _ = klt.process(gray)
            uv = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
        sparse.update(float(cam_t[k]), uv, T_wc.tolist(), None, None)
        ids, rr, vc, vr, cvg, tl = sparse.range_summary()
        okp = vc > 0
        post = {int(a): (float(b), float(np.sqrt(c) * b)) for a, b, c in zip(ids[okp], rr[okp], vc[okp])}
        if post and fdf is not None:
            fdf.set_posterior(post)
        gt = zbuf_median(zbuffer(P_w, T_wc, f, cx, cy, W, H, args.splat), args.zbuf_median)
        gs = None
        if st is not None:
            raw1 = cv2.imread(str(root / "mav0/cam1/data" / cam_names[k]), cv2.IMREAD_GRAYSCALE)
            if raw1 is not None:
                Pc0 = stereo_cloud_cam0(raw, raw1, st)                      # cam0 frame
                Pw_st = Pc0 @ T_wc[:3, :3].T + T_wc[:3, 3]
                gs = zbuffer(Pw_st, T_wc, f, cx, cy, W, H, 1)
                both = np.isfinite(gt) & np.isfinite(gs) & (gt > 0.3)
                if both.any():
                    sl_rows.append(np.stack([gt[both], gs[both]], 1)[::7].astype(np.float32))
        lev = {o[0]: o[3] for o in fdf.last_observations()} if fdf is not None else {}
        items = []
        for fid, r_, vc_, vr_, cv1, tl_ in zip(ids, rr, vc, vr, cvg, tl):
            fid = int(fid)
            if not cv1 or vc_ <= 0 or fid not in uv or tl_ < args.min_track:
                continue
            u, v = uv[fid]
            items.append((u, v, lev.get(fid, 0), float(np.log(r_)), float(vc_), 0.0))
            gx, gy = int(round(u)), int(round(v))
            if not (R_MARGIN < gx < W - R_MARGIN and R_MARGIN < gy < H - R_MARGIN):
                continue
            g = gt[gy, gx]
            if not np.isfinite(g) or g < 0.3:
                continue
            win = gt[gy - 2:gy + 3, gx - 2:gx + 3]; win = win[np.isfinite(win)]
            spread = (win.max() - win.min()) / np.median(win) if win.size else 9.0
            rows.append((k, tl_, r_, g, (np.log(r_) - np.log(g)) ** 2 / vc_, spread))
            if gs is not None and np.isfinite(gs[gy, gx]) and gs[gy, gx] > 0.3:
                stereo_rows.append((r_, gs[gy, gx], g))
        n_conv.append(len(items))
        if items and i % 2 == 0:
            eta_p, var_p, covm = dz.fuse(items, H, W)["nearest_disagree_slant"]
            vg = np.isfinite(gt) & (gt > 0.3)
            m = covm & vg
            e = eta_p[m] - np.log(gt[m])
            dense.append(np.stack([np.exp(e) - 1.0, e * e / np.maximum(var_p[m], 1e-12)], 1).astype(np.float32)[::2])
            dense_cov.append(float(covm.mean()))
        if (i + 1) % 50 == 0:
            print(f"  frame {k}: {len(items)} converged, {time.time() - t0:.0f} s", flush=True)

    A = np.array(rows, float)
    if len(A) == 0:
        print("no scored landmarks"); return
    k_, tl, re, g, nees, sp = A.T
    rel = re / g - 1.0; ua = sp <= 0.2
    print(f"\nLANDMARKS (converged, tl >= {args.min_track}): {len(A)} obs over {args.frames} frames ({np.median(n_conv):.0f} converged/frame)")
    print(f"  all:   abs err median {100 * np.median(np.abs(rel)):.1f}%  p90 {100 * np.percentile(np.abs(rel), 90):.1f}%  signed median {100 * np.median(rel):+.1f}%  NEES median {np.median(nees):.3f}  >3.84 {100 * np.mean(nees > 3.84):.1f}%  >6.63 {100 * np.mean(nees > 6.63):.1f}%")
    print(f"  GT-unambiguous ({100 * ua.mean():.0f}%): abs err median {100 * np.median(np.abs(rel[ua])):.1f}%  p90 {100 * np.percentile(np.abs(rel[ua]), 90):.1f}%  signed {100 * np.median(rel[ua]):+.1f}%  NEES median {np.median(nees[ua]):.3f}  >3.84 {100 * np.mean(nees[ua] > 3.84):.1f}%  >6.63 {100 * np.mean(nees[ua] > 6.63):.1f}%")
    # Audit 3: scale / offset
    X = np.stack([g[ua], np.ones(ua.sum())], 1); coef = np.linalg.lstsq(X, re[ua], rcond=None)[0]
    print(f"AUDIT 3 FlowDep-F range vs GT range (unambiguous): slope {coef[0]:.4f}  intercept {coef[1]:+.3f} m  (1, 0 = no scale / frame offset);  GT range median {np.median(g):.2f} m")
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 100)):
        m = ua & (g >= lo) & (g < hi)
        if m.sum() > 100:
            print(f"    GT {lo:.1f}-{hi:.0f} m: n {m.sum():6d}  signed median {100 * np.median(rel[m]):+.1f}%  abs {100 * np.median(np.abs(rel[m])):.1f}%")
    if stereo_rows:
        Sr = np.array(stereo_rows, float); r_s, g_s, g_l = Sr.T
        rel_s = r_s / g_s - 1.0; rel_ls = g_l / g_s - 1.0
        X = np.stack([g_s, np.ones(len(g_s))], 1)
        c1 = np.linalg.lstsq(X, r_s, rcond=None)[0]; c2 = np.linalg.lstsq(X, g_l, rcond=None)[0]
        print(f"STEREO (SGBM cam0+cam1) at the scored landmark pixels, n {len(Sr)}:")
        print(f"  FlowDep-F vs stereo: abs err median {100 * np.median(np.abs(rel_s)):.1f}%  signed {100 * np.median(rel_s):+.1f}%  slope {c1[0]:.4f} intercept {c1[1]:+.3f} m")
        print(f"  Leica    vs stereo: abs err median {100 * np.median(np.abs(rel_ls)):.1f}%  signed {100 * np.median(rel_ls):+.1f}%  slope {c2[0]:.4f} intercept {c2[1]:+.3f} m")
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 100)):
            m = (g_s >= lo) & (g_s < hi)
            if m.sum() > 100:
                print(f"    stereo {lo:.1f}-{hi:.0f} m: n {m.sum():6d}  FlowDep-F signed {100 * np.median(rel_s[m]):+.1f}%  Leica signed {100 * np.median(rel_ls[m]):+.1f}%")
    if sl_rows:
        SL = np.concatenate(sl_rows); rel = SL[:, 0] / SL[:, 1] - 1.0
        X = np.stack([SL[:, 1], np.ones(len(SL))], 1); c = np.linalg.lstsq(X, SL[:, 0], rcond=None)[0]
        print(f"LEICA z-buffer vs STEREO over all pixels where both exist (n {len(SL)}): abs median {100 * np.median(np.abs(rel)):.1f}%  signed {100 * np.median(rel):+.1f}%  slope {c[0]:.4f} intercept {c[1]:+.3f} m")
        for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 100)):
            m = (SL[:, 1] >= lo) & (SL[:, 1] < hi)
            if m.sum() > 100:
                print(f"    stereo {lo:.1f}-{hi:.0f} m: n {m.sum():7d}  Leica signed {100 * np.median(rel[m]):+.1f}%  abs {100 * np.median(np.abs(rel[m])):.1f}%")
    if dense:
        Dn = np.concatenate(dense)
        print(f"DENSE (nearest+disagreement+footprint, every 2nd frame): coverage {100 * np.mean(dense_cov):.1f}%  err median {100 * np.median(np.abs(Dn[:, 0])):.1f}%  p90 {100 * np.percentile(np.abs(Dn[:, 0]), 90):.1f}%  NEES median {np.median(Dn[:, 1]):.3f}  >3.84 {100 * np.mean(Dn[:, 1] > 3.84):.1f}%")
    if args.save_npz:
        np.savez_compressed(args.save_npz, rows=A, cons=cons, dense=np.concatenate(dense) if dense else np.zeros((0, 2)))


if __name__ == "__main__":
    main()
