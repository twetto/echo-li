"""Sparse3D consistency scorecard on Mid-Air with exact GT pose/depth.

This is the Mid-Air analogue of real_depth_eval.py: feed tracked pixels and exact
camera poses into Sparse3DFilter, then score the filter's reported depth/3D
covariance against the exact world point attached to each feature at birth.

The intended use is to decide whether Sparse3D becomes inconsistent even when the
pose and depth truth are synthetic/exact, and whether that inconsistency follows
track age, measurement drift, occlusion, or depth discontinuities.

  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_sparse3d_nees.py \
      --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --set VO_test --cond sunny --traj 0 --frames 300 --scale 0.5

Use --measurements exact as a control: the same birth points are observed by
exact reprojection instead of tracker pixels, isolating Sparse3D from front-end
drift. Use --measurements rudolf to feed the actual Rudolf-V frontend.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import echo_li

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import photometric_klt_ab as pk  # noqa: E402


LV, R, ITERS = md.LV, md.R, md.ITERS
SIGMA_PX = 0.5
SPARSE_VOG_KEYS = [
    "max_pool_size", "min_track_length", "conv_inlier_ratio", "conv_variance_threshold",
    "init_depth_var", "init_invdepth_var", "sigma_pixel", "flow_age_rate_px_per_frame",
    "uniform_z_max", "uniform_rho_max", "uniform_d_min", "uniform_d_max",
    "a_init", "b_init", "ab_min", "ab_max",
    "min_inlier_ratio", "mahalanobis_reset_chi2", "process_depth_var", "min_parallax",
    "min_cos_sim", "min_depth", "max_depth", "reanchor_flow_px", "use_equivariant_output",
    "iekf_iterations", "range_walk_var", "rotation_unscented",
]


def camera_point(Xw, T_wb):
    """World point -> current camera coordinates, using Mid-Air body<-camera Rt."""
    T_bw = np.linalg.inv(T_wb)
    pb = T_bw[:3, :3] @ Xw + T_bw[:3, 3]
    return md.RT_BC[:3, :3].T @ pb


def camera_point_fast(Xw, T_bw):
    pb = T_bw[:3, :3] @ Xw + T_bw[:3, 3]
    return md.RT_BC[:3, :3].T @ pb


def project_world_fast(Xw, T_bw, f, cx, cy):
    pc = camera_point_fast(Xw, T_bw)
    if pc[2] <= 1e-6:
        return None, None, None, pc
    u = f * pc[0] / pc[2] + cx
    v = f * pc[1] / pc[2] + cy
    return np.array([u, v]), float(np.linalg.norm(pc)), float(pc[2]), pc


def finite_mahalanobis(err, cov):
    cov = np.asarray(cov, float)
    try:
        return float(err @ np.linalg.solve(cov + 1e-12 * np.eye(3), err))
    except np.linalg.LinAlgError:
        return np.nan


def finite_quadratic(err, cov):
    err = np.asarray(err, float)
    cov = np.asarray(cov, float)
    if not np.isfinite(err).all() or not np.isfinite(cov).all():
        return np.nan
    try:
        return float(err @ np.linalg.solve(cov + 1e-12 * np.eye(len(err)), err))
    except np.linalg.LinAlgError:
        return np.nan


def nees_decomposition(err, cov, est, pc, f):
    """Return tangent/radial 3D consistency diagnostics.

    `z2` is the marginal camera-z NEES already printed by the script.
    `xy_marg` is the marginal NEES of the camera x/y Cartesian error.
    `xy_cond` is the conditional NEES of x/y after accounting for z error:

        full 3D NEES = z2 + xy_cond

    up to numerical jitter. This makes the radial/tangent split exact for the
    reported Euclidean covariance instead of comparing unrelated projections.
    `pix_nees` projects the same 3D covariance into the image plane for a
    measurement-space tangent check.
    """
    cov = 0.5 * (np.asarray(cov, float) + np.asarray(cov, float).T)
    err = np.asarray(err, float)
    est = np.asarray(est, float)
    pc = np.asarray(pc, float)

    pzz = float(cov[2, 2])
    if pzz <= 0 or not np.isfinite(pzz):
        return np.nan, np.nan, np.nan, np.nan

    xy_marg = finite_quadratic(err[:2], cov[:2, :2])

    pxz = cov[:2, 2]
    pxy_cond = cov[:2, :2] - np.outer(pxz, pxz) / pzz
    exy_cond = err[:2] - pxz * (err[2] / pzz)
    xy_cond = finite_quadratic(exy_cond, pxy_cond)

    pix_nees = np.nan
    if est[2] > 1e-6 and pc[2] > 1e-6:
        uv_est = f * est[:2] / est[2]
        uv_gt = f * pc[:2] / pc[2]
        pix_err = uv_est - uv_gt
        z = est[2]
        j_proj = np.array([
            [f / z, 0.0, -f * est[0] / (z * z)],
            [0.0, f / z, -f * est[1] / (z * z)],
        ])
        pix_cov = j_proj @ cov @ j_proj.T
        pix_nees = finite_quadratic(pix_err, pix_cov)

    full_from_parts = err[2] * err[2] / pzz + xy_cond
    return xy_marg, xy_cond, pix_nees, full_from_parts


def print_bin_stats(name, values, mask, label):
    m = mask & np.isfinite(values)
    if m.sum() < 20:
        return
    v = values[m]
    print(f"  {name:>12s} {label:>12s}: n={int(m.sum()):6d}  "
          f"median {np.median(v):8.2f}  mean {np.mean(v):9.2f}  p90 {np.percentile(v, 90):9.2f}")


def chart_from_config(name):
    if name in ("bearing_invdepth_additive3d", "bearing_invdepth_additive"):
        return "bearing_invdepth_additive"
    if name in ("invdepth_additive3d", "invdepth_additive"):
        return "invdepth_additive"
    raise ValueError(f"unsupported SparseVog parametrization for this diagnostic: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--redetect", type=int, default=250)
    ap.add_argument("--min-track", type=int, default=10)
    ap.add_argument("--sigma-pixel", type=float, default=None)
    ap.add_argument("--flow-age-rate-px-per-frame", type=float, default=None,
                    help="age-dependent Sparse3D pixel-noise rate r in sigma^2+(r*age)^2")
    ap.add_argument("--init-depth-var", type=float, default=None)
    ap.add_argument("--process-depth-var", type=float, default=None)
    ap.add_argument("--range-walk-var", type=float, default=None)
    ap.add_argument("--min-parallax", type=float, default=None)
    ap.add_argument("--max-depth", type=float, default=None)
    ap.add_argument("--a-init", type=float, default=None,
                    help="override SparseVog Gaussian-Beta inlier prior a")
    ap.add_argument("--b-init", type=float, default=None,
                    help="override SparseVog Gaussian-Beta outlier prior b")
    ap.add_argument("--ab-max", type=float, default=None,
                    help="override SparseVog Gaussian-Beta concentration cap")
    ap.add_argument("--mahalanobis-reset-chi2", type=float, default=None,
                    help="override hard reset gate; <=0 disables in Sparse3D")
    ap.add_argument("--measurements", default="rudolf", choices=["rudolf", "klt", "exact"],
                    help="feed Rudolf-V, diagnostic Python KLT, or exact reprojections")
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--video", action="store_true",
                    help="save an overlay video comparing exact and tracker measurements")
    ap.add_argument("--video-only", action="store_true",
                    help="render the measurement overlay without Sparse3D update/scoring")
    ap.add_argument("--video-out", default="midair_sparse3d_measurements.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--draw-max", type=int, default=160,
                    help="max tracks to draw per frame, chosen by largest tracker/exact error")
    ap.add_argument("--border", type=int, default=24,
                    help="pixels from image edge considered border-risk for score splits")
    ap.add_argument("--chart", default="config",
                    choices=["config", "invdepth_additive", "bearing_invdepth_additive"])
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.1f}")

    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    chart = chart_from_config(sparse_cfg.get("parametrization", "invdepth_additive3d"))
    if args.chart != "config":
        chart = args.chart
    settings = {k: sparse_cfg[k] for k in SPARSE_VOG_KEYS if k in sparse_cfg}
    settings.setdefault("sigma_pixel", SIGMA_PX)
    settings.setdefault("min_track_length", 1)
    if args.sigma_pixel is not None:
        settings["sigma_pixel"] = args.sigma_pixel
    for key, value in [
        ("init_depth_var", args.init_depth_var),
        ("flow_age_rate_px_per_frame", args.flow_age_rate_px_per_frame),
        ("process_depth_var", args.process_depth_var),
        ("range_walk_var", args.range_walk_var),
        ("min_parallax", args.min_parallax),
        ("max_depth", args.max_depth),
        ("a_init", args.a_init),
        ("b_init", args.b_init),
        ("ab_max", args.ab_max),
        ("mahalanobis_reset_chi2", args.mahalanobis_reset_chi2),
    ]:
        if value is not None:
            settings[key] = value
    if args.video_only and not args.video:
        ap.error("--video-only requires --video")

    filt = None
    if chart == "bearing_invdepth_additive":
        filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
            echo_li.PinholeCamera(f, f, cx, cy), **settings) if not args.video_only else None
    else:
        filt = echo_li.Sparse3DFilter.invdepth_additive3d(
            f, f, cx, cy, **settings) if not args.video_only else None
    print(f"chart: {chart}  settings: {settings}")
    tracker = None
    if args.measurements == "rudolf":
        fcfg = echo_li.FrontendConfig.from_yaml(args.config)
        fcfg.set_camera(f, f, cx, cy, W, H, [])
        tracker = echo_li.Frontend(fcfg, W, H)
        print(f"frontend: Rudolf-V  config={Path(args.config).name}  {fcfg}")

    Xw = {}
    born = {}
    pos_prev = {}
    seen_ids = set()
    nid = 0
    prev = None
    rows = []
    n_live_updates = 0
    t0 = time.time()
    last = min(args.start + args.frames, ds.n)
    writer = vpath = None
    if args.video:
        writer, vpath = md.open_writer(args.video_out, W, H, args.fps)

    for i in range(args.start, last):
        img = ds.image(i)
        depth = None
        T_wb = ds.pose(i)
        T_bw = np.linalg.inv(T_wb)
        T_wc = T_wb @ md.RT_BC
        cur_pyr = cgx = cgy = None
        if args.measurements == "klt":
            cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)

        rudolf_uvs = {}
        if tracker is not None:
            feats, stats = tracker.process(img)
            rudolf_uvs = {
                int(fd["id"]): np.array([float(fd["x"]), float(fd["y"])])
                for fd in feats
            }
            new_rudolf = [(j, p) for j, p in rudolf_uvs.items() if j not in born]
            if new_rudolf:
                depth = ds.depth(i)
            for j, p in rudolf_uvs.items():
                if j in born:
                    pos_prev[j] = p
                    continue
                gx, gy = int(round(p[0])), int(round(p[1]))
                if not (R + 2 <= gx < W - R - 2 and R + 2 <= gy < H - R - 2):
                    continue
                d_rng = float(depth[gy, gx])
                if not (1.0 < d_rng < md.SKY):
                    continue
                Xw[j] = md.backproject_world((float(p[0]), float(p[1])), d_rng, T_wb, f, cx, cy)
                born[j] = i
                pos_prev[j] = p
                seen_ids.add(j)
            # In Rudolf mode, pos_prev must be the current frontend output only.
            # Keeping old IDs after Rudolf-V has dropped them draws frozen points
            # and feeds stale measurements to Sparse3D.
            pos_prev = {j: p for j, p in rudolf_uvs.items() if j in born}
        else:
            stats = None

        if prev is not None and pos_prev and args.measurements == "klt":
            ppyr, _pgx, _pgy = prev
            ids = list(pos_prev)
            u, valid, _ = pk.klt_track(
                ppyr, cur_pyr, cgx, cgy,
                np.array([pos_prev[j] for j in ids]), R, ITERS, "ssd")
            for k, j in enumerate(ids):
                if valid[k]:
                    pos_prev[j] = u[k]
                else:
                    pos_prev.pop(j, None)

        drop = []
        gt_pix = {}
        gt_cam = {}
        drift_px = {}
        for j in list(born):
            gp, rng, _z, pc = project_world_fast(Xw[j], T_bw, f, cx, cy)
            inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
            alive = j in pos_prev if args.measurements in ("klt", "rudolf") else True
            if not inb or not alive:
                drop.append(j)
                continue
            if args.measurements == "exact":
                pos_prev[j] = gp.copy()
            gt_pix[j] = gp
            gt_cam[j] = pc
            drift_px[j] = float(np.linalg.norm(pos_prev[j] - gp))

        for j in drop:
            for d in (Xw, born, pos_prev):
                d.pop(j, None)

        if args.measurements in ("klt", "exact") and len(pos_prev) < args.redetect:
            if depth is None:
                depth = ds.depth(i)
            mask = np.uint8((depth < md.SKY) & (depth > 1.0)) * 255
            for p in pos_prev.values():
                cv2.circle(mask, (int(p[0]), int(p[1])), R + 2, 0, -1)
            corners = cv2.goodFeaturesToTrack(
                img, args.max_tracks - len(pos_prev), 0.01, 2 * R + 3, mask=mask)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    x, y = float(c[0]), float(c[1])
                    gx, gy = int(round(x)), int(round(y))
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < md.SKY):
                        continue
                    Xw[nid] = md.backproject_world((x, y), d_rng, T_wb, f, cx, cy)
                    born[nid] = i
                    pos_prev[nid] = np.array([x, y])
                    seen_ids.add(nid)
                    nid += 1

        if pos_prev and filt is not None:
            if args.measurements == "exact":
                uvs = {}
                for j in list(pos_prev):
                    gp, _rng, _z, _pc = project_world_fast(Xw[j], T_bw, f, cx, cy)
                    if gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2:
                        uvs[int(j)] = (float(gp[0]), float(gp[1]))
            else:
                uvs = {int(j): (float(p[0]), float(p[1])) for j, p in pos_prev.items()}
            filt.update(float(i) / 25.0, uvs, T_wc.tolist(), None, None)
            n_live_updates += len(uvs)

        if writer is not None:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            draw_ids = [j for j in gt_pix if j in pos_prev]
            draw_ids.sort(key=lambda j: drift_px.get(j, 0.0), reverse=True)
            for j in draw_ids[:args.draw_max]:
                gp = gt_pix[j]
                kp = pos_prev[j]
                g = (int(round(gp[0])), int(round(gp[1])))
                k = (int(round(kp[0])), int(round(kp[1])))
                if drift_px[j] > 0.75:
                    cv2.line(vis, g, k, (0, 80, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(vis, g, (0, 255, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                cv2.rectangle(vis, (g[0] - R, g[1] - R), (g[0] + R, g[1] + R),
                              (0, 180, 0), 1, cv2.LINE_AA)
                cv2.drawMarker(vis, k, (0, 215, 255), cv2.MARKER_TILTED_CROSS,
                               8, 1, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (W, 44), (0, 0, 0), -1)
            drift_vals = [drift_px[j] for j in draw_ids]
            med_drift = np.median(drift_vals) if drift_vals else 0.0
            p90_drift = np.percentile(drift_vals, 90) if drift_vals else 0.0
            stat = ""
            if stats is not None:
                stat = (f"  tracked {stats.get('tracked', 0)} lost {stats.get('lost', 0)} "
                        f"new {stats.get('new_detections', 0)}")
            cv2.putText(vis, f"frame {i}  live {len(pos_prev)}  shown {min(len(draw_ids), args.draw_max)}{stat}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"exact +/square   {args.measurements} x   "
                        f"drift median {med_drift:.2f}px p90 {p90_drift:.2f}px",
                        (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(vis)

        if args.video_only:
            if args.measurements == "klt":
                prev = (cur_pyr, cgx, cgy)
            if (i - args.start) % 50 == 0:
                fps = (i - args.start + 1) / max(time.time() - t0, 1e-9)
                print(f"  [{i - args.start:4d}/{last - args.start}] active={len(pos_prev):4d} "
                      f"shown={min(len(gt_pix), args.draw_max):4d} {fps:5.1f} fps")
            continue

        fdict = filt.get_features()
        for fid, fd in fdict.items():
            j = int(fid)
            if j not in gt_cam or j not in pos_prev:
                continue
            track_len = int(fd["track_length"])
            if track_len < args.min_track:
                continue
            est = np.asarray(fd["position"], float)
            cov = np.asarray(fd["covariance_euclidean"], float)
            pc = gt_cam[j]
            if pc[2] <= 1e-6 or not np.isfinite(cov).all():
                continue
            var_z = float(cov[2, 2])
            if var_z <= 0:
                continue
            err = est - pc
            z_score = err[2] / np.sqrt(var_z)
            nees3 = finite_mahalanobis(err, cov)
            xy_marg, xy_cond, pix_nees, nees3_parts = nees_decomposition(err, cov, est, pc, f)

            gp, rng, _, _pc = project_world_fast(Xw[j], T_bw, f, cx, cy)
            if depth is None:
                depth = ds.depth(i)
            gx, gy = int(round(gp[0])), int(round(gp[1]))
            dwin = depth[max(0, gy - R):gy + R + 1, max(0, gx - R):gx + R + 1]
            dwin = dwin[dwin < md.SKY]
            dstd = float(np.std(dwin)) if dwin.size > 4 else np.nan
            map_rng = float(depth[gy, gx])
            occluded = int(map_rng < rng - max(0.05 * rng, 0.5))
            border = int(gp[0] < args.border or gp[0] >= W - args.border
                         or gp[1] < args.border or gp[1] >= H - args.border
                         or pos_prev[j][0] < args.border or pos_prev[j][0] >= W - args.border
                         or pos_prev[j][1] < args.border or pos_prev[j][1] >= H - args.border)
            inlier_ratio = fd.get("inlier_ratio", np.nan)
            nis_value = fd.get("nis", np.nan)
            nis = float(nis_value) if np.isfinite(nis_value) else np.nan
            rows.append((
                i - born[j], track_len, z_score, z_score * z_score, nees3,
                xy_marg, xy_cond, pix_nees, nees3_parts,
                err[2] / pc[2], np.linalg.norm(err) / np.linalg.norm(pc),
                drift_px[j], dstd, occluded, border, pc[2], float(inlier_ratio), nis))

        if args.measurements == "klt":
            prev = (cur_pyr, cgx, cgy)
        if (i - args.start) % 50 == 0:
            fps = (i - args.start + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i - args.start:4d}/{last - args.start}] active={len(pos_prev):4d} "
                  f"scored={len(rows):7d} {fps:5.1f} fps")

    if writer is not None:
        writer.release()
        print(f"saved video: {vpath}")

    if args.video_only:
        print(f"\n=== Mid-Air measurement overlay ({len(seen_ids)} tracks) ===")
        print(f"measurements:              {args.measurements}")
        print("Sparse3D update/scoring skipped by --video-only.")
        return

    A = np.array(rows, float)
    print(f"\n=== Mid-Air Sparse3D exact-GT consistency "
          f"({len(A)} scored obs, {len(seen_ids)} tracks, {n_live_updates} updates) ===")
    print(f"measurements:              {args.measurements}")
    if len(A) == 0:
        print("No scored observations. Lower --min-track, increase --frames, or check dataset path.")
        return

    (age, tl, z, z2, nees3, xy_marg, xy_cond, pix_nees, nees3_parts,
     relz, rel3, drift, dstd, occ, border, depth_z, inlier, nis) = A.T
    print(f"depth NEES-1D mean/median: {np.mean(z2):8.2f} / {np.median(z2):8.2f}  (ideal 1)")
    print(f"3D NEES mean/median:       {np.nanmean(nees3):8.2f} / {np.nanmedian(nees3):8.2f}  (ideal 3)")
    print(f"3D split mean/median:      z {np.nanmean(z2):8.2f} / {np.nanmedian(z2):8.2f}  "
          f"xy|z {np.nanmean(xy_cond):8.2f} / {np.nanmedian(xy_cond):8.2f}  "
          f"(ideal 1 + 2)")
    print(f"XY marginal mean/median:   {np.nanmean(xy_marg):8.2f} / {np.nanmedian(xy_marg):8.2f}  "
          f"(diagnostic only)")
    pix_f = pix_nees[np.isfinite(pix_nees)]
    if len(pix_f):
        print(f"pixel-plane NEES mean/med: {np.mean(pix_f):8.2f} / {np.median(pix_f):8.2f}  "
              f"(ideal 2 if linearized projection matches)")
    decomp_resid = nees3 - nees3_parts
    decomp_f = decomp_resid[np.isfinite(decomp_resid)]
    if len(decomp_f):
        print(f"3D split check |full-(z+xy|z)|: "
              f"median {np.median(np.abs(decomp_f)):.2e}  p99 {np.percentile(np.abs(decomp_f), 99):.2e}")
    print(f"|z|<1/<2/<3:               {np.mean(np.abs(z) < 1)*100:5.1f}% / "
          f"{np.mean(np.abs(z) < 2)*100:5.1f}% / {np.mean(np.abs(z) < 3)*100:5.1f}%")
    print(f"signed rel depth err:      median {np.median(relz)*100:+6.2f}%  "
          f"mean {np.mean(relz)*100:+6.2f}%")
    print(f"abs 3D rel err:            median {np.median(np.abs(rel3))*100:6.2f}%  "
          f"p90 {np.percentile(np.abs(rel3), 90)*100:6.2f}%")
    print(f"measurement drift vs GT:   median {np.median(drift):6.3f}px  "
          f"p90 {np.percentile(drift, 90):6.3f}px")
    print(f"occluded obs:              {np.mean(occ == 1)*100:5.1f}%")
    print(f"border-risk obs:           {np.mean(border == 1)*100:5.1f}%")
    nis_f = nis[np.isfinite(nis)]
    if len(nis_f):
        print(f"NIS chi2(2):               mean {np.mean(nis_f):6.2f}  "
              f"median {np.median(nis_f):6.2f}  p95 {np.percentile(nis_f, 95):6.2f}")

    print("\n--- depth NEES-1D by track length ---")
    for lo, hi in [(10, 20), (20, 40), (40, 80), (80, 160), (160, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if m.any():
            label = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"
            print_bin_stats("track_len", z2, m, label)

    print("\n--- full 3D NEES split by track length ---")
    for lo, hi in [(10, 20), (20, 40), (40, 80), (80, 160), (160, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if m.any():
            label = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"
            print_bin_stats("z", z2, m, label)
            print_bin_stats("xy|z", xy_cond, m, label)
            print_bin_stats("full3", nees3, m, label)

    print("\n--- depth NEES-1D by measurement drift and scene covariates ---")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]:
        m = (drift >= lo) & (drift < hi)
        print_bin_stats("drift_px", z2, m, f"{lo:g}-{hi:g}")
    print_bin_stats("occlusion", z2, occ == 0, "visible")
    print_bin_stats("occlusion", z2, occ == 1, "occluded")
    print_bin_stats("border", z2, border == 0, "interior")
    print_bin_stats("border", z2, border == 1, "border")
    finite_dstd = np.isfinite(dstd)
    if finite_dstd.sum() > 100:
        q25, q75 = np.percentile(dstd[finite_dstd], [25, 75])
        print_bin_stats("depth_std", z2, finite_dstd & (dstd < q25), "low edge")
        print_bin_stats("depth_std", z2, finite_dstd & (dstd >= q25) & (dstd < q75), "mid")
        print_bin_stats("depth_std", z2, finite_dstd & (dstd >= q75), "high edge")
        clean = (occ == 0) & (border == 0) & finite_dstd & (dstd < q75)
        print("\n--- gated valid-domain NEES ---")
        print_bin_stats("valid", z2, clean, "vis/int/lowmid")
        print_bin_stats("valid xy|z", xy_cond, clean, "vis/int/lowmid")
        print_bin_stats("valid full3", nees3, clean, "vis/int/lowmid")
        clean_core = clean & (drift <= 3.0)
        print_bin_stats("valid", z2, clean_core, "drift<=3px")
        print_bin_stats("valid xy|z", xy_cond, clean_core, "drift<=3px")
        print_bin_stats("valid full3", nees3, clean_core, "drift<=3px")

    print("\n--- NEES mass concentration ---")
    order = np.argsort(z2)[::-1]
    total = np.sum(z2)
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(frac * len(z2)))
        print(f"top {frac*100:4.0f}% obs carry {np.sum(z2[order[:k]]) / total * 100:5.1f}% "
              f"of depth NEES mass; median age {np.median(age[order[:k]]):.0f}, "
              f"median drift {np.median(drift[order[:k]]):.2f}px")
    order3 = np.argsort(nees3)[::-1]
    total3 = np.nansum(nees3)
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(frac * len(nees3)))
        top = order3[:k]
        print(f"top {frac*100:4.0f}% obs carry {np.nansum(nees3[top]) / total3 * 100:5.1f}% "
              f"of 3D NEES mass; median z {np.nanmedian(z2[top]):.2f}, "
              f"median xy|z {np.nanmedian(xy_cond[top]):.2f}, "
              f"median drift {np.median(drift[top]):.2f}px")


if __name__ == "__main__":
    main()
