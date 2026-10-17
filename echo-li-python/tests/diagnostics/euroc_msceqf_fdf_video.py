"""MSCEqF (the shipping VIOFilter backend, ZUPT, static init) + FlowDep-F + Sparse3D on a EuRoC
Vicon-room sequence, scored against the Leica scan, rendered as a high-quality video.

Pose source: the backend's own pose, fed by Rudolf-V on the full-resolution distorted cam0
frame with the RadTan camera (exactly the node's pipeline); FlowDep-F runs on the undistorted
pinhole frame at --scale with that pose, and its Sparse3D is driven through the backend's
clone-relative `sparse_feed`. GT range = radial z-buffer of `pointcloud0/data.ply` through the
fused Vicon+IMU pose at the frame (the true camera, so the depth error is defined in the
physical camera frame whatever the estimated pose is).

Video, 3x2 panels at 2x the working size:
  undistorted frame + Rudolf-V tracks | dense range (measured full, filled paler) | Leica GT range
  dense signed error vs GT            | per-pixel sigma                           | trajectory est (SE3-aligned) vs GT
Written as H.264 at --crf (default 16) to --out, plus a _config.yaml and a .npz of the scores.

Usage:
  .venv/bin/python echo-li-python/tests/diagnostics/euroc_msceqf_fdf_video.py \
      --root ~/18TB/datasets/EuRoC/vicon_room1/V1_01_easy/V1_01_easy --cache $SCRATCH/euroc \
      --out /mnt/18TB/chen_fu_yeh/echo-li-videos/euroc/V1_01_easy.mp4
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
import fdf_densify_study as dz  # noqa: E402
from euroc_fdf_depth import load_cam, load_cloud, load_gt, zbuffer  # noqa: E402
from tartanair_e2e_depth_nees import SPARSE_KEYS  # noqa: E402

GRAVITY = 9.81


def umeyama_se3(est, gt):
    mu_e, mu_g = est.mean(0), gt.mean(0)
    Hm = (est - mu_e).T @ (gt - mu_g)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mu_g - R @ mu_e


def txt(img, s, org, col=(235, 235, 235), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "configs/eqvio_euroc_rho.yaml"))
    ap.add_argument("--sparse-config", default=str(Path(__file__).resolve().parents[3] / "configs/diagnostics_midair_e2e_depth_nees.yaml"),
                    help="SparseVog section for the FlowDep-F filter (the Kite-validated settings)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-slam", type=int, default=50)
    ap.add_argument("--frontend-features", type=int, default=120)
    ap.add_argument("--fdf", action="append", default=["max_mean_abs_residual=30"], metavar="KEY=VAL")
    ap.add_argument("--fdf-sparse", action="append", default=["bias_walk_var=0.3"], metavar="KEY=VAL")
    ap.add_argument("--slant-deg", type=float, default=35.0)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--min-track", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    gt_t, gt_p, gt_R = load_gt(root)
    T_BS, (fu, fv, cu, cv_), D, (W0, H0), cam_t, cam_names = load_cam(root)
    imu = np.loadtxt(root / "mav0/imu0/data.csv", delimiter=",", skiprows=1)
    imu[:, 0] *= 1e-9
    from scipy.spatial.transform import Slerp
    slerp = Slerp(gt_t, gt_R)

    def gt_pose_at(t):
        t = float(np.clip(t, gt_t[0], gt_t[-1]))
        T = np.eye(4); T[:3, :3] = slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, gt_t, gt_p[:, i]) for i in range(3)]
        return T @ T_BS

    s = args.scale
    W, H = int(round(W0 * s)), int(round(H0 * s))
    f = float(0.5 * (fu + fv) * s); cx = float(cu * s); cy = float(cv_ * s)
    K0 = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], float)
    K1 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    map1, map2 = cv2.initUndistortRectifyMap(K0, D, np.eye(3), K1, (W, H), cv2.CV_32FC1)
    P_w = load_cloud(root, args.cache)

    # --- the shipping VIO: VIOFilter + MSCEqF section + ZUPT, Rudolf-V on the distorted full frame ---
    full_cfg = yaml.safe_load(open(args.config))
    full_cfg["MSCEqF"] = {
        "enabled": True, "numClones": args.num_clones, "pixelStd": 1.0, "chi2Mult": 1.0,
        "maxSlam": args.max_slam, "maxPromoteRange": 0.0, "directProductBias": False,
        "cloneAtFrameStamp": False, "civeraBirthDepth": 0.0, "civeraBirthRelStd": 1.0,
        "gravity": [0.0, 0.0, -GRAVITY],
        "initialStd": {"attitude": 1e-2, "velocity": 1e-1, "position": 1e-4, "bias": 1e-2, "extrinsics": 1e-4},
        # EuRoC ADIS16448 densities, as the native ATE harness uses them.
        "processNoise": {"accelDensity": 2.0e-3, "gyroDensity": 1.6968e-4,
                         "accelRandomWalk": 3.0e-3, "gyroRandomWalk": 1.9393e-5, "transitionOrder": 0},
    }
    full_cfg["ZUPT"] = {"enabled": True, "window": 25, "gyroStdThreshold": 0.05, "accStdThreshold": 0.15, "sigmaV": 1e-2}
    run_cfg = str(out.with_suffix("")) + "_config.yaml"
    with open(run_cfg, "w") as fh:
        yaml.safe_dump(full_cfg, fh, sort_keys=False)
    cam_full = echo_li.RadTanCamera(fu, fv, cu, cv_, *D[:4])
    fcfg = echo_li.FrontendConfig.from_yaml(run_cfg)
    fcfg.set_camera(fu, fv, cu, cv_, W0, H0, list(D[:4]), "radtan")
    fcfg.max_features = args.frontend_features
    tracker = echo_li.Frontend(fcfg, W0, H0)
    vio = echo_li.VIOFilter(run_cfg, cam_full, fx=fu)
    vio.set_camera_extrinsics(np.ascontiguousarray(T_BS))
    print(f"backend={vio.backend_kind}  ZUPT on  clones {args.num_clones}  maxSlam {args.max_slam}  config {run_cfg}")

    # --- FlowDep-F + its Sparse3D on the undistorted pinhole frame at the working scale ---
    scfg = (yaml.safe_load(open(args.sparse_config)) or {}).get("SparseVog", {}) or {}
    settings = {k: scfg[k] for k in SPARSE_KEYS if k in scfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings["max_pool_size"] = 4000
    settings["min_track_length"] = args.min_track
    for kv in args.fdf_sparse:
        k, v = kv.split("=", 1)
        cur = settings.get(k, scfg.get(k, 0.0))
        settings[k] = (v.lower() == "true") if isinstance(cur, bool) else type(cur)(v) if isinstance(cur, (int, float)) else float(v)
    # sigma_s(E) from the IMU excitation, the operational rule of the Kite and rig harnesses.
    _E = float(np.std(imu[:, 3]) * np.std(imu[:, 5]))
    settings["scale_sigma"] = max(0.02, 0.0072 * _E ** (-0.710)) if _E > 0 else 0.02
    fdf_settings = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
    fdf = echo_li.FlowDepF(f, f, cx, cy, W, H, fdf_settings or None)
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(echo_li.PinholeCamera(f, f, cx, cy), **settings)
    dz.SLANT_G[0] = np.tan(np.radians(args.slant_deg)) / f
    print(f"FlowDep-F {fdf_settings}; Sparse3D pinhole pool 4000 min_track {args.min_track} walk {settings.get('bias_walk_var')} scale_sigma {settings['scale_sigma']:.4f}")

    # --- init: replay the 100 IMU samples before the first frame (the node's StationaryInit) ---
    k0 = args.start
    n = len(cam_t) - k0 if args.frames <= 0 else min(args.frames, len(cam_t) - k0)
    i_start = int(np.searchsorted(imu[:, 0], cam_t[k0]))
    i0 = max(100, i_start)
    for j in range(i0 - 100, i0):
        vio.process_imu(float(imu[j, 0]), imu[j, 1:4].tolist(), imu[j, 4:7].tolist())
    ii = i0

    # --- video ---
    SC = 2
    PW, PH = W * SC, H * SC
    raw_out = str(out.with_suffix("")) + "_mp4v.mp4"
    vw = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (3 * PW, 2 * PH))
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)[:, 0, :]
    lo, hi = np.log(0.8), np.log(12.0)

    def range_panel(r, valid, base):
        t = np.clip((np.log(np.clip(np.nan_to_num(r, nan=1.0), 1e-3, None)) - lo) / (hi - lo), 0, 1)
        c = jet[((1.0 - t) * 255).astype(np.uint8)]
        pnl = base.copy(); pnl[valid] = c[valid]
        return pnl

    est_p, gt_pp = [], []
    rows = []; dense_err = []; dense_nees = []; cover = []
    t0 = time.time()
    for i in range(n):
        k = k0 + i
        ts = float(cam_t[k])
        while ii < len(imu) and imu[ii, 0] <= ts:
            vio.process_imu(float(imu[ii, 0]), imu[ii, 1:4].tolist(), imu[ii, 4:7].tolist()); ii += 1
        raw = cv2.imread(str(root / "mav0/cam0/data" / cam_names[k]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        feats, _ = tracker.process(np.ascontiguousarray(raw))
        uvs = {int(fd["id"]): [float(fd["x"]), float(fd["y"])] for fd in feats}
        vio.process_vision(ts, uvs)
        pp, qq = vio.get_pose()
        T_wb = np.eye(4); T_wb[:3, :3] = Rot.from_quat(np.asarray(qq, float)).as_matrix(); T_wb[:3, 3] = np.asarray(pp, float)
        T_wc = T_wb @ T_BS
        est_p.append(T_wc[:3, 3].copy()); T_gt = gt_pose_at(ts); gt_pp.append(T_gt[:3, 3].copy())

        und = np.ascontiguousarray(cv2.remap(raw, map1, map2, cv2.INTER_LINEAR))
        uv = fdf.process(und, T_wc.tolist())
        vio.sparse_feed(sparse, ts, uv, T_wc.tolist())
        ids, rr, vc, vr, cvg, tl = sparse.range_summary()
        okp = vc > 0
        post = {int(a): (float(b), float(np.sqrt(c) * b)) for a, b, c in zip(ids[okp], rr[okp], vc[okp])}
        if post:
            fdf.set_posterior(post)

        gt = zbuffer(P_w, T_gt, f, cx, cy, W, H, 1)
        vg = np.isfinite(gt) & (gt > 0.3)
        lev = {o[0]: o[3] for o in fdf.last_observations()}
        items = []
        for fid, r_, vc_, vr_, cv1, tl_ in zip(ids, rr, vc, vr, cvg, tl):
            fid = int(fid)
            if not cv1 or vc_ <= 0 or fid not in uv or tl_ < args.min_track:
                continue
            u, v = uv[fid]
            items.append((u, v, lev.get(fid, 0), float(np.log(r_)), float(vc_), float(np.sqrt(max(vr_ / (r_ * r_) - vc_, 0.0)))))
            gx, gy = int(round(u)), int(round(v))
            if 4 < gx < W - 4 and 4 < gy < H - 4 and vg[gy, gx]:
                g = gt[gy, gx]
                rows.append((k, tl_, r_, g, (np.log(r_) - np.log(g)) ** 2 / max(vr_ / (r_ * r_), 1e-12)))
        if items:
            eta_p, var_p, covm = dz.fuse(items, H, W)["nearest_disagree_slant"]
            cls = covm.astype(np.uint8)
            if not args.no_fill and covm.any():
                ef, vf, filled, _ = dz.fill(eta_p, var_p, covm)
                eta_p = np.where(filled, ef, eta_p); var_p = np.where(filled, vf, var_p); cls[filled] = 2
        else:
            eta_p = np.zeros((H, W)); var_p = np.ones((H, W)); cls = np.zeros((H, W), np.uint8)
        covered = cls > 0
        m = covered & vg
        e = eta_p[m] - np.log(gt[m]); nn = e * e / np.maximum(var_p[m], 1e-12)
        dense_err.append((np.exp(e) - 1.0).astype(np.float32)[::3]); dense_nees.append(nn.astype(np.float32)[::3]); cover.append((float((cls == 1).mean()), float((cls == 2).mean())))

        # ---- render ----
        base = cv2.cvtColor(cv2.resize(und, (PW, PH), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        p1 = base.copy()
        for fid, (u, v) in uvs.items():
            cv2.circle(p1, (int(u * s * SC), int(v * s * SC)), 3, (0, 255, 255), 1, cv2.LINE_AA)
        txt(p1, f"EuRoC {root.name}  frame {k}  t+{ts - cam_t[k0]:.1f}s   MSCEqF (ZUPT, {args.max_slam} in-state) + Rudolf-V {len(uvs)} tracks", (8, 22))
        r_big = cv2.resize(np.exp(eta_p).astype(np.float32), (PW, PH), interpolation=cv2.INTER_NEAREST)
        cls_big = cv2.resize(cls, (PW, PH), interpolation=cv2.INTER_NEAREST)
        p2 = base.copy()
        c = range_panel(r_big, cls_big > 0, base)
        mm = cls_big == 1; p2[mm] = (0.25 * base[mm] + 0.75 * c[mm]).astype(np.uint8)
        mm = cls_big == 2; p2[mm] = (0.6 * base[mm] + 0.4 * c[mm]).astype(np.uint8)
        for (u, v, *_r) in items:
            cv2.circle(p2, (int(u * SC), int(v * SC)), 2, (255, 255, 255), 1, cv2.LINE_AA)
        txt(p2, f"FlowDep-F dense range: measured {100 * cover[-1][0]:.0f}% + filled {100 * cover[-1][1]:.0f}% (paler)   {len(items)} converged   0.8 m red .. 12 m blue", (8, 22))
        g_big = cv2.resize(np.nan_to_num(gt, nan=0.0).astype(np.float32), (PW, PH), interpolation=cv2.INTER_NEAREST)
        p3 = range_panel(g_big, g_big > 0.3, (base * 0.3).astype(np.uint8))
        txt(p3, "Leica scan range through the Vicon GT pose", (8, 22))
        err_full = np.zeros((H, W), np.float32); err_full[m] = (np.exp(eta_p[m]) / gt[m] - 1.0)
        e_big = cv2.resize(err_full, (PW, PH), interpolation=cv2.INTER_NEAREST); m_big = cv2.resize(m.astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST) > 0
        t_e = np.clip(e_big / 0.2, -1, 1)
        col = np.zeros((PH, PW, 3), np.uint8)
        col[..., 0] = np.where(t_e < 0, 255, (255 * (1 - t_e)).clip(0, 255)); col[..., 1] = (255 * (1 - np.abs(t_e))).astype(np.uint8); col[..., 2] = np.where(t_e > 0, 255, (255 * (1 + t_e)).clip(0, 255))
        p4 = (base * 0.3).astype(np.uint8); p4[m_big] = col[m_big]
        de = np.concatenate(dense_err) if dense_err else np.zeros(0); dn = np.concatenate(dense_nees) if dense_nees else np.zeros(0)
        txt(p4, f"signed error vs Leica: blue short, red long (sat. 20%)   this frame med|err| {100 * np.median(np.abs(err_full[m])) if m.any() else 0:.1f}%", (8, 22))
        if de.size:
            txt(p4, f"running: med|err| {100 * np.median(np.abs(de)):.1f}%  p90 {100 * np.percentile(np.abs(de), 90):.1f}%  NEES med {np.median(dn):.2f}  >chi2_95 {100 * np.mean(dn > 3.84):.1f}%", (8, 44))
        sig = np.sqrt(np.maximum(var_p, 1e-12)); s_big = cv2.resize(sig.astype(np.float32), (PW, PH), interpolation=cv2.INTER_NEAREST)
        t_s = np.clip((np.log(np.maximum(s_big, 1e-4)) - np.log(0.01)) / (np.log(0.30) - np.log(0.01)), 0, 1)
        p5 = (base * 0.3).astype(np.uint8); cb = cls_big > 0; p5[cb] = jet[((1.0 - t_s) * 255).astype(np.uint8)][cb]
        txt(p5, f"per-pixel sigma (log-range): red 1% .. blue 30%   median {100 * np.median(sig[covered]) if covered.any() else 0:.1f}%", (8, 22))
        p6 = np.full((PH, PW, 3), 24, np.uint8)
        E_ = np.array(est_p); G_ = np.array(gt_pp)
        if len(E_) > 10:
            R_, t_ = umeyama_se3(E_, G_); A_ = E_ @ R_.T + t_
            allp = np.vstack([A_, G_]); mn, mx = allp[:, :2].min(0), allp[:, :2].max(0); span = max((mx - mn).max(), 1e-3) * 1.15; mid = (mx + mn) / 2
            P_ = lambda xy: (int((xy[0] - mid[0]) / span * (PW - 40) + PW / 2), int(-(xy[1] - mid[1]) / span * (PH - 40) + PH / 2))  # noqa: E731
            cv2.polylines(p6, [np.array([P_(g) for g in G_[:, :2]], np.int32)], False, (120, 120, 120), 2, cv2.LINE_AA)
            cv2.polylines(p6, [np.array([P_(a) for a in A_[:, :2]], np.int32)], False, (90, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(p6, P_(A_[-1, :2]), 5, (0, 255, 255), -1, cv2.LINE_AA)
            ate = float(np.sqrt(np.mean(np.sum((A_ - G_) ** 2, 1))))
            txt(p6, f"trajectory (top view): grey GT, orange MSCEqF SE3-aligned   ATE so far {ate:.3f} m over {np.sum(np.linalg.norm(np.diff(G_, axis=0), axis=1)):.1f} m", (8, 22))
        frame = np.vstack([np.hstack([p1, p2, p3]), np.hstack([p4, p5, p6])])
        vw.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] {len(items)} converged, dense cov {100 * (cover[-1][0] + cover[-1][1]):.0f}%, {time.time() - t0:.0f} s", flush=True)
    vw.release()
    rc = subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out, "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
    if rc == 0:
        os.remove(raw_out)
    A = np.array(rows, float)
    if len(A):
        k_, tl, re, g, nees = A.T; rel = re / g - 1.0
        print(f"\nLANDMARKS (converged, tl >= {args.min_track}, MSCEqF pose): {len(A)} obs, abs err median {100 * np.median(np.abs(rel)):.1f}%  p90 {100 * np.percentile(np.abs(rel), 90):.1f}%  signed {100 * np.median(rel):+.1f}%  NEES median {np.median(nees):.3f}  >3.84 {100 * np.mean(nees > 3.84):.1f}%  >6.63 {100 * np.mean(nees > 6.63):.1f}%")
    de = np.concatenate(dense_err) if dense_err else np.zeros(1); dn = np.concatenate(dense_nees) if dense_nees else np.zeros(1); cv_ = np.array(cover) if cover else np.zeros((1, 2))
    print(f"DENSE: coverage measured {100 * cv_[:, 0].mean():.1f}% + filled {100 * cv_[:, 1].mean():.1f}%  err median {100 * np.median(np.abs(de)):.1f}%  p90 {100 * np.percentile(np.abs(de), 90):.1f}%  NEES median {np.median(dn):.3f}  >3.84 {100 * np.mean(dn > 3.84):.1f}%  >6.63 {100 * np.mean(dn > 6.63):.1f}%")
    E_ = np.array(est_p); G_ = np.array(gt_pp); R_, t_ = umeyama_se3(E_, G_); A_ = E_ @ R_.T + t_
    print(f"ATE (SE3-aligned, camera positions): {np.sqrt(np.mean(np.sum((A_ - G_) ** 2, 1))):.3f} m over {np.sum(np.linalg.norm(np.diff(G_, axis=0), axis=1)):.1f} m path")
    np.savez_compressed(str(out.with_suffix("")) + "_scores.npz", rows=A, dense_err=de, dense_nees=dn, cover=cv_, est=E_, gt=G_)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
