"""Run the core EqVIO on Mid-Air (the deferred end-to-end test). Drives the VIO exactly like
rot_odom_diag.py but builds the IMU+image event stream from Mid-Air's sensor_records.hdf5
(noisy imu/*, 100 Hz) and JPEG frames (25 Hz), with t_bs = RT_BC. Records estimated vs GT
poses; reports ATE after SE(3)/Umeyama alignment as the first sanity milestone (does the VIO
track, are the conventions right?) before the Sparse3D depth-NEES step.

  PY=echo-li-python/venv/bin/python
  $PY midair_vio_run.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 200 --config configs/eqvio_euroc_rho.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import run_manifest  # noqa: E402
import echo_li  # noqa: E402


def umeyama(src, dst):
    """SE(3) (no scale) aligning src->dst; returns R,t and aligned-RMSE."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; t = mu_d - R @ mu_s
    a = (R @ src.T).T + t
    return R, t, float(np.sqrt(((a - dst) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--stereo", action="store_true",
                    help="drive full stereo VIO: per-frame left-right range priors fed via "
                    "process_vision_with_depth_priors (metric-scale observable).")
    ap.add_argument("--stereo-baseline-m", type=float, default=1.0)
    ap.add_argument("--stereo-sigma-pixel-scale", type=float, default=20.0)
    ap.add_argument("--eqf-max-obs", type=int, default=0,
                    help="cap features fed to the EqF, prioritizing existing landmarks "
                    "(matches prior_ab when >0). <=0 feeds all (default; measured better here).")
    ap.add_argument("--true-depth-seed", action="store_true",
                    help="seed new landmarks with GT range from the depth map instead of the "
                    "median-depth pin (birth-only prior). Skips occluded/depth-edge pixels "
                    "(5x5 range spread >15%% of range) so occluders don't inject wrong depth.")
    ap.add_argument("--true-depth-edge-frac", type=float, default=0.15)
    ap.add_argument("--seed-scale", type=float, default=1.0,
                    help="multiply every true-depth seed by this constant: a CONSISTENT but "
                    "wrong-scale seed set (error lies in the unobservable scale mode).")
    ap.add_argument("--seed-lognormal", type=float, default=0.0,
                    help="per-landmark lognormal(0,sigma) multiplier on the true-depth seed: "
                    "median-preserving but MUTUALLY INCONSISTENT seeds (error has an "
                    "observable-subspace component). Reported prior variance stays at 2%%, "
                    "mirroring Sparse3D reporting ~5%% rel-sigma while being far off.")
    ap.add_argument("--probe-scale", action="store_true",
                    help="per-frame stage decomposition of the SCALE error: how much of\n                    log(|v_est|/|v_gt|) is moved by IMU propagation vs by the vision update.")
    ap.add_argument("--track-lifetimes", action="store_true",
                    help="histogram in-state landmark lifetimes and filter occupancy: does a\n                    landmark live long enough to accumulate parallax and converge in depth?")
    ap.add_argument("--extrinsic", default="rtbc", choices=["rtbc", "inv", "identity", "rtbc_T"])
    ap.add_argument("--ext-euler", default="", help="rx,ry,rz deg: camera-frame mounting "
                    "rotation post-multiplied onto the extrinsic (R_bc @ Rz@Ry@Rx)")
    ap.add_argument("--gyro-frame", default="repaired_gt",
                    choices=["body", "spatial_est", "repaired_gt", "spatial_gt", "world", "world_gt"],
                    help="How to feed MidAir gyro samples to EqF. 'body' trusts the HDF5 "
                    "metadata. 'spatial_est'/'world' treats the released channel as "
                    "spatial/world and rotates it with the current estimate. "
                    "'repaired_gt'/'spatial_gt'/'world_gt' uses MidAir GT attitude to "
                    "repair the released spatial channel into the body-frame gyro a real "
                    "IMU should have provided.")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    nimg = ds.n - args.start if args.frames <= 0 else min(args.frames, ds.n - args.start)
    print(f"Mid-Air {args.cond}/{ds.traj}  {W}x{H} f={f:.1f}  frames={nimg}")
    if args.gyro_frame in ("repaired_gt", "spatial_gt", "world_gt"):
        print("gyro-frame: repaired_gt uses MidAir GT attitude to repair the released "
              "spatial gyro into the body-frame gyro that a real IMU should provide.")
    elif args.gyro_frame in ("spatial_est", "world"):
        print("gyro-frame: spatial_est repairs MidAir's spatial gyro with estimated attitude; "
              "this can feed attitude error back into the IMU adapter.")

    # IMU (noisy) at 100 Hz; camera at 25 Hz -> imu index = 4*k
    imu = ds.db[ds.traj]["imu"]
    accel = imu["accelerometer"][:]; gyro = imu["gyroscope"][:]
    imu0 = args.start * 4
    imu1 = min(len(accel), (args.start + nimg) * 4 + 4)
    imu_ev = [(i / 100.0, "imu", (gyro[i].tolist(), accel[i].tolist())) for i in range(imu0, imu1)]
    img_ev = [(k / 25.0, "img", k) for k in range(args.start, args.start + nimg)]
    events = sorted(imu_ev + img_ev, key=lambda e: e[0])

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    vio = echo_li.VIOFilter(args.config, cam)
    stereo = None
    if args.stereo:
        stereo = echo_li.Stereo.from_pinhole(
            f, f, cx, cy, W, H, [-args.stereo_baseline_m, 0.0, 0.0], None, args.config)
        print(f"stereo: rectified pinhole baseline={args.stereo_baseline_m:.3f}m "
              f"sigma_pixel_scale={args.stereo_sigma_pixel_scale:g}")

    def right_gray(k):
        p = ds.dir / "color_right" / ds.traj / f"{k:06d}.JPEG"
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if args.scale != 1.0:
            im = cv2.resize(im, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        return im
    ext = {"rtbc": md.RT_BC, "inv": np.linalg.inv(md.RT_BC),
           "rtbc_T": md.RT_BC.T, "identity": np.eye(4)}[args.extrinsic]
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    print(f"extrinsic={args.extrinsic}\n{ext[:3,:3]}")
    # Mid-Air VO_test starts mid-flight (~7 m/s); the stationary auto-init fails, so seed the
    # initial state from GT. Mid-Air is NED (Z down); the filter's world is Z-up (gravity along
    # -Z), so map NED->NWU (T=diag(1,-1,-1), a 180deg rotation about X) so gravity is consistent.
    T = np.diag([1.0, -1.0, -1.0])
    gt0 = ds.pose(args.start)
    v0 = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][args.start * 4])
    R0 = T @ gt0[:3, :3]
    v0_body = R0.T @ (T @ v0)
    vio.set_initial_state((T @ gt0[:3, 3]).tolist(), np.ascontiguousarray(R0),
                          v0_body.tolist())

    seed_rng = np.random.default_rng(0)  # reproducible seed perturbation
    lm_birth = {}; lm_life = []; lm_occ = []; conv = []; probe = []
    rec = []; n = 0; t0 = time.time()
    td_seeded = 0; td_occ_skipped = 0  # true-depth-seed stats
    for stamp, et, data in events:
        if et == "imu":
            gyro_s = data[0]
            if args.gyro_frame in ("spatial_est", "world"):
                # Empirical MidAir adapter: metadata says local/body, but the stored channel
                # matches spatial/world attitude finite differences on VO_test/sunny.
                # Rotate into body with the current attitude estimate (body->NWU).
                _, q = vio.get_pose()
                R_est = Rot.from_quat(np.asarray(q)).as_matrix()
                gyro_s = (R_est.T @ (T @ np.asarray(data[0]))).tolist()
            elif args.gyro_frame in ("repaired_gt", "spatial_gt", "world_gt"):
                # MidAir dataset repair: rotate the released spatial gyro into the
                # body-frame gyro that a real IMU would directly measure.
                gi = min(int(round(stamp * 100.0)), len(ds.att) - 1)
                q = ds.att[gi]
                R_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                gyro_s = (R_gt.T @ np.asarray(data[0])).tolist()
            vio.process_imu(stamp, gyro_s, data[1])
            continue
        k = data
        gray = md.to_u8(ds.image(k)) if hasattr(md, "to_u8") else np.asarray(ds.image(k)).astype(np.uint8)
        feats, stats = tracker.process(gray)
        n += 1
        if not vio.is_initialized:
            continue
        uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
        if args.probe_scale:
            _gv = ds.db[ds.traj]["groundtruth"]["velocity"]
            vg_n = float(np.linalg.norm(_gv[min(k * 4, len(_gv) - 1)]))
            v_pre = float(np.linalg.norm(np.asarray(vio.get_velocity())))
        # Match prior_ab: cap EqF observations to its landmark budget, keeping
        # existing landmarks first (continuity). Feeding all ~300 frontend
        # features into a 40-landmark EqF churns landmarks and degrades tracking.
        if args.eqf_max_obs > 0 and len(uvs) > args.eqf_max_obs:
            existing = {int(x) for x in vio.get_landmarks().keys()}
            ordered = list(uvs)
            keep = ([f for f in ordered if f in existing] +
                    [f for f in ordered if f not in existing])[:args.eqf_max_obs]
            uvs = {f: uvs[f] for f in keep}
        if stereo is not None and uvs:
            priors = dict(stereo.range_priors(right_gray(k), tracker, args.stereo_sigma_pixel_scale))
            vio.process_vision_with_depth_priors(stamp, uvs, priors)
        elif args.true_depth_seed and uvs:
            existing = {int(x) for x in vio.get_landmarks().keys()}
            dmap = ds.depth(k); hh, ww = dmap.shape
            priors = {}
            for fid, (u, v) in uvs.items():
                if fid in existing:      # only seed births; tracked landmarks ignore priors
                    continue
                gx, gy = int(round(u)), int(round(v))
                if not (2 <= gx < ww - 2 and 2 <= gy < hh - 2):
                    continue
                r = float(dmap[gy, gx])
                if not (1.0 < r < md.SKY):
                    continue
                patch = dmap[gy - 2:gy + 3, gx - 2:gx + 3]
                pv = patch[(patch > 1.0) & (patch < md.SKY)]
                if pv.size < 9 or (pv.max() - pv.min()) / max(r, 1e-3) > args.true_depth_edge_frac:
                    td_occ_skipped += 1  # depth-edge / occlusion boundary -> don't seed true depth
                    continue
                rp = r * args.seed_scale
                if args.seed_lognormal > 0.0:
                    rp *= float(np.exp(seed_rng.normal(0.0, args.seed_lognormal)))
                priors[fid] = (rp, (0.02 * rp) ** 2)
            td_seeded += len(priors)
            vio.process_vision_with_depth_priors(stamp, uvs, priors)
        else:
            vio.process_vision(stamp, uvs)
        if args.track_lifetimes:
            cur = {int(x) for x in vio.get_landmarks().keys()}
            lm_occ.append(len(cur))
            for lid in cur - set(lm_birth):
                lm_birth[lid] = n
            for lid in set(lm_birth) - cur:
                lm_life.append(n - lm_birth.pop(lid))
            if n % 5 == 0:
                lms = vio.get_landmarks()
                pos_b, quat_b = vio.get_pose()
                Rwb = Rot.from_quat(np.asarray(quat_b)).as_matrix()
                cam_w = np.asarray(pos_b) + Rwb @ ext[:3, 3]
                dmap_c = ds.depth(k); hh_c, ww_c = dmap_c.shape
                for lid, pw in lms.items():
                    lid = int(lid)
                    if lid not in uvs:
                        continue
                    u_, v_ = uvs[lid]
                    gx, gy = int(round(u_)), int(round(v_))
                    if not (2 <= gx < ww_c - 2 and 2 <= gy < hh_c - 2):
                        continue
                    gr = float(dmap_c[gy, gx])
                    if not (1.0 < gr < md.SKY):
                        continue
                    pat = dmap_c[gy - 2:gy + 3, gx - 2:gx + 3]
                    pv = pat[(pat > 1.0) & (pat < md.SKY)]
                    if pv.size < 9 or (pv.max() - pv.min()) / max(gr, 1e-3) > 0.15:
                        continue
                    er = float(np.linalg.norm(np.asarray(pw) - cam_w))
                    conv.append((n - lm_birth.get(lid, n), er, gr, n))
        if args.probe_scale:
            v_post = float(np.linalg.norm(np.asarray(vio.get_velocity())))
            gb, ab = vio.get_biases()
            _p, _q = vio.get_pose()
            R_est = Rot.from_quat(np.asarray(_q)).as_matrix()
            qg = ds.att[min(k * 4, len(ds.att) - 1)]
            R_gt = T @ Rot.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
            g_est = R_est.T @ np.array([0.0, 0.0, 1.0])
            g_gt = R_gt.T @ np.array([0.0, 0.0, 1.0])
            tilt = float(np.degrees(np.arccos(np.clip(g_est @ g_gt, -1, 1))))
            probe.append((k, vg_n, v_pre, v_post,
                          float(np.linalg.norm(np.asarray(ab))),
                          float(np.linalg.norm(np.asarray(gb))), tilt))
        pos, quat = vio.get_pose()
        pcov = vio.get_camera_pose_covariance()
        pvv, pww = (np.zeros((3, 3)), np.zeros((3, 3))) if pcov is None else \
            (np.asarray(pcov[0]), np.asarray(pcov[1]))
        gt = ds.pose(k)
        gt_pos_nwu = T @ gt[:3, 3]
        rec.append((k, np.asarray(pos), np.asarray(quat), gt_pos_nwu.copy(), stats["tracked"], pvv, pww))
        if n % 100 == 0:
            print(f"  [{n}/{nimg}] t={stamp:5.1f}s tracked={stats['tracked']:3d} "
                  f"|est|={np.linalg.norm(pos):5.1f} |gt|={np.linalg.norm(gt_pos_nwu):5.1f} "
                  f"{n/(time.time()-t0):.0f}fps")

    if len(rec) < 20:
        print(f"\nVIO produced only {len(rec)} poses (init failed or diverged?)"); return
    est = np.array([r[1] for r in rec]); gtp = np.array([r[3] for r in rec])
    R, t, ate = umeyama(est, gtp)
    traj_len = float(np.linalg.norm(np.diff(gtp, axis=0), axis=1).sum())
    print(f"\n=== {len(rec)} VIO poses ===")
    print(f"GT path length {traj_len:.1f} m over {len(rec)} frames")
    print(f"ATE (SE3-aligned RMSE): {ate:.3f} m   =  {100*ate/max(traj_len,1e-6):.1f}% of path")
    print(f"final drift: est vs gt (aligned) = {np.linalg.norm((R@est[-1]+t)-gtp[-1]):.3f} m")
    if args.true_depth_seed:
        tot = td_seeded + td_occ_skipped
        print(f"true-depth-seed: {td_seeded} births seeded, {td_occ_skipped} skipped as "
              f"occlusion/depth-edge ({100*td_occ_skipped/max(tot,1):.1f}% of birth candidates)")
    if args.probe_scale and len(probe) > 5:
        P = np.array(probe, float)
        kk, vg, vpre, vpost, ab, gb, tilt = P.T
        ok = vg > 2.0
        r_pre, r_post = np.log(vpre[ok] / vg[ok]), np.log(vpost[ok] / vg[ok])
        d_upd = r_post - r_pre                       # moved BY the vision update
        d_prop = r_pre[1:] - r_post[:-1]             # moved BY IMU propagation between frames
        print("\n=== SCALE DRIFT STAGE DECOMPOSITION  (log |v_est|/|v_gt|) ===")
        print(f"  start log-scale {r_post[0]:+.3f}   end {r_post[-1]:+.3f}   NET drift {r_post[-1]-r_post[0]:+.3f}")
        print(f"  sum d_update (vision) = {d_upd.sum():+8.2f}   mean/frame {d_upd.mean():+.5f}")
        print(f"  sum d_propag (IMU)    = {d_prop.sum():+8.2f}   mean/frame {d_prop.mean():+.5f}")
        tot = abs(d_upd.sum()) + abs(d_prop.sum())
        print(f"  => vision accounts for {100*abs(d_upd.sum())/max(tot,1e-9):.1f}% of |motion|, "
              f"IMU {100*abs(d_prop.sum())/max(tot,1e-9):.1f}%")
        print(f"  accel-bias |b_a|: start {ab[0]:.4f} end {ab[-1]:.4f} max {ab.max():.4f} (MidAir true ~4e-4)")
        print(f"  gyro-bias  |b_g|: start {gb[0]:.5f} end {gb[-1]:.5f} max {gb.max():.5f}")
        print(f"  gravity-dir (tilt) err deg: median {np.median(tilt):.2f} p90 {np.percentile(tilt,90):.2f} max {tilt.max():.2f}")
        print(f"  corr(|b_a|, log-scale) = {np.corrcoef(ab[ok], r_post)[0,1]:+.2f}   "
              f"corr(tilt, log-scale) = {np.corrcoef(tilt[ok], r_post)[0,1]:+.2f}")
        # --- is the propagation drag explained by gravity leakage from attitude tilt? ---
        tl = tilt[ok]; vgo = vg[ok]
        dprop = d_prop                      # per-frame propagation-induced log-scale change
        tl_p = tl[1:]                       # align with dprop
        vg_p = vgo[1:]
        print("\n  --- gravity-leakage test (propagation stage only) ---")
        print(f"  corr(tilt, d_prop)            = {np.corrcoef(tl_p, dprop)[0,1]:+.3f}"
              f"   (negative => more tilt, more downward drag)")
        # magnitude: leaked accel g*sin(tilt) over dt, relative to speed
        dt_f = 1.0 / 25.0
        pred = 9.81 * np.sin(np.radians(tl_p)) * dt_f / np.maximum(vg_p, 1e-6)
        print(f"  |d_prop| observed median      = {np.median(np.abs(dprop)):.5f} /frame")
        print(f"  g*sin(tilt)*dt/|v| predicted  = {np.median(pred):.5f} /frame"
              f"   ratio obs/pred = {np.median(np.abs(dprop))/max(np.median(pred),1e-12):.2f}")
        # --- causality: lagged cross-correlation of INCREMENTS (detrended) ---
        dt_tilt = np.diff(tl); dsc = np.diff(r_post)
        n = min(len(dt_tilt), len(dsc)); dt_tilt, dsc = dt_tilt[:n], dsc[:n]
        best = []
        for L in range(-30, 31, 5):
            if L < 0:   c = np.corrcoef(dt_tilt[-L:], dsc[:L])[0, 1]
            elif L > 0: c = np.corrcoef(dt_tilt[:-L], dsc[L:])[0, 1]
            else:       c = np.corrcoef(dt_tilt, dsc)[0, 1]
            best.append((L, c))
        pk = max(best, key=lambda t: abs(t[1]))
        print(f"  lagged xcorr(d_tilt -> d_logscale) peak at lag {pk[0]:+d} frames, r={pk[1]:+.3f}"
              f"   (lag>0 => TILT LEADS scale)")
        print("   " + "  ".join(f"{L:+d}:{c:+.2f}" for L, c in best))
    if args.track_lifetimes and conv:
        C = np.array(conv, float)
        age, er, gr, fr = C[:,0], C[:,1], C[:,2], C[:,3]
        lr = np.log(er / gr)
        # remove per-frame global scale: subtract that frame's median log-ratio
        norm = np.empty_like(lr)
        for f in np.unique(fr):
            m = fr == f
            norm[m] = lr[m] - np.median(lr[m])
        print("\n=== depth convergence vs landmark age (in-state landmarks vs GT depth) ===")
        print(f"{'age bin':>10} {'n':>7} {'med|log ratio|':>15} {'med ratio':>10} {'scale-norm med|log|':>20}")
        for lo, hi in [(0,2),(3,5),(6,10),(11,20),(21,50),(51,1000)]:
            m = (age >= lo) & (age <= hi)
            if m.sum() < 20: continue
            print(f"{str(lo)+'-'+str(hi):>10} {m.sum():7d} {np.median(np.abs(lr[m])):15.3f} "
                  f"{np.median(er[m]/gr[m]):10.3f} {np.median(np.abs(norm[m])):20.3f}")
        print(f"  corr(log age, |log ratio|) = {np.corrcoef(np.log(age+1), np.abs(lr))[0,1]:+.3f}")
        print(f"  corr(log age, scale-norm |log ratio|) = {np.corrcoef(np.log(age+1), np.abs(norm))[0,1]:+.3f}")
    if args.track_lifetimes:
        lifes = np.array(lm_life + [n - b for b in lm_birth.values()], float)
        occ = np.array(lm_occ, float)
        print(f"\n=== landmark lifetimes (frames in EqF state) ===")
        print(f"  n_tracks={len(lifes)}  mean={lifes.mean():.2f}  median={np.median(lifes):.1f}  "
              f"p90={np.percentile(lifes,90):.1f}  p99={np.percentile(lifes,99):.1f}  max={lifes.max():.0f}")
        print(f"  frac lifetime<=2: {100*np.mean(lifes<=2):.1f}%   <=5: {100*np.mean(lifes<=5):.1f}%   "
              f">=20: {100*np.mean(lifes>=20):.1f}%")
        print(f"  occupancy: mean={occ.mean():.1f} median={np.median(occ):.0f} max={occ.max():.0f} "
              f"(cap from config)")
        print(f"  births/frame={len(lifes)/max(len(occ),1):.1f}")
    if args.save_npz:
        np.savez(args.save_npz, k=[r[0] for r in rec], est=est, quat=[r[2] for r in rec],
                 gt=gtp, R=R, t=t, ate=ate, tracked=[r[4] for r in rec],
                 pvv=np.array([r[5] for r in rec]), pww=np.array([r[6] for r in rec]))
        print("saved ->", args.save_npz)
        run_manifest.save_run_manifest(args.save_npz, args.config, extra={
            "traj": args.traj, "frames": nimg, "scale": args.scale,
            "stereo": args.stereo, "eqf_max_obs": args.eqf_max_obs,
            "true_depth_seed": args.true_depth_seed,
            "gyro_frame": args.gyro_frame, "extrinsic": args.extrinsic,
            "ate_m": round(ate, 3), "ate_pct": round(100 * ate / max(traj_len, 1e-6), 3)})


if __name__ == "__main__":
    main()
