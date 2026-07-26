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

    rec = []; n = 0; t0 = time.time()
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
        else:
            vio.process_vision(stamp, uvs)
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
    if args.save_npz:
        np.savez(args.save_npz, k=[r[0] for r in rec], est=est, quat=[r[2] for r in rec],
                 gt=gtp, R=R, t=t, ate=ate, tracked=[r[4] for r in rec],
                 pvv=np.array([r[5] for r in rec]), pww=np.array([r[6] for r in rec]))
        print("saved ->", args.save_npz)
        run_manifest.save_run_manifest(args.save_npz, args.config, extra={
            "traj": args.traj, "frames": nimg, "scale": args.scale,
            "stereo": args.stereo, "eqf_max_obs": args.eqf_max_obs,
            "gyro_frame": args.gyro_frame, "extrinsic": args.extrinsic,
            "ate_m": round(ate, 3), "ate_pct": round(100 * ate / max(traj_len, 1e-6), 3)})


if __name__ == "__main__":
    main()
