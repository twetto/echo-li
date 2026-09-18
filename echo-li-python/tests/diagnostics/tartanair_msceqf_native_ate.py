#!/usr/bin/env python3
"""Drive the MSCEqF-native fork-b filter on TartanAir v2 and report SE3-ATE.

Third dataset after MidAir and EuRoC. CONVENTIONS PINNED EMPIRICALLY (not assumed):
  * gravity `g = [0,0,+9.81]` (NED, z DOWN) — residual `|(acc + R^T g) - a_body|` = 0.010
    vs 19.610 for -9.81.
  * attitude `R = Rx(roll) Ry(pitch) Rz(yaw)` from `imu/ori_global.npy` (body->world) —
    verified `|R^T a_world - a_body|` = 0.0000.
  * `imu/acc.npy` is BODY specific force INCLUDING gravity (what the filter wants);
    `acc_nograv_body.npy` is true body acceleration, used only for those checks.
  * IMU 100 Hz, camera 10 Hz, exactly 10 IMU/frame; `imu_time`/`cam_time` provided.
  * camera 640x640, 90 deg fov => fx=fy=cx=cy=320, NO distortion.
  * GT (`pos_global`/`vel_global`/`ori_global`) is at IMU rate — no interpolation needed.

DAY/NIGHT CONFOUND: OldScandinavia P000 is day/night cyclic — measured frame brightness
swings between ~2/255 and ~150/255. Running the whole sequence measures the renderer's
exposure, not the filter. `--k0/--k1` select a window; daylight windows are roughly
k = 280-360, 640-720, 1000-1080, 1360-1440.

Usage:
  python tests/diagnostics/tartanair_msceqf_native_ate.py --k0 280 --k1 360
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
from echo_li import MSCEqFNativeFilter  # noqa: E402

ROOT = "/mnt/18TB/chen_fu_yeh/datasets/tartanair_v2/OldScandinavia/Data_easy/P000"


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def R_from_euler(e):
    """TartanAir `ori_global` -> body->world. XYZ order, verified to 0.0 residual."""
    return _Rx(e[0]) @ _Ry(e[1]) @ _Rz(e[2])


def align_umeyama(est, gt):
    """SE3 (no-scale) alignment — SAME convention as the MidAir/EuRoC harnesses."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    H = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ est.T).T + (mu_g - R @ mu_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--k0", type=int, default=280, help="first frame (daylight window)")
    ap.add_argument("--k1", type=int, default=360, help="last frame")
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-slam", type=int, default=0,
                    help="in-state SLAM landmarks (0 = structureless only). On a "
                         "high-turnover sequence these extract information from short "
                         "tracks that the structureless path discards.")
    ap.add_argument("--slam-min-len", type=int, default=0,
                    help="min obs before promotion (0 => num_clones)")
    ap.add_argument("--slam-chi2-mult", type=float, default=0.0)
    ap.add_argument("--slam-immediate", action="store_true",
                    help="promote as soon as a track has --slam-min-len observations, "
                         "without waiting for it to reach the marginalized clone. This "
                         "is what turns fork-b from an MSCKF into an EKF-SLAM: the "
                         "structureless path becomes vestigial.")
    ap.add_argument("--chi2-mult", type=float, default=1.0)
    ap.add_argument("--pixel-mult", type=float, default=1.0)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--max-track-len", type=int, default=200)
    ap.add_argument("--curvature", type=int, default=0)
    ap.add_argument("--acc-density", type=float, default=1.6798e-3)
    ap.add_argument("--gyr-density", type=float, default=1.2724e-3)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.0e-4)
    ap.add_argument("--delta-std", type=float, default=1.0e-2)
    ap.add_argument("--frontend-config", default=str(
        Path(__file__).resolve().parents[3] / "configs/eqvio_midair.yaml"))
    ap.add_argument("--no-ransac", action="store_true",
                    help="disable front-end RANSAC: measured to cull ~1/3 of TartanAir "
                         "tracks (mean length 4.1->6.0, %%>=11obs 8%%->14%%)")
    ap.add_argument("--klt-fb", type=float, default=0.0,
                    help="forward-backward KLT consistency threshold (px). OFF by "
                         "default. MEASURED on TartanAir: without it ~half the "
                         "correspondences are FALSE (GT-pose reproj p90 = 42 px); at "
                         "fb=1.0 p90 drops to 1.6 px and %%<3px goes 53%% -> 95%%.")
    ap.add_argument("--lbp", action="store_true", help="descriptor verification")
    ap.add_argument("--stereo", type=int, default=0,
                    help="use image_rcam_front for a metric range channel at landmark "
                         "birth. TartanAir v2 stereo is IDEAL for this: baseline exactly "
                         "0.250 m, inter-camera rotation exactly 0.0 (rectified), no "
                         "distortion, zero spread over the run (verified from "
                         "pose_lcam_front vs pose_rcam_front).")
    ap.add_argument("--baseline", type=float, default=0.25)
    ap.add_argument("--sigma-pixel-scale", type=float, default=20.0)
    ap.add_argument("--klt-window", type=int, default=0)
    ap.add_argument("--pyramid-levels", type=int, default=0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    T = Path(args.root)
    acc = np.load(T / "imu" / "acc.npy")
    gyr = np.load(T / "imu" / "gyro.npy")
    pos = np.load(T / "imu" / "pos_global.npy")
    vel = np.load(T / "imu" / "vel_global.npy")
    ori = np.load(T / "imu" / "ori_global.npy")
    itime = np.load(T / "imu" / "imu_time.npy")
    ctime = np.load(T / "imu" / "cam_time.npy")
    nimg = len(ctime)
    # IMU-per-frame from the RATES, not integer division of the array lengths: the IMU
    # array is one frame short of nimg*10, so `len(itime)//nimg` silently returns 9 —
    # which under-propagates the IMU by 10% every frame AND indexes GT at k*9, scoring
    # against truth that slips a whole frame every 9. Both were live until 2026-09-16.
    IPF = int(round((1.0 / float(np.median(np.diff(itime)))) / 10.0))  # imu_fps / img_fps
    DT = float(np.median(np.diff(itime)))

    k0 = max(0, args.k0)
    k1 = min(nimg - 1, args.k1 if args.k1 > 0 else nimg - 1)
    W = Hh = 640
    fx = fy = cx = cy = 320.0
    print(f"TartanAir {T.parent.parent.name}/{T.name}  frames {k0}..{k1} of {nimg}  "
          f"IMU {len(itime)} @ {1/DT:.0f}Hz  ({IPF}/frame)")

    i0 = k0 * IPF
    R0, p0, v0 = R_from_euler(ori[i0]), pos[i0].copy(), vel[i0].copy()
    b0 = np.zeros(6)
    grav = np.array([0.0, 0.0, 9.81])          # NED, z down (verified)
    # TartanAir lcam_front: camera looks along +x(body) with z down => cam->body is the
    # same axis permutation MidAir uses (cam x-right,y-down,z-fwd -> body x-fwd,y-right,z-down).
    P = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    def rx(a):
        c, sn = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
    def ry(a):
        c, sn = np.cos(a), np.sin(a)
        return np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    # cam->body: TartanAir optical frame (x right, y down, z fwd) -> pose frame
    # (x fwd, y right, z down). Verified against pose_lcam_front/pose_rcam_front.
    S0 = np.eye(4); S0[:3, :3] = P
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
    filt = MSCEqFNativeFilter(
        R0, p0, v0, b0, S0, grav, d_std, np.full(6, args.delta_std), np.full(6, 1e-4),
        args.acc_density, args.gyr_density, args.acc_rw, args.gyr_rw, 0, args.num_clones)

    fcfg = echo_li.FrontendConfig.from_yaml(args.frontend_config)
    fcfg.set_camera(fx, fy, cx, cy, W, Hh, [])
    fcfg.lbp_verification = args.lbp
    if args.klt_fb > 0:
        fcfg.klt_fb_threshold_px = args.klt_fb
    fcfg.max_features = args.max_features
    if args.no_ransac:
        fcfg.enable_ransac = False
    if args.klt_window > 0:
        fcfg.klt_window = args.klt_window
    if args.pyramid_levels > 0:
        fcfg.pyramid_levels = args.pyramid_levels
    tracker = echo_li.Frontend(fcfg, W, Hh)
    matcher = None
    if args.stereo:
        matcher = echo_li.Stereo.from_pinhole(
            fx, fy, cx, cy, W, Hh, [-args.baseline, 0.0, 0.0], None, None)

    pixel_std = args.pixel_mult / fx
    buf, est_p, gt_p, est_R, gt_R, est_v, gt_v = {}, [], [], [], [], [], []
    n_upd = n_slam_upd = n_promoted = n_stereo = n_stream_range = 0
    n_try = 0
    slam = set()
    slam_min_len = args.slam_min_len if args.slam_min_len > 0 else args.num_clones
    slam_chi2 = args.slam_chi2_mult if args.slam_chi2_mult > 0 else args.chi2_mult
    bright = []

    def snap(i):
        p, Rm, vv = filt.nav_body_pose()
        est_p.append(p.copy()); est_R.append(Rm.copy()); est_v.append(vv.copy())
        gt_p.append(pos[i].copy()); gt_R.append(R_from_euler(ori[i])); gt_v.append(vel[i].copy())

    T_wc_prev = None

    def cam_pose():
        p, Rm, _ = filt.nav_body_pose()
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = p
        return T @ S0                      # body->world composed with cam->body

    for k in range(k0, k1 + 1):
        if k > k0:
            for i in range((k - 1) * IPF, k * IPF):
                filt.process_imu(gyr[i], acc[i], DT)
        img = cv2.imread(str(T / "image_lcam_front" / f"{k:06d}_lcam_front.png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            snap(k * IPF); continue
        bright.append(float(img.mean()))
        feats, _ = tracker.process(np.ascontiguousarray(img))
        if not feats:
            snap(k * IPF); continue
        pri = None
        if matcher is not None:
            rimg = cv2.imread(str(T / "image_rcam_front" / f"{k:06d}_rcam_front.png"),
                              cv2.IMREAD_GRAYSCALE)
            if rimg is not None:
                pri = matcher.range_priors(np.ascontiguousarray(rimg), tracker,
                                           args.sigma_pixel_scale)
                n_stereo += len(pri)
        filt.clone_pose(k, float(ctime[k]))
        cur, und_by_id = set(), {}
        for f in feats:
            tid = int(f["id"]); cur.add(tid)
            un, vn = (float(f["x"]) - cx) / fx, (float(f["y"]) - cy) / fy
            und_by_id[tid] = (un, vn)
            if tid not in slam:
                buf.setdefault(tid, []).append((k, un, vn))
        marg = filt.clone_ids()[0] if filt.n_clones() > args.num_clones else None

        if args.max_slam > 0 and (marg is not None or args.slam_immediate) \
                and len(slam) < args.max_slam:
            # NOTE: requiring the track to touch the MARGINALIZED clone (the MidAir
            # harness's rule) implicitly forces ~num_clones observations, because tracks
            # are contiguous — which makes `slam_min_len` inert and "immediate promotion"
            # unreachable. Promote on observation count alone; `--slam-immediate` drops
            # the marginalization requirement entirely.
            cand = [t for t, o in buf.items()
                    if t in cur and len(o) >= slam_min_len
                    and (args.slam_immediate or any(fk == marg for (fk, *_) in o))]
            cand.sort(key=lambda t: len(buf[t]), reverse=True)
            for tid in cand[: args.max_slam - len(slam)]:
                rp = None
                if pri is not None and tid in pri:
                    rr, vv = pri[tid]
                    if np.isfinite(rr) and 0.0 < rr < 400.0 and np.isfinite(vv) and vv > 0.0:
                        rp = (float(rr), float(vv) / (float(rr) * float(rr)))
                if filt.birth_landmark(tid, buf[tid], pixel_std, slam_chi2, rp):
                    filt.reanchor_landmark(tid, k)
                    slam.add(tid); buf.pop(tid, None); n_promoted += 1
        if slam:
            supd = {}
            for tid in list(slam):
                if tid in cur:
                    supd[tid] = [(k, *und_by_id[tid])]
                else:
                    filt.marginalize_landmark(tid); slam.discard(tid)
            srange = {}
            if pri is not None:
                for tid in supd:
                    if tid in pri:
                        rr, vv = pri[tid]
                        if np.isfinite(rr) and 0.0 < rr < 400.0 and np.isfinite(vv) and vv > 0.0:
                            srange[tid] = [(k, -float(np.log(rr)),
                                            float(vv) / (float(rr) * float(rr)))]
                            n_stream_range += 1
            if supd:
                n_slam_upd += filt.landmark_stream_update(
                    supd, pixel_std, slam_chi2, srange or None)
        ready, done = {}, []
        for tid, obs in buf.items():
            if (tid not in cur) or len(obs) >= args.max_track_len or (
                    marg is not None and any(fk == marg for (fk, *_) in obs)):
                if len(obs) >= 2:
                    ready[tid] = obs
                done.append(tid)
        if ready:
            n_try += len(ready)
            n_upd += filt.msc_update(ready, pixel_std, args.chi2_mult, bool(args.curvature))
        for tid in done:
            buf.pop(tid, None)
        while filt.n_clones() > args.num_clones:
            filt.marginalize_oldest()
        snap(k * IPF)

    est_p = np.array(est_p); gt_p = np.array(gt_p)
    plen = np.linalg.norm(np.diff(gt_p, axis=0), axis=1).sum()
    elen = np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum()
    err = np.linalg.norm(align_umeyama(est_p, gt_p) - gt_p, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"tracks offered to update={n_try}  accepted={n_upd}  "
          f"accept-rate={100*n_upd/max(n_try,1):.1f}%  "
          f"slam_promoted={n_promoted}  slam_updates={n_slam_upd}  "
          f"stereo_births={n_stereo}  stream_range_rows={n_stream_range}")
    print(f"frames={len(est_p)}  accepted_updates={n_upd}  "
          f"mean frame brightness={np.mean(bright):.0f}/255")
    print(f"GT path length   = {plen:8.2f} m")
    print(f"est path length  = {elen:8.2f} m   (est/gt = {elen/max(plen,1e-9):.4f})")
    print(f"ATE RMSE (SE3)   = {rmse:8.4f} m")
    print(f"ATE %            = {100*rmse/max(plen,1e-9):7.3f} %")
    print(f"ATE median/mean/max = {np.median(err):.4f} / {err.mean():.4f} / {err.max():.4f} m")
    if args.dump:
        np.savez(args.dump, est=est_p, gt=gt_p, est_R=np.array(est_R),
                 gt_R=np.array(gt_R), est_v=np.array(est_v), gt_v=np.array(gt_v))


if __name__ == "__main__":
    main()
