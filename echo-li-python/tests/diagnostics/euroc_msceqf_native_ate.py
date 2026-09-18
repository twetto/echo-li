#!/usr/bin/env python3
"""Drive the MSCEqF-native fork-b filter on EuRoC and report SE3-ATE.

WHY: every fork-b number so far is from MidAir, whose IMU noise is randomised
per-trajectory and unpublished ([[midair-imu-noise-random-per-trajectory]]) — a confound
we cannot control. EuRoC is the dataset the MSCEqF paper actually validated on
(|D.p| <~ 25 m, indoor, well-excited), with published IMU noise. If fork-b behaves on
EuRoC, the port is validated end-to-end and MidAir t1 is a specific hard case; if it
does not, there is a general defect the MidAir work never isolated.

Mono + structureless by default, to match the t1 comparison exactly.

Reuses the loaders from `examples/euroc_tracking.py` rather than rebuilding them
(per the standing "check diagnostics/ before re-implementing a dataloader" rule); only
the GT reader is extended here because we need attitude + velocity for the GT-seeded
given-origin init, and the example returns positions only.

Usage:
  python tests/diagnostics/euroc_msceqf_native_ate.py \
      --seq /mnt/18TB/chen_fu_yeh/datasets/EuRoC/vicon_room1/V1_01_easy/V1_01_easy
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
from echo_li import MSCEqFNativeFilter  # noqa: E402


def R_wb(q):
    """Rotation body->world from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_gt_full(root):
    """EuRoC state_groundtruth_estimate0: ts, p(3), q_wxyz(4), v(3), b_w(3), b_a(3)."""
    ts, p, q, v = [], [], [], []
    with open(root / "state_groundtruth_estimate0" / "data.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts.append(int(row[0]) * 1e-9)
            p.append([float(row[1]), float(row[2]), float(row[3])])
            q.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            v.append([float(row[8]), float(row[9]), float(row[10])])
    return (np.array(ts), np.array(p), np.array(q), np.array(v))


def load_imu(root):
    ts, gyr, acc = [], [], []
    with open(root / "imu0" / "data.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts.append(int(row[0]) * 1e-9)
            gyr.append([float(row[1]), float(row[2]), float(row[3])])
            acc.append([float(row[4]), float(row[5]), float(row[6])])
    return np.array(ts), np.array(gyr), np.array(acc)


def load_images(root, cam="cam0"):
    out = []
    with open(root / cam / "data.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            out.append((int(row[0]) * 1e-9, root / cam / "data" / row[1].strip()))
    return out


def load_cam(root, cam="cam0"):
    import yaml
    with open(root / cam / "sensor.yaml") as f:
        c = yaml.safe_load(f)
    w, h = c["resolution"]
    fx, fy, cx, cy = c["intrinsics"]
    d = np.array(c.get("distortion_coefficients", []), dtype=np.float64)
    t_bs = np.array(c["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return w, h, fx, fy, cx, cy, d, t_bs


def align_umeyama(est, gt):
    """SE3 (no-scale) alignment — SAME convention as the MidAir harness."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    H = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_g - R @ mu_e
    return (R @ est.T).T + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="…/V1_01_easy (containing mav0/)")
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--start-speed", type=float, default=0.2,
                    help="skip frames until GT speed exceeds this (m/s). EuRoC sequences "
                         "begin AT REST, which is degenerate for a structureless MSCKF "
                         "(zero baseline => unconditioned triangulation). The reference "
                         "C++ has a zero-velocity updater; our Rust has none, so we start "
                         "from motion onset instead of pretending to handle rest.")
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--stereo", type=int, default=0,
                    help="metric range channel from cam1 via Stereo.from_euroc(), which "
                         "builds the rig from the two sensor.yaml files directly — it "
                         "handles the non-rectified inter-camera rotation (0.0141) and "
                         "the per-camera distortion internally, so no cv2.stereoRectify "
                         "pre-pass is needed.")
    ap.add_argument("--sigma-pixel-scale", type=float, default=20.0)
    ap.add_argument("--max-slam", type=int, default=0,
                    help="in-state SLAM landmarks (0 = structureless only)")
    ap.add_argument("--slam-min-len", type=int, default=0,
                    help="min observations before promotion (0 => num_clones). Set 2 to "
                         "promote as early as the birth triangulation allows.")
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
    # EuRoC published IMU noise (ADIS16448, the standard Kalibr values).
    ap.add_argument("--acc-density", type=float, default=2.0e-3)
    ap.add_argument("--gyr-density", type=float, default=1.6968e-4)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.9393e-5)
    ap.add_argument("--delta-std", type=float, default=1.0e-2)
    ap.add_argument("--frontend-config", default=str(
        Path(__file__).resolve().parents[3] / "configs/eqvio_midair.yaml"))
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    root = Path(args.seq)
    if (root / "mav0").exists():
        root = root / "mav0"

    W, H, fx, fy, cx, cy, dist, T_BS = load_cam(root)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    gts, gtp, gtq, gtv = load_gt_full(root)
    its, igyr, iacc = load_imu(root)
    imgs = load_images(root)

    # Restrict to the window GT actually covers.
    imgs = [(t, p) for (t, p) in imgs if gts[0] <= t <= gts[-1]]
    if args.start_speed > 0:
        spd = np.linalg.norm(gtv, axis=1)
        mv = np.where(spd > args.start_speed)[0]
        if len(mv):
            t_go = gts[mv[0]]
            n_before = len(imgs)
            imgs = [(t, p) for (t, p) in imgs if t >= t_go]
            print(f"skipping {n_before - len(imgs)} at-rest frames; motion starts "
                  f"t+{t_go - gts[0]:.1f}s (no ZUPT in this filter)")
    if args.nframes > 0:
        imgs = imgs[: args.nframes]
    nimg = len(imgs)
    print(f"EuRoC {root.parent.name}: {nimg} frames, {len(its)} IMU, "
          f"GT {gts[0]:.1f}..{gts[-1]:.1f}s  cam {W}x{H} f={fx:.1f}")

    def gt_at(t):
        i = np.clip(np.searchsorted(gts, t), 1, len(gts) - 1)
        a = (t - gts[i - 1]) / max(gts[i] - gts[i - 1], 1e-9)
        p = (1 - a) * gtp[i - 1] + a * gtp[i]
        v = (1 - a) * gtv[i - 1] + a * gtv[i]
        q = gtq[i - 1] if a < 0.5 else gtq[i]   # nearest — avoids quat lerp sign issues
        return p, R_wb(q), v

    t0 = imgs[0][0]
    p0, R0, v0 = gt_at(t0)
    b0 = np.zeros(6)
    # EuRoC world frame is gravity-aligned with z UP => gravity acceleration is -z.
    grav = np.array([0.0, 0.0, -9.81])
    S0 = T_BS.copy()                      # cam0 -> body, exactly what the filter wants
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
    delta_std = np.full(6, args.delta_std)
    e_std = np.full(6, 1e-4)

    filt = MSCEqFNativeFilter(
        R0, p0, v0, b0, S0, grav, d_std, delta_std, e_std,
        args.acc_density, args.gyr_density, args.acc_rw, args.gyr_rw,
        0, args.num_clones)

    fcfg = echo_li.FrontendConfig.from_yaml(args.frontend_config)
    fcfg.set_camera(fx, fy, cx, cy, W, H, [])   # we undistort feature coords ourselves
    fcfg.lbp_verification = False
    fcfg.max_features = args.max_features
    tracker = echo_li.Frontend(fcfg, W, H)
    matcher = None
    if args.stereo:
        matcher = echo_li.Stereo.from_euroc(
            str(root / "cam0" / "sensor.yaml"), str(root / "cam1" / "sensor.yaml"), W, H, None)
        cam1 = {}
        import csv as _csv
        with open(root / "cam1" / "data.csv") as _f:
            _r = _csv.reader(_f); next(_r)
            for _row in _r:
                cam1[int(_row[0]) * 1e-9] = root / "cam1" / "data" / _row[1].strip()
        cam1_ts = np.array(sorted(cam1.keys()))
        print(f"stereo: cam1 {len(cam1)} frames, baseline "
              f"{matcher.baseline_meters:.4f} m")

    pixel_std = args.pixel_mult / fx
    buf, est_p, gt_p, est_R, gt_R, est_v, gt_v = {}, [], [], [], [], [], []
    slam = set()
    slam_min_len = args.slam_min_len if args.slam_min_len > 0 else args.num_clones
    slam_chi2 = args.slam_chi2_mult if args.slam_chi2_mult > 0 else args.chi2_mult
    n_upd = n_slam_upd = n_promoted = n_stereo = n_stream_range = 0
    imu_i = int(np.searchsorted(its, t0))

    def snap(t):
        p, Rm, vv = filt.nav_body_pose()
        gp, gR, gv = gt_at(t)
        est_p.append(p.copy()); est_R.append(Rm.copy()); est_v.append(vv.copy())
        gt_p.append(gp.copy()); gt_R.append(gR.copy()); gt_v.append(gv.copy())

    for k, (tk, path) in enumerate(imgs):
        # --- propagate IMU up to this image time, with REAL dt (EuRoC is ~200 Hz) ---
        while imu_i < len(its) and its[imu_i] <= tk:
            dt = its[imu_i] - its[imu_i - 1] if imu_i > 0 else 0.005
            if 0.0 < dt < 0.1:
                filt.process_imu(igyr[imu_i], iacc[imu_i], float(dt))
            imu_i += 1

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        feats, _ = tracker.process(np.ascontiguousarray(gray))
        if not feats:
            snap(tk)
            continue
        pri = None
        if matcher is not None:
            j = int(np.argmin(np.abs(cam1_ts - tk)))
            if abs(cam1_ts[j] - tk) < 0.005:
                rimg = cv2.imread(str(cam1[cam1_ts[j]]), cv2.IMREAD_GRAYSCALE)
                if rimg is not None:
                    pri = matcher.range_priors(np.ascontiguousarray(rimg), tracker,
                                               args.sigma_pixel_scale)
                    n_stereo += len(pri)

        # Undistort feature pixels -> normalized (Z1) coordinates.
        px = np.array([[f["x"], f["y"]] for f in feats], dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dist).reshape(-1, 2)
        ids = [int(f["id"]) for f in feats]

        filt.clone_pose(k, tk)
        cur = set(ids)
        und_by_id = {t: (float(u), float(v)) for t, (u, v) in zip(ids, und)}
        for tid, (un, vn) in zip(ids, und):
            if tid in slam:
                continue
            buf.setdefault(tid, []).append((k, float(un), float(vn)))

        marg = filt.clone_ids()[0] if filt.n_clones() > args.num_clones else None

        # (0) PROMOTE window-spanning tracks about to be marginalized (mirrors the
        #     MidAir harness: birth from the buffered obs, then re-anchor to `k`).
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

        # (1) per-frame stream update for live landmarks; drop on track loss.
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
            lost = tid not in cur
            toolong = len(obs) >= args.max_track_len
            touches = marg is not None and any(fk == marg for (fk, *_) in obs)
            if lost or toolong or touches:
                if len(obs) >= 2:
                    ready[tid] = obs
                done.append(tid)
        if ready:
            n_upd += filt.msc_update(ready, pixel_std, args.chi2_mult, bool(args.curvature))
        for tid in done:
            buf.pop(tid, None)
        while filt.n_clones() > args.num_clones:
            filt.marginalize_oldest()
        snap(tk)
        if k % 400 == 0:
            print(f"  frame {k}/{nimg}  feats {len(feats)}  updates {n_upd}")

    est_p = np.array(est_p); gt_p = np.array(gt_p)
    plen = np.linalg.norm(np.diff(gt_p, axis=0), axis=1).sum()
    elen = np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum()
    al = align_umeyama(est_p, gt_p)
    err = np.linalg.norm(al - gt_p, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"frames={len(est_p)}  accepted_updates={n_upd}  "
          f"slam_promoted={n_promoted}  slam_updates={n_slam_upd}  "
          f"stereo_births={n_stereo}  stream_range_rows={n_stream_range}")
    print(f"GT path length   = {plen:8.2f} m")
    print(f"est path length  = {elen:8.2f} m   (est/gt = {elen/max(plen,1e-9):.4f})")
    print(f"ATE RMSE (SE3)   = {rmse:8.4f} m    [EqVIO-path baseline V1_01 ~0.12 m]")
    print(f"ATE %            = {100*rmse/max(plen,1e-9):7.3f} %")
    print(f"ATE median/mean/max = {np.median(err):.4f} / {err.mean():.4f} / {err.max():.4f} m")
    if args.dump:
        np.savez(args.dump, est=est_p, gt=gt_p, est_R=np.array(est_R),
                 gt_R=np.array(gt_R), est_v=np.array(est_v), gt_v=np.array(gt_v))
        print(f"dumped -> {args.dump}")


if __name__ == "__main__":
    main()
