#!/usr/bin/env python3
"""Drive the MSCEqF-native fork-b filter on TUM-VI and report SE3-ATE.

Adapted from `euroc_msceqf_native_ate.py` (copied rather than switched on a --dataset
flag: that harness is committed and validated, and this one differs in four
dataset-specific places). Everything else -- main loop, promotion, Civera birth, the
per-frame ledger, align_umeyama -- is shared verbatim so numbers stay comparable.

What differs from EuRoC:
  * FISHEYE. Camera is pinhole + EQUIDISTANT (Kannala-Brandt); undistortion must go
    through cv2.fisheye.undistortPoints, not the radtan cv2.undistortPoints.
  * GT is `mocap0`, which has NO velocity column (EuRoC's state_groundtruth_estimate0
    does), so velocity is finite-differenced from the mocap positions.
  * Calibration lives in `dso/camchain.yaml` (Kalibr); the per-sensor sensor.yaml files
    ship EMPTY. Extrinsics are given as T_cam_imu, so cam->body is its inverse.
  * IMU noise from the sequence's own dso/imu_config.yaml (the inflated values).

Verified before writing this (no harness needed): the mocap world frame IS gravity-aligned
z-up, mocap attitude ~ IMU attitude (total error <=3.5 deg incl. yaw over all 6 rooms), and
all 6 room sequences START AT REST (|v0| <= 0.05 m/s) -- so v0=0 is TRUE here and this is a
tight-sigma dataset like EuRoC, not a loose-sigma one like MidAir.

Original EuRoC docstring follows.
----
Drive the MSCEqF-native fork-b filter on EuRoC and report SE3-ATE.

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
  python tests/diagnostics/tumvi_msceqf_native_ate.py \
      --seq /mnt/18TB/chen_fu_yeh/datasets/TUM-VI/dataset-room1_512_16
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
    """TUM-VI mocap0: ts, p(3), q_wxyz(4) -- and NOTHING else. EuRoC ships velocity in
    its GT; the mocap does not, so we central-difference the positions. The mocap runs
    at ~120 Hz, so the raw difference is noisy; a short boxcar is applied because this
    velocity is used as the GT-init seed and as the ledger's reference speed, where a
    per-sample spike would masquerade as a filter error."""
    ts, p, q = [], [], []
    with open(root / "mocap0" / "data.csv") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].lstrip().startswith("#"):
                continue
            ts.append(int(row[0]) * 1e-9)
            p.append([float(row[1]), float(row[2]), float(row[3])])
            q.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    ts, p, q = np.array(ts), np.array(p), np.array(q)
    v = np.gradient(p, ts, axis=0)
    w = 5
    if len(v) > w:
        k = np.ones(w) / w
        v = np.stack([np.convolve(v[:, i], k, mode="same") for i in range(3)], axis=1)
    return (ts, p, q, v)


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
    """TUM-VI: Kalibr camchain in <seq>/dso/, NOT mav0/<cam>/sensor.yaml (empty here)."""
    import yaml
    with open(root.parent / "dso" / "camchain.yaml") as f:
        c = yaml.safe_load(f)[cam]
    w, h = c["resolution"]
    fx, fy, cx, cy = c["intrinsics"]
    d = np.array(c["distortion_coeffs"], dtype=np.float64)
    if c["distortion_model"] != "equidistant":
        raise SystemExit(f"expected equidistant (fisheye), got {c['distortion_model']}")
    # camchain gives T_cam_imu; the filter wants cam -> body(imu).
    t_bs = np.linalg.inv(np.array(c["T_cam_imu"], dtype=np.float64))
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
    ap.add_argument("--seq", required=True, help="…/dataset-room1_512_16 (containing mav0/ + dso/)")
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--start-speed", type=float, default=0.2,
                    help="skip frames until GT speed exceeds this (m/s). EuRoC sequences "
                         "begin AT REST, which is degenerate for a structureless MSCKF "
                         "(zero baseline => unconditioned triangulation). The reference "
                         "C++ has a zero-velocity updater; our Rust has none, so we start "
                         "from motion onset instead of pretending to handle rest.")
    ap.add_argument("--init", choices=("gt", "gravity"), default="gt",
                    help="gt: seed R0/p0/v0 from ground truth (benchmarking/debugging). "
                         "gravity: gravity-align the mean accel of the --init-imu "
                         "samples ending at t0, p0=0, v0=0 -- no ground truth at all.")
    ap.add_argument("--civera-depth", type=float, default=0.0,
                    help="if >0, birth in-state landmarks by Civera undelayed "
                         "inverse-depth init with this depth PRIOR (m) instead of "
                         "triangulating. Needs no parallax, so it works at motion onset. "
                         "Pick it as the scene's rough scale (EuRoC rooms ~2-4 m).")
    ap.add_argument("--diag-csv", default="",
                    help="per-FRAME update ledger: attempted vs accepted updates, "
                         "in-state landmark depth distribution, and state-vs-GT error. "
                         "This is the 'when does it go bad, and what differs' probe -- "
                         "aggregate end-of-run ATE cannot localise a divergence.")
    ap.add_argument("--civera-frames", type=int, default=0,
                    help="apply Civera+immediate promotion ONLY for the first N frames "
                         "(the initialization phase), then revert to the normal "
                         "window-spanning triangulated promotion. 0 = whole run. The "
                         "init phase is where triangulation has no parallax to work "
                         "with; after it, the standard path has earned its baseline.")
    ap.add_argument("--civera-rho-rel", type=float, default=1.0,
                    help="sigma_rho / rho0. 1.0 = Civera's band (infinity .. d0/2 at 1sigma)")
    ap.add_argument("--init-r0-gt", action="store_true", help="ablation: GT attitude")
    ap.add_argument("--init-p0-gt", action="store_true", help="ablation: GT position")
    ap.add_argument("--init-v0-gt", action="store_true", help="ablation: GT velocity")
    ap.add_argument("--init-std-gt", action="store_true", help="ablation: GT-init d_std")
    ap.add_argument("--init-imu", type=int, default=200,
                    help="accel samples averaged for the align (EuRoC IMU is ~200 Hz)")
    ap.add_argument("--init-vel-std", type=float, default=10.0,
                    help="initial velocity std for --init gravity")
    ap.add_argument("--init-att-std", type=float, default=0.2,
                    help="initial attitude std (rad) for --init gravity")
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
    # TUM-VI dso/imu_config.yaml, the INFLATED values the dataset recommends
    # (white noise x2, bias random walk x10 vs the raw Allan values).
    ap.add_argument("--acc-density", type=float, default=2.8e-3)
    ap.add_argument("--gyr-density", type=float, default=1.6e-4)
    ap.add_argument("--acc-rw", type=float, default=8.6e-4)
    ap.add_argument("--gyr-rw", type=float, default=2.2e-5)
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
    print(f"TUM-VI {root.parent.name}: {nimg} frames, {len(its)} IMU, "
          f"GT {gts[0]:.1f}..{gts[-1]:.1f}s  cam {W}x{H} f={fx:.1f}")

    def gt_at(t):
        i = np.clip(np.searchsorted(gts, t), 1, len(gts) - 1)
        a = (t - gts[i - 1]) / max(gts[i] - gts[i - 1], 1e-9)
        p = (1 - a) * gtp[i - 1] + a * gtp[i]
        v = (1 - a) * gtv[i - 1] + a * gtv[i]
        q = gtq[i - 1] if a < 0.5 else gtq[i]   # nearest — avoids quat lerp sign issues
        return p, R_wb(q), v

    t0 = imgs[0][0]
    b0 = np.zeros(6)
    # EuRoC world frame is gravity-aligned with z UP => gravity acceleration is -z.
    grav = np.array([0.0, 0.0, -9.81])
    S0 = T_BS.copy()                      # cam0 -> body, exactly what the filter wants
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])

    if args.init == "gt":
        p0, R0, v0 = gt_at(t0)
    else:
        # Gravity-align, as echo_li_core::initialization::estimate_initial_pose does.
        # Accel reads specific force f = R^T (a - g_vec), so at rest f_world = -g_vec
        # and the align target is -grav/|grav| (= +z here, z-up).
        #
        # The samples are taken from the window ENDING at t0, not from the start of the
        # recording. EuRoC begins at rest but the filter starts at motion onset (no
        # ZUPT), and the vehicle is rotated by then -- aligning off the first samples
        # would hand the filter a stale attitude from tens of seconds earlier. Ending
        # at t0 keeps the align consistent with the state it initializes, and those
        # samples are still at rest whenever start_speed onset is the first motion.
        j1 = int(np.searchsorted(its, t0))
        j0 = max(0, j1 - max(1, args.init_imu))
        f_body = iacc[j0:j1].mean(axis=0)
        a_hat = f_body / np.linalg.norm(f_body)
        b_hat = -grav / np.linalg.norm(grav)
        v = np.cross(a_hat, b_hat); c = float(a_hat @ b_hat)
        if c < -0.999999:
            ax = np.array([1., 0., 0.]) if abs(a_hat[0]) < 0.9 else np.array([0., 1., 0.])
            v = np.cross(a_hat, ax); v /= np.linalg.norm(v)
            Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R0 = np.eye(3) + 2.0 * Kx @ Kx
        else:
            Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R0 = np.eye(3) + Kx + Kx @ Kx / (1.0 + c)
        p0 = np.zeros(3); v0 = np.zeros(3)
        d_std[0:3] = args.init_att_std
        d_std[3:6] = args.init_vel_std
        _p, Rg, vg = gt_at(t0)
        # Ablation: put back one GT component at a time to find which one of
        # {R0, p0, v0, d_std} is responsible for updates never being accepted.
        if args.init_r0_gt:
            R0 = Rg.copy()
        if args.init_p0_gt:
            p0 = _p.copy()
        if args.init_v0_gt:
            v0 = vg.copy()
        if args.init_std_gt:
            d_std[0:3] = 1e-2; d_std[3:6] = 1e-1
        # Yaw-invariant (observable) error: R^T u is unchanged by a yaw about u, so this
        # isolates the tilt the align is actually responsible for from the free gauge.
        u = b_hat
        tilt = np.degrees(np.arccos(np.clip((R0.T @ u) @ (Rg.T @ u), -1.0, 1.0)))
        tot = np.degrees(np.arccos(np.clip((np.trace(R0.T @ Rg) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"init=gravity: {j1-j0} accel samples ending at t0, |f|={np.linalg.norm(f_body):.3f} "
              f"m/s^2 (|f|-g = {np.linalg.norm(f_body)-9.81:+.3f})")
        print(f"  TILT err={tilt:.2f} deg (observable)   total={tot:.1f} deg (incl. free yaw)"
              f"   |v_gt(t0)|={np.linalg.norm(vg):.2f} m/s asserted as 0")
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

    diag = [] if args.diag_csv else None
    for k, (tk, path) in enumerate(imgs):
        n_acc_k = 0; n_slam_acc = 0; n_supd_try = 0; n_promoted_k = 0
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
        # FISHEYE: equidistant model -> cv2.fisheye, not the radtan undistortPoints.
        und = cv2.fisheye.undistortPoints(px, K, dist.reshape(4, 1)).reshape(-1, 2)
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
        in_init = args.civera_frames <= 0 or k < args.civera_frames
        use_civera = args.civera_depth > 0.0 and in_init
        immediate = args.slam_immediate and in_init
        if args.max_slam > 0 and (marg is not None or immediate) \
                and len(slam) < args.max_slam:
            # NOTE: requiring the track to touch the MARGINALIZED clone (the MidAir
            # harness's rule) implicitly forces ~num_clones observations, because tracks
            # are contiguous — which makes `slam_min_len` inert and "immediate promotion"
            # unreachable. Promote on observation count alone; `--slam-immediate` drops
            # the marginalization requirement entirely.
            cand = [t for t, o in buf.items()
                    if t in cur and len(o) >= slam_min_len
                    and (immediate or any(fk == marg for (fk, *_) in o))]
            cand.sort(key=lambda t: len(buf[t]), reverse=True)
            for tid in cand[: args.max_slam - len(slam)]:
                rp = None
                if pri is not None and tid in pri:
                    rr, vv = pri[tid]
                    if np.isfinite(rr) and 0.0 < rr < 400.0 and np.isfinite(vv) and vv > 0.0:
                        rp = (float(rr), float(vv) / (float(rr) * float(rr)))
                if use_civera:
                    # Undelayed inverse-depth birth: no parallax needed, so promotion
                    # is not gated on the track having earned a baseline.
                    born = filt.birth_landmark_civera(
                        tid, buf[tid], pixel_std, args.civera_depth, args.civera_rho_rel)
                else:
                    born = filt.birth_landmark(tid, buf[tid], pixel_std, slam_chi2, rp)
                if born:
                    filt.reanchor_landmark(tid, k)
                    slam.add(tid); buf.pop(tid, None); n_promoted += 1; n_promoted_k += 1

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
            n_supd_try = len(supd)
            if supd:
                n_slam_acc = filt.landmark_stream_update(
                    supd, pixel_std, slam_chi2, srange or None)
                n_slam_upd += n_slam_acc

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
            n_acc_k = filt.msc_update(ready, pixel_std, args.chi2_mult, bool(args.curvature))
            n_upd += n_acc_k
        for tid in done:
            buf.pop(tid, None)
        while filt.n_clones() > args.num_clones:
            filt.marginalize_oldest()
        if diag is not None:
            ep, eR, ev = filt.nav_body_pose()
            gp, gR, gv = gt_at(tk)
            dep = [d for d in (filt.landmark_range_anchor(t) for t in slam)
                   if d is not None and np.isfinite(d)]
            dep = np.array(dep) if dep else np.array([np.nan])
            # Gauge-free error channels: speed magnitude and attitude TILT need no
            # alignment, so they are comparable frame-by-frame without Umeyama.
            u_g = np.array([0.0, 0.0, 1.0])
            tilt = np.degrees(np.arccos(np.clip((eR.T @ u_g) @ (gR.T @ u_g), -1, 1)))
            diag.append([k, tk - t0, len(ready), n_acc_k, n_supd_try, n_slam_acc,
                         len(slam), n_promoted_k,
                         np.nanmedian(dep), np.nanmin(dep), np.nanmax(dep),
                         np.linalg.norm(ev), np.linalg.norm(gv), tilt,
                         float(np.trace(filt.nav_vel_cov())) if hasattr(filt, "nav_vel_cov") else np.nan])
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
    print(f"ATE RMSE (SE3)   = {rmse:8.4f} m")
    print(f"ATE %            = {100*rmse/max(plen,1e-9):7.3f} %")
    print(f"ATE median/mean/max = {np.median(err):.4f} / {err.mean():.4f} / {err.max():.4f} m")
    if diag:
        hdr = ("k,t,msc_try,msc_acc,slam_try,slam_acc,n_slam,promoted,"
               "dep_med,dep_min,dep_max,est_speed,gt_speed,tilt_deg,vel_cov_tr")
        np.savetxt(args.diag_csv, np.array(diag, dtype=float), delimiter=",",
                   header=hdr, comments="")
        print(f"per-frame ledger -> {args.diag_csv}")
    if args.dump:
        np.savez(args.dump, est=est_p, gt=gt_p, est_R=np.array(est_R),
                 gt_R=np.array(gt_R), est_v=np.array(est_v), gt_v=np.array(gt_v))
        print(f"dumped -> {args.dump}")


if __name__ == "__main__":
    main()
