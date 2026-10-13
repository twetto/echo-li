"""Run ECHO-LI offline on the Pi 4 rig's bike recording and render what it sees.

Two panels, the same layout as the MidAir diagnostic video:
  LEFT   the rig's own 320x200 frame, with the front-end's tracked features drawn
         hollow and the filter's IN-STATE landmarks drawn solid, coloured by range.
         Only ~21 of ~300 tracked features are in the state (maxFeatures is the
         state size), so the two layers are genuinely different populations and
         drawing them the same way would overstate what the filter knows.
  RIGHT  the estimated trajectory, top-down, with the trajectory the bag already
         carries (/echo_li/odometry, from an earlier on-rig run) drawn behind it.

WHY THE RECORDED ODOMETRY IS DRAWN. This harness re-creates, offline, a pipeline
that normally runs inside a ROS 2 node: extrinsics, the 12.583 ms camera-IMU
timeshift, gravity, and the IMU/frame interleave all have to be reproduced by hand.
Every one of those is a silent failure mode -- the same class that put five of them
into a previous hand-rolled adapter. The bag's own odometry is therefore a CONTROL,
not decoration: if this run does not track it, the wiring is wrong, and the picture
is not worth reading.

Pixels come from frames320.bin, built with IMREAD_REDUCED_GRAYSCALE_4 "the same
decode as the live rig", so the front-end sees what the rig saw. Intrinsics are the
authoritative calib_final values scaled by 0.25; distortion is normalised and does
not scale.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import echo_li  # noqa: E402

BIKE = "/mnt/18TB/chen_fu_yeh/bikevid"
W, H = 320, 200
# calib_final-camchain-imucam.yaml, scaled by 0.25 for the raw_scale_denom=4 stream.
FX, FY, CX, CY = 226.7062860357, 226.1687971677, 150.0624337420, 99.3898662352
DIST = (0.046897611239, -0.062989153081, -0.001026069440, -0.001348024507)
TIMESHIFT = 0.012583408676        # camera trigger -> exposure, measured
GRAVITY = 9.7893                  # Hsinchu local g, NOT 9.81 (calib README)
# T_imu_cam from myrig_calib_320.yaml: camera-to-IMU rotation, camera position in IMU.
T_BS = np.array([
    [-0.0174737788,  0.0260945687, -0.9995067486, -0.0607532420],
    [-0.9998079935, -0.0093224586,  0.0172356597,  0.0013138133],
    [-0.0088681032,  0.9996160089,  0.0262524569, -0.0284133329],
    [0.0, 0.0, 0.0, 1.0]])


def clut(cm, t):
    return tuple(int(v) for v in cm[int(np.clip(t, 0, 1) * 255)])


def txt(img, s, org, col=(235, 235, 235)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", required=True, help="bike_sensors.npz from the extractor")
    ap.add_argument("--config", required=True, help="eqvio_myrig_rt_320.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nframes", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--no-render", action="store_true", help="filter pass only")
    ap.add_argument("--no-vision", action="store_true",
                    help="IMU dead-reckoning only: feed process_imu but never process_vision. "
                         "Discriminates 'vision is correcting the IMU' from 'vision is being "
                         "rejected and the filter is open-loop'.")
    ap.add_argument("--scene-depth", type=float, default=0.0,
                    help="override eqf.initialValue.sceneDepth (m); 0 = keep config value")
    ap.add_argument("--outlier-abs", type=float, default=0.0,
                    help="override eqf.measurementNoise.featureOutlierAbs (px); 0 = keep")
    ap.add_argument("--depth-trace", default="", help="npz of per-landmark depth over time")
    ap.add_argument("--backend", default="eqvio", choices=["eqvio", "msceqf"],
                    help="pose backend. msceqf drives echo_li_core::backend::MsceqfBackend, "
                         "the MSCKF-style filter, on the same frames and IMU.")
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-slam", type=int, default=0)
    ap.add_argument("--zupt", action="store_true",
                    help="zero-velocity update while the IMU says the rig is at rest. "
                         "A structureless MSCKF has no vision update without parallax, "
                         "so without this it dead-reckons through the bag's 25 s idle.")
    ap.add_argument("--frontend-features", type=int, default=0,
                    help="override RudolfV.maxFeatures. The rig config sets 21 because "
                         "EqVIO holds 21 landmarks IN STATE; a structureless MSCKF "
                         "consumes every track and wants far more (MidAir fed it ~118).")
    ap.add_argument("--feature-dist", type=float, default=0.0,
                    help="override RudolfV.featureDist (minimum feature spacing, px). "
                         "The rig sets 32 on a 320x200 image, which caps the detector "
                         "at ~60 features regardless of maxFeatures.")
    ap.add_argument("--no-lbp", action="store_true",
                    help="disable the frontend's LBP descriptor verification")
    ap.add_argument("--direct-product-bias", action="store_true",
                    help="static (abelian) bias chart instead of semi-direct")
    ap.add_argument("--timeshift", type=float, default=None,
                    help="override the camera-IMU time offset (s). A wrong offset costs "
                         "v*dt of position, i.e. an error PROPORTIONAL TO SPEED.")
    ap.add_argument("--max-promote-range", type=float, default=0.0,
                    help="refuse to promote landmarks triangulated beyond this range (m)")
    ap.add_argument("--accel-bias-std", type=float, default=1e-2,
                    help="initial accel-bias std (m/s^2). The rig's EqVIO config uses "
                         "1.258; the MidAir harness value of 1e-2 assumes a GT-seeded "
                         "start and cannot absorb this rig's 0.14 m/s^2 gravity-"
                         "magnitude error, which lands on scale.")
    ap.add_argument("--gyro-bias-std", type=float, default=1e-2,
                    help="initial gyro-bias std (rad/s)")
    ap.add_argument("--accel-bias-rw", type=float, default=0.0,
                    help="override ONLY the accel bias random walk (m/s^3/sqrt(Hz)); "
                         "0 = keep whatever --imu-noise selects. The shipped 4.1e-3 is "
                         "64x this IMU's measured Allan floor of 6.44e-5 and was fitted "
                         "on short handheld bags, so over a 560 s ride it permits "
                         "0.097 m/s^2 of bias drift against the sensor's own 0.0015 -- "
                         "enough for the bias to absorb a tilt.")
    ap.add_argument("--imu-noise", default="tuned", choices=["tuned", "allan"],
                    help="tuned = the inflated values the rig's EqVIO config ships "
                         "(acc 4.2e-3, gyr 1.7e-4, accRW 4.1e-3, gyrRW 1.2e-4); "
                         "allan = raw static-bench Allan variance from imu0.yaml. "
                         "The rig's own config inflates for a reason -- a bench floor "
                         "understates a vibrating vehicle.")
    ap.add_argument("--msceqf-pixel-std", type=float, default=1.0,
                    help="MSCEqF feature noise in PIXELS (divided by fx internally)")
    ap.add_argument("--msceqf-chi2", type=float, default=1.0,
                    help="MSCEqF chi2 gate multiplier")
    ap.add_argument("--civera-birth-depth", type=float, default=0.0,
                    help="birth in-state landmarks from this prior depth (m) instead of "
                         "from a triangulation; 0 = triangulated. Triangulated birth can "
                         "only admit the near scene here (measured in-state range "
                         "0.56-12.9 m on an outdoor ride), so WHICH part of the scene "
                         "constrains the filter is a selection effect. EqVIO, same bag "
                         "and same tracks, admits every track against sceneDepth 5.0 and "
                         "holds scale to 1.04x.")
    ap.add_argument("--civera-birth-rel-std", type=float, default=1.0,
                    help="inverse-depth relative std for --civera-birth-depth; the band "
                         "is [d0*exp(-rel), d0*exp(+rel)], so 1.0 spans 0.37x-2.7x. Wide "
                         "priors are known to be catastrophic in this chart.")
    ap.add_argument("--vision-stride", type=int, default=1,
                    help="give the filter only every Nth frame's observations (the "
                         "front-end still tracks every frame, so tracks stay long). "
                         "11 clones at the full 62 Hz span 0.19 s = 0.9 m at riding "
                         "speed, which is a very short baseline for an outdoor scene; "
                         "a stride multiplies the window's baseline by N.")
    ap.add_argument("--clone-at-frame-stamp", action="store_true",
                    help="propagate to the frame's own stamp before cloning instead of "
                         "cloning whatever the last IMU sample left behind. This rig's "
                         "IMU is 122 Hz against a 62 Hz camera, so the default leaves "
                         "each clone a median 3.7 ms / 1.8 cm-at-4.7-m/s behind the "
                         "image it is attached to, and the lag varies per frame.")
    ap.add_argument("--freeze-bias", action="store_true",
                    help="BOUND on the whole bias-chart family: initial bias std and "
                         "bias random walk -> 0, so the bias never moves from its "
                         "seeded value. If the underscale survives this, no bias chart "
                         "(semi-direct, direct-product, anything) can be its mechanism.")
    ap.add_argument("--zupt-sigma", type=float, default=1e-2,
                    help="assumed stddev of the zero-velocity measurement, m/s")
    args = ap.parse_args()

    global TIMESHIFT
    if args.timeshift is not None:
        TIMESHIFT = args.timeshift
        print(f'camera-IMU timeshift overridden -> {TIMESHIFT*1e3:.2f} ms')
    z = np.load(args.sensors)
    imu, img_t, odom = z["imu"], z["img_t"], z["odom"]
    idx = np.load(f"{BIKE}/frames320_index.npy")
    cache = np.memmap(f"{BIKE}/frames320.bin", dtype=np.uint8, mode="r").reshape(-1, H, W)
    n_all = min(len(idx), len(cache))
    n = n_all if args.nframes <= 0 else min(args.nframes, n_all - args.start)
    print(f"frames {n} of {n_all}   imu {imu.shape}   recorded odom {odom.shape}")

    cfg_path = args.config
    if args.scene_depth > 0 or args.outlier_abs > 0:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        if args.scene_depth > 0:
            cfg["eqf"]["initialValue"]["sceneDepth"] = float(args.scene_depth)
        if args.outlier_abs > 0:
            cfg["eqf"]["measurementNoise"]["featureOutlierAbs"] = float(args.outlier_abs)
        cfg_path = args.out.replace(".mp4", "_cfg.yaml")
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        print(f"config override -> {cfg_path}  sceneDepth="
              f"{cfg['eqf']['initialValue']['sceneDepth']}  "
              f"featureOutlierAbs={cfg['eqf']['measurementNoise']['featureOutlierAbs']}")

    cam = echo_li.RadTanCamera(FX, FY, CX, CY, *DIST)
    fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
    fcfg.set_camera(FX, FY, CX, CY, W, H, list(DIST))
    if args.no_lbp:
        fcfg.lbp_verification = False
        print("frontend LBP verification DISABLED")
    if args.feature_dist > 0:
        fcfg.cell_size = int(args.feature_dist)
        print(f"frontend featureDist overridden -> {args.feature_dist} px")
    if args.frontend_features > 0:
        fcfg.max_features = args.frontend_features
        print(f"frontend maxFeatures overridden -> {args.frontend_features}")
    tracker = echo_li.Frontend(fcfg, W, H)
    use_msceqf = args.backend == "msceqf"

    # Init EXACTLY as the shipped node does (echo-li-ros2 voxl2_vio_node.rs:853).
    # It buffers IMU until n_init samples, calls estimate_initial_pose -- which averages
    # those samples, normalises, and rotates measured-gravity-in-body onto world +z with
    # zero position and zero velocity -- rebuilds the filter at that pose, then REPLAYS the
    # buffered samples. Reproducing the replay matters: those samples are the filter's only
    # evidence of the bias before the first frame. set_camera_extrinsics must come first,
    # because set_initial_state reads the camera offset to build the sensor state.
    N_INIT = 100
    # Align from the samples immediately BEFORE the first frame used, not from the
    # bag's beginning. With --start they are not the same window, and aligning at t=0
    # while feeding frames from mid-ride starts the filter in a state unrelated to the
    # data it is about to see.
    i_start = int(np.searchsorted(imu[:, 0], img_t[args.start] + TIMESHIFT))
    i0 = max(N_INIT, i_start)
    # Everything below used to be re-implemented here: a gravity-align from the rest
    # window, a hand-seeded gyro bias, a hand-built MsceqfBackend with twenty-two typed
    # arguments, and a Python zero-velocity detector. Two of this investigation's
    # harness bugs were exactly those typed numbers. Now the harness writes its sweep
    # flags into an `MSCEqF:` / `ZUPT:` section and hands the config to `VIOFilter`, the
    # same chooser, initialiser and ZUPT the rig node runs.
    with open(cfg_path) as fh:
        full_cfg = yaml.safe_load(fh)
    if use_msceqf:
        acc_d, gyr_d, acc_rw, gyr_rw = ((4.2e-03, 1.7e-04, 4.1e-03, 1.2e-04) if args.imu_noise == "tuned"
                                        else (7.47e-04, 1.63e-04, 2.08e-04, 6.80e-05))
        if args.accel_bias_rw:
            acc_rw = args.accel_bias_rw
        if args.freeze_bias:
            acc_rw, gyr_rw = 0.0, 0.0
        full_cfg["MSCEqF"] = {
            "enabled": True,
            "numClones": args.num_clones,
            "pixelStd": args.msceqf_pixel_std,
            "chi2Mult": args.msceqf_chi2,
            "maxSlam": args.max_slam,
            "maxPromoteRange": args.max_promote_range,
            "directProductBias": bool(args.direct_product_bias),
            "cloneAtFrameStamp": bool(args.clone_at_frame_stamp),
            "civeraBirthDepth": args.civera_birth_depth,
            "civeraBirthRelStd": args.civera_birth_rel_std,
            # World +z is UP after gravity alignment, so gravity ACCELERATION is -z.
            "gravity": [0.0, 0.0, -GRAVITY],
            # One bias std in the config; the accel value is the one this rig's 0.14 m/s^2
            # gravity-magnitude error depends on (see --accel-bias-std).
            "initialStd": {"attitude": 1e-2, "velocity": 1e-1, "position": 1e-4,
                           "bias": 1e-9 if args.freeze_bias else args.accel_bias_std,
                           "extrinsics": 1e-4},
            "processNoise": {"accelDensity": acc_d, "gyroDensity": gyr_d,
                             "accelRandomWalk": acc_rw, "gyroRandomWalk": gyr_rw,
                             "transitionOrder": 0},
        }
    if args.zupt:
        full_cfg["ZUPT"] = {"enabled": True, "window": 25, "gyroStdThreshold": 0.05,
                            "accStdThreshold": 0.15, "sigmaV": args.zupt_sigma}
    run_cfg = args.out.replace(".mp4", "_config.yaml")
    with open(run_cfg, "w") as fh:
        yaml.safe_dump(full_cfg, fh, sort_keys=False)
    vio = echo_li.VIOFilter(run_cfg, cam, fx=FX)
    vio.set_camera_extrinsics(np.ascontiguousarray(T_BS))
    print(f"backend={vio.backend_kind}  zupt={bool(args.zupt)}  config={run_cfg}")

    def push_imu(row):
        vio.process_imu(float(row[0]), row[1:4].tolist(), row[4:7].tolist())

    # The initialiser buffers 100 samples, gravity-aligns, seeds the gyro bias if the
    # window was stationary, rebuilds the backend at that pose and replays the buffer.
    for j in range(i0 - N_INIT, i0):
        push_imu(imu[j])

    traj, stamps, feats_per_frame, lm_rng = [], [], {}, {}
    diag = []
    grav_audit = []
    bias_trace = []
    yaw_trace = []
    rr_all = {} if args.depth_trace else None
    ii = i0
    t0 = time.perf_counter()
    for k in range(args.start, args.start + n):
        ts = img_t[k] + TIMESHIFT
        while ii < len(imu) and imu[ii, 0] <= ts:
            push_imu(imu[ii])
            ii += 1
        gray = np.ascontiguousarray(cache[k])
        fs, _ = tracker.process(gray)
        uvs = {int(f["id"]): [float(f["x"]), float(f["y"])] for f in fs}
        if not args.no_vision and (k - args.start) % args.vision_stride == 0:
            vio.process_vision(float(ts), uvs)
        if use_msceqf:
            p, _rm, _v = vio.nav_pose()
            p = np.asarray(p, float)
            # GRAVITY-LEAK AUDIT, no ground truth needed. Over a closed 9-minute loop the
            # long-time average of specific force MUST be vertical -- a sustained
            # horizontal acceleration is impossible. So rotating the measured accel into
            # the world frame with the filter's OWN attitude and averaging has to give
            # (0, 0, g). Any horizontal residual is gravity leaking through a tilt bias,
            # and it integrates straight into velocity: a mean residual `r` adds `r*T` of
            # speed, which is why a 0.1 deg tilt is worth metres per second over a ride.
            if ii > 0:
                grav_audit.append(np.asarray(_rm, float) @ imu[ii - 1, 4:7])
            bias_trace.append(np.asarray(vio.imu_bias(), float).copy())
            # Gyro bias resolved along WORLD VERTICAL vs perpendicular. The semi-direct
            # bias chart transports bias error as eps_b_dot = ad_psi eps_b, which
            # CONSERVES the component along the rotation axis while the perpendicular
            # part precesses away. A bike loop is a sequence of turns about gravity, so
            # that axis is the vertical one, and a conserved vertical gyro bias
            # integrates straight into yaw -- which does not leak gravity, but does wreck
            # loop closure, the one metric that has not improved in any arm.
            _R = np.asarray(_rm, float)
            _bg = np.asarray(vio.imu_bias(), float)[:3]
            _up = _R.T @ np.array([0.0, 0.0, 1.0])   # world vertical in BODY frame
            _par = float(_bg @ _up)
            yaw_trace.append((_par, float(np.linalg.norm(_bg - _par * _up))))
            _off, _acc = vio.msc_update_counts()
            diag.append((float(ts), float(np.linalg.norm(_v)),
                         float(np.asarray(vio.nav_cov_diag())[3:6].sum()),
                         vio.n_clones(), vio.n_slam(), float(_off), float(_acc),
                         float(len(uvs))))
        else:
            p, q = vio.get_pose()
            p = np.asarray(p, float)
        traj.append(p)
        stamps.append(ts)
        feats_per_frame[k] = uvs
        # Landmark RANGE from the camera: get_landmarks() is in the world frame.
        lms = None if use_msceqf else vio.get_landmarks()
        if use_msceqf:
            # Frame-clean anchor-frame range, not world-minus-body (which mixes frames).
            rr = {int(k): float(v) for k, v in vio.landmark_ranges().items()}
            if rr:
                lm_rng[k] = rr
            lms = {}
        if lms:
            rr = {}
            for lid, wp in lms.items():
                d = float(np.linalg.norm(np.asarray(wp, float) - p))
                if np.isfinite(d) and d > 1e-6:
                    rr[int(lid)] = d
            lm_rng[k] = rr
        if args.depth_trace and rr_all is not None and lm_rng.get(k):
            for lid, dd in lm_rng[k].items():
                rr_all.setdefault(lid, []).append((float(ts), dd))
        if (k - args.start) % 2000 == 0:
            print(f"  [{k - args.start}/{n}] feats {len(uvs)} in-state {len(lm_rng.get(k, {}))} "
                  f"|p| {np.linalg.norm(p):.2f} m")
    traj = np.array(traj)
    stamps = np.array(stamps)
    if diag:
        np.savez(args.out.replace(".mp4", "_diag.npz"), diag=np.array(diag),
                 grav=np.array(grav_audit) if grav_audit else np.zeros((0, 3)),
                 bias=np.array(bias_trace) if bias_trace else np.zeros((0, 6)),
                 yaw=np.array(yaw_trace) if yaw_trace else np.zeros((0, 2)))
    if use_msceqf:
        off, acc = vio.msc_update_counts()
        print(f"structureless tracks: offered {off}, accepted {acc} "
              f"({100.0*acc/max(off,1):.1f}%)")
        rj = vio.msc_reject_counts()
        names = ["too_few_obs", "stale_clone", "triangulation", "jacobian", "chi2"]
        print("  dropped by: " + "  ".join(
            f"{nm} {c} ({100.0*c/max(off,1):.1f}%)" for nm, c in zip(names, rj)))
        # A rejected track just outside the gate means the gate is too tight for an
        # honest innovation; one far outside means the covariance is wrong or the track
        # is a real outlier. The count alone cannot tell those apart.
        cr = np.asarray(vio.take_msc_chi2_ratios(), float)
        if len(cr):
            q = np.percentile(cr, [5, 25, 50, 75, 90, 95, 99])
            print(f"  chi2/gate over {len(cr)} gated tracks: "
                  f"median {np.median(cr):.2f}  mean {cr.mean():.2f}  "
                  "pct 5/25/50/75/90/95/99 " + "/".join(f"{v:.2f}" for v in q))
            for thr in (1.0, 2.0, 5.0, 20.0, 100.0):
                print(f"    > {thr:6.1f}x gate: {100.0*np.mean(cr > thr):5.1f}%")
        # Does the update ASK the velocity channel for anything, and does the channel
        # claim that ask is possible? maha >> 3 = corrections the covariance calls
        # impossible (channel throttled); maha << 3 = the updates are not asking.
        vc = np.asarray(vio.take_msc_vel_corr(), float)
        if len(vc):
            dv, sv, mh = vc[:, 0], vc[:, 1], vc[:, 2]
            q = lambda a: "/".join(f"{v:.3g}" for v in np.percentile(a, [5, 25, 50, 75, 95]))
            print(f"  velocity correction over {len(vc)} MSC updates, chart units:")
            print(f"    |dv| (m/s)   median {np.median(dv):.4g}  mean {dv.mean():.4g}  "
                  f"pct 5/25/50/75/95 {q(dv)}")
            print(f"    sigma_v      median {np.median(sv):.4g}  mean {sv.mean():.4g}  "
                  f"pct 5/25/50/75/95 {q(sv)}")
            print(f"    |dv|/sigma_v median {np.median(dv/np.maximum(sv,1e-12)):.3g}")
            good = np.isfinite(mh)
            if good.any():
                m = mh[good]
                print(f"    3-DOF Mahalanobis  median {np.median(m):.3g}  mean {m.mean():.3g}  "
                      f"pct 5/25/50/75/95 {q(m)}   (ideal median 2.37, chi2_3 95% = 7.81)")
                for thr in (7.81, 30.0, 100.0):
                    print(f"      > {thr:6.2f}: {100.0*np.mean(m > thr):5.1f}%")
        # K = cov[:, clones] C^T S^-1, so a core block uncorrelated with the clones
        # cannot be corrected at all. Where did the correction land, and could it have
        # landed anywhere else?
        cs = np.asarray(vio.take_msc_corr_split(), float)
        if len(cs):
            names = ["att", "vel", "pos", "bias", "E", "clones"]
            tot = cs[:, :6].sum(axis=1)
            print(f"  where the correction lands, {len(cs)} MSC updates (median share of "
                  "the per-update total):")
            for i, nm in enumerate(names):
                print(f"    {nm:7s} |d| median {np.median(cs[:, i]):.4g}   share "
                      f"{100.0*np.median(cs[:, i]/np.maximum(tot,1e-30)):5.1f}%")
            print("  largest |correlation| between each core block and one clone "
                  "(oldest = the anchor that carries the baseline; the newest clone is a "
                  "fresh copy of E, so its att/pos correlation is 1 by construction):")
            for i, nm in zip((6, 7, 8), ("att", "vel", "pos")):
                o, n_ = cs[:, i], cs[:, i + 3]
                print(f"    rho({nm:3s}) oldest median {np.median(o):.4f} "
                      "(5/95 " + "/".join(f"{v:.4f}" for v in np.percentile(o, [5, 95]))
                      + f")   newest median {np.median(n_):.4f}")
    if grav_audit:
        ga = np.array(grav_audit)
        m = ga.mean(axis=0)
        hor = float(np.linalg.norm(m[:2]))
        T = float(stamps[-1] - stamps[0])
        print(f"gravity-leak audit over {len(ga)} frames, {T:.1f} s:")
        print(f"  mean world-frame specific force [{m[0]:+.4f} {m[1]:+.4f} {m[2]:+.4f}] m/s^2"
              f"  (|vertical| should be {GRAVITY}, horizontal should be 0)")
        print(f"  horizontal residual {hor:.4f} m/s^2 -> {hor*T:.2f} m/s of speed over the "
              f"ride, and {0.5*hor*T*T:.0f} m of position")
        print(f"  implied tilt bias {np.degrees(np.arcsin(min(1.0, hor/GRAVITY))):.3f} deg")
        # A CONSTANT tilt set at initialisation and a tilt that DRIFTS during the ride
        # give the same mean, and call for completely different fixes, so bin it.
        nb = 14
        edges = np.linspace(0, len(ga), nb + 1).astype(int)
        print("  horizontal residual by time bin (m/s^2), and its direction (deg):")
        for b0, b1 in zip(edges[:-1], edges[1:]):
            if b1 - b0 < 10:
                continue
            mb = ga[b0:b1].mean(axis=0)
            hb = float(np.linalg.norm(mb[:2]))
            ang = np.degrees(np.arctan2(mb[1], mb[0]))
            t0b = float(stamps[b0]) - float(stamps[0])
            print(f"    t {t0b:6.1f} s  |h| {hb:.4f}  dir {ang:+7.1f}  "
                  f"tilt {np.degrees(np.arcsin(min(1.0, hb/GRAVITY))):.3f} deg")
    if bias_trace:
        bt = np.array(bias_trace)
        ba = bt[:, 3:]
        print("accel-bias estimate (seeded at ZERO; this rig's measured |g| is 1.43% high,"
              " = 0.14 m/s^2, which is 0.82 deg of apparent tilt if it stays unlearned):")
        for lab, sl in (("first 10 s", slice(0, min(len(ba), 620))),
                        ("last 10 s", slice(max(0, len(ba) - 620), len(ba)))):
            m = ba[sl].mean(axis=0)
            print(f"  {lab:10s} ba [{m[0]:+.4f} {m[1]:+.4f} {m[2]:+.4f}]  "
                  f"|ba| {np.linalg.norm(m):.4f}  |ba_horiz| {np.linalg.norm(m[:2]):.4f}")
        print(f"  |ba| over the run: median {np.median(np.linalg.norm(ba,axis=1)):.4f}  "
              f"max {np.linalg.norm(ba,axis=1).max():.4f} m/s^2")
    if yaw_trace:
        yt = np.array(yaw_trace)
        print("gyro bias resolved about WORLD VERTICAL (the axis a bike loop turns about;"
              " the semi-direct chart conserves the parallel component):")
        nb = 8
        ed = np.linspace(0, len(yt), nb + 1).astype(int)
        for b0, b1 in zip(ed[:-1], ed[1:]):
            if b1 - b0 < 10:
                continue
            print(f"    t {float(stamps[b0])-float(stamps[0]):6.1f} s  "
                  f"bg_parallel {yt[b0:b1, 0].mean():+.6f}  "
                  f"|bg_perp| {yt[b0:b1, 1].mean():.6f} rad/s")
        tot = float(stamps[-1] - stamps[0])
        print(f"  mean bg_parallel {yt[:,0].mean():+.6f} rad/s -> "
              f"{np.degrees(yt[:,0].mean()*tot):+.1f} deg of yaw if it integrates at unit "
              "gain")
    if args.zupt:
        print(f"ZUPT accepted on {vio.zupt_fired} IMU samples")
    print(f"run done in {time.perf_counter()-t0:.1f}s   path length "
          f"{np.sum(np.linalg.norm(np.diff(traj,axis=0),axis=1)):.1f} m")

    # Depth distribution of the IN-STATE landmarks. On an outdoor ride these must leave
    # the birth prior; if they do not, the colour map is showing the prior, not the scene.
    alld = np.array([d for fr in lm_rng.values() for d in fr.values()])
    if len(alld):
        q = np.percentile(alld, [1, 5, 25, 50, 75, 95, 99])
        print("in-state landmark RANGE (m) pct 1/5/25/50/75/95/99: "
              + "/".join(f"{v:.2f}" for v in q))
        ride = [d for k, fr in lm_rng.items() for d in fr.values()
                if stamps[0] + 200 <= img_t[k] <= stamps[0] + 480]
        if ride:
            rq = np.percentile(ride, [5, 50, 95])
            print(f"   during the RIDE only (200-480 s, 3.8-5.8 m/s): "
                  f"5/50/95 = {rq[0]:.2f}/{rq[1]:.2f}/{rq[2]:.2f} m  (n={len(ride)})")
    if rr_all is not None:
        long_lived = {k: v for k, v in rr_all.items() if len(v) >= 50}
        print(f"landmarks seen >=50 frames: {len(long_lived)} of {len(rr_all)}")
        if long_lived:
            spans = [(v[-1][1] / max(v[0][1], 1e-9)) for v in long_lived.values()]
            print(f"   last/first range ratio: median {np.median(spans):.3f}, "
                  f"5-95 pct {np.percentile(spans,5):.3f}-{np.percentile(spans,95):.3f} "
                  "(1.0 = never moved off the prior)")
        np.savez(args.depth_trace,
                 **{str(k): np.array(v) for k, v in list(long_lived.items())[:400]})

    # --- control: the trajectory the bag already carries -------------------
    ctrl = None
    if len(odom):
        m = (odom[:, 0] >= stamps[0]) & (odom[:, 0] <= stamps[-1])
        if m.sum() > 10:
            ctrl = odom[m, 1:4]
            ctrl = ctrl - ctrl[0]
    np.savez(args.out.replace(".mp4", "_traj.npz"), traj=traj, stamps=stamps,
             ctrl=(ctrl if ctrl is not None else np.zeros((0, 3))))

    if args.no_render:
        print("--no-render: stopping after the filter pass")
        return

    # --- render ------------------------------------------------------------
    jetr = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[::-1, 0]
    allr = np.array([r for d in lm_rng.values() for r in d.values()])
    vlo, vhi = (np.percentile(allr, 5), np.percentile(allr, 95)) if len(allr) else (0.5, 10.0)
    print(f"range colour scale: {vlo:.2f}-{vhi:.2f} m over {len(allr)} landmark-frames")

    RW, SC = 420, 3
    pts = traj[:, :2]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max((hi - lo).max(), 1e-3) * 1.15
    mid = (hi + lo) / 2
    def P(xy):
        u = (xy - mid) / span + 0.5
        return int(u[0] * (RW - 20)) + 10, int((1 - u[1]) * (H * SC - 20)) + 10

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (W * SC + RW, H * SC))
    base = np.full((H * SC, RW, 3), 28, np.uint8)
    if ctrl is not None and len(ctrl) > 1:
        cp = [P(c[:2]) for c in ctrl]
        cv2.polylines(base, [np.array(cp, np.int32)], False, (105, 105, 105), 1, cv2.LINE_AA)
    rendered = 0
    for k in range(args.start, args.start + n, args.stride):
        left = cv2.cvtColor(cv2.resize(np.asarray(cache[k]), (W * SC, H * SC),
                                       interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        rr = lm_rng.get(k, {})
        nn = ns = 0
        for fid, (u, v) in feats_per_frame.get(k, {}).items():
            x, y = int(u * SC), int(v * SC)
            if fid in rr:
                col = clut(jetr, (rr[fid] - vlo) / (vhi - vlo + 1e-9))
                cv2.circle(left, (x, y), 5, col, -1, cv2.LINE_AA)
                cv2.circle(left, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
                ns += 1
            else:
                cv2.circle(left, (x, y), 4, (150, 150, 150), 1, cv2.LINE_AA)
                nn += 1
        j = k - args.start
        txt(left, f"t {stamps[j]-stamps[0]:6.1f}s   tracked(o) {nn}   in-state(*) {ns}", (8, 20))
        txt(left, f"range {vlo:.1f}-{vhi:.1f} m (jet_r: near=red)", (8, 38))
        right = base.copy()
        cv2.polylines(right, [np.array([P(p[:2]) for p in traj[:j + 1]], np.int32)],
                      False, (90, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(right, P(traj[j][:2]), 4, (0, 255, 255), -1, cv2.LINE_AA)
        txt(right, "estimate (this run)", (10, 20), (90, 200, 255))
        if ctrl is not None:
            txt(right, "bag /echo_li/odometry (control)", (10, 38), (160, 160, 160))
        txt(right, f"path {np.sum(np.linalg.norm(np.diff(traj[:j+1],axis=0),axis=1)):.1f} m", (10, 56))
        vw.write(np.hstack([left, right]))
        rendered += 1
    vw.release()
    print(f"wrote {args.out}  ({rendered} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
