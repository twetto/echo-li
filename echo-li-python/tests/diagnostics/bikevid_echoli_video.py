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

Any rig bag works through `--bag-cache DIR` (frames320.bin + frames320_index.npy from
the same decode, sensors.npz from the extractor with HEADER image stamps), or simply by
name with `--bag vio_live_20260925-055046`: the cache is looked up under `--bag-root`
and, when it is not there yet, extracted from the rosbag2 under `--bag-src` by running
extract_rigbag.py inside the ROS 2 distrobox (`--distrobox`). `--bag bikevid` is the
bike recording. With `--bag`, `--out` defaults to <bag>.mp4 in the working directory
and `--config` to the rig MSCEqF + Sparse3D config kept next to the caches. With
`--sparse3d` the frame's tracks also go through the shipped `VIOFilter.sparse_feed`
seam (CloneRelFeed: clone-relative covariances, re-anchoring) into a Sparse3D filter
built from the config's `SparseVog` section; its CONVERGED landmarks are drawn as
hollow circles next to the in-state dots. The estimate is SE(3)-aligned (Umeyama, no
scale) to the recorded odometry over matched stamps, and the right panel colours it by
the residual against that control -- which is an on-rig EqVIO run, not ground truth.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import echo_li  # noqa: E402
import fdf_densify_study as dz  # noqa: E402
from euroc_msceqf_native_ate import align_umeyama  # noqa: E402
from midair_e2e_depth_nees import SPARSE_KEYS  # noqa: E402

BIKE = "/mnt/18TB/chen_fu_yeh/bikevid"
# Rig recordings: raw rosbag2 directories, and the extracted caches the harness reads.
RIG_BAG_SRC = "/mnt/18TB/chen_fu_yeh/bagmkv/bags"
RIG_BAG_ROOT = "/mnt/18TB/chen_fu_yeh/bagmkv/rig_offline/bags"
RIG_CONFIG = "/mnt/18TB/chen_fu_yeh/bagmkv/rig_offline/rig_sparse.yaml"
W, H = 320, 200
# calib_final-camchain-imucam.yaml, scaled by 0.25 for the raw_scale_denom=4 stream.
FX, FY, CX, CY = 226.7062860357, 226.1687971677, 150.0624337420, 99.3898662352
DIST = (0.046897611239, -0.062989153081, -0.001026069440, -0.001348024507)
CAM_MODEL = "radtan"              # or "equidistant" (Kannala-Brandt k1..k4), from --calib camera_model
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


def ensure_bag_cache(name, root, src, distrobox):
    """Return the cache directory for rig bag `name`, extracting it first if needed.

    Extraction decodes the rosbag2 with rclpy, which only exists inside the ROS 2
    distrobox, so it is run there; the harness itself stays in the venv.
    """
    cache = os.path.join(root, name)
    need = ["sensors.npz", "frames320.bin", "frames320_index.npy"]
    if all(os.path.exists(os.path.join(cache, f)) for f in need):
        return cache
    bagdir = os.path.join(src, name)
    if not os.path.isdir(bagdir):
        sys.exit(f"bag '{name}': no cache under {root} and no rosbag2 directory {bagdir}")
    extractor = str(Path(__file__).resolve().parent / "extract_rigbag.py")
    cmd = ["distrobox", "enter", distrobox, "--", "bash", "-c",
           f". /opt/ros/humble/setup.bash && python3 {extractor} {bagdir} {cache}"]
    print(f"extracting {name} -> {cache}\n  " + " ".join(cmd), flush=True)
    import subprocess
    rc = subprocess.call(cmd)
    if rc != 0 or not all(os.path.exists(os.path.join(cache, f)) for f in need):
        sys.exit(f"extraction of '{name}' failed (exit {rc})")
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default="",
                    help="rig bag by name (directory name under --bag-src, e.g. "
                         "vio_live_20260925-055046), or 'bikevid'. Replaces --sensors/--bag-cache.")
    ap.add_argument("--bag-root", default=RIG_BAG_ROOT,
                    help="where extracted caches live: <bag-root>/<bag>/{sensors.npz,frames320.bin,"
                         "frames320_index.npy}")
    ap.add_argument("--bag-src", default=RIG_BAG_SRC,
                    help="where the raw rosbag2 directories live (extracted on first use)")
    ap.add_argument("--distrobox", default="ubuntu-22-04",
                    help="distrobox with ROS 2 humble + rclpy, used only to extract a missing cache")
    ap.add_argument("--calib", default="",
                    help="rig calibration in the ROS 2 node's format (echo-li-ros2/config/myrig_calib.yaml: "
                         "intrinsics, distortion_coefficients, t_bs, camera_time_offset_sec at the "
                         "source image_width; camera_model radtan or equidistant). Intrinsics are "
                         "scaled to the 320-wide cache. Default: "
                         "the first rig's constants baked into this file. Needed for any other rig.")
    ap.add_argument("--sensors", default="", help="bike_sensors.npz from the extractor (or use --bag)")
    ap.add_argument("--config", default="",
                    help=f"run config; with --bag defaults to {RIG_CONFIG}")
    ap.add_argument("--out", default="", help="output mp4; with --bag defaults to <bag>.mp4")
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
    ap.add_argument("--bag-cache", default=BIKE,
                    help="directory holding frames320.bin + frames320_index.npy for the "
                         "bag named by --sensors (default: the bikevid directory)")
    ap.add_argument("--sparse3d", action="store_true",
                    help="run Sparse3D on every frame's tracks through VIOFilter.sparse_feed "
                         "(the shipped CloneRelFeed path) and draw its converged landmarks "
                         "as hollow circles")
    ap.add_argument("--patch-mapper", action="store_true",
                    help="run the shipped PatchDepthMapper offline on each frame, seeded exactly as "
                         "the ROS2 node seeds it (Sparse3D converged landmarks with eta_var = var/r^2, "
                         "then in-state landmarks not covered by Sparse3D at eta_var 0.25), in THREE "
                         "instances: both sources / Sparse3D only / in-state only. Status map and "
                         "seed sources are overlaid on the left panel; per-frame pixel counts by "
                         "status, SeedOnly disc components and the pixels only the in-state seeds "
                         "produce go to <out>_mapper.npz. Needs --sparse3d.")
    ap.add_argument("--track-source", default="rudolf", choices=["rudolf", "fdf", "dense"],
                    help="fdf: ALSO run FlowDep-F (anchor-referenced epipolar patch tracker) on the "
                         "fisheye-undistorted frame, driven by the backend's pose, feeding a second "
                         "Sparse3D (pinhole camera) through the same sparse_feed; the rendered video "
                         "gains a second row with the densified range (rule of the design note row "
                         "5b: nearest patch + disagreement + footprint term) and its per-pixel sigma. "
                         "The pose source still runs on Rudolf-V. Needs --sparse3d. Dense frames are "
                         "kept in memory for rendered frames only (~130 kB each).")
    ap.add_argument("--fdf", action="append", default=[], metavar="KEY=VAL", help="FlowDepF setting override")
    ap.add_argument("--slant-deg", type=float, default=35.0, help="footprint slant prior of the densify rule")
    ap.add_argument("--fill", action="store_true",
                    help="add the FILLED class to the dense row (design note rows 5k/5l): normalised-convolution "
                         "interpolation between measured patches, kept only where evidence surrounds the pixel "
                         "(balance gate), drawn at lower opacity; sigma = nearest measured + (2 g0 d)^2")
    ap.add_argument("--fdf-sparse", action="append", default=[], metavar="KEY=VAL",
                    help="Sparse3D override for the FlowDep-F arm only (e.g. bias_walk_var=0.3, design note row 5h)")
    ap.add_argument("--fdf-clahe", action="store_true", help="same as --fdf-input clahe")
    ap.add_argument("--fdf-input", default="raw", choices=["raw", "clahe", "global", "gain"],
                    help="intensity map applied to the undistorted frame FlowDep-F sees. raw: none (on the rig's "
                         "dim frames the per-cell curvature median is 4-6 against a birth gate of 25); clahe: the "
                         "front-end config's CLAHE (local, frame-dependent map); global: cv2.equalizeHist (one "
                         "monotone curve per frame); gain: linear stretch of the 1st..99th percentile to 0..255 "
                         "(one affine map per frame, the most photometrically consistent)")
    ap.add_argument("--sparse-config", default="",
                    help="YAML whose SparseVog section configures Sparse3D (default: the "
                         "run config)")
    args = ap.parse_args()

    if args.bag:
        if args.bag == "bikevid":
            args.bag_cache = BIKE
            args.sensors = os.path.join(BIKE, "bike_sensors_hdr.npz")
        else:
            args.bag_cache = ensure_bag_cache(args.bag, args.bag_root, args.bag_src, args.distrobox)
            args.sensors = os.path.join(args.bag_cache, "sensors.npz")
        args.out = args.out or f"{args.bag}.mp4"
        args.config = args.config or RIG_CONFIG
        print(f"bag {args.bag}: cache {args.bag_cache}  config {args.config}  out {args.out}")
    if not (args.sensors and args.config and args.out):
        ap.error("either --bag NAME, or all of --sensors, --config and --out")

    global TIMESHIFT, FX, FY, CX, CY, DIST, T_BS, CAM_MODEL
    if args.calib:
        with open(args.calib) as fh:
            cal = next(iter(yaml.safe_load(fh).values()))["ros__parameters"]
        sc = W / float(cal["image_width"])
        FX, FY, CX, CY = (float(v) * sc for v in cal["intrinsics"])
        DIST = tuple(float(v) for v in cal["distortion_coefficients"])
        CAM_MODEL = cal.get("camera_model", "radtan").lower()
        if CAM_MODEL not in ("radtan", "equidistant"):
            sys.exit(f"--calib camera_model '{CAM_MODEL}': this harness handles radtan or equidistant")
        T_BS = np.array(cal["t_bs"], float).reshape(4, 4)
        TIMESHIFT = float(cal.get("camera_time_offset_sec", TIMESHIFT))
        print(f"calibration {args.calib}: source {cal['image_width']}x{cal['image_height']} -> x{sc:.3g}  "
              f"{CAM_MODEL}  fx {FX:.3f} fy {FY:.3f} cx {CX:.3f} cy {CY:.3f}  dist {DIST}  timeshift {TIMESHIFT*1e3:.3f} ms\n"
              f"  T_BS\n{np.array2string(T_BS, precision=6, suppress_small=True)}")
    if args.timeshift is not None:
        TIMESHIFT = args.timeshift
        print(f'camera-IMU timeshift overridden -> {TIMESHIFT*1e3:.2f} ms')
    z = np.load(args.sensors)
    imu, img_t, odom = z["imu"], z["img_t"], z["odom"]
    idx = np.load(f"{args.bag_cache}/frames320_index.npy")
    cache = np.memmap(f"{args.bag_cache}/frames320.bin", dtype=np.uint8, mode="r").reshape(-1, H, W)
    n_all = min(len(idx), len(cache), len(img_t))
    # Some bags start the camera before the IMU; the filter cannot see a frame that
    # precedes its 100-sample init window, so start at the first one that follows it.
    if img_t[args.start] + TIMESHIFT < imu[100, 0]:
        k0 = int(np.searchsorted(img_t + TIMESHIFT, imu[100, 0]))
        print(f"first {k0} frames precede the IMU init window (camera starts "
              f"{imu[0, 0] - img_t[0]:.2f} s before the IMU): starting at frame {k0}")
        args.start = k0
    n = n_all - args.start if args.nframes <= 0 else min(args.nframes, n_all - args.start)
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

    cam_cls = echo_li.EquidistantCamera if CAM_MODEL == "equidistant" else echo_li.RadTanCamera
    cam = cam_cls(FX, FY, CX, CY, *DIST)
    fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
    fcfg.set_camera(FX, FY, CX, CY, W, H, list(DIST), CAM_MODEL)
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
    elif "MSCEqF" in full_cfg:
        # --backend eqvio must win over a config that carries an enabled MSCEqF section,
        # otherwise the "EqVIO arm" silently runs MSCEqF (it did, 2026-10-15).
        full_cfg["MSCEqF"] = dict(full_cfg["MSCEqF"], enabled=False)
    if args.zupt:
        full_cfg["ZUPT"] = {"enabled": True, "window": 25, "gyroStdThreshold": 0.05,
                            "accStdThreshold": 0.15, "sigmaV": args.zupt_sigma}
    run_cfg = args.out.replace(".mp4", "_config.yaml")
    with open(run_cfg, "w") as fh:
        yaml.safe_dump(full_cfg, fh, sort_keys=False)
    vio = echo_li.VIOFilter(run_cfg, cam, fx=FX)
    vio.set_camera_extrinsics(np.ascontiguousarray(T_BS))
    print(f"backend={vio.backend_kind}  zupt={bool(args.zupt)}  config={run_cfg}")

    sparse = None
    fdf = None; sparse_fdf = None; undist_maps = None
    dense = None; dense_last = {}   # --track-source dense: the node's DenseDepthMapper (Rust), the wired path
    dense_frames = {}      # k -> (undistorted gray, eta_p f16, sigma f16, covered) for rendered frames
    fdf_rng = {}           # k -> {fid: (u, v, range, converged)} on the pinhole frame
    fdf_stat = [0, 0]      # converged landmark-frames: fdf, klt (for the summary)
    if args.sparse3d and args.track_source == "fdf":
        pass
    if args.sparse3d:
        with open(args.sparse_config or cfg_path) as fh:
            scfg = yaml.safe_load(fh).get("SparseVog") or {}
        settings = {k: scfg[k] for k in SPARSE_KEYS if k in scfg}
        settings["parametrization"] = "bearing_invdepth_additive3d"
        # sigma_s(E) from the IMU excitation over the run, the same operational rule
        # the Kite harness applies (`--scale-sigma-auto`): E = std(w_z) * std(a_y) on
        # the raw IMU, body axes as recorded.
        _seg = imu[(imu[:, 0] >= img_t[args.start]) & (imu[:, 0] <= img_t[args.start + n - 1])]
        _E = float(np.std(_seg[:, 3]) * np.std(_seg[:, 5])) if len(_seg) > 10 else 0.0
        settings["scale_sigma"] = max(0.02, 0.0072 * _E ** (-0.710)) if _E > 0 else 0.02
        sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)
        if args.track_source == "fdf":
            # FlowDep-F is pinhole-only: undistort every frame to a pinhole with the SAME
            # fx fy cx cy (pixels outside the fisheye's valid region stay black and are
            # never born: no gradient). The second Sparse3D sees those pinhole pixels.
            _K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)
            if CAM_MODEL == "equidistant":
                _m1, _m2 = cv2.fisheye.initUndistortRectifyMap(_K, np.array(DIST, float), np.eye(3), _K, (W, H), cv2.CV_32FC1)
            else:
                _m1, _m2 = cv2.initUndistortRectifyMap(_K, np.array(DIST, float), np.eye(3), _K, (W, H), cv2.CV_32FC1)
            undist_maps = (_m1, _m2)
            _fs = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
            fdf = echo_li.FlowDepF(FX, FY, CX, CY, W, H, _fs or None)
            _ps = dict(settings); _ps["max_pool_size"] = max(int(_ps.get("max_pool_size", 300)), 4000)
            for _kv in args.fdf_sparse:
                _k, _v = _kv.split("=", 1)
                _cur = _ps.get(_k, scfg.get(_k, 0.0))
                _ps[_k] = (_v.lower() == "true") if isinstance(_cur, bool) else type(_cur)(_v) if isinstance(_cur, (int, float)) else float(_v)
                print(f"FlowDep-F Sparse3D override {_k} = {_ps[_k]!r}")
            sparse_fdf = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(echo_li.PinholeCamera(FX, FY, CX, CY), **_ps)
            dz.SLANT_G[0] = np.tan(np.radians(args.slant_deg)) / FX
            if args.fdf_clahe:
                args.fdf_input = "clahe"
            fdf_map = None
            if args.fdf_input == "clahe":
                with open(cfg_path) as fh:
                    _rv = (yaml.safe_load(fh) or {}).get("RudolfV", {}) or {}
                _tile = int(_rv.get("claheTileSize", 64)); _clip = float(_rv.get("claheClipLimit", 4.0))
                _cl = cv2.createCLAHE(clipLimit=_clip, tileGridSize=(max(1, W // _tile), max(1, H // _tile)))
                fdf_map = _cl.apply
                print(f"FlowDep-F input: CLAHE tile {_tile} px (grid {max(1, W // _tile)}x{max(1, H // _tile)}) clip {_clip}")
            elif args.fdf_input == "global":
                fdf_map = cv2.equalizeHist
                print("FlowDep-F input: global histogram equalisation")
            elif args.fdf_input == "gain":
                def fdf_map(im):
                    lo_, hi_ = np.percentile(im, (1, 99))
                    return np.clip((im.astype(np.float32) - lo_) * (255.0 / max(hi_ - lo_, 1.0)), 0, 255).astype(np.uint8)
                print("FlowDep-F input: global 1..99 percentile stretch")
            print(f"FlowDep-F on the undistorted pinhole frame, settings {_fs}; second Sparse3D (pinhole, pool {_ps['max_pool_size']}); densify slant {args.slant_deg:.0f} deg")
        elif args.track_source == "dense":
            # The node's path end to end: echo_li.DenseDepthMapper equalises, undistorts and
            # tracks the raw cache frame itself, runs its own clone-relative Sparse3D, fuses and
            # fills in Rust, and is told the corner tracker's features so it births none there.
            _fs = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
            _ps = dict(settings); _ps["max_pool_size"] = max(int(_ps.get("max_pool_size", 300)), 4000)
            for _kv in args.fdf_sparse:
                _k, _v = _kv.split("=", 1)
                _cur = _ps.get(_k, scfg.get(_k, 0.0))
                _ps[_k] = (_v.lower() == "true") if isinstance(_cur, bool) else type(_cur)(_v) if isinstance(_cur, (int, float)) else float(_v)
            with open(cfg_path) as fh:
                _rv = (yaml.safe_load(fh) or {}).get("RudolfV", {}) or {}
            _tile = int(_rv.get("claheTileSize", 64)); _clip = float(_rv.get("claheClipLimit", 4.0))
            if args.fdf_clahe:
                args.fdf_input = "clahe"
            _he = args.fdf_input if args.fdf_input in ("clahe", "global") else "none"
            dense = echo_li.DenseDepthMapper(cam, FX, FY, CX, CY, W, H, _ps, tracker=_fs or None, scale=1.0,
                                             histeq=_he, clahe_tile_size=_tile, clahe_clip_limit=_clip,
                                             slant_deg=args.slant_deg, fill=bool(args.fill), exclude_corners=True)
            dz.SLANT_G[0] = np.tan(np.radians(args.slant_deg)) / FX
            print(f"DenseDepthMapper (the node's path, Rust): histeq {_he} tile {_tile} clip {_clip}; tracker {_fs}; "
                  f"Sparse3D pool {_ps['max_pool_size']} walk {_ps.get('bias_walk_var')}; fill {bool(args.fill)}; births excluded on corners")
        print(f"Sparse3D via VIOFilter.sparse_feed: direct={settings.get('pose_range_direct', False)} "
              f"consider={settings.get('consider_relpose', False)} "
              f"bias_walk_var={settings.get('bias_walk_var', 0.0)} "
              f"E={_E:.4f} -> scale_sigma={settings['scale_sigma']:.4f}  "
              f"min_track_length={settings.get('min_track_length', '?')}")

    mappers = {}
    mp_rows = []              # per frame: see MP_COLS
    mp_status = {}            # k -> (status_both, status_s3d) uint8 maps on render frames
    mp_seeds = {}             # k -> (set of Sparse3D seed ids, set of in-state seed ids)
    MP_COLS = ("k", "n_s3d_seeds", "n_lm_seeds",
               "both_seedonly", "both_photo", "both_rej", "both_cc", "both_cc_med",
               "s3d_seedonly", "s3d_photo", "s3d_rej", "lm_seedonly", "lm_photo", "lm_rej",
               "only_from_lm", "only_from_s3d",
               # no GT on the rig, so two proxies for false depth: FLICKER = covered pixels
               # whose log-range jumps > 0.3 (35 %) from the previous mapper frame, split by
               # whether the in-state seeds alone produce them; and seeds with a neighbour
               # inside the seed radius whose range differs > 2x (a disc straddling a
               # discontinuity), per seed source.
               "flicker_px", "flicker_px_only_lm", "covered_prev_and_now",
               "s3d_seeds_straddle", "lm_seeds_straddle")
    mp_prev = {"eta": None, "only_lm": None}
    if args.patch_mapper:
        if sparse is None:
            sys.exit("--patch-mapper needs --sparse3d")
        with open(run_cfg) as fh:
            _mcfg = yaml.safe_load(fh)
        _pd = dict(_mcfg.get("PatchDepth") or {})
        # The rig config is written for the 1280x800 stream and downscales by `scale`;
        # the cache already holds the quarter-resolution frames, so run at scale 1 and
        # shrink the seed radius by the same factor (the mapper multiplies it by scale).
        _native_scale = float(_pd.get("scale", 1.0))
        _pd["scale"] = 1.0
        _pd["seed_radius_px"] = float(_pd.get("seed_radius_px", 32.0)) * _native_scale
        _mcfg["PatchDepth"] = _pd
        _mcfg_path = args.out.replace(".mp4", "_mapper_cfg.yaml")
        with open(_mcfg_path, "w") as fh:
            yaml.safe_dump(_mcfg, fh, sort_keys=False)
        for _name in ("both", "s3d", "lm"):
            mappers[_name] = echo_li.PatchDepthMapper(cam, FX, FY, CX, CY, W, H, config=_mcfg_path)
        print(f"PatchDepthMapper x3 (both / s3d / lm): {_pd}  seed coords={mappers['both'].seed_coordinates}")

    def push_imu(row):
        vio.process_imu(float(row[0]), row[1:4].tolist(), row[4:7].tolist())

    # The initialiser buffers 100 samples, gravity-aligns, seeds the gyro bias if the
    # window was stationary, rebuilds the backend at that pose and replays the buffer.
    for j in range(i0 - N_INIT, i0):
        push_imu(imu[j])

    traj, stamps, feats_per_frame, lm_rng = [], [], {}, {}
    sp_rng, sp_stats = {}, [0, 0, 0]       # per-frame Sparse3D landmarks; reanchored, n_rel, clone-relative frames
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
            if sparse is not None:
                # World camera pose for Sparse3D: body pose from the backend, camera
                # offset from the SAME extrinsics the filter was given.
                _pp, _qq = vio.get_pose()
                _T = np.eye(4)
                _T[:3, :3] = Rot.from_quat(np.asarray(_qq, float)).as_matrix()
                _T[:3, 3] = np.asarray(_pp, float)
                _rep = vio.sparse_feed(sparse, float(ts), uvs, (_T @ T_BS).tolist())
                sp_stats[0] += int(_rep[1]); sp_stats[1] += int(_rep[2]); sp_stats[2] += int(_rep[0])
                _d = {}
                for _fid, _fd in sparse.get_features().items():
                    _r = float(np.linalg.norm(np.asarray(_fd["position"], float)))
                    if not (np.isfinite(_r) and _r > 1e-6):
                        continue
                    _rv = float(_fd["range_var_reported"])
                    _conv = sparse.query_range(int(_fid))[0] > 0.0
                    _d[int(_fid)] = (_r, float(np.sqrt(max(_rv, 0.0))), bool(_conv))
                sp_rng[k] = _d
            if fdf is not None:
                # FlowDep-F on the undistorted frame with the backend's pose of this frame
                # (the same body pose @ T_BS the KLT Sparse3D just got).
                _ug = cv2.remap(gray, undist_maps[0], undist_maps[1], cv2.INTER_LINEAR)
                if fdf_map is not None:
                    _ug = fdf_map(_ug)
                _T = np.eye(4)
                _pp, _qq = vio.get_pose()
                _T[:3, :3] = Rot.from_quat(np.asarray(_qq, float)).as_matrix(); _T[:3, 3] = np.asarray(_pp, float)
                _Twc = _T @ T_BS
                _uvf = fdf.process(np.ascontiguousarray(_ug), _Twc.tolist())
                vio.sparse_feed(sparse_fdf, float(ts), _uvf, _Twc.tolist())
                _post = {}; _items = []; _dd = {}
                _lev = {o[0]: o[3] for o in fdf.last_observations()}
                for _fid, _fd in sparse_fdf.get_features().items():
                    _est = np.asarray(_fd["position"], float); _cov = np.asarray(_fd["covariance_euclidean"], float)
                    _r = float(np.linalg.norm(_est))
                    if not (np.isfinite(_r) and _r > 1e-6 and np.isfinite(_cov).all()):
                        continue
                    _rh = _est / _r; _vr = float(_rh @ _cov @ _rh)
                    if _vr > 0:
                        _post[int(_fid)] = (_r, float(np.sqrt(_vr)))
                    _conv = sparse_fdf.query_range(int(_fid))[0] > 0.0
                    if _fid in _uvf:
                        _dd[int(_fid)] = (_uvf[_fid][0], _uvf[_fid][1], _r, bool(_conv))
                        if _conv and _vr > 0:
                            _vrep = float(_fd["range_var_reported"]) / (_r * _r)
                            _vc = _vr / (_r * _r)
                            _items.append((_uvf[_fid][0], _uvf[_fid][1], _lev.get(_fid, 0), np.log(_r), _vc, float(np.sqrt(max(_vrep - _vc, 0.0)))))
                if _post:
                    fdf.set_posterior(_post)
                fdf_rng[k] = _dd
                if os.environ.get("DBG_FDF") and (k - args.start) % 100 == 0:
                    _rf = [v[2] for v in _dd.values() if v[3]]; _rk = [v[0] for v in sp_rng.get(k, {}).values() if v[2]]
                    _ra = [v[2] for v in _dd.values()]
                    print(f"  DBG_FDF k={k} fdf conv {len(_rf)} med range {np.median(_rf) if _rf else 0:.2f} (all tracked {len(_ra)} med {np.median(_ra) if _ra else 0:.2f})  klt conv {len(_rk)} med {np.median(_rk) if _rk else 0:.2f}  stats {fdf.stats()}", file=sys.stderr)
                fdf_stat[0] += sum(1 for v in _dd.values() if v[3]); fdf_stat[1] += sum(1 for v in sp_rng.get(k, {}).values() if v[2])
                if (k - args.start) % args.stride == 0 and not args.no_render:
                    if _items:
                        _eta, _var, _cov_m = dz.fuse(_items, H, W)["nearest_disagree_slant"]
                    else:
                        _eta = np.zeros((H, W)); _var = np.ones((H, W)); _cov_m = np.zeros((H, W), bool)
                    _cls = _cov_m.astype(np.uint8)                       # 1 = Measured, 2 = Filled, 0 = Unknown
                    if args.fill and _cov_m.any():
                        _ef, _vf, _filled, _ = dz.fill(_eta, _var, _cov_m)
                        _eta = np.where(_filled, _ef, _eta); _var = np.where(_filled, _vf, _var)
                        _cls[_filled] = 2
                    dense_frames[k] = (_ug.copy(), _eta.astype(np.float16), np.sqrt(np.maximum(_var, 1e-12)).astype(np.float16), _cls)
            elif dense is not None:
                _T = np.eye(4)
                _pp, _qq = vio.get_pose()
                _T[:3, :3] = Rot.from_quat(np.asarray(_qq, float)).as_matrix(); _T[:3, 3] = np.asarray(_pp, float)
                _Twc = _T @ T_BS
                _corn = [(float(v[0]), float(v[1])) for v in uvs.values()]
                _eta, _var, _cls, _st = dense.process(np.ascontiguousarray(gray), float(ts), np.ascontiguousarray(_Twc), vio, _corn)
                dense_last = dict(_st)
                fdf_rng[k] = {}
                fdf_stat[0] += int(_st["landmarks_converged"]); fdf_stat[1] += sum(1 for v in sp_rng.get(k, {}).values() if v[2])
                if os.environ.get("DBG_FDF") and (k - args.start) % 100 == 0:
                    print(f"  DBG_DENSE k={k} {_st}", file=sys.stderr)
                if (k - args.start) % args.stride == 0 and not args.no_render:
                    dense_frames[k] = (np.asarray(dense.last_frame()).copy(), _eta.astype(np.float16),
                                       np.sqrt(np.maximum(_var.astype(np.float64), 1e-12)).astype(np.float16), _cls.astype(np.uint8))
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
        if mappers and not args.no_vision and (k - args.start) % args.vision_stride == 0:
            # Seeds exactly as voxl2_vio_node.rs builds its priors (raw pixels of this
            # frame's features): Sparse3D converged landmarks first, eta_var = var/r^2;
            # then in-state landmarks not already covered, eta_var 0.25, r >= 0.1.
            _sp = sp_rng.get(k, {})
            _lm = lm_rng.get(k, {})
            pri_s3d, pri_lm, ids_s3d, ids_lm = [], [], set(), set()
            for _fid, (_u, _v) in uvs.items():
                if _fid in _sp and _sp[_fid][2]:
                    _r, _sig, _ = _sp[_fid]
                    pri_s3d.append((_u, _v, float(np.log(_r)), float(_sig * _sig / (_r * _r)) if _r > 0.01 else 1.0))
                    ids_s3d.add(_fid)
            for _fid, (_u, _v) in uvs.items():
                if _fid in ids_s3d or _fid not in _lm or not (_lm[_fid] >= 0.1):
                    continue
                pri_lm.append((_u, _v, float(np.log(_lm[_fid])), 0.25))
                ids_lm.add(_fid)
            _pp, _qq = vio.get_pose()
            _Twb = np.eye(4)
            _Twb[:3, :3] = Rot.from_quat(np.asarray(_qq, float)).as_matrix()
            _Twb[:3, 3] = np.asarray(_pp, float)
            _Twc = (_Twb @ T_BS).tolist()
            outs = {}
            for _name, _pri in (("both", pri_s3d + pri_lm), ("s3d", pri_s3d), ("lm", pri_lm)):
                outs[_name] = mappers[_name].update(float(ts), int(k), gray, _Twc, _pri)
            def _cnt(o):
                if o is None:
                    return (0, 0, 0), None
                st = o["status"]
                return (int((st == 1).sum()), int((st == 2).sum()), int((st == 3).sum())), st
            (b1, b2, b3), st_b = _cnt(outs["both"])
            (s1, s2, s3), st_s = _cnt(outs["s3d"])
            (l1, l2, l3), st_l = _cnt(outs["lm"])
            n_cc, cc_med = 0, 0.0
            if st_b is not None and b1 > 0:
                _n, _, _stats, _ = cv2.connectedComponentsWithStats((st_b == 1).astype(np.uint8), connectivity=8)
                n_cc = int(_n - 1)
                cc_med = float(np.median(_stats[1:, cv2.CC_STAT_AREA])) if _n > 1 else 0.0
            only_lm = only_s3d = 0
            if st_b is not None:
                _cov_b = st_b > 0
                _cov_s = (st_s > 0) if st_s is not None else np.zeros_like(_cov_b)
                _cov_l = (st_l > 0) if st_l is not None else np.zeros_like(_cov_b)
                only_lm = int((_cov_b & ~_cov_s).sum())    # pixels the in-state seeds alone bring
                only_s3d = int((_cov_b & ~_cov_l).sum())   # pixels the Sparse3D seeds alone bring
            flick = flick_lm = both_cov = 0
            if st_b is not None:
                _eta = outs["both"]["eta"]
                _only = (st_b > 0) & ~((st_s > 0) if st_s is not None else np.zeros_like(st_b, bool))
                if mp_prev["eta"] is not None:
                    _c = np.isfinite(_eta) & np.isfinite(mp_prev["eta"])
                    _j = _c & (np.abs(_eta - mp_prev["eta"]) > 0.3)
                    both_cov = int(_c.sum()); flick = int(_j.sum()); flick_lm = int((_j & _only).sum())
                mp_prev["eta"] = _eta.copy(); mp_prev["only_lm"] = _only
            _rad = float(_pd["seed_radius_px"])
            def _straddle(pri):
                if len(pri) < 2:
                    return 0
                _a = np.array([(u, v, e) for u, v, e, _ in pri]); n_ = 0
                for i in range(len(_a)):
                    d = np.hypot(_a[:, 0] - _a[i, 0], _a[:, 1] - _a[i, 1])
                    nb = (d < _rad) & (d > 0)
                    if nb.any() and np.abs(_a[nb, 2] - _a[i, 2]).max() > np.log(2.0):
                        n_ += 1
                return n_
            mp_rows.append((k, len(pri_s3d), len(pri_lm), b1, b2, b3, n_cc, cc_med,
                            s1, s2, s3, l1, l2, l3, only_lm, only_s3d,
                            flick, flick_lm, both_cov, _straddle(pri_s3d + pri_lm) - _straddle(pri_lm) if False else _straddle(pri_s3d), _straddle(pri_lm)))
            if (k - args.start) % args.stride == 0:
                mp_status[k] = (st_b.copy() if st_b is not None else None,
                                st_s.copy() if st_s is not None else None)
                mp_seeds[k] = (ids_s3d, ids_lm)
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
    if mp_rows:
        _M = np.array(mp_rows, dtype=float)
        np.savez(args.out.replace(".mp4", "_mapper.npz"), rows=_M, cols=np.array(MP_COLS))
        _med = {c: float(np.median(_M[:, i])) for i, c in enumerate(MP_COLS)}
        _tot = {c: float(np.sum(_M[:, i])) for i, c in enumerate(MP_COLS)}
        _npx = float(W * H)
        print("PATCH MAPPER (per-frame medians; pixel counts as % of the 320x200 frame):")
        print(f"  seeds/frame: Sparse3D converged {_med['n_s3d_seeds']:.0f}, in-state fallback {_med['n_lm_seeds']:.0f}")
        for _nm in ("both", "s3d", "lm"):
            print(f"  {_nm:4s}: SeedOnly {100*_med[_nm+'_seedonly']/_npx:5.1f}%  PhotoRefined {100*_med[_nm+'_photo']/_npx:5.1f}%  "
                  f"Rejected {100*_med[_nm+'_rej']/_npx:5.1f}%"
                  + (f"  SeedOnly discs/frame {_med['both_cc']:.0f}, median disc {_med['both_cc_med']:.0f} px" if _nm == "both" else ""))
        print(f"  covered pixels that only the in-state seeds bring: {100*_med['only_from_lm']/_npx:5.1f}% of frame "
              f"(sum over run {100*_tot['only_from_lm']/max(_tot['both_seedonly']+_tot['both_photo'],1):.1f}% of all covered pixel-frames); "
              f"only Sparse3D brings: {100*_med['only_from_s3d']/_npx:5.1f}%")
        _fl = 100 * _tot['flicker_px'] / max(_tot['covered_prev_and_now'], 1)
        _fl_lm = 100 * _tot['flicker_px_only_lm'] / max(_tot['flicker_px'], 1)
        print(f"  FLICKER (|d eta| > 0.3 frame-to-frame): {_fl:.1f}% of covered pixel-frames; "
              f"{_fl_lm:.1f}% of the flickering pixels exist only because of in-state seeds "
              f"(they are {100*_tot['only_from_lm']/max(_tot['both_seedonly']+_tot['both_photo'],1):.1f}% of coverage)")
        print(f"  seeds with a >2x-range neighbour inside the seed radius: Sparse3D {100*_tot['s3d_seeds_straddle']/max(_tot['n_s3d_seeds'],1):.1f}%, "
              f"in-state {100*_tot['lm_seeds_straddle']/max(_tot['n_lm_seeds'],1):.1f}%")
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

    if sparse is not None:
        n_conv = [sum(1 for v in d.values() if v[2]) for d in sp_rng.values()]
        n_all_sp = [len(d) for d in sp_rng.values()]
        cr = np.array([v[0] for d in sp_rng.values() for v in d.values() if v[2]])
        cs = np.array([v[1] for d in sp_rng.values() for v in d.values() if v[2]])
        print(f"Sparse3D: clone-relative on {sp_stats[2]} frames, {sp_stats[1]} relative "
              f"covariances, {sp_stats[0]} re-anchors; per frame live median "
              f"{np.median(n_all_sp) if n_all_sp else 0:.0f}, converged median "
              f"{np.median(n_conv) if n_conv else 0:.0f}")
        if len(cr):
            print("  converged landmark range (m) pct 5/25/50/75/95: "
                  + "/".join(f"{v:.2f}" for v in np.percentile(cr, [5, 25, 50, 75, 95]))
                  + f"   reported sigma_r median {np.median(cs):.2f} m, "
                  f"sigma_r/r median {np.median(cs / cr):.3f}")

    # --- control: the trajectory the bag already carries -------------------
    # Matched by stamp (the on-rig odometry carries the frame stamps), then SE(3)-aligned
    # with no scale so the residual still shows a scale error. Both runs start at rest
    # with gravity aligned, so the alignment is in principle a yaw + translation; the
    # full SE(3) is used anyway and its tilt is reported as a wiring check.
    ctrl = None
    aligned = None
    err = None
    if len(odom):
        j = np.searchsorted(odom[:, 0], stamps)
        j = np.clip(j, 1, len(odom) - 1)
        j = np.where(np.abs(odom[j - 1, 0] - stamps) < np.abs(odom[j, 0] - stamps), j - 1, j)
        ok = np.abs(odom[j, 0] - stamps) < 0.02
        if ok.sum() > 10:
            ctrl_all = odom[j, 1:4]
            ctrl_all = ctrl_all - ctrl_all[ok][0]
            aligned_ok = align_umeyama(traj[ok], ctrl_all[ok])
            # Apply the same transform to every pose for drawing.
            mu_e, mu_a = traj[ok].mean(0), aligned_ok.mean(0)
            Hm = (traj[ok] - mu_e).T @ (aligned_ok - mu_a)
            U, _, Vt = np.linalg.svd(Hm)
            Ra = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            aligned = (Ra @ (traj - mu_e).T).T + mu_a
            ctrl = np.where(ok[:, None], ctrl_all, np.nan)
            err = np.linalg.norm(aligned - ctrl, axis=1)
            e = err[ok]
            pl = lambda p: float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sm = lambda p, w: np.array([p[max(0, i - w):i + w + 1].mean(0) for i in range(len(p))])
            w = max(1, int(round(0.25 / np.median(np.diff(stamps)))))
            tilt = np.degrees(np.arccos(np.clip(Ra[2, 2], -1, 1)))
            # Scale as the Umeyama similarity would report it, from the same matches.
            ec, gc = traj[ok] - mu_e, ctrl_all[ok] - ctrl_all[ok].mean(0)
            s_sim = float(np.trace(np.diag(np.linalg.svd(ec.T @ gc)[1]) @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])) / np.sum(ec ** 2))
            print(f"control (/echo_li/odometry): {ok.sum()} of {len(stamps)} frames matched "
                  f"within 20 ms; SE(3) alignment tilt {tilt:.2f} deg")
            print(f"  residual |est - ctrl| (m): rmse {np.sqrt(np.mean(e**2)):.2f}  median "
                  f"{np.median(e):.2f}  p95 {np.percentile(e, 95):.2f}  max {e.max():.2f}")
            print(f"  path length est/ctrl raw {pl(traj[ok]):.1f}/{pl(ctrl_all[ok]):.1f} m "
                  f"= {pl(traj[ok]) / max(pl(ctrl_all[ok]), 1e-9):.3f}x; smoothed 0.5 s "
                  f"{pl(sm(traj[ok], w)):.1f}/{pl(sm(ctrl_all[ok], w)):.1f} m = "
                  f"{pl(sm(traj[ok], w)) / max(pl(sm(ctrl_all[ok], w)), 1e-9):.3f}x; "
                  f"similarity scale est->ctrl {s_sim:.3f}")
            print(f"  start->end displacement est {np.linalg.norm(traj[-1] - traj[0]):.2f} m, "
                  f"ctrl {np.linalg.norm(ctrl_all[ok][-1] - ctrl_all[ok][0]):.2f} m")
            for q0, q1 in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
                i0b, i1b = int(q0 * ok.sum()), max(int(q1 * ok.sum()), int(q0 * ok.sum()) + 1)
                print(f"    quarter {q0:.2f}-{q1:.2f}: residual median {np.median(e[i0b:i1b]):.2f}  "
                      f"p95 {np.percentile(e[i0b:i1b], 95):.2f} m")
        else:
            print("control: fewer than 10 odometry stamps match the frames -- not aligned")
    np.savez(args.out.replace(".mp4", "_traj.npz"), traj=traj, stamps=stamps,
             ctrl=(ctrl if ctrl is not None else np.zeros((0, 3))),
             aligned=(aligned if aligned is not None else np.zeros((0, 3))),
             err=(err if err is not None else np.zeros(0)))

    if args.no_render:
        print("--no-render: stopping after the filter pass")
        return

    if fdf is not None or dense is not None:
        _cv = [float((v[3] > 0).mean()) for v in dense_frames.values()]
        _cvm = [float((v[3] == 1).mean()) for v in dense_frames.values()]
        _st = fdf.stats() if fdf is not None else dense_last
        print(f"FlowDep-F: converged landmark-frames {fdf_stat[0]} vs KLT {fdf_stat[1]} over {n} frames "
              f"({fdf_stat[0] / max(n, 1):.1f} vs {fdf_stat[1] / max(n, 1):.1f} per frame); live {fdf.n_live() if fdf is not None else dense_last.get('patches_live', 0)}; "
              f"densified coverage per rendered frame median {100 * np.median(_cv) if _cv else 0:.1f}% p10 {100 * np.percentile(_cv, 10) if _cv else 0:.1f}% p90 {100 * np.percentile(_cv, 90) if _cv else 0:.1f}% (measured {100 * np.median(_cvm) if _cvm else 0:.1f}%)")

    # --- render ------------------------------------------------------------
    jetr = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[::-1, 0]
    # One colour scale for both populations (in-state dots and Sparse3D circles).
    allr = np.array([r for d in lm_rng.values() for r in d.values()]
                    + [v[0] for d in sp_rng.values() for v in d.values() if v[2]])
    vlo, vhi = (np.percentile(allr, 5), np.percentile(allr, 95)) if len(allr) else (0.5, 10.0)
    print(f"range colour scale: {vlo:.2f}-{vhi:.2f} m over {len(allr)} landmark-frames")

    RW, SC = 420, 3
    draw = aligned if aligned is not None else traj
    pts = draw[:, :2]
    if ctrl is not None:
        pts = np.vstack([pts, ctrl[np.isfinite(ctrl[:, 0]), :2]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max((hi - lo).max(), 1e-3) * 1.15
    mid = (hi + lo) / 2
    def P(xy):
        u = (xy - mid) / span + 0.5
        return int(u[0] * (RW - 20)) + 10, int((1 - u[1]) * (H * SC - 20)) + 10

    rows = 2 if (fdf is not None or dense is not None) else 1
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (W * SC + RW, H * SC * rows))
    if fdf is not None or dense is not None:
        _lr = np.array([v[2] for d in fdf_rng.values() for v in d.values() if v[3]])
        if len(_lr):
            vlo, vhi = min(vlo, np.percentile(_lr, 5)), max(vhi, np.percentile(_lr, 95))
    base = np.full((H * SC, RW, 3), 28, np.uint8)
    if ctrl is not None and len(ctrl) > 1:
        cp = [P(c[:2]) for c in ctrl if np.isfinite(c[0])]
        cv2.polylines(base, [np.array(cp, np.int32)], False, (105, 105, 105), 1, cv2.LINE_AA)
    # Residual colour scale for the estimate: green at 0, red at the p95 residual.
    e95 = float(np.nanpercentile(err, 95)) if err is not None and np.isfinite(err).any() else 1.0
    def ecol(ev):
        if ev is None or not np.isfinite(ev):
            return (90, 200, 255)
        t = float(np.clip(ev / max(e95, 1e-6), 0, 1))
        return (0, int(255 * (1 - t)), int(255 * t))
    rendered = 0
    for k in range(args.start, args.start + n, args.stride):
        left = cv2.cvtColor(cv2.resize(np.asarray(cache[k]), (W * SC, H * SC),
                                       interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        rr = lm_rng.get(k, {})
        sp = sp_rng.get(k, {})
        nn = ns = nc = 0
        if k in mp_status and mp_status[k][0] is not None:
            # Status overlay: yellow = SeedOnly (seed disc, no photometric refinement),
            # green = PhotoRefined, red = Rejected; magenta = covered ONLY because of the
            # in-state fallback seeds (absent from the Sparse3D-only instance).
            st_b, st_s = mp_status[k]
            ov = np.zeros((H, W, 3), np.uint8)
            ov[st_b == 1] = (0, 255, 255)
            ov[st_b == 2] = (0, 200, 0)
            ov[st_b == 3] = (0, 0, 255)
            if st_s is not None:
                ov[(st_b > 0) & ~(st_s > 0)] = (255, 0, 255)
            ov = cv2.resize(ov, (W * SC, H * SC), interpolation=cv2.INTER_NEAREST)
            m = ov.any(axis=2)
            left[m] = (0.55 * left[m] + 0.45 * ov[m]).astype(np.uint8)
            ids_s3d, ids_lm = mp_seeds.get(k, (set(), set()))
            for fid, (u, v) in feats_per_frame.get(k, {}).items():
                if fid in ids_lm:
                    x, y = int(u * SC), int(v * SC)
                    cv2.rectangle(left, (x - 6, y - 6), (x + 6, y + 6), (255, 0, 255), 2)
            txt(left, f"mapper: yellow SeedOnly  green PhotoRefined  red Rejected  magenta only-from-in-state-seed (square)",
                (8, H * SC - 10), (255, 255, 255))
        for fid, (u, v) in feats_per_frame.get(k, {}).items():
            x, y = int(u * SC), int(v * SC)
            if fid in sp and sp[fid][2]:
                # Sparse3D converged landmark: hollow circle coloured by its range.
                col = clut(jetr, (sp[fid][0] - vlo) / (vhi - vlo + 1e-9))
                cv2.circle(left, (x, y), 9, col, 2, cv2.LINE_AA)
                nc += 1
            if fid in rr:
                col = clut(jetr, (rr[fid] - vlo) / (vhi - vlo + 1e-9))
                cv2.circle(left, (x, y), 5, col, -1, cv2.LINE_AA)
                cv2.circle(left, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
                ns += 1
            elif fid not in sp or not sp[fid][2]:
                cv2.circle(left, (x, y), 2, (150, 150, 150), -1, cv2.LINE_AA)
                nn += 1
        j = k - args.start
        line = f"t {stamps[j]-stamps[0]:6.1f}s   tracked(.) {nn}   in-state(*) {ns}"
        if sparse is not None:
            line += f"   sparse3d converged(o) {nc}"
        txt(left, line, (8, 20))
        txt(left, f"range {vlo:.1f}-{vhi:.1f} m (jet_r: near=red)", (8, 38))
        right = base.copy()
        seg = draw[:j + 1]
        for i in range(1, len(seg)):
            cv2.line(right, P(seg[i - 1][:2]), P(seg[i][:2]),
                     ecol(err[i] if err is not None else None), 2, cv2.LINE_AA)
        cv2.circle(right, P(draw[j][:2]), 4, (0, 255, 255), -1, cv2.LINE_AA)
        txt(right, f"estimate ({vio.backend_kind}), SE3-aligned, green->red = 0..{e95:.1f} m off control",
            (10, 20), (90, 200, 255))
        if ctrl is not None:
            txt(right, "bag /echo_li/odometry (on-rig EqVIO control, not GT)", (10, 38), (160, 160, 160))
        line = f"path {np.sum(np.linalg.norm(np.diff(traj[:j+1],axis=0),axis=1)):.1f} m"
        if err is not None and np.isfinite(err[j]):
            line += f"   |est-ctrl| {err[j]:.2f} m"
        txt(right, line, (10, 56))
        frame_out = np.hstack([left, right])
        if fdf is not None or dense is not None:
            # Second row: FlowDep-F converged landmarks on the undistorted frame + the
            # densified range (nearest patch + disagreement + footprint sigma) | per-pixel sigma.
            _ug, _eta, _sig, _cls = dense_frames.get(k, (None, None, None, None))
            if _ug is None:
                _ug = cv2.remap(np.asarray(cache[k]), undist_maps[0], undist_maps[1], cv2.INTER_LINEAR)
                _eta = np.zeros((H, W), np.float16); _sig = np.ones((H, W), np.float16); _cls = np.zeros((H, W), np.uint8)
            _cov_m = _cls > 0
            dl = cv2.cvtColor(cv2.resize(_ug, (W * SC, H * SC), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            t_d = np.clip((np.exp(_eta.astype(np.float64)) - vlo) / (vhi - vlo + 1e-9), 0, 1)
            col = jetr[(t_d * 255).astype(np.uint8)]          # same lookup as clut(jetr, t) above
            col = cv2.resize(col, (W * SC, H * SC), interpolation=cv2.INTER_NEAREST)
            cls_big = cv2.resize(_cls, (W * SC, H * SC), interpolation=cv2.INTER_NEAREST)
            m_meas = cls_big == 1; m_fill = cls_big == 2
            dl[m_meas] = (0.3 * dl[m_meas] + 0.7 * col[m_meas]).astype(np.uint8)
            dl[m_fill] = (0.6 * dl[m_fill] + 0.4 * col[m_fill]).astype(np.uint8)   # Filled: paler, same colour scale
            nf = 0
            for fid, (u, v, r, conv) in fdf_rng.get(k, {}).items():
                if conv:
                    cv2.circle(dl, (int(u * SC), int(v * SC)), 3, (255, 255, 255), 1, cv2.LINE_AA); nf += 1
            txt(dl, f"FlowDep-F converged {nf}   dense range: measured {100 * (_cls == 1).mean():.0f}%" + (f" + filled {100 * (_cls == 2).mean():.0f}% (paler)" if args.fill else "") + f"   unknown {100 * (_cls == 0).mean():.0f}%", (8, 20))
            txt(dl, "undistorted pinhole frame; same colour scale as above; filled = interpolated between measured patches only", (8, 38))
            sg = np.full((H * SC, RW, 3), 28, np.uint8)
            sw = RW; sh = int(H * sw / W)
            t_s = np.clip((np.log(np.maximum(_sig.astype(np.float64), 1e-4)) - np.log(0.01)) / (np.log(0.30) - np.log(0.01)), 0, 1)
            scol = jetr[(t_s * 255).astype(np.uint8)]          # 1 % red .. 30 % blue
            scol[~_cov_m] = (28, 28, 28)
            sg[:sh] = cv2.resize(scol, (sw, sh), interpolation=cv2.INTER_NEAREST)
            txt(sg, "per-pixel sigma: red 1% .. blue 30%", (10, sh + 20), (200, 200, 200))
            if _cov_m.any():
                txt(sg, f"median sigma {100 * np.median(_sig[_cov_m].astype(np.float64)):.1f}%  (log-range)", (10, sh + 38), (200, 200, 200))
            txt(sg, "no GT on the rig: sigma is the", (10, sh + 64), (160, 160, 160))
            txt(sg, "rule validated on TartanAir/MidAir", (10, sh + 80), (160, 160, 160))
            frame_out = np.vstack([frame_out, np.hstack([dl, sg])])
        vw.write(frame_out)
        rendered += 1
    vw.release()
    print(f"wrote {args.out}  ({rendered} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
