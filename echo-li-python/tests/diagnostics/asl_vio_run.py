"""Drive EqVIO on an ASL-format sequence (EuRoC or TUM-VI) and capture pose covariance.

Companion to midair_vio_run.py, for cross-dataset verification of the pose-covariance
results (midair_relpose_randomwalk_test.py, midair_pac_crossterm_probe.py). EuRoC is
the right control for those specifically because they need only GT *pose*, never GT
depth -- so EuRoC's excellent Vicon pose ground truth is used and its known-buggy
pointcloud depth is never touched.

Use ground truth that MEASURES orientation, not one that infers it. EuRoC Vicon rooms
(V1_*, V2_*) give true 6-DOF Vicon; EuRoC Machine Hall does not (Leica is position-only,
attitude is fused). TUM-VI room1-6 give 120 Hz OptiTrack 6-DOF, oversampled 6x against
the 20 Hz camera, which is what keeps the short-lag regime clean.

CAUTION on TUM-VI: the mocap->camera extrinsic is documented as error-prone
(fisheye_nis_by_radius.py). A constant extrinsic error E leaves the GT relative pose
conjugated while the estimate carries none, so the residual grows with the relative
motion -- error ~ lag, variance ~ lag^2. That MANUFACTURES a super-diffusive signature.
Before trusting a TUM-VI exponent, re-run with the extrinsic perturbed and confirm the
exponent is insensitive.

Ground truth is interpolated to each image stamp (linear on position, SLERP on
rotation) rather than nearest-neighbour: GT runs at 200 Hz, so nearest-neighbour would
inject up to 2.5 ms of pose error, which at ~1 rad/s is ~0.14 deg -- comparable to the
one-frame relative rotation error the probes measure, i.e. it would contaminate exactly
the short-lag regime that carries the headline numbers.

Saves the self-contained `baseline_*` npz schema (including GT attitude and the
body->camera extrinsic), so both probes read it with no dataset access.

    PY=echo-li-python/venv/bin/python
    $PY euroc_vio_run.py <path>/V1_01_easy --config configs/eqvio_euroc_rho.yaml \
        --save-npz v1_01.npz
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import echo_li  # noqa: E402
from euroc_tracking import load_camera_config, load_images, load_imu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest  # noqa: E402


def load_calib(root: Path):
    """Intrinsics + body->camera extrinsic, mirroring echo-li-core's asl_dataset.rs.

    EuRoC puts both in cam0/sensor.yaml (T_BS = camera pose in the body frame).
    TUM-VI instead ships a Kalibr camchain at <dataset>/dso/camchain.yaml, whose
    `T_cam_imu` is IMU->camera -- the INVERSE of what we want. Getting that
    inversion wrong would displace the camera by a constant, and a constant
    extrinsic error grows with the relative motion (error ~ lag, variance ~ lag^2),
    i.e. it would fabricate the very super-diffusive signature under test.
    """
    import yaml
    try:
        w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs = load_camera_config(root)
    except FileNotFoundError:
        # TUM-VI has no cam0/sensor.yaml at all; fall through to the camchain.
        fx, t_bs = None, None
    if fx and t_bs is not None:
        return w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs, "sensor.yaml"

    camchain = root.parent / "dso" / "camchain.yaml"
    if not camchain.exists():
        raise SystemExit(f"no usable calibration: {root}/cam0/sensor.yaml incomplete "
                         f"and {camchain} missing")
    cam0 = yaml.safe_load(open(camchain))["cam0"]
    fx, fy, cx, cy = cam0["intrinsics"]
    w, h = cam0["resolution"]
    dist_model = cam0.get("distortion_model")
    dist_coeffs = list(cam0.get("distortion_coeffs", []))
    t_ci = np.array(cam0["T_cam_imu"], dtype=float).reshape(4, 4)
    t_bs = np.linalg.inv(t_ci)          # IMU->cam inverted to cam-in-body, as the Rust does
    return w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs, "dso/camchain.yaml"


def load_gt_pose(root: Path):
    """(times, positions[N,3], quats[N,4] xyzw) from state_groundtruth_estimate0.

    EuRoC columns: timestamp, p_RS_R_{x,y,z}, q_RS_{w,x,y,z}, ...
    """
    # EuRoC: state_groundtruth_estimate0. TUM-VI euroc export: mocap0. Both start
    # with timestamp, p_{x,y,z}, q_{w,x,y,z}, so one parser serves both.
    path = root / "state_groundtruth_estimate0" / "data.csv"
    if not path.exists():
        path = root / "mocap0" / "data.csv"
    if not path.exists():
        raise SystemExit(f"no ground truth under {root} (tried state_groundtruth_estimate0, mocap0)")
    t, p, q = [], [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]) * 1e-9)
            p.append([float(row[1]), float(row[2]), float(row[3])])
            # csv is (w,x,y,z); scipy wants (x,y,z,w)
            q.append([float(row[5]), float(row[6]), float(row[7]), float(row[4])])
    return np.array(t), np.array(p), np.array(q)


def align_se3(src, dst):
    """Umeyama without scale -- the 'SE3-aligned' of the ATE baselines note."""
    ms, mdst = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - mdst))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return (R @ (src - ms).T).T + mdst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs, src = load_calib(root)
    print(f"camera {w}x{h} fx={fx:.1f} cx={cx:.1f}  dist={dist_model}  (calib from {src})")
    print(f"T_BS translation (lever arm) = {np.round(t_bs[:3, 3], 4).tolist()} m")

    cfg = echo_li.FrontendConfig.from_yaml(args.config)
    cfg.set_camera(fx, fy, cx, cy, w, h, dist_coeffs if dist_coeffs else [])
    tracker = echo_li.Frontend(cfg, w, h)
    dm = (dist_model or "").lower()
    if "equidistant" in dm or "fisheye" in dm or "kannala" in dm:
        # TUM-VI: Kannala-Brandt fisheye. Using a radtan model here would blow up at
        # the edge of a 195-deg lens, so this branch is load-bearing.
        cam = echo_li.EquidistantCamera(fx, fy, cx, cy, *dist_coeffs[:4])
    elif "radial" in dm:
        cam = echo_li.RadTanCamera(fx, fy, cx, cy, *dist_coeffs[:4])
    else:
        cam = echo_li.PinholeCamera(fx, fy, cx, cy)
    print(f"camera model -> {type(cam).__name__}")
    vio = echo_li.VIOFilter(args.config, cam)
    vio.set_camera_extrinsics(t_bs)

    gt_t, gt_p, gt_q = load_gt_pose(root)
    slerp = Slerp(gt_t, Rot.from_quat(gt_q))
    print(f"ground truth: {len(gt_t)} poses, {gt_t[0]:.2f}..{gt_t[-1]:.2f} s")

    events = sorted([(t, "imu", d) for t, *d in ((t, g, a) for t, g, a in load_imu(root))]
                    + [(t, "img", p) for t, p in load_images(root)], key=lambda e: e[0])
    print(f"events: {sum(e[1]=='imu' for e in events)} imu, {sum(e[1]=='img' for e in events)} img")

    rec = []
    for stamp, etype, data in events:
        if etype == "imu":
            gyr, acc = data
            vio.process_imu(stamp, gyr, acc)
            continue
        if not Path(data).exists():
            continue
        gray = cv2.imread(str(data), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        feats, stats = tracker.process(gray)
        if not vio.is_initialized:
            continue
        vio.process_vision(stamp, {f["id"]: (f["x"], f["y"]) for f in feats})
        if not (gt_t[0] <= stamp <= gt_t[-1]):
            continue
        pos, quat = vio.get_pose()
        pcov = vio.get_camera_pose_covariance()
        if pcov is None:
            continue
        gp = np.array([np.interp(stamp, gt_t, gt_p[:, i]) for i in range(3)])
        gq = slerp(stamp).as_quat()
        gb, ab = vio.get_biases()   # IMU bias estimates; their DRIFT RATE is the best
                                    # online predictor of the per-track error twist
        rec.append((len(rec), np.asarray(pos), np.asarray(quat), gp, gq,
                    np.asarray(pcov[0], float), np.asarray(pcov[1], float),
                    np.asarray(gb, float), np.asarray(ab, float)))
        if len(rec) % 200 == 0:
            print(f"  [{len(rec):5d}] t={stamp:.2f} tracked={stats['tracked']}")
        if args.max_frames and len(rec) >= args.max_frames:
            break

    if not rec:
        raise SystemExit("no frames captured (filter never initialised?)")
    est = np.array([r[1] for r in rec])
    gt = np.array([r[3] for r in rec])
    ea = align_se3(est, gt)
    ate = float(np.sqrt(((ea - gt) ** 2).sum(1).mean()))
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    print(f"\n=== {len(rec)} frames ===")
    print(f"GT path {path_len:.1f} m")
    print(f"ATE (SE3-aligned RMSE): {ate:.4f} m = {100*ate/max(path_len,1e-9):.2f}% of path")
    print("  compare against docs/eqvio/euroc_ate_baselines.md before trusting any probe")

    if args.save_npz:
        np.savez(args.save_npz,
                 baseline_k=np.array([r[0] for r in rec]),
                 baseline_est=est,
                 baseline_quat=np.array([r[2] for r in rec]),
                 baseline_gt=gt,
                 baseline_gt_quat=np.array([r[4] for r in rec]),
                 baseline_pcov_pos=np.array([r[5] for r in rec]),
                 baseline_pcov_att=np.array([r[6] for r in rec]),
                 baseline_gyro_bias=np.array([r[7] for r in rec]),
                 baseline_accel_bias=np.array([r[8] for r in rec]),
                 baseline_t_bc=t_bs, ate=ate, path_len=path_len)
        print("saved ->", args.save_npz)
        run_manifest.save_run_manifest(args.save_npz, args.config,
                                       extra={"dataset": str(root), "frames": len(rec),
                                              "ate_m": round(ate, 4),
                                              "ate_pct": round(100 * ate / max(path_len, 1e-9), 3)})


if __name__ == "__main__":
    main()
