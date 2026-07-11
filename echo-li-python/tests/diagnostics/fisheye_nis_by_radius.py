"""Fisheye Sparse3D NIS stratified by image radius (bearing-migration step 5).

The wide-FoV consistency proof. TUM-VI has **no depth GT** (no Leica scan), so we
cannot score depth NEES on fisheye — but NIS is GT-free (innovation vs its
predicted covariance S), so it validates the *camera-model* consistency directly.

The filter is driven by the EqF's own estimated poses (VIOFilter handles the
`T_cam_imu` extrinsic internally, avoiding the error-prone TUM-VI mocap->cam
chain). Global pose drift inflates NIS at all radii ~equally; stratifying NIS by
image radius therefore isolates the fisheye camera model: if the equidistant
undistortion is correct, NIS stays flat from center to edge; if the model were
wrong (e.g. radtan on a fisheye), edge features (strongest distortion) blow up.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/fisheye_nis_by_radius.py \
      ../Rudolf-V/target/tumvi/dataset-room1_512_16 [--chart bearing] [--every 1]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

import echo_li

REPO = Path(__file__).resolve().parents[3]


def parse_camchain_cam0(path):
    """Parse cam0 from a Kalibr camchain: intrinsics, distortion, T_cam_imu, res."""
    with open(path) as f:
        chain = yaml.safe_load(f)
    cam = chain["cam0"]
    fx, fy, cx, cy = cam["intrinsics"]
    dist = list(cam["distortion_coeffs"])
    w, h = cam["resolution"]
    model = cam["distortion_model"]
    t_cam_imu = np.array(cam["T_cam_imu"], dtype=float)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, w=w, h=h,
                model=model, t_cam_imu=t_cam_imu)


def load_imu(path):
    rows = []
    with open(path) as f:
        rd = csv.reader(f)
        for r in rd:
            if not r or r[0].startswith("#"):
                continue
            v = [float(x) for x in r]
            rows.append((v[0] * 1e-9, v[1:4], v[4:7]))  # t(s), gyro, accel
    return rows


def load_frames(root):
    idir = root / "mav0" / "cam0" / "data"
    frames = []
    with open(root / "mav0" / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        for r in rd:
            if not r or r[0].startswith("#"):
                continue
            frames.append((int(r[0]) * 1e-9, idir / r[1].strip()))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(REPO / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--chart", default="bearing", choices=["bearing", "invdepth"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--min-track", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = Path(args.dataset)
    cc = parse_camchain_cam0(root / "dso" / "camchain.yaml")
    fx, fy, cx, cy = cc["fx"], cc["fy"], cc["cx"], cc["cy"]
    w, h = cc["w"], cc["h"]
    dist = cc["dist"]
    assert cc["model"] == "equidistant", f"expected equidistant, got {cc['model']}"
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array(dist, dtype=float)
    print(f"cam {w}x{h} equidistant fx={fx:.1f} cx={cx:.1f} cy={cy:.1f} dist={dist}")

    t_bs = np.linalg.inv(cc["t_cam_imu"])  # T_BS (cam->body) = inv(T_cam_imu)

    # Tracker (gate off: no pose prior is supplied to process()).
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dist)
    tracker = echo_li.Frontend(fcfg, w, h)

    # VIO for poses (handles T_cam_imu extrinsic itself).
    cam = echo_li.EquidistantCamera(fx, fy, cx, cy, dist[0], dist[1], dist[2], dist[3])
    vio = echo_li.VIOFilter(args.config, cam, n_init_samples=100)
    vio.set_camera_extrinsics(t_bs)

    # Sparse depth bank.
    sf_kwargs = dict(sigma_pixel=0.5, min_track_length=1)
    if args.chart == "bearing":
        filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **sf_kwargs)
    else:
        filt = echo_li.Sparse3DFilter.invdepth_additive3d(fx, fy, cx, cy, **sf_kwargs)
    print(f"chart: {args.chart}")

    imu = load_imu(root / "mav0" / "imu0" / "data.csv")
    frames = load_frames(root)
    if args.max_frames:
        frames = frames[: args.max_frames]

    imu_i = 0
    nis_r = []  # (nis, radius_px)
    n_vis = 0
    for k, (t, p) in enumerate(frames):
        # feed all IMU up to this image time
        while imu_i < len(imu) and imu[imu_i][0] <= t:
            ti, g, a = imu[imu_i]
            vio.process_imu(ti, g, a)
            imu_i += 1

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        if not feats:
            continue
        ids = [int(f["id"]) for f in feats]
        raw = {int(f["id"]): [float(f["x"]), float(f["y"])] for f in feats}
        radius = {int(f["id"]): float(np.hypot(f["x"] - cx, f["y"] - cy)) for f in feats}

        vio.process_vision(t, {i: raw[i] for i in ids})
        if not vio.is_initialized:
            continue

        pos, quat = vio.get_pose()  # quat xyzw, body pose T_wb
        from scipy.spatial.transform import Rotation as Rot
        t_wb = np.eye(4)
        t_wb[:3, :3] = Rot.from_quat(quat).as_matrix()
        t_wb[:3, 3] = pos
        t_wc = t_wb @ t_bs

        if args.chart == "bearing":
            meas = {i: raw[i] for i in ids}
        else:
            px = np.array([raw[i] for i in ids], np.float64).reshape(-1, 1, 2)
            und = cv2.fisheye.undistortPoints(px, K, D, P=K).reshape(-1, 2)
            meas = {i: [float(u), float(v)] for i, (u, v) in zip(ids, und)}

        filt.update(t, meas, t_wc.tolist(), None, None)
        n_vis += 1

        if k % args.every == 0:
            fdict = filt.get_features()
            for fid, fd in fdict.items():
                if fd["track_length"] >= args.min_track and np.isfinite(fd["nis"]):
                    if fid in radius:
                        nis_r.append((fd["nis"], radius[fid]))

        if k % 400 == 0:
            print(f"  [{k}/{len(frames)}] init={vio.is_initialized} "
                  f"feats={len(feats)} samples={len(nis_r)}")

    if not nis_r:
        print("no NIS samples collected (VIO may not have initialized)")
        return

    nis = np.array([x[0] for x in nis_r])
    rad = np.array([x[1] for x in nis_r])
    rmax = np.hypot(cx, cy)  # ~half-diagonal

    print(f"\n=== fisheye Sparse3D NIS by image radius ({args.chart}, "
          f"{n_vis} vis frames, {len(nis)} feature-obs) ===")
    print(f"NIS ideal chi2(2): mean 2.0 / median 1.39 / p95 5.99")
    print(f"overall: mean {nis.mean():6.2f}  median {np.median(nis):6.2f}  "
          f"p95 {np.percentile(nis, 95):6.2f}")
    print(f"{'radius band':>18} | {'r_px':>10} | {'FoV angle':>10} | "
          f"{'n':>6} | {'mean':>6} | {'median':>6} | {'p95':>6}")
    edges = [0.0, 0.33, 0.66, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rad >= lo * rmax) & (rad < hi * rmax)
        if not m.any():
            continue
        # representative FoV half-angle at the band's outer radius
        r_outer = min(hi, 1.0) * rmax
        u_edge = np.array([cx + r_outer, cy])
        b = cam.undistort(u_edge)
        ang = np.degrees(np.arctan2(np.hypot(b[0], b[1]), b[2]))
        band = f"{lo:.2f}-{hi:.2f} R"
        print(f"{band:>18} | {rad[m].mean():10.1f} | {ang:9.1f}d | "
              f"{m.sum():6d} | {nis[m].mean():6.2f} | {np.median(nis[m]):6.2f} | "
              f"{np.percentile(nis[m], 95):6.2f}")


if __name__ == "__main__":
    main()
