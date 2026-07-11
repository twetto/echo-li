"""Real-data depth-calibration scorecard for the Sparse3DFilter.

Uses EuRoC's survey-grade Leica scan (mav0/pointcloud0/data.ply, world frame) as
depth ground truth: project the cloud into each eval frame with the Vicon GT
pose (z-buffered), look up GT depth at each tracked-feature pixel, and score the
sparse filter's depth estimate + reported sigma against it. Also histograms the
filter's own NIS (GT-free online consistency; ideal chi2(2)).

The filter is fed **GT poses** so the scorecard isolates the sparse filter's
real-data calibration (tracker noise, real texture, real geometry) from VIO pose
error. Optional --sigma-t/--sigma-phi feed constant p_vv/p_ww covariances.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/real_depth_eval.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--every 10]
"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
import echo_li


def load_csv(path):
    with open(path) as f:
        return np.array([r for r in csv.reader(f) if r and not r[0].startswith("#")], dtype=float)


def load_cloud(ply_path):
    npy = ply_path.with_suffix(".npy")
    if npy.exists():
        return np.load(npy)
    with open(ply_path, "rb") as f:
        n_header = 0
        n_vert = 0
        for line in f:
            n_header += 1
            if line.startswith(b"element vertex"):
                n_vert = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
    print(f"parsing {ply_path.name}: {n_vert} pts (once; cached to .npy)...")
    data = np.loadtxt(ply_path, skiprows=n_header, usecols=(0, 1, 2), dtype=np.float32)
    assert len(data) == n_vert, (len(data), n_vert)
    np.save(npy, data)
    return data


def build_zbuf(cloud, t_cw, fx, fy, cx, cy, w, h, cell=4):
    """Full-resolution z-buffer of the (world-frame f32) cloud for one camera
    pose, on a cell-px grid. f32 throughout (no f64 promotion) and copy-free:
    behind-camera / out-of-view points are routed to a dummy bin instead of
    boolean-index copies (those dominated the runtime on the 3.2M-pt scan)."""
    r = t_cw[:3, :3].astype(np.float32)
    t = t_cw[:3, 3].astype(np.float32)
    cc = cloud @ r.T + t
    x, y, z = cc[:, 0], cc[:, 1], cc[:, 2]
    zs = np.where(z > np.float32(0.1), z, np.float32(np.inf))
    inv = np.float32(1.0) / zs
    u = np.float32(fx) * x * inv + np.float32(cx)
    v = np.float32(fy) * y * inv + np.float32(cy)
    gw, gh = w // cell + 1, h // cell + 1
    dummy = gw * gh
    iu = (u * np.float32(1.0 / cell)).astype(np.int32)
    iv = (v * np.float32(1.0 / cell)).astype(np.int32)
    idx = iv * gw + iu
    bad = (u < 0) | (u >= w) | (v < 0) | (v >= h)
    idx[bad] = dummy
    zbuf = np.full(dummy + 1, np.inf, np.float32)
    np.minimum.at(zbuf, idx, zs)
    return zbuf[:dummy]


def zbuf_lookup(zbuf, uvs, w, h, cell=4):
    """GT depth at each (undistorted-pixel) uv; NaN where no scan point landed."""
    gw = w // cell + 1
    out = np.full(len(uvs), np.nan)
    for k, (uu, vv) in enumerate(uvs):
        if 0 <= uu < w and 0 <= vv < h:
            d = zbuf[int(vv / cell) * gw + int(uu / cell)]
            if np.isfinite(d):
                out[k] = d
    return out


def gt_depth_at(cloud_c, uvs, fx, fy, cx, cy, w, h, cell=4):
    """Back-compat wrapper: camera-frame cloud -> depths at uvs (one shot)."""
    return zbuf_lookup(
        build_zbuf(cloud_c, np.eye(4), fx, fy, cx, cy, w, h, cell), uvs, w, h, cell
    )


def zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h, cell=4):
    """Per-dataset cache of full-resolution z-buffers for every frame stamp.
    Built once (~3 min for V1_03), memmapped thereafter -- later runs never even
    load the point cloud. Invalidated if the frame stamp list changes."""
    import time as _time
    gw, gh = w // cell + 1, h // cell + 1
    zpath = root / "pointcloud0" / f"zbuf_c{cell}.npy"
    spath = root / "pointcloud0" / f"zbuf_c{cell}.stamps.npy"
    stamps = np.array([t for t, _ in frames])
    if zpath.exists() and spath.exists() and np.array_equal(np.load(spath), stamps):
        return np.load(zpath, mmap_mode="r")
    cloud = load_cloud(root / "pointcloud0" / "data.ply")
    print(f"building z-buffer cache: {len(frames)} frames x {gh}x{gw} cells "
          f"(full {len(cloud)}-pt resolution, once)...")
    mm = np.lib.format.open_memmap(zpath, mode="w+", dtype=np.float32,
                                   shape=(len(frames), gh * gw))
    t0 = _time.time()
    for i, (t, _) in enumerate(frames):
        mm[i] = build_zbuf(cloud, np.linalg.inv(cam_pose(t)), fx, fy, cx, cy, w, h, cell)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] {i / max(_time.time() - t0, 1e-9):.0f} fps")
    mm.flush()
    np.save(spath, stamps)
    return np.load(zpath, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=10, help="GT-render every K frames")
    ap.add_argument("--min-track", type=int, default=10,
                    help="score only features with track_length >= this")
    ap.add_argument("--sigma-t", type=float, default=0.0, help="constant p_vv = s^2 I")
    ap.add_argument("--sigma-phi", type=float, default=0.0, help="constant p_ww = s^2 I")
    ap.add_argument("--chart", default="invdepth", choices=["invdepth", "bearing"],
                    help="sparse chart to score")
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root/"mav0").exists():
        root = root/"mav0"

    cfg = yaml.safe_load(open(root/"cam0"/"sensor.yaml"))
    w, h = cfg["resolution"]; fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    gt = load_csv(root/"state_groundtruth_estimate0"/"data.csv")
    gt_t = gt[:, 0]*1e-9; gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))
    cloud = load_cloud(root/"pointcloud0"/"data.ply")
    print(f"cloud {len(cloud)} pts; cam {w}x{h}")

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    sf_kwargs = dict(sigma_pixel=0.5, min_track_length=1)
    if args.chart == "bearing":
        # This instrument pre-undistorts pixels into the pinhole-K domain (and does
        # its GT-depth lookup with a pinhole projection), so the bearing chart's
        # camera is a PinholeCamera here — the *domain* camera, not the dataset
        # model. Passing the real radtan/fisheye camera would double-undistort.
        filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
            echo_li.PinholeCamera(fx, fy, cx, cy), **sf_kwargs)
    else:
        filt = echo_li.Sparse3DFilter.invdepth_additive3d(fx, fy, cx, cy, **sf_kwargs)
    print(f"chart: {args.chart}")
    p_vv = ((args.sigma_t) ** 2 * np.eye(3)).tolist() if args.sigma_t > 0 else None
    p_ww = ((args.sigma_phi) ** 2 * np.eye(3)).tolist() if args.sigma_phi > 0 else None

    idir = root/"cam0"/"data"
    with open(root/"cam0"/"data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(r[0])*1e-9, idir/r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]

    zscores, relerrs, nises, gtds = [], [], [], []
    n_eval = 0
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        if not feats:
            continue
        # undistort feature pixels -> pinhole-K pixel domain (matches the filter's K
        # and the pinhole cloud projection)
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
        ids = [int(f["id"]) for f in feats]
        uvs = {fid: (float(u), float(v)) for fid, (u, v) in zip(ids, und)}

        # GT camera pose at stamp
        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(t).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        t_wc = t_wb @ t_bs

        filt.update(t, uvs, t_wc.tolist(), p_vv, p_ww)
        fdict = filt.get_features()

        live = {fid: fd for fid, fd in fdict.items()
                if fd["track_length"] >= args.min_track}
        for fd in live.values():
            if np.isfinite(fd["nis"]):
                nises.append(fd["nis"])

        if i % args.every == 0 and live:
            t_cw = np.linalg.inv(t_wc)
            cloud_c = cloud @ t_cw[:3, :3].T + t_cw[:3, 3]
            fids = list(live.keys())
            f_uvs = [uvs[fid] for fid in fids]
            gtd = gt_depth_at(cloud_c, f_uvs, fx, fy, cx, cy, w, h)
            for fid, d in zip(fids, gtd):
                if not np.isfinite(d):
                    continue
                fd = live[fid]
                est_z = fd["position"][2]
                var_z = np.asarray(fd["covariance_euclidean"])[2, 2]
                if var_z <= 0:
                    continue
                zscores.append((est_z - d) / np.sqrt(var_z))
                relerrs.append((est_z - d) / d)
                gtds.append(d)
            n_eval += 1
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] t={t-gt_t[0]:5.1f}s live={len(live)} "
                  f"scored={len(zscores)} {i/max(time.time()-tstart,1e-9):.0f}fps")

    z = np.array(zscores); rel = np.array(relerrs); nis = np.array(nises)
    print(f"\n=== depth scorecard (GT poses, track>={args.min_track}, "
          f"{n_eval} eval frames, {len(z)} feature-obs) ===")
    print(f"config: sigma_t={args.sigma_t} sigma_phi={args.sigma_phi}")
    if len(z):
        print(f"depth NEES-1D (mean z^2): {np.mean(z**2):7.2f}   (ideal 1)")
        print(f"|z|<1 / <2 / <3:          {np.mean(np.abs(z)<1)*100:5.1f}% / "
              f"{np.mean(np.abs(z)<2)*100:5.1f}% / {np.mean(np.abs(z)<3)*100:5.1f}%  "
              f"(ideal 68/95/99.7)")
        print(f"bias (mean z):            {np.mean(z):7.2f}")
        print(f"rel depth err:            median {np.median(np.abs(rel))*100:5.1f}%  "
              f"p90 {np.percentile(np.abs(rel), 90)*100:5.1f}%")
        print(f"SIGNED rel depth err:     median {np.median(rel)*100:+5.1f}%  "
              f"mean {np.mean(rel)*100:+5.1f}%  (neg = estimate too close)")
        print(f"GT depth range:           {np.min(gtds):.1f}..{np.max(gtds):.1f} m "
              f"(median {np.median(gtds):.1f})")
    if len(nis):
        print(f"NIS (chi2(2), GT-free):   mean {np.mean(nis):5.2f}  median "
              f"{np.median(nis):5.2f}  p95 {np.percentile(nis, 95):5.2f}   "
              f"(ideal 2 / 1.39 / 5.99)")


if __name__ == "__main__":
    main()
