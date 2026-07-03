"""Real-data depth-calibration scorecard for the Sparse3DFilter.

Uses EuRoC's survey-grade Leica scan (mav0/pointcloud0/data.ply, world frame) as
depth ground truth: project the cloud into each eval frame with the Vicon GT
pose (z-buffered), look up GT depth at each tracked-feature pixel, and score the
sparse filter's depth estimate + reported sigma against it. Also histograms the
filter's own NIS (GT-free online consistency; ideal chi2(2)).

The filter is fed **GT poses** so the scorecard isolates the sparse filter's
real-data calibration (tracker noise, real texture, real geometry) from VIO pose
error. Optional --sigma-t/--sigma-phi feed a constant p_vv/p_ww and
--pose-measurement/--anchor exercise the consider-R treatment.

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


def gt_depth_at(cloud_c, uvs, fx, fy, cx, cy, w, h, cell=4):
    """Z-buffer the camera-frame cloud on a cell-px grid; return GT depth at each
    (undistorted-pixel) uv, NaN where no scan point lands in the cell."""
    z = cloud_c[:, 2]
    front = z > 0.1
    pc = cloud_c[front]
    u = fx * pc[:, 0] / pc[:, 2] + cx
    v = fy * pc[:, 1] / pc[:, 2] + cy
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inb], v[inb], pc[inb, 2]
    gw, gh = w // cell + 1, h // cell + 1
    zbuf = np.full(gw * gh, np.inf, np.float32)
    idx = (v / cell).astype(np.int32) * gw + (u / cell).astype(np.int32)
    np.minimum.at(zbuf, idx, z.astype(np.float32))
    out = np.full(len(uvs), np.nan)
    for k, (uu, vv) in enumerate(uvs):
        if 0 <= uu < w and 0 <= vv < h:
            d = zbuf[int(vv / cell) * gw + int(uu / cell)]
            if np.isfinite(d):
                out[k] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=10, help="GT-render every K frames")
    ap.add_argument("--min-track", type=int, default=10,
                    help="score only features with track_length >= this")
    ap.add_argument("--sigma-t", type=float, default=0.0, help="constant p_vv = (s/dt)^2 I")
    ap.add_argument("--sigma-phi", type=float, default=0.0, help="constant p_ww = (s/dt)^2 I")
    ap.add_argument("--pose-measurement", action="store_true")
    ap.add_argument("--anchor", action="store_true")
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

    sf_kwargs = dict(sigma_pixel=0.5, min_track_length=1,
                     pose_measurement=args.pose_measurement,
                     anchor_measurement=args.anchor)
    filt = echo_li.Sparse3DFilter.invdepth_additive3d(fx, fy, cx, cy, **sf_kwargs)
    DT = 0.05
    p_vv = ((args.sigma_t / DT) ** 2 * np.eye(3)).tolist() if args.sigma_t > 0 else None
    p_ww = ((args.sigma_phi / DT) ** 2 * np.eye(3)).tolist() if args.sigma_phi > 0 else None

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
    print(f"config: pose_measurement={args.pose_measurement} anchor={args.anchor} "
          f"sigma_t={args.sigma_t} sigma_phi={args.sigma_phi}")
    if len(z):
        print(f"depth NEES-1D (mean z^2): {np.mean(z**2):7.2f}   (ideal 1)")
        print(f"|z|<1 / <2 / <3:          {np.mean(np.abs(z)<1)*100:5.1f}% / "
              f"{np.mean(np.abs(z)<2)*100:5.1f}% / {np.mean(np.abs(z)<3)*100:5.1f}%  "
              f"(ideal 68/95/99.7)")
        print(f"bias (mean z):            {np.mean(z):7.2f}")
        print(f"rel depth err:            median {np.median(np.abs(rel))*100:5.1f}%  "
              f"p90 {np.percentile(np.abs(rel), 90)*100:5.1f}%")
        print(f"GT depth range:           {np.min(gtds):.1f}..{np.max(gtds):.1f} m "
              f"(median {np.median(gtds):.1f})")
    if len(nis):
        print(f"NIS (chi2(2), GT-free):   mean {np.mean(nis):5.2f}  median "
              f"{np.median(nis):5.2f}  p95 {np.percentile(nis, 95):5.2f}   "
              f"(ideal 2 / 1.39 / 5.99)")


if __name__ == "__main__":
    main()
