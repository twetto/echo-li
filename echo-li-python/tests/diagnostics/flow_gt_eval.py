"""Ground-truth optical-flow scorecard for the tracker (Rudolf-V frontend).

GT flow from the EuRoC Leica scan + Vicon poses (rigid static scene): z-buffer GT
depth at each tracked feature's pixel in frame k, backproject to a 3D world
point, reproject into frame k+1 with the GT pose -> GT correspondence. The
tracker's uv_{k+1} minus that is the TRUE per-frame tracking error.

Measures, on real data:
  * effective sigma_pixel of the KLT (inlier core, per-axis) -- the filter
    currently assumes 0.5 px;
  * outlier rate/magnitude (err > 3 px);
  * degradation vs GT angular rate (rotation bursts).

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/flow_gt_eval.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--every 5]
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, load_cloud, gt_depth_at  # noqa: E402
import echo_li  # noqa: E402


def quat_ang_rate(t, quat):
    def qmul(a, b):
        aw, ax, ay, az = a.T; bw, bx, by, bz = b.T
        return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], axis=1)
    qc = quat.copy(); qc[:, 1:] *= -1
    dq = qmul(qc[:-1], quat[1:]); dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    w = 2*np.arccos(np.clip(np.abs(dq[:, 0]), -1, 1))/np.diff(t)
    return np.concatenate([w, w[-1:]])


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=5, help="evaluate every K frame-pairs")
    ap.add_argument("--out", default="flow_gt_hist.png")
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
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])
    cloud = load_cloud(root/"pointcloud0"/"data.ply")

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    def cam_pose(t):
        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(t).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return t_wb @ t_bs

    idir = root/"cam0"/"data"
    with open(root/"cam0"/"data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(r[0])*1e-9, idir/r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]

    prev = None   # (t, {fid: und_uv})
    errs, werrs = [], []   # flow error [px], |w| at that time
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2) if len(feats) else np.zeros((0, 2))
        cur = {int(f["id"]): uv for f, uv in zip(feats, und)}

        if prev is not None and i % args.every == 0 and cur:
            t0, uv0 = prev
            common = [fid for fid in cur if fid in uv0]
            if common:
                t_wc0 = cam_pose(t0); t_cw0 = np.linalg.inv(t_wc0)
                cloud_c = cloud @ t_cw0[:3, :3].T + t_cw0[:3, 3]
                p0 = [uv0[fid] for fid in common]
                d0 = gt_depth_at(cloud_c, p0, fx, fy, cx, cy, w, h)
                t_cw1 = np.linalg.inv(cam_pose(t))
                wmag = float(np.interp(t, gt_t, gt_w))
                for fid, (u0, v0), d in zip(common, p0, d0):
                    if not np.isfinite(d):
                        continue
                    pc0 = np.array([(u0-cx)/fx*d, (v0-cy)/fy*d, d])
                    xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
                    pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
                    if pc1[2] < 0.1:
                        continue
                    ugt = fx*pc1[0]/pc1[2] + cx
                    vgt = fy*pc1[1]/pc1[2] + cy
                    u1, v1 = cur[fid]
                    errs.append(np.hypot(u1-ugt, v1-vgt))
                    werrs.append(wmag)
        prev = (t, cur)
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] scored={len(errs)} "
                  f"{i/max(time.time()-tstart,1e-9):.0f}fps")

    e = np.array(errs); wm = np.array(werrs)
    inl = e < 3.0
    print(f"\n=== GT-flow tracker scorecard ({len(e)} feature-pairs, "
          f"every {args.every} frames) ===")
    print(f"flow err: median {np.median(e):.3f} px  p90 {np.percentile(e,90):.3f}  "
          f"p99 {np.percentile(e,99):.2f}")
    print(f"outliers (>3 px): {100*np.mean(~inl):.1f}%   (>1 px: {100*np.mean(e>1):.1f}%)")
    sig = np.sqrt(np.mean(e[inl]**2) / 2.0)
    print(f"effective sigma_pixel (inlier RMS/sqrt2): {sig:.3f} px   "
          f"(filter assumes 0.5)")
    hi = wm > np.percentile(wm, 75)
    print(f"by rotation: hi-|w| median {np.median(e[hi]):.3f} px, outliers "
          f"{100*np.mean(e[hi]>=3):.1f}%   calm {np.median(e[~hi]):.3f} px, "
          f"{100*np.mean(e[~hi]>=3):.1f}%")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(e, 0, 5), bins=100)
    ax[0].set_xlabel("flow error [px]"); ax[0].set_title("GT-flow error (clip 5px)")
    ax[1].scatter(wm, np.clip(e, 0, 10), s=3, alpha=0.15)
    ax[1].set_xlabel("GT |w| [rad/s]"); ax[1].set_ylabel("flow err [px] (clip 10)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
