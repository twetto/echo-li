"""GT-flow in action: video of tracked points vs pointcloud+pose ground truth.

Per frame k: take features tracked from k-1, look up GT depth at their k-1 pixel
(Leica scan z-buffered with the Vicon pose), reproject the 3D point into frame k
-> GT-predicted position. Draw on the UNDISTORTED frame k:
  dot   = tracker position, colored by |err|: green <1 px, yellow 1-3, red >3
  line  = error vector tracker->GT, magnified x10 (sub-px errors invisible at x1)
HUD: time, GT |w|, median err, outlier fraction.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/flow_gt_viz.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
from flow_gt_eval import quat_ang_rate  # noqa: E402
import echo_li  # noqa: E402

MAG = 10.0  # error-vector magnification


def distort_radtan(pc, fx, fy, cx, cy, d):
    """Project camera-frame point through the full radtan model -> RAW pixel."""
    x, y = pc[0]/pc[2], pc[1]/pc[2]
    k1, k2, p1, p2 = d[:4]
    r2 = x*x + y*y
    rad = 1.0 + k1*r2 + k2*r2*r2
    xd = x*rad + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*rad + p1*(r2 + 2*y*y) + 2*p2*x*y
    return fx*xd + cx, fy*yd + cy


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--domain", choices=["raw", "undist"], default="raw",
                    help="raw = distorted frames (where KLT actually tracks)")
    args = ap.parse_args()
    out = args.out or f"flow_gt_viz_{args.domain}.mp4"
    root = Path(args.dataset)
    if (root/"mav0").exists():
        root = root/"mav0"

    cfg = yaml.safe_load(open(root/"cam0"/"sensor.yaml"))
    w, h = cfg["resolution"]; fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    undist_map = cv2.initUndistortRectifyMap(K, dcoef[:4], None, K, (w, h), cv2.CV_32FC1)

    gt = load_csv(root/"state_groundtruth_estimate0"/"data.csv")
    gt_t = gt[:, 0]*1e-9; gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])

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
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    prev = None  # (frame index, t, {fid: (und_uv, disp_uv)})
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2) if len(feats) else np.zeros((0, 2))
        # cur: fid -> (undistorted uv for geometry, display uv in the chosen domain)
        raw_mode = args.domain == "raw"
        cur = {int(f["id"]): (uv, (f["x"], f["y"]) if raw_mode else tuple(uv))
               for f, uv in zip(feats, und)}

        # display with global histeq — matches what the tracker actually sees
        # (equaliseImageHistogram: true in the config)
        base = cv2.equalizeHist(img if raw_mode
                                else cv2.remap(img, *undist_map, cv2.INTER_LINEAR))
        vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        errs = []
        if prev is not None and cur:
            i0, t0, uv0 = prev
            common = [fid for fid in cur if fid in uv0]
            if common:
                t_wc0 = cam_pose(t0)
                p0 = [uv0[fid][0] for fid in common]   # undistorted uv (geometry)
                d0 = zbuf_lookup(zbufs[i0], p0, w, h)
                t_cw1 = np.linalg.inv(cam_pose(t))
                for fid, (u0, v0), d in zip(common, p0, d0):
                    u1, v1 = cur[fid][1]               # display-domain position
                    if not np.isfinite(d):
                        cv2.circle(vis, (int(u1), int(v1)), 2, (128, 128, 128), -1)
                        continue
                    pc0 = np.array([(u0-cx)/fx*d, (v0-cy)/fy*d, d])
                    xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
                    pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
                    if pc1[2] < 0.1:
                        continue
                    if raw_mode:
                        ugt, vgt = distort_radtan(pc1, fx, fy, cx, cy, dcoef)
                    else:
                        ugt = fx*pc1[0]/pc1[2] + cx
                        vgt = fy*pc1[1]/pc1[2] + cy
                    e = float(np.hypot(u1-ugt, v1-vgt))
                    errs.append(e)
                    col = ((0, 200, 0) if e < 1.0 else
                           (0, 220, 220) if e < 3.0 else (0, 0, 255))
                    tip = (int(u1 + MAG*(ugt-u1)), int(v1 + MAG*(vgt-v1)))
                    cv2.line(vis, (int(u1), int(v1)), tip, col, 1, cv2.LINE_AA)
                    cv2.circle(vis, (int(u1), int(v1)), 2, col, -1)
        wmag = float(np.interp(t, gt_t, gt_w))
        med = np.median(errs) if errs else float("nan")
        outl = 100*np.mean(np.array(errs) > 3.0) if errs else 0.0
        cv2.putText(vis, f"[{args.domain}] t={t-gt_t[0]:5.1f}s |w|={wmag:4.2f} "
                    f"tracked-vs-GT flow: med {med:4.2f}px  >3px {outl:3.0f}%  (err x{MAG:.0f})",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        vw.write(vis)
        prev = (i, t, cur)
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] {i/max(time.time()-tstart,1e-9):.0f}fps")
    vw.release()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
