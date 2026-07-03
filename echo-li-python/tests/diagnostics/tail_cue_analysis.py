"""Which signal predicts a bad track? GT-supervised feature-importance study.

For every GT-scored feature pair (flow error from the pointcloud+pose GT, as in
flow_gt_eval), record Rudolf-V's internal per-track signals plus externally
computable cues, and rank them by how well they discriminate outliers (>3 px):

  internal:  age, klt_quality (needs kltResidual: true), lbp_distance,
             reservoir_score, corner score (birth-time detection strength)
  external:  Shi-Tomasi min-eig at pyramid L0/L1/L2 (the coarse-to-fine gate cue),
             depth-edge proximity (3x3 z-buffer max-min; occlusion cue),
             Sampson epipolar distance under the GT essential matrix
             (the idealized 'RANSAC residual'), flow magnitude

Outputs per-signal AUC (rank-based, sign-adjusted) + outlier-rate by decile for
the top signals.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/tail_cue_analysis.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
from flow_gt_eval import quat_ang_rate  # noqa: E402
import echo_li  # noqa: E402

CELL = 4
OUTLIER_PX = 3.0


def sampson_px(p0n, p1n, e, f_px):
    """Sampson distance (in ~pixels) of normalized correspondences under E."""
    x0 = np.hstack([p0n, np.ones((len(p0n), 1))])
    x1 = np.hstack([p1n, np.ones((len(p1n), 1))])
    ex0 = x0 @ e.T          # E x0
    etx1 = x1 @ e           # E^T x1
    num = np.einsum("ij,ij->i", x1, ex0) ** 2
    den = ex0[:, 0]**2 + ex0[:, 1]**2 + etx1[:, 0]**2 + etx1[:, 1]**2
    return f_px * np.sqrt(num / np.maximum(den, 1e-30))


def auc(signal, is_out):
    """Rank AUC of signal for predicting is_out; sign-adjusted (>=0.5)."""
    g = np.isfinite(signal)
    s, y = signal[g], is_out[g]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, 1
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    a = (r[y].sum() - n1*(n1+1)/2) / (n1*n0)
    return (a, 1) if a >= 0.5 else (1-a, -1)


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
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

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.klt_residual = True   # make klt_quality live for this analysis
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)
    # full VIO alongside: the runtime-realistic pose source for the epipolar cue
    cam = echo_li.RadTanCamera(fx, fy, cx, cy, *dcoef[:4])
    vio = echo_li.VIOFilter(args.config, cam)
    vio.set_camera_extrinsics(t_bs)
    imu = load_csv(root/"imu0"/"data.csv")
    imu_ev = [(r[0]*1e-9, r[1:4].tolist(), r[4:7].tolist()) for r in imu]

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
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h, CELL)
    gw = w // CELL + 1; gh = h // CELL + 1

    cols = ["err", "age", "klt_quality", "lbp_distance", "reservoir_score",
            "corner_score", "mineig_L0", "mineig_L1", "mineig_L2",
            "depth_edge", "sampson_gtE", "sampson_vio", "flow_mag"]
    rows = []
    prev = None   # (i, t, {fid: (und, raw)}, t_wc_vio or None)
    imu_i = 0
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        while imu_i < len(imu_ev) and imu_ev[imu_i][0] <= t:
            vio.process_imu(*imu_ev[imu_i])
            imu_i += 1
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        t_wc_vio = None
        if vio.is_initialized:
            vio.process_vision(t, {f["id"]: (f["x"], f["y"]) for f in feats})
            pos, quat = vio.get_pose()
            t_wb = np.eye(4)
            t_wb[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
            t_wb[:3, 3] = np.asarray(pos)
            t_wc_vio = t_wb @ t_bs
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2) if len(feats) else np.zeros((0, 2))
        cur = {int(f["id"]): (uv, (f["x"], f["y"]), f["score"])
               for f, uv in zip(feats, und)}
        # min-eig pyramid on the histeq'd image (what the tracker sees)
        eq = cv2.equalizeHist(img)
        me = [cv2.cornerMinEigenVal(eq, 3)]
        lv = eq
        for _ in range(2):
            lv = cv2.pyrDown(lv)
            me.append(cv2.cornerMinEigenVal(lv, 3))

        if prev is not None and cur:
            i0, t0, uv0, t_wc0_vio = prev
            common = [fid for fid in cur if fid in uv0 and fid in meta]
            if common:
                def e_from(t_wc0_, t_wc1_):
                    t_10 = np.linalg.inv(t_wc1_) @ t_wc0_    # c0 -> c1
                    r10, t10 = t_10[:3, :3], t_10[:3, 3]
                    tn = t10 / max(np.linalg.norm(t10), 1e-12)
                    return np.array([[0, -tn[2], tn[1]], [tn[2], 0, -tn[0]],
                                     [-tn[1], tn[0], 0]]) @ r10
                t_wc0 = cam_pose(t0)
                p0u = np.array([uv0[fid][0] for fid in common])
                d0 = zbuf_lookup(zbufs[i0], p0u, w, h, CELL)
                t_wc1 = cam_pose(t)
                p1u = np.array([cur[fid][0] for fid in common])
                p0n = (p0u - [cx, cy]) / [fx, fy]
                p1n = (p1u - [cx, cy]) / [fx, fy]
                samp = sampson_px(p0n, p1n, e_from(t_wc0, t_wc1), fx)
                if t_wc_vio is not None and t_wc0_vio is not None:
                    samp_v = sampson_px(p0n, p1n, e_from(t_wc0_vio, t_wc_vio), fx)
                else:
                    samp_v = np.full(len(common), np.nan)
                t_cw1 = np.linalg.inv(t_wc1)
                zb0 = zbufs[i0]
                for k, fid in enumerate(common):
                    d = d0[k]
                    if not np.isfinite(d):
                        continue
                    u0, v0 = p0u[k]
                    pc0 = np.array([(u0-cx)/fx*d, (v0-cy)/fy*d, d])
                    xw = t_wc0[:3, :3] @ pc0 + t_wc0[:3, 3]
                    pc1 = t_cw1[:3, :3] @ xw + t_cw1[:3, 3]
                    if pc1[2] < 0.1:
                        continue
                    ugt = fx*pc1[0]/pc1[2] + cx
                    vgt = fy*pc1[1]/pc1[2] + cy
                    u1, v1 = p1u[k]
                    err = float(np.hypot(u1-ugt, v1-vgt))
                    m = meta[fid]
                    # depth-edge: max-min finite depth in 3x3 z-buffer cells at k-1
                    ci, cj = int(v0/CELL), int(u0/CELL)
                    nb = zb0[np.ravel_multi_index(
                        np.meshgrid(np.clip(np.arange(ci-1, ci+2), 0, gh-1),
                                    np.clip(np.arange(cj-1, cj+2), 0, gw-1),
                                    indexing="ij"),
                        (gh, gw)).ravel()]
                    nb = nb[np.isfinite(nb)]
                    dedge = float(nb.max() - nb.min()) if len(nb) else np.nan
                    # min-eig per level at the current (raw) position
                    rx, ry = cur[fid][1]
                    mes = []
                    for L, mmap_ in enumerate(me):
                        xl, yl = int(rx / 2**L), int(ry / 2**L)
                        hL, wL = mmap_.shape
                        mes.append(float(mmap_[min(yl, hL-1), min(xl, wL-1)]))
                    rows.append([err, m["age"], m["klt_quality"], m["lbp_distance"],
                                 m["reservoir_score"], cur[fid][2],
                                 mes[0], mes[1], mes[2], dedge, float(samp[k]),
                                 float(samp_v[k]), float(np.hypot(u1-u0, v1-v0))])
        prev = (i, t, cur, t_wc_vio)
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] rows={len(rows)} "
                  f"{i/max(time.time()-tstart,1e-9):.0f}fps")

    d = np.array(rows)
    err = d[:, 0]
    out = err > OUTLIER_PX
    print(f"\n=== tail cue ranking ({len(d)} pairs, outliers {100*out.mean():.1f}% "
          f">{OUTLIER_PX}px) ===")
    print(f"{'signal':>18} {'AUC':>6} {'direction':>22}")
    results = []
    for j, name in enumerate(cols[1:], start=1):
        a, sgn = auc(d[:, j], out)
        results.append((a, name, j, sgn))
        arrow = "higher => outlier" if sgn > 0 else "LOWER => outlier"
        print(f"{name:>18} {a:6.3f}   {arrow:>22}")
    results.sort(reverse=True)
    print("\noutlier % by decile (top 4 signals; deciles low->high signal value):")
    for a, name, j, sgn in results[:4]:
        s = d[:, j]
        g = np.isfinite(s)
        q = np.quantile(s[g], np.linspace(0, 1, 11))
        rates = []
        for b in range(10):
            m_ = g & (s >= q[b]) & (s <= q[b+1])
            rates.append(100*out[m_].mean() if m_.sum() else np.nan)
        print(f"  {name:>18}: " + " ".join(f"{r:5.1f}" for r in rates))


if __name__ == "__main__":
    main()
