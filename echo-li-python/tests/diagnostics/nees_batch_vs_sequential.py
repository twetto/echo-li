"""Batch vs sequential depth NEES on the SAME real tracks (the decisive cut).

Isolates filter-formulation inconsistency (sequential Sigma over-collapse) from
measurement inconsistency (correlated flow bias). Both estimators see identical
observations {(T_k, u_k)} and GT poses; the batch estimator solves the joint
Gauss-Newton least-squares per landmark with its Fisher covariance
P_q = sigma^2 (sum J_k^T J_k)^-1 — provably consistent for the linear-Gaussian
part, so it shares the measurement/geometry error but removes the sequential term.

  batch NEES flat ~1 in track length   -> fault is the FILTER (fix: batch/IEKF/decimate)
  batch NEES also grows / large        -> fault is the MEASUREMENTS (correlated flow bias)

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/nees_batch_vs_sequential.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--chart invdepth] [--every 10]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

import echo_li

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, load_cloud, gt_depth_at  # noqa: E402

SIGMA_PX = 0.5
R_AGES = (0.0, 0.15, 0.30, 0.45)  # px/frame template-drift rate for R(age) sweep


def batch_solve(obs, fx, fy, cx, cy, q_init, r_age=0.0, iters=10):
    """Joint GN for q (world) from obs=[(T_cw 4x4, u 2)], ordered by age.
    Per-obs noise rho_k^2 = SIGMA_PX^2 + (r_age*age)^2 models template drift;
    P_q = (sum rho_k^-2 J_k^T J_k)^-1. Returns (q, P_q) or None."""
    q = q_init.astype(float).copy()
    JtJ = None
    for _ in range(iters):
        JtJ = np.zeros((3, 3))
        Jtr = np.zeros(3)
        for age, (T_cw, u) in enumerate(obs):
            R = T_cw[:3, :3]
            x = R @ q + T_cw[:3, 3]
            if x[2] <= 1e-6:
                return None
            inv = 1.0 / x[2]
            up = np.array([fx * x[0] * inv + cx, fy * x[1] * inv + cy])
            resid = u - up
            dudx = np.array([[fx * inv, 0, -fx * x[0] * inv * inv],
                             [0, fy * inv, -fy * x[1] * inv * inv]])
            J = dudx @ R
            wk = 1.0 / (SIGMA_PX ** 2 + (r_age * age) ** 2)
            JtJ += wk * (J.T @ J)
            Jtr += wk * (J.T @ resid)
        try:
            dq = np.linalg.solve(JtJ + 1e-12 * np.eye(3), Jtr)
        except np.linalg.LinAlgError:
            return None
        q += dq
        if np.linalg.norm(dq) < 1e-7:
            break
    try:
        Pq = np.linalg.inv(JtJ)  # weighting already carries 1/rho^2
    except np.linalg.LinAlgError:
        return None
    return q, Pq


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--min-track", type=int, default=10)
    ap.add_argument("--chart", default="invdepth", choices=["invdepth", "bearing"])
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))
    cloud = load_cloud(root / "pointcloud0" / "data.ply")
    print(f"cloud {len(cloud)} pts; cam {w}x{h}")

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)
    sf = dict(sigma_pixel=SIGMA_PX, min_track_length=1)
    if args.chart == "bearing":
        filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
            echo_li.PinholeCamera(fx, fy, cx, cy), **sf)
    else:
        filt = echo_li.Sparse3DFilter.invdepth_additive3d(fx, fy, cx, cy, **sf)

    idir = root / "cam0" / "data"
    frames = []
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r:
                frames.append((int(r[0]) * 1e-9, idir / r[1].strip()))
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]

    obs = {}  # fid -> list of (T_cw 4x4, u_pinholeK 2)
    rec = []  # (z_seq, z_batch, track_len, seq_err, batch_err)
    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        if not feats:
            continue
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
        ids = [int(f["id"]) for f in feats]
        uvs = {fid: (float(u), float(v)) for fid, (u, v) in zip(ids, und)}

        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(t).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        t_wc = t_wb @ t_bs
        t_cw = np.linalg.inv(t_wc)

        filt.update(t, uvs, t_wc.tolist(), None, None)
        for fid in ids:
            obs.setdefault(fid, []).append((t_cw, np.array(uvs[fid])))

        if i % args.every != 0:
            continue
        fdict = filt.get_features()
        live = {fid: fd for fid, fd in fdict.items()
                if fd["track_length"] >= args.min_track and fid in uvs}
        if not live:
            continue
        cloud_c = cloud @ t_cw[:3, :3].T + t_cw[:3, 3]
        fids = list(live.keys())
        gtd = gt_depth_at(cloud_c, [uvs[f] for f in fids], fx, fy, cx, cy, w, h)
        for fid, d in zip(fids, gtd):
            if not np.isfinite(d):
                continue
            fd = live[fid]
            est_z = fd["position"][2]
            var_z = np.asarray(fd["covariance_euclidean"])[2, 2]
            if var_z <= 0:
                continue
            # batch R(age) sweep, initialised from the filter estimate -> world
            q_init = t_wc[:3, :3] @ np.asarray(fd["position"]) + t_wc[:3, 3]
            hb = t_cw[2, :3]
            zbs = []
            ok = True
            for r_age in R_AGES:
                bt = batch_solve(obs[fid], fx, fy, cx, cy, q_init, r_age=r_age)
                if bt is None:
                    ok = False
                    break
                q_w, Pq = bt
                zc = (t_cw[:3, :3] @ q_w + t_cw[:3, 3])[2]
                pzz = hb @ Pq @ hb
                if pzz <= 0:
                    ok = False
                    break
                zbs.append((zc - d) / np.sqrt(pzz))
            if not ok:
                continue
            rec.append((fd["track_length"], (est_z - d) / np.sqrt(var_z), *zbs))
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] live={len(live)} scored={len(rec)}")

    a = np.array(rec)
    tl = a[:, 0]
    zs = a[:, 1]
    zbat = {r: a[:, 2 + j] for j, r in enumerate(R_AGES)}
    cols = "  ".join(f"r={r:.2f}" for r in R_AGES)
    print(f"\n=== batch R(age) sweep vs sequential ({args.chart}, {len(a)} obs) ===")
    print(f"{'':>12} | {'seq':>8} | {cols}")
    print(f"{'NEES median':>12} | {np.median(zs**2):8.1f} | "
          + "  ".join(f"{np.median(zbat[r]**2):6.1f}" for r in R_AGES))
    print("median NEES-1D by track length (r = px/frame drift rate in R(age)):")
    print(f"{'len':>10} | {'n':>6} | {'seq':>8} | " + "  ".join(f"r={r:.2f}" for r in R_AGES))
    for lo, hi in [(10, 20), (20, 40), (40, 80), (80, 160), (160, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if not m.any():
            continue
        hs = "inf" if hi > 1e8 else f"{int(hi)}"
        row = f"  {lo:>3}-{hs:>4} | {int(m.sum()):6d} | {np.median(zs[m]**2):8.1f} | "
        row += "  ".join(f"{np.median(zbat[r][m]**2):6.1f}" for r in R_AGES)
        print(row)


if __name__ == "__main__":
    main()
