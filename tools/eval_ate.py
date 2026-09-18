#!/usr/bin/env python3
"""Compute ATE (Absolute Trajectory Error) between an estimated and ground-truth
trajectory, both in TUM format.  Prints RMSE, mean, and matched pose count.

Usage:
    python3 tools/eval_ate.py est.tum gt.tum [--max-gap 0.1] [--align se3|sim3]
    python3 tools/eval_ate.py est.tum gt.tum --t-bc "row-major 4x4 floats"
"""
import argparse
import sys
import numpy as np
from scipy.spatial.transform import Rotation


def load_tum(path):
    stamps, positions, quats = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            qxyzw = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            stamps.append(t)
            positions.append(xyz)
            quats.append(qxyzw)
    return np.array(stamps), np.array(positions), np.array(quats)


def associate(stamps_est, stamps_gt, max_gap):
    pairs = []
    j = 0
    for i, t in enumerate(stamps_est):
        while j < len(stamps_gt) - 1 and stamps_gt[j + 1] <= t:
            j += 1
        best = j
        if j + 1 < len(stamps_gt) and abs(stamps_gt[j + 1] - t) < abs(stamps_gt[j] - t):
            best = j + 1
        if abs(stamps_gt[best] - t) <= max_gap:
            pairs.append((i, best))
    return pairs


def umeyama_se3(src, dst):
    """SE(3) alignment (no scale): R, t = argmin ||dst - (R @ src + t)||."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, d])
    R = Vt.T @ S @ U.T
    t = mu_dst - R @ mu_src
    return R, t, 1.0


def umeyama_sim3(src, dst):
    """Sim(3) alignment (with scale)."""
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c
    U, S_diag, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S_mat = np.diag([1, 1, d])
    R = Vt.T @ S_mat @ U.T
    var_src = np.sum(src_c ** 2) / n
    s = np.sum(S_diag * np.diag(S_mat)[:len(S_diag)]) / var_src if var_src > 0 else 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def main():
    ap = argparse.ArgumentParser(description="Compute ATE between TUM trajectories")
    ap.add_argument("est", help="Estimated trajectory (TUM format)")
    ap.add_argument("gt", help="Ground-truth trajectory (TUM format)")
    ap.add_argument("--max-gap", type=float, default=0.1,
                    help="Max timestamp gap for association (default: 0.1s)")
    ap.add_argument("--align", choices=["se3", "sim3"], default="se3",
                    help="Alignment type (default: se3)")
    ap.add_argument("--t-bc", type=str, default=None,
                    help="Body-to-camera SE(3) as 16 row-major floats (space-separated), "
                         "or path to a file with a 4x4 matrix. Transforms body-frame est "
                         "to camera frame before comparing with camera-frame GT.")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="Add this offset to estimated timestamps before association "
                         "(e.g. timeshift_cam_imu from Kalibr).")
    args = ap.parse_args()

    stamps_est, pos_est, quat_est = load_tum(args.est)
    stamps_gt, pos_gt, _ = load_tum(args.gt)

    if args.time_offset != 0.0:
        stamps_est = stamps_est + args.time_offset

    if args.t_bc is not None:
        import os
        if os.path.isfile(args.t_bc):
            t_bc = np.loadtxt(args.t_bc).reshape(4, 4)
        else:
            t_bc = np.array([float(x) for x in args.t_bc.split()]).reshape(4, 4)
        for i in range(len(pos_est)):
            R_wb = Rotation.from_quat(quat_est[i]).as_matrix()
            T_wb = np.eye(4)
            T_wb[:3, :3] = R_wb
            T_wb[:3, 3] = pos_est[i]
            T_wc = T_wb @ t_bc
            pos_est[i] = T_wc[:3, 3]
            quat_est[i] = Rotation.from_matrix(T_wc[:3, :3]).as_quat()

    if len(stamps_est) == 0:
        print("ERROR: estimated trajectory is empty", file=sys.stderr)
        sys.exit(1)
    if len(stamps_gt) == 0:
        print("ERROR: ground-truth trajectory is empty", file=sys.stderr)
        sys.exit(1)

    pairs = associate(stamps_est, stamps_gt, args.max_gap)
    if len(pairs) < 3:
        print(f"ERROR: only {len(pairs)} matched poses (need >= 3)", file=sys.stderr)
        sys.exit(1)

    idx_est = [p[0] for p in pairs]
    idx_gt = [p[1] for p in pairs]
    est = pos_est[idx_est]
    gt = pos_gt[idx_gt]

    if args.align == "sim3":
        R, t, s = umeyama_sim3(est, gt)
    else:
        R, t, s = umeyama_se3(est, gt)

    aligned = s * (est @ R.T) + t
    errors = np.linalg.norm(aligned - gt, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean = np.mean(errors)
    median = np.median(errors)
    mx = np.max(errors)

    path_est = np.sum(np.linalg.norm(np.diff(pos_est, axis=0), axis=1))
    path_gt = np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))

    print(f"ATE ({args.align.upper()} alignment, scale={s:.4f}):")
    print(f"  RMSE:    {rmse:.4f} m  ({rmse*100:.2f} cm)")
    print(f"  Mean:    {mean:.4f} m")
    print(f"  Median:  {median:.4f} m")
    print(f"  Max:     {mx:.4f} m")
    print(f"  Matched: {len(pairs)} / {len(stamps_est)} est, {len(stamps_gt)} gt")
    print(f"  Path:    est={path_est:.2f} m, gt={path_gt:.2f} m (ratio={path_est/path_gt:.3f})")

    sys.exit(0 if rmse < 0.10 else 1)


if __name__ == "__main__":
    main()
