"""Offline test: can mature Sparse3D landmarks retain previously observed scale?

The diagnostic never feeds Sparse3D back into EqVIO.  It finds a sustained
excited-to-degenerate transition from the raw-IMU proxy E, freezes mature
Sparse3D points in the transition camera frame, and later estimates scale from
both surviving landmark bearings and a robust rigid 3-D alignment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md
import run_manifest
from midair_vio_sparse3d_prior_ab import (
    cap_eqf_observations, gt_body_velocity, init_vio, nwu_body_pose,
    read_gray, sparse_settings, vio_body_pose,
)
import echo_li


def rolling_excitation(ds, window=500, hop=100):
    imu = ds.db[ds.traj]["imu"]
    acc = np.asarray(imu["accelerometer"], float)
    gyr = np.asarray(imu["gyroscope"], float)
    starts = np.arange(0, len(acc) - window + 1, hop)
    score = np.array([
        np.std(gyr[i:i + window, 2]) * np.std(acc[i:i + window, 1])
        for i in starts
    ])
    centers = (starts + window // 2) / 100.0
    return centers, score


def find_transition(centers, score, safe=0.05, degenerate=0.005, sustain=3):
    candidates = []
    for j in range(sustain - 1, len(score) - sustain):
        if (np.median(score[j - sustain + 1:j + 1]) > safe and
                np.median(score[j + 1:j + 1 + sustain]) < degenerate):
            candidates.append(j)
    if not candidates:
        raise RuntimeError("no sustained SAFE->DEGENERATE transition")
    j = candidates[-1]
    return float(centers[j]), j


def bearing(uv, f, cx, cy):
    b = np.array([(uv[0] - cx) / f, (uv[1] - cy) / f, 1.0])
    return b / np.linalg.norm(b)


def robust_scale(points_s, bearings_c, r_cs, t_cs):
    def objective(alpha):
        q = (r_cs @ points_s.T).T + alpha * t_cs
        q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
        angle = np.arccos(np.clip(np.sum(q * bearings_c, axis=1), -1.0, 1.0))
        # Truncated angular loss: surviving KLT tracks can still contain swaps.
        return float(np.mean(np.minimum(angle * angle, np.deg2rad(3.0) ** 2)))
    out = minimize_scalar(objective, bounds=(0.1, 3.0), method="bounded")
    return float(out.x), float(out.fun)


def rigid_umeyama(src, dst):
    """Least-squares SE(3) mapping src to dst: dst ~= R @ src + t."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    u, _, vt = np.linalg.svd(xd.T @ xs)
    fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        fix[-1, -1] = -1.0
    rot = u @ fix @ vt
    trans = mu_d - rot @ mu_s
    return rot, trans


def robust_rigid_umeyama(src, dst, trim_fraction=0.75, iterations=4):
    """Rigid Umeyama with iterative trimming of large 3-D residuals."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    keep_count = min(len(src), max(5, int(np.ceil(trim_fraction * len(src)))))
    keep = np.arange(len(src))
    for _ in range(iterations):
        rot, trans = rigid_umeyama(src[keep], dst[keep])
        pred = (rot @ src.T).T + trans
        new_keep = np.argsort(np.linalg.norm(pred - dst, axis=1))[:keep_count]
        if np.array_equal(np.sort(new_keep), np.sort(keep)):
            break
        keep = new_keep
    rot, trans = rigid_umeyama(src[keep], dst[keep])
    pred = (rot @ src[keep].T).T + trans
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst[keep]) ** 2, axis=1))))
    return rot, trans, rmse, len(keep)


def similarity_scale(x, y):
    x = np.asarray(x); y = np.asarray(y)
    xc = x - x.mean(0); yc = y - y.mean(0)
    den = float(np.sum(xc * xc))
    return float(np.sum(xc * yc) / den) if den > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="Kite_training")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eqf-selection", choices=["grid", "preserve_order"], default="grid")
    ap.add_argument("--eqf-max-obs", type=int, default=40)
    ap.add_argument("--min-track", type=int, default=20)
    ap.add_argument("--min-snapshot-features", type=int, default=12)
    ap.add_argument("--max-snapshot-features", type=int, default=80)
    ap.add_argument("--save-npz", required=True)
    args = ap.parse_args()
    args.start = 0

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, 0.5)
    im0 = read_gray(ds, 0); h, w = im0.shape
    f, cx, cy = md.intrinsics(w, h)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    ext = md.RT_BC.copy()
    centers, excitation = rolling_excitation(ds)
    transition_s, transition_idx = find_transition(centers, excitation)
    transition_frame = int(round(transition_s * 25.0))

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.max_features = 300
    fcfg.set_camera(f, f, cx, cy, w, h, [])
    tracker = echo_li.Frontend(fcfg, w, h)
    vio = init_vio(args, cam, ext, ds)
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
        cam, **sparse_settings(args.config))

    imu = ds.db[ds.traj]["imu"]
    acc = np.asarray(imu["accelerometer"])
    gyr = np.asarray(imu["gyroscope"])
    events = [(i / 100.0, "imu", i) for i in range(len(acc))]
    events += [(k / 25.0, "cam", k) for k in range(ds.n)]
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    frozen = None
    t_wc_ref = None
    gt_ref = None
    est_hist, gt_hist, rows = [], [], []
    for stamp, kind, idx in events:
        if kind == "imu":
            gi = min(idx, len(ds.att) - 1)
            q = ds.att[gi]
            from scipy.spatial.transform import Rotation as Rot
            r_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            vio.process_imu(stamp, (r_gt.T @ gyr[idx]).tolist(), acc[idx].tolist())
            continue

        k = idx
        feats, _ = tracker.process(read_gray(ds, k))
        all_uvs = {int(x["id"]): (float(x["x"]), float(x["y"])) for x in feats}
        t_wc = vio_body_pose(vio) @ ext
        pc = vio.get_camera_pose_covariance()
        pvv = None if pc is None else np.asarray(pc[0], float).tolist()
        pww = None if pc is None else np.asarray(pc[1], float).tolist()
        sparse.update(stamp, all_uvs, t_wc.tolist(), pvv, pww)
        existing = {int(x) for x in vio.get_landmarks()}
        vio_uvs = cap_eqf_observations(
            all_uvs, existing, {}, args.eqf_max_obs, args.eqf_selection, w, h)
        vio.process_vision(stamp, vio_uvs)
        t_wc = vio_body_pose(vio) @ ext
        gt_wc = nwu_body_pose(ds, k) @ ext
        est_hist.append(t_wc[:3, 3].copy()); gt_hist.append(gt_wc[:3, 3].copy())

        if frozen is None and k >= transition_frame:
            sf = sparse.get_features()
            candidate_points = {}
            candidate_uvs = {}
            for fid, uv in all_uvs.items():
                fd = sf.get(fid)
                if fd is None or int(fd["track_length"]) < args.min_track:
                    continue
                p = np.asarray(fd["position"], float)
                if np.isfinite(p).all() and 1.0 < np.linalg.norm(p) < 5000.0:
                    candidate_points[fid] = p.copy()
                    candidate_uvs[fid] = uv
            chosen = cap_eqf_observations(
                candidate_uvs, set(), {}, args.max_snapshot_features, "grid", w, h)
            if len(chosen) >= args.min_snapshot_features:
                frozen = {fid: candidate_points[fid] for fid in chosen}
                t_wc_ref = t_wc.copy(); gt_ref = gt_wc.copy()
                transition_frame = k

        if frozen is not None:
            sf_live = sparse.get_features()
            ids = [fid for fid in frozen if fid in all_uvs]
            cloud_ids = [fid for fid in ids if fid in sf_live]
            alpha = loss = np.nan
            rigid_alpha = rigid_rmse = np.nan
            rigid_inliers = 0
            if len(ids) >= 5:
                points = np.array([frozen[fid] for fid in ids])
                bearings = np.array([bearing(all_uvs[fid], f, cx, cy) for fid in ids])
                t_cs = np.linalg.inv(t_wc) @ t_wc_ref
                alpha, loss = robust_scale(points, bearings, t_cs[:3, :3], t_cs[:3, 3])
            start = max(0, transition_frame - 125)
            s_ref = similarity_scale(np.array(est_hist[start:transition_frame + 1]),
                                     np.array(gt_hist[start:transition_frame + 1]))
            de = np.linalg.norm(t_wc[:3, 3] - t_wc_ref[:3, 3])
            dg = np.linalg.norm(gt_wc[:3, 3] - gt_ref[:3, 3])
            oracle = dg / (s_ref * de) if de > 1e-6 and s_ref > 1e-9 else np.nan
            if len(cloud_ids) >= 5:
                points_ref = np.array([frozen[fid] for fid in cloud_ids])
                points_live = np.array(
                    [sf_live[fid]["position"] for fid in cloud_ids], float)
                finite = np.isfinite(points_live).all(axis=1)
                if finite.sum() >= 5:
                    _, rigid_t, rigid_rmse, rigid_inliers = robust_rigid_umeyama(
                        points_ref[finite], points_live[finite])
                    if de > 1e-6:
                        rigid_alpha = np.linalg.norm(rigid_t) / de
            rows.append((k, stamp, len(ids), alpha, oracle, loss, de, dg,
                         rigid_alpha, rigid_rmse, rigid_inliers))

    if frozen is None:
        raise RuntimeError("transition found but insufficient mature snapshot landmarks")
    out = np.asarray(rows, float)
    np.savez(args.save_npz, data=out,
             cols=np.array(["frame", "time", "survivors", "alpha", "oracle",
                            "loss", "est_disp", "gt_disp", "rigid_alpha",
                            "rigid_rmse", "rigid_inliers"]),
             excitation_centers=centers, excitation=excitation,
             transition=np.array([transition_frame, transition_s, transition_idx]),
             frozen_ids=np.array(list(frozen), int))
    run_manifest.save_run_manifest(args.save_npz, args.config, extra=vars(args))
    valid = np.isfinite(out[:, 3]) & np.isfinite(out[:, 4])
    print(f"traj {args.traj}: transition frame {transition_frame}, frozen {len(frozen)}, "
          f"valid scale frames {valid.sum()}/{len(out)}")
    if valid.any():
        err = np.abs(np.log(out[valid, 3] / out[valid, 4]))
        print(f"median |log(alpha/oracle)|={np.median(err):.3f}, "
              f"p90={np.quantile(err, .9):.3f}, last valid frame={int(out[valid, 0][-1])}")
    rigid_valid = np.isfinite(out[:, 8]) & np.isfinite(out[:, 4]) & (out[:, 8] > 0)
    if rigid_valid.any():
        err = np.abs(np.log(out[rigid_valid, 8] / out[rigid_valid, 4]))
        print(f"rigid Umeyama median |log(alpha/oracle)|={np.median(err):.3f}, "
              f"p90={np.quantile(err, .9):.3f}, "
              f"last valid frame={int(out[rigid_valid, 0][-1])}")


if __name__ == "__main__":
    main()
