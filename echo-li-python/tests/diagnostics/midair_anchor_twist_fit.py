"""Can shared-anchor image residuals estimate Sparse3D's coherent pose lean?

Runs the normal MidAir frontend + EqVIO + Sparse3D pipeline.  When a feature
reaches ``min_track`` its current Sparse3D point is transformed back to its
recorded birth/anchor camera frame and frozen.  Features born in the same frame
form one anchor cohort.  A constant SE(3) twist rate is fitted from an early set
of cohort observations and scored on disjoint later observations.

Ground truth is never used by the fit.  It is used only to score whether the
image-fitted correction improves the held-out anchor-to-current relative pose.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import midair_drift as md  # noqa: E402
from _se3 import exp_se3  # noqa: E402
from midair_vio_sparse3d_prior_ab import (  # noqa: E402
    init_vio, nwu_body_pose, read_gray, sparse_settings, vio_body_pose,
)
import echo_li  # noqa: E402


def project(q, f, cx, cy):
    if q[2] <= 1e-6:
        return np.array([1e6, 1e6])
    return np.array([f * q[0] / q[2] + cx, f * q[1] / q[2] + cy])


def relative_error(est_rel, gt_rel):
    d = gt_rel @ np.linalg.inv(est_rel)
    return np.linalg.norm(d[:3, 3]), np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec())


def fit_cohort(cohort, observations, est_poses, gt_poses, f, cx, cy,
               train_frames, test_frames, max_rate, depth_prior_sigma):
    ids = sorted(cohort)
    anchor = cohort[ids[0]]["anchor"]
    t0 = cohort[ids[0]]["stamp"]
    t_wc_a = cohort[ids[0]]["t_wc_anchor"]
    gt_wc_a = gt_poses[anchor]

    train = [(k, fid, observations[k][fid]) for k in train_frames
             for fid in ids if fid in observations[k]]
    test = [(k, fid, observations[k][fid]) for k in test_frames
            for fid in ids if fid in observations[k]]
    if len(train) < 40 or len(test) < 30:
        return None

    id_slot = {fid: j for j, fid in enumerate(ids)}

    def residual(x, rows, regularize=False):
        out = []
        for k, fid, uv in rows:
            dt = k / 25.0 - t0
            t_ca = np.linalg.inv(est_poses[k]) @ t_wc_a
            point = cohort[fid]["point_anchor"] * np.exp(x[6 + id_slot[fid]])
            q = t_ca[:3, :3] @ point + t_ca[:3, 3]
            q = (exp_se3(dt * x[:6]) @ np.r_[q, 1.0])[:3]
            out.extend(project(q, f, cx, cy) - uv)
        if regularize:
            out.extend(x[6:] / depth_prior_sigma)
        return np.asarray(out)

    x0 = np.zeros(6 + len(ids))
    zero_train = residual(x0, train)
    lo = np.r_[np.full(6, -max_rate), np.full(len(ids), -1.0)]
    hi = np.r_[np.full(6, max_rate), np.full(len(ids), 1.0)]
    opt = least_squares(
        lambda x: residual(x, train, regularize=True), x0,
        loss="soft_l1", f_scale=1.0, bounds=(lo, hi), max_nfev=60,
    )
    zero_test = residual(x0, test).reshape(-1, 2)
    fit_test = residual(opt.x, test).reshape(-1, 2)
    pix0 = float(np.median(np.linalg.norm(zero_test, axis=1)))
    pix1 = float(np.median(np.linalg.norm(fit_test, axis=1)))

    pose0, pose1 = [], []
    for k in test_frames:
        dt = k / 25.0 - t0
        xe = np.linalg.inv(est_poses[k]) @ t_wc_a
        xg = np.linalg.inv(gt_poses[k]) @ gt_wc_a
        pose0.append(relative_error(xe, xg))
        pose1.append(relative_error(exp_se3(dt * opt.x[:6]) @ xe, xg))
    pose0 = np.asarray(pose0); pose1 = np.asarray(pose1)
    return {
        "anchor": anchor, "features": len(ids), "train_obs": len(train),
        "test_obs": len(test), "twist": opt.x[:6], "cost": float(opt.cost),
        "train_pix0": float(np.median(np.linalg.norm(zero_train.reshape(-1, 2), axis=1))),
        "train_pix1": float(np.median(np.linalg.norm(residual(opt.x, train).reshape(-1, 2), axis=1))),
        "test_pix0": pix0, "test_pix1": pix1,
        "test_trans0": float(np.median(pose0[:, 0])),
        "test_trans1": float(np.median(pose1[:, 0])),
        "test_rot0": float(np.median(pose0[:, 1])),
        "test_rot1": float(np.median(pose1[:, 1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--min-track", type=int, default=20)
    ap.add_argument("--min-cohort", type=int, default=6)
    ap.add_argument("--train-length", type=int, default=24)
    ap.add_argument("--test-length", type=int, default=32)
    ap.add_argument("--max-rate", type=float, default=0.5,
                    help="symmetric bound on every twist-rate component per second")
    ap.add_argument("--depth-prior-sigma", type=float, default=0.25,
                    help="regularization sigma for per-feature log-range correction")
    ap.add_argument("--save-npz", required=True)
    args = ap.parse_args()
    args.start = 0

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, 0.5)
    im0 = read_gray(ds, 0); h, w = im0.shape
    f, cx, cy = md.intrinsics(w, h)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    ext = md.RT_BC.copy()
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.max_features = 300
    fcfg.set_camera(f, f, cx, cy, w, h, [])
    tracker = echo_li.Frontend(fcfg, w, h)
    vio = init_vio(args, cam, ext, ds)
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
        cam, **sparse_settings(args.config))

    imu = ds.db[ds.traj]["imu"]
    acc = np.asarray(imu["accelerometer"]); gyr = np.asarray(imu["gyroscope"])
    events = [(i / 100.0, "imu", i) for i in range(len(acc))]
    events += [(k / 25.0, "cam", k) for k in range(ds.n)]
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    births = {}
    frozen = {}
    observations = [dict() for _ in range(ds.n)]
    est_poses = [None] * ds.n; gt_poses = [None] * ds.n
    for stamp, kind, idx in events:
        if kind == "imu":
            gi = min(idx, len(ds.att) - 1); q = ds.att[gi]
            r_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            vio.process_imu(stamp, (r_gt.T @ gyr[idx]).tolist(), acc[idx].tolist())
            continue
        k = idx
        feats, _ = tracker.process(read_gray(ds, k))
        uvs = {int(x["id"]): np.array([float(x["x"]), float(x["y"])]) for x in feats}
        observations[k] = uvs
        t_wc = vio_body_pose(vio) @ ext
        pc = vio.get_camera_pose_covariance()
        pvv = None if pc is None else np.asarray(pc[0], float).tolist()
        pww = None if pc is None else np.asarray(pc[1], float).tolist()
        sparse.update(stamp, {fid: tuple(uv) for fid, uv in uvs.items()},
                      t_wc.tolist(), pvv, pww)
        existing = {int(x) for x in vio.get_landmarks()}
        ordered = sorted(uvs, key=lambda fid: (0 if fid in existing else 1, fid))[:40]
        vio.process_vision(stamp, {fid: tuple(uvs[fid]) for fid in ordered})
        t_wc = vio_body_pose(vio) @ ext
        est_poses[k] = t_wc
        gt_poses[k] = nwu_body_pose(ds, k) @ ext

        sf = sparse.get_features()
        for fid, fd in sf.items():
            fid = int(fid)
            if fid not in births:
                births[fid] = {"anchor": k, "stamp": stamp, "t_wc_anchor": t_wc.copy()}
            if fid in frozen or int(fd["track_length"]) < args.min_track:
                continue
            q_c = np.asarray(fd["position"], float)
            b = births[fid]
            t_ca = np.linalg.inv(t_wc) @ b["t_wc_anchor"]
            p_a = np.linalg.inv(t_ca[:3, :3]) @ (q_c - t_ca[:3, 3])
            if np.isfinite(p_a).all() and 1.0 < np.linalg.norm(p_a) < 5000.0:
                frozen[fid] = {**b, "point_anchor": p_a, "mature": k}

    cohorts = defaultdict(dict)
    for fid, item in frozen.items():
        cohorts[item["anchor"]][fid] = item
    rows = []
    for anchor, cohort in sorted(cohorts.items()):
        if len(cohort) < args.min_cohort:
            continue
        start = max(x["mature"] for x in cohort.values()) + 1
        stop = start + args.train_length + args.test_length
        if stop >= ds.n:
            continue
        train = range(start, start + args.train_length)
        test = range(start + args.train_length, stop)
        result = fit_cohort(cohort, observations, est_poses, gt_poses, f, cx, cy,
                            train, test, args.max_rate, args.depth_prior_sigma)
        if result is not None:
            rows.append(result)

    keys = ["anchor", "features", "train_obs", "test_obs", "cost", "train_pix0",
            "train_pix1", "test_pix0", "test_pix1", "test_trans0", "test_trans1",
            "test_rot0", "test_rot1"]
    data = np.array([[r[k] for k in keys] for r in rows], float) if rows else np.zeros((0, len(keys)))
    twists = np.array([r["twist"] for r in rows], float) if rows else np.zeros((0, 6))
    np.savez(args.save_npz, data=data, cols=np.array(keys), twists=twists)
    print(f"traj {args.traj}: frozen={len(frozen)}, cohorts={len(cohorts)}, fitted={len(rows)}")
    if rows:
        print(f"held-out pixel median {np.median(data[:, 7]):.3f} -> {np.median(data[:, 8]):.3f}")
        print(f"held-out translation median {np.median(data[:, 9]):.3f} -> {np.median(data[:, 10]):.3f} m")
        print(f"held-out rotation median {np.rad2deg(np.median(data[:, 11])):.3f} -> "
              f"{np.rad2deg(np.median(data[:, 12])):.3f} deg")


if __name__ == "__main__":
    main()
