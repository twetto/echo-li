#!/usr/bin/env python3
"""Drive the MSCEqF-native symmetry-group filter on MidAir VO_test traj2 and
report ATE against the C++ MSCEqF reference (0.6%) / OpenVINS (0.2%).

Front-end is ISOLATED: we feed MSCEqF's OWN KLT tracks
(`artifacts/msceqf_midair/stagediff/msceqf_own_tracks.txt`, undistorted pixels)
so any ATE gap is the filter, not the tracker. The filter is the parallel
`MSCEqFNativeFilter` (covariance on the SE_2(3)⋉bias / SE3 group algebra) — the
EqVIO EqF path is untouched.

Init = GT-seeded given-origin at frame 0 (MidAir starts mid-flight; MSCEqF has
only static init), mirroring the golden config
`artifacts/msceqf_midair/run/t2full/config/config.yaml`.

Usage (opt-in; uncommitted):
  source ../.venv/bin/activate   # from echo-li-python/
  python tests/diagnostics/midair_msceqf_native_ate.py
"""
import argparse
from collections import deque, defaultdict
from pathlib import Path

import h5py
import numpy as np
from echo_li import MSCEqFNativeFilter

ROOT = "/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir"
TRACKS = Path(__file__).resolve().parents[3] / \
    "artifacts/msceqf_midair/stagediff/msceqf_own_tracks.txt"

DT = 0.01          # 100 Hz IMU
IMU_PER_IMG = 4    # 25 Hz images
FX = CX = 512.0    # MidAir 90deg @1024, no distortion


def R_wb(q):
    """Rotation body->world from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_tracks(nimg):
    """frame k -> {track_id: (un, vn)} normalized Z1 coords."""
    by_frame = defaultdict(dict)
    with open(TRACKS) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            ts = float(p[0]); tid = int(p[2]); u = float(p[3]); v = float(p[4])
            k = int(round(ts / (DT * IMU_PER_IMG)))
            if k >= nimg:
                continue
            by_frame[k][tid] = ((u - CX) / FX, (v - CX) / FX)
    return by_frame


def align_umeyama(est, gt):
    """SE3 (no-scale) alignment of est to gt; returns (aligned, R, t)."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    H = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_g - R @ mu_e
    return (R @ est.T).T + t, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-track-len", type=int, default=200)
    ap.add_argument("--curvature", type=int, default=1)
    ap.add_argument("--chi2-mult", type=float, default=1.0)
    ap.add_argument("--acc-density", type=float, default=1.6798e-3)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-density", type=float, default=1.2724e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.0e-4)
    ap.add_argument("--dump", default="", help="save est/gt/R/t/k npz for the video")
    args = ap.parse_args()

    base = f"{ROOT}/{args.set}/{args.cond}"
    tr = f"trajectory_{args.traj:04d}"
    g = h5py.File(f"{base}/sensor_records.hdf5", "r")[tr]
    acc = np.asarray(g["imu"]["accelerometer"], float)
    gyr = np.asarray(g["imu"]["gyroscope"], float)
    att = np.asarray(g["groundtruth"]["attitude"], float)   # wxyz body->world
    pos = np.asarray(g["groundtruth"]["position"], float)
    vel = np.asarray(g["groundtruth"]["velocity"], float)
    m = min(len(gyr), len(att))
    Rwb = [R_wb(att[i]) for i in range(m)]
    gyr_body = np.array([Rwb[i].T @ gyr[i] for i in range(m)])  # world->body
    acc = acc[:m]
    nimg_all = m // IMU_PER_IMG
    nimg = nimg_all if args.nframes <= 0 else min(args.nframes, nimg_all)

    by_frame = load_tracks(nimg)

    # --- GT-seeded given-origin init (frame 0) ---
    R0 = Rwb[0]
    p0 = pos[0].copy()
    v0 = vel[0].copy()
    b0 = np.zeros(6)
    S0 = np.array([[0., 0., 1., 0.],
                   [1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.]])          # T_imu_cam (cam->body)
    grav = np.array([0., 0., 9.81])            # MidAir NED: +z
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])  # att,vel,pos
    delta_std = np.full(6, 1e-2)
    e_std = np.full(6, 1e-4)

    filt = MSCEqFNativeFilter(
        R0, p0, v0, b0, S0, grav, d_std, delta_std, e_std,
        args.acc_density, args.gyr_density, args.acc_rw, args.gyr_rw,
        0,                       # transition_order != 1 -> full matrix exp
        args.num_clones)

    pixel_std = 1.0 / FX
    buf = {}                     # tid -> [(frame_k, un, vn)]
    est_p, gt_p = [], []
    n_upd = 0

    def add_frame(k):
        filt.clone_pose(k, k * DT * IMU_PER_IMG)
        for tid, (un, vn) in by_frame.get(k, {}).items():
            buf.setdefault(tid, []).append((k, un, vn))

    add_frame(0)
    est_p.append(filt.nav_body_pose()[0].copy()); gt_p.append(pos[0].copy())

    for k in range(1, nimg):
        for i in range((k - 1) * IMU_PER_IMG, k * IMU_PER_IMG):
            filt.process_imu(gyr_body[i], acc[i], DT)
        add_frame(k)

        # Structureless MSCKF update. A track's full measurement set is consumed
        # exactly ONCE (then the track is removed), triggered when it is (a) lost
        # this frame, (b) at max length, or (c) about to lose its oldest
        # observation to window marginalization (OpenVINS feats_lost/maxtracks/marg).
        # Silently dropping a track's measurement at the marginalized clone would
        # discard most geometric constraints and let scale drift.
        cur = set(by_frame.get(k, {}).keys())
        marg = filt.clone_ids()[0] if filt.n_clones() > args.num_clones else None
        ready, done = {}, []
        for tid, obs in buf.items():
            lost = tid not in cur
            toolong = len(obs) >= args.max_track_len
            touches_marg = marg is not None and any(fk == marg for (fk, _, _) in obs)
            if lost or toolong or touches_marg:
                if len(obs) >= 2:
                    ready[tid] = [(fk, un, vn) for (fk, un, vn) in obs]
                done.append(tid)
        if ready:
            n_upd += filt.msc_update(
                ready, pixel_std, args.chi2_mult, bool(args.curvature))
        for tid in done:
            del buf[tid]

        # Slide the window: every track touching the oldest clone was just
        # consumed, so marginalization now drops no live measurements.
        while filt.n_clones() > args.num_clones:
            old = filt.marginalize_oldest()
            for tid in list(buf):
                b = [(fk, un, vn) for (fk, un, vn) in buf[tid] if fk != old]
                if b:
                    buf[tid] = b
                else:
                    del buf[tid]

        est_p.append(filt.nav_body_pose()[0].copy())
        gt_p.append(pos[k * IMU_PER_IMG].copy())

    est_p = np.array(est_p); gt_p = np.array(gt_p)
    aligned, Ralign, talign = align_umeyama(est_p, gt_p)
    err = np.linalg.norm(aligned - gt_p, axis=1)
    seg = np.linalg.norm(np.diff(gt_p, axis=0), axis=1)
    path = seg.sum()
    ate_rmse = np.sqrt((err ** 2).mean())
    est_len = np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum()

    print(f"frames={nimg}  accepted_updates={n_upd}")
    print(f"GT path length   = {path:8.2f} m")
    print(f"est path length  = {est_len:8.2f} m   (est/gt = {est_len/path:.4f})")
    print(f"ATE RMSE (SE3)   = {ate_rmse:8.3f} m")
    print(f"ATE %            = {100*ate_rmse/path:7.3f} %   "
          f"[MSCEqF 0.6% / OV 0.2%]")
    print(f"ATE median/mean/max = "
          f"{np.median(err):.3f} / {err.mean():.3f} / {err.max():.3f} m")
    print(f"final drift      = {np.linalg.norm(est_p[-1]-gt_p[-1]):8.3f} m")

    if args.dump:
        # Match midair_vio_video.py's schema: est=raw, gt, SE3 alignment R/t, frame k.
        np.savez(args.dump, est=est_p, gt=gt_p, R=Ralign, t=talign,
                 k=np.arange(nimg, dtype=int))
        print(f"dumped trajectory -> {args.dump}")


if __name__ == "__main__":
    main()
