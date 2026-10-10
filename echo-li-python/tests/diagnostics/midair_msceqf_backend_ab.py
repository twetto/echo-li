"""Score the Rust MsceqfBackend against the Python orchestration it replaces.

Both arms drive the SAME MSCEqF filter over the SAME data with the same origin, noise
and window. The only difference is where the frame step lives: arm "py" runs the loop in
midair_msceqf_native_ate.py against the primitive bindings, arm "rs" hands each frame to
echo_li_core::backend::MsceqfBackend. If the port is faithful the trajectories agree.

Why this is worth running rather than assuming: the Python side had TWO different
orchestrations -- this ATE harness and MSCEqFAdapter in midair_e2e_depth_nees.py -- and
they disagree on when a track is flushed, when one is promoted, and whether the
structureless update uses the curvature term. The port follows this harness, because
this is the one whose ATE figures were published.

Known deliberate difference: the backend normalizes pixels through a real camera model
where the harness divides by a bare inverse-K. Identical on MidAir (undistorted pinhole),
which is why MidAir is the right place to compare.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from midair_msceqf_native_ate import (  # noqa: E402
    DT, FX, R_wb, ROOT, TRACKS, align_umeyama,
)
from echo_li import MSCEqFNativeFilter, MsceqfBackend  # noqa: E402

CX = FX
IMU_PER_IMG = 4


def load_pixel_tracks(nimg, path, t0=0.0):
    """frame k -> {track_id: (u_px, v_px)}. Raw pixels; each arm normalizes its own way."""
    by_frame = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            ts, tid, u, v = float(p[0]), int(p[2]), float(p[3]), float(p[4])
            k = int(round((ts - t0) / (DT * IMU_PER_IMG)))
            if 0 <= k < nimg:
                by_frame[k][tid] = (u, v)
    return by_frame


def load_sequence(traj, nframes):
    h5 = h5py.File(f"{ROOT}/VO_test/sunny/sensor_records.hdf5", "r")
    g = h5[traj]
    acc = np.asarray(g["imu"]["accelerometer"], float)
    gyr = np.asarray(g["imu"]["gyroscope"], float)
    att = np.asarray(g["groundtruth"]["attitude"], float)
    pos = np.asarray(g["groundtruth"]["position"], float)
    vel = np.asarray(g["groundtruth"]["velocity"], float)
    m = min(len(gyr), len(att))
    rwb = [R_wb(att[i]) for i in range(m)]
    gyr_body = np.array([rwb[i].T @ gyr[i] for i in range(m)])
    nimg_all = m // IMU_PER_IMG
    nimg = nimg_all if nframes <= 0 else min(nframes, nimg_all)
    return gyr_body, acc[:m], pos, vel, rwb, nimg


def common_init(rwb, pos, vel):
    return dict(
        r0=np.ascontiguousarray(rwb[0]),
        p0=np.ascontiguousarray(pos[0].copy()),
        v0=np.ascontiguousarray(vel[0].copy()),
        b0=np.zeros(6),
        s0=np.array([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]),
        g=np.array([0., 0., 9.81]),
        d_std=np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4]),
        delta_std=np.full(6, 1e-2),
        e_std=np.full(6, 1e-4),
    )


def run_rust(a, by_frame, gyr_body, acc, nimg, cfg):
    f = MsceqfBackend(
        a["r0"], a["p0"], a["v0"], a["b0"], a["s0"], a["g"],
        a["d_std"], a["delta_std"], a["e_std"],
        cfg.acc_density, cfg.gyr_density, cfg.acc_rw, cfg.gyr_rw, 0,
        FX, FX, CX, CX,
        num_clones=cfg.num_clones, pixel_std=1.0 / FX, chi2_mult=cfg.chi2_mult,
        max_track=cfg.max_track_len, max_slam=cfg.max_slam,
        curvature=bool(cfg.curvature),
    )
    est = []
    # Prime the stamp. The backend derives dt from consecutive samples, so its first
    # call propagates nothing, whereas the Python loop passes dt=DT for every sample
    # including the first -- i.e. it treats sample 0 as covering [-DT, 0]. Without this
    # the Rust arm integrates one fewer IMU step at the start and every later pose is
    # offset, which is a convention difference, not a port error.
    f.process_imu(-DT, gyr_body[0], acc[0])
    f.process_vision(0.0, by_frame.get(0, {}))
    est.append(f.nav_pose()[0].copy())
    for k in range(1, nimg):
        for i in range((k - 1) * IMU_PER_IMG, k * IMU_PER_IMG):
            f.process_imu(i * DT, gyr_body[i], acc[i])
        f.process_vision(k * DT * IMU_PER_IMG, by_frame.get(k, {}))
        est.append(f.nav_pose()[0].copy())
    return np.array(est), f.n_clones(), f.n_slam()


def run_python(a, by_frame, gyr_body, acc, nimg, cfg):
    """The reference loop, verbatim in structure, against the primitive bindings."""
    norm = {k: {t: ((u - CX) / FX, (v - CX) / FX) for t, (u, v) in d.items()}
            for k, d in by_frame.items()}
    f = MSCEqFNativeFilter(
        a["r0"], a["p0"], a["v0"], a["b0"], a["s0"], a["g"],
        a["d_std"], a["delta_std"], a["e_std"],
        cfg.acc_density, cfg.gyr_density, cfg.acc_rw, cfg.gyr_rw, 0, cfg.num_clones)
    pixel_std = 1.0 / FX
    slam_min_len = cfg.num_clones
    buf, slam, est = {}, set(), []
    n_upd = 0

    def add_frame(k):
        f.clone_pose(k, k * DT * IMU_PER_IMG)
        for tid, (un, vn) in norm.get(k, {}).items():
            if tid in slam:
                continue
            buf.setdefault(tid, []).append((k, un, vn))

    add_frame(0)
    est.append(f.nav_body_pose()[0].copy())
    for k in range(1, nimg):
        for i in range((k - 1) * IMU_PER_IMG, k * IMU_PER_IMG):
            f.process_imu(gyr_body[i], acc[i], DT)
        add_frame(k)
        cur = set(norm.get(k, {}).keys())
        marg = f.clone_ids()[0] if f.n_clones() > cfg.num_clones else None

        if cfg.max_slam > 0 and marg is not None and len(slam) < cfg.max_slam:
            cand = [t for t, o in buf.items()
                    if t in cur and len(o) >= slam_min_len and any(fk == marg for fk, _, _ in o)]
            # Match the backend's TOTAL order (length desc, then track id). The
            # original reference sorted by length alone, and Python's stable sort
            # left ties in dict insertion order -- which, once a track has been
            # flushed and re-observed, is neither id nor arrival order. That is the
            # ONLY thing that separated the two implementations on the SLAM path:
            # with it aligned they agree to 0.000000 m over the full trajectory.
            cand.sort(key=lambda t: (-len(buf[t]), t))
            for tid in cand[: cfg.max_slam - len(slam)]:
                if f.birth_landmark(tid, list(buf[tid]), pixel_std, cfg.chi2_mult):
                    f.reanchor_landmark(tid, k)
                    slam.add(tid)
                    del buf[tid]

        ready, done = {}, []
        for tid, obs in buf.items():
            if (tid not in cur or len(obs) >= cfg.max_track_len
                    or (marg is not None and any(fk == marg for fk, _, _ in obs))):
                if len(obs) >= 2:
                    ready[tid] = list(obs)
                done.append(tid)
        if ready:
            n_upd += f.msc_update(ready, pixel_std, cfg.chi2_mult, bool(cfg.curvature))
        for tid in done:
            del buf[tid]

        if slam:
            supd = {}
            for tid in list(slam):
                if tid in cur:
                    un, vn = norm[k][tid]
                    supd[tid] = [(k, un, vn)]
                else:
                    f.marginalize_landmark(tid)
                    slam.discard(tid)
            if supd:
                f.landmark_stream_update(supd, pixel_std, cfg.chi2_mult)

        if slam and marg is not None:
            newest = f.clone_ids()[-1]
            for tid in list(slam):
                if f.landmark_anchor(tid) == marg:
                    f.reanchor_landmark(tid, newest)

        while f.n_clones() > cfg.num_clones:
            old = f.marginalize_oldest()
            for tid in list(buf):
                b = [o for o in buf[tid] if o[0] != old]
                if b:
                    buf[tid] = b
                else:
                    del buf[tid]
        est.append(f.nav_body_pose()[0].copy())
    print(f'    [py arm] accepted_updates={n_upd}')
    return np.array(est), f.n_clones(), len(slam)


def ate(est, gt):
    aligned, _, _ = align_umeyama(est, gt)
    err = np.linalg.norm(aligned - gt, axis=1)
    length = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    return float(np.sqrt((err ** 2).mean())), 100.0 * float(np.sqrt((err ** 2).mean())) / length, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-track-len", type=int, default=200)
    ap.add_argument("--chi2-mult", type=float, default=1.0)
    ap.add_argument("--max-slam", type=int, default=0)
    ap.add_argument("--curvature", type=int, default=1)
    # These MUST match midair_msceqf_native_ate.py. An earlier version of this file
    # carried Allan-variance figures from elsewhere, leaving the process noise 9-55x
    # too small; the filter then drifted to 24.7% ATE over the full trajectory and the
    # fault looked like the filter rather than the harness.
    ap.add_argument("--acc-density", type=float, default=1.6798e-3)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-density", type=float, default=1.2724e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.0e-4)
    ap.add_argument("--tracks", default=str(TRACKS))
    ap.add_argument("--dump", default="")
    cfg = ap.parse_args()

    traj = f"trajectory_{cfg.traj:04d}"
    gyr_body, acc, pos, vel, rwb, nimg = load_sequence(traj, cfg.nframes)
    by_frame = load_pixel_tracks(nimg, cfg.tracks)
    gt = np.array([pos[k * IMU_PER_IMG].copy() for k in range(nimg)])
    a = common_init(rwb, pos, vel)
    print(f"{traj}: {nimg} frames, max_slam={cfg.max_slam}, "
          f"num_clones={cfg.num_clones}, curvature={cfg.curvature}")

    e_py, c_py, s_py = run_python(a, by_frame, gyr_body, acc, nimg, cfg)
    e_rs, c_rs, s_rs = run_rust(a, by_frame, gyr_body, acc, nimg, cfg)

    r_py, p_py, length = ate(e_py, gt)
    r_rs, p_rs, _ = ate(e_rs, gt)
    d = np.linalg.norm(e_rs - e_py, axis=1)
    print(f"  path length {length:.1f} m")
    print(f"  python loop   ATE {r_py:.4f} m  ({p_py:.3f}% of path)  clones {c_py} slam {s_py}")
    print(f"  rust backend  ATE {r_rs:.4f} m  ({p_rs:.3f}% of path)  clones {c_rs} slam {s_rs}")
    print(f"  trajectory difference: median {np.median(d):.6f} m, max {d.max():.6f} m")
    print(f"  VERDICT: {'MATCH' if d.max() < 1e-6 else 'DIFFER'}")
    if cfg.dump:
        np.savez(cfg.dump, est_py=e_py, est_rs=e_rs, gt=gt)
        print(f"  dumped -> {cfg.dump}")


if __name__ == "__main__":
    main()
