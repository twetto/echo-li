#!/usr/bin/env python3
"""Score an ECHO-LI trajectory (run_offline.py output) against mocap.

Steps
  1. Mocap cleanup: sort by header stamp, drop equal stamps and poses
     bit-identical to their predecessor. VRPN re-sends every pose, and a
     rigid-body dropout keeps re-sending the last one; dropping repeats turns
     a freeze into a gap, and GT is only used where bracketing samples are
     < max_gap apart.
  2. Clock offset (t_mocap = t_imu + offset): a coarse value from the bag
     receive stamps, refined to ~1 ms by cross-correlating |gyro| with the
     mocap rotation rate. |w| is frame-independent, so this needs no
     extrinsics.
  3. GT at estimate times: interpolated position, and world velocity from
     a central difference over +-vel_half_window.
  4. SE(3) Umeyama alignment of estimated positions onto GT. The same
     rotation maps the estimated world velocity (R_wb v_body).

Metrics
  ate_rmse   position error after alignment [m]
  vel_rmse   world-velocity error [m/s] (includes the yaw drift)
  vel_body_rmse  body-frame velocity error [m/s], GT rotated with the GT
             attitude: velocity quality without the pose drift
  vel_hf_rmse  velocity error minus its 1 s moving average [m/s]: the
               "wobble" that the slow drift terms do not explain
  hf_ratio   RMS high-pass velocity of the estimate / of the GT: > 1 means
             the estimate shakes more than the rig did
  rot_rmse   attitude error [deg], after fitting the constant mocap-body <->
             IMU rotation
  rpe1/rpe5  translation error over 1 s / 5 s segments [m]
  score      ate_rmse + vel_rmse (1 s time constant)

numpy only, so it runs in the ROS runtime venv and on the host.
"""
import argparse
import json
import os

import numpy as np

NS = 1_000_000_000


def quat_to_rot(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_angle(R):
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ E @ Vt


def umeyama(src, dst, with_scale=False):
    """R, t, s minimising |s R src + t - dst|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / xs.var(0).sum()) if with_scale else 1.0
    return R, mu_d - s * R @ mu_s, s


VERTICAL_AXIS = {"vrpn": 1, "vp": 2}   # VRPN/Motive is Y-up; vision_pose is ENU


def load_mocap(sensors, key, glitch_m=0.005, glitch_half_window=5):
    t = sensors[f"{key}_hdr"]
    pos = sensors[f"{key}_pos"]
    quat = sensors[f"{key}_quat"]
    order = np.argsort(t, kind="stable")
    t, pos, quat = t[order], pos[order], quat[order]
    keep = np.ones(len(t), bool)
    keep[1:] = np.diff(t) > 0
    t, pos, quat = t[keep], pos[keep], quat[keep]
    pose = np.hstack([pos, quat])
    keep = np.ones(len(t), bool)
    keep[1:] = np.any(pose[1:] != pose[:-1], axis=1)
    t, pos, quat = t[keep], pos[keep], quat[keep].copy()
    # Drop isolated position glitches: a sample more than glitch_m from the
    # median of its +-w neighbours. Smooth motion keeps the centre sample
    # on the per-axis median (<1 mm at these speeds); a marker glitch does not.
    w = glitch_half_window
    if len(t) > 2 * w + 1:
        med = np.median(np.lib.stride_tricks.sliding_window_view(pos, 2 * w + 1, axis=0), axis=-1)
        keep = np.ones(len(t), bool)
        keep[w:-w] = np.linalg.norm(pos[w:-w] - med, axis=1) < glitch_m
        t, pos, quat = t[keep], pos[keep], quat[keep]
    flip = np.cumsum(np.r_[False, np.sum(quat[1:] * quat[:-1], axis=1) < 0]) % 2 == 1
    quat[flip] *= -1.0
    return t, pos, quat


def interp_pose(t, pos, quat, tq, max_gap_ns):
    """Linear position / normalised-lerp attitude at tq; valid only inside
    intervals shorter than max_gap_ns."""
    i1 = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
    i0 = i1 - 1
    gap = t[i1] - t[i0]
    valid = (tq >= t[0]) & (tq <= t[-1]) & (gap <= max_gap_ns)
    a = ((tq - t[i0]) / np.maximum(gap, 1))[:, None]
    p = pos[i0] + a * (pos[i1] - pos[i0])
    q = quat[i0] + a * (quat[i1] - quat[i0])
    return p, q / np.linalg.norm(q, axis=1, keepdims=True), valid


def moving_average(x, n):
    """Centred moving average along axis 0, shrinking at the edges."""
    n = max(1, int(n) | 1)
    k = np.ones(n)
    cnt = np.convolve(np.ones(len(x)), k, mode="same")
    if x.ndim == 1:
        return np.convolve(x, k, mode="same") / cnt
    return np.column_stack([np.convolve(x[:, j], k, mode="same") / cnt
                            for j in range(x.shape[1])])


def estimate_clock_offset(sensors, t_moc, quat_moc, key, search_s=1.0, rate_hz=200.0,
                          max_gap_ns=30_000_000):
    """t_mocap = t_imu + offset [ns], from bag stamps refined by |w| correlation."""
    coarse = int(np.median(sensors[f"{key}_hdr"] - sensors[f"{key}_bag"])
                 - np.median(sensors["imu_hdr"] - sensors["imu_bag"]))
    imu_t = sensors["imu_hdr"]
    base = int(imu_t[0])
    t_imu = (imu_t - base) / NS
    w_imu = moving_average(np.linalg.norm(sensors["imu_gyr"], axis=1), 5)

    # Mocap rotation rate on a uniform grid in (coarse-mapped) IMU time.
    tm = (t_moc - base - coarse) / NS
    step = 1.0 / rate_hz
    grid_m = np.arange(tm[0], tm[-1], step)
    _, q, ok = interp_pose((tm * NS).astype(np.int64), np.zeros((len(tm), 3)), quat_moc,
                           (grid_m * NS).astype(np.int64), max_gap_ns)
    R = quat_to_rot(q)
    w_m = rot_angle(np.einsum("nji,njk->nik", R[:-1], R[1:])) / step
    ok_m = ok[:-1] & ok[1:]
    mid = grid_m[:-1] + step / 2

    grid = np.arange(max(t_imu[0], mid[0]) + search_s, min(t_imu[-1], mid[-1]) - search_s, step)
    wi = np.interp(grid, t_imu, w_imu)
    lags = np.arange(-search_s, search_s + 1e-9, 0.001)
    corr = np.full(len(lags), -np.inf)
    for n, d in enumerate(lags):
        wm = np.interp(grid + d, mid, w_m)
        good = np.interp(grid + d, mid, ok_m.astype(float)) > 0.999
        if good.sum() > 100:
            corr[n] = np.corrcoef(wi[good], wm[good])[0, 1]
    b = int(np.argmax(corr))
    d = lags[b]
    if 0 < b < len(lags) - 1:
        y0, y1, y2 = corr[b - 1], corr[b], corr[b + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            d += 0.001 * 0.5 * (y0 - y2) / den
    return dict(offset_ns=coarse + int(round(d * NS)), coarse_ns=coarse,
                refine_s=float(d), corr=float(corr[b]))


def cached_clock_offset(cache, sensors, t_moc, quat_moc, key):
    path = os.path.join(cache, f"sync_{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    sync = estimate_clock_offset(sensors, t_moc, quat_moc, key)
    with open(path, "w") as f:
        json.dump(sync, f, indent=1)
    return sync


def evaluate(traj, mocap, offset_ns, max_gap_ms=30.0, vel_half_window_ms=25.0,
             hf_window_s=1.0, skip_s=0.0, vertical_axis=None):
    t_moc, pos, quat = mocap
    gap = int(max_gap_ms * 1e6)
    h = int(vel_half_window_ms * 1e6)
    t_est = traj["t_ns"]
    if len(t_est) < 10:
        return dict(ok=False, reason="too few estimates")
    tq = t_est + offset_ns
    p_gt, q_gt, ok = interp_pose(t_moc, pos, quat, tq, gap)
    p_plus, _, ok_p = interp_pose(t_moc, pos, quat, tq + h, gap)
    p_minus, _, ok_m = interp_pose(t_moc, pos, quat, tq - h, gap)
    ok &= ok_p & ok_m & (t_est >= t_est[0] + int(skip_s * NS))
    if ok.sum() < 10:
        return dict(ok=False, reason="no overlap with mocap")
    v_gt = (p_plus - p_minus) / (2 * h / NS)

    t = (t_est[ok] - t_est[0]) / NS
    G, VG = p_gt[ok], v_gt[ok]
    P = traj["p"][ok]
    R_est = quat_to_rot(traj["q"][ok])
    R, tr, _ = umeyama(P, G)
    _, _, scale = umeyama(P, G, with_scale=True)
    P_al = P @ R.T + tr
    V_al = np.einsum("ij,njk,nk->ni", R, R_est, traj["v"][ok])
    ate = np.linalg.norm(P_al - G, axis=1)
    ev = V_al - VG

    rate = (len(t) - 1) / max(t[-1] - t[0], 1e-9)
    n_hf = int(round(hf_window_s * rate))
    ev_hf = ev - moving_average(ev, n_hf)
    est_hf = V_al - moving_average(V_al, n_hf)
    gt_hf = VG - moving_average(VG, n_hf)

    R_gt = quat_to_rot(q_gt[ok])
    R_bm = project_so3(np.einsum("nji,kj,nkl->il", R_est, R, R_gt) / len(R_gt))
    rot_err = np.degrees(rot_angle(np.einsum("ij,njk,kl,nml->nim", R, R_est, R_bm, R_gt)))
    # Body-frame velocity error: GT world velocity rotated into the IMU frame
    # with the GT attitude, v_b = R_bm R_gt^T v_w. It carries no yaw drift, so
    # it isolates velocity quality from the pose error.
    VBG = np.einsum("ij,nkj,nk->ni", R_bm, R_gt, VG)
    eb = traj["v"][ok] - VBG

    def rpe(delta):
        j = np.searchsorted(t, t + delta)
        use = j < len(t)
        i, j = np.flatnonzero(use), j[use]
        good = np.abs(t[j] - t[i] - delta) < 0.1
        i, j = i[good], j[good]
        if len(i) == 0:
            return float("nan")
        e = (P_al[j] - P_al[i]) - (G[j] - G[i])
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))

    def rms(x):
        return float(np.sqrt(np.mean(np.sum(np.atleast_2d(x.T).T ** 2, axis=-1))))

    m = dict(ok=True,
             ate_rmse=rms(P_al - G), ate_max=float(ate.max()),
             vel_rmse=rms(ev), vel_hf_rmse=rms(ev_hf),
             vel_body_rmse=rms(eb), vel_body_hf_rmse=rms(eb - moving_average(eb, n_hf)),
             hf_ratio=rms(est_hf) / max(rms(gt_hf), 1e-9),
             rot_rmse=float(np.sqrt(np.mean(rot_err ** 2))),
             rpe1=rpe(1.0), rpe5=rpe(5.0), sim3_scale=scale,
             gt_speed_rms=rms(VG), coverage=float(ok.mean()),
             duration_s=float(t[-1] - t[0]), n=int(ok.sum()))
    if vertical_axis is not None:
        horiz = [k for k in range(3) if k != vertical_axis]
        m["ate_vert_rmse"] = rms((P_al - G)[:, vertical_axis])
        m["ate_horiz_rmse"] = rms((P_al - G)[:, horiz])
        m["vel_vert_rmse"] = rms(ev[:, vertical_axis])
        m["vel_horiz_rmse"] = rms(ev[:, horiz])
    m["score"] = m["ate_rmse"] + m["vel_rmse"]
    details = dict(t=t, t_ns=t_est[ok], P_al=P_al, G=G, V_al=V_al, VG=VG,
                   VB=traj["v"][ok], VBG=VBG, ate=ate, rot_err=rot_err)
    return m, details


def plot(details, metrics, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = details
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(4, 3)
    ax = fig.add_subplot(grid[:2, 0])
    a, b = sorted(np.argsort(d["G"].std(0))[-2:])   # the two widest GT axes
    ax.plot(d["G"][:, a], d["G"][:, b], "k-", lw=1, label="mocap")
    ax.plot(d["P_al"][:, a], d["P_al"][:, b], "-", color="tab:orange", lw=1, label="ECHO-LI")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.set_title(f"plan view ({'xyz'[a]}, {'xyz'[b]}), aligned [m]")
    ax = fig.add_subplot(grid[2:, 0])
    ax.plot(d["t"], d["ate"], lw=0.8, label="position [m]")
    ax.plot(d["t"], d["rot_err"] / 10.0, lw=0.8, label="attitude [10 deg]")
    ax.legend(loc="upper left")
    ax.set_xlabel("time since first estimate [s]")
    ax.set_title("errors")
    axes = [fig.add_subplot(grid[k, 1:]) for k in range(4)]
    for k, name in enumerate("xyz"):
        axes[k].plot(d["t"], d["VG"][:, k], "k-", lw=0.8, label="mocap")
        axes[k].plot(d["t"], d["V_al"][:, k], "-", color="tab:orange", lw=0.8, label="ECHO-LI")
        axes[k].set_ylabel(f"v{name} [m/s]")
    axes[0].legend(loc="upper right")
    axes[3].plot(d["t"], np.linalg.norm(d["V_al"] - d["VG"], axis=1), lw=0.8)
    axes[3].set_ylabel("|v error| [m/s]")
    axes[3].set_xlabel("time since first estimate [s]")
    fig.suptitle(title + "  " + "  ".join(
        f"{k}={metrics[k]:.3f}" for k in ("ate_rmse", "vel_rmse", "vel_hf_rmse", "hf_ratio",
                                           "rot_rmse")))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("traj")
    ap.add_argument("--mocap", default="vrpn", choices=["vrpn", "vp"])
    ap.add_argument("--skip", type=float, default=0.0, help="ignore the first SKIP s")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    sensors = np.load(os.path.join(args.cache, "sensors.npz"))
    traj = np.load(args.traj)
    mocap = load_mocap(sensors, args.mocap)
    sync = cached_clock_offset(args.cache, sensors, mocap[0], mocap[2], args.mocap)
    result = evaluate(traj, mocap, sync["offset_ns"], skip_s=args.skip,
                      vertical_axis=VERTICAL_AXIS[args.mocap])
    if isinstance(result, dict):
        print(json.dumps(result))
        return
    metrics, details = result
    metrics["clock"] = sync
    print(json.dumps(metrics, indent=1))
    if args.plot:
        plot(details, metrics, args.plot, title=os.path.basename(args.traj))


if __name__ == "__main__":
    main()
