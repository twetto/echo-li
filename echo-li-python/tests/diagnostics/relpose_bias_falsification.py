"""Falsification test for the constant-bias explanation of the super-diffusive pose error.

The derivation behind the coherent range-noise term claims the relative pose error over a
window is dominated by a PERSISTENT bias: for a residual gyro-bias error db_g, the relative
rotation error over Delta frames is

    dphi_rel ~ -db_g * Delta        (a single constant 3-vector, times the window length)

so its variance grows as Delta^2. Everything else in the derivation (the Delta^2 scaling, the
tau/dt = Delta ratio between truth and the per-frame-increment surrogate, the exclusion of
accel bias) follows from that one claim.

It is directly falsifiable WITHOUT the bias covariance, which no saved run contains:

  1. Least-squares fit a single per-frame bias vector w_b from all (window, lag) pairs:
     w_b = sum(Delta * dphi) / sum(Delta^2).
  2. Subtract it: dphi_res = dphi_rel - w_b * Delta.
  3. Re-fit the growth exponent of the residual.

  derivation TRUE  => exponent collapses toward 1 (only white noise left), and the fitted
                      w_b is consistent across lags and explains most of the error energy.
  derivation FALSE => exponent barely moves; the coherent part is not a constant bias.

A constant body-frame bias appears as a constant vector in the anchor CAMERA frame (a
body-fixed frame), which is where dphi_rel is already expressed, so no extra transport.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

LAGS = [4, 8, 16, 32, 64]


def se3(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def load(npz, kite=None, root=None):
    d = np.load(npz)
    if "baseline_pcov_pos" in d.files:
        ep, eq = d["baseline_est"], d["baseline_quat"]
        gp, gq = d["baseline_gt"], d["baseline_gt_quat"]
        tbc = d["baseline_t_bc"] if "baseline_t_bc" in d.files else md.RT_BC
    else:
        ep, eq, k = d["est"], d["quat"], d["k"]
        ds = md.MidAir(root, "Kite_training", "sunny", kite, 1.0)
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        gp = np.zeros_like(ep)
        gq = np.zeros((len(k), 4))
        for i, kk in enumerate(k):
            M = T @ ds.pose(int(kk))
            gp[i], gq[i] = M[:3, 3], Rot.from_matrix(M[:3, :3]).as_quat()
        tbc = md.RT_BC
    n = len(ep)
    Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ tbc for i in range(n)])
    Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ tbc for i in range(n)])
    return Te, Tg, n


def collect(Te, Tg, n, burn=200, stride=5):
    """(lag, dphi[3]) for every window."""
    out = []
    for L in LAGS:
        a = np.arange(burn, n - L, stride)
        if len(a) < 20:
            continue
        Xe = np.linalg.inv(Te[a]) @ Te[a + L]
        Xg = np.linalg.inv(Tg[a]) @ Tg[a + L]
        dR = np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]
        out.append((L, Rot.from_matrix(dR).as_rotvec()))
    return out


def expo(pairs, wb=None):
    """Growth exponent of median |dphi|^2 vs lag, optionally after removing wb*lag."""
    xs, ys = [], []
    for L, dphi in pairs:
        r = dphi if wb is None else dphi - wb * L
        xs.append(L)
        ys.append(np.median((r * r).sum(1)))
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0]), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir")
    args = ap.parse_args()
    SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
          "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")
    runs = [("MidAir", "VO t0", f"{SP}/mono_run_fig_traj0.npz", None),
            ("MidAir", "VO t1", f"{SP}/mono_run_fig_traj1.npz", None),
            ("MidAir", "VO t2", f"{SP}/mono_run_fig_traj2.npz", None),
            ("MidAir", "Kite t0", f"{SP}/kite_traj0.npz", 0),
            ("MidAir", "Kite t1", f"{SP}/kite_traj1.npz", 1)]
    runs += [("EuRoC", s, f"{SP}/euroc_{s}.npz", None) for s in
             ["V1_01", "V1_02_medium", "V1_03_difficult",
              "V2_01_easy", "V2_02_medium", "V2_03_difficult"]]
    runs += [("TUM-VI", f"room{i}", f"{SP}/tumvi_room{i}.npz", None) for i in range(1, 7)]

    print("PREDICTION: removing one constant bias vector collapses the exponent toward 1.\n")
    print(f"{'ds':<8}{'run':<16}{'exp raw':>9}{'exp resid':>11}{'drop':>8}"
          f"{'|w_b| deg/frame':>17}{'energy expl.':>14}")
    print("-" * 84)
    rows = []
    for ds, name, f, kite in runs:
        if not Path(f).exists():
            print(f"{ds:<8}{name:<16}   (missing)")
            continue
        Te, Tg, n = load(f, kite, args.root)
        pairs = collect(Te, Tg, n)
        if len(pairs) < 3:
            continue
        # least-squares single bias vector: w_b = sum(L*dphi) / sum(L^2)
        num = np.zeros(3)
        den = 0.0
        for L, dphi in pairs:
            num += L * dphi.sum(0)
            den += (L ** 2) * len(dphi)
        wb = num / den
        e_raw, y_raw = expo(pairs)
        e_res, y_res = expo(pairs, wb)
        expl = 1.0 - y_res.sum() / y_raw.sum()
        rows.append((ds, name, e_raw, e_res, expl))
        print(f"{ds:<8}{name:<16}{e_raw:>9.2f}{e_res:>11.2f}{e_raw - e_res:>8.2f}"
              f"{np.degrees(np.linalg.norm(wb)):>17.4f}{100 * expl:>13.1f}%")

    print("\nverdict per run: drop >~0.6 and exponent_resid ~1 => constant bias CONFIRMED")
    print("                 exponent_resid still >>1              => derivation FALSIFIED")
    r = np.array([[x[2], x[3]] for x in rows])
    print(f"\nmean exponent {r[:,0].mean():.2f} raw -> {r[:,1].mean():.2f} after removing one "
          f"constant vector  (n={len(r)} runs)")


if __name__ == "__main__":
    main()
