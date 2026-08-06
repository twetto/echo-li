"""Can the per-track error twist be predicted from anything the filter already reports,
and is it predictable from its own past (i.e. estimable online)?

Established by relpose_pertrack_full.py: the relative pose error within a track is a constant
SE(3) twist xi_a = (w_a, v_a) times the window length, xi_a wanders slowly across anchors
(lag-1 autocorrelation ~0.7), and the filter's accumulated P_ww predicts |w_a| only weakly
(corr +0.23). A coherent q_k term needs E[|xi_a|^2], so the question is whether any ONLINE
quantity supplies it.

TEST 1 -- correlate |w_a| and |v_a| against every filter-reported quantity available, all of
which are computable online (no ground truth):
    dPvv, dPww    accumulated pose-covariance increment over the window
    |dbg|, |dba|  DRIFT RATE of the bias estimates over the window -- if the filter is still
                  chasing the true bias, the residual error should be larger
    |bg|, |ba|    bias estimate magnitudes
    speed, |dv|   body speed and its change (excitation proxies)
    ntrack        tracked feature count

TEST 2 -- predictability horizon: corr(xi_a, xi_{a+D}) versus anchor separation D. High
correlation at useful D means an online estimator (a twist state, like the correspondence-bias
state) could track it; rapid decay means it cannot.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIT = list(range(4, 65, 8))
L0 = 32
SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
      "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")


def se3(R, p):
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, p; return T


def spear(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    runs = ["mono_run_fig_traj0", "mono_run_fig_traj1", "mono_run_fig_traj2"]
    names = ["VO t0", "VO t1", "VO t2"]
    print("TEST 1 -- Spearman corr of ONLINE quantities vs the true per-track twist magnitude")
    print("         (target: something with |rho| >> 0.23, the P_ww baseline)\n")
    keys = ["dPvv", "dPww", "|dbg|", "|dba|", "|bg|", "|ba|", "speed", "|dv|", "ntrack"]
    print(f"{'run':<7}{'chan':<6}" + "".join(f"{k:>8}" for k in keys))
    print("-" * (13 + 8 * len(keys)))
    store = {}
    for f, nm in zip(runs, names):
        d = np.load(f"{SP}/{f}.npz"); p = "baseline_"
        ep, eq = d[p + "est"], d[p + "quat"]
        gp, gq = d[p + "gt"], d[p + "gt_quat"]
        pvv, pww = d[p + "pcov_pos"], d[p + "pcov_att"]
        bg, ba = d[p + "gyro_bias"], d[p + "accel_bias"]
        vel, ntr = d[p + "vel_est_body"], d[p + "counts"]
        n = len(ep)
        import midair_drift as md
        Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ md.RT_BC for i in range(n)])
        Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ md.RT_BC for i in range(n)])
        A = np.arange(200, n - 70, 17)
        numw = np.zeros((len(A), 3)); numv = np.zeros((len(A), 3)); den = 0.0
        for L in FIT:
            Xe = np.linalg.inv(Te[A]) @ Te[A + L]
            Xg = np.linalg.inv(Tg[A]) @ Tg[A + L]
            numw += L * Rot.from_matrix(np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]).as_rotvec()
            numv += L * (Xe[:, :3, 3] - Xg[:, :3, 3]); den += L * L
        w_a, v_a = numw / den, numv / den
        store[nm] = (w_a, v_a)
        cw = np.zeros(n); cv = np.zeros(n)
        for i in range(1, n):
            cw[i] = cw[i - 1] + max(np.trace(pww[i] - pww[i - 1]) / 3, 0)
            cv[i] = cv[i - 1] + max(np.trace(pvv[i] - pvv[i - 1]) / 3, 0)
        nt = np.asarray(ntr, float)
        nt = nt[:, 0] if nt.ndim > 1 else nt
        P = {
            "dPvv": np.array([cv[a + L0] - cv[a] for a in A]),
            "dPww": np.array([cw[a + L0] - cw[a] for a in A]),
            "|dbg|": np.array([np.linalg.norm(bg[a + L0] - bg[a]) for a in A]),
            "|dba|": np.array([np.linalg.norm(ba[a + L0] - ba[a]) for a in A]),
            "|bg|":  np.array([np.linalg.norm(bg[a]) for a in A]),
            "|ba|":  np.array([np.linalg.norm(ba[a]) for a in A]),
            "speed": np.array([np.linalg.norm(vel[a]) for a in A]),
            "|dv|":  np.array([np.linalg.norm(vel[a + L0] - vel[a]) for a in A]),
            "ntrack": np.array([nt[a] for a in A]),
        }
        for chan, tgt in [("rot", np.linalg.norm(w_a, axis=1)),
                          ("trans", np.linalg.norm(v_a, axis=1))]:
            print(f"{nm:<7}{chan:<6}" + "".join(f"{spear(P[k], tgt):>8.2f}" for k in keys))

    print("\nTEST 2 -- predictability horizon: corr(xi_a, xi_{a+D}) vs anchor separation")
    print("         anchors are 17 frames apart, so D=1 is ~17 frames\n")
    print(f"{'run':<7}{'chan':<6}" + "".join(f"{'D='+str(D):>8}" for D in [1, 2, 4, 8, 16]))
    print("-" * 53)
    for nm in names:
        for chan, X in [("rot", store[nm][0]), ("trans", store[nm][1])]:
            row = []
            Xc = X - X.mean(0)
            for D in [1, 2, 4, 8, 16]:
                num = (Xc[:-D] * Xc[D:]).sum()
                den = np.sqrt((Xc[:-D] ** 2).sum() * (Xc[D:] ** 2).sum())
                row.append(num / max(den, 1e-30))
            print(f"{nm:<7}{chan:<6}" + "".join(f"{r:>8.2f}" for r in row))


if __name__ == "__main__":
    main()
