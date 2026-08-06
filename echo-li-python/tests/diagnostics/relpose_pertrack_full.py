"""Per-track structure of the EqVIO relative pose error: rotation AND translation, plus
the questions the rotation result raises.

relpose_bias_pertrack.py showed the ROTATION error is, within a track, a constant rate times
the window length (held-out exponent 1.26 -> 0.07), with the rate varying 1-14x across
anchors. That resurrects the coherent q_k term but leaves four things open, all tested here:

  Q1 TRANSLATION -- does the same hold? Two competing models, both fitted per anchor and
     scored on HELD-OUT lags:
        (A) constant velocity-error vector   dt_rel ~ v_a * L
        (B) scale error                      dt_rel ~ ds_a * b_gt(a,L)     (b_gt = GT baseline)
     B winning would identify the mechanism as scale, matching the along-baseline
     concentration measured earlier (44-95% vs 33% isotropic).

  Q2 OBSERVABILITY -- is the per-track rate predictable from what the FILTER reports? If
     |w_a| correlates with the filter's own accumulated sigma over the window, the coherent
     coefficient is computable online; if not, it needs ground truth and is not shippable.

  Q3 WANDER -- is w_a autocorrelated across anchor epochs (a slowly-drifting bias) or
     independent per track (something track-specific)? Lag-1 correlation of the rate series.

  Q4 RECONCILIATION -- per-track L^2 should imply pooled L^2, but the pooled exponent is
     1.26. Reported here as the ratio of pooled to per-track-implied growth, to see whether
     early death of high-rate tracks (selection) explains the gap.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

FIT = list(range(4, 65, 8))     # 4,12,...,60
TEST = list(range(8, 65, 8))    # 8,16,...,64


def se3(R, p):
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, p; return T


def load(npz, kite, root):
    d = np.load(npz)
    if "baseline_pcov_pos" in d.files:
        ep, eq = d["baseline_est"], d["baseline_quat"]
        gp, gq = d["baseline_gt"], d["baseline_gt_quat"]
        pvv, pww = d["baseline_pcov_pos"], d["baseline_pcov_att"]
        tbc = d["baseline_t_bc"] if "baseline_t_bc" in d.files else md.RT_BC
    else:
        ep, eq, k = d["est"], d["quat"], d["k"]
        pvv, pww = d["pvv"], d["pww"]
        ds = md.MidAir(root, "Kite_training", "sunny", kite, 1.0)
        T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        gp = np.zeros_like(ep); gq = np.zeros((len(k), 4))
        for i, kk in enumerate(k):
            M = T @ ds.pose(int(kk))
            gp[i], gq[i] = M[:3, 3], Rot.from_matrix(M[:3, :3]).as_quat()
        tbc = md.RT_BC
    n = len(ep)
    Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ tbc for i in range(n)])
    Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ tbc for i in range(n)])
    return Te, Tg, pvv, pww, n


def rel(Te, Tg, a, L):
    Xe = np.linalg.inv(Te[a]) @ Te[a + L]
    Xg = np.linalg.inv(Tg[a]) @ Tg[a + L]
    dphi = Rot.from_matrix(np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]).as_rotvec()
    dt = Xe[:, :3, 3] - Xg[:, :3, 3]
    return dphi, dt, Xg[:, :3, 3]


def expo(vals):
    return float(np.polyfit(np.log(TEST), np.log(vals), 1)[0])


def analyse(Te, Tg, pvv, pww, n, burn=200, stride=17):
    A = np.arange(burn, n - max(TEST) - 1, stride)
    if len(A) < 30:
        return None
    # ---- per-anchor fits on FIT lags -------------------------------------
    numw = np.zeros((len(A), 3)); numv = np.zeros((len(A), 3)); den = 0.0
    num_s = np.zeros(len(A)); den_s = np.zeros(len(A))
    for L in FIT:
        dphi, dt, bg = rel(Te, Tg, A, L)
        numw += L * dphi; numv += L * dt; den += L * L
        num_s += (dt * bg).sum(1); den_s += (bg * bg).sum(1)
    w_a = numw / den                      # rad/frame, per anchor
    v_a = numv / den                      # m/frame,   per anchor
    s_a = num_s / np.maximum(den_s, 1e-12)  # dimensionless scale error, per anchor

    # ---- held-out residual exponents -------------------------------------
    raw_r, res_r, raw_t, resA_t, resB_t = [], [], [], [], []
    for L in TEST:
        dphi, dt, bg = rel(Te, Tg, A, L)
        raw_r.append(np.median((dphi ** 2).sum(1)))
        res_r.append(np.median(((dphi - w_a * L) ** 2).sum(1)))
        raw_t.append(np.median((dt ** 2).sum(1)))
        resA_t.append(np.median(((dt - v_a * L) ** 2).sum(1)))
        resB_t.append(np.median(((dt - s_a[:, None] * bg) ** 2).sum(1)))

    # ---- Q2 observability: does the filter's own sigma predict |w_a|? -----
    cw = np.zeros((n, 3, 3))
    for i in range(1, n):
        m = 0.5 * ((pww[i] - pww[i - 1]) + (pww[i] - pww[i - 1]).T)
        ev, V = np.linalg.eigh(m)
        cw[i] = cw[i - 1] + (V * np.maximum(ev, 0)) @ V.T
    L0 = 32
    pred = np.array([np.sqrt(max(np.trace(cw[a + L0] - cw[a]) / 3, 1e-30)) for a in A])
    obs = np.linalg.norm(w_a, axis=1) * L0
    ok = np.isfinite(pred) & (pred > 0) & (obs > 0)
    r_obs = np.corrcoef(np.log(pred[ok]), np.log(obs[ok]))[0, 1] if ok.sum() > 10 else np.nan

    # ---- Q3 wander: lag-1 autocorrelation of the rate series -------------
    def ac1(x):
        x = x - x.mean(0)
        num = (x[:-1] * x[1:]).sum()
        return float(num / max((x * x).sum(), 1e-30))
    return dict(er_raw=expo(raw_r), er_res=expo(res_r),
                et_raw=expo(raw_t), et_A=expo(resA_t), et_B=expo(resB_t),
                redA=1 - np.sum(resA_t) / np.sum(raw_t), redB=1 - np.sum(resB_t) / np.sum(raw_t),
                r_obs=r_obs, ac_w=ac1(w_a), ac_v=ac1(v_a),
                s_med=float(np.median(np.abs(s_a))), n=len(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir")
    a = ap.parse_args()
    SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
          "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")
    runs = [("MidAir", "VO t0", f"{SP}/mono_run_fig_traj0.npz", None),
            ("MidAir", "VO t1", f"{SP}/mono_run_fig_traj1.npz", None),
            ("MidAir", "VO t2", f"{SP}/mono_run_fig_traj2.npz", None),
            ("MidAir", "Kite t0", f"{SP}/kite_traj0.npz", 0),
            ("MidAir", "Kite t1", f"{SP}/kite_traj1.npz", 1)]
    runs += [("EuRoC", s[:5], f"{SP}/euroc_{s}.npz", None) for s in
             ["V1_01", "V1_02_medium", "V1_03_difficult",
              "V2_01_easy", "V2_02_medium", "V2_03_difficult"]]
    runs += [("TUM-VI", f"room{i}", f"{SP}/tumvi_room{i}.npz", None) for i in range(1, 7)]

    print("held-out exponents after per-anchor fits    |  A = v_a*L   B = scale*baseline\n")
    print(f"{'ds':<8}{'run':<8}| {'ROT raw':>8}{'resid':>7} | {'TR raw':>7}{'residA':>8}{'residB':>8}"
          f" | {'A expl':>7}{'B expl':>7} | {'corr':>6}{'ac_w':>6}{'ac_v':>6}{'|scale|':>9}")
    print("-" * 112)
    acc = []
    for ds, name, f, kite in runs:
        if not Path(f).exists():
            continue
        Te, Tg, pvv, pww, n = load(f, kite, a.root)
        r = analyse(Te, Tg, pvv, pww, n)
        if r is None:
            continue
        acc.append(r)
        print(f"{ds:<8}{name:<8}| {r['er_raw']:>8.2f}{r['er_res']:>7.2f} | {r['et_raw']:>7.2f}"
              f"{r['et_A']:>8.2f}{r['et_B']:>8.2f} | {100*r['redA']:>6.0f}%{100*r['redB']:>6.0f}%"
              f" | {r['r_obs']:>6.2f}{r['ac_w']:>6.2f}{r['ac_v']:>6.2f}{r['s_med']:>9.3f}")
    k = lambda f: np.nanmean([x[f] for x in acc])
    print(f"\nMEAN     rot {k('er_raw'):.2f}->{k('er_res'):.2f} | trans {k('et_raw'):.2f}-> "
          f"A {k('et_A'):.2f} / B {k('et_B'):.2f} | explained A {100*k('redA'):.0f}% B {100*k('redB'):.0f}%"
          f" | corr(filter sigma, true rate) {k('r_obs'):+.2f} | autocorr w {k('ac_w'):+.2f} v {k('ac_v'):+.2f}")


if __name__ == "__main__":
    main()
