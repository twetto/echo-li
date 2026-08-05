"""The P_ac cross-term probe of half_schmidt_covariance.md section 7.

That note asks, before any pose-clone machinery is built: how much does DROPPING the
anchor-current cross-covariance P_ac actually change the relative pose covariance that
Sparse3D triangulates with? It proposes comparing

    P_rel_diag       = A P_aa A^T + C P_cc C^T          (no cross term)
    P_rel_true_clone = the full expression including P_ac

projected along the depth direction. "If the missing cross term is small in real
sequences, clone machinery is not the right next cost."

P_ac is precisely what a current-state EqVIO does not retain, so it cannot be read out.
It can be BACKED OUT instead. To first order, with the relative error expressed in the
ANCHOR camera frame (gauge-free), the relative position error is

    dt_rel ~ R_ac dp_c - dp_a - [b]x dphi_a

so if the anchor and current pose errors were INDEPENDENT the relative covariance would be

    P_diag = P_aa^pos + R_ac P_cc^pos R_ac^T + [b]x P_aa^rot [b]x^T .

The measured covariance of the real relative error is the truth. The gap between them is
exactly what the cross term contributes:

    cross-term cancellation = 1 - P_measured / P_diag

Large cancellation => anchor and current errors are strongly common-mode, the cross term
carries most of the covariance, and a kept-pose window is doing real work.
Small cancellation => the poses are effectively independent and clones buy little.

This also places the SHIPPED surrogate (accumulated PSD-clipped per-frame increments of
the absolute marginal) on the same axis, so all three constructions are comparable at the
system's own operating point.

Everything is projected onto the BASELINE direction, which is the depth-relevant
direction: eq. (11) has sigma_r/r ~ sigma_b/b along the baseline.

CAVEAT: `sparse_camera_pose_covariances` returns only the two diagonal blocks (P_vv, P_ww);
the intra-frame translation-rotation cross block Sigma^{t,phi} is not exposed (see
filter_formulation V-B, where it is noted as not separately characterized), so P_diag omits
it. That biases P_diag by an unknown but bounded amount and is a floor on this probe's
precision, not on its sign.

    PY=echo-li-python/venv/bin/python
    $PY midair_pac_crossterm_probe.py --npz mono_run_fig_traj2.npz --traj 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402


def se3(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def clip_psd(m):
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    return (v * np.maximum(w, 0.0)) @ v.T


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--root", default="/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir")
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--lags", default="4,8,16,32,64,128,256")
    ap.add_argument("--stride", type=int, default=5, help="subsample window starts")
    args = ap.parse_args()

    d = np.load(args.npz)
    if "baseline_pcov_pos" in d.files:
        p = "baseline_"
        ep, eq = d[p + "est"], d[p + "quat"]
        gp, gq = d[p + "gt"], d[p + "gt_quat"]
        pvv, pww = d[p + "pcov_pos"], d[p + "pcov_att"]
    else:
        # midair_vio_run format: GT attitude comes from the dataset, mapped by the
        # same diag(1,-1,-1) that midair_vio_run applies.
        ep, eq = d["est"], d["quat"]
        pvv, pww, kidx = d["pvv"], d["pww"], d["k"]
        ds = md.MidAir(args.root, args.subset, args.cond, args.traj, 1.0)
        T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        gp = np.zeros_like(ep); gq = np.zeros((len(kidx), 4))
        for i, k in enumerate(kidx):
            M = T @ ds.pose(int(k))
            gp[i], gq[i] = M[:3, 3], Rot.from_matrix(M[:3, :3]).as_quat()
    n = len(ep)
    # Body->camera extrinsic. EuRoC has a real lever arm, MidAir's RT_BC has none,
    # so compose the FULL SE(3): T_wc = T_wb @ T_bc (reduces to the old rotation-only
    # form exactly when the translation is zero).
    T_bc = d["baseline_t_bc"] if "baseline_t_bc" in d.files else md.RT_BC

    Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ T_bc for i in range(n)])
    Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ T_bc for i in range(n)])

    cum_t = np.zeros((n, 3, 3))
    for i in range(1, n):
        cum_t[i] = cum_t[i - 1] + clip_psd(pvv[i] - pvv[i - 1])

    print(f"# {Path(args.npz).name}  traj {args.traj}  n={n}")
    print("# All variances projected on the BASELINE direction (the depth-relevant direction).")
    print("# 'diag' drops P_ac (treats anchor & current pose error as independent);")
    print("# 'surrogate' is what ships; 'measured' is the real relative error.\n")

    hdr = (f"{'lag':>5} {'n':>6} | {'diag':>11} {'surrogate':>11} {'measured':>11} | "
           f"{'diag/meas':>10} {'surr/meas':>10} | {'cross-term cancels':>19}")
    print(hdr)
    print("-" * len(hdr))

    for lag in [int(x) for x in args.lags.split(",")]:
        a = np.arange(args.burn_in, n - lag, args.stride)
        c = a + lag
        v_diag, v_sur, v_meas = [], [], []
        for i, j in zip(a, c):
            Xe = np.linalg.inv(Te[i]) @ Te[j]
            Xg = np.linalg.inv(Tg[i]) @ Tg[j]
            b = Xg[:3, 3]
            nb = np.linalg.norm(b)
            if nb < 1e-6:
                continue
            bh = b / nb
            dt = Xe[:3, 3] - Xg[:3, 3]

            R_ac = Te[i][:3, :3].T @ Te[j][:3, :3]
            P_diag = (pvv[i]
                      + R_ac @ pvv[j] @ R_ac.T
                      + skew(b) @ pww[i] @ skew(b).T)
            v_diag.append(float(bh @ P_diag @ bh))
            v_sur.append(float(bh @ (cum_t[j] - cum_t[i]) @ bh))
            v_meas.append(float(dt @ bh) ** 2)

        v_diag, v_sur, v_meas = np.array(v_diag), np.array(v_sur), np.array(v_meas)
        if v_meas.size == 0:
            continue
        # medians throughout: the error distribution is heavy-tailed, so the mean of
        # v_meas is dominated by the same outlier tail documented in V-E.
        md_, ms_, mm_ = np.median(v_diag), np.median(v_sur), np.median(v_meas)
        canc = 100.0 * (1.0 - mm_ / md_)
        print(f"{lag:>5} {v_meas.size:>6} | {md_:>11.3e} {ms_:>11.3e} {mm_:>11.3e} | "
              f"{md_/mm_:>10.1f} {ms_/mm_:>10.3f} | {canc:>18.2f}%")

    print("\n# cross-term cancellation = 1 - measured/diag: the fraction of the naive")
    print("#   cross-term-free covariance that the anchor-current correlation removes.")
    print("# ~0%   => poses effectively independent, P_ac negligible, clones buy little.")
    print("# ~100% => the relative error is almost entirely common-mode; the cross term")
    print("#          carries the covariance and a kept-pose window is doing real work.")


if __name__ == "__main__":
    main()
