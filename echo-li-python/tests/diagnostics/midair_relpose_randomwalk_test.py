"""Is the EqVIO relative (anchor->current) pose error actually a RANDOM WALK?

Sparse3D's range-process term (filter_formulation.md eq. 10-11) needs the RELATIVE
pose covariance over a landmark's anchor->current window,
    sigma^2_rel = Sigma(t_k) - Sigma(t_a),
but a current-state EqVIO retains only the ABSOLUTE marginal Sigma(t_k). The shipped
code uses the increment of the absolute marginal as a random-walk SURROGATE for the
relative covariance. That is exact only if the pose error really is a random walk
(independent increments); otherwise the anchor-current cross-covariance matters and
you need a kept-pose window (filter_formulation.md section X.1 -- the open approximation).

Every existing harness (midair_pose_window_test, midair_posecov_nees_test) INJECTS
iid-vs-random-walk pose error. None measures which one the filter actually produces.
This does: it scores the REAL relative pose error of a real EqVIO run against the
surrogate covariance the code would have used, as a function of window length.

    surrogate honest  => NEES ~ chi2(3), flat in lag   => the pose window buys nothing
    surrogate too small => NEES >> 3, growing with lag => cross-covariance matters

RESULT (2026-08-05, MidAir sunny, tuned per-traj mono configs, lags 4-256): the
random-walk assumption is FALSE. Fitting variance ~ lag^p:

    channel        true error        shipped surrogate
    translation    lag^1.70-1.99     lag^1.00-1.01
    rotation       lag^1.67-1.97     lag^1.00-1.01

The surrogate is exactly diffusive BY CONSTRUCTION (a cumulative sum of roughly
stationary per-frame increments can only grow linearly); the real error is near
BALLISTIC, i.e. coherent drift. The deficit is therefore in the EXPONENT, not the
coefficient, so the scalar `pose_range_scale` cannot repair it -- it slides the curve
without changing its slope. The surrogate over-covers short windows and under-covers
long ones (at lag 256: 9.3x / 91x / 1.4x too small on traj 0 / 1 / 2).

It simultaneously answers a second question. Eq (11) needs SCALAR relative rotation and
relative BASELINE variances. The shipped update isotropizes, sigma_t^2 = trace(P_vv)/3,
while the BIRTH path (init_cov_3d/baseline_tau in sparse_3d/mod.rs) projects onto the
baseline, t_hat^T P_vv t_hat. This reports both against the measured along-baseline
error. MEASURED: they differ by only 8-16% at every lag on every trajectory, so the
accumulated increment is nearly isotropic and unifying the update with the birth path
is correct but numerically irrelevant. The exponent is what matters.

NOTE the lag-2 rows are unreliable: the accumulated increment is near-singular there
and the PSD floor binds (visible as a nonzero neg-eig count). Fit from lag 4.

    PY=echo-li-python/venv/bin/python
    # needs a run with per-frame pose covariance; prior_ab also stores GT attitude,
    # so that format needs no dataset access. Use a TUNED per-traj config: the
    # untuned diagnostics config gives 6.7% ATE on traj 2 vs the 3.0% baseline, and
    # a mis-tuned filter confounds "surrogate is wrong" with "filter is wrong".
    $PY midair_vio_sparse3d_prior_ab.py ... --save-npz mono_run_fig_traj2.npz
    $PY midair_relpose_randomwalk_test.py --npz mono_run_fig_traj2.npz \
        --root <MidAir> --traj 2 --lags 4,8,16,32,64,128,256
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

CHI2_3_MED, CHI2_3_95, CHI2_3_99 = 2.3660, 7.8147, 11.3449


def psd_inv(m, floor=1e-14):
    """Symmetrize, clip eigenvalues to a floor, invert. Mirrors the psd_clip the
    shipped surrogate needs because Sigma(t_k) - Sigma(t_a) is not PSD in general."""
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    n_neg = int((w <= floor).sum())
    w = np.maximum(w, floor)
    return (v * (1.0 / w)) @ v.T, n_neg


def se3(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--burn-in", type=int, default=200,
                    help="skip early frames while the filter is still converging")
    ap.add_argument("--lags", default="5,10,20,40,80,160")
    args = ap.parse_args()

    d = np.load(args.npz)
    if "baseline_pcov_pos" in d.files:
        # midair_vio_sparse3d_prior_ab format: GT already mapped to the VIO world by
        # nwu_body_pose, and GT attitude stored, so no dataset access is needed.
        p = "baseline_"
        est_p, est_q = d[p + "est"], d[p + "quat"]
        gt_p, gt_q = d[p + "gt"], d[p + "gt_quat"]
        pvv, pww, kidx = d[p + "pcov_pos"], d[p + "pcov_att"], d[p + "k"]
    else:
        # midair_vio_run format: GT attitude not stored, so read it from the dataset
        # and apply the same diag(1,-1,-1) map that midair_vio_run uses.
        est_p, est_q = d["est"], d["quat"]
        pvv, pww, kidx = d["pvv"], d["pww"], d["k"]
        ds = md.MidAir(args.root, args.subset, args.cond, args.traj, 1.0)
        T_nwu = np.eye(4)
        T_nwu[:3, :3] = np.diag([1.0, -1.0, -1.0])
        gt_p = np.zeros_like(est_p)
        gt_q = np.zeros((len(kidx), 4))
        for i, k in enumerate(kidx):
            T = T_nwu @ ds.pose(int(k))
            gt_p[i], gt_q[i] = T[:3, 3], Rot.from_matrix(T[:3, :3]).as_quat()
    n = len(kidx)
    R_bc = md.RT_BC[:3, :3]

    # Camera poses, estimated and GT, in a common world convention. Zero lever arm
    # (RT_BC has no translation), so T_wc = T_wb @ R_bc.
    Twc_est, Twc_gt = np.zeros((n, 4, 4)), np.zeros((n, 4, 4))
    finite = np.zeros(n, bool)
    for i in range(n):
        Twc_est[i] = se3(Rot.from_quat(np.asarray(est_q[i])).as_matrix() @ R_bc, est_p[i])
        Twc_gt[i] = se3(Rot.from_quat(np.asarray(gt_q[i])).as_matrix() @ R_bc, gt_p[i])
        finite[i] = np.isfinite(pvv[i]).all() and np.isfinite(pww[i]).all()

    # SANITY must be gauge-free: the VIO world frame differs from GT's by a constant
    # SE(3) (that is what the ATE's SE3 alignment removes), so the ABSOLUTE attitude
    # error is dominated by that constant and says nothing. The relative rotation over
    # a short window cancels it exactly -- and is what this test actually scores.
    rel1 = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(
        (np.linalg.inv(Twc_gt[i - 1]) @ Twc_gt[i])[:3, :3].T
        @ (np.linalg.inv(Twc_est[i - 1]) @ Twc_est[i])[:3, :3]).as_rotvec()))
        for i in range(1, n)])
    print(f"# SANITY (gauge-free): 1-frame RELATIVE rotation error, median {np.median(rel1):.3f} deg "
          f"(p90 {np.percentile(rel1,90):.3f}) -- a frame bug would show tens of degrees")
    print(f"# frames with finite pose covariance: {finite.sum()}/{n}")

    # The SHIPPED surrogate: the per-frame increment of the absolute marginal, PSD-clipped,
    # injected as process noise every frame. The anchor->current covariance the filter
    # effectively carries is therefore the ACCUMULATED SUM of those clipped increments --
    # not clip(Sigma(t_k) - Sigma(t_a)), which differs precisely on the frames where a
    # vision update SHRINKS Sigma (clipping drops those, the difference keeps them).
    def clip_psd(m):
        m = 0.5 * (m + m.T)
        w, v = np.linalg.eigh(m)
        return (v * np.maximum(w, 0.0)) @ v.T

    cum_t = np.zeros((n, 3, 3))
    cum_w = np.zeros((n, 3, 3))
    for i in range(1, n):
        dt_i = clip_psd(pvv[i] - pvv[i - 1]) if (finite[i] and finite[i - 1]) else np.zeros((3, 3))
        dw_i = clip_psd(pww[i] - pww[i - 1]) if (finite[i] and finite[i - 1]) else np.zeros((3, 3))
        cum_t[i], cum_w[i] = cum_t[i - 1] + dt_i, cum_w[i - 1] + dw_i

    print(f"# {args.npz}  n={n} frames, burn-in {args.burn_in}, traj {args.traj}/{args.cond}")
    print("# Scoring the REAL relative pose error against the shipped random-walk surrogate")
    print(f"# ideal chi2(3): median {CHI2_3_MED:.3f}  mean 3.0  %>chi2_95 = 5  %>chi2_99 = 1\n")

    hdr = (f"{'lag':>5} {'n':>6} | {'ROT med':>8} {'mean':>9} {'%>95':>6} {'%>99':>6} |"
           f" {'TRANS med':>10} {'mean':>10} {'%>95':>6} {'%>99':>6} | {'neg-eig':>7}")
    print(hdr); print("-" * len(hdr))

    scal = []
    for lag in [int(x) for x in args.lags.split(",")]:
        rot_n, tr_n, negs = [], [], 0
        s_iso, s_proj, s_emp, s_rot_sur, s_rot_emp = [], [], [], [], []
        for i in range(args.burn_in + lag, n):
            if not (finite[i] and finite[i - lag]):
                continue
            a = i - lag
            X_e = np.linalg.inv(Twc_est[a]) @ Twc_est[i]
            X_g = np.linalg.inv(Twc_gt[a]) @ Twc_gt[i]
            E = np.linalg.inv(X_g) @ X_e            # error, right/local chart: log(T_true^-1 T_est)
            dphi = Rot.from_matrix(E[:3, :3]).as_rotvec()
            dt = X_e[:3, 3] - X_g[:3, 3]            # baseline error in the ANCHOR camera frame

            Sr, nr = psd_inv(cum_w[i] - cum_w[a])
            St, nt = psd_inv(cum_t[i] - cum_t[a])
            negs += nr + nt
            rot_n.append(float(dphi @ Sr @ dphi))
            tr_n.append(float(dt @ St @ dt))

            # Eq (11) scalars: isotropized (shipped update) vs baseline-projected (birth path)
            b = X_g[:3, 3]
            nb = np.linalg.norm(b)
            if nb > 1e-6:
                bh = b / nb
                dS = cum_t[i] - cum_t[a]
                s_iso.append(np.trace(dS) / 3.0)
                s_proj.append(float(bh @ dS @ bh))
                s_emp.append(float(dt @ bh) ** 2)
                dW = cum_w[i] - cum_w[a]
                s_rot_sur.append(np.trace(dW) / 3.0)
                s_rot_emp.append(float(dphi @ dphi) / 3.0)

        rot_n, tr_n = np.array(rot_n), np.array(tr_n)
        if rot_n.size == 0:
            continue
        print(f"{lag:>5} {rot_n.size:>6} | {np.median(rot_n):>8.2f} {rot_n.mean():>9.1f} "
              f"{100*np.mean(rot_n>CHI2_3_95):>5.1f}% {100*np.mean(rot_n>CHI2_3_99):>5.1f}% | "
              f"{np.median(tr_n):>10.2f} {tr_n.mean():>10.1f} "
              f"{100*np.mean(tr_n>CHI2_3_95):>5.1f}% {100*np.mean(tr_n>CHI2_3_99):>5.1f}% | {negs:>7}")
        scal.append((lag, np.median(s_iso), np.median(s_proj), np.median(s_emp),
                     np.median(s_rot_sur), np.median(s_rot_emp)))

    print(f"\n# Eq (11) scalars, medians. sigma_b^2: shipped update uses trace/3; birth path")
    print(f"# (baseline_tau) uses the baseline projection. 'measured' is the real along-baseline error^2.")
    h2 = (f"{'lag':>5} | {'trace/3':>10} {'t^T.S.t':>10} {'measured':>10} | "
          f"{'iso/meas':>9} {'proj/meas':>9} | {'rot sur':>10} {'rot meas':>10} {'ratio':>8}")
    print(h2); print("-" * len(h2))
    for lag, iso, proj, emp, rs, re_ in scal:
        print(f"{lag:>5} | {iso:>10.3e} {proj:>10.3e} {emp:>10.3e} | "
              f"{iso/max(emp,1e-30):>9.3f} {proj/max(emp,1e-30):>9.3f} | "
              f"{rs:>10.3e} {re_:>10.3e} {rs/max(re_,1e-30):>8.3f}")


if __name__ == "__main__":
    main()
