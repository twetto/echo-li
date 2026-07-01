"""Pose-noise NEES diagnostic for the Rust Sparse3DFilter (quick-look figure).

Drives the *shipped* Rust filter through the binding. Each frame the pixel is
generated from the TRUE pose but the filter is fed a POSE-NOISE-perturbed pose
(via _se3), so pose error shows up as depth error. NEES is scored in world
frame using the Rust filter's own euclidean covariance (covariance_euclidean),
so no chart math is reimplemented and no eqvio dependency is needed.

Four arms tell the story:
  A no pose noise            -> calibrated baseline (NEES ~ dim = 3)
  B translation noise, naive -> overconfident (no pose term in the model)
  C translation noise, p_vv  -> the 3x3 velocity cov recovers calibration
  D rotation noise, naive    -> overconfident, and NO 3x3 p_vv can fix it
                                (motivates the 6x6 pose covariance)

Run from a cwd WITHOUT the source echo_li/ dir (e.g. the repo root) so the
pip-installed wheel is imported, not the flat-layout source package:
    echo-li-python/.venv/bin/python echo-li-python/tests/make_pose_noise_nees_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _se3 import exp_se3  # noqa: E402

from echo_li import Sparse3DFilter  # noqa: E402

# --- geometry / experiment knobs --------------------------------------------
FX = FY = 458.0
CX, CY = 376.0, 240.0
Z_TRUE = float(os.environ.get("Z_TRUE", "80.0"))
U0, V0 = 400.0, 250.0
DT = 0.05
N = int(os.environ.get("N_STEPS", "100"))
N_MC = int(os.environ.get("N_MC", "80"))
# BASELINE per frame [m]. If PARALLAX_PX is set, scale it so translation yields
# that many px/frame at Z_TRUE (matched parallax across depths -> decouples depth
# from parallax; fixed baseline conflates them). Landmark pixel sweep = N*px/frame.
BASELINE = float(os.environ.get("BASELINE", "0.05"))
if os.environ.get("PARALLAX_PX"):
    BASELINE = float(os.environ["PARALLAX_PX"]) * Z_TRUE / FX
DIM = 3

SIGMA_PX = 0.5
SIGMA_T = 0.01     # translation pose-noise std [m] per frame
SIGMA_PHI = 0.0015  # rotation pose-noise std [rad] per frame
P_VV_SCALE = 2.0  # 2x because the original test over-inflated; kept for parity
P_VV_T = (P_VV_SCALE * SIGMA_T**2 / DT**2) * np.eye(3)
P_WW_T = (P_VV_SCALE * SIGMA_PHI**2 / DT**2) * np.eye(3)
P_VV_T1 = (SIGMA_T**2 / DT**2) * np.eye(3)  # factor-1 variants
P_WW_T1 = (SIGMA_PHI**2 / DT**2) * np.eye(3)

SETTINGS = dict(
    # filter's assumed pixel noise; decoupled from the injected SIGMA_PX so we can
    # test whether inflating measurement noise alone recalibrates rotation error.
    sigma_pixel=float(os.environ.get("SIGMA_PXF", str(SIGMA_PX))),
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=float(os.environ.get("MAX_DEPTH", "150.0")),
    mahalanobis_reset_chi2=1e9,  # disable resets so we see pure (over)confidence
    process_depth_var=0.0,  # no fallback process noise: the ONLY pose term is p_vv
    anchor_measurement=os.environ.get("ANCHOR", "0") == "1",  # add anchor-pose to R
)

P_W = np.array([(U0 - CX) / FX * Z_TRUE, (V0 - CY) / FY * Z_TRUE, Z_TRUE])


def run_trial(rng, pose_noise, p_vv, p_ww=None, chart="polar3d", unscented=False,
              measurement=False, range_walk=0.0, gate_px_mult=0.0):
    """One MC trial -> per-step NEES (NaN where no live feature).

    gate_px_mult>0: measurement decimation -- only fuse a frame when the tracked
    pixel has moved >= gate_px_mult*SIGMA_PX since the last fused update (fewer,
    stronger-parallax updates -> fewer sequential linearisations)."""
    ctor = getattr(Sparse3DFilter, chart)
    filt = ctor(FX, FY, CX, CY, **{**SETTINGS, "rotation_unscented": unscented,
                                   "pose_measurement": measurement,
                                   "range_walk_var": range_walk})
    nees = np.full(N, np.nan)
    errm = np.full(N, np.nan)   # actual position error [m]
    sigm = np.full(N, np.nan)   # reported 1-sigma = sqrt(trace(cov)/3) [m]
    p_vv_arg = None if p_vv is None else p_vv.tolist()
    p_ww_arg = None if p_ww is None else p_ww.tolist()

    last_uv = None
    for i in range(N):
        t_true = np.eye(4)
        t_true[0, 3] = i * BASELINE
        t_cw = np.linalg.inv(t_true)
        pc = t_cw[:3, :3] @ P_W + t_cw[:3, 3]
        uv = (
            FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
            FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX),
        )

        # fed pose = true pose perturbed by pose noise (xi = [rho, phi])
        xi = pose_noise(rng)
        t_fed = t_true @ exp_se3(xi)

        if gate_px_mult > 0.0 and last_uv is not None:
            if np.hypot(uv[0] - last_uv[0], uv[1] - last_uv[1]) < gate_px_mult * SIGMA_PX:
                continue  # decimate: skip this low-parallax frame (NEES left NaN)

        filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg, p_ww_arg)
        last_uv = uv

        feats = filt.get_features()
        if 42 not in feats:
            continue
        fd = feats[42]
        est = np.asarray(fd["position"])
        cov_euc = np.asarray(fd["covariance_euclidean"])
        r_fed = t_fed[:3, :3]
        est_world = r_fed @ est + t_fed[:3, 3]
        cov_world = r_fed @ cov_euc @ r_fed.T
        err = est_world - P_W
        errm[i] = np.linalg.norm(err)
        sigm[i] = np.sqrt(np.trace(cov_world) / 3.0)
        try:
            nees[i] = float(err @ np.linalg.solve(cov_world, err))
        except np.linalg.LinAlgError:
            pass
    return nees, errm, sigm


def no_noise(rng):
    return np.zeros(6)


def trans_noise(rng):
    return np.concatenate([rng.normal(0, SIGMA_T, 3), np.zeros(3)])


def rot_noise(rng):
    return np.concatenate([np.zeros(3), rng.normal(0, SIGMA_PHI, 3)])


def trans_rot_noise(rng):
    return np.concatenate([rng.normal(0, SIGMA_T, 3), rng.normal(0, SIGMA_PHI, 3)])


SIGMA_PHI_SMALL = 0.0003  # 5x smaller rotation noise

def rot_noise_small(rng):
    return np.concatenate([np.zeros(3), rng.normal(0, SIGMA_PHI_SMALL, 3)])

P_WW_SMALL = (SIGMA_PHI_SMALL**2 / DT**2) * np.eye(3)

# (label, noise_fn, p_vv, p_ww, chart, color)
ARMS = [
    # Baselines
    ("A: no noise (polar3d)", no_noise, None, None, "polar3d", "tab:green"),
    ("A2: no noise (inv_add)", no_noise, None, None, "invdepth_additive3d", "limegreen"),
    # Translation calibration
    ("B: trans, naive", trans_noise, None, None, "polar3d", "tab:orange"),
    ("C: trans, p_vv (polar3d)", trans_noise, P_VV_T1, None, "polar3d", "tab:blue"),
    # Rotation — polar3d (SOT(3) IEKF breaks with rotation process noise)
    ("D: rot, naive (polar3d)", rot_noise, None, None, "polar3d", "tab:red"),
    ("E: rot, p_ww (polar3d)", rot_noise, None, P_WW_T1, "polar3d", "tab:purple"),
    # Rotation — invdepth_additive (well-behaved chart)
    ("F: rot, naive (inv_add)", rot_noise, None, None, "invdepth_additive3d", "salmon"),
    ("G: rot, p_ww 1st-order (inv_add)", rot_noise, None, P_WW_T1, "invdepth_additive3d", "darkviolet"),
    ("G2: rot, p_ww UNSCENTED (inv_add)", rot_noise, None, P_WW_T1, "invdepth_additive3d", "black", True),
    ("G3: rot, p_ww MEASUREMENT (inv_add)", rot_noise, None, P_WW_T1, "invdepth_additive3d", "magenta", False, True),
    ("G4: rot, MEASUREMENT + range_walk floor", rot_noise, None, P_WW_T1, "invdepth_additive3d", "teal", False, True, 1e-8),
    ("G0: rot, range_walk floor only (no rot model)", rot_noise, None, None, "invdepth_additive3d", "silver", False, False, 1e-8),
    # Small rotation — verifies linearized model is correct
    ("H: rot_small, naive (inv_add)", rot_noise_small, None, None, "invdepth_additive3d", "gold"),
    ("I: rot_small, p_ww (inv_add)", rot_noise_small, None, P_WW_SMALL, "invdepth_additive3d", "darkgoldenrod"),
    # Joint noise
    ("J: trans+rot, p_vv+p_ww (inv_add)", trans_rot_noise, P_VV_T1, P_WW_T1, "invdepth_additive3d", "tab:cyan"),
    ("J2: trans+rot UNSCENTED (inv_add)", trans_rot_noise, P_VV_T1, P_WW_T1, "invdepth_additive3d", "tab:brown", True),
    ("J3: FULL POSE measurement p_vv+p_ww->R", trans_rot_noise, P_VV_T1, P_WW_T1, "invdepth_additive3d", "olive", False, True),
    ("J0: trans p_vv only, rot UNMODELED (inv_add)", trans_rot_noise, P_VV_T1, None, "invdepth_additive3d", "lightblue"),
    ("J0g2: J0 + parallax gate 2*sig_px", trans_rot_noise, P_VV_T1, None, "invdepth_additive3d", "tab:pink", False, False, 0.0, 2.0),
    ("J0g5: J0 + parallax gate 5*sig_px", trans_rot_noise, P_VV_T1, None, "invdepth_additive3d", "tab:gray", False, False, 0.0, 5.0),
]


def main():
    steps = np.arange(N)
    fig, ax = plt.subplots(figsize=(11, 6))
    sweep_px = FX * (N * BASELINE) / Z_TRUE
    print(f"pose-noise NEES (dim={DIM}, N_MC={N_MC}, N={N}, Z={Z_TRUE}m); "
          f"landmark pixel sweep = {sweep_px:.0f} px of {int(2*CX)} wide "
          f"({100*sweep_px/(2*CX):.0f}% of view)")
    bin_edges = list(range(0, N, 50))
    print("  " + "arm".ljust(36) + "".join(f"{s+50:>7}" for s in bin_edges)
          + "     (mean NEES per 50 steps, ideal 3)")
    for arm in ARMS:
        label, noise, p_vv, p_ww, chart, color = arm[:6]
        unscented = arm[6] if len(arm) > 6 else False
        measurement = arm[7] if len(arm) > 7 else False
        range_walk = arm[8] if len(arm) > 8 else 0.0
        gate = arm[9] if len(arm) > 9 else 0.0
        alln = np.full((N_MC, N), np.nan)
        alle = np.full((N_MC, N), np.nan)
        alls = np.full((N_MC, N), np.nan)
        for mc in range(N_MC):
            alln[mc], alle[mc], alls[mc] = run_trial(
                np.random.default_rng(7000 + mc), noise, p_vv, p_ww,
                chart, unscented, measurement, range_walk, gate)
        mean_nees = np.nanmean(alln, axis=0)
        mean_err = np.nanmean(alle, axis=0)
        mean_sig = np.nanmean(alls, axis=0)
        ax.plot(steps, mean_nees, color=color, lw=1.6, label=label)
        nbins = "".join(f"{np.nanmean(mean_nees[s:s+50]):7.2f}" for s in bin_edges)
        ebins = "".join(f"{100*np.nanmean(mean_err[s:s+50]):7.2f}" for s in bin_edges)
        sbins = "".join(f"{100*np.nanmean(mean_sig[s:s+50]):7.2f}" for s in bin_edges)
        print(f"  {label:36s}{nbins}")
        if os.environ.get("SHOW_ERR") == "1":
            print(f"  {'    err[cm]':36s}{ebins}")
            print(f"  {'    sig[cm]':36s}{sbins}")

    lo = chi2.ppf(0.025, DIM * N_MC) / N_MC
    hi = chi2.ppf(0.975, DIM * N_MC) / N_MC
    ax.axhline(DIM, color="k", ls=":", lw=1.0, label=f"E[NEES]={DIM} (calibrated)")
    ax.axhspan(lo, hi, color="gray", alpha=0.15, label="95% band")
    ax.set_yscale("log")
    ax.set_xlabel("observation step")
    ax.set_ylabel("mean NEES (log scale)")
    ax.set_title(
        "Sparse3D depth NEES under pose noise\n"
        f"sigma_px={SIGMA_PX}, sigma_t={SIGMA_T} m, sigma_phi={SIGMA_PHI} rad, Z={Z_TRUE} m"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    out = os.path.join(os.environ.get("TMPDIR", "."), "pose_noise_nees.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
