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
Z_TRUE = 80.0
U0, V0 = 400.0, 250.0
BASELINE = 0.05
DT = 0.05
N = 100
N_MC = 80
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
    sigma_pixel=SIGMA_PX,
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=150.0,
    mahalanobis_reset_chi2=1e9,  # disable resets so we see pure (over)confidence
    process_depth_var=0.0,  # no fallback process noise: the ONLY pose term is p_vv
)

P_W = np.array([(U0 - CX) / FX * Z_TRUE, (V0 - CY) / FY * Z_TRUE, Z_TRUE])


def run_trial(rng, pose_noise, p_vv, p_ww=None, chart="polar3d"):
    """One MC trial -> per-step NEES (NaN where no live feature)."""
    ctor = getattr(Sparse3DFilter, chart)
    filt = ctor(FX, FY, CX, CY, **SETTINGS)
    nees = np.full(N, np.nan)
    p_vv_arg = None if p_vv is None else p_vv.tolist()
    p_ww_arg = None if p_ww is None else p_ww.tolist()

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

        filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg, p_ww_arg)

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
        try:
            nees[i] = float(err @ np.linalg.solve(cov_world, err))
        except np.linalg.LinAlgError:
            pass
    return nees


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
    ("G: rot, p_ww (inv_add)", rot_noise, None, P_WW_T1, "invdepth_additive3d", "darkviolet"),
    # Small rotation — verifies linearized model is correct
    ("H: rot_small, naive (inv_add)", rot_noise_small, None, None, "invdepth_additive3d", "gold"),
    ("I: rot_small, p_ww (inv_add)", rot_noise_small, None, P_WW_SMALL, "invdepth_additive3d", "darkgoldenrod"),
    # Joint noise
    ("J: trans+rot, p_vv+p_ww (inv_add)", trans_rot_noise, P_VV_T1, P_WW_T1, "invdepth_additive3d", "tab:cyan"),
]


def main():
    steps = np.arange(N)
    fig, ax = plt.subplots(figsize=(11, 6))
    print(f"pose-noise NEES (dim={DIM}, N_MC={N_MC}, Z={Z_TRUE}m)")
    for label, noise, p_vv, p_ww, chart, color in ARMS:
        alln = np.full((N_MC, N), np.nan)
        for mc in range(N_MC):
            alln[mc] = run_trial(np.random.default_rng(7000 + mc), noise, p_vv, p_ww, chart)
        mean_nees = np.nanmean(alln, axis=0)
        ax.plot(steps, mean_nees, color=color, lw=1.6, label=label)
        settled = np.nanmean(mean_nees[N // 2:])
        print(f"  {label:36s} settled mean NEES = {settled:6.2f}  (ideal {DIM})")

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
    out = "/tmp/pose_noise_nees.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
