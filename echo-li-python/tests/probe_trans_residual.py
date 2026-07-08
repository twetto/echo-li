"""Isolate the translation NEES residual by axis.

Motion is along +x, so the triangulation baseline is x. Along-baseline
translation noise perturbs the baseline length and can corrupt the range
direction, while perpendicular noise is a lateral pixel shift captured by
the projection Jacobian.

Config: invdepth_additive with p_vv/p_ww fed to match the injected covariance.
Reports settled NEES over stable windows.

  .venv/Scripts/python.exe echo-li-python/tests/probe_trans_residual.py
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _se3 import exp_se3  # noqa: E402
from echo_li import Sparse3DFilter  # noqa: E402

FX = FY = 458.0
CX, CY = 376.0, 240.0
Z_TRUE = float(os.environ.get("Z_TRUE", "80.0"))
U0, V0 = 400.0, 250.0
DT = 0.05
N = int(os.environ.get("N_STEPS", "1000"))
N_MC = int(os.environ.get("N_MC", "40"))
PARALLAX_PX = float(os.environ.get("PARALLAX_PX", "1.0"))
BASELINE = PARALLAX_PX * Z_TRUE / FX
SCORE_OUTPUT_POSE_COV = os.environ.get("SCORE_OUTPUT_POSE_COV", "0") == "1"
SIGMA_PX = 0.5
SIGMA_T = 0.01
SIGMA_PHI = 0.0015
PWW_SCALE = float(os.environ.get("PWW_SCALE", "1.0"))
WIN = (int(0.35 * N), int(0.85 * N))  # settled window (skip init transient + view-exit)

SETTINGS = dict(
    sigma_pixel=SIGMA_PX, min_track_length=1, init_depth_var=0.01,
    max_depth=float(os.environ.get("MAX_DEPTH", "2000.0")),
    mahalanobis_reset_chi2=1e9, process_depth_var=0.0,
    rotation_unscented=os.environ.get("ROTATION_UNSCENTED", "0") == "1",
)
P_W = np.array([(U0 - CX) / FX * Z_TRUE, (V0 - CY) / FY * Z_TRUE, Z_TRUE])


def run_trial(rng, t_mask, rot_on, extra=None):
    """t_mask: 3-vector of {0,1} selecting which translation axes get noise."""
    filt = Sparse3DFilter.invdepth_additive3d(FX, FY, CX, CY, **{**SETTINGS, **(extra or {})})
    t_mask = np.asarray(t_mask, float)
    # p_vv / p_ww fed = the ACTUAL injected covariance (consider treatment).
    p_vv = (SIGMA_T**2) * np.diag(t_mask)
    p_ww = (PWW_SCALE * SIGMA_PHI**2) * np.eye(3) if rot_on else None
    p_vv_arg = p_vv.tolist()
    p_ww_arg = None if p_ww is None else p_ww.tolist()
    nees = np.full(N, np.nan)
    for i in range(N):
        t_true = np.eye(4); t_true[0, 3] = i * BASELINE
        t_cw = np.linalg.inv(t_true)
        pc = t_cw[:3, :3] @ P_W + t_cw[:3, 3]
        uv = (FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
              FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX))
        xi = np.concatenate([rng.normal(0, SIGMA_T, 3) * t_mask,
                             rng.normal(0, SIGMA_PHI, 3) if rot_on else np.zeros(3)])
        t_fed = t_true @ exp_se3(xi)
        filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg, p_ww_arg)
        feats = filt.get_features()
        if 42 not in feats:
            continue
        fd = feats[42]
        est = np.asarray(fd["position"]); cov = np.asarray(fd["covariance_euclidean"])
        r_fed = t_fed[:3, :3]
        ew = r_fed @ est + t_fed[:3, 3]; cw = r_fed @ cov @ r_fed.T
        if SCORE_OUTPUT_POSE_COV:
            if p_vv is not None:
                cw += r_fed @ (p_vv) @ r_fed.T
            if p_ww is not None:
                ex = np.array([[0.0, -est[2], est[1]],
                               [est[2], 0.0, -est[0]],
                               [-est[1], est[0], 0.0]])
                cw += r_fed @ (ex @ (p_ww) @ ex.T) @ r_fed.T
        err = ew - P_W
        try:
            nees[i] = float(err @ np.linalg.solve(cw, err))
        except np.linalg.LinAlgError:
            pass
    return nees


# Diagnose the no-noise baseline: is ~3.5 from update-count drift, the init
# transient, or Gaussian-Beta inlier weighting? Windows early/mid/late show drift;
# GB-off (a huge, b tiny -> inlier prob ~1) isolates the weighting overhead.
GB_OFF = dict(a_init=1e12, b_init=1e-12)
ARMS = [
    ("no noise                 ", [0, 0, 0], False, None),
    ("no noise, GB off         ", [0, 0, 0], False, GB_OFF),
    ("trans x (baseline)       ", [1, 0, 0], False, None),
    ("trans yz (perp)          ", [0, 1, 1], False, None),
    ("full pose (J3)           ", [1, 1, 1], True, None),
    ("full pose, rot unscented ", [1, 1, 1], True, dict(rotation_unscented=True)),
    ("full pose, GB off        ", [1, 1, 1], True, GB_OFF),
]
WINDOWS = [("early", int(0.10 * N), int(0.30 * N)),
           ("mid", int(0.35 * N), int(0.65 * N)),
           ("late", int(0.70 * N), int(0.95 * N))]

print(f"NEES by window (N_MC={N_MC}, N={N}, Z={Z_TRUE}, parallax {PARALLAX_PX}px/frame); ideal 3")
print("  " + "arm".ljust(27) + "".join(f"{w[0]:>8}" for w in WINDOWS))
for label, tmask, rot, extra in ARMS:
    alln = np.full((N_MC, N), np.nan)
    for mc in range(N_MC):
        alln[mc] = run_trial(np.random.default_rng(7000 + mc), tmask, rot, extra)
    mean_curve = np.nanmean(alln, axis=0)
    cells = "".join(f"{np.nanmean(mean_curve[a:b]):8.2f}" for _, a, b in WINDOWS)
    print(f"  {label}{cells}")
