"""Sparse3D IEKF: joint p_vv x range_walk_var sweep under injected pose noise.

Each frame the pixel is generated from the TRUE pose but the filter is fed a
translation-pose-noise-perturbed pose (via _se3), so pose error shows up as
landmark error. We sweep the two Σ-floor sources -- the per-step velocity
covariance `p_vv` and the radial `range_walk_var` -- and report the settled NEES,
to see which (and at what magnitude) recalibrates the filter.

Findings at Z=320,
translation pose-noise σ_t = 0.01 m/frame, Analytic 2nd-order:
  * range_walk_var ~ 1e-8 ALONE recalibrates even with pose noise (NEES ~ 3.1):
    the radial floor absorbs both the triangulation bias and the pose-induced
    error.
  * the "physical" p_vv = 2σ_t²/dt² is ~10-30x too large for this INDEPENDENT
    per-frame pose noise (it assumes a random walk); scale ~0.03-0.05 hits 3.
    For a genuine random-walk pose process the nominal magnitude is right.
  * p_vv and range_walk_var are partially redundant (both inject a Σ-floor);
    stacking them over-inflates (NEES < 3, under-confident).

Run from the REPO ROOT (so the installed wheel imports, not the source dir):
    echo-li-python/venv/bin/python echo-li-python/tests/make_pose_uncertainty_joint_sweep.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _se3 import exp_se3  # noqa: E402

from echo_li import Sparse3DFilter  # noqa: E402

FX = FY = 458.0
CX, CY = 376.0, 240.0
U0, V0 = 400.0, 250.0
Z = 320.0
BASELINE = 0.05
DT = 0.05
SIGMA_PX = 0.5
SIGMA_T = 0.01  # translation pose-noise std [m] per frame
N = 600
N_MC = 15

PVV0 = (2.0 * SIGMA_T**2 / DT**2) * np.eye(3)  # nominal rel-translation cov
P_W = np.array([(U0 - CX) / FX * Z, (V0 - CY) / FY * Z, Z])

RANGE_WALK_VAR = [0.0, 1e-9, 1e-8]
PVV_SCALE = [0.0, 0.1, 0.3, 1.0]


def settled_nees(range_walk_var, pvv_scale):
    pvv = None if pvv_scale == 0.0 else (pvv_scale * PVV0)
    p_vv_arg = None if pvv is None else pvv.tolist()
    settings = dict(
        sigma_pixel=SIGMA_PX,
        min_track_length=1,
        init_depth_var=0.01,
        max_depth=8000.0,
        mahalanobis_reset_chi2=1e9,
        second_order_mode="analytic",
        range_walk_var=range_walk_var,
    )
    alln = np.full((N_MC, N), np.nan)
    for mc in range(N_MC):
        rng = np.random.default_rng(7000 + mc)
        filt = Sparse3DFilter.polar3d(FX, FY, CX, CY, **settings)
        for i in range(N):
            t_true = np.eye(4)
            t_true[0, 3] = i * BASELINE
            t_cw = np.linalg.inv(t_true)
            pc = t_cw[:3, :3] @ P_W + t_cw[:3, 3]
            uv = (
                FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
                FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX),
            )
            xi = np.concatenate([rng.normal(0, SIGMA_T, 3), np.zeros(3)])
            t_fed = t_true @ exp_se3(xi)
            filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg)
            feats = filt.get_features()
            if 42 not in feats:
                continue
            est = np.asarray(feats[42]["position"])
            cov = np.asarray(feats[42]["covariance_euclidean"])
            r_fed = t_fed[:3, :3]
            ew = r_fed @ est + t_fed[:3, 3] - P_W
            cw = r_fed @ cov @ r_fed.T
            try:
                alln[mc, i] = float(ew @ np.linalg.solve(cw, ew))
            except np.linalg.LinAlgError:
                pass
    return float(np.nanmean(np.nanmean(alln, axis=0)[N // 2:]))


def main():
    print(
        f"settled NEES, Analytic, Z={Z:.0f}, translation pose-noise σ_t={SIGMA_T} "
        f"(N={N}, N_MC={N_MC}). ideal=3"
    )
    print("rows = p_vv scale (x 2σ_t²/dt²),  cols = range_walk_var\n")
    print("  scale\\rwv " + "".join(f"{r:>10.0e}" for r in RANGE_WALK_VAR))
    for sc in PVV_SCALE:
        print(f"{sc:9.2f} " + "".join(f"{settled_nees(r, sc):10.2f}" for r in RANGE_WALK_VAR))


if __name__ == "__main__":
    main()
