"""Sparse3D IEKF: range random-walk floor (`range_walk_var`) vs depth.

Tests the hypothesis that the depth-growing converged-tail NEES of the Sparse3D
IEKF is an *accumulated triangulation/range bias* (a covariance that has
collapsed below a deterministic error), not a per-step covariance fault: a tiny
per-step radial process-noise floor should stop Σ collapsing and flatten NEES
across depth.

Finding:
  range_walk_var = 0      -> tail NEES 4.4 / 3.9 / 5.4  (Z = 320/640/1280, grows)
  range_walk_var = 1e-9   -> tail NEES 1.9 / 1.7 / 2.0  (depth-FLAT, mild under-conf)
  range_walk_var >~ 1e-6  -> destabilises (the floor is ACCUMULATING; too large
                            forces single-frame triangulation -> NEES explodes).
The `‖q_c‖²` scaling matches the bias' `∝ depth` growth, so one constant
calibrates every depth. ~3e-10 centres the tail on 3.

Caveat: the floor accumulates each step, so its sweet spot is track-length
sensitive; a Σ eigenvalue lower-bound clamp would be the track-length-insensitive
alternative (same calibration, no instability cliff) -- not yet implemented.

Run from the REPO ROOT (so the installed wheel imports, not the source dir):
    echo-li-python/venv/bin/python echo-li-python/tests/make_process_floor_sweep.py
"""

from __future__ import annotations

import numpy as np

from echo_li import Sparse3DFilter

FX = FY = 458.0
CX, CY = 376.0, 240.0
IMG_W = 752
U0, V0 = 400.0, 250.0
DT = 0.05
BASELINE = 0.05
SIGMA_PX = 0.5
N_MC = 12
TARGET_PX = IMG_W / 2.0

Z_VALUES = [320.0, 640.0, 1280.0]
RANGE_WALK_VAR = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]

BASE = dict(
    sigma_pixel=SIGMA_PX,
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=8000.0,
    mahalanobis_reset_chi2=1e9,
    second_order_mode="analytic",
)


def steps_for_half_image(z):
    return int(round(TARGET_PX * z / (FX * BASELINE)))


def tail_nees(z, n_steps, settings):
    p_w = np.array([(U0 - CX) / FX * z, (V0 - CY) / FY * z, z])
    alln = np.full((N_MC, n_steps), np.nan)
    for mc in range(N_MC):
        rng = np.random.default_rng(11 + mc)
        filt = Sparse3DFilter.polar3d(FX, FY, CX, CY, **settings)
        for i in range(n_steps):
            t_wc = np.eye(4)
            t_wc[0, 3] = i * BASELINE
            t_cw = np.linalg.inv(t_wc)
            pc = t_cw[:3, :3] @ p_w + t_cw[:3, 3]
            uv = (
                FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
                FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX),
            )
            filt.update(i * DT, {42: uv}, t_wc.tolist(), None)
            feats = filt.get_features()
            if 42 in feats:
                e = np.asarray(feats[42]["position"]) - pc
                cov = np.asarray(feats[42]["covariance_euclidean"])
                try:
                    alln[mc, i] = float(e @ np.linalg.solve(cov, e))
                except np.linalg.LinAlgError:
                    pass
    curve = np.nanmean(alln, axis=0)
    return float(np.nanmean(curve[int(0.8 * n_steps):]))


def main():
    print(f"converged-tail NEES, Analytic 2nd-order, no pose noise (N_MC={N_MC}). ideal=3")
    print("range_walk_var (rows) x Z (cols)\n")
    print("   rwv \\ Z " + "".join(f"{z:>9.0f}" for z in Z_VALUES))
    for rwv in RANGE_WALK_VAR:
        row = []
        for z in Z_VALUES:
            n = steps_for_half_image(z)
            row.append(tail_nees(z, n, dict(BASE, range_walk_var=rwv)))
        print(f"{rwv:9.0e} " + "".join(f"{v:9.2f}" for v in row))


if __name__ == "__main__":
    main()
