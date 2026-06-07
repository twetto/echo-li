"""Per-timestep NEES trajectory of the Sparse3D IEKF, one panel per depth.

Realistic constant-speed flight (fixed per-frame baseline). Each depth is run for
as many steps as it takes the landmark to sweep ~half the image, so even the
far landmarks accumulate enough parallax to converge -- the point being that the
NEES settles to the ideal once the geometry is informative, regardless of depth.

3D Euclidean NEES (dim = 3, ideal = 3), mean over Monte-Carlo trials at every
step, scored in the current camera frame via covariance_euclidean. Dotted line =
ideal; shaded band = 95% interval for the mean of N_MC chi-square(3) samples.

Run from the REPO ROOT (so the installed wheel imports, not the source dir):
    echo-li-python/.venv/bin/python echo-li-python/tests/make_depth_nees_curves_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from echo_li import Sparse3DFilter

FX = FY = 458.0
CX, CY = 376.0, 240.0
IMG_W = 752
U0, V0 = 400.0, 250.0
DT = 0.05
BASELINE = 0.05  # m/frame, constant speed
SIGMA_PX = 0.5
N_MC = 40
DIM = 3
TARGET_PX = IMG_W / 2.0  # each landmark sweeps ~half the image
Z_VALUES = [80.0, 320.0, 640.0, 1280.0]

SETTINGS = dict(
    sigma_pixel=SIGMA_PX,
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=8000.0,
    mahalanobis_reset_chi2=1e9,
    process_depth_var=0.0,
)


def steps_for_half_image(z):
    """Frames so the landmark sweeps TARGET_PX at constant baseline."""
    return int(round(TARGET_PX * z / (FX * BASELINE)))


def nees_curve(z, n_steps):
    p_w = np.array([(U0 - CX) / FX * z, (V0 - CY) / FY * z, z])
    alln = np.full((N_MC, n_steps), np.nan)
    for mc in range(N_MC):
        rng = np.random.default_rng(11 + mc)
        filt = Sparse3DFilter.polar3d(FX, FY, CX, CY, **SETTINGS)
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
                est = np.asarray(feats[42]["position"])
                cov = np.asarray(feats[42]["covariance_euclidean"])
                e = est - pc
                try:
                    alln[mc, i] = float(e @ np.linalg.solve(cov, e))
                except np.linalg.LinAlgError:
                    pass
    return np.nanmean(alln, axis=0)


def main():
    lo = chi2.ppf(0.025, DIM * N_MC) / N_MC
    hi = chi2.ppf(0.975, DIM * N_MC) / N_MC
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, z in zip(axes.ravel(), Z_VALUES):
        n = steps_for_half_image(z)
        m = nees_curve(z, n)
        steps = np.arange(n) + 1
        ax.plot(steps, m, color="tab:blue", lw=1.3)
        ax.axhline(DIM, color="k", ls=":", lw=1.0, label=f"ideal = {DIM}")
        ax.axhspan(lo, hi, color="gray", alpha=0.15, label="95% band")
        ax.set_yscale("log")
        ax.set_xlabel("observation step")
        ax.set_ylabel("mean 3D NEES (log)")
        ax.set_title(f"Z = {int(z)} m  ({n} steps to sweep ~half image)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Sparse3D IEKF NEES per depth (constant speed, run to half-image travel)  "
        f"|  sigma_px={SIGMA_PX}, N_MC={N_MC}",
        fontsize=12,
    )
    out = "/tmp/depth_nees_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    for z in Z_VALUES:
        print(f"  Z={z:6.0f} -> {steps_for_half_image(z)} steps")


if __name__ == "__main__":
    main()
