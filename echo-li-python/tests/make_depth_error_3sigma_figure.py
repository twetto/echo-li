"""Range error vs ±3σ bound for the Sparse3D IEKF, one panel per depth (EqVIO-style).

Companion to make_depth_nees_curves_figure.py (same per-depth layout, same
constant-speed run-to-half-image setup). Instead of NEES this plots the classic
filter-consistency view: the actual range (depth) error trajectories together
with the ±3σ envelope derived from the filter's own reported covariance. A
consistent filter keeps ~99.7% of the error inside ±3σ; transient overconfidence
shows up as the error poking outside the (too-tight) envelope.

Range error e_r = ||est|| - ||true||; σ_r = sqrt(uᵀ Σ_euc u), u = est/||est||
(the radial direction), Σ_euc = covariance_euclidean.

Run from the REPO ROOT (so the installed wheel imports, not the source dir):
    echo-li-python/.venv/bin/python echo-li-python/tests/make_depth_error_3sigma_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from echo_li import Sparse3DFilter

FX = FY = 458.0
CX, CY = 376.0, 240.0
IMG_W = 752
U0, V0 = 400.0, 250.0
DT = 0.05
BASELINE = 0.05  # m/frame, constant speed
SIGMA_PX = 0.5
N_MC = 25
TARGET_PX = IMG_W / 2.0
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
    return int(round(TARGET_PX * z / (FX * BASELINE)))


def run(z, n_steps):
    """Return (err[N_MC, n], sigma[N_MC, n]) of range error and reported range std."""
    p_w = np.array([(U0 - CX) / FX * z, (V0 - CY) / FY * z, z])
    err = np.full((N_MC, n_steps), np.nan)
    sig = np.full((N_MC, n_steps), np.nan)
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
                r_est = np.linalg.norm(est)
                u = est / r_est
                err[mc, i] = r_est - np.linalg.norm(pc)
                sig[mc, i] = np.sqrt(max(float(u @ cov @ u), 0.0))
    return err, sig


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, z in zip(axes.ravel(), Z_VALUES):
        n = steps_for_half_image(z)
        err, sig = run(z, n)
        steps = np.arange(n) + 1
        for mc in range(N_MC):
            ax.plot(steps, err[mc], color="tab:blue", lw=0.4, alpha=0.35)
        s3 = 3.0 * np.nanmedian(sig, axis=0)
        ax.plot(steps, s3, color="k", ls="--", lw=1.2, label="±3σ (reported)")
        ax.plot(steps, -s3, color="k", ls="--", lw=1.2)
        ax.axhline(0.0, color="gray", lw=0.6)
        # focus the y-axis on the post-transient regime
        tail = err[:, n // 4 :]
        lim = max(np.nanpercentile(np.abs(tail), 99.5), s3[n // 4 :].max()) * 1.3
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("observation step")
        ax.set_ylabel("range error [m]")
        ax.set_title(f"Z = {int(z)} m  ({n} steps, ~half image)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Sparse3D IEKF range error vs ±3σ per depth  |  sigma_px={SIGMA_PX}, "
        f"N_MC={N_MC} (blue), constant speed",
        fontsize=12,
    )
    out = "/tmp/depth_error_3sigma.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
