"""Altitude sweep of pose-noise NEES for the Rust Sparse3DFilter.

Companion to make_pose_noise_nees_figure.py. That figure fixes the depth; this
one sweeps it to expose the depth-scaling asymmetry behind the 6x6 argument:

  - translation pose error projects as ~ (f/Z) * dt  -> parallax-dependent,
    shrinks with altitude, captured by the 3x3 velocity covariance.
  - rotation pose error projects as ~ f * dphi       -> depth-INDEPENDENT, so
    its metric impact (Z * dphi) grows with altitude and cannot be modelled by
    any translation-only 3x3.

IMPORTANT (observability control): triangulation quality depends on how far the
landmark travels in the image, not on the depth per se. With a fixed frame
count a far landmark barely moves (e.g. ~6 px at Z=640 m), so it is essentially
un-triangulated and the result is dominated by weak parallax rather than the
pose-noise effect. We therefore run EACH depth long enough that the landmark
sweeps the same pixel travel (TARGET_PX, ~half the image width). At constant
speed that means N_z ~ Z (farther landmarks are observed for longer).

Run from the repo root (so the installed wheel imports, not the source dir):
    echo-li-python/.venv/bin/python echo-li-python/tests/make_altitude_sweep_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _se3 import exp_se3  # noqa: E402

from echo_li import Sparse3DFilter  # noqa: E402

FX = FY = 458.0
CX, CY = 376.0, 240.0
IMG_W = 752
U0, V0 = 400.0, 250.0
BASELINE = 0.05      # m/frame (constant speed)
DT = 0.05
TARGET_PX = IMG_W / 2.0  # each landmark sweeps >= half the image
N_MC = 30
DIM = 3

SIGMA_PX = 0.5
SIGMA_T = 0.01
SIGMA_PHI = 0.0015
P_VV_T = (2.0 * SIGMA_T**2 / DT**2) * np.eye(3)

Z_VALUES = [20.0, 40.0, 80.0, 160.0, 320.0, 640.0]

SETTINGS = dict(
    sigma_pixel=SIGMA_PX,
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=2000.0,
    mahalanobis_reset_chi2=1e9,
    process_depth_var=0.0,
)


def frames_for(z):
    """Frame count so the landmark sweeps TARGET_PX of image travel at depth z."""
    return int(round(TARGET_PX * z / (FX * BASELINE)))


def run_trial(rng, z, n_frames, pose_noise, p_vv):
    p_w = np.array([(U0 - CX) / FX * z, (V0 - CY) / FY * z, z])
    filt = Sparse3DFilter.polar3d(FX, FY, CX, CY, **SETTINGS)
    p_vv_arg = None if p_vv is None else p_vv.tolist()
    settle = max(40, n_frames // 4)
    acc, cnt = 0.0, 0

    for i in range(n_frames):
        t_true = np.eye(4)
        t_true[0, 3] = i * BASELINE
        t_cw = np.linalg.inv(t_true)
        pc = t_cw[:3, :3] @ p_w + t_cw[:3, 3]
        uv = (
            FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
            FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX),
        )
        t_fed = t_true @ exp_se3(pose_noise(rng))
        filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg)

        if i < n_frames - settle:
            continue
        feats = filt.get_features()
        if 42 not in feats:
            continue
        fd = feats[42]
        est = np.asarray(fd["position"])
        cov_euc = np.asarray(fd["covariance_euclidean"])
        r_fed = t_fed[:3, :3]
        err = (r_fed @ est + t_fed[:3, 3]) - p_w
        cov_world = r_fed @ cov_euc @ r_fed.T
        try:
            acc += float(err @ np.linalg.solve(cov_world, err))
            cnt += 1
        except np.linalg.LinAlgError:
            pass
    return acc / cnt if cnt else np.nan


def no_noise(rng):
    return np.zeros(6)


def trans_noise(rng):
    return np.concatenate([rng.normal(0, SIGMA_T, 3), np.zeros(3)])


def rot_noise(rng):
    return np.concatenate([np.zeros(3), rng.normal(0, SIGMA_PHI, 3)])


ARMS = [
    ("no pose noise", no_noise, None, "tab:green", "o"),
    ("translation noise + p_vv (3x3)", trans_noise, P_VV_T, "tab:blue", "s"),
    ("rotation noise + p_vv (3x3)", rot_noise, P_VV_T, "tab:purple", "D"),
]


def main():
    n_by_z = {z: frames_for(z) for z in Z_VALUES}
    print(f"altitude sweep @ equal pixel travel ~{TARGET_PX:.0f}px (dim={DIM}, N_MC={N_MC})")
    print("  frames/Z: " + "  ".join(f"Z={z:.0f}:{n}" for z, n in n_by_z.items()))

    results = {}
    for label, noise, p_vv, color, marker in ARMS:
        vals = []
        for z in Z_VALUES:
            trials = [run_trial(np.random.default_rng(9000 + mc), z, n_by_z[z], noise, p_vv)
                      for mc in range(N_MC)]
            vals.append(np.nanmean(trials))
        results[label] = np.array(vals)
        print(f"  {label:34s} " + "  ".join(f"Z={z:>5.0f}:{v:8.2f}"
                                            for z, v in zip(Z_VALUES, vals)))

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(15, 6))

    for label, _, _, color, marker in ARMS:
        ax.plot(Z_VALUES, results[label], color=color, marker=marker, lw=1.8, label=label)
    ax.axhline(DIM, color="k", ls=":", lw=1.0, label=f"E[NEES]={DIM} (calibrated)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Z_VALUES)
    ax.set_xticklabels([f"{int(z)}" for z in Z_VALUES])
    ax.set_xlabel("scene depth Z [m]  (each landmark sweeps ~half the image)")
    ax.set_ylabel("settled mean NEES (log scale)")
    ax.set_title("Settled depth NEES vs altitude\n(equal pixel travel across Z)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    ratio = results["rotation noise + p_vv (3x3)"] / results["translation noise + p_vv (3x3)"]
    axr.plot(Z_VALUES, ratio, color="tab:purple", marker="D", lw=2.0)
    axr.axhline(1.0, color="k", ls=":", lw=1.0, label="equal (3x3 handles both)")
    axr.set_xscale("log")
    axr.set_xticks(Z_VALUES)
    axr.set_xticklabels([f"{int(z)}" for z in Z_VALUES])
    axr.set_xlabel("scene depth Z [m]")
    axr.set_ylabel("NEES ratio:  rotation+p_vv / translation+p_vv")
    axr.set_title("Same 3x3 p_vv, rotation vs translation\n(gap vs altitude at equal observability)")
    axr.grid(True, which="both", alpha=0.3)
    axr.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"sigma_px={SIGMA_PX}, sigma_t={SIGMA_T} m, sigma_phi={SIGMA_PHI} rad, "
        f"baseline={BASELINE} m/frame, N_MC={N_MC}",
        fontsize=10,
    )
    out = "/tmp/altitude_sweep_nees.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
