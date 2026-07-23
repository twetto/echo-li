from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _se3 import exp_se3  # noqa: E402
from echo_li import Sparse3DFilter  # noqa: E402

FX = FY = 458.0
CX, CY = 376.0, 240.0
DT = 0.05
N = int(os.environ.get("N_STEPS", "160"))
N_MC = int(os.environ.get("N_MC", "20"))
SIGMA_PX = 0.5
SIGMA_T = 0.01
SIGMA_PHI = 0.0015
Z0 = float(os.environ.get("Z_TRUE", "80.0"))
FORWARD_FRAC = float(os.environ.get("FORWARD_FRAC", "0.16"))
DZ = float(os.environ.get("FORWARD_DZ", str(FORWARD_FRAC * Z0 / max(N - 1, 1))))
YAW_STEP = float(os.environ.get("YAW_STEP", "0.0006"))

P_W = np.array([(410.0 - CX) / FX * Z0, (252.0 - CY) / FY * Z0, Z0])
P_VV = (SIGMA_T**2) * np.eye(3)
P_WW = (SIGMA_PHI**2) * np.eye(3)
SETTINGS = dict(
    sigma_pixel=SIGMA_PX,
    min_track_length=1,
    init_depth_var=0.01,
    max_depth=500.0,
    mahalanobis_reset_chi2=1e9,
    process_depth_var=0.0,
)


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def run_trial(rng, pose_noise: bool, feed_trans_cov: bool, feed_rot_cov: bool) -> np.ndarray:
    filt = Sparse3DFilter.invdepth_additive3d(FX, FY, CX, CY, **SETTINGS)
    p_vv_arg = P_VV.tolist() if feed_trans_cov else None
    p_ww_arg = P_WW.tolist() if feed_rot_cov else None
    nees = np.full(N, np.nan)
    for i in range(N):
        t_true = np.eye(4)
        t_true[:3, :3] = rot_y(i * YAW_STEP)
        t_true[2, 3] = i * DZ
        t_cw = np.linalg.inv(t_true)
        pc = t_cw[:3, :3] @ P_W + t_cw[:3, 3]
        if pc[2] <= 0.5:
            break
        uv = (
            FX * pc[0] / pc[2] + CX + rng.normal(0, SIGMA_PX),
            FY * pc[1] / pc[2] + CY + rng.normal(0, SIGMA_PX),
        )
        if pose_noise:
            xi = np.concatenate([rng.normal(0, SIGMA_T, 3), rng.normal(0, SIGMA_PHI, 3)])
        else:
            xi = np.zeros(6)
        t_fed = t_true @ exp_se3(xi)
        filt.update(i * DT, {42: uv}, t_fed.tolist(), p_vv_arg, p_ww_arg)
        feats = filt.get_features()
        if 42 not in feats:
            continue
        fd = feats[42]
        est = np.asarray(fd["position"])
        cov = np.asarray(fd["covariance_euclidean"])
        r_fed = t_fed[:3, :3]
        est_world = r_fed @ est + t_fed[:3, 3]
        cov_world = r_fed @ cov @ r_fed.T
        err = est_world - P_W
        try:
            nees[i] = float(err @ np.linalg.solve(cov_world, err))
        except np.linalg.LinAlgError:
            pass
    return nees


def summarize(label: str, curves: np.ndarray) -> None:
    mean_curve = np.nanmean(curves, axis=0)
    cells = []
    for start in range(0, N, 50):
        cells.append(f"{np.nanmean(mean_curve[start:start+50]):7.2f}")
    print(f"  {label:28s}{''.join(cells)}")


print(
    f"Forward+rotation NEES (N_MC={N_MC}, N={N}, Z0={Z0}, dz={DZ}, "
    f"yaw_step={YAW_STEP}); ideal 3"
)
print("  " + "arm".ljust(28) + "".join(f"{s+50:>7}" for s in range(0, N, 50)))
for label, pose_noise, feed_trans_cov, feed_rot_cov in [
    ("no pose noise", False, False, False),
    ("full pose cov", True, True, True),
    ("translation cov only", True, True, False),
]:
    alln = np.full((N_MC, N), np.nan)
    for mc in range(N_MC):
        alln[mc] = run_trial(np.random.default_rng(9000 + mc), pose_noise, feed_trans_cov, feed_rot_cov)
    summarize(label, alln)
