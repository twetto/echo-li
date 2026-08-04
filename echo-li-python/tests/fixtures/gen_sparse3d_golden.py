"""Generate the sparse-3D golden fixture from the pure-Python reference filter.

This is the cross-language contract for test_parity_sparse3d.py. It runs a
fixed, NOISE-FREE (T_WC, uv) trace through eqvio's SparseVogiatzisFilter3D
(polar chart) and dumps the per-step (position, covariance) it produces,
together with the exact input trace and the pinned settings, so the Rust
Sparse3DFilter can replay byte-identical input and be asserted against it.

RUN FROM THE eqvio VENV (it imports eqvio/liepp/cv2), e.g.:
    ../ECHO-LI-python/.venv/bin/python \
        echo-li-python/tests/fixtures/gen_sparse3d_golden.py

Regenerate only when the Python reference deliberately changes; the diff to
sparse3d_golden.npz is then a visible, reviewable record of that change.
"""

from __future__ import annotations

import json
import os

import numpy as np

from eqvio.sparse_vogiatzis import SparseVogiatzisFilter3D, SparseVogSettings
from eqvio.mathematical.vision_measurement import VisionMeasurement

# --- camera + trace geometry (noise-free, pure x-translation) ---------------
FX = FY = 458.0
CX, CY = 376.0, 240.0
Z_TRUE = 80.0  # safely inside both filters' max_depth gate
U0, V0 = 400.0, 250.0
BASELINE = 0.05  # metres per frame
DT = 0.05
N_FRAMES = 40
FEAT_ID = 42

# --- settings pinned identically on both sides ------------------------------
# Rust and Python defaults have diverged (init_depth_var, ab_max,
# min_inlier_ratio, max_depth), so every trajectory-affecting knob is set
# explicitly here and replayed on the Rust side via the same kwarg names.
SETTINGS = {
    "max_pool_size": 300,
    "min_track_length": 1,
    "conv_inlier_ratio": 0.7,
    "conv_variance_threshold": 0.5,
    "init_depth_var": 0.01,
    "sigma_pixel": 0.5,
    "uniform_z_max": 20.0,
    "uniform_rho_max": 10.0,
    "uniform_d_min": -5.0,
    "uniform_d_max": 5.0,
    "a_init": 10.0,
    "b_init": 2.0,
    "ab_min": 1.0,
    "ab_max": 100.0,
    "min_inlier_ratio": 0.3,
    "mahalanobis_reset_chi2": 9.0,
    "process_depth_var": 0.01,
    "min_parallax": 1e-4,
    "min_cos_sim": 0.95,
    "min_depth": 0.1,
    "max_depth": 150.0,
    "birth_min_flow_px": 3.0,
}

K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def build_trace():
    """Return stamps (N,), uvs (N,2), poses (N,4,4) for the noise-free track."""
    x_n = (U0 - CX) / FX
    y_n = (V0 - CY) / FY
    p_w = np.array([x_n * Z_TRUE, y_n * Z_TRUE, Z_TRUE])

    stamps = np.zeros(N_FRAMES)
    uvs = np.zeros((N_FRAMES, 2))
    poses = np.zeros((N_FRAMES, 4, 4))
    for i in range(N_FRAMES):
        t_wc = np.eye(4)
        t_wc[0, 3] = i * BASELINE
        t_cw = np.linalg.inv(t_wc)
        p_c = t_cw[:3, :3] @ p_w + t_cw[:3, 3]
        stamps[i] = i * DT
        uvs[i] = [FX * p_c[0] / p_c[2] + CX, FY * p_c[1] / p_c[2] + CY]
        poses[i] = t_wc
    return stamps, uvs, poses


def main():
    stamps, uvs, poses = build_trace()
    settings = SparseVogSettings(**SETTINGS)
    filt = SparseVogiatzisFilter3D(K, settings)

    positions = np.full((N_FRAMES, 3), np.nan)
    covariances = np.full((N_FRAMES, 3, 3), np.nan)
    track_lengths = np.full(N_FRAMES, -1, dtype=np.int64)

    for i in range(N_FRAMES):
        meas = VisionMeasurement(stamp=float(stamps[i]), cam_coordinates={FEAT_ID: uvs[i]})
        filt.update(meas, poses[i])
        feat = filt.features.get(FEAT_ID)
        if feat is not None:
            positions[i] = np.asarray(feat.position, dtype=float)
            covariances[i] = np.asarray(feat.covariance, dtype=float)
            track_lengths[i] = int(feat.track_length)

    n_valid = int(np.sum(track_lengths >= 0))
    print(f"generated {N_FRAMES} steps, {n_valid} with a live feature")
    if n_valid == 0:
        raise SystemExit("no feature ever materialised -- check trace/settings")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse3d_golden.npz")
    np.savez(
        out,
        chart=np.array("polar3d"),
        feat_id=np.array(FEAT_ID),
        k=K,
        settings_json=np.array(json.dumps(SETTINGS)),
        stamps=stamps,
        uvs=uvs,
        poses=poses,
        positions=positions,
        covariances=covariances,
        track_lengths=track_lengths,
    )
    print(f"wrote {out}")
    # quick sanity: final depth should be near Z_TRUE
    last = np.where(track_lengths >= 0)[0][-1]
    print(f"final position={positions[last]}  (Z_true={Z_TRUE})  track_len={track_lengths[last]}")


if __name__ == "__main__":
    main()
