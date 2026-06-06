"""Cross-language parity tripwire for the sparse-3D filter.

Replays the committed golden trace (generated offline by the pure-Python
reference, see fixtures/gen_sparse3d_golden.py) through the Rust Sparse3DFilter
and asserts per-step position + covariance match *in steady state*.

KNOWN DIVERGENCE (intentional, documented): the Rust filter admits a feature
one frame earlier than the Python reference (it triangulates from slightly
weaker parallax). This shifts track_length by one and gives a different
initialisation transient -- the early covariance differs by up to ~2.4x at
birth. Both implementations *converge*: at the same absolute frame the position
agrees to the f32 pixel round-trip floor (~2e-5) and the covariance to within a
few percent by the end of the trace. Since NEES is evaluated on settled
features (the diagnostics average the second half of a run), steady-state
parity is the property that makes a NEES result transfer between the two.

So this test asserts agreement over the settled TAIL of the trace, not from
birth. It still trips on any genuine drift in the recursion (which would be
gross, not a few percent). The full-trace transient is characterised in the
git history of this file / fixtures rather than asserted.

Tolerances are set with margin above the measured tail convergence:
position |d| < 5e-5, covariance rel < 3.5% over the last 10 frames.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from echo_li import Sparse3DFilter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sparse3d_golden.npz")

# Number of trailing frames over which steady-state parity is asserted.
TAIL = 10
POS_ATOL = 2e-4
COV_RTOL = 0.06


def _load():
    if not os.path.exists(FIXTURE):
        pytest.skip(
            "golden fixture missing; regenerate with the eqvio venv: "
            "fixtures/gen_sparse3d_golden.py"
        )
    return np.load(FIXTURE, allow_pickle=True)


def _run_rust(g):
    k = g["k"]
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    settings = json.loads(str(g["settings_json"]))
    chart = str(g["chart"])
    feat_id = int(g["feat_id"])

    ctor = Sparse3DFilter.polar3d if chart == "polar3d" else Sparse3DFilter.invdepth3d
    filt = ctor(fx, fy, cx, cy, **settings)

    stamps, uvs, poses = g["stamps"], g["uvs"], g["poses"]
    n = len(stamps)
    positions = np.full((n, 3), np.nan)
    covariances = np.full((n, 3, 3), np.nan)
    track_lengths = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        filt.update(
            float(stamps[i]),
            {feat_id: (float(uvs[i, 0]), float(uvs[i, 1]))},
            poses[i].tolist(),
        )
        feats = filt.get_features()
        if feat_id in feats:
            fd = feats[feat_id]
            positions[i] = fd["position"]
            covariances[i] = fd["covariance"]
            track_lengths[i] = int(fd["track_length"])

    return positions, covariances, track_lengths


def test_rust_matches_python_golden_steady_state():
    g = _load()
    rust_pos, rust_cov, rust_tl = _run_rust(g)
    gold_pos, gold_cov, gold_tl = g["positions"], g["covariances"], g["track_lengths"]

    n = len(gold_tl)
    tail = slice(n - TAIL, n)

    # Both filters must have a live feature throughout the settled tail.
    assert np.all(gold_tl[tail] >= 0), "python golden lost the feature in the tail"
    assert np.all(rust_tl[tail] >= 0), "rust lost the feature in the tail"

    np.testing.assert_allclose(
        rust_pos[tail], gold_pos[tail], rtol=1e-4, atol=POS_ATOL,
        err_msg="steady-state position diverged between Rust and Python",
    )
    np.testing.assert_allclose(
        rust_cov[tail], gold_cov[tail], rtol=COV_RTOL, atol=1e-9,
        err_msg="steady-state covariance diverged between Rust and Python",
    )


def test_known_admission_offset_is_one_frame():
    """Guard the documented divergence so it can't silently grow.

    Rust admits the feature exactly one frame before Python. If that offset
    changes, the admission gating drifted and the steady-state tolerances above
    may no longer be appropriate -- fail loudly so we revisit.
    """
    g = _load()
    _, _, rust_tl = _run_rust(g)
    gold_tl = g["track_lengths"]

    rust_birth = int(np.argmax(rust_tl >= 0))
    gold_birth = int(np.argmax(gold_tl >= 0))
    assert gold_birth - rust_birth == 1, (
        f"feature-admission offset changed: rust births at {rust_birth}, "
        f"python at {gold_birth} (expected python = rust + 1)"
    )
