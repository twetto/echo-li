"""Cross-checks for the test-side SE(3) oracle (_se3.py).

These pin _se3 against *independent* implementations (scipy's rotation vector
and the generic matrix exponential), so the noise oracle can't silently drift.
This is intentionally not testing any echo-li Rust code -- it validates the
pure-numpy perturbation machinery the NEES harness relies on.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from _se3 import exp_se3, exp_so3, hat_so3, left_jacobian_so3


def _rng(seed):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("seed", range(8))
def test_exp_so3_matches_scipy_rotvec(seed):
    """exp_so3(phi) == Rotation.from_rotvec(phi) for generic angles."""
    phi = _rng(seed).standard_normal(3) * (0.1 + seed * 0.3)
    R_ours = exp_so3(phi)
    R_scipy = Rotation.from_rotvec(phi).as_matrix()
    assert np.allclose(R_ours, R_scipy, atol=1e-12)
    # and it's a proper rotation
    assert np.allclose(R_ours @ R_ours.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R_ours), 1.0, atol=1e-12)


def test_exp_so3_zero_and_small_angle():
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3), atol=1e-15)
    phi = np.array([1e-10, -2e-10, 5e-11])
    assert np.allclose(exp_so3(phi), Rotation.from_rotvec(phi).as_matrix(), atol=1e-12)


def _se3_wedge(xi):
    """4x4 se(3) algebra element for xi = [rho(3), phi(3)]."""
    rho, phi = xi[:3], xi[3:]
    m = np.zeros((4, 4))
    m[:3, :3] = hat_so3(phi)
    m[:3, 3] = rho
    return m


@pytest.mark.parametrize("seed", range(8))
def test_exp_se3_matches_matrix_exponential(seed):
    """exp_se3(xi) == expm(wedge(xi)) -- validates R and the V-matrix at once."""
    rng = _rng(100 + seed)
    xi = np.concatenate(
        [rng.standard_normal(3) * 0.5, rng.standard_normal(3) * (0.1 + seed * 0.2)]
    )
    T_ours = exp_se3(xi)
    T_ref = expm(_se3_wedge(xi))
    assert np.allclose(T_ours, T_ref, atol=1e-10)
    assert np.allclose(T_ours[3, :], [0, 0, 0, 1], atol=1e-15)


def test_exp_se3_pure_translation():
    """Zero rotation -> V = I, translation passes through unchanged."""
    xi = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
    T = exp_se3(xi)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-15)
    assert np.allclose(T[:3, 3], [1.0, -2.0, 3.0], atol=1e-15)


def test_left_jacobian_small_angle_is_identity():
    assert np.allclose(left_jacobian_so3(np.zeros(3)), np.eye(3), atol=1e-15)
