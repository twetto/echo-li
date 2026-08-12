"""Minimal SE(3) exponential map for pose-noise injection.

Self-contained on purpose: the pose-noise NEES harness must not pull in
``liepp``/``eqvio`` from the pure-Python repo, so we keep this isolated venv
free of that stack. Only numpy is required.

Twist convention: xi = [rho (translation, 3), phi (rotation, 3)], so a 6x6
pose covariance Sigma_pose is ordered [translation-block, rotation-block].
exp(xi) = [[R, V@rho], [0, 1]] with R = exp(phi^) and V the SO(3) left
Jacobian (so that the translation part is exact, not just first order).

SCOPE -- DO NOT GROW THIS INTO A LIE TOOLBOX.
This module is *only* the test-side noise oracle: sampling pose
perturbations to corrupt a fed pose. It is deliberately independent of the
code under test (a shared exp could let a bug cancel itself and pass a wrong
test) and deliberately tiny (~exp/log). The canonical Lie toolbox already
exists in Rust (`echo-lie`: SO3/SE3/SOT3/...). The moment a test needs group
composition, adjoints, or chart conversions, that is system-under-test math
-> expose `echo-lie` to Python and use it, never add it here.
"""

from __future__ import annotations

import numpy as np


def hat_so3(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector."""
    w = np.asarray(w, dtype=float)
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential via Rodrigues' formula."""
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + hat_so3(phi)
    k = phi / theta
    kx = hat_so3(k)
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian V, used for the translation part of SE(3) exp."""
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + 0.5 * hat_so3(phi)
    k = phi / theta
    kx = hat_so3(k)
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta * kx
        + (1.0 - np.sin(theta) / theta) * (kx @ kx)
    )


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential. xi = [rho(3) translation, phi(3) rotation] -> 4x4."""
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    R = exp_so3(phi)
    V = left_jacobian_so3(phi)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T
