"""Dense depth mapper using DIS optical flow + two-ray triangulation.

Drop-in replacement for ``echo_li.PatchDepthMapper``.  Computes
``cv2.DISOpticalFlow`` between the current frame and the best keyframe,
triangulates per-pixel depth from the flow correspondences and the known
relative pose, and outputs ``{eta, eta_var, status}`` in the same format
the occupancy map and Rerun visualisation expect.

Camera model: Kannala-Brandt equidistant (same as VOXL2 tracking camera).
All heavy work is vectorised NumPy or OpenCV C++.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Equidistant (Kannala-Brandt) fisheye helpers — vectorised NumPy
# ---------------------------------------------------------------------------

def _equidistant_undistort(mx: np.ndarray, my: np.ndarray,
                           k1: float, k2: float, k3: float, k4: float,
                           n_iters: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undistort normalised coords → unit bearing vectors (bx, by, bz).

    Parameters
    ----------
    mx, my : (H, W) normalised pixel coords ``(u - cx) / fx``.
    k1-k4  : equidistant distortion coefficients.

    Returns
    -------
    bx, by, bz : (H, W) unit-bearing components.
    """
    r = np.sqrt(mx * mx + my * my)
    # theta_d == r in normalised coords; solve f(theta) = theta_d for theta.
    theta = r.copy()
    for _ in range(n_iters):
        th2 = theta * theta
        th4 = th2 * th2
        th6 = th4 * th2
        th8 = th4 * th4
        f = theta + k1 * th2 * theta + k2 * th4 * theta + k3 * th6 * theta + k4 * th8 * theta - r
        fp = 1.0 + 3.0 * k1 * th2 + 5.0 * k2 * th4 + 7.0 * k3 * th6 + 9.0 * k4 * th8
        theta -= f / np.where(fp > 1e-12, fp, 1e-12)
        theta = np.clip(theta, 0.0, np.pi)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # Direction in the tangent plane is (mx/r, my/r).
    safe_r = np.where(r > 1e-12, r, 1.0)
    bx = sin_theta * (mx / safe_r)
    by = sin_theta * (my / safe_r)
    bz = cos_theta
    # Where r ≈ 0 the bearing is straight ahead.
    on_axis = r < 1e-12
    bx = np.where(on_axis, 0.0, bx)
    by = np.where(on_axis, 0.0, by)
    bz = np.where(on_axis, 1.0, bz)
    return bx, by, bz


def _build_bearing_lut(w: int, h: int,
                       fx: float, fy: float, cx: float, cy: float,
                       k1: float, k2: float, k3: float, k4: float
                       ) -> np.ndarray:
    """Build (H, W, 3) unit-bearing LUT for every pixel."""
    us = np.arange(w, dtype=np.float64)
    vs = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    mx = (uu - cx) / fx
    my = (vv - cy) / fy
    bx, by, bz = _equidistant_undistort(mx, my, k1, k2, k3, k4)
    lut = np.stack([bx, by, bz], axis=-1).astype(np.float32)
    return lut


# ---------------------------------------------------------------------------
# Keyframe
# ---------------------------------------------------------------------------

@dataclass
class _Keyframe:
    gray: np.ndarray      # uint8 at working resolution
    t_wc: np.ndarray      # 4×4 float64
    stamp: float


# ---------------------------------------------------------------------------
# DIS Depth Mapper
# ---------------------------------------------------------------------------

class DISDepthMapper:
    """Dense depth from DIS optical flow + two-ray triangulation.

    Interface mirrors ``echo_li.PatchDepthMapper`` so the VIO node can swap
    them without other changes.
    """

    # PatchStatus values matching PatchDepthMapper's output:
    # 0 = Unknown, 1 = SeedOnly, 2 = PhotoRefined, 3 = Rejected.
    STATUS_UNKNOWN = np.uint8(0)
    STATUS_SEED_ONLY = np.uint8(1)
    STATUS_REFINED = np.uint8(2)

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        distortion: list[float],
        *,
        scale: float = 0.25,
        max_depth: float = 12.0,
        min_depth: float = 0.1,
        min_baseline_ratio: float = 0.005,
        max_baseline_ratio: float = 0.3,
        min_parallax_sin2: float = 1e-5,
        sigma_flow_px: float = 1.0,
        max_keyframes: int = 5,
        dis_preset: str = 'medium',
    ):
        self.full_w = width
        self.full_h = height
        self.scale = scale
        self.work_w = int(width * scale)
        self.work_h = int(height * scale)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_baseline_ratio = min_baseline_ratio
        self.max_baseline_ratio = max_baseline_ratio
        self.min_parallax_sin2 = min_parallax_sin2
        self.sigma_flow_px = sigma_flow_px

        # Intrinsics at working resolution.
        self.fx = fx * scale
        self.fy = fy * scale
        self.cx = cx * scale
        self.cy = cy * scale
        self.distortion = list(distortion)

        # Bearing look-up table at working resolution.
        k1, k2, k3, k4 = self.distortion[:4]
        self.bearing_lut = _build_bearing_lut(
            self.work_w, self.work_h,
            self.fx, self.fy, self.cx, self.cy,
            k1, k2, k3, k4)

        # DIS optical flow.
        presets = {
            'ultrafast': cv2.DISOpticalFlow_PRESET_ULTRAFAST,
            'fast': cv2.DISOpticalFlow_PRESET_FAST,
            'medium': cv2.DISOpticalFlow_PRESET_MEDIUM,
        }
        self._dis = cv2.DISOpticalFlow.create(
            presets.get(dis_preset, cv2.DISOpticalFlow_PRESET_MEDIUM))

        # Keyframe pool.
        self._keyframes: deque[_Keyframe] = deque(maxlen=max_keyframes)
        self._frame_count = 0

    # -- public interface (matches PatchDepthMapper) ------------------------

    @property
    def seed_coordinates(self) -> str:
        return 'raw'

    def keyframe_count(self) -> int:
        return len(self._keyframes)

    def update(
        self,
        stamp: float,
        frame_id: int,
        gray: np.ndarray,
        t_wc,
        priors: list[tuple[float, float, float, float]],
    ) -> dict | None:
        """Run one frame.

        Parameters
        ----------
        stamp : float – seconds.
        frame_id : int.
        gray : (H, W) uint8 at full resolution.
        t_wc : 4×4 camera-to-world SE(3), as nested list or ndarray.
        priors : list of (u, v, eta, eta_var) seed priors (full-res pixels).

        Returns
        -------
        dict with ``eta``, ``eta_var``, ``status`` (all (work_H, work_W)
        float32 / uint8), or ``None`` if depth cannot be estimated yet.
        """
        t_wc = np.asarray(t_wc, dtype=np.float64).reshape(4, 4)
        gray_small = cv2.resize(
            gray, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA)

        # Median seed depth for baseline selection.
        if priors:
            median_depth = float(np.median(
                [np.exp(p[2]) for p in priors]))
        else:
            median_depth = 3.0  # fallback

        # Select best keyframe.
        kf = self._select_keyframe(t_wc, median_depth)

        # Always add the current frame as a potential keyframe.
        self._add_keyframe(gray_small, t_wc, stamp, median_depth)

        if kf is None:
            return None

        # Compute DIS flow: current → keyframe.
        # flow[y, x] = (du, dv) such that current(x, y) ≈ kf(x+du, y+dv).
        flow = self._dis.calc(gray_small, kf.gray, None)

        # Triangulate depth.
        eta, eta_var, valid = self._triangulate(flow, t_wc, kf.t_wc)

        # Merge seed priors as fallback for regions where flow fails.
        eta, eta_var, valid = self._merge_seeds(
            eta, eta_var, valid, priors)

        # Build output arrays.
        status = np.where(valid, self.STATUS_REFINED, self.STATUS_UNKNOWN)
        eta = np.where(valid, eta, np.float32(np.nan))
        eta_var = np.where(valid, eta_var, np.float32(np.nan))

        return {
            'eta': eta.astype(np.float32),
            'eta_var': eta_var.astype(np.float32),
            'status': status,
        }

    # -- keyframe management ------------------------------------------------

    def _add_keyframe(self, gray_small: np.ndarray, t_wc: np.ndarray,
                      stamp: float, median_depth: float):
        """Conditionally add a keyframe.  Always adds the first two frames;
        after that, only if the baseline exceeds the minimum."""
        if len(self._keyframes) < 2:
            self._keyframes.append(_Keyframe(gray_small.copy(), t_wc.copy(), stamp))
            return

        newest = self._keyframes[-1]
        baseline = np.linalg.norm(t_wc[:3, 3] - newest.t_wc[:3, 3])
        if baseline > self.min_baseline_ratio * median_depth:
            self._keyframes.append(_Keyframe(gray_small.copy(), t_wc.copy(), stamp))

    def _select_keyframe(
        self, t_wc: np.ndarray, median_depth: float
    ) -> _Keyframe | None:
        """Pick the keyframe with the largest usable baseline."""
        min_bl = self.min_baseline_ratio * median_depth
        max_bl = self.max_baseline_ratio * median_depth
        best: _Keyframe | None = None
        best_bl = 0.0
        for kf in self._keyframes:
            bl = float(np.linalg.norm(t_wc[:3, 3] - kf.t_wc[:3, 3]))
            if bl < min_bl:
                continue
            if bl > max_bl:
                continue
            if bl > best_bl:
                best_bl = bl
                best = kf
        # If nothing in the sweet spot, take the largest baseline even if
        # it exceeds max — clamped triangulation is better than nothing.
        if best is None:
            for kf in self._keyframes:
                bl = float(np.linalg.norm(t_wc[:3, 3] - kf.t_wc[:3, 3]))
                if bl > min_bl and bl > best_bl:
                    best_bl = bl
                    best = kf
        return best

    # -- triangulation ------------------------------------------------------

    def _triangulate(
        self,
        flow: np.ndarray,
        t_wc_curr: np.ndarray,
        t_wc_kf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-ray triangulation from flow correspondences.

        Returns (eta, eta_var, valid) at working resolution.
        """
        H, W = self.work_h, self.work_w

        # Relative pose: T_curr_kf = inv(T_wc_curr) @ T_wc_kf
        t_cw_curr = np.linalg.inv(t_wc_curr)
        t_curr_kf = t_cw_curr @ t_wc_kf
        R = t_curr_kf[:3, :3].astype(np.float64)
        t = t_curr_kf[:3, 3].astype(np.float64)

        baseline = float(np.linalg.norm(t))
        if baseline < 1e-6:
            return (np.zeros((H, W), np.float32),
                    np.full((H, W), 100.0, np.float32),
                    np.zeros((H, W), dtype=bool))

        # Current-frame bearings: direct LUT lookup.
        b_curr = self.bearing_lut.astype(np.float64)  # (H, W, 3)

        # Keyframe-frame bearings: look up at (u + flow_u, v + flow_v).
        us = np.arange(W, dtype=np.float32)
        vs = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        u_kf = uu + flow[:, :, 0]
        v_kf = vv + flow[:, :, 1]

        # Bounds check.
        in_bounds = ((u_kf >= 0) & (u_kf < W - 1) &
                     (v_kf >= 0) & (v_kf < H - 1))

        # Bilinear interpolation of keyframe bearings.
        u_kf_c = np.clip(u_kf, 0, W - 1.001).astype(np.float64)
        v_kf_c = np.clip(v_kf, 0, H - 1.001).astype(np.float64)
        u0 = np.floor(u_kf_c).astype(int)
        v0 = np.floor(v_kf_c).astype(int)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        du = (u_kf_c - u0).astype(np.float64)[..., None]
        dv = (v_kf_c - v0).astype(np.float64)[..., None]

        lut = self.bearing_lut.astype(np.float64)
        b_kf = ((1 - dv) * ((1 - du) * lut[v0, u0] + du * lut[v0, u1]) +
                 dv * ((1 - du) * lut[v1, u0] + du * lut[v1, u1]))
        # Renormalise (bilinear interp of unit vectors isn't unit).
        b_kf_norm = np.linalg.norm(b_kf, axis=-1, keepdims=True)
        b_kf = b_kf / np.where(b_kf_norm > 1e-12, b_kf_norm, 1.0)

        # Rotate keyframe bearing into current frame.
        # b_kf_rot[i,j] = R @ b_kf[i,j]
        b_kf_rot = np.einsum('ab,hwb->hwa', R, b_kf)

        # Two-ray triangulation:
        #   d_c * b_c = d_r * b_kf_rot + t
        # Dot with b_c:      d_c - d_r * cos_alpha = A
        # Dot with b_kf_rot: d_c * cos_alpha - d_r = B
        # => d_c = (A - B * cos_alpha) / sin2_alpha
        cos_alpha = np.sum(b_curr * b_kf_rot, axis=-1)  # (H, W)
        A = np.sum(b_curr * t, axis=-1)
        B = np.sum(b_kf_rot * t, axis=-1)
        sin2_alpha = 1.0 - cos_alpha * cos_alpha

        # Parallax gate: skip near-zero parallax.
        good_parallax = sin2_alpha > self.min_parallax_sin2

        # Safe division.
        sin2_safe = np.where(good_parallax, sin2_alpha, 1.0)
        d_c = (A - B * cos_alpha) / sin2_safe
        d_r = (d_c * cos_alpha - B)

        # Validity: both depths positive, in-bounds flow, good parallax.
        valid = (in_bounds & good_parallax &
                 (d_c > self.min_depth) & (d_c < self.max_depth) &
                 (d_r > 0.0))

        # Depth → log-range.
        d_c_safe = np.where(valid, d_c, 1.0)
        eta = np.log(d_c_safe).astype(np.float32)

        # Uncertainty: σ_eta ≈ σ_flow / (f * sqrt(sin²α) * baseline) * depth
        #            → η_var = (σ_flow * depth / (f * sin_α * baseline))²
        # This is the geometric depth-from-flow noise model.
        f_mean = 0.5 * (self.fx + self.fy)
        sin_alpha = np.sqrt(np.maximum(sin2_alpha, 1e-12))
        eta_var = (self.sigma_flow_px * d_c_safe /
                   (f_mean * sin_alpha * baseline)) ** 2
        eta_var = np.clip(eta_var, 1e-4, 100.0).astype(np.float32)

        return eta, eta_var, valid

    # -- seed merging -------------------------------------------------------

    def _merge_seeds(
        self,
        eta: np.ndarray,
        eta_var: np.ndarray,
        valid: np.ndarray,
        priors: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fill holes in the flow-based depth with sparse seed priors."""
        if not priors:
            return eta, eta_var, valid

        eta = eta.copy()
        eta_var = eta_var.copy()
        valid = valid.copy()

        H, W = self.work_h, self.work_w
        s = self.scale

        for u_full, v_full, p_eta, p_var in priors:
            u = int(round(u_full * s))
            v = int(round(v_full * s))
            if u < 0 or u >= W or v < 0 or v >= H:
                continue
            if p_var <= 0:
                continue
            # Splat the seed into a small radius around the pixel.
            r = 4  # pixels at working resolution
            v0 = max(0, v - r)
            v1 = min(H, v + r + 1)
            u0 = max(0, u - r)
            u1 = min(W, u + r + 1)
            patch = valid[v0:v1, u0:u1]
            # Only fill where flow didn't produce a valid depth.
            fill_mask = ~patch
            if not np.any(fill_mask):
                continue
            eta[v0:v1, u0:u1] = np.where(fill_mask, p_eta, eta[v0:v1, u0:u1])
            eta_var[v0:v1, u0:u1] = np.where(
                fill_mask, p_var, eta_var[v0:v1, u0:u1])
            valid[v0:v1, u0:u1] = valid[v0:v1, u0:u1] | fill_mask

        return eta, eta_var, valid


def from_config(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    distortion: list[float],
    config_path: str | None = None,
) -> DISDepthMapper:
    """Construct a ``DISDepthMapper`` from a YAML config file.

    Reads the ``PatchDepth`` and ``DISDepth`` sections if present; falls
    back to sensible defaults otherwise.
    """
    kwargs: dict = {}
    if config_path is not None:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        # Pull shared settings from PatchDepth section.
        pd = cfg.get('PatchDepth', {})
        kwargs['scale'] = pd.get('scale', 0.25)
        kwargs['max_depth'] = pd.get('max_depth', 12.0)
        kwargs['min_depth'] = pd.get('min_depth', 0.1)
        kwargs['min_baseline_ratio'] = pd.get('min_baseline_ratio', 0.005)
        kwargs['max_baseline_ratio'] = pd.get('max_baseline_ratio', 0.3)
        # DISDepth section for flow-specific overrides.
        dd = cfg.get('DISDepth', {})
        kwargs['dis_preset'] = dd.get('preset', 'medium')
        kwargs['sigma_flow_px'] = dd.get('sigma_flow_px', 1.0)
        kwargs['min_parallax_sin2'] = dd.get('min_parallax_sin2', 1e-5)
        kwargs['max_keyframes'] = dd.get('max_keyframes', 5)

    return DISDepthMapper(
        fx, fy, cx, cy, width, height, distortion, **kwargs)
