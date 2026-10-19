"""The Rust dense fusion / fill against the prototype it replaced (design note row 5q).

`fuse` must be bit-identical to the numpy prototype (the rule `nearest_disagree_slant` of
`diagnostics/fdf_densify_study.py`, reproduced here so the test needs no dataset). `fill` is
checked for its contract: it interpolates only between surrounding evidence, never past the
last patch, and its variance grows with the distance to evidence. The whole mapper is smoke-
tested on synthetic frames.

Run: .venv/bin/pytest echo-li-python/tests/test_dense_port.py
"""
import numpy as np
import pytest

import echo_li


def fuse_reference(items, H, W, slant_g):
    """Verbatim arithmetic of the prototype's `fuse`, rule nearest_disagree_slant."""
    S_a = np.zeros((H, W)); S_ae = np.zeros((H, W)); S_ae2 = np.zeros((H, W))
    best_d2 = np.full((H, W), np.inf); n_eta = np.zeros((H, W)); n_var = np.zeros((H, W)); n_s = np.zeros((H, W))
    for px, py, level, eta, vc, ss in items:
        half = 4 * (2 ** int(level))
        x0, x1 = max(0, int(round(px)) - half), min(W, int(round(px)) + half + 1)
        y0, y1 = max(0, int(round(py)) - half), min(H, int(round(py)) + half + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) - px; ys = np.arange(y0, y1) - py
        ax = np.clip(1.0 - np.abs(xs) / half, 0.0, None); ay = np.clip(1.0 - np.abs(ys) / half, 0.0, None)
        a = ay[:, None] * ax[None, :]
        S_a[y0:y1, x0:x1] += a; S_ae[y0:y1, x0:x1] += a * eta; S_ae2[y0:y1, x0:x1] += a * eta * eta
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        better = d2 < best_d2[y0:y1, x0:x1]
        sub = best_d2[y0:y1, x0:x1]; sub[better] = d2[better]
        n_eta[y0:y1, x0:x1][better] = eta; n_var[y0:y1, x0:x1][better] = vc; n_s[y0:y1, x0:x1][better] = ss
    covered = S_a > 1e-9
    Sa = np.maximum(S_a, 1e-12)
    t_mean = S_ae / Sa
    disagree = np.maximum(S_ae2 / Sa - t_mean * t_mean, 0.0)
    d_near = np.sqrt(np.where(np.isfinite(best_d2), best_d2, 0.0))
    slant = (slant_g * d_near) ** 2
    return n_eta, n_var + n_s * n_s + disagree + slant, covered, d_near


def surface_items(rng, n, H, W):
    k = max(1, n // 40)
    cx = rng.uniform(0, W, k); cy = rng.uniform(0, H, k)
    j = rng.integers(0, k, n)
    px = cx[j] + rng.normal(0, 18, n); py = cy[j] + rng.normal(0, 18, n)
    lev = rng.integers(0, 2, n)
    eta = 1.5 + 0.004 * py + 0.3 * np.sin(px / 70.0) + rng.normal(0, 0.02, n)
    vc = np.exp(rng.normal(-5, 0.7, n)); ss = np.full(n, 0.02)
    px[: n // 20] = np.round(px[: n // 20]) + 0.5      # exact ties for the rounding rule
    return np.stack([px, py, lev, eta, vc, ss], axis=1)


@pytest.mark.parametrize("n", [30, 400, 1500])
def test_fuse_is_bit_identical_to_the_prototype(n):
    rng = np.random.default_rng(n)
    H, W = 240, 376
    g0 = np.tan(np.radians(35.0)) / 230.0
    items = surface_items(rng, n, H, W)
    e_r, v_r, c_r, d_r = fuse_reference(list(map(tuple, items)), H, W, g0)
    e, v, c, d = echo_li.dense_fuse(items, H, W, g0)
    assert np.array_equal(c, c_r)
    assert np.array_equal(e, e_r)
    assert np.array_equal(v, v_r)
    assert np.array_equal(d, d_r)


def test_fill_interpolates_only_between_evidence():
    H, W = 64, 96
    g0 = 0.003
    a = np.array([[24.0, 32.0, 0, 1.0, 0.01, 0.0], [72.0, 32.0, 0, 1.2, 0.01, 0.0]])
    e, v, c, d = echo_li.dense_fuse(a, H, W, g0)
    ef, vf, filled, gated, dist = echo_li.dense_fill(e, v, c, 12.0, 0.15, 12.0, 48.0, 2.0 * g0)
    assert filled[32, 48], "the midpoint between two patches is filled"
    assert abs(ef[32, 48] - 1.1) < 0.02
    assert not filled[32, 4], "left of the last evidence is not filled (centroid far right)"
    assert not filled[4, 48], "above both patches is not filled (one-sided evidence)"
    assert not (filled & c).any(), "measured pixels are never re-filled"
    # variance grows with the distance to evidence
    ys, xs = np.nonzero(filled)
    far = dist[ys, xs] > 6; near = dist[ys, xs] <= 2
    assert vf[ys, xs][far].mean() > vf[ys, xs][near].mean()


def test_fill_refuses_disagreeing_evidence():
    H, W = 64, 96
    a = np.array([[24.0, 32.0, 0, 1.0, 0.01, 0.0], [72.0, 32.0, 0, 2.0, 0.01, 0.0]])  # 1 m vs 7 m
    e, v, c, d = echo_li.dense_fuse(a, H, W, 0.0)
    ef, vf, filled, gated, dist = echo_li.dense_fill(e, v, c, 12.0, 0.15, 12.0, 48.0, 0.0)
    assert not filled[32, 48], "a depth edge between the patches stays Unknown"
    assert gated[32, 48]


def test_mapper_runs_on_synthetic_frames():
    cam = echo_li.PinholeCamera(150.0, 150.0, 160.0, 100.0)
    sparse = {"parametrization": "bearing_invdepth_additive3d", "max_pool_size": 4000, "min_track_length": 5,
              "bias_walk_var": 0.3, "pose_range_direct": True, "scale_sigma": 0.02, "sigma_pixel": 1.0,
              "mahalanobis_reset_chi2": 0.2, "conv_variance_threshold": 2.0, "init_depth_var": 10.0,
              "a_init": 1e8, "b_init": 1e-8, "ab_max": 200.0, "ab_min": 1.0, "max_depth": 12.0, "min_depth": 0.1}
    m = echo_li.DenseDepthMapper(cam, 150.0, 150.0, 160.0, 100.0, 320, 200, sparse,
                                 tracker={"max_mean_abs_residual": 30.0, "min_birth_curvature": 10.0,
                                          "min_track_curvature": 160.0}, scale=0.5, histeq="clahe")
    assert m.size() == (160, 100)
    fx, fy, cx, cy = m.intrinsics()
    assert abs(fx - 75.0) < 1e-9 and abs(cx - 80.0) < 1e-9
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (200, 320)).astype(np.uint8)
    T = np.eye(4)
    for i in range(3):
        T[0, 3] = 0.05 * i
        eta, var, cls, st = m.process(img, 0.05 * i, T, None, [(10.0, 10.0)])
        assert eta.shape == (100, 160) and cls.dtype == np.uint8
        assert set(np.unique(cls)) <= {0, 1, 2}
    assert np.asarray(m.last_frame()).shape == (100, 160)
