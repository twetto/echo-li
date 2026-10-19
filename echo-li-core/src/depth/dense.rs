//! Dense range frame from the converged patch landmarks: the fusion rule and the honest fill.
//!
//! This is the Rust home of what the `fdf_densify_study.py` prototype measured (design note
//! `dense_sparse3d_patch_filter_design.md` §2.5, rows 5b–5l). Two steps:
//!
//! 1. [`fuse`]: every pixel inside at least one patch footprint takes the range of the
//!    NEAREST patch centre (no averaging: averaging overlapping patches as if independent
//!    manufactures precision, measured 3× over-confident on real poses), with a variance of
//!    four parts: the patch's own conditional variance, the shared (scale) term squared,
//!    the α-weighted disagreement of all covering patches, and a footprint term
//!    `(g0 d)^2` for the distance `d` from the centre with `g0 = tan(slant)/f` the log-range
//!    gradient of a surface at that slant. The footprint is `8 px` at the patch's pyramid
//!    level (half-width 4·2^level) and the weights are a tent on that footprint.
//! 2. [`fill`]: pixels outside every footprint but SURROUNDED by evidence take a
//!    normalised-convolution interpolation of the measured pixels (Gaussian kernel), kept only
//!    where the kernel-weighted centroid of the evidence lies within `balance` px of the pixel
//!    (interpolate, never extrapolate), where the measured log-ranges inside the kernel agree
//!    (weighted std ≤ `gate`), and within `max_d` px of the nearest measured pixel. Their
//!    variance is that nearest pixel's plus `(g d)^2`, `g` defaulting to twice the footprint
//!    gradient. These are the `Filled` class; the occupancy consumer may carve free space with
//!    them and must never assert an obstacle from them.
//!
//! Fidelity to the prototype: the fusion is the same arithmetic in the same order as the
//! Python (`numpy` accumulates per item in f64), so it reproduces it to the last bit; window
//! rounding uses round-half-to-even like Python's `round`. The fill uses an exact Euclidean
//! distance transform (Meijster) with nearest-pixel labels and an f64-accumulated separable
//! Gaussian with reflect-101 borders, where the prototype used OpenCV's 5×5 chamfer
//! approximation of the distance and its float32 blur; the two agree to the chamfer error,
//! which the port's test reports rather than hides.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One converged landmark as the fusion sees it: pixel centre at level 0, pyramid level of
/// the patch, log-range, conditional log-range variance (the filter's own), and the shared
/// log-range sigma (the scale term, fully correlated across the frame).
#[derive(Clone, Copy, Debug)]
pub struct DenseItem {
    pub px: f64,
    pub py: f64,
    pub level: u32,
    pub eta: f64,
    pub var_cond: f64,
    pub sigma_shared: f64,
}

/// The fused (`Measured`) frame, row-major `h × w`.
#[derive(Clone, Debug)]
pub struct FusedFrame {
    pub w: usize,
    pub h: usize,
    /// Log-range of the nearest covering patch (0 where not covered).
    pub eta: Vec<f64>,
    /// Its variance: conditional + shared² + disagreement + footprint.
    pub var: Vec<f64>,
    pub covered: Vec<bool>,
    /// Distance (px) to the centre of the patch the pixel took its range from (0 if none).
    pub dist: Vec<f64>,
}

/// Fusion rule `nearest_disagree_slant` of the prototype.
pub fn fuse(items: &[DenseItem], w: usize, h: usize, slant_g: f64) -> FusedFrame {
    let n = w * h;
    let mut s_a = vec![0.0f64; n];
    let mut s_ae = vec![0.0f64; n];
    let mut s_ae2 = vec![0.0f64; n];
    let mut s_as = vec![0.0f64; n];
    let mut best_d2 = vec![f64::INFINITY; n];
    let mut n_eta = vec![0.0f64; n];
    let mut n_var = vec![0.0f64; n];
    let mut n_s = vec![0.0f64; n];
    for it in items {
        let half = 4usize << it.level;
        let cx = it.px.round_ties_even() as i64;
        let cy = it.py.round_ties_even() as i64;
        let x0 = (cx - half as i64).max(0);
        let x1 = (cx + half as i64 + 1).min(w as i64);
        let y0 = (cy - half as i64).max(0);
        let y1 = (cy + half as i64 + 1).min(h as i64);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let halff = half as f64;
        for y in y0..y1 {
            let ys = y as f64 - it.py;
            let ay = (1.0 - ys.abs() / halff).max(0.0);
            let ys2 = ys * ys;
            let row = y as usize * w;
            for x in x0..x1 {
                let xs = x as f64 - it.px;
                let ax = (1.0 - xs.abs() / halff).max(0.0);
                let a = ay * ax;
                let i = row + x as usize;
                s_a[i] += a;
                s_ae[i] += a * it.eta;
                s_ae2[i] += a * it.eta * it.eta;
                s_as[i] += a * it.sigma_shared;
                let d2 = ys2 + xs * xs;
                if d2 < best_d2[i] {
                    best_d2[i] = d2;
                    n_eta[i] = it.eta;
                    n_var[i] = it.var_cond;
                    n_s[i] = it.sigma_shared;
                }
            }
        }
    }
    let mut eta = vec![0.0f64; n];
    let mut var = vec![0.0f64; n];
    let mut covered = vec![false; n];
    let mut dist = vec![0.0f64; n];
    for i in 0..n {
        covered[i] = s_a[i] > 1e-9;
        let sa = s_a[i].max(1e-12);
        let t_mean = s_ae[i] / sa;
        let disagree = (s_ae2[i] / sa - t_mean * t_mean).max(0.0);
        let d_near = if best_d2[i].is_finite() {
            best_d2[i].sqrt()
        } else {
            0.0
        };
        let slant = (slant_g * d_near) * (slant_g * d_near);
        eta[i] = n_eta[i];
        var[i] = n_var[i] + n_s[i] * n_s[i] + disagree + slant;
        dist[i] = d_near;
    }
    FusedFrame {
        w,
        h,
        eta,
        var,
        covered,
        dist,
    }
}

/// Parameters of the fill; the defaults are the prototype's (rows 5k/5l).
#[derive(Clone, Copy, Debug)]
pub struct FillParams {
    /// Gaussian kernel sigma (px) of the normalised convolution.
    pub sigma_px: f64,
    /// Max weighted std of the measured log-range inside the kernel.
    pub gate: f64,
    /// Max offset (px) of the evidence centroid from the pixel: the interpolate-only rule.
    pub balance: f64,
    /// Max distance (px) to the nearest measured pixel.
    pub max_d: f64,
    /// Log-range gradient prior of the filled variance, `(g d)^2`.
    pub g: f64,
}

impl FillParams {
    pub fn with_slant(slant_g: f64) -> Self {
        Self {
            sigma_px: 12.0,
            gate: 0.15,
            balance: 12.0,
            max_d: 48.0,
            g: 2.0 * slant_g,
        }
    }
}

/// The `Filled` class on top of a fused frame.
#[derive(Clone, Debug)]
pub struct FilledFrame {
    /// Interpolated log-range wherever the kernel saw evidence (0 elsewhere); meaningful on
    /// `filled` pixels.
    pub eta: Vec<f64>,
    /// Nearest measured variance + `(g d)^2`, on every pixel.
    pub var: Vec<f64>,
    pub filled: Vec<bool>,
    /// Fillable pixels refused by the agreement or balance gate.
    pub gated: Vec<bool>,
    /// Exact Euclidean distance (px) to the nearest measured pixel (0 on measured pixels).
    pub dist: Vec<f64>,
}

pub fn fill(fused: &FusedFrame, p: &FillParams) -> FilledFrame {
    let (w, h) = (fused.w, fused.h);
    let n = w * h;
    // Weighted images for the normalised convolution: w, eta·w, eta²·w, x·w, y·w.
    let mut ch = vec![0.0f32; 5 * n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if fused.covered[i] {
                let e = fused.eta[i] as f32;
                ch[i] = 1.0;
                ch[n + i] = e;
                ch[2 * n + i] = e * e;
                ch[3 * n + i] = x as f32;
                ch[4 * n + i] = y as f32;
            }
        }
    }
    let kernel = gaussian_kernel(p.sigma_px);
    let blurred = gaussian_blur_multi(&ch, 5, w, h, &kernel);
    let (den, num, num2, cxw, cyw) = (
        &blurred[0..n],
        &blurred[n..2 * n],
        &blurred[2 * n..3 * n],
        &blurred[3 * n..4 * n],
        &blurred[4 * n..5 * n],
    );
    let (dist, nearest) = edt_with_labels(&fused.covered, w, h);
    let mut eta = vec![0.0f64; n];
    let mut var = vec![0.0f64; n];
    let mut filled = vec![false; n];
    let mut gated = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let d = den[i] as f64;
            let seen = d > 1e-3;
            let dm = d.max(1e-6);
            let ef = if seen { num[i] as f64 / dm } else { 0.0 };
            let lvar = if seen {
                (num2[i] as f64 / dm - ef * ef).max(0.0)
            } else {
                0.0
            };
            let lstd = lvar.sqrt();
            let off = (((cxw[i] as f64 / dm) - x as f64).powi(2)
                + ((cyw[i] as f64 / dm) - y as f64).powi(2))
            .sqrt();
            let near_var = fused.var[nearest[i]];
            let dd = dist[i];
            eta[i] = ef;
            var[i] = near_var + (p.g * dd) * (p.g * dd);
            let fillable = !fused.covered[i] && seen && dd <= p.max_d;
            let g = fillable && (lstd > p.gate || off > p.balance);
            gated[i] = g;
            filled[i] = fillable && !g;
        }
    }
    FilledFrame {
        eta,
        var,
        filled,
        gated,
        dist,
    }
}

/// Normalised Gaussian kernel of size `(6σ as int) | 1`, as OpenCV's `getGaussianKernel`.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let ks = ((6.0 * sigma) as usize) | 1;
    let c = (ks / 2) as f64;
    let mut k: Vec<f64> = (0..ks)
        .map(|i| (-((i as f64 - c) * (i as f64 - c)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

#[inline]
fn reflect101(i: i64, n: i64) -> usize {
    // OpenCV BORDER_REFLECT_101: ... 3 2 | 0 1 2 3 | 2 1 0 ...
    let mut i = i;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}

/// Separable blur of `c` channel images stored back to back, f64 accumulation. Border
/// indices are tabulated once per pass; interior pixels take a straight dot product.
fn gaussian_blur_multi(src: &[f32], c: usize, w: usize, h: usize, k: &[f64]) -> Vec<f32> {
    let n = w * h;
    let r = (k.len() / 2) as i64;
    // Reflected source indices for the `r` pad pixels on each side of a line of length `len`:
    // entries 0..r for positions -r..-1, entries r..2r for positions len..len+r-1.
    let taps = |len: usize| -> Vec<usize> {
        let left = (0..r).map(|j| reflect101(j - r, len as i64));
        let right = (0..r).map(|j| reflect101(len as i64 + j, len as i64));
        left.chain(right).collect()
    };
    let tx = taps(w);
    let ty = taps(h);
    // One line: copy into an f64 buffer padded with reflect-101 borders, then accumulate
    // tap by tap over the whole line (an axpy per tap, which vectorises; a per-pixel dot
    // product would not, f64 reductions being order-sensitive).
    let line = |get: &dyn Fn(usize) -> f32, len: usize, tab: &[usize], out: &mut [f32]| {
        let ru = r as usize;
        let mut pad = vec![0.0f64; len + 2 * ru];
        for i in 0..len {
            pad[ru + i] = get(i) as f64;
        }
        for j in 0..ru {
            pad[j] = get(tab[j]) as f64;
            pad[ru + len + j] = get(tab[ru + j]) as f64;
        }
        let mut acc = vec![0.0f64; len];
        for (t, kv) in k.iter().enumerate() {
            let src = &pad[t..t + len];
            for i in 0..len {
                acc[i] += *kv * src[i];
            }
        }
        for i in 0..len {
            out[i] = acc[i] as f32;
        }
    };
    let mut tmp = vec![0.0f32; c * n];
    let hpass = |(row_idx, out_row): (usize, &mut [f32])| {
        let ch = row_idx / h;
        let y = row_idx % h;
        let src_row = &src[ch * n + y * w..ch * n + (y + 1) * w];
        line(&|i| src_row[i], w, &tx, out_row);
    };
    #[cfg(feature = "parallel")]
    tmp.par_chunks_mut(w).enumerate().for_each(hpass);
    #[cfg(not(feature = "parallel"))]
    tmp.chunks_mut(w).enumerate().for_each(hpass);
    // Vertical pass: per (channel, column), written into a transposed scratch then back.
    let mut out = vec![0.0f32; c * n];
    let vpass = |(col_idx, out_col): (usize, &mut [f32])| {
        let ch = col_idx / w;
        let x = col_idx % w;
        let base = ch * n;
        line(&|i| tmp[base + i * w + x], h, &ty, out_col);
    };
    let mut cols = vec![0.0f32; c * n];
    #[cfg(feature = "parallel")]
    cols.par_chunks_mut(h).enumerate().for_each(vpass);
    #[cfg(not(feature = "parallel"))]
    cols.chunks_mut(h).enumerate().for_each(vpass);
    for ch in 0..c {
        for x in 0..w {
            let col = &cols[ch * n + x * h..ch * n + (x + 1) * h];
            for y in 0..h {
                out[ch * n + y * w + x] = col[y];
            }
        }
    }
    out
}

/// Exact Euclidean distance transform with nearest-pixel labels (Meijster et al. 2000):
/// distance from every pixel to the nearest `true` pixel, and that pixel's index. Pixels
/// that are `true` get distance 0 and their own index. If nothing is `true`, distances are
/// `inf` and labels are 0.
fn edt_with_labels(mask: &[bool], w: usize, h: usize) -> (Vec<f64>, Vec<usize>) {
    let n = w * h;
    let inf = (w + h + 1) as i64;
    // Phase 1: per column, vertical distance g and the row ny of the nearest true pixel.
    let mut g = vec![inf; n];
    let mut ny = vec![0usize; n];
    for x in 0..w {
        if mask[x] {
            g[x] = 0;
            ny[x] = 0;
        }
        for y in 1..h {
            let i = y * w + x;
            if mask[i] {
                g[i] = 0;
                ny[i] = y;
            } else {
                g[i] = g[i - w] + 1;
                ny[i] = ny[i - w];
            }
        }
        for y in (0..h.saturating_sub(1)).rev() {
            let i = y * w + x;
            if g[i + w] + 1 < g[i] {
                g[i] = g[i + w] + 1;
                ny[i] = ny[i + w];
            }
        }
    }
    // Phase 2: per row, lower envelope of parabolas (x-i)^2 + g[i]^2.
    let mut dist = vec![f64::INFINITY; n];
    let mut label = vec![0usize; n];
    let mut s = vec![0i64; w];
    let mut t = vec![0i64; w];
    for y in 0..h {
        let row = &g[y * w..(y + 1) * w];
        let f = |x: i64, i: i64| -> i64 { (x - i) * (x - i) + row[i as usize] * row[i as usize] };
        let sep = |i: i64, u: i64| -> i64 {
            (u * u - i * i + row[u as usize] * row[u as usize] - row[i as usize] * row[i as usize])
                / (2 * (u - i))
        };
        let mut q: i64 = 0;
        s[0] = 0;
        t[0] = 0;
        for u in 1..w as i64 {
            while q >= 0 && f(t[q as usize], s[q as usize]) > f(t[q as usize], u) {
                q -= 1;
            }
            if q < 0 {
                q = 0;
                s[0] = u;
            } else {
                let ww = 1 + sep(s[q as usize], u);
                if ww < w as i64 {
                    q += 1;
                    s[q as usize] = u;
                    t[q as usize] = ww;
                }
            }
        }
        for u in (0..w as i64).rev() {
            let i = s[q as usize];
            let d2 = f(u, i);
            let idx = y * w + u as usize;
            if row[i as usize] >= inf {
                dist[idx] = f64::INFINITY;
                label[idx] = 0;
            } else {
                dist[idx] = (d2 as f64).sqrt();
                label[idx] = ny[i as usize + y * w] * w + i as usize;
            }
            if u == t[q as usize] {
                q -= 1;
            }
        }
    }
    (dist, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_matches_brute_force() {
        let (w, h) = (23, 17);
        let mut mask = vec![false; w * h];
        for &(x, y) in &[(3usize, 4usize), (20, 2), (11, 15), (0, 0), (22, 16)] {
            mask[y * w + x] = true;
        }
        let (d, l) = edt_with_labels(&mask, w, h);
        for y in 0..h {
            for x in 0..w {
                let mut best = f64::INFINITY;
                for yy in 0..h {
                    for xx in 0..w {
                        if mask[yy * w + xx] {
                            let dd = (((x as f64 - xx as f64).powi(2))
                                + (y as f64 - yy as f64).powi(2))
                            .sqrt();
                            if dd < best {
                                best = dd;
                            }
                        }
                    }
                }
                let i = y * w + x;
                assert!((d[i] - best).abs() < 1e-9, "({x},{y}) {} vs {}", d[i], best);
                let (lx, ly) = (l[i] % w, l[i] / w);
                assert!(mask[l[i]]);
                let dl = (((x as f64 - lx as f64).powi(2)) + (y as f64 - ly as f64).powi(2)).sqrt();
                assert!((dl - best).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn fuse_single_patch_footprint_and_variance() {
        let it = DenseItem {
            px: 10.0,
            py: 10.0,
            level: 0,
            eta: 1.5,
            var_cond: 0.01,
            sigma_shared: 0.1,
        };
        let f = fuse(&[it], 32, 32, 0.01);
        let c = f.covered.iter().filter(|&&b| b).count();
        // tent weights vanish at |dx| = half, so 7x7 pixels are covered, not 9x9
        assert_eq!(c, 49);
        let i = 10 * 32 + 10;
        assert!((f.var[i] - (0.01 + 0.01)).abs() < 1e-12);
        let j = 10 * 32 + 12;
        assert!((f.var[j] - (0.01 + 0.01 + (0.01 * 2.0f64).powi(2))).abs() < 1e-12);
        assert!((f.dist[j] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn fill_interpolates_between_two_patches_and_refuses_outside() {
        let a = DenseItem {
            px: 20.0,
            py: 32.0,
            level: 0,
            eta: 1.0,
            var_cond: 0.01,
            sigma_shared: 0.0,
        };
        let b = DenseItem {
            px: 44.0,
            py: 32.0,
            level: 0,
            eta: 1.2,
            var_cond: 0.01,
            sigma_shared: 0.0,
        };
        let f = fuse(&[a, b], 64, 64, 0.0);
        let p = FillParams {
            sigma_px: 12.0,
            gate: 0.15,
            balance: 12.0,
            max_d: 48.0,
            g: 0.0,
        };
        let r = fill(&f, &p);
        let mid = 32 * 64 + 32;
        assert!(r.filled[mid], "midpoint between the patches is filled");
        assert!((r.eta[mid] - 1.1).abs() < 0.02);
        let far = 32 * 64 + 2; // beyond the last evidence on the left: centroid is far right
        assert!(!r.filled[far]);
    }
}
