//! Matrix exponential and logarithm for general square matrices.
//!
//! expm: scaling-and-squaring with Padé(13,13) — matches scipy/MATLAB.
//! logm: inverse scaling-and-squaring with Padé approximant.

use nalgebra::DMatrix;

// Padé(13,13) coefficients for expm (Higham 2005, Table 10.4)
const PADE13_B: [f64; 14] = [
    1.0,
    0.5,
    0.12,
    1.833_333_333_333_333_4e-2,
    1.992_753_623_188_405_8e-3,
    1.630_434_782_608_695_7e-4,
    1.035_196_687_401_6e-5,
    5.175_983_436_853_2e-7,
    2.043_151_356_652_5e-8,
    6.306_022_705_717_6e-10,
    1.483_770_048_404_1e-11,
    2.529_153_491_597_9e-13,
    2.810_170_546_219_96e-15,
    1.544_049_750_670_308_9e-17,
];

/// 1-norm of a matrix.
fn onenorm(a: &DMatrix<f64>) -> f64 {
    let n = a.ncols();
    (0..n)
        .map(|j| (0..a.nrows()).map(|i| a[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Matrix exponential via scaling-and-squaring with Padé(13,13).
pub fn expm(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());

    let norm1 = onenorm(a);
    if norm1 == 0.0 {
        return DMatrix::identity(n, n);
    }

    // Determine scaling: choose s such that ‖A/2^s‖₁ ≤ θ₁₃ = 5.37
    let theta13 = 5.371920351148152;
    let s = ((norm1 / theta13).log2().ceil().max(0.0)) as u32;
    let a_scaled = a * 2.0_f64.powi(-(s as i32));

    // Compute matrix powers
    let id = DMatrix::identity(n, n);
    let a2 = &a_scaled * &a_scaled;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    // U = A(b₁I + b₃A² + b₅A⁴ + b₇A⁶ + A⁶(b₉A² + b₁₁A⁴ + b₁₃A⁶))
    let inner_u = &a2 * PADE13_B[9] + &a4 * PADE13_B[11] + &a6 * PADE13_B[13];
    let u = &a_scaled
        * (&id * PADE13_B[1] + &a2 * PADE13_B[3] + &a4 * PADE13_B[5] + &a6 * PADE13_B[7]
            + &a6 * inner_u);

    // V = b₀I + b₂A² + b₄A⁴ + b₆A⁶ + A⁶(b₈A² + b₁₀A⁴ + b₁₂A⁶)
    let inner_v = &a2 * PADE13_B[8] + &a4 * PADE13_B[10] + &a6 * PADE13_B[12];
    let v = &id * PADE13_B[0] + &a2 * PADE13_B[2] + &a4 * PADE13_B[4] + &a6 * PADE13_B[6]
        + &a6 * inner_v;

    // r₁₃ = (V - U)⁻¹(V + U)
    let lhs = &v - &u;
    let rhs = &v + &u;
    let mut result = lhs.lu().solve(&rhs).unwrap_or(rhs);

    // Squaring phase
    for _ in 0..s {
        result = &result * &result;
    }

    result
}

/// Matrix logarithm via inverse scaling-and-squaring.
pub fn logm(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());

    let id = DMatrix::identity(n, n);

    // Repeated square roots to bring matrix close to identity
    let mut s = 0u32;
    let mut a_root = a.clone();
    for _ in 0..64 {
        let diff_norm = onenorm(&(&a_root - &id));
        if diff_norm < 0.1 {
            break;
        }
        a_root = matrix_sqrt(&a_root);
        s += 1;
    }

    // Padé approximation of log(I + X) via partial fractions
    // Using [m/m] Padé with m = 7 for good accuracy
    let x = &a_root - &id;
    let result = pade_log(&x, n);

    // Undo the square roots: log(A) = 2^s · log(A^{1/2^s})
    result * (2.0_f64.powi(s as i32))
}

/// Diagonal Padé approximation for log(I + X), |X| < 1.
///
/// Uses the identity: log(I + X) = X (I + X/2)⁻¹ · correction terms
/// Implemented as Gauss-Legendre quadrature on the integral representation:
///   log(I + X) = ∫₀¹ (I + tX)⁻¹ X dt
///
/// 8-point Gauss-Legendre.
fn pade_log(x: &DMatrix<f64>, n: usize) -> DMatrix<f64> {
    // 8-point Gauss-Legendre nodes and weights on [0, 1]
    let nodes: [f64; 8] = [
        0.019_855_071_751_231_884,
        0.101_666_761_293_186_63,
        0.237_233_795_041_835_51,
        0.408_282_678_752_175_1,
        0.591_717_321_247_824_9,
        0.762_766_204_958_164_49,
        0.898_333_238_706_813_37,
        0.980_144_928_248_768_12,
    ];
    let weights: [f64; 8] = [
        0.050_614_268_145_188_13,
        0.111_190_517_226_687_24,
        0.156_853_322_938_943_64,
        0.181_341_891_689_180_99,
        0.181_341_891_689_180_99,
        0.156_853_322_938_943_64,
        0.111_190_517_226_687_24,
        0.050_614_268_145_188_13,
    ];

    let id = DMatrix::identity(n, n);
    let mut result = DMatrix::zeros(n, n);

    for k in 0..8 {
        let t = nodes[k];
        // (I + t·X)⁻¹ · X
        let m = &id + x * t;
        let m_inv = m.try_inverse().unwrap_or_else(|| id.clone());
        result += (&m_inv * x) * weights[k];
    }

    result
}

/// Denman-Beavers iteration for matrix square root.
fn matrix_sqrt(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let mut y = a.clone();
    let mut z = DMatrix::identity(n, n);

    for _ in 0..64 {
        let z_inv = z
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(n, n));
        let y_inv = y
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(n, n));
        let y_next = 0.5 * (&y + z_inv);
        let z_next = 0.5 * (&z + y_inv);
        let diff = onenorm(&(&y_next - &y));
        y = y_next;
        z = z_next;
        if diff < 1e-15 {
            break;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn expm_identity() {
        let z = DMatrix::zeros(3, 3);
        let e = expm(&z);
        assert_abs_diff_eq!(e, DMatrix::identity(3, 3), epsilon = 1e-14);
    }

    #[test]
    fn expm_logm_roundtrip() {
        for _ in 0..20 {
            let a = DMatrix::from_fn(3, 3, |_, _| 0.3 * (rand::random::<f64>() - 0.5));
            let ea = expm(&a);
            let log_ea = logm(&ea);
            let ea2 = expm(&log_ea);
            assert_abs_diff_eq!(ea, ea2, epsilon = 1e-10);
        }
    }

    #[test]
    fn expm_skew_symmetric_is_rotation() {
        let w = nalgebra::Vector3::new(0.3, -0.5, 0.7);
        let mut s = DMatrix::zeros(3, 3);
        s[(0, 1)] = -w[2];
        s[(0, 2)] = w[1];
        s[(1, 0)] = w[2];
        s[(1, 2)] = -w[0];
        s[(2, 0)] = -w[1];
        s[(2, 1)] = w[0];
        let r = expm(&s);
        let rrt = &r * r.transpose();
        assert_abs_diff_eq!(rrt, DMatrix::identity(3, 3), epsilon = 1e-12);
    }
}
