use nalgebra::{Matrix3, Vector3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Per-pixel depth triangulation via derotated epipolar geometry.
pub fn depth_densification(
    k: &Matrix3<f64>,
    dr: &Matrix3<f64>,
    p: &Vector3<f64>,
    flow: &[f32], // (H, W, 2) flattened
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    let mut invdepth_map = vec![-1.0f32; h * w];
    let mut geom_drive_map = vec![0.0f32; h * w];

    let tx = p[0];
    let ty = p[1];
    let tz = p[2];

    let compute_row = |v: usize, row_inv: &mut [f32], row_geom: &mut [f32]| {
        for u in 0..w {
            let idx = (v * w + u) * 2;
            let flow_u = flow[idx] as f64;
            let flow_v = flow[idx + 1] as f64;

            let x_curr_norm = (u as f64 - cx) / fx;
            let y_curr_norm = (v as f64 - cy) / fy;

            let u_prev = u as f64 - flow_u;
            let v_prev = v as f64 - flow_v;

            if u_prev >= 0.0 && u_prev < w as f64 && v_prev >= 0.0 && v_prev < h as f64 {
                let x_prev_norm = (u_prev - cx) / fx;
                let y_prev_norm = (v_prev - cy) / fy;

                let bearing_prev_raw = Vector3::new(x_prev_norm, y_prev_norm, 1.0);
                let bearing_prev_aligned = dr * bearing_prev_raw;

                if bearing_prev_aligned[2] > 1e-6 {
                    let x_prev_rect = bearing_prev_aligned[0] / bearing_prev_aligned[2];
                    let y_prev_rect = bearing_prev_aligned[1] / bearing_prev_aligned[2];

                    let num_x = x_prev_rect * tz - tx;
                    let den_x = x_prev_rect - x_curr_norm;
                    let num_y = y_prev_rect * tz - ty;
                    let den_y = y_prev_rect - y_curr_norm;

                    let geom_mag_sq = num_x * num_x + num_y * num_y;
                    let ideal_mag = geom_mag_sq.sqrt();
                    let obs_mag = (den_x * den_x + den_y * den_y).sqrt();

                    if ideal_mag > 0.0 && obs_mag > 0.0 {
                        row_geom[u] = ideal_mag as f32;
                        let dot_product = num_x * den_x + num_y * den_y;
                        let cosine_sim = dot_product / (ideal_mag * obs_mag);

                        if dot_product > 1e-6 && cosine_sim > 0.95 {
                            let z_curr = geom_mag_sq / dot_product;
                            if z_curr > 0.1 {
                                row_inv[u] = (1.0 / z_curr) as f32;
                            }
                        }
                    }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    invdepth_map
        .par_chunks_mut(w)
        .enumerate()
        .zip(geom_drive_map.par_chunks_mut(w))
        .for_each(|((v, row_inv), row_geom)| compute_row(v, row_inv, row_geom));
    #[cfg(not(feature = "parallel"))]
    for (v, (row_inv, row_geom)) in invdepth_map
        .chunks_mut(w)
        .zip(geom_drive_map.chunks_mut(w))
        .enumerate()
    {
        compute_row(v, row_inv, row_geom);
    }

    (invdepth_map, geom_drive_map)
}

/// Forward-warp inverse depth + variance via bilinear splatting.
pub fn bilinear_splatting(
    u_proj: &[f32],
    v_proj: &[f32],
    inv_z_proj: &[f32],
    propagated_var: &[f32],
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut predicted_invdepth_accum = vec![0.0f32; h * w];
    let mut predicted_var_accum = vec![0.0f32; h * w];
    let mut weights_accum = vec![0.0f32; h * w];

    for i in 0..u_proj.len() {
        let up = u_proj[i];
        let vp = v_proj[i];
        let uf = up.floor() as i32;
        let vf = vp.floor() as i32;

        if uf >= 0 && uf < (w - 1) as i32 && vf >= 0 && vf < (h - 1) as i32 {
            let uf = uf as usize;
            let vf = vf as usize;
            let inv_z_p = inv_z_proj[i];
            let var_p = propagated_var[i];
            let dx = up - uf as f32;
            let dy = vp - vf as f32;

            let w_ll = (1.0 - dx) * (1.0 - dy);
            let w_lr = dx * (1.0 - dy);
            let w_ul = (1.0 - dx) * dy;
            let w_ur = dx * dy;

            let add = |map: &mut Vec<f32>, r: usize, c: usize, val: f32| {
                map[r * w + c] += val;
            };

            add(&mut predicted_invdepth_accum, vf, uf, w_ll * inv_z_p);
            add(&mut predicted_var_accum, vf, uf, w_ll * var_p);
            add(&mut weights_accum, vf, uf, w_ll);

            add(&mut predicted_invdepth_accum, vf, uf + 1, w_lr * inv_z_p);
            add(&mut predicted_var_accum, vf, uf + 1, w_lr * var_p);
            add(&mut weights_accum, vf, uf + 1, w_lr);

            add(&mut predicted_invdepth_accum, vf + 1, uf, w_ul * inv_z_p);
            add(&mut predicted_var_accum, vf + 1, uf, w_ul * var_p);
            add(&mut weights_accum, vf + 1, uf, w_ul);

            add(
                &mut predicted_invdepth_accum,
                vf + 1,
                uf + 1,
                w_ur * inv_z_p,
            );
            add(&mut predicted_var_accum, vf + 1, uf + 1, w_ur * var_p);
            add(&mut weights_accum, vf + 1, uf + 1, w_ur);
        }
    }
    (predicted_invdepth_accum, predicted_var_accum, weights_accum)
}

/// Forward-warp Beta (a, b) counts via bilinear splatting.
pub fn bilinear_splatting_ab(
    u_proj: &[f32],
    v_proj: &[f32],
    a_vals: &[f32],
    b_vals: &[f32],
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut a_accum = vec![0.0f32; h * w];
    let mut b_accum = vec![0.0f32; h * w];
    let mut weights_accum = vec![0.0f32; h * w];

    for i in 0..u_proj.len() {
        let up = u_proj[i];
        let vp = v_proj[i];
        let uf = up.floor() as i32;
        let vf = vp.floor() as i32;

        if uf >= 0 && uf < (w - 1) as i32 && vf >= 0 && vf < (h - 1) as i32 {
            let uf = uf as usize;
            let vf = vf as usize;
            let ap = a_vals[i];
            let bp = b_vals[i];
            let dx = up - uf as f32;
            let dy = vp - vf as f32;

            let w_ll = (1.0 - dx) * (1.0 - dy);
            let w_lr = dx * (1.0 - dy);
            let w_ul = (1.0 - dx) * dy;
            let w_ur = dx * dy;

            let add = |map: &mut Vec<f32>, r: usize, c: usize, val: f32| {
                map[r * w + c] += val;
            };

            add(&mut a_accum, vf, uf, w_ll * ap);
            add(&mut b_accum, vf, uf, w_ll * bp);
            add(&mut weights_accum, vf, uf, w_ll);

            add(&mut a_accum, vf, uf + 1, w_lr * ap);
            add(&mut b_accum, vf, uf + 1, w_lr * bp);
            add(&mut weights_accum, vf, uf + 1, w_lr);

            add(&mut a_accum, vf + 1, uf, w_ul * ap);
            add(&mut b_accum, vf + 1, uf, w_ul * bp);
            add(&mut weights_accum, vf + 1, uf, w_ul);

            add(&mut a_accum, vf + 1, uf + 1, w_ur * ap);
            add(&mut b_accum, vf + 1, uf + 1, w_ur * bp);
            add(&mut weights_accum, vf + 1, uf + 1, w_ur);
        }
    }
    (a_accum, b_accum, weights_accum)
}

/// Vogiatzis Gaussian-Beta mixture fusion.
pub fn vogiatzis_update(
    predicted_invdepth: &[f32],
    predicted_var: &[f32],
    predicted_a: &[f32],
    predicted_b: &[f32],
    observed_invdepth: &[f32],
    geom_drive: &[f32],
    sigma_norm: f64,
    init_var: f32,
    uniform_rho_max: f64,
    a_init: f32,
    b_init: f32,
    ab_min: f32,
    ab_max: f32,
    min_inlier_ratio: f32,
    mahal_reset_chi2: f32,
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut updated_invdepth = vec![-1.0f32; h * w];
    let mut updated_var = vec![init_var; h * w];
    let mut updated_a = vec![a_init; h * w];
    let mut updated_b = vec![b_init; h * w];

    let u_rho = uniform_rho_max;

    let update_row = |r: usize,
                      row_inv: &mut [f32],
                      row_var: &mut [f32],
                      row_a: &mut [f32],
                      row_b: &mut [f32]| {
        for c in 0..w {
            let idx = r * w + c;
            let mu = predicted_invdepth[idx] as f64;
            let sigma_sq = predicted_var[idx] as f64;
            let a = predicted_a[idx] as f64;
            let b = predicted_b[idx] as f64;

            let x = observed_invdepth[idx] as f64;
            let tau_sq = (sigma_norm / geom_drive[idx].max(1e-8) as f64).powi(2);

            let valid_pred = mu > 0.0;
            let valid_obs = x > 0.0;

            if valid_pred && valid_obs {
                let s_total = sigma_sq + tau_sq;
                let m_dist_sq = (x - mu).powi(2) / s_total;

                // Outlier reset branch
                if a + b > 0.0
                    && (a / (a + b)) < min_inlier_ratio as f64
                    && m_dist_sq > mahal_reset_chi2 as f64
                {
                    row_inv[c] = x as f32;
                    row_var[c] = tau_sq as f32;
                    row_a[c] = a_init;
                    row_b[c] = b_init;
                    continue;
                }

                // Inlier Kalman branch
                let m = (mu * tau_sq + x * sigma_sq) / s_total;
                let s_sq = sigma_sq * tau_sq / s_total;

                let gauss_pdf = if m_dist_sq > 100.0 {
                    0.0
                } else {
                    (-0.5 * m_dist_sq).exp() / (2.0 * std::f64::consts::PI * s_total).sqrt()
                };

                let ab_sum = a + b;
                let c1 = (a / ab_sum) * gauss_pdf;
                let c2 = (b / ab_sum) * u_rho;
                let z_norm = c1 + c2;

                let (new_mu, new_sigma_sq, mut new_a, mut new_b);

                if z_norm < 1e-30 {
                    new_mu = mu;
                    new_sigma_sq = sigma_sq;
                    new_a = a;
                    new_b = b + 1.0;
                } else {
                    let w1 = c1 / z_norm;
                    let w2 = c2 / z_norm;

                    new_mu = w1 * m + w2 * mu;
                    let e_rho2 = w1 * (s_sq + m * m) + w2 * (sigma_sq + mu * mu);
                    new_sigma_sq = (e_rho2 - new_mu * new_mu).max(1e-8);

                    let denom1 = ab_sum + 1.0;
                    let denom2 = denom1 * (ab_sum + 2.0);
                    let e_pi = (w1 * (a + 1.0) + w2 * a) / denom1;
                    let e_pi2 = (w1 * (a + 1.0) * (a + 2.0) + w2 * a * (a + 1.0)) / denom2;
                    let v_pi = e_pi2 - e_pi * e_pi;

                    if v_pi < 1e-6 || e_pi <= 1e-6 || e_pi >= 1.0 - 1e-6 {
                        new_a = a + w1;
                        new_b = b + w2;
                    } else {
                        let mut factor = e_pi * (1.0 - e_pi) / v_pi - 1.0;
                        if factor < 0.5 {
                            factor = 0.5;
                        }
                        new_a = e_pi * factor;
                        new_b = (1.0 - e_pi) * factor;
                    }
                }

                new_a = new_a.clamp(ab_min as f64, ab_max as f64);
                new_b = new_b.clamp(ab_min as f64, ab_max as f64);

                row_inv[c] = new_mu as f32;
                row_var[c] = new_sigma_sq as f32;
                row_a[c] = new_a as f32;
                row_b[c] = new_b as f32;
            } else if valid_obs {
                row_inv[c] = x as f32;
                row_var[c] = init_var;
                row_a[c] = a_init;
                row_b[c] = b_init;
            } else if valid_pred {
                row_inv[c] = mu as f32;
                row_var[c] = sigma_sq as f32;
                row_a[c] = a as f32;
                row_b[c] = b as f32;
            }
        }
    };

    #[cfg(feature = "parallel")]
    updated_invdepth
        .par_chunks_mut(w)
        .enumerate()
        .zip(updated_var.par_chunks_mut(w))
        .zip(updated_a.par_chunks_mut(w))
        .zip(updated_b.par_chunks_mut(w))
        .for_each(|((((r, row_inv), row_var), row_a), row_b)| {
            update_row(r, row_inv, row_var, row_a, row_b)
        });
    #[cfg(not(feature = "parallel"))]
    for (r, (((row_inv, row_var), row_a), row_b)) in updated_invdepth
        .chunks_mut(w)
        .zip(updated_var.chunks_mut(w))
        .zip(updated_a.chunks_mut(w))
        .zip(updated_b.chunks_mut(w))
        .enumerate()
    {
        update_row(r, row_inv, row_var, row_a, row_b);
    }

    (updated_invdepth, updated_var, updated_a, updated_b)
}
