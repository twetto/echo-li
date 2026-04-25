use nalgebra::{Matrix3, Vector3};
use echo_lie::{SO3, SE3};

use crate::mathematical::StampedPose;

/// Umeyama SE(3) alignment between two point sets.
///
/// Finds (R, t) to minimize sum_i ||(R * p1_i + t) - p2_i||^2.
fn align_umeyama(points1: &[Vector3<f64>], points2: &[Vector3<f64>]) -> SE3 {
    assert_eq!(points1.len(), points2.len());
    let n = points1.len() as f64;

    let mu1: Vector3<f64> = points1.iter().sum::<Vector3<f64>>() / n;
    let mu2: Vector3<f64> = points2.iter().sum::<Vector3<f64>>() / n;

    let sigma1_sq: f64 = points1.iter()
        .map(|p| (p - mu1).norm_squared())
        .sum::<f64>() / n;

    // sigma12 = (1/n) * sum (q_i - mu2) * (p_i - mu1)^T
    let mut sigma12 = Matrix3::zeros();
    for (p, q) in points1.iter().zip(points2.iter()) {
        sigma12 += (q - mu2) * (p - mu1).transpose();
    }
    sigma12 /= n;

    let svd = sigma12.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let s_vals = svd.singular_values;

    let mut s_mat = Matrix3::identity();
    let scale_sum = if sigma12.determinant() < 0.0 {
        s_mat[(2, 2)] = -1.0;
        s_vals[0] + s_vals[1] - s_vals[2]
    } else {
        s_vals.sum()
    };

    let r_mat = u * s_mat * vt;
    let r = SO3::from_matrix(&r_mat);

    let s = if sigma1_sq > 1e-12 { scale_sum / sigma1_sq } else { 1.0 };
    let t = mu2 - s * (r.act(&mu1));

    SE3::new(r, t)
}

/// Align estimated trajectory to ground truth using Umeyama's method.
///
/// Matches trajectories by timestamp, then computes SE(3) alignment.
/// Falls back to first-pose alignment if fewer than 100 matched points.
pub fn align_trajectories(est: &[(f64, SE3)], gt: &[StampedPose]) -> SE3 {
    if est.is_empty() || gt.is_empty() {
        return SE3::identity();
    }

    let min_time = est[0].0.max(gt[0].stamp);
    let max_time = est.last().unwrap().0.min(gt.last().unwrap().stamp);

    let ref_period = (gt.last().unwrap().stamp - gt[0].stamp) / gt.len() as f64;
    let est_period = (est.last().unwrap().0 - est[0].0) / est.len().max(1) as f64;
    let use_period = ref_period.max(est_period);

    let mut est_matched = Vec::new();
    let mut ref_matched = Vec::new();
    let mut est_it = 0usize;
    let mut ref_it = 0usize;

    let mut t = min_time;
    while t < max_time {
        while est_it < est.len() && est[est_it].0 < t {
            est_it += 1;
        }
        while ref_it < gt.len() && gt[ref_it].stamp < t {
            ref_it += 1;
        }
        if est_it >= est.len() || ref_it >= gt.len() {
            break;
        }

        est_matched.push(est[est_it].1.translation);
        ref_matched.push(gt[ref_it].pose.translation);

        t += use_period;
    }

    if est_matched.is_empty() {
        return SE3::identity();
    }

    if est_matched.len() <= 100 {
        // Too few points — first-pose alignment
        return gt[0].pose.compose(&est[0].1.inverse());
    }

    align_umeyama(&est_matched, &ref_matched)
}
