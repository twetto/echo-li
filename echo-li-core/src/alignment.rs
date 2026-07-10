use echo_lie::{SE3, SO3};
use nalgebra::{Matrix3, Vector3};

use crate::mathematical::StampedPose;

/// Umeyama SE(3) alignment between two point sets.
///
/// Finds (R, t) to minimize sum_i ||(R * p1_i + t) - p2_i||^2.
fn align_umeyama(points1: &[Vector3<f64>], points2: &[Vector3<f64>]) -> SE3 {
    assert_eq!(points1.len(), points2.len());
    let n = points1.len() as f64;

    let mu1: Vector3<f64> = points1.iter().sum::<Vector3<f64>>() / n;
    let mu2: Vector3<f64> = points2.iter().sum::<Vector3<f64>>() / n;

    let sigma1_sq: f64 = points1
        .iter()
        .map(|p| (p - mu1).norm_squared())
        .sum::<f64>()
        / n;

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

    let s = if sigma1_sq > 1e-12 {
        scale_sum / sigma1_sq
    } else {
        1.0
    };
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

    // Match by interpolating the GT translation at each estimate timestamp —
    // the same convention as `compute_ate_metrics`. The previous coarse
    // nearest-forward resampling (at the ~image period) mismatched pairs during
    // fast motion and biased the fit, inflating reported ATE velocity-dependently
    // (~9% on fast sequences, ~2% on slow ones).
    let mut est_matched = Vec::new();
    let mut ref_matched = Vec::new();
    let mut cursor = 0usize;
    for (stamp, pose) in est {
        if let Some(gt_pos) = interpolate_gt_translation(gt, *stamp, &mut cursor) {
            est_matched.push(pose.translation);
            ref_matched.push(gt_pos);
        }
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

/// Linear interpolation of the GT translation at `stamp`. `cursor` is advanced
/// monotonically, so this is O(1) amortized when called over time-sorted stamps.
fn interpolate_gt_translation(
    gt: &[StampedPose],
    stamp: f64,
    cursor: &mut usize,
) -> Option<Vector3<f64>> {
    if stamp < gt.first()?.stamp || stamp > gt.last()?.stamp {
        return None;
    }
    while *cursor + 1 < gt.len() && gt[*cursor + 1].stamp < stamp {
        *cursor += 1;
    }
    if *cursor + 1 >= gt.len() {
        return None;
    }
    let a = &gt[*cursor];
    let b = &gt[*cursor + 1];
    let dt = b.stamp - a.stamp;
    if dt <= 0.0 {
        return Some(a.pose.translation);
    }
    let alpha = ((stamp - a.stamp) / dt).clamp(0.0, 1.0);
    Some(a.pose.translation + alpha * (b.pose.translation - a.pose.translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematical::StampedPose;
    use approx::assert_relative_eq;

    fn helix(n: usize) -> Vec<Vector3<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.2;
                Vector3::new(2.0 * t.cos(), 2.0 * t.sin(), i as f64 * 0.05)
            })
            .collect()
    }

    fn make_gt(pts: &[Vector3<f64>]) -> Vec<StampedPose> {
        pts.iter()
            .enumerate()
            .map(|(i, p)| StampedPose::new(i as f64, SE3::new(SO3::identity(), *p)))
            .collect()
    }

    fn make_est(pts: &[Vector3<f64>]) -> Vec<(f64, SE3)> {
        pts.iter()
            .enumerate()
            .map(|(i, p)| (i as f64, SE3::new(SO3::identity(), *p)))
            .collect()
    }

    #[test]
    fn empty_est_returns_identity() {
        let gt = make_gt(&helix(10));
        let result = align_trajectories(&[], &gt);
        assert_relative_eq!(result.translation.norm(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn empty_gt_returns_identity() {
        let est = make_est(&helix(10));
        let result = align_trajectories(&est, &[]);
        assert_relative_eq!(result.translation.norm(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn few_matched_points_uses_first_pose_alignment() {
        // 50 points → ≤100 matched → first-pose fallback: gt[0].pose.compose(&est[0].1.inverse())
        let pts = helix(50);
        let offset = Vector3::new(1.0, 2.0, 3.0);
        let gt: Vec<StampedPose> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| StampedPose::new(i as f64, SE3::new(SO3::identity(), p + offset)))
            .collect();
        let est = make_est(&pts);

        let result = align_trajectories(&est, &gt);
        // gt[0].pose = SE3(id, pts[0]+offset), est[0].1 = SE3(id, pts[0])
        // compose gives SE3(id, pts[0]+offset - pts[0]) = SE3(id, offset)
        assert_relative_eq!(result.translation, offset, epsilon = 1e-9);
        assert_relative_eq!(
            result.rotation.as_matrix(),
            Matrix3::identity(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn umeyama_recovers_pure_translation() {
        let pts = helix(200);
        let delta = Vector3::new(5.0, -3.0, 2.0);
        let gt = make_gt(&pts);
        let est: Vec<(f64, SE3)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (i as f64, SE3::new(SO3::identity(), p + delta)))
            .collect();

        let result = align_trajectories(&est, &gt);
        // Umeyama: R≈I, t≈-delta (maps est→gt)
        assert_relative_eq!(
            result.rotation.as_matrix(),
            Matrix3::identity(),
            epsilon = 1e-6
        );
        assert_relative_eq!(result.translation, -delta, epsilon = 1e-6);
    }

    #[test]
    fn umeyama_recovers_pure_rotation() {
        let r_true = SO3::exp(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        let pts = helix(200);
        let gt = make_gt(&pts);
        let est: Vec<(f64, SE3)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (i as f64, SE3::new(SO3::identity(), r_true.inverse().act(p))))
            .collect();

        let result = align_trajectories(&est, &gt);
        assert_relative_eq!(
            result.rotation.as_matrix(),
            r_true.as_matrix(),
            epsilon = 1e-6
        );
        assert_relative_eq!(result.translation.norm(), 0.0, epsilon = 1e-5);
    }
}
