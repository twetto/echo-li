use echo_lie::{SE3, SO3};

use crate::mathematical::StampedPose;

#[derive(Debug, Clone, Copy)]
pub struct ErrorStats {
    pub rmse: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TrajectoryMetrics {
    pub matched_poses: usize,
    pub ate_position_m: ErrorStats,
    pub ate_attitude_deg: ErrorStats,
}

pub fn compute_ate_metrics(
    est: &[(f64, SE3)],
    gt: &[StampedPose],
    alignment: &SE3,
) -> Option<TrajectoryMetrics> {
    if est.is_empty() || gt.len() < 2 {
        return None;
    }

    let mut gt_cursor = 0usize;
    let mut pos_errors = Vec::new();
    let mut att_errors = Vec::new();

    for (stamp, pose) in est {
        let Some(gt_pose) = interpolate_groundtruth(gt, *stamp, &mut gt_cursor) else {
            continue;
        };
        let aligned = alignment.compose(pose);
        pos_errors.push((aligned.translation - gt_pose.translation).norm());

        let r_err = gt_pose.rotation.inverse().compose(&aligned.rotation);
        att_errors.push(rotation_angle_rad(&r_err).to_degrees());
    }

    if pos_errors.is_empty() {
        return None;
    }

    Some(TrajectoryMetrics {
        matched_poses: pos_errors.len(),
        ate_position_m: error_stats(&mut pos_errors),
        ate_attitude_deg: error_stats(&mut att_errors),
    })
}

fn interpolate_groundtruth(gt: &[StampedPose], stamp: f64, cursor: &mut usize) -> Option<SE3> {
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
        return Some(a.pose.clone());
    }

    let alpha = ((stamp - a.stamp) / dt).clamp(0.0, 1.0);
    let translation = a.pose.translation + alpha * (b.pose.translation - a.pose.translation);
    let rotation = SO3::from_quaternion(a.pose.rotation.q.slerp(&b.pose.rotation.q, alpha));
    Some(SE3::new(rotation, translation))
}

fn rotation_angle_rad(r: &SO3) -> f64 {
    let q = r.q.as_ref();
    let w = q.w.abs().clamp(-1.0, 1.0);
    2.0 * w.acos()
}

fn error_stats(values: &mut [f64]) -> ErrorStats {
    let n = values.len() as f64;
    let rmse = (values.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let mean = values.iter().sum::<f64>() / n;
    values.sort_by(|a, b| a.total_cmp(b));
    let median = if values.len() % 2 == 0 {
        0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
    } else {
        values[values.len() / 2]
    };
    let max = *values.last().unwrap();

    ErrorStats {
        rmse,
        mean,
        median,
        max,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use super::*;

    fn pose(x: f64, yaw: f64) -> SE3 {
        SE3::new(
            SO3::exp(&Vector3::new(0.0, 0.0, yaw)),
            Vector3::new(x, 0.0, 0.0),
        )
    }

    #[test]
    fn ate_metrics_are_zero_for_identical_interpolated_trajectory() {
        let gt = vec![
            StampedPose::new(0.0, pose(0.0, 0.0)),
            StampedPose::new(1.0, pose(1.0, 0.1)),
            StampedPose::new(2.0, pose(2.0, 0.2)),
        ];
        let est = vec![(0.5, pose(0.5, 0.05)), (1.5, pose(1.5, 0.15))];

        let metrics = compute_ate_metrics(&est, &gt, &SE3::identity()).unwrap();

        assert_eq!(metrics.matched_poses, 2);
        assert_abs_diff_eq!(metrics.ate_position_m.rmse, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(metrics.ate_attitude_deg.rmse, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ate_position_reports_translation_error_after_alignment() {
        let gt = vec![
            StampedPose::new(0.0, pose(0.0, 0.0)),
            StampedPose::new(1.0, pose(1.0, 0.0)),
        ];
        let est = vec![(0.5, pose(0.5, 0.0))];
        let alignment = SE3::new(SO3::identity(), Vector3::new(0.0, 1.0, 0.0));

        let metrics = compute_ate_metrics(&est, &gt, &alignment).unwrap();

        assert_abs_diff_eq!(metrics.ate_position_m.rmse, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(metrics.ate_position_m.max, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn attitude_error_uses_shortest_rotation_angle() {
        let almost_full_turn = SO3::exp(&Vector3::new(0.0, 0.0, std::f64::consts::TAU - 0.1));

        assert_abs_diff_eq!(rotation_angle_rad(&almost_full_turn), 0.1, epsilon = 1e-12);
    }
}
