use nalgebra::{DMatrix, DVector, Matrix2x3, Vector2, Vector3};

use crate::mathematical::vio_state::VIOState;
use crate::mathematical::vio_group::{VIOGroup, VIOAlgebra};
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::camera::CameraModel;
use std::collections::HashMap;

pub trait EqFCoordinateSuite: Send + Sync {
    fn state_chart(&self, xi: &VIOState, xi0: &VIOState) -> DVector<f64>;
    fn state_chart_inv(&self, eps: &DVector<f64>, xi0: &VIOState) -> VIOState;

    fn state_matrix_a(&self, x: &VIOGroup, xi0: &VIOState, imu_vel: &IMUVelocity) -> DMatrix<f64>;
    fn input_matrix_b(&self, x: &VIOGroup, xi0: &VIOState) -> DMatrix<f64>;
    
    fn output_matrix_ci_star(&self, q0: &Vector3<f64>, q_hat: &echo_lie::SOT3, cam: &dyn CameraModel, y: &Vector2<f64>) -> Matrix2x3<f64>;

    fn output_matrix_C(&self, xi0: &VIOState, x_hat: &VIOGroup, y_ids: &[u64], y_obs: &HashMap<u64, Vector2<f64>>, cam: &dyn CameraModel, use_equivariance: bool) -> DMatrix<f64> {
        let dim = xi0.dim();
        let mut c = DMatrix::<f64>::zeros(2 * y_ids.len(), dim);

        for (i, &id) in y_ids.iter().enumerate() {
            if let Some(pos) = xi0.camera_landmarks.iter().position(|l| l.id == id) {
                let q0 = xi0.camera_landmarks[pos].p;
                let q_hat = &x_hat.q[pos];
                let uv = y_obs[&id];

                let ci_star = if use_equivariance {
                    self.output_matrix_ci_star(&q0, q_hat, cam, &uv)
                } else {
                    // Non-equivariant: evaluate C* at the predicted measurement
                    // so the averaging in ci_star is a no-op (y_hat = y_tru).
                    let p_c = q_hat.inverse().act(&q0);
                    let y_hat = cam.project(&p_c);
                    self.output_matrix_ci_star(&q0, q_hat, cam, &y_hat)
                };
                c.fixed_view_mut::<2, 3>(2 * i, 21 + 3 * pos).copy_from(&ci_star);
            }
        }
        c
    }

    fn lift_innovation(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOAlgebra;
    fn lift_innovation_discrete(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOGroup;
}
