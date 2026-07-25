use nalgebra::{DMatrix, DVector, Matrix2x3, RowVector3, SMatrix, Vector2, Vector3};

use crate::mathematical::camera::CameraModel;
use crate::mathematical::imu_velocity::IMUVelocity;
use crate::mathematical::vio_group::{VIOAlgebra, VIOGroup};
use crate::mathematical::vio_state::{VIOSensorState, VIOState};
use std::collections::HashMap;

pub struct RiccatiPropagationBlocks {
    pub a_ss: SMatrix<f64, 21, 21>,
    pub a_lm_s: DMatrix<f64>,
    pub a_lm_lm: Vec<SMatrix<f64, 3, 3>>,
    pub b_s: SMatrix<f64, 21, 12>,
    pub b_lm: DMatrix<f64>,
}

pub trait EqFCoordinateSuite: Send + Sync {
    fn state_chart(&self, xi: &VIOState, xi0: &VIOState) -> DVector<f64>;
    fn state_chart_inv(&self, eps: &DVector<f64>, xi0: &VIOState) -> VIOState;

    fn state_matrix_a(&self, x: &VIOGroup, xi0: &VIOState, imu_vel: &IMUVelocity) -> DMatrix<f64>;
    fn input_matrix_b(&self, x: &VIOGroup, xi0: &VIOState) -> DMatrix<f64>;

    fn propagation_blocks(
        &self,
        x: &VIOGroup,
        xi0: &VIOState,
        imu_vel: &IMUVelocity,
    ) -> RiccatiPropagationBlocks {
        let a = self.state_matrix_a(x, xi0, imu_vel);
        let b = self.input_matrix_b(x, xi0);
        let s = VIOSensorState::CDIM;
        let n_lm = xi0.camera_landmarks.len();

        let a_ss = a.fixed_view::<21, 21>(0, 0).into_owned();
        let mut a_lm_s = DMatrix::<f64>::zeros(3 * n_lm, s);
        let mut a_lm_lm = Vec::with_capacity(n_lm);
        for i in 0..n_lm {
            let row = s + 3 * i;
            a_lm_s
                .fixed_view_mut::<3, 21>(3 * i, 0)
                .copy_from(&a.fixed_view::<3, 21>(row, 0));
            a_lm_lm.push(a.fixed_view::<3, 3>(row, row).into_owned());
        }

        let b_s = b.fixed_view::<21, 12>(0, 0).into_owned();
        let mut b_lm = DMatrix::<f64>::zeros(3 * n_lm, 12);
        if n_lm > 0 {
            b_lm.copy_from(&b.view((s, 0), (3 * n_lm, 12)));
        }

        RiccatiPropagationBlocks {
            a_ss,
            a_lm_s,
            a_lm_lm,
            b_s,
            b_lm,
        }
    }

    fn output_matrix_ci_star(
        &self,
        q0: &Vector3<f64>,
        q_hat: &echo_lie::SOT3,
        cam: &dyn CameraModel,
        y: &Vector2<f64>,
    ) -> Matrix2x3<f64>;

    fn output_matrix_C(
        &self,
        xi0: &VIOState,
        x_hat: &VIOGroup,
        y_ids: &[u64],
        y_obs: &HashMap<u64, Vector2<f64>>,
        cam: &dyn CameraModel,
        use_equivariance: bool,
    ) -> DMatrix<f64> {
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
                c.fixed_view_mut::<2, 3>(2 * i, 21 + 3 * pos)
                    .copy_from(&ci_star);
            }
        }
        c
    }

    /// Log-inverse-range output row for the stereo measurement channel:
    /// `ℓ(q) = -log‖q‖`, `∂ℓ/∂(chart)` as a 1×3 row in this suite's landmark
    /// chart. Default is the Euclidean form `-q0ᵀ/‖q0‖²`; charts override by
    /// mapping through their `conv_*2euc`. See
    /// ECHO-LI-notes/docs/eqvio/stereo_output_matrix_derivation.md.
    fn output_range_row(&self, q0: &Vector3<f64>) -> RowVector3<f64> {
        -q0.transpose() / q0.norm_squared()
    }

    fn lift_innovation(&self, total_innovation: &DVector<f64>, xi0: &VIOState) -> VIOAlgebra;
    fn lift_innovation_discrete(&self, total_innovation: &DVector<f64>, xi0: &VIOState)
    -> VIOGroup;
}
