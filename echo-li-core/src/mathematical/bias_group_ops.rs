use echo_lie::SE3;
use nalgebra::{SMatrix, Vector6};

use crate::mathematical::vio_group::{VIOAlgebra, VIOGroup};
use crate::ImuBiasGroup;

#[derive(Debug, Clone, Copy)]
pub struct BiasGroupOps {
    kind: ImuBiasGroup,
}

impl BiasGroupOps {
    pub fn new(kind: ImuBiasGroup) -> Self {
        Self { kind }
    }

    pub fn kind(self) -> ImuBiasGroup {
        self.kind
    }

    pub fn bias_action_matrix(x: &VIOGroup) -> SE3 {
        SE3::new(x.a.rotation.clone(), x.w)
    }

    pub fn compose_beta(self, x: &VIOGroup, other_beta: &Vector6<f64>) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => x.beta + other_beta,
            ImuBiasGroup::SemiDirect => x.beta + Self::bias_action_matrix(x).adjoint() * other_beta,
        }
    }

    pub fn inverse_beta(self, x: &VIOGroup) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => -x.beta,
            ImuBiasGroup::SemiDirect => -(Self::bias_action_matrix(x).inverse().adjoint() * x.beta),
        }
    }

    pub fn act_bias(self, x: &VIOGroup, bias: &Vector6<f64>) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => bias + x.beta,
            ImuBiasGroup::SemiDirect => {
                Self::bias_action_matrix(x).inverse().adjoint() * (bias + x.beta)
            }
        }
    }

    pub fn physical_bias_noise_matrix(self, x: &VIOGroup) -> SMatrix<f64, 6, 6> {
        match self.kind {
            ImuBiasGroup::Additive => SMatrix::<f64, 6, 6>::identity(),
            ImuBiasGroup::SemiDirect => Self::bias_action_matrix(x).adjoint(),
        }
    }

    pub fn beta_for_physical_bias_update(
        self,
        current: &VIOGroup,
        delta_geometry: &VIOGroup,
        bias_origin: &Vector6<f64>,
        physical_bias_delta: &Vector6<f64>,
    ) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => *physical_bias_delta,
            ImuBiasGroup::SemiDirect => {
                let ad_delta = Self::bias_action_matrix(delta_geometry).adjoint();
                let ad_current = Self::bias_action_matrix(current).adjoint();
                (ad_delta - SMatrix::<f64, 6, 6>::identity()) * bias_origin
                    + ad_delta * ad_current * physical_bias_delta
            }
        }
    }

    pub fn exp_beta(self, lam: &VIOAlgebra) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => lam.u_beta,
            ImuBiasGroup::SemiDirect => {
                let mut b_tangent = Vector6::zeros();
                b_tangent
                    .fixed_rows_mut::<3>(0)
                    .copy_from(&lam.u_a.fixed_rows::<3>(0));
                b_tangent.fixed_rows_mut::<3>(3).copy_from(&lam.u_w);
                SE3::left_jacobian(&b_tangent) * lam.u_beta
            }
        }
    }

    pub fn observer_increment_beta(
        self,
        lifted: &VIOGroup,
        state_bias: &Vector6<f64>,
        additive_bias_delta: &Vector6<f64>,
    ) -> Vector6<f64> {
        match self.kind {
            ImuBiasGroup::Additive => lifted.beta,
            ImuBiasGroup::SemiDirect => {
                Self::bias_action_matrix(lifted).adjoint() * (state_bias + additive_bias_delta)
                    - state_bias
            }
        }
    }
}
