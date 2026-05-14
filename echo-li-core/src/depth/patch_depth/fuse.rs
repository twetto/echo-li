use super::{PatchDepthOutput, PatchDepthSettings, PatchEstimate, PatchStatus};
use crate::core_types::DepthMap;

#[cfg(feature = "parallel")]
pub(super) fn patch_centers(
    width: usize,
    height: usize,
    settings: &PatchDepthSettings,
) -> Vec<(usize, usize)> {
    let half = settings.patch_size / 2;
    let u_count = width
        .saturating_sub(half)
        .saturating_sub(half)
        .saturating_add(settings.patch_stride - 1)
        / settings.patch_stride;
    let v_count = height
        .saturating_sub(half)
        .saturating_sub(half)
        .saturating_add(settings.patch_stride - 1)
        / settings.patch_stride;
    let mut centers = Vec::with_capacity(u_count * v_count);
    for v in (half..height.saturating_sub(half)).step_by(settings.patch_stride) {
        for u in (half..width.saturating_sub(half)).step_by(settings.patch_stride) {
            centers.push((u, v));
        }
    }
    centers
}

pub(super) struct FuseAccumulator {
    n_cells_u: usize,
    n_cells_v: usize,
    rho_acc: Vec<f64>,
    w_acc: Vec<f64>,
    status: Vec<PatchStatus>,
}

impl FuseAccumulator {
    pub(super) fn new(width: usize, height: usize, settings: &PatchDepthSettings) -> Self {
        let n_cells_u = (width / settings.cell_size).max(1);
        let n_cells_v = (height / settings.cell_size).max(1);
        let n = n_cells_u * n_cells_v;
        Self {
            n_cells_u,
            n_cells_v,
            rho_acc: vec![0.0; n],
            w_acc: vec![0.0; n],
            status: vec![PatchStatus::Unknown; n],
        }
    }

    pub(super) fn add(
        &mut self,
        cu: f64,
        cv: f64,
        estimate: PatchEstimate,
        settings: &PatchDepthSettings,
    ) {
        if estimate.status == PatchStatus::Unknown || estimate.status == PatchStatus::Rejected {
            return;
        }
        let ci = (cu as usize / settings.cell_size).min(self.n_cells_u - 1);
        let cj = (cv as usize / settings.cell_size).min(self.n_cells_v - 1);
        let idx = cj * self.n_cells_u + ci;
        let status_weight = match estimate.status {
            PatchStatus::PhotoRefined => settings.status_weight_photo,
            PatchStatus::SeedOnly => settings.status_weight_seed,
            _ => 0.0,
        };
        let w = status_weight / estimate.var.max(settings.var_floor);
        self.rho_acc[idx] += w * estimate.rho;
        self.w_acc[idx] += w;
        if (estimate.status as u8) > (self.status[idx] as u8) {
            self.status[idx] = estimate.status;
        }
    }

    pub(super) fn finish(self) -> PatchDepthOutput {
        let mut depth = vec![f32::NAN; self.rho_acc.len()];
        let mut variance = vec![f32::INFINITY; self.rho_acc.len()];
        for i in 0..self.rho_acc.len() {
            if self.w_acc[i] > 0.0 {
                depth[i] = (1.0 / (self.rho_acc[i] / self.w_acc[i])) as f32;
                variance[i] = (1.0 / self.w_acc[i]) as f32;
            }
        }

        PatchDepthOutput {
            depth_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, depth)
                .expect("depth cell size"),
            variance_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, variance)
                .expect("variance cell size"),
            status_cells: DepthMap::from_vec(self.n_cells_u, self.n_cells_v, self.status)
                .expect("status cell size"),
        }
    }
}

#[cfg(feature = "parallel")]
pub(super) fn fuse_cells(
    patches: &[(f64, f64, PatchEstimate)],
    width: usize,
    height: usize,
    settings: &PatchDepthSettings,
) -> PatchDepthOutput {
    let mut fuse = FuseAccumulator::new(width, height, settings);
    for &(cu, cv, estimate) in patches {
        fuse.add(cu, cv, estimate, settings);
    }
    fuse.finish()
}
