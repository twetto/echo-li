use super::{PatchDepthSettings, SparseDepthPrior};

#[derive(Debug, Clone)]
pub(super) struct SeedGrid {
    pub(super) ids: Vec<usize>,
    pub(super) starts: Vec<usize>,
    pub(super) cols: usize,
    pub(super) rows: usize,
    pub(super) cell_size: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct NearbySeed {
    pub(super) idx: usize,
    pub(super) w_spatial: f64,
    pub(super) precision: f64,
}

#[derive(Debug, Clone)]
pub(super) struct NearbySeeds {
    len: usize,
    items: [NearbySeed; NearbySeeds::MAX],
}

impl NearbySeeds {
    const MAX: usize = 64;

    pub(super) fn new() -> Self {
        Self {
            len: 0,
            items: [NearbySeed::default(); Self::MAX],
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn push(&mut self, item: NearbySeed) -> bool {
        if self.len >= Self::MAX {
            return false;
        }
        self.items[self.len] = item;
        self.len += 1;
        true
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &NearbySeed> {
        self.items[..self.len].iter()
    }
}

impl SeedGrid {
    pub(super) fn new(
        seeds: &[SparseDepthPrior],
        cell_size: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let cell_size = cell_size.max(1.0);
        let cols = ((width as f64) / cell_size).ceil().max(1.0) as usize;
        let rows = ((height as f64) / cell_size).ceil().max(1.0) as usize;
        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); cols * rows];
        for (idx, seed) in seeds.iter().enumerate() {
            let ci = ((seed.uv[0] / cell_size) as usize).min(cols - 1);
            let cj = ((seed.uv[1] / cell_size) as usize).min(rows - 1);
            bins[cj * cols + ci].push(idx);
        }
        let mut ids = Vec::with_capacity(seeds.len());
        let mut starts = Vec::with_capacity(cols * rows + 1);
        starts.push(0);
        for bin in bins {
            ids.extend(bin);
            starts.push(ids.len());
        }
        Self {
            ids,
            starts,
            cols,
            rows,
            cell_size,
        }
    }
}

pub(super) fn median_seed_depth(seeds: &[SparseDepthPrior]) -> Option<f64> {
    if seeds.is_empty() {
        return None;
    }
    let mut depths: Vec<f64> = seeds
        .iter()
        .filter_map(|seed| (seed.rho > 0.0).then_some(1.0 / seed.rho))
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.total_cmp(b));
    Some(depths[depths.len() / 2])
}

pub(super) fn scale_seeds(seeds: &[SparseDepthPrior], scale: f64) -> Vec<SparseDepthPrior> {
    seeds
        .iter()
        .map(|seed| SparseDepthPrior {
            uv: seed.uv * scale,
            rho: seed.rho,
            rho_var: seed.rho_var,
        })
        .collect()
}

pub(super) fn nearby_seed_weights(
    cu: f64,
    cv: f64,
    seeds: &[SparseDepthPrior],
    seed_grid: &SeedGrid,
    settings: &PatchDepthSettings,
) -> NearbySeeds {
    let radius = seed_grid.cell_size;
    let radius_sq = radius * radius;
    let ci_min = ((cu - radius) / seed_grid.cell_size).floor().max(0.0) as usize;
    let cj_min = ((cv - radius) / seed_grid.cell_size).floor().max(0.0) as usize;
    let ci_max = ((cu + radius) / seed_grid.cell_size)
        .floor()
        .min((seed_grid.cols - 1) as f64) as usize;
    let cj_max = ((cv + radius) / seed_grid.cell_size)
        .floor()
        .min((seed_grid.rows - 1) as f64) as usize;
    let mut out = NearbySeeds::new();
    for cj in cj_min..=cj_max {
        for ci in ci_min..=ci_max {
            let cell_idx = cj * seed_grid.cols + ci;
            for k in seed_grid.starts[cell_idx]..seed_grid.starts[cell_idx + 1] {
                let idx = seed_grid.ids[k];
                let du = seeds[idx].uv[0] - cu;
                let dv = seeds[idx].uv[1] - cv;
                let dist_sq = du * du + dv * dv;
                if dist_sq > radius_sq {
                    continue;
                }
                let dist = dist_sq.sqrt();
                let w_spatial = 1.0 - dist / radius;
                let var_capped = seeds[idx]
                    .rho_var
                    .max(settings.sigma_seed_floor * settings.sigma_seed_floor);
                if !out.push(NearbySeed {
                    idx,
                    w_spatial,
                    precision: 1.0 / var_capped,
                }) {
                    return out;
                }
            }
        }
    }
    out
}
