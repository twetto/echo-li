use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

use crate::core_types::CameraIntrinsics;
use crate::depth::patch_depth::{PatchDepthOutput, PatchDepthSeedCoordinates, PatchStatus};
use crate::mathematical::camera::CameraModel;

/// How a ray writes evidence into the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OccupancyUpdateMode {
    /// **v0** — fixed `log_odds_hit`/`log_odds_miss` per voxel; `var(η)` is used
    /// only as a gate. Robust and calibration-independent.
    #[default]
    FixedIncrement,
    /// **v1** — σ-shaped inverse sensor model. The metric band
    /// `σ_m = range·sqrt(var(η))` sets the width of a Gaussian occupied bump
    /// centred at the measured range, with confidence-weighted carve/hit
    /// magnitudes, so uncertain rays smear weak evidence and confident rays
    /// write a sharp surface. Uses the *absolute* scale of `var(η)` as a metric
    /// σ (see docs/3d_mapping_for_navigation.md §"Uncertainty-aware update").
    UncertaintyAware,
}

#[derive(Debug, Clone)]
pub struct LocalOccupancySettings {
    pub enabled: bool,
    pub resolution: f64,
    pub width_cells: usize,
    pub height_cells: usize,
    pub sample_stride: usize,
    pub min_range: f64,
    pub max_range: f64,
    pub max_eta_std: f64,
    pub log_odds_hit: f32,
    pub log_odds_miss: f32,
    pub log_odds_min: f32,
    pub log_odds_max: f32,
    pub occupied_threshold: f32,
    pub free_threshold: f32,
    /// Fixed-increment (v0) vs σ-shaped (v1) ray integration.
    pub update_mode: OccupancyUpdateMode,
    /// v1 only — band half-width in units of σ_m. The occupied bump spans
    /// `range ± band_k·σ_m`; beyond it the ray is treated as occluded (no
    /// update), before it as free.
    pub band_k: f64,
    /// v1 only — σ floor as a fraction of `resolution`. Caps how sharp the
    /// surface can get: `σ_m = max(range·sqrt(var(η)), sigma_floor_factor·res)`.
    /// Below the floor the bump would collapse to sub-voxel and is meaningless.
    pub sigma_floor_factor: f64,
    /// v1 only — lower clamp on the confidence weight `w = sigma_floor/σ_m ∈
    /// (0,1]` that scales both carve and hit magnitudes, so a very uncertain ray
    /// still contributes a little rather than nothing.
    pub min_confidence_weight: f64,
    /// Vertical extent of the (ego-centric) 3D grid, **relative to the camera**:
    /// the grid spans `cam_z + [min_obstacle_height, max_obstacle_height]`,
    /// discretised into `round((max - min) / resolution)` z-layers. The map is a
    /// true 3D voxel grid; these two fields only size its vertical extent.
    pub min_obstacle_height: f64,
    pub max_obstacle_height: f64,
}

impl Default for LocalOccupancySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution: 0.10,
            width_cells: 160,
            height_cells: 160,
            sample_stride: 4,
            min_range: 0.25,
            max_range: 12.0,
            max_eta_std: 0.50,
            log_odds_hit: 0.85,
            log_odds_miss: -0.35,
            log_odds_min: -4.0,
            log_odds_max: 4.0,
            occupied_threshold: 0.8,
            free_threshold: -0.8,
            update_mode: OccupancyUpdateMode::FixedIncrement,
            band_k: 2.0,
            sigma_floor_factor: 0.5,
            min_confidence_weight: 0.1,
            min_obstacle_height: -1.0,
            max_obstacle_height: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyCell {
    Unknown,
    Free,
    Occupied,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OccupancyUpdateStats {
    pub rays_considered: usize,
    pub rays_integrated: usize,
    pub occupied_updates: usize,
    pub free_updates: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyGridSnapshot {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub origin_z: f64,
    pub log_odds: Vec<f32>,
}

/// Ego-centric, scrolling **3D** log-odds occupancy grid fed by dense
/// patch-depth rays. Free space is carved along each ray with a 3D
/// Amanatides–Woo traversal; the ray endpoint marks an occupied voxel.
#[derive(Debug, Clone)]
pub struct LocalOccupancyMap {
    settings: LocalOccupancySettings,
    depth_cells: usize,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    center_ix: i64,
    center_iy: i64,
    center_iz: i64,
    log_odds: Vec<f32>,
}

impl LocalOccupancyMap {
    pub fn new(settings: LocalOccupancySettings) -> anyhow::Result<Self> {
        anyhow::ensure!(
            settings.resolution > 0.0,
            "occupancy resolution must be positive"
        );
        anyhow::ensure!(
            settings.width_cells > 0 && settings.height_cells > 0,
            "occupancy grid dimensions must be non-zero"
        );
        anyhow::ensure!(
            settings.sample_stride > 0,
            "occupancy sample_stride must be non-zero"
        );
        anyhow::ensure!(
            settings.max_range > settings.min_range,
            "occupancy max_range must be greater than min_range"
        );
        anyhow::ensure!(
            settings.max_obstacle_height > settings.min_obstacle_height,
            "occupancy vertical extent (max_obstacle_height - min_obstacle_height) must be positive"
        );
        let depth_cells = (((settings.max_obstacle_height - settings.min_obstacle_height)
            / settings.resolution)
            .round() as usize)
            .max(1);
        let w = settings.width_cells;
        let h = settings.height_cells;
        let n = w * h * depth_cells;
        let origin_x = -(w as f64) * 0.5 * settings.resolution;
        let origin_y = -(h as f64) * 0.5 * settings.resolution;
        let origin_z = -(depth_cells as f64) * 0.5 * settings.resolution;
        Ok(Self {
            settings,
            depth_cells,
            origin_x,
            origin_y,
            origin_z,
            center_ix: 0,
            center_iy: 0,
            center_iz: 0,
            log_odds: vec![0.0; n],
        })
    }

    pub fn settings(&self) -> &LocalOccupancySettings {
        &self.settings
    }

    /// Number of vertical (z) layers in the grid.
    pub fn depth_cells(&self) -> usize {
        self.depth_cells
    }

    pub fn snapshot(&self) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot {
            width: self.settings.width_cells,
            height: self.settings.height_cells,
            depth: self.depth_cells,
            resolution: self.settings.resolution,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            origin_z: self.origin_z,
            log_odds: self.log_odds.clone(),
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.settings.height_cells + y) * self.settings.width_cells + x
    }

    pub fn cell_state(&self, x: usize, y: usize, z: usize) -> Option<OccupancyCell> {
        if x >= self.settings.width_cells
            || y >= self.settings.height_cells
            || z >= self.depth_cells
        {
            return None;
        }
        let value = self.log_odds[self.index(x, y, z)];
        Some(if value >= self.settings.occupied_threshold {
            OccupancyCell::Occupied
        } else if value <= self.settings.free_threshold {
            OccupancyCell::Free
        } else {
            OccupancyCell::Unknown
        })
    }

    /// Returns `(unknown, free, occupied)` voxel counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut unknown = 0;
        let mut free = 0;
        let mut occupied = 0;
        for &value in &self.log_odds {
            if value >= self.settings.occupied_threshold {
                occupied += 1;
            } else if value <= self.settings.free_threshold {
                free += 1;
            } else {
                unknown += 1;
            }
        }
        (unknown, free, occupied)
    }

    pub fn update_from_patch_depth(
        &mut self,
        output: &PatchDepthOutput,
        camera: &dyn CameraModel,
        intrinsics: CameraIntrinsics,
        coordinates: PatchDepthSeedCoordinates,
        image_width: usize,
        image_height: usize,
        t_wc: &Matrix4<f64>,
    ) -> OccupancyUpdateStats {
        self.recenter(t_wc[(0, 3)], t_wc[(1, 3)], t_wc[(2, 3)]);

        let mut stats = OccupancyUpdateStats::default();
        if output.eta.width == 0 || output.eta.height == 0 || image_width == 0 || image_height == 0
        {
            return stats;
        }

        let cam_origin = Vector3::new(t_wc[(0, 3)], t_wc[(1, 3)], t_wc[(2, 3)]);
        let sx = image_width as f64 / output.eta.width as f64;
        let sy = image_height as f64 / output.eta.height as f64;

        for y in (0..output.eta.height).step_by(self.settings.sample_stride) {
            for x in (0..output.eta.width).step_by(self.settings.sample_stride) {
                stats.rays_considered += 1;
                let idx = y * output.eta.width + x;
                if !matches!(
                    output.status.data[idx],
                    PatchStatus::SeedOnly | PatchStatus::PhotoRefined
                ) {
                    continue;
                }
                let eta = output.eta.data[idx];
                let eta_var = output.eta_var.data[idx];
                if !eta.is_finite() || !eta_var.is_finite() || eta_var < 0.0 {
                    continue;
                }
                // sqrt(var(η)) ≈ σ_range/range is the relative range std. v0 uses
                // it as a gate only; v1 turns it into the metric band σ_m below.
                let rel_std = (eta_var as f64).sqrt();
                if rel_std > self.settings.max_eta_std {
                    continue;
                }

                let range = (eta as f64).exp();
                if range < self.settings.min_range || range > self.settings.max_range {
                    continue;
                }

                let uv = Vector2::new((x as f64 + 0.5) * sx - 0.5, (y as f64 + 0.5) * sy - 0.5);
                let bearing_c = bearing_for_pixel(camera, intrinsics, coordinates, &uv);
                if !bearing_c.iter().all(|v| v.is_finite()) {
                    continue;
                }
                let p_c = bearing_c * range;
                let p_w_h = t_wc * Vector4::new(p_c[0], p_c[1], p_c[2], 1.0);
                let endpoint = Vector3::new(p_w_h[0], p_w_h[1], p_w_h[2]);

                stats.rays_integrated += 1;
                match self.settings.update_mode {
                    OccupancyUpdateMode::FixedIncrement => {
                        self.integrate_ray(cam_origin, endpoint, &mut stats);
                    }
                    OccupancyUpdateMode::UncertaintyAware => {
                        self.integrate_ray_sigma(cam_origin, endpoint, rel_std, &mut stats);
                    }
                }
            }
        }
        stats
    }

    fn recenter(&mut self, center_x: f64, center_y: f64, center_z: f64) {
        let res = self.settings.resolution;
        let new_center_ix = (center_x / res).floor() as i64;
        let new_center_iy = (center_y / res).floor() as i64;
        let new_center_iz = (center_z / res).floor() as i64;
        let dx = new_center_ix - self.center_ix;
        let dy = new_center_iy - self.center_iy;
        let dz = new_center_iz - self.center_iz;
        if dx == 0 && dy == 0 && dz == 0 {
            return;
        }

        let w = self.settings.width_cells;
        let h = self.settings.height_cells;
        let d = self.depth_cells;
        let mut shifted = vec![0.0; self.log_odds.len()];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let src_x = x as i64 + dx;
                    let src_y = y as i64 + dy;
                    let src_z = z as i64 + dz;
                    if src_x >= 0
                        && src_x < w as i64
                        && src_y >= 0
                        && src_y < h as i64
                        && src_z >= 0
                        && src_z < d as i64
                    {
                        shifted[self.index(x, y, z)] = self.log_odds
                            [self.index(src_x as usize, src_y as usize, src_z as usize)];
                    }
                }
            }
        }
        self.log_odds = shifted;
        self.center_ix = new_center_ix;
        self.center_iy = new_center_iy;
        self.center_iz = new_center_iz;
        self.origin_x = (self.center_ix as f64 - w as f64 * 0.5) * res;
        self.origin_y = (self.center_iy as f64 - h as f64 * 0.5) * res;
        self.origin_z = (self.center_iz as f64 - d as f64 * 0.5) * res;
    }

    /// 3D Amanatides–Woo voxel traversal from the camera to the ray endpoint.
    /// Carves free space for every in-bounds voxel up to the boundary, and marks
    /// the endpoint occupied only if it is itself in bounds (clip-and-carve: an
    /// off-grid endpoint still contributes its near free space).
    fn integrate_ray(
        &mut self,
        origin: Vector3<f64>,
        endpoint: Vector3<f64>,
        stats: &mut OccupancyUpdateStats,
    ) {
        let res = self.settings.resolution;
        // Continuous voxel-space coordinates of the ray.
        let p0 = [
            (origin.x - self.origin_x) / res,
            (origin.y - self.origin_y) / res,
            (origin.z - self.origin_z) / res,
        ];
        let p1 = [
            (endpoint.x - self.origin_x) / res,
            (endpoint.y - self.origin_y) / res,
            (endpoint.z - self.origin_z) / res,
        ];
        let dir = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];

        let mut cell = [
            p0[0].floor() as i64,
            p0[1].floor() as i64,
            p0[2].floor() as i64,
        ];
        let end = [
            p1[0].floor() as i64,
            p1[1].floor() as i64,
            p1[2].floor() as i64,
        ];
        let endpoint_in_bounds = self.cell_in_bounds(end[0], end[1], end[2]);

        // Per-axis Amanatides–Woo: (step, t to first boundary, t to cross a voxel).
        let setup = |d: f64, i: i64, p: f64| -> (i64, f64, f64) {
            if d > 0.0 {
                (1, ((i + 1) as f64 - p) / d, 1.0 / d)
            } else if d < 0.0 {
                (-1, (i as f64 - p) / d, -1.0 / d)
            } else {
                (0, f64::INFINITY, f64::INFINITY)
            }
        };
        let (step_x, mut tmax_x, tdelta_x) = setup(dir[0], cell[0], p0[0]);
        let (step_y, mut tmax_y, tdelta_y) = setup(dir[1], cell[1], p0[1]);
        let (step_z, mut tmax_z, tdelta_z) = setup(dir[2], cell[2], p0[2]);

        let mut entered = false;
        loop {
            if cell == end {
                if endpoint_in_bounds {
                    self.add_log_odds(cell[0], cell[1], cell[2], self.settings.log_odds_hit);
                    stats.occupied_updates += 1;
                }
                break;
            }
            if self.cell_in_bounds(cell[0], cell[1], cell[2]) {
                self.add_log_odds(cell[0], cell[1], cell[2], self.settings.log_odds_miss);
                stats.free_updates += 1;
                entered = true;
            } else if entered {
                // Left the grid; a straight ray will not re-enter.
                break;
            }
            // Safety: stop once the next crossing is past the endpoint (t > 1).
            if tmax_x.min(tmax_y).min(tmax_z) > 1.0 {
                break;
            }
            if tmax_x <= tmax_y && tmax_x <= tmax_z {
                cell[0] += step_x;
                tmax_x += tdelta_x;
            } else if tmax_y <= tmax_z {
                cell[1] += step_y;
                tmax_y += tdelta_y;
            } else {
                cell[2] += step_z;
                tmax_z += tdelta_z;
            }
        }
    }

    /// **v1** σ-shaped integration. Walks the ray with the same 3D
    /// Amanatides–Woo traversal as [`integrate_ray`], but the per-voxel
    /// increment is an inverse sensor model shaped by the calibrated metric band
    /// `σ_m = range·sqrt(var(η))` rather than a fixed hit/miss:
    ///
    /// * `d < range − band_k·σ_m` → free: `log_odds_miss · w`
    /// * `|d − range| ≤ band_k·σ_m` → occupied: `log_odds_hit · w · exp(−½(z/σ_m)²)`
    /// * `d > range + band_k·σ_m` → occluded: no update
    ///
    /// where `d` is the along-ray distance to the voxel centre, `z = d − range`,
    /// and `w = clamp(sigma_floor/σ_m, min_confidence_weight, 1)` is the
    /// confidence weight (1 for a depth at the σ floor, shrinking as the band
    /// widens). The traversal is extended to `range + band_k·σ_m` so the far
    /// half of the occupied bump is written.
    fn integrate_ray_sigma(
        &mut self,
        origin: Vector3<f64>,
        endpoint: Vector3<f64>,
        rel_std: f64,
        stats: &mut OccupancyUpdateStats,
    ) {
        let res = self.settings.resolution;
        let diff = endpoint - origin;
        let range = diff.norm();
        if range <= 0.0 {
            return;
        }
        let unit = diff / range;
        let sigma_floor = res * self.settings.sigma_floor_factor;
        let sigma_m = (range * rel_std).max(sigma_floor);
        let half_band = self.settings.band_k * sigma_m;
        let w = (sigma_floor / sigma_m).clamp(self.settings.min_confidence_weight, 1.0);
        // Traverse up to the far edge of the occupied band so both sides of the
        // Gaussian bump are written; beyond it the ray is occluded.
        let far = origin + unit * (range + half_band);

        let to_voxel = |p: &Vector3<f64>| {
            [
                (p.x - self.origin_x) / res,
                (p.y - self.origin_y) / res,
                (p.z - self.origin_z) / res,
            ]
        };
        let p0 = to_voxel(&origin);
        let p1 = to_voxel(&far);
        let dir = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];

        let mut cell = [
            p0[0].floor() as i64,
            p0[1].floor() as i64,
            p0[2].floor() as i64,
        ];
        let end = [
            p1[0].floor() as i64,
            p1[1].floor() as i64,
            p1[2].floor() as i64,
        ];

        let setup = |d: f64, i: i64, p: f64| -> (i64, f64, f64) {
            if d > 0.0 {
                (1, ((i + 1) as f64 - p) / d, 1.0 / d)
            } else if d < 0.0 {
                (-1, (i as f64 - p) / d, -1.0 / d)
            } else {
                (0, f64::INFINITY, f64::INFINITY)
            }
        };
        let (step_x, mut tmax_x, tdelta_x) = setup(dir[0], cell[0], p0[0]);
        let (step_y, mut tmax_y, tdelta_y) = setup(dir[1], cell[1], p0[1]);
        let (step_z, mut tmax_z, tdelta_z) = setup(dir[2], cell[2], p0[2]);

        let mut entered = false;
        loop {
            if self.cell_in_bounds(cell[0], cell[1], cell[2]) {
                let center = Vector3::new(
                    self.origin_x + (cell[0] as f64 + 0.5) * res,
                    self.origin_y + (cell[1] as f64 + 0.5) * res,
                    self.origin_z + (cell[2] as f64 + 0.5) * res,
                );
                let d = (center - origin).dot(&unit);
                let z = d - range;
                if z <= half_band {
                    let delta = if z < -half_band {
                        self.settings.log_odds_miss as f64 * w
                    } else {
                        let g = (-0.5 * (z / sigma_m).powi(2)).exp();
                        self.settings.log_odds_hit as f64 * w * g
                    };
                    self.add_log_odds(cell[0], cell[1], cell[2], delta as f32);
                    if z < -half_band {
                        stats.free_updates += 1;
                    } else {
                        stats.occupied_updates += 1;
                    }
                }
                entered = true;
            } else if entered {
                break;
            }
            if cell == end {
                break;
            }
            if tmax_x.min(tmax_y).min(tmax_z) > 1.0 {
                break;
            }
            if tmax_x <= tmax_y && tmax_x <= tmax_z {
                cell[0] += step_x;
                tmax_x += tdelta_x;
            } else if tmax_y <= tmax_z {
                cell[1] += step_y;
                tmax_y += tdelta_y;
            } else {
                cell[2] += step_z;
                tmax_z += tdelta_z;
            }
        }
    }

    fn add_log_odds(&mut self, x: i64, y: i64, z: i64, delta: f32) {
        if !self.cell_in_bounds(x, y, z) {
            return;
        }
        let idx = self.index(x as usize, y as usize, z as usize);
        self.log_odds[idx] = (self.log_odds[idx] + delta)
            .clamp(self.settings.log_odds_min, self.settings.log_odds_max);
    }

    fn cell_in_bounds(&self, x: i64, y: i64, z: i64) -> bool {
        x >= 0
            && x < self.settings.width_cells as i64
            && y >= 0
            && y < self.settings.height_cells as i64
            && z >= 0
            && z < self.depth_cells as i64
    }
}

fn bearing_for_pixel(
    camera: &dyn CameraModel,
    intrinsics: CameraIntrinsics,
    coordinates: PatchDepthSeedCoordinates,
    uv: &Vector2<f64>,
) -> Vector3<f64> {
    match coordinates {
        PatchDepthSeedCoordinates::RawDistorted => camera.undistort(uv),
        PatchDepthSeedCoordinates::UndistortedPinhole => intrinsics.normalize_pixel(uv).normalize(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_types::DepthMap;
    use crate::depth::patch_depth::PatchDepthOutput;
    use crate::mathematical::camera::PinholeModel;

    fn single_depth_output(range: f32) -> PatchDepthOutput {
        single_depth_output_var(range, 0.01)
    }

    fn single_depth_output_var(range: f32, eta_var: f32) -> PatchDepthOutput {
        PatchDepthOutput {
            eta: DepthMap::from_vec(1, 1, vec![range.ln()]).unwrap(),
            eta_var: DepthMap::from_vec(1, 1, vec![eta_var]).unwrap(),
            status: DepthMap::from_vec(1, 1, vec![PatchStatus::PhotoRefined]).unwrap(),
        }
    }

    // Camera looking along +Y_world in a planar (z=0) configuration. With
    // min/max_obstacle_height = -1/1 and resolution 1, the grid has 2 z-layers
    // and the z=0 plane falls in layer kz=1.
    fn y_forward_pose() -> Matrix4<f64> {
        let mut pose = Matrix4::identity();
        pose[(1, 1)] = 0.0;
        pose[(1, 2)] = 1.0;
        pose[(2, 1)] = 1.0;
        pose[(2, 2)] = 0.0;
        pose
    }

    #[test]
    fn integrates_endpoint_and_free_space() {
        let settings = LocalOccupancySettings {
            enabled: true,
            resolution: 1.0,
            width_cells: 11,
            height_cells: 11,
            sample_stride: 1,
            log_odds_hit: 1.0,
            log_odds_miss: -1.0,
            occupied_threshold: 0.5,
            free_threshold: -0.5,
            ..Default::default()
        };
        let mut map = LocalOccupancyMap::new(settings).unwrap();
        let camera = PinholeModel {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let pose = y_forward_pose();
        let intrinsics = CameraIntrinsics::new(1.0, 1.0, 0.0, 0.0);
        let stats = map.update_from_patch_depth(
            &single_depth_output(3.0),
            &camera,
            intrinsics,
            PatchDepthSeedCoordinates::UndistortedPinhole,
            1,
            1,
            &pose,
        );

        assert_eq!(stats.rays_integrated, 1);
        assert_eq!(map.cell_state(5, 5, 1), Some(OccupancyCell::Free));
        assert_eq!(map.cell_state(5, 8, 1), Some(OccupancyCell::Occupied));
    }

    #[test]
    fn off_grid_endpoint_carves_free_without_hit() {
        // Endpoint beyond the grid extent: the ray must still carve free space up
        // to the boundary and mark NO occupied cell (the obstacle is off-map).
        let settings = LocalOccupancySettings {
            enabled: true,
            resolution: 1.0,
            width_cells: 11,
            height_cells: 11,
            sample_stride: 1,
            log_odds_hit: 1.0,
            log_odds_miss: -1.0,
            occupied_threshold: 0.5,
            free_threshold: -0.5,
            max_range: 20.0, // let the 8 m range pass the range gate
            ..Default::default()
        };
        let mut map = LocalOccupancyMap::new(settings).unwrap();
        let camera = PinholeModel {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let pose = y_forward_pose();
        let intrinsics = CameraIntrinsics::new(1.0, 1.0, 0.0, 0.0);

        // range 8 -> endpoint cell (5, 13, 1), outside the 11x11 grid (max idx 10).
        let stats = map.update_from_patch_depth(
            &single_depth_output(8.0),
            &camera,
            intrinsics,
            PatchDepthSeedCoordinates::UndistortedPinhole,
            1,
            1,
            &pose,
        );

        assert_eq!(stats.rays_integrated, 1);
        assert_eq!(
            stats.occupied_updates, 0,
            "off-grid endpoint must not mark a hit"
        );
        assert!(
            stats.free_updates > 0,
            "should still carve free space up to the boundary"
        );
        assert_eq!(map.cell_state(5, 5, 1), Some(OccupancyCell::Free)); // camera cell
        assert_eq!(map.cell_state(5, 10, 1), Some(OccupancyCell::Free)); // last in-bounds cell
    }

    #[test]
    fn recenters_without_losing_overlap() {
        let settings = LocalOccupancySettings {
            enabled: true,
            resolution: 1.0,
            width_cells: 7,
            height_cells: 7,
            sample_stride: 1,
            log_odds_hit: 1.0,
            occupied_threshold: 0.5,
            ..Default::default()
        };
        let mut map = LocalOccupancyMap::new(settings).unwrap();
        map.add_log_odds(4, 3, 1, 1.0);

        map.recenter(1.0, 0.0, 0.0);

        assert_eq!(map.cell_state(3, 3, 1), Some(OccupancyCell::Occupied));
    }

    fn v1_settings() -> LocalOccupancySettings {
        LocalOccupancySettings {
            enabled: true,
            resolution: 1.0,
            width_cells: 11,
            height_cells: 11,
            sample_stride: 1,
            update_mode: OccupancyUpdateMode::UncertaintyAware,
            band_k: 2.0,
            sigma_floor_factor: 0.5,
            min_confidence_weight: 0.1,
            max_eta_std: 1.0,
            log_odds_hit: 2.0,
            log_odds_miss: -2.0,
            occupied_threshold: 0.5,
            free_threshold: -0.5,
            ..Default::default()
        }
    }

    fn run_single_ray(
        map: &mut LocalOccupancyMap,
        output: &PatchDepthOutput,
    ) -> OccupancyUpdateStats {
        let camera = PinholeModel {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let intrinsics = CameraIntrinsics::new(1.0, 1.0, 0.0, 0.0);
        map.update_from_patch_depth(
            output,
            &camera,
            intrinsics,
            PatchDepthSeedCoordinates::UndistortedPinhole,
            1,
            1,
            &y_forward_pose(),
        )
    }

    #[test]
    fn v1_marks_surface_band_and_carves_free() {
        // Confident depth (rel_std 0.1): σ_m floors at 0.5 m, band ±1 m about the
        // range-3 surface. The surface voxel is occupied; the camera voxel is free.
        let mut map = LocalOccupancyMap::new(v1_settings()).unwrap();
        let stats = run_single_ray(&mut map, &single_depth_output(3.0));

        assert_eq!(stats.rays_integrated, 1);
        assert!(stats.occupied_updates > 0 && stats.free_updates > 0);
        assert_eq!(map.cell_state(5, 8, 1), Some(OccupancyCell::Occupied)); // surface
        assert_eq!(map.cell_state(5, 5, 1), Some(OccupancyCell::Free)); // camera
    }

    #[test]
    fn v1_uncertain_ray_writes_weaker_and_more_spread_occupied() {
        // The v1 ablation claim: an uncertain depth writes a *weaker* peak at the
        // surface (fewer false-confident occupied cells) but spreads more evidence
        // into neighbouring cells than a confident depth of the same range.
        let surface = 214; // index(5, 8, 1)
        let beyond = 225; // index(5, 9, 1), one voxel past the surface

        let mut confident = LocalOccupancyMap::new(v1_settings()).unwrap();
        run_single_ray(&mut confident, &single_depth_output_var(3.0, 0.0001));
        let conf = confident.snapshot().log_odds;

        let mut uncertain = LocalOccupancyMap::new(v1_settings()).unwrap();
        run_single_ray(&mut uncertain, &single_depth_output_var(3.0, 0.25)); // rel_std 0.5
        let unc = uncertain.snapshot().log_odds;

        assert!(
            conf[surface] > unc[surface],
            "confident depth must write a stronger surface peak ({} vs {})",
            conf[surface],
            unc[surface]
        );
        assert!(
            unc[beyond] > conf[beyond],
            "uncertain depth must spread more evidence past the surface ({} vs {})",
            unc[beyond],
            conf[beyond]
        );
    }

    #[test]
    fn vertical_extent_resolves_into_layers() {
        // 4 m vertical extent at 1 m resolution -> 4 z-layers (true 3D, not BEV).
        let settings = LocalOccupancySettings {
            resolution: 1.0,
            width_cells: 11,
            height_cells: 11,
            min_obstacle_height: -2.0,
            max_obstacle_height: 2.0,
            ..Default::default()
        };
        let map = LocalOccupancyMap::new(settings).unwrap();
        assert_eq!(map.depth_cells(), 4);
        assert_eq!(map.snapshot().log_odds.len(), 11 * 11 * 4);
    }
}
