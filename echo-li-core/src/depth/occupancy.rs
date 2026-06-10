use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

use crate::core_types::CameraIntrinsics;
use crate::depth::patch_depth::{PatchDepthOutput, PatchDepthSeedCoordinates, PatchStatus};
use crate::mathematical::camera::CameraModel;

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
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub log_odds: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LocalOccupancyMap {
    settings: LocalOccupancySettings,
    origin_x: f64,
    origin_y: f64,
    center_ix: i64,
    center_iy: i64,
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
        let n = settings.width_cells * settings.height_cells;
        let origin_x = -(settings.width_cells as f64) * 0.5 * settings.resolution;
        let origin_y = -(settings.height_cells as f64) * 0.5 * settings.resolution;
        Ok(Self {
            settings,
            origin_x,
            origin_y,
            center_ix: 0,
            center_iy: 0,
            log_odds: vec![0.0; n],
        })
    }

    pub fn settings(&self) -> &LocalOccupancySettings {
        &self.settings
    }

    pub fn snapshot(&self) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot {
            width: self.settings.width_cells,
            height: self.settings.height_cells,
            resolution: self.settings.resolution,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            log_odds: self.log_odds.clone(),
        }
    }

    pub fn cell_state(&self, x: usize, y: usize) -> Option<OccupancyCell> {
        if x >= self.settings.width_cells || y >= self.settings.height_cells {
            return None;
        }
        let value = self.log_odds[y * self.settings.width_cells + x];
        Some(if value >= self.settings.occupied_threshold {
            OccupancyCell::Occupied
        } else if value <= self.settings.free_threshold {
            OccupancyCell::Free
        } else {
            OccupancyCell::Unknown
        })
    }

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
        self.recenter(t_wc[(0, 3)], t_wc[(1, 3)]);

        let mut stats = OccupancyUpdateStats::default();
        if output.eta.width == 0 || output.eta.height == 0 || image_width == 0 || image_height == 0
        {
            return stats;
        }

        let cam_origin = Vector2::new(t_wc[(0, 3)], t_wc[(1, 3)]);
        let cam_z = t_wc[(2, 3)];
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
                if !eta.is_finite()
                    || !eta_var.is_finite()
                    || eta_var < 0.0
                    || (eta_var as f64).sqrt() > self.settings.max_eta_std
                {
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
                let height = p_w_h[2] - cam_z;
                if height < self.settings.min_obstacle_height
                    || height > self.settings.max_obstacle_height
                {
                    continue;
                }

                let endpoint = Vector2::new(p_w_h[0], p_w_h[1]);
                stats.rays_integrated += 1;
                self.integrate_ray(cam_origin, endpoint, &mut stats);
            }
        }
        stats
    }

    fn recenter(&mut self, center_x: f64, center_y: f64) {
        let new_center_ix = (center_x / self.settings.resolution).floor() as i64;
        let new_center_iy = (center_y / self.settings.resolution).floor() as i64;
        let dx = new_center_ix - self.center_ix;
        let dy = new_center_iy - self.center_iy;
        if dx == 0 && dy == 0 {
            return;
        }

        let w = self.settings.width_cells;
        let h = self.settings.height_cells;
        let mut shifted = vec![0.0; self.log_odds.len()];
        for y in 0..h {
            for x in 0..w {
                let src_x = x as i64 + dx;
                let src_y = y as i64 + dy;
                if src_x >= 0 && src_x < w as i64 && src_y >= 0 && src_y < h as i64 {
                    shifted[y * w + x] = self.log_odds[src_y as usize * w + src_x as usize];
                }
            }
        }
        self.log_odds = shifted;
        self.center_ix = new_center_ix;
        self.center_iy = new_center_iy;
        self.origin_x = (self.center_ix as f64 - w as f64 * 0.5) * self.settings.resolution;
        self.origin_y = (self.center_iy as f64 - h as f64 * 0.5) * self.settings.resolution;
    }

    fn integrate_ray(
        &mut self,
        origin: Vector2<f64>,
        endpoint: Vector2<f64>,
        stats: &mut OccupancyUpdateStats,
    ) {
        // Cell coordinates may lie outside the grid (e.g. an endpoint beyond
        // `max_range` for the ego-centric extent). We walk the *whole* ray and
        // clip per-cell: carve free space for every in-bounds traversed cell up to
        // the boundary, and mark the endpoint occupied only if it is itself
        // in bounds. A ray whose endpoint is off-map thus still contributes its
        // near free space (the obstacle is simply "beyond the map / unknown").
        let (x0, y0) = self.world_to_cell_raw(origin);
        let (x1, y1) = self.world_to_cell_raw(endpoint);
        let endpoint_in_bounds = self.cell_in_bounds(x1, y1);

        let mut x = x0;
        let mut y = y0;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut entered = false;

        loop {
            if x == x1 && y == y1 {
                if endpoint_in_bounds {
                    self.add_log_odds(x, y, self.settings.log_odds_hit);
                    stats.occupied_updates += 1;
                }
                break;
            }
            if self.cell_in_bounds(x, y) {
                self.add_log_odds(x, y, self.settings.log_odds_miss);
                stats.free_updates += 1;
                entered = true;
            } else if entered {
                // We were inside the grid and have now crossed the boundary; a
                // straight ray will not re-enter, so stop (don't walk the off-map
                // tail toward a far endpoint).
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    fn add_log_odds(&mut self, x: i64, y: i64, delta: f32) {
        if !self.cell_in_bounds(x, y) {
            return;
        }
        let idx = y as usize * self.settings.width_cells + x as usize;
        self.log_odds[idx] = (self.log_odds[idx] + delta)
            .clamp(self.settings.log_odds_min, self.settings.log_odds_max);
    }

    /// Cell index for a world point, without bounds checking (may be off-grid).
    fn world_to_cell_raw(&self, p: Vector2<f64>) -> (i64, i64) {
        let x = ((p[0] - self.origin_x) / self.settings.resolution).floor() as i64;
        let y = ((p[1] - self.origin_y) / self.settings.resolution).floor() as i64;
        (x, y)
    }

    fn cell_in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0
            && x < self.settings.width_cells as i64
            && y >= 0
            && y < self.settings.height_cells as i64
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
        PatchDepthOutput {
            eta: DepthMap::from_vec(1, 1, vec![range.ln()]).unwrap(),
            eta_var: DepthMap::from_vec(1, 1, vec![0.01]).unwrap(),
            status: DepthMap::from_vec(1, 1, vec![PatchStatus::PhotoRefined]).unwrap(),
        }
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
        let mut pose = Matrix4::identity();
        pose[(1, 1)] = 0.0;
        pose[(1, 2)] = 1.0;
        pose[(2, 1)] = 1.0;
        pose[(2, 2)] = 0.0;

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
        assert_eq!(map.cell_state(5, 5), Some(OccupancyCell::Free));
        assert_eq!(map.cell_state(5, 8), Some(OccupancyCell::Occupied));
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
        let mut pose = Matrix4::identity();
        pose[(1, 1)] = 0.0;
        pose[(1, 2)] = 1.0;
        pose[(2, 1)] = 1.0;
        pose[(2, 2)] = 0.0;
        let intrinsics = CameraIntrinsics::new(1.0, 1.0, 0.0, 0.0);

        // range 8 -> endpoint cell (5, 13), outside the 11x11 grid (max index 10).
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
        assert_eq!(map.cell_state(5, 5), Some(OccupancyCell::Free)); // camera cell
        assert_eq!(map.cell_state(5, 10), Some(OccupancyCell::Free)); // last in-bounds cell
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
        map.add_log_odds(4, 3, 1.0);

        let mut pose = Matrix4::identity();
        pose[(0, 3)] = 1.0;
        map.recenter(pose[(0, 3)], pose[(1, 3)]);

        assert_eq!(map.cell_state(3, 3), Some(OccupancyCell::Occupied));
    }
}
