use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use echo_li_core::config::VIOConfig;
use echo_li_core::core_types::{CameraIntrinsics, DepthMap};
use echo_li_core::depth::occupancy::{LocalOccupancyMap, LocalOccupancySettings};
use echo_li_core::depth::patch_depth::{PatchDepthOutput, PatchDepthSeedCoordinates, PatchStatus};
use nalgebra::Matrix4;

use crate::camera::to_camera_arc;

#[pyclass(name = "LocalOccupancyMap")]
pub struct PyLocalOccupancyMap {
    inner: LocalOccupancyMap,
    camera: std::sync::Arc<dyn echo_li_core::mathematical::camera::CameraModel>,
    intrinsics: CameraIntrinsics,
    seed_coordinates: PatchDepthSeedCoordinates,
    image_width: usize,
    image_height: usize,
}

#[pymethods]
impl PyLocalOccupancyMap {
    /// Build a local occupancy map.
    ///
    /// `seed_coordinates`: `"raw"` (raw distorted / per-patch-bearing) or
    /// `"pinhole"` (undistorted-pinhole / tiled-bearing).  Must match the
    /// `PatchDepthMapper`'s `seed_coordinates` property.
    #[new]
    #[pyo3(signature = (camera, fx, fy, cx, cy, width, height, config=None, seed_coordinates="raw"))]
    fn new(
        camera: &Bound<'_, PyAny>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        config: Option<&str>,
        seed_coordinates: &str,
    ) -> PyResult<Self> {
        let cam = to_camera_arc(camera)?;
        let intrinsics = CameraIntrinsics { fx, fy, cx, cy };
        let coords = match seed_coordinates {
            "raw" => PatchDepthSeedCoordinates::RawDistorted,
            "pinhole" => PatchDepthSeedCoordinates::UndistortedPinhole,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "seed_coordinates must be \"raw\" or \"pinhole\", got {other:?}"
                )));
            }
        };

        let settings = match config {
            Some(path) => {
                let vio = VIOConfig::from_yaml(path).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to load config {path}: {e}"
                    ))
                })?;
                vio.local_occupancy
                    .as_ref()
                    .map(|c| c.to_local_occupancy_settings())
                    .unwrap_or_default()
            }
            None => LocalOccupancySettings::default(),
        };

        let inner = LocalOccupancyMap::new(settings).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create LocalOccupancyMap: {e}"
            ))
        })?;

        Ok(Self {
            inner,
            camera: cam,
            intrinsics,
            seed_coordinates: coords,
            image_width: width,
            image_height: height,
        })
    }

    /// Integrate one frame's dense depth into the occupancy grid.
    ///
    /// `eta`, `eta_var`, `status` are the arrays returned by
    /// `PatchDepthMapper.update()`.  `t_wc` is the 4×4 camera pose (world ← cam).
    ///
    /// Returns a dict `{rays_considered, rays_integrated, occupied_updates,
    /// free_updates}`.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        eta: PyReadonlyArray2<'py, f32>,
        eta_var: PyReadonlyArray2<'py, f32>,
        status: PyReadonlyArray2<'py, u8>,
        t_wc: [[f64; 4]; 4],
    ) -> PyResult<PyObject> {
        let [h, w] = *eta.shape() else {
            return Err(pyo3::exceptions::PyValueError::new_err("eta must be 2-D"));
        };
        if eta_var.shape() != [h, w] || status.shape() != [h, w] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eta, eta_var, and status must have the same shape",
            ));
        }

        let eta_data = eta.as_slice()?.to_vec();
        let eta_var_data = eta_var.as_slice()?.to_vec();
        let status_data: Vec<PatchStatus> = status
            .as_slice()?
            .iter()
            .map(|&s| match s {
                1 => PatchStatus::SeedOnly,
                2 => PatchStatus::PhotoRefined,
                3 => PatchStatus::Rejected,
                _ => PatchStatus::Unknown,
            })
            .collect();

        let output = PatchDepthOutput {
            eta: DepthMap::from_vec(w, h, eta_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            eta_var: DepthMap::from_vec(w, h, eta_var_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            status: DepthMap::from_vec(w, h, status_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        };

        let pose = array_to_matrix4(&t_wc);

        let stats = self.inner.update_from_patch_depth(
            &output,
            self.camera.as_ref(),
            self.intrinsics,
            self.seed_coordinates,
            self.image_width,
            self.image_height,
            &pose,
        );

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("rays_considered", stats.rays_considered)?;
        dict.set_item("rays_integrated", stats.rays_integrated)?;
        dict.set_item("occupied_updates", stats.occupied_updates)?;
        dict.set_item("free_updates", stats.free_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// `(unknown, free, occupied)` voxel counts.
    fn counts(&self) -> (usize, usize, usize) {
        self.inner.counts()
    }

    /// Occupied voxel centres as an (N, 3) float32 array in world coordinates.
    fn occupied_cells<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let snap = self.inner.snapshot();
        let occ_thr = self.inner.settings().occupied_threshold;
        let r = snap.resolution;
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for cz in 0..snap.depth {
            for cy in 0..snap.height {
                for cx in 0..snap.width {
                    let l = snap.log_odds[(cz * snap.height + cy) * snap.width + cx];
                    if l >= occ_thr {
                        pts.push([
                            (snap.origin_x + (cx as f64 + 0.5) * r) as f32,
                            (snap.origin_y + (cy as f64 + 0.5) * r) as f32,
                            (snap.origin_z + (cz as f64 + 0.5) * r) as f32,
                        ]);
                    }
                }
            }
        }
        let n = pts.len();
        let flat: Vec<f32> = pts.into_iter().flat_map(|p| p).collect();
        let arr = Array2::from_shape_vec((n, 3), flat).unwrap_or_else(|_| Array2::zeros((0, 3)));
        PyArray2::from_owned_array(py, arr)
    }

    /// Free voxel centres as an (N, 3) float32 array in world coordinates.
    /// Decimated by stride 2 per axis (1/8 the full free volume).
    fn free_cells<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let snap = self.inner.snapshot();
        let free_thr = self.inner.settings().free_threshold;
        let occ_thr = self.inner.settings().occupied_threshold;
        let r = snap.resolution;
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for cz in (0..snap.depth).step_by(2) {
            for cy in (0..snap.height).step_by(2) {
                for cx in (0..snap.width).step_by(2) {
                    let l = snap.log_odds[(cz * snap.height + cy) * snap.width + cx];
                    if l <= free_thr && l < occ_thr {
                        pts.push([
                            (snap.origin_x + (cx as f64 + 0.5) * r) as f32,
                            (snap.origin_y + (cy as f64 + 0.5) * r) as f32,
                            (snap.origin_z + (cz as f64 + 0.5) * r) as f32,
                        ]);
                    }
                }
            }
        }
        let n = pts.len();
        let flat: Vec<f32> = pts.into_iter().flat_map(|p| p).collect();
        let arr = Array2::from_shape_vec((n, 3), flat).unwrap_or_else(|_| Array2::zeros((0, 3)));
        PyArray2::from_owned_array(py, arr)
    }

    /// Full 3D log-odds grid as (depth, height, width) float32 array.
    fn log_odds_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let snap = self.inner.snapshot();
        let arr = Array3::from_shape_vec((snap.depth, snap.height, snap.width), snap.log_odds)
            .expect("occupancy grid shape is valid");
        PyArray3::from_owned_array(py, arr)
    }

    /// The σ channel as (depth, height, width) float32 metres; `inf` where no ray's band reached.
    fn sigma_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let snap = self.inner.snapshot();
        let arr = Array3::from_shape_vec((snap.depth, snap.height, snap.width), snap.sigma)
            .expect("occupancy grid shape is valid");
        PyArray3::from_owned_array(py, arr)
    }

    /// `LocalOccupancyMap.from_settings({...})`: a map from a dict of
    /// `LocalOccupancySettings` fields (resolution, width_cells, height_cells, sample_stride,
    /// min_range, max_range, max_eta_std, log_odds_hit, log_odds_miss, log_odds_min,
    /// log_odds_max, occupied_threshold, free_threshold, update_mode "fixed"|"sigma", band_k,
    /// sigma_floor_factor, min_confidence_weight, min_obstacle_height, max_obstacle_height).
    /// For `update_dense` only; `update` needs the camera the other constructor takes.
    #[staticmethod]
    fn from_settings(settings: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        use echo_li_core::depth::occupancy::OccupancyUpdateMode;
        let mut st = LocalOccupancySettings::default();
        for (k, v) in settings.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "enabled" => st.enabled = v.extract()?,
                "resolution" => st.resolution = v.extract()?,
                "width_cells" => st.width_cells = v.extract()?,
                "height_cells" => st.height_cells = v.extract()?,
                "sample_stride" => st.sample_stride = v.extract()?,
                "min_range" => st.min_range = v.extract()?,
                "max_range" => st.max_range = v.extract()?,
                "max_eta_std" => st.max_eta_std = v.extract()?,
                "log_odds_hit" => st.log_odds_hit = v.extract()?,
                "log_odds_miss" => st.log_odds_miss = v.extract()?,
                "log_odds_min" => st.log_odds_min = v.extract()?,
                "log_odds_max" => st.log_odds_max = v.extract()?,
                "occupied_threshold" => st.occupied_threshold = v.extract()?,
                "free_threshold" => st.free_threshold = v.extract()?,
                "band_k" => st.band_k = v.extract()?,
                "sigma_floor_factor" => st.sigma_floor_factor = v.extract()?,
                "min_confidence_weight" => st.min_confidence_weight = v.extract()?,
                "min_obstacle_height" => st.min_obstacle_height = v.extract()?,
                "max_obstacle_height" => st.max_obstacle_height = v.extract()?,
                "update_mode" => {
                    let m: String = v.extract()?;
                    st.update_mode = match m.as_str() {
                        "sigma" | "uncertainty_aware" | "v1" => {
                            OccupancyUpdateMode::UncertaintyAware
                        }
                        "one_sided" | "v2" => OccupancyUpdateMode::OneSided,
                        _ => OccupancyUpdateMode::FixedIncrement,
                    };
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown LocalOccupancy setting {other:?}"
                    )))
                }
            }
        }
        let inner = LocalOccupancyMap::new(st).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create LocalOccupancyMap: {e}"
            ))
        })?;
        let cam: std::sync::Arc<dyn echo_li_core::mathematical::camera::CameraModel> =
            std::sync::Arc::new(echo_li_core::mathematical::camera::PinholeModel {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            });
        Ok(Self {
            inner,
            camera: cam,
            intrinsics: CameraIntrinsics {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            },
            seed_coordinates: PatchDepthSeedCoordinates::UndistortedPinhole,
            image_width: 0,
            image_height: 0,
        })
    }

    /// Integrate a dense frame from `DenseDepthMapper` (pinhole frame with intrinsics
    /// `fx, fy, cx, cy` at the frame's own resolution; `cls` 0 unknown / 1 measured / 2 filled).
    /// Measured pixels integrate as returns; Filled pixels only carve free space.
    #[allow(clippy::too_many_arguments)]
    fn update_dense<'py>(
        &mut self,
        py: Python<'py>,
        eta: PyReadonlyArray2<'py, f32>,
        eta_var: PyReadonlyArray2<'py, f32>,
        cls: PyReadonlyArray2<'py, u8>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        t_wc: [[f64; 4]; 4],
    ) -> PyResult<PyObject> {
        use echo_li_core::depth::dense_mapper::{DenseClass, DenseOutput, DenseStats};
        let [h, w] = *eta.shape() else {
            return Err(pyo3::exceptions::PyValueError::new_err("eta must be 2-D"));
        };
        if eta_var.shape() != [h, w] || cls.shape() != [h, w] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eta, eta_var and cls must share a shape",
            ));
        }
        let class: Vec<DenseClass> = cls
            .as_slice()?
            .iter()
            .map(|&c| match c {
                1 => DenseClass::Measured,
                2 => DenseClass::Filled,
                _ => DenseClass::Unknown,
            })
            .collect();
        let out = DenseOutput {
            width: w,
            height: h,
            eta: eta.as_slice()?.to_vec(),
            eta_var: eta_var.as_slice()?.to_vec(),
            class,
            stats: DenseStats::default(),
        };
        let pose = array_to_matrix4(&t_wc);
        let stats = self
            .inner
            .update_from_dense(&out, CameraIntrinsics { fx, fy, cx, cy }, &pose);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("rays_considered", stats.rays_considered)?;
        dict.set_item("rays_integrated", stats.rays_integrated)?;
        dict.set_item("occupied_updates", stats.occupied_updates)?;
        dict.set_item("free_updates", stats.free_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// `(origin_x, origin_y, origin_z, resolution, width, height, depth)` of the grid as it
    /// stands (the map recentres on the camera; cell `(z, y, x)` of `log_odds_grid()` has its
    /// centre at `origin + (idx + 0.5) * resolution`).
    fn grid_info(&self) -> (f64, f64, f64, f64, usize, usize, usize) {
        let snap = self.inner.snapshot();
        (
            snap.origin_x,
            snap.origin_y,
            snap.origin_z,
            snap.resolution,
            snap.width,
            snap.height,
            snap.depth,
        )
    }

    fn __repr__(&self) -> String {
        let (u, f, o) = self.inner.counts();
        format!(
            "LocalOccupancyMap(unknown={u}, free={f}, occupied={o}, depth_cells={})",
            self.inner.depth_cells()
        )
    }
}

fn array_to_matrix4(a: &[[f64; 4]; 4]) -> Matrix4<f64> {
    Matrix4::new(
        a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0], a[2][1],
        a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3],
    )
}
