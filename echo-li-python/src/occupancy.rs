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
            eta: DepthMap::from_vec(w, h, eta_data).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?,
            eta_var: DepthMap::from_vec(w, h, eta_var_data).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?,
            status: DepthMap::from_vec(w, h, status_data).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?,
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
        let arr = Array3::from_shape_vec(
            (snap.depth, snap.height, snap.width),
            snap.log_odds,
        )
        .expect("occupancy grid shape is valid");
        PyArray3::from_owned_array(py, arr)
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
