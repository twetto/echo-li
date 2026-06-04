use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::path::PathBuf;

use echo_li_core::config::VIOConfig;
use rudolf_v::camera::StereoRig;
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::stereo::{StereoConfig as RudolfStereoConfig, StereoMatcher};

use crate::frontend::PyFrontend;

/// Stereo matcher wrapping `rudolf_v::stereo::StereoMatcher`.
///
/// The matcher owns cam0/cam1 intrinsics + rig and a cam1 image pyramid. It
/// pairs cam0 features (from the Python `Frontend`) with cam1 patches and
/// returns per-feature inverse-depth + residual. Use `range_priors()` to
/// convert matches into the `(range, range_var)` form consumed by
/// `VIOFilter.process_vision_with_depth_priors`.
#[pyclass(name = "Stereo")]
pub struct PyStereo {
    inner: StereoMatcher,
    img_w: usize,
    img_h: usize,
}

#[pymethods]
impl PyStereo {
    /// Build a stereo matcher from a pair of EuRoC `sensor.yaml` files.
    ///
    /// When `vio_config` is given, the `Stereo:` section of that YAML
    /// overrides the defaults (pyramid levels, patch size, residual gate,
    /// histeq, etc.). Pass the same config you use for `VIOFilter` so the
    /// matcher settings stay in sync with the CLI.
    #[staticmethod]
    #[pyo3(signature = (cam0_yaml, cam1_yaml, img_w, img_h, vio_config=None))]
    fn from_euroc(
        cam0_yaml: &str,
        cam1_yaml: &str,
        img_w: usize,
        img_h: usize,
        vio_config: Option<&str>,
    ) -> PyResult<Self> {
        let rig = StereoRig::from_euroc(&PathBuf::from(cam0_yaml), &PathBuf::from(cam1_yaml))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build StereoRig: {e}"))
            })?;
        let mut cfg = RudolfStereoConfig::default();
        if let Some(path) = vio_config {
            let vio = VIOConfig::from_yaml(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load config: {e}"))
            })?;
            if let Some(s) = vio.stereo.as_ref() {
                if let Some(v) = s.pyramid_levels {
                    cfg.pyramid_levels = v;
                }
                if let Some(v) = s.patch_half_size {
                    cfg.patch_half_size = v;
                }
                if let Some(v) = s.max_iterations {
                    cfg.max_iterations = v;
                }
                if let Some(v) = s.convergence_eps {
                    cfg.convergence_eps = v;
                }
                if let Some(v) = s.min_inv_depth {
                    cfg.min_inv_depth = v;
                }
                if let Some(v) = s.max_inv_depth {
                    cfg.max_inv_depth = v;
                }
                if let Some(v) = s.init_inv_depth {
                    cfg.init_inv_depth = v;
                }
                if let Some(v) = s.max_residual {
                    cfg.max_residual = v;
                }
                if let Some(v) = s.n_search_candidates {
                    cfg.n_search_candidates = v;
                }
                if let Some(v) = s.knn_propagation {
                    cfg.knn_propagation = v;
                }
                if let Some(h) = &s.histeq {
                    cfg.histeq = match h.to_ascii_lowercase().as_str() {
                        "global" => HistEqMethod::Global,
                        _ => HistEqMethod::None,
                    };
                }
            }
        }
        Ok(Self {
            inner: StereoMatcher::new(rig, cfg, img_w, img_h),
            img_w,
            img_h,
        })
    }

    #[getter]
    fn baseline_meters(&self) -> f64 {
        self.inner.rig().baseline_meters()
    }

    /// Match the current frontend features into `cam1_image` and return
    /// `(range, range_var)` priors, ready for
    /// `VIOFilter.process_vision_with_depth_priors`.
    ///
    /// `range` is the euclidean distance from cam0 origin to the
    /// triangulated point. `range_var` follows the CLI's heuristic:
    /// `(residual_px / sigma_pixel_scale)² · range² / baseline²`, clamped
    /// to `>= 1e-4`.
    #[pyo3(signature = (cam1_image, frontend, sigma_pixel_scale=20.0))]
    fn range_priors<'py>(
        &mut self,
        py: Python<'py>,
        cam1_image: PyReadonlyArray2<'py, u8>,
        frontend: PyRef<'py, PyFrontend>,
        sigma_pixel_scale: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let matches = self.match_into(&cam1_image, &frontend)?;
        let rig = self.inner.rig();
        let baseline = rig.baseline_meters().max(1e-9);
        let features = frontend.inner().features();
        let dict = pyo3::types::PyDict::new(py);
        for (feat, m) in features.iter().zip(matches.iter()) {
            let Some(p) = m.point_cam0(rig, feat) else {
                continue;
            };
            let range = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if range <= 0.0 {
                continue;
            }
            let range_var = (m.residual as f64 / sigma_pixel_scale).powi(2) * range * range
                / (baseline * baseline);
            dict.set_item(feat.id, (range, range_var.max(1e-4)))?;
        }
        Ok(dict)
    }

    /// Lower-level access: return a list of per-feature stereo matches with
    /// all the matcher's intermediates (cam1 pixel, inverse depth, residual,
    /// success flag). Useful when you want to do your own RANSAC validation
    /// before promoting matches to depth priors.
    fn match_features<'py>(
        &mut self,
        py: Python<'py>,
        cam1_image: PyReadonlyArray2<'py, u8>,
        frontend: PyRef<'py, PyFrontend>,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let matches = self.match_into(&cam1_image, &frontend)?;
        let features = frontend.inner().features();
        let rig = self.inner.rig();
        let list = pyo3::types::PyList::empty(py);
        for (feat, m) in features.iter().zip(matches.iter()) {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", feat.id)?;
            dict.set_item("u1", m.u1)?;
            dict.set_item("v1", m.v1)?;
            dict.set_item("inv_depth", m.inv_depth)?;
            dict.set_item("residual", m.residual)?;
            dict.set_item("matched", m.matched)?;
            if let Some(p) = m.point_cam0(rig, feat) {
                dict.set_item("point_cam0", (p[0], p[1], p[2]))?;
            } else {
                dict.set_item("point_cam0", py.None())?;
            }
            list.append(dict)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "Stereo(img={}x{}, baseline={:.4}m)",
            self.img_w,
            self.img_h,
            self.inner.rig().baseline_meters()
        )
    }
}

impl PyStereo {
    fn match_into<'py>(
        &mut self,
        cam1_image: &PyReadonlyArray2<'py, u8>,
        frontend: &PyRef<'py, PyFrontend>,
    ) -> PyResult<Vec<rudolf_v::stereo::StereoMatch>> {
        let shape = cam1_image.shape();
        let (w, h) = frontend.dimensions();
        if shape[0] != h || shape[1] != w {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cam1 shape ({}, {}) doesn't match frontend ({}, {})",
                shape[0], shape[1], h, w
            )));
        }
        let data = cam1_image.as_slice()?.to_vec();
        let img = RudolfImage::from_vec(w, h, data);
        let fe = frontend.inner();
        Ok(self
            .inner
            .match_features(&img, fe.features(), fe.current_pyramid()))
    }
}
