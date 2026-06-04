use echo_li_core::config::VIOConfig;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::frontend::{self, Frontend, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::klt::LkMethod;

#[pyclass]
#[derive(Clone)]
pub struct FrontendConfig {
    #[pyo3(get, set)]
    pub max_features: usize,
    #[pyo3(get, set)]
    pub fast_threshold: u8,
    #[pyo3(get, set)]
    pub pyramid_levels: usize,
    #[pyo3(get, set)]
    pub cell_size: usize,
    #[pyo3(get, set)]
    pub klt_window: usize,
    #[pyo3(get, set)]
    pub klt_max_iter: usize,
    #[pyo3(get, set)]
    pub lbp_verification: bool,
    #[pyo3(get, set)]
    pub lbp_policy: String,
    #[pyo3(get, set)]
    pub histeq: String,
    intrinsics: Option<(f64, f64, f64, f64, usize, usize, Vec<f64>)>,
}

#[pymethods]
impl FrontendConfig {
    #[new]
    #[pyo3(signature = (
        max_features = 200,
        fast_threshold = 20,
        pyramid_levels = 3,
        cell_size = 64,
        klt_window = 21,
        klt_max_iter = 30,
        lbp_verification = true,
        lbp_policy = "soft".to_string(),
        histeq = "global".to_string(),
    ))]
    fn new(
        max_features: usize,
        fast_threshold: u8,
        pyramid_levels: usize,
        cell_size: usize,
        klt_window: usize,
        klt_max_iter: usize,
        lbp_verification: bool,
        lbp_policy: String,
        histeq: String,
    ) -> Self {
        Self {
            max_features,
            fast_threshold,
            pyramid_levels,
            cell_size,
            klt_window,
            klt_max_iter,
            lbp_verification,
            lbp_policy,
            histeq,
            intrinsics: None,
        }
    }

    #[staticmethod]
    fn from_yaml(path: &str) -> PyResult<Self> {
        let vio_config = VIOConfig::from_yaml(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load config: {e}"))
        })?;
        let rv = &vio_config.rudolf_v;
        let histeq = if rv.equalise_image_histogram {
            "global".to_string()
        } else {
            "none".to_string()
        };
        let lbp_policy = rv.lbp_policy.as_deref().unwrap_or("soft").to_string();
        let defaults = frontend::FrontendConfig::default();
        Ok(Self {
            max_features: rv.max_features,
            fast_threshold: defaults.fast_threshold,
            pyramid_levels: rv.max_level,
            cell_size: rv.feature_dist as usize,
            klt_window: defaults.klt_window,
            klt_max_iter: defaults.klt_max_iter,
            lbp_verification: defaults.lbp_verification_enabled,
            lbp_policy,
            histeq,
            intrinsics: None,
        })
    }

    fn set_camera(
        &mut self,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        distortion: Vec<f64>,
    ) {
        self.intrinsics = Some((fx, fy, cx, cy, width, height, distortion));
    }

    fn __repr__(&self) -> String {
        let cam = if self.intrinsics.is_some() {
            "camera=set"
        } else {
            "camera=none"
        };
        format!(
            "FrontendConfig(max_features={}, fast_threshold={}, pyramid_levels={}, cell_size={}, lbp={}, histeq={}, {})",
            self.max_features, self.fast_threshold, self.pyramid_levels, self.cell_size, self.lbp_policy, self.histeq, cam
        )
    }
}

impl FrontendConfig {
    fn to_rust(&self) -> frontend::FrontendConfig {
        let mut cfg = frontend::FrontendConfig::default();
        cfg.max_features = self.max_features;
        cfg.fast_threshold = self.fast_threshold;
        cfg.pyramid_levels = self.pyramid_levels;
        cfg.cell_size = self.cell_size;
        cfg.klt_window = self.klt_window;
        cfg.klt_max_iter = self.klt_max_iter;
        cfg.lbp_verification_enabled = self.lbp_verification;
        cfg.lbp_policy = match self.lbp_policy.to_ascii_lowercase().as_str() {
            "hardreject" | "hard_reject" | "hard-reject" | "hard" => LbpPolicy::HardReject,
            _ => LbpPolicy::SoftPenalty,
        };
        cfg.klt_method = LkMethod::InverseCompositional;
        cfg.histeq = match self.histeq.to_ascii_lowercase().as_str() {
            "global" => HistEqMethod::Global,
            "clahe" => HistEqMethod::Clahe {
                tile_size: 8,
                clip_limit: 4.0,
            },
            _ => HistEqMethod::None,
        };
        if let Some((fx, fy, cx, cy, w, h, ref dist)) = self.intrinsics {
            let mut cam = CameraIntrinsics::new(fx, fy, cx, cy, w, h);
            cam.distortion = dist.clone();
            cfg.camera = Some(cam);
        }
        cfg
    }
}

#[pyclass(name = "Frontend")]
pub struct PyFrontend {
    inner: Frontend,
    width: usize,
    height: usize,
}

impl PyFrontend {
    pub(crate) fn inner(&self) -> &Frontend {
        &self.inner
    }

    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

#[pymethods]
impl PyFrontend {
    #[new]
    fn new(config: &FrontendConfig, width: usize, height: usize) -> Self {
        let inner = Frontend::new(config.to_rust(), width, height);
        Self {
            inner,
            width,
            height,
        }
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyList>, PyObject)> {
        let shape = image.shape();
        if shape[0] != self.height || shape[1] != self.width {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Image shape ({}, {}) doesn't match frontend ({}, {})",
                shape[0], shape[1], self.height, self.width
            )));
        }
        let data = image.as_slice()?;
        let rudolf_img = RudolfImage::from_vec(self.width, self.height, data.to_vec());
        let (features, stats) = self.inner.process(&rudolf_img);

        let feat_list = pyo3::types::PyList::empty(py);
        for f in features {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", f.id)?;
            dict.set_item("x", f.x)?;
            dict.set_item("y", f.y)?;
            dict.set_item("score", f.score)?;
            dict.set_item("level", f.level)?;
            feat_list.append(dict)?;
        }

        let stats_dict = pyo3::types::PyDict::new(py);
        stats_dict.set_item("tracked", stats.tracked)?;
        stats_dict.set_item("lost", stats.lost)?;
        stats_dict.set_item("rejected", stats.rejected)?;
        stats_dict.set_item("new_detections", stats.new_detections)?;
        stats_dict.set_item("total", stats.total)?;
        stats_dict.set_item("timing_ms", stats.timing.total_ms())?;

        Ok((feat_list, stats_dict.into_any().unbind()))
    }

    fn drop_tracks(&mut self, ids: Vec<u64>) -> usize {
        self.inner.drop_tracks(&ids)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let feat_list = pyo3::types::PyList::empty(py);
        for f in self.inner.features() {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", f.id)?;
            dict.set_item("x", f.x)?;
            dict.set_item("y", f.y)?;
            dict.set_item("score", f.score)?;
            dict.set_item("level", f.level)?;
            feat_list.append(dict)?;
        }
        Ok(feat_list)
    }

    fn track_meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let meta_list = pyo3::types::PyList::empty(py);
        for m in self.inner.track_meta() {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", m.id)?;
            dict.set_item("age", m.age)?;
            dict.set_item("klt_quality", m.klt_quality)?;
            dict.set_item("reservoir_score", m.reservoir_score)?;
            dict.set_item("is_ekf_landmark", m.is_ekf_landmark)?;
            meta_list.append(dict)?;
        }
        Ok(meta_list)
    }

    fn __repr__(&self) -> String {
        format!(
            "Frontend({}x{}, {} features)",
            self.width,
            self.height,
            self.inner.features().len()
        )
    }
}
