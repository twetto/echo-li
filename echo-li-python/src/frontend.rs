use echo_li_core::config::VIOConfig;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::frontend::{self, Frontend, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::klt::LkMethod;
use rudolf_v::klt_reference::{KltTemplatePolicy, ReferenceKltWarp};

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
    pub klt_warp: String,
    #[pyo3(get, set)]
    pub klt_template_policy: String,
    #[pyo3(get, set)]
    pub klt_reference_warp: String,
    #[pyo3(get, set)]
    pub klt_residual: bool,
    #[pyo3(get, set)]
    pub enable_ransac: bool,
    #[pyo3(get, set)]
    pub epipolar_gate_threshold: f64,
    #[pyo3(get, set)]
    pub epipolar_refine: bool,
    #[pyo3(get, set)]
    pub epipolar_min_baseline: f64,
    #[pyo3(get, set)]
    pub epipolar_max_reject_frac: f64,
    #[pyo3(get, set)]
    pub lbp_verification: bool,
    #[pyo3(get, set)]
    pub lbp_policy: String,
    #[pyo3(get, set)]
    pub histeq: String,
    #[pyo3(get, set)]
    pub clahe_tile_size: usize,
    #[pyo3(get, set)]
    pub clahe_clip_limit: f32,
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
        klt_warp = "translation".to_string(),
        klt_template_policy = "previous".to_string(),
        klt_reference_warp = "translation".to_string(),
        klt_residual = false,
        enable_ransac = true,
        epipolar_gate_threshold = 0.0,
        epipolar_refine = false,
        epipolar_min_baseline = 1e-3,
        epipolar_max_reject_frac = 0.5,
        lbp_verification = true,
        lbp_policy = "soft".to_string(),
        histeq = "global".to_string(),
        clahe_tile_size = 256,
        clahe_clip_limit = 4.0,
    ))]
    fn new(
        max_features: usize,
        fast_threshold: u8,
        pyramid_levels: usize,
        cell_size: usize,
        klt_window: usize,
        klt_max_iter: usize,
        klt_warp: String,
        klt_template_policy: String,
        klt_reference_warp: String,
        klt_residual: bool,
        enable_ransac: bool,
        epipolar_gate_threshold: f64,
        epipolar_refine: bool,
        epipolar_min_baseline: f64,
        epipolar_max_reject_frac: f64,
        lbp_verification: bool,
        lbp_policy: String,
        histeq: String,
        clahe_tile_size: usize,
        clahe_clip_limit: f32,
    ) -> Self {
        Self {
            max_features,
            fast_threshold,
            pyramid_levels,
            cell_size,
            klt_window,
            klt_max_iter,
            klt_warp,
            klt_template_policy,
            klt_reference_warp,
            klt_residual,
            enable_ransac,
            epipolar_gate_threshold,
            epipolar_refine,
            epipolar_min_baseline,
            epipolar_max_reject_frac,
            lbp_verification,
            lbp_policy,
            histeq,
            clahe_tile_size,
            clahe_clip_limit,
            intrinsics: None,
        }
    }

    #[staticmethod]
    fn from_yaml(path: &str) -> PyResult<Self> {
        let vio_config = VIOConfig::from_yaml(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load config: {e}"))
        })?;
        let rv = &vio_config.rudolf_v;
        let histeq = rv.histeq.clone().unwrap_or_else(|| {
            if rv.equalise_image_histogram {
                "global".to_string()
            } else {
                "none".to_string()
            }
        });
        let lbp_policy = rv.lbp_policy.as_deref().unwrap_or("soft").to_string();
        let defaults = frontend::FrontendConfig::default();
        Ok(Self {
            max_features: rv.max_features,
            fast_threshold: defaults.fast_threshold,
            pyramid_levels: rv.max_level,
            cell_size: rv.feature_dist as usize,
            klt_window: defaults.klt_window,
            klt_max_iter: defaults.klt_max_iter,
            klt_warp: "translation".to_string(),
            klt_template_policy: "previous".to_string(),
            klt_reference_warp: "translation".to_string(),
            klt_residual: rv.klt_residual,
            enable_ransac: rv.enable_ransac,
            epipolar_gate_threshold: rv.epipolar_gate_threshold,
            epipolar_refine: rv.epipolar_refine,
            epipolar_min_baseline: rv.epipolar_min_baseline,
            epipolar_max_reject_frac: rv.epipolar_max_reject_frac,
            lbp_verification: defaults.lbp_verification_enabled,
            lbp_policy,
            histeq,
            clahe_tile_size: rv.clahe_tile_size,
            clahe_clip_limit: rv.clahe_clip_limit,
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
            self.max_features,
            self.fast_threshold,
            self.pyramid_levels,
            self.cell_size,
            self.lbp_policy,
            self.histeq,
            cam
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
        cfg.klt_method = match self.klt_warp.to_ascii_lowercase().as_str() {
            "affine" => LkMethod::InverseCompositionalAffine,
            _ => LkMethod::InverseCompositional,
        };
        cfg.klt_template_policy = match self.klt_template_policy.to_ascii_lowercase().as_str() {
            "first" | "first_observation" | "first-observation" | "reference" | "anchor" => {
                KltTemplatePolicy::FirstObservation
            }
            _ => KltTemplatePolicy::PreviousFrame,
        };
        cfg.klt_reference_warp = match self.klt_reference_warp.to_ascii_lowercase().as_str() {
            "affine" => ReferenceKltWarp::Affine,
            _ => ReferenceKltWarp::Translation,
        };
        cfg.klt_residual_enabled = self.klt_residual;
        cfg.enable_internal_ransac = self.enable_ransac;
        cfg.epipolar_gate_threshold = self.epipolar_gate_threshold;
        cfg.epipolar_refine = self.epipolar_refine;
        cfg.epipolar_min_baseline = self.epipolar_min_baseline;
        cfg.epipolar_max_reject_frac = self.epipolar_max_reject_frac;
        cfg.lbp_verification_enabled = self.lbp_verification;
        cfg.lbp_policy = match self.lbp_policy.to_ascii_lowercase().as_str() {
            "hardreject" | "hard_reject" | "hard-reject" | "hard" => LbpPolicy::HardReject,
            _ => LbpPolicy::SoftPenalty,
        };
        cfg.histeq = match self.histeq.to_ascii_lowercase().as_str() {
            "global" => HistEqMethod::Global,
            "clahe" => HistEqMethod::Clahe {
                tile_size: self.clahe_tile_size,
                clip_limit: self.clahe_clip_limit,
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

    /// Relative-pose prior for the NEXT process() call: 4x4 T (prev camera ->
    /// current camera, i.e. x_curr ~ R x_prev + t). Enables the epipolar gate
    /// when epipolar_gate_threshold > 0.
    fn set_pose_prior(&mut self, t_rel: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let a = t_rel.as_array();
        if a.shape() != [4, 4] {
            return Err(pyo3::exceptions::PyValueError::new_err("t_rel must be 4x4"));
        }
        let mut r = [[0.0f64; 3]; 3];
        let mut t = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[[i, j]];
            }
            t[i] = a[[i, 3]];
        }
        self.inner.set_pose_prior(r, t);
        Ok(())
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
            dict.set_item("lbp_distance", m.lbp_distance)?;
            dict.set_item("reservoir_score", m.reservoir_score)?;
            dict.set_item("is_ekf_landmark", m.is_ekf_landmark)?;
            meta_list.append(dict)?;
        }
        Ok(meta_list)
    }

    fn preprocessed_image<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<u8>>>> {
        let Some(img) = self.inner.preprocessed_image() else {
            return Ok(None);
        };

        let width = img.width();
        let height = img.height();
        let stride = img.stride();
        let src = img.as_slice();
        let data = if stride == width {
            src[..width * height].to_vec()
        } else {
            let mut compact = Vec::with_capacity(width * height);
            for y in 0..height {
                let row = &src[y * stride..y * stride + width];
                compact.extend_from_slice(row);
            }
            compact
        };
        let arr = Array2::from_shape_vec((height, width), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Some(PyArray2::from_owned_array(py, arr)))
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
