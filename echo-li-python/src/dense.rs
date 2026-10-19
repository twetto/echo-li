//! Python face of `echo_li_core::depth::dense`: the fusion rule and the honest fill, as two
//! functions over numpy arrays, so the diagnostics can score the Rust frame against the
//! prototype pixel for pixel and the node can share the code.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use echo_li_core::depth::dense::{fill, fuse, DenseItem, FillParams, FusedFrame};

fn to_array<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    v: &[T],
    h: usize,
    w: usize,
) -> Bound<'py, PyArray2<T>> {
    PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), v.to_vec()).unwrap())
}

/// `dense_fuse(items, h, w, slant_g) -> (eta, var, covered, dist)`.
///
/// `items`: `(N, 6)` float64 rows `(px, py, level, eta, var_cond, sigma_shared)` at the working
/// resolution; `slant_g = tan(slant) / f`. Returns `h × w` arrays: nearest-patch log-range,
/// its variance (conditional + shared² + disagreement + footprint), the covered mask and the
/// distance to the patch centre each pixel took its range from.
#[pyfunction]
pub fn dense_fuse<'py>(
    py: Python<'py>,
    items: PyReadonlyArray2<f64>,
    h: usize,
    w: usize,
    slant_g: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let a = items.as_array();
    if a.ncols() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "items must be (N, 6)",
        ));
    }
    let v: Vec<DenseItem> = a
        .rows()
        .into_iter()
        .map(|r| DenseItem {
            px: r[0],
            py: r[1],
            level: r[2] as u32,
            eta: r[3],
            var_cond: r[4],
            sigma_shared: r[5],
        })
        .collect();
    let f = py.allow_threads(|| fuse(&v, w, h, slant_g));
    Ok((
        to_array(py, &f.eta, h, w),
        to_array(py, &f.var, h, w),
        to_array(py, &f.covered, h, w),
        to_array(py, &f.dist, h, w),
    ))
}

/// `dense_fill(eta, var, covered, sigma_px, gate, balance, max_d, g) -> (eta_f, var_f, filled, gated, dist)`.
///
/// The `Filled` class on top of a fused frame; defaults are the prototype's (`g` defaults to
/// twice the footprint gradient when passed as `None`, so pass `2 * slant_g` explicitly).
#[pyfunction]
#[pyo3(signature = (eta, var, covered, sigma_px=12.0, gate=0.15, balance=12.0, max_d=48.0, g=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn dense_fill<'py>(
    py: Python<'py>,
    eta: PyReadonlyArray2<f64>,
    var: PyReadonlyArray2<f64>,
    covered: PyReadonlyArray2<bool>,
    sigma_px: f64,
    gate: f64,
    balance: f64,
    max_d: f64,
    g: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let e = eta.as_array();
    let (h, w) = (e.nrows(), e.ncols());
    let fused = FusedFrame {
        w,
        h,
        eta: e.iter().copied().collect(),
        var: var.as_array().iter().copied().collect(),
        covered: covered.as_array().iter().copied().collect(),
        dist: vec![0.0; w * h],
    };
    let p = FillParams {
        sigma_px,
        gate,
        balance,
        max_d,
        g,
    };
    let r = py.allow_threads(|| fill(&fused, &p));
    Ok((
        to_array(py, &r.eta, h, w),
        to_array(py, &r.var, h, w),
        to_array(py, &r.filled, h, w),
        to_array(py, &r.gated, h, w),
        to_array(py, &r.dist, h, w),
    ))
}

// ---------------------------------------------------------------------------------------
// The whole dense mapper, as the node runs it, for the rig harness to drive offline.

use numpy::{PyArray2 as PyArray2U8, PyReadonlyArray2 as PyReadonlyArray2U8};
use pyo3::types::PyDict;
use std::collections::HashMap;

use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::depth::dense_mapper::{DenseClass, DenseDepthMapper, DenseMapperSettings};
use rudolf_v::histeq::HistEqMethod;

/// `DenseDepthMapper(camera, fx, fy, cx, cy, width, height, sparse, tracker=None, ...)`.
///
/// `camera` is one of the echo_li camera classes (the rig's model); `fx..cy` are its
/// intrinsics at `width × height`, the raw frame size the mapper is given; `sparse` is a dict
/// of Sparse3D settings (the same keys as `Sparse3DFilter`), already carrying the patch-track
/// values (walk variance, pool); `tracker` is a dict of FlowDepF settings. The mapper
/// equalises, undistorts and downscales the frame itself.
#[pyclass(name = "DenseDepthMapper")]
pub struct PyDenseDepthMapper {
    inner: DenseDepthMapper,
}

#[pymethods]
impl PyDenseDepthMapper {
    #[new]
    #[pyo3(signature = (camera, fx, fy, cx, cy, width, height, sparse, tracker=None, scale=1.0,
                        histeq="clahe", clahe_tile_size=64, clahe_clip_limit=4.0, slant_deg=35.0,
                        fill=true, fill_sigma=12.0, fill_gate=0.15, fill_balance=12.0, fill_max_d=48.0,
                        exclude_corners=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera: &Bound<'_, PyAny>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        sparse: &Bound<'_, PyDict>,
        tracker: Option<HashMap<String, f64>>,
        scale: f64,
        histeq: &str,
        clahe_tile_size: usize,
        clahe_clip_limit: f32,
        slant_deg: f64,
        fill: bool,
        fill_sigma: f64,
        fill_gate: f64,
        fill_balance: f64,
        fill_max_d: f64,
        exclude_corners: bool,
    ) -> PyResult<Self> {
        let cam = crate::camera::to_camera_arc(camera)?;
        let sparse_settings = crate::sparse_3d::parse_settings(Some(sparse))?;
        let tracker_settings = crate::flowdep_f::settings_from(tracker)?;
        let histeq = match histeq {
            "none" => HistEqMethod::None,
            "global" => HistEqMethod::Global,
            _ => HistEqMethod::Clahe {
                tile_size: clahe_tile_size,
                clip_limit: clahe_clip_limit,
            },
        };
        let s = DenseMapperSettings {
            scale,
            histeq,
            tracker: tracker_settings,
            sparse: sparse_settings,
            slant_deg,
            fill: fill.then_some(FillParams {
                sigma_px: fill_sigma,
                gate: fill_gate,
                balance: fill_balance,
                max_d: fill_max_d,
                g: 0.0,
            }),
            exclude_corners,
        };
        Ok(Self {
            inner: DenseDepthMapper::new(
                cam,
                CameraIntrinsics { fx, fy, cx, cy },
                width,
                height,
                s,
            ),
        })
    }

    /// `process(gray, stamp, t_wc, vio, corners) -> (eta, eta_var, class, stats)`.
    ///
    /// `gray`: the raw frame (`height × width` uint8); `t_wc`: 4×4 camera-to-world pose of this
    /// frame; `vio`: the `VIOFilter` whose pose covariance the patch landmarks consume (None =
    /// no pose term); `corners`: the corner tracker's live features in raw pixels. Returns
    /// `height' × width'` arrays at the working scale: log-range (f32), its variance (f32),
    /// the class (uint8: 0 unknown, 1 measured, 2 filled) and a dict of per-frame counters.
    #[pyo3(signature = (gray, stamp, t_wc, vio=None, corners=vec![]))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        gray: PyReadonlyArray2U8<u8>,
        stamp: f64,
        t_wc: PyReadonlyArray2<f64>,
        vio: Option<&crate::vio_filter::PyVIOFilter>,
        corners: Vec<(f64, f64)>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2U8<u8>>,
        Bound<'py, PyDict>,
    )> {
        let g = gray.as_array();
        let gv: Vec<u8> = g.iter().copied().collect();
        let t = t_wc.as_array();
        let mut m = nalgebra::Matrix4::<f64>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                m[(r, c)] = t[[r, c]];
            }
        }
        let cov = vio.map(|v| v.backend_ref().pose_covariance()).flatten();
        let out = self.inner.process(&gv, stamp, &m, cov, &corners);
        let (h, w) = (out.height, out.width);
        let cls: Vec<u8> = out.class.iter().map(|c| *c as u8).collect();
        let st = PyDict::new(py);
        st.set_item("patches_live", out.stats.patches_live)?;
        st.set_item("patches_observed", out.stats.patches_observed)?;
        st.set_item("landmarks_converged", out.stats.landmarks_converged)?;
        st.set_item("items_fused", out.stats.items_fused)?;
        st.set_item("measured_frac", out.stats.measured_frac)?;
        st.set_item("filled_frac", out.stats.filled_frac)?;
        st.set_item("ms_equalise_remap", out.stats.ms_equalise_remap)?;
        st.set_item("ms_track", out.stats.ms_track)?;
        st.set_item("ms_filter", out.stats.ms_filter)?;
        st.set_item("ms_fuse_fill", out.stats.ms_fuse_fill)?;
        let _ = DenseClass::Unknown;
        Ok((
            to_array(py, &out.eta, h, w),
            to_array(py, &out.eta_var, h, w),
            to_array(py, &cls, h, w),
            st,
        ))
    }

    /// The equalised, undistorted, downscaled frame the tracker last saw.
    fn last_frame<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2U8<u8>> {
        let (w, h) = self.inner.size();
        to_array(py, self.inner.last_frame(), h, w)
    }

    /// Pinhole intrinsics of the working frame `(fx, fy, cx, cy)`.
    fn intrinsics(&self) -> (f64, f64, f64, f64) {
        let i = self.inner.intrinsics();
        (i.fx, i.fy, i.cx, i.cy)
    }

    fn size(&self) -> (usize, usize) {
        self.inner.size()
    }
}
