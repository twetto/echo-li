use numpy::ndarray::Array1;
use numpy::PyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;

use echo_li_core::depth::sparse_3d::Sparse3DFilter;
use echo_li_core::depth::sparse_gb::SparseVogSettings;
use echo_li_core::mathematical::vision_measurement::VisionMeasurement;
use nalgebra::{Matrix3, Matrix4, Vector2};

#[pyclass(name = "Sparse3DFilter")]
pub struct PySparse3DFilter {
    inner: Sparse3DFilter,
}

#[pymethods]
impl PySparse3DFilter {
    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, **kwargs))]
    fn polar3d(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let k = intrinsics_matrix(fx, fy, cx, cy);
        let settings = parse_settings(kwargs)?;
        Ok(Self {
            inner: Sparse3DFilter::polar3d(k, settings),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, **kwargs))]
    fn invdepth3d(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let k = intrinsics_matrix(fx, fy, cx, cy);
        let settings = parse_settings(kwargs)?;
        Ok(Self {
            inner: Sparse3DFilter::invdepth3d(k, settings),
        })
    }

    fn update(&mut self, stamp: f64, feature_uvs: HashMap<u64, [f32; 2]>, t_wc: [[f64; 4]; 4]) {
        let cam_coords: HashMap<u64, Vector2<f32>> = feature_uvs
            .into_iter()
            .map(|(id, uv)| (id, Vector2::new(uv[0], uv[1])))
            .collect();
        let measurement = VisionMeasurement::new(stamp, cam_coords);
        let t = array_to_matrix4(&t_wc);
        self.inner.update(&measurement, &t, None);
    }

    fn query(&self, feature_id: u64) -> (f64, f64) {
        self.inner.query(feature_id)
    }

    fn get_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        for feat in self.inner.features_iter() {
            let feat_dict = pyo3::types::PyDict::new(py);
            feat_dict.set_item(
                "position",
                PyArray1::from_owned_array(
                    py,
                    Array1::from_vec(vec![feat.position[0], feat.position[1], feat.position[2]]),
                ),
            )?;
            feat_dict.set_item("track_length", feat.track_length)?;
            feat_dict.set_item("inlier_ratio", feat.inlier_ratio())?;
            dict.set_item(feat.feat_id, feat_dict)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("Sparse3DFilter({} features)", self.inner.feature_count())
    }
}

fn intrinsics_matrix(fx: f64, fy: f64, cx: f64, cy: f64) -> Matrix3<f64> {
    Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

fn array_to_matrix4(a: &[[f64; 4]; 4]) -> Matrix4<f64> {
    Matrix4::new(
        a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0], a[2][1],
        a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3],
    )
}

fn parse_settings(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<SparseVogSettings> {
    let mut s = SparseVogSettings::default();
    let Some(kw) = kwargs else { return Ok(s) };

    if let Some(v) = kw.get_item("max_pool_size")? {
        s.max_pool_size = v.extract()?;
    }
    if let Some(v) = kw.get_item("min_track_length")? {
        s.min_track_length = v.extract()?;
    }
    if let Some(v) = kw.get_item("sigma_pixel")? {
        s.sigma_pixel = v.extract()?;
    }
    if let Some(v) = kw.get_item("conv_inlier_ratio")? {
        s.conv_inlier_ratio = v.extract()?;
    }
    if let Some(v) = kw.get_item("conv_variance_threshold")? {
        s.conv_variance_threshold = v.extract()?;
    }
    if let Some(v) = kw.get_item("min_depth")? {
        s.min_depth = v.extract()?;
    }
    if let Some(v) = kw.get_item("max_depth")? {
        s.max_depth = v.extract()?;
    }

    Ok(s)
}
