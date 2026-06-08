use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

use echo_li_core::depth::sparse_3d::Sparse3DFilter;
use echo_li_core::depth::sparse_gb::{SecondOrderMode, SparseVogSettings};
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

    #[pyo3(signature = (stamp, feature_uvs, t_wc, p_vv=None))]
    fn update(
        &mut self,
        stamp: f64,
        feature_uvs: HashMap<u64, [f32; 2]>,
        t_wc: [[f64; 4]; 4],
        p_vv: Option<[[f64; 3]; 3]>,
    ) {
        let cam_coords: HashMap<u64, Vector2<f32>> = feature_uvs
            .into_iter()
            .map(|(id, uv)| (id, Vector2::new(uv[0], uv[1])))
            .collect();
        let measurement = VisionMeasurement::new(stamp, cam_coords);
        let t = array_to_matrix4(&t_wc);
        // p_vv is the 3x3 velocity (translation-rate) covariance consumed by the
        // core as process noise (q_euc = p_vv * dt^2). Rotation is not yet
        // ingested; a 6x6 pose covariance would be added as a separate kwarg.
        let p_vv_mat = p_vv.map(|m| array_to_matrix3(&m));
        self.inner.update(&measurement, &t, p_vv_mat.as_ref());
    }

    fn query(&self, feature_id: u64) -> (f64, f64) {
        self.inner.query(feature_id)
    }

    fn get_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        let chart = self.inner.chart();
        for feat in self.inner.features_iter() {
            let feat_dict = pyo3::types::PyDict::new(py);
            feat_dict.set_item(
                "position",
                PyArray1::from_owned_array(
                    py,
                    Array1::from_vec(vec![feat.position[0], feat.position[1], feat.position[2]]),
                ),
            )?;
            let cov = feat.covariance;
            let cov_arr = Array2::from_shape_vec(
                (3, 3),
                vec![
                    cov[(0, 0)],
                    cov[(0, 1)],
                    cov[(0, 2)],
                    cov[(1, 0)],
                    cov[(1, 1)],
                    cov[(1, 2)],
                    cov[(2, 0)],
                    cov[(2, 1)],
                    cov[(2, 2)],
                ],
            )
            .expect("3x3 covariance shape is valid");
            feat_dict.set_item("covariance", PyArray2::from_owned_array(py, cov_arr))?;
            let cov_euc = feat.covariance_euclidean(chart);
            let cov_euc_arr = Array2::from_shape_vec(
                (3, 3),
                vec![
                    cov_euc[(0, 0)],
                    cov_euc[(0, 1)],
                    cov_euc[(0, 2)],
                    cov_euc[(1, 0)],
                    cov_euc[(1, 1)],
                    cov_euc[(1, 2)],
                    cov_euc[(2, 0)],
                    cov_euc[(2, 1)],
                    cov_euc[(2, 2)],
                ],
            )
            .expect("3x3 euclidean covariance shape is valid");
            feat_dict.set_item(
                "covariance_euclidean",
                PyArray2::from_owned_array(py, cov_euc_arr),
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

fn array_to_matrix3(a: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    )
}

fn parse_settings(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<SparseVogSettings> {
    let mut s = SparseVogSettings::default();
    let Some(kw) = kwargs else { return Ok(s) };

    // Expose every trajectory-affecting setting so callers (e.g. the golden
    // parity fixture) can pin them identically to the pure-Python filter.
    // Rust and Python defaults have diverged (init_depth_var, ab_max,
    // min_inlier_ratio, max_depth), so relying on defaults is not safe.
    macro_rules! set {
        ($name:literal, $field:ident) => {
            if let Some(v) = kw.get_item($name)? {
                s.$field = v.extract()?;
            }
        };
    }

    set!("max_pool_size", max_pool_size);
    set!("min_track_length", min_track_length);
    set!("conv_inlier_ratio", conv_inlier_ratio);
    set!("conv_variance_threshold", conv_variance_threshold);
    set!("init_depth_var", init_depth_var);
    set!("init_invdepth_var", init_invdepth_var);
    set!("sigma_pixel", sigma_pixel);
    set!("uniform_z_max", uniform_z_max);
    set!("uniform_rho_max", uniform_rho_max);
    set!("uniform_d_min", uniform_d_min);
    set!("uniform_d_max", uniform_d_max);
    set!("a_init", a_init);
    set!("b_init", b_init);
    set!("ab_min", ab_min);
    set!("ab_max", ab_max);
    set!("min_inlier_ratio", min_inlier_ratio);
    set!("mahalanobis_reset_chi2", mahalanobis_reset_chi2);
    set!("process_depth_var", process_depth_var);
    set!("min_parallax", min_parallax);
    set!("min_cos_sim", min_cos_sim);
    set!("min_depth", min_depth);
    set!("max_depth", max_depth);
    set!("reanchor_flow_px", reanchor_flow_px);
    set!("use_equivariant_output", use_equivariant_output);
    set!("iekf_iterations", iekf_iterations);
    set!("range_walk_var", range_walk_var);

    if let Some(v) = kw.get_item("second_order_mode")? {
        let mode: String = v.extract()?;
        s.second_order_mode = match mode.as_str() {
            "off" => SecondOrderMode::Off,
            "analytic" | "second_order" => SecondOrderMode::Analytic,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown second_order_mode {other:?} (expected \"off\" or \"analytic\")"
                )));
            }
        };
    }

    Ok(s)
}
