use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::Arc;

use echo_li_core::mathematical::camera::{CameraModel, PinholeModel, RadTanModel};

pub(crate) fn to_camera_arc(cam: &Bound<'_, PyAny>) -> PyResult<Arc<dyn CameraModel>> {
    if let Ok(p) = cam.downcast::<PinholeCamera>() {
        let inner = p.borrow();
        Ok(Arc::new(PinholeModel {
            fx: inner.fx,
            fy: inner.fy,
            cx: inner.cx,
            cy: inner.cy,
        }))
    } else if let Ok(r) = cam.downcast::<RadTanCamera>() {
        let inner = r.borrow();
        Ok(Arc::new(RadTanModel {
            fx: inner.fx,
            fy: inner.fy,
            cx: inner.cx,
            cy: inner.cy,
            k1: inner.k1,
            k2: inner.k2,
            p1: inner.p1,
            p2: inner.p2,
        }))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected PinholeCamera or RadTanCamera",
        ))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PinholeCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

#[pymethods]
impl PinholeCamera {
    #[new]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = point.as_slice()?;
        if p.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("point must have 3 elements"));
        }
        let model = PinholeModel {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        };
        let v = nalgebra::Vector3::new(p[0], p[1], p[2]);
        let uv = model.project(&v);
        Ok(PyArray1::from_owned_array(py, Array1::from_vec(vec![uv[0], uv[1]])))
    }

    fn undistort<'py>(
        &self,
        py: Python<'py>,
        uv: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = uv.as_slice()?;
        if s.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("uv must have 2 elements"));
        }
        let model = PinholeModel {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        };
        let bearing = model.undistort(&nalgebra::Vector2::new(s[0], s[1]));
        Ok(PyArray1::from_owned_array(
            py,
            Array1::from_vec(vec![bearing[0], bearing[1], bearing[2]]),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "PinholeCamera(fx={}, fy={}, cx={}, cy={})",
            self.fx, self.fy, self.cx, self.cy
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct RadTanCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
}

#[pymethods]
impl RadTanCamera {
    #[new]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
        }
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = point.as_slice()?;
        if p.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("point must have 3 elements"));
        }
        let model = RadTanModel {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: self.p2,
        };
        let v = nalgebra::Vector3::new(p[0], p[1], p[2]);
        let uv = model.project(&v);
        Ok(PyArray1::from_owned_array(py, Array1::from_vec(vec![uv[0], uv[1]])))
    }

    fn undistort<'py>(
        &self,
        py: Python<'py>,
        uv: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = uv.as_slice()?;
        if s.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("uv must have 2 elements"));
        }
        let model = RadTanModel {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: self.p2,
        };
        let bearing = model.undistort(&nalgebra::Vector2::new(s[0], s[1]));
        Ok(PyArray1::from_owned_array(
            py,
            Array1::from_vec(vec![bearing[0], bearing[1], bearing[2]]),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "RadTanCamera(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, p1={}, p2={})",
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2
        )
    }
}
