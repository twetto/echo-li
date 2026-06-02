use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::depth::patch_depth::{
    FrameProducts, PatchDepthMapper, PatchDepthSettings, SparseDepthPrior,
};
use nalgebra::{Matrix4, Vector2};

use crate::camera::to_camera_arc;

#[pyclass(name = "PatchDepthMapper")]
pub struct PyPatchDepthMapper {
    inner: PatchDepthMapper,
}

#[pymethods]
impl PyPatchDepthMapper {
    #[new]
    fn new(
        camera: &Bound<'_, PyAny>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
    ) -> PyResult<Self> {
        let cam = to_camera_arc(camera)?;
        let intrinsics = CameraIntrinsics { fx, fy, cx, cy };
        let settings = PatchDepthSettings::default();
        let inner =
            PatchDepthMapper::new_undistorted_pinhole(cam, intrinsics, width, height, settings)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create PatchDepthMapper: {e}"
                    ))
                })?;
        Ok(Self { inner })
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        stamp: f64,
        frame_id: u64,
        gray_image: PyReadonlyArray2<'py, u8>,
        t_wc: [[f64; 4]; 4],
        priors: Vec<(f64, f64, f64, f64)>,
    ) -> PyResult<Option<PyObject>> {
        let shape = gray_image.shape();
        let h = shape[0];
        let w = shape[1];
        let gray_data = gray_image.as_slice()?.to_vec();

        let pose = array_to_matrix4(&t_wc);
        let frame = FrameProducts {
            frame_id,
            stamp,
            gray: gray_data,
            width: w,
            height: h,
            pose_t_wc: pose,
        };

        // Priors are log-range: (u, v, eta = ln(range), eta_var). The caller holds
        // the camera model, so it converts metric/inverse depth → eta on its side.
        let seeds: Vec<SparseDepthPrior> = priors
            .into_iter()
            .map(|(u, v, eta, eta_var)| SparseDepthPrior {
                uv: Vector2::new(u, v),
                eta,
                eta_var,
            })
            .collect();

        let dt = 0.05;
        let result = self.inner.update_with_priors(frame, &seeds, None, dt);

        match result {
            Some(output) => {
                let dict = pyo3::types::PyDict::new(py);
                let dw = output.eta.width;
                let dh = output.eta.height;

                // Output is log-range η = ln(range); convert on the Python side with
                // np.exp(eta) for range. eta_var is a relative range variance.
                let eta = Array2::from_shape_vec((dh, dw), output.eta.data)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                dict.set_item("eta", PyArray2::from_owned_array(py, eta))?;

                let eta_var = Array2::from_shape_vec((dh, dw), output.eta_var.data)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                dict.set_item("eta_var", PyArray2::from_owned_array(py, eta_var))?;

                let status_u8: Vec<u8> = output.status.data.iter().map(|s| *s as u8).collect();
                let status = Array2::from_shape_vec((dh, dw), status_u8)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                dict.set_item("status", PyArray2::from_owned_array(py, status))?;

                Ok(Some(dict.into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    #[getter]
    fn keyframe_count(&self) -> usize {
        self.inner.keyframe_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "PatchDepthMapper(keyframes={})",
            self.inner.keyframe_count()
        )
    }
}

fn array_to_matrix4(a: &[[f64; 4]; 4]) -> Matrix4<f64> {
    Matrix4::new(
        a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0], a[2][1],
        a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3],
    )
}
