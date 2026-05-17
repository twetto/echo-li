use pyo3::prelude::*;

mod camera;
mod frontend;
mod patch_depth;
mod sparse_3d;
mod vio_filter;

#[pymodule]
fn _echo_li(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<camera::PinholeCamera>()?;
    m.add_class::<camera::RadTanCamera>()?;
    m.add_class::<frontend::FrontendConfig>()?;
    m.add_class::<frontend::PyFrontend>()?;
    m.add_class::<vio_filter::PyVIOFilter>()?;
    m.add_class::<sparse_3d::PySparse3DFilter>()?;
    m.add_class::<patch_depth::PyPatchDepthMapper>()?;
    Ok(())
}
