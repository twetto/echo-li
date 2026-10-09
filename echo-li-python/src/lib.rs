use pyo3::prelude::*;

mod camera;
mod frontend;
mod msceqf_native;
mod patch_depth;
mod sparse_3d;
mod stereo;
mod vio_filter;

#[pymodule]
fn _echo_li(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<camera::PinholeCamera>()?;
    m.add_class::<camera::RadTanCamera>()?;
    m.add_class::<camera::EquidistantCamera>()?;
    m.add_class::<frontend::FrontendConfig>()?;
    m.add_class::<frontend::PyFrontend>()?;
    m.add_class::<vio_filter::PyVIOFilter>()?;
    m.add_class::<msceqf_native::PyMSCEqFNativeFilter>()?;
    m.add_class::<sparse_3d::PySparse3DFilter>()?;
    m.add_class::<patch_depth::PyPatchDepthMapper>()?;
    m.add_class::<stereo::PyStereo>()?;
    m.add_function(pyo3::wrap_pyfunction!(static_initialize, m)?)?;
    Ok(())
}

/// Detect an at-rest window and seed the IMU state from it — direct binding to
/// `echo_li_core::initialization::static_initialize` (port of OpenVINS
/// `StaticInitializer`). `stamps`/`gyr`/`acc` are the raw IMU stream.
/// Returns `(stamp, R_wb 3x3, bg 3, ba 3, std_att, std_pos, std_vel)` or None when the
/// excitation pattern does not match, so the caller keeps buffering.
#[pyo3::pyfunction]
#[pyo3(signature = (stamps, gyr, acc, window_time=1.0, imu_thresh=0.1, gravity_mag=9.81, wait_for_jerk=false))]
#[allow(clippy::too_many_arguments)]
fn static_initialize(
    py: pyo3::Python<'_>,
    stamps: Vec<f64>,
    gyr: Vec<[f64; 3]>,
    acc: Vec<[f64; 3]>,
    window_time: f64,
    imu_thresh: f64,
    gravity_mag: f64,
    wait_for_jerk: bool,
) -> Option<(
    f64,
    pyo3::Bound<'_, numpy::PyArray2<f64>>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    f64,
)> {
    use echo_li_core::initialization::static_initialize as si;
    use echo_li_core::mathematical::IMUVelocity;
    use nalgebra::Vector3;
    let n = stamps.len().min(gyr.len()).min(acc.len());
    let imu: Vec<IMUVelocity> = (0..n)
        .map(|i| IMUVelocity {
            stamp: stamps[i],
            gyr: Vector3::new(gyr[i][0], gyr[i][1], gyr[i][2]),
            acc: Vector3::new(acc[i][0], acc[i][1], acc[i][2]),
            gyr_bias_vel: Vector3::zeros(),
            acc_bias_vel: Vector3::zeros(),
        })
        .collect();
    let out = si(&imu, window_time, imu_thresh, gravity_mag, wait_for_jerk)?;
    let m = out.r_wb.as_matrix();
    let d: Vec<f64> = (0..3)
        .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
        .collect();
    let r = numpy::PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array2::from_shape_vec((3, 3), d).unwrap(),
    );
    Some((
        out.stamp,
        r,
        vec![out.bg[0], out.bg[1], out.bg[2]],
        vec![out.ba[0], out.ba[1], out.ba[2]],
        out.std_att,
        out.std_pos,
        out.std_vel,
    ))
}
