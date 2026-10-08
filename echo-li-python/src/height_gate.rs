use pyo3::prelude::*;

use echo_li_core::config::VIOConfig;
use echo_li_core::depth::height_gate::{FlightHeightGate, FlightHeightGateSettings};

/// Occupancy flight-height gate (echo_li_core::depth::height_gate), built
/// from the params file's `FlightHeightGate:` section.
#[pyclass(name = "FlightHeightGate")]
pub struct PyFlightHeightGate {
    inner: FlightHeightGate,
}

#[pymethods]
impl PyFlightHeightGate {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&str>) -> PyResult<Self> {
        let settings = match config {
            Some(path) => VIOConfig::from_yaml(path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load config {path}: {e}")))?
                .flight_height_gate
                .unwrap_or_default(),
            None => FlightHeightGateSettings::default(),
        };
        let inner = FlightHeightGate::new(settings).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.settings().enabled
    }

    /// (low, high) edges of the accepted height band, metres.
    fn band(&self) -> (f64, f64) {
        self.inner.band()
    }

    fn update_range(&mut self, t_s: f64, range_m: f64) {
        self.inner.update_range(t_s, range_m);
    }

    /// (open, vio_height or None, range_height or None) for a frame at t_s
    /// with the VIO body z (odom frame).
    fn evaluate(&mut self, t_s: f64, vio_z: f64) -> (bool, Option<f64>, Option<f64>) {
        let d = self.inner.evaluate(t_s, vio_z);
        (d.open, d.vio_height, d.range_height)
    }
}
