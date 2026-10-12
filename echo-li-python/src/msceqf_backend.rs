//! Python binding for [`echo_li_core::backend::MsceqfBackend`].
//!
//! This exists to VALIDATE the Rust port against the Python orchestration it replaces.
//! The reference harness drives `MSCEqFNativeFilter` primitives from a Python loop;
//! this class drives the same filter from the Rust frame step, so the two can be run on
//! identical data and scored against each other. Once the port is trusted, this is also
//! how a harness picks the MSCEqF backend.
//!
//! Feature coordinates are RAW PIXELS here, not the normalized pairs the primitive
//! binding takes: the backend owns a camera model and normalizes internally.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use echo_li_core::backend::{MsceqfBackend, MsceqfSettings, VioBackend};
use echo_li_core::mathematical::camera::CameraModel;
use echo_li_core::mathematical::imu_velocity::IMUVelocity;
use echo_li_core::mathematical::msceqf_filter::{MSCEqFFilter, ProcessNoise, SystemOrigin};
use echo_li_core::mathematical::vision_measurement::VisionMeasurement;
use echo_lie::{SE23, SE3, SO3};
use nalgebra::{Matrix3, Matrix4, SMatrix, Vector2, Vector3, Vector6};

fn mat3(a: PyReadonlyArray2<'_, f64>) -> Matrix3<f64> {
    let v = a.as_array();
    Matrix3::from_fn(|i, j| v[[i, j]])
}
fn mat4(a: PyReadonlyArray2<'_, f64>) -> Matrix4<f64> {
    let v = a.as_array();
    Matrix4::from_fn(|i, j| v[[i, j]])
}
fn vec3(a: &PyReadonlyArray1<'_, f64>) -> Vector3<f64> {
    let v = a.as_array();
    Vector3::new(v[0], v[1], v[2])
}
fn vec6(a: &PyReadonlyArray1<'_, f64>) -> Vector6<f64> {
    let v = a.as_array();
    Vector6::new(v[0], v[1], v[2], v[3], v[4], v[5])
}

#[pyclass(name = "MsceqfBackend")]
pub struct PyMsceqfBackend {
    inner: MsceqfBackend,
}

#[pymethods]
impl PyMsceqfBackend {
    /// `camera` is a real camera model (PinholeCamera / RadTanCamera /
    /// EquidistantCamera), NOT four intrinsics. The backend undistorts through it to
    /// get bearings, so handing a bare pinhole to a distorted lens silently biases
    /// every observation -- which is what an earlier version of this binding did, and
    /// it diverged on the first real-rig bag it met.
    ///
    /// `fx` is used only to convert `pixel_std` into the normalized units the filter
    /// works in.
    #[new]
    #[pyo3(signature = (
        r0, p0, v0, b0, s0, g, d_std, delta_std, e_std,
        accel_density, gyro_density, accel_rw, gyro_rw, transition_order,
        camera, fx,
        num_clones=11, pixel_std=1.0, chi2_mult=1.0, slam_chi2_mult=0.0,
        max_track=200, slam_min_len=0, max_slam=0, curvature=true, update_span=0,
        max_promote_range=0.0, direct_product_bias=false,
        clone_at_frame_stamp=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        r0: PyReadonlyArray2<'_, f64>,
        p0: PyReadonlyArray1<'_, f64>,
        v0: PyReadonlyArray1<'_, f64>,
        b0: PyReadonlyArray1<'_, f64>,
        s0: PyReadonlyArray2<'_, f64>,
        g: PyReadonlyArray1<'_, f64>,
        d_std: PyReadonlyArray1<'_, f64>,
        delta_std: PyReadonlyArray1<'_, f64>,
        e_std: PyReadonlyArray1<'_, f64>,
        accel_density: f64,
        gyro_density: f64,
        accel_rw: f64,
        gyro_rw: f64,
        transition_order: usize,
        camera: &Bound<'_, PyAny>,
        fx: f64,
        num_clones: usize,
        pixel_std: f64,
        chi2_mult: f64,
        slam_chi2_mult: f64,
        max_track: usize,
        slam_min_len: usize,
        max_slam: usize,
        curvature: bool,
        update_span: usize,
        max_promote_range: f64,
        direct_product_bias: bool,
        clone_at_frame_stamp: bool,
    ) -> PyResult<Self> {
        let t0 = SE23::new(SO3::from_matrix(&mat3(r0)), vec3(&p0), vec3(&v0));
        let origin = SystemOrigin::new(t0, vec6(&b0), SE3::from_matrix(&mat4(s0)), vec3(&g));

        let ds = d_std.as_array();
        let d_init = SMatrix::<f64, 9, 9>::from_fn(|i, j| if i == j { ds[i] * ds[i] } else { 0.0 });
        let dl = delta_std.as_array();
        let delta_init =
            SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { dl[i] * dl[i] } else { 0.0 });
        let es = e_std.as_array();
        let e_init = SMatrix::<f64, 6, 6>::from_fn(|i, j| if i == j { es[i] * es[i] } else { 0.0 });

        let noise = ProcessNoise {
            angular_velocity_std: gyro_density,
            acceleration_std: accel_density,
            angular_velocity_bias_std: gyro_rw,
            acceleration_bias_std: accel_rw,
            state_transition_order: transition_order,
        };
        let filter = if direct_product_bias {
            MSCEqFFilter::new_direct_product_bias(origin, &d_init, &delta_init, &e_init, noise)
        } else {
            MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, noise)
        };
        let cam: Arc<dyn CameraModel> = crate::camera::to_camera_arc(camera)?;
        let settings = MsceqfSettings {
            num_clones,
            pixel_std,
            chi2_mult,
            slam_chi2_mult: (slam_chi2_mult > 0.0).then_some(slam_chi2_mult),
            max_track,
            slam_min_len: (slam_min_len > 0).then_some(slam_min_len),
            max_slam,
            curvature,
            update_span,
            max_promote_range,
            direct_product_bias,
        };
        let mut inner = MsceqfBackend::new(filter, cam, settings);
        inner.set_clone_at_frame_stamp(clone_at_frame_stamp);
        Ok(Self { inner })
    }

    /// One IMU sample. The backend derives dt from consecutive stamps, so the FIRST
    /// call only establishes a timestamp and propagates nothing.
    fn process_imu(
        &mut self,
        stamp: f64,
        gyro: PyReadonlyArray1<'_, f64>,
        accel: PyReadonlyArray1<'_, f64>,
    ) {
        self.inner
            .process_imu(IMUVelocity::new(stamp, vec3(&gyro), vec3(&accel)));
    }

    /// One frame: `{track_id: (u_px, v_px)}` in RAW PIXELS.
    fn process_vision(&mut self, stamp: f64, features: HashMap<u64, (f64, f64)>) {
        let coords = features
            .into_iter()
            .map(|(id, (u, v))| (id, Vector2::new(u as f32, v as f32)))
            .collect();
        self.inner
            .process_vision(VisionMeasurement::new(stamp, coords), &HashMap::new());
    }

    /// `(position, R_body_to_world, velocity_BODY)`.
    fn nav_pose<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let nav = self.inner.nav_estimate();
        let r = nav.pose.rotation.as_matrix();
        let mut rm = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                rm[[i, j]] = r[(i, j)];
            }
        }
        (
            PyArray1::from_slice(py, nav.pose.translation.as_slice()),
            PyArray2::from_array(py, &rm),
            PyArray1::from_slice(py, nav.velocity.as_slice()),
        )
    }

    /// Assert zero velocity. Returns False if the chi2 gate refused it.
    fn zero_velocity_update(&mut self, sigma_v: f64) -> bool {
        self.inner.zero_velocity_update(sigma_v)
    }

    /// Cumulative per-reason structureless drop counts:
    /// `[too_few_obs, stale_clone, triangulation_failed, jacobian_failed, chi2_rejected]`.
    fn msc_reject_counts(&self) -> [u64; 5] {
        self.inner.msc_reject_counts()
    }

    /// Drain the per-track `chi2 / gate` ratios recorded since the last call.
    fn take_msc_chi2_ratios<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.take_msc_chi2_ratios())
    }

    /// Core-state covariance diagonal [att(3) vel(3) pos(3) bias(6) | E(6)], for
    /// watching where a divergence starts.
    fn nav_cov_diag<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let c = &self.inner.filter().cov;
        let n = 21.min(c.nrows());
        let d: Vec<f64> = (0..n).map(|i| c[(i, i)]).collect();
        PyArray1::from_slice(py, &d)
    }

    /// (offered, accepted) structureless tracks.
    fn msc_update_counts(&self) -> (u64, u64) {
        self.inner.msc_update_counts()
    }

    /// In-state landmark positions in the world frame, by track id.
    fn landmarks_world(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        use echo_li_core::backend::LandmarkMap;
        let d = pyo3::types::PyDict::new(py);
        for (id, p) in self.inner.landmarks_world() {
            d.set_item(id, PyArray1::from_slice(py, p.as_slice()))?;
        }
        Ok(d.unbind())
    }

    /// Metric range of each in-state landmark in its OWN anchor camera frame.
    ///
    /// Frame-clean: `landmarks_world` composes with the anchor clone pose and is then
    /// usually differenced against the CURRENT body position, which mixes frames and
    /// inflates the number for any landmark whose anchor is not the current pose. This
    /// is the quantity the filter actually assigned.
    fn landmark_ranges(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let f = self.inner.filter();
        for (j, id) in self.inner.landmark_ids_snapshot().iter().enumerate() {
            d.set_item(id, f.x.landmarks[j].point().norm())?;
        }
        Ok(d.unbind())
    }

    fn n_clones(&self) -> usize {
        self.inner.n_clones()
    }
    fn n_slam(&self) -> usize {
        self.inner.n_slam()
    }
}
