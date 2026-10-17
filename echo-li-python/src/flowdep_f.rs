use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;

use echo_li_core::depth::flowdep_f::{FlowDepF, FlowDepFSettings, Observation};
use nalgebra::Matrix4;

/// FlowDep-F track source (see echo-li-core/src/depth/flowdep_f.rs). Pinhole only.
#[pyclass(name = "FlowDepF")]
pub struct PyFlowDepF {
    inner: FlowDepF,
    last: Vec<Observation>,
}

fn settings_from(map: Option<HashMap<String, f64>>) -> PyResult<FlowDepFSettings> {
    let mut s = FlowDepFSettings::default();
    if let Some(m) = map {
        for (k, v) in m {
            match k.as_str() {
                "levels" => s.levels = v as usize,
                "patch_half" => s.patch_half = v as usize,
                "cell" => s.cell = v as usize,
                "min_birth_curvature" => s.min_birth_curvature = v as f32,
                "min_track_curvature" => s.min_track_curvature = v as f32,
                "d_min" => s.d_min = v,
                "d_max" => s.d_max = v,
                "scan_step_px" => s.scan_step_px = v,
                "max_scan_samples" => s.max_scan_samples = v as usize,
                "bound_sigma" => s.bound_sigma = v,
                "eta_sigma_floor" => s.eta_sigma_floor = v,
                "eta_sigma_growth" => s.eta_sigma_growth = v,
                "sigma_photo" => s.sigma_photo = v,
                "gn_max_iters" => s.gn_max_iters = v as usize,
                "gn_tol" => s.gn_tol = v,
                "max_mean_abs_residual" => s.max_mean_abs_residual = v as f32,
                "min_valid_frac" => s.min_valid_frac = v as f32,
                "ambiguity_ratio" => s.ambiguity_ratio = v as f32,
                "max_misses" => s.max_misses = v as u32,
                "keyframe_every" => s.keyframe_every = v as u32,
                "min_live_frac" => s.min_live_frac = v as f32,
                "max_patches" => s.max_patches = v as usize,
                "min_birth_dist_px" => s.min_birth_dist_px = v,
                "reanchor_br" => s.reanchor_br = v,
                "max_obs_br" => s.max_obs_br = v,
                other => {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "unknown FlowDepF setting {other}"
                    )))
                }
            }
        }
    }
    Ok(s)
}

#[pymethods]
impl PyFlowDepF {
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, width, height, settings=None))]
    fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        settings: Option<HashMap<String, f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FlowDepF::new(fx, fy, cx, cy, width, height, settings_from(settings)?),
            last: Vec::new(),
        })
    }

    /// One frame: `image` is (H, W) uint8, `t_wc` the 4x4 camera-to-world pose. Returns
    /// `{id: (u, v)}` of the accepted observations, level-0 pixels.
    fn process(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArray2<'_, u8>,
        t_wc: [[f64; 4]; 4],
    ) -> PyResult<HashMap<u64, (f64, f64)>> {
        let shape = image.shape();
        let img = image.as_slice()?;
        let mut t = Matrix4::<f64>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                t[(r, c)] = t_wc[r][c];
            }
        }
        let _ = shape;
        let inner = &mut self.inner;
        self.last = py.allow_threads(|| inner.process(img, &t));
        Ok(self.last.iter().map(|o| (o.id, (o.u, o.v))).collect())
    }

    /// Full observations of the last frame:
    /// (id, u, v, level, eta, eta_sigma, sigma_px, dir_x, dir_y, mean_abs_residual,
    ///  anchor_u, anchor_v, anchor_frame).
    fn last_observations(
        &self,
    ) -> Vec<(
        u64,
        f64,
        f64,
        usize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f32,
        (f64, f64, u32),
    )> {
        self.last
            .iter()
            .map(|o| {
                (
                    o.id,
                    o.u,
                    o.v,
                    o.level,
                    o.eta,
                    o.eta_sigma,
                    o.sigma_px,
                    o.dir[0],
                    o.dir[1],
                    o.mean_abs_residual,
                    (o.anchor_u, o.anchor_v, o.anchor_frame),
                )
            })
            .collect()
    }

    fn stats(&self) -> HashMap<String, f64> {
        let s = self.inner.stats;
        HashMap::from([
            ("live".into(), s.live as f64),
            ("observed".into(), s.observed as f64),
            ("born".into(), s.born as f64),
            ("dropped".into(), s.dropped as f64),
            ("rejected_residual".into(), s.rejected_residual as f64),
            ("rejected_ambiguous".into(), s.rejected_ambiguous as f64),
            ("rejected_curvature".into(), s.rejected_curvature as f64),
            ("rejected_outside".into(), s.rejected_outside as f64),
            ("new_anchor".into(), if s.new_anchor { 1.0 } else { 0.0 }),
            ("reanchored".into(), s.reanchored as f64),
            ("withheld_br".into(), s.withheld_br as f64),
        ])
    }

    /// `{id: (range_m, sigma_range_m)}` from the downstream filter replaces the tracker's
    /// own search bound for those patches.
    fn set_posterior(&mut self, posteriors: HashMap<u64, (f64, f64)>) {
        self.inner.set_posterior(&posteriors);
    }

    /// Debug: cost profile of one live patch on an image, see core `cost_profile`.
    fn cost_profile(
        &self,
        id: u64,
        image: PyReadonlyArray2<'_, u8>,
        t_wc: [[f64; 4]; 4],
        n: usize,
    ) -> PyResult<Vec<(f64, f32, f32, f64, f64)>> {
        let img = image.as_slice()?;
        let mut t = Matrix4::<f64>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                t[(r, c)] = t_wc[r][c];
            }
        }
        Ok(self.inner.cost_profile(id, img, &t, n))
    }

    /// (id, age, misses, last outcome: 0 ok, 1 residual, 2 ambiguous, 3 curvature, 4 outside).
    fn live_outcomes(&self) -> Vec<(u64, u32, u32, u8)> {
        self.inner.live_outcomes()
    }

    fn n_live(&self) -> usize {
        self.inner.n_live()
    }

    /// (id, level, u0, v0, eta, eta_sigma, age) per live patch; (u0, v0) = anchor pixel, level 0.
    fn live_patches(&self) -> Vec<(u64, usize, f64, f64, f64, f64, u32)> {
        self.inner.live_patches()
    }
}
