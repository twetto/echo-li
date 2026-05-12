use std::collections::BTreeMap;

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};

pub type Timestamp = f64;
pub type FeatureId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    pub fn from_matrix(k: &Matrix3<f64>) -> Self {
        Self {
            fx: k[(0, 0)],
            fy: k[(1, 1)],
            cx: k[(0, 2)],
            cy: k[(1, 2)],
        }
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn normalize_pixel(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        Vector3::new(
            (uv[0] - self.cx) / self.fx,
            (uv[1] - self.cy) / self.fy,
            1.0,
        )
    }

    pub fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        Vector2::new(
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    pub t_wc: Matrix4<f64>,
}

impl CameraPose {
    pub fn identity() -> Self {
        Self {
            t_wc: Matrix4::identity(),
        }
    }

    pub fn new(t_wc: Matrix4<f64>) -> Self {
        Self { t_wc }
    }

    pub fn inverse(&self) -> Matrix4<f64> {
        self.t_wc.try_inverse().unwrap_or_else(Matrix4::identity)
    }

    pub fn current_from_reference(&self, reference: &Self) -> Matrix4<f64> {
        self.inverse() * reference.t_wc
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureObservation {
    pub id: FeatureId,
    pub uv: Vector2<f64>,
    pub stamp: Timestamp,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeatureTrack {
    pub id: FeatureId,
    pub observations: Vec<FeatureObservation>,
}

impl FeatureTrack {
    pub fn new(id: FeatureId) -> Self {
        Self {
            id,
            observations: Vec::new(),
        }
    }

    pub fn push(&mut self, uv: Vector2<f64>, stamp: Timestamp) {
        self.observations.push(FeatureObservation {
            id: self.id,
            uv,
            stamp,
        });
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseDepthSeed {
    pub id: FeatureId,
    pub uv: Vector2<f64>,
    pub depth: f64,
    pub variance: f64,
    pub stamp: Timestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthStatus {
    Unknown,
    Candidate,
    Valid,
    Outlier,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DepthMap<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

impl<T: Clone> DepthMap<T> {
    pub fn new(width: usize, height: usize, value: T) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> anyhow::Result<Self> {
        anyhow::ensure!(data.len() == width * height, "depth map size mismatch");
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        (x < self.width && y < self.height).then(|| &self.data[y * self.width + x])
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        (x < self.width && y < self.height).then(|| &mut self.data[y * self.width + x])
    }
}

pub type DepthImage = DepthMap<f32>;
pub type VarianceImage = DepthMap<f32>;
pub type StatusImage = DepthMap<DepthStatus>;
pub type FeatureObservations = BTreeMap<FeatureId, Vector2<f64>>;
