use nalgebra::Matrix4;

#[derive(Debug, Clone)]
pub struct Keyframe {
    pub gray: Vec<u8>, // Grayscale image data
    pub width: usize,
    pub height: usize,
    pub t_wc: Matrix4<f64>, // camera-to-world SE3
    pub stamp: f64,
    pub score: f64,
}

pub struct KeyframePool {
    max_keyframes: usize,
    pool: Vec<Keyframe>,
}

impl KeyframePool {
    pub fn new(max_keyframes: usize) -> Self {
        Self {
            max_keyframes,
            pool: Vec::new(),
        }
    }

    pub fn add_keyframe(&mut self, gray: Vec<u8>, width: usize, height: usize, t_wc: Matrix4<f64>, stamp: f64) {
        if self.pool.len() >= self.max_keyframes {
            self.pool.remove(0); // drop oldest
        }
        self.pool.push(Keyframe {
            gray,
            width,
            height,
            t_wc,
            stamp,
            score: 0.0,
        });
    }

    pub fn select_best(&self, t_wc_curr: &Matrix4<f64>) -> Option<&Keyframe> {
        if self.pool.is_empty() { return None; }
        
        let t_cw_curr = t_wc_curr.try_inverse().unwrap_or_else(Matrix4::identity);
        let mut best_kf = None;
        let mut best_baseline = 0.0;

        for kf in &self.pool {
            let t_curr_kf = t_cw_curr * kf.t_wc;
            let baseline = t_curr_kf.fixed_view::<3, 1>(0, 3).norm();
            if baseline > best_baseline {
                best_baseline = baseline;
                best_kf = Some(kf);
            }
        }
        best_kf
    }

    pub fn remove_keyframe(&mut self, stamp: f64) {
        self.pool.retain(|kf| kf.stamp != stamp);
    }

    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}
