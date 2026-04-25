use nalgebra::{Matrix3, Matrix4, Vector3};
use crate::depth::flowdep_kernels::{depth_densification, bilinear_splatting, bilinear_splatting_ab, vogiatzis_update};
use crate::depth::keyframe_pool::KeyframePool;
use crate::depth::sparse_gb::SparseVogSettings;

pub trait OpticalFlowBackend {
    /// Compute dense optical flow between two grayscale frames.
    /// Returns (H, W, 2) float32 flow array (u_curr - u_prev convention).
    fn compute(&self, prev_gray: &[u8], curr_gray: &[u8], h: usize, w: usize) -> Vec<f32>;
}

pub struct FlowDepFilter {
    k: Matrix3<f64>,
    settings: SparseVogSettings,
    sigma_norm: f64,

    invdepth_state: Option<Vec<f32>>,
    invdepth_var: Option<Vec<f32>>,
    a_state: Option<Vec<f32>>,
    b_state: Option<Vec<f32>>,

    keyframe_pool: KeyframePool,
    
    prev_gray: Option<Vec<u8>>,
    prev_t_wc: Option<Matrix4<f64>>,
    prev_stamp: f64,
    
    width: usize,
    height: usize,
}

impl FlowDepFilter {
    pub fn new(k: Matrix3<f64>, settings: SparseVogSettings, width: usize, height: usize) -> Self {
        let sigma_norm = settings.sigma_pixel / k[(0, 0)];
        
        Self {
            k,
            settings,
            sigma_norm,
            invdepth_state: None,
            invdepth_var: None,
            a_state: None,
            b_state: None,
            keyframe_pool: KeyframePool::new(5),
            prev_gray: None,
            prev_t_wc: None,
            prev_stamp: -1.0,
            width,
            height,
        }
    }

    pub fn process_frame<F: OpticalFlowBackend>(
        &mut self,
        curr_gray: &[u8],
        t_wc_curr: &Matrix4<f64>,
        stamp: f64,
        flow_backend: &F,
    ) -> bool {
        if self.prev_gray.is_none() {
            self.prev_gray = Some(curr_gray.to_vec());
            self.prev_t_wc = Some(*t_wc_curr);
            self.prev_stamp = stamp;
            self.keyframe_pool.add_keyframe(curr_gray.to_vec(), self.width, self.height, *t_wc_curr, stamp);
            return false;
        }

        let mut use_keyframe = false;
        let mut r_curr_ref = Matrix3::identity();
        let mut t_curr_ref = Vector3::zeros();
        let mut flow = Vec::new();

        if let Some(best_kf) = self.keyframe_pool.select_best(t_wc_curr) {
            flow = flow_backend.compute(&best_kf.gray, curr_gray, self.height, self.width);
            
            let t_cw_curr = t_wc_curr.try_inverse().unwrap_or_else(Matrix4::identity);
            let t_curr_kf = t_cw_curr * best_kf.t_wc;
            r_curr_ref = t_curr_kf.fixed_view::<3, 3>(0, 0).into_owned();
            t_curr_ref = t_curr_kf.fixed_view::<3, 1>(0, 3).into_owned();
            use_keyframe = true;
        }

        if !use_keyframe {
            let prev_gray = self.prev_gray.as_ref().unwrap();
            flow = flow_backend.compute(prev_gray, curr_gray, self.height, self.width);
            let t_cw_curr = t_wc_curr.try_inverse().unwrap_or_else(Matrix4::identity);
            let t_curr_prev = t_cw_curr * self.prev_t_wc.unwrap();
            r_curr_ref = t_curr_prev.fixed_view::<3, 3>(0, 0).into_owned();
            t_curr_ref = t_curr_prev.fixed_view::<3, 1>(0, 3).into_owned();
        }

        self.keyframe_pool.add_keyframe(curr_gray.to_vec(), self.width, self.height, *t_wc_curr, stamp);

        let (observed_invdepth, geom_drive_map) = depth_densification(
            &self.k, &r_curr_ref, &t_curr_ref, &flow, self.height, self.width
        );

        let dt = (stamp - self.prev_stamp).max(0.0);
        let (predicted_invdepth, predicted_var, predicted_a, predicted_b) = self.predict(t_wc_curr, dt);

        if predicted_invdepth.is_none() {
            self.init_all_states(&observed_invdepth, &geom_drive_map);
        } else {
            let (updated_invdepth, updated_var, updated_a, updated_b) = vogiatzis_update(
                &predicted_invdepth.unwrap(),
                &predicted_var.unwrap(),
                &predicted_a.unwrap(),
                &predicted_b.unwrap(),
                &observed_invdepth,
                &geom_drive_map,
                self.sigma_norm,
                self.settings.init_invdepth_var as f32,
                self.settings.uniform_rho_max,
                self.settings.a_init as f32,
                self.settings.b_init as f32,
                self.settings.ab_min as f32,
                self.settings.ab_max as f32,
                self.settings.min_inlier_ratio as f32,
                self.settings.mahalanobis_reset_chi2 as f32,
                self.height,
                self.width
            );
            self.invdepth_state = Some(updated_invdepth);
            self.invdepth_var = Some(updated_var);
            self.a_state = Some(updated_a);
            self.b_state = Some(updated_b);
        }

        self.prev_gray = Some(curr_gray.to_vec());
        self.prev_t_wc = Some(*t_wc_curr);
        self.prev_stamp = stamp;

        true
    }

    fn predict(&self, t_wc_curr: &Matrix4<f64>, _dt: f64) -> (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>) {
        if self.invdepth_state.is_none() {
            return (None, None, None, None);
        }

        let invdepth_state = self.invdepth_state.as_ref().unwrap();
        let invdepth_var = self.invdepth_var.as_ref().unwrap();
        let a_state = self.a_state.as_ref().unwrap();
        let b_state = self.b_state.as_ref().unwrap();

        let t_cw_curr = t_wc_curr.try_inverse().unwrap_or_else(Matrix4::identity);
        let t_curr_prev = t_cw_curr * self.prev_t_wc.unwrap();
        let r = t_curr_prev.fixed_view::<3, 3>(0, 0).into_owned();
        let t = t_curr_prev.fixed_view::<3, 1>(0, 3).into_owned();

        let mut u_proj = Vec::new();
        let mut v_proj = Vec::new();
        let mut inv_z_proj = Vec::new();
        let mut propagated_var = Vec::new();
        let mut a_vals = Vec::new();
        let mut b_vals = Vec::new();

        let fx = self.k[(0, 0)];
        let fy = self.k[(1, 1)];
        let cx = self.k[(0, 2)];
        let cy = self.k[(1, 2)];

        for v in 0..self.height {
            for u in 0..self.width {
                let idx = v * self.width + u;
                let rho = invdepth_state[idx] as f64;
                let var = invdepth_var[idx] as f64;
                
                if rho > 0.0 && var < 3.0 { // propagate_crit_var = 3.0
                    let x_norm = (u as f64 - cx) / fx;
                    let y_norm = (v as f64 - cy) / fy;
                    let z = 1.0 / rho;
                    
                    let p_prev = Vector3::new(x_norm * z, y_norm * z, z);
                    let p_curr = r * p_prev + t;
                    let z_new = p_curr[2];
                    
                    if z_new > 0.1 {
                        let inv_z_new = 1.0 / z_new;
                        u_proj.push(((p_curr[0] * inv_z_new) * fx + cx) as f32);
                        v_proj.push(((p_curr[1] * inv_z_new) * fy + cy) as f32);
                        inv_z_proj.push(inv_z_new as f32);
                        
                        let g = r[(2, 0)] * x_norm + r[(2, 1)] * y_norm + r[(2, 2)];
                        let j = g * (z / z_new).powi(2);
                        let proc_noise = self.settings.process_depth_var;
                        propagated_var.push(((j * j * var) + proc_noise) as f32);
                        
                        a_vals.push(a_state[idx]);
                        b_vals.push(b_state[idx]);
                    }
                }
            }
        }

        let (pred_inv, pred_var, w_accum) = bilinear_splatting(
            &u_proj, &v_proj, &inv_z_proj, &propagated_var, self.height, self.width
        );
        let (pred_a_accum, pred_b_accum, _) = bilinear_splatting_ab(
            &u_proj, &v_proj, &a_vals, &b_vals, self.height, self.width
        );

        let mut final_inv = vec![-1.0f32; self.height * self.width];
        let mut final_var = vec![self.settings.init_invdepth_var as f32; self.height * self.width];
        let mut final_a = vec![self.settings.a_init as f32; self.height * self.width];
        let mut final_b = vec![self.settings.b_init as f32; self.height * self.width];

        for i in 0..(self.height * self.width) {
            let w = w_accum[i];
            if w > 1e-6 {
                final_inv[i] = pred_inv[i] / w;
                final_var[i] = pred_var[i] / w;
                final_a[i] = pred_a_accum[i] / w;
                final_b[i] = pred_b_accum[i] / w;
            }
        }

        (Some(final_inv), Some(final_var), Some(final_a), Some(final_b))
    }

    fn init_all_states(&mut self, inv_depth_map: &[f32], geom_drive: &[f32]) {
        let mut inv_var = vec![self.settings.init_invdepth_var as f32; self.height * self.width];
        for i in 0..(self.height * self.width) {
            if inv_depth_map[i] > 0.0 {
                let drive = geom_drive[i] as f64;
                inv_var[i] = (self.sigma_norm / drive.max(1e-8)).powi(2) as f32;
            }
        }
        self.invdepth_state = Some(inv_depth_map.to_vec());
        self.invdepth_var = Some(inv_var);
        self.a_state = Some(vec![self.settings.a_init as f32; self.height * self.width]);
        self.b_state = Some(vec![self.settings.b_init as f32; self.height * self.width]);
    }
}
