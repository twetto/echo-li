use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use camera_geometry::CameraProjection;
use echo_li_core::config::VIOConfig;
use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::depth::occupancy::LocalOccupancyMap;
use echo_li_core::depth::patch_depth::{
    FrameProducts, PatchDepthMapper, PatchDepthSeedCoordinates, SparseDepthPrior,
};
use echo_li_core::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use echo_li_core::initialization::estimate_initial_pose;
use echo_li_core::mathematical::camera::CameraModel;
use echo_li_core::mathematical::imu_velocity::IMUVelocity;
use echo_li_core::mathematical::vio_state::{VIOSensorState, VIOState};
use echo_li_core::mathematical::vision_measurement::VisionMeasurement;
use echo_li_core::{VIOFilter, landmarks_to_global};
use echo_li_ros2::{NSEC_PER_SEC, best_effort_qos, quat_to_se3, stamp_ns, to_stamp};
use echo_lie::SE3;
use futures::StreamExt;
use nalgebra::{Matrix3, Matrix4, Vector2, Vector3, Vector6};
use rudolf_v::camera::{
    CameraIntrinsics as RudolfCameraIntrinsics, DistortionModel as RudolfDistortionModel,
};
use rudolf_v::fast::Feature;
use rudolf_v::frontend::{Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image as RudolfImage;

// ── Parameters ──────────────────────────────────────────────────────────────

struct NodeParams {
    config_path: String,
    imu_topic: String,
    image_topic: String,
    odometry_topic: String,
    odom_frame: String,
    body_frame: String,
    publish_tf: bool,
    camera_offset_ns: i64,
    width: usize,
    height: usize,
    intrinsics: [f64; 4],
    distortion: [f64; 4],
    camera_model: String,
    t_bs: Matrix4<f64>,
    n_init: usize,
    imu_qos_depth: usize,
    image_qos_depth: usize,
    image_qos_reliable: bool,
    max_imu_queue: usize,
    max_image_queue: usize,
    stats_period_sec: f64,
    patch_depth_enabled: bool,
    occupancy_enabled: bool,
    mapping_stride: usize,
    mocap_topic: String,
    vio_topic: String,
    #[cfg(feature = "rerun")]
    rerun_enabled: bool,
    #[cfg(feature = "rerun")]
    rerun_url: String,
    #[cfg(feature = "rerun")]
    rerun_world_stride: usize,
}

// ── Parameter helpers ────────────────────────────────────────────────────────
// r2r populates node.params from --ros-args -p name:=value and --params-file.

fn get_str(node: &r2r::Node, name: &str, default: &str) -> String {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::String(s)) => s.clone(),
        _ => default.to_string(),
    }
}

fn get_f64(node: &r2r::Node, name: &str, default: f64) -> f64 {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::Double(v)) => *v,
        Some(r2r::ParameterValue::Integer(v)) => *v as f64,
        _ => default,
    }
}

fn get_i64(node: &r2r::Node, name: &str, default: i64) -> i64 {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::Integer(v)) => *v,
        Some(r2r::ParameterValue::Double(v)) => *v as i64,
        _ => default,
    }
}

fn get_bool(node: &r2r::Node, name: &str, default: bool) -> bool {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::Bool(v)) => *v,
        _ => default,
    }
}

fn get_f64_vec(node: &r2r::Node, name: &str) -> Option<Vec<f64>> {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::DoubleArray(v)) => Some(v.clone()),
        _ => None,
    }
}

fn load_params(node: &r2r::Node) -> NodeParams {
    let config_path = get_str(node, "echo_config_path", "config/eqvio_voxl2.yaml");
    let offset_sec = get_f64(node, "camera_time_offset_sec", -0.0264);

    let [fx, fy, cx, cy] = if let Some(v) = get_f64_vec(node, "intrinsics") {
        assert!(v.len() == 4, "intrinsics must have 4 elements");
        [v[0], v[1], v[2], v[3]]
    } else {
        [
            get_f64(node, "fx", 462.459_008_454_092_61),
            get_f64(node, "fy", 462.540_269_670_589_45),
            get_f64(node, "cx", 670.414_298_091_828),
            get_f64(node, "cy", 398.445_515_810_873_47),
        ]
    };

    let [k1, k2, k3, k4] = if let Some(v) = get_f64_vec(node, "distortion_coefficients") {
        assert!(v.len() == 4, "distortion_coefficients must have 4 elements");
        [v[0], v[1], v[2], v[3]]
    } else {
        [
            get_f64(node, "k1", 0.067_992_633_732_965_42),
            get_f64(node, "k2", 0.002_231_546_482_175_450_2),
            get_f64(node, "k3", 0.004_627_575_172_719_192),
            get_f64(node, "k4", -0.003_319_341_100_961_203_3),
        ]
    };

    let t_bs = if let Some(v) = get_f64_vec(node, "t_bs") {
        assert!(v.len() == 16, "t_bs must have 16 elements");
        Matrix4::from_row_slice(&v)
    } else {
        let t_bs_default: [f64; 16] = [
            0.0, 0.0, 1.0, 0.037, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0006, 0.0, 0.0, 0.0, 1.0,
        ];
        Matrix4::from_row_slice(&t_bs_default)
    };

    NodeParams {
        config_path,
        imu_topic: get_str(node, "imu_topic", "/voxl/raw_imu"),
        image_topic: get_str(node, "image_topic", "/tracking_front/decoded"),
        odometry_topic: get_str(node, "odometry_topic", "/echo_li/odometry"),
        odom_frame: get_str(node, "odom_frame_id", "echo_li_odom"),
        body_frame: get_str(node, "body_frame_id", "imu_link"),
        publish_tf: get_bool(node, "publish_tf", true),
        camera_offset_ns: (offset_sec * NSEC_PER_SEC as f64).round() as i64,
        width: get_i64(node, "image_width", 1280) as usize,
        height: get_i64(node, "image_height", 800) as usize,
        intrinsics: [fx, fy, cx, cy],
        distortion: [k1, k2, k3, k4],
        camera_model: get_str(node, "camera_model", "equidistant"),
        t_bs,
        n_init: get_i64(node, "initialization_imu_samples", 100) as usize,
        imu_qos_depth: get_i64(node, "imu_qos_depth", 2000) as usize,
        image_qos_depth: get_i64(node, "image_qos_depth", 5) as usize,
        image_qos_reliable: get_str(node, "image_qos_reliability", "best_effort") == "reliable",
        max_imu_queue: get_i64(node, "max_imu_queue", 5000) as usize,
        max_image_queue: get_i64(node, "max_image_queue", 4) as usize,
        stats_period_sec: get_f64(node, "statistics_period_sec", 5.0),
        patch_depth_enabled: get_bool(node, "patch_depth_enabled", true),
        occupancy_enabled: get_bool(node, "occupancy_enabled", true),
        mapping_stride: (get_i64(node, "mapping_stride", 1) as usize).max(1),
        mocap_topic: get_str(node, "mocap_topic", ""),
        vio_topic: get_str(node, "vio_topic", ""),
        #[cfg(feature = "rerun")]
        rerun_enabled: get_bool(node, "rerun_enabled", false),
        #[cfg(feature = "rerun")]
        rerun_url: get_str(node, "rerun_url", ""),
        #[cfg(feature = "rerun")]
        rerun_world_stride: (get_i64(node, "rerun_world_stride", 3) as usize).max(1),
    }
}

// ── Reference trajectory tracker ────────────────────────────────────────────

struct ReferenceTrack {
    stamps_ns: Vec<i64>,
    positions: Vec<[f64; 3]>,
    offsets_ns: VecDeque<i64>,
    last_stamp_ns: Option<i64>,
    last_pose: Option<([f64; 3], [f64; 4])>,
}

impl ReferenceTrack {
    fn new() -> Self {
        Self {
            stamps_ns: Vec::new(),
            positions: Vec::new(),
            offsets_ns: VecDeque::with_capacity(2000),
            last_stamp_ns: None,
            last_pose: None,
        }
    }

    fn add(
        &mut self,
        stamp_ns: i64,
        position: [f64; 3],
        orientation: [f64; 4],
        latest_imu_ns: Option<i64>,
    ) {
        let pose = (position, orientation);
        if let Some(last) = self.last_stamp_ns {
            if stamp_ns <= last {
                return;
            }
        }
        if self.last_pose.as_ref() == Some(&pose) {
            return;
        }
        self.last_stamp_ns = Some(stamp_ns);
        self.last_pose = Some(pose);
        if let Some(imu_ns) = latest_imu_ns {
            if self.offsets_ns.len() >= 2000 {
                self.offsets_ns.pop_front();
            }
            self.offsets_ns.push_back(stamp_ns - imu_ns);
        }
        self.stamps_ns.push(stamp_ns);
        self.positions.push(position);
    }

    fn offset_ns(&self) -> Option<i64> {
        if self.offsets_ns.is_empty() {
            return None;
        }
        let mut sorted: Vec<i64> = self.offsets_ns.iter().copied().collect();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2])
    }

    fn snapshot(&self) -> Option<(Vec<f64>, Vec<[f64; 3]>)> {
        let offset = self.offset_ns()?;
        let n = self.stamps_ns.len().min(self.positions.len());
        if n < 2 {
            return None;
        }
        let t: Vec<f64> = self.stamps_ns[..n]
            .iter()
            .map(|&s| (s - offset) as f64 / NSEC_PER_SEC as f64)
            .collect();
        Some((t, self.positions[..n].to_vec()))
    }
}

// ── Main node state ─────────────────────────────────────────────────────────

struct VioNode {
    // VIO pipeline
    filter: VIOFilter,
    camera: Arc<dyn CameraModel>,
    frontend: Frontend,
    t_bc: Matrix4<f64>,
    initialized: bool,
    n_init: usize,
    imu_buffer: Vec<IMUVelocity>,

    // Sparse 3D
    sparse_3d: Option<Sparse3DFilter>,

    // Dense mapping
    depth_mapper: Option<PatchDepthMapper>,
    depth_seed_coords: PatchDepthSeedCoordinates,
    occupancy_map: Option<LocalOccupancyMap>,
    mapping_stride: usize,

    // Message queues
    imu_queue: VecDeque<(i64, [f64; 3], [f64; 3])>,
    image_queue: VecDeque<(i64, r2r::sensor_msgs::msg::Image)>,
    max_imu_queue: usize,
    max_image_queue: usize,
    camera_offset_ns: i64,

    // State tracking
    latest_imu_ns: Option<i64>,
    last_imu_received_ns: Option<i64>,
    last_imu_processed_ns: Option<i64>,
    last_image_received_ns: Option<i64>,
    imu_processed: u64,
    imu_received: u64,
    images_received: u64,
    images_processed: u64,
    dropped_imu: u64,
    dropped_images: u64,
    bad_images: u64,
    patch_depth_frame_count: u64,

    // Timing stats
    imu_times_us: VecDeque<f64>,
    gray_times_ms: VecDeque<f64>,
    frontend_times_ms: VecDeque<f64>,
    vision_times_ms: VecDeque<f64>,
    total_times_ms: VecDeque<f64>,
    track_counts: VecDeque<u64>,
    patch_depth_times_ms: VecDeque<f64>,
    occupancy_times_ms: VecDeque<f64>,
    seed_counts: VecDeque<usize>,

    // Reference trajectories
    mocap_track: ReferenceTrack,
    vio_track: ReferenceTrack,
    estimated_stamps: VecDeque<f64>,
    estimated_positions: VecDeque<[f64; 3]>,

    // Image parameters
    width: usize,
    height: usize,
    intrinsics: CameraIntrinsics,

    // Frame IDs
    odom_frame: String,
    body_frame: String,
    publish_tf: bool,

    started_at: Instant,
}

impl VioNode {
    fn new(params: &NodeParams) -> Self {
        let [fx, fy, cx, cy] = params.intrinsics;
        let [k1, k2, k3, k4] = params.distortion;
        let w = params.width;
        let h = params.height;

        // Camera model
        let mut cam_intrinsics = RudolfCameraIntrinsics::new(fx, fy, cx, cy, w, h);
        cam_intrinsics.distortion = vec![k1, k2, k3, k4];
        let camera: Arc<dyn CameraModel> = match params.camera_model.as_str() {
            "radtan" | "radial-tangential" | "plumb_bob" => {
                cam_intrinsics.model = RudolfDistortionModel::RadTan;
                Arc::new(CameraProjection::pinhole_radtan(
                    [fx, fy, cx, cy],
                    [k1, k2, k3, k4],
                    [0, 0],
                ))
            }
            _ => {
                cam_intrinsics.model = RudolfDistortionModel::Equidistant;
                Arc::new(CameraProjection::pinhole_equidistant(
                    [fx, fy, cx, cy],
                    [k1, k2, k3, k4],
                    [0, 0],
                ))
            }
        };

        // Frontend
        let vio_config = VIOConfig::from_yaml(&params.config_path)
            .unwrap_or_else(|e| panic!("Failed to load config {}: {e}", params.config_path));
        let rv = &vio_config.rudolf_v;
        let mut frontend_cfg = FrontendConfig::default();
        frontend_cfg.max_features = rv.max_features;
        if let Some(t) = rv.fast_threshold {
            frontend_cfg.fast_threshold = t;
        }
        frontend_cfg.pyramid_levels = rv.max_level;
        frontend_cfg.cell_size = rv.feature_dist as usize;
        frontend_cfg.klt_residual_enabled = rv.klt_residual;
        frontend_cfg.enable_internal_ransac = rv.enable_ransac;
        frontend_cfg.epipolar_gate_threshold = rv.epipolar_gate_threshold;
        frontend_cfg.epipolar_refine = rv.epipolar_refine;
        frontend_cfg.epipolar_min_baseline = rv.epipolar_min_baseline;
        frontend_cfg.epipolar_max_reject_frac = rv.epipolar_max_reject_frac;
        frontend_cfg.lbp_policy = match rv.lbp_policy.as_deref() {
            Some("hardreject" | "hard_reject" | "hard-reject" | "hard") => LbpPolicy::HardReject,
            _ => LbpPolicy::SoftPenalty,
        };
        frontend_cfg.histeq = match rv.histeq.as_deref() {
            Some("global") | None => {
                if rv.equalise_image_histogram {
                    HistEqMethod::Global
                } else {
                    HistEqMethod::None
                }
            }
            Some("clahe") => HistEqMethod::Clahe {
                tile_size: rv.clahe_tile_size,
                clip_limit: rv.clahe_clip_limit,
            },
            _ => HistEqMethod::None,
        };
        frontend_cfg.camera = Some(cam_intrinsics);
        let frontend = Frontend::new(frontend_cfg, w, h);

        // VIO filter
        let filter_settings = vio_config.to_filter_settings();
        let cam_offset = SE3::from_matrix(&params.t_bs);
        let sensor = VIOSensorState {
            input_bias: Vector6::zeros(),
            pose: SE3::identity(),
            velocity: Vector3::zeros(),
            camera_offset: cam_offset,
        };
        let xi0 = VIOState::new(sensor, vec![]);
        let filter = VIOFilter::new(filter_settings, xi0);

        // Sparse 3D filter
        let sparse_3d = vio_config.sparse_vog.as_ref().and_then(|sv| {
            if !sv.enabled {
                return None;
            }
            let chart = match sv.parametrization.as_str() {
                s if s.starts_with("bearing") => Sparse3DChart::BearingInvDepthAdditive,
                "invdepth" | "invdepth3d" | "inverse-depth" => Sparse3DChart::InvDepth,
                _ => Sparse3DChart::Polar,
            };
            let settings = sv.to_sparse_settings();
            let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
            let filter = Sparse3DFilter::new(k, chart, settings);
            let filter = if chart == Sparse3DChart::BearingInvDepthAdditive {
                filter.with_camera(camera.clone())
            } else {
                filter
            };
            log::info!(
                "Sparse3DFilter: {:?}, pool={}",
                chart,
                sv.max_pool_size.unwrap_or(300)
            );
            Some(filter)
        });

        // Patch depth mapper (Rust backend only — DIS depth dropped)
        let mut depth_mapper = None;
        let mut depth_seed_coords = PatchDepthSeedCoordinates::RawDistorted;
        let mut occupancy_map = None;
        if params.patch_depth_enabled {
            let intrinsics = CameraIntrinsics { fx, fy, cx, cy };
            let pd_settings = vio_config
                .patch_depth
                .as_ref()
                .map(|c| c.to_patch_depth_settings())
                .unwrap_or_default();
            match PatchDepthMapper::new(camera.clone(), intrinsics, w, h, pd_settings) {
                Ok(mapper) => {
                    depth_seed_coords = mapper.expected_seed_coordinates();
                    log::info!("PatchDepthMapper: enabled");
                    if params.occupancy_enabled {
                        let occ_settings = vio_config
                            .local_occupancy
                            .as_ref()
                            .map(|c| c.to_local_occupancy_settings())
                            .unwrap_or_default();
                        match LocalOccupancyMap::new(occ_settings) {
                            Ok(occ) => {
                                log::info!("LocalOccupancyMap: enabled");
                                occupancy_map = Some(occ);
                            }
                            Err(e) => log::error!("Failed to create LocalOccupancyMap: {e}"),
                        }
                    }
                    depth_mapper = Some(mapper);
                }
                Err(e) => log::error!("Failed to create PatchDepthMapper: {e}"),
            }
        }

        Self {
            filter,
            camera,
            frontend,
            t_bc: params.t_bs,
            initialized: false,
            n_init: params.n_init,
            imu_buffer: Vec::new(),
            sparse_3d,
            depth_mapper,
            depth_seed_coords,
            occupancy_map,
            mapping_stride: params.mapping_stride,
            imu_queue: VecDeque::new(),
            image_queue: VecDeque::new(),
            max_imu_queue: params.max_imu_queue,
            max_image_queue: params.max_image_queue,
            camera_offset_ns: params.camera_offset_ns,
            latest_imu_ns: None,
            last_imu_received_ns: None,
            last_imu_processed_ns: None,
            last_image_received_ns: None,
            imu_processed: 0,
            imu_received: 0,
            images_received: 0,
            images_processed: 0,
            dropped_imu: 0,
            dropped_images: 0,
            bad_images: 0,
            patch_depth_frame_count: 0,
            imu_times_us: VecDeque::with_capacity(300),
            gray_times_ms: VecDeque::with_capacity(300),
            frontend_times_ms: VecDeque::with_capacity(300),
            vision_times_ms: VecDeque::with_capacity(300),
            total_times_ms: VecDeque::with_capacity(300),
            track_counts: VecDeque::with_capacity(300),
            patch_depth_times_ms: VecDeque::with_capacity(300),
            occupancy_times_ms: VecDeque::with_capacity(300),
            seed_counts: VecDeque::with_capacity(300),
            mocap_track: ReferenceTrack::new(),
            vio_track: ReferenceTrack::new(),
            estimated_stamps: VecDeque::with_capacity(20000),
            estimated_positions: VecDeque::with_capacity(20000),
            width: w,
            height: h,
            intrinsics: CameraIntrinsics { fx, fy, cx, cy },
            odom_frame: params.odom_frame.clone(),
            body_frame: params.body_frame.clone(),
            publish_tf: params.publish_tf,
            started_at: Instant::now(),
        }
    }

    fn on_imu(&mut self, msg: &r2r::sensor_msgs::msg::Imu) {
        let ns = stamp_ns(&msg.header.stamp);
        self.imu_received += 1;
        if let Some(last) = self.last_imu_received_ns {
            if ns <= last {
                self.dropped_imu += 1;
                return;
            }
        }
        self.last_imu_received_ns = Some(ns);
        self.latest_imu_ns = Some(ns);
        let gyro = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ];
        let accel = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ];
        self.imu_queue.push_back((ns, gyro, accel));
        if self.imu_queue.len() > self.max_imu_queue {
            self.imu_queue.pop_front();
            self.dropped_imu += 1;
        }
    }

    fn on_image(&mut self, msg: r2r::sensor_msgs::msg::Image) {
        self.images_received += 1;
        let ns = stamp_ns(&msg.header.stamp) + self.camera_offset_ns;
        if let Some(last) = self.last_image_received_ns {
            if ns <= last {
                self.dropped_images += 1;
                return;
            }
        }
        self.last_image_received_ns = Some(ns);
        self.image_queue.push_back((ns, msg));
        if self.image_queue.len() > self.max_image_queue {
            self.image_queue.pop_front();
            self.dropped_images += 1;
        }
    }

    fn drain_queues(
        &mut self,
        odom_pub: &r2r::Publisher<r2r::nav_msgs::msg::Odometry>,
        tf_pub: Option<&r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>>,
    ) {
        while !self.image_queue.is_empty() && !self.imu_queue.is_empty() {
            let (image_ns, _) = &self.image_queue[0];
            let image_ns = *image_ns;
            if let Some(latest) = self.latest_imu_ns {
                if latest < image_ns {
                    break;
                }
            } else {
                break;
            }
            if let Some(last_proc) = self.last_imu_processed_ns {
                if image_ns <= last_proc {
                    self.image_queue.pop_front();
                    self.dropped_images += 1;
                    continue;
                }
            }

            let mut processed_for_frame = 0u64;
            while let Some(&(imu_ns, _, _)) = self.imu_queue.front() {
                if imu_ns > image_ns {
                    break;
                }
                let (imu_ns, gyro, accel) = self.imu_queue.pop_front().unwrap();
                if let Some(last_proc) = self.last_imu_processed_ns {
                    if imu_ns <= last_proc {
                        self.dropped_imu += 1;
                        continue;
                    }
                }
                self.process_imu_sample(imu_ns, gyro, accel);
                self.last_imu_processed_ns = Some(imu_ns);
                self.imu_processed += 1;
                processed_for_frame += 1;
            }

            if processed_for_frame == 0 {
                self.image_queue.pop_front();
                self.dropped_images += 1;
                continue;
            }
            let (ns, msg) = self.image_queue.pop_front().unwrap();
            self.process_image(ns, &msg, odom_pub, tf_pub);
        }
    }

    fn process_imu_sample(&mut self, stamp_ns: i64, gyro: [f64; 3], accel: [f64; 3]) {
        let stamp = stamp_ns as f64 / NSEC_PER_SEC as f64;
        let imu = IMUVelocity::new(
            stamp,
            Vector3::new(gyro[0], gyro[1], gyro[2]),
            Vector3::new(accel[0], accel[1], accel[2]),
        );

        if !self.initialized {
            self.imu_buffer.push(imu);
            if self.imu_buffer.len() >= self.n_init {
                let pose = estimate_initial_pose(&self.imu_buffer, self.n_init);
                let cam_offset = SE3::from_matrix(&self.t_bc);
                let sensor = VIOSensorState {
                    input_bias: Vector6::zeros(),
                    pose,
                    velocity: Vector3::zeros(),
                    camera_offset: cam_offset,
                };
                let xi0 = VIOState::new(sensor, vec![]);
                self.filter = VIOFilter::new(self.filter.settings.clone(), xi0);
                for buffered in self.imu_buffer.drain(..) {
                    self.filter.process_imu(buffered);
                }
                self.initialized = true;
            }
            return;
        }
        let imu_t = Instant::now();
        self.filter.process_imu(imu);
        push_capped(
            &mut self.imu_times_us,
            imu_t.elapsed().as_secs_f64() * 1e6,
            300,
        );
    }

    fn process_image(
        &mut self,
        stamp_ns: i64,
        msg: &r2r::sensor_msgs::msg::Image,
        odom_pub: &r2r::Publisher<r2r::nav_msgs::msg::Odometry>,
        tf_pub: Option<&r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>>,
    ) {
        let started = Instant::now();
        let gray = match self.to_gray(msg) {
            Ok(g) => g,
            Err(e) => {
                self.bad_images += 1;
                log::error!("{e}");
                return;
            }
        };
        let gray_ms = started.elapsed().as_secs_f64() * 1000.0;

        let frontend_started = Instant::now();
        let rudolf_img = RudolfImage::from_vec(self.width, self.height, gray.clone());
        let (features_ref, frame_stats) = self.frontend.process(&rudolf_img);
        // Copy features out so we release the borrow on self.frontend.
        let features: Vec<Feature> = features_ref.to_vec();
        let total_tracks = frame_stats.total;
        let frontend_ms = frontend_started.elapsed().as_secs_f64() * 1000.0;
        log::debug!("frontend: {}", frame_stats.timing);

        let observations: HashMap<u64, Vector2<f32>> = features
            .iter()
            .map(|f| (f.id, Vector2::new(f.x, f.y)))
            .collect();
        let stamp = stamp_ns as f64 / NSEC_PER_SEC as f64;
        let measurement = VisionMeasurement::new(stamp, observations);

        let vision_started = Instant::now();
        self.filter
            .process_vision(measurement, self.camera.as_ref());
        let vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;

        self.images_processed += 1;
        push_capped(&mut self.gray_times_ms, gray_ms, 300);
        push_capped(&mut self.frontend_times_ms, frontend_ms, 300);
        push_capped(&mut self.vision_times_ms, vision_ms, 300);
        push_capped(
            &mut self.total_times_ms,
            started.elapsed().as_secs_f64() * 1000.0,
            300,
        );
        push_capped(&mut self.track_counts, total_tracks as u64, 300);

        if self.imu_processed < self.n_init as u64 {
            return;
        }

        let state = self.filter.state_estimate();
        let pos = state.sensor.pose.translation;
        let q = state.sensor.pose.rotation.as_xyzw();
        let vel = state.sensor.velocity;
        let position = [pos[0], pos[1], pos[2]];
        let quaternion = [q[0], q[1], q[2], q[3]];
        let velocity = [vel[0], vel[1], vel[2]];

        if !position.iter().all(|v| v.is_finite())
            || !quaternion.iter().all(|v| v.is_finite())
            || !velocity.iter().all(|v| v.is_finite())
        {
            log::error!("ECHO-LI produced a non-finite state");
            return;
        }

        self.publish_odometry(
            stamp_ns,
            &position,
            &quaternion,
            &velocity,
            odom_pub,
            tf_pub,
        );

        // Camera pose: T_wc = T_wb @ T_bc
        let t_wc = quat_to_se3(&position, &quaternion) * self.t_bc;

        // Update sparse 3D filter
        if let Some(ref mut sparse) = self.sparse_3d {
            let feature_uvs: HashMap<u64, Vector2<f32>> = features
                .iter()
                .map(|f| (f.id, Vector2::new(f.x, f.y)))
                .collect();
            let (p_vv, p_ww) = self
                .filter
                .sparse_camera_pose_covariances()
                .map(|(v, w)| (Some(v), Some(w)))
                .unwrap_or((None, None));
            let measurement = VisionMeasurement::new(stamp, feature_uvs);
            sparse.update(&measurement, &t_wc, p_vv.as_ref(), p_ww.as_ref());
        }

        // Dense mapping: patch depth → occupancy
        if self.depth_mapper.is_some() && self.images_processed % self.mapping_stride as u64 == 0 {
            self.run_mapping(stamp_ns, &gray, &t_wc, &features);
        }

        // Track estimated trajectory
        if self.estimated_positions.len() >= 20000 {
            self.estimated_positions.pop_front();
            self.estimated_stamps.pop_front();
        }
        self.estimated_positions.push_back(position);
        self.estimated_stamps.push_back(stamp);
    }

    fn run_mapping(
        &mut self,
        stamp_ns: i64,
        gray: &[u8],
        t_wc: &Matrix4<f64>,
        features: &[Feature],
    ) {
        let cam_pos = [t_wc[(0, 3)], t_wc[(1, 3)], t_wc[(2, 3)]];
        let stamp = stamp_ns as f64 / NSEC_PER_SEC as f64;

        // Build sparse depth priors from Sparse3DFilter + EqF landmarks
        let mut priors = Vec::new();
        let mut sparse3d_fids = std::collections::HashSet::new();

        if let Some(ref sparse) = self.sparse_3d {
            for feat in features {
                let (rng, rng_var) = sparse.query_range(feat.id);
                if rng < 0.0 {
                    continue;
                }
                sparse3d_fids.insert(feat.id);
                let eta = rng.ln();
                let eta_var = if rng > 0.01 {
                    rng_var / (rng * rng)
                } else {
                    1.0
                };
                priors.push(SparseDepthPrior {
                    uv: Vector2::new(feat.x as f64, feat.y as f64),
                    eta,
                    eta_var,
                });
            }
        }

        // Fill from EqF landmarks
        let state = self.filter.state_estimate();
        let (global_lm, _, _) = landmarks_to_global(&state);
        for feat in features {
            if sparse3d_fids.contains(&feat.id) {
                continue;
            }
            if let Some(lm_pos) = global_lm.get(&feat.id) {
                let dx = lm_pos[0] - cam_pos[0];
                let dy = lm_pos[1] - cam_pos[1];
                let dz = lm_pos[2] - cam_pos[2];
                let rng = (dx * dx + dy * dy + dz * dz).sqrt();
                if rng < 0.1 {
                    continue;
                }
                priors.push(SparseDepthPrior {
                    uv: Vector2::new(feat.x as f64, feat.y as f64),
                    eta: rng.ln(),
                    eta_var: 0.25,
                });
            }
        }

        push_capped_usize(&mut self.seed_counts, priors.len(), 300);
        self.patch_depth_frame_count += 1;

        let pd_start = Instant::now();
        let frame = FrameProducts {
            frame_id: self.patch_depth_frame_count,
            stamp,
            gray: gray.to_vec(),
            width: self.width,
            height: self.height,
            pose_t_wc: *t_wc,
        };
        let depth_result = self
            .depth_mapper
            .as_mut()
            .unwrap()
            .update_with_priors(frame, &priors, None, 0.05);
        push_capped(
            &mut self.patch_depth_times_ms,
            pd_start.elapsed().as_secs_f64() * 1000.0,
            300,
        );

        if let (Some(output), Some(occ)) = (&depth_result, &mut self.occupancy_map) {
            let occ_start = Instant::now();
            occ.update_from_patch_depth(
                output,
                self.camera.as_ref(),
                self.intrinsics,
                self.depth_seed_coords,
                self.width,
                self.height,
                t_wc,
            );
            push_capped(
                &mut self.occupancy_times_ms,
                occ_start.elapsed().as_secs_f64() * 1000.0,
                300,
            );
        }
    }

    fn to_gray(&self, msg: &r2r::sensor_msgs::msg::Image) -> Result<Vec<u8>, String> {
        let w = msg.width as usize;
        let h = msg.height as usize;
        if w != self.width || h != self.height {
            return Err(format!(
                "unexpected image size {w}x{h}; expected {}x{}",
                self.width, self.height
            ));
        }
        let encoding = msg.encoding.to_lowercase();
        match encoding.as_str() {
            "mono8" | "8uc1" => {
                let step = msg.step as usize;
                let mut gray = Vec::with_capacity(w * h);
                for row in 0..h {
                    gray.extend_from_slice(&msg.data[row * step..row * step + w]);
                }
                Ok(gray)
            }
            "rgb8" | "bgr8" => {
                let step = msg.step as usize;
                let is_bgr = encoding == "bgr8";
                let mut gray = Vec::with_capacity(w * h);
                for row in 0..h {
                    let row_start = row * step;
                    for col in 0..w {
                        let i = row_start + col * 3;
                        let (r, g, b) = if is_bgr {
                            (msg.data[i + 2], msg.data[i + 1], msg.data[i])
                        } else {
                            (msg.data[i], msg.data[i + 1], msg.data[i + 2])
                        };
                        gray.push((0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8);
                    }
                }
                Ok(gray)
            }
            "rgba8" | "bgra8" => {
                let step = msg.step as usize;
                let is_bgr = encoding == "bgra8";
                let mut gray = Vec::with_capacity(w * h);
                for row in 0..h {
                    let row_start = row * step;
                    for col in 0..w {
                        let i = row_start + col * 4;
                        let (r, g, b) = if is_bgr {
                            (msg.data[i + 2], msg.data[i + 1], msg.data[i])
                        } else {
                            (msg.data[i], msg.data[i + 1], msg.data[i + 2])
                        };
                        gray.push((0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8);
                    }
                }
                Ok(gray)
            }
            _ => Err(format!("unsupported image encoding: {}", msg.encoding)),
        }
    }

    fn publish_odometry(
        &self,
        stamp_ns: i64,
        position: &[f64; 3],
        quaternion: &[f64; 4],
        velocity: &[f64; 3],
        odom_pub: &r2r::Publisher<r2r::nav_msgs::msg::Odometry>,
        tf_pub: Option<&r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>>,
    ) {
        let stamp = to_stamp(stamp_ns);
        let mut odom = r2r::nav_msgs::msg::Odometry::default();
        odom.header.stamp = stamp.clone();
        odom.header.frame_id = self.odom_frame.clone();
        odom.child_frame_id = self.body_frame.clone();
        odom.pose.pose.position.x = position[0];
        odom.pose.pose.position.y = position[1];
        odom.pose.pose.position.z = position[2];
        odom.pose.pose.orientation.x = quaternion[0];
        odom.pose.pose.orientation.y = quaternion[1];
        odom.pose.pose.orientation.z = quaternion[2];
        odom.pose.pose.orientation.w = quaternion[3];
        odom.twist.twist.linear.x = velocity[0];
        odom.twist.twist.linear.y = velocity[1];
        odom.twist.twist.linear.z = velocity[2];
        let _ = odom_pub.publish(&odom);

        if self.publish_tf {
            if let Some(tf_pub) = tf_pub {
                let mut tf = r2r::geometry_msgs::msg::TransformStamped::default();
                tf.header.stamp = stamp;
                tf.header.frame_id = self.odom_frame.clone();
                tf.child_frame_id = self.body_frame.clone();
                tf.transform.translation.x = position[0];
                tf.transform.translation.y = position[1];
                tf.transform.translation.z = position[2];
                tf.transform.rotation = odom.pose.pose.orientation.clone();
                let tf_msg = r2r::tf2_msgs::msg::TFMessage {
                    transforms: vec![tf],
                };
                let _ = tf_pub.publish(&tf_msg);
            }
        }
    }

    fn on_mocap(&mut self, msg: &r2r::geometry_msgs::msg::PoseStamped) {
        let p = &msg.pose.position;
        let o = &msg.pose.orientation;
        self.mocap_track.add(
            stamp_ns(&msg.header.stamp),
            [p.x, p.y, p.z],
            [o.x, o.y, o.z, o.w],
            self.latest_imu_ns,
        );
    }

    fn on_vio(&mut self, msg: &r2r::nav_msgs::msg::Odometry) {
        let p = &msg.pose.pose.position;
        let o = &msg.pose.pose.orientation;
        self.vio_track.add(
            stamp_ns(&msg.header.stamp),
            [p.x, p.y, p.z],
            [o.x, o.y, o.z, o.w],
            self.latest_imu_ns,
        );
    }

    fn report_statistics(&self) {
        let elapsed = self.started_at.elapsed().as_secs_f64().max(1e-9);
        let imu_us = median_deque(&self.imu_times_us).unwrap_or(0.0);
        let gray_ms = median_deque(&self.gray_times_ms).unwrap_or(0.0);
        let frontend_ms = median_deque(&self.frontend_times_ms).unwrap_or(0.0);
        let vision_ms = median_deque(&self.vision_times_ms).unwrap_or(0.0);
        let total_ms = median_deque(&self.total_times_ms).unwrap_or(0.0);
        let tracks = median_deque_u64(&self.track_counts).unwrap_or(0);

        let mut mapping_part = String::new();
        if let Some(pd_ms) = median_deque(&self.patch_depth_times_ms) {
            let seeds_med = median_deque_usize(&self.seed_counts).unwrap_or(0);
            mapping_part.push_str(&format!(
                "; seeds_med={seeds_med} patch_depth_med={pd_ms:.1}ms"
            ));
        }
        if let Some(occ_ms) = median_deque(&self.occupancy_times_ms) {
            if let Some(ref occ) = self.occupancy_map {
                let (u, f, o) = occ.counts();
                mapping_part.push_str(&format!(
                    " occupancy_med={occ_ms:.1}ms (unk={u} free={f} occ={o})"
                ));
            } else {
                mapping_part.push_str(&format!(" occupancy_med={occ_ms:.1}ms"));
            }
        }
        for (name, track) in [("mocap", &self.mocap_track), ("vio", &self.vio_track)] {
            if let Some(offset) = track.offset_ns() {
                mapping_part.push_str(&format!(
                    "; {name}_clock_offset={:+.3}s n={}",
                    offset as f64 / NSEC_PER_SEC as f64,
                    track.stamps_ns.len()
                ));
            }
        }
        log::info!(
            "input: imu={:.1}Hz image={:.1}Hz; \
             processed: imu={} image={}; \
             queues: imu={} image={}; \
             dropped: imu={} image={}; \
             imu_med={imu_us:.0}us gray_med={gray_ms:.1}ms frontend_med={frontend_ms:.1}ms \
             vision_med={vision_ms:.1}ms total_med={total_ms:.1}ms \
             tracks_med={tracks}{mapping_part}",
            self.imu_received as f64 / elapsed,
            self.images_received as f64 / elapsed,
            self.imu_processed,
            self.images_processed,
            self.imu_queue.len(),
            self.image_queue.len(),
            self.dropped_imu,
            self.dropped_images,
        );
    }
}

// ── Utility ─────────────────────────────────────────────────────────────────

fn push_capped<T>(deque: &mut VecDeque<T>, val: T, max: usize) {
    if deque.len() >= max {
        deque.pop_front();
    }
    deque.push_back(val);
}

fn push_capped_usize(deque: &mut VecDeque<usize>, val: usize, max: usize) {
    if deque.len() >= max {
        deque.pop_front();
    }
    deque.push_back(val);
}

fn median_deque(d: &VecDeque<f64>) -> Option<f64> {
    if d.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = d.iter().copied().collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

fn median_deque_u64(d: &VecDeque<u64>) -> Option<u64> {
    if d.is_empty() {
        return None;
    }
    let mut v: Vec<u64> = d.iter().copied().collect();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

fn median_deque_usize(d: &VecDeque<usize>) -> Option<usize> {
    if d.is_empty() {
        return None;
    }
    let mut v: Vec<usize> = d.iter().copied().collect();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "echo_li_voxl2", "")?;

    let params = load_params(&node);

    let imu_qos = best_effort_qos(params.imu_qos_depth);
    let image_qos = r2r::QosProfile {
        depth: params.image_qos_depth,
        reliability: if params.image_qos_reliable {
            r2r::qos::ReliabilityPolicy::Reliable
        } else {
            r2r::qos::ReliabilityPolicy::BestEffort
        },
        ..r2r::QosProfile::sensor_data()
    };
    let ref_qos = best_effort_qos(200);

    // Publishers
    let odom_pub = node.create_publisher::<r2r::nav_msgs::msg::Odometry>(
        &params.odometry_topic,
        r2r::QosProfile::default(),
    )?;
    let tf_pub =
        if params.publish_tf {
            Some(node.create_publisher::<r2r::tf2_msgs::msg::TFMessage>(
                "/tf",
                r2r::QosProfile::default(),
            )?)
        } else {
            None
        };

    // Subscribers
    let mut imu_sub = node.subscribe::<r2r::sensor_msgs::msg::Imu>(&params.imu_topic, imu_qos)?;
    let mut image_sub =
        node.subscribe::<r2r::sensor_msgs::msg::Image>(&params.image_topic, image_qos)?;

    let mut mocap_sub = if !params.mocap_topic.is_empty() {
        log::info!("Mocap: subscribing to {}", params.mocap_topic);
        Some(node.subscribe::<r2r::geometry_msgs::msg::PoseStamped>(
            &params.mocap_topic,
            ref_qos.clone(),
        )?)
    } else {
        None
    };
    let mut vio_sub = if !params.vio_topic.is_empty() {
        log::info!("VIO ref: subscribing to {}", params.vio_topic);
        Some(node.subscribe::<r2r::nav_msgs::msg::Odometry>(&params.vio_topic, ref_qos)?)
    } else {
        None
    };

    let stats_dur = std::time::Duration::from_secs_f64(params.stats_period_sec);
    let mut stats_timer = node.create_wall_timer(stats_dur)?;

    log::info!(
        "ECHO-LI ready: imu={}, image={}, camera={}x{} {}, \
         camera_time_offset={:+.6}s, image_qos={}",
        params.imu_topic,
        params.image_topic,
        params.width,
        params.height,
        params.camera_model,
        params.camera_offset_ns as f64 / NSEC_PER_SEC as f64,
        if params.image_qos_reliable {
            "reliable"
        } else {
            "best_effort"
        },
    );

    let mut state = VioNode::new(&params);

    // Spin the underlying rcl node in a background thread
    let _spin = tokio::task::spawn_blocking(move || {
        loop {
            node.spin_once(std::time::Duration::from_millis(1));
        }
    });

    loop {
        tokio::select! {
            Some(msg) = imu_sub.next() => {
                state.on_imu(&msg);
                state.drain_queues(&odom_pub, tf_pub.as_ref());
            }
            Some(msg) = image_sub.next() => {
                state.on_image(msg);
                state.drain_queues(&odom_pub, tf_pub.as_ref());
            }
            msg = async {
                match mocap_sub.as_mut() {
                    Some(s) => s.next().await,
                    None => futures::future::pending().await,
                }
            } => {
                if let Some(msg) = msg {
                    state.on_mocap(&msg);
                }
            }
            msg = async {
                match vio_sub.as_mut() {
                    Some(s) => s.next().await,
                    None => futures::future::pending().await,
                }
            } => {
                if let Some(msg) = msg {
                    state.on_vio(&msg);
                }
            }
            _ = stats_timer.tick() => {
                state.report_statistics();
            }
        }
    }
}
