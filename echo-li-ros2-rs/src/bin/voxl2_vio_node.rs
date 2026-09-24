use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use camera_geometry::CameraProjection;
use echo_li_core::config::VIOConfig;
use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::depth::occupancy::LocalOccupancyMap;
use echo_li_core::depth::patch_depth::{
    FrameProducts, PatchDepthMapper, PatchDepthOutput, PatchDepthSeedCoordinates, PatchStatus,
    SparseDepthPrior,
};
use echo_li_core::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use echo_li_core::initialization::estimate_initial_pose;
use echo_li_core::mathematical::camera::CameraModel;
use echo_li_core::mathematical::imu_velocity::IMUVelocity;
use echo_li_core::mathematical::vio_state::{VIOSensorState, VIOState};
use echo_li_core::mathematical::vision_measurement::VisionMeasurement;
use echo_li_core::{VIOFilter, landmarks_to_global};
use echo_li_ros2::{NSEC_PER_SEC, best_effort_qos, quat_to_se3, stamp_ns, to_stamp, xyzi_cloud};
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
use rudolf_v::klt::LkMethod;

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
    image_scale: f64,
    klt_method: String,
    trajectory_output: String,
    mocap_topic: String,
    vio_topic: String,
    path_topic: String,
    mocap_path_topic: String,
    path_period_sec: f64,
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

// Anything describing the physical rig is required rather than defaulted: a
// built-in value would silently run the filter on some other rig's geometry.
fn missing(name: &str) -> ! {
    panic!("parameter '{name}' is missing; pass the rig's calibration with --params-file")
}

fn require_f64_array<const N: usize>(node: &r2r::Node, name: &str) -> [f64; N] {
    let v = get_f64_vec(node, name).unwrap_or_else(|| missing(name));
    let len = v.len();
    v.try_into()
        .unwrap_or_else(|_| panic!("parameter '{name}' needs {N} values, got {len}"))
}

fn require_i64(node: &r2r::Node, name: &str) -> i64 {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::Integer(v)) => *v,
        Some(r2r::ParameterValue::Double(v)) => *v as i64,
        _ => missing(name),
    }
}

fn require_str(node: &r2r::Node, name: &str) -> String {
    let params = node.params.lock().unwrap();
    match params.get(name).map(|p| &p.value) {
        Some(r2r::ParameterValue::String(s)) => s.clone(),
        _ => missing(name),
    }
}

fn load_params(node: &r2r::Node) -> NodeParams {
    let config_path = get_str(node, "echo_config_path", "config/eqvio_voxl2.yaml");
    // No offset unless the rig's calibration measures one. A guessed value is
    // an angular error during every turn, which the filter pays for as
    // translation.
    let offset_sec = get_f64(node, "camera_time_offset_sec", 0.0);

    let intrinsics = require_f64_array::<4>(node, "intrinsics");
    let distortion = require_f64_array::<4>(node, "distortion_coefficients");
    let t_bs = Matrix4::from_row_slice(&require_f64_array::<16>(node, "t_bs"));

    NodeParams {
        config_path,
        imu_topic: get_str(node, "imu_topic", "/voxl/raw_imu"),
        image_topic: get_str(node, "image_topic", "/tracking_front/decoded"),
        odometry_topic: get_str(node, "odometry_topic", "/echo_li/odometry"),
        odom_frame: get_str(node, "odom_frame_id", "echo_li_odom"),
        body_frame: get_str(node, "body_frame_id", "imu_link"),
        publish_tf: get_bool(node, "publish_tf", true),
        camera_offset_ns: (offset_sec * NSEC_PER_SEC as f64).round() as i64,
        width: require_i64(node, "image_width") as usize,
        height: require_i64(node, "image_height") as usize,
        intrinsics,
        distortion,
        camera_model: require_str(node, "camera_model"),
        t_bs,
        n_init: get_i64(node, "initialization_imu_samples", 100) as usize,
        imu_qos_depth: get_i64(node, "imu_qos_depth", 2000) as usize,
        image_qos_depth: get_i64(node, "image_qos_depth", 30) as usize,
        // Reliable by default: see the note in bag_time_relay.rs. Whole images
        // are lost to single dropped UDP fragments otherwise.
        image_qos_reliable: get_str(node, "image_qos_reliability", "reliable") == "reliable",
        max_imu_queue: get_i64(node, "max_imu_queue", 5000) as usize,
        max_image_queue: get_i64(node, "max_image_queue", 4) as usize,
        stats_period_sec: get_f64(node, "statistics_period_sec", 5.0),
        patch_depth_enabled: get_bool(node, "patch_depth_enabled", true),
        occupancy_enabled: get_bool(node, "occupancy_enabled", true),
        image_scale: get_f64(node, "image_scale", 1.0).clamp(0.25, 1.0),
        // forward_additive is the FrontendConfig default and is the only KLT
        // variant with no SIMD on any architecture -- it is scalar bilinear
        // loops. inverse_compositional reaches extract_template_gradients and
        // ic_iterate_patch, both of which have NEON kernels on aarch64.
        klt_method: get_str(node, "klt_method", "forward_additive"),
        trajectory_output: get_str(node, "trajectory_output", ""),
        mapping_stride: (get_i64(node, "mapping_stride", 1) as usize).max(1),
        mocap_topic: get_str(node, "mocap_topic", ""),
        vio_topic: get_str(node, "vio_topic", ""),
        path_topic: get_str(node, "path_topic", "/echo_li/path"),
        mocap_path_topic: get_str(node, "mocap_path_topic", "/echo_li/mocap_path"),
        path_period_sec: get_f64(node, "path_period_sec", 0.5).max(0.05),
    }
}

// ── Trajectory plotting ─────────────────────────────────────────────────────

/// Longest gap that still counts as the same instant when pairing a reference
/// pose with an estimated one.
const PATH_MATCH_DT: f64 = 0.02;
/// Below this many pairs, or this much horizontal travel (m), the heading fit
/// is noise.
const PATH_MIN_PAIRS: usize = 50;
const PATH_MIN_SPREAD: f64 = 0.3;
/// Poses per published path. Mocap runs at a few hundred hertz, and rviz2 does
/// not need every sample to draw the line.
const PATH_MAX_POINTS: usize = 4000;

/// Yaw and translation putting a reference track into the VIO's odom frame.
/// Both are gravity-aligned, so heading and origin are all that differ.
#[derive(Clone, Copy)]
struct PlanarAlignment {
    yaw: f64,
    t: [f64; 3],
    rmse: f64,
    pairs: usize,
}

impl PlanarAlignment {
    fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let (s, c) = self.yaw.sin_cos();
        [
            c * p[0] - s * p[1] + self.t[0],
            s * p[0] + c * p[1] + self.t[1],
            p[2] + self.t[2],
        ]
    }
}

fn decimate<'a>(points: impl Iterator<Item = &'a [f64; 3]>, max: usize) -> Vec<[f64; 3]> {
    let all: Vec<[f64; 3]> = points.copied().collect();
    if all.len() <= max {
        return all;
    }
    let stride = (all.len() + max - 1) / max;
    let mut out: Vec<[f64; 3]> = all.iter().step_by(stride).copied().collect();
    match (out.last(), all.last()) {
        (Some(last), Some(end)) if last != end => out.push(*end),
        _ => {}
    }
    out
}

// ── Reference trajectory tracker ────────────────────────────────────────────

/// How much reference history to retain. Bounded by time, not sample count:
/// mocap and external VIO arrive at different rates, so a shared count cap would
/// cover very different spans. Only the part overlapping the estimate is ever
/// used — `estimated_positions` itself holds about ten minutes at image rate.
const REF_TRACK_WINDOW_NS: i64 = 600 * NSEC_PER_SEC;

struct ReferenceTrack {
    stamps_ns: VecDeque<i64>,
    positions: VecDeque<[f64; 3]>,
    offsets_ns: VecDeque<i64>,
    last_stamp_ns: Option<i64>,
    last_pose: Option<([f64; 3], [f64; 4])>,
}

impl ReferenceTrack {
    fn new() -> Self {
        Self {
            stamps_ns: VecDeque::new(),
            positions: VecDeque::new(),
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
        while let Some(&oldest) = self.stamps_ns.front() {
            if stamp_ns - oldest <= REF_TRACK_WINDOW_NS {
                break;
            }
            self.stamps_ns.pop_front();
            self.positions.pop_front();
        }
        self.stamps_ns.push_back(stamp_ns);
        self.positions.push_back(position);
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
        let t: Vec<f64> = self
            .stamps_ns
            .iter()
            .take(n)
            .map(|&s| (s - offset) as f64 / NSEC_PER_SEC as f64)
            .collect();
        Some((t, self.positions.iter().take(n).copied().collect()))
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
    depth_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::Image>>,
    depth_status_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::Image>>,
    sparse_landmark_pub: r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>,
    occupancy_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>>,
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
    patch_depth_frame_count: u64,

    // Timing stats
    imu_times_us: VecDeque<f64>,
    gray_times_ms: VecDeque<f64>,
    frontend_times_ms: VecDeque<f64>,
    // Frontend sub-stages, straight from rudolf_v FrameStats.timing. The
    // frontend was a single 5.7 ms number with no way to tell pyramid from
    // KLT from detect; these split it.
    fe_histeq_ms: VecDeque<f64>,
    fe_pyramid_ms: VecDeque<f64>,
    fe_klt_ms: VecDeque<f64>,
    fe_ransac_ms: VecDeque<f64>,
    fe_detect_ms: VecDeque<f64>,
    vision_times_ms: VecDeque<f64>,
    total_times_ms: VecDeque<f64>,
    track_counts: VecDeque<u64>,
    patch_depth_times_ms: VecDeque<f64>,
    depth_valid_pct: VecDeque<f64>,
    depth_seedonly_pct: VecDeque<f64>,
    occupancy_times_ms: VecDeque<f64>,
    seed_counts: VecDeque<usize>,
    sparse_seed_counts: VecDeque<usize>,
    sparse_census: [usize; 6],
    sparse_range_var: [f64; 7],
    landmark_counts: VecDeque<usize>,

    // Reference trajectories
    mocap_track: ReferenceTrack,
    vio_track: ReferenceTrack,
    estimated_stamps: VecDeque<f64>,
    estimated_positions: VecDeque<[f64; 3]>,
    estimated_quaternions: VecDeque<[f64; 4]>,
    mocap_fit: Option<PlanarAlignment>,
    /// Frame the mocap poses arrive in. The fit is broadcast as odom -> this,
    /// so anything else published in that frame lands where the mocap path is.
    mocap_frame_id: String,

    // Image parameters
    input_width: usize,
    input_height: usize,
    width: usize,
    height: usize,
    image_scale: f64,
    intrinsics: CameraIntrinsics,

    // Frame IDs
    odom_frame: String,
    body_frame: String,
    publish_tf: bool,

    started_at: Instant,
}

impl VioNode {
    fn new(
        params: &NodeParams,
        depth_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::Image>>,
        depth_status_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::Image>>,
        sparse_landmark_pub: r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>,
        occupancy_pub: Option<r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>>,
    ) -> Self {
        let s = params.image_scale;
        let [fx, fy, cx, cy] = [
            params.intrinsics[0] * s,
            params.intrinsics[1] * s,
            params.intrinsics[2] * s,
            params.intrinsics[3] * s,
        ];
        let [k1, k2, k3, k4] = params.distortion;
        let w = (params.width as f64 * s).round() as usize;
        let h = (params.height as f64 * s).round() as usize;

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
        frontend_cfg.klt_method = match params.klt_method.as_str() {
            "inverse_compositional" | "ic" => LkMethod::InverseCompositional,
            "inverse_compositional_fixed" | "ic_fixed" => LkMethod::InverseCompositionalFixed,
            "forward_additive" | "fa" => LkMethod::ForwardAdditive,
            other => panic!(
                "unknown klt_method {other:?}; expected forward_additive, \
                 inverse_compositional or inverse_compositional_fixed"
            ),
        };
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
            depth_pub,
            depth_status_pub,
            sparse_landmark_pub,
            occupancy_pub,
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
            patch_depth_frame_count: 0,
            imu_times_us: VecDeque::with_capacity(300),
            gray_times_ms: VecDeque::with_capacity(300),
            frontend_times_ms: VecDeque::with_capacity(300),
            fe_histeq_ms: VecDeque::with_capacity(300),
            fe_pyramid_ms: VecDeque::with_capacity(300),
            fe_klt_ms: VecDeque::with_capacity(300),
            fe_ransac_ms: VecDeque::with_capacity(300),
            fe_detect_ms: VecDeque::with_capacity(300),
            vision_times_ms: VecDeque::with_capacity(300),
            total_times_ms: VecDeque::with_capacity(300),
            track_counts: VecDeque::with_capacity(300),
            patch_depth_times_ms: VecDeque::with_capacity(300),
            depth_valid_pct: VecDeque::with_capacity(300),
            depth_seedonly_pct: VecDeque::with_capacity(300),
            occupancy_times_ms: VecDeque::with_capacity(300),
            seed_counts: VecDeque::with_capacity(300),
            sparse_seed_counts: VecDeque::with_capacity(300),
            sparse_census: [0; 6],
            sparse_range_var: [-1.0; 7],
            landmark_counts: VecDeque::with_capacity(300),
            mocap_track: ReferenceTrack::new(),
            vio_track: ReferenceTrack::new(),
            estimated_stamps: VecDeque::with_capacity(20000),
            estimated_positions: VecDeque::with_capacity(20000),
            estimated_quaternions: VecDeque::with_capacity(20000),
            mocap_fit: None,
            mocap_frame_id: String::new(),
            input_width: params.width,
            input_height: params.height,
            width: w,
            height: h,
            image_scale: s,
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
        landmark_pub: &r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>,
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
            self.process_image(ns, &msg, odom_pub, tf_pub, landmark_pub);
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
        landmark_pub: &r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>,
    ) {
        let started = Instant::now();
        let gray = match self.to_gray(msg) {
            Ok(g) => g,
            Err(e) => {
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
        {
            let t = &frame_stats.timing;
            push_capped(&mut self.fe_histeq_ms, t.histeq * 1000.0, 300);
            push_capped(&mut self.fe_pyramid_ms, t.pyramid * 1000.0, 300);
            push_capped(&mut self.fe_klt_ms, t.klt * 1000.0, 300);
            push_capped(&mut self.fe_ransac_ms, t.ransac * 1000.0, 300);
            push_capped(&mut self.fe_detect_ms, t.detect * 1000.0, 300);
        }
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

        let n_lm = state.camera_landmarks.len();
        push_capped(&mut self.landmark_counts, n_lm, 300);
        let (global_landmarks, _, _) = landmarks_to_global(&state);

        // Camera pose: T_wc = T_wb @ T_bc
        let t_wc = quat_to_se3(&position, &quaternion) * self.t_bc;
        let cam_origin = [t_wc[(0, 3)], t_wc[(1, 3)], t_wc[(2, 3)]];
        Self::publish_landmarks(
            stamp_ns,
            &self.odom_frame,
            &global_landmarks,
            &cam_origin,
            landmark_pub,
        );

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
            self.estimated_quaternions.pop_front();
        }
        self.estimated_positions.push_back(position);
        self.estimated_quaternions.push_back(quaternion);
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

        let mut sparse_world: HashMap<u64, Vector3<f64>> = HashMap::new();
        let mut census = [0usize; 6];
        let mut range_var = [-1.0f64; 7];

        if let Some(ref sparse) = self.sparse_3d {
            census = sparse.gate_census();
            range_var = sparse.range_var_percentiles();
            for feat in features {
                let (rng, rng_var) = sparse.query_range(feat.id);
                if rng < 0.0 {
                    continue;
                }
                sparse3d_fids.insert(feat.id);
                if let Some(state) = sparse.feature(feat.id) {
                    // position is cached in the current camera frame.
                    let p = state.position;
                    sparse_world.insert(
                        feat.id,
                        Vector3::new(
                            t_wc[(0, 0)] * p[0]
                                + t_wc[(0, 1)] * p[1]
                                + t_wc[(0, 2)] * p[2]
                                + t_wc[(0, 3)],
                            t_wc[(1, 0)] * p[0]
                                + t_wc[(1, 1)] * p[1]
                                + t_wc[(1, 2)] * p[2]
                                + t_wc[(1, 3)],
                            t_wc[(2, 0)] * p[0]
                                + t_wc[(2, 1)] * p[1]
                                + t_wc[(2, 2)] * p[2]
                                + t_wc[(2, 3)],
                        ),
                    );
                }
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

        push_capped(&mut self.seed_counts, priors.len(), 300);
        push_capped(&mut self.sparse_seed_counts, sparse_world.len(), 300);
        self.sparse_census = census;
        self.sparse_range_var = range_var;
        Self::publish_landmarks(
            stamp_ns,
            &self.odom_frame,
            &sparse_world,
            &cam_pos,
            &self.sparse_landmark_pub,
        );
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

        if let Some(output) = &depth_result {
            self.publish_depth_maps(stamp_ns, output);
        }

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
        if w != self.input_width || h != self.input_height {
            return Err(format!(
                "unexpected image size {w}x{h}; expected {}x{}",
                self.input_width, self.input_height
            ));
        }
        let encoding = msg.encoding.to_lowercase();
        let full = match encoding.as_str() {
            "mono8" | "8uc1" => {
                let step = msg.step as usize;
                let mut gray = Vec::with_capacity(w * h);
                for row in 0..h {
                    gray.extend_from_slice(&msg.data[row * step..row * step + w]);
                }
                gray
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
                gray
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
                gray
            }
            _ => return Err(format!("unsupported image encoding: {}", msg.encoding)),
        };
        if self.image_scale >= 1.0 {
            return Ok(full);
        }
        // Area-average, not point-sample. Rudolf-V's pyramid applies a proper
        // blur+decimate, but only for the levels above level 0 — level 0 is a
        // verbatim copy of whatever it is handed, and KLT tracks against it. So
        // a point-sampled input aliases the gradients the tracker keys on and
        // every pyramid level above inherits that.
        let ow = self.width;
        let oh = self.height;
        let mut out = Vec::with_capacity(ow * oh);
        for oy in 0..oh {
            let y0 = oy * h / oh;
            let y1 = (((oy + 1) * h / oh).max(y0 + 1)).min(h);
            for ox in 0..ow {
                let x0 = ox * w / ow;
                let x1 = (((ox + 1) * w / ow).max(x0 + 1)).min(w);
                let mut sum = 0u32;
                for sy in y0..y1 {
                    let row = sy * w;
                    for sx in x0..x1 {
                        sum += full[row + sx] as u32;
                    }
                }
                out.push((sum / (((y1 - y0) * (x1 - x0)) as u32)) as u8);
            }
        }
        Ok(out)
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

    /// Publish the dense map so its coverage is inspectable from ROS: a 32FC1
    /// range image, NaN where the mapper produced nothing, alongside a mono8
    /// status image. The status image is what separates "never had texture to
    /// work with" from "solved and then rejected" when the map comes out sparse.
    fn publish_depth_maps(&mut self, stamp_ns: i64, output: &PatchDepthOutput) {
        let (w, h) = (output.eta.width, output.eta.height);
        let pixels = (w * h) as f64;

        let mut range = Vec::with_capacity(w * h * 4);
        let mut valid = 0usize;
        for &eta in &output.eta.data {
            let metres = if eta.is_finite() {
                valid += 1;
                eta.exp()
            } else {
                f32::NAN
            };
            range.extend_from_slice(&metres.to_le_bytes());
        }

        let mut status = Vec::with_capacity(w * h);
        let mut seed_only = 0usize;
        for s in &output.status.data {
            status.push(match s {
                PatchStatus::Unknown => 0u8,
                PatchStatus::SeedOnly => {
                    seed_only += 1;
                    85
                }
                PatchStatus::Rejected => 170,
                PatchStatus::PhotoRefined => 255,
            });
        }

        push_capped(
            &mut self.depth_valid_pct,
            100.0 * valid as f64 / pixels,
            300,
        );
        push_capped(
            &mut self.depth_seedonly_pct,
            100.0 * seed_only as f64 / pixels,
            300,
        );

        let stamp = to_stamp(stamp_ns);
        let header = r2r::std_msgs::msg::Header {
            stamp,
            frame_id: self.body_frame.clone(),
        };
        if let Some(p) = &self.depth_pub {
            let _ = p.publish(&r2r::sensor_msgs::msg::Image {
                header: header.clone(),
                height: h as u32,
                width: w as u32,
                encoding: "32FC1".into(),
                is_bigendian: 0,
                step: (w * 4) as u32,
                data: range,
            });
        }
        if let Some(p) = &self.depth_status_pub {
            let _ = p.publish(&r2r::sensor_msgs::msg::Image {
                header,
                height: h as u32,
                width: w as u32,
                encoding: "mono8".into(),
                is_bigendian: 0,
                step: w as u32,
                data: status,
            });
        }
    }

    fn publish_landmarks(
        stamp_ns: i64,
        frame_id: &str,
        landmarks: &HashMap<u64, Vector3<f64>>,
        cam_origin: &[f64; 3],
        pub_: &r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>,
    ) {
        if landmarks.is_empty() {
            return;
        }
        let stamp = to_stamp(stamp_ns);
        let point_step: u32 = 16; // x, y, z, depth — 4 × f32
        let n = landmarks.len() as u32;
        let mut data = Vec::with_capacity(point_step as usize * n as usize);
        for pos in landmarks.values() {
            data.extend_from_slice(&(pos[0] as f32).to_le_bytes());
            data.extend_from_slice(&(pos[1] as f32).to_le_bytes());
            data.extend_from_slice(&(pos[2] as f32).to_le_bytes());
            let dx = pos[0] - cam_origin[0];
            let dy = pos[1] - cam_origin[1];
            let dz = pos[2] - cam_origin[2];
            let depth = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
            data.extend_from_slice(&depth.to_le_bytes());
        }
        let _ = pub_.publish(&xyzi_cloud(stamp, frame_id, data, n));
    }

    /// Publish the occupied voxels of the local occupancy grid as a point cloud,
    /// intensity carrying log-odds. Rides the path timer rather than the image
    /// path: the grid is ~500k voxels and occupancy changes far more slowly than
    /// frames arrive. Render it in rviz2 with Style=Boxes and Size = resolution.
    fn publish_occupancy(&self, pub_: &r2r::Publisher<r2r::sensor_msgs::msg::PointCloud2>) {
        let (Some(occ), Some(&last_stamp)) = (&self.occupancy_map, self.estimated_stamps.back())
        else {
            return;
        };
        let snap = occ.snapshot();
        let occupied = occ.settings().occupied_threshold;
        let res = snap.resolution;
        let mut data: Vec<u8> = Vec::new();
        let mut n: u32 = 0;
        for z in 0..snap.depth {
            for y in 0..snap.height {
                for x in 0..snap.width {
                    let log_odds = snap.log_odds[(z * snap.height + y) * snap.width + x];
                    if log_odds < occupied {
                        continue;
                    }
                    let wx = snap.origin_x + (x as f64 + 0.5) * res;
                    let wy = snap.origin_y + (y as f64 + 0.5) * res;
                    let wz = snap.origin_z + (z as f64 + 0.5) * res;
                    data.extend_from_slice(&(wx as f32).to_le_bytes());
                    data.extend_from_slice(&(wy as f32).to_le_bytes());
                    data.extend_from_slice(&(wz as f32).to_le_bytes());
                    data.extend_from_slice(&log_odds.to_le_bytes());
                    n += 1;
                }
            }
        }
        if n == 0 {
            return;
        }
        let stamp = to_stamp((last_stamp * NSEC_PER_SEC as f64).round() as i64);
        let _ = pub_.publish(&xyzi_cloud(stamp, &self.odom_frame, data, n));
    }

    /// Fit a reference track onto the estimated trajectory: pair the two by
    /// time (the track's own clock offset is already known), then solve for the
    /// heading and origin that put the reference into the odom frame.
    fn fit_planar(&self, track: &ReferenceTrack) -> Option<PlanarAlignment> {
        let (ref_t, ref_p) = track.snapshot()?;
        let mut j = 0usize;
        let mut pairs: Vec<([f64; 3], [f64; 3])> = Vec::new();
        for i in 0..self.estimated_stamps.len() {
            let t = self.estimated_stamps[i];
            while j + 1 < ref_t.len() && (ref_t[j + 1] - t).abs() <= (ref_t[j] - t).abs() {
                j += 1;
            }
            if (ref_t[j] - t).abs() <= PATH_MATCH_DT {
                pairs.push((ref_p[j], self.estimated_positions[i]));
            }
        }
        let n = pairs.len();
        if n < PATH_MIN_PAIRS {
            return None;
        }
        let (mut mr, mut me) = ([0.0f64; 3], [0.0f64; 3]);
        for (r, e) in &pairs {
            for k in 0..3 {
                mr[k] += r[k];
                me[k] += e[k];
            }
        }
        for k in 0..3 {
            mr[k] /= n as f64;
            me[k] /= n as f64;
        }
        let (mut sxx, mut sxy, mut spread) = (0.0f64, 0.0f64, 0.0f64);
        for (r, e) in &pairs {
            let (rx, ry) = (r[0] - mr[0], r[1] - mr[1]);
            let (ex, ey) = (e[0] - me[0], e[1] - me[1]);
            sxx += rx * ex + ry * ey;
            sxy += rx * ey - ry * ex;
            spread = spread.max((rx * rx + ry * ry).sqrt());
        }
        if spread < PATH_MIN_SPREAD {
            return None;
        }
        let yaw = sxy.atan2(sxx);
        let (sin_y, cos_y) = yaw.sin_cos();
        let fit = PlanarAlignment {
            yaw,
            t: [
                me[0] - (cos_y * mr[0] - sin_y * mr[1]),
                me[1] - (sin_y * mr[0] + cos_y * mr[1]),
                me[2] - mr[2],
            ],
            rmse: 0.0,
            pairs: n,
        };
        let mut sq = 0.0;
        for (r, e) in &pairs {
            let a = fit.apply(r);
            sq += (a[0] - e[0]).powi(2) + (a[1] - e[1]).powi(2) + (a[2] - e[2]).powi(2);
        }
        Some(PlanarAlignment {
            rmse: (sq / n as f64).sqrt(),
            ..fit
        })
    }

    fn make_path(
        &self,
        stamp: &r2r::builtin_interfaces::msg::Time,
        points: &[[f64; 3]],
    ) -> r2r::nav_msgs::msg::Path {
        let mut path = r2r::nav_msgs::msg::Path::default();
        path.header.stamp = stamp.clone();
        path.header.frame_id = self.odom_frame.clone();
        path.poses = points
            .iter()
            .map(|p| {
                let mut ps = r2r::geometry_msgs::msg::PoseStamped::default();
                ps.header.stamp = stamp.clone();
                ps.header.frame_id = self.odom_frame.clone();
                ps.pose.position.x = p[0];
                ps.pose.position.y = p[1];
                ps.pose.position.z = p[2];
                ps.pose.orientation.w = 1.0;
                ps
            })
            .collect();
        path
    }

    /// Draw the estimated trajectory, and the mocap one beside it once enough
    /// of both overlap to fit the heading.
    fn publish_paths(
        &mut self,
        path_pub: &r2r::Publisher<r2r::nav_msgs::msg::Path>,
        mocap_path_pub: Option<&r2r::Publisher<r2r::nav_msgs::msg::Path>>,
        tf_pub: Option<&r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>>,
    ) {
        let Some(&last_stamp) = self.estimated_stamps.back() else {
            return;
        };
        let stamp = to_stamp((last_stamp * NSEC_PER_SEC as f64).round() as i64);
        let estimate = decimate(self.estimated_positions.iter(), PATH_MAX_POINTS);
        let _ = path_pub.publish(&self.make_path(&stamp, &estimate));

        let Some(mocap_path_pub) = mocap_path_pub else {
            return;
        };
        let Some(fit) = self.fit_planar(&self.mocap_track) else {
            return;
        };
        if self.mocap_fit.is_none() {
            log::info!(
                "Mocap fitted to the estimate: yaw={:+.1}deg over {} pairs; \
                 broadcasting TF {} -> {}",
                fit.yaw.to_degrees(),
                fit.pairs,
                self.odom_frame,
                self.mocap_frame_id
            );
        }
        self.mocap_fit = Some(fit);
        // The same fit as TF, so obstacle markers or other bodies published in
        // the mocap frame render where the mocap path is drawn. Refreshed with
        // the path, since the fit keeps improving as the tracks grow.
        if self.publish_tf
            && !self.mocap_frame_id.is_empty()
            && self.mocap_frame_id != self.odom_frame
        {
            if let Some(tf_pub) = tf_pub {
                let (half_sin, half_cos) = (fit.yaw / 2.0).sin_cos();
                let mut tf = r2r::geometry_msgs::msg::TransformStamped::default();
                tf.header.stamp = stamp.clone();
                tf.header.frame_id = self.odom_frame.clone();
                tf.child_frame_id = self.mocap_frame_id.clone();
                tf.transform.translation.x = fit.t[0];
                tf.transform.translation.y = fit.t[1];
                tf.transform.translation.z = fit.t[2];
                tf.transform.rotation.z = half_sin;
                tf.transform.rotation.w = half_cos;
                let _ = tf_pub.publish(&r2r::tf2_msgs::msg::TFMessage {
                    transforms: vec![tf],
                });
            }
        }
        let points: Vec<[f64; 3]> = decimate(self.mocap_track.positions.iter(), PATH_MAX_POINTS)
            .iter()
            .map(|p| fit.apply(p))
            .collect();
        let _ = mocap_path_pub.publish(&self.make_path(&stamp, &points));
    }

    fn save_trajectory(&self, path: &str) {
        use std::io::Write;
        let Ok(mut f) = std::fs::File::create(path) else {
            log::error!("Cannot write trajectory to {path}");
            return;
        };
        let _ = writeln!(f, "# TUM format: timestamp x y z qx qy qz qw");
        for i in 0..self.estimated_stamps.len() {
            let t = self.estimated_stamps[i];
            let p = self.estimated_positions[i];
            let q = self.estimated_quaternions[i];
            let _ = writeln!(
                f,
                "{t:.9} {:.6} {:.6} {:.6} {:.9} {:.9} {:.9} {:.9}",
                p[0], p[1], p[2], q[0], q[1], q[2], q[3]
            );
        }
        log::info!("Saved {} poses to {path}", self.estimated_stamps.len());
    }

    fn on_mocap(&mut self, msg: &r2r::geometry_msgs::msg::PoseStamped) {
        if self.mocap_frame_id.is_empty() && !msg.header.frame_id.is_empty() {
            self.mocap_frame_id = msg.header.frame_id.clone();
        }
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
        let fe_histeq = median_deque(&self.fe_histeq_ms).unwrap_or(0.0);
        let fe_pyr = median_deque(&self.fe_pyramid_ms).unwrap_or(0.0);
        let fe_klt = median_deque(&self.fe_klt_ms).unwrap_or(0.0);
        let fe_ransac = median_deque(&self.fe_ransac_ms).unwrap_or(0.0);
        let fe_detect = median_deque(&self.fe_detect_ms).unwrap_or(0.0);
        let total_ms = median_deque(&self.total_times_ms).unwrap_or(0.0);
        let tracks = median_ord(&self.track_counts).unwrap_or(0);

        let mut mapping_part = String::new();
        if let Some(pd_ms) = median_deque(&self.patch_depth_times_ms) {
            let seeds_med = median_ord(&self.seed_counts).unwrap_or(0);
            let sparse_med = median_ord(&self.sparse_seed_counts).unwrap_or(0);
            let c = self.sparse_census;
            mapping_part.push_str(&format!(
                "; sparse[pending={} pool={} short_track={} low_inlier={} high_var={} usable={}]",
                c[0], c[1], c[2], c[3], c[4], c[5]
            ));
            let rv = self.sparse_range_var;
            mapping_part.push_str(&format!(
                " range(p10/50/90)={:.2}/{:.2}/{:.2}m var(p10/50/90)={:.2}/{:.2}/{:.2} short<0.5m={:.0}%",
                rv[0], rv[1], rv[2], rv[3], rv[4], rv[5], 100.0 * rv[6]
            ));
            let valid = median_deque(&self.depth_valid_pct).unwrap_or(0.0);
            let seed_only = median_deque(&self.depth_seedonly_pct).unwrap_or(0.0);
            mapping_part.push_str(&format!(
                "; seeds_med={seeds_med} sparse_seeds_med={sparse_med} patch_depth_med={pd_ms:.1}ms \
                 depth_valid={valid:.1}% depth_seedonly={seed_only:.1}%"
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
        if let Some(fit) = self.mocap_fit {
            mapping_part.push_str(&format!(
                "; mocap_fit yaw={:+.1}deg rmse={:.3}m pairs={}",
                fit.yaw.to_degrees(),
                fit.rmse,
                fit.pairs
            ));
        }
        log::info!(
            "input: imu={:.1}Hz image={:.1}Hz; \
             processed: imu={} image={}; \
             queues: imu={} image={}; \
             dropped: imu={} image={}; \
             imu_med={imu_us:.0}us gray_med={gray_ms:.1}ms frontend_med={frontend_ms:.1}ms \
             fe[histeq={fe_histeq:.2} pyr={fe_pyr:.2} klt={fe_klt:.2} \
             ransac={fe_ransac:.2} detect={fe_detect:.2}]ms \
             vision_med={vision_ms:.1}ms total_med={total_ms:.1}ms \
             tracks_med={tracks} lm_med={}{mapping_part}",
            self.imu_received as f64 / elapsed,
            self.images_received as f64 / elapsed,
            self.imu_processed,
            self.images_processed,
            self.imu_queue.len(),
            self.image_queue.len(),
            self.dropped_imu,
            self.dropped_images,
            median_ord(&self.landmark_counts).unwrap_or(0),
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

fn median_deque(d: &VecDeque<f64>) -> Option<f64> {
    if d.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = d.iter().copied().collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// Median for the integer counters. `median_deque` stays separate because f64
/// has no total order.
fn median_ord<T: Ord + Copy>(d: &VecDeque<T>) -> Option<T> {
    if d.is_empty() {
        return None;
    }
    let mut v: Vec<T> = d.iter().copied().collect();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

// ── Main ────────────────────────────────────────────────────────────────────

/// Drain a subscription into an unbounded channel. r2r gives each subscription
/// a 10-message channel and its spin thread drops whatever does not fit
/// (try_send, logged at debug only). The main loop blocks for tens of
/// milliseconds per image, so at 1 kHz roughly 40 % of the IMU samples were
/// being lost before the node ever saw them, which diverged the filter; mocap
/// at a few hundred hertz has the same problem. A pump task keeps every
/// subscription drained while the loop is busy.
fn spawn_pump<S, T>(sub: Option<S>) -> tokio::sync::mpsc::UnboundedReceiver<T>
where
    S: futures::Stream<Item = T> + Unpin + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    if let Some(mut sub) = sub {
        tokio::spawn(async move {
            while let Some(msg) = sub.next().await {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

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
    let landmark_pub = node.create_publisher::<r2r::sensor_msgs::msg::PointCloud2>(
        "/echo_li/landmarks",
        r2r::QosProfile::default(),
    )?;
    // Separate from /echo_li/landmarks (EqF SLAM points) so the two seed sources
    // can be told apart: this one carries the Sparse3D features that actually
    // passed the convergence gate and therefore seed the patch mapper.
    let sparse_landmark_pub = node.create_publisher::<r2r::sensor_msgs::msg::PointCloud2>(
        "/echo_li/sparse3d_landmarks",
        r2r::QosProfile::default(),
    )?;
    let occupancy_pub = if params.occupancy_enabled {
        Some(node.create_publisher::<r2r::sensor_msgs::msg::PointCloud2>(
            "/echo_li/occupancy",
            r2r::QosProfile::default(),
        )?)
    } else {
        None
    };
    // Dense depth is only worth a topic when the mapper is actually running.
    let (depth_pub, depth_status_pub) = if params.patch_depth_enabled {
        (
            Some(node.create_publisher::<r2r::sensor_msgs::msg::Image>(
                "/echo_li/depth",
                r2r::QosProfile::default(),
            )?),
            Some(node.create_publisher::<r2r::sensor_msgs::msg::Image>(
                "/echo_li/depth_status",
                r2r::QosProfile::default(),
            )?),
        )
    } else {
        (None, None)
    };
    let path_pub = node.create_publisher::<r2r::nav_msgs::msg::Path>(
        &params.path_topic,
        r2r::QosProfile::default(),
    )?;
    let mocap_path_pub = if params.mocap_topic.is_empty() {
        None
    } else {
        Some(node.create_publisher::<r2r::nav_msgs::msg::Path>(
            &params.mocap_path_topic,
            r2r::QosProfile::default(),
        )?)
    };

    // Subscribers
    let imu_sub = node.subscribe::<r2r::sensor_msgs::msg::Imu>(&params.imu_topic, imu_qos)?;
    let mut image_sub =
        node.subscribe::<r2r::sensor_msgs::msg::Image>(&params.image_topic, image_qos)?;

    let mocap_sub = if !params.mocap_topic.is_empty() {
        log::info!("Mocap: subscribing to {}", params.mocap_topic);
        Some(node.subscribe::<r2r::geometry_msgs::msg::PoseStamped>(
            &params.mocap_topic,
            ref_qos.clone(),
        )?)
    } else {
        None
    };
    let vio_sub = if !params.vio_topic.is_empty() {
        log::info!("VIO ref: subscribing to {}", params.vio_topic);
        Some(node.subscribe::<r2r::nav_msgs::msg::Odometry>(&params.vio_topic, ref_qos)?)
    } else {
        None
    };

    let stats_dur = std::time::Duration::from_secs_f64(params.stats_period_sec);
    let mut stats_timer = node.create_wall_timer(stats_dur)?;
    let mut path_timer =
        node.create_wall_timer(std::time::Duration::from_secs_f64(params.path_period_sec))?;

    log::info!(
        "ECHO-LI ready: imu={}, image={}, camera={}x{} (scale={}) {}, \
         camera_time_offset={:+.6}s, image_qos={}, klt={}",
        params.imu_topic,
        params.image_topic,
        (params.width as f64 * params.image_scale).round() as usize,
        (params.height as f64 * params.image_scale).round() as usize,
        params.image_scale,
        params.camera_model,
        params.camera_offset_ns as f64 / NSEC_PER_SEC as f64,
        if params.image_qos_reliable {
            "reliable"
        } else {
            "best_effort"
        },
        params.klt_method,
    );

    let mut state = VioNode::new(
        &params,
        depth_pub,
        depth_status_pub,
        sparse_landmark_pub,
        occupancy_pub,
    );
    let traj_output = params.trajectory_output.clone();

    // Spin the underlying rcl node in a background thread.
    //
    // This must be able to stop. `spawn_blocking` tasks cannot be cancelled, so
    // an endless loop here leaves the runtime's Drop waiting for it forever: the
    // select below exits cleanly on SIGINT, `main` returns, and then shutdown
    // hangs with the spin thread still burning a core. From outside that looks
    // exactly like a node ignoring Ctrl-C, and only SIGKILL ends it.
    let spinning = Arc::new(AtomicBool::new(true));
    let spin_flag = Arc::clone(&spinning);
    let _spin = tokio::task::spawn_blocking(move || {
        while spin_flag.load(Ordering::Relaxed) {
            node.spin_once(std::time::Duration::from_millis(1));
        }
    });

    let mut imu_rx = spawn_pump(Some(imu_sub));
    let mut mocap_rx = spawn_pump(mocap_sub);
    let mut vio_rx = spawn_pump(vio_sub);

    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    loop {
        tokio::select! {
            Some(msg) = imu_rx.recv() => {
                state.on_imu(&msg);
                state.drain_queues(&odom_pub, tf_pub.as_ref(), &landmark_pub);
            }
            Some(msg) = image_sub.next() => {
                state.on_image(msg);
                state.drain_queues(&odom_pub, tf_pub.as_ref(), &landmark_pub);
            }
            Some(msg) = mocap_rx.recv() => {
                state.on_mocap(&msg);
            }
            Some(msg) = vio_rx.recv() => {
                state.on_vio(&msg);
            }
            _ = path_timer.tick() => {
                state.publish_paths(&path_pub, mocap_path_pub.as_ref(), tf_pub.as_ref());
                if let Some(occ_pub) = &state.occupancy_pub {
                    state.publish_occupancy(occ_pub);
                }
            }
            _ = stats_timer.tick() => {
                state.report_statistics();
            }
            _ = sigint.recv() => { break; }
            _ = sigterm.recv() => { break; }
        }
    }

    // Release the spin thread before the runtime is dropped, or its Drop blocks.
    spinning.store(false, Ordering::Relaxed);

    state.report_statistics();
    if !traj_output.is_empty() {
        state.save_trajectory(&traj_output);
    }

    Ok(())
}
