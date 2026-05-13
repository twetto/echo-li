use clap::Parser;
use echo_li_core::config::VIOConfig;
use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::dataserver::ASLDatasetReader;
use echo_li_core::depth::patch_depth::{
    FrameProducts, PatchDepthCameraMode, PatchDepthMapper, PatchDepthOutput,
    PatchDepthSeedCoordinates, PatchDepthSettings, PatchStatus,
};
use echo_li_core::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use echo_li_core::depth::sparse_gb::SparseVogSettings;
use echo_li_core::initialization::{check_stationary, estimate_initial_pose};
use echo_li_core::mathematical::camera::{CameraModel, PinholeModel, RadTanModel};
use echo_li_core::mathematical::*;
use echo_li_core::{VIOFilter, VIOFilterSettings};
use nalgebra::{Matrix3, Matrix4, Vector2};
use rudolf_v::frontend::{Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::klt::LkMethod;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    dataset: String,

    #[arg(short, long)]
    config: Option<String>,

    #[arg(long, default_value = "Normal")]
    coord: String,

    #[arg(short = 'l', long, default_value_t = 0.0)]
    cam_lag: f64,

    /// Output directory for trajectory files (default: eqvio_output_<dataset_name>)
    #[arg(short, long)]
    output: Option<String>,

    /// Enable Rerun visualization (requires --features rerun)
    #[arg(long, default_value_t = false)]
    vis: bool,

    /// Run the out-of-state sparse depth filter alongside EqVIO.
    #[arg(long, default_value_t = true)]
    sparse: bool,

    /// Sparse filter chart: polar3d or invdepth3d.
    #[arg(long, default_value = "polar3d")]
    sparse_chart: String,

    /// Run patch-grid direct depth mapper. Enabled automatically by PatchDepth config.
    #[arg(long, default_value_t = false)]
    patch_depth: bool,
}

fn write_trajectory(path: &std::path::Path, entries: &[(f64, VIOState)]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for (t, state) in entries {
        let pos = state.sensor.pose.translation;
        let q = state.sensor.pose.rotation.as_xyzw();
        writeln!(
            f,
            "{:.9} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            t, pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3]
        )?;
    }
    Ok(())
}

fn write_groundtruth(path: &std::path::Path, poses: &[StampedPose]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for sp in poses {
        let pos = sp.pose.translation;
        let q = sp.pose.rotation.as_xyzw();
        writeln!(
            f,
            "{:.9} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            sp.stamp, pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3]
        )?;
    }
    Ok(())
}

fn parse_sparse_chart(name: &str) -> Sparse3DChart {
    match name.to_ascii_lowercase().as_str() {
        "invdepth" | "invdepth3d" | "inverse-depth" => Sparse3DChart::InvDepth,
        _ => Sparse3DChart::Polar,
    }
}

fn camera_pose_matrix(state: &VIOState) -> Matrix4<f64> {
    state
        .sensor
        .pose
        .compose(&state.sensor.camera_offset)
        .as_matrix()
}

fn undistorted_pinhole_measurement(
    stamp: f64,
    raw_uvs: &HashMap<u64, Vector2<f32>>,
    cam_model: &dyn CameraModel,
    k: &Matrix3<f64>,
) -> VisionMeasurement {
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let mut undistorted_uvs = HashMap::with_capacity(raw_uvs.len());

    for (&id, uv) in raw_uvs {
        let raw = Vector2::new(uv[0] as f64, uv[1] as f64);
        let bearing = cam_model.undistort(&raw);
        if bearing[2].abs() <= 1e-12 {
            continue;
        }
        let x = bearing[0] / bearing[2];
        let y = bearing[1] / bearing[2];
        undistorted_uvs.insert(id, Vector2::new((fx * x + cx) as f32, (fy * y + cy) as f32));
    }

    VisionMeasurement::new(stamp, undistorted_uvs)
}

fn patch_depth_status_counts(output: &PatchDepthOutput) -> (usize, usize, usize, usize) {
    let mut unknown = 0;
    let mut seed_only = 0;
    let mut photo_refined = 0;
    let mut rejected = 0;
    for status in &output.status_cells.data {
        match status {
            PatchStatus::Unknown => unknown += 1,
            PatchStatus::SeedOnly => seed_only += 1,
            PatchStatus::PhotoRefined => photo_refined += 1,
            PatchStatus::Rejected => rejected += 1,
        }
    }
    (unknown, seed_only, photo_refined, rejected)
}

#[cfg(feature = "rerun")]
fn patch_depth_rgb_for_vis(
    output: &PatchDepthOutput,
    img_w: usize,
    img_h: usize,
    vis_min_depth: f64,
    vis_max_depth: f64,
) -> Vec<u8> {
    let mut rgb = vec![0u8; img_w * img_h * 3];
    if vis_min_depth <= 0.0 || vis_max_depth <= vis_min_depth {
        return rgb;
    }
    let cell_w = (img_w / output.depth_cells.width.max(1)).max(1);
    let cell_h = (img_h / output.depth_cells.height.max(1)).max(1);

    for cy in 0..output.depth_cells.height {
        for cx in 0..output.depth_cells.width {
            let idx = cy * output.depth_cells.width + cx;
            let depth = output.depth_cells.data[idx];
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }
            let color = color_for_depth(depth as f64, vis_min_depth, vis_max_depth);
            let r = ((color >> 24) & 0xFF) as u8;
            let g = ((color >> 16) & 0xFF) as u8;
            let b = ((color >> 8) & 0xFF) as u8;
            let x0 = cx * cell_w;
            let y0 = cy * cell_h;
            let x1 = if cx + 1 == output.depth_cells.width {
                img_w
            } else {
                ((cx + 1) * cell_w).min(img_w)
            };
            let y1 = if cy + 1 == output.depth_cells.height {
                img_h
            } else {
                ((cy + 1) * cell_h).min(img_h)
            };
            for y in y0..y1 {
                for x in x0..x1 {
                    let out = (y * img_w + x) * 3;
                    rgb[out] = r;
                    rgb[out + 1] = g;
                    rgb[out + 2] = b;
                }
            }
        }
    }

    rgb
}

#[cfg(feature = "rerun")]
fn clip_image_point(x: f64, y: f64, img_w: usize, img_h: usize) -> Option<(f32, f32)> {
    (x >= 0.0 && x < img_w as f64 && y >= 0.0 && y < img_h as f64).then_some((x as f32, y as f32))
}

#[cfg(feature = "rerun")]
fn color_for_inverse_depth(invdepth: f64, min_invdepth: f64, max_invdepth: f64) -> u32 {
    color_for_scalar(invdepth, min_invdepth, max_invdepth)
}

#[cfg(feature = "rerun")]
fn color_for_depth(depth: f64, min_depth: f64, max_depth: f64) -> u32 {
    let flipped_depth = max_depth + min_depth - depth;
    color_for_scalar(flipped_depth, min_depth, max_depth)
}

#[cfg(feature = "rerun")]
fn color_for_scalar(value: f64, min_value: f64, max_value: f64) -> u32 {
    // Approximate OpenCV COLORMAP_JET in RGB order. OpenCV's LUT has saturated
    // blue/red shoulders, which makes nearby inverse-depth values easier to
    // separate than a simple dark-blue -> cyan -> green -> yellow -> dark-red ramp.
    const JET: [(f64, u8, u8, u8); 6] = [
        (0.0, 0, 0, 128),
        (0.125, 0, 0, 255),
        (0.375, 0, 255, 255),
        (0.625, 255, 255, 0),
        (0.875, 255, 0, 0),
        (1.0, 128, 0, 0),
    ];

    let t = if max_value > min_value {
        ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let idx = JET
        .windows(2)
        .position(|window| t <= window[1].0)
        .unwrap_or(JET.len() - 2);
    let (t0, r0, g0, b0) = JET[idx];
    let (t1, r1, g1, b1) = JET[idx + 1];
    let local_t = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    let lerp =
        |a: u8, b: u8| -> u32 { (a as f64 + (b as f64 - a as f64) * local_t).round() as u32 };

    (lerp(r0, r1) << 24) | (lerp(g0, g1) << 16) | (lerp(b0, b1) << 8) | 0xFF
}

#[cfg(feature = "rerun")]
fn sparse_points_for_vis(
    sparse_filter: &Sparse3DFilter,
    cam_model: &dyn CameraModel,
    p_cam: &nalgebra::Vector3<f64>,
    r_cam: &echo_lie::SO3,
    img_w: usize,
    img_h: usize,
) -> (Vec<(f32, f32)>, Vec<u32>, Vec<(f32, f32, f32)>, Vec<u32>) {
    struct SparseVisPoint {
        image_point: Option<(f32, f32)>,
        world_point: (f32, f32, f32),
        invdepth: f64,
    }

    let mut points = Vec::new();

    for feat in sparse_filter.features().values() {
        let q = feat.position;
        if q[2] <= 1e-6 {
            continue;
        }

        let uv = cam_model.project(&q);
        let image_point = clip_image_point(uv[0], uv[1], img_w, img_h);
        let p_world = p_cam + r_cam.act(&q);
        points.push(SparseVisPoint {
            image_point,
            world_point: (p_world[0] as f32, p_world[1] as f32, p_world[2] as f32),
            invdepth: 1.0 / q[2],
        });
    }

    let mut image_points = Vec::new();
    let mut image_colors = Vec::new();
    let mut world_points = Vec::new();
    let mut world_colors = Vec::new();

    let min_invdepth = points
        .iter()
        .map(|point| point.invdepth)
        .fold(f64::INFINITY, f64::min);
    let max_invdepth = points
        .iter()
        .map(|point| point.invdepth)
        .fold(f64::NEG_INFINITY, f64::max);

    for point in points {
        let color = color_for_inverse_depth(point.invdepth, min_invdepth, max_invdepth);
        if let Some(image_point) = point.image_point {
            image_points.push(image_point);
            image_colors.push(color);
        }
        world_points.push(point.world_point);
        world_colors.push(color);
    }

    (image_points, image_colors, world_points, world_colors)
}

#[cfg(feature = "rerun")]
fn send_rerun_blueprint(
    rec: &rerun::RecordingStream,
    img_w: usize,
    img_h: usize,
) -> rerun::RecordingStreamResult<()> {
    use rerun::external::re_log_types::{BlueprintActivationCommand, LogMsg, RecordingId};
    use rerun::external::re_sdk_types::blueprint::archetypes::{
        ContainerBlueprint, ViewBlueprint, ViewContents, ViewportBlueprint, VisualBounds2D,
    };
    use rerun::external::re_sdk_types::blueprint::components::{
        AutoLayout, AutoViews, ContainerKind, IncludedContent, RootContainer, ViewOrigin,
    };
    use rerun::external::re_sdk_types::components::{Name, Visible};
    use rerun::external::re_sdk_types::datatypes::{Bool, EntityPath, Range2D, Uuid};

    let app_id = rec
        .store_info()
        .map(|info| info.application_id().to_string())
        .unwrap_or_else(|| "echo-li".to_owned());

    let (bp, storage) = rerun::RecordingStreamBuilder::new(app_id)
        .recording_id(RecordingId::random())
        .blueprint()
        .memory()?;
    bp.set_time_sequence("blueprint", 0);

    let camera_view_id = Uuid::random();
    let patch_depth_view_id = Uuid::random();
    let world_view_id = Uuid::random();
    let left_container_id = Uuid::random();
    let root_container_id = Uuid::random();
    let camera_view_path = format!("view/{camera_view_id}");
    let patch_depth_view_path = format!("view/{patch_depth_view_id}");
    let world_view_path = format!("view/{world_view_id}");
    let left_container_path = format!("container/{left_container_id}");
    let root_container_path = format!("container/{root_container_id}");

    bp.log(
        camera_view_path.as_str(),
        &ViewBlueprint::new("2D")
            .with_display_name(Name("Camera".into()))
            .with_space_origin(ViewOrigin("camera".into()))
            .with_visible(Visible(Bool(true))),
    )?;
    bp.log(
        format!("{camera_view_path}/ViewContents"),
        &ViewContents::new(["camera/**"]),
    )?;
    bp.log(
        format!("{camera_view_path}/VisualBounds2D"),
        &VisualBounds2D::new(Range2D {
            x_range: [0.0, img_w as f64].into(),
            y_range: [0.0, img_h as f64].into(),
        }),
    )?;

    bp.log(
        patch_depth_view_path.as_str(),
        &ViewBlueprint::new("2D")
            .with_display_name(Name("Patch Depth".into()))
            .with_space_origin(ViewOrigin("patch_depth".into()))
            .with_visible(Visible(Bool(true))),
    )?;
    bp.log(
        format!("{patch_depth_view_path}/ViewContents"),
        &ViewContents::new(["patch_depth/**"]),
    )?;
    bp.log(
        format!("{patch_depth_view_path}/VisualBounds2D"),
        &VisualBounds2D::new(Range2D {
            x_range: [0.0, img_w as f64].into(),
            y_range: [0.0, img_h as f64].into(),
        }),
    )?;

    bp.log(
        world_view_path.as_str(),
        &ViewBlueprint::new("3D")
            .with_display_name(Name("World".into()))
            .with_space_origin(ViewOrigin("world".into()))
            .with_visible(Visible(Bool(true))),
    )?;
    bp.log(
        format!("{world_view_path}/ViewContents"),
        &ViewContents::new(["world/**"]),
    )?;

    bp.log(
        left_container_path.as_str(),
        &ContainerBlueprint::new(ContainerKind::Vertical).with_contents([
            IncludedContent(EntityPath(camera_view_path.into())),
            IncludedContent(EntityPath(patch_depth_view_path.into())),
        ]),
    )?;
    bp.log(
        root_container_path.as_str(),
        &ContainerBlueprint::new(ContainerKind::Horizontal).with_contents([
            IncludedContent(EntityPath(left_container_path.into())),
            IncludedContent(EntityPath(world_view_path.into())),
        ]),
    )?;
    bp.log(
        "viewport",
        &ViewportBlueprint::new()
            .with_root_container(RootContainer(root_container_id))
            .with_auto_layout(AutoLayout(Bool(false)))
            .with_auto_views(AutoViews(Bool(false))),
    )?;

    let msgs = storage.take();
    let blueprint_id = msgs
        .first()
        .and_then(|msg| match msg {
            LogMsg::SetStoreInfo(info) => Some(info.info.store_id.clone()),
            _ => None,
        })
        .expect("blueprint memory stream should contain SetStoreInfo");

    rec.send_blueprint(
        msgs,
        BlueprintActivationCommand {
            blueprint_id,
            make_active: true,
            make_default: true,
        },
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading dataset: {}", args.dataset);

    // Load config if provided
    let vio_config = if let Some(conf_path) = &args.config {
        println!("Loading config from: {}", conf_path);
        Some(VIOConfig::from_yaml(conf_path)?)
    } else {
        None
    };

    let cam_lag = if let Some(conf) = &vio_config {
        conf.main.camera_lag
    } else {
        args.cam_lag
    };

    let reader = ASLDatasetReader::new(&args.dataset, cam_lag);
    let mut imu_it = reader.imu_iter().peekable();
    let mut image_it = reader.image_iter().peekable();

    let settings = if let Some(conf) = &vio_config {
        conf.to_filter_settings()
    } else {
        let mut s = VIOFilterSettings::default();
        s.coordinate_choice = args.coord.clone();
        s
    };

    println!(
        "Filter: chart={}, max_landmarks={}",
        settings.coordinate_choice, settings.max_landmarks
    );

    let (cam_model, k_matrix, img_w, img_h): (Arc<dyn CameraModel>, Matrix3<f64>, usize, usize) =
        if let Some(intr) = &reader.intrinsics {
            println!(
                "Camera intrinsics: {}x{} fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
                intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
            );
            let model: Arc<dyn CameraModel> = match (
                intr.distortion_model.as_deref(),
                &intr.distortion_coefficients,
            ) {
                (Some("radial-tangential"), Some(d)) if d.len() >= 4 => {
                    println!(
                        "Distortion: radial-tangential k1={:.4} k2={:.4} p1={:.6} p2={:.6}",
                        d[0], d[1], d[2], d[3]
                    );
                    Arc::new(RadTanModel {
                        fx: intr.fx,
                        fy: intr.fy,
                        cx: intr.cx,
                        cy: intr.cy,
                        k1: d[0],
                        k2: d[1],
                        p1: d[2],
                        p2: d[3],
                    })
                }
                _ => {
                    println!("Distortion: none (pinhole)");
                    Arc::new(PinholeModel {
                        fx: intr.fx,
                        fy: intr.fy,
                        cx: intr.cx,
                        cy: intr.cy,
                    })
                }
            };
            (
                model,
                Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0),
                intr.width,
                intr.height,
            )
        } else {
            println!("No intrinsics found, using EuRoC defaults");
            (
                Arc::new(PinholeModel {
                    fx: 458.65,
                    fy: 457.3,
                    cx: 367.2,
                    cy: 248.3,
                }) as Arc<dyn CameraModel>,
                Matrix3::new(458.65, 0.0, 367.2, 0.0, 457.3, 248.3, 0.0, 0.0, 1.0),
                752,
                480,
            )
        };

    let patch_depth_enabled = args.patch_depth
        || vio_config
            .as_ref()
            .and_then(|conf| conf.patch_depth.as_ref())
            .is_some();
    let patch_depth_settings = vio_config
        .as_ref()
        .and_then(|conf| conf.patch_depth.as_ref())
        .map(|conf| conf.to_patch_depth_settings())
        .unwrap_or_else(PatchDepthSettings::default);
    #[cfg(feature = "rerun")]
    let (patch_depth_vis_min_depth, patch_depth_vis_max_depth) = vio_config
        .as_ref()
        .and_then(|conf| conf.patch_depth.as_ref())
        .map(|conf| {
            (
                conf.vis_min_depth.unwrap_or(0.1),
                conf.vis_max_depth.unwrap_or(5.0),
            )
        })
        .unwrap_or((0.1, 5.0));

    // Initialize Rudolf-V Frontend
    let mut frontend_config = FrontendConfig::default();
    if let Some(conf) = &vio_config {
        frontend_config.max_features = conf.rudolf_v.max_features;
        frontend_config.pyramid_levels = conf.rudolf_v.max_level;
        if conf.rudolf_v.equalise_image_histogram {
            frontend_config.histeq = rudolf_v::histeq::HistEqMethod::Global;
        }
        frontend_config.cell_size = conf.rudolf_v.feature_dist as usize;
        if let Some(policy) = &conf.rudolf_v.lbp_policy {
            frontend_config.lbp_policy = match policy.to_ascii_lowercase().as_str() {
                "softpenalty" | "soft_penalty" | "soft-penalty" => LbpPolicy::SoftPenalty,
                "hardreject" | "hard_reject" | "hard-reject" => LbpPolicy::HardReject,
                _ => {
                    return Err(format!(
                        "unsupported RudolfV.lbpPolicy '{}'; expected SoftPenalty or HardReject",
                        policy
                    )
                    .into());
                }
            };
        }
    } else {
        frontend_config.max_features = 40;
        frontend_config.cell_size = 100;
        frontend_config.histeq = rudolf_v::histeq::HistEqMethod::Global;
    }
    frontend_config.klt_method = LkMethod::InverseCompositional;
    let tracker_max_features = frontend_config.max_features;
    let mut frontend = Frontend::new(frontend_config, img_w, img_h);
    println!("Tracker: Rudolf-V, max_features={}", tracker_max_features);

    // Initialize Rerun visualization
    #[cfg(feature = "rerun")]
    let rec: Option<rerun::RecordingStream> = if args.vis {
        match rerun::RecordingStreamBuilder::new("echo-li").spawn() {
            Ok(r) => {
                println!("Rerun viewer connected");
                send_rerun_blueprint(&r, img_w, img_h).ok();

                Some(r)
            }
            Err(e) => {
                eprintln!("Warning: could not spawn Rerun viewer: {e}");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "rerun"))]
    if args.vis {
        eprintln!("Warning: --vis requires building with --features rerun");
    }

    let mut filter: Option<VIOFilter> = None;
    let mut patch_depth_mapper = if patch_depth_enabled {
        let mapper = PatchDepthMapper::new(
            Arc::clone(&cam_model),
            CameraIntrinsics::from_matrix(&k_matrix),
            img_w,
            img_h,
            patch_depth_settings.clone(),
        )?;
        println!(
            "Patch depth: enabled mode={:?} scale={:.2}, patch={} stride={} cell={} levels={}",
            patch_depth_settings.camera_mode,
            patch_depth_settings.scale,
            patch_depth_settings.patch_size,
            patch_depth_settings.patch_stride,
            patch_depth_settings.cell_size,
            patch_depth_settings.n_pyramid_levels
        );
        #[cfg(feature = "rerun")]
        println!(
            "Patch depth visualization: fixed depth range [{:.2}, {:.2}] m",
            patch_depth_vis_min_depth, patch_depth_vis_max_depth
        );
        Some(mapper)
    } else {
        println!("Patch depth: disabled");
        None
    };
    let mut sparse_filter = if let Some(conf) = &vio_config {
        if let Some(sparse_conf) = &conf.sparse_vog {
            if sparse_conf.enabled {
                let sparse_chart = parse_sparse_chart(&sparse_conf.parametrization);
                let sparse_settings = sparse_conf.to_sparse_settings();
                println!(
                    "Sparse filter: {:?}, max_pool_size={}",
                    sparse_chart, sparse_settings.max_pool_size
                );
                Some(Sparse3DFilter::new(k_matrix, sparse_chart, sparse_settings))
            } else {
                println!("Sparse filter: disabled by config");
                None
            }
        } else if args.sparse {
            let sparse_chart = parse_sparse_chart(&args.sparse_chart);
            let mut sparse_settings = SparseVogSettings::default();
            sparse_settings.max_pool_size = tracker_max_features.max(300);
            println!(
                "Sparse filter: {:?}, max_pool_size={}",
                sparse_chart, sparse_settings.max_pool_size
            );
            Some(Sparse3DFilter::new(k_matrix, sparse_chart, sparse_settings))
        } else {
            None
        }
    } else if args.sparse {
        let sparse_chart = parse_sparse_chart(&args.sparse_chart);
        let mut sparse_settings = SparseVogSettings::default();
        sparse_settings.max_pool_size = tracker_max_features.max(300);
        println!(
            "Sparse filter: {:?}, max_pool_size={}",
            sparse_chart, sparse_settings.max_pool_size
        );
        Some(Sparse3DFilter::new(k_matrix, sparse_chart, sparse_settings))
    } else {
        None
    };
    let mut initial_imu = Vec::new();
    let mut initialized = false;
    let mut states_out: Vec<(f64, VIOState)> = Vec::new();
    let mut imu_count: usize = 0;
    let mut vision_count: usize = 0;
    let mut last_patch_depth_counts: Option<(usize, usize, usize, usize)> = None;
    #[cfg(feature = "rerun")]
    let mut last_patch_depth_output: Option<PatchDepthOutput> = None;
    #[cfg(feature = "rerun")]
    let mut patch_depth_vis_announced = false;
    let t_start = std::time::Instant::now();

    #[cfg(feature = "rerun")]
    let mut trajectory_vis: Vec<nalgebra::Vector3<f64>> = Vec::new();

    println!("\nRunning filter...");

    loop {
        if let Some(next_img) = image_it.peek() {
            while let Some(imu) = imu_it.peek() {
                if imu.stamp > next_img.stamp {
                    break;
                }
                let imu = imu_it.next().unwrap();
                imu_count += 1;

                if !initialized {
                    initial_imu.push(imu);
                    if initial_imu.len() >= 100 {
                        if check_stationary(&initial_imu, 100, 0.1, 0.5) {
                            let mut xi0 = VIOState::new(VIOSensorState::identity(), Vec::new());
                            xi0.sensor.pose = estimate_initial_pose(&initial_imu, 100);
                            if let Some(ext) = &reader.camera_extrinsics {
                                xi0.sensor.camera_offset = ext.clone();
                            }
                            println!("Initialized at t={:.3}", imu.stamp);
                            filter = Some(VIOFilter::new(settings.clone(), xi0));
                            // Warm up the covariance by replaying initial IMU
                            let f = filter.as_mut().unwrap();
                            for past_imu in &initial_imu {
                                f.process_imu(*past_imu);
                            }
                            initialized = true;
                        } else {
                            initial_imu.remove(0);
                        }
                    }
                } else if let Some(f) = &mut filter {
                    f.process_imu(imu);
                }
            }
        }

        if let Some(img_data) = image_it.next() {
            if let Ok(dynamic_img) = image::open(&img_data.image_path) {
                let gray_img = dynamic_img.to_luma8();
                let gray_data = gray_img.into_raw();

                // Clone raw pixels for Rerun before consuming into Rudolf-V
                #[cfg(feature = "rerun")]
                let rerun_gray_data = if rec.is_some() {
                    gray_data.clone()
                } else {
                    vec![]
                };

                let rudolf_img = RudolfImage::from_vec(img_w, img_h, gray_data);

                let (features, _stats) = frontend.process(&rudolf_img);

                if let Some(f) = &mut filter {
                    #[cfg(feature = "rerun")]
                    let tracker_points: Vec<(f32, f32)> = features
                        .iter()
                        .filter_map(|feat| {
                            clip_image_point(feat.x as f64, feat.y as f64, img_w, img_h)
                        })
                        .collect();

                    let mut feat_uvs = HashMap::new();
                    for feat in features {
                        feat_uvs.insert(feat.id, Vector2::new(feat.x, feat.y));
                    }

                    let measurement = VisionMeasurement::new(img_data.stamp, feat_uvs);
                    let sparse_measurement = undistorted_pinhole_measurement(
                        img_data.stamp,
                        &measurement.cam_coordinates,
                        cam_model.as_ref(),
                        &k_matrix,
                    );
                    let patch_gray_data = if patch_depth_mapper.is_some() {
                        frontend
                            .preprocessed_image()
                            .map(|img| img.as_slice()[..img_w * img_h].to_vec())
                            .unwrap_or_else(|| rudolf_img.as_slice()[..img_w * img_h].to_vec())
                    } else {
                        Vec::new()
                    };
                    f.process_vision(measurement.clone(), cam_model.as_ref());
                    vision_count += 1;

                    let state = f.eqf.state_estimate();
                    let t_wc = camera_pose_matrix(&state);
                    if let Some(sparse) = &mut sparse_filter {
                        sparse.update(&sparse_measurement, &t_wc, None);
                        if let Some(mapper) = &mut patch_depth_mapper {
                            if !patch_gray_data.is_empty() {
                                let patch_measurement = match mapper.camera_mode() {
                                    PatchDepthCameraMode::RawDistorted => &measurement,
                                    PatchDepthCameraMode::UndistortedPinhole => &sparse_measurement,
                                };
                                let patch_seed_coordinates = match mapper.camera_mode() {
                                    PatchDepthCameraMode::RawDistorted => {
                                        PatchDepthSeedCoordinates::RawDistorted
                                    }
                                    PatchDepthCameraMode::UndistortedPinhole => {
                                        PatchDepthSeedCoordinates::UndistortedPinhole
                                    }
                                };
                                let frame = FrameProducts {
                                    frame_id: vision_count as u64,
                                    stamp: img_data.stamp,
                                    gray: patch_gray_data.clone(),
                                    width: img_w,
                                    height: img_h,
                                    pose_t_wc: t_wc,
                                };
                                let patch_output = mapper.update(
                                    sparse,
                                    patch_measurement,
                                    patch_seed_coordinates,
                                    frame,
                                );
                                last_patch_depth_counts =
                                    patch_output.as_ref().map(patch_depth_status_counts);
                                #[cfg(feature = "rerun")]
                                {
                                    last_patch_depth_output = patch_output;
                                }
                            }
                        }
                    }

                    #[cfg(feature = "rerun")]
                    let (feat_global, p_cam, r_cam) = echo_li_core::landmarks_to_global(&state);
                    #[cfg(not(feature = "rerun"))]
                    let (feat_global, _, _) = echo_li_core::landmarks_to_global(&state);
                    states_out.push((f.eqf.current_time, state));

                    // Rerun per-frame logging
                    #[cfg(feature = "rerun")]
                    if let Some(rec) = &rec {
                        use std::time::Duration;
                        rec.set_time("log_time", Duration::from_secs_f64(img_data.stamp));

                        // Camera image (grayscale)
                        rec.log(
                            "camera/image",
                            &rerun::Image::from_l8(rerun_gray_data, [img_w as u32, img_h as u32]),
                        )
                        .ok();

                        if let Some(output) = &last_patch_depth_output {
                            let rgb = patch_depth_rgb_for_vis(
                                output,
                                img_w,
                                img_h,
                                patch_depth_vis_min_depth,
                                patch_depth_vis_max_depth,
                            );
                            rec.log(
                                "patch_depth/image",
                                &rerun::Image::from_rgb24(rgb, [img_w as u32, img_h as u32]),
                            )
                            .ok();
                            if !patch_depth_vis_announced {
                                println!("Patch depth Rerun entity: patch_depth/image");
                                patch_depth_vis_announced = true;
                            }
                        }

                        // Tracked features on image
                        if !tracker_points.is_empty() {
                            rec.log(
                                "camera/image/features",
                                &rerun::Points2D::new(tracker_points)
                                    .with_colors([0xFFFF00FFu32])
                                    .with_radii([2.0f32]),
                            )
                            .ok();
                        }

                        let (
                            sparse_img_pts,
                            sparse_img_colors,
                            sparse_world_pts,
                            sparse_world_colors,
                        ) = if let Some(sparse) = &sparse_filter {
                            sparse_points_for_vis(
                                sparse,
                                cam_model.as_ref(),
                                &p_cam,
                                &r_cam,
                                img_w,
                                img_h,
                            )
                        } else {
                            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                        };
                        if !sparse_img_pts.is_empty() {
                            rec.log(
                                "camera/image/sparse_out_of_state",
                                &rerun::Points2D::new(sparse_img_pts)
                                    .with_colors(sparse_img_colors)
                                    .with_radii([3.0f32]),
                            )
                            .ok();
                        }

                        // Accumulate trajectory
                        trajectory_vis.push(p_cam);

                        // 3D trajectory line
                        if trajectory_vis.len() >= 2 {
                            let strip: Vec<[f32; 3]> = trajectory_vis
                                .iter()
                                .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                                .collect();
                            rec.log(
                                "world/trajectory",
                                &rerun::LineStrips3D::new([strip]).with_colors([0x00FFFFFFu32]),
                            )
                            .ok();
                        }

                        // 3D landmarks
                        let lm_pts: Vec<(f32, f32, f32)> = feat_global
                            .values()
                            .map(|p| (p[0] as f32, p[1] as f32, p[2] as f32))
                            .collect();
                        if !lm_pts.is_empty() {
                            rec.log(
                                "world/landmarks",
                                &rerun::Points3D::new(lm_pts)
                                    .with_colors([0x00FF00FFu32])
                                    .with_radii([0.02f32]),
                            )
                            .ok();
                        }

                        if !sparse_world_pts.is_empty() {
                            rec.log(
                                "world/sparse_out_of_state",
                                &rerun::Points3D::new(sparse_world_pts)
                                    .with_colors(sparse_world_colors)
                                    .with_radii([0.015f32]),
                            )
                            .ok();
                        }

                        // Camera pose as RGB arrows (X=red, Y=green, Z=blue)
                        let origin = [p_cam[0] as f32, p_cam[1] as f32, p_cam[2] as f32];
                        let r = r_cam.as_matrix();
                        let scale = 0.1f32;
                        rec.log(
                            "world/camera_axes",
                            &rerun::Arrows3D::from_vectors([
                                [
                                    r[(0, 0)] as f32 * scale,
                                    r[(1, 0)] as f32 * scale,
                                    r[(2, 0)] as f32 * scale,
                                ],
                                [
                                    r[(0, 1)] as f32 * scale,
                                    r[(1, 1)] as f32 * scale,
                                    r[(2, 1)] as f32 * scale,
                                ],
                                [
                                    r[(0, 2)] as f32 * scale,
                                    r[(1, 2)] as f32 * scale,
                                    r[(2, 2)] as f32 * scale,
                                ],
                            ])
                            .with_origins([origin, origin, origin])
                            .with_colors([
                                0xFF0000FFu32,
                                0x00FF00FFu32,
                                0x0000FFFFu32,
                            ]),
                        )
                        .ok();
                    }

                    if vision_count % 100 == 0 || vision_count <= 5 {
                        let pos = states_out.last().unwrap().1.sensor.pose.translation;
                        let vel = states_out.last().unwrap().1.sensor.velocity;
                        if let Some((unk, seed, photo, rej)) = last_patch_depth_counts {
                            println!(
                                "  [{:4}] t={:.3}  pos=({:+.2}, {:+.2}, {:+.2})  vel=({:+.3}, {:+.3}, {:+.3})  lm={}  patch=(photo:{} seed:{} unk:{} rej:{})",
                                vision_count,
                                img_data.stamp,
                                pos[0],
                                pos[1],
                                pos[2],
                                vel[0],
                                vel[1],
                                vel[2],
                                feat_global.len(),
                                photo,
                                seed,
                                unk,
                                rej
                            );
                        } else {
                            println!(
                                "  [{:4}] t={:.3}  pos=({:+.2}, {:+.2}, {:+.2})  vel=({:+.3}, {:+.3}, {:+.3})  lm={}",
                                vision_count,
                                img_data.stamp,
                                pos[0],
                                pos[1],
                                pos[2],
                                vel[0],
                                vel[1],
                                vel[2],
                                feat_global.len()
                            );
                        }
                    }
                }
            }
        } else {
            break;
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "\nProcessed {} IMU + {} vision in {:.2}s",
        imu_count, vision_count, elapsed
    );

    // Write trajectory output
    let dataset_name = PathBuf::from(&args.dataset)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let output_dir = PathBuf::from(
        args.output
            .unwrap_or_else(|| format!("eqvio_output_{}", dataset_name)),
    );

    if !states_out.is_empty() {
        let traj_file = output_dir.join("estimated_trajectory.txt");
        write_trajectory(&traj_file, &states_out)?;
        println!("Trajectory written to {}", traj_file.display());
    }

    let gt_poses = reader.groundtruth();
    if !gt_poses.is_empty() {
        let gt_file = output_dir.join("groundtruth_trajectory.txt");
        write_groundtruth(&gt_file, &gt_poses)?;
        println!("Ground truth written to {}", gt_file.display());

        // Aligned trajectory
        if !states_out.is_empty() {
            let est_poses: Vec<(f64, echo_lie::SE3)> = states_out
                .iter()
                .map(|(t, s)| (*t, s.sensor.pose.clone()))
                .collect();
            let alignment = echo_li_core::alignment::align_trajectories(&est_poses, &gt_poses);
            let aligned: Vec<(f64, VIOState)> = states_out
                .iter()
                .map(|(t, s)| {
                    let mut s_aligned = s.clone();
                    let aligned_pose = alignment.compose(&s.sensor.pose);
                    s_aligned.sensor.pose = aligned_pose;
                    (*t, s_aligned)
                })
                .collect();
            let aligned_file = output_dir.join("aligned_trajectory.txt");
            write_trajectory(&aligned_file, &aligned)?;
            println!("Aligned trajectory written to {}", aligned_file.display());
        }
    }

    println!("Done.");
    Ok(())
}
