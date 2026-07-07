use clap::Parser;
use echo_li_core::config::VIOConfig;
use echo_li_core::core_types::CameraIntrinsics;
use echo_li_core::dataserver::ASLDatasetReader;
use echo_li_core::depth::occupancy::{LocalOccupancyMap, LocalOccupancySettings};
use echo_li_core::depth::patch_depth::{
    FrameProducts, PatchDepthCameraMode, PatchDepthMapper, PatchDepthOutput,
    PatchDepthSeedCoordinates, PatchDepthSettings, PatchStatus,
};
use echo_li_core::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use echo_li_core::depth::sparse_gb::SparseVogSettings;
use echo_li_core::initialization::{check_stationary, estimate_initial_pose};
use echo_li_core::mathematical::camera::CameraModel;
use echo_li_core::mathematical::*;
use echo_li_core::trajectory_metrics::TrajectoryMetrics;
use echo_li_core::{LandmarkDepthPrior, VIOFilter, VIOFilterSettings};
use nalgebra::{Matrix3, Matrix4, Vector2};
use rudolf_v::camera::CameraIntrinsics as RudolfCameraIntrinsics;
use rudolf_v::camera::StereoRig;
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::klt::LkMethod;
use rudolf_v::rigid_ransac::{Correspondence3d, Rigid3dRansacConfig};
use rudolf_v::stereo::{StereoConfig as RudolfStereoConfig, StereoMatcher};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
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

    /// Sparse filter chart: polar3d, invdepth3d, or invdepth_additive3d (ρ-first).
    #[arg(long, default_value = "polar3d")]
    sparse_chart: String,

    /// Run patch-grid direct depth mapper. Enabled automatically by PatchDepth config.
    #[arg(long, default_value_t = false)]
    patch_depth: bool,

    /// Disable patch-grid direct depth mapper even when PatchDepth exists in YAML.
    #[arg(long, default_value_t = false)]
    no_patch_depth: bool,

    /// Enable stereo matching for EqF landmark depth initialization (requires cam1).
    #[arg(long, default_value_t = false)]
    stereo: bool,

    /// Build a rolling local occupancy map from patch-depth rays.
    #[arg(long, default_value_t = false)]
    occupancy_map: bool,
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

fn write_trajectory_metrics(
    path: &std::path::Path,
    metrics: &TrajectoryMetrics,
) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "matched_poses {}", metrics.matched_poses)?;
    writeln!(f, "ate_position_rmse_m {:.9}", metrics.ate_position_m.rmse)?;
    writeln!(f, "ate_position_mean_m {:.9}", metrics.ate_position_m.mean)?;
    writeln!(
        f,
        "ate_position_median_m {:.9}",
        metrics.ate_position_m.median
    )?;
    writeln!(f, "ate_position_max_m {:.9}", metrics.ate_position_m.max)?;
    writeln!(
        f,
        "ate_attitude_rmse_deg {:.9}",
        metrics.ate_attitude_deg.rmse
    )?;
    writeln!(
        f,
        "ate_attitude_mean_deg {:.9}",
        metrics.ate_attitude_deg.mean
    )?;
    writeln!(
        f,
        "ate_attitude_median_deg {:.9}",
        metrics.ate_attitude_deg.median
    )?;
    writeln!(
        f,
        "ate_attitude_max_deg {:.9}",
        metrics.ate_attitude_deg.max
    )?;
    Ok(())
}

fn parse_sparse_chart(name: &str) -> Sparse3DChart {
    match name.to_ascii_lowercase().as_str() {
        "invdepth_additive3d" | "invdepth-additive" | "rho3d" | "rho" | "additive" => {
            Sparse3DChart::InvDepthAdditive
        }
        "invdepth" | "invdepth3d" | "inverse-depth" => Sparse3DChart::InvDepth,
        _ => Sparse3DChart::Polar,
    }
}

fn hash_u64(hasher: &mut DefaultHasher, value: u64) {
    value.hash(hasher);
}

fn hash_f64(hasher: &mut DefaultHasher, value: f64) {
    value.to_bits().hash(hasher);
}

fn hash_f32(hasher: &mut DefaultHasher, value: f32) {
    value.to_bits().hash(hasher);
}

fn hash_features(features: &[rudolf_v::fast::Feature]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for feat in features {
        hash_u64(&mut hasher, feat.id);
        hash_f32(&mut hasher, feat.x);
        hash_f32(&mut hasher, feat.y);
        hash_f32(&mut hasher, feat.score);
        feat.descriptor.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_depth_priors(depth_priors: &HashMap<u64, LandmarkDepthPrior>) -> u64 {
    let mut ids: Vec<_> = depth_priors.keys().copied().collect();
    ids.sort_unstable();
    let mut hasher = DefaultHasher::new();
    for id in ids {
        let prior = depth_priors[&id];
        hash_u64(&mut hasher, id);
        hash_f64(&mut hasher, prior.range);
        hash_f64(&mut hasher, prior.range_var);
    }
    hasher.finish()
}

fn hash_sparse_filter(sparse: Option<&Sparse3DFilter>) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(sparse) = sparse {
        let mut ids: Vec<_> = sparse.features_iter().map(|feat| feat.feat_id).collect();
        ids.sort_unstable();
        for id in ids {
            let (range, range_var) = sparse.query_range(id);
            hash_u64(&mut hasher, id);
            hash_f64(&mut hasher, range);
            hash_f64(&mut hasher, range_var);
        }
    }
    hasher.finish()
}

fn hash_state(state: &VIOState) -> u64 {
    let mut hasher = DefaultHasher::new();
    let pose = &state.sensor.pose;
    let q = pose.rotation.as_xyzw();
    for v in q.iter() {
        hash_f64(&mut hasher, *v);
    }
    for v in pose.translation.iter() {
        hash_f64(&mut hasher, *v);
    }
    for v in state.sensor.velocity.iter() {
        hash_f64(&mut hasher, *v);
    }
    for v in state.sensor.input_bias.iter() {
        hash_f64(&mut hasher, *v);
    }
    for lm in &state.camera_landmarks {
        hash_u64(&mut hasher, lm.id);
        for v in lm.p.iter() {
            hash_f64(&mut hasher, *v);
        }
    }
    hasher.finish()
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
    for status in &output.status.data {
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
    let dw = output.eta.width;
    let dh = output.eta.height;
    for y in 0..img_h {
        let dy = y * dh / img_h;
        for x in 0..img_w {
            let dx = x * dw / img_w;
            let eta = output.eta.data[dy * dw + dx];
            if !eta.is_finite() {
                continue;
            }
            // Output is log-range η; show range = exp(η) (thin edge conversion).
            let range = (eta as f64).exp();
            let color = color_for_depth(range, vis_min_depth, vis_max_depth);
            let out = (y * img_w + x) * 3;
            rgb[out] = ((color >> 24) & 0xFF) as u8;
            rgb[out + 1] = ((color >> 16) & 0xFF) as u8;
            rgb[out + 2] = ((color >> 8) & 0xFF) as u8;
        }
    }

    rgb
}

/// Render the patch-depth confidence as a relative range-std image.
/// `output.eta_var` is `var(η)` for `η = ln(range)`, so `sqrt(var(η)) ≈ σ_range/range`
/// is a dimensionless relative depth uncertainty — no `z` conversion needed. Note
/// its absolute scale is uncalibrated (it bakes in status/photo weights), so the
/// thresholds are relative. Low (confident) maps to blue, high (uncertain) to red.
#[cfg(feature = "rerun")]
fn patch_depth_cov_rgb_for_vis(
    output: &PatchDepthOutput,
    img_w: usize,
    img_h: usize,
    cov_vis_min: f64,
    cov_vis_max: f64,
) -> Vec<u8> {
    let mut rgb = vec![0u8; img_w * img_h * 3];
    if cov_vis_max <= cov_vis_min {
        return rgb;
    }
    let dw = output.eta.width;
    let dh = output.eta.height;
    for y in 0..img_h {
        let dy = y * dh / img_h;
        for x in 0..img_w {
            let dx = x * dw / img_w;
            let idx = dy * dw + dx;
            let eta = output.eta.data[idx];
            let eta_var = output.eta_var.data[idx];
            if !eta.is_finite() || !eta_var.is_finite() || eta_var <= 0.0 {
                continue;
            }
            // sqrt(var(η)) ≈ σ_range/range: relative range std (dimensionless).
            let rel_std = (eta_var as f64).sqrt();
            let color = color_for_scalar(rel_std, cov_vis_min, cov_vis_max);
            let out = (y * img_w + x) * 3;
            rgb[out] = ((color >> 24) & 0xFF) as u8;
            rgb[out + 1] = ((color >> 16) & 0xFF) as u8;
            rgb[out + 2] = ((color >> 8) & 0xFF) as u8;
        }
    }

    rgb
}

#[cfg(feature = "rerun")]
fn clip_image_point(x: f64, y: f64, img_w: usize, img_h: usize) -> Option<(f32, f32)> {
    (x >= 0.0 && x < img_w as f64 && y >= 0.0 && y < img_h as f64).then_some((x as f32, y as f32))
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
    vis_min_depth: f64,
    vis_max_depth: f64,
) -> (Vec<(f32, f32)>, Vec<u32>, Vec<(f32, f32, f32)>, Vec<u32>) {
    struct SparseVisPoint {
        image_point: Option<(f32, f32)>,
        world_point: (f32, f32, f32),
        depth: f64,
    }

    let mut points = Vec::new();

    for feat in sparse_filter.features_iter() {
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
            depth: q[2],
        });
    }

    let mut image_points = Vec::new();
    let mut image_colors = Vec::new();
    let mut world_points = Vec::new();
    let mut world_colors = Vec::new();

    for point in points {
        let color = color_for_depth(point.depth, vis_min_depth, vis_max_depth);
        if let Some(image_point) = point.image_point {
            image_points.push(image_point);
            image_colors.push(color);
        }
        world_points.push(point.world_point);
        world_colors.push(color);
    }

    (image_points, image_colors, world_points, world_colors)
}

/// Occupied / free voxel centres of the local 3D map as world points for Rerun.
/// Occupied voxels are returned in full; free voxels are decimated by
/// `FREE_VIS_STRIDE` per axis (the free volume is otherwise far too many cells to
/// stream/draw). Unknown voxels are omitted.
#[cfg(feature = "rerun")]
fn occupancy_cells_for_vis(map: &LocalOccupancyMap) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    // Free thinning: keep every Nth voxel per axis (2 -> 1/8 the count).
    const FREE_VIS_STRIDE: usize = 2;
    let snap = map.snapshot();
    let occ_thr = map.settings().occupied_threshold;
    let free_thr = map.settings().free_threshold;
    let r = snap.resolution;
    let (mut occupied, mut free) = (Vec::new(), Vec::new());
    for cz in 0..snap.depth {
        for cy in 0..snap.height {
            for cx in 0..snap.width {
                let l = snap.log_odds[(cz * snap.height + cy) * snap.width + cx];
                let occupied_cell = l >= occ_thr;
                if !occupied_cell && l > free_thr {
                    continue; // unknown
                }
                if !occupied_cell
                    && (cx % FREE_VIS_STRIDE != 0
                        || cy % FREE_VIS_STRIDE != 0
                        || cz % FREE_VIS_STRIDE != 0)
                {
                    continue; // thinned-out free voxel
                }
                let p = [
                    (snap.origin_x + (cx as f64 + 0.5) * r) as f32,
                    (snap.origin_y + (cy as f64 + 0.5) * r) as f32,
                    (snap.origin_z + (cz as f64 + 0.5) * r) as f32,
                ];
                if occupied_cell {
                    occupied.push(p);
                } else {
                    free.push(p);
                }
            }
        }
    }
    (occupied, free)
}

#[cfg(feature = "rerun")]
fn send_rerun_blueprint(
    rec: &rerun::RecordingStream,
    k_matrix: &Matrix3<f64>,
    img_w: usize,
    img_h: usize,
) -> rerun::RecordingStreamResult<()> {
    use rerun::external::re_log_types::{BlueprintActivationCommand, LogMsg, RecordingId};
    use rerun::external::re_sdk_types::blueprint::archetypes::{
        ContainerBlueprint, EyeControls3D, ViewBlueprint, ViewContents, ViewportBlueprint,
        VisualBounds2D,
    };
    use rerun::external::re_sdk_types::blueprint::components::{
        AutoLayout, AutoViews, ContainerKind, IncludedContent, RootContainer, ViewOrigin,
    };
    use rerun::external::re_sdk_types::components::{Name, Visible};
    use rerun::external::re_sdk_types::datatypes::{Bool, EntityPath, Range2D, Uuid};

    let pinhole = || {
        rerun::Pinhole::from_focal_length_and_resolution(
            [k_matrix[(0, 0)] as f32, k_matrix[(1, 1)] as f32],
            [img_w as f32, img_h as f32],
        )
        .with_principal_point([k_matrix[(0, 2)] as f32, k_matrix[(1, 2)] as f32])
        // CamerasVisualizer registers the camera for tracking before drawing
        // its frustum. Degenerate, transparent drawing properties therefore
        // hide the helper without removing it from the tracking camera list.
        .with_image_plane_distance(0.0)
        .with_line_width(0.0)
        .with_color(rerun::Color::TRANSPARENT)
    };
    // Only the virtual camera needs pinhole semantics: that makes Rerun adopt
    // its full pose when tracking it. The physical pose remains a transform.
    rec.log_static("world/view_camera", &pinhole())?;

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
    let patch_depth_cov_view_id = Uuid::random();
    let world_view_id = Uuid::random();
    let left_container_id = Uuid::random();
    let root_container_id = Uuid::random();
    let camera_view_path = format!("view/{camera_view_id}");
    let patch_depth_view_path = format!("view/{patch_depth_view_id}");
    let patch_depth_cov_view_path = format!("view/{patch_depth_cov_view_id}");
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
        patch_depth_cov_view_path.as_str(),
        &ViewBlueprint::new("2D")
            .with_display_name(Name("Patch Depth Covariance".into()))
            .with_space_origin(ViewOrigin("patch_depth_cov".into()))
            .with_visible(Visible(Bool(true))),
    )?;
    bp.log(
        format!("{patch_depth_cov_view_path}/ViewContents"),
        &ViewContents::new(["patch_depth_cov/**"]),
    )?;
    bp.log(
        format!("{patch_depth_cov_view_path}/VisualBounds2D"),
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
        format!("{world_view_path}/EyeControls3D"),
        &EyeControls3D::default().with_tracking_entity("world/view_camera"),
    )?;

    bp.log(
        left_container_path.as_str(),
        &ContainerBlueprint::new(ContainerKind::Vertical).with_contents([
            IncludedContent(EntityPath(camera_view_path.into())),
            IncludedContent(EntityPath(patch_depth_view_path.into())),
            IncludedContent(EntityPath(patch_depth_cov_view_path.into())),
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

fn build_camera_model(
    reader: &ASLDatasetReader,
) -> (Arc<dyn CameraModel>, Matrix3<f64>, usize, usize) {
    if let Some(intr) = &reader.intrinsics {
        println!(
            "Camera intrinsics: {}x{} fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
            intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
        );
        let distortion = intr
            .distortion_coefficients
            .as_deref()
            .unwrap_or_default()
            .to_vec();
        if distortion.len() >= 4 {
            println!(
                "Distortion: radial-tangential k1={:.4} k2={:.4} p1={:.6} p2={:.6}",
                distortion[0], distortion[1], distortion[2], distortion[3]
            );
        } else {
            println!("Distortion: none (pinhole)");
        }
        let rudolf_cam = RudolfCameraIntrinsics {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            resolution: [intr.width, intr.height],
            distortion,
        };
        (
            Arc::new(rudolf_cam) as Arc<dyn CameraModel>,
            Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0),
            intr.width,
            intr.height,
        )
    } else {
        println!("No intrinsics found, using EuRoC defaults");
        let rudolf_cam = RudolfCameraIntrinsics::new(458.65, 457.3, 367.2, 248.3, 752, 480);
        (
            Arc::new(rudolf_cam) as Arc<dyn CameraModel>,
            Matrix3::new(458.65, 0.0, 367.2, 0.0, 457.3, 248.3, 0.0, 0.0, 1.0),
            752,
            480,
        )
    }
}

fn build_frontend(
    vio_config: Option<&VIOConfig>,
    img_w: usize,
    img_h: usize,
    camera: Option<RudolfCameraIntrinsics>,
) -> Result<(Frontend, usize), Box<dyn std::error::Error>> {
    let mut config = FrontendConfig::default();
    // Intrinsics enable the geometric-verification paths (pose-prior epipolar
    // gate / internal RANSAC); without them both are inert.
    config.camera = camera;
    if let Some(conf) = vio_config {
        config.max_features = conf.rudolf_v.max_features;
        config.klt_residual_enabled = conf.rudolf_v.klt_residual;
        config.enable_internal_ransac = conf.rudolf_v.enable_ransac;
        config.epipolar_gate_threshold = conf.rudolf_v.epipolar_gate_threshold;
        config.epipolar_refine = conf.rudolf_v.epipolar_refine;
        config.epipolar_min_baseline = conf.rudolf_v.epipolar_min_baseline;
        config.epipolar_max_reject_frac = conf.rudolf_v.epipolar_max_reject_frac;
        config.pyramid_levels = conf.rudolf_v.max_level;
        if let Some(fast_threshold) = conf.rudolf_v.fast_threshold {
            config.fast_threshold = fast_threshold;
        }
        if let Some(detector) = &conf.rudolf_v.detector {
            config.detector = match detector.to_ascii_lowercase().as_str() {
                "fast" => DetectorType::Fast,
                "harris" => DetectorType::Harris,
                "shi_tomasi" | "shitomasi" | "shi-tomasi" => DetectorType::ShiTomasi,
                _ => {
                    return Err(format!(
                        "unsupported RudolfV.detector '{}'; expected fast, harris, or shi_tomasi",
                        detector
                    )
                    .into());
                }
            };
        }
        if let Some(v) = conf.rudolf_v.shi_tomasi_threshold {
            config.shi_tomasi_threshold = v;
        }
        if let Some(v) = conf.rudolf_v.shi_tomasi_block_size {
            config.shi_tomasi_block_size = v;
        }
        config.histeq = match conf.rudolf_v.histeq.as_deref() {
            Some("none") => rudolf_v::histeq::HistEqMethod::None,
            Some("global") => rudolf_v::histeq::HistEqMethod::Global,
            Some("clahe") => rudolf_v::histeq::HistEqMethod::Clahe {
                tile_size: conf.rudolf_v.clahe_tile_size,
                clip_limit: conf.rudolf_v.clahe_clip_limit,
            },
            Some(other) => {
                return Err(format!(
                    "unsupported RudolfV.histeq '{other}'; expected none, global, or clahe"
                )
                .into());
            }
            // Legacy fallback: the equaliseImageHistogram bool.
            None if conf.rudolf_v.equalise_image_histogram => {
                rudolf_v::histeq::HistEqMethod::Global
            }
            None => rudolf_v::histeq::HistEqMethod::None,
        };
        config.cell_size = conf.rudolf_v.feature_dist as usize;
        if let Some(policy) = &conf.rudolf_v.lbp_policy {
            config.lbp_policy = match policy.to_ascii_lowercase().as_str() {
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
        config.max_features = 40;
        config.cell_size = 100;
        config.histeq = rudolf_v::histeq::HistEqMethod::Global;
    }
    config.klt_method = LkMethod::InverseCompositional;
    let tracker_max_features = config.max_features;
    println!(
        "Tracker: Rudolf-V, detector={:?}, max_features={}",
        config.detector, tracker_max_features
    );
    Ok((Frontend::new(config, img_w, img_h), tracker_max_features))
}

fn build_stereo_matcher(
    args: &Args,
    vio_config: Option<&VIOConfig>,
    reader: &ASLDatasetReader,
    img_w: usize,
    img_h: usize,
) -> Option<StereoMatcher> {
    let stereo_enabled = args.stereo
        || vio_config
            .and_then(|c| c.stereo.as_ref())
            .map_or(false, |s| s.enabled);
    if !stereo_enabled || !reader.has_cam1() {
        if stereo_enabled && !reader.has_cam1() {
            eprintln!("Warning: stereo requested but cam1 data not found");
        }
        return None;
    }
    let root = reader.root_path();
    let rig = StereoRig::from_euroc(
        &root.join("cam0/sensor.yaml"),
        &root.join("cam1/sensor.yaml"),
    )
    .expect("Failed to load stereo rig");
    let mut stereo_cfg = RudolfStereoConfig::default();
    if let Some(conf) = vio_config.and_then(|c| c.stereo.as_ref()) {
        if let Some(v) = conf.pyramid_levels {
            stereo_cfg.pyramid_levels = v;
        }
        if let Some(v) = conf.patch_half_size {
            stereo_cfg.patch_half_size = v;
        }
        if let Some(v) = conf.max_iterations {
            stereo_cfg.max_iterations = v;
        }
        if let Some(v) = conf.convergence_eps {
            stereo_cfg.convergence_eps = v;
        }
        if let Some(v) = conf.min_inv_depth {
            stereo_cfg.min_inv_depth = v;
        }
        if let Some(v) = conf.max_inv_depth {
            stereo_cfg.max_inv_depth = v;
        }
        if let Some(v) = conf.init_inv_depth {
            stereo_cfg.init_inv_depth = v;
        }
        if let Some(v) = conf.max_residual {
            stereo_cfg.max_residual = v;
        }
        if let Some(v) = conf.n_search_candidates {
            stereo_cfg.n_search_candidates = v;
        }
        if let Some(v) = conf.knn_propagation {
            stereo_cfg.knn_propagation = v;
        }
        if let Some(h) = &conf.histeq {
            stereo_cfg.histeq = match h.to_ascii_lowercase().as_str() {
                "global" => rudolf_v::histeq::HistEqMethod::Global,
                _ => rudolf_v::histeq::HistEqMethod::None,
            };
        }
    }
    println!(
        "Stereo: baseline={:.4}m cam1 {}×{} patch_half={} levels={} iters={} search={}",
        rig.baseline_meters(),
        rig.cam1.resolution[0],
        rig.cam1.resolution[1],
        stereo_cfg.patch_half_size,
        stereo_cfg.pyramid_levels,
        stereo_cfg.max_iterations,
        stereo_cfg.n_search_candidates,
    );
    Some(StereoMatcher::new(rig, stereo_cfg, img_w, img_h))
}

fn build_patch_depth_mapper(
    enabled: bool,
    settings: PatchDepthSettings,
    cam_model: Arc<dyn CameraModel>,
    k_matrix: Matrix3<f64>,
    img_w: usize,
    img_h: usize,
    stereo_matcher: Option<&StereoMatcher>,
) -> Result<Option<PatchDepthMapper>, Box<dyn std::error::Error>> {
    if !enabled {
        println!("Patch depth: disabled");
        return Ok(None);
    }
    let intrinsics = CameraIntrinsics::from_matrix(&k_matrix);
    println!(
        "Patch depth: enabled mode={:?} warp={:?} scale={:.2}, patch={} stride={} levels={}",
        settings.camera_mode,
        settings.warp_mode,
        settings.scale,
        settings.patch_size,
        settings.patch_stride,
        settings.n_pyramid_levels
    );
    let mut mapper = PatchDepthMapper::new(cam_model, intrinsics, img_w, img_h, settings)?;
    if let Some(matcher) = stereo_matcher {
        let rig = matcher.rig();
        let mut t_c1_c0 = Matrix4::<f64>::identity();
        for r in 0..3 {
            for c in 0..3 {
                t_c1_c0[(r, c)] = rig.r_10[r][c];
            }
            t_c1_c0[(r, 3)] = rig.t_10[r];
        }
        mapper.init_stereo_ref(&rig.cam1, t_c1_c0);
        println!(
            "Patch depth: stereo ref from cam1 (baseline={:.4}m)",
            rig.baseline_meters()
        );
    }
    Ok(Some(mapper))
}

fn build_sparse_filter(
    args: &Args,
    vio_config: Option<&VIOConfig>,
    k_matrix: Matrix3<f64>,
    tracker_max_features: usize,
) -> Option<Sparse3DFilter> {
    if let Some(sparse_conf) = vio_config.and_then(|c| c.sparse_vog.as_ref()) {
        if !sparse_conf.enabled {
            println!("Sparse filter: disabled by config");
            return None;
        }
        let chart = parse_sparse_chart(&sparse_conf.parametrization);
        let settings = sparse_conf.to_sparse_settings();
        println!(
            "Sparse filter: {:?}, max_pool_size={}",
            chart, settings.max_pool_size
        );
        return Some(Sparse3DFilter::new(k_matrix, chart, settings));
    }
    if args.sparse {
        let chart = parse_sparse_chart(&args.sparse_chart);
        let mut settings = SparseVogSettings::default();
        settings.max_pool_size = tracker_max_features.max(300);
        println!(
            "Sparse filter: {:?}, max_pool_size={}",
            chart, settings.max_pool_size
        );
        return Some(Sparse3DFilter::new(k_matrix, chart, settings));
    }
    None
}

fn build_local_occupancy_map(
    args: &Args,
    vio_config: Option<&VIOConfig>,
    patch_depth_enabled: bool,
) -> Result<Option<LocalOccupancyMap>, Box<dyn std::error::Error>> {
    let mut settings = vio_config
        .and_then(|conf| conf.local_occupancy.as_ref())
        .map(|conf| conf.to_local_occupancy_settings())
        .unwrap_or_else(LocalOccupancySettings::default);
    settings.enabled |= args.occupancy_map;

    if !settings.enabled {
        println!("Local occupancy: disabled");
        return Ok(None);
    }
    if !patch_depth_enabled {
        println!("Local occupancy: disabled (requires patch depth)");
        return Ok(None);
    }

    let map = LocalOccupancyMap::new(settings.clone())?;
    println!(
        "Local occupancy: enabled {}x{}x{} voxels @ {:.3}m (z extent [{:.2}, {:.2}]m), range=[{:.2}, {:.2}]m, stride={}",
        settings.width_cells,
        settings.height_cells,
        map.depth_cells(),
        settings.resolution,
        settings.min_obstacle_height,
        settings.max_obstacle_height,
        settings.min_range,
        settings.max_range,
        settings.sample_stride
    );
    Ok(Some(map))
}

fn write_outputs(
    output_dir: &std::path::Path,
    states_out: &[(f64, VIOState)],
    gt_poses: &[StampedPose],
) -> Result<(), Box<dyn std::error::Error>> {
    if !states_out.is_empty() {
        let traj_file = output_dir.join("estimated_trajectory.txt");
        write_trajectory(&traj_file, states_out)?;
        println!("Trajectory written to {}", traj_file.display());
    }

    if !gt_poses.is_empty() {
        let gt_file = output_dir.join("groundtruth_trajectory.txt");
        write_groundtruth(&gt_file, gt_poses)?;
        println!("Ground truth written to {}", gt_file.display());

        if !states_out.is_empty() {
            let est_poses: Vec<(f64, echo_lie::SE3)> = states_out
                .iter()
                .map(|(t, s)| (*t, s.sensor.pose.clone()))
                .collect();
            let alignment = echo_li_core::alignment::align_trajectories(&est_poses, gt_poses);
            let aligned: Vec<(f64, VIOState)> = states_out
                .iter()
                .map(|(t, s)| {
                    let mut s_aligned = s.clone();
                    s_aligned.sensor.pose = alignment.compose(&s.sensor.pose);
                    (*t, s_aligned)
                })
                .collect();
            let aligned_file = output_dir.join("aligned_trajectory.txt");
            write_trajectory(&aligned_file, &aligned)?;
            println!("Aligned trajectory written to {}", aligned_file.display());

            if let Some(metrics) = echo_li_core::trajectory_metrics::compute_ate_metrics(
                &est_poses, gt_poses, &alignment,
            ) {
                println!(
                    "ATE position: rmse={:.4} m mean={:.4} m median={:.4} m max={:.4} m",
                    metrics.ate_position_m.rmse,
                    metrics.ate_position_m.mean,
                    metrics.ate_position_m.median,
                    metrics.ate_position_m.max
                );
                println!(
                    "ATE attitude: rmse={:.4} deg mean={:.4} deg median={:.4} deg max={:.4} deg",
                    metrics.ate_attitude_deg.rmse,
                    metrics.ate_attitude_deg.mean,
                    metrics.ate_attitude_deg.median,
                    metrics.ate_attitude_deg.max
                );
                let metrics_file = output_dir.join("trajectory_metrics.txt");
                write_trajectory_metrics(&metrics_file, &metrics)?;
                println!("Trajectory metrics written to {}", metrics_file.display());
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // RUST_LOG controls core telemetry (e.g. RUST_LOG=echo_li_core=debug for the
    // landmark-init dump); default shows warnings and up.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
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
    let mut raw_imu_it = reader.imu_iter();
    let mut image_it = reader.image_iter().peekable();

    let settings = if let Some(conf) = &vio_config {
        conf.to_filter_settings()
    } else {
        let mut s = VIOFilterSettings::default();
        s.coordinate_choice = args.coord.clone();
        s
    };

    println!(
        "Filter: chart={}, imu_bias_group={}, max_landmarks={}",
        settings.coordinate_choice,
        settings.imu_bias_group.as_str(),
        settings.max_landmarks
    );

    let (cam_model, k_matrix, img_w, img_h) = build_camera_model(&reader);

    let occupancy_requested = args.occupancy_map
        || vio_config
            .as_ref()
            .and_then(|conf| conf.local_occupancy.as_ref())
            .map(|conf| conf.enabled)
            .unwrap_or(false);
    let patch_depth_enabled = !args.no_patch_depth
        && (args.patch_depth
            || occupancy_requested
            || vio_config
                .as_ref()
                .and_then(|conf| conf.patch_depth.as_ref())
                .is_some());
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
    #[cfg(feature = "rerun")]
    let (patch_depth_cov_vis_min, patch_depth_cov_vis_max) = vio_config
        .as_ref()
        .and_then(|conf| conf.patch_depth.as_ref())
        .map(|conf| {
            (
                conf.cov_vis_min.unwrap_or(0.0),
                conf.cov_vis_max.unwrap_or(0.5),
            )
        })
        .unwrap_or((0.0, 0.5));
    #[cfg(feature = "rerun")]
    let (sparse_vis_min_depth, sparse_vis_max_depth) = vio_config
        .as_ref()
        .and_then(|conf| conf.sparse_vog.as_ref())
        .map(|conf| {
            (
                conf.vis_min_depth.unwrap_or(0.1),
                conf.vis_max_depth.unwrap_or(5.0),
            )
        })
        .unwrap_or((0.1, 5.0));

    let frontend_cam = reader
        .intrinsics
        .as_ref()
        .map(|intr| RudolfCameraIntrinsics {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            resolution: [intr.width, intr.height],
            distortion: intr
                .distortion_coefficients
                .as_deref()
                .unwrap_or_default()
                .to_vec(),
        });
    let (mut frontend, tracker_max_features) =
        build_frontend(vio_config.as_ref(), img_w, img_h, frontend_cam)?;

    let mut stereo_matcher =
        build_stereo_matcher(&args, vio_config.as_ref(), &reader, img_w, img_h);
    let mut cam1_image_it = if stereo_matcher.is_some() {
        Some(reader.cam1_image_iter().peekable())
    } else {
        None
    };

    // Initialize Rerun visualization
    #[cfg(feature = "rerun")]
    let rec: Option<rerun::RecordingStream> = if args.vis {
        // Three sinks, by env:
        //   ECHO_LI_RERUN_URL=rerun+http://<host>:9876/proxy  -> stream to an
        //     already-running viewer (e.g. native Rerun on Windows = hardware GPU);
        //   ECHO_LI_RRD=<path>  -> record to a file (headless);
        //   otherwise spawn a local (WSL software) viewer.
        let builder = rerun::RecordingStreamBuilder::new("echo-li");
        let built = if let Ok(url) = std::env::var("ECHO_LI_RERUN_URL") {
            println!("Rerun: connecting to viewer at {url}");
            builder.connect_grpc_opts(url)
        } else if let Ok(path) = std::env::var("ECHO_LI_RRD") {
            builder.save(&path)
        } else {
            builder.spawn()
        };
        match built {
            Ok(r) => {
                println!("Rerun stream ready");
                send_rerun_blueprint(&r, &k_matrix, img_w, img_h).ok();

                Some(r)
            }
            Err(e) => {
                eprintln!("Warning: could not start Rerun stream: {e}");
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

    // ------------------------------------------------------------------
    // Pose initialization — mirrors run_euroc.py / initialization.py.
    // Eager-init: build filter once with gravity-aligned xi0, then process
    // every event (IMU + image) through it in time order. No batch IMU
    // replay; no dropping of early images.
    // ------------------------------------------------------------------
    let initial_imu: Vec<IMUVelocity> = (&mut raw_imu_it).take(100).collect();
    // Gravity-align from the FIRST accel reading, unconditionally. The old
    // stationarity gate fell back to identity attitude when the start was in
    // motion, which diverges immediately (gravity integrates as phantom
    // acceleration; EuRoC MH_01/V2_03). Even an in-flight accel sample is
    // within ~10-15 deg of gravity — inside the filter's initial attitude
    // sigma — whereas identity can be 180 deg off. Moving starts additionally
    // need a loose eqf initialVariance.velocity (the old 9e-8 asserts v=0).
    let stationary = check_stationary(&initial_imu, 100, 0.1, 0.5);
    let init_pose = estimate_initial_pose(&initial_imu, 1);
    {
        let r = init_pose.rotation.as_matrix();
        let pitch_deg = (-r[(2, 0)]).clamp(-1.0, 1.0).asin().to_degrees();
        let roll_deg = r[(2, 1)].atan2(r[(2, 2)]).to_degrees();
        let t0 = initial_imu.first().map(|imu| imu.stamp).unwrap_or(0.0);
        println!(
            "IMU gravity-align at t={:.3} (first sample, stationary={}): roll={:.1}° pitch={:.1}°",
            t0, stationary, roll_deg, pitch_deg
        );
    }

    let mut xi0 = VIOState::new(VIOSensorState::identity(), Vec::new());
    xi0.sensor.pose = init_pose;
    if let Some(ext) = &reader.camera_extrinsics {
        xi0.sensor.camera_offset = ext.clone();
    }
    let mut filter: Option<VIOFilter> = Some(VIOFilter::new(settings.clone(), xi0));

    // Re-attach the buffered IMU samples to the front of the stream so they
    // pass through the filter at their original timestamps, interleaved with
    // images, just like every other IMU sample.
    let mut imu_it = initial_imu.into_iter().chain(raw_imu_it).peekable();

    let mut patch_depth_mapper = build_patch_depth_mapper(
        patch_depth_enabled,
        patch_depth_settings,
        Arc::clone(&cam_model),
        k_matrix,
        img_w,
        img_h,
        stereo_matcher.as_ref(),
    )?;
    let mut local_occupancy =
        build_local_occupancy_map(&args, vio_config.as_ref(), patch_depth_mapper.is_some())?;
    let occupancy_intrinsics = CameraIntrinsics::from_matrix(&k_matrix);
    #[cfg(feature = "rerun")]
    if patch_depth_mapper.is_some() {
        println!(
            "Patch depth visualization: fixed depth range [{:.2}, {:.2}] m, \
             covariance std-dev range [{:.2}, {:.2}] m",
            patch_depth_vis_min_depth,
            patch_depth_vis_max_depth,
            patch_depth_cov_vis_min,
            patch_depth_cov_vis_max
        );
    }
    let mut sparse_filter =
        build_sparse_filter(&args, vio_config.as_ref(), k_matrix, tracker_max_features);
    let mut states_out: Vec<(f64, VIOState)> = Vec::new();
    let mut imu_count: usize = 0;
    let mut vision_count: usize = 0;
    let mut prev_stereo_3d: HashMap<u64, [f64; 3]> = HashMap::new();
    let stereo_ransac_cfg = Rigid3dRansacConfig::default();
    let mut last_patch_depth_counts: Option<(usize, usize, usize, usize)> = None;
    let mut last_occupancy_counts: Option<(usize, usize, usize)> = None;
    #[cfg(feature = "rerun")]
    let mut last_patch_depth_output: Option<PatchDepthOutput> = None;
    let trace_determinism = std::env::var_os("ECHO_LI_TRACE_DETERMINISM").is_some();
    #[cfg(feature = "rerun")]
    let mut patch_depth_vis_announced = false;
    let t_start = std::time::Instant::now();

    #[cfg(feature = "rerun")]
    let mut trajectory_vis: Vec<nalgebra::Vector3<f64>> = Vec::new();
    // Ground-truth trajectory + its rolling alignment to the estimate (Rerun).
    #[cfg(feature = "rerun")]
    let gt_poses_vis = reader.groundtruth();
    #[cfg(feature = "rerun")]
    let mut gt_align: Option<echo_lie::SE3> = None;
    #[cfg(feature = "rerun")]
    let vis_cfg = vio_config
        .as_ref()
        .map(|c| c.rerun.clone())
        .unwrap_or_default();

    println!("\nRunning filter...");

    // Previous frame's post-update camera pose, for the epipolar-gate prior.
    let use_pose_prior = vio_config
        .as_ref()
        .is_some_and(|c| c.rudolf_v.epipolar_gate_threshold > 0.0);
    let mut prev_cam_pose: Option<Matrix4<f64>> = None;

    loop {
        if let Some(next_img) = image_it.peek() {
            while let Some(imu) = imu_it.peek() {
                if imu.stamp > next_img.stamp {
                    break;
                }
                let imu = imu_it.next().unwrap();
                imu_count += 1;
                if let Some(f) = &mut filter {
                    f.process_imu(imu);
                }
            }
        }

        if let Some(img_data) = image_it.next() {
            if let Ok(dynamic_img) = image::open(&img_data.image_path) {
                let gray_img = dynamic_img.to_luma8();
                let gray_data = gray_img.into_raw();

                // Clone raw pixels for Rerun before consuming into Rudolf-V
                // (fallback when the preprocessed/histeq image is unavailable).
                #[cfg(feature = "rerun")]
                let rerun_gray_data = if rec.is_some() && vis_cfg.image {
                    gray_data.clone()
                } else {
                    vec![]
                };

                let rudolf_img = RudolfImage::from_vec(img_w, img_h, gray_data);

                // Epipolar-gate prior: relative camera motion prev -> curr from
                // the IMU-propagated (pre-vision) EqF prediction.
                if use_pose_prior {
                    if let (Some(f), Some(prev)) = (&filter, &prev_cam_pose) {
                        let pred = camera_pose_matrix(&f.eqf.state_estimate());
                        if let Some(pred_inv) = pred.try_inverse() {
                            let t_rel = pred_inv * prev;
                            let mut r = [[0.0f64; 3]; 3];
                            for (i, row) in r.iter_mut().enumerate() {
                                for (j, v) in row.iter_mut().enumerate() {
                                    *v = t_rel[(i, j)];
                                }
                            }
                            frontend
                                .set_pose_prior(r, [t_rel[(0, 3)], t_rel[(1, 3)], t_rel[(2, 3)]]);
                        }
                    }
                }

                let (features_ref, stats) = frontend.process(&rudolf_img);
                let features: Vec<rudolf_v::fast::Feature> = features_ref.to_vec();
                let feature_hash = trace_determinism.then(|| hash_features(&features));

                // --- Stereo matching + deferred RANSAC validation ---
                let mut stereo_depth_priors: HashMap<u64, LandmarkDepthPrior> = HashMap::new();
                let mut cam1_gray_for_patch: Option<Vec<u8>> = None;
                if let Some(matcher) = &mut stereo_matcher {
                    let cam1_loaded = if let Some(cam1_it) = &mut cam1_image_it {
                        while let Some(next_cam1) = cam1_it.peek() {
                            if next_cam1.stamp >= img_data.stamp {
                                break;
                            }
                            cam1_it.next();
                        }
                        cam1_it.next().and_then(|cam1_data| {
                            image::open(&cam1_data.image_path).ok().map(|dyn_img| {
                                RudolfImage::from_vec(img_w, img_h, dyn_img.to_luma8().into_raw())
                            })
                        })
                    } else {
                        None
                    };

                    if let Some(cam1_img) = cam1_loaded {
                        if patch_depth_mapper
                            .as_ref()
                            .map_or(false, |m| m.has_stereo_ref())
                        {
                            let cam1_eq = rudolf_v::histeq::equalize_histogram(&cam1_img);
                            cam1_gray_for_patch =
                                Some(cam1_eq.as_slice()[..img_w * img_h].to_vec());
                        }
                        let cam0_pyramid = frontend.current_pyramid();
                        let matches = matcher.match_features(&cam1_img, &features, cam0_pyramid);
                        let rig = matcher.rig();

                        let mut curr_3d: HashMap<u64, [f64; 3]> = HashMap::new();
                        for (feat, m) in features.iter().zip(matches.iter()) {
                            if let Some(p) = m.point_cam0(rig, feat) {
                                curr_3d.insert(feat.id, p);
                            }
                        }

                        // 3D-3D RANSAC against previous frame's stereo points.
                        let mut corrs_with_id: Vec<(u64, Correspondence3d)> = Vec::new();
                        let mut curr_3d_ids: Vec<u64> = curr_3d.keys().copied().collect();
                        curr_3d_ids.sort_unstable();
                        for id in curr_3d_ids {
                            let p_curr = curr_3d[&id];
                            if let Some(&p_prev) = prev_stereo_3d.get(&id) {
                                corrs_with_id.push((
                                    id,
                                    Correspondence3d {
                                        p1: p_prev,
                                        p2: p_curr,
                                    },
                                ));
                            }
                        }
                        let mut outlier_ids: Vec<u64> = Vec::new();
                        if corrs_with_id.len() >= 3 {
                            let corrs_only: Vec<Correspondence3d> =
                                corrs_with_id.iter().map(|(_, c)| *c).collect();
                            if let Some(result) = rudolf_v::rigid_ransac::estimate_rigid_ransac(
                                &corrs_only,
                                &stereo_ransac_cfg,
                            ) {
                                for ((id, _), &is_in) in
                                    corrs_with_id.iter().zip(result.inliers.iter())
                                {
                                    if is_in {
                                        // RANSAC-validated: promote to depth prior.
                                        let p = curr_3d[id];
                                        let range =
                                            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                                        if range > 0.0 {
                                            let baseline = rig.baseline_meters();
                                            let m_res = matches
                                                .iter()
                                                .find(|m| m.id == *id)
                                                .map(|m| m.residual)
                                                .unwrap_or(10.0);
                                            let range_var =
                                                (m_res as f64 / 20.0).powi(2) * range * range
                                                    / (baseline * baseline);
                                            stereo_depth_priors.insert(
                                                *id,
                                                LandmarkDepthPrior {
                                                    range,
                                                    range_var: range_var.max(1e-4),
                                                },
                                            );
                                        }
                                    } else {
                                        outlier_ids.push(*id);
                                    }
                                }
                            }
                        }
                        if !outlier_ids.is_empty() {
                            frontend.drop_tracks(&outlier_ids);
                            for id in &outlier_ids {
                                curr_3d.remove(id);
                            }
                        }
                        prev_stereo_3d = curr_3d;
                    }
                }

                if let Some(f) = &mut filter {
                    #[cfg(feature = "rerun")]
                    let tracker_points: Vec<(f32, f32)> = features
                        .iter()
                        .filter_map(|feat| {
                            clip_image_point(feat.x as f64, feat.y as f64, img_w, img_h)
                        })
                        .collect();

                    let mut feat_uvs = HashMap::new();
                    for feat in &features {
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
                    let (mut depth_priors, defer_fallback_ids) =
                        if let Some(sparse) = &sparse_filter {
                            let mut priors = HashMap::new();
                            let mut deferred = HashSet::new();
                            for &fid in measurement.cam_coordinates.keys() {
                                let (range, range_var) = sparse.query_range(fid);
                                if range > 0.0 && range_var.is_finite() {
                                    priors.insert(fid, LandmarkDepthPrior { range, range_var });
                                } else if sparse.has_track(fid) {
                                    deferred.insert(fid);
                                }
                            }
                            (priors, deferred)
                        } else {
                            (HashMap::new(), HashSet::new())
                        };
                    // Stereo priors override SparseVog (known baseline → tighter variance).
                    depth_priors.extend(stereo_depth_priors.iter().map(|(&id, p)| (id, *p)));
                    let depth_prior_hash =
                        trace_determinism.then(|| hash_depth_priors(&depth_priors));
                    f.process_vision_with_depth_priors_and_deferred_fallbacks(
                        measurement.clone(),
                        cam_model.as_ref(),
                        &depth_priors,
                        &defer_fallback_ids,
                    );
                    vision_count += 1;

                    let state = f.eqf.state_estimate();
                    let state_hash = trace_determinism.then(|| hash_state(&state));
                    let t_wc = camera_pose_matrix(&state);
                    prev_cam_pose = Some(t_wc);
                    if let Some(sparse) = &mut sparse_filter {
                        sparse.update(&sparse_measurement, &t_wc, None, None);
                        if let Some(mapper) = &mut patch_depth_mapper {
                            if !patch_gray_data.is_empty() {
                                let frame = FrameProducts {
                                    frame_id: vision_count as u64,
                                    stamp: img_data.stamp,
                                    gray: patch_gray_data.clone(),
                                    width: img_w,
                                    height: img_h,
                                    pose_t_wc: t_wc,
                                };
                                // RawDistorted/PerPatchBearing rectify the raw image,
                                // so seeds stay in raw distorted coords; the pinhole-
                                // based modes consume undistorted-pinhole coords.
                                let (patch_measurement, patch_seed_coordinates) =
                                    match mapper.camera_mode() {
                                        PatchDepthCameraMode::RawDistorted
                                        | PatchDepthCameraMode::PerPatchBearing => {
                                            (&measurement, PatchDepthSeedCoordinates::RawDistorted)
                                        }
                                        PatchDepthCameraMode::UndistortedPinhole
                                        | PatchDepthCameraMode::TiledBearing => (
                                            &sparse_measurement,
                                            PatchDepthSeedCoordinates::UndistortedPinhole,
                                        ),
                                    };
                                let patch_output: Option<PatchDepthOutput> =
                                    if let Some(cam1_gray) = &cam1_gray_for_patch {
                                        mapper.update_with_stereo_ref(
                                            sparse,
                                            patch_measurement,
                                            patch_seed_coordinates,
                                            frame,
                                            cam1_gray,
                                            img_w,
                                            img_h,
                                        )
                                    } else {
                                        mapper.update(
                                            sparse,
                                            patch_measurement,
                                            patch_seed_coordinates,
                                            frame,
                                        )
                                    };
                                last_patch_depth_counts =
                                    patch_output.as_ref().map(patch_depth_status_counts);
                                if let (Some(output), Some(occupancy)) =
                                    (patch_output.as_ref(), local_occupancy.as_mut())
                                {
                                    occupancy.update_from_patch_depth(
                                        output,
                                        cam_model.as_ref(),
                                        occupancy_intrinsics,
                                        patch_seed_coordinates,
                                        img_w,
                                        img_h,
                                        &t_wc,
                                    );
                                    last_occupancy_counts = Some(occupancy.counts());
                                }
                                #[cfg(feature = "rerun")]
                                {
                                    last_patch_depth_output = patch_output;
                                }
                            }
                        }
                    }
                    if trace_determinism {
                        eprintln!(
                            "det frame={} stamp={:.9} features={} feature_hash={:016x} tracked={} lost={} rejected={} new={} occupied={}/{} priors={} prior_hash={:016x} sparse_hash={:016x} eqf_landmarks={} state_hash={:016x}",
                            vision_count,
                            img_data.stamp,
                            measurement.cam_coordinates.len(),
                            feature_hash.unwrap_or(0),
                            stats.tracked,
                            stats.lost,
                            stats.rejected,
                            stats.new_detections,
                            stats.occupied_cells,
                            stats.total_cells,
                            depth_priors.len(),
                            depth_prior_hash.unwrap_or(0),
                            hash_sparse_filter(sparse_filter.as_ref()),
                            state.camera_landmarks.len(),
                            state_hash.unwrap_or(0),
                        );
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

                        // Camera image (grayscale) — the tracker's preprocessed
                        // (histeq'd) frame when enabled/available, else raw.
                        if vis_cfg.image {
                            let l8: Vec<u8> = match frontend
                                .preprocessed_image()
                                .filter(|_| vis_cfg.histeq_image)
                            {
                                Some(img) => {
                                    let (w, h, stride) = (img.width(), img.height(), img.stride());
                                    let src = img.as_slice();
                                    if stride == w {
                                        src[..w * h].to_vec()
                                    } else {
                                        let mut out = Vec::with_capacity(w * h);
                                        for y in 0..h {
                                            out.extend_from_slice(&src[y * stride..y * stride + w]);
                                        }
                                        out
                                    }
                                }
                                None => rerun_gray_data,
                            };
                            rec.log(
                                "camera/image",
                                &rerun::Image::from_l8(l8, [img_w as u32, img_h as u32]),
                            )
                            .ok();
                        }

                        if let Some(output) = &last_patch_depth_output {
                            if vis_cfg.patch_depth {
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
                            }
                            if vis_cfg.patch_depth_cov {
                                let cov_rgb = patch_depth_cov_rgb_for_vis(
                                    output,
                                    img_w,
                                    img_h,
                                    patch_depth_cov_vis_min,
                                    patch_depth_cov_vis_max,
                                );
                                rec.log(
                                    "patch_depth_cov/image",
                                    &rerun::Image::from_rgb24(
                                        cov_rgb,
                                        [img_w as u32, img_h as u32],
                                    ),
                                )
                                .ok();
                            }
                            if !patch_depth_vis_announced {
                                println!(
                                    "Patch depth Rerun entities: patch_depth/image, \
                                     patch_depth_cov/image"
                                );
                                patch_depth_vis_announced = true;
                            }
                        }

                        // Tracked features on image
                        if vis_cfg.features && !tracker_points.is_empty() {
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
                                sparse_vis_min_depth,
                                sparse_vis_max_depth,
                            )
                        } else {
                            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                        };
                        if vis_cfg.sparse_image && !sparse_img_pts.is_empty() {
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
                        if vis_cfg.trajectory && trajectory_vis.len() >= 2 {
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

                        // Ground truth, aligned to the estimate in real time —
                        // like eqvio's visualiser.py: a Umeyama fit over the
                        // matched estimate↔GT poses so far, recomputed every
                        // few frames once enough have accumulated. GT is shown
                        // in the estimate frame (inverse transform), subsampled
                        // and time-masked to "now" so it grows with the run.
                        let now = states_out.last().map(|(t, _)| *t).unwrap_or(0.0);
                        if states_out.len() >= 100
                            && states_out.len() % 5 == 0
                            && !gt_poses_vis.is_empty()
                        {
                            let est_poses: Vec<(f64, echo_lie::SE3)> = states_out
                                .iter()
                                .map(|(t, s)| (*t, s.sensor.pose.clone()))
                                .collect();
                            gt_align = Some(echo_li_core::alignment::align_trajectories(
                                &est_poses,
                                &gt_poses_vis,
                            ));
                        }
                        if vis_cfg.groundtruth
                            && let Some(align) = &gt_align
                        {
                            let inv = align.inverse();
                            let gt_strip: Vec<[f32; 3]> = gt_poses_vis
                                .iter()
                                .filter(|sp| sp.stamp <= now)
                                .step_by(10)
                                .map(|sp| {
                                    let p = inv.compose(&sp.pose).translation;
                                    [p[0] as f32, p[1] as f32, p[2] as f32]
                                })
                                .collect();
                            if gt_strip.len() >= 2 {
                                rec.log(
                                    "world/groundtruth",
                                    &rerun::LineStrips3D::new([gt_strip])
                                        .with_colors([0xFFFFFFFFu32])
                                        .with_radii([0.02f32]),
                                )
                                .ok();
                            }
                        }

                        // 3D landmarks
                        let lm_pts: Vec<(f32, f32, f32)> = feat_global
                            .values()
                            .map(|p| (p[0] as f32, p[1] as f32, p[2] as f32))
                            .collect();
                        if vis_cfg.landmarks && !lm_pts.is_empty() {
                            rec.log(
                                "world/landmarks",
                                &rerun::Points3D::new(lm_pts)
                                    .with_colors([0x00FF00FFu32])
                                    .with_radii([0.02f32]),
                            )
                            .ok();
                        }

                        if vis_cfg.sparse_world && !sparse_world_pts.is_empty() {
                            rec.log(
                                "world/sparse_out_of_state",
                                &rerun::Points3D::new(sparse_world_pts)
                                    .with_colors(sparse_world_colors)
                                    .with_radii([0.015f32]),
                            )
                            .ok();
                        }

                        // Local 3D occupancy map, both as solid voxel cubes:
                        // occupied = opaque red, free = translucent green. NB: use
                        // 2-level entity paths (siblings of world/landmarks) and
                        // FillMode::Solid — 3-level paths don't render in the 3D
                        // view, and the default Boxes3D fill is invisible wireframe.
                        // Unlike Points3D (which ignore colour alpha), solid Boxes3D
                        // DO blend: a colour with alpha < 0xFF becomes the mesh
                        // additive_tint and is routed to rerun's premultiplied-alpha
                        // transparent pass — so the free voxels are genuinely
                        // see-through (lower the alpha byte for more transparency).
                        if let Some(occ_map) = local_occupancy.as_ref() {
                            let hx = (occ_map.settings().resolution * 0.5) as f32;
                            let (occ_pts, free_pts) = occupancy_cells_for_vis(occ_map);
                            if vis_cfg.occupied_cells && !occ_pts.is_empty() {
                                let occ_half = vec![[hx, hx, hx]; occ_pts.len()];
                                rec.log(
                                    "world/occupied_cells",
                                    &rerun::Boxes3D::from_centers_and_half_sizes(occ_pts, occ_half)
                                        .with_fill_mode(rerun::FillMode::Solid)
                                        .with_colors([0xFF0000FFu32]),
                                )
                                .ok();
                            }
                            if vis_cfg.free_cells && !free_pts.is_empty() {
                                let free_half = vec![[hx, hx, hx]; free_pts.len()];
                                rec.log(
                                    "world/free_cells",
                                    &rerun::Boxes3D::from_centers_and_half_sizes(
                                        free_pts, free_half,
                                    )
                                    .with_fill_mode(rerun::FillMode::Solid)
                                    .with_colors([0x33C03301u32]),
                                )
                                .ok();
                            }
                        }

                        // Build the third-person pose from the estimated
                        // world-from-camera pose. Mat3x3 values are columns.
                        let camera_origin = [p_cam[0] as f32, p_cam[1] as f32, p_cam[2] as f32];
                        let camera_rotation = r_cam.as_matrix();
                        let camera_rotation_cols = [
                            [
                                camera_rotation[(0, 0)] as f32,
                                camera_rotation[(1, 0)] as f32,
                                camera_rotation[(2, 0)] as f32,
                            ],
                            [
                                camera_rotation[(0, 1)] as f32,
                                camera_rotation[(1, 1)] as f32,
                                camera_rotation[(2, 1)] as f32,
                            ],
                            [
                                camera_rotation[(0, 2)] as f32,
                                camera_rotation[(1, 2)] as f32,
                                camera_rotation[(2, 2)] as f32,
                            ],
                        ];
                        // Camera coordinates are RDF, so -Y is above and -Z is
                        // behind. Keep the chase eye above and behind the camera.
                        const VIEW_ABOVE_M: f32 = 0.5;
                        const VIEW_BEHIND_M: f32 = 2.0;
                        let view_origin = std::array::from_fn(|i| {
                            camera_origin[i]
                                - VIEW_ABOVE_M * camera_rotation_cols[1][i]
                                - VIEW_BEHIND_M * camera_rotation_cols[2][i]
                        });
                        rec.log(
                            "world/view_camera",
                            &rerun::Transform3D::from_translation_mat3x3(
                                view_origin,
                                camera_rotation_cols,
                            ),
                        )
                        .ok();

                        // Camera pose as RGB arrows (X=red, Y=green, Z=blue)
                        if vis_cfg.camera_axes {
                            let scale = 0.1f32;
                            rec.log(
                                "world/camera_axes",
                                &rerun::Arrows3D::from_vectors([
                                    [
                                        camera_rotation[(0, 0)] as f32 * scale,
                                        camera_rotation[(1, 0)] as f32 * scale,
                                        camera_rotation[(2, 0)] as f32 * scale,
                                    ],
                                    [
                                        camera_rotation[(0, 1)] as f32 * scale,
                                        camera_rotation[(1, 1)] as f32 * scale,
                                        camera_rotation[(2, 1)] as f32 * scale,
                                    ],
                                    [
                                        camera_rotation[(0, 2)] as f32 * scale,
                                        camera_rotation[(1, 2)] as f32 * scale,
                                        camera_rotation[(2, 2)] as f32 * scale,
                                    ],
                                ])
                                .with_origins([camera_origin, camera_origin, camera_origin])
                                .with_colors([
                                    0xFF0000FFu32,
                                    0x00FF00FFu32,
                                    0x0000FFFFu32,
                                ]),
                            )
                            .ok();
                        }
                    }

                    if vision_count % 100 == 0 || vision_count <= 5 {
                        let pos = states_out.last().unwrap().1.sensor.pose.translation;
                        let vel = states_out.last().unwrap().1.sensor.velocity;
                        let occ_suffix = last_occupancy_counts
                            .map(|(unk, free, occ)| {
                                format!("  occ=(occ:{} free:{} unk:{})", occ, free, unk)
                            })
                            .unwrap_or_default();
                        if let Some((unk, seed, photo, rej)) = last_patch_depth_counts {
                            println!(
                                "  [{:4}] t={:.3}  pos=({:+.2}, {:+.2}, {:+.2})  vel=({:+.3}, {:+.3}, {:+.3})  lm={}  patch=(photo:{} seed:{} unk:{} rej:{}){}",
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
                                rej,
                                occ_suffix
                            );
                        } else {
                            println!(
                                "  [{:4}] t={:.3}  pos=({:+.2}, {:+.2}, {:+.2})  vel=({:+.3}, {:+.3}, {:+.3})  lm={}{}",
                                vision_count,
                                img_data.stamp,
                                pos[0],
                                pos[1],
                                pos[2],
                                vel[0],
                                vel[1],
                                vel[2],
                                feat_global.len(),
                                occ_suffix
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
    if let Some(occupancy) = &local_occupancy {
        let (unknown, free, occupied) = occupancy.counts();
        println!(
            "Local occupancy final: occupied={} free={} unknown={}",
            occupied, free, unknown
        );
    }

    let dataset_name = PathBuf::from(&args.dataset)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let output_dir = PathBuf::from(
        args.output
            .unwrap_or_else(|| format!("eqvio_output_{}", dataset_name)),
    );
    let gt_poses = reader.groundtruth();
    write_outputs(&output_dir, &states_out, &gt_poses)?;

    println!("Done.");
    Ok(())
}
