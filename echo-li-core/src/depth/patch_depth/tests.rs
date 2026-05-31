use std::sync::Arc;

use nalgebra::{Matrix4, Vector2};

use super::structure_tensor_passes;
use super::*;
use crate::core_types::CameraIntrinsics;
use crate::depth::sparse_3d::{Sparse3DChart, Sparse3DFilter};
use crate::depth::sparse_gb::SparseVogSettings;
use crate::mathematical::camera::{CameraModel, PinholeModel};
use crate::mathematical::vision_measurement::VisionMeasurement;

fn camera() -> (Arc<dyn CameraModel>, CameraIntrinsics) {
    let intr = CameraIntrinsics::new(40.0, 40.0, 16.0, 16.0);
    (
        Arc::new(PinholeModel {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
        }),
        intr,
    )
}

fn frame(id: u64, tx: f64, gray: Vec<u8>) -> FrameProducts {
    let mut pose = Matrix4::identity();
    pose[(0, 3)] = tx;
    FrameProducts {
        frame_id: id,
        stamp: id as f64,
        gray,
        width: 32,
        height: 32,
        pose_t_wc: pose,
    }
}

fn textured_image() -> Vec<u8> {
    let mut data = vec![0u8; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            data[y * 32 + x] = ((x * 5 + y * 3) % 255) as u8;
        }
    }
    data
}

#[test]
fn keyframe_buffer_uses_baseline_gate() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings::default();
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];

    let img = textured_image();
    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    assert_eq!(mapper.keyframe_count(), 1);
    assert!(mapper
        .update_with_priors(frame(1, 0.001, img.clone()), &seeds, None, 0.0)
        .is_none());
    assert_eq!(mapper.keyframe_count(), 2);
    assert!(mapper
        .update_with_priors(frame(2, 0.02, img), &seeds, None, 0.0)
        .is_some());
    assert_eq!(mapper.keyframe_count(), 2);
}

#[test]
fn seed_priors_produce_cell_depths() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        min_photo_curvature: 1e12,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = vec![120u8; 32 * 32];

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out
        .depth
        .data
        .iter()
        .any(|z| z.is_finite() && (*z - 2.0).abs() < 0.1));
    assert!(out.status.data.contains(&PatchStatus::SeedOnly));
}

#[test]
fn bearing_lut_handles_distorted_path_for_photometric_solve() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn fast_translation_refines_undistorted_pinhole_patches() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::UndistortedPinhole,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn fast_translation_refines_undistorted_pinhole_4x4_patches() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::UndistortedPinhole,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn raw_bearing_lut_interpolates_fractional_pixels() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings::default();
    let mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();

    let bearing = mapper.bearing_at_original(16.5, 16.25).unwrap();
    assert!((bearing[0] - 0.5 / 40.0).abs() < 1e-12);
    assert!((bearing[1] - 0.25 / 40.0).abs() < 1e-12);
    assert!((bearing[2] - 1.0).abs() < 1e-12);
}

#[test]
fn tiled_bearing_mode_runs_tile_local_update() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.depth.data.iter().any(|z| z.is_finite() && *z > 0.0));
    assert!(out.status.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn tiled_bearing_fusion_converts_range_to_z_depth() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        ..PatchDepthSettings::default()
    };
    let mapper = PatchDepthMapper::new(Arc::clone(&camera), intr, 32, 32, settings).unwrap();
    let layout = mapper.tiled_bearing_levels.as_ref().unwrap();
    let level = &layout[0];
    let tile_idx = level.owning_tile_for_patch(28.0, 28.0, 2).unwrap();
    let tile = &level.tiles[tile_idx];
    let local = tile.global_to_local(Vector2::new(28.0, 28.0));
    let estimate = PatchEstimate::seed_only(0.5, 0.01, &mapper.settings);
    let mut rho_acc = vec![0.0f32; 32 * 32];
    let mut w_acc = vec![0.0f32; 32 * 32];
    let mut status = vec![PatchStatus::Unknown; 32 * 32];

    mapper.accumulate_tiled_patch(
        &mut rho_acc,
        &mut w_acc,
        &mut status,
        32,
        32,
        tile,
        local[0],
        local[1],
        estimate,
    );

    let idx = 28 * 32 + 28;
    let z_depth = w_acc[idx] / rho_acc[idx];
    let range = estimate.eta.exp() as f32;
    let bearing_z = tile.bearing_at_level_pixel(28.0, 28.0)[2] as f32;
    assert!((z_depth - range * bearing_z).abs() < 1e-5);
    assert!(z_depth < range);
}

#[test]
fn tiled_bearing_fusion_feathers_internal_tile_edges() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        ..PatchDepthSettings::default()
    };
    let mapper = PatchDepthMapper::new(Arc::clone(&camera), intr, 32, 32, settings).unwrap();
    let layout = mapper.tiled_bearing_levels.as_ref().unwrap();
    let tile = layout[0]
        .tiles
        .iter()
        .find(|tile| tile.x0 > 0 && tile.y0 > 0 && tile.x0 + tile.width < 32)
        .expect("test layout should contain an internal tile");

    let center_w = mapper.tiled_blend_weight(
        tile,
        0.5 * (tile.width - 1) as f64,
        0.5 * (tile.height - 1) as f64,
        32,
        32,
    );
    let edge_w = mapper.tiled_blend_weight(tile, 0.0, 0.5 * (tile.height - 1) as f64, 32, 32);

    assert!(center_w > 0.9);
    assert_eq!(edge_w, 0.0);
}

#[test]
fn tiled_bearing_requires_fast_translation_warp() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::Exact,
        ..PatchDepthSettings::default()
    };
    let err = match PatchDepthMapper::new(camera, intr, 32, 32, settings) {
        Ok(_) => panic!("tiled_bearing should reject exact warp mode"),
        Err(err) => err,
    };
    assert!(err
        .to_string()
        .contains("requires warp_mode=fast_translation"));
}

#[test]
fn tiled_bearing_mode_runs_with_two_pyramid_levels() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        n_pyramid_levels: 2,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.depth.data.iter().any(|z| z.is_finite() && *z > 0.0));
    assert!(out.status.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn tiled_bearing_stereo_ref_initializes_tile_layout() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        n_pyramid_levels: 2,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(Arc::clone(&camera), intr, 32, 32, settings).unwrap();
    assert!(!mapper.has_stereo_ref());

    mapper.init_stereo_ref(camera.as_ref(), Matrix4::identity());

    assert!(mapper.has_stereo_ref());
    assert!(mapper.stereo_undistort_luts.is_none());
    let levels = mapper.stereo_tiled_bearing_levels.as_ref().unwrap();
    assert_eq!(levels.len(), 2);
    assert!(!levels[0].tiles.is_empty());
    assert!(!levels[1].tiles.is_empty());
}

#[test]
fn tiled_bearing_stereo_ref_update_uses_tile_path() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::TiledBearing,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        tiled_tile_size: 16,
        tiled_tile_overlap: 12,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(Arc::clone(&camera), intr, 32, 32, settings).unwrap();
    mapper.init_stereo_ref(camera.as_ref(), Matrix4::identity());

    let k = nalgebra::Matrix3::new(intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0);
    let sparse_filter = Sparse3DFilter::new(k, Sparse3DChart::Polar, SparseVogSettings::default());
    let measurement = VisionMeasurement::new(0.0, std::collections::HashMap::new());
    let img = textured_image();
    let out = mapper.update_with_stereo_ref(
        &sparse_filter,
        &measurement,
        PatchDepthSeedCoordinates::UndistortedPinhole,
        frame(0, 0.0, img.clone()),
        &img,
        32,
        32,
    );

    assert!(out.is_some());
}

#[test]
fn tiled_bearing_tiles_cover_level_and_keep_local_basis_orthonormal() {
    let (camera, intr) = camera();
    let levels = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].width, 32);
    assert_eq!(levels[0].height, 32);
    assert!(!levels[0].tiles.is_empty());

    let first = &levels[0].tiles[0];
    assert!(first.contains_patch(8.0, 8.0, 4));
    assert!((first.center_bearing.norm() - 1.0).abs() < 1e-12);
    assert!((first.tangent_u.norm() - 1.0).abs() < 1e-12);
    assert!((first.tangent_v.norm() - 1.0).abs() < 1e-12);
    assert!(first.center_bearing.dot(&first.tangent_u).abs() < 1e-12);
    assert!(first.center_bearing.dot(&first.tangent_v).abs() < 1e-12);
    assert!(first.tangent_u.dot(&first.tangent_v).abs() < 1e-12);

    let bottom_right = levels[0]
        .tiles
        .iter()
        .find(|tile| tile.x0 + tile.width == 32 && tile.y0 + tile.height == 32)
        .expect("tile layout should cover the bottom-right corner");
    assert!(bottom_right.contains_patch(28.0, 28.0, 2));
}

#[test]
fn tiled_bearing_tangent_axes_follow_image_axes() {
    let (camera, intr) = camera();
    let levels = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);

    for tile in &levels[0].tiles {
        let center = tile.bearing_at_level_pixel(tile.center_u, tile.center_v);
        let u_step = tile.bearing_at_level_pixel(tile.center_u + 1.0, tile.center_v) - center;
        let v_step = tile.bearing_at_level_pixel(tile.center_u, tile.center_v + 1.0) - center;

        assert!(
            u_step[0] > 0.0,
            "tile ({}, {}) local u axis should increase bearing x",
            tile.x0,
            tile.y0
        );
        assert!(
            v_step[1] > 0.0,
            "tile ({}, {}) local v axis should increase bearing y",
            tile.x0,
            tile.y0
        );
        assert!(
            u_step[0].abs() > u_step[1].abs(),
            "tile ({}, {}) local u axis should be mostly image x",
            tile.x0,
            tile.y0
        );
        assert!(
            v_step[1].abs() > v_step[0].abs(),
            "tile ({}, {}) local v axis should be mostly image y",
            tile.x0,
            tile.y0
        );
    }
}

#[test]
fn tiled_bearing_lut_remaps_from_full_level_into_tile_image() {
    let (camera, intr) = camera();
    let levels = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let tile = &levels[0].tiles[0];
    let raw = Image::from_vec(32, 32, (0..32 * 32).map(|i| (i % 251) as f32).collect());

    let remapped = tile.lut.undistort_level(&raw);
    assert_eq!(remapped.width(), tile.width);
    assert_eq!(remapped.height(), tile.height);
    assert_eq!(remapped.as_slice().len(), tile.width * tile.height);
    assert!(remapped.as_slice().iter().any(|v| *v > 0.0));
}

#[test]
fn tiled_bearing_local_projection_round_trips_chart_pixels() {
    let (camera, intr) = camera();
    let levels = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let tile = &levels[0].tiles[0];
    let rel_pose = RelativePose {
        r: nalgebra::Matrix3::identity(),
        t: nalgebra::Vector3::zeros(),
    };

    for (u, v) in [(4.0, 5.0), (8.0, 8.0), (13.0, 11.0)] {
        let bearing = tile.bearing_at_level_pixel(tile.x0 as f64 + u, tile.y0 as f64 + v);
        let projected = tile.project_to_local_pixel(&bearing).unwrap();
        assert!((projected[0] - u).abs() < 1e-10);
        assert!((projected[1] - v).abs() < 1e-10);

        let (u_ref, v_ref, _, _) = tile.warp_local_pixel(u, v, 0.5, &rel_pose).unwrap();
        assert!((u_ref - u).abs() < 1e-10);
        assert!((v_ref - v).abs() < 1e-10);
    }
}

#[test]
fn tiled_bearing_projection_jacobian_matches_finite_difference() {
    let (camera, intr) = camera();
    let levels = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let tile = &levels[0].tiles[0];
    let p =
        tile.bearing_at_level_pixel(9.0, 10.0) * 2.0 + nalgebra::Vector3::new(0.03, -0.02, 0.04);
    let analytic = tile.projection_jacobian(&p).unwrap();
    let eps = 1e-6;

    for axis in 0..3 {
        let mut dp = nalgebra::Vector3::zeros();
        dp[axis] = eps;
        let plus = tile.project_to_local_pixel(&(p + dp)).unwrap();
        let minus = tile.project_to_local_pixel(&(p - dp)).unwrap();
        let numeric = (plus - minus) / (2.0 * eps);
        assert!((analytic[(0, axis)] - numeric[0]).abs() < 1e-6);
        assert!((analytic[(1, axis)] - numeric[1]).abs() < 1e-6);
    }
}

#[test]
fn tiled_bearing_fast_translation_accumulates_scalar_jacobian() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::UndistortedPinhole,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        sigma_photo: 1.0,
        photo_huber_delta: 255.0,
        ..PatchDepthSettings::default()
    };
    let mapper = PatchDepthMapper::new(Arc::clone(&camera), intr, 32, 32, settings).unwrap();
    let layout = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let tile = &layout[0].tiles[0];
    let raw = Image::from_vec(
        32,
        32,
        (0..32 * 32)
            .map(|i| {
                let x = (i % 32) as f32;
                let y = (i / 32) as f32;
                x + 2.0 * y
            })
            .collect(),
    );
    let remapped = tile.lut.undistort_level(&raw);
    let valid = bilinear_valid_image_from_mask(&tile.lut.valid_image());
    let (grad_x, grad_y) = gradients(&remapped);

    let identity = RelativePose {
        r: nalgebra::Matrix3::identity(),
        t: nalgebra::Vector3::zeros(),
    };
    let (_, h_identity, _, n_identity) = mapper
        .patch_residual_jacobian_fast_translation_tiled_level(
            tile,
            tile,
            8.0,
            8.0,
            0.5,
            &remapped,
            Some(&valid),
            &remapped,
            Some(&valid),
            &grad_x,
            &grad_y,
            &identity,
            0.0,
        );
    assert_eq!(n_identity, 16);
    assert!(h_identity.abs() < 1e-12);

    let translated = RelativePose {
        r: nalgebra::Matrix3::identity(),
        t: nalgebra::Vector3::new(0.05, 0.0, 0.0),
    };
    let (_, h_translated, _, n_translated) = mapper
        .patch_residual_jacobian_fast_translation_tiled_level(
            tile,
            tile,
            8.0,
            8.0,
            0.5,
            &remapped,
            Some(&valid),
            &remapped,
            Some(&valid),
            &grad_x,
            &grad_y,
            &translated,
            0.0,
        );
    assert_eq!(n_translated, 16);
    assert!(h_translated > 0.0);
}

#[test]
fn tiled_bearing_frame_levels_remap_each_tile() {
    let (camera, intr) = camera();
    let layout = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let raw = Image::from_vec(32, 32, (0..32 * 32).map(|i| (i % 251) as f32).collect());

    let levels = build_tiled_bearing_frame_levels(&layout, &[raw]);
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].tiles.len(), layout[0].tiles.len());
    for image_tile in &levels[0].tiles {
        assert_eq!(image_tile.image.width(), image_tile.tile.width);
        assert_eq!(image_tile.image.height(), image_tile.tile.height);
        assert_eq!(image_tile.valid.width(), image_tile.tile.width);
        assert_eq!(image_tile.valid.height(), image_tile.tile.height);
    }

    let valid = bilinear_valid_image_from_mask(&levels[0].tiles[0].valid);
    assert_eq!(valid.width(), levels[0].tiles[0].tile.width);
    assert_eq!(valid.height(), levels[0].tiles[0].tile.height);
}

#[test]
fn tiled_bearing_patch_and_seed_overlap_is_deterministic() {
    let (camera, intr) = camera();
    let layout = build_tiled_bearing_levels(camera.as_ref(), &intr, 32, 32, 1.0, 1, 16, 8);
    let settings = PatchDepthSettings {
        patch_size: 4,
        patch_stride: 2,
        ..PatchDepthSettings::default()
    };
    let level = &layout[0];
    let half = settings.patch_size / 2;

    let mut total_centers = 0usize;
    let mut unique_centers = std::collections::HashSet::new();
    for tile_idx in 0..level.tiles.len() {
        let tile = &level.tiles[tile_idx];
        for center in level.patch_centers_in_tile(tile_idx, &settings) {
            total_centers += 1;
            unique_centers.insert(center);
            assert!(tile.contains_patch(center.0 as f64, center.1 as f64, half));
        }
    }
    assert!(!unique_centers.is_empty());
    assert!(
        total_centers > unique_centers.len(),
        "overlap should solve some patch centers in multiple tiles"
    );

    let seeds = vec![
        SparseDepthPrior {
            uv: Vector2::new(8.0, 8.0),
            eta: 0.693,
            eta_var: 0.04,
        },
        SparseDepthPrior {
            uv: Vector2::new(28.0, 28.0),
            eta: 1.386,
            eta_var: 0.32,
        },
    ];
    let assigned = assign_tiled_bearing_seeds(level, &seeds, 1.0, half);
    assert!(assigned.iter().map(Vec::len).sum::<usize>() >= seeds.len());
    for (tile_idx, tile_seeds) in assigned.iter().enumerate() {
        let tile = &level.tiles[tile_idx];
        for seed in tile_seeds {
            assert!(seed.uv[0] >= 0.0 && seed.uv[0] < tile.width as f64);
            assert!(seed.uv[1] >= 0.0 && seed.uv[1] < tile.height as f64);
        }
    }
}

// --- structure_tensor_passes unit tests ---
// Uses known analytic eigenvalues:
//   [[3,1],[1,3]] → λ_min=2, λ_max=4, condition=2
//   [[5,0],[0,1]] → λ_min=1, λ_max=5, condition=5
//   [[4,0],[0,0]] → λ_min=0, λ_max=4  (edge-like)
//   [[2,0],[0,2]] → λ_min=2, λ_max=2, condition=1  (isotropic)

#[test]
fn structure_tensor_both_disabled_always_passes() {
    assert!(structure_tensor_passes(0.0, 0.0, 0.0, 0.0, 0.0, None));
    assert!(structure_tensor_passes(4.0, 0.0, 0.0, 0.0, 0.0, None));
}

#[test]
fn structure_tensor_min_eigen_rejects_edge_patch() {
    // [[4,0],[0,0]] → λ_min=0 < 1.0
    assert!(!structure_tensor_passes(4.0, 0.0, 0.0, 1.0, 0.0, None));
}

#[test]
fn structure_tensor_min_eigen_accepts_corner_patch() {
    // [[2,0],[0,2]] → λ_min=2 ≥ 1.0
    assert!(structure_tensor_passes(2.0, 0.0, 2.0, 1.0, 0.0, None));
}

#[test]
fn structure_tensor_min_eigen_boundary_is_inclusive() {
    // [[3,1],[1,3]] → λ_min=2; threshold=2.0 → pass (not strictly less than)
    assert!(structure_tensor_passes(3.0, 1.0, 3.0, 2.0, 0.0, None));
    // threshold=2.01 → fail
    assert!(!structure_tensor_passes(3.0, 1.0, 3.0, 2.01, 0.0, None));
}

#[test]
fn structure_tensor_condition_rejects_anisotropic_patch() {
    // [[5,0],[0,1]] → condition=5 > 3.0
    assert!(!structure_tensor_passes(5.0, 0.0, 1.0, 0.0, 3.0, None));
}

#[test]
fn structure_tensor_condition_accepts_isotropic_patch() {
    // [[3,1],[1,3]] → condition=2 ≤ 3.0
    assert!(structure_tensor_passes(3.0, 1.0, 3.0, 0.0, 3.0, None));
}

#[test]
fn structure_tensor_degenerate_gradient_fails_condition_check() {
    // λ_min=0 ≤ 1e-12 guard fires before division
    assert!(!structure_tensor_passes(0.0, 0.0, 0.0, 0.0, 1.0, None));
}

#[test]
fn structure_tensor_both_active_passes_both_gates() {
    // [[3,1],[1,3]] → λ_min=2 ≥ 1.5, condition=2 ≤ 3.0
    assert!(structure_tensor_passes(3.0, 1.0, 3.0, 1.5, 3.0, None));
}

#[test]
fn structure_tensor_both_active_fails_condition_despite_min_eigen_ok() {
    // [[5,0],[0,1]] → λ_min=1 ≥ 0.5, condition=5 > 3.0
    assert!(!structure_tensor_passes(5.0, 0.0, 1.0, 0.5, 3.0, None));
}

// --- epipolar direction tests ---
// Edge patch: gxx=4, gxy=0, gyy=0 → λ_min=0, λ_max=4 (gradient only in x).

#[test]
fn structure_tensor_epipolar_accepts_edge_perpendicular_to_epi_line() {
    // Epipolar direction = (1,0): gradient is entirely along depth-sensing direction.
    // ê^T S ê = 4 ≥ min_eigen=1 → passes, despite λ_min=0.
    assert!(structure_tensor_passes(
        4.0,
        0.0,
        0.0,
        1.0,
        0.0,
        Some((1.0, 0.0))
    ));
}

#[test]
fn structure_tensor_epipolar_rejects_edge_parallel_to_epi_line() {
    // Epipolar direction = (0,1): gradient is orthogonal to depth-sensing direction.
    // ê^T S ê = 0 < min_eigen=1 → fails, correctly: this edge gives no depth info.
    assert!(!structure_tensor_passes(
        4.0,
        0.0,
        0.0,
        1.0,
        0.0,
        Some((0.0, 1.0))
    ));
}

#[test]
fn structure_tensor_epipolar_diagonal_edge() {
    // Edge at 45° and epipolar at 45°: ê = (1/√2, 1/√2), gxx=2, gxy=2, gyy=2.
    // ê^T S ê = 2*(0.5) + 2*2*(0.5) + 2*(0.5) = 1 + 2 + 1 = 4 ≥ min_eigen=1.
    let e = 1.0_f64 / 2.0_f64.sqrt();
    assert!(structure_tensor_passes(
        2.0,
        2.0,
        2.0,
        1.0,
        0.0,
        Some((e, e))
    ));
}

#[test]
fn structure_tensor_epipolar_skips_condition_check_for_pure_edge() {
    // With epipolar known, condition check is skipped. A pure edge (λ_min=0)
    // that would fail the condition guard (λ_min ≤ 1e-12) still passes when
    // the epipolar direction is aligned with the gradient.
    assert!(structure_tensor_passes(
        4.0,
        0.0,
        0.0,
        0.5,
        2.0,
        Some((1.0, 0.0))
    ));
}

#[test]
fn structure_tensor_min_eigen_active_suppresses_all_depth_output() {
    // Rejected patches have inv_var_w=0 and are skipped in the densify step,
    // so they leave the output pixel as Unknown. With min_structure_eigen=1e6
    // no real image can pass, so the entire depth output should be Unknown.
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        min_structure_eigen: 1e6,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        eta: 0.693,
        eta_var: 0.04,
    }];
    let img = textured_image();
    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(!out.status.data.contains(&PatchStatus::PhotoRefined));
    assert!(!out.status.data.contains(&PatchStatus::SeedOnly));
    assert!(out.status.data.iter().all(|s| *s == PatchStatus::Unknown));
}
