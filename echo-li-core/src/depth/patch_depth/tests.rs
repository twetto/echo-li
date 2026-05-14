use std::sync::Arc;

use nalgebra::{Matrix4, Vector2};

use super::*;
use crate::core_types::CameraIntrinsics;
use crate::mathematical::camera::{CameraModel, PinholeModel};

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
        rho: 0.5,
        rho_var: 0.01,
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
        rho: 0.5,
        rho_var: 0.01,
    }];
    let img = vec![120u8; 32 * 32];

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out
        .depth_cells
        .data
        .iter()
        .any(|z| z.is_finite() && (*z - 2.0).abs() < 0.1));
    assert!(out.status_cells.data.contains(&PatchStatus::SeedOnly));
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
        rho: 0.5,
        rho_var: 0.01,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
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
        rho: 0.5,
        rho_var: 0.01,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
}

#[test]
fn fast_translation_refines_undistorted_pinhole_4x4_patches() {
    let (camera, intr) = camera();
    let settings = PatchDepthSettings {
        camera_mode: PatchDepthCameraMode::UndistortedPinhole,
        warp_mode: PatchDepthWarpMode::FastTranslation,
        patch_size: 4,
        patch_stride: 2,
        cell_size: 4,
        min_photo_curvature: 0.0,
        max_photo_residual: 255.0,
        ..PatchDepthSettings::default()
    };
    let mut mapper = PatchDepthMapper::new(camera, intr, 32, 32, settings).unwrap();
    let seeds = vec![SparseDepthPrior {
        uv: Vector2::new(16.0, 16.0),
        rho: 0.5,
        rho_var: 0.01,
    }];
    let img = textured_image();

    assert!(mapper
        .update_with_priors(frame(0, 0.0, img.clone()), &seeds, None, 0.0)
        .is_none());
    let out = mapper
        .update_with_priors(frame(1, 0.02, img), &seeds, None, 0.0)
        .unwrap();
    assert!(out.status_cells.data.contains(&PatchStatus::PhotoRefined));
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
