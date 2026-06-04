use super::*;

#[allow(dead_code)]
pub(super) fn build_tiled_bearing_levels(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    width: usize,
    height: usize,
    scale: f64,
    levels: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Vec<TiledBearingLevel> {
    let specs = undistort_level_specs(width, height, scale, levels);
    specs
        .iter()
        .enumerate()
        .map(|(level, spec)| {
            let tiles = build_tiled_bearing_tiles_for_level(
                camera,
                intrinsics,
                spec,
                level,
                tile_size,
                tile_overlap,
            );
            TiledBearingLevel {
                level,
                width: spec.lw,
                height: spec.lh,
                tiles,
            }
        })
        .collect()
}

#[allow(dead_code)]
pub(super) fn build_tiled_bearing_frame_levels(
    layout: &[TiledBearingLevel],
    raw_pyramid: &[Image<f32>],
) -> Vec<TiledBearingFrameLevel> {
    layout
        .iter()
        .zip(raw_pyramid)
        .map(|(level, raw)| {
            let tiles = level
                .tiles
                .iter()
                .map(|tile| TiledBearingImageTile {
                    tile: tile.clone(),
                    image: tile.lut.undistort_level(raw),
                    valid: tile.lut.valid_image(),
                })
                .collect();
            TiledBearingFrameLevel {
                level: level.level,
                width: level.width,
                height: level.height,
                tiles,
            }
        })
        .collect()
}

#[allow(dead_code)]
pub(super) fn assign_tiled_bearing_seeds(
    level: &TiledBearingLevel,
    seeds: &[SparseDepthPrior],
    scale_from_original: f64,
    patch_half: usize,
) -> Vec<Vec<SparseDepthPrior>> {
    let mut by_tile = vec![Vec::new(); level.tiles.len()];
    for seed in seeds {
        let scaled_uv = seed.uv * scale_from_original;
        for (tile_idx, tile) in level.tiles.iter().enumerate() {
            if !tile.contains_patch(scaled_uv[0], scaled_uv[1], patch_half)
                && !tile.contains_point(scaled_uv[0], scaled_uv[1])
            {
                continue;
            }
            by_tile[tile_idx].push(SparseDepthPrior {
                uv: tile.global_to_local(scaled_uv),
                eta: seed.eta,
                eta_var: seed.eta_var,
            });
        }
    }
    by_tile
}

#[allow(dead_code)]
pub(super) fn bilinear_valid_image_from_mask(mask: &Image<f32>) -> Image<f32> {
    build_bilinear_valid_pyramid(std::slice::from_ref(mask))
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            Image::from_vec(
                mask.width(),
                mask.height(),
                vec![0.0; mask.width() * mask.height()],
            )
        })
}

#[allow(dead_code)]
pub(super) fn build_tiled_bearing_tiles_for_level(
    camera: &dyn CameraModel,
    intrinsics: &CameraIntrinsics,
    spec: &image_ops::UndistortLevelSpec,
    level: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Vec<TiledBearingTile> {
    let tile_size = tile_size.max(2);
    let tile_overlap = tile_overlap.min(tile_size.saturating_sub(1));
    let stride = (tile_size - tile_overlap).max(1);
    let mut tiles = Vec::new();
    let mut y0 = 0;
    loop {
        let h = tile_size.min(spec.lh - y0);
        let mut x0 = 0;
        loop {
            let w = tile_size.min(spec.lw - x0);
            tiles.push(build_tiled_bearing_tile(
                camera, intrinsics, spec, level, x0, y0, w, h,
            ));
            if x0 + w >= spec.lw {
                break;
            }
            x0 = (x0 + stride).min(spec.lw - 1);
        }
        if y0 + h >= spec.lh {
            break;
        }
        y0 = (y0 + stride).min(spec.lh - 1);
    }
    tiles
}
