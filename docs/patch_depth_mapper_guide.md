# Patch Depth Mapper Guide

This guide describes the intended Rust port of
`../ECHO-LI-python/eqvio/patch_depth_mapper.py`.

The mapper is a direct patch inverse-depth estimator with sparse seed priors. It
is not the same module as the old optical-flow dense filter. Prefer names like
`patch_depth`, `direct_patch_depth`, or `PatchDepthMapper` over `FlowDep 2.0`
inside the code.

## Purpose

For each current frame, the mapper:

1. Collects converged sparse out-of-state seeds.
2. Selects a usable reference keyframe by baseline.
3. Solves inverse depth on a regular patch grid using photometric residuals and
   sparse seed priors.
4. Fuses overlapping patch estimates into cell-level depth, variance, and
   status maps.

The first integration target is visualization and performance evaluation, not
EqVIO landmark initialization.

## Boundary With Rudolf-V

The mapper consumes ordinary frame products:

```rust
struct FrameProducts {
    frame_id: FrameId,
    stamp: f64,
    gray: GrayImage,
    pose_t_wc: Matrix4<f64>,
}
```

It should build mapper-specific products on demand:

```rust
struct DepthKeyframe {
    frame: Arc<FrameProducts>,
    ref_pyramid: Vec<GrayImageF32>,
    grad_x_pyramid: Vec<ImageF32>,
    grad_y_pyramid: Vec<ImageF32>,
    intrinsics_by_level: Vec<ScaledIntrinsics>,
    settings_hash: u64,
}
```

Do not require Rudolf-V to compute gradients for every frame. Compute gradients
only when a frame is promoted or selected as a depth keyframe.

## Lazy Keyframe Policy

The Python mapper keeps a two-keyframe buffer. Preserve that policy first:

```text
on frame:
    gather sparse seeds
    estimate median seed depth
    select reference keyframe by baseline
    maybe insert current frame into keyframe buffer
    if no reference: return None
    solve patch depths against reference
```

Reference selection:

- minimum baseline = `min_baseline_ratio * median_depth`
- maximum baseline = `max_baseline_ratio * median_depth`
- choose the usable keyframe with the largest baseline

Keyframe insertion:

- fill the buffer until it has two frames
- afterward, insert the current frame only if it has enough baseline from the
  newest keyframe
- evict the oldest keyframe when inserting into a full buffer

## Settings To Port First

Port these settings directly from Python:

- `scale`
- `patch_size`
- `patch_stride`
- `cell_size`
- `min_depth`
- `max_depth`
- `photo_huber_delta`
- `sigma_photo`
- `n_gn_iters`
- `fd_eps`
- `lambda_seed`
- `seed_radius_px`
- `sigma_seed_floor`
- `n_search_candidates`
- `search_half_range`
- `min_baseline_ratio`
- `max_baseline_ratio`
- `min_photo_curvature`
- `max_photo_residual`
- `n_pyramid_levels`
- `var_floor`
- `status_weight_photo`
- `status_weight_seed`

Keep defaults close to Python until benchmarks suggest otherwise.

## Outputs

Expose the same output shape as Python:

```rust
struct PatchDepthOutput {
    depth_cells: DepthImage,
    variance_cells: VarianceImage,
    status_cells: StatusImage,
}
```

Status values should mirror Python:

```rust
enum PatchStatus {
    Unknown = 0,
    SeedOnly = 1,
    PhotoRefined = 2,
    Rejected = 3,
}
```

## Sparse Seed Input

The mapper should consume converged sparse seeds from `Sparse3DFilter`.

For each feature:

- use current-frame image position
- call `query(fid)` for depth and variance
- reject invalid or out-of-range depth
- convert depth to inverse depth:

```text
rho = 1 / z
rho_var = z_var / z^4
```

Build a spatial seed grid inside the mapper for neighborhood lookup. Do not ask
Rudolf-V to own this grid.

## Solver Structure

The initial Rust port should follow the Python structure:

1. Scale current/reference images if `scale < 1`.
2. Build current/reference pyramids.
3. Compute reference gradients per pyramid level.
4. Generate patch centers.
5. Build seed spatial grid.
6. For each patch:
   - initialize inverse depth from nearby seeds or discrete search
   - run GN iterations
   - accumulate photometric gradient/Hessian over all pyramid levels
   - add seed prior gradient/Hessian
   - clamp inverse depth to `[1/max_depth, 1/min_depth]`
   - classify status
7. Fuse patch estimates into cell maps.

Keep the first port CPU-only and scalar/straightforward. Add Rayon after the
single-threaded behavior matches Python.

## Pose Convention

Use the same convention as the Python code:

- incoming pose is camera-to-world `T_WC`
- reference/current transform is:

```text
T_ref_curr = inv(T_WC_ref) * T_WC_curr
```

The photometric warp maps a current-frame pixel and inverse depth into the
reference frame.

## Distortion Policy

Prefer the raw-image path if it benchmarks well. The patch mapper can stay in
raw distorted image coordinates without asking Rudolf-V to undistort images.

The trick is a mapper-owned bearing lookup table:

```text
for every raw pixel (u, v):
    bearing_lut[v, u] = camera.undistort((u, v))
```

This table is calibration-specific, not frame-specific. Build it once for a
given camera model and image size, then reuse it across frames.

Then each photometric residual uses:

```text
b_c = bearing_lut[v, u]
X_c = b_c / rho
X_r = R_ref_curr * X_c + t_ref_curr
u_ref = project_distorted(X_r)
r = I_ref_raw(u_ref) - I_curr_raw(u, v)
```

This avoids full-frame undistortion/remap and keeps Rudolf-V untouched:

```text
Rudolf-V:
    raw image tracking, raw feature coordinates

Patch mapper:
    owns bearing_lut
    owns direct-depth warping
    samples raw current/reference images
```

Do not run iterative undistortion inside the patch residual loop. Use the
precomputed LUT for current patch pixels, and use the camera model's analytic
distorted projection for the warped reference sample.

The projection Jacobian for GN is:

```text
J = d r / d rho
  = grad I_ref(u_ref)^T * d u_ref / d rho

d u_ref / d rho =
    d project_distorted(X_r) / d X_r
  * R_ref_curr
  * (-b_c / rho^2)
```

For radial-tangential projection:

```text
x = X / Z
y = Y / Z
r2 = x^2 + y^2
rad = 1 + k1*r2 + k2*r2^2

x_d = x*rad + 2*p1*x*y + p2*(r2 + 2*x^2)
y_d = y*rad + p1*(r2 + 2*y^2) + 2*p2*x*y

u = fx*x_d + cx
v = fy*y_d + cy
```

Use this analytic distorted projection and its Jacobian rather than pretending
the raw image is pinhole. Mixing raw distorted pixels with pinhole projection
will bias depth, especially near image edges.

If the raw-image path proves slower than expected because patch coverage is very
dense, keep an explicit fallback mode that builds undistorted/pinhole working
images inside the mapper. That fallback should still be mapper-owned, not a
Rudolf-V responsibility.

## Raw Distorted vs Undistorted Pinhole

Keep both designs in mind. They trade different costs.

### Raw Distorted

Raw distorted mode keeps the original image and evaluates the real camera model
inside the patch warp:

```text
b_c = bearing_lut[v, u]
X_c = b_c / rho
X_r = R_ref_curr * X_c + t_ref_curr
u_ref = project_distorted(X_r)
r = I_ref_raw(u_ref) - I_curr_raw(u, v)
```

Benefits:

- no full-frame image remap
- no extra undistorted image buffers
- Rudolf-V stays completely untouched
- natural for sparse direct methods such as ROVIO, where only a relatively
  small set of patches is maintained

Costs:

- distorted projection Jacobian is evaluated in the residual loop
- each patch pixel has a nonlinear projection path
- SIMD is harder because lanes have more math and more per-lane validity
  divergence

This is attractive when the mapper evaluates a small or moderate number of
patch pixels.

### Undistorted Pinhole

Undistorted pinhole mode remaps raw images into a pinhole working image before
the patch solve:

```text
I_raw -> I_pinhole

x = (u - cx) / fx
y = (v - cy) / fy
X_c = [x, y, 1] / rho
X_r = R_ref_curr * X_c + t_ref_curr
u_ref = project_pinhole(X_r)
r = I_ref_pinhole(u_ref) - I_curr_pinhole(u, v)
```

Benefits:

- simpler inner-loop projection
- simpler Jacobian
- easier to validate against the current Python reference
- closer to the optimized KLT-style patch arithmetic

Costs:

- full-frame remap work
- extra image buffers
- all seeds, patch centers, gradients, and intrinsics must live in the same
  pinhole coordinate system

This can be faster for a dense patch grid because the full-frame remap is paid
once, while the projection/Jacobian is evaluated millions of times:

```text
example 752x480, patch_size=8, patch_stride=4, n_gn_iters=5
    full-frame remap:       ~0.36M pixels
    patch residual evals:   ~7M patch-pixels
```

So the raw distorted path is not automatically faster for this mapper. It avoids
temporary images, but it adds projection work to the dominant residual loop.

### Practical Policy

ROVIO-style sparse direct patches can choose raw distorted easily because they
maintain a smaller set of patches. This patch mapper is denser, so the right
answer should be benchmarked.

Recommended implementation:

```rust
enum PatchMapperImageModel {
    RawDistorted,
    UndistortedPinhole,
}
```

For the first port, prioritize correctness and clear ownership:

- keep all remap/LUT/gradient work mapper-owned
- do not make Rudolf-V produce mapper-specific images
- implement one mode cleanly first
- retain the type boundary so the other mode can be added without redesigning
  the frontend

Benchmark both modes single-threaded before deciding the default.

### Tiled Pinhole

Wide-FOV camera models such as equidistant fisheye or double-sphere should not
be forced through one global undistorted pinhole image. A single pinhole target
either crops away useful field of view or stretches the image border enough to
damage patch photometry.

A future wide-FOV mode should use tiled pinhole rectification:

```text
I_raw -> multiple overlapping local pinhole/tangent-plane images

for each tile:
    use pinhole projection/Jacobian inside the patch solve
    keep a tile-local remap LUT from pinhole pixels to raw pixels
    accept seeds whose raw rays fall into the tile footprint
```

Benefits:

- preserves more FOV than one global pinhole image
- keeps the hot residual loop close to the SIMD-friendly pinhole path
- isolates camera-model complexity in per-tile remap/LUT construction
- allows different tile layouts for radtan, fisheye, and double-sphere cameras

Costs:

- duplicate work in tile overlap regions
- seed ownership/fusion must handle cells covered by more than one tile
- visualization and output fusion need a clear target coordinate system

Treat this as a third mode, not a replacement:

```rust
enum PatchMapperImageModel {
    RawDistorted,
    UndistortedPinhole,
    TiledPinhole,
}
```

Use `UndistortedPinhole` for EuRoC/Python-reference parity. Keep
`RawDistorted` for native-FOV correctness. Add `TiledPinhole` when wide-FOV
performance becomes important.

## Rerun Visualization

Before wiring mapper outputs into EqVIO, visualize:

- depth cells with a colormap
- variance cells
- status cells
- optional seed overlay
- selected keyframe age/baseline as text or scalar logs

This is the fastest way to judge whether performance and failure modes match the
Python reference.

## Integration Order

Recommended order:

1. Port settings, status enum, and output maps.
2. Add `FrameProducts` and a small mapper-owned keyframe cache.
3. Port image scaling, pyramid, and reference gradients.
4. Port seed gathering and seed grid.
5. Port patch solve and fusion.
6. Add Rerun visualization.
7. Benchmark with and without `--features parallel`.
8. Only then consider using mapper depths to initialize in-state landmarks.
