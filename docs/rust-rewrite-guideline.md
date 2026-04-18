# Rust Rewrite Guideline

Consolidating three Python packages — **ECHO-LI-python**, **gift-python**, and **liepp-python** — into a Rust-native VIO stack with ROS 2 integration.

## Feasibility Assessment

The rewrite is feasible and well-motivated:

- **Small, well-scoped codebases** — liepp-python (~1.9k LOC), gift-python (~2.2k LOC), ECHO-LI-python (~9.4k LOC). Total ~13.5k Python LOC.
- **Rudolf-V already exists** (~13k LOC Rust) covering most of gift-python's functionality (FAST, Harris, KLT, image pyramids, camera models, essential matrix, NMS). Only ZNCC/LBP/ORB occlusion checks remain to be ported.
- **Math-heavy, performance-critical code** — matrix exponentials, Lie group operations, inverse-compositional KLT, Gaussian-Beta depth filters with numba-JIT kernels, dense per-pixel Kalman filtering. Rust's zero-cost abstractions and SIMD support are a natural fit.
- **No complex Python-specific dependencies** — NumPy/SciPy operations map directly to nalgebra; numba kernels (FlowDep splatting, depth triangulation, Gaussian-Beta update) become native Rust loops with no JIT overhead.
- **Existing Rust ecosystem** — nalgebra for linear algebra + Lie groups, image/imageproc for I/O, rayon for parallelism, `r2r` or `rclrs` for ROS 2.

## Repository Mapping

| Python Package | Rust Target | Notes |
|----------------|-------------|-------|
| **liepp-python** | `echo-lie` crate (new) | SO(3), SE(3), SEn(3), SOT(3), SO(n), SL(n), GL(n) |
| **gift-python** | **Rudolf-V** (existing) | After ZNCC/LBP/ORB feature porting is complete |
| **ECHO-LI-python** | **ECHO-LI** (this repo) | EqF VIO with planar landmarks, FlowDep, sparse Gaussian-Beta filter |

## Architecture

Cargo workspace with ROS-agnostic core:

```
echo-li/
├── Cargo.toml                          # workspace root
├── echo-lie/                           # Lie group library
├── echo-li-core/                       # pure algorithm library (no ROS dependency)
│   ├── Cargo.toml                      # depth::flowdep is pub — other projects can depend on echo-li-core for standalone FlowDep use
│   └── src/
│       ├── lib.rs
│       ├── mathematical/
│       │   ├── mod.rs
│       │   ├── vio_state.rs            # VIO state on SE₂(3) × SOT(3)ⁿ
│       │   ├── vio_group.rs            # Symmetry group definition
│       │   ├── vio_eqf.rs              # Equivariant filter core
│       │   ├── eqf_matrices.rs         # A, B, C matrices (lift, innovation)
│       │   ├── vision_measurement.rs
│       │   ├── imu_velocity.rs
│       │   └── plane_measurement.rs
│       ├── coordinate_suite/
│       │   ├── mod.rs
│       │   ├── euclid.rs               # Euclidean landmark coords
│       │   ├── invdepth.rs             # Inverse-depth coords
│       │   └── normal.rs               # Normal (polar) coords
│       ├── depth/
│       │   ├── mod.rs
│       │   ├── flowdep.rs              # Dense per-pixel inverse-depth Kalman filter
│       │   ├── flowdep_kernels.rs      # depth_densification, bilinear_splatting, vogiatzis_update
│       │   ├── sparse_gb.rs            # Per-feature 1D Gaussian-Beta filter (EUCLIDEAN/INVDEPTH/POLAR)
│       │   ├── sparse_gb_3d.rs         # 3D Local IEKF on Normal chart (SOT(3) manifold)
│       │   ├── keyframe_pool.rs        # Keyframe selection + DIS flow management
│       │   └── optical_flow.rs         # Dense optical flow backend (DIS or custom)
│       ├── plane_detection/
│       │   ├── mod.rs
│       │   ├── detector.rs             # Delaunay + RANSAC plane fitting
│       │   └── fitting.rs              # SVD plane fitting, CP refinement
│       ├── dataserver/
│       │   ├── mod.rs
│       │   └── asl_dataset.rs          # EuRoC ASL dataset reader
│       ├── alignment.rs                # Trajectory alignment (Umeyama)
│       ├── initialization.rs           # Filter initialization
│       └── visualization.rs            # Optional: minifb or egui live display
├── echo-li-ros2/                       # ROS 2 node wrapping echo-li-core
│   ├── Cargo.toml
│   ├── src/
│   │   ├── node.rs                     # main ROS 2 node
│   │   ├── subscribers.rs              # IMU + image topic subscribers
│   │   ├── publishers.rs               # odometry, pose, pointcloud, depth image publishers
│   │   ├── tf.rs                       # TF2 broadcaster for camera/IMU/world frames
│   │   └── config.rs                   # ROS param ↔ EqF config bridge
│   ├── launch/
│   │   └── echo_li.launch.py
│   └── config/
│       └── params.yaml
├── echo-li-py/                         # Python bindings via PyO3 + maturin (pip install echo-li)
│   ├── Cargo.toml
│   ├── pyproject.toml                  # maturin build config
│   └── src/
│       ├── lib.rs                      # #[pymodule] entry point
│       ├── pipeline.rs                 # run_euroc() one-liner, internal orchestration
│       ├── flowdep.rs                  # FlowDepFilter + FlowDepSettings as #[pyclass]
│       ├── types.rs                    # numpy ↔ nalgebra, result dataclasses
│       └── config.rs                   # YAML/dict → Rust settings bridge
├── echo-li-cli/                        # standalone CLI (no ROS dependency)
│   ├── Cargo.toml
│   └── src/main.rs                     # clap CLI, dataset iteration, optional visualization
├── configs/
│   └── *.yaml
└── tests/
```

## Phase 1: liepp → Rust `echo-lie` Crate

**Scope:** ~1.9k Python LOC → new Rust library crate.

### Design Decisions

- **Use nalgebra as the matrix backend.** nalgebra provides const-generic stack-allocated matrices, existing rotation/isometry types, and broad ecosystem adoption. This also replaces lalir in Rudolf-V.
- **Wrap nalgebra primitives with Lie-group semantics.** nalgebra has `UnitQuaternion` (SO(3)) and `Isometry3` (SE(3)), but lacks SEn(3), SOT(3). Implement these as newtypes over nalgebra matrices.
- **Expose exp/log/Adjoint/action as traits.** Define a `LieGroup` trait with associated `Algebra` type, enabling generic EqF code.

### Modules

| Python Module | Rust Module | Strategy |
|---------------|-------------|----------|
| `so3.py` (396 LOC) | `so3.rs` | Thin wrapper around `nalgebra::UnitQuaternion` + custom exp/log with Rodrigues |
| `se3.py` (296 LOC) | `se3.rs` | Newtype over `nalgebra::Isometry3<f64>` |
| `sen3.py` (299 LOC) | `sen3.rs` | Custom: rotation + N translation vectors |
| `sot3.py` (253 LOC) | `sot3.rs` | Custom: SO(3) × R⁺ (rotation + scale) |
| `so_n.py`, `sln.py`, `gln.py` | `so_n.rs`, `sln.rs`, `gln.rs` | Generic over dimension using nalgebra's `SMatrix<f64, N, N>` |
| `base.py` (89 LOC) | `traits.rs` | `LieGroup` trait: `exp`, `log`, `Adj`, `inverse`, `compose`, `act` |

### Key Considerations

- Preserve the Python test suite structure. Port all pytest cases to `#[test]` functions.
- Numerical tolerances: match the Python thresholds (typically 1e-8 for group axioms).
- Consider `#[derive(Clone, Copy)]` for stack-allocated groups (SO3, SE3) for ergonomic value semantics.

## Phase 2: gift-python → Rudolf-V (Feature Completion)

**Scope:** Port remaining occlusion-check features from gift-python to Rudolf-V.

### Remaining Features

Rudolf-V already implements: image pyramids, camera models (pinhole, radtan, equidistant, double-sphere), FAST detection, Harris detection, inverse-compositional KLT tracking, NMS, occupancy grid, essential matrix RANSAC, histogram equalization.

Still needed:

| Feature | gift-python Location | Description |
|---------|---------------------|-------------|
| **ZNCC verification** | `tracker.py:46` (`_zncc`) | Reference-patch zero-normalized cross-correlation for occlusion detection |
| **LBP descriptors** | `tracker.py:102-161` | Local Binary Pattern vector computation + chi-squared distance |
| **ORB descriptors** | `tracker.py:168-189` | ORB descriptor computation + Hamming distance |
| **Occlusion check framework** | `tracker.py:233-265` | `OcclusionCheckMethod` enum, threshold config, per-feature reference storage |

A plan for ZNCC already exists at `Rudolf-V/reference_patch_zncc_plan.md`.

### Migration Checklist

1. Add `reference_patch` field to `Feature` struct
2. Implement ZNCC as a standalone function in a new `zncc.rs` module
3. Implement LBP computation (uniform LBP, rotation-invariant histogram)
4. Implement ORB descriptor (oriented BRIEF) — or depend on a Rust ORB crate if one matures
5. Add `OcclusionCheckMethod` enum and integrate into `frontend.rs` tracking loop
6. Validate against gift-python's test suite on EuRoC sequences

### lalir → nalgebra Migration

Rudolf-V currently depends on lalir for geometric verification (essential matrix). This should be migrated to nalgebra as part of this phase to unify the linear algebra backend across the stack. The essential matrix code in `essential.rs` (~611 LOC) uses SVD and matrix operations that map directly to nalgebra.

## Phase 3: ECHO-LI-python → ECHO-LI (This Repo)

**Scope:** ~9.4k Python LOC → Rust workspace (echo-li-core + echo-li-cli).

### Dependencies

- `echo-lie` crate (Phase 1)
- `rudolf-v` crate (Phase 2, as the visual frontend)
- `nalgebra` (matrices, rotations)
- `image` (I/O)
- `serde` + `serde_yaml` (config parsing, replacing PyYAML)
- `rayon` (parallel plane fitting, feature processing, dense depth kernels)

### Module Porting Order

Port bottom-up following the dependency graph:

1. **`dataserver/`** — EuRoC dataset reader (file I/O, CSV parsing, image loading). No math dependencies. Good warm-up.
2. **`coordinate_suite/`** — Landmark coordinate representations. Depends on echo-lie (SO3, SE3). Small, self-contained.
3. **`depth/flowdep_kernels.rs`** — The numba-JIT kernels (`_depth_densification`, `_bilinear_splatting`, `_bilinear_splatting_ab`, `_vogiatzis_update`). These are pure numerical loops over 2D grids — translate directly to Rust with no JIT overhead. Prime candidates for rayon parallelism and SIMD.
4. **`depth/sparse_gb.rs`** — Per-feature 1D Gaussian-Beta filter. Depends on coordinate_suite (Normal chart conversions: `conv_euc2normal`, `conv_normal2euc`, `point_chart_normal_inv`). Three parametrizations (EUCLIDEAN, INVDEPTH, POLAR) with unified prediction/update.
5. **`depth/sparse_gb_3d.rs`** — 3D Local IEKF variant. Full 3×3 covariance on SOT(3) manifold with sequential bearing + depth updates. Depends on Normal chart Jacobians from coordinate_suite.
6. **`depth/flowdep.rs`** + **`depth/keyframe_pool.rs`** — Dense depth filter orchestration. Keyframe pool management, DIS optical flow wrapping, predict/observe/update cycle. Note: OpenCV's DIS optical flow will need either an OpenCV-rust binding or a custom Rust implementation.
7. **`mathematical/`** — Core EqF. Port in order: `vio_state` → `vio_group` → `eqf_matrices` → `vision_measurement` / `imu_velocity` / `plane_measurement` → `vio_eqf`.
8. **`plane_detection/`** — Delaunay, RANSAC fitting, CP refinement. Depends on nalgebra SVD. Consider `spade` crate for Delaunay triangulation.
9. **`lib.rs`** — Top-level filter orchestration (from `vio_filter.py`).
10. **`alignment.rs`** — Trajectory evaluation (Umeyama alignment, APE/RPE metrics).
11. **`echo-li-cli/src/main.rs`** — CLI with `clap`, config loading, dataset iteration, optional visualization.

### Key Porting Considerations

- **Numba kernels → native Rust loops.** The four FlowDep numba kernels (`_depth_densification`, `_bilinear_splatting`, `_bilinear_splatting_ab`, `_vogiatzis_update`) are pixel-parallel loops that translate directly. Use rayon `par_iter` for row-parallel execution. SIMD intrinsics can further accelerate the inner loops.
- **SciPy `least_squares` with Cauchy loss** (used in plane fitting): Use `levenberg-marquardt` or `argmin` crate. The Cauchy robust kernel is straightforward to implement manually.
- **Delaunay triangulation** (used in plane detection): Use the `spade` crate (pure Rust, well-maintained).
- **Dense optical flow**: FlowDep uses OpenCV's DIS optical flow. Options: (a) `opencv-rust` bindings for DIS, (b) custom Rust DIS implementation, (c) alternative dense flow (e.g. Farneback via opencv-rust). Decision can be deferred.
- **Matrix exponentials / Jacobians**: Already handled by echo-lie crate. The Python code uses SymPy for validation — keep a Python script for cross-validation during development.
- **Config system**: Replace PyYAML with `serde_yaml`. The existing YAML configs can be reused as-is.
- **Visualization**: Optional. Use `minifb` (already in Rudolf-V) or `egui` for live display. Can be feature-gated.

### Allocation Strategy: Fixed-Capacity Heap

The only dynamically-sized dimension in the system is the number of active landmarks — in-state for EqF, and out-of-state for the Gaussian-Beta filter. These two have fundamentally different data layouts and should use different strategies.

#### EqF Covariance (in-state) — Boxed Fixed-Capacity Matrix

The EqF Riccati covariance `Σ` is a dense `d × d` matrix where `d = 21 + 3·n_pt + 6·n_plane`. In Python, every landmark add/remove triggers a numpy resize (heap allocation + copy). In Rust, we use a **Boxed fixed-capacity matrix** to eliminate reallocations while maintaining memory safety.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N_MAX_PT` | 40 | Config default `max_landmarks: 30`, tests go up to 60 |
| `N_MAX_PLANE` | 10 | Typical planar scenes have 3–8 active planes |
| **Max state dim** | **201** | `21 + 3×40 + 6×10 = 201` |
| **Σ memory** | ~314 KB | `201 × 201 × 8 bytes`. Fixed heap allocation. |

**Why Boxed instead of Stack?** 
A 314 KB object is too large for many thread stacks (which often default to 2 MB or less). Wrapping the fixed-size `SMatrix` in a `Box` puts it on the heap exactly once, preventing stack overflows while still allowing the compiler to use fixed-size SIMD optimizations.

**Marginalization: Swap and Pop**
To remove a landmark from the state without reallocating:
1. Identify the index of the landmark to be marginalized.
2. **Swap** its rows and columns with the rows and columns of the *last active* landmark in the matrix.
3. Decrement the `active_dim` counter.
4. Use `.fixed_view::<D, D>(0, 0)` for subsequent operations to strictly process the contiguous active block.

Implementation:

```rust
use nalgebra::SMatrix;

const STATE_DIM_MAX: usize = 201; // 21 + 3*40 + 6*10
type CovMatrix = SMatrix<f64, STATE_DIM_MAX, STATE_DIM_MAX>;

struct EqFState {
    /// Active state dimension (≤ STATE_DIM_MAX).
    dim: usize,
    /// Riccati covariance. Only the top-left dim × dim block is live.
    /// Boxed to avoid stack overflow (314 KB is too large for many stacks).
    sigma: Box<CovMatrix>,
    // ...
}
```
```

All matrix operations (predict, update, landmark add/marginalize) operate on `sigma.view((0,0), (dim, dim))` slices. Landmark removal is a row/column deletion on the live block — no reallocation. Landmark addition extends the live block and initializes the new rows/columns from the prior.

The `Box<SMatrix>` gives the same contiguous fixed-size memory layout as a raw `SMatrix` (cache-friendly, SIMD-compatible), but safely heap-allocated once at filter construction. No resize, no fragmentation, no stack overflow risk.

#### Sparse Gaussian-Beta Filter (out-of-state) — HashMap

Each feature in the GB filter carries independent per-feature state: canonical depth, variance, Beta a/b counts. There is **no joint covariance** — features are decoupled. The access pattern is keyed by feature ID (add on first observation, remove on track loss, query by ID).

```rust
struct GBFeatureState {
    depth: f64,
    variance: f64,
    beta_a: f64,
    beta_b: f64,
    track_length: u32,
    // ...  (~48 bytes per feature)
}

struct SparseGBFilter {
    features: FxHashMap<u32, GBFeatureState>, // rustc_hash, NOT std HashMap
    max_pool_size: usize,                     // default: 300
    // ...
}
```

**Do not use `std::collections::HashMap` here.** Rust's std HashMap uses SipHash (DoS-resistant, cryptographic) which is needlessly slow for integer keys queried hundreds of times per frame. Use `rustc_hash::FxHashMap` instead — the same non-cryptographic hash used by the Rust compiler, a drop-in replacement with significantly lower per-lookup cost.

Alternative: the `slab` crate provides an arena-backed pool where insert returns a `usize` key and lookup is a direct array index (zero hashing). Better cache locality than any hashmap. Trade-off: feature IDs from the tracker must be mapped to slab keys, adding a small indirection layer.

#### FlowDep Dense Grid — Heap

FlowDep operates on per-pixel grids (`depth_map`, `variance_map`, `beta_a`, `beta_b`) at configurable `image_scale` (typically 0.25–0.5 of input resolution). For EuRoC at 752×480 with `image_scale=0.25`, grids are 188×120 ≈ 22.5K pixels × 4 maps × 8 bytes ≈ 720 KB. This should be **heap-allocated** (`Vec<f64>` or `ndarray::Array2`) as the size depends on runtime image resolution.

## Phase 4: ROS 2 Integration

**Goal:** First-class ROS 2 node for real-time VIO on robot platforms.

### Approach

- Use `rclrs` (the official Rust ROS 2 client library) or `r2r` (community alternative with better ergonomics). Both support ROS 2 Humble+.
- The core library (`echo-li-core`) stays ROS-agnostic. The ROS node (`echo-li-ros2`) is a thin wrapper that converts messages and calls the core API.

### ROS 2 Topics

| Direction | Topic | Message Type | Description |
|-----------|-------|-------------|-------------|
| **Subscribe** | `/imu/data` | `sensor_msgs/msg/Imu` | IMU acceleration + angular velocity |
| **Subscribe** | `/camera/image_raw` | `sensor_msgs/msg/Image` | Grayscale camera frames |
| **Subscribe** | `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsics + distortion |
| **Publish** | `/echo_li/odom` | `nav_msgs/msg/Odometry` | Filtered pose + twist with covariance |
| **Publish** | `/echo_li/pose` | `geometry_msgs/msg/PoseStamped` | Pose only (for rviz) |
| **Publish** | `/echo_li/features` | `sensor_msgs/msg/PointCloud2` | Tracked 3D landmarks |
| **Publish** | `/echo_li/planes` | `visualization_msgs/msg/MarkerArray` | Detected planes (for rviz) |
| **Publish** | `/echo_li/depth` | `sensor_msgs/msg/Image` | FlowDep dense depth map |
| **Publish** | TF2 | `geometry_msgs/msg/TransformStamped` | world → imu → camera frames |

### Design Principles

- **Zero-copy where possible.** Use ROS 2 zero-copy transport for images. The `Image` message buffer can be borrowed directly into Rudolf-V's `Image<u8>` type.
- **Parameter server integration.** Expose all `FlowDepSettings`, `SparseGBSettings`, and EqF config as dynamic ROS parameters with runtime reconfiguration.
- **Lifecycle node.** Use ROS 2 lifecycle management for clean startup/shutdown, sensor discovery, and parameter validation.

## Phase 5: Python Wrapper (`echo-li-py`)

**Goal:** `pip install echo-li` gives colleagues a drop-in replacement for `ECHO-LI-python` with Rust performance. All tuning and feature toggling is done through the YAML config — the Python API is a one-liner.

### Stack

- **PyO3** for Rust → Python bindings
- **maturin** for build/packaging (`pip install .` or `maturin develop`)
- **numpy** for array I/O (trajectory results, depth maps)

### Python API

```python
import echo_li

# One-liner: run full VIO pipeline on a EuRoC sequence
result = echo_li.run_euroc("/path/to/V1_01_easy", config="configs/eqvio_euroc.yaml")

# result.timestamps       — np.ndarray (N,)
# result.positions         — np.ndarray (N, 3)
# result.quaternions       — np.ndarray (N, 4) [x, y, z, w]
# result.trajectory_tum()  — str in TUM format (for evo evaluation)
# result.write_tum("out/estimated_trajectory.txt")
```

All pipeline options (FlowDep, sparse Gaussian-Beta filter, plane detection, feature tracker settings, coordinate chart, max landmarks, etc.) are controlled by the YAML config, matching the same format as the CLI and `ECHO-LI-python`.

#### Standalone FlowDep

FlowDep is also exposed as a standalone module for use in other projects (e.g. dense depth estimation without the full VIO pipeline):

```python
from echo_li import FlowDepFilter, FlowDepSettings

settings = FlowDepSettings.from_yaml("configs/flowdep.yaml")
fdf = FlowDepFilter(settings, intrinsics)

# Per-frame update cycle
fdf.predict(pose_delta)
fdf.observe(image, pose)
depth_map = fdf.query()           # np.ndarray (H, W), inverse-depth
```

> **Future:** The sparse Gaussian-Beta depth filter (`SparseGBFilter`) may be exposed in the same way for standalone per-feature depth estimation.

### Implementation

The `echo-li-py` crate depends on `echo-li-core` and re-uses the same orchestration logic as `echo-li-cli`. The PyO3 layer is thin:

1. **`config.rs`** — Parse a YAML file path or Python dict into Rust `VIOFilterSettings` / `FlowDepSettings` / etc. via serde.
2. **`pipeline.rs`** — `#[pyfunction] fn run_euroc(dataset: &str, config: &str) -> PyResult<RunResult>`. Internally constructs the full event loop (dataset reader → IMU propagation → feature tracking → vision update → depth filters → plane detection), identical to what `echo-li-cli/src/main.rs` does.
3. **`flowdep.rs`** — `FlowDepFilter` and `FlowDepSettings` as `#[pyclass]` wrappers around `echo_li_core::depth::flowdep`. Exposes `predict()`, `observe()`, `query()` with numpy I/O. (Future: `SparseGBFilter` may be added similarly.)
4. **`types.rs`** — `RunResult` as a `#[pyclass]` with numpy array accessors via `numpy` crate. Trajectory, aligned trajectory, timing stats.
5. **`lib.rs`** — `#[pymodule]` registering `run_euroc`, `RunResult`, `FlowDepFilter`, `FlowDepSettings`.

### Build & Distribution

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[project]
name = "echo-li"
requires-python = ">=3.10"
dependencies = ["numpy"]
```

```bash
# Development install (editable, debug build)
cd echo-li-py && maturin develop

# Release wheel
maturin build --release
```

### Design Principles

- **No Python-side orchestration.** The entire VIO pipeline runs in Rust. Python only calls in and reads results out. This avoids GIL overhead and keeps the hot loop in native code.
- **Config-driven, not API-driven.** The Python interface exposes one function, not a zoo of classes. Colleagues who need fine-grained control use the Rust API or CLI directly.
- **Numpy for output only.** Input is file paths and config. Output is numpy arrays and convenience methods (`write_tum`, `trajectory_tum`).

## Testing Strategy

- **Unit tests**: Port all Python pytest cases to `#[cfg(test)]` modules. Maintain the same numerical tolerances.
- **Integration tests**: Run on EuRoC MAV sequences, compare trajectory outputs against the Python baseline.
- **Cross-validation**: During development, run Python and Rust side-by-side on the same dataset, compare state vectors at each timestep. A Python script that loads both outputs and computes max divergence is useful.
- **Benchmarks**: Use `criterion` (already in Rudolf-V) for performance-critical paths: EqF prediction/update, Gaussian-Beta kernels, plane fitting, KLT tracking.
- **ROS 2 tests**: rosbag replay with recorded EuRoC bags, latency measurement, topic rate verification.

## Milestones

| Milestone | Deliverable | Blocked By |
|-----------|-------------|------------|
| **M0** | echo-lie crate with full test coverage | — |
| **M1** | Rudolf-V: ZNCC/LBP/ORB ported, lalir→nalgebra migration | — |
| **M2** | ECHO-LI core: EqF filter running on EuRoC (point landmarks only) | M0, M1 |
| **M3** | ECHO-LI core: FlowDep + sparse Gaussian-Beta depth filters | M2 |
| **M4** | ECHO-LI core: Planar landmark support | M3 |
| **M5** | ECHO-LI core: Feature parity with ECHO-LI-python, trajectory validation | M4 |
| **M6** | ECHO-LI ROS 2: Node with full topic interface, launch files, param config | M5 |
| **M7** | echo-li-py: `pip install echo-li` with `run_euroc()` one-liner, maturin wheel | M5 |

M0 and M1 can proceed in parallel. M6 and M7 can proceed in parallel.
