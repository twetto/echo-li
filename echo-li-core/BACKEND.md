# The backend seam

How a pose backend is plugged in, what was decided, and what is still owed.

## Why

Three wrappers drive a filter: `echo-li-cli`, `echo-li-ros2`'s node, and the pyo3
bindings. All three already call the same surface -- feed IMU, feed a frame, read the
state -- but none of them could choose a *different* backend, because `VIOFilter`
hard-wires the EqVIO filter. The consequence was measured, not hypothetical: driving the
MSCEqF filter from a diagnostic meant hand-rolling an adapter in Python
(`midair_e2e_depth_nees.py::MSCEqFAdapter`), which shipped five silent bugs -- a gravity
sign, a quaternion ordering invisible to the harness's own error metric, a conflated
update-span and anchor-window, body-vs-world velocity, and borrowed IMU noise that
understated the gyro by 20x.

Those three wrappers also each carry their own copy of filter setup. As of this writing
the CLI gravity-aligns from **1** IMU sample while python and ROS 2 average **100**, and
only the CLI calls `check_stationary`. Nobody chose that; it is what three copies drift
into.

## Shape

`VioBackend` is deliberately small:

```rust
fn process_imu(&mut self, imu: IMUVelocity);
fn process_vision(&mut self, m: VisionMeasurement, priors: &HashMap<u64, LandmarkDepthPrior>);
fn nav_estimate(&self) -> NavEstimate;
```

Everything a backend might *not* have is a separate capability trait -- `LandmarkMap`,
`CameraPoseCovariance` -- so a backend advertises what it actually has instead of
returning a placeholder, and a consumer names what it needs in its own bounds:

```rust
fn drive<B: VioBackend + CameraPoseCovariance>(b: &mut B) { ... }
```

Three decisions behind that shape.

**The currency is `NavEstimate`, not `VIOState`.** `VIOState` is EqVIO's own state: a
sensor block plus camera-frame landmarks in EqVIO's parametrization. Making it the
interface would force every other backend to project itself into EqVIO's shape on every
read -- an MSCKF would invent a `camera_landmarks` vector it does not keep that way, and
silently drop its pose clones, which are not an EqVIO concept at all. Lossy in one
direction, fictional in the other. `NavEstimate` carries only what every VIO backend has
by definition: pose, body-frame velocity, gyro bias, accel bias. Landmarks move to
`LandmarkMap`; anything genuinely EqVIO-specific is reached through the concrete type,
which `EqvioBackend::filter()` hands out. Composition, not encapsulation: the wrapper
does not hide the thing it wraps.

**Depth priors are a per-frame parameter.** The alternative -- a setter called before
each frame -- is order-dependent hidden state, and order-dependent hidden state is
exactly what produced the bugs above. Only a source with its own metric scale belongs
here: stereo with a known baseline, a range sensor, or accel-vs-gravity. Sparse3D does
not qualify and was removed as a prior source in `e755139`; feeding back a depth derived
from this filter's own poses closes a loop on the unobservable scale direction and was
measured to compound rather than converge.

**The camera is held, not passed.** A backend is built against one camera and does not
change it mid-run, and both implementors need it in places other than the vision call.

## Status

Done:

- `VioBackend`, `NavEstimate`, `LandmarkMap`, `CameraPoseCovariance`
- `EqvioBackend`, verified byte-identical to the inherent calls on attitude, position,
  velocity, bias, landmark set and covariance after an IMU+vision sequence
- `defer_fallback_ids` retired (`9f52ecb`) -- it existed only to make features wait for a
  Sparse3D prior that no longer arrives
- **One start-up policy.** `initialization::StationaryInit` buffers the first IMU
  samples, gravity-aligns, and hands back the initial state plus the samples to replay.
  All three wrappers use it, so the divergence described above is gone: the CLI now
  averages 100 samples like the other two, which on EuRoC V1 improves 2 of 3 (V1_02
  0.155 -> 0.149 m, V1_03 0.206 -> 0.179 m) and costs 0.010 m on V1_01. The type also
  enforces a 5-sample floor, because below that `estimate_initial_pose` silently returns
  IDENTITY -- the one attitude the whole module exists to avoid, and 180 deg from gravity
  in the worst case. A unit test pins that.

  The CLI still differs in one respect, deliberately: it builds its filter eagerly and
  splices `start.replay` onto the front of its own event stream, where the other two
  replay into the filter via `start_filter`. Doing both would integrate those samples
  twice.

- **`MsceqfBackend`** -- the MSCEqF filter plus the track bookkeeping that turns its
  primitives into one frame step. Owns `clone_ids`, `landmark_ids`, the structureless
  observation buffer, the in-state SLAM set and the frame counter; implements
  `VioBackend` and `LandmarkMap`. Settings are `MsceqfSettings`.

  Two things the port had to get right and now pins with tests. MSCEqF carries velocity
  in the WORLD frame while `NavEstimate` is BODY, so `nav_estimate` rotates -- the
  difference is invisible to a position-only error metric and is one of the five bugs
  the Python adapter shipped. And buffered observations are keyed by clone ID, never by
  window position, because positions shift under marginalization.

  **VALIDATED against the Python reference loop.** `midair_msceqf_backend_ab.py` drives
  both over identical MidAir data and compares trajectories pointwise:

  | arm | frames | SLAM | ATE | max difference |
  |---|---|---|---|---|
  | structureless | 400 | 0 | 0.303% | 0.000000 m |
  | structureless | 7866 (full) | 0 | 0.624% | 0.000000 m |
  | with SLAM | 400 | 50 | 0.317% | 0.000000 m |
  | with SLAM | 7866 (full) | 100 | 0.401% | 0.000000 m |

  The ATE column matches the reference harness exactly (0.303% at 400 frames, 0.624%
  full), and accepted-update counts agree to the unit (2569 at 400 frames). Keep the
  noise defaults in the A/B harness tied to the reference: a first version carried
  Allan-variance figures from elsewhere, leaving acc_rw 55x, gyr_density 11x and gyr_rw
  9x too small. Both arms still agreed byte-for-byte -- the equivalence result was
  unaffected -- but the absolute ATE read 24.7% on the full trajectory and looked like
  the filter being inaccurate rather than the harness being misconfigured.

  Two convention differences had to be settled to get there, and both are worth knowing.
  The backend derives `dt` from consecutive IMU stamps, so its FIRST sample propagates
  nothing, where the Python loop passes `dt=DT` for every sample including the first --
  a driver primes the stamp. And the backend breaks promotion ties by track id, giving a
  total order, where the reference inherited dict insertion order from a stable sort.

Owed, in dependency order: `MSCEqFFilter` is a primitives API -- `stochastic_clone`,
   `insert_landmark`, `marginalize_clone`, `msc_update`, `birth_landmark`,
   `landmark_stream_update`, `reanchor_landmark` -- with no `process_vision`. The policy
   that turns those into one frame step was ~90 lines living in a Python test harness.
   The policy is ported; what remains is proving it reproduces the harness's numbers.
   Note the port deliberately differs in one respect: it normalizes pixels through the
   REAL camera model, where the Python harness used a bare inverse-K. Those agree on an
   undistorted pinhole (MidAir) and diverge on a distorted one, so MidAir is the right
   place to compare.
2. **`MsceqfBackend::pose_covariance`** -- DONE. The absolute pair is the newest
   clone's own 6x6 marginal (in an MSCKF the current camera pose IS a clone, and
   `phi()`'s camera pose is invariant to the navigation direction, so there is no
   core-state Jacobian route). The pair that matters is the clone-to-clone
   anchor->current RELATIVE covariance, reached through `CameraPoseCovariance`'s
   `relative_pose_covariances` / `live_clone_ids` / `current_clone_id` /
   `last_clone_eviction`, and consumed by `CloneRelFeed` -- the one shared feed the CLI
   and the ROS 2 node run, which reproduces the depth NEES harness's construction
   (clone-relative, PSD-clipped increments, and the harness's re-anchor rule: every
   frame, landmarks on the oldest live clone move onto the previous frame's clone,
   WITHOUT adding the oldest->new relative covariance -- `reanchor_transport_cov`,
   off since 2026-10-15: under the measured error channel that block is an
   independence artefact, not covariance, and the next update removes it; keeping it
   cost 6.6 vs 5.7 % range error, 26 vs 27/30 honest, sigma_r 5.15 vs 4.40 m, see the
   field's comment). Validated on all 30 Kite_training
   trajectories: the harness's `--pose-source backend --feed rust`, which routes the
   whole clone-relative block through `VIOFilter.sparse_feed` = this `CloneRelFeed`, is
   byte-identical to the harness's own Python block, and the harness's hand-built
   adapter (`--pose-source msceqf`) reproduces its 2026-10-09 ledger exactly. The
   eviction-based alternative (`--reanchor-eviction`, via `last_clone_eviction`) was
   measured and is NOT shipped: honest on 22/30 vs 26/30 for the harness rule, with a
   systematically heavier >chi2_99 tail; why is not established.
   One trap, pinned by `msceqf_eviction_is_recorded_before_the_clone_dies`: a clone
   stores the group element `E_k`, not a pose; its world camera pose is
   `SE3(T0) S0 E_k` (`clone_world_pose`), and the first version handed out `E_k`.

### Selecting MSCEqF

An `MSCEqF:` section in the YAML selects it; absent or `enabled: false` leaves EqVIO.
The section carries only what no other section can supply -- `SystemOrigin` and
`ProcessNoise` -- and deliberately NOT the initial attitude, position or extrinsic:
those come from gravity alignment and from the `t_bs` the rest of the system already
uses, and duplicating them is how two sources of truth start to disagree.

`pixelStd` is in PIXELS, like EqVIO's `measurementNoise.feature`, and is divided by fx
at construction because the backend normalizes observations before they reach the
filter. Initial bias is zero, matching the EqVIO path: gravity alignment at rest cannot
observe accelerometer bias, since the bias is parallel to measured gravity by
construction.

`config_path_matches_direct_construction` pins the mapping against direct construction
-- covariance blocks first, then 30 driven frames compared by fingerprint. The config
path has a transposition on every field (Dd tangent order, the pixel conversion, the
gravity sign, density-vs-random-walk), each of which still runs and still produces a
plausible trajectory. Three mutations were checked and all three fail the test.

The test also carries a non-vacuity control, because the first version of it did not
work: a backend with 100x the pixel noise must diverge. Without that, the comparison
was empty -- the synthetic rig started at rest while the imagery implied 2 m/s, so every
update was chi2-rejected and both arms agreed by doing nothing.

Starting the node with MSCEqF while Sparse3D is enabled logs a warning: this backend
reports no camera-pose covariance yet, so the depth filter runs on missing information.

### MSCEqF needs a ZUPT; EqVIO is papering over not having one

On a bag that starts with 25 s of rest, MSCEqF reaches 61 km where EqVIO reaches
43 m. Measured over the stationary window alone (1400 frames):

| arm | path |
|---|---|
| MSCEqF with vision | 764.0 m |
| MSCEqF without vision | 764.0 m -- IDENTICAL |
| EqVIO with vision | 0.5 m |
| EqVIO without vision | 764.6 m |

MSCEqF's vision contributes NOTHING at rest: 2447 triangulations, every one rejected
by the condition-number gate (`ECHO_MSC_TRIFAIL=1` tallies them). That is correct
behaviour -- a structureless MSCKF has no update without parallax -- so it dead-reckons
for 25 s and never recovers. An earlier note here guessed that degenerate zero-baseline
updates were being ACCEPTED; that is refuted, they are all gated.

Stationary initialisation does not help, and is not supposed to: both backends get the
identical initial state from `StationaryInit`. It fixes the attitude at t=0 and says
nothing about the 25 s after.

EqVIO survives because its update does not need parallax -- it births at a fixed scene
depth, and zero optical flow is consistent with zero motion, which pins the state. That
is the EqVIO formulation masking a missing component, not a property MSCEqF should be
expected to share.

**There is no zero-velocity update anywhere in this tree.** OpenVINS has
`zero_velocity_updater.cpp` for exactly this. The gap is already recorded as fatal on
EuRoC sequences that start at rest; it now also decides whether MSCEqF is usable on a
rig that idles before it moves. This is the blocker for running MSCEqF on real
recordings, ahead of the pose-covariance derivation.

### Widening the trait

Adding a method that only one backend implements is NOT forbidden, and
`pose_covariance() -> Option<&dyn CameraPoseCovariance>` already is one. What matters is
whether ABSENCE can be expressed. An `Option` return says "this backend cannot tell
you", which a consumer must handle; that is honest and costs the other implementor
nothing.

The thing to avoid is narrower: a method that bakes one backend's REPRESENTATION into
the interface. `state_estimate() -> VIOState` was exactly that -- EqVIO's struct, with
camera-frame landmarks in EqVIO's parametrization, which every other backend would have
to manufacture. The fix was not "never widen", it was to return what the question
actually is (`NavEstimate`, `landmarks_world()`, `camera_pose()`) rather than one
implementor's container.

### The CLI

Converted. It holds a concrete `EqvioBackend` rather than `Box<dyn VioBackendFull>`,
because it only ever runs EqVIO and the concrete type keeps `filter()` available; the
point was to stop it reading EqVIO's state layout, and it no longer does:

| was | now |
|---|---|
| `camera_pose_matrix(&f.eqf.state_estimate())` | `f.camera_pose()` |
| `hash_state(&state)` | `f.state_fingerprint()` |
| `landmarks_to_global(&state)` | `f.landmarks_world()` |
| `f.sparse_camera_pose_covariances()` | `f.camera_pose_covariances()` |
| `f.eqf.current_time` | `measurement.stamp`, which the caller already had |
| `Vec<(f64, VIOState)>` | `Vec<(f64, NavEstimate)>` |

`hash_state` and `camera_pose_matrix` are deleted. `state_fingerprint` was the one
genuinely new trait method, and it is a general question -- "hash your own state" --
not an EqVIO one; the EqVIO implementation reproduces the CLI's old field order
(attitude xyzw, position, velocity, bias, then each landmark id and camera-frame point)
so determinism traces recorded either side of the change stay comparable.

Verified digit-identical on EuRoC V1 after conversion: V1_01 0.2073 m, V1_02 0.1494 m,
V1_03 0.1785 m, all unchanged, and two traced runs produce byte-identical 2912-line
determinism logs.

## Deferred: priors into `VisionMeasurement`

Folding per-feature depth priors into `VisionMeasurement` is the cleaner design -- one
argument instead of two, and priors travelling with the observations that produced them.
Deliberately **not** done now: `VisionMeasurement` is used by every diagnostic harness in
`echo-li-python/tests/diagnostics`, and those harnesses are the instruments that produce
the research results. The blast radius lands on them for a mostly cosmetic gain.

Revisit when the harnesses are otherwise being touched. The migration is mechanical: add
an optional prior map to `VisionMeasurement`, change `process_vision` to take only the
measurement, update constructors, then delete the parameter. Doing it in that order keeps
each step compiling.
