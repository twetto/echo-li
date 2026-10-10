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
2. **A backend choice the ROS 2 node can actually make.** The node holds the seam, but
   EqVIO is the only thing it can construct: `MsceqfBackend` needs a `SystemOrigin`
   (R0, p0, v0, b0, S0, g) and a `ProcessNoise`, none of which `VIOConfig` carries. The
   remaining work is config plumbing, not architecture.

3. **`MsceqfBackend::pose_covariance`.** It returns `None` today, so a node running
   MSCEqF would starve the Sparse3D depth filter of pose uncertainty. MSCEqF carries
   covariance on the group algebra over its clone window; projecting that onto a single
   camera pose is the missing derivation.

### Not converted, on purpose

**The CLI keeps a concrete `VIOFilter`.** It is the EqVIO replay and diagnostic tool: it
hashes filter state for determinism traces, reads `eqf.current_time` and the covariance
blocks, and writes trajectories as `VIOState`. Putting it behind a trait object would
mean widening the shared trait with EqVIO-only methods, which is the thing the seam
exists to avoid. A consumer that needs one backend's internals should hold that
backend -- `EqvioBackend::filter()` is there for exactly this.

**Config loading stays per-wrapper.** It is two lines each (`VIOConfig::from_yaml` then
`to_filter_settings`), and the part that actually drifted -- initialisation -- is now
shared. A wrapper type for two lines would be cost without benefit.

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
