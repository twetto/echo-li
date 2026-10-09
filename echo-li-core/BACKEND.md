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

Owed, in dependency order:

1. **`MsceqfBackend`.** `MSCEqFFilter` is a primitives API -- `stochastic_clone`,
   `insert_landmark`, `marginalize_clone`, `msc_update`, `birth_landmark`,
   `landmark_stream_update`, `reanchor_landmark` -- with no `process_vision`. The policy
   that turns those into one frame step is ~90 lines living in a Python test harness:
   clone before update; buffer observations excluding in-state SLAM; promote
   window-spanning tracks and re-anchor them to the current clone; stream-update live
   SLAM and marginalize lost ones; flush a track for `msc_update` on any of lost /
   too-long / touches-the-marginalized-clone / spanned; re-anchor SLAM landmarks off a
   dying clone to the newest; marginalize clones down and prune buffers. That is backend
   state, not caller state, and it must come inside. Porting it must reproduce the
   harness's validated numbers -- note that the update span and the anchor window are
   independent, and tying them together breaks one or the other.
2. **One constructor in core.** Fold the three copies of config-load, gravity-align,
   buffer-rebuild-replay into a single entry point and delete the per-wrapper versions,
   including the CLI's 1-sample divergence.
3. **Wrappers hold `Box<dyn VioBackend>`.** Only after 1 and 2, so the switch is a
   capability change rather than a silent behaviour change.

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
