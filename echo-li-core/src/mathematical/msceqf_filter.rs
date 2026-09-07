// MSCEqF-native filter (parallel to the EqVIO `vio_eqf`).
//
// This module is a faithful Rust port of the C++ MSCEqF core
// (`MSCEqF/source/msceqf/...`, Fornasier et al., arXiv:2311.11649). It exists
// SEPARATELY from `vio_eqf` — the EqVIO EqF — and is selected by config. The
// EqVIO path is never touched: its covariance lives on a state-normal-coordinate
// chart, whereas THIS filter's covariance lives on the symmetry-GROUP ALGEBRA by
// construction (Dd = SE_2(3) ⋉ bias, E = SE3 extrinsics, clones = SE3). That is
// the whole point of the "consistent build": nothing to reconcile, because P is
// already on the algebra the propagator/updater linearize on.
//
// # Tangent-order convention (READ THIS BEFORE EDITING BLOCK INDICES)
//
// The covariance and every Jacobian/propagator block are stored in
// **MSCEqF-native tangent order**, line-for-line with the C++ so the ported
// matrices need no permutation:
//
//   Dd  (dof 15): [ att(0:3) | vel(3:6) | pos(6:9) | delta(9:15) ]
//   E   (dof 6) : [ att(0:3) | pos(3:6) ]   (SE3, [omega, v])
//   clone_i (6) : [ att(0:3) | pos(3:6) ]   (SE3), appended at the tail
//
// echo-lie's `SE23`/`SemiDirectBias` use a DIFFERENT SE_2(3) tangent order,
// [att, pos, vel] (its SE23 stores translations as (position, velocity)). The
// ONLY place that difference is bridged is [`se23_incr_msceqf_to_echo`], applied
// to the 9-dim SE_2(3) part of an increment right before it hits the SDB mean via
// echo-lie's tested `SE23::exp`. The covariance never gets permuted, and the
// structureless update touches only clone columns (SE3), so it is agnostic to the
// Dd internal order.

use echo_lie::base::LieGroup;
use echo_lie::{SE3, SE23, SO3, SOT3, SemiDirectBias};
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector2, Vector3, Vector4, Vector6};

use crate::coordinate_suite::invdepth::conv_ind2euc;
use crate::mathematical::camera::CameraModel;
use crate::mathematical::msckf;

/// Semi-Direct-Bias block degrees of freedom (SE_2(3) 9 + bias 6).
pub const DD_DOF: usize = 15;
/// Extrinsics (SE3) block degrees of freedom.
pub const E_DOF: usize = 6;
/// Per-clone (SE3) degrees of freedom.
pub const CLONE_DOF: usize = 6;
/// Starting index of the Dd block in the covariance.
pub const DD_IDX: usize = 0;
/// Starting index of the E block in the covariance.
pub const E_IDX: usize = DD_DOF;
/// Starting index of the first clone.
pub const CLONE_BASE: usize = DD_DOF + E_DOF;
/// Per-landmark (SOT3-reduced) degrees of freedom carried in the covariance:
/// 2 bearing + 1 depth. The SOT3 group element is 4-dof; the 4th (rotation about
/// the ray) is the SO(2) gauge and is not carried. See the notes-repo design
/// `docs/eqvio/msceqf_native_instate_landmarks_design.md`.
pub const LM_DOF: usize = 3;

// Covariance layout: [ sensor(21) | clones(6·n) | landmarks(3·m) ]. Clones are a
// bounded window; landmarks are the unbounded, high-churn tail — so landmark
// birth/death is a pure tail resize that never reindexes clones, and only the
// bounded clone birth/marg shifts the landmark tail. `landmark_base()` = the
// dynamic split CLONE_BASE + CLONE_DOF·n.

/// Fixed origin `xi0` the lifted group element acts on. The MSCEqF state group
/// element `X` is initialized at identity; `xi0` carries "how close to ground
/// truth" the estimate is. The nav estimate is reconstructed via the `phi`
/// action of `X` on this origin (see the updater/output wiring).
#[derive(Debug, Clone)]
pub struct SystemOrigin {
    /// Origin extended pose `T0 = SE_2(3)` in echo-lie storage (R0, position0,
    /// velocity0). Kept in echo-lie order because it is a *group element*, not a
    /// covariance block.
    pub t0: SE23,
    /// Origin IMU bias `b0 = [gyro(3); accel(3)]` (R^6).
    pub b0: Vector6<f64>,
    /// Origin camera extrinsics `S0 = SE3` (I→C).
    pub s0: SE3,
    /// World-frame gravity vector `ge3` (e.g. [0,0,-9.81] ENU or [0,0,+9.81]
    /// NED-fixed; the harness supplies the sign). Enters as `R0^T g` in the
    /// propagator and `R^T g` in the lift.
    pub g: Vector3<f64>,
}

impl SystemOrigin {
    pub fn new(t0: SE23, b0: Vector6<f64>, s0: SE3, g: Vector3<f64>) -> Self {
        Self { t0, b0, s0, g }
    }
}

/// A single IMU reading. `ang` = angular velocity (rad/s), `acc` = specific
/// force (m/s^2). `w()` is the stacked SE3 tangent [ang; acc] the propagator and
/// lift consume.
#[derive(Debug, Clone, Copy)]
pub struct Imu {
    pub ang: Vector3<f64>,
    pub acc: Vector3<f64>,
}

impl Imu {
    pub fn w(&self) -> Vector6<f64> {
        let mut w = Vector6::zeros();
        w.fixed_rows_mut::<3>(0).copy_from(&self.ang);
        w.fixed_rows_mut::<3>(3).copy_from(&self.acc);
        w
    }
}

/// Continuous-time IMU + bias-random-walk process-noise parameters, matching the
/// C++ `PropagatorOptions` (`propagator.cpp:36-40`). Std values are densities.
#[derive(Debug, Clone, Copy)]
pub struct ProcessNoise {
    pub angular_velocity_std: f64,
    pub acceleration_std: f64,
    pub angular_velocity_bias_std: f64,
    pub acceleration_bias_std: f64,
    /// State-transition Taylor order: 1 => `I + H*dt`, else `exp(H*dt)`.
    pub state_transition_order: usize,
}

impl ProcessNoise {
    /// Continuous-time 12x12 `Q` = blkdiag(gyro, accel, gyro-bias, accel-bias).
    pub fn q(&self) -> SMatrix<f64, 12, 12> {
        let mut q = SMatrix::<f64, 12, 12>::zeros();
        let s = |v: f64| v * v;
        q.view_mut((0, 0), (3, 3))
            .copy_from(&(SMatrix::<f64, 3, 3>::identity() * s(self.angular_velocity_std)));
        q.view_mut((3, 3), (3, 3))
            .copy_from(&(SMatrix::<f64, 3, 3>::identity() * s(self.acceleration_std)));
        q.view_mut((6, 6), (3, 3))
            .copy_from(&(SMatrix::<f64, 3, 3>::identity() * s(self.angular_velocity_bias_std)));
        q.view_mut((9, 9), (3, 3))
            .copy_from(&(SMatrix::<f64, 3, 3>::identity() * s(self.acceleration_bias_std)));
        q
    }
}

/// Current system (nav) state produced by the `phi` action of `X` on the origin.
/// This is what gets output and fed into `lift`.
#[derive(Debug, Clone)]
pub struct NavState {
    /// Extended pose (R, position, velocity) in echo-lie order.
    pub t: SE23,
    /// IMU bias [gyro; accel].
    pub b: Vector6<f64>,
    /// Camera extrinsics.
    pub s: SE3,
}

/// The lift `lambda = Lift(xi, u)` (symmetry.cpp:76-135), structureless subset.
#[derive(Debug, Clone)]
pub struct Lift {
    /// SE_2(3) tangent (MSCEqF order [att, vel, pos]).
    pub lambda_t: SVector<f64, 9>,
    /// Bias tangent (R^6).
    pub lambda_b: Vector6<f64>,
    /// Extrinsics SE3 tangent [omega, v].
    pub lambda_s: Vector6<f64>,
}

/// 3x3 skew-symmetric (hat) operator.
fn skew(v: &Vector3<f64>) -> nalgebra::Matrix3<f64> {
    SO3::wedge(v)
}

/// SE_2(3) GROUP adjoint (9x9) in MSCEqF tangent order [att, vel, pos]:
/// `Ad = [[R,0,0],[skew(v)R,R,0],[skew(p)R,0,R]]`.
fn se23_group_adjoint_msceqf(d: &SE23) -> SMatrix<f64, 9, 9> {
    let r = d.rotation.as_matrix();
    let mut ad = SMatrix::<f64, 9, 9>::zeros();
    ad.view_mut((0, 0), (3, 3)).copy_from(&r);
    ad.view_mut((3, 3), (3, 3)).copy_from(&r);
    ad.view_mut((6, 6), (3, 3)).copy_from(&r);
    ad.view_mut((3, 0), (3, 3))
        .copy_from(&(skew(&d.velocity) * r));
    ad.view_mut((6, 0), (3, 3))
        .copy_from(&(skew(&d.position) * r));
    ad
}

/// The lifted symmetry-group element `X = (Dd, E, {clones})`. All elements start
/// at identity (see MSCEqF `state.hpp`), and evolve by the propagator/updater.
#[derive(Debug, Clone)]
pub struct StateGroup {
    /// Semi-Direct-Bias element `Dd = (D in SE_2(3), delta in R^6)`.
    pub sdb: SemiDirectBias,
    /// Extrinsics element `E in SE3`.
    pub e: SE3,
    /// Stochastic clones of `E`, in ascending timestamp order (matches the C++
    /// `std::map<fp, ...>` iteration and the covariance clone-block layout).
    pub clones: Vec<Clone>,
    /// Persistent in-state (SLAM) landmarks, appended at the covariance tail
    /// after the clones (birth order). Each is a per-feature SOT3 factor anchored
    /// to a clone. Empty on the structureless path.
    pub landmarks: Vec<Landmark>,
}

/// One stochastic clone: a frozen copy of `E` tagged by capture time.
#[derive(Debug, Clone)]
pub struct Clone {
    pub stamp: f64,
    pub pose: SE3,
}

/// One persistent in-state landmark: a per-feature SOT3 group element (born at
/// identity, evolves under the update) anchored to a clone, plus the fixed
/// anchor-frame reference point it deviates from. Error is carried in the
/// stereographic-bearing + inverse-depth chart (`LM_DOF` = 3).
#[derive(Debug, Clone)]
pub struct Landmark {
    /// SOT3 group element (SO(3)×R⁺); identity at birth.
    pub q: SOT3,
    /// Stamp of the anchor clone this landmark is expressed relative to (a stable
    /// tag; clone positions shift on marginalization, stamps do not).
    pub anchor: f64,
    /// Fixed anchor-frame reference point `f_a` (the landmark origin `ξ0`).
    pub origin: Vector3<f64>,
}

impl Landmark {
    /// Current anchor-frame point estimate `q · f_a` (SOT3 acting on the fixed
    /// origin). Equals `origin` at birth (`q = identity`).
    pub fn point(&self) -> Vector3<f64> {
        self.q.act(&self.origin)
    }
}

impl StateGroup {
    pub fn identity() -> Self {
        Self {
            sdb: SemiDirectBias::identity(),
            e: SE3::identity(),
            clones: Vec::new(),
            landmarks: Vec::new(),
        }
    }
}

/// MSCEqF-native filter state: the lifted group element, the fixed origin, and
/// the group-algebra covariance.
#[derive(Debug, Clone)]
pub struct MSCEqFFilter {
    pub origin: SystemOrigin,
    pub x: StateGroup,
    /// Covariance on the group algebra, MSCEqF-native tangent order (see the
    /// module-level convention note). Size = `CLONE_BASE + CLONE_DOF * n_clones`.
    pub cov: DMatrix<f64>,
    /// IMU/bias process noise + discretization order.
    pub noise: ProcessNoise,
}

impl MSCEqFFilter {
    /// Total covariance dimension: sensor + clones + landmarks.
    pub fn dim(&self) -> usize {
        self.landmark_base() + LM_DOF * self.x.landmarks.len()
    }

    /// Covariance index of the clone at position `i` (0 = oldest).
    pub fn clone_idx(&self, i: usize) -> usize {
        CLONE_BASE + CLONE_DOF * i
    }

    /// Number of in-state landmarks.
    pub fn n_landmarks(&self) -> usize {
        self.x.landmarks.len()
    }

    /// The sensor|clones ↔ landmarks split: first landmark column. Dynamic in the
    /// clone count, so a landmark's index only moves when a clone is born/removed.
    pub fn landmark_base(&self) -> usize {
        CLONE_BASE + CLONE_DOF * self.x.clones.len()
    }

    /// Covariance index of the landmark at position `j` (0 = oldest birth).
    pub fn landmark_idx(&self, j: usize) -> usize {
        self.landmark_base() + LM_DOF * j
    }

    /// Construct the filter at the given origin with the supplied per-block
    /// initial covariances, applying the exact MSCEqF origin cross-correlation
    /// transform (`state.cpp:31-56`).
    ///
    /// * `d_init` — 9×9 SE_2(3) block cov, MSCEqF order [att, vel, pos].
    /// * `delta_init` — 6×6 bias block cov.
    /// * `e_init` — 6×6 extrinsics block cov.
    ///
    /// The state group starts at identity; `xi0` alone controls closeness to
    /// ground truth. Covariance is then conjugated by the origin map `D` so the
    /// bias/extrinsics blocks inherit the correct cross-correlation to the SE_2(3)
    /// block whenever the origin is not identity.
    pub fn new(
        origin: SystemOrigin,
        d_init: &SMatrix<f64, 9, 9>,
        delta_init: &SMatrix<f64, 6, 6>,
        e_init: &SMatrix<f64, 6, 6>,
        noise: ProcessNoise,
    ) -> Self {
        let n = CLONE_BASE; // 21: Dd(15) + E(6), no clones at construction
        let mut cov = DMatrix::<f64>::zeros(n, n);
        // Block-diagonal seed.
        cov.view_mut((DD_IDX, DD_IDX), (9, 9)).copy_from(d_init);
        cov.view_mut((DD_IDX + 9, DD_IDX + 9), (6, 6))
            .copy_from(delta_init);
        cov.view_mut((E_IDX, E_IDX), (6, 6)).copy_from(e_init);

        // Origin transform D (state.cpp:48-56). Identity except:
        //   D[delta, Dd]   (6x6) = ad_se3(b0)
        //   D[E,     att]  (6x3) = AdS0inv[0:6, 0:3]
        //   D[E+3,   pos]  (3x3) = AdS0inv[3:6, 3:6]
        // Indices are MSCEqF-native: att=Dd+0, vel=Dd+3, pos=Dd+6, delta=Dd+9.
        let ad_s0_inv = origin.s0.inverse().adjoint(); // AdS0inv = Ad(S0^{-1}) (6x6)
        let ad_b0 = SE3::adjoint_algebra(&origin.b0); // ad_se3(b0) (6x6)

        let mut dmat = DMatrix::<f64>::identity(n, n);
        let delta_idx = DD_IDX + 9;
        // D[delta, Dd att..pos] — but C++ writes the 6x6 at (delta_idx, D_idx)
        // where D_idx is the START of the SE_2(3) tangent (att). ad_b0 is 6x6 and
        // acts on the [att, vel] = B(D) sub-tangent... in MSCEqF the first 6 cols
        // of the SE_2(3) tangent are [att, vel], which is exactly B(D). So the 6x6
        // block lands at columns (att:vel) = Dd+0 .. Dd+6.
        dmat.view_mut((delta_idx, DD_IDX), (6, 6)).copy_from(&ad_b0);
        // D[E, att] (6x3)
        dmat.view_mut((E_IDX, DD_IDX), (6, 3))
            .copy_from(&ad_s0_inv.view((0, 0), (6, 3)));
        // D[E+3, pos] (3x3) — pos is Dd+6 in MSCEqF order.
        dmat.view_mut((E_IDX + 3, DD_IDX + 6), (3, 3))
            .copy_from(&ad_s0_inv.view((3, 3), (3, 3)));

        cov = &dmat * &cov * dmat.transpose();

        Self {
            origin,
            x: StateGroup::identity(),
            cov,
            noise,
        }
    }

    /// Stochastic cloning of the current `E` element (`state.cpp:296-330`).
    /// Appends a clone of `E` at the tail: copies E's diagonal block, and E's full
    /// cross-column/row into the new tail block, so the clone shares E's
    /// covariance and cross-correlations at birth.
    pub fn stochastic_clone(&mut self, stamp: f64) {
        let old = self.dim();
        // Insert the new 6-block at the sensor|clones ↔ landmarks split, so the
        // clone lands AFTER the existing clones but BEFORE the landmark tail. When
        // there are no landmarks this is `old` (byte-identical to a tail append).
        let ins = self.landmark_base();
        let new = old + CLONE_DOF;
        // Remap old covariance index r → new index: rows at/after the split slide
        // down by CLONE_DOF to make room for the clone block at [ins, ins+6).
        let remap = |r: usize| if r < ins { r } else { r + CLONE_DOF };

        let mut cov = DMatrix::<f64>::zeros(new, new);
        for r in 0..old {
            let nr = remap(r);
            for c in 0..old {
                cov[(nr, remap(c))] = self.cov[(r, c)];
            }
        }
        // New clone block: diagonal = E block, cross to every old state = that
        // state's cross to E (the clone is born perfectly correlated with E).
        let e_diag = self.cov.view((E_IDX, E_IDX), (E_DOF, E_DOF)).into_owned();
        cov.view_mut((ins, ins), (E_DOF, E_DOF)).copy_from(&e_diag);
        for r in 0..old {
            let nr = remap(r);
            for k in 0..E_DOF {
                let v = self.cov[(r, E_IDX + k)];
                cov[(nr, ins + k)] = v;
                cov[(ins + k, nr)] = v;
            }
        }

        self.cov = cov;
        self.x.clones.push(Clone {
            stamp,
            pose: self.x.e.clone(),
        });
    }

    /// Append an in-state landmark at the covariance tail with a `LM_DOF`-block
    /// prior `p_block` and optional cross-covariance `cross` (an `old × LM_DOF`
    /// matrix over the current state; `None` = independent birth, zero cross).
    /// Returns the new landmark's position. This is the state-plumbing primitive;
    /// the delayed-init birth supplies the geometry-derived correlated `cross`.
    pub fn insert_landmark(
        &mut self,
        lm: Landmark,
        p_block: &SMatrix<f64, LM_DOF, LM_DOF>,
        cross: Option<&DMatrix<f64>>,
    ) -> usize {
        let old = self.dim();
        let mut cov = self.cov.clone().resize(old + LM_DOF, old + LM_DOF, 0.0);
        cov.view_mut((old, old), (LM_DOF, LM_DOF))
            .copy_from(p_block);
        if let Some(c) = cross {
            debug_assert_eq!((c.nrows(), c.ncols()), (old, LM_DOF));
            cov.view_mut((0, old), (old, LM_DOF)).copy_from(c);
            cov.view_mut((old, 0), (LM_DOF, old))
                .copy_from(&c.transpose());
        }
        self.cov = cov;
        self.x.landmarks.push(lm);
        self.x.landmarks.len() - 1
    }

    /// Marginalize (exact block deletion) the landmark at position `j`. Reuses the
    /// same block-agnostic keep-filter as `marginalize_clone`; only later
    /// landmarks shift, clones and sensor are untouched.
    pub fn marginalize_landmark(&mut self, j: usize) {
        assert!(j < self.x.landmarks.len(), "landmark index out of range");
        let idx = self.landmark_idx(j);
        let dim = self.dim();
        let keep: Vec<usize> = (0..dim).filter(|&r| r < idx || r >= idx + LM_DOF).collect();
        let new_dim = dim - LM_DOF;
        let mut cov = DMatrix::<f64>::zeros(new_dim, new_dim);
        for (rr, &r) in keep.iter().enumerate() {
            for (cc, &c) in keep.iter().enumerate() {
                cov[(rr, cc)] = self.cov[(r, c)];
            }
        }
        self.cov = cov;
        self.x.landmarks.remove(j);
    }

    /// Marginalize the clone at index `i` (`state.cpp:332-353`): drop its
    /// rows/cols and shift the remaining clone indices down.
    pub fn marginalize_clone(&mut self, i: usize) {
        assert!(i < self.x.clones.len(), "clone index out of range");
        let idx = self.clone_idx(i);
        let dim = self.dim();
        let keep: Vec<usize> = (0..dim)
            .filter(|&r| r < idx || r >= idx + CLONE_DOF)
            .collect();
        let new_dim = dim - CLONE_DOF;
        let mut cov = DMatrix::<f64>::zeros(new_dim, new_dim);
        for (rr, &r) in keep.iter().enumerate() {
            for (cc, &c) in keep.iter().enumerate() {
                cov[(rr, cc)] = self.cov[(r, c)];
            }
        }
        self.cov = cov;
        self.x.clones.remove(i);
    }

    /// Index of the oldest clone (the marginalization target, matching the C++
    /// `cloneTimestampToMarginalize` = `clones_.cbegin()`).
    pub fn oldest_clone(&self) -> Option<usize> {
        if self.x.clones.is_empty() {
            None
        } else {
            Some(0)
        }
    }
}

// ---------------------------------------------------------------------------
// Mean action (phi) and lift, ported from symmetry.cpp (structureless subset).
// ---------------------------------------------------------------------------

impl MSCEqFFilter {
    /// System-state action `xi = phi(X, xi0)` (symmetry.cpp:32-74), structureless
    /// subset (T, b, S only; no features/intrinsics):
    ///   T = xi0.T * D,  b = B^{-Adj} (xi0.b - delta),  S = C^{-1} * xi0.S * E.
    pub fn phi(&self) -> NavState {
        let d = &self.x.sdb;
        let t = self.origin.t0.compose(&d.d); // xi0.T().multiplyRight(X.D())
        let b_inv_adj = d
            .b()
            .adjoint()
            .try_inverse()
            .expect("B(D) adjoint invertible");
        let b = b_inv_adj * (self.origin.b0 - d.delta);
        // S = C^{-1} * xi0.S * E  (multiplyLeft(C.inv()) then multiplyRight(E)).
        let s = d.c().inverse().compose(&self.origin.s0).compose(&self.x.e);
        NavState { t, b, s }
    }

    /// The lift `lambda = Lift(xi, u)` (symmetry.cpp:76-135), structureless
    /// subset. `xi` is the CURRENT nav state (from `phi`).
    pub fn lift(&self, xi: &NavState, u: &Imu) -> Lift {
        let r = xi.t.rotation.as_matrix();
        let bg = xi.b.fixed_rows::<3>(0).into_owned();
        let ba = xi.b.fixed_rows::<3>(3).into_owned();

        let mut lambda_t = SVector::<f64, 9>::zeros();
        lambda_t.fixed_rows_mut::<3>(0).copy_from(&(u.ang - bg));
        lambda_t
            .fixed_rows_mut::<3>(3)
            .copy_from(&(u.acc - ba + r.transpose() * self.origin.g));
        lambda_t
            .fixed_rows_mut::<3>(6)
            .copy_from(&(r.transpose() * xi.t.velocity));

        // lambda_S = S^{-Adj} * [lambda_T[0:3]; lambda_T[6:9]]  (att, pos).
        let mut att_pos = Vector6::zeros();
        att_pos
            .fixed_rows_mut::<3>(0)
            .copy_from(&lambda_t.fixed_rows::<3>(0));
        att_pos
            .fixed_rows_mut::<3>(3)
            .copy_from(&lambda_t.fixed_rows::<3>(6));
        let s_inv_adj = xi.s.adjoint().try_inverse().expect("S adjoint invertible");
        let lambda_s = s_inv_adj * att_pos;

        // lambda_b = ad_se3(b) * lambda_T[0:6].
        let lambda_b = SE3::adjoint_algebra(&xi.b) * lambda_t.fixed_rows::<6>(0).into_owned();

        Lift {
            lambda_t,
            lambda_b,
            lambda_s,
        }
    }
}

// ---------------------------------------------------------------------------
// Covariance propagation, ported from propagator.cpp.
// ---------------------------------------------------------------------------

impl MSCEqFFilter {
    /// Continuous-time state matrix `A` (21x21, MSCEqF order [att,vel,pos | delta
    /// | E]), ported line-for-line from propagator.cpp:250-331.
    pub fn state_matrix(&self, u: &Imu) -> SMatrix<f64, 21, 21> {
        let d = &self.x.sdb;
        let dp = d.d.position; // X.D().p()
        let dv = d.d.velocity; // X.D().v()
        let dr = d.d.rotation.as_matrix(); // X.D().R()
        let delta = d.delta; // X.delta()

        let r0 = self.origin.t0.rotation.as_matrix();
        let r0tg = r0.transpose() * self.origin.g; // R0^T g
        let r0tv0 = r0.transpose() * self.origin.t0.velocity; // R0^T v0
        let b0 = self.origin.b0;
        let b0g = b0.fixed_rows::<3>(0).into_owned(); // gyro bias

        let adb0 = SE3::adjoint_algebra(&b0); // ad_se3(b0)

        // Psi: Psi[3:6,0:3] = wedge(R0Tg).
        let mut psi = SMatrix::<f64, 6, 6>::zeros();
        psi.view_mut((3, 0), (3, 3)).copy_from(&skew(&r0tg));

        // theta = AdB * u.w() + [0; R0Tg].
        let adb = d.b().adjoint();
        let mut theta = adb * u.w();
        let theta_acc = theta.fixed_rows::<3>(3).into_owned() + r0tg;
        theta.fixed_rows_mut::<3>(3).copy_from(&theta_acc);
        let delta_plus_theta = delta + theta;
        let ad_dpt = SE3::adjoint_algebra(&delta_plus_theta);

        let (di, ei) = (DD_IDX, E_IDX);
        let mut a = SMatrix::<f64, 21, 21>::zeros();

        // A1
        a.view_mut((di, di), (6, 6)).copy_from(&(psi - adb0));
        a.view_mut((di + 6, di), (3, 3))
            .copy_from(&(skew(&r0tv0) - skew(&dp) * skew(&b0g)));
        a.view_mut((di + 6, di + 3), (3, 3))
            .copy_from(&nalgebra::Matrix3::identity());

        // A2
        a.view_mut((di, di + 9), (6, 6))
            .copy_from(&SMatrix::<f64, 6, 6>::identity());
        a.view_mut((di + 6, di + 9), (3, 3)).copy_from(&skew(&dp));

        // A3
        a.view_mut((di + 9, di), (6, 6))
            .copy_from(&(adb0 * psi - ad_dpt * adb0));

        // A4
        a.view_mut((di + 9, di + 9), (6, 6)).copy_from(&ad_dpt);

        // A5..A7 (E block).
        let ad_s0_inv = self.origin.s0.inverse().adjoint(); // AdS0inv

        let psi1 = dr * u.ang + delta.fixed_rows::<3>(0).into_owned();
        let psi2 = psi1 - b0g;
        let psi3 = dv + skew(&dp) * psi1;
        let psi4 = dv + skew(&dp) * psi2 + r0tv0;
        let mut rho = Vector6::zeros();
        rho.fixed_rows_mut::<3>(0).copy_from(&psi2);
        rho.fixed_rows_mut::<3>(3).copy_from(&psi4);

        // Xi (6x9).
        let mut xi_m = SMatrix::<f64, 6, 9>::zeros();
        xi_m.view_mut((0, 0), (3, 3)).copy_from(&(-skew(&psi1)));
        xi_m.view_mut((3, 0), (3, 3))
            .copy_from(&(-skew(&psi3) - skew(&b0g) * skew(&dp)));
        xi_m.view_mut((3, 3), (3, 3))
            .copy_from(&nalgebra::Matrix3::identity());
        xi_m.view_mut((3, 6), (3, 3)).copy_from(&(-skew(&psi2)));

        // Gamma (6x6).
        let mut gamma = SMatrix::<f64, 6, 6>::zeros();
        gamma
            .view_mut((0, 0), (3, 3))
            .copy_from(&nalgebra::Matrix3::identity());
        gamma.view_mut((3, 0), (3, 3)).copy_from(&skew(&dp));

        a.view_mut((ei, di), (6, 9)).copy_from(&(ad_s0_inv * xi_m)); // A5
        a.view_mut((ei, di + 9), (6, 6))
            .copy_from(&(ad_s0_inv * gamma)); // A6
        a.view_mut((ei, ei), (6, 6))
            .copy_from(&SE3::adjoint_algebra(&(ad_s0_inv * rho))); // A7

        a
    }

    /// Continuous-time input matrix `B` (21x12), ported from
    /// propagator.cpp:333-356.
    pub fn input_matrix(&self) -> SMatrix<f64, 21, 12> {
        let d = &self.x.sdb;
        let ad_d = se23_group_adjoint_msceqf(&d.d); // AdD (9x9, MSCEqF order)
        let adb0 = SE3::adjoint_algebra(&self.origin.b0);
        let s0r = self.origin.s0.rotation.as_matrix();
        let dr = d.d.rotation.as_matrix();

        let (di, ei) = (DD_IDX, E_IDX);
        let mut b = SMatrix::<f64, 21, 12>::zeros();
        // B[D, 0:6] = AdD[0:9, 0:6].
        b.view_mut((di, 0), (9, 6))
            .copy_from(&ad_d.view((0, 0), (9, 6)));
        // B[delta, 0:6] = ad_se3(b0) * AdD[0:6, 0:6].
        let ad_d_66 = ad_d.view((0, 0), (6, 6)).into_owned();
        b.view_mut((di + 9, 0), (6, 6))
            .copy_from(&(adb0 * &ad_d_66));
        // B[delta, 6:12] = -AdD[0:6, 0:6].
        b.view_mut((di + 9, 6), (6, 6)).copy_from(&(-ad_d_66));
        // B[E, 0:3] = S0.R^T * D.R.
        b.view_mut((ei, 0), (3, 3))
            .copy_from(&(s0r.transpose() * dr));
        b
    }

    /// Van Loan discrete-time transition + noise. Returns `(Phi, Qd)` where
    /// `Phi` is the state transition (21x21) and `Qd` the discrete process noise
    /// (21x21), matching propagator.cpp:358-374 + 243-247.
    fn discrete(
        &self,
        a: &SMatrix<f64, 21, 21>,
        bmat: &SMatrix<f64, 21, 12>,
        dt: f64,
    ) -> (SMatrix<f64, 21, 21>, SMatrix<f64, 21, 21>) {
        let q = self.noise.q();
        let bqbt = bmat * q * bmat.transpose();
        // H = [[A, BQB^T], [0, -A^T]] (42x42).
        let mut h = SMatrix::<f64, 42, 42>::zeros();
        h.view_mut((0, 0), (21, 21)).copy_from(a);
        h.view_mut((21, 21), (21, 21)).copy_from(&(-a.transpose()));
        h.view_mut((0, 21), (21, 21)).copy_from(&bqbt);

        let disc: SMatrix<f64, 42, 42> = if self.noise.state_transition_order == 1 {
            SMatrix::<f64, 42, 42>::identity() + h * dt
        } else {
            (h * dt).exp()
        };

        let phi = disc.fixed_view::<21, 21>(0, 0).into_owned();
        // Qd = upper-tri-symmetrized( disc[0:21, 21:42] * phi^T ).
        let m = disc.fixed_view::<21, 21>(0, 21).into_owned() * phi.transpose();
        // Symmetrize from the upper triangle (matches M.selfadjointView<Upper>()).
        let mut qd = SMatrix::<f64, 21, 21>::zeros();
        for i in 0..21 {
            for j in i..21 {
                let v = m[(i, j)];
                qd[(i, j)] = v;
                qd[(j, i)] = v;
            }
        }
        (phi, qd)
    }

    /// Propagate covariance for one IMU step (propagator.cpp:225-248). Only the
    /// 21x21 core (Dd, E) is transitioned; clones are frozen but their
    /// cross-correlation to the core is rotated by `Phi`.
    pub fn propagate_covariance(&mut self, u: &Imu, dt: f64) {
        let a = self.state_matrix(u);
        let b = self.input_matrix();
        let (phi, qd) = self.discrete(&a, &b, dt);
        let n = CLONE_BASE; // 21
        let dim = self.dim();

        // Core: cov[0:n,0:n] = Phi * cov * Phi^T + Qd.
        let core = self.cov.view((0, 0), (n, n)).into_owned();
        let new_core = &phi * core * phi.transpose() + qd;
        self.cov.view_mut((0, 0), (n, n)).copy_from(&new_core);

        if dim > n {
            let tail = dim - n;
            // Cross cols: cov[0:n, n:] = Phi * cov[0:n, n:].
            let cc = self.cov.view((0, n), (n, tail)).into_owned();
            let new_cc = &phi * cc;
            self.cov.view_mut((0, n), (n, tail)).copy_from(&new_cc);
            // Cross rows: cov[n:, 0:n] = cov[n:, 0:n] * Phi^T.
            let cr = self.cov.view((n, 0), (tail, n)).into_owned();
            let new_cr = cr * phi.transpose();
            self.cov.view_mut((n, 0), (tail, n)).copy_from(&new_cr);
        }
    }

    /// Propagate the mean for one IMU step (propagator.cpp:203-223): RIGHT-update
    /// Dd and E by the lifted, `dt`-scaled algebra element.
    pub fn propagate_mean(&mut self, u: &Imu, dt: f64) {
        let xi = self.phi();
        let lambda = self.lift(&xi, u);

        // Dd.multiplyRight(SDB::exp(dt * [lambda_T; lambda_b])).
        let mut inn = SVector::<f64, 15>::zeros();
        inn.fixed_rows_mut::<9>(0)
            .copy_from(&(lambda.lambda_t * dt));
        inn.fixed_rows_mut::<6>(9)
            .copy_from(&(lambda.lambda_b * dt));
        self.x.sdb = sdb_right_increment(&self.x.sdb, &inn);

        // E.multiplyRight(SE3::exp(dt * lambda_S)).
        self.x.e = self.x.e.compose(&SE3::exp(&(lambda.lambda_s * dt)));
    }

    /// One full propagation step (covariance then mean, matching the C++ order).
    pub fn propagate_step(&mut self, u: &Imu, dt: f64) {
        self.propagate_covariance(u, dt);
        self.propagate_mean(u, dt);
    }
}

/// Apply a 15-dim SDB increment (MSCEqF tangent order) by RIGHT multiplication
/// `Dd <- Dd * exp(inn)` (propagator mean update). Mirrors `sdb_left_increment`
/// but composes on the right.
pub fn sdb_right_increment(dd: &SemiDirectBias, inn_msceqf: &SVector<f64, 15>) -> SemiDirectBias {
    let se23_echo = se23_incr_msceqf_to_echo(&inn_msceqf.fixed_rows::<9>(0).into_owned());
    let mut inn_echo = SVector::<f64, 15>::zeros();
    inn_echo.fixed_rows_mut::<9>(0).copy_from(&se23_echo);
    inn_echo
        .fixed_rows_mut::<6>(9)
        .copy_from(&inn_msceqf.fixed_rows::<6>(9));
    dd.compose(&SemiDirectBias::exp(&inn_echo))
}

/// Permute a 9-dim SE_2(3) increment from MSCEqF order [att, vel, pos] to
/// echo-lie order [att, pos, vel], so it can be fed to echo-lie's `SE23::exp`.
/// This is the single convention bridge between the (MSCEqF-native) covariance
/// tangent and echo-lie's tested group primitives.
pub fn se23_incr_msceqf_to_echo(u_msceqf: &SVector<f64, 9>) -> SVector<f64, 9> {
    let mut out = SVector::<f64, 9>::zeros();
    out.fixed_rows_mut::<3>(0)
        .copy_from(&u_msceqf.fixed_rows::<3>(0)); // att
    out.fixed_rows_mut::<3>(3)
        .copy_from(&u_msceqf.fixed_rows::<3>(6)); // pos <- pos
    out.fixed_rows_mut::<3>(6)
        .copy_from(&u_msceqf.fixed_rows::<3>(3)); // vel <- vel
    out
}

/// Inverse of [`se23_incr_msceqf_to_echo`]: echo order → MSCEqF order.
pub fn se23_incr_echo_to_msceqf(u_echo: &SVector<f64, 9>) -> SVector<f64, 9> {
    let mut out = SVector::<f64, 9>::zeros();
    out.fixed_rows_mut::<3>(0)
        .copy_from(&u_echo.fixed_rows::<3>(0)); // att
    out.fixed_rows_mut::<3>(3)
        .copy_from(&u_echo.fixed_rows::<3>(6)); // vel(MSCEqF@3) <- vel(echo@6)
    out.fixed_rows_mut::<3>(6)
        .copy_from(&u_echo.fixed_rows::<3>(3)); // pos(MSCEqF@6) <- pos(echo@3)
    out
}

/// Apply a full 15-dim SDB increment (MSCEqF tangent order) to an SDB mean by
/// LEFT multiplication `Dd <- exp(inn) * Dd` (`state_elements.hpp:150`). The
/// SE_2(3) part is permuted into echo order before `SE23::exp`; the bias part is
/// passed through unchanged (bias tangent order is shared).
pub fn sdb_left_increment(dd: &SemiDirectBias, inn_msceqf: &SVector<f64, 15>) -> SemiDirectBias {
    let se23_echo = se23_incr_msceqf_to_echo(&inn_msceqf.fixed_rows::<9>(0).into_owned());
    let mut inn_echo = SVector::<f64, 15>::zeros();
    inn_echo.fixed_rows_mut::<9>(0).copy_from(&se23_echo);
    inn_echo
        .fixed_rows_mut::<6>(9)
        .copy_from(&inn_msceqf.fixed_rows::<6>(9));
    SemiDirectBias::exp(&inn_echo).compose(dd)
}

/// Apply a 6-dim SE3 increment (order [omega, v], shared with MSCEqF) by LEFT
/// multiplication `E <- exp(inn) * E`.
pub fn se3_left_increment(e: &SE3, inn: &Vector6<f64>) -> SE3 {
    SE3::exp(inn).compose(e)
}

/// Apply an inverse-depth chart increment (`LM_DOF` = 3 tangent
/// `[bearing_stereo(2), δ_invdepth(1)]`) to a landmark's SOT3 factor by LEFT
/// multiplication `q <- exp(w) · q`. The chart tangent is first mapped to a
/// Euclidean anchor-frame point delta `γ = conv_ind2euc(origin) · chart_inn`,
/// then lifted to the SOT3 algebra `w = [ -origin×γ/‖origin‖² ; -origin·γ/‖origin‖² ]`
/// — the same per-landmark lift as `invdepth.rs::lift_innovation`, evaluated at the
/// fixed chart center `origin`. The SO(2)-about-ray gauge (SOT3's 4th dof) is not
/// excited, matching `LM_DOF = 3`.
pub fn sot3_left_increment_invdepth(
    q: &SOT3,
    origin: &Vector3<f64>,
    chart_inn: &Vector3<f64>,
) -> SOT3 {
    let gamma = conv_ind2euc(origin) * chart_inn;
    let qq = origin.norm_squared();
    let cross = origin.cross(&gamma);
    let w = Vector4::new(
        -cross[0] / qq,
        -cross[1] / qq,
        -cross[2] / qq,
        -origin.dot(&gamma) / qq,
    );
    SOT3::exp(&w).compose(q)
}

/// Jacobian `∂p̂_a/∂ε` (3×3) of the anchor-frame landmark point `p̂_a = q·origin`
/// with respect to its inverse-depth chart increment `ε`, in the EXACT direction
/// [`sot3_left_increment_invdepth`] applies it. Because both are built from the
/// same lift `L` and chart map `conv_ind2euc(origin)`, the measurement Jacobian
/// `h_f · landmark_chart_jacobian` is sign-consistent with the clone columns of
/// [`msckf::feature_jacobians_anchored_left`] (same left-perturbation linearization),
/// so a `+K·res` correction reduces the residual.
///
/// Derivation: the lift maps `ε → w = L·conv·ε` with
/// `L = [[−[origin]×/‖origin‖²]; [−originᵀ/‖origin‖²]]` (4×3), and applying
/// `exp(w)·q` moves `p̂_a` by `[−[p̂_a]× | p̂_a]·w`. Hence
/// `∂p̂_a/∂ε = [−[p̂_a]× | p̂_a] · L · conv_ind2euc(origin)`. At `q = identity`
/// (`p̂_a = origin`) this collapses to `−conv_ind2euc(origin)`.
pub fn landmark_chart_jacobian(q: &SOT3, origin: &Vector3<f64>) -> Matrix3<f64> {
    let p_a = q.act(origin); // current anchor-frame point
    let qq = origin.norm_squared();
    // L (4×3): ε-chart (via conv) → SOT3 algebra [ω(3); scale(1)].
    let mut l = SMatrix::<f64, 4, 3>::zeros();
    l.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-skew(origin) / qq));
    l.fixed_view_mut::<1, 3>(3, 0)
        .copy_from(&(-origin.transpose() / qq));
    // J_pw (3×4): SOT3 algebra → euclidean motion of p̂_a.
    let mut j_pw = SMatrix::<f64, 3, 4>::zeros();
    j_pw.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew(&p_a)));
    j_pw.fixed_view_mut::<3, 1>(0, 3).copy_from(&p_a);
    j_pw * l * conv_ind2euc(origin)
}

/// SE_2(3) ALGEBRA adjoint `ad_ξ` (9x9) in MSCEqF tangent order [att, vel, pos]:
/// `ad = [[ω^,0,0],[ν1^,ω^,0],[ν2^,0,ω^]]` with `ω = att`, `ν1 = vel`, `ν2 = pos`.
/// Used only by the curvature correction (`SE23::adjoint(vector)` in the C++ takes
/// a Lie-algebra vector, so it is the ALGEBRA adjoint, not the group Adjoint).
fn se23_algebra_adjoint_msceqf(v: &SVector<f64, 9>) -> SMatrix<f64, 9, 9> {
    let w = v.fixed_rows::<3>(0).into_owned();
    let v1 = v.fixed_rows::<3>(3).into_owned(); // vel
    let v2 = v.fixed_rows::<3>(6).into_owned(); // pos
    let sw = skew(&w);
    let mut ad = SMatrix::<f64, 9, 9>::zeros();
    ad.view_mut((0, 0), (3, 3)).copy_from(&sw);
    ad.view_mut((3, 3), (3, 3)).copy_from(&sw);
    ad.view_mut((6, 6), (3, 3)).copy_from(&sw);
    ad.view_mut((3, 0), (3, 3)).copy_from(&skew(&v1));
    ad.view_mut((6, 0), (3, 3)).copy_from(&skew(&v2));
    ad
}

// ---------------------------------------------------------------------------
// Structureless MSCEqF vision update, ported from updater.cpp:56-798.
// ---------------------------------------------------------------------------

/// One observation of a track: the global clone index (position in
/// `filter.x.clones`, ascending timestamp) and the NORMALIZED image coordinate
/// (Z1 plane, so the harness feeds a normalized pinhole `fx=fy=1, cx=cy=0`).
#[derive(Debug, Clone, Copy)]
pub struct MscTrackObs {
    pub clone: usize,
    pub uvn: Vector2<f64>,
}

/// A single feature track observed by several clones. The ANCHOR frame is the
/// observation with the smallest clone index (oldest clone), matching MSCEqF's
/// `track.timestamps_.front()`.
#[derive(Debug, Clone)]
pub struct MscTrack {
    pub obs: Vec<MscTrackObs>,
}

/// One in-state (SLAM) landmark's fresh observations for a measurement update:
/// the landmark's index `j` in `filter.x.landmarks` and its observations at the
/// current clones. The landmark's stored anchor clone MUST be among `obs` (it is
/// kept live by reanchoring before its anchor marginalizes); the update linearizes
/// at the current landmark estimate, NOT a fresh triangulation.
#[derive(Debug, Clone)]
pub struct LmUpdate {
    pub j: usize,
    pub obs: Vec<MscTrackObs>,
}

/// One in-state landmark's *fresh* observations for a STREAMING measurement update
/// (the OpenVINS SLAM-update analog). Unlike `LmUpdate`, the anchor is DECOUPLED —
/// it enters only through its representation Jacobian and is NEVER re-observed — so
/// each physical measurement is consumed exactly once (the birth consumed the track
/// history; every later frame contributes only its NEW bearing). `obs` therefore
/// holds only observations at clones OTHER than the anchor; any obs at the anchor
/// clone is ignored (it would double-count the anchor measurement).
#[derive(Debug, Clone)]
pub struct LmStreamUpdate {
    pub j: usize,
    pub obs: Vec<MscTrackObs>,
}

impl MSCEqFFilter {
    /// Build the block-diagonal curvature-correction transform `expΓ`
    /// (`symmetry.cpp:137-169`): second-order BCH term, `Γ = blkdiag(ad's) * -0.5`,
    /// returned as `Γ.exp()`. `inn` is the full-state left-update increment.
    fn curvature_correction(&self, inn: &DVector<f64>) -> DMatrix<f64> {
        let dim = self.dim();
        let mut gamma = DMatrix::<f64>::zeros(dim, dim);
        // Dd[0:9,0:9] = ad_SE23(inn[Dd .. +9]).
        let se23 = SVector::<f64, 9>::from_iterator(inn.rows(DD_IDX, 9).iter().copied());
        gamma
            .view_mut((DD_IDX, DD_IDX), (9, 9))
            .copy_from(&se23_algebra_adjoint_msceqf(&se23));
        // Dd[9:15, 0:6] = ad_se3(inn[Dd+9 .. +6])  (delta<->D coupling).
        let d_delta: Vector6<f64> = inn.fixed_rows::<6>(DD_IDX + 9).into_owned();
        gamma
            .view_mut((DD_IDX + 9, DD_IDX), (6, 6))
            .copy_from(&SE3::adjoint_algebra(&d_delta));
        // Dd[9:15, 9:15] = ad_se3(inn[Dd .. +6])  (delta<->delta).
        let d_att_vel: Vector6<f64> = inn.fixed_rows::<6>(DD_IDX).into_owned();
        gamma
            .view_mut((DD_IDX + 9, DD_IDX + 9), (6, 6))
            .copy_from(&SE3::adjoint_algebra(&d_att_vel));
        // E[15:21,15:21] = ad_se3(inn[E .. +6]).
        let e_inn: Vector6<f64> = inn.fixed_rows::<6>(E_IDX).into_owned();
        gamma
            .view_mut((E_IDX, E_IDX), (6, 6))
            .copy_from(&SE3::adjoint_algebra(&e_inn));
        // clone_i = ad_se3(inn[clone_i .. +6]).
        for i in 0..self.x.clones.len() {
            let ci = self.clone_idx(i);
            let c_inn: Vector6<f64> = inn.fixed_rows::<6>(ci).into_owned();
            gamma
                .view_mut((ci, ci), (6, 6))
                .copy_from(&SE3::adjoint_algebra(&c_inn));
        }
        gamma *= -0.5;
        gamma.exp()
    }

    /// Structureless multi-state-constraint update (`mscUpdate` + `UpdateMSCEqF`,
    /// updater.cpp:56-798). For each track: triangulate in the anchor clone frame,
    /// build the LEFT-chart residual/feature Jacobians over the clone columns,
    /// null-project the feature out, χ² gate, then a single stacked EKF update on
    /// the group algebra with LEFT-multiply corrections and optional curvature
    /// correction. The nav (Dd) and extrinsics (E) blocks carry NO measurement
    /// columns — they are corrected purely through their covariance cross-blocks to
    /// the clones, exactly as in MSCEqF.
    ///
    /// `cam` is the normalized (Z1) projection model; `pixel_std` is in the SAME
    /// (normalized) units. Returns the number of accepted tracks.
    pub fn msc_update(
        &mut self,
        tracks: &[MscTrack],
        cam: &dyn CameraModel,
        pixel_std: f64,
        chi2_mult: f64,
        curvature: bool,
    ) -> usize {
        let nclones = self.x.clones.len();
        if nclones < 2 {
            return 0;
        }
        let nc = CLONE_DOF * nclones; // clone-tail width
        let dim = self.dim();
        let r2 = pixel_std * pixel_std;
        // Pre-update clone-tail covariance (used by both the per-track χ² gate and
        // the final innovation), taken BEFORE any downdate.
        let p_cc = self
            .cov
            .view((CLONE_BASE, CLONE_BASE), (nc, nc))
            .into_owned();

        let mut c_rows: Vec<DMatrix<f64>> = Vec::new();
        let mut d_rows: Vec<DVector<f64>> = Vec::new();
        let mut accepted = 0usize;

        for track in tracks {
            if track.obs.len() < 2 {
                continue;
            }
            // Order observations by clone index; the front (oldest) is the anchor.
            let mut obs = track.obs.clone();
            obs.sort_by_key(|o| o.clone);
            if obs.iter().any(|o| o.clone >= nclones) {
                continue; // stale clone reference
            }
            let clone_of: Vec<usize> = obs.iter().map(|o| o.clone).collect();
            let mscobs: Vec<msckf::MscObs> = obs
                .iter()
                .map(|o| {
                    let pose = self.x.clones[o.clone].pose.clone();
                    msckf::MscObs {
                        pose: pose.clone(),
                        pose_fej: pose,
                        uv: o.uvn,
                    }
                })
                .collect();

            // Triangulate the track into a WORLD point (anchor-frame gates inside).
            let x_f = match msckf::triangulate(&mscobs, cam) {
                Some(p) => p,
                None => continue,
            };
            // LEFT-chart residual + feature Jacobians (per-observation 6-col bands).
            let (h_f, h_x_local, res) =
                match msckf::feature_jacobians_anchored_left(&x_f, &mscobs, cam) {
                    Some(v) => v,
                    None => continue,
                };
            // Null-project the feature out of the constraint.
            let (h_o, r_o) = match msckf::left_nullspace_project(&h_f, &h_x_local, &res) {
                Some(v) => v,
                None => continue,
            };
            let rows = h_o.nrows();
            if rows == 0 {
                continue;
            }
            // Scatter the per-observation 6-col bands into the global clone columns.
            let mut c_track = DMatrix::<f64>::zeros(rows, nc);
            for (j, &ci) in clone_of.iter().enumerate() {
                let src = h_o.view((0, 6 * j), (rows, 6));
                let dst = ci * CLONE_DOF;
                let mut tgt = c_track.view_mut((0, dst), (rows, 6));
                tgt += src;
            }
            // Per-track χ² gate: S = C·P_cc·Cᵀ + σ²I, χ² = rᵀ S⁻¹ r.
            let s_t =
                &c_track * &p_cc * c_track.transpose() + DMatrix::<f64>::identity(rows, rows) * r2;
            let chol = match s_t.clone().cholesky() {
                Some(c) => c,
                None => continue,
            };
            let chi2 = r_o.dot(&chol.solve(&r_o));
            if !chi2.is_finite() || chi2 > chi2_mult * msckf::chi2_095(rows) {
                continue;
            }
            c_rows.push(c_track);
            d_rows.push(r_o);
            accepted += 1;
        }

        if accepted == 0 {
            return 0;
        }

        // Stack accepted tracks into one system.
        let total_rows: usize = c_rows.iter().map(|c| c.nrows()).sum();
        let mut c = DMatrix::<f64>::zeros(total_rows, nc);
        let mut delta = DVector::<f64>::zeros(total_rows);
        let mut r0 = 0usize;
        for (cm, dv) in c_rows.iter().zip(d_rows.iter()) {
            let m = cm.nrows();
            c.view_mut((r0, 0), (m, nc)).copy_from(cm);
            delta.rows_mut(r0, m).copy_from(dv);
            r0 += m;
        }

        // QR-compress a tall system to `nc` rows (updater_helper.cpp:169-180).
        if c.nrows() > c.ncols() {
            let qr = c.clone().qr();
            let mut d = delta.clone();
            qr.q_tr_mul(&mut d);
            let r = qr.r(); // (nc × nc) upper-triangular
            c = r;
            delta = d.rows(0, nc).into_owned();
        }

        let rows = c.nrows();
        // EKF update on the algebra (UpdateMSCEqF, updater.cpp:431-706).
        // G = cov[:, clones]·Cᵀ ; S = C·P_cc·Cᵀ + R ; K = G·S⁻¹ ; inn = K·δ.
        let sub_cov_cols = self.cov.view((0, CLONE_BASE), (dim, nc)).into_owned();
        let g = &sub_cov_cols * c.transpose(); // (dim × rows)
        let s = &c * &p_cc * c.transpose() + DMatrix::<f64>::identity(rows, rows) * r2;
        let chol = match s.clone().cholesky() {
            Some(ch) => ch,
            None => return 0,
        };
        // K = G·S⁻¹  ⇒  Kᵀ = S⁻¹·Gᵀ (solve with the SPD Cholesky), K = (S⁻¹Gᵀ)ᵀ.
        let k = chol.solve(&g.transpose()).transpose(); // (dim × rows)
        let inn_vec = &k * &delta; // (dim)

        // Apply LEFT-multiply corrections.
        let dd_inn = SVector::<f64, 15>::from_iterator(inn_vec.rows(DD_IDX, 15).iter().copied());
        self.x.sdb = sdb_left_increment(&self.x.sdb, &dd_inn);
        let e_inn: Vector6<f64> = inn_vec.fixed_rows::<6>(E_IDX).into_owned();
        self.x.e = se3_left_increment(&self.x.e, &e_inn);
        for i in 0..nclones {
            let ci = self.clone_idx(i);
            let c_inn: Vector6<f64> = inn_vec.fixed_rows::<6>(ci).into_owned();
            self.x.clones[i].pose = se3_left_increment(&self.x.clones[i].pose, &c_inn);
        }

        // Covariance downdate: cov -= K·Gᵀ, then symmetrize from the upper triangle.
        let downdate = &k * g.transpose();
        self.cov -= downdate;
        let sym = (&self.cov + self.cov.transpose()) * 0.5;
        self.cov = sym;

        // Optional curvature correction.
        if curvature {
            let exp_gamma = self.curvature_correction(&inn_vec);
            self.cov = &exp_gamma * &self.cov * exp_gamma.transpose();
        }

        accepted
    }

    /// Apply a full-state LEFT-multiply correction `inn_vec` (length `dim()`) to the
    /// mean: Dd via [`sdb_left_increment`], E and every clone via
    /// [`se3_left_increment`], and every in-state landmark via
    /// [`sot3_left_increment_invdepth`] (its own inverse-depth chart block). Used by
    /// the in-state landmark birth/update where the correction reaches landmarks
    /// through their covariance cross-blocks.
    fn apply_left_increment(&mut self, inn_vec: &DVector<f64>) {
        let dd_inn = SVector::<f64, 15>::from_iterator(inn_vec.rows(DD_IDX, 15).iter().copied());
        self.x.sdb = sdb_left_increment(&self.x.sdb, &dd_inn);
        let e_inn: Vector6<f64> = inn_vec.fixed_rows::<6>(E_IDX).into_owned();
        self.x.e = se3_left_increment(&self.x.e, &e_inn);
        for i in 0..self.x.clones.len() {
            let ci = self.clone_idx(i);
            let c_inn: Vector6<f64> = inn_vec.fixed_rows::<6>(ci).into_owned();
            self.x.clones[i].pose = se3_left_increment(&self.x.clones[i].pose, &c_inn);
        }
        for j in 0..self.x.landmarks.len() {
            let li = self.landmark_idx(j);
            let chart_inn = Vector3::new(inn_vec[li], inn_vec[li + 1], inn_vec[li + 2]);
            let origin = self.x.landmarks[j].origin;
            self.x.landmarks[j].q =
                sot3_left_increment_invdepth(&self.x.landmarks[j].q, &origin, &chart_inn);
        }
    }

    /// Delayed-initialize one persistent in-state (SLAM) landmark from a track that
    /// spans the clone window — the equivariant analog of OpenVINS
    /// `UpdaterSLAM::delayed_init` + `StateHelper::initialize`, ported onto the
    /// symmetry group (see `vio_eqf.rs:add_landmark_delayed` for the EqVIO-chart
    /// sibling). Steps:
    ///
    /// 1. triangulate the track → world point → anchor(obs[0])-frame reference `f_a`;
    /// 2. LEFT-chart feature Jacobians, mapped into the inverse-depth landmark chart
    ///    (`H_L = H_f · conv_ind2euc(f_a)`);
    /// 3. QR-split into an invertible init block (3 rows) + nullspace-projected
    ///    update rows (`initialize_split`);
    /// 4. χ² gate on the (feature-free) update rows;
    /// 5. `initialize_invertible` in the chart: `M = H_R P_marg H_Rᵀ + σ²I`,
    ///    `P_LL = H_L⁻¹ M H_L⁻ᵀ`, cross-cov `−(Σ H_Rᵀ) H_L⁻ᵀ`, insert at the tail
    ///    (no clone shift — landmarks live after the clones);
    /// 6. mean-nudge the new landmark by `H_L⁻¹ res_init` through its chart;
    /// 7. apply the update rows as a structureless clone correction (landmarks and
    ///    nav move through cross-cov).
    ///
    /// Curvature (second-order) correction on the update rows is deferred, matching
    /// both OpenVINS' delayed-init and EqVIO's landmark path. Returns the new
    /// landmark's position, or `None` if the track is degenerate or χ²-rejected.
    pub fn birth_landmark(
        &mut self,
        track: &MscTrack,
        cam: &dyn CameraModel,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> Option<usize> {
        let nclones = self.x.clones.len();
        if nclones < 2 {
            return None;
        }
        let sigma2 = pixel_std * pixel_std;

        // Order observations by clone index; the front (oldest) is the anchor.
        let mut obs = track.obs.clone();
        obs.sort_by_key(|o| o.clone);
        if obs.len() < 2 || obs.iter().any(|o| o.clone >= nclones) {
            return None;
        }
        let clone_of: Vec<usize> = obs.iter().map(|o| o.clone).collect();
        let mscobs: Vec<msckf::MscObs> = obs
            .iter()
            .map(|o| {
                let pose = self.x.clones[o.clone].pose.clone();
                msckf::MscObs {
                    pose: pose.clone(),
                    pose_fej: pose,
                    uv: o.uvn,
                }
            })
            .collect();
        let nobs = obs.len();

        // 1. Triangulate → world point → anchor(obs[0])-frame reference f_a.
        let x_f = msckf::triangulate(&mscobs, cam)?;
        let anchor = mscobs[0].pose.clone();
        let f_a: Vector3<f64> = anchor.inverse().act(&x_f);
        if f_a[2] <= 1e-6 {
            return None;
        }

        // 2. LEFT-chart feature Jacobians, mapped into the inverse-depth chart at
        //    q0 = f_a (q̂ = identity at birth).
        let (h_f, h_x, res) = msckf::feature_jacobians_anchored_left(&x_f, &mscobs, cam)?;
        // Chart Jacobian in the LIFT's tangent direction (q̂ = identity at birth, so
        // this equals `−conv_ind2euc(f_a)`). Using `landmark_chart_jacobian` keeps the
        // column sign-consistent with the update path and the clone columns.
        let jl = landmark_chart_jacobian(&SOT3::identity(), &f_a);
        let jl_d = DMatrix::from_column_slice(3, 3, jl.as_slice());
        let h_l = &h_f * &jl_d; // 2N×3 chart Jacobian

        // 3. QR-split → invertible init block + nullspace-projected update rows.
        let (h_finit, hx_init, res_init, hup, res_up) = msckf::initialize_split(&h_l, &h_x, &res)?;

        // Marginal cov of the involved clone blocks (6N×6N), in obs order.
        let dim = self.dim();
        let mut p_marg = DMatrix::<f64>::zeros(6 * nobs, 6 * nobs);
        for a in 0..nobs {
            let ca = self.clone_idx(clone_of[a]);
            for b in 0..nobs {
                let cb = self.clone_idx(clone_of[b]);
                let src = self.cov.view((ca, cb), (6, 6)).into_owned();
                p_marg.view_mut((6 * a, 6 * b), (6, 6)).copy_from(&src);
            }
        }

        // 4. χ² gate on the (feature-free) update rows.
        let dof = res_up.len();
        if dof > 0 {
            let mut s = &hup * &p_marg * hup.transpose();
            for k in 0..dof {
                s[(k, k)] += sigma2;
            }
            let s_inv = s.try_inverse()?;
            let d = (res_up.transpose() * &s_inv * &res_up)[(0, 0)];
            if !d.is_finite() || d > chi2_mult * msckf::chi2_095(dof) {
                return None;
            }
        }

        // 5. initialize_invertible in the landmark chart.
        //    M = H_R P_marg H_Rᵀ + σ²I  (H_R = hx_init, the init-row clone Jacobian).
        let mut m_mat = &hx_init * &p_marg * hx_init.transpose();
        for k in 0..3 {
            m_mat[(k, k)] += sigma2;
        }
        let h_linv = h_finit.try_inverse()?; // 3×3, chart ← pixel
        let p_ll_d = &h_linv * &m_mat * h_linv.transpose(); // new landmark chart block

        // Cross-cov to the OLD full state: scatter hx_init's 6-blocks into the native
        // clone columns, then cross_old = −(Σ H_Rᵀ) H_L⁻ᵀ (dim×3, OLD ordering).
        let mut h_r_full = DMatrix::<f64>::zeros(3, dim);
        for a in 0..nobs {
            let col = self.clone_idx(clone_of[a]);
            h_r_full
                .view_mut((0, col), (3, 6))
                .copy_from(&hx_init.view((0, 6 * a), (3, 6)));
        }
        let m_a = &self.cov * h_r_full.transpose(); // dim×3
        let cross_old = -(m_a * h_linv.transpose()); // dim×3

        if !p_ll_d.iter().all(|v| v.is_finite()) || !cross_old.iter().all(|v| v.is_finite()) {
            return None;
        }

        // 6. Mean nudge: chart innovation δ_l = H_L⁻¹ res_init → SOT3 factor.
        let mut q = SOT3::identity();
        let delta_l = &h_linv * &res_init; // 3-vec, invdepth chart tangent
        if delta_l.iter().all(|v| v.is_finite()) && delta_l.norm() > 0.0 {
            let dl = Vector3::new(delta_l[0], delta_l[1], delta_l[2]);
            q = sot3_left_increment_invdepth(&q, &f_a, &dl);
        }

        // Insert the landmark at the tail with the correlated birth covariance.
        let p_ll = SMatrix::<f64, LM_DOF, LM_DOF>::from_fn(|i, j| p_ll_d[(i, j)]);
        let lm = Landmark {
            q,
            anchor: self.x.clones[clone_of[0]].stamp,
            origin: f_a,
        };
        let j = self.insert_landmark(lm, &p_ll, Some(&cross_old));
        // Symmetrize after the correlated insert (guards accumulated round-off).
        let sym = (&self.cov + self.cov.transpose()) * 0.5;
        self.cov = sym;

        // 7. Update rows: a structureless clone correction (feature already removed
        //    by the QR). Scatter hup into the clone-tail columns and run the EKF core;
        //    nav/clones/landmarks all move through their covariance cross-blocks.
        if dof > 0 {
            let dim = self.dim(); // grew by LM_DOF
            let nc = CLONE_DOF * nclones;
            let mut c = DMatrix::<f64>::zeros(dof, nc);
            for a in 0..nobs {
                let dst = clone_of[a] * CLONE_DOF;
                let mut tgt = c.view_mut((0, dst), (dof, 6));
                tgt += hup.view((0, 6 * a), (dof, 6));
            }
            let p_cc = self
                .cov
                .view((CLONE_BASE, CLONE_BASE), (nc, nc))
                .into_owned();
            let sub_cov_cols = self.cov.view((0, CLONE_BASE), (dim, nc)).into_owned();
            let g = &sub_cov_cols * c.transpose(); // dim×dof
            let s = &c * &p_cc * c.transpose() + DMatrix::<f64>::identity(dof, dof) * sigma2;
            if let Some(chol) = s.clone().cholesky() {
                let k = chol.solve(&g.transpose()).transpose(); // dim×dof
                let inn_vec = &k * &res_up; // dim
                self.apply_left_increment(&inn_vec);
                let downdate = &k * g.transpose();
                self.cov -= downdate;
                let sym = (&self.cov + self.cov.transpose()) * 0.5;
                self.cov = sym;
            }
        }

        Some(j)
    }

    /// In-state (SLAM) landmark measurement update — the equivariant analog of
    /// OpenVINS `UpdaterSLAM::update`. Unlike the structureless [`Self::msc_update`],
    /// the landmark is a state, so its feature column is KEPT (no nullspace
    /// projection): the update jointly corrects the observing clones, the anchor
    /// clone (which enters every row through the anchored Jacobian's band-0 coupling,
    /// exactly OpenVINS' representation-Jacobian `dpfg_dx`), and the landmark chart.
    ///
    /// The measurement is linearized at the CURRENT landmark estimate
    /// `G = anchor_pose · q·origin` (not a fresh triangulation), so this is a genuine
    /// EKF step on the in-state feature. The chart column uses
    /// [`landmark_chart_jacobian`] — the `∂p̂_a/∂ε` consistent with the lift — so a
    /// `+K·res` correction reduces the residual. Curvature (second-order) correction
    /// on the landmark block is deferred (matches OpenVINS + EqVIO).
    ///
    /// Each landmark's stored anchor clone must be among its observations; otherwise
    /// that landmark is skipped this frame (its anchor is kept live by reanchoring
    /// before marginalization). χ²-rejected updates leave the state untouched.
    /// Returns the number of landmarks whose update was applied.
    pub fn landmark_update(
        &mut self,
        updates: &[LmUpdate],
        cam: &dyn CameraModel,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> usize {
        let sigma2 = pixel_std * pixel_std;
        let nclones = self.x.clones.len();
        let mut accepted = 0;

        for upd in updates {
            if upd.j >= self.x.landmarks.len() {
                continue;
            }
            let lm = self.x.landmarks[upd.j].clone();
            // Resolve the anchor clone position by its stored stamp.
            let anchor_pos = match self.x.clones.iter().position(|c| c.stamp == lm.anchor) {
                Some(p) => p,
                None => continue, // anchor not live (should have been reanchored)
            };

            // Keep only live observations; the anchor MUST be present.
            let mut obs: Vec<MscTrackObs> = upd
                .obs
                .iter()
                .copied()
                .filter(|o| o.clone < nclones)
                .collect();
            if !obs.iter().any(|o| o.clone == anchor_pos) || obs.len() < 2 {
                continue;
            }
            // Anchor first (obs[0]), matching `feature_jacobians_anchored_left`.
            obs.sort_by_key(|o| {
                if o.clone == anchor_pos {
                    0
                } else {
                    o.clone + 1
                }
            });
            let clone_of: Vec<usize> = obs.iter().map(|o| o.clone).collect();
            let nobs = obs.len();

            // Current world estimate of the landmark and the anchored Jacobians.
            let anchor_pose = self.x.clones[anchor_pos].pose.clone();
            let x_f = anchor_pose.act(&lm.point());
            let mscobs: Vec<msckf::MscObs> = obs
                .iter()
                .map(|o| {
                    let pose = self.x.clones[o.clone].pose.clone();
                    msckf::MscObs {
                        pose: pose.clone(),
                        pose_fej: pose,
                        uv: o.uvn,
                    }
                })
                .collect();
            let (h_f, h_x, res) = match msckf::feature_jacobians_anchored_left(&x_f, &mscobs, cam) {
                Some(t) => t,
                None => continue,
            };
            let jl = landmark_chart_jacobian(&lm.q, &lm.origin);
            let jl_d = DMatrix::from_column_slice(3, 3, jl.as_slice());
            let h_l = &h_f * &jl_d; // 2N×3 chart column

            // Assemble the full measurement Jacobian H (2N × dim): clone bands into
            // their native columns (band 0 = anchor coupling already folded in by
            // `feature_jacobians_anchored_left`), and the chart column at the landmark.
            let rows = 2 * nobs;
            let dim = self.dim();
            let mut h = DMatrix::<f64>::zeros(rows, dim);
            for a in 0..nobs {
                let col = self.clone_idx(clone_of[a]);
                let src = h_x.view((0, 6 * a), (rows, 6)).into_owned();
                h.view_mut((0, col), (rows, 6)).copy_from(&src);
            }
            let lcol = self.landmark_idx(upd.j);
            h.view_mut((0, lcol), (rows, 3)).copy_from(&h_l);

            // EKF innovation covariance S = H P Hᵀ + σ²I, χ² gate on all rows.
            let g = &self.cov * h.transpose(); // dim×rows  (= P Hᵀ)
            let mut s = &h * &g; // rows×rows  (= H P Hᵀ)
            for k in 0..rows {
                s[(k, k)] += sigma2;
            }
            let s_inv = match s.clone().try_inverse() {
                Some(m) => m,
                None => continue,
            };
            let d = (res.transpose() * &s_inv * &res)[(0, 0)];
            if !d.is_finite() || d > chi2_mult * msckf::chi2_095(rows) {
                continue;
            }

            // Kalman gain and update: K = P Hᵀ S⁻¹, δ = K·res, downdate P −= K (H P).
            let chol = match s.cholesky() {
                Some(c) => c,
                None => continue,
            };
            let k = chol.solve(&g.transpose()).transpose(); // dim×rows
            let inn_vec = &k * &res; // dim
            self.apply_left_increment(&inn_vec);
            let downdate = &k * g.transpose();
            self.cov -= downdate;
            let sym = (&self.cov + self.cov.transpose()) * 0.5;
            self.cov = sym;
            accepted += 1;
        }

        accepted
    }

    /// Streaming (per-frame) in-state landmark update — the measurement-consistent
    /// OpenVINS `UpdaterSLAM` analog. Each landmark is updated with only its NEW
    /// observations (at clones OTHER than the anchor); the anchor is decoupled and
    /// enters through its representation Jacobian, so no measurement is used twice
    /// (birth already consumed the track history). This is the update the driver
    /// calls every frame for persistent features — `landmark_update` (batch, anchor
    /// re-observed) is only for one-shot re-triangulation, not streaming.
    ///
    /// For a new observation at clone `k` (≠ anchor `a`), with world point
    /// `G = anchor_pose · q·origin`:
    ///   `dz_dG = Jπ · R_kᵀ`  (2×3),
    ///   landmark chart col `H_L = dz_dG · R_a · landmark_chart_jacobian(q,origin)`,
    ///   observing-clone col `H_k = dz_dG · [ [G]× | −I ]`,
    ///   anchor col `H_a = dz_dG · [ −[G]× | I ] = −H_k`
    /// (a rigid pose applied to BOTH the anchor and the observing clone leaves the
    /// bearing invariant, so the two pose columns are negatives — the same
    /// band-0 = −clone relation `feature_jacobians_anchored_left` uses). Standard
    /// EKF with a χ² gate on the `2M` measurement rows. Returns the accepted count.
    pub fn landmark_stream_update(
        &mut self,
        updates: &[LmStreamUpdate],
        cam: &dyn CameraModel,
        pixel_std: f64,
        chi2_mult: f64,
    ) -> usize {
        let sigma2 = pixel_std * pixel_std;
        let nclones = self.x.clones.len();
        let mut accepted = 0;

        for upd in updates {
            if upd.j >= self.x.landmarks.len() {
                continue;
            }
            let lm = self.x.landmarks[upd.j].clone();
            let anchor_pos = match self.x.clones.iter().position(|c| c.stamp == lm.anchor) {
                Some(p) => p,
                None => continue, // anchor not live — must reanchor first
            };
            // Fresh observations at live, non-anchor clones (anchor never re-observed).
            let obs: Vec<MscTrackObs> = upd
                .obs
                .iter()
                .copied()
                .filter(|o| o.clone < nclones && o.clone != anchor_pos)
                .collect();
            if obs.is_empty() {
                continue;
            }

            let anchor_pose = self.x.clones[anchor_pos].pose.clone();
            let g = anchor_pose.act(&lm.point()); // world point
            let r_a = anchor_pose.rotation.as_matrix();
            // dG/dε (landmark chart) = R_a · J_lift.
            let jl = landmark_chart_jacobian(&lm.q, &lm.origin);
            let dg_deps = r_a * jl; // 3×3
            // A = [ [G]× | −I ] (3×6): observing-clone pose Jacobian factor.
            let mut a_mat = SMatrix::<f64, 3, 6>::zeros();
            a_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&g));
            a_mat
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-Matrix3::identity()));

            let m = obs.len();
            let rows = 2 * m;
            let dim = self.dim();
            let mut h = DMatrix::<f64>::zeros(rows, dim);
            let mut res = DVector::<f64>::zeros(rows);
            let lcol = self.landmark_idx(upd.j);
            let acol = self.clone_idx(anchor_pos);
            let mut ok = true;
            for (i, o) in obs.iter().enumerate() {
                let clone_pose = &self.x.clones[o.clone].pose;
                let q = clone_pose.inverse().act(&g); // point in clone-k camera frame
                if q[2] <= 1e-6 {
                    ok = false;
                    break;
                }
                let jpi = cam.projection_jacobian(&q); // 2×3
                let rt_k = clone_pose.rotation.inverse().as_matrix(); // R_kᵀ
                let dz_dg = jpi * rt_k; // 2×3
                let h_l = dz_dg * dg_deps; // 2×3 landmark chart
                let clone_col = dz_dg * a_mat; // 2×6 observing clone
                let r = o.uvn - cam.project(&q); // residual
                let ri = 2 * i;
                for a in 0..2 {
                    res[ri + a] = r[a];
                    for c in 0..3 {
                        h[(ri + a, lcol + c)] = h_l[(a, c)];
                    }
                    let kcol = self.clone_idx(o.clone);
                    for c in 0..6 {
                        h[(ri + a, kcol + c)] = clone_col[(a, c)];
                        // Anchor col = −clone col; ACCUMULATE (multiple obs share it).
                        h[(ri + a, acol + c)] -= clone_col[(a, c)];
                    }
                }
            }
            if !ok || !h.iter().all(|v| v.is_finite()) {
                continue;
            }

            // EKF innovation covariance, χ² gate, gain, update, downdate.
            let g_mat = &self.cov * h.transpose(); // dim×rows
            let mut s = &h * &g_mat;
            for k in 0..rows {
                s[(k, k)] += sigma2;
            }
            let s_inv = match s.clone().try_inverse() {
                Some(mm) => mm,
                None => continue,
            };
            let d = (res.transpose() * &s_inv * &res)[(0, 0)];
            if !d.is_finite() || d > chi2_mult * msckf::chi2_095(rows) {
                continue;
            }
            let chol = match s.cholesky() {
                Some(c) => c,
                None => continue,
            };
            let k_gain = chol.solve(&g_mat.transpose()).transpose(); // dim×rows
            let inn_vec = &k_gain * &res;
            self.apply_left_increment(&inn_vec);
            let downdate = &k_gain * g_mat.transpose();
            self.cov -= downdate;
            let sym = (&self.cov + self.cov.transpose()) * 0.5;
            self.cov = sym;
            accepted += 1;
        }

        accepted
    }

    /// Covariance-consistent anchor change — the equivariant analog of OpenVINS
    /// `UpdaterSLAM::perform_anchor_change`. Re-expresses landmark `j` relative to a
    /// new anchor clone (`new_anchor_pos`) WITHOUT changing the world point it
    /// represents, and maps the covariance through the exact change-of-variables so
    /// no information is created or lost. Run this on any in-state landmark whose
    /// anchor clone is about to marginalize, reanchoring to a clone that survives
    /// (OpenVINS reanchors to the newest clone; `change_anchors` before marg).
    ///
    /// Since the world point `G = anchor_old · q·origin` is invariant, the new
    /// anchor-frame origin is `origin_new = anchor_new⁻¹·G` (chart reset: `q_new =
    /// identity`). The landmark error maps as
    /// `δp_new = H_f_new⁻¹(H_f_old·δp_old + H_g·δa_old − H_g·δa_new)` with
    /// representation Jacobians `H_f = R_anchor·landmark_chart_jacobian` and
    /// `H_g = [−[G]× | I]` (both anchors move `G` rigidly), applied as a full-state
    /// `Φ P Φᵀ` propagation (identity except the landmark rows). Returns `false` if
    /// the target clone is not live, equals the current anchor, or the point falls
    /// behind the new anchor.
    pub fn reanchor_landmark(&mut self, j: usize, new_anchor_pos: usize) -> bool {
        if j >= self.x.landmarks.len() || new_anchor_pos >= self.x.clones.len() {
            return false;
        }
        let lm = self.x.landmarks[j].clone();
        let old_anchor_pos = match self.x.clones.iter().position(|c| c.stamp == lm.anchor) {
            Some(p) => p,
            None => return false,
        };
        if old_anchor_pos == new_anchor_pos {
            return false;
        }

        let pose_old = self.x.clones[old_anchor_pos].pose.clone();
        let pose_new = self.x.clones[new_anchor_pos].pose.clone();
        let g = pose_old.act(&lm.point()); // invariant world point
        let origin_new = pose_new.inverse().act(&g);
        if origin_new[2] <= 1e-6 {
            return false; // behind the new anchor
        }

        // Representation Jacobians ∂G/∂(·). H_f = R_anchor · ∂p̂_a/∂ε.
        let h_f_old = pose_old.rotation.as_matrix() * landmark_chart_jacobian(&lm.q, &lm.origin);
        let h_f_new =
            pose_new.rotation.as_matrix() * landmark_chart_jacobian(&SOT3::identity(), &origin_new);
        let h_f_new_inv = match h_f_new.try_inverse() {
            Some(m) => m,
            None => return false,
        };
        // H_g = ∂G/∂(anchor left-pert) = [−[G]× | I] (3×6); both anchors share it.
        let mut h_g = SMatrix::<f64, 3, 6>::zeros();
        h_g.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew(&g)));
        h_g.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());

        let phi_feat = h_f_new_inv * h_f_old; // 3×3, landmark ← old chart
        let phi_g = h_f_new_inv * h_g; // 3×6, landmark ← anchor pose

        // Full-state Φ = I except the landmark's 3 rows (chart change).
        let dim = self.dim();
        let li = self.landmark_idx(j);
        let old_col = self.clone_idx(old_anchor_pos);
        let new_col = self.clone_idx(new_anchor_pos);
        let mut phi = DMatrix::<f64>::identity(dim, dim);
        // Zero the landmark's identity diagonal, then place the change-of-variables.
        phi.view_mut((li, li), (3, 3)).fill(0.0);
        let pf = DMatrix::from_column_slice(3, 3, phi_feat.as_slice());
        phi.view_mut((li, li), (3, 3)).copy_from(&pf);
        let pg = DMatrix::from_column_slice(3, 6, phi_g.as_slice());
        phi.view_mut((li, old_col), (3, 6)).copy_from(&pg);
        phi.view_mut((li, new_col), (3, 6)).copy_from(&(-&pg));

        self.cov = &phi * &self.cov * phi.transpose();
        let sym = (&self.cov + self.cov.transpose()) * 0.5;
        self.cov = sym;

        // Reset the landmark to the new anchor (chart center = current point).
        self.x.landmarks[j].q = SOT3::identity();
        self.x.landmarks[j].origin = origin_new;
        self.x.landmarks[j].anchor = self.x.clones[new_anchor_pos].stamp;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64) -> bool {
        a.shape() == b.shape() && (a - b).amax() < tol
    }

    fn test_noise() -> ProcessNoise {
        ProcessNoise {
            angular_velocity_std: 1e-3,
            acceleration_std: 1e-2,
            angular_velocity_bias_std: 1e-4,
            acceleration_bias_std: 1e-3,
            state_transition_order: 1,
        }
    }

    fn g_enu() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, -9.81)
    }

    #[test]
    fn se23_permutation_roundtrips() {
        let u = SVector::<f64, 9>::from_iterator((0..9).map(|i| i as f64 + 1.0));
        let back = se23_incr_echo_to_msceqf(&se23_incr_msceqf_to_echo(&u));
        assert!((u - back).amax() < 1e-15);
        // att fixed, vel<->pos swapped.
        let e = se23_incr_msceqf_to_echo(&u);
        assert_eq!(e[0], 1.0); // att
        assert_eq!(e[3], 7.0); // echo pos <- MSCEqF pos (idx 6)
        assert_eq!(e[6], 4.0); // echo vel <- MSCEqF vel (idx 3)
    }

    #[test]
    fn identity_origin_transform_matches_hand_computation() {
        // The origin map D (state.cpp:48-56) is applied UNCONDITIONALLY. At
        // identity origin AdS0inv = I6, whose sub-blocks land off-diagonal in D,
        // so E (the extrinsics *symmetry*) couples to (att, pos) with unit gain.
        // With d_init=2*I, e_init=5*I: E-omega rows += att (2), E-v rows += pos
        // (2) -> diagonal 7; cross E<->att = 2. This is faithful MSCEqF behavior.
        let origin =
            SystemOrigin::new(SE23::identity(), Vector6::zeros(), SE3::identity(), g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity() * 2.0;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 3.0;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 5.0;
        let f = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise());
        assert_eq!(f.dim(), CLONE_BASE);
        // Dd block itself is untouched (D has identity on the Dd rows/cols).
        assert_eq!(f.cov[(0, 0)], 2.0);
        assert_eq!(f.cov[(9, 9)], 3.0);
        // b0 = 0 -> ad_se3(b0) = 0 -> no delta<->Dd coupling from the bias term.
        assert!(f.cov.view((DD_IDX + 9, DD_IDX), (6, 6)).amax() < 1e-12);
        // E diagonal accumulates att (omega rows) / pos (v rows) variance.
        for k in 0..3 {
            assert!((f.cov[(E_IDX + k, E_IDX + k)] - 7.0).abs() < 1e-12);
            assert!((f.cov[(E_IDX + 3 + k, E_IDX + 3 + k)] - 7.0).abs() < 1e-12);
            // Cross E-omega <-> att and E-v <-> pos are the source d-variance.
            assert!((f.cov[(E_IDX + k, DD_IDX + k)] - 2.0).abs() < 1e-12);
            assert!((f.cov[(E_IDX + 3 + k, DD_IDX + 6 + k)] - 2.0).abs() < 1e-12);
        }
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    #[test]
    fn origin_transform_is_symmetric_and_conjugation() {
        // Non-identity origin: D * cov * D^T must stay symmetric, and equal the
        // hand-built conjugation.
        let t0 = SE23::new(
            SO3::exp(&Vector3::new(0.1, -0.2, 0.3)),
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.4, -0.5, 0.6),
        );
        let b0 = Vector6::new(0.01, -0.02, 0.03, 0.1, -0.2, 0.15);
        let s0 = SE3::new(
            SO3::exp(&Vector3::new(-0.05, 0.07, 0.02)),
            Vector3::new(0.2, 0.1, -0.3),
        );
        let origin = SystemOrigin::new(t0, b0, s0, g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity() * 1.5;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 0.7;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 2.2;
        let f = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise());
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
        // A non-identity origin MUST induce Dd<->E cross-correlation.
        assert!(f.cov.view((E_IDX, DD_IDX), (6, 3)).amax() > 1e-6);
    }

    #[test]
    fn clone_shares_e_covariance() {
        let origin =
            SystemOrigin::new(SE23::identity(), Vector6::zeros(), SE3::identity(), g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity() * 2.0;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 3.0;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 5.0;
        let mut f = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise());
        f.stochastic_clone(1.0);
        assert_eq!(f.x.clones.len(), 1);
        assert_eq!(f.dim(), CLONE_BASE + CLONE_DOF);
        let cidx = f.clone_idx(0);
        // Clone diagonal == E diagonal.
        let e_diag = f.cov.view((E_IDX, E_IDX), (6, 6)).into_owned();
        let c_diag = f.cov.view((cidx, cidx), (6, 6)).into_owned();
        assert!((e_diag - c_diag).amax() < 1e-12);
        // Clone<->E cross == E diagonal (perfect correlation at birth).
        let cross = f.cov.view((cidx, E_IDX), (6, 6)).into_owned();
        assert!((cross - f.cov.view((E_IDX, E_IDX), (6, 6)).into_owned()).amax() < 1e-12);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    fn diag3(a: f64, b: f64, c: f64) -> SMatrix<f64, LM_DOF, LM_DOF> {
        let mut p = SMatrix::<f64, LM_DOF, LM_DOF>::zeros();
        p[(0, 0)] = a;
        p[(1, 1)] = b;
        p[(2, 2)] = c;
        p
    }

    fn dummy_landmark(anchor: f64) -> Landmark {
        Landmark {
            q: SOT3::new(SO3::identity(), 1.0),
            anchor,
            origin: Vector3::new(0.0, 0.0, 1.0),
        }
    }

    #[test]
    fn insert_landmark_appends_at_tail() {
        let mut f = hover_filter();
        f.stochastic_clone(1.0);
        let base = f.landmark_base();
        assert_eq!(base, CLONE_BASE + CLONE_DOF); // one clone
        let j = f.insert_landmark(dummy_landmark(1.0), &diag3(7.0, 8.0, 9.0), None) as usize;
        assert_eq!(j, 0);
        assert_eq!(f.n_landmarks(), 1);
        assert_eq!(f.dim(), base + LM_DOF);
        assert_eq!(f.landmark_idx(0), base);
        // Block preserved, zero cross to sensor/clone, still symmetric.
        assert!((f.cov[(base, base)] - 7.0).abs() < 1e-12);
        assert!((f.cov[(base + 2, base + 2)] - 9.0).abs() < 1e-12);
        assert!(f.cov.view((0, base), (base, LM_DOF)).amax() < 1e-12);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    #[test]
    fn clone_birth_shifts_landmark_tail() {
        // A landmark born after one clone must slide down by CLONE_DOF when the
        // next clone is inserted at the sensor|clones ↔ landmarks split, and its
        // covariance (block AND cross to E) must be carried intact.
        let mut f = hover_filter();
        f.stochastic_clone(1.0);
        let l0 = f.landmark_idx(0); // insertion target (base, one clone)
        f.insert_landmark(dummy_landmark(1.0), &diag3(7.0, 8.0, 9.0), None);
        // Plant a recognizable landmark↔E cross by hand.
        let l0 = {
            let _ = l0;
            f.landmark_idx(0)
        };
        f.cov[(l0, E_IDX)] = 0.5;
        f.cov[(E_IDX, l0)] = 0.5;

        f.stochastic_clone(2.0);
        assert_eq!(f.x.clones.len(), 2);
        let l1 = f.landmark_idx(0);
        assert_eq!(l1, l0 + CLONE_DOF); // shifted down by the new clone block
        // Landmark block intact at the new location.
        assert!((f.cov[(l1, l1)] - 7.0).abs() < 1e-12);
        assert!((f.cov[(l1 + 2, l1 + 2)] - 9.0).abs() < 1e-12);
        // The planted cross to E (E is before the split, index unchanged) survived.
        assert!((f.cov[(l1, E_IDX)] - 0.5).abs() < 1e-12);
        // New clone block sits at clone position 1 with the E diagonal.
        let c1 = f.clone_idx(1);
        let e_diag = f.cov.view((E_IDX, E_IDX), (6, 6)).into_owned();
        assert!((f.cov.view((c1, c1), (6, 6)).into_owned() - e_diag).amax() < 1e-12);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    #[test]
    fn marginalize_landmark_removes_block() {
        let mut f = hover_filter();
        f.stochastic_clone(1.0);
        f.insert_landmark(dummy_landmark(1.0), &diag3(7.0, 7.0, 7.0), None);
        f.insert_landmark(dummy_landmark(1.0), &diag3(9.0, 9.0, 9.0), None);
        let dim0 = f.dim();
        f.marginalize_landmark(0); // drop the first
        assert_eq!(f.n_landmarks(), 1);
        assert_eq!(f.dim(), dim0 - LM_DOF);
        // The surviving (second) landmark's block is now at position 0.
        let idx = f.landmark_idx(0);
        assert!((f.cov[(idx, idx)] - 9.0).abs() < 1e-12);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    #[test]
    fn marginalize_clone_compacts_landmark_tail() {
        // Marginalizing a clone must shift the landmark tail down (reusing the
        // block-agnostic keep-filter) and preserve landmark covariance.
        let mut f = hover_filter();
        f.stochastic_clone(1.0);
        f.stochastic_clone(2.0);
        f.insert_landmark(dummy_landmark(2.0), &diag3(5.0, 6.0, 7.0), None);
        let before = f.landmark_idx(0);
        f.marginalize_clone(0); // drop oldest clone
        assert_eq!(f.x.clones.len(), 1);
        let after = f.landmark_idx(0);
        assert_eq!(after, before - CLONE_DOF);
        assert!((f.cov[(after, after)] - 5.0).abs() < 1e-12);
        assert!((f.cov[(after + 2, after + 2)] - 7.0).abs() < 1e-12);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }

    fn hover_filter() -> MSCEqFFilter {
        let origin =
            SystemOrigin::new(SE23::identity(), Vector6::zeros(), SE3::identity(), g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity() * 1e-4;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 1e-4;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise())
    }

    #[test]
    fn discrete_order1_is_identity_plus_a_dt() {
        let f = hover_filter();
        let u = Imu {
            ang: Vector3::new(0.0, 0.0, 0.0),
            acc: Vector3::new(0.0, 0.0, 9.81),
        };
        let dt = 0.005;
        let a = f.state_matrix(&u);
        let b = f.input_matrix();
        let (phi, _qd) = f.discrete(&a, &b, dt);
        let expected = SMatrix::<f64, 21, 21>::identity() + a * dt;
        assert!((phi - expected).amax() < 1e-12);
    }

    #[test]
    fn state_matrix_structure_at_identity() {
        // Identity state + identity origin, b0=0. R0Tg = g. Check the hand blocks.
        let f = hover_filter();
        let u = Imu {
            ang: Vector3::zeros(),
            acc: Vector3::new(0.0, 0.0, 9.81),
        };
        let a = f.state_matrix(&u);
        // A[att, delta] = I6 (A2, top-left 6x6 of the delta block).
        let a2 = a.fixed_view::<6, 6>(DD_IDX, DD_IDX + 9).into_owned();
        assert!((a2 - SMatrix::<f64, 6, 6>::identity()).amax() < 1e-12);
        // A[pos, vel] = I3 (A1 lower).
        let a1 = a.fixed_view::<3, 3>(DD_IDX + 6, DD_IDX + 3).into_owned();
        assert!((a1 - nalgebra::Matrix3::identity()).amax() < 1e-12);
        // A1[att:vel block] = Psi (since adb0=0): only [vel,att]=wedge(g) nonzero.
        let psi_block = a.fixed_view::<3, 3>(DD_IDX + 3, DD_IDX).into_owned();
        assert!((psi_block - skew(&f.origin.g)).amax() < 1e-12);
        // A[att,att] = 0.
        assert!(a.fixed_view::<3, 3>(DD_IDX, DD_IDX).amax() < 1e-12);
    }

    #[test]
    fn hover_keeps_state_at_origin() {
        // Feed IMU that exactly cancels gravity (acc = -R^T g at identity).
        // lambda must be ~0, so phi stays at the origin: no drift.
        let mut f = hover_filter();
        let u = Imu {
            ang: Vector3::zeros(),
            acc: Vector3::new(0.0, 0.0, 9.81),
        };
        let dt = 0.01;
        for _ in 0..2000 {
            f.propagate_mean(&u, dt);
        }
        let nav = f.phi();
        assert!(
            nav.t.position.norm() < 1e-9,
            "pos drift {}",
            nav.t.position.norm()
        );
        assert!(
            nav.t.velocity.norm() < 1e-9,
            "vel drift {}",
            nav.t.velocity.norm()
        );
        assert!(nav.t.rotation.log().norm() < 1e-9);
        // Bias unchanged.
        assert!(nav.b.norm() < 1e-12);
    }

    #[test]
    fn covariance_propagation_symmetric_and_grows() {
        let mut f = hover_filter();
        let u = Imu {
            ang: Vector3::new(0.01, -0.02, 0.005),
            acc: Vector3::new(0.1, 0.0, 9.81),
        };
        let tr0 = f.cov.view((0, 0), (CLONE_BASE, CLONE_BASE)).trace();
        for _ in 0..50 {
            f.propagate_covariance(&u, 0.01);
        }
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-10));
        let tr1 = f.cov.view((0, 0), (CLONE_BASE, CLONE_BASE)).trace();
        assert!(tr1 > tr0, "trace should grow: {tr0} -> {tr1}");
    }

    #[test]
    fn clone_cross_covariance_propagates() {
        // With a clone present, propagation must rotate its cross-block by Phi and
        // keep the clone's own diagonal frozen.
        let mut f = hover_filter();
        f.stochastic_clone(0.0);
        let cidx = f.clone_idx(0);
        let clone_diag_before = f.cov.view((cidx, cidx), (6, 6)).into_owned();
        let u = Imu {
            ang: Vector3::new(0.02, 0.01, -0.03),
            acc: Vector3::new(0.0, 0.2, 9.81),
        };
        f.propagate_covariance(&u, 0.01);
        let clone_diag_after = f.cov.view((cidx, cidx), (6, 6)).into_owned();
        assert!(
            (clone_diag_before - clone_diag_after).amax() < 1e-12,
            "clone diagonal frozen"
        );
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-10));
    }

    fn norm_cam() -> crate::mathematical::camera::PinholeModel {
        // Z1 normalized model: project(q) = (q.x/q.z, q.y/q.z).
        crate::mathematical::camera::PinholeModel {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        }
    }

    /// Build a filter with three clones at distinct world<-camera poses and a
    /// track of one world point observed by all three (exact, noise-free obs).
    fn scene_filter() -> (MSCEqFFilter, MscTrack, Vector3<f64>) {
        let origin =
            SystemOrigin::new(SE23::identity(), Vector6::zeros(), SE3::identity(), g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity() * 1e-2;
        let delta_init = SMatrix::<f64, 6, 6>::identity() * 1e-2;
        let e_init = SMatrix::<f64, 6, 6>::identity() * 1e-3;
        let mut f = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise());
        let poses = [
            SE3::new(SO3::identity(), Vector3::new(0.0, 0.0, 0.0)),
            SE3::new(
                SO3::exp(&Vector3::new(0.0, 0.03, 0.0)),
                Vector3::new(1.0, 0.0, 0.0),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(0.0, 0.06, 0.0)),
                Vector3::new(2.0, 0.1, -0.2),
            ),
        ];
        // Propagate covariance between clones so they DECORRELATE (inject process
        // noise). Without this the clones are born as perfect copies of E, the
        // clone-tail covariance is rank-6, and the structureless constraint (which
        // sees only relative pose) has zero leverage — a correct no-op, but a
        // useless test. A few IMU steps give the clone tail full-rank structure.
        let u = Imu {
            ang: Vector3::new(0.02, -0.01, 0.03),
            acc: Vector3::new(0.1, -0.2, 9.81),
        };
        for (i, p) in poses.iter().enumerate() {
            f.stochastic_clone(i as f64);
            let last = f.x.clones.len() - 1;
            f.x.clones[last].pose = p.clone();
            for _ in 0..20 {
                f.propagate_covariance(&u, 0.01);
            }
        }
        let cam = norm_cam();
        let x_f = Vector3::new(0.5, -0.3, 5.0);
        let obs: Vec<MscTrackObs> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let q = p.inverse().act(&x_f);
                MscTrackObs {
                    clone: i,
                    uvn: cam.project(&q),
                }
            })
            .collect();
        (f, MscTrack { obs }, x_f)
    }

    #[test]
    fn msc_update_perfect_track_downdates_without_moving_state() {
        let (mut f, track, _x_f) = scene_filter();
        let cam = norm_cam();
        let tr_before = f.cov.trace();
        let sdb_before = f.x.sdb.clone();
        let accepted = f.msc_update(&[track], &cam, 1e-3, 1.0, false);
        assert_eq!(accepted, 1, "exact track must pass the χ² gate");
        // Perfect (noise-free) observations ⇒ residual ≈ 0 ⇒ state barely moves.
        let dd = f.x.sdb.inverse().compose(&sdb_before);
        assert!(
            SemiDirectBias::log(&dd).norm() < 1e-6,
            "state should not move on zero residual"
        );
        // Downdate still reduces uncertainty and keeps symmetry.
        assert!(
            f.cov.trace() < tr_before,
            "covariance must shrink: {} -> {}",
            tr_before,
            f.cov.trace()
        );
        assert!((f.cov.clone() - f.cov.transpose()).amax() < 1e-9);
    }

    #[test]
    fn msc_update_curvature_keeps_symmetry() {
        let (mut f, track, _x_f) = scene_filter();
        let cam = norm_cam();
        let accepted = f.msc_update(&[track], &cam, 1e-3, 1.0, true);
        assert_eq!(accepted, 1);
        assert!(
            (f.cov.clone() - f.cov.transpose()).amax() < 1e-8,
            "curvature-corrected cov symmetric"
        );
        assert!(f.cov.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn msc_update_rejects_outlier_track() {
        let (mut f, mut track, _x_f) = scene_filter();
        let cam = norm_cam();
        // Corrupt a SINGLE observation by a large offset: no 3-D point can explain
        // it, so re-triangulation cannot absorb it and the residual stays large.
        // (A constant offset on ALL obs would just shift the triangulated point.)
        track.obs[1].uvn += Vector2::new(0.5, -0.4);
        let sdb_before = f.x.sdb.clone();
        let accepted = f.msc_update(&[track], &cam, 1e-3, 1.0, false);
        assert_eq!(
            accepted, 0,
            "grossly inconsistent track must be χ²-rejected"
        );
        // Rejected update leaves the state untouched.
        let dd = f.x.sdb.inverse().compose(&sdb_before);
        assert!(SemiDirectBias::log(&dd).norm() < 1e-12);
    }

    #[test]
    fn birth_landmark_augments_correlated_and_recovers_geometry() {
        let (mut f, track, x_f) = scene_filter();
        let cam = norm_cam();
        let dim_before = f.dim();
        let j = f
            .birth_landmark(&track, &cam, 1e-3, 1.0)
            .expect("exact window-spanning track must birth a landmark");
        assert_eq!(j, 0);
        assert_eq!(f.n_landmarks(), 1);
        assert_eq!(f.dim(), dim_before + LM_DOF);

        // Geometry recovered: reconstructed anchor-frame point reprojects onto every
        // observation (anchor = clone 0). The born point is in the anchor frame.
        let anchor = f.x.clones[0].pose.clone();
        let pt_anchor = f.x.landmarks[0].point();
        let world = anchor.act(&pt_anchor);
        assert!(
            (world - x_f).norm() < 1e-6,
            "world point {world:?} vs {x_f:?}"
        );
        for o in &track.obs {
            let q = f.x.clones[o.clone].pose.inverse().act(&world);
            assert!(
                (cam.project(&q) - o.uvn).norm() < 1e-6,
                "reprojection mismatch"
            );
        }

        // P_LL is a proper (positive-diagonal) block; global cov stays symmetric.
        let li = f.landmark_idx(0);
        for d in 0..LM_DOF {
            assert!(
                f.cov[(li + d, li + d)] > 0.0,
                "P_LL diagonal must be positive"
            );
        }
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-9));

        // Correlated birth: cross-cov to the clone tail is NOT all zero (this is the
        // whole point of delayed init over a guessed-diagonal insert).
        let nc = CLONE_DOF * f.x.clones.len();
        let cross = f.cov.view((li, CLONE_BASE), (LM_DOF, nc)).into_owned();
        assert!(
            cross.amax() > 1e-9,
            "landmark must be born correlated to clones"
        );
    }

    #[test]
    fn birth_landmark_rejects_outlier_track() {
        let (mut f, mut track, _x_f) = scene_filter();
        let cam = norm_cam();
        // One grossly inconsistent observation: no 3-D point explains it, so the
        // update-row χ² gate must reject the birth and leave the state unaugmented.
        track.obs[1].uvn += Vector2::new(0.5, -0.4);
        let dim_before = f.dim();
        let out = f.birth_landmark(&track, &cam, 1e-3, 1.0);
        assert!(out.is_none(), "inconsistent track must be χ²-rejected");
        assert_eq!(f.n_landmarks(), 0);
        assert_eq!(f.dim(), dim_before);
    }

    #[test]
    fn landmark_update_pulls_estimate_toward_truth() {
        // Sign-adjudicating test (birth can't do this — its res_init ≈ 0). Birth an
        // exact landmark, then PERTURB its mean off truth. Feeding exact observations
        // of the true point must pull the reconstruction back toward truth. A flipped
        // chart-column sign (or wrong cross-cov sign) would push it AWAY.
        let (mut f, track, x_f) = scene_filter();
        let cam = norm_cam();
        let j = f
            .birth_landmark(&track, &cam, 1e-3, 1.0)
            .expect("exact track births a landmark");

        // Nudge the landmark mean through its own chart so point() is wrong.
        let origin = f.x.landmarks[j].origin;
        let delta = Vector3::new(0.005, -0.004, 0.008);
        f.x.landmarks[j].q = sot3_left_increment_invdepth(&SOT3::identity(), &origin, &delta);

        // Isolate the landmark with a loose, decoupled prior so the single EKF step is
        // well-scaled (≈ Gauss-Newton) rather than an overshoot from res ≫ √S. This
        // cleanly adjudicates the chart-column SIGN: correct sign lands near truth,
        // a flipped sign reflects the estimate PAST truth (error grows).
        let li = f.landmark_idx(j);
        let dim = f.dim();
        for d in 0..LM_DOF {
            for r in 0..dim {
                f.cov[(r, li + d)] = 0.0;
                f.cov[(li + d, r)] = 0.0;
            }
            f.cov[(li + d, li + d)] = 0.1;
        }

        let world_err = |f: &MSCEqFFilter| -> f64 {
            let anchor = f.x.clones[0].pose.clone();
            (anchor.act(&f.x.landmarks[j].point()) - x_f).norm()
        };
        let before = world_err(&f);
        assert!(
            before > 1e-3,
            "perturbation should displace the estimate: {before:.3e}"
        );

        // Exact observations of the TRUE point; loose χ² gate (deliberate mean error).
        let upd = LmUpdate {
            j,
            obs: track.obs.clone(),
        };
        let n = f.landmark_update(&[upd], &cam, 1e-3, 1e12);
        assert_eq!(n, 1, "consistent update must be accepted");

        let after = world_err(&f);
        assert!(
            after < 0.5 * before,
            "update must pull the landmark toward truth: {before:.3e} -> {after:.3e}"
        );
        assert!(
            (f.cov.clone() - f.cov.transpose()).amax() < 1e-9,
            "cov stays symmetric"
        );
        for d in 0..LM_DOF {
            assert!(
                f.cov[(li + d, li + d)] > 0.0,
                "P_LL diagonal stays positive"
            );
        }
    }

    #[test]
    fn landmark_update_rejects_outlier_and_leaves_state() {
        // A grossly inconsistent observation must be χ²-rejected, leaving the landmark
        // mean and covariance untouched.
        let (mut f, mut track, _x_f) = scene_filter();
        let cam = norm_cam();
        let j = f.birth_landmark(&track, &cam, 1e-3, 1.0).expect("birth");
        let pt_before = f.x.landmarks[j].point();
        let tr_before = f.cov.trace();

        track.obs[1].uvn += Vector2::new(0.5, -0.4); // no 3-D point explains it
        let upd = LmUpdate {
            j,
            obs: track.obs.clone(),
        };
        let n = f.landmark_update(&[upd], &cam, 1e-3, 1.0);
        assert_eq!(n, 0, "grossly inconsistent update must be χ²-rejected");
        assert!(
            (f.x.landmarks[j].point() - pt_before).norm() < 1e-12,
            "mean untouched"
        );
        assert!(
            (f.cov.trace() - tr_before).abs() < 1e-12,
            "covariance untouched"
        );
    }

    #[test]
    fn landmark_stream_update_pulls_toward_truth_with_decoupled_anchor() {
        // Streaming update: the anchor is NOT re-observed. Birth an exact landmark
        // (anchor = clone 0), perturb its mean, then feed ONE fresh observation at a
        // NON-anchor clone of the true point. A correct chart/anchor-column sign pulls
        // the estimate toward truth; a flipped sign pushes it away.
        let (mut f, track, x_f) = scene_filter();
        let cam = norm_cam();
        let j = f.birth_landmark(&track, &cam, 1e-3, 1.0).expect("birth");
        assert_eq!(f.x.landmarks[j].anchor, f.x.clones[0].stamp);

        let origin = f.x.landmarks[j].origin;
        f.x.landmarks[j].q = sot3_left_increment_invdepth(
            &SOT3::identity(),
            &origin,
            &Vector3::new(0.005, -0.004, 0.008),
        );

        // Isolate with a loose decoupled prior (≈ Gauss-Newton single step).
        let li = f.landmark_idx(j);
        let dim = f.dim();
        for d in 0..LM_DOF {
            for r in 0..dim {
                f.cov[(r, li + d)] = 0.0;
                f.cov[(li + d, r)] = 0.0;
            }
            f.cov[(li + d, li + d)] = 0.1;
        }
        let world_err = |f: &MSCEqFFilter| -> f64 {
            (f.x.clones[0].pose.act(&f.x.landmarks[j].point()) - x_f).norm()
        };
        let before = world_err(&f);

        // A single NEW observation at clone 1 (≠ anchor clone 0).
        let obs1 = vec![track.obs[1]]; // clone 1, exact bearing of the true point
        let upd = LmStreamUpdate { j, obs: obs1 };
        let n = f.landmark_stream_update(&[upd], &cam, 1e-3, 1e12);
        assert_eq!(n, 1, "single fresh observation must be accepted");
        assert!(
            world_err(&f) < 0.5 * before,
            "streaming update must pull toward truth: {before:.3e} -> {:.3e}",
            world_err(&f)
        );
        assert!(
            (f.cov.clone() - f.cov.transpose()).amax() < 1e-9,
            "cov symmetric"
        );
        for d in 0..LM_DOF {
            assert!(f.cov[(li + d, li + d)] > 0.0, "P_LL diagonal positive");
        }
    }

    #[test]
    fn landmark_stream_update_ignores_anchor_only_and_rejects_outlier() {
        let (mut f, mut track, _x_f) = scene_filter();
        let cam = norm_cam();
        let j = f.birth_landmark(&track, &cam, 1e-3, 1.0).expect("birth");
        let pt_before = f.x.landmarks[j].point();
        let tr_before = f.cov.trace();

        // Anchor-only observation (clone 0): filtered out ⇒ no-op.
        let anchor_only = LmStreamUpdate {
            j,
            obs: vec![track.obs[0]],
        };
        assert_eq!(
            f.landmark_stream_update(&[anchor_only], &cam, 1e-3, 1e12),
            0
        );
        assert!(
            (f.x.landmarks[j].point() - pt_before).norm() < 1e-12,
            "no-op on anchor-only"
        );
        assert!(
            (f.cov.trace() - tr_before).abs() < 1e-12,
            "cov untouched on anchor-only"
        );

        // Grossly inconsistent fresh obs at clone 1 ⇒ χ²-rejected.
        track.obs[1].uvn += Vector2::new(0.5, -0.4);
        let bad = LmStreamUpdate {
            j,
            obs: vec![track.obs[1]],
        };
        assert_eq!(
            f.landmark_stream_update(&[bad], &cam, 1e-3, 1.0),
            0,
            "outlier rejected"
        );
        assert!(
            (f.x.landmarks[j].point() - pt_before).norm() < 1e-12,
            "mean untouched"
        );
        assert!((f.cov.trace() - tr_before).abs() < 1e-12, "cov untouched");
    }

    #[test]
    fn reanchor_landmark_preserves_world_point_and_stays_consistent() {
        // Change-of-anchor (OV `perform_anchor_change` analog) must be a pure
        // change-of-variables: the WORLD point the landmark represents is invariant,
        // the covariance stays symmetric + PSD, and an update from the NEW anchor
        // still pulls the estimate toward truth.
        let (mut f, track, x_f) = scene_filter();
        let cam = norm_cam();
        let j = f.birth_landmark(&track, &cam, 1e-3, 1.0).expect("birth");
        assert_eq!(
            f.x.landmarks[j].anchor, f.x.clones[0].stamp,
            "born on clone 0"
        );

        // World point before reanchor (anchor = clone 0).
        let g_before = f.x.clones[0].pose.act(&f.x.landmarks[j].point());

        // Reanchor from clone 0 to clone 2.
        assert!(
            f.reanchor_landmark(j, 2),
            "reanchor to a live, distinct clone"
        );
        assert_eq!(
            f.x.landmarks[j].anchor, f.x.clones[2].stamp,
            "anchor now clone 2"
        );

        // World point is invariant; reconstructed via the NEW anchor.
        let g_after = f.x.clones[2].pose.act(&f.x.landmarks[j].point());
        assert!(
            (g_after - g_before).norm() < 1e-9,
            "world point must be invariant: {g_before:?} -> {g_after:?}"
        );

        // Covariance stays symmetric and PSD (Cholesky succeeds).
        assert!(
            (f.cov.clone() - f.cov.transpose()).amax() < 1e-9,
            "cov stays symmetric"
        );
        assert!(
            nalgebra::Cholesky::new(f.cov.clone()).is_some(),
            "reanchored covariance must stay PSD"
        );

        // No-op guards: same anchor and out-of-range clone are rejected.
        assert!(
            !f.reanchor_landmark(j, 2),
            "reanchor to the current anchor is a no-op"
        );
        assert!(!f.reanchor_landmark(j, 99), "out-of-range clone rejected");

        // The reanchored landmark is still usable: perturb its mean and update from
        // the new anchor (isolated loose prior so the single EKF step is well-scaled).
        let origin = f.x.landmarks[j].origin;
        f.x.landmarks[j].q = sot3_left_increment_invdepth(
            &SOT3::identity(),
            &origin,
            &Vector3::new(0.004, -0.003, 0.006),
        );
        let li = f.landmark_idx(j);
        let dim = f.dim();
        for d in 0..LM_DOF {
            for r in 0..dim {
                f.cov[(r, li + d)] = 0.0;
                f.cov[(li + d, r)] = 0.0;
            }
            f.cov[(li + d, li + d)] = 0.1;
        }
        let world_err = |f: &MSCEqFFilter| -> f64 {
            (f.x.clones[2].pose.act(&f.x.landmarks[j].point()) - x_f).norm()
        };
        let before = world_err(&f);
        let upd = LmUpdate {
            j,
            obs: track.obs.clone(),
        };
        let n = f.landmark_update(&[upd], &cam, 1e-3, 1e12);
        assert_eq!(n, 1, "update from the new anchor must be accepted");
        assert!(
            world_err(&f) < 0.5 * before,
            "update from new anchor must pull toward truth: {before:.3e} -> {:.3e}",
            world_err(&f)
        );
    }

    #[test]
    fn propagation_leaves_landmark_block_static() {
        // A born landmark lives in the propagation "tail" (Φ=I, no injection), so
        // its diagonal block must be frozen across a propagation step, and the cov
        // must stay symmetric/finite with the state dim unchanged.
        let (mut f, track, _x_f) = scene_filter();
        let cam = norm_cam();
        f.birth_landmark(&track, &cam, 1e-3, 1.0).expect("births");
        let li = f.landmark_idx(0);
        let lm_diag_before = f.cov.view((li, li), (LM_DOF, LM_DOF)).into_owned();
        let dim_before = f.dim();
        let u = Imu {
            ang: Vector3::new(0.02, -0.01, 0.03),
            acc: Vector3::new(0.1, -0.2, 9.81),
        };
        f.propagate_covariance(&u, 0.01);
        assert_eq!(f.dim(), dim_before);
        let lm_diag_after = f.cov.view((li, li), (LM_DOF, LM_DOF)).into_owned();
        assert!(
            (lm_diag_before - lm_diag_after).amax() < 1e-12,
            "landmark diagonal must be frozen under propagation"
        );
        assert!(f.cov.iter().all(|v| v.is_finite()));
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-10));
    }

    #[test]
    fn marginalize_removes_oldest() {
        let origin =
            SystemOrigin::new(SE23::identity(), Vector6::zeros(), SE3::identity(), g_enu());
        let d_init = SMatrix::<f64, 9, 9>::identity();
        let delta_init = SMatrix::<f64, 6, 6>::identity();
        let e_init = SMatrix::<f64, 6, 6>::identity();
        let mut f = MSCEqFFilter::new(origin, &d_init, &delta_init, &e_init, test_noise());
        f.stochastic_clone(1.0);
        f.stochastic_clone(2.0);
        assert_eq!(f.dim(), CLONE_BASE + 2 * CLONE_DOF);
        let i = f.oldest_clone().unwrap();
        assert_eq!(f.x.clones[i].stamp, 1.0);
        f.marginalize_clone(i);
        assert_eq!(f.x.clones.len(), 1);
        assert_eq!(f.x.clones[0].stamp, 2.0);
        assert_eq!(f.dim(), CLONE_BASE + CLONE_DOF);
        assert!(approx(&f.cov, &f.cov.transpose(), 1e-12));
    }
}
