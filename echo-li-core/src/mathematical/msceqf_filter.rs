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
use echo_lie::{SE3, SE23, SO3, SemiDirectBias};
use nalgebra::{DMatrix, DVector, SMatrix, SVector, Vector2, Vector3, Vector6};

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
/// Starting index of the first clone (before any clones this is the tail).
pub const CLONE_BASE: usize = DD_DOF + E_DOF;

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
    /// `std::map<fp, ...>` iteration and the covariance tail layout).
    pub clones: Vec<Clone>,
}

/// One stochastic clone: a frozen copy of `E` tagged by capture time.
#[derive(Debug, Clone)]
pub struct Clone {
    pub stamp: f64,
    pub pose: SE3,
}

impl StateGroup {
    pub fn identity() -> Self {
        Self {
            sdb: SemiDirectBias::identity(),
            e: SE3::identity(),
            clones: Vec::new(),
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
    /// Total covariance dimension for the current number of clones.
    pub fn dim(&self) -> usize {
        CLONE_BASE + CLONE_DOF * self.x.clones.len()
    }

    /// Covariance index of the clone at position `i` (0 = oldest).
    pub fn clone_idx(&self, i: usize) -> usize {
        CLONE_BASE + CLONE_DOF * i
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
        let mut cov = self
            .cov
            .clone()
            .resize(old + CLONE_DOF, old + CLONE_DOF, 0.0);

        // Diagonal: clone block = E block.
        let e_diag = self.cov.view((E_IDX, E_IDX), (E_DOF, E_DOF)).into_owned();
        cov.view_mut((old, old), (E_DOF, E_DOF)).copy_from(&e_diag);
        // Cross columns: new tail column = E column, over the old range.
        let e_col = self.cov.view((0, E_IDX), (old, E_DOF)).into_owned();
        cov.view_mut((0, old), (old, E_DOF)).copy_from(&e_col);
        // Cross rows: symmetric.
        let e_row = self.cov.view((E_IDX, 0), (E_DOF, old)).into_owned();
        cov.view_mut((old, 0), (E_DOF, old)).copy_from(&e_row);

        self.cov = cov;
        self.x.clones.push(Clone {
            stamp,
            pose: self.x.e.clone(),
        });
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
