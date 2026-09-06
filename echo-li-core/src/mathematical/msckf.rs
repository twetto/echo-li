//! Structureless multi-state-constraint (MSCKF) vision update helpers.
//!
//! Mirrors OpenVINS `UpdaterHelper` (`open_vins/ov_msckf/src/update/`): a track
//! observed by several pose clones is triangulated, the feature error is projected
//! out of the constraint (left-nullspace), and the surviving residual — now a pure
//! function of the CLONE POSES — corrects the nav+clone state through the shared
//! covariance. Sparse3D keeps its own per-landmark depth bank; this update touches
//! only the EqF pose window.
//!
//! ## Conventions (must match the rest of ECHO-LI)
//! - Clone pose `T` is world←camera (SE3), right-perturbed in the CAMERA frame:
//!   `T(δ) = T·exp(δ^)`, tangent `δ = [ω; v]` (rotation-first) — the exact
//!   convention `sparse_camera_pose_jacobian` / `sparse_relative_pose_covariance`
//!   and the clone covariance block already live in.
//! - Measurement residual is `z − h` (measured − predicted), and every Jacobian is
//!   `∂h/∂(·)` (of the PREDICTION), so the stacked KF update `γ = Σ Cᵀ S⁻¹ (z−h)`
//!   estimates the error in the same coordinates — identical sign semantics to the
//!   existing bearing update in `perform_stacked_update`.
//!
//! With `q = T⁻¹ X_f` the feature in the clone camera frame and `Jπ = ∂π/∂q` the
//! camera projection Jacobian (`cam.projection_jacobian`):
//! - `H_f = Jπ · Rᵀ`                (2×3, feature Euclidean error; `R = T.rotation`)
//! - `H_x = Jπ · [ [q]× | −I ]`     (2×6, clone right-perturbation `δ = [ω;v]`)
//! because a right perturbation moves the point as `q(δ) ≈ q + [q]× ω − v`.

use echo_lie::SE3;
use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3};

use crate::mathematical::camera::CameraModel;

/// One observation of a track: the observing clone's world←camera pose and the
/// measured (undistorted-domain) pixel.
///
/// `pose` is the CURRENT clone estimate (used for triangulation and the residual);
/// `pose_fej` is the FIRST-ESTIMATE clone pose (frozen at clone birth) used as the
/// Jacobian linearization point when first-estimate Jacobians (FEJ) are enabled.
/// With FEJ OFF the caller sets `pose_fej == pose`, so [`feature_jacobians`] is
/// byte-identical to the plain current-estimate linearization (the no-op guarantee).
pub struct MscObs {
    pub pose: SE3,
    pub pose_fej: SE3,
    pub uv: Vector2<f64>,
}

/// Skew-symmetric matrix `[v]×`.
#[inline]
fn skew3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// Triangulation quality gates, mirroring OpenVINS `FeatureInitializerOptions`
/// (`open_vins/ov_core/src/feat/FeatureInitializerOptions.h`). Without these an
/// ill-conditioned track (tiny baseline / far point / rank-deficient ray system)
/// still triangulates — to a point with huge reprojection error — and a loose
/// clone covariance lets it pass the χ² gate, so the update fits garbage and
/// corrupts the nav state. A correct MSCKF rejects the triangulation, not the
/// residual. The two SCALE-FREE gates (condition number, baseline ratio) are the
/// principled ones; `max_dist` is scene-specific and off by default (MidAir is
/// aerial with legitimately far structure). Env-overridable for tuning:
/// `ECHO_MSC_MAXCOND`, `ECHO_MSC_MINDIST`, `ECHO_MSC_MAXDIST`, `ECHO_MSC_MAXBASE`.
struct TriGates {
    max_cond: f64,
    min_dist: f64,
    max_dist: f64,
    max_baseline: f64,
    max_rms_px: f64,
}

impl TriGates {
    fn from_env() -> Self {
        let g = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(d)
        };
        TriGates {
            max_cond: g("ECHO_MSC_MAXCOND", 10_000.0), // OpenVINS max_cond_number
            min_dist: g("ECHO_MSC_MINDIST", 0.10),     // OpenVINS min_dist
            max_dist: g("ECHO_MSC_MAXDIST", f64::INFINITY), // OpenVINS 60 (scene-specific; off here)
            max_baseline: g("ECHO_MSC_MAXBASE", 40.0), // OpenVINS max_baseline (range/perp-baseline)
            // Post-triangulation reprojection-RMS gate (px). NOT in OpenVINS —
            // its clone poses/cov are self-consistent so triangulations fit and its
            // χ² gate suffices. ECHO-LI's EqVIO clones are mutually inconsistent, so
            // a track can fit to many px yet be admitted by a loose-cov χ². This
            // rejects geometrically inconsistent tracks the χ² gate misses. Off
            // (∞) by default; the reprojection error is what actually discriminates
            // a corrupting update (9 px) from a corrective one (1.5 px) here.
            max_rms_px: g("ECHO_MSC_MAXRMS", f64::INFINITY),
        }
    }
}

/// Multi-view triangulation of a track into a WORLD 3-D point.
///
/// Linear midpoint seed (intersection of bearing rays) followed by a few
/// Gauss-Newton reprojection-error iterations (`single_triangulation` +
/// `single_gaussnewton` in OpenVINS). Returns `None` if the geometry is
/// degenerate (rank-deficient ray system, non-finite, the point lands behind any
/// observing camera) OR fails the OpenVINS-mirrored quality gates ([`TriGates`]:
/// condition number, min depth, baseline ratio) — the latter is what stops
/// low-parallax/far tracks from corrupting the update.
pub fn triangulate(obs: &[MscObs], cam: &dyn CameraModel) -> Option<Vector3<f64>> {
    if obs.len() < 2 {
        return None;
    }
    let gates = TriGates::from_env();
    // TRIFAIL (ECHO_MSC_TRIFAIL=1): log which gate rejects a triangulation, to
    // localize why echo drops short low-parallax tracks MSCEqF keeps.
    let trifail = std::env::var("ECHO_MSC_TRIFAIL").as_deref() == Ok("1");
    macro_rules! fail {
        ($reason:expr) => {{
            if trifail {
                eprintln!("TRIFAIL nobs={} reason={}", obs.len(), $reason);
            }
            return None;
        }};
    }

    // ECHO_MSC_TRI_INVDEPTH=1 mirrors MSCEqF triangulation (linearTriangulation +
    // nonlinearTriangulation): un-normalized bearing weighting in the seed AND an
    // inverse-depth GN refine below. Default (unset) keeps echo's unit-ray seed +
    // Euclidean GN.
    let use_invdepth = std::env::var("ECHO_MSC_TRI_INVDEPTH").as_deref() == Ok("1");

    // --- Linear seed: minimize Σ ‖(I − dₖdₖᵀ)(X − oₖ)‖² over rays dₖ from oₖ ---
    let mut a = Matrix3::<f64>::zeros();
    let mut b = Vector3::<f64>::zeros();
    for o in obs {
        let bearing_cam = cam.undistort(&o.uv);
        let nb = bearing_cam.norm();
        if !(nb > 1e-9) {
            fail!("bearing_norm0");
        }
        if use_invdepth {
            // MSCEqF `linearTriangulation`: A_bf = R·(uvn.x,uvn.y,1) UN-normalized,
            // Ai = −⌊A_bf⌋² = ‖A_bf‖²·(I − v̂v̂ᵀ) — each ray weighted by ‖(uvn,1)‖²,
            // so off-axis observations count more (echo's unit-ray P weights all 1).
            let uvn = Vector3::new(
                bearing_cam[0] / bearing_cam[2],
                bearing_cam[1] / bearing_cam[2],
                1.0,
            );
            let a_bf = o.pose.rotation.act(&uvn); // un-normalized world ray
            let p = skew3(&a_bf) * skew3(&a_bf).transpose(); // −⌊a⌋² = ‖a‖²I − aaᵀ
            a += p;
            b += p * o.pose.translation;
        } else {
            let d = o.pose.rotation.act(&(bearing_cam / nb)); // world-frame unit ray
            let origin = o.pose.translation; // camera centre in world
            let p = Matrix3::identity() - d * d.transpose();
            a += p;
            b += p * origin;
        }
    }

    // Condition number of the linear ray system (mirrors OpenVINS `condA`): a
    // rank-deficient / near-parallel ray set (tiny parallax) has σ_max/σ_min ≫ 1
    // and triangulates to a wildly uncertain point.
    let sv = a.svd_unordered(false, false).singular_values;
    let smax = sv[0].max(sv[1]).max(sv[2]);
    let smin = sv[0].min(sv[1]).min(sv[2]);
    let cond = if smin > 0.0 {
        smax / smin
    } else {
        f64::INFINITY
    };
    if !(cond.is_finite()) || cond > gates.max_cond {
        fail!(format!("cond={:.3e}", cond));
    }

    let mut x = match a.try_inverse() {
        Some(ai) => ai * b,
        None => fail!("seed_singular"),
    };
    if x.iter().any(|v| !v.is_finite()) {
        fail!("seed_nonfinite");
    }

    // SEEDDBG (ECHO_MSC_TRIFAIL=1): capture the linear-seed anchor-frame depth so
    // we can compare it to the GN-refined depth — tests whether undamped Euclidean
    // GN wanders deep from a well-conditioned near seed on low-parallax tracks.
    let seed_depth_a = obs[0].pose.inverse().act(&x)[2];

    // Refinement: two modes. The default (Euclidean-XYZ undamped GN) BLOWS UP the
    // depth of low-parallax tracks — one step along the near-flat cost direction
    // rockets a far point to 10-40× its seed depth (measured: seed 47m → 1021m).
    // With ECHO_MSC_TRI_INVDEPTH the refine mirrors MSCEqF `nonlinearTriangulation`
    // / OpenVINS `single_gaussnewton`: refine in ANCHOR-frame INVERSE-DEPTH
    // (α=x/z, β=y/z, ρ=1/z), where ρ→0 is a bounded limit so the far-point step
    // stays finite, PLUS the revert-to-seed safeguard (if GN worsens the residual,
    // keep the linear seed) — the two ingredients echo's Euclidean GN lacks.
    if use_invdepth {
        let anchor_pose = &obs[0].pose;
        let anchor_r = anchor_pose.rotation.as_matrix();
        // Normalized (retina) coords per obs: undistort → bearing, divide by z.
        let uvns: Vec<Vector2<f64>> = obs
            .iter()
            .map(|o| {
                let b = cam.undistort(&o.uv);
                Vector2::new(b[0] / b[2], b[1] / b[2])
            })
            .collect();
        let a_f_init = anchor_pose.inverse().act(&x); // anchor-frame Euclidean seed
        let mut a_f = a_f_init;
        let mut invd = Vector3::new(a_f[0] / a_f[2], a_f[1] / a_f[2], 1.0 / a_f[2]);
        let mut initial_res_norm = 0.0_f64;
        let mut actual_res_norm = 0.0_f64;
        let mut converged = false;
        for it in 0..10 {
            // J_rep = a_f(2) · [ e0 e1 (−a_f) ] : ∂(anchor-Euclidean)/∂(inv-depth)
            let mut j_rep = Matrix3::<f64>::identity();
            j_rep.set_column(2, &(-a_f));
            j_rep *= a_f[2];
            let gf0 = anchor_pose.act(&a_f); // world point
            // Solve the RECTANGULAR least-squares J·delta = res via SVD (mirrors
            // MSCEqF's `J.colPivHouseholderQr().solve(res)`, NOT normal equations —
            // JᵀJ squares the condition number and detonates the near-unobservable
            // ρ direction on low-parallax tracks). SVD truncation zeroes that
            // direction so the step stays bounded and ρ keeps its seed value.
            let n = obs.len();
            let mut jmat = DMatrix::<f64>::zeros(2 * n, 3);
            let mut rvec = DVector::<f64>::zeros(2 * n);
            let mut rn2 = 0.0_f64;
            for (idx, (o, uvn)) in obs.iter().zip(uvns.iter()).enumerate() {
                let ci = o.pose.inverse().act(&gf0); // point in clone frame
                if ci[2] <= 1e-6 {
                    fail!(format!("invd_behind depth={:.3}", ci[2]));
                }
                let ci_invd = Vector3::new(ci[0] / ci[2], ci[1] / ci[2], 1.0 / ci[2]);
                let r = Vector2::new(uvn[0] - ci_invd[0], uvn[1] - ci_invd[1]); // z − h
                // ∂h/∂(inv-depth) = ρ_i·[I₂|−(α_i,β_i)] · Rᵢᵀ · R_A · J_rep  (2×3)
                let m = o.pose.rotation.inverse().as_matrix() * anchor_r * j_rep;
                let m0 = m.row(0).transpose();
                let m1 = m.row(1).transpose();
                let m2 = m.row(2).transpose();
                let jr0 = ci_invd[2] * (m0 - ci_invd[0] * m2); // 1st residual row (as col)
                let jr1 = ci_invd[2] * (m1 - ci_invd[1] * m2);
                jmat.row_mut(2 * idx).copy_from(&jr0.transpose());
                jmat.row_mut(2 * idx + 1).copy_from(&jr1.transpose());
                rvec[2 * idx] = r[0];
                rvec[2 * idx + 1] = r[1];
                rn2 += r.norm_squared();
            }
            actual_res_norm = rn2.sqrt();
            if it == 0 {
                initial_res_norm = actual_res_norm;
            }
            let svd = jmat.svd(true, true);
            let smax = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
            let delta_d = match svd.solve(&rvec, (1e-6 * smax).max(1e-12)) {
                Ok(d) => d,
                Err(_) => fail!("gn_singular"),
            };
            invd += Vector3::new(delta_d[0], delta_d[1], delta_d[2]);
            a_f = Vector3::new(invd[0] / invd[2], invd[1] / invd[2], 1.0 / invd[2]);
            if delta_d.norm() < 1e-6 {
                converged = true;
                break;
            }
        }
        // Revert-to-seed safeguard (MSCEqF updater.cpp:384-392): if GN did not
        // converge AND made the fit worse, keep the well-conditioned linear seed.
        if !converged && actual_res_norm > initial_res_norm {
            a_f = a_f_init;
        }
        x = anchor_pose.act(&a_f); // back to world frame for the shared gates below
    } else {
        // --- Legacy: Euclidean-XYZ undamped Gauss-Newton (blows up low-parallax) ---
        for _ in 0..5 {
            let mut jtj = Matrix3::<f64>::zeros();
            let mut jtr = Vector3::<f64>::zeros();
            for o in obs {
                let inv = o.pose.inverse();
                let q = inv.act(&x);
                if q[2] <= 1e-6 {
                    fail!(format!("gn_behind depth={:.3}", q[2]));
                }
                let r = o.uv - cam.project(&q); // z − h
                let jpi = cam.projection_jacobian(&q); // 2×3
                let h = jpi * o.pose.rotation.inverse().as_matrix(); // ∂h/∂X = Jπ·Rᵀ
                jtj += h.transpose() * h;
                jtr += h.transpose() * r;
            }
            let dx = match jtj.try_inverse() {
                Some(ji) => ji * jtr,
                None => fail!("gn_singular"),
            };
            x += dx;
            if dx.norm() < 1e-8 {
                break;
            }
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        fail!("post_gn_nonfinite");
    }
    // Cheirality: the refined point must sit in front of every observing camera.
    for o in obs {
        if o.pose.inverse().act(&x)[2] <= 1e-6 {
            fail!(format!(
                "cheirality depth={:.3}",
                o.pose.inverse().act(&x)[2]
            ));
        }
    }

    // --- OpenVINS distance + baseline-ratio gates (anchor = obs[0]) ------------
    // p_FinA = feature in the anchor camera frame; base_line_max = the largest
    // PERPENDICULAR baseline (component of a clone's anchor-frame position ⟂ the
    // anchor→feature ray). range/base_line_max is the scale-free inverse-parallax:
    // a far point seen over a short baseline (poor triangulation) has a large ratio.
    let anchor = &obs[0].pose;
    let p_fin_a = anchor.inverse().act(&x); // R_GtoA·(x − p_AinG)
    let depth = p_fin_a[2];
    let range = p_fin_a.norm();
    if trifail {
        eprintln!(
            "SEEDDBG nobs={} seed_depth={:.3} refined_depth={:.3}",
            obs.len(),
            seed_depth_a,
            depth
        );
    }
    // TRIINPUT (ECHO_MSC_TRIDUMP=1): dump the geometric INPUT (anchor_w, rel_t,
    // per-obs normalized uv) + result BEFORE the depth gate, so tracks echo rejects
    // (e.g. {83,151,184} at depth_range) still print. Match by anchor_w to MSCEqF's
    // .tript to byte-compare poses+uv on identical input.
    if std::env::var("ECHO_MSC_TRIDUMP").as_deref() == Ok("1") {
        let aw = anchor.translation;
        let last = &obs.last().unwrap().pose;
        let rel_t = anchor.inverse().act(&last.translation);
        // gauge-invariant relative rotation angle anchor->last (mirrors MSCEqF
        // rel_ang = acos(0.5(tr(R_a^T R_l)-1))): with identical bearings + rel_t,
        // any residual depth diff must live in THIS attitude term.
        let r_rel = anchor.rotation.inverse() * last.rotation;
        let ct = (0.5 * (r_rel.as_matrix().trace() - 1.0)).clamp(-1.0, 1.0);
        let rel_ang = ct.acos() * 180.0 / std::f64::consts::PI;
        eprint!(
            "TRIINPUT nobs={} depth={:.4} rel_ang={:.5} A_f=[{:.4},{:.4},{:.4}] anchor_w=[{:.6},{:.6},{:.6}] rel_t=[{:.6},{:.6},{:.6}] uvn=",
            obs.len(),
            depth,
            rel_ang,
            p_fin_a[0],
            p_fin_a[1],
            p_fin_a[2],
            aw[0],
            aw[1],
            aw[2],
            rel_t[0],
            rel_t[1],
            rel_t[2],
        );
        for o in obs {
            let bc = cam.undistort(&o.uv);
            eprint!("({:.6},{:.6})", bc[0] / bc[2], bc[1] / bc[2]);
        }
        eprintln!();
    }
    if !(range.is_finite()) || depth < gates.min_dist || depth > gates.max_dist {
        fail!(format!("depth_range depth={:.3} range={:.3}", depth, range));
    }
    let u = p_fin_a / range; // unit anchor→feature ray
    let mut base_line_max: f64 = 0.0;
    for o in obs {
        // clone centre in the anchor frame, then its component ⟂ the ray u.
        let p_ci_in_a = anchor.inverse().act(&o.pose.translation);
        let perp = (p_ci_in_a - u * p_ci_in_a.dot(&u)).norm();
        base_line_max = base_line_max.max(perp);
    }
    if !(base_line_max > 1e-9) || range / base_line_max > gates.max_baseline {
        fail!(format!(
            "baseline inv_par={:.3}",
            range / base_line_max.max(1e-12)
        ));
    }

    // Reprojection-RMS gate: reject a triangulation whose multi-view fit is poor
    // (the clone poses do not agree on a single 3-D point), which the loose-cov χ²
    // gate would otherwise admit and corrupt the nav with.
    if gates.max_rms_px.is_finite() {
        let mut ss = 0.0;
        for o in obs {
            let q = o.pose.inverse().act(&x);
            let r = o.uv - cam.project(&q);
            ss += r.norm_squared();
        }
        let rms = (ss / obs.len() as f64).sqrt();
        if !(rms <= gates.max_rms_px) {
            return None;
        }
    }
    // TRIDUMP (ECHO_MSC_TRIDUMP=1): per-track fit diagnostics under the CURRENT
    // clone poses. Under --gt-clones (GT poses) this measures how well the real
    // pixel tracks fit a SINGLE static point given perfect geometry: rms ≫ σ_pix ⇒
    // the tracks are inconsistent with a static point (front-end bias/noise), the
    // measurement-consistency defect; rms ≈ 0 ⇒ tracks consistent, leak is machinery.
    if std::env::var("ECHO_MSC_TRIDUMP").as_deref() == Ok("1") {
        let mut ss = 0.0;
        for o in obs {
            let q = o.pose.inverse().act(&x);
            ss += (o.uv - cam.project(&q)).norm_squared();
        }
        let rms = (ss / obs.len() as f64).sqrt();
        let inv_parallax = range / base_line_max; // large = poor parallax (far/short-baseline)
        // anchor_w / rel_t mirror MSCEqF's .tript fields so we can match a track by
        // its anchor and byte-compare the geometric INPUT (poses + normalized uv)
        // echo feeds vs MSCEqF, before triangulation numerics.
        let anchor_w = obs[0].pose.translation;
        let rel_t = anchor.inverse().act(&obs.last().unwrap().pose.translation);
        eprint!(
            "TRIDUMP nobs={} rms_px={:.4} depth={:.3} inv_parallax={:.3} anchor_w=[{:.6},{:.6},{:.6}] rel_t=[{:.6},{:.6},{:.6}] uvn=",
            obs.len(),
            rms,
            depth,
            inv_parallax,
            anchor_w[0],
            anchor_w[1],
            anchor_w[2],
            rel_t[0],
            rel_t[1],
            rel_t[2],
        );
        for o in obs {
            let bc = cam.undistort(&o.uv);
            eprint!("({:.6},{:.6})", bc[0] / bc[2], bc[1] / bc[2]);
        }
        eprintln!();
    }
    Some(x)
}

/// Per-observation Jacobians for a triangulated track.
///
/// Returns `(h_f, h_x, res)` stacked over the `N = obs.len()` observations:
/// - `h_f`  : `2N×3`  feature Euclidean-error Jacobian (`∂h/∂X_f`)
/// - `h_x`  : `2N×6N` block-diagonal clone-pose Jacobian (`∂h/∂δₖ`)
/// - `res`  : `2N`    residual `z − h`
///
/// Column band `[6k, 6k+6)` of `h_x` belongs to observation `k`'s clone. The caller
/// maps those bands onto the global covariance columns (clones may repeat / be
/// non-contiguous), so this function keeps them one-per-observation.
pub fn feature_jacobians(
    x_f: &Vector3<f64>,
    obs: &[MscObs],
    cam: &dyn CameraModel,
) -> Option<(DMatrix<f64>, DMatrix<f64>, DVector<f64>)> {
    let n = obs.len();
    let mut h_f = DMatrix::<f64>::zeros(2 * n, 3);
    let mut h_x = DMatrix::<f64>::zeros(2 * n, 6 * n);
    let mut res = DVector::<f64>::zeros(2 * n);

    for (k, o) in obs.iter().enumerate() {
        // Residual is evaluated at the CURRENT clone estimate (`pose`).
        let q = o.pose.inverse().act(x_f); // feature in clone camera frame
        if q[2] <= 1e-6 {
            return None;
        }
        let r = o.uv - cam.project(&q);

        // Jacobians are evaluated at the FIRST-ESTIMATE clone pose (`pose_fej`).
        // With FEJ off `pose_fej == pose`, so `q_fej == q` and this reduces exactly
        // to the current-estimate linearization (mirrors OpenVINS: residual at the
        // current state, ∂h/∂X at the first estimate).
        let q_fej = o.pose_fej.inverse().act(x_f);
        if q_fej[2] <= 1e-6 {
            return None;
        }
        let jpi = cam.projection_jacobian(&q_fej); // 2×3
        let rt = o.pose_fej.rotation.inverse().as_matrix(); // Rᵀ (world→cam)

        // H_f = Jπ · Rᵀ
        let hf = jpi * rt;
        // H_x = Jπ · [ [q]× | −I ]   (right camera-pose perturbation δ = [ω; v])
        let mut dq = DMatrix::<f64>::zeros(3, 6);
        dq.view_mut((0, 0), (3, 3)).copy_from(&skew3(&q_fej));
        dq.view_mut((0, 3), (3, 3))
            .copy_from(&(-Matrix3::identity()));
        let hx = jpi * dq;
        for a in 0..2 {
            for c in 0..3 {
                h_f[(2 * k + a, c)] = hf[(a, c)];
            }
            for c in 0..6 {
                h_x[(2 * k + a, 6 * k + c)] = hx[(a, c)];
            }
            res[2 * k + a] = r[a];
        }
    }
    if !h_f.iter().all(|v| v.is_finite())
        || !h_x.iter().all(|v| v.is_finite())
        || !res.iter().all(|v| v.is_finite())
    {
        return None;
    }
    Some((h_f, h_x, res))
}

/// ANCHORED per-observation Jacobians (MSCEqF `ProjectionHelperZ1` mirror, in
/// ECHO-LI's right-perturbation chart).
///
/// Same output shape as [`feature_jacobians`] — `(h_f, h_x, res)` stacked over the
/// `N` observations — but the feature is parameterized in the ANCHOR clone's frame
/// (`f_a = T_a⁻¹ X`, anchor = `obs[0]`) instead of the world frame, and every
/// observing clone `k ≠ a` contributes to BOTH its own column band `k` AND the
/// anchor's band `0`:
/// - `H_clone_k` (band `k`) `= Jπ·[ [q_k]× | −I ]`               (unchanged from absolute)
/// - `H_anchor_k` (band `0`) `= Jπ·R_kᵀR_a·[ −[f_a]× | +I ]`     (verified vs FD)
/// - `H_f_k` (feature) `= Jπ·R_kᵀR_a`                             (`∂q_k/∂f_a`)
/// The anchor observation `k = a` has ZERO pose Jacobian (its measurement `π(f_a)`
/// is independent of `T_a`), mirroring MSCEqF's `clone_ts != anchor_ts` guard.
///
/// The point: a common WORLD-frame (gauge) perturbation moves both `T_k` and the
/// anchored point `X = T_a f_a` together, leaving `q_k = T_k⁻¹X` — and hence the
/// residual — invariant. So `H_x·g = 0` for the gauge direction `g` (band `k` =
/// `Ad_{T_k⁻¹}ξ`) BEFORE projection, whereas the absolute form's `H_x·g ≠ 0`
/// (only equal after the feature is nullspace-projected). This keeps the
/// constraint in the observable RELATIVE-pose channel.
pub fn feature_jacobians_anchored(
    x_f: &Vector3<f64>,
    obs: &[MscObs],
    cam: &dyn CameraModel,
) -> Option<(DMatrix<f64>, DMatrix<f64>, DVector<f64>)> {
    let n = obs.len();
    if n < 2 {
        return None;
    }
    let mut h_f = DMatrix::<f64>::zeros(2 * n, 3);
    let mut h_x = DMatrix::<f64>::zeros(2 * n, 6 * n);
    let mut res = DVector::<f64>::zeros(2 * n);

    // Anchor = obs[0]. f_a is the feature in the anchor camera frame.
    let anchor = &obs[0].pose;
    let r_a = anchor.rotation.as_matrix(); // R_a (camera→world)
    let f_a = anchor.inverse().act(x_f); // T_a⁻¹ X

    for (k, o) in obs.iter().enumerate() {
        let inv = o.pose.inverse();
        let q = inv.act(x_f); // feature in clone-k camera frame
        if q[2] <= 1e-6 {
            return None;
        }
        let jpi = cam.projection_jacobian(&q); // 2×3
        let rt_k = o.pose.rotation.inverse().as_matrix(); // R_kᵀ (world→cam k)

        // H_f = Jπ · R_kᵀ R_a  (∂q_k/∂f_a)
        let hf = jpi * rt_k * r_a;

        let r = o.uv - cam.project(&q);
        for a in 0..2 {
            for c in 0..3 {
                h_f[(2 * k + a, c)] = hf[(a, c)];
            }
            res[2 * k + a] = r[a];
        }

        if k == 0 {
            continue; // anchor observation: zero pose Jacobian
        }

        // H_clone_k (band k) = Jπ · [ [q_k]× | −I ]
        let mut dq_k = DMatrix::<f64>::zeros(3, 6);
        dq_k.view_mut((0, 0), (3, 3)).copy_from(&skew3(&q));
        dq_k.view_mut((0, 3), (3, 3))
            .copy_from(&(-Matrix3::identity()));
        let hx_k = jpi * dq_k;

        // H_anchor_k (band 0) = Jπ · R_kᵀ R_a · [ −[f_a]× | +I ]
        // (∂q_k/∂δ_a: a right perturbation of T_a moves the anchored point X=T_a f_a
        // by δ(exp(δ_a^)f_a) = ω_a×f_a + v_a = [ −[f_a]× | I ]·δ_a, then R_kᵀR_a into q_k.)
        let mut dq_a = DMatrix::<f64>::zeros(3, 6);
        dq_a.view_mut((0, 0), (3, 3)).copy_from(&(-skew3(&f_a)));
        dq_a.view_mut((0, 3), (3, 3))
            .copy_from(&Matrix3::identity());
        let hx_a = jpi * rt_k * r_a * dq_a;

        for a in 0..2 {
            for c in 0..6 {
                h_x[(2 * k + a, 6 * k + c)] = hx_k[(a, c)];
                h_x[(2 * k + a, c)] += hx_a[(a, c)]; // band 0 = anchor
            }
        }
    }
    if !h_f.iter().all(|v| v.is_finite())
        || !h_x.iter().all(|v| v.is_finite())
        || !res.iter().all(|v| v.is_finite())
    {
        return None;
    }
    Some((h_f, h_x, res))
}

/// LEFT/global-chart port of [`feature_jacobians_anchored`], byte-mirroring
/// MSCEqF's `ProjectionHelperZ1::residualJacobianBlock`.
///
/// The ONLY difference from `feature_jacobians_anchored` is the CHART of the pose
/// columns: instead of ECHO-LI's right camera-frame perturbation `T·exp(δ^)`
/// (`H = Jπ·[[q]×|−I]`), the clone pose is perturbed in the LEFT/GLOBAL frame
/// `exp(δ^)·T` (`E → exp(δ^)E`), matching MSCEqF exactly. Under a left
/// perturbation the world point `G0 = x_f` is fixed and the pose moves the
/// camera-frame point as `q(δ) ≈ q − Rᵀ([ω]×G0 + v)`, so with `A = [[G0]× | −I]`:
/// - `H_clone_k` (band `k`) `= Jπ · R_kᵀ · A`             (LEFT/global clone col)
/// - `H_anchor_k` (band `0`) `= −H_clone_k`                (anchor = −clone col)
/// - `H_f_k` (feature) `= Jπ · R_kᵀ · R_a`                 (unchanged — chart-free)
/// The anchor's own observation (`k = 0`) gives `+clone − clone = 0` net pose
/// Jacobian, so it is skipped exactly as in the right-chart form.
///
/// This exists so ECHO-LI's clone window runs in the SAME chart as its nav state
/// (SDB left action) AND as MSCEqF — removing the mixed left-nav/right-clone
/// coupling. Selected by `ECHO_MSC_LEFTCHART=1` in the update path.
pub fn feature_jacobians_anchored_left(
    x_f: &Vector3<f64>,
    obs: &[MscObs],
    cam: &dyn CameraModel,
) -> Option<(DMatrix<f64>, DMatrix<f64>, DVector<f64>)> {
    let n = obs.len();
    if n < 2 {
        return None;
    }
    let mut h_f = DMatrix::<f64>::zeros(2 * n, 3);
    let mut h_x = DMatrix::<f64>::zeros(2 * n, 6 * n);
    let mut res = DVector::<f64>::zeros(2 * n);

    let anchor = &obs[0].pose;
    let r_a = anchor.rotation.as_matrix(); // R_a (camera→world)
    let g0 = *x_f; // world-frame point (== MSCEqF G0_f), fixed under a left perturbation

    // A = [ [G0]× | −I ]  (3×6): ∂(exp(δ^)·G0)/∂δ, δ = [ω; v], point part.
    let mut a_mat = DMatrix::<f64>::zeros(3, 6);
    a_mat.view_mut((0, 0), (3, 3)).copy_from(&skew3(&g0));
    a_mat
        .view_mut((0, 3), (3, 3))
        .copy_from(&(-Matrix3::identity()));

    for (k, o) in obs.iter().enumerate() {
        let inv = o.pose.inverse();
        let q = inv.act(x_f); // feature in clone-k camera frame
        if q[2] <= 1e-6 {
            return None;
        }
        let jpi = cam.projection_jacobian(&q); // 2×3 (D)
        let rt_k = o.pose.rotation.inverse().as_matrix(); // R_kᵀ (world→cam k)

        // H_f = Jπ · R_kᵀ R_a  (∂q_k/∂f_a) — identical to the right-chart form.
        let hf = jpi * rt_k * r_a;

        let r = o.uv - cam.project(&q);
        for a in 0..2 {
            for c in 0..3 {
                h_f[(2 * k + a, c)] = hf[(a, c)];
            }
            res[2 * k + a] = r[a];
        }

        if k == 0 {
            continue; // anchor obs: +clone − clone = 0 net pose Jacobian
        }

        // LEFT/global clone col = Jπ · R_kᵀ · [ [G0]× | −I ]; anchor band = −clone.
        let clone_col = (jpi * rt_k) * a_mat.clone(); // 2×6
        for a in 0..2 {
            for c in 0..6 {
                h_x[(2 * k + a, 6 * k + c)] = clone_col[(a, c)]; // band k (clone)
                h_x[(2 * k + a, c)] -= clone_col[(a, c)]; // band 0 (anchor) = −clone
            }
        }
    }
    if !h_f.iter().all(|v| v.is_finite())
        || !h_x.iter().all(|v| v.is_finite())
        || !res.iter().all(|v| v.is_finite())
    {
        return None;
    }
    Some((h_f, h_x, res))
}

/// Orthonormal basis (`2N×(2N−3)`) of the LEFT nullspace of `h_f` (`2N×3`).
///
/// `h_f` is a tall thin `∂h/∂X` with 3 columns; its left nullspace has dimension
/// `2N−3`. Built by modified Gram-Schmidt: orthonormalize `h_f`'s columns (thin
/// QR), then sweep the standard basis, project each candidate off the column space
/// and the accumulated nullspace, and keep the survivors. Returns `None` if fewer
/// than `2N−3` independent survivors are found (rank-deficient / degenerate track).
fn left_nullspace_basis(h_f: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let m = h_f.nrows();
    let n = h_f.ncols();
    if m <= n {
        return None;
    }
    let col_basis = h_f.clone().qr().q(); // m×n, orthonormal cols spanning col(h_f)
    let mut null: Vec<DVector<f64>> = Vec::with_capacity(m - n);
    for i in 0..m {
        let mut v = DVector::<f64>::zeros(m);
        v[i] = 1.0;
        for c in 0..col_basis.ncols() {
            let qc = col_basis.column(c);
            v -= qc.dot(&v) * qc;
        }
        for u in &null {
            v -= u.dot(&v) * u;
        }
        let nv = v.norm();
        if nv > 1e-9 {
            v /= nv;
            null.push(v);
        }
        if null.len() == m - n {
            break;
        }
    }
    if null.len() != m - n {
        return None;
    }
    let mut basis = DMatrix::<f64>::zeros(m, m - n);
    for (c, u) in null.iter().enumerate() {
        basis.set_column(c, u);
    }
    Some(basis)
}

/// Left-nullspace projection: remove the feature (`h_f`) from the constraint.
///
/// Mirrors `UpdaterHelper::nullspace_project_inplace`. With `N` an orthonormal
/// basis of `left-null(h_f)` (dim `2N−3`), projecting gives
///   `r_o = Nᵀ res`,  `h_o = Nᵀ h_x`,
/// and because `N` is orthonormal the measurement noise stays isotropic
/// (`Nᵀ(σ²I)N = σ²I`), so the caller reuses `σ_pix²·I`. Returns `None` if
/// `2N ≤ 3` (no constraint survives) or the basis is degenerate / non-finite.
pub fn left_nullspace_project(
    h_f: &DMatrix<f64>,
    h_x: &DMatrix<f64>,
    res: &DVector<f64>,
) -> Option<(DMatrix<f64>, DVector<f64>)> {
    if h_f.nrows() <= 3 {
        return None;
    }
    let null = left_nullspace_basis(h_f)?;
    let nt = null.transpose(); // (2N−3)×2N
    let h_o = &nt * h_x;
    let r_o = &nt * res;
    if !h_o.iter().all(|v| v.is_finite()) || !r_o.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((h_o, r_o))
}

/// QR split for **delayed feature initialization** — mirrors the Givens step of
/// OpenVINS `StateHelper::initialize` (`StateHelper.cpp`).
///
/// Where [`left_nullspace_project`] discards the feature (structureless MSCKF),
/// delayed init *keeps* it: the stacked system is rotated by `Qᵀ` (`Q` from the
/// thin QR of the feature Jacobian `h_l`, 2N×3 in the feature's chart) so that
/// - the **top 3 rows** become the *invertible initializing* system that depends
///   on the new feature (`h_finit` upper-triangular 3×3, `hx_init` 3×6N,
///   `res_init` 3) — used to augment the state with the new landmark, and
/// - the **bottom 2N−3 rows** are the *nullspace-projected updating* system
///   (feature-independent: `hup` (2N−3)×6N, `res_up`) — an ordinary MSCKF
///   constraint applied after augmentation.
///
/// `Qᵀ` is orthogonal, so the σ²·I measurement noise stays isotropic in both
/// blocks (same argument as the nullspace projection). Returns `None` if there is
/// no updating constraint (`2N ≤ 3`), the QR is degenerate, or `h_finit` is
/// singular (rank-deficient triangulation geometry).
pub fn initialize_split(
    h_l: &DMatrix<f64>,
    h_x: &DMatrix<f64>,
    res: &DVector<f64>,
) -> Option<(
    DMatrix<f64>, // h_finit  3×3 (upper-triangular, invertible)
    DMatrix<f64>, // hx_init  3×6N
    DVector<f64>, // res_init 3
    DMatrix<f64>, // hup      (2N−3)×6N
    DVector<f64>, // res_up   (2N−3)
)> {
    let m = h_l.nrows();
    if m <= 3 || h_l.ncols() != 3 {
        return None;
    }
    let qr = h_l.clone().qr();
    // Apply Qᵀ to h_l itself (→ R stacked, top 3 rows upper-tri), h_x and res.
    // Using q_tr_mul (full Householder) keeps all m rows, unlike the economy r().
    let mut hl_r = h_l.clone();
    qr.q_tr_mul(&mut hl_r);
    let mut hx_r = h_x.clone();
    qr.q_tr_mul(&mut hx_r);
    let mut res_r = res.clone();
    qr.q_tr_mul(&mut res_r);

    let h_finit = hl_r.rows(0, 3).into_owned();
    // Invertibility guard on the triangular init block.
    let det = h_finit[(0, 0)] * h_finit[(1, 1)] * h_finit[(2, 2)];
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let hx_init = hx_r.rows(0, 3).into_owned();
    let res_init = res_r.rows(0, 3).into_owned();
    let hup = hx_r.rows(3, m - 3).into_owned();
    let res_up = res_r.rows(3, m - 3).into_owned();
    if !h_finit.iter().all(|v| v.is_finite())
        || !hx_init.iter().all(|v| v.is_finite())
        || !hup.iter().all(|v| v.is_finite())
        || !res_init.iter().all(|v| v.is_finite())
        || !res_up.iter().all(|v| v.is_finite())
    {
        return None;
    }
    Some((h_finit, hx_init, res_init, hup, res_up))
}

/// χ²(0.95) quantile for `dof` degrees of freedom, Wilson–Hilferty approximation
/// (accurate to <1% for dof≥1; ample for a rejection gate). Used by the MSC
/// innovation gate `rᵀS⁻¹r < mult·χ²₀.₉₅(dof)`.
pub fn chi2_095(dof: usize) -> f64 {
    let k = dof as f64;
    let z = 1.6448536269514722_f64; // Φ⁻¹(0.95)
    let t = 1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt();
    k * t * t * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematical::camera::PinholeModel;
    use echo_lie::SO3;

    fn cam() -> PinholeModel {
        PinholeModel {
            fx: 256.0,
            fy: 256.0,
            cx: 256.0,
            cy: 256.0,
        }
    }

    /// A synthetic 3-D point seen from a few poses triangulates back to itself.
    #[test]
    fn triangulation_recovers_synthetic_point() {
        let c = cam();
        let x_true = Vector3::new(0.7, -0.4, 6.0);
        let poses = [
            SE3::identity(),
            SE3::new(SO3::identity(), Vector3::new(1.0, 0.0, 0.0)),
            SE3::new(
                SO3::exp(&Vector3::new(0.0, 0.05, 0.0)),
                Vector3::new(2.0, 0.2, -0.3),
            ),
        ];
        let obs: Vec<MscObs> = poses
            .iter()
            .map(|p| {
                let q = p.inverse().act(&x_true);
                MscObs {
                    pose: p.clone(),
                    pose_fej: p.clone(),
                    uv: c.project(&q),
                }
            })
            .collect();
        let x_hat = triangulate(&obs, &c).expect("triangulation");
        assert!(
            (x_hat - x_true).norm() < 1e-6,
            "triangulated {x_hat:?} vs {x_true:?}"
        );
    }

    /// FEJ: `feature_jacobians` evaluates `h_f`/`h_x` at `pose_fej` (first estimate)
    /// while the residual is evaluated at `pose` (current). When they differ, the
    /// Jacobian must match the all-at-fej reference and the residual the current one.
    #[test]
    fn fej_linearizes_jacobian_at_first_estimate() {
        let c = cam();
        let x_f = Vector3::new(0.2, -0.1, 4.0);
        // First-estimate pose A and the (mean-corrected) current pose B ≠ A.
        let pose_a = SE3::new(
            SO3::exp(&Vector3::new(0.0, 0.02, 0.0)),
            Vector3::new(0.1, 0.0, 0.0),
        );
        let pose_b = SE3::new(
            SO3::exp(&Vector3::new(0.01, -0.03, 0.005)),
            Vector3::new(0.25, -0.05, 0.1),
        );
        // A fixed measurement offset so the residual is a KNOWN nonzero vector iff it
        // is evaluated at the current pose B (prediction = project at B).
        let off = Vector2::new(0.5, -0.3);
        let uv0 = c.project(&pose_b.inverse().act(&x_f)) + off;

        let mixed = vec![MscObs {
            pose: pose_b.clone(),
            pose_fej: pose_a.clone(),
            uv: uv0,
        }];
        // References: all-at-A (Jacobian) and the current-pose-B residual prediction.
        let ref_a = vec![MscObs {
            pose: pose_a.clone(),
            pose_fej: pose_a.clone(),
            uv: uv0,
        }];

        let (hf_a, hx_a, _res_a) = feature_jacobians(&x_f, &ref_a, &c).unwrap();
        let (hf_m, hx_m, res_m) = feature_jacobians(&x_f, &mixed, &c).unwrap();

        assert!(
            (hf_m.clone() - hf_a).norm() < 1e-9,
            "H_f must be linearized at pose_fej (A)"
        );
        assert!(
            (hx_m - hx_a).norm() < 1e-9,
            "H_x must be linearized at pose_fej (A)"
        );
        // Residual = uv0 − project(at B) = off, proving it uses the CURRENT pose.
        assert!(
            (res_m - DVector::from_row_slice(&[off[0], off[1]])).norm() < 1e-9,
            "residual must be evaluated at the current pose (B)"
        );
        // Sanity: A and B genuinely give different H_f (so the test isn't vacuous).
        let mixed_at_b = vec![MscObs {
            pose: pose_b.clone(),
            pose_fej: pose_b.clone(),
            uv: uv0,
        }];
        let (hf_b, _, _) = feature_jacobians(&x_f, &mixed_at_b, &c).unwrap();
        assert!(
            (hf_b - hf_m).norm() > 1e-6,
            "pose A and B must differ (non-vacuous)"
        );
    }

    /// The nullspace projection annihilates the feature direction: `Nᵀ h_f = 0`.
    #[test]
    fn nullspace_annihilates_feature() {
        let c = cam();
        let x_true = Vector3::new(-0.3, 0.5, 5.0);
        let poses = [
            SE3::identity(),
            SE3::new(SO3::identity(), Vector3::new(0.8, 0.1, 0.0)),
            SE3::new(SO3::identity(), Vector3::new(1.6, -0.1, 0.2)),
        ];
        let obs: Vec<MscObs> = poses
            .iter()
            .map(|p| {
                let q = p.inverse().act(&x_true);
                MscObs {
                    pose: p.clone(),
                    pose_fej: p.clone(),
                    uv: c.project(&q),
                }
            })
            .collect();
        let (h_f, h_x, res) = feature_jacobians(&x_true, &obs, &c).unwrap();
        let (h_o, r_o) = left_nullspace_project(&h_f, &h_x, &res).unwrap();
        // Nᵀ h_f = 0  ⇒  reconstruct N from the projected relation is awkward;
        // instead verify dims and that the projector removed exactly 3 rows and
        // that h_o still carries the clone constraint (nonzero).
        assert_eq!(h_o.nrows(), 2 * obs.len() - 3);
        assert_eq!(h_o.ncols(), 6 * obs.len());
        // Perfect (noise-free) observations ⇒ projected residual ≈ 0.
        assert!(r_o.norm() < 1e-6, "r_o norm {}", r_o.norm());
        // Direct annihilation check: Nᵀ h_f must vanish, and N is orthonormal.
        let null = left_nullspace_basis(&h_f).unwrap();
        assert_eq!(null.ncols(), 2 * obs.len() - 3);
        let should_be_zero = null.transpose() * &h_f;
        assert!(
            should_be_zero.amax() < 1e-9,
            "Nᵀh_f amax {}",
            should_be_zero.amax()
        );
        let gram = null.transpose() * &null;
        assert!(
            (gram - DMatrix::identity(null.ncols(), null.ncols())).amax() < 1e-9,
            "N not orthonormal"
        );
    }

    /// DECISIVE gauge test: does the constraint see the global-gauge (common
    /// world-frame) perturbation of the clones? Builds the stacked gauge direction
    /// `g` (band k = `Ad_{T_k⁻¹} ξ`) and measures `‖H_x·g‖` (raw) and `‖h_o·g‖`
    /// (after nullspace projection) for the ABSOLUTE vs the ANCHORED Jacobian.
    ///
    /// Distinguishes the mechanism: if the absolute RAW `H_x·g ≠ 0` but the
    /// anchored RAW `H_x·g = 0`, the anchored form keeps the constraint out of the
    /// absolute/gauge channel BEFORE projection (the honest-relative-channel claim).
    /// If BOTH projected `h_o·g = 0`, single-update gauge is NOT the differentiator
    /// and the fix must be temporal/linearization consistency instead.
    #[test]
    fn gauge_sensitivity_absolute_vs_anchored() {
        let c = cam();
        let x_true = Vector3::new(0.6, -0.3, 5.5);
        // Non-trivial rotations AND translations so Ad_{T⁻¹} differs per clone.
        let poses = [
            SE3::new(
                SO3::exp(&Vector3::new(0.02, -0.03, 0.01)),
                Vector3::new(0.0, 0.0, 0.0),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(-0.01, 0.04, 0.02)),
                Vector3::new(0.7, 0.1, -0.05),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(0.03, 0.02, -0.02)),
                Vector3::new(1.5, -0.2, 0.15),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(-0.02, -0.01, 0.03)),
                Vector3::new(2.1, 0.3, -0.1),
            ),
        ];
        let obs: Vec<MscObs> = poses
            .iter()
            .map(|p| {
                let q = p.inverse().act(&x_true);
                MscObs {
                    pose: p.clone(),
                    pose_fej: p.clone(),
                    uv: c.project(&q),
                }
            })
            .collect();
        let n = obs.len();

        // Stacked gauge direction: a common world-frame perturbation ξ maps to the
        // right-chart clone increment δ_k = Ad_{T_k⁻¹} ξ.
        let xi = nalgebra::Vector6::new(0.05, -0.03, 0.04, 0.2, -0.1, 0.15);
        let mut g = DVector::<f64>::zeros(6 * n);
        for (k, o) in obs.iter().enumerate() {
            let adj_inv = o.pose.inverse().adjoint(); // Ad_{T_k⁻¹}
            let gk = adj_inv * xi;
            for t in 0..6 {
                g[6 * k + t] = gk[t];
            }
        }

        // Absolute form.
        let (hf_abs, hx_abs, res_abs) = feature_jacobians(&x_true, &obs, &c).unwrap();
        let (ho_abs, _) = left_nullspace_project(&hf_abs, &hx_abs, &res_abs).unwrap();
        let raw_abs = (&hx_abs * &g).norm();
        let proj_abs = (&ho_abs * &g).norm();

        // Anchored form.
        let (hf_anc, hx_anc, res_anc) = feature_jacobians_anchored(&x_true, &obs, &c).unwrap();
        let (ho_anc, _) = left_nullspace_project(&hf_anc, &hx_anc, &res_anc).unwrap();
        let raw_anc = (&hx_anc * &g).norm();
        let proj_anc = (&ho_anc * &g).norm();

        println!("RAW   |H_x·g|  absolute={raw_abs:.3e}  anchored={raw_anc:.3e}");
        println!("PROJ  |h_o·g|  absolute={proj_abs:.3e}  anchored={proj_anc:.3e}");

        // Anchored annihilates the gauge BEFORE projection; absolute does not.
        assert!(
            raw_anc < 1e-9,
            "anchored raw H_x·g should vanish, got {raw_anc:.3e}"
        );
        assert!(
            raw_abs > 1e-3,
            "absolute raw H_x·g should be nonzero, got {raw_abs:.3e}"
        );
    }

    /// Finite-difference check of the FULL anchored `H_x` (both the observing-clone
    /// bands `k≥1` and the accumulated anchor band `0`), holding the anchor-frame
    /// feature `f_a = T_a⁻¹X` FIXED as the estimated quantity — exactly how the MSC
    /// update treats it. Perturbing a clone `T_j` (`j≥1`) moves only `res_j`;
    /// perturbing the anchor `T_a` moves the world point `X=T_a f_a` and hence every
    /// observing clone's residual (band 0), while the anchor's OWN residual (`k=0`)
    /// is invariant. Distinguishes a sign/chart bug in the anchor block from a mere
    /// stale doc comment — the observable part the gauge test does not probe.
    #[test]
    fn anchored_jacobian_matches_finite_difference() {
        let c = cam();
        let x_true = Vector3::new(0.6, -0.3, 5.5);
        let poses = [
            SE3::new(
                SO3::exp(&Vector3::new(0.02, -0.03, 0.01)),
                Vector3::new(0.0, 0.0, 0.0),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(-0.01, 0.04, 0.02)),
                Vector3::new(0.7, 0.1, -0.05),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(0.03, 0.02, -0.02)),
                Vector3::new(1.5, -0.2, 0.15),
            ),
        ];
        let obs: Vec<MscObs> = poses
            .iter()
            .map(|p| {
                let q = p.inverse().act(&x_true);
                MscObs {
                    pose: p.clone(),
                    pose_fej: p.clone(),
                    uv: c.project(&q),
                }
            })
            .collect();
        let n = obs.len();
        let (_hf, hx, _res) = feature_jacobians_anchored(&x_true, &obs, &c).unwrap();

        // f_a fixed as the state; residual of obs k as a function of the poses.
        let f_a = obs[0].pose.inverse().act(&x_true);
        let h = 1e-6;
        // Helper: residual row-pair for obs k given a (possibly perturbed) anchor
        // pose and clone-k pose. X = anchor·f_a.
        let res_of = |anchor: &SE3, pose_k: &SE3, uv_k: &Vector2<f64>| -> Vector2<f64> {
            let x_w = anchor.act(&f_a);
            let q = pose_k.inverse().act(&x_w);
            uv_k - c.project(&q)
        };

        // The code's H_x follows the standard EKF measurement convention
        // H = ∂h/∂δ = ∂project(q)/∂δ = −∂res/∂δ (res = uv − project(q)); the FD
        // below differentiates res, so the analytic block is compared against −num.
        let mut max_err = 0.0_f64;
        // Clone bands j≥1: perturb T_j only (anchor fixed), affects res_j only.
        for j in 1..n {
            for t in 0..6 {
                let mut d = nalgebra::Vector6::zeros();
                d[t] = h;
                let pj = obs[j].pose.compose(&SE3::exp(&d));
                let rp = res_of(&obs[0].pose, &pj, &obs[j].uv);
                d[t] = -h;
                let pm = obs[j].pose.compose(&SE3::exp(&d));
                let rm = res_of(&obs[0].pose, &pm, &obs[j].uv);
                let num = (rp - rm) / (2.0 * h);
                for a in 0..2 {
                    max_err = max_err.max((-num[a] - hx[(2 * j + a, 6 * j + t)]).abs());
                }
            }
        }
        // Anchor band 0: perturb T_a only (f_a fixed → world point X=T_a·f_a moves),
        // affects every OBSERVING clone k≥1. For k=0 the anchor's own camera IS the
        // perturbed pose, so X and the camera move together and q_0=T_a⁻¹X=f_a is
        // invariant ⇒ zero Jacobian (the code's `continue` at the anchor obs). The FD
        // must therefore perturb the anchor's own projection pose too for k=0.
        for t in 0..6 {
            let mut d = nalgebra::Vector6::zeros();
            d[t] = h;
            let ap = obs[0].pose.compose(&SE3::exp(&d));
            d[t] = -h;
            let am = obs[0].pose.compose(&SE3::exp(&d));
            for k in 0..n {
                let (pose_kp, pose_km) = if k == 0 {
                    (&ap, &am)
                } else {
                    (&obs[k].pose, &obs[k].pose)
                };
                let rp = res_of(&ap, pose_kp, &obs[k].uv);
                let rm = res_of(&am, pose_km, &obs[k].uv);
                let num = (rp - rm) / (2.0 * h);
                for a in 0..2 {
                    max_err = max_err.max((-num[a] - hx[(2 * k + a, t)]).abs());
                }
            }
        }
        assert!(
            max_err < 1e-4,
            "anchored H_x FD mismatch: max_err={max_err:.3e}"
        );
    }

    /// `initialize_split` (delayed-init QR split) must be consistent with the
    /// structureless `left_nullspace_project`: its bottom `2N−3` rows span the same
    /// left-nullspace of `h_f`, so the innovation χ² of the updating block is
    /// IDENTICAL under any state covariance `P` (both are orthonormal bases of the
    /// same subspace ⇒ related by an orthogonal change of basis, which the quadratic
    /// form is invariant to). Also checks `Qᵀ` energy preservation and that the
    /// init block `h_finit` is upper-triangular and invertible.
    #[test]
    fn initialize_split_matches_nullspace_projection() {
        let c = cam();
        let x_true = Vector3::new(0.4, -0.2, 5.0);
        let poses = [
            SE3::new(
                SO3::exp(&Vector3::new(0.02, -0.03, 0.01)),
                Vector3::new(0.0, 0.0, 0.0),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(-0.01, 0.04, 0.02)),
                Vector3::new(0.7, 0.1, -0.05),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(0.03, 0.02, -0.02)),
                Vector3::new(1.5, -0.2, 0.15),
            ),
            SE3::new(
                SO3::exp(&Vector3::new(-0.02, -0.01, 0.03)),
                Vector3::new(2.1, 0.3, -0.1),
            ),
        ];
        // Perfect pixels from x_true, but linearize at a PERTURBED point so the
        // residual is a known nonzero vector (exercises res_init/res_up).
        let obs: Vec<MscObs> = poses
            .iter()
            .map(|p| {
                let q = p.inverse().act(&x_true);
                MscObs {
                    pose: p.clone(),
                    pose_fej: p.clone(),
                    uv: c.project(&q),
                }
            })
            .collect();
        let n = obs.len();
        let x_pert = x_true + Vector3::new(0.05, -0.04, 0.1);
        let (h_f, h_x, res) = feature_jacobians(&x_pert, &obs, &c).unwrap();

        let (h_finit, _hx_init, res_init, hup, res_up) =
            initialize_split(&h_f, &h_x, &res).expect("split");

        // Shapes.
        assert_eq!(h_finit.shape(), (3, 3));
        assert_eq!(hup.nrows(), 2 * n - 3);
        assert_eq!(hup.ncols(), 6 * n);
        assert_eq!(res_up.len(), 2 * n - 3);
        // h_finit upper-triangular and invertible.
        assert!(h_finit[(1, 0)].abs() < 1e-9);
        assert!(h_finit[(2, 0)].abs() < 1e-9);
        assert!(h_finit[(2, 1)].abs() < 1e-9);
        assert!(h_finit.determinant().abs() > 1e-6, "h_finit singular");
        // Qᵀ preserves residual energy.
        let e_in = res.norm_squared();
        let e_out = res_init.norm_squared() + res_up.norm_squared();
        assert!((e_in - e_out).abs() < 1e-9, "energy {e_in} vs {e_out}");

        // χ² invariance vs nullspace projection under an arbitrary SPD P.
        let (h_o, r_o) = left_nullspace_project(&h_f, &h_x, &res).unwrap();
        let sigma2 = 1.5_f64;
        let dim = 6 * n;
        let mut jm = DMatrix::<f64>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                jm[(i, j)] = (((i * 7 + j * 3) % 5) as f64) * 0.1;
            }
        }
        let p = &jm * jm.transpose() + DMatrix::identity(dim, dim);
        let chi2_of = |h: &DMatrix<f64>, r: &DVector<f64>| -> f64 {
            let mut s = h * &p * h.transpose();
            for k in 0..s.nrows() {
                s[(k, k)] += sigma2;
            }
            (r.transpose() * s.try_inverse().unwrap() * r)[(0, 0)]
        };
        let chi2_split = chi2_of(&hup, &res_up);
        let chi2_null = chi2_of(&h_o, &r_o);
        assert!(
            (chi2_split - chi2_null).abs() < 1e-6 * (1.0 + chi2_null.abs()),
            "χ² split {chi2_split} vs nullspace {chi2_null}"
        );
    }
}
