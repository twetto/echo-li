"""Direct clone-relative POSE NEES: real VIO + real frontend, scored against GT pose.

This is the CLEAN honesty test for the pose-clone covariance — the quantity that
drives the MSCKF over-downdate divergence. It removes Sparse3D entirely (that filter
is downstream: it CONSUMES the clone-relative pose covariance via §V-D and adds its
own depth-filter + triangulation/aperture noise, so a depth NEES only reports the
clone-pose honesty through a convolution). Here we score the clone-relative pose
error directly against the clone-relative pose covariance the EqF reports.

For every live clone `cid` at the current frame `k`:

  estimate  T̄_rel = T̄_clone⁻¹ · T̄_curr        (both = EqF camera poses in world)
  truth     T_rel_gt = ext⁻¹ · P_gt(cid)⁻¹ · P_gt(k) · ext   (camera-frame; the
            NED→NWU world flip cancels in the relative transform, so GT is read in
            its native NED frame — same identity the E2E harness uses at L455)
  error     δ = log(T̄_rel⁻¹ · T_rel_gt) ∈ se(3), ordered [ω(rot); v(trans)]

matching EXACTLY the perturbation convention the covariance is linearized in
(sparse_relative_pose_covariance, echo-li-core/src/lib.rs:851; g = log(T̄_rel⁻¹ ·
T_rel_actual), Jacobian [−Ad_{T_rel⁻¹} | I]; SE3::log in echo-lie/src/se3.rs:157).

The binding returns the two MARGINAL 3×3 blocks (p_vv, p_ww). Under a correct joint
Gaussian each marginal channel is itself Gaussian, so
    rot   NEES = ωᵀ p_ww⁻¹ ω  ~ χ²(3)
    trans NEES = vᵀ p_vv⁻¹ v  ~ χ²(3)
are valid marginal consistency checks (cross ω–v coupling is irrelevant to a marginal
channel). Per-channel is also what we WANT: it isolates the suspected attitude channel
(clone-window-phase0-gate.md measured rotation var ~lag^0.47, over-confidence GROWING
with track length) from translation, on the pose directly rather than through depth.

Reports full distributions per channel (median, mean, %>χ²₉₅, %>χ²₉₉ with dof=3) plus
a lag breakdown to expose the sub-linear cov growth signature directly on the pose.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md
import echo_li

# χ² thresholds, dof=3 (per-channel marginal NEES).
CHI2_95_3 = 7.8147
CHI2_99_3 = 11.3449


def se3_log(T):
    """SE(3) log matching echo-lie/src/se3.rs:157 exactly. Returns [ω; v] (rot first)."""
    R = T[:3, :3]
    t = T[:3, 3]
    omega = Rot.from_matrix(R).as_rotvec()
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]])
    if theta > 1e-6:
        coeff = (1.0 / (theta * theta)) * (
            1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
    else:
        coeff = 1.0 / 12.0
    v_inv = np.eye(3) - 0.5 * K + coeff * (K @ K)
    v = v_inv @ t
    return np.concatenate([omega, v])


def _deep_copy(d):
    import copy
    return copy.deepcopy(d)


def channel_stats(nees):
    nees = np.asarray(nees, float)
    nees = nees[np.isfinite(nees)]
    if nees.size == 0:
        return None
    return dict(
        n=int(nees.size),
        med=float(np.median(nees)),
        mean=float(np.mean(nees)),
        p95=100.0 * float(np.mean(nees > CHI2_95_3)),
        p99=100.0 * float(np.mean(nees > CHI2_99_3)),
    )


def print_channel(name, s):
    if s is None:
        print(f"  {name:6s}: (no samples)")
        return
    print(f"  {name:6s}: n={s['n']:6d}  med={s['med']:8.3f}  mean={s['mean']:11.2f}  "
          f">χ²₉₅={s['p95']:5.1f}%  >χ²₉₉={s['p99']:5.1f}%   (ideal med≈2.37, "
          f"tail 5.0/1.0%)")


def main():
    ap = argparse.ArgumentParser(
        description="Direct clone-relative POSE NEES for the EqF (no Sparse3D).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eqf-max-obs", type=int, default=40)
    ap.add_argument("--eqf-selection", choices=("grid", "existing-first"), default="grid")
    ap.add_argument("--eqf-max-depth", type=float, default=500.0)
    ap.add_argument("--clone-window", type=int, default=120,
                    help="rolling clone-window length (frames)")
    ap.add_argument("--min-lag", type=int, default=1,
                    help="skip clones younger than this many frames (lag=0 is the "
                         "degenerate just-cloned pose with ~0 error and ~0 cov)")
    # EqF noise overrides (same semantics as midair_e2e_depth_nees.py).
    ap.add_argument("--bias-acc", type=float, default=None)
    ap.add_argument("--bias-gyr", type=float, default=None)
    ap.add_argument("--vel-acc", type=float, default=None)
    ap.add_argument("--vel-gyr", type=float, default=None)
    ap.add_argument("--scene-depth", type=float, default=None)
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--external-tracks", default="",
                    help="SHARED-TRACKER BRIDGE: consume an external per-frame feature "
                         "track dump (OpenVINS KLT front-end via OV_TRACKDUMP: lines "
                         "'ts cam id u v') INSTEAD of the Rudolf-V tracker, so this filter "
                         "and OpenVINS run on byte-identical tracks. Removes the front-end "
                         "as a confound in the clone-pose-cov honesty comparison.")
    ap.add_argument("--external-t0", type=float, default=1.0,
                    help="bag epoch: external frame k = round((ts - t0) * fps)")
    ap.add_argument("--external-fps", type=float, default=25.0)
    ap.add_argument("--external-px-scale", type=float, default=None,
                    help="multiply external pixel coords by this to match the working "
                         "resolution (default = --scale, since OV runs at native 1024px "
                         "and this harness at args.scale*1024). Use 1.0 with --scale 1.0.")
    # --- Active MSCKF (makes this filter architecturally EQUIVALENT to OpenVINS) ---
    # Default OFF => passive clone mirror (covariance only, no measurement update).
    # ON => the additive structureless multi-state-constraint update: buffer per-track
    # obs at clone frames, and on track termination triangulate + nullspace-project +
    # gate + correct nav & clone poses (vio.msc_update). This is the "equivalent
    # component" that makes the ECHO-LI vs OpenVINS trajectory comparison honest —
    # both are then MSCKFs on the same KLT tracks.
    ap.add_argument("--msckf", action="store_true",
                    help="enable the ACTIVE MSCKF structureless update (vio.msc_update); "
                         "default off = passive clone mirror. Turn ON for the comparable "
                         "same-tracker + same-architecture trajectory vs OpenVINS.")
    ap.add_argument("--msckf-min-track", type=int, default=3,
                    help="min live observations for a track to enter msc_update.")
    ap.add_argument("--msckf-retry", action="store_true",
                    help="MSCEqF-faithful RETRY: a track that fails triangulation/gate is "
                         "NOT consumed — it is retried each frame (with progressively "
                         "corrected clones) until it passes OR its anchor clone marginalizes. "
                         "Mirrors updater.cpp:235-255 (removeTracksId removes only ACCEPTED "
                         "tracks). Default off = legacy use-once pop (the U3-split bug).")
    ap.add_argument("--msckf-chi2-mult", type=float, default=1.0,
                    help="multiplier on the 95%% chi² innovation gate in msc_update.")
    ap.add_argument("--msckf-sigma-pix", type=float, default=0.0,
                    help="pixel measurement noise for the MSC update only (0 => reuse "
                         "base-EqF sigma_bearing). Weights the structureless update.")
    ap.add_argument("--msckf-suppress-sensor", action="store_true",
                    help="DIAGNOSTIC: zero the MSC nav(sensor-21) mean-correction "
                         "(localizes scale injection to the sensor cross-cov).")
    ap.add_argument("--msckf-suppress-landmarks", action="store_true",
                    help="DIAGNOSTIC: zero the MSC in-state-landmark mean-correction.")
    ap.add_argument("--msckf-only", action="store_true",
                    help="DIAGNOSTIC: skip the every-frame EqVIO landmark update "
                         "(process_vision) entirely, so the ONLY vision correction is the "
                         "structureless MSCKF. Mirrors OpenVINS's pure-MSCKF architecture "
                         "(nav = seed+IMU+structureless update, no in-state landmark filter) "
                         "for an apples-to-apples comparison. Implies --msckf.")
    ap.add_argument("--gt-clones", action="store_true",
                    help="DIAGNOSTIC: overwrite each clone's stored pose GEOMETRY with "
                         "the GT camera pose (cov untouched) — isolates MSC update-math "
                         "bugs from drifted EqVIO clone geometry.")
    # --- Delayed (SLAM-feature) landmark initialization, mirroring OpenVINS
    # UpdaterSLAM::delayed_init + StateHelper::initialize_invertible. Promotes a
    # still-ALIVE track with enough multi-view observations to a CORRELATED in-state
    # landmark (geometry-derived P_LL + clone cross-cov), vs process_vision's guessed
    # DIAGONAL birth. Runs on the SAME external-track stream so ECHO-LI's SLAM features
    # match OpenVINS's. Compose with --msckf for the full OV (SLAM + structureless) mix.
    ap.add_argument("--delayed-init", action="store_true",
                    help="promote ready-and-alive tracks to correlated in-state landmarks "
                         "via vio.delayed_init (OpenVINS-mirror). Default off.")
    ap.add_argument("--delayed-init-min-obs", type=int, default=3,
                    help="min live observations before a track is promoted by delayed_init.")
    ap.add_argument("--delayed-init-chi2-mult", type=float, default=1.0,
                    help="multiplier on the chi² gate inside delayed_init.")
    ap.add_argument("--delayed-init-sigma-pix", type=float, default=0.0,
                    help="pixel measurement noise for delayed_init (0 => reuse base sigma).")
    # --- Seed ECHO-LI from OpenVINS's DYNAMIC-init state (bypass ECHO-LI static init).
    # Reads OV's total-state est dump (ts q p v bg ba ...), takes the row at ECHO-LI's
    # start timestamp, removes ONLY the unobservable 4-DOF gauge (yaw + position) using
    # GT, and hands OV's pose/velocity/bias — WITH their estimation error — to ECHO-LI.
    # Makes the head-to-head a FILTER comparison, not an init-method comparison.
    ap.add_argument("--seed-from-ov", default="",
                    help="path to OpenVINS total-state est file (ov_cc_*.txt); seeds "
                         "ECHO-LI from OV's dynamic-init state instead of GT.")
    ap.add_argument("--seed-ov-t0", type=float, default=1.0,
                    help="OV bag start time (s) matching --external-t0; the OV est row is "
                         "looked up at seed_ov_t0 + start/external_fps.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    if args.msckf_only:
        args.msckf = True  # MSCKF-only mirrors OV: structureless update is the ONLY vision

    # --- Dataset ---
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    ext = md.RT_BC.copy()
    ext_inv = np.linalg.inv(ext)

    # --- Shared-tracker bridge: load external (OpenVINS KLT) tracks ---
    # ext_tracks[k] = {track_id: (u, v)} at the working resolution. Frame index
    # k = round((ts - t0) * fps) aligns OV bag-time to this harness's camera frame
    # (echo-li frame k ↔ absolute MidAir time k/25; OV placed image k at ts=t0+k/25).
    ext_tracks = None
    if args.external_tracks:
        px_scale = args.external_px_scale if args.external_px_scale is not None else args.scale
        ext_tracks = {}
        n_obs = 0
        with open(args.external_tracks) as fh:
            for ln in fh:
                if ln.startswith("#") or not ln.strip():
                    continue
                p = ln.split()
                ts = float(p[0]); tid = int(p[2]); u = float(p[3]); v = float(p[4])
                k = int(round((ts - args.external_t0) * args.external_fps))
                ext_tracks.setdefault(k, {})[tid] = (u * px_scale, v * px_scale)
                n_obs += 1
        ks = sorted(ext_tracks)
        print(f"External tracks: {args.external_tracks}\n  {n_obs} obs over "
              f"{len(ks)} frames (k={ks[0]}..{ks[-1]}), px_scale={px_scale}, "
              f"median feats/frame={int(np.median([len(ext_tracks[k]) for k in ks]))}")
        if args.start < ks[0] or args.start + args.frames > ks[-1] + 1:
            print(f"  NOTE: requested frames {args.start}..{args.start + args.frames} "
                  f"extend beyond external-track coverage {ks[0]}..{ks[-1]}; "
                  f"uncovered frames get no vision update.")

    # --- Config merge (EqF overrides only; no Sparse3D) ---
    import yaml
    cfg = _deep_copy(yaml.safe_load(open(args.config)) or {})
    if args.bias_acc is not None:
        cfg.setdefault("eqf", {}).setdefault("initialVariance", {})["biasAcc"] = args.bias_acc
    if args.bias_gyr is not None:
        cfg.setdefault("eqf", {}).setdefault("initialVariance", {})["biasGyr"] = args.bias_gyr
    imu_sqrt_dt = 0.1  # 100 Hz IMU; velocityNoise expects density σ·√dt.
    if args.vel_acc is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["acc"] = args.vel_acc * imu_sqrt_dt
    if args.vel_gyr is not None:
        cfg.setdefault("eqf", {}).setdefault("velocityNoise", {})["gyr"] = args.vel_gyr * imu_sqrt_dt
    if args.scene_depth is not None:
        cfg.setdefault("eqf", {}).setdefault("initialValue", {})["sceneDepth"] = args.scene_depth
    if args.msckf:
        # Turn the additive MSCKF update ON in the Rust core (default OFF).
        eqf_settings = cfg.setdefault("eqf", {}).setdefault("settings", {})
        eqf_settings["enableMsckf"] = True
        eqf_settings["msckfWindow"] = args.clone_window
        eqf_settings["msckfMinTrack"] = args.msckf_min_track
        eqf_settings["msckfChi2Mult"] = args.msckf_chi2_mult
        if args.msckf_sigma_pix > 0.0:
            eqf_settings["msckfSigmaPix"] = args.msckf_sigma_pix
        if args.msckf_suppress_sensor:
            eqf_settings["msckfSuppressSensor"] = True
        if args.msckf_suppress_landmarks:
            eqf_settings["msckfSuppressLandmarks"] = True
    if args.delayed_init:
        eqf_settings = cfg.setdefault("eqf", {}).setdefault("settings", {})
        eqf_settings["enableDelayedInit"] = True
        eqf_settings["delayedInitMinObs"] = args.delayed_init_min_obs
        eqf_settings["delayedInitChi2Mult"] = args.delayed_init_chi2_mult
        if args.delayed_init_sigma_pix > 0.0:
            eqf_settings["delayedInitSigmaPix"] = args.delayed_init_sigma_pix
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        merged_config_path = tf.name

    # --- Frontend + VIO ---
    fcfg = echo_li.FrontendConfig.from_yaml(merged_config_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)

    vio = echo_li.VIOFilter(merged_config_path, cam)
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    gt0 = ds.pose(args.start)
    v0 = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][args.start * 4])
    R0 = t_ned_to_nwu @ gt0[:3, :3]           # GT body->NWU at the start frame
    p0_nwu = t_ned_to_nwu @ gt0[:3, 3]
    v0_nwu = t_ned_to_nwu @ v0
    v0_body = R0.T @ v0_nwu
    if args.seed_from_ov:
        # --- Seed from OpenVINS's dynamic-init state, bypassing ECHO-LI static init. ---
        # OV's world frame is GRAVITY-ALIGNED z-up (its init rotates the world so measured
        # gravity -> world -z), i.e. already NWU-like; it differs from ECHO-LI's GT-aligned
        # NWU by only a YAW gauge (OV anchors yaw to bag-frame-0, not this mid-flight
        # start). Verified empirically: G = R0_nwu @ R_ItoG^T is a near-pure yaw (z-axis
        # ~[0,0,1]) that is CONSTANT across frames, and the inherited attitude error below
        # is a few deg (a wrong axis/transpose gives ~180deg). So NO NED flip is applied to
        # the OV side; the residual yaw + position gauge (both UNOBSERVABLE) is removed with
        # GT. OV's OBSERVABLE errors (roll/pitch tilt, velocity scale, bias) are PRESERVED
        # and inherited by ECHO-LI -- the point of a fair FILTER comparison.
        seed_ts = args.seed_ov_t0 + args.start / args.external_fps
        est = np.loadtxt(args.seed_from_ov, comments="#")
        j = int(np.argmin(np.abs(est[:, 0] - seed_ts)))
        dt_seed = abs(est[j, 0] - seed_ts)
        if dt_seed > 1.0 / args.external_fps:
            raise SystemExit(
                f"--seed-from-ov: no OV est row near seed_ts={seed_ts:.3f}s "
                f"(nearest {est[j, 0]:.3f}s, dt={dt_seed:.3f}s). OV likely had not "
                f"initialized by --start={args.start}; pick a later start.")
        q = est[j, 1:5]                # JPL q_GtoI as (qx,qy,qz,qw)
        p_ov, v_ov = est[j, 5:8], est[j, 8:11]
        bg, ba = est[j, 11:14], est[j, 14:17]
        # scipy (Hamilton) on JPL components gives R_ItoG = body->OV-world directly
        # (R_Ham(q) = R_JPL(q)^T = R_GtoI^T). No frame flip: OV-world is already z-up.
        R_ItoG = Rot.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        Rp = R_ItoG                    # body->NWU up to the residual yaw gauge below
        pp, vp = p_ov, v_ov
        yaw = lambda R: np.arctan2(R[1, 0], R[0, 0])
        psi = yaw(R0) - yaw(Rp)        # match GT yaw gauge about NWU z (gravity)
        c, s = np.cos(psi), np.sin(psi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R_seed = Rz @ Rp
        v_seed = Rz @ vp
        p_seed = p0_nwu                # position origin is a pure gauge -> anchor to GT
        # Report the OBSERVABLE error ECHO-LI inherits (post yaw+pos gauge removal).
        tilt = np.degrees(np.arccos(np.clip((np.trace(R_seed.T @ R0) - 1.0) / 2.0, -1.0, 1.0)))
        v_err = float(np.linalg.norm(v_seed - v0_nwu))
        print(f"[seed-from-ov] OV row @ {est[j,0]:.3f}s (dt={dt_seed*1e3:.0f}ms)  "
              f"inherited attitude err {tilt:.2f}deg | vel err {v_err:.3f} m/s "
              f"(|v_gt|={np.linalg.norm(v0_nwu):.2f}) | bg={bg} ba={ba}")
        vio.set_initial_state_full(
            p_seed.tolist(), np.ascontiguousarray(R_seed), v_seed.tolist(),
            bg.tolist(), ba.tolist(),
        )
    else:
        vio.set_initial_state(
            p0_nwu.tolist(),
            np.ascontiguousarray(R0),
            v0_body.tolist(),
        )

    print(f"Config: {args.config}")
    print(f"Traj {args.traj}  frames {args.start}..{args.start + args.frames}  "
          f"clone_window={args.clone_window}  min_lag={args.min_lag}")
    print("NO Sparse3D — scoring the EqF clone-relative pose covariance directly "
          "against GT pose.\n")

    def vio_body_pose():
        pos, quat = vio.get_pose()
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
        T[:3, 3] = np.asarray(pos)
        return T

    def _grid_select(all_uvs, existing, budget):
        import math
        n_cols = n_rows = 5
        n_cells = n_cols * n_rows
        per_cell = math.ceil(budget / n_cells)
        cw, ch = W / n_cols, H / n_rows
        cells = [[] for _ in range(n_cells)]
        for fid, (u, v) in all_uvs.items():
            ci = min(int(u / cw), n_cols - 1)
            ri = min(int(v / ch), n_rows - 1)
            cells[ri * n_cols + ci].append(fid)
        selected = []
        for cell in cells:
            cell.sort(key=lambda fid: (0 if fid in existing else 1, fid))
            selected.extend(cell[:per_cell])
        if len(selected) > budget:
            selected.sort(key=lambda fid: (0 if fid in existing else 1, fid))
            selected = selected[:budget]
        return selected

    # --- Event list (IMU @100Hz, cam @25Hz) ---
    nimg = min(args.frames, ds.n - args.start)
    imu_data = ds.db[ds.traj]["imu"]
    accel = imu_data["accelerometer"][:]
    gyro_raw = imu_data["gyroscope"][:]
    imu0 = args.start * 4
    imu1 = min(len(accel), (args.start + nimg) * 4 + 4)
    events = [(i / 100.0, "imu", (gyro_raw[i].tolist(), accel[i].tolist()))
              for i in range(imu0, imu1)]
    for n in range(nimg):
        k = args.start + n
        events.append((k / 25.0, "cam", k))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "imu" else 1))

    clone_pose = {}   # clone_id (=frame k) -> estimated camera pose T_wc at clone time
    # Per-observation records: (lag, rot_nees, trans_nees, rot_err_deg, trans_err_m).
    rows = []
    # Per-observation ROTATION-cov term decomposition (lag, tr_curr, tr_clone,
    # tr_cross, tr_pww) — localizes the lag^0.47 sub-linear attitude-cov growth to
    # a specific propagation term: p_ww = term_curr + term_clone - term_cross.
    term_rows = []
    # Per-cam-frame trajectory from THIS SAME run (so the plotted path is exactly the
    # filter run scored above): (k, est_pos_nwu, gt_pos_nwu). ext has zero translation,
    # so the camera and body positions coincide -> est = T_curr_est[:3,3].
    traj_rows = []
    _speed_rows = []
    _bias_rows = []  # (frame, |gyro_bias| rad/s, gbx, gby, gbz, tilt°) — spurious-bias probe
    _att_corr_rows = []  # (frame, applied dtheta_att°, |dvel|, n_acc) per msc_update
    _att_nees_rows = []  # (frame, |e_att| rad, σ_att rad, NEES, tilt°) attitude consistency
    _vel_nees_rows = []  # (frame,|e_v|,σ_v,NEES3,along_err,along_nees,|v_est|,|v_gt|,tilt) vel/scale consistency
    _clone_dir_rows = []  # (cos_dir, |corr|°, |e_pre|°) clone-attitude corrections (directly-informed)
    _vel_corr_rows = []  # (frame,cos_vel,|a_vel|,|e_vel|,along_sign,along_ratio,tilt) velocity-correction direction (world frame)
    # Active-MSCKF bookkeeping (mirrors midair_e2e_depth_nees.py): per-track obs
    # buffer at clone frames + the disjoint set of ids the EqF ever held in-state
    # (those are already estimated by process_vision; reusing them as a structureless
    # constraint would double-count the same pixels).
    track_obs = {}        # track_id -> list[(clone_id, (u, v))]
    eqf_ever = set()
    msckf_accepted = 0
    _cadence = [0, 0, 0, 0]  # [frames_processed, frames_ready(msc called), frames_acc(n_acc>0), sum_ready_tracks]
    _vpseudo_dbg = [0]  # c92 velocity-pseudo-measurement 1-frame convergence-check print counter
    delayed_born = 0
    _MSC_DBG = bool(os.environ.get("ECHO_MSC_DBG"))
    _msc_dbg_fires = [0]
    _msc_sk = []   # per-accepted-track (chi2, dof, s_geom, s_full, dx_rot, dx_pos, dx_vel)
    _msc_dnav = []  # per-FRAME-batch nav delta (|dp| m, |dtheta| rad) — matches OV OV_MSCDUMP per-update |dp|/|dtheta|
    _msc_covba = []  # per-update nav-attitude cov (trace P_ww rad²) BEFORE/AFTER the MSC update
                     # — pins the ~10× under-tightening to PROPAGATION (before re-inflates each
                     # frame ⇒ apply_transport) vs UPDATE (before≈after ⇒ constraint info too weak).
    if args.msckf:
        print(f"ACTIVE MSCKF update ON  (min_track={args.msckf_min_track}, "
              f"chi2_mult={args.msckf_chi2_mult}, window={args.clone_window}) "
              f"— architecturally equivalent to OpenVINS MSCKF.\n")
    t0 = time.time()
    n_frames = 0
    # MSCEqF-identity: the reference filter uses given-origin init, whose
    # initialize(TriangulatedFeatures&) calls setGivenOrigin and RETURNS without
    # processFeatures — the first vision frame's features are discarded (no track,
    # no clone). echo-li is already state-initialized from config, so to mirror the
    # cadence we must likewise skip the FIRST vision frame entirely: no clone, no
    # obs buffering, no update. Otherwise echo-li carries a permanent one-clone lead
    # (first clone at t=0 vs MSCEqF t=0.04) and one extra obs per track, which trips
    # min_track_length a frame early and desyncs every subsequent per-update set.
    _init_frame_skipped = False

    for stamp, et, data in events:
        if et == "imu":
            gyr, acc = data
            gi = min(int(round(stamp * 100.0)), len(ds.att) - 1)
            q = ds.att[gi]
            R_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            gyr_body = (R_gt.T @ np.asarray(gyr)).tolist()
            vio.process_imu(stamp, gyr_body, acc)
            continue

        k = data
        if not _init_frame_skipped:
            _init_frame_skipped = True  # mirror MSCEqF given-origin init (frame 0 discarded)
            continue
        if ext_tracks is not None:
            # Shared-tracker bridge: consume OV's KLT tracks; skip Rudolf-V + image I/O.
            all_uvs = ext_tracks.get(k, {})
        else:
            gray = ds.image(k)
            feats, _stats = tracker.process(gray)
            all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}

        existing = {int(x) for x in vio.get_landmarks().keys()}
        # --msckf-only: skip the every-frame EqVIO landmark update so the ONLY vision
        # correction is the structureless MSCKF (OV-mirror). Tracks are still buffered
        # for msc_update below; no in-state landmarks are ever added, so `existing`/
        # `eqf_ever` stay empty and every track feeds the MSCKF (like OV structureless).
        if not args.msckf_only:
            obs_ids = sorted(all_uvs.keys())
            if args.eqf_max_obs > 0 and len(obs_ids) > args.eqf_max_obs:
                if args.eqf_selection == "grid":
                    obs_ids = _grid_select(all_uvs, existing, args.eqf_max_obs)
                else:
                    keep_e = [fid for fid in obs_ids if fid in existing]
                    keep_n = [fid for fid in obs_ids if fid not in existing]
                    obs_ids = (keep_e + keep_n)[:args.eqf_max_obs]
            vio.process_vision(stamp, {fid: all_uvs[fid] for fid in obs_ids})

        # Current EqF camera pose, and GT camera pose at k.
        T_curr_est = vio_body_pose() @ ext

        # Clone THIS frame's pose, then (if active) run the MSCKF update, then roll
        # the window.
        vio.clone_pose(int(k), stamp)
        clone_pose[int(k)] = T_curr_est.copy()

        # DIAGNOSTIC (--gt-clones): overwrite the fresh clone's stored GEOMETRY with
        # the GT camera pose (covariance untouched). Distinguishes "the MSC nav
        # correction is buggy" from "the EqVIO clone poses have drifted": with GT
        # clone geometry the multi-view triangulation + Jacobians are globally
        # consistent, so a correct active-sensor MSC should track; if it still
        # diverges, the update math/cov is the culprit, not the drifted geometry.
        if args.gt_clones:
            M = np.eye(4); M[:3, :3] = t_ned_to_nwu
            T_gt_wc = M @ ds.pose(int(k)) @ ext
            vio.set_clone_pose_value(int(k), np.ascontiguousarray(T_gt_wc))
            clone_pose[int(k)] = T_gt_wc.copy()

        # Cap the clone window to EXACTLY args.clone_window by COUNT (mirrors MSCEqF
        # num_clones). Identify the oldest clones that must roll off THIS frame, but
        # DON'T marginalize yet: any still-alive track observed at a rolling-off clone
        # must first be USED in the MSC update (case (b) below), else its multi-view
        # scale constraint is silently discarded — the information-loss that drives the
        # compounding scale drift. MSCEqF/OpenVINS use these "max-length" features at
        # marginalization time; echo-li previously only used fully-terminated tracks.
        # MSCEqF marginalizes when clonesSize()==num_clones (checked AFTER cloning,
        # applied AFTER the update): the update sees exactly num_clones clones and
        # num_clones-1 persist between frames. Mirror that here — persist
        # (clone_window-1) so the update sees exactly clone_window (== MSCEqF
        # num_clones). Previously we persisted clone_window, so the update saw
        # clone_window+1 (12 vs MSCEqF's 11): a top-end off-by-one.
        live_sorted = sorted(int(c) for c in vio.clone_ids())
        n_over = max(0, len(live_sorted) - (args.clone_window - 1))
        to_marg = set(live_sorted[:n_over])   # clones that will be removed after the update

        # --- Active MSCKF structureless update (equivalent component to OpenVINS) ---
        # Buffer this frame's obs at the fresh clone k, flush terminated tracks
        # (present before, absent now) through msc_update — triangulate + nullspace-
        # project + gate + correct nav & clone poses. Done BEFORE marginalization so a
        # terminating track's anchor clones are still live. The nav pose is corrected,
        # so re-read T_curr_est afterwards (this is what the trajectory dump records).
        if args.msckf or args.delayed_init:
            eqf_ever.update(existing)  # ids the EqF holds/held in-state
            seen = set()
            for fid, uv in all_uvs.items():
                track_obs.setdefault(int(fid), []).append(
                    (int(k), (float(uv[0]), float(uv[1]))))
                seen.add(int(fid))
            live_ids = {int(c) for c in vio.clone_ids()}

            # --- Delayed (SLAM-feature) init on still-ALIVE tracks ---
            # Promote a track ECHO-LI does NOT already hold in-state to a CORRELATED
            # in-state landmark once it has enough multi-view obs. Guarded by
            # existing/eqf_ever so we never double-birth process_vision's tracks; birthed
            # ids join eqf_ever so the structureless MSCKF below leaves them alone
            # (mirrors OpenVINS SLAM vs MSCKF partition). Done while the track is alive so
            # future frames keep updating it through the EqF (that is the point of SLAM
            # features). Buffer NOT popped for alive tracks.
            if args.delayed_init:
                births = {}
                for tid in list(seen):
                    if tid in existing or tid in eqf_ever:
                        continue
                    obs = [(c, uv) for (c, uv) in track_obs.get(tid, []) if c in live_ids]
                    if len(obs) >= args.delayed_init_min_obs:
                        births[tid] = obs
                if births:
                    n_born = vio.delayed_init(births)
                    delayed_born += n_born
                    if n_born:
                        eqf_ever.update(births.keys())
                        T_curr_est = vio_body_pose() @ ext  # birth nudges the mean
                        clone_pose[int(k)] = T_curr_est.copy()

            # --- Flush tracks for the MSC update ---
            # (a) TERMINATED tracks (present before, absent now): always pop.
            # (b) MAX-LENGTH tracks: still-alive tracks observed at a clone that is
            #     about to be marginalized (to_marg). Use them now — their scale
            #     constraint would otherwise be discarded when the clone rolls off.
            #     This is the OpenVINS/MSCEqF "marginalized features" path echo-li was
            #     missing. Used tracks are popped (structureless = use-once); if the
            #     feature keeps being tracked it simply restarts a fresh track.
            ready = {}
            flush_ids = [t for t in track_obs if t not in seen]           # (a)
            if to_marg:
                flush_ids += [t for t in track_obs
                              if t in seen
                              and any(c in to_marg for (c, _) in track_obs[t])]  # (b)
            # U3 FLUSH DIAGNOSTIC (ECHO_MSC_U3DBG=1): why do MSCEqF-kept tracks
            # {83,151,184} not enter echo's msc_update at U3 (frame 10)?
            import os as _os
            _u3dbg = _os.environ.get("ECHO_MSC_U3DBG") == "1" and int(k) in (6, 7, 8, 9, 10)
            if _u3dbg:
                print(f"[U3DBG] --- frame k={k} ---")
                for _t in (83, 151, 184):
                    _all = track_obs.get(_t, [])
                    _liv = [c for (c, _) in _all if c in live_ids]
                    print(f"[U3DBG] tid={_t} in_track_obs={_t in track_obs} "
                          f"n_all={len(_all)} n_live={len(_liv)} "
                          f"alive={_t in seen} terminated={_t in track_obs and _t not in seen} "
                          f"in_flush={_t in flush_ids} in_eqf_ever={_t in eqf_ever} "
                          f"live_clone_span={min(_liv) if _liv else None}..{max(_liv) if _liv else None} "
                          f"to_marg={sorted(to_marg)} n_clones={len(live_sorted)}")
            for tid in flush_ids:
                obs = [(c, uv) for (c, uv) in track_obs[tid] if c in live_ids]
                if tid in eqf_ever:  # disjoint from in-state EqF landmarks
                    track_obs.pop(tid, None)
                    continue
                if len(obs) >= args.msckf_min_track:
                    ready[tid] = obs
                    if not args.msckf_retry:
                        track_obs.pop(tid, None)   # legacy use-once: consume at flush
                else:
                    # too few LIVE obs to triangulate
                    if (not args.msckf_retry) or (tid not in seen and not obs):
                        # legacy always drops; retry drops only a terminated track whose
                        # every clone has marginalized (anchor gone => can never pass) —
                        # mirrors removeTracksTail trimming a track below viability.
                        track_obs.pop(tid, None)
            # ATTITUDE-AUTHORITY TRAJECTORY DUMP (ECHO_MSC_TRAJDUMP=1). Capture the
            # nav pose+velocity right before the frame's msc_update so we can record
            # the APPLIED per-frame correction (dtheta_att, |dvel|, |dpos|) alongside
            # the running tilt. Distinguishes CHRONIC attitude under-correction (the
            # applied dtheta_att stays a ~constant fraction of what tilt-growth needs)
            # from AUTHORITY COLLAPSE (dtheta_att drops toward 0 at the runaway onset
            # while tilt accelerates) — different fixes.
            _td_on = os.environ.get("ECHO_MSC_TRAJDUMP") == "1"
            _td_pre_R = _td_pre_v = None
            _clone_pre = None
            if _td_on and args.msckf and ready:
                _td_pre_R = Rot.from_quat(np.asarray(vio.get_pose()[1])).as_matrix()
                _td_pre_v = np.asarray(vio.get_velocity(), float).copy()
                # Snapshot every live clone's attitude BEFORE the update so we can
                # measure whether the DIRECTLY-informed clone corrections are well-
                # directed (vs the nav-attitude which is corrected only via Σ[nav,clone]).
                # If clones correct toward GT but nav-attitude doesn't ⇒ the mis-route
                # is the nav↔clone CROSS-BLOCK orientation, not the clone update u.
                _clone_pre = {}
                for _cid in vio.clone_ids():
                    _cm = vio.clone_pose_value(int(_cid))
                    if _cm is not None:
                        _clone_pre[int(_cid)] = np.asarray(_cm, float)[:3, :3].copy()
            # c94 GAMMACMP: stash GT WORLD velocity so the vision msc_update prints
            # cos(vision γ_v, pseudo γ_v). Set BEFORE the update. c95 LAG discriminator:
            # ECHO_GAMMA_CMP_LAG=<frames> feeds the pseudo-reference a TIME-LAGGED GT
            # velocity v_gt(k−lag). Monocular vision infers WINDOW-AVERAGE velocity from
            # clone positions, which lags the instantaneous v_gt(k). If early-regime cos
            # rises toward +1 at lag≈window/2, the c94 cos<0 was a lag artifact (NOT a
            # misdirection bug); if it stays negative at ALL lags, it is a real defect.
            if os.environ.get("ECHO_GAMMA_CMP") == "1" and args.msckf and ready:
                _lag = int(os.environ.get("ECHO_GAMMA_CMP_LAG", "0"))
                _ki = max(0, int(k) - _lag)  # GT vel at 100Hz vs frames 25Hz -> *4
                _vgt_w = t_ned_to_nwu @ np.asarray(
                    ds.db[ds.traj]["groundtruth"]["velocity"][_ki * 4], float)
                vio.set_dbg_v_gt_body(_vgt_w.tolist())
            if args.msckf:
                _cadence[0] += 1
                if ready:
                    _cadence[1] += 1
                    _cadence[3] += len(ready)
            if args.msckf and ready:
                if _MSC_DBG:
                    T_before = vio_body_pose().copy()
                    p_before = T_before[:3, 3].copy()
                    import numpy as _np
                    _cb = vio.get_camera_pose_covariance()
                    _pww_before = float(_np.trace(_cb[1])) if _cb is not None else float("nan")
                    n_acc, dbg_rows = vio.msc_update_debug(ready)
                    msckf_accepted += n_acc
                    T_after = vio_body_pose().copy()
                    p_after = T_after[:3, 3].copy()
                    _ca = vio.get_camera_pose_covariance()
                    _pww_after = float(_np.trace(_ca[1])) if _ca is not None else float("nan")
                    if n_acc:
                        _msc_covba.append((_pww_before, _pww_after))
                    # per-FRAME-batch nav delta (one msc_update call = one applied gamma,
                    # aggregating all ready tracks) — the apples-to-apples match for OV's
                    # per-EKFUpdate |dp|/|dtheta| (OV also batches a frame's lost feats).
                    if n_acc:
                        dR = T_before[:3, :3].T @ T_after[:3, :3]
                        _dtheta = _np.arccos(min(1.0, max(-1.0, (_np.trace(dR) - 1) / 2)))
                        _msc_dnav.append((float(_np.linalg.norm(p_after - p_before)), float(_dtheta)))
                    # Accumulate S/K over ALL accepted tracks, whole run (row =
                    # tid,nobs,rms,chi2,dof,triDepth,triRange,acc,s_geom,s_full,dxRot,dxPos,dxVel)
                    for r in dbg_rows:
                        if r[7]:
                            _msc_sk.append((r[3], r[4], r[8], r[9], r[10], r[11], r[12], r[2], r[1]))
                    if _msc_dbg_fires[0] < 8:
                        _msc_dbg_fires[0] += 1
                        gt_p = t_ned_to_nwu @ P_k[:3, 3]
                        acc = [r for r in dbg_rows if r[7]]
                        def _med(xs):
                            return float(_np.median(xs)) if xs else float("nan")
                        chi = _med([r[3] for r in acc]); dof = _med([r[4] for r in acc])
                        sg = _med([r[8] for r in acc]); sf = _med([r[9] for r in acc])
                        dxp = _med([r[11] for r in acc]); dxr = _med([r[10] for r in acc])
                        dxv = _med([r[12] for r in acc])
                        print(f"  [MSC-DBG k={int(k)}] ready={len(ready)} acc={n_acc}/{len(dbg_rows)} "
                              f"|dnav|={_np.linalg.norm(p_after - p_before):.3f}m  "
                              f"navErr {_np.linalg.norm(p_before - gt_p):.2f}->{_np.linalg.norm(p_after - gt_p):.2f}m | "
                              f"ACC med: chi2/dof={chi/max(dof,1):.2f} s_geom={sg:.2f} s_full={sf:.2f}px² "
                              f"dx_pos={dxp*100:.2f}cm dx_rot={_np.degrees(dxr):.3f}° dx_vel={dxv:.4f}m/s", flush=True)
                else:
                    # Route through the debug variant so we always learn WHICH tracks were
                    # accepted (needed for the MSCEqF-faithful retry). Correction is identical
                    # to msc_update (both call msc_update_impl).
                    n_acc, dbg_rows = vio.msc_update_debug(ready)
                    msckf_accepted += n_acc
                T_curr_est = vio_body_pose() @ ext   # nav pose corrected by the update
                clone_pose[int(k)] = T_curr_est.copy()
                if n_acc:
                    _cadence[2] += 1
                if _td_on and _td_pre_R is not None and n_acc:
                    _R_post = Rot.from_quat(np.asarray(vio.get_pose()[1])).as_matrix()
                    _v_post = np.asarray(vio.get_velocity(), float)
                    _dR = _td_pre_R.T @ _R_post
                    _dth = float(np.degrees(np.arccos(
                        min(1.0, max(-1.0, (np.trace(_dR) - 1) / 2)))))
                    # MIS-DIRECTION test (decisive fix-direction discriminator): does
                    # the applied attitude correction actually rotate the estimate
                    # TOWARD GT? In the pre-update body frame the correction needed to
                    # reach GT is e_pre = log(R_pre^T R_gt); the applied correction is
                    # a_vec = log(R_pre^T R_post). cos_dir=<a,e_pre>/(|a||e_pre|): ~+1 =
                    # well-directed (deficit is pure magnitude), ~0/neg = mis-routed
                    # (magnitude knobs can't help — matches razor-thin ATTSCALE/PROCATT).
                    # dred = |e_pre|-|e_post| (deg): >0 the update reduced tilt error.
                    _Rgt_k = t_ned_to_nwu @ ds.pose(int(k))[:3, :3]
                    _e_pre = Rot.from_matrix(_td_pre_R.T @ _Rgt_k).as_rotvec()
                    _e_post = Rot.from_matrix(_R_post.T @ _Rgt_k).as_rotvec()
                    _a_vec = Rot.from_matrix(_dR).as_rotvec()
                    _na = float(np.linalg.norm(_a_vec)); _ne = float(np.linalg.norm(_e_pre))
                    _cos_dir = (float(np.dot(_a_vec, _e_pre) / (_na * _ne))
                                if _na > 1e-12 and _ne > 1e-12 else float("nan"))
                    _dred = float(np.degrees(_ne - np.linalg.norm(_e_post)))
                    # PROPAGATION-vs-UPDATE decomposition (rules out the "drift is
                    # propagation/gyro-bias, roll/pitch weakly observable" alternative):
                    # gravity-based tilt (yaw-gauge-free) BEFORE and AFTER the update.
                    # Per frame: prop increment = tilt_pre[k]-tilt_post[k-1] (IMU-only
                    # drift), update change = tilt_post[k]-tilt_pre[k]. If prop increment
                    # ≪ available authority and the update fails to reduce it ⇒ routing;
                    # if prop increment ≫ authority ⇒ propagation-limited.
                    _gn = np.array([0.0, 0.0, -9.81])
                    _gg = _Rgt_k.T @ _gn
                    def _tilt(_R):
                        _ge = _R.T @ _gn
                        _c = float(np.dot(_ge, _gg) / (np.linalg.norm(_ge) * np.linalg.norm(_gg)))
                        return float(np.degrees(np.arccos(max(-1.0, min(1.0, _c)))))
                    # whitened innovation magnitude √Σχ² (common-driver of ALL channel
                    # corrections) — to partial out of corr(dtheta_att,|dvel|).
                    _innov = float(np.sqrt(sum(r[3] for r in dbg_rows if r[7])))
                    _att_corr_rows.append((
                        int(k), _dth, float(np.linalg.norm(_v_post - _td_pre_v)),
                        int(n_acc), _cos_dir, _dred, _tilt(_td_pre_R), _tilt(_R_post), _innov))
                    # VELOCITY-CORRECTION DIRECTION (echo-only analog of the attitude
                    # cos_dir; the decisive discriminator for the velocity/scale under-
                    # correction seed). Physical velocity in WORLD frame makes the applied
                    # correction attitude-artifact-free: a_vel = R_post·v_post − R_pre·v_pre,
                    # error-to-GT e_vel = v_gt_world − R_pre·v_pre. cos_vel≈+1 ⇒ the update
                    # pushes |v| toward GT and the deficit is pure MAGNITUDE (gain fix);
                    # cos_vel≈0/neg ⇒ MIS-DIRECTED (routing fix — align_vel 0.83× seam).
                    # along_ratio = (v̂·a_vel)/(v̂·e_vel) on the pre-update speed axis is the
                    # scale-channel gain: 1 = fully corrected, <1 = under-corrected.
                    _vpre_w = _td_pre_R @ _td_pre_v
                    _vpost_w = _R_post @ _v_post
                    _vgt_w = t_ned_to_nwu @ np.asarray(
                        ds.db[ds.traj]["groundtruth"]["velocity"][int(k) * 4], float)
                    _av = _vpost_w - _vpre_w
                    _ev = _vgt_w - _vpre_w
                    _nav_ = float(np.linalg.norm(_av)); _nev = float(np.linalg.norm(_ev))
                    _cos_vel = (float(np.dot(_av, _ev) / (_nav_ * _nev))
                                if _nav_ > 1e-12 and _nev > 1e-12 else float("nan"))
                    _spd = float(np.linalg.norm(_vpre_w))
                    if _spd > 1e-9:
                        _vhat = _vpre_w / _spd
                        _a_al = float(np.dot(_vhat, _av)); _e_al = float(np.dot(_vhat, _ev))
                        _al_sign = float(np.sign(_a_al * _e_al))
                        _al_ratio = (_a_al / _e_al) if abs(_e_al) > 1e-9 else float("nan")
                    else:
                        _al_sign = _al_ratio = float("nan")
                    _vel_corr_rows.append((
                        int(k), _cos_vel, _nav_, _nev, _al_sign, _al_ratio, _tilt(_td_pre_R)))
                    # CLONE-vs-NAV routing bisection: the update informs clones DIRECTLY
                    # (measurement Jacobian has clone columns) but the nav attitude only
                    # through Σ[nav,clone]. Measure the DIRECTLY-informed clone-attitude
                    # correction's alignment with its own GT error. Well-directed clones
                    # + mis-directed nav-attitude ⇒ the bug is the cross-block orientation.
                    if _clone_pre:
                        _re3 = ext[:3, :3]
                        for _cid, _Rc_pre in _clone_pre.items():
                            _cm2 = vio.clone_pose_value(int(_cid))
                            if _cm2 is None:
                                continue
                            _Rc_post = np.asarray(_cm2, float)[:3, :3]
                            _Rc_gt = (t_ned_to_nwu @ ds.pose(int(_cid))[:3, :3]) @ _re3
                            _ac = Rot.from_matrix(_Rc_pre.T @ _Rc_post).as_rotvec()
                            _ec = Rot.from_matrix(_Rc_pre.T @ _Rc_gt).as_rotvec()
                            _nac = float(np.linalg.norm(_ac)); _nec = float(np.linalg.norm(_ec))
                            if _nac > 1e-9 and _nec > 1e-9:
                                _clone_dir_rows.append((
                                    float(np.dot(_ac, _ec) / (_nac * _nec)),
                                    float(np.degrees(_nac)), float(np.degrees(_nec))))
                # --- consume vs retry (mirrors updater.cpp:235-255 + removeTracksId): only
                # ACCEPTED tracks are removed; rejected ones stay in track_obs and are retried
                # next frame with progressively corrected clones. Legacy already popped ready
                # tracks at flush above. ---
                if args.msckf_retry:
                    accepted_tids = {int(r[0]) for r in dbg_rows if r[7]}
                    for tid in list(ready):
                        if tid in accepted_tids:
                            track_obs.pop(tid, None)   # consumed (removeTracksId)
                        # else: rejected -> keep for retry

        # CLONEEVO: watch the FIXED (frame1, frame5) clone pair — tid83's anchor/last —
        # evolve through U0->U3. Distinguishes pure-cadence (rel_ang reaches MSCEqF 2.22511
        # by U3 => delaying the flush alone fixes it) from cadence+under-correction
        # (rel_ang stays ~2.16 => echo also under-corrects clone attitude). Zero-perturbation
        # readback via clone_pose_value. Env-gated.
        if os.environ.get("ECHO_CLONEEVO") == "1":
            M1 = vio.clone_pose_value(1)
            M5 = vio.clone_pose_value(5)
            if M1 is not None and M5 is not None:
                Rrel = M1[:3, :3].T @ M5[:3, :3]
                ang = np.degrees(np.arccos(min(1.0, max(-1.0, (np.trace(Rrel) - 1) / 2))))
                relt = M1[:3, :3].T @ (M5[:3, 3] - M1[:3, 3])
                # GT reference: rel-rotation ANGLE is invariant to the fixed cam extrinsic
                # and the NED/NWU axis flip (both similarities), so ds.pose (body/NED) is valid.
                Rg = ds.pose(1)[:3, :3].T @ ds.pose(5)[:3, :3]
                ang_gt = np.degrees(np.arccos(min(1.0, max(-1.0, (np.trace(Rg) - 1) / 2))))
                print(f"CLONEEVO k={int(k)} rel_ang15={ang:.5f} (gt {ang_gt:.5f}) "
                      f"rel_bl15={np.linalg.norm(relt):.5f}", flush=True)

        # Now marginalize the rolled-off clones — AFTER the MSC update has consumed
        # every max-length track observed at them (case (b) above). Caps the window at
        # exactly args.clone_window, matching MSCEqF num_clones.
        for cid in to_marg:
            vio.marginalize_clone(int(cid))
            clone_pose.pop(cid, None)

        # Score every live clone's relative pose against GT.
        P_k = ds.pose(k)
        traj_rows.append((k, T_curr_est[:3, 3].copy(), (t_ned_to_nwu @ P_k[:3, 3]).copy()))
        # SPEED RUNAWAY (frame-invariant): filter body-speed vs GT speed. |v| is
        # invariant to attitude/frame, so this isolates the velocity/scale channel
        # from any rotation-convention confound — does the ESTIMATE's speed blow up?
        if os.environ.get("ECHO_MSC_TRAJDUMP") == "1":
            try:
                vf = float(np.linalg.norm(np.asarray(vio.get_velocity(), float)))
                vg = float(np.linalg.norm(np.asarray(
                    ds.db[ds.traj]["groundtruth"]["velocity"][k * 4], float)))
                # TILT error vs GT via the gravity direction (frame/yaw-gauge robust):
                # g in each body frame = R_body_to_world^T · g_world; the angle between
                # est and GT body-gravity is exactly the roll/pitch (gravity-relevant)
                # attitude error. gravity-leak ⇒ this LEADS the |v| runaway with
                # d|v|/dt ≈ g·sin(tilt); direct scale ⇒ tilt stays small.
                g_nwu = np.array([0.0, 0.0, -9.81])
                R_est = Rot.from_quat(np.asarray(vio.get_pose()[1])).as_matrix()
                R_gt = t_ned_to_nwu @ ds.pose(k)[:3, :3]
                ge = R_est.T @ g_nwu
                gg = R_gt.T @ g_nwu
                cs = float(np.dot(ge, gg) / (np.linalg.norm(ge) * np.linalg.norm(gg)))
                tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, cs)))))
                _speed_rows.append((int(k), vf, vg, tilt))
                # SPURIOUS GYRO-BIAS PROBE: true MidAir gyro bias ≈ 0 (0.0005 rad/s,
                # measured). If echo's vision update drives an estimated bias to a
                # significant nonzero value it directly corrupts propagation via
                # exp(dt·(gyr − b)). ~0.003 rad/s would explain the 0.0034→0.0109°/f
                # propagation inflation (pure-IMU vs with-vision).
                _gb, _ba = vio.get_biases()
                _gb = np.asarray(_gb, float)
                _ba = np.asarray(_ba, float)
                # Σ[gyro_bias, clone/att] cross-cov: the leverage that lets the mono
                # structureless update dump residual into gyro bias. tangent order
                # bg[0:3], att[6:9]; clones start at 21 (msckf-only ⇒ no in-state lm).
                _rho_bg_att = _rho_bg_cl = float("nan")
                try:
                    _C = np.asarray(vio.get_full_covariance(), float)
                    _tr = lambda a: np.trace(_C[a:a+3, a:a+3])
                    _fro = lambda r, c, m: np.sqrt(np.sum(_C[r:r+3, c:c+m]**2))
                    _dbg = np.sqrt(max(_tr(0), 1e-30))
                    _datt = np.sqrt(max(_tr(6), 1e-30))
                    _rho_bg_att = _fro(0, 6, 3) / (_dbg * _datt)
                    if _C.shape[0] > 21:  # clone block present
                        _dcl = np.sqrt(max(np.trace(_C[21:, 21:]), 1e-30))
                        _rho_bg_cl = _fro(0, 21, _C.shape[0]-21) / (_dbg * _dcl)
                except Exception:
                    pass
                # per-axis sigma_bg (rad/s) and sigma_att (deg) — absolute MAGNITUDE of
                # the bias/attitude cov, to compare vs MSCEqF (whose sigma_bg converges
                # to ~4e-4 rad/s tight while rho matches echo's ~0.3).
                _sig_bg = _dbg / np.sqrt(3.0)
                _sig_att = np.degrees(_datt / np.sqrt(3.0))
                _bias_rows.append((int(k), float(np.linalg.norm(_gb)),
                                   float(_gb[0]), float(_gb[1]), float(_gb[2]), tilt,
                                   _rho_bg_att, _rho_bg_cl, _sig_bg, _sig_att,
                                   float(np.linalg.norm(_ba))))
                # ATTITUDE OVER-CONFIDENCE (self-contained NEES): score echo's OWN
                # attitude error against echo's OWN attitude covariance P_ww. If the
                # filter is over-confident in attitude (P_ww too small), the Kalman
                # gain starves the attitude correction → chronic under-correction →
                # the tilt runaway. e_att = log(R_est^T R_gt) (body-frame right error);
                # P_ww is camera-local — trace is rotation-invariant so the σ-magnitude
                # ratio is frame-robust; the 3-dof NEES carries a small fixed-extrinsic
                # frame caveat (noted at report time).
                cov = vio.get_camera_pose_covariance()
                if cov is not None:
                    P_ww = np.asarray(cov[1], float)
                    e_att = Rot.from_matrix(R_est.T @ R_gt).as_rotvec()  # rad, body
                    try:
                        nees = float(e_att @ np.linalg.solve(P_ww, e_att))
                    except np.linalg.LinAlgError:
                        nees = float("nan")
                    sig = float(np.sqrt(np.trace(P_ww) / 3.0))  # per-axis σ, rad
                    _att_nees_rows.append(
                        (int(k), float(np.linalg.norm(e_att)), sig, nees, tilt))
                # VELOCITY/SCALE OVER-CONFIDENCE (self-contained NEES): score echo's OWN
                # body-velocity error against echo's OWN P_vv (state 12..15, body frame).
                # The coupling test showed the runaway is a velocity↔attitude 2-cycle; the
                # hypothesis is that P_vv becomes over-confident (esp. ALONG v̂ = the scale
                # channel) by the frame-600-800 deceleration, so the filter rejects the
                # vision correction that would hold scale. GT world velocity → est-body via
                # R_est^T; the along-v̂ magnitude channel is attitude-independent (norm
                # preserved). Split by regime at report time to test onset-at-deceleration.
                Pvv = vio.get_velocity_covariance()
                if Pvv is not None:
                    P_vv = np.asarray(Pvv, float)
                    v_est_b = np.asarray(vio.get_velocity(), float)
                    v_gt_w = t_ned_to_nwu @ np.asarray(
                        ds.db[ds.traj]["groundtruth"]["velocity"][k * 4], float)
                    v_gt_b = R_est.T @ v_gt_w
                    e_v = v_est_b - v_gt_b
                    try:
                        nees_v = float(e_v @ np.linalg.solve(P_vv, e_v))
                    except np.linalg.LinAlgError:
                        nees_v = float("nan")
                    sig_v = float(np.sqrt(np.trace(P_vv) / 3.0))
                    nv = float(np.linalg.norm(v_est_b))
                    if nv > 1e-6:
                        vh = v_est_b / nv
                        along_e = float(vh @ e_v)
                        var_al = float(vh @ P_vv @ vh)
                        along_n = along_e * along_e / var_al if var_al > 1e-18 else float("nan")
                    else:
                        along_e = along_n = float("nan")
                    _vel_nees_rows.append((int(k), float(np.linalg.norm(e_v)), sig_v,
                                           nees_v, along_e, along_n, nv,
                                           float(np.linalg.norm(v_gt_w)), tilt))
            except Exception:
                pass

        # GRAVITY-LEAK CAUSAL TEST (diagnostic, env-gated). Pin the nav-state MEAN
        # to GT each frame WITHOUT touching Σ or the clone window (overwrite_nav_mean
        # inverts the EqF group action; mean-only nudge, small per-frame so the
        # un-transported Σ mismatch is negligible). Applied at END of frame so the
        # diagnostic recording above still shows the pre-reset drift; the reset only
        # bounds the NEXT propagation.
        #   ECHO_MSC_GTATT=1 -> pin roll/pitch/yaw attitude to GT (tests: does bounding
        #     tilt bound the |v| runaway? gravity-leak ⇒ yes.)
        #   ECHO_MSC_GTVEL=1 -> pin body velocity to GT (tests: direct scale channel.)
        _gtatt = os.environ.get("ECHO_MSC_GTATT") == "1"
        _gtvel = os.environ.get("ECHO_MSC_GTVEL") == "1"
        if _gtatt or _gtvel:
            R_gt_k = t_ned_to_nwu @ ds.pose(int(k))[:3, :3]
            v_gt_k = t_ned_to_nwu @ np.asarray(
                ds.db[ds.traj]["groundtruth"]["velocity"][int(k) * 4], float)
            vio.overwrite_nav_mean(
                np.ascontiguousarray(R_gt_k), v_gt_k.tolist(), _gtatt, _gtvel)

        # c92 DE-CONFOUND: velocity PSEUDO-MEASUREMENT through the gain machinery
        # (updates mean AND covariance, unlike overwrite_nav_mean). ECHO_MSC_VPSEUDO=1;
        # σ_v via ECHO_VPSEUDO_SIGMA (default 0.05 m/s = hard pull); sign via
        # ECHO_VPSEUDO_SIGN (default -1 = self-consistent lift convention). The
        # binding returns (before-after) residual norm; with the correct sign it is
        # >0 (residual shrinks). First few frames print the 1-frame convergence check.
        if os.environ.get("ECHO_MSC_VPSEUDO") == "1":
            v_gt_w = t_ned_to_nwu @ np.asarray(
                ds.db[ds.traj]["groundtruth"]["velocity"][int(k) * 4], float)
            _sv = float(os.environ.get("ECHO_VPSEUDO_SIGMA", "0.05"))
            _sg = float(os.environ.get("ECHO_VPSEUDO_SIGN", "-1"))
            _shrink = vio.velocity_pseudo_update(v_gt_w.tolist(), _sv, _sg)
            if _vpseudo_dbg[0] < 8:
                _vpseudo_dbg[0] += 1
                print(f"VPSEUDO frame={k} sign={_sg} residual_shrink={_shrink:+.4e} "
                      f"(>0 ⇒ correct sign)")

        for cid in vio.clone_ids():
            cid = int(cid)
            lag = k - cid
            if lag < args.min_lag:
                continue
            Tc = clone_pose.get(cid)
            if Tc is None:
                continue
            rc = vio.get_relative_pose_covariance(cid, np.ascontiguousarray(Tc))
            if rc is None:
                continue
            p_vv = np.asarray(rc[0], float)   # translation 3×3
            p_ww = np.asarray(rc[1], float)   # rotation    3×3

            T_rel_est = np.linalg.inv(Tc) @ T_curr_est
            T_rel_gt = ext_inv @ np.linalg.inv(ds.pose(cid)) @ P_k @ ext
            err = np.linalg.inv(T_rel_est) @ T_rel_gt
            delta = se3_log(err)
            w, v = delta[:3], delta[3:]

            try:
                rot_nees = float(w @ np.linalg.solve(p_ww, w))
                trans_nees = float(v @ np.linalg.solve(p_vv, v))
            except np.linalg.LinAlgError:
                continue
            rot_err_deg = float(np.degrees(np.linalg.norm(
                Rot.from_matrix(err[:3, :3]).as_rotvec())))
            trans_err_m = float(np.linalg.norm(err[:3, 3]))
            rows.append((lag, rot_nees, trans_nees, rot_err_deg, trans_err_m))

            # ROTATION-cov term decomposition (trace = total variance in the
            # rotation channel). Records how each additive term scales with lag.
            tt = vio.get_relative_pose_cov_terms_rot(cid, np.ascontiguousarray(Tc))
            if tt is not None:
                tc = float(np.trace(np.asarray(tt[0], float)))   # term_curr  (sig_kk)
                tl = float(np.trace(np.asarray(tt[1], float)))   # term_clone (a·sig_cc·aᵀ)
                tx = float(np.trace(np.asarray(tt[2], float)))   # term_cross (subtracted)
                tp = float(np.trace(p_ww))                       # = tc + tl - tx
                term_rows.append((lag, tc, tl, tx, tp))

        n_frames += 1
        if not args.no_progress and n_frames % 40 == 0:
            gt_p = t_ned_to_nwu @ P_k[:3, 3]
            nav_err = float(np.linalg.norm(vio_body_pose()[:3, 3] - gt_p))
            msc_s = f" mscAcc={msckf_accepted}" if args.msckf else ""
            di_s = f" slamBorn={delayed_born}" if args.delayed_init else ""
            print(f"  [k={k}] n_clones={vio.n_clones()} scored={len(rows)} "
                  f"navErr={nav_err:.3f}m{msc_s}{di_s}", file=sys.stderr)

    rows = np.asarray(rows, float)
    dt = time.time() - t0
    print(f"\nProcessed {n_frames} frames in {dt:.1f}s, {len(rows)} clone-obs scored.\n")
    if args.msckf:
        _cp, _cr, _ca2, _crt = _cadence
        print(f"  MSC CADENCE: msc_update entered on {_cp} frames | ready(≥1 flush track) on "
              f"{_cr} ({100*_cr/max(_cp,1):.1f}%) | n_acc>0 on {_ca2} ({100*_ca2/max(_cp,1):.1f}%) | "
              f"mean ready-tracks/ready-frame={_crt/max(_cr,1):.2f} | total accepted={msckf_accepted}")
    if len(rows) == 0:
        print("No samples.")
        return

    lag, rot_nees, trans_nees, rot_deg, trans_m = rows.T
    print("=== Clone-relative POSE NEES (dof=3 per channel) ===")
    print_channel("ROT", channel_stats(rot_nees))
    print_channel("TRANS", channel_stats(trans_nees))
    print(f"\nGT relative-error magnitudes: rot median {np.median(rot_deg):.2f}° "
          f"(p90 {np.percentile(rot_deg, 90):.2f}°), "
          f"trans median {np.median(trans_m):.3f} m "
          f"(p90 {np.percentile(trans_m, 90):.3f} m)")

    # Lag breakdown — exposes the sub-linear cov-growth signature on the pose directly.
    print("\n=== NEES by lag (frames since clone) ===")
    print(f"  {'lag bin':>10s}  {'n':>6s}  {'ROT med':>8s}  {'ROT>χ²₉₅':>9s}  "
          f"{'TRANS med':>9s}  {'TRANS>χ²₉₅':>10s}  {'rot° med':>8s}")
    bins = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 10 ** 9)]
    for lo, hi in bins:
        m = (lag >= lo) & (lag < hi)
        if not m.any():
            continue
        rn, tn = rot_nees[m], trans_nees[m]
        label = f"{lo}-{hi if hi < 10 ** 9 else '+'}"
        print(f"  {label:>10s}  {int(m.sum()):>6d}  {np.median(rn):>8.3f}  "
              f"{100 * np.mean(rn > CHI2_95_3):>8.1f}%  {np.median(tn):>9.3f}  "
              f"{100 * np.mean(tn > CHI2_95_3):>9.1f}%  {np.median(rot_deg[m]):>8.2f}")

    # ROTATION-cov term decomposition by lag — the distinguishing test for the
    # lag^0.47 sub-linear growth. If p_ww (=curr+clone-cross) grows sub-linearly,
    # this table shows WHICH term is responsible: does term_curr (sig_kk, the
    # current camera-pose cov) fail to grow, or does term_cross (the persistent
    # correlation with the old clone) stay too large and cancel curr's growth?
    if term_rows:
        tr = np.asarray(term_rows, float)
        tlag, tcurr, tclone, tcross, tpww = tr.T
        print("\n=== ROTATION-cov terms by lag  (trace; p_ww = curr + clone - cross) ===")
        print(f"  {'lag bin':>10s}  {'n':>6s}  {'curr':>11s}  {'clone':>11s}  "
              f"{'cross':>11s}  {'p_ww':>11s}  {'cross/curr':>10s}")
        for lo, hi in bins:
            m = (tlag >= lo) & (tlag < hi)
            if not m.any():
                continue
            mc, ml, mx, mp = (np.median(tcurr[m]), np.median(tclone[m]),
                              np.median(tcross[m]), np.median(tpww[m]))
            label = f"{lo}-{hi if hi < 10 ** 9 else '+'}"
            ratio = mx / mc if mc > 1e-300 else float('nan')
            print(f"  {label:>10s}  {int(m.sum()):>6d}  {mc:>11.3e}  {ml:>11.3e}  "
                  f"{mx:>11.3e}  {mp:>11.3e}  {ratio:>10.3f}")

    # Trajectory from this same run (SE3 no-scale ATE for a sanity readout).
    tk = np.asarray([r[0] for r in traj_rows], int)
    est = np.asarray([r[1] for r in traj_rows], float)
    gtp = np.asarray([r[2] for r in traj_rows], float)
    if len(est) > 10:
        mu_s, mu_d = est.mean(0), gtp.mean(0)
        U, _, Vt = np.linalg.svd((est - mu_s).T @ (gtp - mu_d))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rr = Vt.T @ np.diag([1, 1, d]) @ U.T
        tt = mu_d - Rr @ mu_s
        ate = float(np.sqrt(np.mean(np.sum(((Rr @ est.T).T + tt - gtp) ** 2, 1))))
        glen = float(np.linalg.norm(np.diff(gtp, axis=0), axis=1).sum())
        elen = float(np.linalg.norm(np.diff(est, axis=0), axis=1).sum())
        print(f"\n=== trajectory (same run) ===\n  {len(est)} cam poses  "
              f"ATE(SE3,no-scale)={ate:.2f} m = {100*ate/max(glen,1e-6):.1f}% of {glen:.0f} m path  "
              f"est_len/gt_len={elen/max(glen,1e-6):.2f}")
        # SCALE-RUNAWAY ONSET (ECHO_MSC_TRAJDUMP=1): per-frame cumulative est vs gt
        # path length. Localizes WHEN the 14.5x est_len runaway begins — before the
        # update helps (open-loop) it should track ~1.0; a per-step ratio spiking
        # above 1 at update k pins the runaway to that update's over-correction.
        if os.environ.get("ECHO_MSC_TRAJDUMP") == "1":
            de = np.linalg.norm(np.diff(est, axis=0), axis=1)
            dg = np.linalg.norm(np.diff(gtp, axis=0), axis=1)
            ce, cg = np.cumsum(de), np.cumsum(dg)
            if _speed_rows:
                sp = np.asarray(_speed_rows, float)
                # RAW per-frame dump (frame, est|v|, gt|v|, tilt°) for offline
                # lead-lag: does the scale-ratio |v_est|/|v_gt| drift LEAD the tilt
                # error (⇒ velocity/scale gain is primary) or LAG it (⇒ tilt drives
                # scale)? The GRAVITY-LEAK LAW below is a kinematic identity and
                # cannot resolve the arrow; the cross-correlation of the two raw
                # series at ± lags can. ECHO_MSC_SPEEDDUMP=<path>.
                _sd = os.environ.get("ECHO_MSC_SPEEDDUMP")
                if _sd:
                    np.savetxt(_sd, sp, fmt="%.6f",
                               header="frame est_v gt_v tilt_deg")
                st2 = max(1, len(sp) // 40)
                print("  --- filter body-SPEED vs GT speed (|v|) + TILT err vs GT (gravity) ---")
                print("  frame   est|v|    gt|v|    ratio   tilt°  gsin(tilt)")
                for i in range(0, len(sp), st2):
                    r = sp[i, 1] / sp[i, 2] if sp[i, 2] > 1e-6 else float('nan')
                    tl = sp[i, 3] if sp.shape[1] > 3 else float('nan')
                    gsin = 9.81 * np.sin(np.radians(tl)) if sp.shape[1] > 3 else float('nan')
                    print(f"  {int(sp[i,0]):>5d}  {sp[i,1]:>7.3f}  {sp[i,2]:>7.3f}  {r:>7.3f}  "
                          f"{tl:>6.2f}  {gsin:>7.3f}")
                # GRAVITY-LEAK LAW test: regress the filter's excess speed-growth rate
                # d|v|/dt on g·sin(tilt). slope≈1 (once |v_est|>>|v_gt|, leak aligned
                # with motion) ⇒ the |v| runaway IS attitude-tilt gravity leak.
                if sp.shape[1] > 3 and len(sp) > 20:
                    b = sp[::25]  # ~1s blocks to suppress frame-to-frame |v| noise
                    fr, vv, tl = b[:, 0], b[:, 1], b[:, 3]
                    dt = np.diff(fr) / 25.0
                    dvdt = np.diff(vv) / np.maximum(dt, 1e-6)
                    gsin = 9.81 * np.sin(np.radians(0.5 * (tl[1:] + tl[:-1])))
                    m = (dt > 0) & np.isfinite(dvdt) & np.isfinite(gsin)
                    if m.sum() > 10:
                        A = np.vstack([gsin[m], np.ones(m.sum())]).T
                        (slope, icpt), *_ = np.linalg.lstsq(A, dvdt[m], rcond=None)
                        pred = A @ np.array([slope, icpt])
                        ss = 1 - np.sum((dvdt[m]-pred)**2)/np.sum((dvdt[m]-dvdt[m].mean())**2)
                        rho = np.corrcoef(gsin[m], dvdt[m])[0, 1]
                        print(f"  GRAVITY-LEAK LAW: d|v|/dt = {slope:.3f}·g·sin(tilt) + {icpt:.3f}  "
                              f"(R²={ss:.3f}, ρ={rho:.3f}, n={int(m.sum())})  "
                              f"[slope~1 & ρ>0 ⇒ tilt drives the runaway]")
                        # CUMULATIVE form (noise-averaged, decisive): accumulated excess
                        # speed vs the time-integral of the leak ∫g·sin(tilt)dt.
                        cum_leak = np.cumsum(gsin * dt)
                        exc = vv[1:] - vv[0]
                        A2 = np.vstack([cum_leak, np.ones(len(cum_leak))]).T
                        (s2, i2), *_ = np.linalg.lstsq(A2, exc, rcond=None)
                        p2 = A2 @ np.array([s2, i2])
                        ss2 = 1 - np.sum((exc-p2)**2)/np.sum((exc-exc.mean())**2)
                        print(f"  CUMULATIVE: (|v_est|-|v0|) = {s2:.3f}·∫g·sin(tilt)dt + {i2:.2f}  "
                              f"(R²={ss2:.3f})  [R²→1 ⇒ the entire runaway IS integrated tilt-leak]")
            if _bias_rows:
                br = np.asarray(_bias_rows, float)
                e = br[:, 0] < 100
                # true bias ≈ 0.0005 rad/s; leak-explaining threshold ≈ 0.003 rad/s
                print("  --- estimated GYRO BIAS trajectory (true ≈ 0.0005 rad/s) ---")
                print(f"  |b_g| rad/s:  early(<100f) med={np.median(br[e,1]):.5f}  "
                      f"ALL med={np.median(br[:,1]):.5f}  max={br[:,1].max():.5f}")
                print(f"  final b_g = [{br[-1,2]:+.5f}, {br[-1,3]:+.5f}, {br[-1,4]:+.5f}] "
                      f"@f{int(br[-1,0])}  (tilt {br[-1,5]:.1f}°)")
                if br.shape[1] > 10:
                    # ACCEL BIAS |b_a| m/s^2 (MSCEqF holds med 0.044, max 0.17). A runaway
                    # here = attitude-INDEPENDENT velocity-PROPAGATION scale driver (splits
                    # vel-propagation from vel-update when combined with GTATT).
                    print(f"  |b_a| m/s^2: early(<100f) med={np.median(br[e,10]):.4f}  "
                          f"ALL med={np.median(br[:,10]):.4f}  max={br[:,10].max():.4f}  "
                          f"final={br[-1,10]:.4f}  (MSCEqF med 0.044 max 0.17)")
                if br.shape[1] > 6:
                    print(f"  ρ[gyro_bias,att]:  early med={np.nanmedian(br[e,6]):.3f}  "
                          f"ALL med={np.nanmedian(br[:,6]):.3f}   "
                          f"ρ[gyro_bias,clones]: early med={np.nanmedian(br[e,7]):.3f}  "
                          f"ALL med={np.nanmedian(br[:,7]):.3f}  "
                          f"[large ⇒ update has leverage to dump residual into gyro bias]")
                if br.shape[1] > 9:
                    # ABSOLUTE cov magnitudes (compare vs MSCEqF: sigma_bg conv→4e-4 rad/s
                    # tight, sigma_att ~0.5°). Same rho but tighter magnitudes ⇒ less leverage.
                    print(f"  σ_bg rad/s:  early med={np.median(br[e,8]):.2e}  "
                          f"ALL med={np.median(br[:,8]):.2e}  final={br[-1,8]:.2e}  "
                          f"(MSCEqF conv→4.1e-4)")
                    print(f"  σ_att deg:   early med={np.median(br[e,9]):.3f}  "
                          f"ALL med={np.median(br[:,9]):.3f}  final={br[-1,9]:.3f}  "
                          f"(MSCEqF ~0.36-0.54°)")
                st3 = max(1, len(br) // 12)
                print("  frame   |b_g|      bx        by        bz       tilt°   ρbg_att  ρbg_cl")
                for i in range(0, len(br), st3):
                    _ra = br[i,6] if br.shape[1] > 6 else float('nan')
                    _rc = br[i,7] if br.shape[1] > 7 else float('nan')
                    print(f"  {int(br[i,0]):>5d}  {br[i,1]:>7.5f}  {br[i,2]:>+8.5f}  "
                          f"{br[i,3]:>+8.5f}  {br[i,4]:>+8.5f}  {br[i,5]:>6.2f}  "
                          f"{_ra:>6.3f}  {_rc:>6.3f}")
                # decisive: does |b_g| track tilt growth (feedback) or stay ~0?
                if len(br) > 20:
                    rho_bt = np.corrcoef(br[:, 1], br[:, 5])[0, 1]
                    print(f"  ρ(|b_g|, tilt) = {rho_bt:+.3f}  "
                          f"[>0 ⇒ estimated bias grows with the runaway = feedback channel]")
            # ATTITUDE-AUTHORITY: applied per-frame attitude correction (dtheta_att)
            # vs the tilt error it should be closing. CHRONIC under-correction ⇒
            # dtheta_att stays a bounded fraction while tilt grows (positive corr,
            # bounded ratio); AUTHORITY COLLAPSE ⇒ dtheta_att → 0 as tilt accelerates.
            if _att_corr_rows and _speed_rows:
                ac = np.asarray(_att_corr_rows, float)   # frame, dth°, |dvel|, n_acc
                # RAW per-accepted-update dump for the PROPAGATION-vs-UPDATE tilt
                # decomposition (cols: k, dtheta_att°, |dvel|, n_acc, cos_dir, dred,
                # tilt_pre°, tilt_post°, innov). update jump = tilt_post−tilt_pre;
                # propagation increment (IMU-only) = tilt_pre[k]−tilt_post[prev update].
                # Tests whether the ⊥ correction INJECTS tilt into a near-zero error.
                _ad = os.environ.get("ECHO_MSC_ATTCORRDUMP")
                if _ad:
                    np.savetxt(_ad, ac, fmt="%.6f",
                               header="k dtheta_att dvel n_acc cos_dir dred tilt_pre tilt_post innov")
                sp2 = np.asarray(_speed_rows, float)
                tilt_at = dict(zip(sp2[:, 0].astype(int), sp2[:, 3]))
                fr = ac[:, 0].astype(int)
                tl = np.array([tilt_at.get(int(f), np.nan) for f in fr])
                good = np.isfinite(tl) & (tl > 1e-6)
                print("  --- ATTITUDE-CORRECTION AUTHORITY (applied per-frame) ---")
                print(f"  updates with correction: {len(ac)}  "
                      f"med dtheta_att={np.median(ac[:,1]):.4f}°  "
                      f"med |dvel|={np.median(ac[:,2]):.4f} m/s")
                if good.sum() > 10:
                    # split at the median tilt: is the correction bigger or smaller
                    # when tilt is large (post-onset) vs small (pre-onset)?
                    tmd = np.median(tl[good])
                    lo = ac[good][tl[good] <= tmd, 1]
                    hi = ac[good][tl[good] > tmd, 1]
                    rho = np.corrcoef(tl[good], ac[good][:, 1])[0, 1]
                    print(f"  tilt≤{tmd:.2f}° (early): med dtheta_att={np.median(lo):.4f}°  |  "
                          f"tilt>{tmd:.2f}° (runaway): med dtheta_att={np.median(hi):.4f}°")
                    print(f"  corr(tilt, applied dtheta_att)={rho:+.3f}  "
                          f"[>0 & hi≳lo ⇒ CHRONIC under-correction; ≈0/<0 & hi≪lo ⇒ AUTHORITY COLLAPSE]")
                # MIS-DIRECTION (decisive magnitude-vs-routing discriminator). cos_dir
                # = alignment of the applied attitude correction with the true
                # error-reduction direction; dred = |e_pre|-|e_post| (deg) reduced.
                if ac.shape[1] >= 6:
                    cd = ac[:, 4]; dr = ac[:, 5]
                    fin = np.isfinite(cd)
                    if fin.sum() > 10:
                        cdf = cd[fin]; drf = dr[fin]
                        frac_pos = 100 * np.mean(cdf > 0)
                        frac_red = 100 * np.mean(drf > 0)
                        print("  --- ATTITUDE MIS-DIRECTION (applied correction vs true error dir) ---")
                        print(f"  cos_dir med={np.median(cdf):+.3f} mean={np.mean(cdf):+.3f}  "
                              f"%cos>0={frac_pos:.0f}  |  tilt-reduced/update: %dred>0={frac_red:.0f}  "
                              f"med dred={np.median(drf):+.4f}°")
                        print("  [cos≈+1 & %red high ⇒ well-directed (pure magnitude deficit); "
                              "cos≈0/neg & %red≈50 ⇒ MIS-ROUTED (magnitude knobs can't fix)]")
                        if good.sum() > 10:
                            hi_m = tl[good] > tmd
                            cdg = cd[good]; drg = dr[good]
                            fh = np.isfinite(cdg) & hi_m; fl = np.isfinite(cdg) & ~hi_m
                            if fh.sum() > 5 and fl.sum() > 5:
                                print(f"  by regime: early cos_dir med={np.median(cdg[fl]):+.3f} "
                                      f"%red>0={100*np.mean(drg[fl]>0):.0f}  |  "
                                      f"runaway cos_dir med={np.median(cdg[fh]):+.3f} "
                                      f"%red>0={100*np.mean(drg[fh]>0):.0f}")
                # PROPAGATION-vs-UPDATE decomposition (cols 6=tilt_pre, 7=tilt_post).
                if ac.shape[1] >= 8:
                    tpre = ac[:, 6]; tpost = ac[:, 7]
                    # frames are consecutive msc updates; prop increment uses prev post.
                    prop_inc = tpre[1:] - tpost[:-1]          # IMU-only tilt drift/frame
                    upd_chg = tpost - tpre                    # update's tilt change/frame
                    finp = np.isfinite(prop_inc)
                    print("  --- PROPAGATION vs UPDATE (gravity tilt, per msc-update frame) ---")
                    print(f"  prop increment (IMU-only)  med={np.median(prop_inc[finp]):+.4f}°  "
                          f"mean={np.mean(prop_inc[finp]):+.4f}°  p90={np.percentile(prop_inc[finp],90):+.4f}°")
                    print(f"  update change              med={np.median(upd_chg):+.4f}°  "
                          f"mean={np.mean(upd_chg):+.4f}°  (neg=reduces tilt)  "
                          f"med|dtheta_att applied|={np.median(ac[:,1]):.4f}°")
                    print("  [prop ≪ applied authority & update fails to reduce ⇒ ROUTING; "
                          "prop ≫ authority ⇒ propagation-limited]")
                    # EARLY vs LATER seed test: is the IMU-only prop increment ALREADY
                    # loose in the healthy 0-100f regime (⇒ clean propagation/gyro-bias
                    # seed defect, compare vs MSCEqF's 0.0011°/frame), or fine early and
                    # only loose in the runaway (⇒ feedback consequence, no new lever)?
                    fr_pi = ac[1:, 0]  # frame for each prop_inc entry
                    e_m = finp & (fr_pi < 100); l_m = finp & (fr_pi >= 100)
                    if e_m.any() and l_m.any():
                        print(f"  EARLY(<100f) prop_inc med={np.median(prop_inc[e_m]):+.4f}° "
                              f"tilt_pre med={np.median(tpre[1:][e_m]):.3f}°  |  "
                              f"LATER(≥100f) prop_inc med={np.median(prop_inc[l_m]):+.4f}° "
                              f"tilt_pre med={np.median(tpre[1:][l_m]):.3f}°")
                    # CROSS-COUPLING LEAK test: is the attitude correction DRIVEN by the
                    # velocity/scale innovation (|dvel|) rather than by attitude info? A
                    # high corr(dtheta_att,|dvel|) ⇒ the update leaks the large translation/
                    # scale residual into a spurious attitude rotation ⇒ net-adds tilt as
                    # scale drifts (explains net-harm despite tiny propagation drift).
                    dth_all = ac[:, 1]; dvel_all = ac[:, 2]
                    fv = np.isfinite(dth_all) & np.isfinite(dvel_all)
                    if fv.sum() > 20 and np.std(dvel_all[fv]) > 1e-12:
                        rho_av = np.corrcoef(dth_all[fv], dvel_all[fv])[0, 1]
                        # also vs tilt-increase magnitude (update change), sign-agnostic
                        uc = (ac[:, 7] - ac[:, 6])
                        rho_uv = np.corrcoef(uc[fv], dvel_all[fv])[0, 1] if np.std(uc[fv])>1e-12 else np.nan
                        print(f"  CROSS-COUPLING: corr(|dtheta_att|,|dvel|)={rho_av:+.3f}  "
                              f"corr(tilt-change,|dvel|)={rho_uv:+.3f}  "
                              f"[high ⇒ scale/vel innovation LEAKS into attitude rotation]")
                        # PARTIAL correlation of (dtheta_att, |dvel|) controlling for the
                        # whitened innovation |δ|: separates a genuine vel→att LEAK from the
                        # benign common-driver (big innovation → big correction everywhere).
                        if ac.shape[1] >= 9:
                            innov = ac[:, 8]
                            fp = fv & np.isfinite(innov) & (innov > 1e-12)
                            if fp.sum() > 30 and np.std(innov[fp]) > 1e-9:
                                def _resid(y, x):
                                    b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
                                ra = _resid(dth_all[fp], innov[fp])
                                rv = _resid(dvel_all[fp], innov[fp])
                                pc = (np.corrcoef(ra, rv)[0, 1]
                                      if np.std(ra) > 1e-12 and np.std(rv) > 1e-12 else np.nan)
                                print(f"  PARTIAL corr(dtheta_att,|dvel| | |δ|)={pc:+.3f}  "
                                      f"[~0 ⇒ benign common-driver; still-high ⇒ genuine vel→att LEAK]")
            # CLONE vs NAV routing bisection. Clones are DIRECTLY informed by the
            # measurement; nav-attitude only via Σ[nav,clone]. If clone cos_dir ≫ nav's
            # (-0.11) ⇒ the update itself is sound and the CROSS-BLOCK mis-maps it to
            # nav-attitude (fix = port MSCEqF cross-cov/transport). If clone cos_dir also
            # ~0 ⇒ the clone-space update u (C/S/δ) is the defect (contradicts U0 matches).
            if _clone_dir_rows:
                cdr = np.asarray(_clone_dir_rows, float)  # cos, |corr|°, |e_pre|°
                cc = cdr[:, 0][np.isfinite(cdr[:, 0])]
                if len(cc) > 10:
                    print("  --- CLONE-ATTITUDE CORRECTION DIRECTION (directly-informed) ---")
                    print(f"  n={len(cc)}  clone cos_dir med={np.median(cc):+.3f} mean={np.mean(cc):+.3f}  "
                          f"%cos>0={100*np.mean(cc>0):.0f}  med|corr|={np.median(cdr[:,1]):.4f}° "
                          f"med|e_pre|={np.median(cdr[:,2]):.3f}°")
                    print("  [clone≫nav(-0.11) ⇒ CROSS-BLOCK Σ[nav,clone] mis-map; "
                          "clone~0 ⇒ clone-space update u (C/S/δ) defect]")
            # VELOCITY-CORRECTION DIRECTION (echo-only; decisive gain-vs-routing test for
            # the velocity/scale UNDER-correction seed). cos_vel = alignment of the applied
            # world-frame velocity correction with the true error-to-GT. along_ratio =
            # scale-axis gain (1 = fully corrected, <1 = under). Split by regime (early vs
            # runaway at ECHO_MSC_DECEL_FRAME) so a healthy-regime signal isn't masked.
            if _vel_corr_rows:
                vcr = np.asarray(_vel_corr_rows, float)  # frame,cos,|a|,|e|,al_sign,al_ratio,tilt
                # c96: raw per-update dump so the MSCEqF-side navdump comparison can bin
                # by |e_vel| identically (cos of realized world-frame Δv vs δp=v_gt−v_pre).
                _vd = os.environ.get("ECHO_MSC_VELCORRDUMP")
                if _vd:
                    np.savetxt(_vd, vcr, fmt="%.6f",
                               header="k cos_vel a_vel e_vel al_sign al_ratio tilt")
                cv = vcr[:, 1][np.isfinite(vcr[:, 1])]
                if len(cv) > 10:
                    _df = int(float(os.environ.get("ECHO_MSC_DECEL_FRAME", "700")))
                    ar = vcr[:, 5][np.isfinite(vcr[:, 5])]
                    print("  --- VELOCITY-CORRECTION DIRECTION (world frame; gain-vs-routing) ---")
                    print(f"  n={len(cv)}  cos_vel med={np.median(cv):+.3f} mean={np.mean(cv):+.3f}  "
                          f"%cos>0={100*np.mean(cv>0):.0f}  med|a_vel|={np.median(vcr[:,2]):.4f} "
                          f"med|e_vel|={np.median(vcr[:,3]):.4f} m/s")
                    if len(ar) > 5:
                        print(f"  along-v̂ scale gain (a/e): med={np.median(ar):+.3f} "
                              f"mean={np.mean(ar):+.3f}  %sign-agree={100*np.mean(vcr[:,4]>0):.0f}  "
                              f"[1=corrected, <1=under, ~0/neg=mis-routed]")
                    _fr = vcr[:, 0]
                    _lo = vcr[(_fr < _df) & np.isfinite(vcr[:, 1])]
                    _hi = vcr[(_fr >= _df) & np.isfinite(vcr[:, 1])]
                    if len(_lo) > 5 and len(_hi) > 5:
                        _alo = _lo[:, 5][np.isfinite(_lo[:, 5])]
                        _ahi = _hi[:, 5][np.isfinite(_hi[:, 5])]
                        print(f"  by regime: early cos_vel med={np.median(_lo[:,1]):+.3f} "
                              f"gain med={np.median(_alo) if len(_alo) else float('nan'):+.3f}  |  "
                              f"runaway cos_vel med={np.median(_hi[:,1]):+.3f} "
                              f"gain med={np.median(_ahi) if len(_ahi) else float('nan'):+.3f}")
            # ATTITUDE NEES (self-contained: echo error vs echo cov). chi2_3 refs:
            # 95%=7.81, 99%=11.34. Whole distribution per reporting rules. NEES≫3 and
            # growing ⇒ attitude OVER-CONFIDENT (P_ww too small) ⇒ gain starves the
            # attitude correction ⇒ the chronic under-correction has a covariance root.
            if _att_nees_rows:
                an = np.asarray(_att_nees_rows, float)  # frame,|e|,σ,NEES,tilt
                ne = an[:, 3][np.isfinite(an[:, 3])]
                if len(ne) > 5:
                    def _q(x, p): return float(np.percentile(x, p))
                    ratio = an[:, 1] / np.maximum(an[:, 2], 1e-12)  # |e_att| / σ_att
                    print("  --- ATTITUDE NEES (echo err vs echo P_ww; chi2_3 95%=7.81 99%=11.34) ---")
                    print(f"  n={len(ne)}  NEES med={np.median(ne):.2f}  mean={np.mean(ne):.2f}  "
                          f"p90={_q(ne,90):.2f}  %>7.81={100*np.mean(ne>7.81):.1f}  %>11.34={100*np.mean(ne>11.34):.1f}")
                    print(f"  |e_att|/σ_att (per-axis, frame-robust): med={np.median(ratio):.1f}σ  "
                          f"p90={_q(ratio,90):.1f}σ  max={ratio.max():.1f}σ  "
                          f"[σ_att med={np.degrees(np.median(an[:,2])):.3f}° vs tilt med={np.median(an[:,4]):.2f}°]")
            # VELOCITY/SCALE NEES — the covariance-evolution hypothesis. If echo's P_vv
            # (esp. ALONG v̂ = scale channel) is over-confident and that over-confidence
            # ONSETS at the deceleration, the filter rejects the scale-holding vision
            # correction ⇒ the coupled runaway. chi2_3 95%=7.81; along is 1-dof (95%=3.84).
            if _vel_nees_rows:
                vn = np.asarray(_vel_nees_rows, float)  # frame,|e_v|,σ_v,NEES3,along_e,along_n,|v_est|,|v_gt|,tilt
                def _q(x, p): return float(np.percentile(x, p))
                fr = vn[:, 0]
                n3 = vn[:, 3][np.isfinite(vn[:, 3])]
                al = vn[:, 5][np.isfinite(vn[:, 5])]
                dec = float(os.environ.get("ECHO_MSC_DECEL_FRAME", "700"))
                print("  --- VELOCITY/SCALE NEES (echo err vs echo P_vv; chi2_3 95%=7.81; along 1-dof 95%=3.84) ---")
                if len(n3) > 5:
                    print(f"  NEES3  n={len(n3)} med={np.median(n3):.2f} mean={np.mean(n3):.2f} "
                          f"p90={_q(n3,90):.2f} %>7.81={100*np.mean(n3>7.81):.1f} %>11.34={100*np.mean(n3>11.34):.1f}")
                if len(al) > 5:
                    print(f"  ALONG-v̂ (scale) n={len(al)} med={np.median(al):.2f} mean={np.mean(al):.2f} "
                          f"p90={_q(al,90):.2f} %>3.84={100*np.mean(al>3.84):.1f}  [σ_v med={np.median(vn[:,2]):.4f} m/s]")
                # ONSET: split at deceleration frame — does along-NEES jump after it?
                pre = vn[fr < dec]; post = vn[fr >= dec]
                def _md(a, c):
                    v = a[:, c][np.isfinite(a[:, c])]
                    return float(np.median(v)) if len(v) else float("nan")
                if len(pre) > 5 and len(post) > 5:
                    print(f"  ONSET @f{int(dec)}: along-NEES med  pre={_md(pre,5):.2f}  post={_md(post,5):.2f}  "
                          f"| NEES3 med pre={_md(pre,3):.2f} post={_md(post,3):.2f} "
                          f"| |v_est|/|v_gt| med pre={_md(pre,6)/max(_md(pre,7),1e-9):.2f} post={_md(post,6)/max(_md(post,7),1e-9):.2f}")
                # per-block onset sweep — separate ERROR (|e_v|) from COVARIANCE (σ_v).
                # over-confidence = σ_v too small; runaway = |e_v| growing. If σ_v is ALREADY
                # tiny in the healthy <100f regime (small |e_v|), the tightness is structural,
                # not caused by the deceleration.
                bs = int(os.environ.get("ECHO_MSC_VNEES_BLOCK", "100"))
                print("  block[frame]   |e_v|med   σ_v med   along-NEES   NEES3   |v_est|/|v_gt|   tilt")
                fmax = int(fr.max())
                for b0 in range(0, fmax + 1, bs):
                    m = (fr >= b0) & (fr < b0 + bs)
                    if m.sum() < 5:
                        continue
                    bl = vn[m]
                    print(f"  {b0:>5d}-{b0+bs:<5d}  {_md(bl,1):>8.3f}  {_md(bl,2):>7.4f}  "
                          f"{_md(bl,5):>10.2f}  {_md(bl,3):>7.2f}  "
                          f"{_md(bl,6)/max(_md(bl,7),1e-9):>12.2f}  {_md(bl,8):>6.2f}")
            stride = int(os.environ.get("ECHO_MSC_TRAJDUMP_STRIDE", "0"))
            print("  frame  step_est   step_gt   step_ratio  cum_est/cum_gt  win_ratio")
            if stride > 0:
                # whole-trajectory sampling: window scale ratio over each stride block
                for i in range(stride, len(ce), stride):
                    wr = (ce[i]-ce[i-stride]) / max(cg[i]-cg[i-stride], 1e-9)
                    print(f"  {int(tk[i]):>5d}  {de[i-1]:>8.4f}  {dg[i-1]:>8.4f}  "
                          f"{de[i-1]/max(dg[i-1],1e-9):>10.3f}  {ce[i]/max(cg[i],1e-9):>8.3f}  {wr:>8.3f}")
            else:
                for i in range(min(len(de), int(os.environ.get("ECHO_MSC_TRAJDUMP_N", "40")))):
                    sr = de[i] / dg[i] if dg[i] > 1e-9 else float('nan')
                    print(f"  {int(tk[i+1]):>5d}  {de[i]:>8.4f}  {dg[i]:>8.4f}  {sr:>10.3f}  "
                          f"{ce[i]/max(cg[i],1e-9):>8.3f}")

    if _MSC_DBG and _msc_sk:
        a = np.asarray(_msc_sk)  # cols: chi2,dof,s_geom,s_full,dx_rot,dx_pos,dx_vel
        chi2, dof = a[:, 0], a[:, 1]
        def q(x, p): return float(np.percentile(x, p))
        print(f"\n=== MSC per-track S/K over whole run ({len(a)} accepted-track updates) ===")
        print(f"  chi2/dof   : med={np.median(chi2/dof):.2f}  (ideal ~1; <1 ⇒ S too large/weak, >1 ⇒ S too small)")
        if a.shape[1] > 7:  # cols 7=rms(px reproj RMS), 8=nobs
            print(f"  reproj RMS : med={np.median(a[:,7]):.3f}px  p10={q(a[:,7],10):.3f}  p90={q(a[:,7],90):.3f}px  "
                  f"(triangulation fit on OV-gold tracks; OV converges ⇒ its RMS is small ⇒ if echo-li's is too, gap is GAIN not residual)")
            print(f"  nobs/track : med={np.median(a[:,8]):.1f}  p90={q(a[:,8],90):.1f}  (obs per accepted track)")
        print(f"  s_geom (px²): med={np.median(a[:,2]):.3f}  p10={q(a[:,2],10):.3f}  p90={q(a[:,2],90):.3f}  "
              f"(pose-induced innovation var; small vs σ² ⇒ clones seen as certain)")
        print(f"  s_full (px²): med={np.median(a[:,3]):.3f}   (= s_geom + σ²; σ²={ (args.msckf_sigma_pix**2 if args.msckf_sigma_pix>0 else 4.0):.1f})")
        print(f"  dx_rot     : med={np.degrees(np.median(a[:,4])):.4f}°  p90={np.degrees(q(a[:,4],90)):.4f}°  (nav attitude correction per update)")
        print(f"  dx_pos     : med={100*np.median(a[:,5]):.3f}cm  p90={100*q(a[:,5],90):.3f}cm  (nav position correction per update)")
        print(f"  dx_vel     : med={np.median(a[:,6]):.4f}m/s  p90={q(a[:,6],90):.4f}m/s  (nav VELOCITY/scale correction per update)")
        if _msc_dnav:
            dn = np.asarray(_msc_dnav)  # per-FRAME-batch (|dp| m, |dtheta| rad)
            print(f"  --- per-FRAME-batch nav delta (n={len(dn)} frames; matches OV OV_MSCDUMP |dp|/|dtheta|) ---")
            print(f"  |dp|/frame : med={100*np.median(dn[:,0]):.3f}cm  p90={100*q(dn[:,0],90):.3f}cm")
            print(f"  |dth|/frame: med={np.degrees(np.median(dn[:,1])):.4f}°  p90={np.degrees(q(dn[:,1],90)):.4f}°")
        if _msc_covba:
            cb = np.asarray(_msc_covba)  # (pww_before, pww_after) nav-attitude cov trace rad²
            bef, aft = cb[:, 0], cb[:, 1]
            drop = np.where(bef > 0, (bef - aft) / bef, np.nan)  # fractional tightening per update
            print(f"  --- nav-attitude cov trace P_ww BEFORE/AFTER MSC update (n={len(cb)}; MSCEqF ref ~2.35e-4) ---")
            print(f"  before     : med={np.median(bef):.3e}  (propagated cov entering the update)")
            print(f"  after      : med={np.median(aft):.3e}  (post-downdate cov)")
            print(f"  drop/update: med={100*np.median(drop):.1f}%  (MSCEqF holds ~2.35e-4; before≈after ⇒ update-info weak, "
                  f"before≫after re-inflating ⇒ propagation/apply_transport)")

    if args.save_npz:
        np.savez(args.save_npz, lag=lag, rot_nees=rot_nees, trans_nees=trans_nees,
                 rot_deg=rot_deg, trans_m=trans_m, traj=args.traj,
                 config=str(args.config),
                 traj_k=tk, traj_est=est, traj_gt=gtp)
        print(f"\nSaved raw records to {args.save_npz}")


if __name__ == "__main__":
    main()
