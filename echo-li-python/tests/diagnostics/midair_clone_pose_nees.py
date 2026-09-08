"""Direct clone-relative POSE NEES: real VIO + real frontend, scored against GT pose.

This is the CLEAN honesty test for the pose-clone covariance the EqF reports. It
removes Sparse3D entirely (that filter is downstream: it CONSUMES the clone-relative
pose covariance via §V-D and adds its own depth-filter + triangulation/aperture noise,
so a depth NEES only reports the clone-pose honesty through a convolution). Here we
score the clone-relative pose error directly against the clone-relative pose
covariance the EqF reports.

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
    _vpseudo_dbg = [0]  # c92 velocity-pseudo-measurement 1-frame convergence-check print counter
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

        # Clone THIS frame's pose, then roll the window.
        vio.clone_pose(int(k), stamp)
        clone_pose[int(k)] = T_curr_est.copy()

        # Cap the clone window to EXACTLY args.clone_window by COUNT (mirrors MSCEqF
        # num_clones). MSCEqF marginalizes when clonesSize()==num_clones (checked AFTER
        # cloning): num_clones-1 persist between frames, so persist (clone_window-1)
        # here. Identify the oldest clones to roll off, then marginalize them below
        # (before this frame's scoring, as in the reference cadence).
        live_sorted = sorted(int(c) for c in vio.clone_ids())
        n_over = max(0, len(live_sorted) - (args.clone_window - 1))
        to_marg = set(live_sorted[:n_over])   # clones removed at end of this frame

        # Marginalize the rolled-off clones. Caps the window at exactly
        # args.clone_window, matching MSCEqF num_clones.
        for cid in to_marg:
            vio.marginalize_clone(int(cid))
            clone_pose.pop(cid, None)

        # Score every live clone's relative pose against GT.
        P_k = ds.pose(k)
        traj_rows.append((k, T_curr_est[:3, 3].copy(), (t_ned_to_nwu @ P_k[:3, 3]).copy()))
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
            print(f"  [k={k}] n_clones={vio.n_clones()} scored={len(rows)} "
                  f"navErr={nav_err:.3f}m", file=sys.stderr)

    rows = np.asarray(rows, float)
    dt = time.time() - t0
    print(f"\nProcessed {n_frames} frames in {dt:.1f}s, {len(rows)} clone-obs scored.\n")
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
    if args.save_npz:
        np.savez(args.save_npz, lag=lag, rot_nees=rot_nees, trans_nees=trans_nees,
                 rot_deg=rot_deg, trans_m=trans_m, traj=args.traj,
                 config=str(args.config),
                 traj_k=tk, traj_est=est, traj_gt=gtp)
        print(f"\nSaved raw records to {args.save_npz}")


if __name__ == "__main__":
    main()
