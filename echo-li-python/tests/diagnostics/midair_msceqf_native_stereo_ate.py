#!/usr/bin/env python3
"""Drive the MSCEqF-native fork-b filter on MidAir with an HONEST-STEREO metric-range
channel (EqVIO's log-inverse-range stereo measurement) and report ATE against
MSCEqF (0.6%) / OpenVINS (0.2%).

Front-end is ISOLATED (same as the mono harness): we feed the Rudolf-V left tracks so
any ATE gap is the filter. The metric range for each in-state landmark comes from
Rudolf-V's REAL stereo matcher (`echo_li.Stereo`): cam0 features are matched into the
actual right image and triangulated with the measured 1.0 m baseline -> per-feature
`(range, range_var)`. NO GROUND-TRUTH DEPTH is used anywhere. Each in-state obs adds one
1-D innovation `ell = -ln(range)` with variance `r_ell = range_var / range²`. Birth is
MONO; scale enters only through this per-frame measured-range channel.

MidAir depth is RADIAL (euclidean ray distance); `echo_li.Stereo.range_priors` returns
the euclidean cam0->point distance = ‖q‖, matching the filter's clone range directly.

`--stereo 0` disables the range channel for a matched mono A/B.

Usage (opt-in; uncommitted):
  source ../.venv/bin/activate   # from echo-li-python/
  python tests/diagnostics/midair_msceqf_native_stereo_ate.py --traj 2 --max-slam 100
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import imageio.v2 as iio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402
from echo_li import MSCEqFNativeFilter  # noqa: E402

ROOT = "/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir"
TRACKS = Path(__file__).resolve().parents[3] / \
    "artifacts/msceqf_midair/stagediff/msceqf_own_tracks.txt"

DT = 0.01          # 100 Hz IMU
IMU_PER_IMG = 4    # 25 Hz images
FX = CX = 512.0    # MidAir 90deg @1024, no distortion
BASELINE = 1.0     # MidAir rectified stereo baseline (m), measured from disparity
                   # (median disp*Z/f = 1.001 over 1604 ground pts; the old 0.25 was
                   #  an unverified constant that collapsed honest stereo triangulation)


def R_wb(q):
    """Rotation body->world from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def bilinear(img, u, v):
    """Bilinear depth sample; NaN if out of bounds or across a depth discontinuity
    (a 2x2 neighbourhood spanning >5% relative depth = an object edge, unreliable)."""
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    if not (0 <= u0 < img.shape[1] - 1 and 0 <= v0 < img.shape[0] - 1):
        return np.nan
    fu, fv = u - u0, v - v0
    a, b, c, d = img[v0, u0], img[v0, u0+1], img[v0+1, u0], img[v0+1, u0+1]
    vals = np.array([a, b, c, d], float)
    if not np.all(np.isfinite(vals)) or vals.max() - vals.min() > 0.05 * vals.min():
        return np.nan
    return a*(1-fu)*(1-fv) + b*fu*(1-fv) + c*(1-fu)*fv + d*fu*fv


def load_tracks_stereo(nimg, tracks_path, t0):
    """frame k -> {track_id: (un, vn, None, None)} normalized Z1 coords. This
    precomputed-tracks path carries NO stereo range (no right-image matcher for
    external tracks); the range channel is only available on the live Rudolf-V path."""
    by_frame = defaultdict(dict)
    with open(tracks_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            ts = float(p[0]); tid = int(p[2]); u = float(p[3]); v = float(p[4])
            k = int(round((ts - t0) / (DT * IMU_PER_IMG)))
            if k < 0 or k >= nimg:
                continue
            un, vn = (u - CX) / FX, (v - CX) / FX
            by_frame[k][tid] = (un, vn, None, None)
    return by_frame


def range_row(rng, rvar, rel_std=0.0):
    """EqVIO log-inverse-range measurement from a MEASURED stereo range `rng` (m) and
    its variance `rvar` (m²) from `echo_li.Stereo`. Returns (ell_obs, r_ell) or None.
    `ell_obs = -ln(range)` (range = ‖q‖, the euclidean cam0->point distance the matcher
    triangulates and the filter's clone also measures). `r_ell = Var(range)/range²` is
    the relative range variance in log space."""
    if rng is None or rvar is None or not (np.isfinite(rng) and 0.0 < rng < 400.0):
        return None
    # FIXED-RELATIVE mode (`--range-rel-std s`): ignore the matcher's `range_var` and use
    # a constant log-range variance `r_ell = s²`. Justified because Rudolf-V documents
    # range_var as a HEURISTIC — `(residual_px/sigma_pixel_scale)²·range²/baseline²` — i.e.
    # NOT a variance in the filter's sense. Measured against MidAir GT depth (n=1685): the
    # range itself is unbiased (median meas/GT 0.9953) but 27% are badly wrong, and their
    # claimed sigma is ~3x TIGHTER (8.18 m) than the good ones' (22.92 m) — the heuristic
    # is ANTI-correlated with accuracy, so confident garbage dominates the channel. A flat
    # variance removes the inversion: bad rows no longer outweigh good ones.
    # Since `ell = -ln(range)`, `sigma_ell = sigma_range/range`, so a RELATIVE range sigma
    # is directly the log-space sigma.
    if rel_std > 0.0:
        return -np.log(rng), rel_std * rel_std
    if not (np.isfinite(rvar) and rvar > 0.0):
        return None
    return -np.log(rng), rvar / (rng * rng)


def build_rudolf_tracks_stereo(root, subset, cond, traj, nimg, cfg_path,
                               stereo, max_features, sigma_pixel_scale):
    """Run Rudolf-V live on the LEFT images; when `stereo`, match each frame's features
    into the REAL right image via echo_li.Stereo and attach the MEASURED (range,
    range_var) — no ground truth. frame k -> {tid: (un, vn, rng|None, rvar|None)}."""
    ds = md.MidAir(root, subset, cond, traj, 1.0)
    im0 = ds.image(0); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    fcfg.lbp_verification = False
    if max_features > 0:
        fcfg.max_features = max_features
    tracker = echo_li.Frontend(fcfg, W, H)
    matcher = None
    if stereo:
        scfg = str(Path(__file__).resolve().parents[3] / "configs/eqvio_midair_stereo.yaml")
        if not Path(scfg).exists():
            scfg = None
        matcher = echo_li.Stereo.from_pinhole(
            f, f, cx, cy, W, H, [-BASELINE, 0.0, 0.0], None, scfg)
    by_frame = defaultdict(dict)
    n_match_tot = 0
    for k in range(nimg):
        gray = np.asarray(ds.image(k))
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        gray = np.ascontiguousarray(gray)
        feats, _stats = tracker.process(gray)
        pri = None
        if matcher is not None:
            rp = ds.dir / "color_right" / ds.traj / f"{k:06d}.JPEG"
            right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if right is not None:
                pri = matcher.range_priors(
                    np.ascontiguousarray(right), tracker, sigma_pixel_scale)
                n_match_tot += len(pri)
        for fd in feats:
            tid = int(fd["id"]); x = float(fd["x"]); y = float(fd["y"])
            un, vn = (x - cx) / f, (y - cy) / f
            rng = rvar = None
            if pri is not None and tid in pri:
                rr, vv = pri[tid]
                if np.isfinite(rr) and 0.0 < rr < 400.0 and np.isfinite(vv) and vv > 0.0:
                    rng, rvar = float(rr), float(vv)
            by_frame[k][tid] = (un, vn, rng, rvar)
        if k % 1000 == 0:
            nm = 0 if pri is None else len(pri)
            print(f"  rudolf-V tracked frame {k}/{nimg}: {len(feats)} feats, "
                  f"{nm} stereo matches")
    if stereo:
        print(f"  total stereo matches: {n_match_tot} "
              f"(~{n_match_tot/max(nimg,1):.0f}/frame)")
    return by_frame


def align_umeyama(est, gt):
    """SE3 (no-scale) alignment of est to gt; returns (aligned, R, t)."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc = est - mu_e, gt - mu_g
    H = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_g - R @ mu_e
    return (R @ est.T).T + t, R, t


def main():
    global IMU_PER_IMG
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="VO_test")
    ap.add_argument("--imu-per-img", type=int, default=4)
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--nframes", type=int, default=0)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-track-len", type=int, default=200)
    # Default OFF: the second-order BCH curvature correction (symmetry.cpp:137) is a
    # byte-faithful port of MSCEqF's, but it assumes the update increment lives in the
    # equivariant algebra. echo's realized `inn` does not (non-equivariant chart), so the
    # same transport leaks scale and is UNIFORMLY worse here — VO_test 2×2 (mono, sl=0):
    #   t1 OV-tracks 0.94→2.81% | t1 native 0.98→2.17% | t2 native 0.42→0.91%.
    # MSCEqF's own code default for the flag is also `false` (it only enables it in its
    # tuned example configs, where the rest of the covariance IS equivariant-consistent).
    ap.add_argument("--curvature", type=int, default=0)
    ap.add_argument("--chi2-mult", type=float, default=1.0)
    ap.add_argument("--pixel-mult", type=float, default=1.0,
                    help="scale the vision measurement std (pixel_std=mult/FX); "
                         ">1 loosens/damps the filter to smooth scale rattle")
    ap.add_argument("--max-slam", type=int, default=0)
    ap.add_argument("--slam-min-len", type=int, default=0)
    ap.add_argument("--slam-chi2-mult", type=float, default=0.0)
    ap.add_argument("--stereo", type=int, default=1,
                    help="1 = MEASURED-stereo log-inverse-range stream channel "
                         "(echo_li.Stereo, real right image); 0 = mono A/B")
    ap.add_argument("--range-rel-std", type=float, default=0.0,
                    help="FIXED relative range sigma (fraction of range) for the metric "
                         "channel, replacing Rudolf-V's range_var heuristic, which is "
                         "documented as not-a-filter-variance and measures 3x tighter on "
                         "the 27%% of ranges that are badly wrong. 0 = use the heuristic.")
    ap.add_argument("--sigma-pixel-scale", type=float, default=20.0,
                    help="echo_li.Stereo residual->range_var scale; larger = more "
                         "trust in the stereo match (range_var∝(residual/scale)²)")
    ap.add_argument("--baseline", type=float, default=BASELINE)
    ap.add_argument("--acc-density", type=float, default=1.6798e-3)
    ap.add_argument("--acc-rw", type=float, default=3.0e-3)
    ap.add_argument("--gyr-density", type=float, default=1.2724e-3)
    ap.add_argument("--gyr-rw", type=float, default=1.0e-4)
    ap.add_argument("--delta-std", type=float, default=1.0e-2,
                    help="init std of the IMU bias block; tiny value ~ freeze bias")
    ap.add_argument("--dump", default="")
    ap.add_argument("--tracks", default=str(TRACKS))
    ap.add_argument("--tracks-t0", type=float, default=0.0)
    ap.add_argument("--frontend", choices=["msceqf", "rudolf"], default="rudolf",
                    help="rudolf = live Rudolf-V tracker (works on ANY traj); "
                         "msceqf = replay the t2-only own-tracks dump")
    ap.add_argument("--external-tracks", default="",
                    help="path to an OpenVINS OV_TRACKDUMP file (lines "
                         "'ts cam id u v', raw pixels). When set, the filter consumes "
                         "these KLT tracks instead of Rudolf-V/MSCEqF (mono; forces "
                         "--stereo 0). Isolates front-end from filter.")
    ap.add_argument("--clone-on-obs", action="store_true",
                    help="clone ONLY on frames that carry observations (matches OV's "
                         "keyframe/clone cadence); IMU still propagates every frame. "
                         "Tests whether echo's every-frame cloning vs OV's sparse tracks "
                         "is the scale-runaway confound.")
    ap.add_argument("--external-t0", type=float, default=1.0,
                    help="OV bag start time (s): frame k = round((ts - t0)*25)")
    ap.add_argument("--frontend-config",
                    default=str(Path(__file__).resolve().parents[3]
                                / "configs/eqvio_midair.yaml"))
    ap.add_argument("--max-features", type=int, default=200,
                    help="Rudolf-V max_features (OV parity = 200)")
    ap.add_argument("--lm-history", default="",
                    help="DIAGNOSTIC (off by default): dump per-(frame,landmark) life "
                         "history — world xyz, anchor range, cov trace, age — to this "
                         ".npz. Used to test whether the in-state landmark ESTIMATE "
                         "wanders with track-age/baseline (the 'landmark-update wall').")
    ap.add_argument("--max-slam-age", type=int, default=0,
                    help="DIAGNOSTIC (0=off): marginalize a SLAM landmark once its age "
                         "(frames since birth) exceeds this. Sweeping it tests whether "
                         "capping landmark lifetime IMPROVES t1 ATE (=long spans hurt).")
    args = ap.parse_args()
    IMU_PER_IMG = args.imu_per_img

    base = f"{ROOT}/{args.set}/{args.cond}"
    tr = f"trajectory_{args.traj:04d}"
    g = h5py.File(f"{base}/sensor_records.hdf5", "r")[tr]
    acc = np.asarray(g["imu"]["accelerometer"], float)
    gyr = np.asarray(g["imu"]["gyroscope"], float)
    att = np.asarray(g["groundtruth"]["attitude"], float)
    pos = np.asarray(g["groundtruth"]["position"], float)
    vel = np.asarray(g["groundtruth"]["velocity"], float)
    m = min(len(gyr), len(att))
    Rwb = [R_wb(att[i]) for i in range(m)]
    gyr_body = np.array([Rwb[i].T @ gyr[i] for i in range(m)])
    acc = acc[:m]
    nimg_all = m // IMU_PER_IMG
    nimg = nimg_all if args.nframes <= 0 else min(args.nframes, nimg_all)

    if args.external_tracks:
        args.stereo = 0  # OV KLT dump is mono
        print(f"front-end: EXTERNAL OV KLT tracks {args.external_tracks} "
              f"(t0={args.external_t0}, mono)")
        by_frame = {}
        n_obs = 0
        with open(args.external_tracks) as fh:
            for ln in fh:
                if ln.startswith("#") or not ln.strip():
                    continue
                p = ln.split()
                ts = float(p[0]); tid = int(p[2]); u = float(p[3]); v = float(p[4])
                k = int(round((ts - args.external_t0) * 25.0))
                if k < 0 or k >= nimg:
                    continue
                un, vn = (u - CX) / FX, (v - CX) / FX
                by_frame.setdefault(k, {})[tid] = (un, vn, None, None)
                n_obs += 1
        ks = sorted(by_frame)
        print(f"  {n_obs} obs over {len(ks)} frames (k={ks[0]}..{ks[-1]}), "
              f"median feats/frame={int(np.median([len(by_frame[k]) for k in ks]))}")
    elif args.frontend == "rudolf":
        print(f"front-end: Rudolf-V live @ native, max_features={args.max_features}")
        by_frame = build_rudolf_tracks_stereo(
            ROOT, args.set, args.cond, args.traj, nimg, args.frontend_config,
            bool(args.stereo), args.max_features, args.sigma_pixel_scale)
    else:
        print("front-end: MSCEqF own KLT dump (replayed; t2-only; mono only)")
        by_frame = load_tracks_stereo(nimg, args.tracks, args.tracks_t0)

    # --- GT-seeded given-origin init (frame 0) ---
    R0 = Rwb[0]; p0 = pos[0].copy(); v0 = vel[0].copy(); b0 = np.zeros(6)
    S0 = np.array([[0., 0., 1., 0.],
                   [1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.]])          # T_imu_cam (cam->body)
    grav = np.array([0., 0., 9.81])
    # Index 2 is rotation about body-z ~ YAW on a level start. Global yaw is
    # UNOBSERVABLE, so whatever the update writes into it is never contradicted by any
    # measurement; the reference pins it (defines the world frame by the initial yaw)
    # while we historically left 1e-2 rad of freedom there.
    d_std = np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
    delta_std = np.full(6, args.delta_std)
    e_std = np.full(6, 1e-4)

    filt = MSCEqFNativeFilter(
        R0, p0, v0, b0, S0, grav, d_std, delta_std, e_std,
        args.acc_density, args.gyr_density, args.acc_rw, args.gyr_rw,
        0, args.num_clones)
    if args.stereo:
        filt.set_stereo_baseline(args.baseline)

    pixel_std = args.pixel_mult / FX
    slam_min_len = args.slam_min_len if args.slam_min_len > 0 else args.num_clones
    slam_chi2 = args.slam_chi2_mult if args.slam_chi2_mult > 0 else args.chi2_mult
    buf = {}                     # tid -> [(frame_k, un, vn, rng)]
    slam = set()
    birth_frame = {}             # tid -> frame k at which it was promoted to in-state
    lm_hist = []                 # per (frame,landmark): [k, tid, age, wx,wy,wz, range, covtr]
    est_p, gt_p = [], []
    est_R, gt_R, est_v, gt_v = [], [], [], []   # attitude+velocity dump (yaw diagnosis)
    scale_ba = []   # per-update [k, |v_est_before|, |v_est_after|, |v_gt|, n_accepted]
    cov_ba = []     # per-update [k, scale_var(v̂ᵀPvvv̂)_before, _after, trace_Pvv_before,
                    #            _after, ||P[vel,E]||_before, _after]
    prop_ba = []    # per-update, PRE-update (post-prop): [k, vvec(3), R_est(9,row-major),
                    #            bias(6)] — for propagation-faithfulness vs physics
    bias_upd = []   # per-update [k, bias_before(6), bias_after(6)] — does the update inject bias?
    biascov_log = []  # per-update [k, dbg_z, cpl_mean, cpl_sum, cpl_absmax, cpl_abssum,
                      #             bg_z<->Dd-att-z, var(bg_z), ncl] — clone<->bias routing

    def snap(gt_i):
        p, Rm, vv = filt.nav_body_pose()
        est_p.append(p.copy()); est_R.append(Rm.copy()); est_v.append(vv.copy())
        gt_p.append(pos[gt_i].copy()); gt_R.append(Rwb[gt_i].copy()); gt_v.append(vel[gt_i].copy())

    n_upd = n_slam_upd = n_promoted = 0
    n_stereo_obs = 0             # births that got a metric range PRIOR
    n_stream_range = 0           # per-frame stream obs carrying a metric range ROW

    def add_frame(k):
        if not by_frame.get(k):
            return  # skip cloning observation-less frames; IMU already propagated by
                    # caller. A clone with no features anchors nothing, and treating an
                    # empty frame as "all tracks lost" would falsely SEVER tracks that
                    # the source keeps continuous across a gap (e.g. OV's KLT dump only
                    # emits on ~74% of frames). Always defer to the next observed frame.
        filt.clone_pose(k, k * DT * IMU_PER_IMG)
        for tid, rec in by_frame.get(k, {}).items():
            if tid in slam:
                continue
            buf.setdefault(tid, []).append((k, *rec))

    add_frame(0)
    snap(0)

    for k in range(1, nimg):
        for i in range((k - 1) * IMU_PER_IMG, k * IMU_PER_IMG):
            filt.process_imu(gyr_body[i], acc[i], DT)
        add_frame(k)

        if not by_frame.get(k):
            # No observations this frame: record pose, defer all update/marg logic to
            # the next observed frame. Skipping (rather than running the loss/flush
            # block with an empty `cur`) is what keeps a track continuous across a
            # source gap instead of severing it every empty frame.
            snap(k * IMU_PER_IMG)
            continue

        cur = set(by_frame.get(k, {}).keys())
        marg = filt.clone_ids()[0] if filt.n_clones() > args.num_clones else None

        # (0) PROMOTE long window-spanning tracks about to be marginalized.
        if args.max_slam > 0 and marg is not None and len(slam) < args.max_slam:
            cand = [tid for tid, obs in buf.items()
                    if tid in cur and len(obs) >= slam_min_len
                    and any(fk == marg for (fk, *_ ) in obs)]
            cand.sort(key=lambda t: len(buf[t]), reverse=True)
            for tid in cand[: args.max_slam - len(slam)]:
                obs = [(fk, un, vn) for (fk, un, vn, rng, rvar) in buf[tid]]
                # TIGHT-METRIC-DEPTH birth: pin the depth from the measured stereo range
                # at the anchor (oldest) obs, tight variance r_ell = Var(range)/range².
                # The bearing track then carries the metric to the trajectory scale.
                rp = None
                if args.stereo:
                    for (fk, un, vn, rng, rvar) in buf[tid]:
                        if rng is not None and rvar is not None and rng > 0.0 and rvar > 0.0:
                            _rv = (args.range_rel_std ** 2 if args.range_rel_std > 0
                                   else float(rvar) / (float(rng) * float(rng)))
                            rp = (float(rng), _rv)
                            n_stereo_obs += 1
                            break
                ok = filt.birth_landmark(tid, obs, pixel_std, slam_chi2, rp)
                if ok:
                    filt.reanchor_landmark(tid, k)
                    slam.add(tid)
                    birth_frame[tid] = k
                    del buf[tid]
                    n_promoted += 1

        # (1) Structureless MSCKF update (mono/bearing — no stereo rows here).
        ready, done = {}, []
        for tid, obs in buf.items():
            lost = tid not in cur
            toolong = len(obs) >= args.max_track_len
            touches_marg = marg is not None and any(fk == marg for (fk, *_ ) in obs)
            if lost or toolong or touches_marg:
                if len(obs) >= 2:
                    ready[tid] = [(fk, un, vn) for (fk, un, vn, rng, rvar) in obs]
                done.append(tid)
        if ready:
            def _scale_cov():
                # scale-direction variance v̂ᵀPvvv̂, trace(Pvv), ||P[vel,E]||_F.
                vv = filt.nav_body_pose()[2]
                nv = float(np.linalg.norm(vv))
                vhat = vv / nv if nv > 1e-9 else vv
                P = filt.nav_cov()            # 21x21 [att0:3 vel3:6 pos6:9 delta9:15 | E15:21]
                Pvv = P[3:6, 3:6]
                return (float(vhat @ Pvv @ vhat), float(np.trace(Pvv)),
                        float(np.linalg.norm(P[3:6, 15:21])))
            p_b, R_b, vvec_b = filt.nav_body_pose()   # post-prop, pre-update
            bias_b = filt.bias()                      # [gyro3; accel3]
            prop_ba.append((float(k), *[float(x) for x in vvec_b],
                            *[float(x) for x in R_b.reshape(-1)],
                            *[float(x) for x in bias_b]))
            v_before = float(np.linalg.norm(filt.nav_body_pose()[2]))
            sv_b, tr_b, cpl_b = _scale_cov()
            nacc = filt.msc_update(
                ready, pixel_std, args.chi2_mult, bool(args.curvature))
            n_upd += nacc
            v_after = float(np.linalg.norm(filt.nav_body_pose()[2]))
            bias_a = filt.bias()   # post-update bias -> did the update inject bias?
            bias_upd.append((float(k), *[float(x) for x in bias_b],
                             *[float(x) for x in bias_a]))
        for tid in done:
            del buf[tid]

        # (2) Streaming SLAM update: fresh BEARING per landmark. Depth is pinned metric
        # at birth (tight r_ell), so the bearing track against the fixed depth is what
        # transfers absolute scale to the trajectory — no per-frame range row (a loose
        # mono depth would absorb it before it reached the baseline; see derivation).
        if slam:
            supd = {}
            srange = {}
            for tid in list(slam):
                if tid in cur:
                    un, vn, rng, rvar = by_frame[k][tid]
                    supd[tid] = [(k, un, vn)]
                    # CONTINUOUS metric channel: one log-inverse-range row per frame for
                    # the landmark's whole life. This is the part that actually holds
                    # scale; the birth prior alone is relaxed away by the bearing stream.
                    if args.stereo:
                        rr = range_row(rng, rvar, args.range_rel_std)
                        if rr is not None:
                            srange[tid] = [(k, rr[0], rr[1])]
                            n_stream_range += 1
                else:
                    filt.marginalize_landmark(tid)
                    slam.discard(tid)
            if supd:
                n_slam_upd += filt.landmark_stream_update(
                    supd, pixel_std, slam_chi2, srange or None)

            # (2b) DIAGNOSTIC: landmark life-history + age cap. Off unless requested.
            if args.lm_history or args.max_slam_age > 0:
                for tid in list(slam):
                    age = k - birth_frame.get(tid, k)
                    if args.max_slam_age > 0 and age > args.max_slam_age:
                        filt.marginalize_landmark(tid)
                        slam.discard(tid)
                        continue
                    if args.lm_history:
                        w = filt.landmark_world(tid)
                        if w is None:
                            continue
                        rng = filt.landmark_range_anchor(tid)
                        cov = filt.landmark_cov(tid)
                        covtr = float(np.trace(cov)) if cov is not None else float("nan")
                        lm_hist.append((float(k), float(tid), float(age),
                                        float(w[0]), float(w[1]), float(w[2]),
                                        float(rng) if rng is not None else float("nan"),
                                        covtr))

        # (3) Reanchor survivors off the marginalizing clone.
        if slam and marg is not None:
            newest = filt.clone_ids()[-1]
            for tid in list(slam):
                if filt.landmark_anchor(tid) == marg:
                    filt.reanchor_landmark(tid, newest)

        # (4) Slide the window.
        while filt.n_clones() > args.num_clones:
            old = filt.marginalize_oldest()
            for tid in list(buf):
                b = [rec for rec in buf[tid] if rec[0] != old]
                if b:
                    buf[tid] = b
                else:
                    del buf[tid]

        snap(k * IMU_PER_IMG)

    est_p = np.array(est_p); gt_p = np.array(gt_p)
    aligned, Ralign, talign = align_umeyama(est_p, gt_p)
    err = np.linalg.norm(aligned - gt_p, axis=1)
    seg = np.linalg.norm(np.diff(gt_p, axis=0), axis=1)
    path = seg.sum()
    ate_rmse = np.sqrt((err ** 2).mean())
    est_len = np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum()

    print(f"traj={args.traj}  stereo={bool(args.stereo)}  baseline={args.baseline}  "
          f"max_slam={args.max_slam}")
    print(f"frames={nimg}  accepted_updates={n_upd}  "
          f"slam_promoted={n_promoted}  slam_updates={n_slam_upd}  "
          f"stereo_births={n_stereo_obs}  stream_range_rows={n_stream_range}")
    print(f"GT path length   = {path:8.2f} m")
    print(f"est path length  = {est_len:8.2f} m   (est/gt = {est_len/path:.4f})")
    print(f"ATE RMSE (SE3)   = {ate_rmse:8.3f} m")
    print(f"ATE %            = {100*ate_rmse/path:7.3f} %   [MSCEqF 0.6% / OV 0.2%]")
    print(f"ATE median/mean/max = "
          f"{np.median(err):.3f} / {err.mean():.3f} / {err.max():.3f} m")
    print(f"final drift      = {np.linalg.norm(est_p[-1]-gt_p[-1]):8.3f} m")

    if args.dump:
        np.savez(args.dump, est=est_p, gt=gt_p, R=Ralign, t=talign,
                 k=np.arange(nimg, dtype=int),
                 est_R=np.array(est_R), gt_R=np.array(gt_R),
                 est_v=np.array(est_v), gt_v=np.array(gt_v),
                 scale_ba=np.array(scale_ba), cov_ba=np.array(cov_ba),
                 prop_ba=np.array(prop_ba), bias_upd=np.array(bias_upd))
        print(f"dumped trajectory -> {args.dump}")



    if args.lm_history and lm_hist:
        lm = np.array(lm_hist)  # [k, tid, age, wx,wy,wz, range, covtr]
        # world-point wander per landmark: std of its estimated world xyz over its life.
        # A static scene point should be ~constant; wander that grows with age = leak.
        wander, ages, spans = [], [], []
        for tid in np.unique(lm[:, 1]):
            r = lm[lm[:, 1] == tid]
            if len(r) < 3:
                continue
            wander.append(float(np.linalg.norm(r[:, 3:6].std(axis=0))))
            ages.append(float(r[:, 2].max()))
            spans.append(len(r))
        wander, ages = np.array(wander), np.array(ages)
        # Per-accepted-landmark-update nav-correction vs anchor baseline: the DIRECT
        # test for the current-estimate relinearization long-baseline leak. Columns:
        # [baseline_m, range_m, yaw_inc, vel_along_vhat_inc, att_inc_norm,
        #  vel_inc_norm, pos_inc_norm, chi2_d, rows].
        upd = np.asarray(filt.lm_upd_diag())
        np.savez(args.lm_history, lm=lm, upd=upd,
                 est_R=np.array(est_R), gt_R=np.array(gt_R), est_p=est_p, gt_p=gt_p)
        print(f"landmark-history -> {args.lm_history}  ({len(np.unique(lm[:,1]))} landmarks)")
        if upd.shape[0] > 10:
            bl = upd[:, 0]; yaw = np.abs(upd[:, 2]); valong = np.abs(upd[:, 3])
            print(f"  lm-update nav-correction vs anchor baseline "
                  f"({upd.shape[0]} accepted updates):")
            print(f"    baseline(m): median {np.median(bl):.2f} p90 {np.percentile(bl,90):.2f} "
                  f"p99 {np.percentile(bl,99):.2f} max {bl.max():.2f}")
            print(f"    |yaw_inc|(rad):   median {np.median(yaw):.2e} mean {yaw.mean():.2e} max {yaw.max():.2e}")
            print(f"    |v_along_inc|:    median {np.median(valong):.2e} mean {valong.mean():.2e} max {valong.max():.2e}")
            print(f"    corr(baseline,|yaw_inc|)     = {np.corrcoef(bl,yaw)[0,1]:+.3f}")
            print(f"    corr(baseline,|v_along_inc|) = {np.corrcoef(bl,valong)[0,1]:+.3f}")
            # binned: does the correction GROW at long baseline?
            edges = np.percentile(bl, [0, 50, 80, 95, 100])
            edges = np.unique(edges)
            print("    baseline-bin(m)   n     |yaw_inc| med   |v_along_inc| med")
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (bl >= lo) & (bl <= hi if hi == edges[-1] else bl < hi)
                if m.sum() == 0:
                    continue
                print(f"      {lo:6.2f}-{hi:<6.2f} {m.sum():5d}   {np.median(yaw[m]):.3e}      {np.median(valong[m]):.3e}")
        if len(wander):
            print(f"  world-wander (m): median {np.median(wander):.3f} "
                  f"mean {wander.mean():.3f} max {wander.max():.3f}")
            # does wander grow with landmark age? (the 'long-span leak' test)
            if len(wander) > 5:
                cc = float(np.corrcoef(ages, wander)[0, 1])
                long = wander[ages >= np.median(ages)]
                short = wander[ages < np.median(ages)]
                print(f"  corr(age, wander) = {cc:+.3f}   "
                      f"wander long-lived {np.median(long):.3f} vs short {np.median(short):.3f} m")


if __name__ == "__main__":
    main()
