"""End-to-end honest-depth test under REAL VIO poses (the deferred goal).

Drives Sparse3D on Mid-Air with EITHER exact GT poses OR the core-EqF's ESTIMATED poses
(from a saved VIO run, midair_vio_run.py --save-npz), and scores the obstacle-relevant
RANGE (camera->point distance) NEES against GT. Measurements are EXACT (true reprojection
of the birth world point via the GT pose), so the ONLY difference between the two modes is
the pose Sparse3D uses to triangulate -> this isolates pose-induced depth error.

Range (not camera-frame 3D error) is the metric because: (a) it's the obstacle-avoidance
quantity, (b) it's a scalar distance, invariant to the VIO's global frame drift (NWU) and
attitude/yaw error -- only the RELATIVE pose error within a track's window bites, which is
exactly the pose-uncertainty we want in the covariance budget.

  PY=echo-li-python/venv/bin/python
  # first: a gyro-fixed VIO run saving per-frame estimated poses
  $PY midair_vio_run.py --traj 2 --gyro-frame world --frames 1500 --save-npz vio.npz ...
  $PY midair_estpose_depth_nees.py --root ... --traj 2 --pose-npz vio.npz --frames 1500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402

SPARSE_KEYS = ["parametrization", "min_track_length", "conv_variance_threshold",
               "init_depth_var", "sigma_pixel", "uniform_z_max", "a_init", "b_init",
               "ab_min", "ab_max", "min_inlier_ratio", "mahalanobis_reset_chi2",
               "process_depth_var", "min_parallax", "min_cos_sim", "min_depth", "max_depth",
               "bias_walk_var", "pose_range_scale", "pose_range_coherent"]
R = 4  # border margin


def load_est_poses(npz):
    """frame index k -> (estimated T_wc = T_wb_est @ RT_BC, P_vv, P_ww). Pose covs are the EqF's
    absolute camera translation/rotation covariance (camera frame); may be absent (older npz)."""
    d = np.load(npz)
    K, pos, quat = d["k"].astype(int), d["est"], d["quat"]
    pvv = d["pvv"] if "pvv" in d.files else None
    pww = d["pww"] if "pww" in d.files else None
    out = {}
    for i, k in enumerate(K):
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(quat[i]).as_matrix()
        T[:3, 3] = pos[i]
        out[int(k)] = (T @ md.RT_BC,
                       None if pvv is None else pvv[i],
                       None if pww is None else pww[i])
    return out


def psd_clip(M):
    """nearest PSD (clip negative eigenvalues); for the incremental cov P(t)-P(t-1)."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.clip(w, 0.0, None)) @ V.T


def project(Xw, T_cw, f, cx, cy):
    pc = (T_cw @ np.append(Xw, 1.0))[:3]
    if pc[2] <= 1e-6:
        return None, None, None
    return np.array([f * pc[0] / pc[2] + cx, f * pc[1] / pc[2] + cy]), float(np.linalg.norm(pc)), pc


def pix_noise(i, j, std):
    """Deterministic per-(frame,landmark) pixel noise so GT and EST modes see the SAME
    measurement realization (apples-to-apples: only the pose fed to the filter differs)."""
    if std <= 0:
        return np.zeros(2)
    return np.random.default_rng(i * 100003 + int(j)).standard_normal(2) * std


def run(mode, ds, est_pose, f, cx, cy, W, H, settings, args):
    filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
        echo_li.PinholeCamera(f, f, cx, cy), **settings)
    Xw, born = {}, {}
    birth_ctr = {}   # track id -> (est cam-centre, gt cam-centre) at birth, for s_eff decomposition
    nid = 0
    rows = []
    prev_pvv = prev_pww = None
    last = min(args.start + args.frames, ds.n)
    for i in range(args.start, last):
        T_wb = ds.pose(i)
        T_wc_gt = T_wb @ md.RT_BC
        T_cw_gt = np.linalg.inv(T_wc_gt)
        p_vv = p_ww = None
        if mode == "est":
            if i not in est_pose:
                continue
            T_wc_filt, pvv_abs, pww_abs = est_pose[i]
            if args.pose_cov == "absolute" and pvv_abs is not None:
                p_vv, p_ww = pvv_abs, pww_abs
            elif args.pose_cov == "incremental" and pvv_abs is not None and prev_pvv is not None:
                # P(t)-P(t-1): per-frame independent pose noise; telescopes over a track window
                # to P(now)-P(birth) = the relative pose uncertainty that bites triangulation.
                p_vv, p_ww = psd_clip(pvv_abs - prev_pvv), psd_clip(pww_abs - prev_pww)
            prev_pvv, prev_pww = pvv_abs, pww_abs
        else:
            T_wc_filt = T_wc_gt

        # measurements: true reprojection of the birth point via GT pose + matched pixel noise;
        # drop out-of-view. Only the pose (T_wc_filt) differs between modes, not the pixels.
        uvs = {}
        drop = []
        for j in list(born):
            uv, _rng, _pc = project(Xw[j], T_cw_gt, f, cx, cy)
            if uv is None or not (R < uv[0] < W - R and R < uv[1] < H - R):
                drop.append(j)
                continue
            uv = uv + pix_noise(i, j, args.pixel_noise)
            uvs[int(j)] = (float(uv[0]), float(uv[1]))
        for j in drop:
            born.pop(j, None); Xw.pop(j, None); birth_ctr.pop(j, None)

        # redetect on a grid using GT depth (truth births, mode-independent)
        if len(born) < args.redetect:
            depth = ds.depth(i)
            step = max(8, int(args.grid))
            for gy in range(R + 2, H - R - 2, step):
                for gx in range(R + 2, W - R - 2, step):
                    if len(born) >= args.max_tracks:
                        break
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < md.SKY):
                        continue
                    Xw[nid] = md.backproject_world((float(gx), float(gy)), d_rng, T_wb, f, cx, cy)
                    born[nid] = i
                    birth_ctr[nid] = (T_wc_filt[:3, 3].copy(), T_wc_gt[:3, 3].copy())
                    bn = np.array([gx, gy], float) + pix_noise(i, nid, args.pixel_noise)
                    uvs[int(nid)] = (float(bn[0]), float(bn[1]))
                    nid += 1

        if uvs:
            pvv_arg = p_vv.tolist() if p_vv is not None else None
            pww_arg = p_ww.tolist() if p_ww is not None else None
            filt.update(float(i) / 25.0, uvs, T_wc_filt.tolist(), pvv_arg, pww_arg)

        # score converged tracks by RANGE
        for fid, fd in filt.get_features().items():
            j = int(fid)
            if j not in born:
                continue
            tl = int(fd["track_length"])
            if tl < args.min_track:
                continue
            est = np.asarray(fd["position"], float)
            cov = np.asarray(fd["covariance_euclidean"], float)
            re = float(np.linalg.norm(est))
            if re <= 1e-6 or not np.isfinite(cov).all():
                continue
            rhat = est / re
            var_r = float(rhat @ cov @ rhat)
            if var_r <= 0:
                continue
            _uv, rng_true, _pc = project(Xw[j], T_cw_gt, f, cx, cy)
            if rng_true is None:
                continue
            derr = re - rng_true
            # s_eff: est-vs-GT camera baseline scale ratio over THIS track's birth->current span
            # (the baseline triangulation actually consumes). par: GT baseline/depth ~ parallax angle.
            # With exact bearings, a pure baseline-scale error gives relerr == s_eff-1 exactly;
            # any excess isolates parallax amplification / drift. GT mode -> s_eff==1 (control).
            cb = birth_ctr.get(j)
            if cb is None:
                continue
            d_est = float(np.linalg.norm(T_wc_filt[:3, 3] - cb[0]))
            d_gt = float(np.linalg.norm(T_wc_gt[:3, 3] - cb[1]))
            s_eff = d_est / d_gt if d_gt > 1e-6 else np.nan
            par = d_gt / rng_true if rng_true > 1e-6 else np.nan
            rows.append((i - born[j], tl, derr, rng_true, derr * derr / var_r, np.sqrt(var_r),
                         s_eff, par))
    return np.array(rows, float)


def summarize(name, A):
    if len(A) == 0:
        print(f"[{name}] no scored obs"); return
    age, tl, derr, rng, nees, sig, s_eff, par = A.T
    relerr = derr / rng
    print(f"\n=== {name} poses  ({len(A)} scored range obs) ===")
    print(f"range NEES-1D  mean/median: {np.mean(nees):8.2f} / {np.median(nees):8.3f}   "
          f"(ideal mean 1, median 0.455)")
    print(f"  %>3.84 / %>6.63:          {100*np.mean(nees>3.84):5.1f}% / {100*np.mean(nees>6.63):5.1f}%   "
          f"(ideal 5% / 1%)")
    print(f"signed range err:           median {100*np.median(relerr):+6.2f}%  mean {100*np.mean(relerr):+6.2f}%")
    print(f"abs range err:              median {100*np.median(np.abs(relerr)):6.2f}%  "
          f"p90 {100*np.percentile(np.abs(relerr),90):6.2f}%")
    print(f"reported range sigma:       median {np.median(sig):6.3f} m   (filter's 1-sigma depth)")
    # s_eff decomposition: does depth error == baseline-scale error (relerr == s_eff-1)?
    b = s_eff - 1.0                                    # baseline-scale error fraction
    resid = relerr - b                                 # excess depth error (amplification/drift)
    g = np.isfinite(b) & np.isfinite(par) & (par > 0)
    if g.sum() >= 10:
        bb, rr, rs, pp, aa = b[g], relerr[g], resid[g], par[g], age[g]
        print(f"baseline-scale err (s_eff-1): median {100*np.median(bb):+6.2f}%  "
              f"abs-med {100*np.median(np.abs(bb)):6.2f}%")
        print(f"relerr vs (s_eff-1):        signed-ratio median {np.median(rr[np.abs(bb)>1e-4]/bb[np.abs(bb)>1e-4]):+6.2f}  "
              f"(1.0 == pure baseline scale)")
        print(f"residual relerr-(s_eff-1):  median {100*np.median(rs):+6.2f}%  "
              f"abs-med {100*np.median(np.abs(rs)):6.2f}%  (0 == no amplification)")
        # is the excess parallax-driven? corr(|resid|, 1/par); and does |resid| grow with age?
        c_par = float(np.corrcoef(np.abs(rs), 1.0 / pp)[0, 1])
        c_age = float(np.corrcoef(np.abs(rs), aa)[0, 1])
        print(f"corr(|residual|, 1/parallax): {c_par:+.3f}   corr(|residual|, track_age): {c_age:+.3f}")
        for plo, phi, lbl in [(0, 0.02, "par<2%"), (0.02, 0.05, "2-5%"), (0.05, 1e9, "par>5%")]:
            m = (pp >= plo) & (pp < phi)
            if m.sum() >= 10:
                print(f"  {lbl:>7}: n={int(m.sum()):5d}  |s_eff-1| med {100*np.median(np.abs(bb[m])):5.2f}%  "
                      f"|resid| med {100*np.median(np.abs(rs[m])):5.2f}%")
    for lo, hi in [(10, 20), (20, 40), (40, 80), (80, 1e9)]:
        m = (tl >= lo) & (tl < hi)
        if m.sum() >= 10:
            print(f"  tl {lo:>3}-{'inf' if hi>1e8 else int(hi):<3}: n={int(m.sum()):5d}  "
                  f"NEES med {np.median(nees[m]):7.3f}  absErr% {100*np.median(np.abs(relerr[m])):5.2f}")
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--pose-npz", required=True, help="VIO run npz (k, est, quat) for estimated poses")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "configs" / "eqvio_midair.yaml"))
    ap.add_argument("--pixel-noise", type=float, default=0.5,
                    help="std of injected measurement noise (px); filter sigma_pixel is set to match "
                    "so the GT-pose baseline is calibrated and est-pose overconfidence shows on top")
    ap.add_argument("--grid", type=int, default=40, help="birth grid stride (px)")
    ap.add_argument("--redetect", type=int, default=250)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--min-track", type=int, default=10)
    ap.add_argument("--pose-cov", default="all", choices=["none", "absolute", "incremental", "all"],
                    help="how to feed the EqF pose covariance to Sparse3D for the estimated-pose run")
    ap.add_argument("--range-walk-var", type=float, default=None,
                    help="override SparseVog range_walk_var: a RANGE process noise (radial) -- the "
                    "channel that CAN inflate depth cov, unlike the measurement-noise pose term")
    ap.add_argument("--pose-range-coherent", type=float, default=None,
                    help="coherent (bias-driven) companion to --pose-range-scale; adds a term "
                         "growing as track_age^2 so the accumulated range variance is no longer "
                         "purely diffusive (filter_formulation X.1)")
    ap.add_argument("--pose-range-scale", type=float, default=None,
                    help="enable the pose-driven, parallax-scaled range process noise (eq 10-11); "
                    "needs --pose-cov incremental so per-frame relative pose cov drives it")
    ap.add_argument("--save-npz", default="")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start); H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    cfg = yaml.safe_load(open(args.config)).get("SparseVog", {}) or {}
    settings = {k: cfg[k] for k in SPARSE_KEYS if k in cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings.setdefault("min_track_length", 1)
    if args.range_walk_var is not None:
        settings["range_walk_var"] = args.range_walk_var
    if args.pose_range_scale is not None:
        settings["pose_range_scale"] = args.pose_range_scale
    if args.pose_range_coherent is not None:
        settings["pose_range_coherent"] = args.pose_range_coherent
    # keep the config's sigma_pixel (its convergence gates are tuned to it); the GT-pose run is
    # the reference and the GT-vs-EST delta isolates pose error under identical filter settings.
    est_pose = load_est_poses(args.pose_npz)
    print(f"Mid-Air {args.cond}/{ds.traj} {W}x{H} f={f:.1f}  est poses: {len(est_pose)} frames  "
          f"settings sigma_pixel={settings.get('sigma_pixel')} bias_walk_var={settings.get('bias_walk_var')}")

    out = {}
    out["gt"] = summarize("GT baseline (exact poses)",
                          run("gt", ds, est_pose, f, cx, cy, W, H, dict(settings), args))
    modes = ["none", "absolute", "incremental"] if args.pose_cov == "all" else [args.pose_cov]
    for pc in modes:
        args.pose_cov = pc
        out[pc] = summarize(f"VIO-ESTIMATED  pose_cov={pc}",
                            run("est", ds, est_pose, f, cx, cy, W, H, dict(settings), args))
    if args.save_npz:
        np.savez(args.save_npz, cols=np.array(["age", "tl", "derr", "rng", "nees", "sig", "s_eff", "par"]),
                 **{k: v for k, v in out.items() if v is not None})
        print("\nsaved ->", args.save_npz)


if __name__ == "__main__":
    main()
