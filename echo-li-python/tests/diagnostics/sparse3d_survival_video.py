"""Video: which Sparse3D landmarks survive, and which the Mahalanobis gate destroys.

`mahalanobis_reset_chi2` is NOT a down-weight -- a single observation with
maha_sq = r^T S^-1 r above the threshold puts the id in `reset_features`, and
mod.rs:507 then REMOVES the landmark outright. Its accumulated depth estimate is
discarded and it must re-triangulate from scratch. At sigma_pixel = 0.42 a threshold
of 0.2 corresponds to a residual of about 0.42*sqrt(0.2) ~ 0.19 px.

Per frame each front-end feature is drawn as:
    GREEN   converged -- query_range returns a usable prior (this is what can seed the EqF)
    YELLOW  live in Sparse3D, not yet converged
    RED     was live last frame and is gone now -> destroyed (gate reset or track loss)
    GREY    tracked by the front-end but not in Sparse3D at all

The point of the video is the ratio and the churn: how much of the frame is green
(usable) versus how much flashes red every frame.

    PY=echo-li-python/venv/bin/python
    $PY sparse3d_survival_video.py --root <MidAir> --traj 2 --frames 600 \
        --config configs/diagnostics_midair_sparse3d.yaml --out survival.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402

SPARSE_KEYS = ["parametrization", "min_track_length", "conv_variance_threshold",
               "conv_inlier_ratio", "sigma_pixel", "init_depth_var", "process_depth_var",
               "min_parallax", "max_depth", "min_depth", "a_init", "b_init", "ab_max",
               "ab_min", "range_walk_var", "birth_min_flow_px", "max_pool_size",
               "mahalanobis_reset_chi2", "bias_walk_var", "pose_range_scale"]


def load_sparse_settings(path):
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    sv = cfg.get("SparseVog", {}) or {}
    return {k: sv[k] for k in SPARSE_KEYS if k in sv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="sparse3d_survival.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-video", action="store_true", help="stats only, skip encoding")
    ap.add_argument("--pose-npz", default="",
                    help="VIO run npz (k, est, quat): drive Sparse3D with ESTIMATED poses. "
                         "Survival under estimated poses is NOT the same as under GT poses -- "
                         "pose error adds to the residual the gate tests.")
    ap.add_argument("--gate", type=float, default=None,
                    help="override mahalanobis_reset_chi2 (chi2 on r^T S^-1 r; at "
                         "sigma_pixel=0.42 the residual limit is 0.42*sqrt(gate) px)")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(0)
    H, W = im0.shape
    f = 512 * args.scale / 2.0          # 90 deg FoV
    cx, cy = W / 2.0, H / 2.0

    settings = load_sparse_settings(args.config)
    if args.gate is not None:
        settings["mahalanobis_reset_chi2"] = args.gate
    print(f"SparseVog settings: {settings}")
    filt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(
        echo_li.PinholeCamera(f, f, cx, cy), **settings)
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    tracker = echo_li.Frontend(fcfg, W, H)

    T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    est_pose = {}
    if args.pose_npz:
        from scipy.spatial.transform import Rotation as _Rot
        _d = np.load(args.pose_npz)
        for _j, _k in enumerate(_d["k"].astype(int)):
            _M = np.eye(4)
            _M[:3, :3] = _Rot.from_quat(_d["quat"][_j]).as_matrix()
            _M[:3, 3] = _d["est"][_j]
            est_pose[int(_k)] = T @ _M      # VIO/NWU frame -> ds.pose()'s raw frame
        _g = [np.linalg.norm(est_pose[k][:3, 3] - ds.pose(k)[:3, 3])
              for k in list(est_pose)[::37] if k < ds.n]
        print(f"estimated poses: {len(est_pose)}   SANITY est-vs-GT gap median "
              f"{np.median(_g):.1f} m (drift-sized = frames agree)")
    writer = vpath = None
    if not args.no_video:
        writer, vpath = md.open_writer(args.out, W, H, args.fps)

    prev_live = set()
    prev_uv_ids = set()
    n_born = n_destroyed = n_gate = n_lost = n_fe_lost = 0
    hist = []
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i)
        feats, _ = tracker.process(img)
        uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
        T_wb = est_pose.get(i, T @ ds.pose(i)) if est_pose else (T @ ds.pose(i))
        T_wc = T_wb @ md.RT_BC
        filt.update(float(i) / 25.0, uvs, T_wc.tolist(), None, None)

        live = set(int(k) for k in filt.get_features().keys())
        converged = {fid for fid in uvs if filt.query_range(fid)[0] > 0.0}
        destroyed = prev_live - live                 # was live, now gone
        # mod.rs:507 removes a feature if the GATE reset it OR the front-end stopped
        # reporting it. Separate them: still tracked => the gate killed it; no longer
        # tracked => ordinary front-end track loss, nothing to do with the gate.
        gate_killed = {f for f in destroyed if f in uvs}
        lost_by_frontend = destroyed - gate_killed
        n_gate += len(gate_killed)
        n_lost += len(lost_by_frontend)
        fe_lost = prev_uv_ids - set(uvs)             # front-end turnover itself
        n_fe_lost += len(fe_lost)
        n_destroyed += len(destroyed)
        n_born += len(live - prev_live)

        if writer is None:
            hist.append((len(uvs), len(live), len(converged), len(destroyed)))
            prev_live = live
            prev_uv_ids = set(uvs)
            continue
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for fid, (u, v) in uvs.items():
            p = (int(u), int(v))
            if fid in converged:
                cv2.circle(vis, p, 4, (0, 220, 0), -1)          # green: usable prior
            elif fid in live:
                cv2.circle(vis, p, 3, (0, 210, 235), 1)         # yellow: live, unconverged
            else:
                cv2.circle(vis, p, 2, (140, 140, 140), 1)       # grey: not in Sparse3D
        for fid in destroyed:                                    # red X where it died
            if fid in uvs:
                u, v = uvs[fid]
                cv2.drawMarker(vis, (int(u), int(v)), (40, 40, 255),
                               cv2.MARKER_TILTED_CROSS, 11, 2)

        hist.append((len(uvs), len(live), len(converged), len(destroyed)))
        cv2.rectangle(vis, (0, 0), (W, 46), (0, 0, 0), -1)
        cv2.putText(vis, f"f{i:4d}  tracked {len(uvs):3d}  live {len(live):3d}  "
                    f"converged {len(converged):3d}  destroyed/frame {len(destroyed):3d}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, "green=usable prior  yellow=live  red X=destroyed  grey=not tracked",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        writer.write(vis)
        prev_live = live

    if writer is not None:
        writer.release()
    a = np.array(hist, float)
    print(f"\ngate={settings.get('mahalanobis_reset_chi2')} "
          f"(~{0.42*float(settings.get('mahalanobis_reset_chi2',0))**0.5:.2f} px)  frames={len(hist)}")
    print(f"{'':<22}{'median':>9}{'mean':>9}")
    for j, lab in enumerate(["front-end tracked", "live in Sparse3D",
                             "converged (usable)", "destroyed per frame"]):
        print(f"{lab:<22}{np.median(a[:, j]):>9.1f}{a[:, j].mean():>9.1f}")
    print(f"\nborn {n_born}, destroyed {n_destroyed} over {len(hist)} frames "
          f"-> {n_destroyed/max(len(hist),1):.1f} destroyed/frame")
    print(f"converged fraction of tracked: {100*a[:,2].sum()/max(a[:,0].sum(),1):.1f}%")
    print(f"\n  CAUSE OF DEATH over {len(hist)} frames:")
    print(f"    gate reset (still tracked by front-end): {n_gate:7d}  "
          f"({100*n_gate/max(n_destroyed,1):5.1f}% of deaths, {n_gate/len(hist):5.1f}/frame)")
    print(f"    front-end lost the track:                {n_lost:7d}  "
          f"({100*n_lost/max(n_destroyed,1):5.1f}% of deaths, {n_lost/len(hist):5.1f}/frame)")
    print(f"    front-end ids dropped per frame:         {n_fe_lost/len(hist):7.1f}"
          f"   -> front-end mean track life ~{a[:,0].mean()/max(n_fe_lost/len(hist),1e-9):.1f} frames"
          f"  (min_track_length gate = 5)")


if __name__ == "__main__":
    main()
