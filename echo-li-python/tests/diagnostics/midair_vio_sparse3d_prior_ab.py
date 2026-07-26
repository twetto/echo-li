"""A/B test: seed EqF landmark births from Sparse3D range priors on MidAir.

This tests the deferred plan:

  baseline        : EqF sees every Rudolf-V feature and births with config sceneDepth.
  sparse3d_seeded : EqF sees every Rudolf-V feature, but any available Sparse3D
                    range prior is passed through at birth.
  stereo_seeded   : EqF sees every Rudolf-V feature, but any available left-right
                    stereo range prior is passed through at birth.
  sparse_defer    : Sparse3D tracks every Rudolf-V feature; EqF only admits a new
                    feature once Sparse3D has a finite, mature range estimate.

Important limitation: the current core EqF API consumes the prior RANGE at birth,
but still initializes the landmark covariance from eqf.initialVariance.point. This
script therefore tests point-estimate/scale benefit first, not covariance transfer.

Example:
  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_vio_sparse3d_prior_ab.py \
      --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 2 --frames 400 --config configs/eqvio_midair.yaml
"""
import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import run_manifest  # noqa: E402
from midair_vio_run import umeyama  # noqa: E402
import echo_li  # noqa: E402


SPARSE_KEYS = [
    "max_pool_size", "min_track_length", "conv_inlier_ratio", "conv_variance_threshold",
    "init_depth_var", "init_invdepth_var", "sigma_pixel", "flow_age_rate_px_per_frame",
    "bias_walk_var", "uniform_z_max", "uniform_rho_max", "uniform_d_min", "uniform_d_max",
    "a_init", "b_init", "ab_min", "ab_max", "min_inlier_ratio",
    "mahalanobis_reset_chi2", "process_depth_var", "range_walk_var", "pose_range_scale",
    "min_parallax", "min_cos_sim", "min_depth", "max_depth", "reanchor_flow_px",
    "use_equivariant_output", "iekf_iterations", "rotation_unscented",
]


def nwu_body_pose(ds, k):
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    gt = ds.pose(k)
    out = np.eye(4)
    out[:3, :3] = t_ned_to_nwu @ gt[:3, :3]
    out[:3, 3] = t_ned_to_nwu @ gt[:3, 3]
    return out


def gt_body_velocity(ds, k):
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    gi = min(k * 4, len(ds.db[ds.traj]["groundtruth"]["velocity"]) - 1)
    v_ned = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][gi], float)
    R_wb = nwu_body_pose(ds, k)[:3, :3]
    return R_wb.T @ (t_ned_to_nwu @ v_ned)


def init_vio(args, cam, ext, ds):
    vio = echo_li.VIOFilter(args.config, cam)
    vio.set_camera_extrinsics(np.ascontiguousarray(ext))
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    gt0 = ds.pose(args.start)
    v0 = np.asarray(ds.db[ds.traj]["groundtruth"]["velocity"][args.start * 4])
    R0 = t_ned_to_nwu @ gt0[:3, :3]
    v0_body = R0.T @ (t_ned_to_nwu @ v0)
    vio.set_initial_state(
        (t_ned_to_nwu @ gt0[:3, 3]).tolist(),
        np.ascontiguousarray(R0),
        v0_body.tolist(),
    )
    return vio


def vio_body_pose(vio):
    pos, quat = vio.get_pose()
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
    T[:3, 3] = np.asarray(pos)
    return T


def sparse_settings(config_path):
    cfg = yaml.safe_load(open(config_path)) or {}
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    settings = {k: sparse_cfg[k] for k in SPARSE_KEYS if k in sparse_cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings.setdefault("min_track_length", 1)
    return settings


def prior_source_for_mode(mode):
    if mode == "sparse3d_seeded":
        return "sparse3d"
    if mode == "stereo_seeded":
        return "stereo"
    if mode == "sparse_defer":
        return "sparse3d"
    return "none"


def parse_modes(text):
    aliases = {
        "all": ["baseline", "sparse3d_seeded", "stereo_seeded"],
        "baseline": ["baseline"],
        "sparse3d": ["sparse3d_seeded"],
        "sparse3d_seeded": ["sparse3d_seeded"],
        "stereo": ["stereo_seeded"],
        "stereo_seeded": ["stereo_seeded"],
    }
    modes = []
    for raw in text.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"unknown mode '{key}'; use all, baseline, sparse3d, stereo, "
                "or a comma-separated list"
            )
        modes.extend(aliases[key])
    if not modes:
        raise ValueError("empty --modes")
    return modes


def read_gray(ds, k):
    img = ds.image(k)
    return md.to_u8(img) if hasattr(md, "to_u8") else np.asarray(img).astype(np.uint8)


def read_right_gray(ds, k):
    p = ds.dir / "color_right" / ds.traj / f"{k:06d}.JPEG"
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(p))
    if ds.scale != 1.0:
        img = cv2.resize(img, None, fx=ds.scale, fy=ds.scale, interpolation=cv2.INTER_AREA)
    return img


class FrameReader:
    def get(self, k):
        raise NotImplementedError

    def close(self):
        pass


class DirectFrameReader(FrameReader):
    def __init__(self, ds, include_right=False):
        self.ds = ds
        self.include_right = include_right

    def get(self, k):
        left = read_gray(self.ds, k)
        if self.include_right:
            return left, read_right_gray(self.ds, k)
        return left


class PrefetchFrameReader(FrameReader):
    def __init__(self, ds, start, nimg, depth, include_right=False):
        self.q = queue.Queue(maxsize=max(1, depth))
        self.thread = threading.Thread(
            target=self._worker,
            args=(ds, start, nimg, include_right),
            daemon=True,
        )
        self.thread.start()

    def _worker(self, ds, start, nimg, include_right):
        try:
            for k in range(start, start + nimg):
                left = read_gray(ds, k)
                item = (left, read_right_gray(ds, k)) if include_right else left
                self.q.put((k, item, None))
        except BaseException as exc:
            self.q.put((None, None, exc))
        finally:
            self.q.put((None, None, None))

    def get(self, k):
        got_k, img, exc = self.q.get()
        if exc is not None:
            raise exc
        if got_k is None:
            raise RuntimeError(f"frame prefetch ended before frame {k}")
        if got_k != k:
            raise RuntimeError(f"frame prefetch order mismatch: expected {k}, got {got_k}")
        return img


def sparse_priors(filt, min_track, max_rel_sigma, var_scale):
    priors = {}
    candidates = 0
    rel_sigmas = []
    for fid, fd in filt.get_features().items():
        if int(fd["track_length"]) < min_track:
            continue
        candidates += 1
        p = np.asarray(fd["position"], float)
        cov = np.asarray(fd["covariance_euclidean"], float)
        rng = float(np.linalg.norm(p))
        if rng <= 1e-6 or not np.isfinite(cov).all():
            continue
        rhat = p / rng
        var_r = float(rhat @ cov @ rhat) * var_scale
        rel_sigma = np.sqrt(max(var_r, 0.0)) / rng
        if np.isfinite(rel_sigma):
            rel_sigmas.append(rel_sigma)
        if var_r > 0.0 and np.isfinite(var_r) and rel_sigma <= max_rel_sigma:
            priors[int(fid)] = [rng, var_r]
    return priors, candidates, rel_sigmas


def prior_rel_sigma(prior):
    rng = float(prior[0])
    var_r = float(prior[1])
    if rng <= 0.0 or var_r < 0.0:
        return float("inf")
    rel = np.sqrt(var_r) / rng
    return float(rel) if np.isfinite(rel) else float("inf")


def cap_eqf_observations(all_uvs, existing_ids, priors, max_obs, selection):
    if max_obs <= 0 or len(all_uvs) <= max_obs:
        return all_uvs

    ordered = list(all_uvs)
    existing = [fid for fid in ordered if fid in existing_ids]
    prior_new = [fid for fid in ordered if fid not in existing_ids and fid in priors]
    other_new = [fid for fid in ordered if fid not in existing_ids and fid not in priors]
    if selection == "prior_uncertainty":
        prior_new.sort(key=lambda fid: prior_rel_sigma(priors[fid]))
    keep = (existing + prior_new + other_new)[:max_obs]
    return {fid: all_uvs[fid] for fid in keep}


def open_writer(path, w, h, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if wr.isOpened():
        return wr, path
    alt = path.with_suffix(".avi")
    wr = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return wr, alt


def rec_field(rec, idx):
    return np.array([x[idx] for x in rec], float)


def summarize_state_monitor(mode, rec):
    if len(rec) < 5:
        return
    vel_err = rec_field(rec, 10)
    gyro_norm = rec_field(rec, 13)
    accel_norm = rec_field(rec, 14)
    speed_gt = np.linalg.norm(rec_field(rec, 9), axis=1)
    finite = np.isfinite(vel_err) & np.isfinite(gyro_norm) & np.isfinite(accel_norm)
    if finite.sum() < 5:
        return
    ve = vel_err[finite]
    gb = gyro_norm[finite]
    ab = accel_norm[finite]
    sp = speed_gt[finite]
    q20, q80 = np.quantile(ve, [0.2, 0.8])
    low = ve <= q20
    high = ve >= q80
    def stats(mask):
        return (float(np.median(ve[mask])), float(np.median(gb[mask])),
                float(np.median(ab[mask])), float(np.median(sp[mask])), int(mask.sum()))
    lo = stats(low)
    hi = stats(high)
    corr_g = float(np.corrcoef(ve, gb)[0, 1]) if len(ve) > 2 and np.std(gb) > 0 else np.nan
    corr_a = float(np.corrcoef(ve, ab)[0, 1]) if len(ve) > 2 and np.std(ab) > 0 else np.nan
    print(f"  state monitor {mode}: low vel-error q20<= {q20:.3f} m/s "
          f"n={lo[4]} med(|dv|,|bg|,|ba|,|v_gt|)="
          f"({lo[0]:.3f}, {lo[1]:.3g}, {lo[2]:.3g}, {lo[3]:.3f})")
    print(f"  state monitor {mode}: high vel-error q80>= {q80:.3f} m/s "
          f"n={hi[4]} med(|dv|,|bg|,|ba|,|v_gt|)="
          f"({hi[0]:.3f}, {hi[1]:.3g}, {hi[2]:.3g}, {hi[3]:.3f}) "
          f"corr(|dv|,|bg|)={corr_g:.3f} corr(|dv|,|ba|)={corr_a:.3f}")


def plot_state_monitor(path, results):
    rows = len(results)
    if rows == 0:
        return
    fig, ax = plt.subplots(rows, 3, figsize=(13, 3.1 * rows), squeeze=False)
    for r_i, r in enumerate(results):
        rec = r.get("rec", [])
        if not rec:
            continue
        k = np.array([x[0] for x in rec], int)
        vel_err = rec_field(rec, 10)
        gyro_norm = rec_field(rec, 13)
        accel_norm = rec_field(rec, 14)
        speed_gt = np.linalg.norm(rec_field(rec, 9), axis=1)
        ax[r_i, 0].plot(k, vel_err, lw=1.0, label="|v_est-v_gt|")
        ax[r_i, 0].plot(k, speed_gt, lw=0.8, alpha=0.7, label="|v_gt|")
        ax[r_i, 0].set_ylabel(r["mode"])
        ax[r_i, 0].legend(loc="upper right", fontsize=8)
        ax[r_i, 0].grid(True, alpha=0.25)
        ax[r_i, 1].plot(k, gyro_norm, lw=1.0, label="|gyro bias|")
        ax[r_i, 1].plot(k, accel_norm, lw=1.0, label="|accel bias|")
        ax[r_i, 1].legend(loc="upper right", fontsize=8)
        ax[r_i, 1].grid(True, alpha=0.25)
        ax[r_i, 2].scatter(vel_err, accel_norm, s=5, alpha=0.35, label="accel")
        ax[r_i, 2].scatter(vel_err, gyro_norm, s=5, alpha=0.35, label="gyro")
        ax[r_i, 2].set_xlabel("|velocity error| [m/s]")
        ax[r_i, 2].legend(loc="upper right", fontsize=8)
        ax[r_i, 2].grid(True, alpha=0.25)
    ax[-1, 0].set_xlabel("frame")
    ax[-1, 1].set_xlabel("frame")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved state monitor plot ->", path)


def draw_points(vis, pts, color, radius=2):
    for uv in pts:
        x, y = int(round(uv[0])), int(round(uv[1]))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), radius, color, -1, cv2.LINE_AA)


def build_jet_range_lut():
    # OpenCV JET maps low values to blue and high values to red.
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)[:, 0, :]


def jet_metric_range_color(range_m, min_range_m, max_range_m, lut):
    rng = float(range_m)
    if not np.isfinite(rng) or rng <= 0.0:
        return (170, 170, 170)
    near = max(float(min_range_m), 1e-6)
    far = max(float(max_range_m), near + 1e-6)
    # Close -> red, far -> blue: invert the metric range before indexing JET.
    t = int(round(255.0 * (1.0 - np.clip((rng - near) / (far - near), 0.0, 1.0))))
    return tuple(int(x) for x in lut[t])


def draw_depth_coded_obs(vis, all_uvs, vio_uvs, priors, args, range_lut):
    for fid, uv in vio_uvs.items():
        x, y = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]):
            continue
        if fid in priors:
            color = jet_metric_range_color(
                priors[fid][0],
                args.vis_min_range_m,
                args.vis_max_range_m,
                range_lut,
            )
        else:
            color = (185, 185, 185)
        cv2.circle(vis, (x, y), args.vis_eqf_radius, color, -1, cv2.LINE_AA)

    for fid, prior in priors.items():
        if fid not in all_uvs:
            continue
        x, y = int(round(all_uvs[fid][0])), int(round(all_uvs[fid][1]))
        if not (0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]):
            continue
        color = jet_metric_range_color(prior[0], args.vis_min_range_m, args.vis_max_range_m, range_lut)
        cv2.circle(vis, (x, y), args.vis_stereo_radius, color, args.vis_stereo_thickness, cv2.LINE_AA)


def topdown_mapper(gt_path, panel_w, panel_h):
    xy = np.asarray(gt_path, float)[:, :2]
    lo = xy.min(0)
    hi = xy.max(0)
    span = np.maximum(hi - lo, 1.0)
    pad = 0.15 * span.max()
    lo -= pad
    hi += pad
    span = np.maximum(hi - lo, 1.0)
    scale = min((panel_w - 36) / span[0], (panel_h - 72) / span[1])
    center = 0.5 * (lo + hi)

    def map_xy(p):
        q = np.asarray(p, float)[:2]
        x = int(round(panel_w * 0.5 + (q[0] - center[0]) * scale))
        y = int(round(panel_h * 0.54 - (q[1] - center[1]) * scale))
        return x, y

    return map_xy


class TrajectoryPanel:
    def __init__(self, width, height, map_xy):
        self.width = width
        self.height = height
        self.map_xy = map_xy
        self.canvas = np.full((height, width, 3), (18, 18, 18), np.uint8)
        self.prev_gt = None
        self.prev_est = None

    def update(self, gt, est):
        gt_pt = self.map_xy(gt)
        est_pt = self.map_xy(est)
        if self.prev_gt is not None:
            cv2.line(self.canvas, self.prev_gt, gt_pt, (80, 255, 80), 2, cv2.LINE_AA)
        if self.prev_est is not None:
            cv2.line(self.canvas, self.prev_est, est_pt, (80, 80, 255), 2, cv2.LINE_AA)
        self.prev_gt = gt_pt
        self.prev_est = est_pt

    def frame(self):
        out = self.canvas.copy()
        if self.prev_gt is not None:
            cv2.circle(out, self.prev_gt, 4, (80, 255, 80), -1, cv2.LINE_AA)
        if self.prev_est is not None:
            cv2.circle(out, self.prev_est, 4, (80, 80, 255), -1, cv2.LINE_AA)
        cv2.putText(out, "top-down trajectory", (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(out, "green GT", (14, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(out, "red estimate", (118, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 255), 1, cv2.LINE_AA)
        return out


def draw_frame(image_vis, mode, k, stats, all_uvs, vio_uvs, priors, births,
               traj_panel, rec_len, args):
    vis = np.zeros((image_vis.shape[0], image_vis.shape[1] + args.traj_panel_width, 3), np.uint8)
    vis[:, :image_vis.shape[1]] = image_vis
    panel_x = image_vis.shape[1]
    vis[:, panel_x:] = traj_panel.frame()

    draw_points(vis, [uv for fid, uv in all_uvs.items() if fid not in vio_uvs], (105, 105, 105), 1)
    draw_depth_coded_obs(vis, all_uvs, vio_uvs, priors, args, args.vis_range_lut)
    draw_points(vis, [all_uvs[j] for j in births if j in all_uvs], (255, 0, 255), 4)
    cv2.rectangle(vis, (0, 0), (image_vis.shape[1], 76), (0, 0, 0), -1)
    cv2.putText(vis, f"{mode}  frame {k}  rec {rec_len}/{args.frames}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"klt_surv {stats.get('tracked', 0)}  reservoir {len(all_uvs)}  "
                f"obs_sent {len(vio_uvs)}  priors {len(priors)}  prior_births {len(births)}",
                (6, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, "filled=EqF obs  hollow=depth prior  Jet=range close red/far blue",
                (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 215, 215), 1, cv2.LINE_AA)
    return vis


class Progress:
    def __init__(self, total, desc, enabled=True):
        self.total = total
        self.desc = desc
        self.n = 0
        self.t0 = time.time()
        self.enabled = enabled
        self.bar = (tqdm(total=total, desc=desc, unit="frame", dynamic_ncols=True)
                    if enabled and tqdm else None)

    def update(self, tracked, vio_obs, priors):
        self.n += 1
        fps = self.n / max(time.time() - self.t0, 1e-9)
        postfix = {"klt_surv": tracked, "obs_sent": vio_obs, "priors": priors, "fps": f"{fps:.1f}"}
        if self.bar is not None:
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(1)
        elif self.enabled and (self.n == self.total or self.n % 100 == 0):
            print(f"  {self.desc:12s} [{self.n:4d}/{self.total}] klt_surv={tracked:3d} "
                  f"obs_sent={vio_obs:3d} priors={priors:3d} {fps:5.1f} fps")

    def close(self):
        if self.bar is not None:
            self.bar.close()


def run_once(mode, args, ds, f, cx, cy, W, H, ext, events, map_xy):
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    if args.sparse_max_features > 0:
        fcfg.max_features = args.sparse_max_features
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    tracker = echo_li.Frontend(fcfg, W, H)
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    vio = init_vio(args, cam, ext, ds)
    prior_source = prior_source_for_mode(mode)
    sparse = None
    stereo = None
    if mode != "baseline" and prior_source == "sparse3d":
        sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **sparse_settings(args.config))
    elif mode != "baseline" and prior_source == "stereo":
        stereo = echo_li.Stereo.from_pinhole(
            f, f, cx, cy, W, H,
            [-args.stereo_baseline_m, 0.0, 0.0],
            None,
            args.config,
        )

    rec = []
    prior_births = 0
    deferred_new = 0
    admitted_new = 0
    prior_candidates_total = 0
    prior_rel_sigmas = []
    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    writer = None
    vpath = None
    traj_panel = None
    video_name = mode
    need_right = stereo is not None
    if args.video_out_dir and mode in ("baseline", "sparse3d_seeded", "stereo_seeded"):
        writer, vpath = open_writer(
            Path(args.video_out_dir) / f"{video_name}.mp4",
            W + args.traj_panel_width,
            H,
            args.fps,
        )
        traj_panel = TrajectoryPanel(args.traj_panel_width, H, map_xy)
    reader = (PrefetchFrameReader(ds, args.start, args.frames, args.prefetch_images, need_right)
              if args.prefetch_images > 0 else DirectFrameReader(ds, need_right))
    progress = Progress(args.frames, video_name, not args.no_progress)

    try:
        for stamp, et, data in events:
            if et == "imu":
                gyro = np.asarray(data[0])
                if args.gyro_frame in ("spatial_est", "world"):
                    R_est = vio_body_pose(vio)[:3, :3]
                    gyro = R_est.T @ (t_ned_to_nwu @ gyro)
                elif args.gyro_frame in ("repaired_gt", "spatial_gt", "world_gt"):
                    gi = min(int(round(stamp * 100.0)), len(ds.att) - 1)
                    q = ds.att[gi]
                    R_gt = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                    gyro = R_gt.T @ gyro
                vio.process_imu(stamp, gyro.tolist(), data[1])
                continue

            k = data
            frame = reader.get(k)
            if need_right:
                gray, right_gray = frame
            else:
                gray, right_gray = frame, None
            feats, stats = tracker.process(gray)
            all_uvs = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
            priors = {}

            if sparse is not None and all_uvs:
                if args.sparse_pose == "gt":
                    T_wc_sparse = nwu_body_pose(ds, k) @ md.RT_BC
                else:
                    T_wc_sparse = vio_body_pose(vio) @ ext
                sparse.update(float(stamp), all_uvs, T_wc_sparse.tolist(), None, None)
                priors, candidates, rel_sigmas = sparse_priors(
                    sparse, args.sparse_min_track, args.max_rel_sigma, args.prior_var_scale)
                prior_candidates_total += candidates
                prior_rel_sigmas.extend(rel_sigmas)
            elif stereo is not None and right_gray is not None and all_uvs:
                priors = dict(stereo.range_priors(
                    right_gray,
                    tracker,
                    args.stereo_sigma_pixel_scale,
                ))
                prior_candidates_total += len(all_uvs)
                for rng, var_r in priors.values():
                    rng = float(rng)
                    var_r = float(var_r)
                    if rng > 0.0 and var_r >= 0.0 and np.isfinite(var_r):
                        prior_rel_sigmas.append(np.sqrt(var_r) / rng)

            existing = {int(x) for x in vio.get_landmarks().keys()}
            if mode == "baseline":
                vio_uvs = cap_eqf_observations(
                    all_uvs, existing, priors, args.eqf_max_obs, args.eqf_selection)
                vio.process_vision(stamp, vio_uvs)
                births = set()
            elif mode in ("sparse3d_seeded", "stereo_seeded"):
                vio_uvs = cap_eqf_observations(
                    all_uvs, existing, priors, args.eqf_max_obs, args.eqf_selection)
                vio.process_vision_with_depth_priors(stamp, vio_uvs, priors)
                after = {int(x) for x in vio.get_landmarks().keys()}
                births = after - existing
                prior_births += len(births & set(priors))
                admitted_new += len(births)
            else:
                new_ids = set(all_uvs) - existing
                allowed_new = new_ids & set(priors)
                deferred_new += len(new_ids - allowed_new)
                keep = existing | allowed_new
                vio_uvs = {fid: uv for fid, uv in all_uvs.items() if fid in keep}
                vio_uvs = cap_eqf_observations(
                    vio_uvs, existing, priors, args.eqf_max_obs, args.eqf_selection)
                vio.process_vision_with_depth_priors(stamp, vio_uvs, priors)
                after = {int(x) for x in vio.get_landmarks().keys()}
                births = after - existing
                admitted_new += len(births)
                prior_births += len(births & set(priors))

            pos, quat = vio.get_pose()
            vel_est = np.asarray(vio.get_velocity(), float)
            gyro_bias, accel_bias = vio.get_biases()
            gyro_bias = np.asarray(gyro_bias, float)
            accel_bias = np.asarray(accel_bias, float)
            vel_gt_body = gt_body_velocity(ds, k)
            vel_err = float(np.linalg.norm(vel_est - vel_gt_body))
            gt = nwu_body_pose(ds, k)
            # Pose-covariance consistency capture: EqF camera-pose cov (P_vv=pos,
            # P_ww=att) + GT attitude, for offline NEES scoring.
            pcov = vio.get_camera_pose_covariance()
            gt_quat = Rot.from_matrix(gt[:3, :3]).as_quat()
            if pcov is None:
                p_vv = np.full((3, 3), np.nan)
                p_ww = np.full((3, 3), np.nan)
            else:
                p_vv = np.asarray(pcov[0], float)
                p_ww = np.asarray(pcov[1], float)
            if traj_panel is not None:
                traj_panel.update(gt[:3, 3], pos)
            if writer is not None and ((len(rec) + 1) % args.video_stride == 0):
                image_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                vis = draw_frame(image_vis, video_name, k, stats, all_uvs, vio_uvs, priors,
                                 births & set(priors), traj_panel, len(rec) + 1, args)
                writer.write(vis)
            rec.append((k, np.asarray(pos), np.asarray(quat), gt[:3, 3].copy(),
                        stats["tracked"], len(all_uvs), len(vio_uvs), len(priors),
                        vel_est, vel_gt_body, vel_err, gyro_bias, accel_bias,
                        float(np.linalg.norm(gyro_bias)), float(np.linalg.norm(accel_bias)),
                        gt_quat, p_vv, p_ww))
            progress.update(stats["tracked"], len(vio_uvs), len(priors))
    finally:
        reader.close()
        progress.close()

    if writer is not None:
        writer.release()
        print(f"  saved video -> {vpath}")
        run_manifest.save_run_manifest(vpath, args.config, extra={
            "mode": mode, "traj": args.traj, "frames": args.frames,
            "scale": args.scale, "gyro_frame": args.gyro_frame,
            "eqf_max_obs": args.eqf_max_obs, "eqf_selection": args.eqf_selection,
            "sparse_max_features": args.sparse_max_features,
            "stereo_baseline_m": args.stereo_baseline_m})
    summarize_state_monitor(mode, rec)

    if len(rec) < 20:
        return {"mode": mode, "n": len(rec), "ate": np.nan, "path": np.nan,
                "final": np.nan, "prior_births": prior_births,
                "deferred_new": deferred_new, "admitted_new": admitted_new,
                "prior_candidates": prior_candidates_total,
                "prior_rel_sigma_p50": np.nan,
                "rec": rec}
    prior_rel_sigma_p50 = float(np.median(prior_rel_sigmas)) if prior_rel_sigmas else np.nan
    est = np.array([r[1] for r in rec])
    gtp = np.array([r[3] for r in rec])
    R, t, ate = umeyama(est, gtp)
    path = float(np.linalg.norm(np.diff(gtp, axis=0), axis=1).sum())
    final = float(np.linalg.norm((R @ est[-1] + t) - gtp[-1]))
    return {"mode": mode, "n": len(rec), "ate": ate, "path": path, "final": final,
            "prior_births": prior_births, "deferred_new": deferred_new,
            "admitted_new": admitted_new, "prior_candidates": prior_candidates_total,
            "prior_rel_sigma_p50": prior_rel_sigma_p50, "rec": rec}


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=400,
                    help="number of image frames; <=0 means all remaining frames")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_midair.yaml"))
    ap.add_argument("--sparse-pose", default="vio", choices=["vio", "gt"],
                    help="pose source used only by Sparse3D before making EqF birth priors")
    ap.add_argument("--modes", default="all",
                    help="real run selector: all, baseline, sparse3d, stereo, or comma-separated list")
    ap.add_argument("--sparse-max-features", type=int, default=300,
                    help="Rudolf-V tracker reservoir size; Sparse3D/stereo prior extraction sees this pool")
    ap.add_argument("--eqf-max-obs", type=int, default=40,
                    help="maximum observations sent into EqF per image; <=0 sends the full reservoir")
    ap.add_argument("--eqf-selection", default="prior_uncertainty",
                    choices=["prior_uncertainty", "preserve_order"],
                    help="when capping EqF observations, rank prior-backed births by sigma/range "
                    "or keep Rudolf-V feature order")
    ap.add_argument("--sparse-min-track", type=int, default=5)
    ap.add_argument("--max-rel-sigma", type=float, default=2.0,
                    help="skip Sparse3D priors whose reported range sigma/range is larger")
    ap.add_argument("--prior-var-scale", type=float, default=1.0,
                    help="logged into the prior map; current EqF core does not consume it yet")
    ap.add_argument("--stereo-baseline-m", type=float, default=1.0,
                    help="MidAir left-right stereo baseline in metres")
    ap.add_argument("--stereo-sigma-pixel-scale", type=float, default=20.0,
                    help="Stereo range variance heuristic scale passed to echo_li.Stereo")
    ap.add_argument("--gyro-frame", default="repaired_gt",
                    choices=["body", "spatial_est", "repaired_gt", "spatial_gt", "world", "world_gt"],
                    help="How to feed MidAir gyro samples to EqF. 'body' trusts the HDF5 "
                    "metadata. 'spatial_est'/'world' treats the released channel as "
                    "spatial/world and rotates it with the current estimate. "
                    "'repaired_gt'/'spatial_gt'/'world_gt' uses MidAir GT attitude to "
                    "repair the released spatial channel into the body-frame gyro a real "
                    "IMU should have provided.")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--no-progress", action="store_true",
                    help="disable tqdm/plain progress output for batch profiling")
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--state-plot-out", default="",
                    help="optional PNG of velocity error and IMU bias estimates")
    ap.add_argument("--video-out-dir", default="",
                    help="directory for per-mode inspection videos")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--video-stride", type=int, default=1,
                    help="write one video frame every N image frames")
    ap.add_argument("--traj-panel-width", type=int, default=360)
    ap.add_argument("--vis-min-range-m", type=float, default=1.0,
                    help="Jet color near end: this range maps to red")
    ap.add_argument("--vis-max-range-m", type=float, default=150.0,
                    help="Jet color far end: this range maps to blue")
    ap.add_argument("--vis-eqf-radius", type=int, default=2,
                    help="filled EqF observation marker radius")
    ap.add_argument("--vis-stereo-radius", type=int, default=5,
                    help="hollow depth-prior marker radius")
    ap.add_argument("--vis-stereo-thickness", type=int, default=1,
                    help="hollow depth-prior marker line thickness")
    ap.add_argument("--prefetch-images", type=int, default=16,
                    help="bounded image decode queue depth; 0 disables threaded prefetch")
    ap.add_argument("--include-defer", action="store_true",
                    help="also run the sparse_defer starvation diagnostic")
    args = ap.parse_args()
    args.vis_range_lut = build_jet_range_lut()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    nimg = ds.n - args.start if args.frames <= 0 else min(args.frames, ds.n - args.start)
    args.frames = nimg
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj} {W}x{H} f={f:.1f} frames={nimg}")
    print("Sparse3D prior limitation: EqF uses prior range at birth, but not prior covariance yet.")
    try:
        modes = parse_modes(args.modes)
    except ValueError as exc:
        ap.error(str(exc))
    if args.include_defer:
        modes.append("sparse_defer")

    print(f"modes: {', '.join(modes)}")
    print(f"feature split: Sparse3D/stereo reservoir={args.sparse_max_features} "
          f"EqF obs cap={'all' if args.eqf_max_obs <= 0 else args.eqf_max_obs} "
          f"selection={args.eqf_selection}")
    if any(prior_source_for_mode(m) == "stereo" for m in modes):
        print(f"stereo: rectified pinhole baseline={args.stereo_baseline_m:.3f}m "
              f"sigma_pixel_scale={args.stereo_sigma_pixel_scale:g}")
    if args.gyro_frame in ("repaired_gt", "spatial_gt", "world_gt"):
        print("gyro-frame: repaired_gt uses MidAir GT attitude to repair the released "
              "spatial gyro into the body-frame gyro that a real IMU should provide.")
    elif args.gyro_frame in ("spatial_est", "world"):
        print("gyro-frame: spatial_est repairs MidAir's spatial gyro with estimated attitude; "
              "this can feed attitude error back into the IMU adapter.")
    print(f"image loading: {'direct' if args.prefetch_images <= 0 else f'prefetch depth {args.prefetch_images}'}")

    imu = ds.db[ds.traj]["imu"]
    accel = imu["accelerometer"][:]
    gyro = imu["gyroscope"][:]
    imu0 = args.start * 4
    imu1 = min(len(accel), (args.start + nimg) * 4 + 4)
    imu_ev = [(i / 100.0, "imu", (gyro[i].tolist(), accel[i].tolist()))
              for i in range(imu0, imu1)]
    img_ev = [(k / 25.0, "img", k) for k in range(args.start, args.start + nimg)]
    events = sorted(imu_ev + img_ev, key=lambda e: e[0])
    ext = md.RT_BC
    gt_path = [nwu_body_pose(ds, k)[:3, 3] for k in range(args.start, args.start + nimg)]
    map_xy = topdown_mapper(gt_path, args.traj_panel_width, H)

    results = []
    for mode in modes:
        print(f"\n--- {mode} ---")
        results.append(run_once(mode, args, ds, f, cx, cy, W, H, ext, events, map_xy))

    print("\n=== Sparse3D-prior EqF birth A/B ===")
    print(f"{'mode':>14s} {'poses':>6s} {'ATE m':>8s} {'ATE%':>7s} {'final m':>8s} "
          f"{'prior_births':>12s} {'deferred':>10s} {'admitted':>9s} "
          f"{'cand':>10s} {'relsig50':>9s}")
    for r in results:
        ate_pct = 100.0 * r["ate"] / max(r["path"], 1e-9)
        print(f"{r['mode']:>14s} {r['n']:6d} {r['ate']:8.3f} {ate_pct:7.2f} "
              f"{r['final']:8.3f} {r['prior_births']:12d} {r['deferred_new']:10d} "
              f"{r['admitted_new']:9d} {r['prior_candidates']:10d} "
              f"{r['prior_rel_sigma_p50']:9.3f}")

    if args.state_plot_out:
        plot_state_monitor(args.state_plot_out, results)

    if args.save_npz:
        payload = {}
        for r in results:
            rec = r.pop("rec")
            payload[r["mode"] + "_k"] = np.array([x[0] for x in rec], int)
            payload[r["mode"] + "_est"] = np.array([x[1] for x in rec], float)
            payload[r["mode"] + "_quat"] = np.array([x[2] for x in rec], float)
            payload[r["mode"] + "_gt"] = np.array([x[3] for x in rec], float)
            payload[r["mode"] + "_counts"] = np.array([x[4:8] for x in rec], float)
            payload[r["mode"] + "_vel_est_body"] = np.array([x[8] for x in rec], float)
            payload[r["mode"] + "_vel_gt_body"] = np.array([x[9] for x in rec], float)
            payload[r["mode"] + "_vel_err"] = np.array([x[10] for x in rec], float)
            payload[r["mode"] + "_gyro_bias"] = np.array([x[11] for x in rec], float)
            payload[r["mode"] + "_accel_bias"] = np.array([x[12] for x in rec], float)
            payload[r["mode"] + "_gt_quat"] = np.array([x[15] for x in rec], float)
            payload[r["mode"] + "_pcov_pos"] = np.array([x[16] for x in rec], float)
            payload[r["mode"] + "_pcov_att"] = np.array([x[17] for x in rec], float)
        payload["summary_names"] = np.array(["n", "ate", "path", "final", "prior_births",
                                             "deferred_new", "admitted_new",
                                             "prior_candidates", "prior_rel_sigma_p50"])
        payload["summary"] = np.array([[r[k] for k in ["n", "ate", "path", "final",
                                                       "prior_births", "deferred_new",
                                                       "admitted_new", "prior_candidates",
                                                       "prior_rel_sigma_p50"]]
                                       for r in results], float)
        payload["summary_modes"] = np.array([r["mode"] for r in results])
        np.savez(args.save_npz, **payload)
        print("saved ->", args.save_npz)
        run_manifest.save_run_manifest(args.save_npz, args.config, extra={
            "modes": args.modes, "traj": args.traj, "frames": args.frames,
            "scale": args.scale, "gyro_frame": args.gyro_frame,
            "eqf_max_obs": args.eqf_max_obs, "sparse_max_features": args.sparse_max_features,
            "stereo_baseline_m": args.stereo_baseline_m})


if __name__ == "__main__":
    main()
