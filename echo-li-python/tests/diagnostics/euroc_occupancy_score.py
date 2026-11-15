"""Score the occupancy map against truth on EuRoC (design note §7 item 5, the next theory item).

The stack as shipped: MSCEqF (ZUPT, in-state landmarks) on Rudolf-V corners of the raw frame,
the DenseDepthMapper (FlowDep-F + its Sparse3D + fusion + fill, Rust) on the raw frame with
the backend's pose, and the LocalOccupancyMap fed the dense frame. Several maps are fed the
SAME dense frames, one per integration rule under test:

  fixed_meas    v0 fixed-increment, Measured pixels only
  fixed_fill    v0 fixed-increment, Measured + Filled (Filled carves free space only)
  sigma_meas    v1 sigma-band, Measured only
  sigma_fill    v1 sigma-band, Measured + Filled
  onesided_*    v2 one-sided: hit at the endpoint, free carved only up to range - band_k*sigma_m

Every map also carries the sigma CHANNEL (per cell, the tightest sigma_m of any ray whose band
covered it; written by every Measured ray, never read by the occupancy). The scorer reads it as a
planner would: inflate each asserted-occupied cell by kappa*sigma(cell) and report how many of
the rule's holes (map-free, truth-occupied) that covers against how much truly-free space it
consumes, next to a constant inflation radius at the same two numbers.

and one TRUTH map is built by the same integrator from the ideal sensor: the Leica z-buffer
through the Vicon pose at every frame (range exact, sigma tiny, hit saturating), so the truth
is what the map would hold if depth and pose were perfect, on the same rays, with the same
occlusion. Score, at the end and every --score-every frames, over the cells the truth labels:

  reliability   P(occupied) from the map's log-odds vs the empirical occupied fraction, in bins
  ECE           expected calibration error of that diagram
  false obstacle  cells the map calls occupied (>= occupied_threshold) that truth calls free
  missed obstacle cells truth calls occupied that the map calls free (<= free_threshold):
                  the dangerous case, free space carved through a real surface
  unknown       fraction of truth-labelled cells the map leaves unknown

The ego-centric grids recentre on their own camera (estimated vs GT), so cells are matched in
WORLD coordinates through grid_info(), not by index.

Usage:
  .venv/bin/python echo-li-python/tests/diagnostics/euroc_occupancy_score.py \
      --root ~/18TB/datasets/EuRoC/vicon_room1/V1_01_easy/V1_01_easy --cache $SCRATCH/euroc \
      --out $SCRATCH/occ/V1_01.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
from euroc_fdf_depth import load_cam, load_cloud, load_gt, zbuffer  # noqa: E402
from midair_e2e_depth_nees import SPARSE_KEYS  # noqa: E402

GRAVITY = 9.81

RULES = {
    "fixed_meas": dict(update_mode="fixed", use_fill=False, pose="est", depth="dense"),
    "fixed_fill": dict(update_mode="fixed", use_fill=True, pose="est", depth="dense"),
    "sigma_meas": dict(update_mode="sigma", use_fill=False, pose="est", depth="dense"),
    "sigma_fill": dict(update_mode="sigma", use_fill=True, pose="est", depth="dense"),
    # v2: hit at the endpoint, free carved only up to range - band_k * sigma_m (the one-sided
    # bound of the formulation note, eq. 14, in map form)
    "onesided_meas": dict(update_mode="one_sided", use_fill=False, pose="est", depth="dense"),
    "onesided_fill": dict(update_mode="one_sided", use_fill=True, pose="est", depth="dense"),
    # controls: the same dense frame at the Vicon pose (depth error only), and the ideal sensor
    # through the default fixed rule (the integrator's own ceiling at this resolution/stride)
    "fixed_meas_gtpose": dict(update_mode="fixed", use_fill=False, pose="gt", depth="dense"),
    "oracle_fixed": dict(update_mode="fixed", use_fill=False, pose="gt", depth="gt"),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_map(test, truth, occ_thr, free_thr, truth_occ_thr, truth_free_thr, nbins=10):
    """Match test cells to truth cells in world coordinates; return a dict of metrics."""
    ox, oy, oz, res, W, H, D = test.grid_info()
    tox, toy, toz, tres, TW, TH, TD = truth.grid_info()
    assert abs(res - tres) < 1e-9
    L = test.log_odds_grid()            # (D, H, W)
    T = truth.log_odds_grid()
    zi, yi, xi = np.mgrid[0:D, 0:H, 0:W]
    # world centres of the test cells -> truth indices
    tx = np.floor((ox + (xi + 0.5) * res - tox) / res).astype(np.int64)
    ty = np.floor((oy + (yi + 0.5) * res - toy) / res).astype(np.int64)
    tz = np.floor((oz + (zi + 0.5) * res - toz) / res).astype(np.int64)
    inside = (tx >= 0) & (tx < TW) & (ty >= 0) & (ty < TH) & (tz >= 0) & (tz < TD)
    tl = np.full(L.shape, np.nan, np.float32)
    tl[inside] = T[tz[inside], ty[inside], tx[inside]]
    t_occ = tl >= truth_occ_thr
    t_free = tl <= truth_free_thr
    labelled = t_occ | t_free
    n_lab = int(labelled.sum())
    out = {"n_truth_labelled": n_lab, "n_truth_occ": int(t_occ.sum()), "n_truth_free": int(t_free.sum())}
    if n_lab == 0:
        return out
    Ll = L[labelled]; y = t_occ[labelled].astype(np.float64)
    known = (Ll >= occ_thr) | (Ll <= free_thr)
    out["unknown_frac"] = float(1.0 - known.mean())
    p = sigmoid(Ll.astype(np.float64))
    # reliability over the cells the map has an opinion on (|L| > 0)
    has = Ll != 0
    edges = np.linspace(0, 1, nbins + 1)
    rel = []
    ece = 0.0
    for b in range(nbins):
        m = has & (p >= edges[b]) & (p < edges[b + 1] if b < nbins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            rel.append((edges[b], edges[b + 1], 0, np.nan, np.nan)); continue
        pm = float(p[m].mean()); ym = float(y[m].mean())
        rel.append((edges[b], edges[b + 1], int(m.sum()), pm, ym))
        ece += m.sum() / max(has.sum(), 1) * abs(pm - ym)
    out["reliability"] = rel
    out["ece"] = float(ece)
    map_occ = Ll >= occ_thr; map_free = Ll <= free_thr
    out["n_map_occ"] = int(map_occ.sum()); out["n_map_free"] = int(map_free.sum())
    # what a planner meets: of the cells the map asserts, how many are wrong
    out["false_obstacle"] = float((map_occ & (y == 0)).sum() / max(map_occ.sum(), 1))   # P(truth free | map occupied)
    out["missed_obstacle"] = float((map_free & (y == 1)).sum() / max(map_free.sum(), 1))  # P(truth occupied | map free)
    # and how much of the truth the map recovers
    out["occ_recall"] = float((map_occ & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["free_recall"] = float((map_free & (y == 0)).sum() / max((y == 0).sum(), 1))
    # the same two rates with a position tolerance: an asserted obstacle within d cells of a
    # true surface is a displaced surface, not a phantom; a free cell on a true surface with an
    # asserted obstacle within d cells is a displaced surface too
    from scipy import ndimage
    full_occ = np.zeros(L.shape, bool); full_occ[labelled] = map_occ
    full_free = np.zeros(L.shape, bool); full_free[labelled] = map_free
    d_truth = ndimage.distance_transform_edt(~t_occ) if t_occ.any() else np.full(L.shape, np.inf)
    d_map = ndimage.distance_transform_edt(~full_occ) if full_occ.any() else np.full(L.shape, np.inf)
    out["false_obstacle_tol"] = [float((full_occ & t_free & (d_truth > d)).sum() / max(full_occ.sum(), 1)) for d in (0, 1, 2, 3)]
    out["missed_obstacle_tol"] = [float((full_free & t_occ & (d_map > d)).sum() / max(full_free.sum(), 1)) for d in (0, 1, 2, 3)]
    # the sigma channel: of the map-free cells that are truly occupied (the holes), how many lie
    # inside some ray's band (finite sigma), i.e. a planner taking the channel's margin covers
    # them; and the price: how much of the truly-free map-free space is inside a band
    S = test.sigma_grid()
    holes = full_free & t_occ
    good_free = full_free & t_free
    out["n_holes"] = int(holes.sum())
    out["sigma_occ_med"] = float(np.median(S[full_occ])) if full_occ.any() else float("nan")
    # a planner's margin: inflate every asserted-occupied cell by kappa * sigma(cell) (the channel
    # at that cell), against a constant radius; score = holes covered vs truly-free space consumed
    def covered_by(radius_cells):
        """cells within `radius_cells[c]` of occupied cell c (per-cell radius), via EDT per bucket"""
        cov = np.zeros(L.shape, bool)
        for rb in (1, 2, 3, 4, 6, 8, 10):
            m = full_occ & (radius_cells >= rb)
            if not m.any():
                continue
            cov |= ndimage.distance_transform_edt(~m) <= rb + 1e-6
        return cov
    out["inflate"] = {}
    for kappa in (1.0, 2.0):
        rc = np.where(full_occ & np.isfinite(S), np.ceil(kappa * S / res), 0.0)
        cov = covered_by(np.minimum(rc, 10.0))
        out["inflate"][f"sigma_k{kappa:g}"] = (float((holes & cov).sum() / max(holes.sum(), 1)), float((good_free & cov).sum() / max(good_free.sum(), 1)))
    for rconst in (1, 2, 3, 4):
        cov = ndimage.distance_transform_edt(~full_occ) <= rconst + 1e-6 if full_occ.any() else np.zeros(L.shape, bool)
        out["inflate"][f"const_{rconst}"] = (float((holes & cov).sum() / max(holes.sum(), 1)), float((good_free & cov).sum() / max(good_free.sum(), 1)))
    # sigma as a soft WEIGHT: among the map's free cells (truth-labelled), does the hole rate rise
    # with the channel's sigma at that cell? Bins of sigma (m), plus "no band reached" as its own
    # bin; and the same against the distance to the nearest asserted obstacle (cells) as the
    # obvious competitor a planner already has
    free_lab = full_free & labelled
    Sf = S[free_lab]; hf = holes[free_lab]
    edges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, np.inf]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (Sf >= a) & (Sf < b)
        rows.append((a, b, int(m.sum()), float(hf[m].mean()) if m.any() else float("nan")))
    m = ~np.isfinite(Sf)
    rows.append((np.inf, np.inf, int(m.sum()), float(hf[m].mean()) if m.any() else float("nan")))
    out["hole_rate_by_sigma"] = rows
    dmap_f = d_map[free_lab]
    rows = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, np.inf)):
        m = (dmap_f > a + 1e-6) & (dmap_f <= b + 1e-6) if b < np.inf else dmap_f > a + 1e-6  # EDT values are often exact integers: bins (a, b]
        rows.append((a, b, int(m.sum()), float(hf[m].mean()) if m.any() else float("nan")))
    out["hole_rate_by_dist"] = rows
    out["hole_rate_overall"] = float(hf.mean()) if hf.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "configs/eqvio_euroc_rho.yaml"))
    ap.add_argument("--sparse-config", default=str(Path(__file__).resolve().parents[3] / "configs/diagnostics_midair_e2e_depth_nees.yaml"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--num-clones", type=int, default=11)
    ap.add_argument("--max-slam", type=int, default=50)
    ap.add_argument("--frontend-features", type=int, default=120)
    ap.add_argument("--fdf", action="append", default=["max_mean_abs_residual=30"], metavar="KEY=VAL")
    ap.add_argument("--fdf-sparse", action="append", default=["bias_walk_var=0.3"], metavar="KEY=VAL")
    ap.add_argument("--min-track", type=int, default=5)
    ap.add_argument("--resolution", type=float, default=0.10)
    ap.add_argument("--cells", type=int, default=160)
    ap.add_argument("--height", type=float, default=1.5, help="grid spans camera z +- this (m)")
    ap.add_argument("--sample-stride", type=int, default=4)
    ap.add_argument("--max-range", type=float, default=12.0)
    ap.add_argument("--band-k", type=float, default=2.0)
    ap.add_argument("--score-every", type=int, default=500)
    ap.add_argument("--video", default="", help="mp4: camera, dense depth, top-down fixed / one-sided / sigma / truth")
    ap.add_argument("--video-every", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--rrd", default="", help="Rerun recording (.rrd): camera, dense depth, trajectory, occupied cells per rule")
    ap.add_argument("--rrd-every", type=int, default=20, help="log the maps every Nth frame")
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    gt_t, gt_p, gt_R = load_gt(root)
    T_BS, (fu, fv, cu, cv_), D, (W0, H0), cam_t, cam_names = load_cam(root)
    imu = np.loadtxt(root / "mav0/imu0/data.csv", delimiter=",", skiprows=1); imu[:, 0] *= 1e-9
    from scipy.spatial.transform import Slerp, Rotation as Rot
    slerp = Slerp(gt_t, gt_R)

    def gt_pose_at(t):
        t = float(np.clip(t, gt_t[0], gt_t[-1]))
        T = np.eye(4); T[:3, :3] = slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, gt_t, gt_p[:, i]) for i in range(3)]
        return T @ T_BS

    P_w = load_cloud(root, Path(args.cache))

    full_cfg = yaml.safe_load(open(args.config))
    full_cfg["MSCEqF"] = {
        "enabled": True, "numClones": args.num_clones, "pixelStd": 1.0, "chi2Mult": 1.0,
        "maxSlam": args.max_slam, "maxPromoteRange": 0.0, "directProductBias": False,
        "cloneAtFrameStamp": False, "civeraBirthDepth": 0.0, "civeraBirthRelStd": 1.0,
        "gravity": [0.0, 0.0, -GRAVITY],
        "initialStd": {"attitude": 1e-2, "velocity": 1e-1, "position": 1e-4, "bias": 1e-2, "extrinsics": 1e-4},
        "processNoise": {"accelDensity": 2.0e-3, "gyroDensity": 1.6968e-4,
                         "accelRandomWalk": 3.0e-3, "gyroRandomWalk": 1.9393e-5, "transitionOrder": 0},
    }
    full_cfg["ZUPT"] = {"enabled": True, "window": 25, "gyroStdThreshold": 0.05, "accStdThreshold": 0.15, "sigmaV": 1e-2}
    run_cfg = str(out.with_suffix("")) + "_config.yaml"
    with open(run_cfg, "w") as fh:
        yaml.safe_dump(full_cfg, fh, sort_keys=False)
    cam_full = echo_li.RadTanCamera(fu, fv, cu, cv_, *D[:4])
    fcfg = echo_li.FrontendConfig.from_yaml(run_cfg)
    fcfg.set_camera(fu, fv, cu, cv_, W0, H0, list(D[:4]), "radtan")
    fcfg.max_features = args.frontend_features
    tracker = echo_li.Frontend(fcfg, W0, H0)
    vio = echo_li.VIOFilter(run_cfg, cam_full, fx=fu)
    vio.set_camera_extrinsics(np.ascontiguousarray(T_BS))

    scfg = (yaml.safe_load(open(args.sparse_config)) or {}).get("SparseVog", {}) or {}
    settings = {k: scfg[k] for k in SPARSE_KEYS if k in scfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings["max_pool_size"] = 4000
    settings["min_track_length"] = args.min_track
    for kv in args.fdf_sparse:
        k, v = kv.split("=", 1)
        cur = settings.get(k, scfg.get(k, 0.0))
        settings[k] = (v.lower() == "true") if isinstance(cur, bool) else type(cur)(v) if isinstance(cur, (int, float)) else float(v)
    _E = float(np.std(imu[:, 3]) * np.std(imu[:, 5]))
    settings["scale_sigma"] = max(0.02, 0.0072 * _E ** (-0.710)) if _E > 0 else 0.02
    fdf_settings = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
    mapper = echo_li.DenseDepthMapper(cam_full, fu, fv, cu, cv_, W0, H0, settings, tracker=fdf_settings or None,
                                      scale=args.scale, histeq="none", slant_deg=35.0, fill=True, exclude_corners=True)
    W, H = mapper.size()
    mfx, mfy, mcx, mcy = mapper.intrinsics()
    f_z = 0.5 * (mfx + mfy)
    print(f"backend={vio.backend_kind}  mapper {W}x{H} f {mfx:.1f}/{mfy:.1f}  walk {settings.get('bias_walk_var')}  scale_sigma {settings['scale_sigma']:.4f}")

    occ_common = dict(resolution=args.resolution, width_cells=args.cells, height_cells=args.cells,
                      sample_stride=args.sample_stride, min_range=0.25, max_range=args.max_range, max_eta_std=0.5,
                      band_k=args.band_k, min_obstacle_height=-args.height, max_obstacle_height=args.height)
    maps = {name: echo_li.LocalOccupancyMap.from_settings(dict(occ_common, update_mode=r["update_mode"]))
            for name, r in RULES.items()}
    # Truth: fixed increments that saturate: one hit -> occupied, one pass -> free; a surface
    # cell seen as both keeps the hit (hit >> |miss|).
    truth = echo_li.LocalOccupancyMap.from_settings(dict(occ_common, update_mode="fixed",
                                                         log_odds_hit=4.0, log_odds_miss=-0.5,
                                                         log_odds_min=-4.0, log_odds_max=8.0,
                                                         occupied_threshold=2.0, free_threshold=-0.5))
    OCC_THR, FREE_THR = 0.8, -0.8

    # --- renderers -------------------------------------------------------------------
    import cv2
    VIDEO_RULES = [("fixed_fill", "v0 fixed"), ("onesided_fill", "v2 one-sided"), ("fixed_fill+sigma", "v0 + planner margin 2 sigma(cell) (yellow)"), ("truth", "truth (Leica + Vicon)")]
    SLAB = 0.5  # top-down panels project the cells within camera z +- SLAB (m), not the floor
    # fixed world window around the Vicon track so the ego-centric grids do not jump
    gx0, gy0 = gt_p[:, 0].min() - 4.0, gt_p[:, 1].min() - 4.0
    gx1, gy1 = gt_p[:, 0].max() + 4.0, gt_p[:, 1].max() + 4.0
    PX = 3  # pixels per cell
    VW, VH = int((gx1 - gx0) / args.resolution) * PX, int((gy1 - gy0) / args.resolution) * PX
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)[:, 0, :]
    lo_, hi_ = np.log(0.8), np.log(12.0)

    def world_to_px(x, y):
        return int((x - gx0) / args.resolution * PX), VH - 1 - int((y - gy0) / args.resolution * PX)

    def topdown(m, est_xy, gt_xy, T_cam, label, sigma_overlay=False):
        """Top-down view of one map: a column is occupied (red by max log-odds) if any cell >= thr,
        else free (light blue) if any cell <= -thr, else unknown (grey). World window, north up.
        Only the slab camera z +- SLAB is projected. With `sigma_overlay`, the cells within
        2 sigma(cell) of an occupied cell (the sigma channel at that cell) are tinted yellow."""
        ox, oy, oz, res, Wg, Hg, Dg = m.grid_info()
        L = m.log_odds_grid()
        z0 = max(0, int((T_cam[2, 3] - SLAB - oz) / res)); z1 = min(Dg, int((T_cam[2, 3] + SLAB - oz) / res) + 1)
        L = L[z0:z1]
        occ = L.max(axis=0); free = L.min(axis=0)
        img = np.full((VH, VW, 3), 235, np.uint8)
        # paste the grid into the window (cells -> PX x PX blocks), y flipped
        col = np.full((Hg, Wg, 3), 235, np.uint8)
        is_free = free <= FREE_THR; is_occ = occ >= OCC_THR
        col[is_free] = (250, 225, 200)
        if sigma_overlay:
            from scipy import ndimage
            S = m.sigma_grid()[z0:z1]
            occ3 = L >= OCC_THR
            rc = np.where(occ3 & np.isfinite(S), np.minimum(np.ceil(2.0 * S / res), 10.0), 0.0)
            cov = np.zeros(L.shape, bool)
            for rb in (1, 2, 3, 4, 6, 8, 10):
                mm = occ3 & (rc >= rb)
                if mm.any():
                    cov |= ndimage.distance_transform_edt(~mm) <= rb + 1e-6
            ring = cov.any(axis=0) & ~is_occ
            col[ring] = (60, 220, 250)
        t = np.clip((occ - OCC_THR) / (4.0 - OCC_THR), 0, 1)
        col[is_occ] = np.stack([40 + 0 * t, 40 + 0 * t, 120 + 135 * t], axis=-1)[is_occ].astype(np.uint8)
        big = cv2.resize(col, (Wg * PX, Hg * PX), interpolation=cv2.INTER_NEAREST)[::-1]
        px0, py1 = world_to_px(ox, oy); px1, py0 = world_to_px(ox + Wg * res, oy + Hg * res)
        py0 += 1; py1 += 1
        sx0, sy0 = max(0, -px0), max(0, -py0)
        dx0, dy0 = max(0, px0), max(0, py0)
        dx1, dy1 = min(VW, px0 + big.shape[1]), min(VH, py0 + big.shape[0])
        if dx1 > dx0 and dy1 > dy0:
            img[dy0:dy1, dx0:dx1] = big[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        for pts, colr in ((gt_xy, (60, 160, 60)), (est_xy, (0, 0, 0))):
            if len(pts) > 1:
                cv2.polylines(img, [np.array([world_to_px(x, y) for x, y in pts], np.int32)], False, colr, 1, cv2.LINE_AA)
        cx_, cy_ = world_to_px(T_cam[0, 3], T_cam[1, 3])
        fwd = T_cam[:3, 2]
        cv2.circle(img, (cx_, cy_), 4, (0, 0, 0), -1)
        cv2.line(img, (cx_, cy_), (cx_ + int(fwd[0] * 12), cy_ - int(fwd[1] * 12)), (0, 0, 0), 2)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def depth_panel(gray, eta, cls):
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = np.clip((eta - lo_) / (hi_ - lo_), 0, 1)
        c = jet[((1.0 - t) * 255).astype(np.uint8)]
        pnl = base.copy()
        m1 = cls == 1; m2 = cls == 2
        pnl[m1] = c[m1]
        pnl[m2] = (0.55 * c[m2] + 0.45 * base[m2]).astype(np.uint8)
        return pnl

    vw = None; raw_video = None
    if args.video:
        raw_video = str(Path(args.video).with_suffix("")) + "_mp4v.mp4"
        PH = VH
        PWc = int(round(W * PH / H))
        FW = PWc + 2 * VW; FH = 2 * PH
        vw = cv2.VideoWriter(raw_video, cv2.VideoWriter_fourcc(*"mp4v"), 20.0 / args.video_every, (FW, FH))
        print(f"video {FW}x{FH} every {args.video_every} frames -> {args.video}")
    rr = None
    if args.rrd:
        import rerun as rr
        rr.init("euroc_occupancy", spawn=False)
        rr.save(args.rrd)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        print(f"rerun recording -> {args.rrd} (maps every {args.rrd_every} frames)")

    def rr_cells(name, m, color_occ, log_free):
        ox, oy, oz, res, Wg, Hg, Dg = m.grid_info()
        L = m.log_odds_grid()
        zi, yi, xi = np.nonzero(L >= OCC_THR)
        pts = np.stack([ox + (xi + 0.5) * res, oy + (yi + 0.5) * res, oz + (zi + 0.5) * res], axis=1).astype(np.float32)
        t = np.clip((L[zi, yi, xi] - OCC_THR) / (4.0 - OCC_THR), 0, 1)
        cols = (np.array(color_occ, np.float32)[None, :] * (0.5 + 0.5 * t[:, None])).astype(np.uint8)
        rr.log(f"world/occ/{name}/occupied", rr.Points3D(pts, colors=cols, radii=0.045))
        if log_free:
            zi, yi, xi = np.nonzero(L <= FREE_THR)
            keep = (zi % 2 == 0) & (yi % 2 == 0) & (xi % 2 == 0)
            zi, yi, xi = zi[keep], yi[keep], xi[keep]
            pts = np.stack([ox + (xi + 0.5) * res, oy + (yi + 0.5) * res, oz + (zi + 0.5) * res], axis=1).astype(np.float32)
            rr.log(f"world/occ/{name}/free", rr.Points3D(pts, colors=[170, 200, 240], radii=0.02))
    def rr_band(name, m):
        """The planner margin: cells within 2 sigma(cell) of an occupied cell, not occupied themselves (yellow),
        and the occupied cells' own sigma as a second entity (colour 0.05 m green → 0.5 m orange)."""
        from scipy import ndimage
        ox, oy, oz, res, Wg, Hg, Dg = m.grid_info()
        L = m.log_odds_grid(); S = m.sigma_grid()
        occ3 = L >= OCC_THR
        rc = np.where(occ3 & np.isfinite(S), np.minimum(np.ceil(2.0 * S / res), 10.0), 0.0)
        cov = np.zeros(L.shape, bool)
        for rb in (1, 2, 3, 4, 6, 8, 10):
            mm = occ3 & (rc >= rb)
            if mm.any():
                cov |= ndimage.distance_transform_edt(~mm) <= rb + 1e-6
        zi, yi, xi = np.nonzero(cov & ~occ3)
        keep = (zi % 2 == 0) & (yi % 2 == 0) & (xi % 2 == 0)
        zi, yi, xi = zi[keep], yi[keep], xi[keep]
        pts = np.stack([ox + (xi + 0.5) * res, oy + (yi + 0.5) * res, oz + (zi + 0.5) * res], axis=1).astype(np.float32)
        rr.log(f"world/occ/{name}/margin_2sigma", rr.Points3D(pts, colors=[250, 220, 60], radii=0.02))
        zi, yi, xi = np.nonzero(occ3 & np.isfinite(S))
        pts = np.stack([ox + (xi + 0.5) * res, oy + (yi + 0.5) * res, oz + (zi + 0.5) * res], axis=1).astype(np.float32)
        u = np.clip((S[zi, yi, xi] - 0.05) / 0.45, 0, 1)
        cols = np.stack([60 + 190 * u, 200 - 60 * u, 60 + 0 * u], axis=1).astype(np.uint8)
        rr.log(f"world/occ/{name}/occupied_sigma", rr.Points3D(pts, colors=cols, radii=0.045))

    est_xy, gt_xy = [], []
    est_traj, gt_traj = [], []

    k0 = args.start
    n = len(cam_t) - k0 if args.frames <= 0 else min(args.frames, len(cam_t) - k0)
    i_start = int(np.searchsorted(imu[:, 0], cam_t[k0])); i0 = max(100, i_start)
    for j in range(i0 - 100, i0):
        vio.process_imu(float(imu[j, 0]), imu[j, 1:4].tolist(), imu[j, 4:7].tolist())
    ii = i0
    scores = []
    cover = []
    T_align = None
    t0 = time.time()
    for i in range(n):
        k = k0 + i
        ts = float(cam_t[k])
        while ii < len(imu) and imu[ii, 0] <= ts:
            vio.process_imu(float(imu[ii, 0]), imu[ii, 1:4].tolist(), imu[ii, 4:7].tolist()); ii += 1
        raw = cv2.imread(str(root / "mav0/cam0/data" / cam_names[k]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        feats, _ = tracker.process(np.ascontiguousarray(raw))
        uvs = {int(fd["id"]): [float(fd["x"]), float(fd["y"])] for fd in feats}
        vio.process_vision(ts, uvs)
        pp, qq = vio.get_pose()
        T_wb = np.eye(4); T_wb[:3, :3] = Rot.from_quat(np.asarray(qq, float)).as_matrix(); T_wb[:3, 3] = np.asarray(pp, float)
        if T_align is None:
            # The VIO's world frame is its own (origin at start, yaw free); the truth lives in
            # the Vicon frame. Align the 4-DoF gauge (yaw + translation) at the first frame so
            # the maps share a frame; roll/pitch and all later drift stay as the real error.
            T_gb = gt_pose_at(ts) @ np.linalg.inv(T_BS)
            M = T_gb[:3, :3] @ T_wb[:3, :3].T
            psi = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
            Rz = Rot.from_euler("z", psi).as_matrix()
            T_align = np.eye(4); T_align[:3, :3] = Rz; T_align[:3, 3] = T_gb[:3, 3] - Rz @ T_wb[:3, 3]
            print(f"gauge aligned at frame {k}: yaw {np.degrees(psi):.1f} deg, shift {T_align[:3, 3].round(3)}")
        T_wb = T_align @ T_wb
        T_wc = T_wb @ T_BS
        eta, var, cls, st = mapper.process(np.ascontiguousarray(raw), ts, np.ascontiguousarray(T_wc), vio,
                                           [(u, v) for u, v in uvs.values()])
        cover.append((st["measured_frac"], st["filled_frac"]))
        # the ideal sensor on the GT pose
        T_gt = gt_pose_at(ts)
        gt = zbuffer(P_w, T_gt, f_z, mcx, mcy, W, H, 1)
        ok = np.isfinite(gt) & (gt > 0.3)
        eta_gt = np.where(ok, np.log(np.where(ok, gt, 1.0)), 0.0).astype(np.float32)
        var_gt = np.full((H, W), 1e-6, np.float32); cls_gt = ok.astype(np.uint8)
        poses = {"est": T_wc.tolist(), "gt": T_gt.tolist()}
        for name, r in RULES.items():
            if r["depth"] == "gt":
                maps[name].update_dense(eta_gt, var_gt, cls_gt, mfx, mfy, mcx, mcy, poses[r["pose"]])
                continue
            c = cls if r["use_fill"] else np.where(cls == 2, 0, cls).astype(np.uint8)
            maps[name].update_dense(eta, var, c, mfx, mfy, mcx, mcy, poses[r["pose"]])
        truth.update_dense(eta_gt, var_gt, cls_gt, mfx, mfy, mcx, mcy, T_gt.tolist())
        est_xy.append((T_wc[0, 3], T_wc[1, 3])); gt_xy.append((T_gt[0, 3], T_gt[1, 3]))
        if vw is not None and i % args.video_every == 0:
            gray = np.asarray(mapper.last_frame())
            left = np.vstack([cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (PWc, PH)),
                              cv2.resize(depth_panel(gray, eta, cls), (PWc, PH))])
            cv2.putText(left, f"frame {k}  t={ts - cam_t[0]:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(left, f"dense: measured {100 * st['measured_frac']:.0f}%  filled {100 * st['filled_frac']:.0f}%", (6, PH + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels = [topdown(truth if nm == "truth" else maps[nm.split("+")[0]], est_xy, gt_xy, T_gt if nm == "truth" else T_wc, lab,
                              sigma_overlay=nm.endswith("+sigma")) for nm, lab in VIDEO_RULES]
            right = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            vw.write(np.hstack([left, right]))
        if rr is not None:
            rr.set_time("frame", sequence=k)
            est_traj.append(T_wc[:3, 3].copy()); gt_traj.append(T_gt[:3, 3].copy())
            if i % 5 == 0:
                rr.log("world/trajectory", rr.LineStrips3D([np.array(est_traj, np.float32)], colors=[30, 30, 30], radii=0.01))
                rr.log("world/groundtruth", rr.LineStrips3D([np.array(gt_traj, np.float32)], colors=[60, 180, 60], radii=0.01))
                rr.log("world/camera", rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))
                rr.log("world/camera", rr.Pinhole(image_from_camera=np.array([[mfx, 0, mcx], [0, mfy, mcy], [0, 0, 1]]), width=W, height=H, image_plane_distance=0.3))
                gray = np.asarray(mapper.last_frame())
                rr.log("world/camera/image", rr.Image(gray).compress(jpeg_quality=70))
                rr.log("world/camera/dense", rr.Image(depth_panel(gray, eta, cls)[:, :, ::-1]).compress(jpeg_quality=70))
            if i % args.rrd_every == 0 or i == n - 1:
                rr_cells("v0_fixed", maps["fixed_fill"], (230, 60, 60), True)
                rr_band("v0_fixed", maps["fixed_fill"])
                rr_cells("v2_one_sided", maps["onesided_fill"], (230, 140, 40), True)
                rr_cells("v1_sigma_band", maps["sigma_fill"], (160, 60, 200), False)
                rr_cells("truth", truth, (40, 150, 40), False)
        if (i + 1) % args.score_every == 0 or i == n - 1:
            row = {"frame": k, "t": time.time() - t0}
            for name in RULES:
                row[name] = score_map(maps[name], truth, OCC_THR, FREE_THR, 2.0, -0.5)
            scores.append(row)
            s = row["sigma_fill"]
            print(f"  [{i + 1}/{n}] {row['t']:.0f} s  truth labelled {s['n_truth_labelled']} (occ {s['n_truth_occ']}, free {s['n_truth_free']})  "
                  f"cover {100 * np.mean([c[0] for c in cover[-args.score_every:]]):.0f}+{100 * np.mean([c[1] for c in cover[-args.score_every:]]):.0f}%  "
                  + "  ".join(f"{nm}: ECE {row[nm].get('ece', np.nan):.3f} false {100 * row[nm].get('false_obstacle', np.nan):.1f}% missed {100 * row[nm].get('missed_obstacle', np.nan):.1f}% unk {100 * row[nm].get('unknown_frac', np.nan):.0f}%" for nm in RULES), flush=True)

    final = scores[-1]
    print("\nFINAL (whole run), cells the truth labels; map occupied >= 0.8 log-odds, free <= -0.8")
    print("false obst = P(truth free | map occupied); missed obst = P(truth occupied | map free)")
    print(f"{'rule':18s} {'ECE':>6s} {'false obst':>10s} {'missed obst':>11s} {'occ recall':>10s} {'free recall':>11s} {'unknown':>8s} {'n occ':>8s} {'n free':>8s}")
    for nm in RULES:
        r = final[nm]
        print(f"{nm:18s} {r['ece']:6.3f} {100 * r['false_obstacle']:9.2f}% {100 * r['missed_obstacle']:10.2f}% {100 * r['occ_recall']:9.1f}% {100 * r['free_recall']:10.1f}% {100 * r['unknown_frac']:7.1f}% {r['n_map_occ']:8d} {r['n_map_free']:8d}")
    print("\nWITH POSITION TOLERANCE (cells of 0.1 m): false obstacle farther than d from any true surface | missed obstacle with no asserted obstacle within d")
    print(f"{'rule':18s} {'false d=0':>9s} {'d=1':>7s} {'d=2':>7s} {'d=3':>7s} | {'missed d=0':>10s} {'d=1':>7s} {'d=2':>7s} {'d=3':>7s}")
    for nm in RULES:
        r = final[nm]
        print(f"{nm:18s} " + " ".join(f"{100 * v:8.2f}%" for v in r["false_obstacle_tol"]) + " | " + " ".join(f"{100 * v:8.2f}%" for v in r["missed_obstacle_tol"]))
    print("\nPLANNER MARGIN: inflate asserted-occupied cells by kappa*sigma(cell) [sigma channel] vs a constant radius;")
    print("each entry = holes covered % / truly-free space consumed %   (holes = map-free & truth-occupied; cells of 0.1 m)")
    keys = ["sigma_k1", "sigma_k2", "const_1", "const_2", "const_3", "const_4"]
    print(f"{'rule':18s} {'holes':>6s} {'sig@occ':>8s} " + " ".join(f"{k:>13s}" for k in keys))
    for nm in RULES:
        r = final[nm]
        print(f"{nm:18s} {r['n_holes']:6d} {r['sigma_occ_med']:7.3f}m " + " ".join(f"{100 * r['inflate'][k][0]:5.1f}/{100 * r['inflate'][k][1]:5.1f}  " for k in keys))
    print("\nSIGMA AS A WEIGHT: hole rate of the v0 map's free cells by the channel's sigma at the cell, and by distance to the nearest asserted obstacle")
    for nm in ("fixed_fill", "fixed_meas_gtpose", "oracle_fixed"):
        r = final[nm]
        print(f"  {nm}: overall hole rate {100 * r['hole_rate_overall']:.2f} %")
        print("    by sigma (m):  " + "  ".join((f"[{a:.2f},{b:.2f}) n={n} {100 * h:.2f}%" if np.isfinite(a) else f"no-band n={n} {100 * h:.2f}%") for a, b, n, h in r["hole_rate_by_sigma"]))
        print("    by dist (cells): " + "  ".join(f"({a:g},{b:g}] n={n} {100 * h:.2f}%" for a, b, n, h in r["hole_rate_by_dist"]))
    for nm in ("fixed_fill", "sigma_fill", "onesided_fill", "oracle_fixed"):
        print(f"\nRELIABILITY ({nm}): bin | n | mean P(occ) | empirical occupied")
        for lo, hi, cnt, pm, ym in final[nm]["reliability"]:
            print(f"  [{lo:.1f},{hi:.1f}) {cnt:9d}  {pm:6.3f}  {ym:6.3f}")
    if vw is not None:
        vw.release()
        import subprocess
        rc = subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_video, "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.video])
        if rc == 0:
            Path(raw_video).unlink()
        print(f"video {args.video} (ffmpeg rc {rc})")
    np.savez_compressed(args.out, scores=np.array(scores, dtype=object), cover=np.array(cover, np.float32), rules=np.array(list(RULES)))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
