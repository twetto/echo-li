"""Step 5 of the FlowDep-F plan: which fusion rule turns the converged patches into a dense
range frame whose per-pixel sigma is honest?

Runs FlowDep-F + Sparse3D on a GT camera pose (so the pose term is zero and only the fusion
rule is under test), and per frame fuses the CONVERGED landmarks into a per-pixel log-range
eta_p and variance under several rules, then scores every covered pixel against the GT range:
relative error and the scalar NEES (eta_p - ln gt)^2 / var_p.

Rules (per pixel, over the patches whose footprint covers it; footprint = 8 px at the patch's
pyramid level, i.e. 8 or 16 px at level 0):
  nearest        the nearest patch centre: its eta and its own variance. No overlap gain at all.
  invvar_indep   inverse-variance mean, var = 1 / sum(w): the independence assumption (eq. 29).
  invvar_min     inverse-variance mean, var = min over covering patches (the 5a prototype).
  tent           tent-kernel interpolation alpha_j, var = sum(alpha^2 var_j) + (sum alpha s_j)^2:
                 the eq. 35 structure with the conditional part independent and the shared
                 (scale) part fully correlated.
  tent_disagree  tent + the alpha-weighted disagreement of the covering patches' eta: a
                 model-free term for surface slant across the footprint and for straddling.
  nearest_disagree  nearest + the same disagreement term.
  *_slant        + a footprint term (g0 * d)^2 with d the distance (px) to the patch centre the
                 pixel takes its eta from and g0 = tan(slant-deg) / f the log-range gradient of
                 a surface at that slant: the patch's eta is its CENTRE's, and on slanted ground
                 the true range 4-8 px away differs by more than the patch sigma.

The conditional variance is the filter's own state (covariance_euclidean projected on the
ray); the shared part is reported - state (the scale_sigma term; zero on the GT-pose arm).

Usage (TartanAir window / MidAir trajectory):
  .venv/bin/python echo-li-python/tests/diagnostics/fdf_densify_study.py --dataset tartanair \
      --root $ROOT --traj P000 --config configs/diagnostics_tartanair_e2e_depth_nees.yaml \
      --start 610 --frames 110 --fdf max_mean_abs_residual=30
  .venv/bin/python echo-li-python/tests/diagnostics/fdf_densify_study.py --dataset midair \
      --root $MIDAIR --traj 12 --config configs/diagnostics_midair_e2e_depth_nees.yaml \
      --start 125 --frames 300 --fdf max_mean_abs_residual=30
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
import midair_drift as md  # noqa: E402
from tartanair_e2e_depth_nees import RT_BC, SKY, SPARSE_KEYS, TartanAir, intrinsics  # noqa: E402

RULES = ("nearest", "invvar_indep", "invvar_min", "tent", "tent_disagree", "nearest_disagree",
         "nearest_slant", "nearest_disagree_slant", "tent_disagree_slant")
SLANT_G = [0.0]   # per-pixel log-range gradient prior (1/px), set from --slant-deg and f in main()


def fuse(items, H, W):
    """items: list of (px, py, level, eta, var_cond, sigma_shared). Returns dict rule ->
    (eta_p, var_p, covered) at full working resolution."""
    S_a = np.zeros((H, W)); S_ae = np.zeros((H, W)); S_a2v = np.zeros((H, W)); S_ae2 = np.zeros((H, W)); S_as = np.zeros((H, W))
    S_w = np.zeros((H, W)); S_we = np.zeros((H, W)); min_v = np.full((H, W), np.inf)
    best_d2 = np.full((H, W), np.inf); n_eta = np.zeros((H, W)); n_var = np.zeros((H, W)); n_s = np.zeros((H, W))
    for px, py, level, eta, vc, ss in items:
        half = 4 * (2 ** level)
        x0, x1 = max(0, int(round(px)) - half), min(W, int(round(px)) + half + 1)
        y0, y1 = max(0, int(round(py)) - half), min(H, int(round(py)) + half + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) - px; ys = np.arange(y0, y1) - py
        ax = np.clip(1.0 - np.abs(xs) / half, 0.0, None); ay = np.clip(1.0 - np.abs(ys) / half, 0.0, None)
        a = ay[:, None] * ax[None, :]
        S_a[y0:y1, x0:x1] += a; S_ae[y0:y1, x0:x1] += a * eta; S_a2v[y0:y1, x0:x1] += a * a * vc
        S_ae2[y0:y1, x0:x1] += a * eta * eta; S_as[y0:y1, x0:x1] += a * ss
        w = 1.0 / vc
        S_w[y0:y1, x0:x1] += w; S_we[y0:y1, x0:x1] += w * eta
        np.minimum(min_v[y0:y1, x0:x1], vc, out=min_v[y0:y1, x0:x1])
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        better = d2 < best_d2[y0:y1, x0:x1]
        sub = best_d2[y0:y1, x0:x1]; sub[better] = d2[better]
        n_eta[y0:y1, x0:x1][better] = eta; n_var[y0:y1, x0:x1][better] = vc; n_s[y0:y1, x0:x1][better] = ss
    covered = S_a > 1e-9
    Sa = np.maximum(S_a, 1e-12); Sw = np.maximum(S_w, 1e-12)
    t_mean = S_ae / Sa
    t_var = S_a2v / (Sa * Sa) + (S_as / Sa) ** 2
    disagree = np.maximum(S_ae2 / Sa - t_mean * t_mean, 0.0)
    d_near = np.sqrt(np.where(np.isfinite(best_d2), best_d2, 0.0))
    slant = (SLANT_G[0] * d_near) ** 2
    out = {
        # The adopted rule with the shared (scale) term dropped: what eq. 35 buys, from the
        # same accumulation (the harness scores both without a second pass).
        "nearest_disagree_slant_noshared": (n_eta, n_var + disagree + slant, covered),
        "nearest_slant": (n_eta, n_var + n_s * n_s + slant, covered),
        "nearest_disagree_slant": (n_eta, n_var + n_s * n_s + disagree + slant, covered),
        "tent_disagree_slant": (t_mean, t_var + disagree + slant, covered),
        "nearest": (n_eta, n_var + n_s * n_s, covered),
        "invvar_indep": (S_we / Sw, 1.0 / Sw, covered),
        "invvar_min": (S_we / Sw, min_v, covered),
        "tent": (t_mean, t_var, covered),
        "tent_disagree": (t_mean, t_var + disagree, covered),
        "nearest_disagree": (n_eta, n_var + n_s * n_s + disagree, covered),
    }
    out["_dist"] = d_near
    return out


def fill(eta_m, var_m, cov_m, sigma_px=12.0, gate=0.15, balance=12.0, max_d=48.0, g=None):
    """The FILLED class (design note rows 5k/5l): normalised-convolution interpolation of the
    measured pixels, kept only where the evidence surrounds the pixel (centroid offset <=
    `balance` px), where the measured eta inside the kernel agrees (weighted std <= `gate`)
    and within `max_d` px of evidence; variance = nearest measured variance + (g d)^2 with
    g defaulting to 2 x the footprint gradient prior. Returns (eta_f, var_f, filled, gated)."""
    H, W = eta_m.shape
    if g is None:
        g = 2.0 * SLANT_G[0]
    w = cov_m.astype(np.float32)
    ks = int(6 * sigma_px) | 1
    num = cv2.GaussianBlur((eta_m * w).astype(np.float32), (ks, ks), sigma_px)
    den = cv2.GaussianBlur(w, (ks, ks), sigma_px)
    num2 = cv2.GaussianBlur((eta_m * eta_m * w).astype(np.float32), (ks, ks), sigma_px)
    seen = den > 1e-3
    eta_f = np.where(seen, num / np.maximum(den, 1e-6), 0.0)
    lstd = np.sqrt(np.maximum(np.where(seen, num2 / np.maximum(den, 1e-6) - eta_f * eta_f, 0.0), 0.0))
    dist, labels = cv2.distanceTransformWithLabels((~cov_m).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(cov_m)
    lab_idx = np.zeros(labels.max() + 1, np.int64)
    if len(ys):
        lab_idx[labels[ys, xs]] = ys * W + xs
    near_var = var_m.ravel()[lab_idx[labels]]
    yy, xx = np.mgrid[0:H, 0:W]
    cxw = cv2.GaussianBlur((xx * w).astype(np.float32), (ks, ks), sigma_px) / np.maximum(den, 1e-6)
    cyw = cv2.GaussianBlur((yy * w).astype(np.float32), (ks, ks), sigma_px) / np.maximum(den, 1e-6)
    off = np.sqrt((cxw - xx) ** 2 + (cyw - yy) ** 2)
    fillable = (~cov_m) & seen & (dist <= max_d)
    gated = fillable & ((lstd > gate) | (off > balance))
    filled = fillable & ~gated
    var_f = near_var + (g * dist) ** 2
    return eta_f, var_f, filled, gated


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("tartanair", "midair"), default="tartanair")
    ap.add_argument("--root", required=True)
    ap.add_argument("--traj", default="P000")
    ap.add_argument("--set", dest="subset", default="Kite_training")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", type=int, default=610)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-pool", type=int, default=4000)
    ap.add_argument("--sparse", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--fdf", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--pixel-stride", type=int, default=2, help="score every Nth pixel in x and y")
    ap.add_argument("--slant-deg", type=float, default=60.0, help="surface slant prior for the *_slant rules")
    ap.add_argument("--save-npz", default="", help="per-pixel scored samples (rule, err, nees, sigma, spread)")
    ap.add_argument("--fill", action="store_true",
                    help="also FILL the uncovered pixels (the honest DIS-style densify): eta by normalised "
                         "convolution of the measured pixels (Gaussian kernel --fill-sigma px), variance = the "
                         "nearest measured pixel's + (g d)^2 with d the distance to the nearest measured pixel and "
                         "g the footprint gradient prior; a pixel whose kernel sees measured eta disagreeing by more "
                         "than --fill-gate (log-range) stays Unknown (an opening or a depth edge). Scored per distance band.")
    ap.add_argument("--fill-sigma", type=float, default=12.0)
    ap.add_argument("--fill-gate", type=float, default=0.3, help="max weighted std of measured eta inside the kernel")
    ap.add_argument("--fill-max-d", type=float, default=48.0, help="do not fill beyond this distance (px) to evidence")
    ap.add_argument("--fill-balance", type=float, default=0.0,
                    help="interpolate, never extrapolate: a pixel is filled only if the kernel-weighted centroid "
                         "of the measured evidence lies within this many px of it (evidence on more than one "
                         "side); 0 = off. Beyond the last evidence in a direction the fill continues a surface "
                         "that may have ended (the P000 260 roof over the trees: 44 % error at 16-32 px)")
    args = ap.parse_args()

    T_nwu = np.eye(4); T_nwu[:3, :3] = np.diag([1.0, -1.0, -1.0])
    if args.dataset == "midair":
        ds = md.MidAir(args.root, args.subset, args.cond, int(args.traj), args.scale)
        H, W = ds.image(args.start).shape
        f, cx, cy = md.intrinsics(W, H); ext = md.RT_BC; sky = md.SKY; n_cam = ds.n
        stamp_of = lambda k: k / 25.0  # noqa: E731
        gt_range_working = lambda k: ds.depth(k)  # noqa: E731
    else:
        ds = TartanAir(args.root, args.traj, args.scale)
        H, W = ds.image(args.start).shape
        f, cx, cy = intrinsics(W, H)
        f_full, cx_full, cy_full = intrinsics(int(round(W / ds.scale)), int(round(H / ds.scale)))
        ext = RT_BC; sky = SKY; n_cam = ds.n_cam
        stamp_of = lambda k: float(ds.cam_t[k])  # noqa: E731
        yy, xx = np.mgrid[0:int(round(H / ds.scale)), 0:int(round(W / ds.scale))]
        ray = np.sqrt(((xx - cx_full) / f_full) ** 2 + ((yy - cy_full) / f_full) ** 2 + 1.0)
        gt_range_working = lambda k: cv2.resize(ds.depth(k) * ray, (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    pose = lambda k: T_nwu @ ds.pose(k) @ ext  # noqa: E731
    SLANT_G[0] = np.tan(np.radians(args.slant_deg)) / f

    cfg = yaml.safe_load(open(args.config)) or {}
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    settings = {k: sparse_cfg[k] for k in SPARSE_KEYS if k in sparse_cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings.setdefault("min_track_length", 1)
    settings["max_pool_size"] = args.max_pool
    for kv in args.sparse:
        k, v = kv.split("=", 1)
        settings[k] = type(sparse_cfg.get(k, 0.0))(v) if not isinstance(sparse_cfg.get(k), bool) else v.lower() == "true"
    fdf_settings = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.fdf)}
    fdf = echo_li.FlowDepF(f, f, cx, cy, W, H, fdf_settings or None)
    sparse = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)
    print(f"{args.dataset} {args.traj} frames {args.start}..{args.start + args.frames}  {W}x{H}  "
          f"FlowDepF {fdf_settings}  scale_sigma {settings.get('scale_sigma', 0.0)}")

    sl = slice(None, None, args.pixel_stride)
    fill_samples = []    # (err_rel, d_px, sigma_near_eta, local_std_eta) of FILLED pixels with valid GT
    fill_cov = []        # (measured share, filled share, gated share) per frame
    samples = {r: [] for r in RULES}   # arrays (err_rel, nees, sigma_eta, gt_spread)
    cover = {r: [] for r in RULES}
    n_patches = []
    n = min(args.frames, n_cam - args.start)
    for i in range(n):
        k = args.start + i
        gray = np.ascontiguousarray(ds.image(k))
        T_wc = pose(k)
        uv = fdf.process(gray, T_wc.tolist())
        sparse.update(stamp_of(k), uv, T_wc.tolist(), None, None)
        post = {}
        feats = sparse.get_features()
        for fid, fd in feats.items():
            est = np.asarray(fd["position"], float); cov = np.asarray(fd["covariance_euclidean"], float)
            rr = float(np.linalg.norm(est))
            if rr > 1e-6 and np.isfinite(cov).all():
                rh = est / rr; vr = float(rh @ cov @ rh)
                if vr > 0:
                    post[int(fid)] = (rr, float(np.sqrt(vr)))
        if post:
            fdf.set_posterior(post)
        lev = {o[0]: o[3] for o in fdf.last_observations()}
        items = []
        for fid, fd in feats.items():
            if sparse.query_range(int(fid))[0] <= 0.0 or fid not in uv:
                continue
            est = np.asarray(fd["position"], float); cov = np.asarray(fd["covariance_euclidean"], float)
            rr = float(np.linalg.norm(est))
            if rr <= 1e-6 or not np.isfinite(cov).all():
                continue
            rh = est / rr
            var_cond = float(rh @ cov @ rh) / (rr * rr)                     # eta units
            var_rep = float(fd["range_var_reported"]) / (rr * rr)
            s_shared = float(np.sqrt(max(var_rep - var_cond, 0.0)))
            if var_cond > 0:
                items.append((uv[fid][0], uv[fid][1], lev.get(fid, 0), np.log(rr), var_cond, s_shared))
        n_patches.append(len(items))
        if not items:
            for r in RULES:
                cover[r].append(0.0)
            continue
        gt = gt_range_working(k)
        valid_gt = np.isfinite(gt) & (gt > 0.5) & (gt < sky)
        lg = np.log(np.where(valid_gt, gt, 1.0))
        # GT ambiguity per pixel: relative spread of GT over 5x5.
        g = np.where(valid_gt, gt, np.nan)
        mn = cv2.erode(np.nan_to_num(g, nan=np.inf).astype(np.float32), np.ones((5, 5), np.uint8))
        mx = cv2.dilate(np.nan_to_num(g, nan=-np.inf).astype(np.float32), np.ones((5, 5), np.uint8))
        spread = np.where(valid_gt & np.isfinite(mn) & np.isfinite(mx), (mx - mn) / np.maximum(g, 1e-6), 9.0)
        fused = fuse(items, H, W)
        for r in RULES:
            eta_p, var_p, covered = fused[r]
            m = (covered & valid_gt)[sl, sl]
            cover[r].append(float(covered.mean()))
            e = eta_p[sl, sl][m] - lg[sl, sl][m]
            v = np.maximum(var_p[sl, sl][m], 1e-12)
            samples[r].append(np.stack([np.exp(e) - 1.0, e * e / v, np.sqrt(v), spread[sl, sl][m], fused["_dist"][sl, sl][m]], axis=1).astype(np.float32))
        if args.fill:
            eta_m, var_m, cov_m = fused["nearest_disagree_slant"]
            w = cov_m.astype(np.float32)
            ks = int(6 * args.fill_sigma) | 1
            num = cv2.GaussianBlur((eta_m * w).astype(np.float32), (ks, ks), args.fill_sigma)
            den = cv2.GaussianBlur(w, (ks, ks), args.fill_sigma)
            num2 = cv2.GaussianBlur((eta_m * eta_m * w).astype(np.float32), (ks, ks), args.fill_sigma)
            seen = den > 1e-3
            eta_f = np.where(seen, num / np.maximum(den, 1e-6), 0.0)
            lstd = np.sqrt(np.maximum(np.where(seen, num2 / np.maximum(den, 1e-6) - eta_f * eta_f, 0.0), 0.0))
            # Distance to the nearest measured pixel and that pixel's variance.
            dist, labels = cv2.distanceTransformWithLabels((~cov_m).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            lab_idx = np.zeros(labels.max() + 1, np.int64)
            ys, xs = np.nonzero(cov_m)
            lab_idx[labels[ys, xs]] = ys * W + xs
            near_var = var_m.ravel()[lab_idx[labels]]
            fillable = (~cov_m) & seen & (dist <= args.fill_max_d)
            gated = fillable & (lstd > args.fill_gate)
            yy, xx = np.mgrid[0:H, 0:W]
            cxw = cv2.GaussianBlur((xx * w).astype(np.float32), (ks, ks), args.fill_sigma) / np.maximum(den, 1e-6)
            cyw = cv2.GaussianBlur((yy * w).astype(np.float32), (ks, ks), args.fill_sigma) / np.maximum(den, 1e-6)
            off = np.sqrt((cxw - xx) ** 2 + (cyw - yy) ** 2)   # evidence-centroid offset: 0 = surrounded, large = one-sided
            if args.fill_balance > 0:
                gated |= fillable & (off > args.fill_balance)
            filled = fillable & ~gated
            fill_cov.append((float(cov_m.mean()), float(filled.mean()), float(gated.mean())))
            m = (filled & valid_gt)[sl, sl]
            e = eta_f[sl, sl][m] - lg[sl, sl][m]
            fill_samples.append(np.stack([np.exp(e) - 1.0, dist[sl, sl][m], np.sqrt(np.maximum(near_var[sl, sl][m], 1e-12)), lstd[sl, sl][m], off[sl, sl][m]], axis=1).astype(np.float32))
        if (i + 1) % 25 == 0:
            print(f"  frame {k}: {len(items)} converged patches, coverage tent {100 * cover['tent'][-1]:.1f}%", flush=True)

    print(f"\nconverged patches per frame: median {np.median(n_patches):.0f}  p10 {np.percentile(n_patches, 10):.0f}  p90 {np.percentile(n_patches, 90):.0f}")
    print("\nPER-PIXEL (all covered pixels with valid GT; GT-unambiguous = 5x5 GT spread <= 20 %)")
    print(f"{'rule':22s} {'cov%':>5s} | {'n':>8s} {'err med%':>8s} {'p90%':>6s} {'gross%':>6s} | {'NEES med':>8s} {'mean':>7s} {'>3.84%':>6s} {'>6.63%':>6s} | unamb: {'n':>8s} {'err%':>5s} {'NEESmed':>7s} {'>3.84%':>6s} {'>6.63%':>6s}")
    for r in RULES:
        A = np.concatenate(samples[r]) if samples[r] else np.zeros((0, 5), np.float32)
        if not len(A):
            print(f"{r:22s} no samples"); continue
        err, nees, sig, sp, dist = A.T; ua = sp <= 0.2
        print(f"{r:22s} {100 * np.mean(cover[r]):5.1f} | {len(A):8d} {100 * np.median(np.abs(err)):8.1f} {100 * np.percentile(np.abs(err), 90):6.1f} {100 * np.mean(np.abs(err) > 0.2):6.1f} | "
              f"{np.median(nees):8.3f} {np.mean(nees):7.1f} {100 * np.mean(nees > 3.84):6.1f} {100 * np.mean(nees > 6.63):6.1f} | "
              f"{ua.sum():8d} {100 * np.median(np.abs(err[ua])):5.1f} {np.median(nees[ua]):7.3f} {100 * np.mean(nees[ua] > 3.84):6.1f} {100 * np.mean(nees[ua] > 6.63):6.1f}")
    print("\nBY DISTANCE to the patch centre the pixel takes its eta from (nearest rule): the footprint error")
    A = np.concatenate(samples["nearest"]); err, nees, sig, sp, dist = A.T
    for lo_d, hi_d in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 100)):
        m = (dist >= lo_d) & (dist < hi_d) & (sp <= 0.2)
        if m.sum() < 100: continue
        print(f"  d [{lo_d},{hi_d}) px: n {m.sum():8d}  err med {100*np.median(np.abs(err[m])):4.1f}%  p90 {100*np.percentile(np.abs(err[m]),90):4.1f}%  NEES med {np.median(nees[m]):.2f}  >3.84 {100*np.mean(nees[m]>3.84):4.1f}%")
    print("\nSIGMA TIERS (reported per-pixel sigma_eta <= tier; coverage = share of all pixels per frame, mean over frames)")
    for r in ("nearest", "nearest_disagree", "nearest_slant", "nearest_disagree_slant", "tent_disagree_slant"):
        A = np.concatenate(samples[r]); err, nees, sig, sp, dist = A.T
        n_all = len(A)
        line = f"{r:22s}"
        for tier in (0.02, 0.05, 0.10, 0.20):
            m = sig <= tier
            if m.sum() < 100:
                line += f" | <={100*tier:.0f}%: -"; continue
            cov_t = np.mean(cover[r]) * m.mean()
            line += f" | <={100*tier:.0f}%: cov {100*cov_t:4.1f}% err {100*np.median(np.abs(err[m])):4.1f}% NEES {np.median(nees[m]):.2f} >95 {100*np.mean(nees[m] > 3.84):4.1f}%"
        print(line)
    if args.fill and fill_samples:
        F = np.concatenate(fill_samples); err, d, sn, ls, off_s = F.T
        fc = np.array(fill_cov)
        print(f"\nFILLED pixels (normalised convolution, kernel sigma {args.fill_sigma:.0f} px, gate {args.fill_gate}, max d {args.fill_max_d:.0f}): "
              f"coverage measured {100 * fc[:, 0].mean():.1f}% + filled {100 * fc[:, 1].mean():.1f}% (gated out {100 * fc[:, 2].mean():.1f}%)")
        g0 = SLANT_G[0]
        print(f"  by distance to evidence (px): n | err med / p90 / gross | NEES with sigma^2 = near + (g d)^2 for g = g0 (35 deg), 2 g0, 4 g0: median, >3.84%")
        for lo_d, hi_d in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1000)):
            mm = (d >= lo_d) & (d < hi_d)
            if mm.sum() < 100: continue
            ee = np.log1p(err[mm]); line = f"  [{lo_d:2d},{hi_d:4d}) {mm.sum():8d} | {100 * np.median(np.abs(err[mm])):4.1f} / {100 * np.percentile(np.abs(err[mm]), 90):4.1f} / {100 * np.mean(np.abs(err[mm]) > 0.2):4.1f} |"
            for mult in (1, 2, 4):
                v = sn[mm] ** 2 + (mult * g0 * d[mm]) ** 2; nn = ee * ee / v
                line += f"  {np.median(nn):5.2f} {100 * np.mean(nn > 3.84):5.1f}%"
            print(line)
    if args.save_npz:
        np.savez_compressed(args.save_npz, **{r: np.concatenate(samples[r]) for r in RULES if samples[r]},
                            cover=np.array([cover[r] for r in RULES]), rules=np.array(RULES), n_patches=np.array(n_patches),
                            fill=np.concatenate(fill_samples) if fill_samples else np.zeros((0, 5), np.float32),
                            fill_cov=np.array(fill_cov) if fill_cov else np.zeros((0, 3)))
        print(f"saved {args.save_npz}")


if __name__ == "__main__":
    main()
