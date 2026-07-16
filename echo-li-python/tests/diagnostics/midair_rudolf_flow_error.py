"""Direct Rudolf-V measurement-error model on Mid-Air exact GT.

This does not tune Sparse3D. It measures the frontend observation model that
Sparse3D should consume:

    e_k = u_rudolf,k - project_exact(X_birth, T_k)

Each Rudolf-V feature ID is anchored at birth to the exact Mid-Air range depth
and pose. Later observations of the same ID are compared to the exact
reprojection of that fixed 3D point. The output splits the residual by age,
occlusion, border, depth-edge proximity, and survival, then prints robust
YAML-ready measurement-model numbers.

Run from repo root after installing the current echo-li-python binding:

  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_rudolf_flow_error.py \
      --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --set VO_test --cond sunny --traj 0 --frames 300 --scale 0.5 \
      --config configs/eqvio_euroc_rho.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import echo_li

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402


R = md.R


def camera_point_fast(Xw, T_bw):
    pb = T_bw[:3, :3] @ Xw + T_bw[:3, 3]
    return md.RT_BC[:3, :3].T @ pb


def project_world_fast(Xw, T_bw, f, cx, cy):
    pc = camera_point_fast(Xw, T_bw)
    if pc[2] <= 1e-6:
        return None, None, None
    u = f * pc[0] / pc[2] + cx
    v = f * pc[1] / pc[2] + cy
    return np.array([u, v]), float(np.linalg.norm(pc)), float(pc[2])


def robust_sigma_axis(x):
    """Normal-equivalent robust per-axis scale from MAD."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def p68_axis(ex, ey):
    """Normal-equivalent per-axis core scale from radial 68% quantile."""
    r = np.hypot(ex, ey)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return np.percentile(r, 68.27) / 1.51  # Rayleigh q68 for per-axis sigma.


def sigma_summary(ex, ey):
    ex = np.asarray(ex, float)
    ey = np.asarray(ey, float)
    return {
        "mad_x": robust_sigma_axis(ex),
        "mad_y": robust_sigma_axis(ey),
        "mad_iso": np.nanmedian([robust_sigma_axis(ex), robust_sigma_axis(ey)]),
        "p68_iso": p68_axis(ex, ey),
        "rms_axis": np.sqrt(np.nanmean(ex * ex + ey * ey) / 2.0),
    }


def print_stats(label, a):
    if len(a) == 0:
        return
    e = np.hypot(a[:, 2], a[:, 3])
    ex, ey = a[:, 2], a[:, 3]
    sig = sigma_summary(ex, ey)
    print(f"{label:>18s}: n={len(a):7d}  med {np.median(e):7.3f}px  "
          f"p68 {np.percentile(e, 68.27):7.3f}  p90 {np.percentile(e, 90):8.3f}  "
          f"p99 {np.percentile(e, 99):9.3f}  >3px {np.mean(e > 3)*100:6.2f}%  "
          f"sigma_p68 {sig['p68_iso']:6.3f}px")


def rayleigh_quantile(sigma, prob):
    return sigma * np.sqrt(-2.0 * np.log(max(1.0 - prob, 1e-12)))


def age_label(lo, hi):
    return f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"


def make_frontend(config, f, cx, cy, w, h):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.set_camera(f, f, cx, cy, w, h, [])
    return echo_li.Frontend(fcfg, w, h), fcfg


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--border", type=int, default=24,
                    help="pixels from image edge considered border-risk")
    ap.add_argument("--depth-edge-quantile", type=float, default=0.75,
                    help="depth std quantile used to define high-depth-edge rows")
    ap.add_argument("--core-max-px", type=float, default=3.0,
                    help="residual threshold for the inlier-core sigma report")
    ap.add_argument("--save-npz", default="",
                    help="optional path for raw rows; empty means print only")
    ap.add_argument("--plot-out", default="",
                    help="optional PNG comparing empirical age curves to white+bias model")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    sparse_cfg = (yaml.safe_load(open(args.config)).get("SparseVog", {})
                  if args.config else {})
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.1f}")
    print(f"frontend: {fcfg}")
    print(f"target SparseVog parametrization: {sparse_cfg.get('parametrization', '<missing>')}")

    Xw = {}
    born = {}
    last_seen = {}
    seen = set()
    rows = []
    stat_rows = []
    t0 = time.time()
    last = min(args.start + args.frames, ds.n)

    for i in range(args.start, last):
        img = ds.image(i)
        depth = ds.depth(i)
        T_wb = ds.pose(i)
        T_bw = np.linalg.inv(T_wb)
        feats, stats = tracker.process(img)
        cur = {int(fd["id"]): np.array([float(fd["x"]), float(fd["y"])])
               for fd in feats}
        stat_rows.append((i, int(stats.get("tracked", 0)), int(stats.get("lost", 0)),
                          int(stats.get("rejected", 0)), int(stats.get("new_detections", 0)),
                          int(stats.get("total", 0))))

        for fid, p in cur.items():
            if fid not in born:
                gx, gy = int(round(p[0])), int(round(p[1]))
                if not (R + 2 <= gx < W - R - 2 and R + 2 <= gy < H - R - 2):
                    continue
                d_rng = float(depth[gy, gx])
                if not (1.0 < d_rng < md.SKY):
                    continue
                Xw[fid] = md.backproject_world((float(p[0]), float(p[1])), d_rng,
                                               T_wb, f, cx, cy)
                born[fid] = i
                seen.add(fid)
            gp, rng, _z = project_world_fast(Xw[fid], T_bw, f, cx, cy)
            if gp is None:
                continue
            inb = R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
            if not inb:
                continue
            gx, gy = int(round(gp[0])), int(round(gp[1]))
            dwin = depth[max(0, gy - R):gy + R + 1, max(0, gx - R):gx + R + 1]
            dwin = dwin[dwin < md.SKY]
            dstd = float(np.std(dwin)) if dwin.size > 4 else np.nan
            map_rng = float(depth[gy, gx])
            occluded = int(map_rng < rng - max(0.05 * rng, 0.5))
            border = int(gp[0] < args.border or gp[0] >= W - args.border
                         or gp[1] < args.border or gp[1] >= H - args.border
                         or p[0] < args.border or p[0] >= W - args.border
                         or p[1] < args.border or p[1] >= H - args.border)
            age = i - born[fid]
            err = p - gp
            radius = float(np.hypot(gp[0] - cx, gp[1] - cy))
            rows.append((age, fid, float(err[0]), float(err[1]), dstd, occluded,
                         border, radius, rng, ds.omega_mag(i), i))
            last_seen[fid] = i

        # Drop frontend-dead tracks from the active truth table. Their completed
        # observations stay in rows; if Rudolf reuses an ID later, treat it as
        # a fresh birth.
        dead = [fid for fid in list(born) if fid not in cur]
        for fid in dead:
            Xw.pop(fid, None)
            born.pop(fid, None)
            last_seen.pop(fid, None)

        if (i - args.start) % 50 == 0:
            fps = (i - args.start + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i - args.start:4d}/{last - args.start}] live={len(cur):4d} "
                  f"obs={len(rows):7d} {fps:5.1f} fps")

    A = np.array(rows, float)
    cols = "age fid ex ey depth_std occluded border radius range omega frame".split()
    if args.save_npz:
        np.savez(args.save_npz, rows=A, cols=np.array(cols),
                 frontend_stats=np.array(stat_rows, int))
        print(f"saved {args.save_npz}")
    if len(A) == 0:
        print("No observations scored.")
        return

    age, _fid, ex, ey, dstd, occ, border, radius, rng, omega, frame = A.T
    err = np.hypot(ex, ey)
    finite_edge = np.isfinite(dstd)
    edge_thr = np.percentile(dstd[finite_edge], args.depth_edge_quantile * 100.0) \
        if finite_edge.any() else np.inf
    visible = occ == 0
    interior = border == 0
    low_edge = (~finite_edge) | (dstd < edge_thr)
    core_domain = visible & interior
    clean_domain = core_domain & low_edge
    nonbirth = age >= 1
    inlier_core = core_domain & nonbirth & (err <= args.core_max_px)
    clean_inlier = clean_domain & nonbirth & (err <= args.core_max_px)

    print(f"\n=== Mid-Air Rudolf-V exact-flow error "
          f"({len(A)} obs, {len(seen)} born tracks) ===")
    print(f"conditions: visible {np.mean(visible)*100:5.1f}%  "
          f"interior {np.mean(interior)*100:5.1f}%  "
          f"low-depth-edge<{edge_thr:.3g}m {np.mean(low_edge)*100:5.1f}%")
    print_stats("all", A)
    print_stats("visible+interior", A[core_domain])
    print_stats("clean domain", A[clean_domain])
    print_stats(f"core <={args.core_max_px:g}px age>=1", A[inlier_core])
    print_stats("clean core age>=1", A[clean_inlier])
    print_stats("occluded", A[occ == 1])
    print_stats("border", A[border == 1])
    print_stats("high depth-edge", A[finite_edge & (dstd >= edge_thr)])

    print("\n--- by age (visible + interior) ---")
    age_bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40),
                (40, 80), (80, 160), (160, 1e9)]
    curve_rows = []
    med_age = []
    sig_age = []
    for lo, hi in age_bins:
        m = core_domain & (age >= lo) & (age < hi)
        if m.sum() < 20:
            continue
        sig = sigma_summary(ex[m & (err <= args.core_max_px)],
                            ey[m & (err <= args.core_max_px)])
        print(f"  age {age_label(lo, hi):>8s}: n={int(m.sum()):6d}  "
              f"med {np.median(err[m]):7.3f}px  p68 {np.percentile(err[m], 68.27):7.3f}  "
              f"p90 {np.percentile(err[m], 90):8.3f}  >3px {np.mean(err[m] > 3)*100:6.2f}%  "
              f"core_sigma {sig['p68_iso']:6.3f}px")
        curve_rows.append((lo, hi, np.median(age[m]), m.sum(), np.median(err[m]),
                           np.percentile(err[m], 68.27), np.percentile(err[m], 90),
                           np.mean(err[m] > args.core_max_px), sig["p68_iso"]))
        if lo >= 1:
            med_age.append(np.median(age[m]))
            sig_age.append(sig["p68_iso"])

    # Evidence-derived age term: fit the robust inlier-core scale, not the RMS
    # tail. This is a covariance model for gated observations, not a claim that
    # the bias is Gaussian.
    sigma0 = sigma_summary(ex[clean_inlier], ey[clean_inlier])["p68_iso"]
    r_age = np.nan
    if len(med_age) >= 3 and np.isfinite(sigma0):
        x = np.asarray(med_age, float)
        y = np.asarray(sig_age, float)
        ok = np.isfinite(y) & (y >= 0)
        if ok.sum() >= 3:
            target = np.maximum(y[ok] ** 2 - sigma0 ** 2, 0.0)
            # target ~= (r * age)^2. Fit r^2 through the origin in variance
            # space, weighted by age^2: argmin_a ||a age^2 - target||^2.
            x2 = x[ok] ** 2
            r2 = np.sum(x2 * target) / max(np.sum(x2 * x2), 1e-12)
            r_age = np.sqrt(max(r2, 0.0))

    sig_core = sigma_summary(ex[inlier_core], ey[inlier_core])
    sig_clean = sigma_summary(ex[clean_inlier], ey[clean_inlier])
    print("\n--- YAML-ready measurement model ingredients ---")
    print("MidAirRudolfFlowModel:")
    print(f"  dataset: {args.subset}/{args.cond}/trajectory_{args.traj:04d}")
    print(f"  frames: {args.start}..{last - 1}")
    print(f"  scale: {args.scale}")
    print("  target_chart: bearing_invdepth_additive3d")
    print("  residual_domain: pixel")
    print(f"  gate_visible: true")
    print(f"  gate_border_px: {args.border}")
    print(f"  gate_depth_std_max_m: {edge_thr:.6g}")
    print(f"  inlier_core_max_px: {args.core_max_px}")
    print(f"  sigma_pixel_core_px: {sig_core['p68_iso']:.6g}")
    print(f"  sigma_pixel_clean_core_px: {sig_clean['p68_iso']:.6g}")
    print(f"  sigma_pixel_mad_iso_px: {sig_clean['mad_iso']:.6g}")
    if np.isfinite(r_age):
        print(f"  robust_age_scale_rate_px_per_frame: {r_age:.6g}")
        print("  R_age_model_px2: sigma_pixel_clean_core_px^2 + "
              "robust_age_scale_rate_px_per_frame^2 * age^2")
    print(f"  outlier_rate_visible_interior_gt_3px: {np.mean(err[core_domain] > 3)*100:.6g}")
    print(f"  outlier_rate_clean_domain_gt_3px: {np.mean(err[clean_domain] > 3)*100:.6g}")
    print("  note: sigma/r_age are derived from gated Rudolf-V vs exact MidAir reprojection; "
          "RMS tail is reported separately and should be modeled by gates/outliers.")

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        C = np.array(curve_rows, float)
        x = C[:, 2]
        n = C[:, 3]
        med = C[:, 4]
        p68 = C[:, 5]
        p90 = C[:, 6]
        out = C[:, 7] * 100.0
        sig_emp = C[:, 8]

        sigma_age = np.sqrt(sig_clean["p68_iso"] ** 2 + (r_age * x) ** 2)
        med_pred = rayleigh_quantile(sigma_age, 0.5)
        p68_pred = rayleigh_quantile(sigma_age, 0.6827)
        p90_pred = rayleigh_quantile(sigma_age, 0.9)
        out_pred = np.exp(-args.core_max_px ** 2 / (2.0 * sigma_age ** 2)) * 100.0

        mqq = clean_domain & nonbirth
        sigma_obs = np.sqrt(sig_clean["p68_iso"] ** 2 + (r_age * age[mqq]) ** 2)
        zrad = err[mqq] / np.maximum(sigma_obs, 1e-9)
        zrad = np.sort(zrad[np.isfinite(zrad)])
        if len(zrad):
            probs = (np.arange(len(zrad)) + 0.5) / len(zrad)
            zray = np.sqrt(-2.0 * np.log(1.0 - probs))
        else:
            zray = np.array([])

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax = ax.ravel()
        ax[0].plot(x, med, "o-", label="emp median")
        ax[0].plot(x, p68, "o-", label="emp p68")
        ax[0].plot(x, p90, "o-", label="emp p90")
        ax[0].plot(x, med_pred, "--", label="model median")
        ax[0].plot(x, p68_pred, "--", label="model p68")
        ax[0].plot(x, p90_pred, "--", label="model p90")
        ax[0].set_xlabel("track age [frames]")
        ax[0].set_ylabel("radial pixel residual [px]")
        ax[0].set_title("Residual quantiles vs white+age model")
        ax[0].legend(fontsize=8)

        ax[1].plot(x, sig_emp, "o-", label="emp core sigma")
        ax[1].plot(x, sigma_age, "--", label="model sigma")
        ax[1].set_xlabel("track age [frames]")
        ax[1].set_ylabel("per-axis sigma [px]")
        ax[1].set_title("Core scale fit")
        ax[1].legend(fontsize=8)

        ax[2].plot(x, out, "o-", label=f"emp >{args.core_max_px:g}px")
        ax[2].plot(x, out_pred, "--", label="model Gaussian tail")
        ax[2].set_xlabel("track age [frames]")
        ax[2].set_ylabel("outlier rate [%]")
        ax[2].set_title("Tail mismatch check")
        ax[2].legend(fontsize=8)

        if len(zrad):
            qn = min(len(zrad), 4000)
            idx = np.linspace(0, len(zrad) - 1, qn).astype(int)
            ax[3].plot(zray[idx], zrad[idx], ".", ms=2, alpha=0.5)
            lim = min(max(np.nanpercentile(zrad, 99), 3.0), 10.0)
            ax[3].plot([0, lim], [0, lim], "k--", lw=1)
            ax[3].set_xlim(0, lim)
            ax[3].set_ylim(0, lim)
        ax[3].set_xlabel("Rayleigh model quantile")
        ax[3].set_ylabel("normalized empirical residual")
        ax[3].set_title("Gated clean-domain Q-Q")

        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle(
            f"MidAir Rudolf-V flow model: sigma={sig_clean['p68_iso']:.3f}px, "
            f"r={r_age:.4f}px/frame, n={len(A)}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.plot_out, dpi=140)
        print(f"saved model-check plot: {args.plot_out}")


if __name__ == "__main__":
    main()
