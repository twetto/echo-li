"""Score Rudolf-V stereo range priors against MidAir dense range depth.

This isolates the stereo input to EqF from the filter dynamics. For each sampled
frame, Rudolf-V tracks/detects the left image, the stereo matcher produces
per-feature range priors from the right image, and MidAir dense range depth gives
the exact range at the left feature pixel.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402


def read_right_gray(ds, k):
    p = ds.dir / "color_right" / ds.traj / f"{k:06d}.JPEG"
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    if ds.scale != 1.0:
        im = cv2.resize(im, None, fx=ds.scale, fy=ds.scale, interpolation=cv2.INTER_AREA)
    return im


def prior_rel_sigma(prior):
    rng, var_r = float(prior[0]), float(prior[1])
    if rng <= 0.0 or var_r < 0.0:
        return float("inf")
    rel = np.sqrt(var_r) / rng
    return float(rel) if np.isfinite(rel) else float("inf")


def select_eqf(all_ids, priors, max_obs):
    if max_obs <= 0:
        return set(all_ids)
    prior_ids = [fid for fid in all_ids if fid in priors]
    other_ids = [fid for fid in all_ids if fid not in priors]
    prior_ids.sort(key=lambda fid: prior_rel_sigma(priors[fid]))
    return set((prior_ids + other_ids)[:max_obs])


def depth_edge(depth, x, y, radius):
    h, w = depth.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    win = depth[y0:y1, x0:x1]
    valid = np.isfinite(win) & (win > 1.0) & (win < md.SKY)
    if valid.sum() < 4:
        return np.nan
    return float(np.std(win[valid]))


def summarize(label, rows):
    if len(rows) == 0:
        print(f"{label}: no valid rows")
        return
    a = np.asarray(rows, float)
    rel = a[:, 4]
    z = a[:, 6]
    dstd = a[:, 8]
    selected = a[:, 9] > 0.5
    for name, mask in [("all priors", np.ones(len(a), bool)), ("EqF-selected", selected)]:
        if mask.sum() == 0:
            continue
        rr = rel[mask]
        zz = z[mask]
        print(f"{label} {name:12s}: n={mask.sum():7d} "
              f"med rel={100*np.median(np.abs(rr)):6.2f}% "
              f"p90={100*np.percentile(np.abs(rr),90):6.2f}% "
              f"out>|20%|={100*np.mean(np.abs(rr)>0.2):5.1f}% "
              f"z med/p90={np.nanmedian(np.abs(zz)):5.2f}/{np.nanpercentile(np.abs(zz),90):5.2f} "
              f"edge med={np.nanmedian(dstd[mask]):5.2f}m")


def plot_rows(path, title, rows):
    if not path or len(rows) == 0:
        return
    a = np.asarray(rows, float)
    frame = a[:, 0]
    rel_abs = np.abs(a[:, 4])
    z_abs = np.abs(a[:, 6])
    dstd = a[:, 8]
    selected = a[:, 9] > 0.5
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax = ax.ravel()
    for mask, name, alpha in [(np.ones(len(a), bool), "all", 0.18), (selected, "selected", 0.5)]:
        ax[0].scatter(frame[mask], 100 * rel_abs[mask], s=4, alpha=alpha, label=name)
        ax[1].hist(100 * rel_abs[mask], bins=np.linspace(0, 100, 80), alpha=0.45, label=name)
        ax[2].scatter(dstd[mask], 100 * rel_abs[mask], s=4, alpha=alpha, label=name)
        ax[3].hist(z_abs[mask][np.isfinite(z_abs[mask])], bins=np.linspace(0, 10, 80),
                   alpha=0.45, label=name)
    ax[0].set_ylabel("|range rel error| [%]")
    ax[0].set_xlabel("frame")
    ax[1].set_xlabel("|range rel error| [%]")
    ax[1].set_ylabel("count")
    ax[2].set_xlabel("local GT range std [m]")
    ax[2].set_ylabel("|range rel error| [%]")
    ax[3].set_xlabel("|error| / reported sigma")
    ax[3].set_ylabel("count")
    for a0 in ax:
        a0.grid(True, alpha=0.25)
        a0.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved plot ->", path)


def run(args):
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    h, w = im0.shape
    f, cx, cy = md.intrinsics(w, h)
    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.max_features = args.sparse_max_features
    fcfg.set_camera(f, f, cx, cy, w, h, [])
    tracker = echo_li.Frontend(fcfg, w, h)
    stereo = echo_li.Stereo.from_pinhole(
        f, f, cx, cy, w, h, [-args.stereo_baseline_m, 0.0, 0.0], None, args.config)

    nimg = ds.n - args.start if args.frames <= 0 else min(args.frames, ds.n - args.start)
    rows = []
    for k in range(args.start, args.start + nimg):
        gray = ds.image(k)
        feats, _stats = tracker.process(gray)
        if (k - args.start) % args.stride != 0:
            continue
        right = read_right_gray(ds, k)
        priors = dict(stereo.range_priors(right, tracker, args.stereo_sigma_pixel_scale))
        all_ids = [int(fd["id"]) for fd in feats]
        selected = select_eqf(all_ids, priors, args.eqf_max_obs)
        depth = ds.depth(k)
        for fd in feats:
            fid = int(fd["id"])
            if fid not in priors:
                continue
            x = int(round(float(fd["x"])))
            y = int(round(float(fd["y"])))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            gt = float(depth[y, x])
            if not np.isfinite(gt) or gt <= 1.0 or gt >= md.SKY:
                continue
            rng, var_r = map(float, priors[fid])
            if rng <= 0.0 or var_r <= 0.0:
                continue
            err = rng - gt
            rel = err / gt
            sigma = np.sqrt(var_r)
            rows.append([k, fid, gt, rng, rel, sigma, err / sigma,
                         prior_rel_sigma(priors[fid]), depth_edge(depth, x, y, args.edge_radius),
                         1.0 if fid in selected else 0.0, float(fd["x"]), float(fd["y"])])
    label = f"{args.subset}/{args.cond}/trajectory_{args.traj:04d}"
    summarize(label, rows)
    if args.save_npz:
        arr = np.asarray(rows, float)
        Path(args.save_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, rows=arr,
                 cols=np.array("frame fid gt_range stereo_range rel_err sigma z rel_sigma "
                               "depth_std selected x y".split()))
        print("saved ->", args.save_npz)
    plot_rows(args.plot_out, label, rows)


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_midair.yaml"))
    ap.add_argument("--sparse-max-features", type=int, default=300)
    ap.add_argument("--eqf-max-obs", type=int, default=40)
    ap.add_argument("--stereo-baseline-m", type=float, default=1.0)
    ap.add_argument("--stereo-sigma-pixel-scale", type=float, default=20.0)
    ap.add_argument("--edge-radius", type=int, default=4)
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--plot-out", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
