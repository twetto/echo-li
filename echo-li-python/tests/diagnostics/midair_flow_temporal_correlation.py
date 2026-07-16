"""Temporal correlation of two-frame Rudolf-V flow residuals on MidAir exact GT.

This is the follow-up to `midair_flow_texture_likelihood.py`. It keeps the
two-frame exact-GT residuals as per-track time series and estimates how much
repeated KLT measurements are correlated in time.

Residual definition:

    e_k = u_rudolf,k - project_exact(X_{k-1}, T_k)

where X_{k-1} is the exact MidAir 3D point under the previous tracked pixel. This
is a one-step optical-flow residual, not cumulative first-frame drift.

Run from repo root:

  echo-li-python/venv/bin/python \
    echo-li-python/tests/diagnostics/midair_flow_temporal_correlation.py \
    --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
    --set VO_test --cond sunny --traj 0 --frames 300 --scale 0.5 \
    --config configs/diagnostics_midair_sparse3d.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import echo_li

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
from midair_flow_texture_likelihood import (  # noqa: E402
    eig_sym2,
    project_world_fast,
    robust_sigma,
    structure_tensor_fields,
)


def make_frontend(config, f, cx, cy, w, h):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.set_camera(f, f, cx, cy, w, h, [])
    return echo_li.Frontend(fcfg, w, h), fcfg


def split_core_runs(track_rows, core_px):
    runs = []
    cur = []
    for row in track_rows:
        if row["err_norm"] <= core_px:
            cur.append(row)
        else:
            if len(cur):
                runs.append(cur)
            cur = []
    if len(cur):
        runs.append(cur)
    return runs


def scalar_series(tracks, key, min_len):
    return [
        np.asarray([row[key] for row in rows], float)
        for rows in tracks
        if len(rows) >= min_len
    ]


def lag_corr(series, max_lag, within_demean=False):
    """Pooled lag correlation over variable-length track series.

    `within_demean=False` keeps persistent per-track offsets in the statistic,
    which is the relevant failure mode for repeated-measurement over-counting.
    `within_demean=True` removes each track's mean and measures only correlated
    jitter around that track's bias.
    """
    out = []
    for lag in range(1, max_lag + 1):
        xs = []
        ys = []
        for s in series:
            if len(s) <= lag:
                continue
            v = s.copy()
            if within_demean:
                v = v - np.mean(v)
            xs.append(v[:-lag])
            ys.append(v[lag:])
        if not xs:
            out.append(np.nan)
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20:
            out.append(np.nan)
            continue
        x = x[m]
        y = y[m]
        sx = np.std(x)
        sy = np.std(y)
        if sx <= 0 or sy <= 0:
            out.append(np.nan)
        else:
            out.append(float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy)))
    return np.asarray(out, float)


def integrated_time(rho, n=None):
    """Initial-positive-sequence integrated autocorrelation time."""
    tau = 1.0
    used = 0
    for i, r in enumerate(rho, start=1):
        if not np.isfinite(r) or r <= 0:
            break
        weight = 1.0 if n is None else 1.0 - i / n
        if weight <= 0.0:
            break
        tau += 2.0 * weight * r
        used = i
    return tau, used


def print_acf_report(label, tracks, keys, max_lag, min_len):
    print(f"\n=== {label} ({len(tracks)} contiguous series, min_len={min_len}) ===")
    lens = np.asarray([len(t) for t in tracks], int)
    if len(lens) == 0:
        print("no usable series")
        return {}
    print(f"series length median/p90/max: {np.median(lens):.0f} / "
          f"{np.percentile(lens, 90):.0f} / {np.max(lens):.0f}")

    result = {}
    for key in keys:
        ser = scalar_series(tracks, key, min_len)
        if not ser:
            continue
        vals = np.concatenate(ser)
        rho_raw = lag_corr(ser, max_lag, within_demean=False)
        rho_dm = lag_corr(ser, max_lag, within_demean=True)
        tau_raw, used_raw = integrated_time(rho_raw)
        tau_dm, used_dm = integrated_time(rho_dm)
        result[key] = (rho_raw, rho_dm)
        first = " ".join(
            f"{v:+.2f}" if np.isfinite(v) else "nan"
            for v in rho_raw[:min(8, len(rho_raw))]
        )
        first_dm = " ".join(
            f"{v:+.2f}" if np.isfinite(v) else "nan"
            for v in rho_dm[:min(8, len(rho_dm))]
        )
        print(f"\n{key}: samples={len(vals)} sigma_mad={robust_sigma(vals):.4f}px")
        print(f"  raw pooled rho lag 1..{min(8, len(rho_raw))}:      {first}")
        print(f"  demeaned rho lag 1..{min(8, len(rho_dm))}:         {first_dm}")
        print(f"  tau_int raw {tau_raw:.2f} (through lag {used_raw}), "
              f"N_eff/N ~= {1.0 / tau_raw:.3f}")
        print(f"  tau_int demeaned {tau_dm:.2f} (through lag {used_dm}), "
              f"N_eff/N ~= {1.0 / tau_dm:.3f}")
        for n in (10, 20, 40, 80):
            tau_n, used_n = integrated_time(rho_raw, n=n)
            print(f"    finite N={n:2d}: tau={tau_n:5.2f}, "
                  f"N_eff={n / tau_n:5.1f} (used lag {used_n})")
    return result


def direction_persistence(tracks, min_len):
    vals = []
    for rows in tracks:
        if len(rows) < min_len:
            continue
        e = np.asarray([[r["ex"], r["ey"]] for r in rows], float)
        n = np.linalg.norm(e, axis=1)
        m = n > 1e-9
        if m.sum() < min_len:
            continue
        u = e[m] / n[m, None]
        if len(u) > 1:
            vals.append(float(np.mean(np.sum(u[:-1] * u[1:], axis=1))))
    return np.asarray(vals, float)


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
    ap.add_argument("--window", type=int, default=0,
                    help="structure tensor radius; 0 uses FrontendConfig.klt_window")
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--core-px", type=float, default=3.0)
    ap.add_argument("--max-lag", type=int, default=60)
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--plot-out", default="")
    args = ap.parse_args()

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = md.intrinsics(W, H)
    tracker, fcfg = make_frontend(args.config, f, cx, cy, W, H)
    r = int(args.window or getattr(fcfg, "klt_window", 7))
    print(f"MidAir {args.subset}/{args.cond}/{ds.traj} work {W}x{H} f={f:.1f}")
    print(f"frontend: {fcfg}")
    print(f"structure tensor radius: {r}")

    prev = None
    tracks = {}
    rows_flat = []
    t0 = time.time()
    last = min(args.start + args.frames, ds.n)

    for i in range(args.start, last):
        img = ds.image(i)
        depth = ds.depth(i)
        T_wb = ds.pose(i)
        T_bw = np.linalg.inv(T_wb)
        feats, _stats = tracker.process(img)
        cur = {
            int(fd["id"]): np.array([float(fd["x"]), float(fd["y"])])
            for fd in feats
        }
        prep = tracker.preprocessed_image()
        prep = np.asarray(prep) if prep is not None else cv2.equalizeHist(img)

        if prev is not None:
            prev_cur, prev_depth, prev_T_wb, prev_prep = prev
            sxx, sxy, syy = structure_tensor_fields(prev_prep, r)
            for fid, p0 in prev_cur.items():
                p1 = cur.get(fid)
                if p1 is None:
                    continue
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                if not (args.border <= p0[0] < W - args.border
                        and args.border <= p0[1] < H - args.border
                        and args.border <= p1[0] < W - args.border
                        and args.border <= p1[1] < H - args.border):
                    continue
                d0 = float(prev_depth[y0, x0])
                if not (1.0 < d0 < md.SKY):
                    continue
                Xw = md.backproject_world((float(p0[0]), float(p0[1])), d0,
                                          prev_T_wb, f, cx, cy)
                gp, rng, _z = project_world_fast(Xw, T_bw, f, cx, cy)
                if gp is None or not (args.border <= gp[0] < W - args.border
                                      and args.border <= gp[1] < H - args.border):
                    continue
                err = p1 - gp
                lmin, lmax, ux, uy, vx, vy = eig_sym2(
                    float(sxx[y0, x0]), float(sxy[y0, x0]), float(syy[y0, x0]))
                row = {
                    "fid": fid,
                    "frame": i,
                    "ex": float(err[0]),
                    "ey": float(err[1]),
                    "e_weak": float(err[0] * ux + err[1] * uy),
                    "e_strong": float(err[0] * vx + err[1] * vy),
                    "err_norm": float(np.hypot(err[0], err[1])),
                    "lambda_min": float(lmin),
                    "lambda_max": float(lmax),
                    "condition": float(lmax / max(lmin, 1e-9)),
                    "range": float(rng),
                    "flow_mag": float(np.hypot(*(p1 - p0))),
                }
                tracks.setdefault(fid, []).append(row)
                rows_flat.append(row)

        prev = (cur, depth, T_wb, prep)
        if (i - args.start) % 50 == 0:
            fps = (i - args.start + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i - args.start:4d}/{last - args.start}] tracks={len(cur):4d} "
                  f"pairs={len(rows_flat):7d} {fps:5.1f} fps")

    all_tracks = [rows for rows in tracks.values() if len(rows) >= args.min_len]
    core_tracks = []
    for rows in tracks.values():
        core_tracks.extend(split_core_runs(rows, args.core_px))
    core_tracks = [rows for rows in core_tracks if len(rows) >= args.min_len]

    if not all_tracks:
        raise SystemExit("no usable residual series; lower --min-len or increase --frames")

    A = np.asarray([
        [r["fid"], r["frame"], r["ex"], r["ey"], r["e_weak"], r["e_strong"],
         r["err_norm"], r["lambda_min"], r["lambda_max"], r["condition"],
         r["range"], r["flow_mag"]]
        for r in rows_flat
    ], float)
    cols = np.array(["fid", "frame", "ex", "ey", "e_weak", "e_strong",
                     "err_norm", "lambda_min", "lambda_max", "condition",
                     "range", "flow_mag"])
    if args.save_npz:
        np.savez(args.save_npz, rows=A, cols=cols)
        print(f"saved {args.save_npz}")

    enorm = A[:, 6]
    print(f"\n=== temporal-correlation input ({len(rows_flat)} pairs) ===")
    print(f"track count usable/all: {len(all_tracks)} / {len(tracks)}")
    print(f"radial |error| median/p90/p99: {np.median(enorm):.3f} / "
          f"{np.percentile(enorm, 90):.3f} / {np.percentile(enorm, 99):.3f} px")
    print(f"outlier >{args.core_px:g}px radial: {np.mean(enorm > args.core_px) * 100:.2f}%")
    print(f"core contiguous usable series: {len(core_tracks)}")

    keys = ["ex", "ey", "e_weak", "e_strong"]
    raw_result = print_acf_report("all residuals", all_tracks, keys, args.max_lag, args.min_len)
    print_acf_report(f"clean-core contiguous runs (|e|<={args.core_px:g}px)",
                     core_tracks, keys, args.max_lag, args.min_len)

    dp = direction_persistence(all_tracks, args.min_len)
    if len(dp):
        print("\n--- vector direction persistence, all residuals ---")
        print(f"consecutive unit-error dot median/p90: "
              f"{np.median(dp):+.3f} / {np.percentile(dp, 90):+.3f}")

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lags = np.arange(1, args.max_lag + 1)
        for key, (rho_raw, rho_dm) in raw_result.items():
            ax[0].plot(lags, rho_raw, marker="o", label=key)
            ax[1].plot(lags, rho_dm, marker="o", label=key)
        ax[0].set_title("raw pooled residual ACF")
        ax[1].set_title("within-track demeaned ACF")
        for a in ax:
            a.axhline(0, color="black", linewidth=0.8)
            a.set_xlabel("lag [frames]")
            a.set_ylabel("correlation")
            a.grid(alpha=0.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=140)
        print(f"saved {args.plot_out}")


if __name__ == "__main__":
    main()
