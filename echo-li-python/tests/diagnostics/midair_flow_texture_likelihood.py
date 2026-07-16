"""Texture-conditioned Rudolf-V flow-error likelihood on MidAir exact GT.

This is the Farnworth-style diagnostic: measure *two-frame* optical-flow error,
rotate it into the KLT structure-tensor eigenbasis, and fit robust scale-vs-texture
tables. It is distinct from cumulative beta diagnostics:

    flow_error_k = (u_rudolf,k - u_rudolf,k-1) - (u_gt,k - u_rudolf,k-1)
                 = u_rudolf,k - project_exact(X_prev, T_k)

where X_prev is the exact 3D point under the previous tracked pixel. The structure
tensor is computed on the previous preprocessed image at the previous pixel, which
matches the two-frame LK likelihood model.

Run from repo root:

  echo-li-python/venv/bin/python \
    echo-li-python/tests/diagnostics/midair_flow_texture_likelihood.py \
    --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
    --set VO_test --cond sunny --traj 0 --frames 300 --scale 0.5 \
    --config configs/diagnostics_midair_sparse3d.yaml --plot-out /tmp/flow_tex.png
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


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def structure_tensor_fields(img, r):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    ks = (2 * r + 1, 2 * r + 1)
    sxx = cv2.boxFilter(gx * gx, cv2.CV_32F, ks, normalize=False)
    sxy = cv2.boxFilter(gx * gy, cv2.CV_32F, ks, normalize=False)
    syy = cv2.boxFilter(gy * gy, cv2.CV_32F, ks, normalize=False)
    return sxx, sxy, syy


def eig_sym2(a, b, c):
    tr = a + c
    d = np.sqrt(max(((a - c) * 0.5) ** 2 + b * b, 0.0))
    lmax = tr * 0.5 + d
    lmin = tr * 0.5 - d
    # eigenvector for lmax; lmin axis is orthogonal.
    vx = b
    vy = lmax - a
    n = np.hypot(vx, vy)
    if n < 1e-12:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = vx / n, vy / n
    ux, uy = -vy, vx
    return lmin, lmax, ux, uy, vx, vy


def bin_edges_log(x, n_bins):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.array([1.0, 10.0])
    lo, hi = np.percentile(x, [1, 99])
    lo = max(lo, 1e-6)
    hi = max(hi, lo * 1.01)
    return np.geomspace(lo, hi, n_bins + 1)


def summarize_component(label, t, e, edges, core_px):
    print(f"\n--- {label}: residual component vs texture eigenvalue ---")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.isfinite(t) & np.isfinite(e) & (t >= lo) & (t < hi)
        if m.sum() < 30:
            continue
        core = m & (np.abs(e) <= core_px)
        sig_all = robust_sigma(e[m])
        sig_core = robust_sigma(e[core])
        p68 = np.percentile(np.abs(e[m] - np.median(e[m])), 68.27)
        rows.append((np.sqrt(lo * hi), m.sum(), sig_all, sig_core, p68,
                     np.mean(np.abs(e[m]) > core_px) * 100.0))
        print(f"  t~{rows[-1][0]:10.3g}: n={int(m.sum()):6d}  "
              f"sigma_mad {sig_all:7.3f}px  core {sig_core:7.3f}px  "
              f"p68_abs {p68:7.3f}px  >{core_px:g}px {rows[-1][5]:6.2f}%")
    return np.array(rows, float)


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
    ap.add_argument("--window", type=int, default=0,
                    help="structure tensor radius; 0 uses FrontendConfig.klt_window")
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--core-px", type=float, default=3.0)
    ap.add_argument("--bins", type=int, default=8)
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
    rows = []
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
                e_weak = err[0] * ux + err[1] * uy
                e_strong = err[0] * vx + err[1] * vy
                flow = p1 - p0
                rows.append((e_weak, e_strong, lmin, lmax, lmax / max(lmin, 1e-9),
                             float(np.hypot(err[0], err[1])), float(np.hypot(flow[0], flow[1])),
                             float(rng), float(ds.omega_mag(i)), float(i)))

        prev = (cur, depth, T_wb, prep)
        if (i - args.start) % 50 == 0:
            fps = (i - args.start + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i - args.start:4d}/{last - args.start}] tracks={len(cur):4d} "
                  f"pairs={len(rows):7d} {fps:5.1f} fps")

    A = np.asarray(rows, float)
    cols = np.array(["e_weak", "e_strong", "lambda_min", "lambda_max", "condition",
                     "err_norm", "flow_mag", "range", "omega", "frame"])
    if args.save_npz:
        np.savez(args.save_npz, rows=A, cols=cols)
        print(f"saved {args.save_npz}")
    if len(A) == 0:
        raise SystemExit("no scored flow pairs")

    eweak, estrong, lmin, lmax, cond, enorm, flow_mag, rng, omega, frame = A.T
    print(f"\n=== texture-conditioned two-frame Rudolf flow error ({len(A)} pairs) ===")
    print(f"radial |error| median/p90/p99: {np.median(enorm):.3f} / "
          f"{np.percentile(enorm, 90):.3f} / {np.percentile(enorm, 99):.3f} px")
    print(f"component sigma MAD: weak {robust_sigma(eweak):.3f}px, "
          f"strong {robust_sigma(estrong):.3f}px")
    print(f"component core sigma MAD |e|<={args.core_px:g}: "
          f"weak {robust_sigma(eweak[np.abs(eweak) <= args.core_px]):.3f}px, "
          f"strong {robust_sigma(estrong[np.abs(estrong) <= args.core_px]):.3f}px")
    print(f"outlier >{args.core_px:g}px radial: {np.mean(enorm > args.core_px) * 100:.2f}%")

    weak_edges = bin_edges_log(lmin, args.bins)
    strong_edges = bin_edges_log(lmax, args.bins)
    weak_rows = summarize_component("weak direction (lambda_min)", lmin, eweak,
                                    weak_edges, args.core_px)
    strong_rows = summarize_component("strong direction (lambda_max)", lmax, estrong,
                                      strong_edges, args.core_px)

    print("\n--- condition-number check ---")
    qs = np.percentile(cond[np.isfinite(cond)], [0, 20, 40, 60, 80, 95, 100])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (cond >= lo) & (cond < hi)
        if m.sum() < 30:
            continue
        print(f"  cond [{lo:8.2f},{hi:8.2f}): n={int(m.sum()):6d}  "
              f"|err| med {np.median(enorm[m]):6.3f}px  p90 {np.percentile(enorm[m], 90):7.3f}  "
              f"sig weak {robust_sigma(eweak[m]):6.3f}  sig strong {robust_sigma(estrong[m]):6.3f}")

    print("\n--- suggested next model ---")
    print("Fit lookup tables sigma_weak(lambda_min), sigma_strong(lambda_max) from")
    print("the 'core' columns first, then test Sparse3D with")
    print("R_pixel = R_eig diag(sigma_weak^2, sigma_strong^2) R_eig^T.")

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].scatter(lmin, np.abs(eweak), s=1, alpha=0.05)
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlabel("lambda_min")
        ax[0, 0].set_ylabel("|weak component error| [px]")
        if len(weak_rows):
            ax[0, 0].plot(weak_rows[:, 0], weak_rows[:, 3], "o-", color="tab:red",
                          label="MAD core sigma")
            ax[0, 0].legend()

        ax[0, 1].scatter(lmax, np.abs(estrong), s=1, alpha=0.05)
        ax[0, 1].set_xscale("log")
        ax[0, 1].set_yscale("log")
        ax[0, 1].set_xlabel("lambda_max")
        ax[0, 1].set_ylabel("|strong component error| [px]")
        if len(strong_rows):
            ax[0, 1].plot(strong_rows[:, 0], strong_rows[:, 3], "o-", color="tab:blue",
                          label="MAD core sigma")
            ax[0, 1].legend()

        ax[1, 0].hist(np.clip(eweak, -5, 5), bins=100, alpha=0.7, label="weak")
        ax[1, 0].hist(np.clip(estrong, -5, 5), bins=100, alpha=0.7, label="strong")
        ax[1, 0].set_xlabel("component error [px], clipped")
        ax[1, 0].legend()

        ax[1, 1].scatter(cond, enorm, s=1, alpha=0.05)
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_xlabel("condition lambda_max/lambda_min")
        ax[1, 1].set_ylabel("radial error [px]")
        for a in ax.ravel():
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=140)
        print(f"saved {args.plot_out}")


if __name__ == "__main__":
    main()
