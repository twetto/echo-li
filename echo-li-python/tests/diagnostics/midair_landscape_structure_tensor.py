"""MidAir diagnostic: structure tensor vs SSD landscape uncertainty.

This directly checks the claim that the local KLT/SSD uncertainty is visible in
the patch landscape shape and predicted by the windowed structure tensor.  It is
separate from the first-observation diagnostics: it uses a fresh previous-frame
template and exact MidAir GT correspondence.

For each selected point:
  1. take a patch in frame k,
  2. reproject that exact 3D point into frame k+dt,
  3. evaluate the SSD translation landscape around the exact current center,
  4. compare the landscape Hessian/eigenvectors with the current-frame structure tensor.

Run from repo root:

  echo-li-python/venv/bin/python \
    echo-li-python/tests/diagnostics/midair_landscape_structure_tensor.py \
    --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
    --set VO_test --cond sunny --traj 0 --frame 120 --dt 1 --scale 0.5 \
    --out /tmp/midair_landscape_structure_tensor.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import photometric_klt_ab as pk  # noqa: E402

R = md.R
OFF = np.arange(-R, R + 1, dtype=np.float32)
OFFX = np.repeat(OFF, len(OFF))
OFFY = np.tile(OFF, len(OFF))


def camera_point_fast(Xw, T_bw):
    pb = T_bw[:3, :3] @ Xw + T_bw[:3, 3]
    return md.RT_BC[:3, :3].T @ pb


def project_world_fast(Xw, T_bw, f, cx, cy):
    pc = camera_point_fast(Xw, T_bw)
    if pc[2] <= 1e-6:
        return None
    return np.array([f * pc[0] / pc[2] + cx, f * pc[1] / pc[2] + cy], float)


def eig2(H):
    vals, vecs = np.linalg.eigh(H)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def folded_angle_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    c = abs(float(np.asarray(a) @ np.asarray(b)) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def structure_tensor_at(img, p, r):
    """Window-summed image-gradient Hessian at p on the same image used by SSD."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    off = np.arange(-r, r + 1, dtype=np.float32)
    offx = np.repeat(off, len(off))
    offy = np.tile(off, len(off))
    jx = pk.sample(gx, np.array([p[0]]), np.array([p[1]]), offx, offy)[0]
    jy = pk.sample(gy, np.array([p[0]]), np.array([p[1]]), offx, offy)[0]
    return np.array([[float(jx @ jx), float(jx @ jy)],
                     [float(jx @ jy), float(jy @ jy)]])


def ssd_landscape(cur_img, tref, center, grid):
    dx, dy = np.meshgrid(grid, grid)
    x = (center[0] + dx).ravel()
    y = (center[1] + dy).ravel()
    Iw = pk.sample(cur_img, x, y, OFFX, OFFY)
    return np.mean((Iw - tref[None, :]) ** 2, axis=1).reshape(len(grid), len(grid))


def local_hessian_from_landscape(land, step):
    c = land.shape[0] // 2
    f00 = float(land[c, c])
    fpx = float(land[c, c + 1])
    fmx = float(land[c, c - 1])
    fpy = float(land[c + 1, c])
    fmy = float(land[c - 1, c])
    fpp = float(land[c + 1, c + 1])
    fpm = float(land[c - 1, c + 1])
    fmp = float(land[c + 1, c - 1])
    fmm = float(land[c - 1, c - 1])
    hxx = (fpx - 2.0 * f00 + fmx) / (step * step)
    hyy = (fpy - 2.0 * f00 + fmy) / (step * step)
    hxy = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
    return np.array([[hxx, hxy], [hxy, hyy]], float)


def context_crop(img, p0, p1, size):
    half = size // 2
    crop = cv2.getRectSubPix(img, (size, size), (float(p1[0]), float(p1[1])))
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    for p, color, label in ((p0, (255, 190, 0), "prev"), (p1, (255, 255, 255), "GT")):
        q = np.round(np.asarray(p) - p1 + half).astype(int)
        x, y = int(q[0]), int(q[1])
        if -R <= x < size + R and -R <= y < size + R:
            cv2.rectangle(
                vis,
                (int(np.clip(x - R, 0, size - 1)), int(np.clip(y - R, 0, size - 1))),
                (int(np.clip(x + R, 0, size - 1)), int(np.clip(y + R, 0, size - 1))),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(vis, label, (int(np.clip(x - R, 0, size - 1)),
                                     int(np.clip(y - R - 2, 8, size - 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    return vis


def collect_cases(args):
    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    if args.frame + args.dt >= ds.n:
        raise SystemExit(f"frame+dt exceeds dataset length: {args.frame}+{args.dt} >= {ds.n}")

    img0_raw = ds.image(args.frame)
    img1_raw = ds.image(args.frame + args.dt)
    img0 = cv2.equalizeHist(img0_raw).astype(np.float32) if args.histeq else img0_raw.astype(np.float32)
    img1 = cv2.equalizeHist(img1_raw).astype(np.float32) if args.histeq else img1_raw.astype(np.float32)
    depth0 = ds.depth(args.frame)
    T0 = ds.pose(args.frame)
    T1_bw = np.linalg.inv(ds.pose(args.frame + args.dt))
    H, W = img0_raw.shape
    f, cx, cy = md.intrinsics(W, H)

    margin = int(max(args.border, R + args.grid_half + 4))
    mask = np.uint8((depth0 > 1.0) & (depth0 < md.SKY)) * 255
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    corners = cv2.goodFeaturesToTrack(
        img0_raw, args.max_features, args.quality, 2 * R + 3, mask=mask
    )
    if corners is None:
        raise SystemExit("no corners found")

    grid = np.arange(-args.grid_half, args.grid_half + 0.5 * args.grid_step, args.grid_step)
    if len(grid) < 5 or abs(grid[len(grid) // 2]) > 1e-9:
        raise SystemExit("--grid-half must be an integer multiple of --grid-step")

    cases = []
    for c in corners.reshape(-1, 2):
        p0 = np.array([float(c[0]), float(c[1])])
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        d0 = float(depth0[y0, x0])
        if not (1.0 < d0 < md.SKY):
            continue
        Xw = md.backproject_world((p0[0], p0[1]), d0, T0, f, cx, cy)
        p1 = project_world_fast(Xw, T1_bw, f, cx, cy)
        if p1 is None:
            continue
        if not (margin <= p1[0] < W - margin and margin <= p1[1] < H - margin):
            continue

        tref = pk.sample(img0, np.array([p0[0]]), np.array([p0[1]]), OFFX, OFFY)[0]
        land = ssd_landscape(img1, tref, p1, grid)
        H_tensor = structure_tensor_at(img1, p1, args.tensor_radius)
        H_land = local_hessian_from_landscape(land, args.grid_step)
        tvals, tvecs = eig2(H_tensor)
        lvals, lvecs = eig2(H_land)
        if tvals[0] <= args.min_eig or lvals[0] <= 0:
            continue

        imin = np.unravel_index(np.argmin(land), land.shape)
        argmin = np.array([grid[imin[1]], grid[imin[0]]], float)
        cases.append({
            "p0": p0,
            "p1": p1,
            "land": land,
            "tensor_vals": tvals,
            "tensor_vecs": tvecs,
            "land_vals": lvals,
            "land_vecs": lvecs,
            "tensor_cond": float(tvals[1] / max(tvals[0], 1e-12)),
            "land_cond": float(lvals[1] / max(lvals[0], 1e-12)),
            "weak_angle": folded_angle_deg(tvecs[:, 0], lvecs[:, 0]),
            "argmin": argmin,
            "context": context_crop(img1_raw, p0, p1, args.context),
        })

    if len(cases) < args.examples:
        raise SystemExit(f"only {len(cases)} usable cases; lower --examples or change frame")
    return ds, grid, cases


def choose_examples(cases, n):
    ordered = sorted(cases, key=lambda z: z["tensor_cond"])
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in idx]


def draw_axis(ax, vec, color, scale, label):
    v = np.asarray(vec, float)
    ax.plot([-v[0] * scale, v[0] * scale], [-v[1] * scale, v[1] * scale],
            color=color, lw=1.8, label=label)


def summarize_cases(args, ds, cases):
    summary = np.array([
        [c["tensor_cond"], c["land_cond"], c["weak_angle"], np.linalg.norm(c["argmin"])]
        for c in cases
    ], float)
    print(f"MidAir {args.subset}/{args.cond}/{ds.traj} frame {args.frame}->{args.frame + args.dt}")
    print(f"usable cases: {len(cases)}")
    print("columns: tensor_cond, landscape_cond, weak_axis_angle_deg, ssd_argmin_offset_px")
    for q in (50, 75, 90, 95):
        vals = np.percentile(summary, q, axis=0)
        print(f"p{q:02d}: " + " ".join(f"{v:8.3f}" for v in vals))
    finite = np.isfinite(summary[:, 2])
    print(f"median weak-axis agreement angle: {np.median(summary[finite, 2]):.1f} deg")
    print("corr log(tensor_cond), log(landscape_cond): "
          f"{np.corrcoef(np.log(summary[:, 0]), np.log(summary[:, 1]))[0, 1]:.3f}")
    return summary


def make_plot(args, ds, grid, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = choose_examples(cases, args.examples)
    fig, axes = plt.subplots(len(examples), 3, figsize=(10.8, 2.75 * len(examples)),
                             constrained_layout=True)
    if len(examples) == 1:
        axes = axes[None, :]

    for r, case in enumerate(examples):
        axes[r, 0].imshow(case["context"])
        axes[r, 0].set_axis_off()
        axes[r, 0].set_title(
            f"context\nT cond {case['tensor_cond']:.1f}, L cond {case['land_cond']:.1f}",
            fontsize=9,
        )

        ax = axes[r, 1]
        land = case["land"]
        vmax = np.percentile(land, 85)
        levels = np.linspace(float(np.min(land)), float(vmax), 14)
        ax.contourf(grid, grid, land, levels=levels, cmap="viridis")
        ax.contour(grid, grid, land, levels=levels[::2], colors="white", linewidths=0.45, alpha=0.5)
        ax.scatter([0], [0], s=20, c="white", marker="+", label="GT")
        ax.scatter([case["argmin"][0]], [case["argmin"][1]], s=20, c="#ef4444", marker="x", label="SSD min")
        draw_axis(ax, case["tensor_vecs"][:, 0], "#fbbf24", args.grid_half * 0.65, "tensor weak")
        draw_axis(ax, case["tensor_vecs"][:, 1], "#38bdf8", args.grid_half * 0.45, "tensor strong")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(grid[-1], grid[0])
        ax.set_title(f"SSD landscape, weak angle {case['weak_angle']:.0f} deg", fontsize=9)
        if r == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=True)

        ax = axes[r, 2]
        t = case["tensor_vals"] / np.sum(case["tensor_vals"])
        l = case["land_vals"] / np.sum(case["land_vals"])
        ax.bar([0, 1], t, width=0.35, label="tensor", color="#f59e0b")
        ax.bar([0.4, 1.4], l, width=0.35, label="landscape", color="#0ea5e9")
        ax.set_xticks([0.2, 1.2], ["weak", "strong"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("normalized curvature")
        ax.set_title(f"argmin |d|={np.linalg.norm(case['argmin']):.2f}px", fontsize=9)
        if r == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"MidAir {args.subset}/{args.cond}/{ds.traj} frame {args.frame}->{args.frame + args.dt}: "
        "structure tensor vs SSD landscape",
        fontsize=12,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    summary = summarize_cases(args, ds, cases)
    print(f"wrote {out}")
    if args.save_npz:
        np.savez(args.save_npz, summary=summary,
                 cols=np.array(["tensor_cond", "landscape_cond",
                                "weak_angle_deg", "argmin_px"]))
        print(f"wrote {args.save_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--dt", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-features", type=int, default=700)
    ap.add_argument("--quality", type=float, default=0.01)
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--tensor-radius", type=int, default=R)
    ap.add_argument("--min-eig", type=float, default=1e-6)
    ap.add_argument("--grid-half", type=float, default=5.0)
    ap.add_argument("--grid-step", type=float, default=0.25)
    ap.add_argument("--context", type=int, default=72)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--no-histeq", dest="histeq", action="store_false")
    ap.set_defaults(histeq=True)
    ap.add_argument("--summary-only", action="store_true",
                    help="compute and print statistics without importing Matplotlib or writing files")
    ap.add_argument("--out", default="/tmp/midair_landscape_structure_tensor.png")
    ap.add_argument("--save-npz", default="")
    args = ap.parse_args()

    ds, grid, cases = collect_cases(args)
    if args.summary_only:
        summarize_cases(args, ds, cases)
    else:
        make_plot(args, ds, grid, cases)


if __name__ == "__main__":
    main()
