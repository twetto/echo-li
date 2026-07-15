"""Synthetic sequence with EXACT ground truth, to settle whether beta is real tracker
drift or GT error.

A single textured 3D plane (texture = a real EuRoC image, so the trackers see realistic
corners) is viewed by a camera on a known trajectory. Each frame is rendered by the
exact plane->image homography (cv2.warpPerspective). Because the scene is one plane, the
image position AND depth of every texture point are known *exactly* in every frame --
no Vicon, no Leica, no time-sync, no depth-lookup. beta measured here is pure tracker
drift.

We seed two self-propagating KLT trackers (first-observation template, previous-frame
template) at exact GT feature births, propagate over the rendered frames, and compare
their landing to the EXACT GT position vs age.

  --verify : just render frame 0 + a later frame with GT dots overlaid, to confirm the
             render + GT are correct before trusting drift numbers.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/synthetic_drift.py \
      --texture <euroc_image.png> [--verify] [--frames 200]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photometric_klt_ab as pk  # noqa: E402  (build_pyramid, klt_track, sample)

W, H = 640, 480
FX = FY = 420.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
R, ITERS, LV = 7, 12, 3


def lookat(eye, target, up=np.array([0.0, -1.0, 0.0])):
    z = target - eye; z /= np.linalg.norm(z)          # camera +z looks at target
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)                # columns = camera axes in world
    T = np.eye(4); T[:3, :3] = R_wc; T[:3, 3] = eye
    return T                                          # T_wc


def make_plane(tw, th, mpp=0.004, tilt_deg=25.0):
    """Plane basis in world: texture pixel (u,v) -> X = C + (u-uc)*mpp*Ax + (v-vc)*mpp*Ay."""
    C = np.array([0.0, 0.0, 3.0])
    a = np.radians(tilt_deg)
    Rt = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Ax = Rt @ np.array([1.0, 0, 0]); Ay = Rt @ np.array([0, 1.0, 0])
    return C, Ax * mpp, Ay * mpp, tw / 2.0, th / 2.0


def homography(T_wc, C, Ax, Ay, uc, vc):
    """H mapping texture (u,v,1) -> image (x,y,1) for this camera pose."""
    T_cw = np.linalg.inv(T_wc)
    Rcw, tcw = T_cw[:3, :3], T_cw[:3, 3]
    c0 = Rcw @ (C - uc * Ax - vc * Ay) + tcw
    M = np.stack([Rcw @ Ax, Rcw @ Ay, c0], axis=1)    # 3x3: [u v 1] -> camera ray
    return K @ M


def project_pts(T_wc, C, Ax, Ay, uc, vc, uv):
    """Exact image position + depth of texture points uv (N,2)."""
    T_cw = np.linalg.inv(T_wc)
    Rcw, tcw = T_cw[:3, :3], T_cw[:3, 3]
    X = C + (uv[:, 0:1] - uc) * Ax + (uv[:, 1:2] - vc) * Ay      # (N,3) world
    Xc = X @ Rcw.T + tcw
    z = Xc[:, 2]
    x = FX * Xc[:, 0] / z + CX
    y = FY * Xc[:, 1] / z + CY
    return np.stack([x, y], axis=1), z


def trajectory(n):
    """Camera slides laterally + a little forward, always looking at the plane centre."""
    C = np.array([0.0, 0.0, 3.0])
    poses = []
    for i in range(n):
        s = i / max(n - 1, 1)
        eye = np.array([-0.5 + 1.0 * s, 0.15 * np.sin(2 * np.pi * s), 0.3 * s])
        poses.append(lookat(eye, C))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--visualize", action="store_true",
                    help="montage: crops centred on EXACT GT with first-obs (red) and "
                         "previous-frame (gold) landings overlaid, over age")
    ap.add_argument("--patch", type=int, default=80)
    ap.add_argument("--out", default="synthetic")
    args = ap.parse_args()

    tex = cv2.imread(args.texture, cv2.IMREAD_GRAYSCALE)
    if tex is None:
        raise SystemExit(f"could not read texture {args.texture}")
    tex = cv2.resize(tex, (tex.shape[1] * 2, tex.shape[0] * 2))   # upscale so plane is big
    th, tw = tex.shape
    C, Ax, Ay, uc, vc = make_plane(tw, th)
    poses = trajectory(args.frames)

    def render(T_wc):
        Hm = homography(T_wc, C, Ax, Ay, uc, vc)
        return cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR)

    # GT feature grid (texture coords), kept away from the texture border
    gx = np.linspace(0.12 * tw, 0.88 * tw, args.grid)
    gy = np.linspace(0.12 * th, 0.88 * th, args.grid)
    uv = np.array([[x, y] for y in gy for x in gx], float)

    if args.verify:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for c, fi in enumerate([0, args.frames // 2, args.frames - 1]):
            im = render(poses[fi]); pts, z = project_pts(poses[fi], C, Ax, Ay, uc, vc, uv)
            ax[c].imshow(im, cmap="gray", vmin=0, vmax=255)
            vis = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            ax[c].scatter(pts[vis, 0], pts[vis, 1], s=8, c="lime", marker="+")
            ax[c].set_title(f"frame {fi}  depth {z.min():.1f}-{z.max():.1f}m", fontsize=9)
            ax[c].set_xticks([]); ax[c].set_yticks([])
        fig.suptitle("Synthetic render + EXACT GT feature positions (green +)", fontsize=11)
        fig.tight_layout(); fig.savefig(args.out + "_verify.png", dpi=130)
        print(f"saved {args.out}_verify.png")
        return

    # ---- self-propagating first-obs and previous-frame KLT vs EXACT GT ----
    off = np.arange(-R, R + 1, dtype=np.float32)
    OFFX = np.repeat(off, len(off)); OFFY = np.tile(off, len(off))
    first_patch = {}; pos_first = {}; pos_prev = {}; born = {}
    rows = []                                    # (age, first_off, prev_off)
    rec = {}                                     # j -> {age: (gp, pfirst, pprev, crop)}
    AGES_VIS = [0, 5, 10, 20, 40, 80, 120, 160]
    ps = args.patch; hpp = ps // 2 + 1
    prev = None
    for i, T in enumerate(poses):
        img = render(T)
        pts, _ = project_pts(T, C, Ax, Ay, uc, vc, uv)
        vis = (pts[:, 0] > R + 1) & (pts[:, 0] < W - R - 1) & (pts[:, 1] > R + 1) & (pts[:, 1] < H - R - 1)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        if prev is not None:
            ppyr, pgx, pgy = prev
            fp = [j for j in pos_prev if vis[j]]
            if fp:
                u, v, _ = pk.klt_track(ppyr, cur_pyr, pgx, pgy,
                                       np.array([pos_prev[j] for j in fp]), R, ITERS, "ssd")
                for k, j in enumerate(fp):
                    pos_prev[j] = u[k] if v[k] else pos_prev[j]
            ff = [j for j in pos_first if vis[j]]
            if ff:
                tref = np.array([first_patch[j] for j in ff])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_first[j] for j in ff]), R, ITERS,
                                       "blendB", "ssd", 1e6, tref)
                for k, j in enumerate(ff):
                    pos_first[j] = u[k] if v[k] else pos_first[j]
            for j in range(len(uv)):
                if vis[j] and j in pos_first:
                    age = i - born[j]
                    if age >= 1:
                        rows.append((age, np.hypot(*(pos_first[j] - pts[j])),
                                     np.hypot(*(pos_prev[j] - pts[j]))))
                        gp = pts[j]
                        if (args.visualize and age in AGES_VIS
                                and hpp <= gp[0] < W - hpp and hpp <= gp[1] < H - hpp):
                            crop = cv2.getRectSubPix(img, (ps, ps), (float(gp[0]), float(gp[1])))
                            rec.setdefault(j, {})[age] = (
                                gp.copy(), pos_first[j].copy(), pos_prev[j].copy(), crop)
        # births (features newly visible): seed BOTH trackers at the EXACT GT position
        for j in range(len(uv)):
            if vis[j] and j not in pos_first:
                born[j] = i
                bx, by = pts[j]
                first_patch[j] = np.stack(
                    [pk.sample(cur_pyr[lv], np.array([bx * 0.5 ** lv]),
                               np.array([by * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])
                pos_first[j] = pts[j].copy(); pos_prev[j] = pts[j].copy()
        for j in [j for j in list(pos_first) if not vis[j]]:
            pos_first.pop(j, None); pos_prev.pop(j, None)
        prev = (cur_pyr, cgx, cgy)

    A = np.array(rows)
    print(f"\n=== synthetic drift, EXACT GT ({len(A)} obs) ===")
    print(f"  {'age':>8} {'n':>7} {'first_off':>10} {'prev_off':>9}")
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            print(f"  {lo:3d}-{hi:<4d} {m.sum():7d} {np.median(A[m,1]):10.3f} {np.median(A[m,2]):9.3f}")

    if args.visualize:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        long = [j for j, rm in rec.items() if max(rm) >= 120]
        sel = long[:: max(1, len(long) // 6)][:6]
        cols = AGES_VIS
        fig, ax = plt.subplots(len(sel), len(cols), figsize=(1.7 * len(cols), 1.8 * len(sel)))
        if len(sel) == 1:
            ax = ax[None, :]
        for r, j in enumerate(sel):
            rm = rec[j]
            for c, age in enumerate(cols):
                a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
                if age not in rm:
                    a.axis("off"); continue
                gp, pf, pp, crop = rm[age]; cen = ps / 2
                a.imshow(crop, cmap="gray", vmin=0, vmax=255)
                a.add_patch(plt.Rectangle((cen - R, cen - R), 2 * R + 1, 2 * R + 1,
                                          fill=False, ec="0.4", lw=0.8))
                a.plot(cen, cen, "+", color="lime", ms=10, mew=2)
                a.plot(pp[0] - gp[0] + cen, pp[1] - gp[1] + cen, "x", color="gold", ms=7, mew=2)
                a.plot(pf[0] - gp[0] + cen, pf[1] - gp[1] + cen, "o", mfc="none", mec="red", ms=9, mew=1.5)
                a.set_title(f"n={age}  1st {np.hypot(*(pf-gp)):.2f} / prev {np.hypot(*(pp-gp)):.2f}px",
                            fontsize=6)
        fig.suptitle("SYNTHETIC (exact GT): green + = truth   red o = first-obs KLT   "
                     "gold x = previous-frame KLT", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(args.out + "_montage.png", dpi=130)
        print(f"saved {args.out}_montage.png")


if __name__ == "__main__":
    main()
