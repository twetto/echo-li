"""Exact-GT KLT drift (beta) on Mid-Air -- realistic 3D scene, NO ground-truth error.

Result 13 (flow_bias_template_drift.md) showed the single textured *plane* synthetic
drifts only ~0.005 px/frame, ~60x less than the 0.3 px/frame beta measured on EuRoC --
so most of EuRoC's beta was Vicon/Leica/time-sync GROUND-TRUTH error, not tracker drift.
But a plane has no depth discontinuity and no occlusion, so it cannot test the two
effects most likely to be the *real* driver.

Mid-Air fixes both axes at once: it is a photorealistic synthetic drone dataset with
EXACT per-pixel dense depth + EXACT pose (no sensor error) on real 3D scenes (buildings,
trees, terrain -> genuine occlusion, depth edges, parallax), and the SAME trajectories
under sunny/foggy/sunset (a lighting ablation at fixed geometry).

Exact GT correspondence is by full SE(3) reprojection (validated to ~0.001 px against the
simulator's own native flow -- see --verify): a feature is anchored at birth to a 3D
world point via the exact range-depth backprojection, and at every later frame that point
is reprojected with the exact pose. beta(age) = tracked_pixel - reprojected_pixel is pure
tracker drift. Depth is stored as EUCLIDEAN RANGE (|point| = depth), camera is a 90-deg
pinhole f = W/2, body<-cam extrinsic Rt_bc, groundtruth at 100 Hz vs camera 25 Hz (i*4).

Per observation we also log the observable causes for the exact-GT redo of result 7:
local depth variance (depth-edge proximity), a z-buffer occlusion flag, inter-frame flow
magnitude, angular rate |w|, image radius, point depth.

  PY=echo-li-python/venv/bin/python
  $PY midair_drift.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --set VO_test --cond sunny --traj 0 --frames 400 [--scale 0.5] [--verify]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import h5py
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photometric_klt_ab as pk  # noqa: E402  (build_pyramid, klt_track, sample)

LV, R, ITERS = 3, 7, 12
RT_BC = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
SKY = 5000.0  # range >= this is sky / no-return


def open_range_depth(path):
    """Mid-Air depth PNG -> float32 Euclidean range in metres (float16 packed as uint16)."""
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    return img.view(np.float16).astype(np.float32)


def decode_flow(path):
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    return raw[..., :2].copy().view(np.float16).astype(np.float32)


class MidAir:
    def __init__(self, root, subset, cond, traj, scale):
        self.dir = Path(root) / subset / cond
        self.traj = f"trajectory_{traj:04d}"
        self.scale = scale
        self.db = h5py.File(self.dir / "sensor_records.hdf5", "r")
        g = self.db[self.traj]["groundtruth"]
        self.pos = g["position"][:]
        self.att = g["attitude"][:]
        self.omega = g["angular_velocity"][:]        # body frame, 100 Hz
        self.n = len(list((self.dir / "color_left" / self.traj).glob("*.JPEG")))

    def pose(self, k):
        """T_wb (body->world) at camera frame k (gt sampled at 100 Hz, camera at 25 Hz)."""
        q = self.att[k * 4]
        r = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        M = np.eye(4)
        M[:3, :3] = r
        M[:3, 3] = self.pos[k * 4]
        return M

    def omega_mag(self, k):
        return float(LA.norm(self.omega[k * 4]))

    def image(self, k):
        p = self.dir / "color_left" / self.traj / f"{k:06d}.JPEG"
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if self.scale != 1.0:
            im = cv2.resize(im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return im

    def depth(self, k):
        p = self.dir / "depth" / self.traj / f"{k:06d}.PNG"
        d = open_range_depth(str(p))
        if self.scale != 1.0:
            d = cv2.resize(d, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        return d

    def native_flow(self, k):
        p = self.dir / "flow" / self.traj / f"{k:06d}.PNG"
        fl = decode_flow(str(p))
        if self.scale != 1.0:
            fl = cv2.resize(fl, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_NEAREST) * self.scale
        return fl


def intrinsics(W, H):
    f = W / 2.0
    return f, W / 2.0, H / 2.0


def backproject_world(uv, d_range, T_wb, f, cx, cy):
    """pixel + range depth -> world point (through Rt_bc)."""
    ray = np.array([uv[0] - cx, uv[1] - cy, f])
    pc = ray / LA.norm(ray) * d_range                       # camera frame, |pc| = range
    pb = RT_BC[:3, :3] @ pc + RT_BC[:3, 3]                   # body
    return T_wb[:3, :3] @ pb + T_wb[:3, 3]                   # world


def project_world(Xw, T_wb, f, cx, cy):
    """world point -> (pixel, range, z) in camera k."""
    Rcb = RT_BC[:3, :3].T
    pb = LA.inv(T_wb)[:3, :3] @ Xw + LA.inv(T_wb)[:3, 3]     # world->body
    pc = Rcb @ pb                                            # body->cam (Rt_bc^-1)
    if pc[2] <= 1e-6:
        return None, None, None
    u = f * pc[0] / pc[2] + cx
    v = f * pc[1] / pc[2] + cy
    return np.array([u, v]), LA.norm(pc), pc[2]


def open_writer(path, w, h, fps):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if wr.isOpened():
        return wr, path
    avi = Path(path).with_suffix(".avi")
    wr = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return wr, avi


def verify(ds, W, H):
    """Re-confirm reprojection == native flow at the working scale (sanity, result-13 lesson)."""
    f, cx, cy = intrinsics(W, H)
    errs = []
    for k in [50, 100, 200]:
        if k >= ds.n:
            continue
        depth = ds.depth(k)
        Tprev, Tcur = ds.pose(k - 1), ds.pose(k)
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        ray = np.stack([us - cx, vs - cy, np.full_like(us, f, float)], -1)
        pc = ray / LA.norm(ray, axis=-1, keepdims=True) * depth[..., None]
        ph = np.concatenate([pc, np.ones((H, W, 1))], -1).reshape(-1, 4).T
        Rt_rel = LA.inv(Tprev) @ Tcur                        # cur-body -> prev-body
        proj = LA.inv(RT_BC) @ Rt_rel @ RT_BC @ ph
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        pix = K @ proj[:3]; pix = pix / pix[2]               # -> pixels (apply intrinsics)
        my = np.stack([us, vs], -1) - np.stack([pix[0], pix[1]], -1).reshape(H, W, 2)
        nat = ds.native_flow(k)
        m = np.isfinite(depth) & (depth < SKY) & np.isfinite(nat[..., 0])
        e = np.hypot(*(my[m] - nat[m]).T)
        errs.append(np.median(e))
        print(f"  frame {k}: native|flow| med {np.median(np.hypot(*nat[m].T)):6.2f}px  "
              f"reproj-vs-native med {np.median(e):.4f}px  p90 {np.percentile(e, 90):.4f}px")
    print(f"VERIFY {'OK' if np.mean(errs) < 0.05 else 'FAIL'}: exact-GT machinery matches "
          f"native flow to {np.mean(errs):.4f}px median.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--redetect", type=int, default=250, help="re-detect when active < this")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--visualize", action="store_true",
                    help="montage of where the trackers land vs exact GT, worst-drift tracks")
    ap.add_argument("--video", action="store_true",
                    help="render the whole run to mp4: GT (green +), prev (gold x), first-obs (red o)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--out", default="midair_drift")
    args = ap.parse_args()

    ds = MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    im0 = ds.image(args.start)
    H, W = im0.shape
    f, cx, cy = intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  {ds.n} frames  work {W}x{H} f={f:.0f}")

    if args.verify:
        verify(ds, W, H)
        return

    off = np.arange(-R, R + 1, dtype=np.float32)
    OFFX = np.repeat(off, len(off)); OFFY = np.tile(off, len(off))

    # per-track state
    Xw = {}; born = {}; first_patch = {}; pos_prev = {}; pos_first = {}
    gt_prevpix = {}
    nid = 0
    rows = []        # age, beta_first, beta_prev, depth_var, occluded, flow_mag, omega, radius, depth
    rec = {}         # id -> {age: (gp, pos_first, pos_prev, crop)}  for the montage
    AGES_VIS = [1, 5, 10, 20, 40, 80, 120]
    ps = args.patch; hpp = ps // 2 + 1
    writer = vpath = None
    if args.video:
        writer, vpath = open_writer(f"{args.out}_{args.cond}.mp4", W, H, args.fps)
    prev = None
    last = min(args.start + args.frames, ds.n)
    for i in range(args.start, last):
        img = ds.image(i)
        depth = ds.depth(i)
        T = ds.pose(i)
        om = ds.omega_mag(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if args.video else None

        if prev is not None:
            ppyr, pgx, pgy = prev
            # propagate previous-frame KLT; DROP a tracker when its KLT invalidates (a real
            # tracker would -- keeping a frozen dead position inflates the tail and clutters
            # the video). Forward-additive: the Jacobian is sampled at the moving iterate in
            # the image being warped, so it needs the CURRENT-frame gradients (cgx,cgy), NOT
            # the previous frame's (passing pgx,pgy silently breaks under fast motion).
            idp = list(pos_prev)
            if idp:
                u, v, _ = pk.klt_track(ppyr, cur_pyr, cgx, cgy,
                                       np.array([pos_prev[j] for j in idp]), R, ITERS, "ssd")
                for k, j in enumerate(idp):
                    if v[k]:
                        pos_prev[j] = u[k]
                    else:
                        pos_prev.pop(j, None)
            # propagate first-observation KLT (reference template, blendB huge lambda)
            idf = list(pos_first)
            if idf:
                tref = np.array([first_patch[j] for j in idf])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_first[j] for j in idf]), R, ITERS,
                                       "blendB", "ssd", 1e6, tref)
                for k, j in enumerate(idf):
                    if v[k]:
                        pos_first[j] = u[k]
                    else:
                        pos_first.pop(j, None)
            # exact GT + covariates over every still-anchored track
            drop = []
            for j in list(born):
                gp, rng, _z = project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                alivep, alivef = j in pos_prev, j in pos_first
                if not inb or not (alivep or alivef):
                    drop.append(j); continue
                age = i - born[j]
                gx, gy = int(round(gp[0])), int(round(gp[1]))
                dwin = depth[max(0, gy - R):gy + R + 1, max(0, gx - R):gx + R + 1]
                dwin = dwin[dwin < SKY]
                dvar = float(np.std(dwin)) if dwin.size > 4 else np.nan
                map_rng = float(depth[gy, gx])
                occluded = int(map_rng < rng - max(0.05 * rng, 0.5))   # z-buffer: scene nearer
                fmag = np.hypot(*(gp - gt_prevpix[j])) if j in gt_prevpix else np.nan
                rad = np.hypot(gp[0] - cx, gp[1] - cy)
                bf = np.hypot(*(pos_first[j] - gp)) if alivef else np.nan
                bp = np.hypot(*(pos_prev[j] - gp)) if alivep else np.nan
                rows.append((age, bf, bp, dvar, occluded, fmag, om, rad, rng))
                gt_prevpix[j] = gp
                if args.video:
                    g = (int(round(gp[0])), int(round(gp[1])))
                    cv2.drawMarker(vis, g, (0, 255, 0), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
                    if alivep:
                        pp = (int(round(pos_prev[j][0])), int(round(pos_prev[j][1])))
                        if bp > 2:
                            cv2.line(vis, g, pp, (0, 215, 255), 1, cv2.LINE_AA)
                        cv2.drawMarker(vis, pp, (0, 215, 255), cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA)
                    if alivef:
                        pf = (int(round(pos_first[j][0])), int(round(pos_first[j][1])))
                        if bf > 2:
                            cv2.line(vis, g, pf, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.circle(vis, pf, 3, (0, 0, 255), 1, cv2.LINE_AA)
                    if occluded:
                        cv2.circle(vis, g, 5, (255, 0, 255), 1, cv2.LINE_AA)
                if (args.visualize and age in AGES_VIS
                        and hpp <= gp[0] < W - hpp and hpp <= gp[1] < H - hpp):
                    crop = cv2.getRectSubPix(img, (ps, ps), (float(gp[0]), float(gp[1])))
                    rec.setdefault(j, {})[age] = (
                        gp.copy(), pos_first[j].copy() if alivef else None,
                        pos_prev[j].copy() if alivep else None, crop)
            for j in drop:
                for d in (Xw, born, first_patch, pos_prev, pos_first, gt_prevpix):
                    d.pop(j, None)

        # births: detect corners on finite-depth, non-sky, away from existing tracks
        if len(pos_prev) < args.redetect:
            mask = np.uint8((depth < SKY) & (depth > 1.0)) * 255
            for j in pos_prev:
                p = pos_prev[j]
                cv2.circle(mask, (int(p[0]), int(p[1])), R + 2, 0, -1)
            corners = cv2.goodFeaturesToTrack(img, args.max_tracks - len(pos_prev),
                                              0.01, 2 * R + 3, mask=mask)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    x, y = float(c[0]), float(c[1])
                    gx, gy = int(round(x)), int(round(y))
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < SKY):
                        continue
                    Xw[nid] = backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = np.stack(
                        [pk.sample(cur_pyr[lv], np.array([x * 0.5 ** lv]),
                                   np.array([y * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    nid += 1
        if args.video:
            cv2.rectangle(vis, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(vis, f"{args.cond} frame {i}  active {len(pos_prev)}  |w|={om:.2f}",
                        (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "GT +   prev x   first-obs o   occluded []",
                        (6, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(vis)
        prev = (cur_pyr, cgx, cgy)
        if (i - args.start) % 100 == 0:
            print(f"  [{i - args.start}/{last - args.start}] active={len(pos_prev)} obs={len(rows)}")

    if writer is not None:
        writer.release()
        print(f"saved {vpath}")

    A = np.array(rows, float)
    cols = "age beta_first beta_prev depth_var occluded flow_mag omega radius depth".split()
    np.savez(f"{args.out}_{args.cond}.npz", rows=A, cols=cols)
    print(f"\n=== Mid-Air {args.subset}/{args.cond} exact-GT drift ({len(A)} obs, {nid} tracks) ===")
    print(f"  {'age':>8} {'n':>7} {'first_off':>10} {'prev_off':>9}  (median |beta| px)")
    fit_x, fit_y = [], []
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 999)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() > 15:
            mf, mp = np.nanmedian(A[m, 1]), np.nanmedian(A[m, 2])
            print(f"  {lo:3d}-{hi:<4d} {int(m.sum()):7d} {mf:10.3f} {mp:9.3f}")
            fit_x.append(0.5 * (lo + hi)); fit_y.append(mp)
    if len(fit_x) >= 2:
        slope = np.polyfit(fit_x, fit_y, 1)[0]
        print(f"  previous-frame drift slope ~ {slope:.4f} px/frame  "
              f"(EuRoC beta was 0.30; clean plane was 0.005)")
    # occlusion / depth-edge split (the result-7 covariates, now on EXACT GT)
    if len(A):
        occ = A[:, 4] == 1
        print(f"\n  occluded obs: {occ.mean()*100:.1f}%  "
              f"median beta_prev  occluded {np.nanmedian(A[occ,2]) if occ.any() else float('nan'):.3f}  "
              f"visible {np.nanmedian(A[~occ,2]):.3f}")
        dv = A[:, 3]
        fin = np.isfinite(dv)
        if fin.sum() > 100:
            q = np.nanpercentile(dv[fin], [25, 50, 75])
            for lbl, lo, hi in [("lowedge", -1, q[0]), ("mid", q[0], q[2]), ("highedge", q[2], 1e9)]:
                mm = fin & (dv >= lo) & (dv < hi)
                if mm.sum() > 30:
                    print(f"    depth_var {lbl:8s} (std {lo:.1f}-{hi:.1f}m): "
                          f"median beta_prev {np.nanmedian(A[mm,2]):.3f}px  n={int(mm.sum())}")

    if args.visualize and rec:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # pick the tracks with the largest drift at their oldest recorded age
        def off(p, gp):
            return np.hypot(*(p - gp)) if p is not None else -1.0
        cand = []
        for j, rm in rec.items():
            old = max(rm)
            gp, pf, pp, _ = rm[old]
            cand.append((-max(off(pf, gp), off(pp, gp)), j))
        cand.sort()
        sel = [j for _, j in cand[:6]]
        cols = AGES_VIS
        fig, ax = plt.subplots(len(sel), len(cols), figsize=(1.7 * len(cols), 1.8 * len(sel)))
        if len(sel) == 1:
            ax = ax[None, :]
        for r_, j in enumerate(sel):
            rm = rec[j]
            for c, age in enumerate(cols):
                a = ax[r_, c]; a.set_xticks([]); a.set_yticks([])
                if age not in rm:
                    a.axis("off"); continue
                gp, pf, pp, crop = rm[age]; cen = ps / 2
                a.imshow(crop, cmap="gray", vmin=0, vmax=255)
                a.add_patch(plt.Rectangle((cen - R, cen - R), 2 * R + 1, 2 * R + 1,
                                          fill=False, ec="0.4", lw=0.8))
                a.plot(cen, cen, "+", color="lime", ms=9, mew=2)
                if pp is not None:
                    a.plot(pp[0] - gp[0] + cen, pp[1] - gp[1] + cen, "x", color="gold", ms=7, mew=2)
                if pf is not None:
                    a.plot(pf[0] - gp[0] + cen, pf[1] - gp[1] + cen, "o", mfc="none", mec="red", ms=9, mew=1.5)
                ds_ = f"1st{off(pf,gp):.1f}" if pf is not None else "1stLOST"
                dp_ = f"prev{off(pp,gp):.1f}" if pp is not None else "prevLOST"
                a.set_title(f"n={age} {ds_}/{dp_}", fontsize=6)
            ax[r_, 0].set_ylabel(f"id {j}", fontsize=8)
        fig.suptitle(f"Mid-Air {args.cond} (EXACT GT): green + truth   red o first-obs KLT   "
                     f"gold x previous-frame KLT", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{args.out}_{args.cond}_montage.png", dpi=130)
        print(f"saved {args.out}_{args.cond}_montage.png")


if __name__ == "__main__":
    main()
