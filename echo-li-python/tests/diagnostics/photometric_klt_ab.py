"""Self-propagating KLT A/B harness for the flow-bias β, with two experiments.

β is a *compounding* bias: it accumulates because the tracker feeds its OWN drifted
position/template forward. So candidate fixes must be tested with self-propagating
trackers (single-step re-tracking is already sub-pixel for every mode and shows
nothing). All modes share the production tracker's births/lifetimes (previous-frame
template policy) and a common forward-backward safety gate, and each carries its own
propagated position; cumulative β = tracked − GT-reprojected pixel (first-frame
Leica/Vicon anchor). Modes are scored PAIRED on the intersection of tracks all modes
keep, so survivorship (each mode's FB gate dropping a different subset) is removed.

--experiment photometric  (does an exposure-compensated residual remove β? -> no):
  ssd       r = I_w − T                    baseline (brightness constancy)
  zeromean  r = (I_w−mean) − (T−mean)      invariant to additive offset
  gainbias  project out {1,T}              invariant to gain·T+offset (== ZNCC)

--experiment joint  (does a first-observation REFERENCE anchor reduce drift?):
  ssd       baseline
  B<λ>      version B: joint 2D + fixed IDENTITY reference. Closed form == tracking
            against the blended template (T_prev + λ·T_ref)/(1+λ).
  C<λ>      version C: joint 2D + ORACLE affine reference. Two-block Gauss-Newton;
            the reference patch is compared at u + A_ref·p with A_ref the GT patch
            affine (Vicon pose + Leica depth) from the birth frame to the current
            frame. Translation stays pinned by the previous-frame term. This is the
            decisive UPPER BOUND: if a *perfect* reference shape can't beat baseline,
            version E (which only estimates A_ref) is dead; if it can, E is worth it.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/photometric_klt_ab.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult --experiment joint
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
import echo_li  # noqa: E402


def build_pyramid(img, nlev, histeq=True):
    """f32 pyramid (optionally histeq'd, matching the tracker) + Sobel gradients."""
    base = (cv2.equalizeHist(img) if histeq else img).astype(np.float32)
    pyr, gx, gy = [base], [], []
    for _ in range(nlev - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    for lv in pyr:
        gx.append(cv2.Sobel(lv, cv2.CV_32F, 1, 0, ksize=3) * 0.125)
        gy.append(cv2.Sobel(lv, cv2.CV_32F, 0, 1, ksize=3) * 0.125)
    return pyr, gx, gy


def sample_abs(img, px, py):
    """Batched bilinear sample at absolute coords px,py (both (N,P)) -> (N,P)."""
    h, w = img.shape
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    fx = px - x0
    fy = py - y0
    x0 = np.clip(x0, 0, w - 2); y0 = np.clip(y0, 0, h - 2)
    x1 = x0 + 1; y1 = y0 + 1
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy)
            + Ic * (1 - fx) * fy + Id * fx * fy)


def sample(img, cx, cy, offx, offy):
    """Bilinear sample of a shared offset grid. cx,cy (N,); offx,offy (P,) -> (N,P)."""
    return sample_abs(img, cx[:, None] + offx[None, :], cy[:, None] + offy[None, :])


def project(photo, jx, jy, r, Tc):
    """Photometric projection of residual r and Jacobian cols jx,jy (see module doc)."""
    if photo == "ssd":
        return r, jx, jy
    r = r - r.mean(1, keepdims=True)
    jx = jx - jx.mean(1, keepdims=True)
    jy = jy - jy.mean(1, keepdims=True)
    if photo == "zeromean":
        return r, jx, jy
    denom = np.sum(Tc * Tc, 1, keepdims=True) + 1e-6
    r = r - (np.sum(r * Tc, 1, keepdims=True) / denom) * Tc
    jx = jx - (np.sum(jx * Tc, 1, keepdims=True) / denom) * Tc
    jy = jy - (np.sum(jy * Tc, 1, keepdims=True) / denom) * Tc
    return r, jx, jy


def klt_track(prev_pyr, cur_pyr, cur_gx, cur_gy, p0, r, iters,
              kind="ssd", photo="ssd", lam=0.0, t_ref=None, a_ref=None):
    """Batched pyramidal forward-additive translation KLT for N features.

    kind: 'ssd'|'zeromean'|'gainbias' (single block, photometric),
          'blendB' (single block vs blended prev/reference template),
          'oracleC' (two-block: prev term + oracle-affine reference term).
    t_ref: (N,nlev,P) stored first-observation patches. a_ref: (N,2,2) GT patch affine.
    Returns (N,2) current positions, valid mask, and |expo|=mean(I_w)-mean(T_prev)."""
    n = len(p0)
    off = np.arange(-r, r + 1, dtype=np.float32)
    offx = np.repeat(off, len(off))
    offy = np.tile(off, len(off))
    u = p0.copy().astype(np.float64)
    valid = np.ones(n, bool)
    expo = np.zeros(n)
    nlev = len(prev_pyr)
    for lv in reversed(range(nlev)):
        s = 0.5 ** lv
        Pl, Cl, Gx, Gy = prev_pyr[lv], cur_pyr[lv], cur_gx[lv], cur_gy[lv]
        c0x = (p0[:, 0] * s).astype(np.float64)
        c0y = (p0[:, 1] * s).astype(np.float64)
        Tprev = sample(Pl, c0x, c0y, offx, offy)
        if kind == "blendB":
            T = (Tprev + lam * t_ref[:, lv, :]) / (1.0 + lam)
        else:
            T = Tprev
        Tc = T - T.mean(1, keepdims=True)
        if kind == "oracleC":
            # warped reference offsets (A_ref is scale-free; offsets in level pixels)
            wox = a_ref[:, 0, 0][:, None] * offx[None, :] + a_ref[:, 0, 1][:, None] * offy[None, :]
            woy = a_ref[:, 1, 0][:, None] * offx[None, :] + a_ref[:, 1, 1][:, None] * offy[None, :]
            Tr = t_ref[:, lv, :]
        ux = u[:, 0] * s; uy = u[:, 1] * s
        for _ in range(iters):
            Iw = sample(Cl, ux, uy, offx, offy)
            jx = sample(Gx, ux, uy, offx, offy)
            jy = sample(Gy, ux, uy, offx, offy)
            res = Iw - T
            if kind == "oracleC":
                pr, pjx, pjy = res, jx, jy                       # prev block (ssd)
                Iwr = sample_abs(Cl, ux[:, None] + wox, uy[:, None] + woy)
                jxr = sample_abs(Gx, ux[:, None] + wox, uy[:, None] + woy)
                jyr = sample_abs(Gy, ux[:, None] + wox, uy[:, None] + woy)
                rr = Iwr - Tr
                Hxx = np.sum(pjx * pjx, 1) + lam * np.sum(jxr * jxr, 1)
                Hxy = np.sum(pjx * pjy, 1) + lam * np.sum(jxr * jyr, 1)
                Hyy = np.sum(pjy * pjy, 1) + lam * np.sum(jyr * jyr, 1)
                bx = -(np.sum(pjx * pr, 1) + lam * np.sum(jxr * rr, 1))
                by = -(np.sum(pjy * pr, 1) + lam * np.sum(jyr * rr, 1))
            else:
                pr, pjx, pjy = project(photo, jx, jy, res, Tc)
                Hxx = np.sum(pjx * pjx, 1); Hxy = np.sum(pjx * pjy, 1); Hyy = np.sum(pjy * pjy, 1)
                bx = -np.sum(pjx * pr, 1); by = -np.sum(pjy * pr, 1)
            reg = 1e-3 * (Hxx + Hyy + 1e-6)
            Hxx = Hxx + reg; Hyy = Hyy + reg
            det = Hxx * Hyy - Hxy * Hxy
            ok = np.abs(det) > 1e-6
            dx = np.where(ok, (Hyy * bx - Hxy * by) / np.where(ok, det, 1), 0.0)
            dy = np.where(ok, (Hxx * by - Hxy * bx) / np.where(ok, det, 1), 0.0)
            step = np.hypot(dx, dy)
            scl = np.where(step > 1.0, 1.0 / np.maximum(step, 1e-12), 1.0)
            ux = ux + dx * scl; uy = uy + dy * scl
        expo = sample(Cl, ux, uy, offx, offy).mean(1) - Tprev.mean(1)
        u[:, 0] = ux / s; u[:, 1] = uy / s
    hh, ww = prev_pyr[0].shape
    valid &= (u[:, 0] > r) & (u[:, 0] < ww - r) & (u[:, 1] > r) & (u[:, 1] < hh - r)
    return u, valid, expo


def make_spec(experiment):
    """Return (mode_names, spec) where spec[name] = (kind, photo, lam)."""
    if experiment == "joint":
        modes = ["ssd", "B0.50", "B1.00", "C0.50", "C1.00"]
        spec = {"ssd": ("ssd", "ssd", 0.0),
                "B0.50": ("blendB", "ssd", 0.5), "B1.00": ("blendB", "ssd", 1.0),
                "C0.50": ("oracleC", "ssd", 0.5), "C1.00": ("oracleC", "ssd", 1.0)}
    else:
        modes = ["ssd", "zeromean", "gainbias"]
        spec = {m: (m, m, 0.0) for m in modes}
    return modes, spec


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--experiment", choices=["photometric", "joint"], default="photometric")
    ap.add_argument("--out", default="photo_klt")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--fb", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    MODES, SPEC = make_spec(args.experiment)
    need_ref = any(SPEC[m][0] in ("blendB", "oracleC") for m in MODES)
    need_aref = any(SPEC[m][0] == "oracleC" for m in MODES)

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = dcoef[:4]

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))

    def cam_pose(t):
        m = np.eye(4)
        m[:3, :3] = slerp(t).as_matrix()
        m[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return m @ t_bs

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(row[0]) * 1e-9, idir / row[1].strip()) for row in rd if row]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.klt_warp = "translation"
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    r = args.radius
    off = np.arange(-r, r + 1, dtype=np.float32)
    OFFX = np.repeat(off, len(off))
    OFFY = np.tile(off, len(off))

    anchors = {}            # fid -> (X_world, birth_i)
    pos = {m: {} for m in MODES}
    ref_patch = {}          # fid -> (nlev,P) first-observation patches
    ref_geom = {}           # fid -> (u0, v0, d0, R0(3,3), t0(3)) undistorted birth geom
    rows = []               # (age, mode_idx, beta, expo, frame, id); mode -1 = prod
    prev = None
    tstart = time.time()

    def oracle_aref(fids, t):
        """GT 2x2 patch affine birth->current for each fid (frontal-planar model)."""
        t_cw1 = np.linalg.inv(cam_pose(t))
        R1, t1 = t_cw1[:3, :3], t_cw1[:3, 3]
        u0 = np.array([ref_geom[f][0] for f in fids])
        v0 = np.array([ref_geom[f][1] for f in fids])
        d0 = np.array([ref_geom[f][2] for f in fids])
        R0 = np.array([ref_geom[f][3] for f in fids])          # (M,3,3)
        t0 = np.array([ref_geom[f][4] for f in fids])          # (M,3)
        d = 3.0
        cols = []
        for ox, oy in [(0.0, 0.0), (d, 0.0), (0.0, d)]:
            pc0 = np.stack([((u0 + ox - cx) / fx) * d0,
                            ((v0 + oy - cy) / fy) * d0, d0], axis=1)   # (M,3)
            xw = np.einsum("mij,mj->mi", R0, pc0) + t0
            pc1 = xw @ R1.T + t1
            uu = fx * pc1[:, 0] / pc1[:, 2] + cx
            vv = fy * pc1[:, 1] / pc1[:, 2] + cy
            cols.append((uu, vv))
        a = np.empty((len(fids), 2, 2))
        a[:, 0, 0] = (cols[1][0] - cols[0][0]) / d
        a[:, 1, 0] = (cols[1][1] - cols[0][1]) / d
        a[:, 0, 1] = (cols[2][0] - cols[0][0]) / d
        a[:, 1, 1] = (cols[2][1] - cols[0][1]) / d
        return {f: a[k] for k, f in enumerate(fids)}

    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        raw = {int(f["id"]): np.array([float(f["x"]), float(f["y"])]) for f in feats}
        alive = set(raw)
        if feats:
            und = cv2.undistortPoints(
                np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2),
                K, D, P=K).reshape(-1, 2)
        else:
            und = np.zeros((0, 2))
        und_by_id = {int(f["id"]): uv for f, uv in zip(feats, und)}
        pyr, gxp, gyp = build_pyramid(img, args.levels, histeq=not args.raw)

        t_cw = np.linalg.inv(cam_pose(t))
        gtpix = {}
        for fid in alive:
            if fid in anchors:
                pc = t_cw[:3, :3] @ anchors[fid][0] + t_cw[:3, 3]
                if pc[2] > 0.1:
                    gtpix[fid] = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3),
                                                   np.zeros(3), K, D)[0].ravel()
        aref_frame = None

        if prev is not None:
            prev_pyr, pgx, pgy = prev
            for mi, m in enumerate(MODES):
                kind, photo, lam = SPEC[m]
                fids = [fid for fid in pos[m] if fid in alive]
                if not fids:
                    continue
                p0 = np.array([pos[m][fid] for fid in fids], np.float64)
                tref = (np.array([ref_patch[fid] for fid in fids])
                        if kind in ("blendB", "oracleC") else None)
                aref = None
                if kind == "oracleC":
                    if aref_frame is None:
                        aref_frame = oracle_aref([f for f in alive if f in ref_geom], t)
                    aref = np.array([aref_frame[fid] for fid in fids])
                u, valid, expo = klt_track(prev_pyr, pyr, pgx, pgy, p0, r, args.iters,
                                           kind, photo, lam, tref, aref)
                # neutral forward-backward gate (plain geometric round-trip)
                uback, vb, _ = klt_track(pyr, prev_pyr, pgx, pgy, u, r, args.iters,
                                         "ssd", "ssd", 0.0)
                fbok = np.hypot(*(uback - p0).T) <= args.fb
                valid = valid & vb & fbok
                for k, fid in enumerate(fids):
                    if not valid[k]:
                        del pos[m][fid]
                        continue
                    pos[m][fid] = u[k]
                    if fid in gtpix and fid in anchors:
                        age = i - anchors[fid][1]
                        if age >= 1:
                            b = float(np.hypot(*(u[k] - gtpix[fid])))
                            rows.append((age, mi, b, float(abs(expo[k])), i, fid))
            for fid in alive:
                if fid in gtpix and fid in anchors:
                    age = i - anchors[fid][1]
                    if age >= 1:
                        rows.append((age, -1, float(np.hypot(*(raw[fid] - gtpix[fid]))),
                                     np.nan, i, fid))

        # drop dead
        for fid in [f for f in anchors if f not in alive]:
            del anchors[fid]
            ref_patch.pop(fid, None); ref_geom.pop(fid, None)
        for m in MODES:
            pos[m] = {fid: q for fid, q in pos[m].items() if fid in alive}
        # births (shared across modes)
        newb = []
        for fid in alive:
            if fid in anchors:
                continue
            uv = und_by_id[fid]
            d0 = zbuf_lookup(zbufs[i], [uv], w, h)[0]
            if not np.isfinite(d0):
                continue
            pc = np.array([(uv[0] - cx) / fx * d0, (uv[1] - cy) / fy * d0, d0])
            t_wc = cam_pose(t)
            anchors[fid] = (t_wc[:3, :3] @ pc + t_wc[:3, 3], i)
            for m in MODES:
                pos[m][fid] = raw[fid].copy()
            if need_aref:
                ref_geom[fid] = (uv[0], uv[1], float(d0), t_wc[:3, :3].copy(), t_wc[:3, 3].copy())
            if need_ref:
                newb.append(fid)
        if newb:
            bx = np.array([raw[fid][0] for fid in newb])
            by = np.array([raw[fid][1] for fid in newb])
            patches = np.stack([sample(pyr[lv], bx * (0.5 ** lv), by * (0.5 ** lv), OFFX, OFFY)
                                for lv in range(args.levels)], axis=1)  # (M,nlev,P)
            for k, fid in enumerate(newb):
                ref_patch[fid] = patches[k]

        prev = (pyr, gxp, gyp)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] rows={len(rows)} tracks={len(anchors)} "
                  f"{i/max(time.time()-tstart,1e-9):.1f}fps")

    A = np.array(rows)
    cols = ["age", "mode", "beta", "expo", "frame", "id"]
    np.savez(args.out + ".npz", d=A, cols=np.array(cols),
             modes=np.array(["prod"] + MODES), experiment=args.experiment)
    age = A[:, 0]; mode = A[:, 1].astype(int); beta = A[:, 2]; expo = A[:, 3]
    key = (A[:, 4].astype(np.int64) << 32) | A[:, 5].astype(np.int64)
    named = [("prod", -1)] + [(m, i) for i, m in enumerate(MODES)]

    print(f"\n=== {args.experiment} KLT ({len(A)} scored obs) ===")
    print(f"{'mode':>10} {'n':>8} {'medianB':>8} {'p90B':>8} {'meanB':>8}")
    for name, mi in named:
        e = beta[mode == mi]
        if len(e):
            print(f"{name:>10} {len(e):8d} {np.median(e):8.3f} "
                  f"{np.percentile(e,90):8.3f} {np.mean(e):8.3f}")

    # ---- PAIRED vs baseline (mode 0 = ssd), survivorship-free ----
    dmap = {mi: dict(zip(key[mode == mi], beta[mode == mi])) for mi in range(len(MODES))}
    agemap = dict(zip(key[mode == 0], age[mode == 0]))
    age_edges = np.array([1, 5, 10, 20, 40, 80, 1e9])
    hdr = "  ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(age_edges[:-1], age_edges[1:-1]))
    print("\n-- PAIRED vs ssd (same tracks): overall + by age (median b_mode - b_ssd) --")
    print(f"  {'mode':>7} {'ncommon':>7} {'med_d':>8} {'bett%':>6} | by age: {hdr}  80+")
    for mi in range(1, len(MODES)):
        common = np.array(sorted(set(dmap[0]) & set(dmap[mi])), dtype=np.int64)
        if len(common) == 0:
            continue
        b0 = np.array([dmap[0][k] for k in common])
        bm = np.array([dmap[mi][k] for k in common])
        ag = np.array([agemap[k] for k in common])
        d = bm - b0
        cells = []
        for lo, hi in zip(age_edges[:-1], age_edges[1:]):
            s = (ag >= lo) & (ag < hi)
            cells.append(f"{np.median(d[s]):+5.3f}" if s.sum() > 20 else "   .  ")
        print(f"  {MODES[mi]:>7} {len(common):7d} {np.median(d):+8.4f} "
              f"{100*np.mean(d<0):5.1f} | {'  '.join(cells)}")

    print("\n-- population median |beta| vs age --")
    print(f"  {'mode':>10}  {hdr}  80+")
    for name, mi in named:
        cells = []
        for lo, hi in zip(age_edges[:-1], age_edges[1:]):
            s = (mode == mi) & (age >= lo) & (age < hi)
            cells.append(f"{np.median(beta[s]):5.2f}" if s.sum() > 20 else "  .  ")
        print(f"  {name:>10}  " + "  ".join(cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [0.5 * (lo + hi) for lo, hi in zip(age_edges[:-2], age_edges[1:-1])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, mi in named:
        ys = []
        for lo, hi in zip(age_edges[:-2], age_edges[1:-1]):
            s = (mode == mi) & (age >= lo) & (age < hi)
            ys.append(np.median(beta[s]) if s.sum() > 20 else np.nan)
        ax[0].plot(centers, ys, "o-", label=name)
    ax[0].set_xlabel("track age [frames]"); ax[0].set_ylabel("median |beta| [px]")
    ax[0].set_title(f"{args.experiment}: drift vs age"); ax[0].legend(); ax[0].grid(alpha=0.3)
    for mi in range(1, len(MODES)):
        common = np.array(sorted(set(dmap[0]) & set(dmap[mi])), dtype=np.int64)
        if not len(common):
            continue
        b0 = np.array([dmap[0][k] for k in common]); bm = np.array([dmap[mi][k] for k in common])
        ag = np.array([agemap[k] for k in common]); d = bm - b0
        ys = []
        for lo, hi in zip(age_edges[:-2], age_edges[1:-1]):
            s = (ag >= lo) & (ag < hi)
            ys.append(np.median(d[s]) if s.sum() > 20 else np.nan)
        ax[1].plot(centers, ys, "s-", label=MODES[mi])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("track age [frames]"); ax[1].set_ylabel("paired median(mode−ssd) [px]")
    ax[1].set_title("paired Δ vs baseline (negative = better)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=120)
    print(f"\nsaved {args.out}.png / .npz")


if __name__ == "__main__":
    main()
