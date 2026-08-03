"""Observable causes of the cumulative flow bias beta.

flow_bias_characterize.py established beta's SHAPE (coherent, ~0.3 px/frame,
accumulates with age) and tested beta-vs-gradient with a SINGLE-PIXEL Sobel
gradient, concluding "not aperture" (median 51 deg). But KLT does not localize
against a single-pixel gradient -- it localizes against the WINDOWED structure
tensor H = sum grad I grad I^T over the (2r+1)^2 patch, on the histeq'd image the
tracker actually sees. The aperture problem lives in the weak eigenvector of THAT
H, not in the raw gradient. This diagnostic redoes the direction test properly and
adds the two leading causal hypotheses for template drift:

  1. Edgeness / aperture: small lambda_min(H) or high condition lambda_max/lambda_min
     => weak localization along the weak eigenvector => beta should align with it.
  2. Motion blur: exposure smears the patch along the motion direction => the
     photometric basin is asymmetric along motion => beta should align with the
     frame-to-frame motion direction, and grow with |motion| / low Laplacian
     variance (blur).

Ground truth for beta is identical to flow_bias_characterize: anchor each track at
its first frame to the 3D scene point (GT Leica depth + Vicon pose); beta(age) =
tracker_pixel - GT_reprojected_pixel (cumulative, beta(0)=0).

Per observation it also records, on the histeq'd image over a (2r+1) window at the
tracker window radius r:
  Sxx,Sxy,Syy      windowed structure tensor  -> lambda_min,lambda_max,cond,weak dir
  lap_var          windowed Laplacian variance (blur proxy; low => blurry)
  mdx,mdy          frame-to-frame motion of this feature (blur/motion direction)
  expo             histeq patch-mean change vs the anchor frame (exposure)
  radius, depth, |omega|, age

Direction tests (the crux):
  ang(beta, weak eigenvector of H)   -- aperture: peaks near 0 if true
  ang(beta, frame-to-frame motion)   -- blur:     peaks near 0 if true
both as per-observation folded (undirected) angles AND as per-track COHERENT
(mean-beta) alignment, because beta's whole signature is that it is coherent.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/flow_bias_causes.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--out flow_bias_causes]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
from flow_gt_eval import quat_ang_rate  # noqa: E402
import echo_li  # noqa: E402


def structure_tensor_fields(eq, r):
    """Windowed structure tensor and Laplacian-variance fields on histeq image eq.

    Returns Sxx,Sxy,Syy (window SUMS of grad products) and lap_var (window
    variance of the Laplacian), all full-resolution, sampled later per feature."""
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    ks = (2 * r + 1, 2 * r + 1)
    # normalize=False => window SUM, i.e. the actual KLT Hessian accumulation.
    sxx = cv2.boxFilter(gx * gx, cv2.CV_32F, ks, normalize=False)
    sxy = cv2.boxFilter(gx * gy, cv2.CV_32F, ks, normalize=False)
    syy = cv2.boxFilter(gy * gy, cv2.CV_32F, ks, normalize=False)
    lap = cv2.Laplacian(eq, cv2.CV_32F, ksize=3)
    m1 = cv2.boxFilter(lap, cv2.CV_32F, ks, normalize=True)
    m2 = cv2.boxFilter(lap * lap, cv2.CV_32F, ks, normalize=True)
    lap_var = np.maximum(m2 - m1 * m1, 0.0)
    mean = cv2.boxFilter(eq.astype(np.float32), cv2.CV_32F, ks, normalize=True)
    return sxx, sxy, syy, lap_var, mean


def eig_sym2(a, b, c):
    """Eigenvalues (lmin<=lmax) and unit weak-eigenvector (of lmin) of [[a,b],[b,c]]."""
    tr = a + c
    d = np.sqrt(np.maximum(((a - c) * 0.5) ** 2 + b * b, 0.0))
    lmax = tr * 0.5 + d
    lmin = tr * 0.5 - d
    # weak eigenvector: eigenvector for lmin. (b, lmin - a) unless degenerate.
    wx = b
    wy = lmin - a
    n = np.hypot(wx, wy)
    deg = n < 1e-12
    # isotropic patch: direction undefined; fall back to (1,0)
    wx = np.where(deg, 1.0, wx / np.where(deg, 1.0, n))
    wy = np.where(deg, 0.0, wy / np.where(deg, 1.0, n))
    return lmin, lmax, wx, wy


def folded_angle(ux, uy, vx, vy):
    """Angle in [0,90] deg between two undirected axes (unit-ish inputs)."""
    un = np.hypot(ux, uy) + 1e-12
    vn = np.hypot(vx, vy) + 1e-12
    cos = np.abs((ux * vx + uy * vy) / (un * vn))
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="flow_bias_causes")
    ap.add_argument("--min-age", type=int, default=5)
    ap.add_argument("--anchor-frames", type=int, default=1)
    ap.add_argument("--window", type=int, default=0, help="structure-tensor radius (0=config klt_window)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

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
    gt_w = quat_ang_rate(gt_t, gt[:, 4:8])

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    r = args.window or int(getattr(fcfg, "klt_window", 7))
    tracker = echo_li.Frontend(fcfg, w, h)

    def cam_pose(t):
        m = np.eye(4)
        m[:3, :3] = slerp(t).as_matrix()
        m[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return m @ t_bs

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f)
        next(rd)
        frames = [(int(row[0]) * 1e-9, idir / row[1].strip()) for row in rd if row]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    anchors = {}      # fid -> (X_world, frame0, anchor_patch_mean)
    anchor_buf = {}   # fid -> list of early backprojected world points
    prev_raw = {}     # fid -> (x, y) previous-frame raw pixel
    per_track = {}    # fid -> list of row tuples
    cols = ["age", "bx", "by", "sxx", "sxy", "syy", "lap_var",
            "mdx", "mdy", "expo", "radius", "depth", "wmag"]

    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats, _ = tracker.process(img)
        eq = cv2.equalizeHist(img)   # tracker uses histeq=global
        sxx, sxy, syy, lap_var, meanf = structure_tensor_fields(eq, r)
        t_wc = cam_pose(t)
        t_cw = np.linalg.inv(t_wc)
        wmag = float(np.interp(t, gt_t, gt_w))
        cur_raw = {}
        if feats:
            und = cv2.undistortPoints(
                np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2),
                K, D, P=K).reshape(-1, 2)
        else:
            und = np.zeros((0, 2))
        for f, (uu, vv) in zip(feats, und):
            fid = int(f["id"])
            rawx, rawy = float(f["x"]), float(f["y"])
            cur_raw[fid] = (rawx, rawy)
            ix, iy = int(round(rawx)), int(round(rawy))
            in_img = 0 <= ix < w and 0 <= iy < h
            if fid not in anchors:
                d0 = zbuf_lookup(zbufs[i], [(uu, vv)], w, h)[0]
                if np.isfinite(d0):
                    pc = np.array([(uu - cx) / fx * d0, (vv - cy) / fy * d0, d0])
                    buf = anchor_buf.setdefault(fid, [])
                    buf.append(t_wc[:3, :3] @ pc + t_wc[:3, 3])
                    if len(buf) >= args.anchor_frames:
                        amean = float(meanf[iy, ix]) if in_img else np.nan
                        anchors[fid] = (np.mean(buf, axis=0), i, amean)
                        per_track[fid] = []
                        del anchor_buf[fid]
                continue
            if not in_img:
                continue
            X, i0, amean = anchors[fid]
            pc1 = t_cw[:3, :3] @ X + t_cw[:3, 3]
            if pc1[2] < 0.1:
                continue
            proj = cv2.projectPoints(pc1.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                     K, D)[0].ravel()
            bx, by = rawx - proj[0], rawy - proj[1]
            pr = prev_raw.get(fid)
            mdx = rawx - pr[0] if pr else np.nan
            mdy = rawy - pr[1] if pr else np.nan
            expo = float(meanf[iy, ix]) - amean
            per_track[fid].append((
                i - i0, bx, by,
                float(sxx[iy, ix]), float(sxy[iy, ix]), float(syy[iy, ix]),
                float(lap_var[iy, ix]), mdx, mdy, expo,
                float(np.hypot(rawx - cx, rawy - cy)), float(pc1[2]), wmag))
        prev_raw = cur_raw
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(per_track)}")

    rows = [row for v in per_track.values() for row in v if row[0] >= 1]
    A = np.array(rows)
    np.savez(args.out + ".npz", d=A, cols=np.array(cols),
             track_lens=np.array([len(v) for v in per_track.values()]))
    print(f"\n=== flow-bias causes ({len(A)} obs, {len(per_track)} tracks), window r={r} ===")

    age = A[:, 0]; bx = A[:, 1]; by = A[:, 2]
    sxx = A[:, 3]; sxy = A[:, 4]; syy = A[:, 5]; lap_var = A[:, 6]
    mdx = A[:, 7]; mdy = A[:, 8]; expo = A[:, 9]
    rad = A[:, 10]; dep = A[:, 11]; wm = A[:, 12]
    bmag = np.hypot(bx, by)
    lmin, lmax, wx, wy = eig_sym2(sxx, sxy, syy)
    cond = lmax / np.maximum(lmin, 1e-6)
    trace = lmax + lmin
    fmag = np.hypot(mdx, mdy)

    # ---- Direction tests (per observation) ----
    ang_weak = folded_angle(bx, by, wx, wy)               # aperture
    have_m = np.isfinite(mdx) & (fmag > 0.05)
    ang_mot = np.full(len(A), np.nan)
    ang_mot[have_m] = folded_angle(bx[have_m], by[have_m], mdx[have_m], mdy[have_m])

    print("\n-- Direction tests (per-observation, folded angle in [0,90]; 0=aligned) --")
    print(f"  ang(beta, weak Hessian eigvec): median {np.nanmedian(ang_weak):.1f} deg  "
          f"frac<30 {100*np.mean(ang_weak<30):.0f}%  frac>60 {100*np.mean(ang_weak>60):.0f}%")
    print(f"    if aperture were the cause this peaks near 0 (beta along weak dir)")
    print(f"  ang(beta, frame-to-frame motion): median {np.nanmedian(ang_mot):.1f} deg  "
          f"frac<30 {100*np.nanmean(ang_mot<30):.0f}%  frac>60 {100*np.nanmean(ang_mot>60):.0f}%")
    print(f"    if motion blur were the cause this peaks near 0 (beta along motion)")

    # SIGNED motion projection: distinguishes tracker LAG (beta against motion,
    # under-shooting displacement) from OVERSHOOT (beta along motion). The folded
    # angle above cannot tell these apart; the sign is the mechanistic tell.
    proj = np.full(len(A), np.nan)
    proj[have_m] = (bx[have_m] * mdx[have_m] + by[have_m] * mdy[have_m]) / fmag[have_m]
    pm = proj[np.isfinite(proj)]
    print(f"  SIGNED beta.motion / |motion|: median {np.median(pm):+.3f} px  "
          f"frac>0(along motion) {100*np.mean(pm>0):.0f}%  "
          f"(>50%% => overshoot/along;  <50%% => lag/against)")

    # weak-direction alignment CONDITIONED on how anisotropic the patch is:
    # aperture should bite only for edge-like (high-condition) patches.
    print("\n-- ang(beta, weak eigvec) vs patch anisotropy (condition number) --")
    qc = np.quantile(cond, np.linspace(0, 1, 6))
    for lo, hi in zip(qc[:-1], qc[1:]):
        m = (cond >= lo) & (cond < hi)
        if m.sum() > 30:
            print(f"  cond in [{lo:7.1f},{hi:7.1f}): n={m.sum():6d}  "
                  f"median ang_weak {np.median(ang_weak[m]):.1f} deg  "
                  f"median |beta| {np.median(bmag[m]):.2f} px")

    # ---- Per-track COHERENT direction test ----
    # beta is coherent; test whether the MEAN beta direction of a track aligns with
    # that track's persistent weak-eigvec / motion direction.
    cmb_weak, cmb_mot, npersist = [], [], 0
    idx = 0
    for v in per_track.values():
        vv = [row for row in v if row[0] >= 1]
        n = len(vv)
        if n < args.min_age:
            idx += n
            continue
        sl = slice(idx, idx + n)
        idx += n
        mb = np.array([bx[sl].mean(), by[sl].mean()])
        if np.hypot(*mb) < 1e-6:
            continue
        npersist += 1
        # persistent weak direction: average outer product then eigvec
        Wxx = np.mean(wx[sl] * wx[sl]); Wxy = np.mean(wx[sl] * wy[sl]); Wyy = np.mean(wy[sl] * wy[sl])
        # dominant axis of the weak-dir distribution
        _, _, dwx, dwy = eig_sym2(Wyy, -Wxy, Wxx)  # eigvec of LARGER -> swap trick
        cmb_weak.append(folded_angle(mb[0], mb[1], dwx, dwy))
        mm = np.array([np.nanmean(mdx[sl]), np.nanmean(mdy[sl])])
        if np.isfinite(mm).all() and np.hypot(*mm) > 0.05:
            cmb_mot.append(folded_angle(mb[0], mb[1], mm[0], mm[1]))
    cmb_weak = np.array(cmb_weak); cmb_mot = np.array(cmb_mot)
    print(f"\n-- Per-track coherent mean-beta alignment (tracks>= {args.min_age}, n={npersist}) --")
    if len(cmb_weak):
        print(f"  ang(mean beta, persistent weak eigvec): median {np.median(cmb_weak):.1f} deg  "
              f"frac<30 {100*np.mean(cmb_weak<30):.0f}%")
    if len(cmb_mot):
        print(f"  ang(mean beta, mean motion dir):        median {np.median(cmb_mot):.1f} deg  "
              f"frac<30 {100*np.mean(cmb_mot<30):.0f}%")

    # ---- |beta| vs each predictor (binned medians) ----
    def binned(x, y, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = np.isfinite(x) & (x >= lo) & (x < hi)
            out.append((0.5 * (lo + hi), np.median(y[m]) if m.sum() > 20 else np.nan, int(m.sum())))
        return np.array(out)

    print("\n-- median |beta| vs candidate causal predictors --")
    preds = [
        ("lambda_min", lmin, np.percentile(lmin, [0, 20, 40, 60, 80, 95, 100])),
        ("condition", cond, np.percentile(cond, [0, 20, 40, 60, 80, 95, 100])),
        ("trace(H)", trace, np.percentile(trace, [0, 20, 40, 60, 80, 95, 100])),
        ("lap_var(blur)", lap_var, np.percentile(lap_var, [0, 20, 40, 60, 80, 95, 100])),
        ("flow_mag", fmag[have_m], np.percentile(fmag[have_m], [0, 20, 40, 60, 80, 95, 100])),
        ("|expo|", np.abs(expo), np.percentile(np.abs(expo), [0, 20, 40, 60, 80, 95, 100])),
        ("|omega|", wm, np.percentile(wm, [0, 20, 40, 60, 80, 95, 100])),
        ("age", age, np.array([1, 5, 10, 20, 40, 80, 1e9])),
    ]
    for name, x, edges in preds:
        yv = bmag[have_m] if name == "flow_mag" else bmag
        b = binned(x, yv, np.unique(edges))
        s = "  ".join(f"{c:.3g}:{m:.2f}" for c, m, n in b if np.isfinite(m))
        print(f"  {name:>14}: {s}")

    # ---- Spearman-ish rank correlation of each predictor with |beta| ----
    from scipy.stats import rankdata
    # Age dominates (rho~0.7) and correlates with the other predictors, so a raw
    # rank corr is confounded by age. Report BOTH the raw rank corr and an
    # age-controlled one: residualize |beta| and the predictor against the age-bin
    # median, then correlate the residuals. This isolates the signal beyond age.
    age_edges = np.array([1, 5, 10, 20, 40, 80, 160, 1e9])
    age_bin = np.digitize(age, age_edges)

    def residualize(x):
        out = np.full_like(x, np.nan, dtype=float)
        for b in np.unique(age_bin):
            m = (age_bin == b) & np.isfinite(x)
            if m.sum() > 20:
                out[m] = x[m] - np.median(x[m])
        return out

    rb = residualize(bmag)
    print("\n-- rank correlation of predictor with |beta|:  raw  |  age-controlled --")
    for name, x in [("lambda_min", lmin), ("condition", cond), ("trace(H)", trace),
                    ("lap_var", lap_var), ("|expo|", np.abs(expo)),
                    ("|omega|", wm), ("age", age), ("flow_mag", fmag)]:
        g = np.isfinite(x) & np.isfinite(bmag)
        if g.sum() < 100:
            continue
        rho_raw = np.corrcoef(rankdata(x[g]), rankdata(bmag[g]))[0, 1]
        rx = residualize(x)
        gg = np.isfinite(rx) & np.isfinite(rb)
        rho_age = (np.corrcoef(rankdata(rx[gg]), rankdata(rb[gg]))[0, 1]
                   if gg.sum() > 100 else np.nan)
        print(f"  {name:>12}: {rho_raw:+.3f}  |  {rho_age:+.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(ang_weak, bins=45)
    ax[0, 0].axvline(np.median(ang_weak), color="r", ls="--")
    ax[0, 0].set_title("ang(beta, weak Hessian eigvec)\n0=aperture-aligned")
    ax[0, 0].set_xlabel("deg")
    ax[0, 1].hist(ang_mot[np.isfinite(ang_mot)], bins=45)
    ax[0, 1].axvline(np.nanmedian(ang_mot), color="r", ls="--")
    ax[0, 1].set_title("ang(beta, frame-to-frame motion)\n0=blur-aligned")
    ax[0, 1].set_xlabel("deg")
    if len(cmb_weak):
        ax[0, 2].hist(cmb_weak, bins=30, alpha=0.6, label="vs weak eigvec")
    if len(cmb_mot):
        ax[0, 2].hist(cmb_mot, bins=30, alpha=0.6, label="vs motion")
    ax[0, 2].set_title("per-track mean-beta alignment")
    ax[0, 2].set_xlabel("deg"); ax[0, 2].legend(fontsize=8)
    bc = binned(cond, bmag, np.percentile(cond, np.linspace(0, 100, 9)))
    ax[1, 0].plot(bc[:, 0], bc[:, 1], "o-"); ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("median |beta| vs condition"); ax[1, 0].set_xlabel("lambda_max/lambda_min")
    ax[1, 0].set_ylabel("|beta| [px]")
    bl = binned(lap_var, bmag, np.percentile(lap_var, np.linspace(0, 100, 9)))
    ax[1, 1].plot(bl[:, 0], bl[:, 1], "o-")
    ax[1, 1].set_title("median |beta| vs Laplacian var (blur)"); ax[1, 1].set_xlabel("lap_var")
    bf = binned(fmag[have_m], bmag[have_m], np.percentile(fmag[have_m], np.linspace(0, 100, 9)))
    ax[1, 2].plot(bf[:, 0], bf[:, 1], "o-")
    ax[1, 2].set_title("median |beta| vs frame-to-frame flow"); ax[1, 2].set_xlabel("|motion| [px]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=120)
    print(f"\nsaved {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
