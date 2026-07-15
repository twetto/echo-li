"""Is descriptor-based localization WHITE (non-accumulating), unlike KLT's coherent β?

Everyone (LARVIO, vilib/SVO, this repo's FAST-RI-LBP) does KLT-track + descriptor-
VERIFY; nobody uses the descriptor as the subpixel measurement. This tests the one
property that would justify doing so: KLT's β is a coherent, age-accumulating walk
(direction autocorr 0.98, slope ~0.3 px/frame) that biases depth and cannot average
down. A descriptor detector re-localizes each corner *independently* every frame, so
its error should be WHITE (autocorr ≈ 0) and flat vs age — larger per-frame, but it
averages down in the filter, fixing the depth *bias*, not just its covariance.

Measurement: the production tracker supplies track identity + the first-frame
Leica/Vicon GT anchor + KLT β (production_px − GT_px). Each frame we detect ORB, and
for each active track pick the ORB keypoint within a window of the GT-predicted pixel
whose descriptor best matches the track's first-observation ORB descriptor; its
position gives the ORB localization error (orb_px − GT_px). We compare ORB error and
KLT β on the SAME (track, frame) samples: magnitude vs age, drift slope, and per-track
direction persistence / consecutive-direction autocorrelation. Hamming distance vs age
also reports descriptor staleness.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/orb_vs_klt_drift.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--out orb_drift]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
import echo_li  # noqa: E402

_BITS = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(ref, cands):
    """Hamming distance of a (32,) uint8 descriptor to (M,32) candidates -> (M,)."""
    return _BITS[np.bitwise_xor(cands, ref[None, :])].sum(1)


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="orb_drift")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--assoc-window", type=float, default=6.0,
                    help="descriptor association window around the GT pixel [px]")
    ap.add_argument("--birth-window", type=float, default=3.0,
                    help="max dist from birth pixel to accept an ORB reference kp [px]")
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
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)
    orb = cv2.ORB_create(nfeatures=args.nfeatures)

    anchors = {}       # fid -> (X_world, birth_i)
    ref_desc = {}      # fid -> (32,) uint8 first-observation ORB descriptor
    per_track = {}     # fid -> list of (age, orb_dx, orb_dy, klt_dx, klt_dy, hamming)

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

        kps, desc = orb.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            continue
        kpx = np.array([kp.pt for kp in kps], np.float64)      # (M,2)
        tree = cKDTree(kpx)

        t_cw = np.linalg.inv(cam_pose(t))
        for fid in alive:
            if fid not in anchors:
                continue
            pc = t_cw[:3, :3] @ anchors[fid][0] + t_cw[:3, 3]
            if pc[2] < 0.1:
                continue
            gtp = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                    K, D)[0].ravel()
            idxs = tree.query_ball_point(gtp, args.assoc_window)
            if not idxs or fid not in ref_desc:
                continue
            idxs = np.array(idxs)
            hd = hamming(ref_desc[fid], desc[idxs])
            best = idxs[np.argmin(hd)]
            orb_err = kpx[best] - gtp
            klt_err = raw[fid] - gtp
            age = i - anchors[fid][1]
            if age >= 1:
                per_track.setdefault(fid, []).append(
                    (age, orb_err[0], orb_err[1], klt_err[0], klt_err[1], float(hd.min())))

        # births: anchor + store first-observation ORB descriptor
        for fid in alive:
            if fid in anchors:
                continue
            uv = und_by_id[fid]
            d0 = zbuf_lookup(zbufs[i], [uv], w, h)[0]
            if not np.isfinite(d0):
                continue
            pc = np.array([(uv[0] - cx) / fx * d0, (uv[1] - cy) / fy * d0, d0])
            t_wc = cam_pose(t)
            dd, jj = tree.query(raw[fid])
            if dd > args.birth_window:
                continue                      # no ORB kp co-located with this feature
            anchors[fid] = (t_wc[:3, :3] @ pc + t_wc[:3, 3], i)
            ref_desc[fid] = desc[jj]
        for fid in [f for f in anchors if f not in alive]:
            del anchors[fid]; ref_desc.pop(fid, None)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(anchors)} "
                  f"orb_kps={len(kps)} scored={sum(len(v) for v in per_track.values())}")

    rows = [r for v in per_track.values() for r in v]
    A = np.array(rows)
    np.savez(args.out + ".npz", d=A,
             cols=np.array(["age", "orb_dx", "orb_dy", "klt_dx", "klt_dy", "hamming"]))
    age = A[:, 0]
    orb = np.hypot(A[:, 1], A[:, 2]); klt = np.hypot(A[:, 3], A[:, 4])
    ham = A[:, 5]
    print(f"\n=== ORB vs KLT drift ({len(A)} obs, {len(per_track)} tracks) ===")
    print(f"  overall median |err|:  ORB {np.median(orb):.3f} px   KLT {np.median(klt):.3f} px")
    print(f"  overall p90    |err|:  ORB {np.percentile(orb,90):.3f} px   "
          f"KLT {np.percentile(klt,90):.3f} px")
    print(f"  descriptor hamming vs first obs: median {np.median(ham):.0f}/256  "
          f"p90 {np.percentile(ham,90):.0f}")

    age_edges = np.array([1, 5, 10, 20, 40, 80, 160, 1e9])
    print("\n-- median |err| vs age (does it accumulate?) --")
    print(f"  {'age':>10}  {'n':>7}  {'ORB':>6}  {'KLT':>6}  {'hamming':>7}")
    for lo, hi in zip(age_edges[:-1], age_edges[1:]):
        m = (age >= lo) & (age < hi)
        if m.sum() > 20:
            print(f"  {int(lo):4d}-{int(min(hi,9999)):>4d}  {m.sum():7d}  "
                  f"{np.median(orb[m]):6.2f}  {np.median(klt[m]):6.2f}  {np.median(ham[m]):7.0f}")

    # per-track temporal structure: is the error a coherent walk or white?
    def temporal(dx_i, dy_i):
        # persistence/dir-autocorr are on the raw error (contaminated by the constant
        # per-track GT-anchor offset). inc_autocorr is on frame-to-frame INCREMENTS,
        # which cancel that offset: coherent drift => aligned increments (>0);
        # white noise => anti-correlated increments (~-0.5). slope = accumulation rate.
        persist, autoc, slope, incac = [], [], [], []
        for v in per_track.values():
            if len(v) < 8:
                continue
            a = np.array([r[0] for r in v])
            b = np.array([(r[dx_i], r[dy_i]) for r in v])
            mag = np.hypot(b[:, 0], b[:, 1])
            persist.append(np.hypot(*b.mean(0)) / (mag.mean() + 1e-9))
            u = b / (mag[:, None] + 1e-9)
            autoc.append(np.mean(np.sum(u[1:] * u[:-1], axis=1)))
            slope.append(np.polyfit(a, mag, 1)[0])
            db = np.diff(b, axis=0)
            dm = np.hypot(db[:, 0], db[:, 1])
            ud = db / (dm[:, None] + 1e-9)
            if len(ud) > 1:
                incac.append(np.mean(np.sum(ud[1:] * ud[:-1], axis=1)))
        return (np.median(persist), np.median(autoc), np.median(slope),
                np.median(incac) if incac else np.nan)

    # --- Is n_klt correlated with n_orb? (the fusion cross-covariance C) ---
    # Raw error corr is contaminated by the shared GT-anchor offset + KLT's coherent
    # beta (both common-mode, inflate correlation). Frame-to-frame INCREMENTS cancel
    # the constant anchor offset and the slow beta, leaving the noise correlation.
    def comp_corr(A, B):
        a = np.concatenate([A[:, 0], A[:, 1]]); b = np.concatenate([B[:, 0], B[:, 1]])
        return np.corrcoef(a, b)[0, 1]
    Ks, Os, dKs, dOs = [], [], [], []
    for v in per_track.values():
        vs = sorted(v)
        Kv = np.array([(r[3], r[4]) for r in vs]); Ov = np.array([(r[1], r[2]) for r in vs])
        Ks.append(Kv); Os.append(Ov)
        if len(vs) >= 2:
            dKs.append(np.diff(Kv, 0)); dOs.append(np.diff(Ov, 0))
    K = np.vstack(Ks); O = np.vstack(Os); dK = np.vstack(dKs); dO = np.vstack(dOs)
    raw_c = comp_corr(K, O)
    inc_c = comp_corr(dK, dO)
    print("\n-- KLT/ORB error correlation (fusion cross-covariance C) --")
    print(f"  raw error corr (klt vs orb):        {raw_c:+.3f}  (contaminated: shares anchor offset + beta)")
    print(f"  increment corr (Delta klt vs orb):  {inc_c:+.3f}  (~ noise correlation rho; ~0 => independent)")

    op, oa, os_, oi = temporal(1, 2)
    kp, ka, ks, ki = temporal(3, 4)
    print("\n-- per-track error structure (coherent walk vs white noise) --")
    print(f"  {'':>6} {'persist':>8} {'dir-acorr':>10} {'slope/fr':>9} {'inc-acorr':>10}")
    print(f"  {'ORB':>6} {op:8.2f} {oa:10.2f} {os_:9.3f} {oi:10.2f}")
    print(f"  {'KLT':>6} {kp:8.2f} {ka:10.2f} {ks:9.3f} {ki:10.2f}")
    print("  inc-acorr: coherent drift > 0 (aligned steps); white ~ -0.5 (anti-corr).")
    print("  persist/dir-acorr are inflated for BOTH by the constant anchor offset.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    centers, om, km = [], [], []
    for lo, hi in zip(age_edges[:-2], age_edges[1:-1]):
        m = (age >= lo) & (age < hi)
        if m.sum() > 20:
            centers.append(0.5 * (lo + hi)); om.append(np.median(orb[m])); km.append(np.median(klt[m]))
    ax[0].plot(centers, om, "o-", label="ORB (descriptor)")
    ax[0].plot(centers, km, "s-", label="KLT (production)")
    ax[0].set_xlabel("track age [frames]"); ax[0].set_ylabel("median |error| [px]")
    ax[0].set_title("localization error vs age"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].scatter(np.clip(klt, 0, 8), np.clip(orb, 0, 8), s=2, alpha=0.08)
    ax[1].plot([0, 8], [0, 8], "k--", lw=1)
    ax[1].set_xlabel("KLT |β| [px]"); ax[1].set_ylabel("ORB |err| [px]")
    ax[1].set_title("paired per-observation error"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=120)
    print(f"\nsaved {args.out}.png / .npz")


if __name__ == "__main__":
    main()
