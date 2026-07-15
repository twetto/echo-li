"""VIO-prior version of orb_vs_klt_drift: does a REAL VIO pose prior gate the
descriptor association well enough (not the GT oracle)?

orb_vs_klt_drift.py centered the ORB association window on the GT-projected pixel
(oracle) -- which both caps the error at the window and hides re-detection failures.
This runs a real echo_li.VIOFilter alongside (IMU + the production tracker's vision,
exactly as tail_cue_analysis) and centers the association window on the VIO-PREDICTED
pixel, computed from the RELATIVE VIO pose birth->current (gauge-consistent) applied to
the landmark's birth-frame 3D point (GT depth, to isolate the POSE prior). It reports:

  pred_err   |VIO-predicted pixel - GT pixel|   -- the actual prior quality, vs age
  recall     fraction of (track,frame) with a descriptor-matched ORB kp in the window
  precision  fraction of those matches within 3 px of GT (correct association)
  orb/klt error + increment-autocorrelation on the VIO-gated matches (does the white,
             bounded property survive a real prior?)

Errors are always measured against GT (Vicon+Leica); only the association GATE uses VIO.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/orb_vs_klt_drift_vio.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult [--window 15 --out orb_vio]
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
    return _BITS[np.bitwise_xor(cands, ref[None, :])].sum(1)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson_px(x0, x1, E, f):
    """Sampson epipolar distance (px) of normalized bearings x0->x1 under E."""
    ex0 = E @ x0
    etx1 = E.T @ x1
    num = (x1 @ ex0) ** 2
    den = ex0[0] ** 2 + ex0[1] ** 2 + etx1[0] ** 2 + etx1[1] ** 2
    return f * np.sqrt(num / max(den, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="orb_vio")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--window", type=float, default=15.0,
                    help="association window around the chosen center [px]")
    ap.add_argument("--center", choices=["klt", "vio", "gt"], default="klt",
                    help="where to center the descriptor search: current KLT position "
                         "(realistic; off by bounded beta), VIO birth-integrated "
                         "prediction, or GT (oracle)")
    ap.add_argument("--epi-thr", type=float, default=999.0,
                    help="prior-epipolar (RANSAC-equivalent) gate: reject a descriptor "
                         "match whose Sampson distance to the birth bearing under the "
                         "relative pose exceeds this [px] (999 = off)")
    ap.add_argument("--epi-pose", choices=["vio", "gt"], default="vio",
                    help="pose source for the epipolar gate: realistic VIO prior or GT oracle")
    ap.add_argument("--octave-band", type=int, default=99,
                    help="ORB-SLAM-style scale gate: only match candidates within this "
                         "many octaves of the reference keypoint (99 = off/scale-blind)")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="Lowe ratio test on Hamming: accept the match only if "
                         "best_dist <= ratio * second_best_dist (1.0 = off). Needed when "
                         "the window is large / global to reject ambiguous matches.")
    ap.add_argument("--refresh", choices=["none", "confident"], default="none",
                    help="ORB-SLAM-style reference management: refresh the reference "
                         "descriptor to the current confident match (bounded staleness) "
                         "vs keep the fixed first observation")
    ap.add_argument("--refresh-thr", type=float, default=32.0,
                    help="refresh the reference only when best Hamming <= this")
    ap.add_argument("--birth-window", type=float, default=3.0)
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

    def gt_cam_pose(t):
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
    zbufs = zbuf_cache(root, frames, gt_cam_pose, fx, fy, cx, cy, w, h)

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)
    orb = cv2.ORB_create(nfeatures=args.nfeatures)

    # real VIO alongside (runtime-realistic pose prior), fed IMU + production vision
    cam = echo_li.RadTanCamera(fx, fy, cx, cy, *dcoef[:4])
    vio = echo_li.VIOFilter(args.config, cam)
    vio.set_camera_extrinsics(t_bs)
    imu = load_csv(root / "imu0" / "data.csv")
    imu_ev = [(r[0] * 1e-9, r[1:4].tolist(), r[4:7].tolist()) for r in imu]
    imu_i = 0

    def vio_pose():
        pos, quat = vio.get_pose()
        m = np.eye(4)
        m[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
        m[:3, 3] = np.asarray(pos)
        return m @ t_bs

    anchors = {}       # fid -> (X_world_gt, birth_i)
    ref_desc = {}      # fid -> (32,) uint8 first-obs ORB descriptor
    ref_oct = {}       # fid -> reference keypoint octave (for scale-band matching)
    ref_bear = {}      # fid -> birth normalized bearing [bx, by, 1] (for epipolar gate)
    gt_birth = {}      # fid -> T_wc GT birth pose (for --epi-pose gt)
    vio_birth = {}     # fid -> (pc0 birth-cam 3D [GT depth], T_wc_vio_birth 4x4)
    per_track = {}     # fid -> list of rows

    for i, (t, p) in enumerate(frames):
        while imu_i < len(imu_ev) and imu_ev[imu_i][0] <= t:
            vio.process_imu(*imu_ev[imu_i]); imu_i += 1
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

        if vio.is_initialized:
            vio.process_vision(t, {int(f["id"]): (f["x"], f["y"]) for f in feats})
        t_wc_vio = vio_pose() if vio.is_initialized else None

        kps, desc = orb.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            kps = []
        else:
            kpx = np.array([kp.pt for kp in kps], np.float64)
            octv = np.array([kp.octave & 0xFF for kp in kps])   # ORB packs octave in low byte
            tree = cKDTree(kpx)
            kpx_n = cv2.undistortPoints(kpx.reshape(-1, 1, 2), K, D).reshape(-1, 2)  # normalized bearings

        t_cw_gt = np.linalg.inv(gt_cam_pose(t))
        t_cw_vio = np.linalg.inv(t_wc_vio) if t_wc_vio is not None else None
        for fid in alive:
            if (fid not in anchors or fid not in ref_desc
                    or fid not in vio_birth or t_cw_vio is None):
                continue
            # GT truth pixel (error reference)
            pc_gt = t_cw_gt[:3, :3] @ anchors[fid][0] + t_cw_gt[:3, 3]
            if pc_gt[2] < 0.1:
                continue
            gtp = cv2.projectPoints(pc_gt.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                    K, D)[0].ravel()
            # VIO-predicted pixel: relative VIO pose birth->current on the birth 3D point
            pc0, twc_vio_b = vio_birth[fid]
            xw_vio = twc_vio_b[:3, :3] @ pc0 + twc_vio_b[:3, 3]
            pc_v = t_cw_vio[:3, :3] @ xw_vio + t_cw_vio[:3, 3]
            if pc_v[2] < 0.1:
                continue
            predp = cv2.projectPoints(pc_v.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                      K, D)[0].ravel()
            age = i - anchors[fid][1]
            if age < 1:
                continue
            pred_err = predp - gtp
            center = raw[fid] if args.center == "klt" else (gtp if args.center == "gt" else predp)
            matched, orb_e, ham = 0, np.array([np.nan, np.nan]), np.nan
            orb_ref_e = np.array([np.nan, np.nan])
            if len(kps):
                idxs = tree.query_ball_point(center, args.window)
                if idxs:
                    idxs = np.array(idxs)
                    if args.octave_band < 90:      # ORB-SLAM-style scale-band gate
                        idxs = idxs[np.abs(octv[idxs] - ref_oct[fid]) <= args.octave_band]
                if len(idxs):
                    hd = hamming(ref_desc[fid], desc[idxs])
                    order = np.argsort(hd)
                    bhd = hd[order[0]]
                    shd = hd[order[1]] if len(order) > 1 else 256   # second-best (other kp)
                    if bhd <= args.ratio * shd:                      # Lowe ratio test
                        best = idxs[order[0]]
                        epi_ok = True
                        if args.epi_thr < 900 and fid in ref_bear:   # prior-epipolar (RANSAC) gate
                            if args.epi_pose == "gt":
                                twc_b, twc_c = gt_birth[fid], gt_cam_pose(t)
                            else:
                                twc_b, twc_c = vio_birth[fid][1], t_wc_vio
                            t_rel = np.linalg.inv(twc_c) @ twc_b     # birth-cam -> current-cam
                            tn = t_rel[:3, 3] / max(np.linalg.norm(t_rel[:3, 3]), 1e-9)
                            E = skew(tn) @ t_rel[:3, :3]
                            x1 = np.array([kpx_n[best, 0], kpx_n[best, 1], 1.0])
                            epi_ok = sampson_px(ref_bear[fid], x1, E, fx) <= args.epi_thr
                        if epi_ok:
                            orb_e = kpx[best] - gtp
                            ham = float(bhd)
                            matched = 1
                            if args.refresh == "confident" and ham <= args.refresh_thr:
                                ref_desc[fid] = desc[best]  # geometrically-verified refresh
                                ref_oct[fid] = octv[best]
                            # cheap geometric refine: snap to the subpixel corner (no BA)
                            pt = np.array([[kpx[best]]], np.float32)
                            cv2.cornerSubPix(img, pt, (5, 5), (-1, -1),
                                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                            orb_ref_e = pt.ravel().astype(np.float64) - gtp
            klt_e = raw[fid] - gtp
            per_track.setdefault(fid, []).append(
                (age, matched, pred_err[0], pred_err[1], orb_e[0], orb_e[1],
                 klt_e[0], klt_e[1], ham, orb_ref_e[0], orb_ref_e[1]))

        # births
        for fid in alive:
            if fid in anchors:
                continue
            uv = und_by_id[fid]
            d0 = zbuf_lookup(zbufs[i], [uv], w, h)[0]
            if not np.isfinite(d0):
                continue
            pc = np.array([(uv[0] - cx) / fx * d0, (uv[1] - cy) / fy * d0, d0])
            t_wc = gt_cam_pose(t)
            anchors[fid] = (t_wc[:3, :3] @ pc + t_wc[:3, 3], i)
            if len(kps):
                dd, jj = tree.query(raw[fid])
                if dd <= args.birth_window:
                    ref_desc[fid] = desc[jj]
                    ref_oct[fid] = octv[jj]
                    ref_bear[fid] = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
                    gt_birth[fid] = t_wc.copy()
            if t_wc_vio is not None:           # need VIO pose at birth for relative pred
                vio_birth[fid] = (pc.copy(), t_wc_vio.copy())
        for fid in [f for f in anchors if f not in alive]:
            del anchors[fid]
            ref_desc.pop(fid, None); ref_oct.pop(fid, None); ref_bear.pop(fid, None)
            gt_birth.pop(fid, None); vio_birth.pop(fid, None)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(anchors)} vio_init={vio.is_initialized} "
                  f"scored={sum(len(v) for v in per_track.values())}")

    rows = [r for v in per_track.values() for r in v]
    A = np.array(rows)
    np.savez(args.out + ".npz", d=A, cols=np.array(
        ["age", "matched", "pred_dx", "pred_dy", "orb_dx", "orb_dy", "klt_dx", "klt_dy",
         "hamming", "orb_ref_dx", "orb_ref_dy"]))
    age = A[:, 0]; matched = A[:, 1].astype(bool)
    pred = np.hypot(A[:, 2], A[:, 3])
    orb = np.hypot(A[:, 4], A[:, 5]); klt = np.hypot(A[:, 6], A[:, 7])
    orbref = np.hypot(A[:, 9], A[:, 10])   # (c) cornerSubPix-refined descriptor position
    print(f"\n=== ENSEMBLE ({len(A)} obs, {len(per_track)} tracks) | center={args.center} "
          f"window={args.window:.0f} octave_band={args.octave_band} ratio={args.ratio} "
          f"epi_thr={args.epi_thr}({args.epi_pose}) refresh={args.refresh} ===")
    print(f"  VIO prediction error |pred-GT|:  median {np.median(pred):.2f}  "
          f"p90 {np.percentile(pred,90):.2f}  p99 {np.percentile(pred,99):.2f} px")
    print(f"  association recall (kp matched in window): {100*matched.mean():.1f}%")
    good = matched & (orb < 3.0)
    print(f"  association precision (matched & <3px GT): {100*good.sum()/max(matched.sum(),1):.1f}% "
          f"of matches")
    m = matched
    print(f"  on matched: median |err| ORB {np.median(orb[m]):.2f}  KLT {np.median(klt[m]):.2f} | "
          f"p90 ORB {np.percentile(orb[m],90):.2f}  KLT {np.percentile(klt[m],90):.2f}")

    print(f"  refined (cornerSubPix) on matched: median {np.median(orbref[m]):.2f}  "
          f"p90 {np.percentile(orbref[m],90):.2f}  (raw ORB {np.median(orb[m]):.2f})")

    ham = A[:, 8]
    print("\n-- Hamming gate sweep: raw ORB vs cornerSubPix-refined vs KLT --")
    print(f"  {'thr':>5} {'kept%':>6} {'ORBraw':>7} {'ORBref':>7} {'KLT':>6} {'refprec%':>8}  "
          f"(prec = within 3px GT)")
    for thr in [256, 48, 32, 24, 16]:
        g = matched & (ham <= thr)
        if g.sum() > 50:
            refprec = 100 * (g & (orbref < 3.0)).sum() / g.sum()
            print(f"  {thr:>5} {100*g.sum()/max(matched.sum(),1):6.1f} "
                  f"{np.median(orb[g]):7.2f} {np.median(orbref[g]):7.2f} {np.median(klt[g]):6.2f} "
                  f"{refprec:8.1f}")

    print("\n-- by age (Hamming<=32 gated): ORBraw / ORBref / KLT median --")
    for lo, hi in zip([1, 5, 10, 20, 40, 80, 160], [5, 10, 20, 40, 80, 160, 1e9]):
        g = matched & (ham <= 32) & (age >= lo) & (age < hi)
        if g.sum() > 15:
            print(f"  {int(lo):3d}-{int(min(hi,9999)):>4d}: "
                  f"{np.median(orb[g]):5.2f} / {np.median(orbref[g]):5.2f} / {np.median(klt[g]):5.2f}")

    age_edges = np.array([1, 5, 10, 20, 40, 80, 160, 1e9])
    ham = A[:, 8]
    print("\n-- vs age:  recall% | Hamming median | ORB/KLT median (matched) --")
    print(f"  {'age':>10} {'rec%':>6} {'ham':>5} {'ORB':>6} {'KLT':>6}")
    for lo, hi in zip(age_edges[:-1], age_edges[1:]):
        s = (age >= lo) & (age < hi)
        if s.sum() > 20:
            ms = s & matched
            print(f"  {int(lo):4d}-{int(min(hi,9999)):>4d} "
                  f"{100*matched[s].mean():6.1f} "
                  f"{np.median(ham[ms]) if ms.sum()>10 else np.nan:5.0f} "
                  f"{np.median(orb[ms]) if ms.sum()>10 else np.nan:6.2f} "
                  f"{np.median(klt[ms]) if ms.sum()>10 else np.nan:6.2f}")

    # increment autocorr on VIO-gated matched series (does whiteness survive?)
    def inc_ac(dxi, dyi):
        vals = []
        for v in per_track.values():
            vs = [r for r in sorted(v) if r[1] > 0.5]   # matched only
            if len(vs) < 8:
                continue
            b = np.array([(r[dxi], r[dyi]) for r in vs])
            db = np.diff(b, axis=0); dm = np.hypot(db[:, 0], db[:, 1])
            ud = db / (dm[:, None] + 1e-9)
            if len(ud) > 1:
                vals.append(np.mean(np.sum(ud[1:] * ud[:-1], axis=1)))
        return np.median(vals) if vals else np.nan
    print(f"\n  increment-autocorr (matched):  ORB {inc_ac(4,5):+.2f}   KLT {inc_ac(6,7):+.2f}")
    print("  (ORB<0 => white survives real prior; KLT>0 => coherent walk)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for lab, arr in [("pred_err", pred)]:
        pass
    cs, pe, rc = [], [], []
    for lo, hi in zip(age_edges[:-2], age_edges[1:-1]):
        s = (age >= lo) & (age < hi)
        if s.sum() > 20:
            cs.append(0.5 * (lo + hi)); pe.append(np.median(pred[s])); rc.append(100 * matched[s].mean())
    ax[0].plot(cs, pe, "o-"); ax[0].set_title("VIO prediction error vs age")
    ax[0].set_xlabel("age"); ax[0].set_ylabel("|pred-GT| [px]"); ax[0].grid(alpha=0.3)
    ax[1].plot(cs, rc, "s-", color="C1"); ax[1].set_title("association recall vs age")
    ax[1].set_xlabel("age"); ax[1].set_ylabel("recall %"); ax[1].set_ylim(0, 100); ax[1].grid(alpha=0.3)
    om, km = [], []
    for lo, hi in zip(age_edges[:-2], age_edges[1:-1]):
        s = (age >= lo) & (age < hi) & matched
        om.append(np.median(orb[s]) if s.sum() > 10 else np.nan)
        km.append(np.median(klt[s]) if s.sum() > 10 else np.nan)
    ax[2].plot(cs, om, "o-", label="ORB"); ax[2].plot(cs, km, "s-", label="KLT")
    ax[2].set_title("error vs age (VIO-gated)"); ax[2].set_xlabel("age")
    ax[2].set_ylabel("median |err| [px]"); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=120)
    print(f"\nsaved {args.out}.png / .npz")


if __name__ == "__main__":
    main()
