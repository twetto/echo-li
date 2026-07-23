"""Visual evidence for WHY first-observation tracking fails: appearance staleness.

A first-observation tracker must keep matching the CURRENT frame to the patch it saw
at birth. This montages, for long-lived GT-anchored tracks, the first-observation patch
next to the *true* patch (current frame at the GT-reprojected pixel) at increasing age.
If the true appearance drifts away from the first observation, a fixed template must
fail -- and the only cure (adapt the template) reintroduces drift. That is the
robustness-vs-fixed-template contradiction, made visible.

Patches are read from the histeq'd image the tracker actually matches on. NCC of the
first-obs patch to each later true patch quantifies the staleness; the aggregate
NCC-vs-age curve is the summary. Contrast line: NCC between CONSECUTIVE true patches
(what a previous-frame template sees) stays ~1 -- fresh -- which is why previous-frame
tracking works and first-observation does not.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/first_obs_visual.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_01_easy [--out first_obs_visual]
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
import echo_li  # noqa: E402

AGES = [0, 5, 10, 20, 40, 80, 120]


def ncc(a, b):
    a = a.astype(np.float64) - a.mean(); b = b.astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="first_obs_visual")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-tracks", type=int, default=6)
    ap.add_argument("--patch", type=int, default=31, help="patch size (odd)")
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
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h)

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    ps = args.patch
    hp = ps // 2 + 1                          # half-patch edge margin
    anchors = {}                 # fid -> (X_world, birth_i)
    patches = {}                 # fid -> {age: uint8 patch at GT-true pixel}
    prev_true = {}               # fid -> (prev true patch, prev age) for consecutive NCC
    consec_ncc = []              # (age, ncc(true@k, true@k-1))
    aged_ncc = []                # (age, ncc(first_obs, true@age)) all target ages

    for i, (t, p) in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        eq = cv2.equalizeHist(img)          # the image the tracker matches on
        feats, _ = tracker.process(img)
        raw = {int(f["id"]): (float(f["x"]), float(f["y"])) for f in feats}
        alive = set(raw)
        if feats:
            und = cv2.undistortPoints(
                np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2),
                K, D, P=K).reshape(-1, 2)
        else:
            und = np.zeros((0, 2))
        und_by = {int(f["id"]): uv for f, uv in zip(feats, und)}
        t_cw = np.linalg.inv(cam_pose(t))

        for fid in alive:
            if fid not in anchors:
                continue
            X, i0 = anchors[fid]
            pc = t_cw[:3, :3] @ X + t_cw[:3, 3]
            if pc[2] < 0.1:
                continue
            gp = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                   K, D)[0].ravel()
            if not (hp <= gp[0] < w - hp and hp <= gp[1] < h - hp):
                continue
            patch = cv2.getRectSubPix(eq, (ps, ps), (float(gp[0]), float(gp[1])))
            age = i - i0
            # consecutive-frame NCC (what a previous-frame template sees: fresh)
            if fid in prev_true:
                pv, pa = prev_true[fid]
                if age - pa == 1:
                    consec_ncc.append((age, ncc(pv, patch)))
            prev_true[fid] = (patch, age)
            # staleness NCC vs first observation, at target ages
            if fid in patches and 0 in patches[fid] and age in AGES:
                aged_ncc.append((age, ncc(patches[fid][0], patch)))
            if age in AGES:
                patches.setdefault(fid, {})[age] = patch

        # births
        for fid in alive:
            if fid in anchors:
                continue
            uv = und_by[fid]
            d0 = zbuf_lookup(zbufs[i], [uv], w, h)[0]
            if not np.isfinite(d0):
                continue
            pc = np.array([(uv[0] - cx) / fx * d0, (uv[1] - cy) / fy * d0, d0])
            t_wc = cam_pose(t)
            anchors[fid] = (t_wc[:3, :3] @ pc + t_wc[:3, 3], i)
            rx, ry = raw[fid]
            if hp <= rx < w - hp and hp <= ry < h - hp:
                patches[fid] = {0: cv2.getRectSubPix(eq, (ps, ps), (float(rx), float(ry)))}
        for fid in [f for f in anchors if f not in alive]:
            del anchors[fid]; prev_true.pop(fid, None)
        if i % 200 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(anchors)} long={sum(1 for v in patches.values() if 80 in v)}")

    # pick long-lived tracks where the first-obs template goes MOST stale (low NCC at
    # the oldest available age) -- i.e. exactly the ones a first-obs tracker loses.
    cand = []
    for fid, pm in patches.items():
        if 0 not in pm:
            continue
        old = max(a for a in pm if a > 0) if len(pm) > 1 else 0
        if old >= 80:
            cand.append((ncc(pm[0], pm[old]), fid, old))
    cand.sort()                                  # lowest NCC (most stale) first
    sel = [fid for _, fid, _ in cand[: args.n_tracks]]
    print(f"selected {len(sel)} long tracks (most-stale first-obs templates)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [a for a in AGES]
    fig, ax = plt.subplots(len(sel), len(cols), figsize=(1.5 * len(cols), 1.6 * len(sel)))
    if len(sel) == 1:
        ax = ax[None, :]
    for r, fid in enumerate(sel):
        pm = patches[fid]
        for c, age in enumerate(cols):
            a = ax[r, c]
            a.set_xticks([]); a.set_yticks([])
            if age in pm:
                a.imshow(pm[age], cmap="gray", vmin=0, vmax=255)
                kw = 7                                   # actual KLT window radius (px)
                a.add_patch(plt.Rectangle((ps / 2 - kw, ps / 2 - kw), 2 * kw + 1,
                                          2 * kw + 1, fill=False, ec="cyan", lw=1.2))
                if age == 0:
                    a.set_ylabel(f"id {fid}", fontsize=8)
                    for s in a.spines.values():
                        s.set_color("red"); s.set_linewidth(2)
                else:
                    c = ps // 2
                    w0 = pm[0][c - 7:c + 8, c - 7:c + 8]     # KLT-window NCC (not big crop)
                    wk = pm[age][c - 7:c + 8, c - 7:c + 8]
                    a.set_title(f"n={age}\nwinNCC {ncc(w0, wk):.2f}", fontsize=7)
            else:
                a.axis("off")
    ax[0, 0].set_title("first obs\n(template)", fontsize=7, color="red")
    fig.suptitle("Feature true appearance vs its FIRST-OBSERVATION template (the fixed "
                 "template a first-obs tracker must keep matching)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out + "_montage.png", dpi=130)
    print(f"saved {args.out}_montage.png")

    # aggregate staleness curve
    A = np.array(aged_ncc); C = np.array(consec_ncc)
    fig2, a2 = plt.subplots(figsize=(7, 4.5))
    targets = [5, 10, 20, 40, 80, 120]
    xs, ys = [], []
    for age in targets:                       # first-obs sampled at exact target ages
        m = A[:, 0] == age
        if m.sum() > 10:
            xs.append(age); ys.append(np.median(A[m, 1]))
    a2.plot(xs, ys, "o-", label="NCC(first-obs, true @ age)")
    xc, yc = [], []
    for age in targets:                       # consecutive-frame NCC near each age
        m = np.abs(C[:, 0] - age) <= 2
        if m.sum() > 10:
            xc.append(age); yc.append(np.median(C[m, 1]))
    a2.plot(xc, yc, "s--", label="NCC(prev frame, true) [consecutive]")
    a2.axhline(0, color="0.6", lw=1)
    a2.set_xlabel("track age [frames]"); a2.set_ylabel("median NCC")
    a2.set_title("Why first-obs fails: its template goes stale;\nthe previous-frame "
                 "template stays fresh")
    a2.set_ylim(0, 1.02); a2.grid(alpha=0.3); a2.legend()
    fig2.tight_layout(); fig2.savefig(args.out + "_ncc.png", dpi=130)
    print(f"saved {args.out}_ncc.png")


if __name__ == "__main__":
    main()
