"""Visualize WHY a front-end helps/hurts the VIO, as a video.

Runs a front-end (OpenCV LK or Rudolf-V) over V1_03 with matched CLAHE, and each
frame classifies tracks by fundamental-matrix RANSAC (a proxy for the geometric
check Rudolf-V applies internally):
  green = inlier, RED = geometric OUTLIER, yellow = new this frame.
Short motion trails show drift. The contrast to look for: OpenCV keeps red/outlier
tracks *alive for many frames* (long red trails) while Rudolf-V kills them at age
1 (few/short red trails) -- which is why Rudolf-V wins on ATE despite more churn.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/tracker_viz.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult --frontend opencv
      [--frontend rudolf]
"""
import argparse, csv
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
import cv2, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        return np.array([r for r in csv.reader(f) if r and not r[0].startswith("#")], dtype=float)


def quat_ang_rate(t, quat):
    def qmul(a, b):
        aw, ax, ay, az = a.T; bw, bx, by, bz = b.T
        return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], axis=1)
    qc = quat.copy(); qc[:, 1:] *= -1
    dq = qmul(qc[:-1], quat[1:]); dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    w = 2*np.arccos(np.clip(np.abs(dq[:, 0]), -1, 1))/np.diff(t)
    return np.concatenate([w, w[-1:]])


def make_opencv(w, h):
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    s = dict(prev=None, pts=np.empty((0, 1, 2), np.float32),
             ids=np.empty(0, int), ages=np.empty(0, int), nid=0)

    def step(img):
        if s["prev"] is not None and len(s["pts"]) > 0:
            new, st, _ = cv2.calcOpticalFlowPyrLK(s["prev"], img, s["pts"], None, **lk)
            st = st.reshape(-1).astype(bool)
            nn = new.reshape(-1, 2)
            st &= (nn[:, 0] >= 1) & (nn[:, 0] < w-1) & (nn[:, 1] >= 1) & (nn[:, 1] < h-1)
            s["pts"] = new[st]; s["ids"] = s["ids"][st]; s["ages"] = s["ages"][st] + 1
        need = 300 - len(s["pts"])
        if need > 0:
            mask = np.full((h, w), 255, np.uint8)
            for q in s["pts"].reshape(-1, 2):
                cv2.circle(mask, (int(q[0]), int(q[1])), 16, 0, -1)
            det = cv2.goodFeaturesToTrack(img, need, 0.01, 16, mask=mask)
            if det is not None:
                k = len(det)
                s["pts"] = np.vstack([s["pts"], det]) if len(s["pts"]) else det
                s["ids"] = np.concatenate([s["ids"], np.arange(s["nid"], s["nid"]+k)])
                s["ages"] = np.concatenate([s["ages"], np.ones(k, int)]); s["nid"] += k
        s["prev"] = img
        pos = {int(i): (float(p[0]), float(p[1]))
               for i, p in zip(s["ids"], s["pts"].reshape(-1, 2))}
        age = {int(i): int(a) for i, a in zip(s["ids"], s["ages"])}
        return pos, age
    return step


def make_rudolf(config, w, h, fx, fy, cx, cy, dcoef):
    import echo_li
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.histeq = "none"   # CLAHE applied in the main loop (matched to OpenCV side)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef if dcoef else [])
    tr = echo_li.Frontend(fcfg, w, h)

    def step(img):
        feats, _ = tr.process(img)
        meta = {int(m["id"]): int(m["age"]) for m in tr.track_meta()}
        pos = {int(f["id"]): (float(f["x"]), float(f["y"])) for f in feats}
        return pos, {i: meta.get(i, 1) for i in pos}
    return step


def open_writer(path, w, h, fps=20.0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if vw.isOpened():
        return vw, path
    path = path.replace(".mp4", ".avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    return vw, path


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--frontend", choices=["opencv", "rudolf"], default="opencv")
    ap.add_argument("--histeq", choices=["clahe", "global", "none"], default="clahe",
                    help="preprocessing applied identically to both front-ends")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root/"mav0").exists():
        root = root/"mav0"
    cfg = yaml.safe_load(open(root/"cam0"/"sensor.yaml"))
    w, h = cfg["resolution"]; fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = cfg.get("distortion_coefficients", [])
    gt = load_csv(root/"state_groundtruth_estimate0"/"data.csv")
    gt_t = gt[:, 0]*1e-9; gt_w = quat_ang_rate(gt_t, gt[:, 4:8]); t0 = gt_t[0]

    idir = root/"cam0"/"data"
    with open(root/"cam0"/"data.csv") as f:
        rd = csv.reader(f); next(rd)
        paths = [(int(r[0])*1e-9, idir/r[1].strip()) for r in rd if r]

    step = (make_opencv(w, h) if args.frontend == "opencv"
            else make_rudolf(args.config, w, h, fx, fy, cx, cy, dcoef))
    ce = cv2.createCLAHE(4.0, (8, 8))
    def pre(g):
        if args.histeq == "clahe":
            return ce.apply(g)
        if args.histeq == "global":
            return cv2.equalizeHist(g)
        return g
    out = args.out or f"tracker_viz_{args.frontend}_{args.histeq}.mp4"
    vw, out = open_writer(out, w, h)
    col = {"in": (0, 200, 0), "out": (0, 0, 255), "new": (0, 220, 220)}
    trails = defaultdict(lambda: deque(maxlen=8))

    prev_pos = {}
    tr_hist, outfrac_hist = [], []
    in_ages, out_ages = [], []
    for n, (t, p) in enumerate(paths):
        img = pre(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        pos, age = step(img)
        # geometric classification (prev->curr matched by id)
        common = [i for i in pos if i in prev_pos]
        status = {}
        outfrac = 0.0
        if len(common) >= 8:
            pv = np.array([prev_pos[i] for i in common], np.float32)
            cu = np.array([pos[i] for i in common], np.float32)
            _, mask = cv2.findFundamentalMat(pv, cu, cv2.FM_RANSAC, 1.0, 0.99)
            inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(common), bool)
            for k, i in enumerate(common):
                status[i] = "in" if inl[k] else "out"
                (in_ages if inl[k] else out_ages).append(age[i])
            outfrac = 1.0 - inl.mean()
        for i in pos:
            status.setdefault(i, "new")
        # trails (drop dead ids)
        for i in list(trails):
            if i not in pos:
                del trails[i]
        for i, xy in pos.items():
            trails[i].append(xy)
        # render
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in pos:
            s = status[i]; tl = trails[i]
            if len(tl) >= 2:
                cv2.polylines(vis, [np.array(tl, np.int32)], False, col[s], 1, cv2.LINE_AA)
            x, y = tl[-1]
            cv2.circle(vis, (int(x), int(y)), 2, col[s], -1)
        n_out = sum(1 for i in pos if status[i] == "out")
        wv = float(np.interp(t-t0, gt_t-t0, gt_w))
        cv2.putText(vis, f"{args.frontend}  t={t-t0:5.1f}s  |w|={wv:.2f}  "
                    f"outliers={n_out}/{len(pos)} ({100*n_out/max(len(pos),1):.0f}%)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        vw.write(vis)
        prev_pos = pos
        tr_hist.append(t-t0); outfrac_hist.append(outfrac)
        if n % 400 == 0:
            print(f"  frame {n}/{len(paths)}  t={t-t0:.1f}s  outliers {100*outfrac:.0f}%")
    vw.release()

    tr_hist = np.array(tr_hist); outfrac_hist = np.array(outfrac_hist)
    gw = np.interp(tr_hist, gt_t-t0, gt_w); hi = gw > np.percentile(gw, 75)
    im = np.median(in_ages) if in_ages else 0; om = np.median(out_ages) if out_ages else 0
    print(f"\n[{args.frontend}] geometric outliers (F-RANSAC proxy):")
    print(f"  outlier fraction: overall {np.mean(outfrac_hist):.2f}  "
          f"hi|w| {np.mean(outfrac_hist[hi]):.2f}  calm {np.mean(outfrac_hist[~hi]):.2f}")
    print(f"  track age:  inliers median {im:.0f}   OUTLIERS median {om:.0f}   "
          f"(higher outlier age = bad tracks kept alive)")
    windows = [(82.2, 83.0), (92.3, 93.6), (55.7, 56.9), (67.4, 68.0), (71.3, 71.8)]
    fig, a2 = plt.subplots(figsize=(12, 3.5))
    for a, b in windows:
        a2.axvspan(a, b, color="red", alpha=0.12)
    a2.plot(tr_hist, outfrac_hist*100, lw=0.9, color="tab:red")
    a2.set_ylabel("outlier %"); a2.set_xlabel("t [s]")
    a2.set_title(f"{args.frontend} front-end geometric-outlier fraction (red = rotation windows)")
    a2.grid(alpha=0.3)
    p2 = out.rsplit(".", 1)[0] + "_outfrac.png"
    fig.tight_layout(); fig.savefig(p2, dpi=130)
    print(f"\nsaved {out}, {p2}")


if __name__ == "__main__":
    main()
