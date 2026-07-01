"""Head-to-head tracker control: Rudolf-V frontend vs OpenCV LK on the same
EuRoC images, to answer "does OpenCV hit the same limitation under rotation?"
and to prototype forward-backward (FB) and window size cheaply in Python before
any Rust work.

Metric: median track *age* (lifetime) and survival through rotation-dominant
windows, split by GT angular rate. Longer age under high |w| == tracks better
through rotation. No VIO here -- this isolates the tracker.

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/tracker_compare.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
"""
import argparse, csv, os
from pathlib import Path
import numpy as np
import cv2, yaml
import echo_li


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


def run_opencv(paths, w, h, ce, n=300, min_dist=16, win=21, levels=3, fb=False,
               fb_thresh=1.0):
    lk = dict(winSize=(win, win), maxLevel=levels,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev = None
    pts = np.empty((0, 1, 2), np.float32); ids = np.empty(0, int); ages = np.empty(0, int)
    nid = 0; rec = []
    for t, p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = ce.apply(img)   # matched preprocessing (identical to Rudolf-V side)
        if prev is not None and len(pts) > 0:
            new, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, None, **lk)
            st = st.reshape(-1).astype(bool)
            if fb:
                back, _, _ = cv2.calcOpticalFlowPyrLK(img, prev, new, None, **lk)
                fberr = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
                st &= fberr < fb_thresh
            nn = new.reshape(-1, 2)
            st &= (nn[:, 0] >= 1) & (nn[:, 0] < w-1) & (nn[:, 1] >= 1) & (nn[:, 1] < h-1)
            pts = new[st]; ids = ids[st]; ages = ages[st] + 1
        need = n - len(pts)
        if need > 0:
            mask = np.full((h, w), 255, np.uint8)
            for q in pts.reshape(-1, 2):
                cv2.circle(mask, (int(q[0]), int(q[1])), min_dist, 0, -1)
            det = cv2.goodFeaturesToTrack(img, need, 0.01, min_dist, mask=mask)
            if det is not None:
                k = len(det)
                pts = np.vstack([pts, det]) if len(pts) else det
                ids = np.concatenate([ids, np.arange(nid, nid+k)])
                ages = np.concatenate([ages, np.ones(k, int)])
                nid += k
        rec.append((t, float(np.median(ages)) if len(ages) else 0.0,
                    set(ids.tolist()), len(pts)))
        prev = img
    return rec


def run_rudolf(paths, config, w, h, fx, fy, cx, cy, dcoef, dist_model, ce):
    fcfg = echo_li.FrontendConfig.from_yaml(config)
    fcfg.histeq = "none"   # CLAHE applied in Python identically for both; no double-process
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef if dcoef else [])
    tr = echo_li.Frontend(fcfg, w, h)
    rec = []
    for t, p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = ce.apply(img)   # matched preprocessing (identical to OpenCV side)
        _, _ = tr.process(img)
        meta = tr.track_meta()
        ages = [m["age"] for m in meta]
        ids = set(int(m["id"]) for m in meta)
        rec.append((t, float(np.median(ages)) if ages else 0.0, ids, len(ages)))
    return rec


def survival(rec, tr_rel, windows):
    """median survival % across the rotation windows (ids at start still at end)."""
    times = np.array([r[0] for r in rec])
    out = []
    for a, b in windows:
        i = int(np.argmin(np.abs(times - a))); j = int(np.argmin(np.abs(times - b)))
        s0 = rec[i][2]
        out.append(len(s0 & rec[j][2]) / len(s0) * 100 if s0 else np.nan)
    return np.nanmedian(out)


def summarize(name, rec, t0, gtw_at, windows):
    tr = np.array([r[0]-t0 for r in rec])
    age = np.array([r[1] for r in rec]); ntrk = np.array([r[3] for r in rec])
    wv = gtw_at(tr)
    hi = wv > np.percentile(wv, 75)
    surv = survival([(r[0]-t0,)+r[1:] for r in rec], tr, windows)
    print(f"  {name:28s} age med {np.median(age):5.1f}  hi|w| {np.median(age[hi]):5.1f}  "
          f"calm {np.median(age[~hi]):5.1f}  tracked {np.median(ntrk):4.0f}  "
          f"rot-win surv {surv:4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo/"configs"/"eqvio_euroc_rho.yaml"))
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root/"mav0").exists():
        root = root/"mav0"

    cfg = yaml.safe_load(open(root/"cam0"/"sensor.yaml"))
    w, h = cfg["resolution"]; fx, fy, cx, cy = cfg["intrinsics"]
    dist_model = cfg.get("distortion_model", ""); dcoef = cfg.get("distortion_coefficients", [])

    gt = load_csv(root/"state_groundtruth_estimate0"/"data.csv")
    gt_t = gt[:, 0]*1e-9; gt_w = quat_ang_rate(gt_t, gt[:, 4:8]); t0 = gt_t[0]
    gtw_at = lambda tr: np.interp(tr, gt_t-t0, gt_w)
    windows = [(82.2, 83.0), (92.3, 93.6), (55.7, 56.9), (67.4, 68.0), (71.3, 71.8)]

    idir = root/"cam0"/"data"
    with open(root/"cam0"/"data.csv") as f:
        rd = csv.reader(f); next(rd)
        paths = [(int(r[0])*1e-9, idir/r[1].strip()) for r in rd if r]

    ce = cv2.createCLAHE(4.0, (8, 8))   # matched CLAHE (Rudolf-V's internal params) for BOTH
    print(f"V1_03 tracker head-to-head ({len(paths)} frames). age = track lifetime; "
          f"higher under hi|w| == holds tracks through rotation.")
    print("CLAHE clip=4.0 tile=8 applied identically to both; Rudolf-V internal histeq off.\n")
    summarize("Rudolf-V (config)", run_rudolf(paths, args.config, w, h, fx, fy, cx, cy,
              dcoef, dist_model, ce), t0, gtw_at, windows)
    for fb in (False, True):
        for win in (15, 21):
            rec = run_opencv(paths, w, h, ce, win=win, fb=fb)
            summarize(f"OpenCV win{win} fb={int(fb)}", rec, t0, gtw_at, windows)


if __name__ == "__main__":
    main()
