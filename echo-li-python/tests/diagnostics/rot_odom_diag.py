"""Rotation-degradation diagnostic: drive the full ECHO-LI VIO (Rust core via
the echo_li binding) over EuRoC V1_03_difficult and correlate trajectory error
against the rotation-dominant windows found by rot_segment_diag.py.

Records per image frame: est body pose, GT pose, tracked/total feature counts,
median track age (Frontend.track_meta), in-state landmark ids. Then SE(3)-aligns
(Umeyama) est->GT and reports:
  * ATE RMSE (whole + per rotation-dominant window)
  * position-error timeseries vs GT angular rate
  * median track age + landmark survival through each window

Usage (run from anywhere; config resolves relative to the repo):
  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/rot_odom_diag.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
"""
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import echo_li

Z_SCENE = 2.5   # nominal Vicon-room depth [m] for the parallax/rotation ratio
RPE_DT = 1.0    # RPE interval [s]


def load_csv(path):
    with open(path) as f:
        return np.array([r for r in csv.reader(f) if r and not r[0].startswith("#")],
                        dtype=float)


def quat_ang_rate(t, quat):
    """|angular velocity| (rad/s) from GT quaternion (w,x,y,z) finite diff."""
    def qmul(a, b):
        aw, ax, ay, az = a.T; bw, bx, by, bz = b.T
        return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], axis=1)
    qc = quat.copy(); qc[:, 1:] *= -1
    dq = qmul(qc[:-1], quat[1:]); dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    ang = 2*np.arccos(np.clip(np.abs(dq[:, 0]), -1, 1))
    w = ang/np.diff(t); return np.concatenate([w, w[-1:]])


def umeyama_se3(src, dst):
    """Rigid (no-scale) alignment src->dst; returns R,t,scale(diag),aligned."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src-mu_s, dst-mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    scale = (sig * np.array([1, 1, d])).sum() / (S**2).sum() * len(src)
    t = mu_d - R @ mu_s
    aligned = (R @ src.T).T + t
    return R, t, scale, aligned


def plot_rpe_intervals(out, tr, est, est_R, gtp, gt_R, ratio, rpe_t, rpe_r, m, kf):
    """Teach RPE: for contrasting Δ-intervals, draw GT vs est sub-trajectory in the
    body-i frame (common origin). Endpoint gap = the RPE translation error."""
    # candidate intervals with fully-valid poses AND real motion (>0.3 m over Δ),
    # so the picture isn't a degenerate near-stationary blob.
    disp = lambda i: np.linalg.norm(gtp[i+kf] - gtp[i])
    cand = [i for i in range(len(tr)-kf)
            if m[i:i+kf+1].all() and np.isfinite(rpe_t[i]) and disp(i) > 0.3]
    cand.sort(key=lambda i: rpe_t[i])
    picks = cand[:3] + cand[-3:]          # 3 smallest RPE, 3 largest RPE (both real motion)
    labels = ["low RPE (accurate)"]*3 + ["high RPE (drifting)"]*3

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, i, lab in zip(axes.ravel(), picks, labels):
        ls = np.arange(i, i+kf+1)
        ep = np.array([est_R[i].T @ (est[l]-est[i]) for l in ls])   # est path, body-i
        gp = np.array([gt_R[i].T @ (gtp[l]-gtp[i]) for l in ls])    # gt  path, body-i
        P = np.vstack([ep, gp])
        _, _, Vt = np.linalg.svd(P - P.mean(0))
        B = Vt[:2].T                                                # faithful 2D plane
        ep2, gp2 = ep @ B, gp @ B
        o2 = np.zeros(3) @ B                                        # common start (origin)
        ax.plot(gp2[:, 0], gp2[:, 1], "-o", ms=2.5, color="tab:green", label="GT motion")
        ax.plot(ep2[:, 0], ep2[:, 1], "-o", ms=2.5, color="tab:blue", label="est motion")
        ax.plot(o2[0], o2[1], "ks", ms=6)                          # common start
        ax.annotate("", xy=ep2[-1], xytext=gp2[-1],
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.6))
        mid = 0.5*(ep2[-1]+gp2[-1])
        ax.text(mid[0], mid[1], f" {rpe_t[i]*100:.0f} cm", color="red", fontsize=9)
        ax.set_title(f"t={tr[i]:.1f}s  {lab}\nratio={ratio[i]:.2f}  "
                     f"RPE={rpe_t[i]*100:.0f}cm / {rpe_r[i]:.1f}deg", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"RPE in action: GT vs estimated motion over each Δ interval "
                 f"(body frame). Red arrow = RPE translation error.", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    return out


def main():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config",
                    default=str(repo_root/"configs"/"eqvio_euroc_rho.yaml"))
    ap.add_argument("--out", default="rot_odom_diag.png")
    ap.add_argument("--rpe-dt", type=float, default=RPE_DT, help="RPE interval [s]")
    args = ap.parse_args()
    rpe_dt = args.rpe_dt

    root = Path(args.dataset)
    if (root/"mav0").exists():
        root = root/"mav0"

    # camera config
    cfg = yaml.safe_load(open(root/"cam0"/"sensor.yaml"))
    w, h = cfg["resolution"]
    fx, fy, cx, cy = cfg["intrinsics"]
    dist_model = cfg.get("distortion_model", "")
    dcoef = cfg.get("distortion_coefficients", [])
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4) if "T_BS" in cfg else None
    print(f"cam {w}x{h} f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f}) dist={dist_model}")

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef if dcoef else [])
    tracker = echo_li.Frontend(fcfg, w, h)
    cam = (echo_li.RadTanCamera(fx, fy, cx, cy, *dcoef[:4])
           if "radial" in dist_model.lower() else echo_li.PinholeCamera(fx, fy, cx, cy))
    vio = echo_li.VIOFilter(args.config, cam)
    if t_bs is not None:
        vio.set_camera_extrinsics(t_bs)

    # GT
    gt = load_csv(root/"state_groundtruth_estimate0"/"data.csv")
    gt_t = gt[:, 0]*1e-9; gt_p = gt[:, 1:4]; gt_q = gt[:, 4:8]
    gt_speed = np.linalg.norm(gt[:, 8:11], axis=1)
    gt_w = quat_ang_rate(gt_t, gt_q)
    gt_rots = Rot.from_quat(gt_q[:, [1, 2, 3, 0]])   # w,x,y,z -> x,y,z,w
    slerp = Slerp(gt_t, gt_rots)
    t0 = gt_t[0]

    # event stream
    imu = load_csv(root/"imu0"/"data.csv")
    imu_ev = [(r[0]*1e-9, "imu", (r[1:4].tolist(), r[4:7].tolist())) for r in imu]
    idir = root/"cam0"/"data"
    with open(root/"cam0"/"data.csv") as f:
        rd = csv.reader(f); next(rd)
        img_ev = [(int(r[0])*1e-9, "img", idir/r[1].strip()) for r in rd if r]
    events = sorted(imu_ev+img_ev, key=lambda e: e[0])
    print(f"{len(imu_ev)} imu, {len(img_ev)} images; duration {gt_t[-1]-t0:.1f}s")

    rec = []   # per image frame: dict(t,p,q,tracked,total,age,ids)
    tstart = time.time()
    n_img = 0
    for stamp, et, data in events:
        if et == "imu":
            vio.process_imu(stamp, data[0], data[1])
            continue
        p = data
        if not p.exists():
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        feats, stats = tracker.process(gray)
        n_img += 1
        est = estq = None; lm_ids = ()
        if vio.is_initialized:
            vio.process_vision(stamp, {f["id"]: (f["x"], f["y"]) for f in feats})
            pos, quat = vio.get_pose(); est = np.array(pos); estq = np.array(quat)
            lm_ids = tuple(int(k) for k in vio.get_landmarks().keys())
        ages = [m["age"] for m in tracker.track_meta()]
        med_age = float(np.median(ages)) if ages else 0.0
        rec.append(dict(t=stamp-t0, p=est, q=estq, tracked=stats["tracked"],
                        total=stats["total"], age=med_age, ids=lm_ids))
        if n_img % 200 == 0:
            print(f"  [{n_img}] t={stamp-t0:5.1f}s tracked={stats['tracked']:3d} "
                  f"lm={len(lm_ids):2d} med_age={med_age:4.0f} "
                  f"{n_img/(time.time()-tstart):.0f}fps")

    # assemble matched trajectory
    tr = np.array([r["t"] for r in rec])
    have = np.array([r["p"] is not None for r in rec])
    est = np.array([r["p"] if r["p"] is not None else [np.nan]*3 for r in rec])
    estq = np.array([r["q"] if r["q"] is not None else [np.nan]*4 for r in rec])
    gtp = np.stack([np.interp(tr, gt_t-t0, gt_p[:, j]) for j in range(3)], axis=1)
    gtw = np.interp(tr, gt_t-t0, gt_w)
    gsp = np.interp(tr, gt_t-t0, gt_speed)
    ratio = (gsp / Z_SCENE) / np.maximum(gtw, 1e-3)      # parallax/rotation, <<1 = rot-dom
    gt_R = slerp(np.clip(tr+t0, gt_t[0], gt_t[-1])).as_matrix()
    tracked = np.array([r["tracked"] for r in rec]); med_age = np.array([r["age"] for r in rec])

    m = have & np.isfinite(est).all(1)
    R, t, sc, al = umeyama_se3(est[m], gtp[m])
    err = np.full(len(tr), np.nan)
    err[m] = np.linalg.norm(al - gtp[m], axis=1)
    ate = np.sqrt(np.nanmean(err[m]**2))
    print(f"\nSE(3)-aligned ATE RMSE = {ate*100:.1f} cm  (sim3 scale={sc:.3f}, "
          f"{m.sum()} matched frames)")

    # --- RPE over a fixed interval (alignment-free; local drift) ---
    kf = max(1, round(rpe_dt / np.median(np.diff(tr))))
    est_R = np.full((len(tr), 3, 3), np.nan)
    est_R[m] = Rot.from_quat(estq[m]).as_matrix()
    rpe_t = np.full(len(tr), np.nan); rpe_r = np.full(len(tr), np.nan)
    surv = np.full(len(tr), np.nan)                       # lm survival over the interval
    for i in range(len(tr) - kf):
        j = i + kf
        if not (m[i] and m[j]):
            continue
        re_t = est_R[i].T @ (est[j] - est[i])             # est rel-translation, body-i frame
        rg_t = gt_R[i].T @ (gtp[j] - gtp[i])              # gt  rel-translation, body-i frame
        rpe_t[i] = np.linalg.norm(re_t - rg_t)
        dR = (gt_R[i].T @ gt_R[j]).T @ (est_R[i].T @ est_R[j])
        rpe_r[i] = np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec()))
        a_ids, b_ids = set(rec[i]["ids"]), set(rec[j]["ids"])
        surv[i] = len(a_ids & b_ids) / len(a_ids) if a_ids else np.nan
    rpe_rms = np.sqrt(np.nanmean(rpe_t**2))
    print(f"RPE (dt={rpe_dt}s, kf={kf}): trans RMS {rpe_rms*100:.1f} cm, "
          f"rot RMS {np.sqrt(np.nanmean(rpe_r**2)):.2f} deg  "
          f"({np.isfinite(rpe_t).sum()} intervals)")
    # where is the worst local drift? (does it cluster near the end?)
    order = np.argsort(-np.nan_to_num(rpe_t))
    print("  top-10 highest-RPE intervals (t, RPE, ratio, |w|, med_age, surv):")
    for i in order[:10]:
        print(f"    t={tr[i]:6.1f}s  {rpe_t[i]*100:5.1f}cm  ratio={ratio[i]:5.2f}  "
              f"w={gtw[i]:4.2f}  age={med_age[i]:4.0f}  surv={surv[i]*100 if np.isfinite(surv[i]) else -1:3.0f}%")
    # RPE in the last 15 s vs the rest
    late = tr > tr[-1] - 15
    print(f"  RPE RMS  last-15s {np.sqrt(np.nanmean(rpe_t[late]**2))*100:.1f} cm  "
          f"vs  earlier {np.sqrt(np.nanmean(rpe_t[~late]**2))*100:.1f} cm")

    # --- whole-sequence correlations: does local drift track parallax starvation? ---
    def corr(x, y, label):
        g = np.isfinite(x) & np.isfinite(y)
        pr = pearsonr(x[g], y[g])[0]; sp = spearmanr(x[g], y[g])[0]
        print(f"  RPE_trans vs {label:22s}: pearson {pr:+.3f}  spearman {sp:+.3f}")
    print("\ncorrelations (whole sequence, per RPE interval):")
    corr(rpe_t, np.log10(np.maximum(ratio, 1e-3)), "log10(parallax ratio)")
    corr(rpe_t, med_age, "median track age")
    corr(rpe_t, surv, "landmark survival")
    corr(rpe_t, gtw, "GT |w|")

    # --- churn-controlled: does rotation act THROUGH churn or directly? ---
    g2 = np.isfinite(rpe_t) & np.isfinite(gtw) & np.isfinite(med_age) & np.isfinite(surv)
    def z(a):
        a = a[g2]; return (a - a.mean()) / a.std()
    Y, W, A, S = z(rpe_t), z(gtw), z(med_age), z(surv)
    def pcorr(x, y, *ctrl):
        """partial corr of x,y controlling for ctrl (regress both on ctrl, corr residuals)."""
        C = np.column_stack([np.ones_like(x)] + list(ctrl))
        rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
        return np.corrcoef(rx, ry)[0, 1]
    def ols(y, X):
        b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
        n, k = X.shape; s2 = r @ r / (n-k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        r2 = 1 - (r @ r) / ((y - y.mean())**2).sum()
        return b, b/se, r2
    print("\nchurn-controlled (standardized; rotation acts THROUGH churn if partials collapse):")
    print(f"  r(RPE, |w|)                 = {np.corrcoef(Y, W)[0,1]:+.3f}")
    print(f"  r(RPE, |w| | age)           = {pcorr(Y, W, A):+.3f}")
    print(f"  r(RPE, |w| | age, survival) = {pcorr(Y, W, A, S):+.3f}")
    print(f"  r(RPE, age | |w|)           = {pcorr(Y, A, W):+.3f}   (churn effect net of rotation)")
    b, tt, r2 = ols(Y, np.column_stack([np.ones_like(Y), W, A, S]))
    print(f"  OLS RPE ~ |w|+age+surv:  betas w {b[1]:+.3f}(t{tt[1]:+.1f})  "
          f"age {b[2]:+.3f}(t{tt[2]:+.1f})  surv {b[3]:+.3f}(t{tt[3]:+.1f})  R2={r2:.3f}")

    # quartile of RPE by parallax-ratio bin (the crisp test)
    g = np.isfinite(rpe_t) & np.isfinite(ratio)
    q = np.quantile(ratio[g], [0, .25, .5, .75, 1.0])
    print(f"\nRPE_trans by parallax-ratio quartile (Q1=most rotation-dominant):")
    print(f"  {'quartile':>10} {'ratio_range':>16} {'medRPE':>8} {'p90RPE':>8} {'medSurv':>8}")
    for k in range(4):
        b = g & (ratio >= q[k]) & (ratio <= q[k+1])
        print(f"  {'Q'+str(k+1):>10} [{q[k]:5.2f},{q[k+1]:5.2f}] "
              f"{np.nanmedian(rpe_t[b])*100:7.1f} {np.nanpercentile(rpe_t[b],90)*100:7.1f} "
              f"{np.nanmedian(surv[b])*100:7.0f}%")

    # rotation-dominant windows (from rot_segment_diag on this seq)
    windows = [(82.2, 83.0), (92.3, 93.6), (55.7, 56.9), (67.4, 68.0), (71.3, 71.8)]
    print("\nper-window (rotation-dominant): local RPE, not global ATE")
    print(f"  {'window':>14} {'w_pk':>6} {'medRPE':>7} {'age':>5} {'surv':>6}")
    for a, b in windows:
        win = (tr >= a) & (tr <= b)
        if win.sum() < 2:
            continue
        print(f"  [{a:5.1f},{b:5.1f}] {np.nanmax(gtw[win]):6.2f} "
              f"{np.nanmedian(rpe_t[win])*100:7.1f} {np.nanmedian(med_age[win]):5.0f} "
              f"{np.nanmedian(surv[win])*100:5.0f}%")

    # figure: timeseries + scatter
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for a, b in windows:
        for x in ax:
            x.axvspan(a, b, color="red", alpha=0.12)
    ax[0].plot(tr[m], err[m]*100, lw=1.0); ax[0].set_ylabel("abs err [cm]")
    ax[0].set_title(f"V1_03_difficult VIO — ATE {ate*100:.1f} cm, RPE {rpe_rms*100:.1f} cm; "
                    f"red = rotation-dominant windows")
    ax[1].plot(tr, rpe_t*100, lw=1.0, color="tab:purple"); ax[1].set_ylabel(f"RPE_t [cm/{rpe_dt}s]")
    ax[2].plot(tr, gtw, lw=1.0, color="tab:red"); ax[2].set_ylabel("GT |w| [rad/s]")
    axr = ax[2].twinx(); axr.plot(tr, ratio, lw=0.8, color="tab:blue", alpha=0.6)
    axr.set_ylabel("parallax ratio", color="tab:blue"); axr.set_ylim(0, 2)
    ax[3].plot(tr, tracked, lw=1.0, label="tracked feats")
    ax[3].plot(tr, med_age, lw=1.0, label="median track age", color="tab:green")
    ax[3].plot(tr, surv*100, lw=1.0, label="lm survival %", color="tab:orange")
    ax[3].set_ylabel("count / %"); ax[3].set_xlabel("t [s]"); ax[3].legend(loc="upper right")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    # scatter: RPE vs parallax ratio
    fig2, a2 = plt.subplots(figsize=(7, 5))
    a2.scatter(ratio[g], rpe_t[g]*100, s=6, alpha=0.3)
    a2.set_xscale("log"); a2.set_xlabel("parallax/rotation ratio (log)")
    a2.set_ylabel(f"RPE_trans [cm/{RPE_DT}s]")
    a2.set_title("local drift vs parallax starvation"); a2.grid(alpha=0.3)
    out2 = args.out.replace(".png", "_scatter.png")
    fig2.tight_layout(); fig2.savefig(out2, dpi=130)
    out3 = plot_rpe_intervals(args.out.replace(".png", "_howto.png"), tr, est, est_R,
                              gtp, gt_R, ratio, rpe_t, rpe_r, m, kf)
    cache = args.out.replace(".png", "_cache.npz")
    np.savez(cache, tr=tr, rpe_t=rpe_t, rpe_r=rpe_r, gtw=gtw, ratio=ratio,
             med_age=med_age, surv=surv, tracked=tracked, err=err, rpe_dt=rpe_dt)
    print(f"\nsaved {args.out}, {out2}, {out3}, {cache}")


if __name__ == "__main__":
    main()
