"""Stage 0 — track population: do long, clean, high-parallax tracks exist?

The MSCKF question reduces to: is the "long AND pure AND high-baseline" quadrant
populated in ECHO-LI's front-end? (an earlier sweep showed naive
long tracks are contaminated; rudolf keeps them short by design.) This measures,
per Rudolf-V track over its whole life:

  length   : frames survived (max track age)
  baseline : GT camera translation birth->last [m]
  parallax : subtended viewing-angle of the birth landmark, birth->last [deg]
  drift    : GT anchor-reprojection error [px] — take the birth pixel's GT 3D
             point (GT depth z-buffer + GT pose at birth), reproject it into each
             later frame's GT pose, compare to the tracked pixel. A pure track
             stays ~0; a drifted / mis-associated track grows. This is the direct
             "long red trail" purity measure, stronger than frame-to-frame error.

Decision output: of tracks that get LONG, what fraction stay CLEAN, and do they
carry usable parallax? If long+clean+high-parallax is well populated, the untested
"retain length + epipolar-gate purity" quadrant has headroom (and long tracks can
help depth/pose); if not, long tracks are inherently contaminated here and the
MSCKF premise is dead. GT depth needs pointcloud0 -> Vicon rooms only (MH has none).

  .venv/Scripts/python.exe echo-li-python/tests/diagnostics/track_population.py \
      ~/Downloads/vicon_room1/vicon_room1/V1_03_difficult
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_depth_eval import load_csv, zbuf_cache, zbuf_lookup  # noqa: E402
import echo_li  # noqa: E402

CELL = 4
DRIFT_PX = 3.0          # a track is "clean" if its max anchor-drift stays below this
LONG_FRAMES = 20        # "long" track threshold (frames); ~1 s at 20 Hz EuRoC
PARALLAX_DEG = 2.0      # "usable" triangulation parallax threshold


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("dataset")
    ap.add_argument("--config", default=str(repo / "configs" / "eqvio_euroc_rho.yaml"))
    ap.add_argument("--gate", type=float, default=0.0,
                    help="epipolar gate threshold (0 = ungated tracker; 1e-5 = shipped)")
    ap.add_argument("--out", default="track_population.png")
    ap.add_argument("--dump", default=None, help="save per-track table (npz)")
    args = ap.parse_args()
    root = Path(args.dataset)
    if (root / "mav0").exists():
        root = root / "mav0"

    cfg = yaml.safe_load(open(root / "cam0" / "sensor.yaml"))
    w, h = cfg["resolution"]; fx, fy, cx, cy = cfg["intrinsics"]
    dcoef = np.array(cfg.get("distortion_coefficients", []), float)
    t_bs = np.array(cfg["T_BS"]["data"], float).reshape(4, 4)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    gt = load_csv(root / "state_groundtruth_estimate0" / "data.csv")
    gt_t = gt[:, 0] * 1e-9; gt_p = gt[:, 1:4]
    slerp = Slerp(gt_t, Rot.from_quat(gt[:, 4:8][:, [1, 2, 3, 0]]))

    def cam_pose(t):
        t_wb = np.eye(4)
        t_wb[:3, :3] = slerp(np.clip(t, gt_t[0], gt_t[-1])).as_matrix()
        t_wb[:3, 3] = [np.interp(t, gt_t, gt_p[:, j]) for j in range(3)]
        return t_wb @ t_bs

    fcfg = echo_li.FrontendConfig.from_yaml(args.config)
    fcfg.epipolar_gate_threshold = args.gate
    fcfg.set_camera(fx, fy, cx, cy, w, h, dcoef.tolist())
    tracker = echo_li.Frontend(fcfg, w, h)

    # Full VIO alongside: the shipped epipolar gate is fed the EqF's IMU-propagated
    # pose prior each frame (RANSAC fallback when absent). Without this a gate>0 run
    # would only exercise estimated-E RANSAC, not the shipped pose-prior gate.
    use_prior = args.gate > 0 and hasattr(tracker, "set_pose_prior")
    cam = (echo_li.RadTanCamera(fx, fy, cx, cy, *dcoef[:4])
           if len(dcoef) >= 4 else echo_li.PinholeCamera(fx, fy, cx, cy))
    vio = echo_li.VIOFilter(args.config, cam)
    vio.set_camera_extrinsics(t_bs)
    imu = load_csv(root / "imu0" / "data.csv")
    imu_ev = [(r[0] * 1e-9, r[1:4].tolist(), r[4:7].tolist()) for r in imu]
    imu_i = 0
    vio_cam_prev = None

    idir = root / "cam0" / "data"
    with open(root / "cam0" / "data.csv") as f:
        rd = csv.reader(f); next(rd)
        frames = [(int(r[0]) * 1e-9, idir / r[1].strip()) for r in rd if r]
    frames = [(t, p) for t, p in frames if gt_t[0] <= t <= gt_t[-1]]
    zbufs = zbuf_cache(root, frames, cam_pose, fx, fy, cx, cy, w, h, CELL)

    # Per-track running state (keyed by frontend id). Scalars only.
    tr = {}   # fid -> dict(birth_world, birth_cam, length, baseline, parallax, drift_max, drift_final, n_obs)
    tstart = time.time()
    for i, (t, p) in enumerate(frames):
        while imu_i < len(imu_ev) and imu_ev[imu_i][0] <= t:
            vio.process_imu(*imu_ev[imu_i])
            imu_i += 1
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Feed the EqF's IMU-propagated pose prior to the epipolar gate (relative
        # c_prev -> c_curr), exactly as the shipped CLI does.
        if use_prior and vio.is_initialized:
            pos_p, quat_p = vio.get_pose()
            t_wb_p = np.eye(4)
            t_wb_p[:3, :3] = Rot.from_quat(np.asarray(quat_p)).as_matrix()
            t_wb_p[:3, 3] = np.asarray(pos_p)
            vio_cam = t_wb_p @ t_bs
            if vio_cam_prev is not None:
                tracker.set_pose_prior(np.linalg.inv(vio_cam) @ vio_cam_prev)
        feats, _ = tracker.process(img)
        if vio.is_initialized:
            vio.process_vision(t, {f["id"]: (f["x"], f["y"]) for f in feats})
            pos, quat = vio.get_pose()
            t_wb_c = np.eye(4)
            t_wb_c[:3, :3] = Rot.from_quat(np.asarray(quat)).as_matrix()
            t_wb_c[:3, 3] = np.asarray(pos)
            vio_cam_prev = t_wb_c @ t_bs
        if not feats:
            continue
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        px = np.array([[f["x"], f["y"]] for f in feats], np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, K, dcoef[:4], P=K).reshape(-1, 2)
        t_wc = cam_pose(t); t_cw = np.linalg.inv(t_wc); cam_c = t_wc[:3, 3]

        # GT depth at every current undistorted pixel (NaN where no pointcloud).
        depth = zbuf_lookup(zbufs[i], und, w, h, CELL)

        for k, f in enumerate(feats):
            fid = int(f["id"])
            uv = und[k]
            age = int(meta[fid]["age"]) if fid in meta else 1
            rec = tr.get(fid)
            if rec is None:
                # Birth: anchor the GT 3D point from GT depth at this pixel.
                d = depth[k]
                world = None
                if np.isfinite(d) and d > 0:
                    pc = np.array([(uv[0] - cx) / fx * d, (uv[1] - cy) / fy * d, d])
                    world = t_wc[:3, :3] @ pc + t_wc[:3, 3]
                tr[fid] = dict(birth_world=world, birth_cam=cam_c, length=age,
                               baseline=0.0, parallax=0.0, drift_max=0.0,
                               drift_final=0.0, n_obs=1)
                continue
            rec["length"] = max(rec["length"], age)
            rec["n_obs"] += 1
            rec["baseline"] = float(np.linalg.norm(cam_c - rec["birth_cam"]))
            world = rec["birth_world"]
            if world is not None:
                # anchor-drift: reproject birth world point into current GT pose
                pc = t_cw[:3, :3] @ world + t_cw[:3, 3]
                if pc[2] > 0.1:
                    pred = np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy])
                    drift = float(np.hypot(uv[0] - pred[0], uv[1] - pred[1]))
                    rec["drift_final"] = drift
                    rec["drift_max"] = max(rec["drift_max"], drift)
                # parallax: angle between the two viewing rays at the landmark
                r0 = rec["birth_cam"] - world; r1 = cam_c - world
                n0, n1 = np.linalg.norm(r0), np.linalg.norm(r1)
                if n0 > 1e-9 and n1 > 1e-9:
                    cosang = np.clip(np.dot(r0, r1) / (n0 * n1), -1, 1)
                    rec["parallax"] = max(rec["parallax"], float(np.degrees(np.arccos(cosang))))
        if i % 400 == 0:
            print(f"  [{i}/{len(frames)}] tracks={len(tr)} "
                  f"{i / max(time.time() - tstart, 1e-9):.0f}fps")

    # Assemble per-track arrays (only tracks with a GT anchor can be scored for drift).
    scored = [r for r in tr.values() if r["birth_world"] is not None and r["n_obs"] >= 2]
    length = np.array([r["length"] for r in scored], float)
    baseline = np.array([r["baseline"] for r in scored], float)
    parallax = np.array([r["parallax"] for r in scored], float)
    drift_max = np.array([r["drift_max"] for r in scored], float)
    clean = drift_max < DRIFT_PX

    print(f"\n=== track population ({len(scored)} GT-anchored tracks; "
          f"{len(tr)} total; gate={args.gate:g}) ===")
    print(f"length  frames: median {np.median(length):.0f}  p90 {np.percentile(length,90):.0f}  "
          f"max {length.max():.0f}")
    print(f"parallax  deg : median {np.median(parallax):.2f}  p90 {np.percentile(parallax,90):.2f}  "
          f"max {parallax.max():.2f}")
    print(f"drift_max px  : median {np.median(drift_max):.2f}  p90 {np.percentile(drift_max,90):.2f}")
    print(f"clean (drift<{DRIFT_PX:g}px) overall: {100*clean.mean():.1f}%\n")

    # The crisp test: as tracks get LONGER, do they stay clean, and gain parallax?
    print(f"{'length>=L':>10} {'#tracks':>8} {'%clean':>7} {'medParallax':>11} "
          f"{'%clean&par>=' + str(PARALLAX_DEG):>14}")
    for L in (5, 10, 20, 40, 80):
        sel = length >= L
        if sel.sum() == 0:
            continue
        good = sel & clean & (parallax >= PARALLAX_DEG)
        print(f"{L:>10} {sel.sum():>8} {100*clean[sel].mean():>6.1f}% "
              f"{np.median(parallax[sel]):>11.2f} {100*good.sum()/sel.sum():>13.1f}%")

    n_quad = int((clean & (length >= LONG_FRAMES) & (parallax >= PARALLAX_DEG)).sum())
    print(f"\nMONEY QUADRANT  long(>={LONG_FRAMES}f) & clean(<{DRIFT_PX:g}px) & "
          f"parallax(>={PARALLAX_DEG:g}deg): {n_quad} tracks "
          f"({100*n_quad/len(scored):.1f}% of scored)")

    if args.dump:
        np.savez(args.dump, length=length, baseline=baseline, parallax=parallax,
                 drift_max=drift_max)

    # Figures: length histogram; length-vs-parallax scatter colored by drift.
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(length, bins=40, color="tab:blue", alpha=0.8)
    ax[0].axvline(LONG_FRAMES, color="red", ls="--", label=f"long={LONG_FRAMES}")
    ax[0].set(xlabel="track length [frames]", ylabel="# tracks", title="length distribution")
    ax[0].legend()
    sc = ax[1].scatter(length, parallax, c=np.clip(drift_max, 0, 10), s=8, cmap="RdYlGn_r",
                       alpha=0.6, vmin=0, vmax=10)
    ax[1].axhline(PARALLAX_DEG, color="k", ls=":", lw=1)
    ax[1].axvline(LONG_FRAMES, color="k", ls=":", lw=1)
    ax[1].set(xlabel="track length [frames]", ylabel="parallax [deg]",
              title="length vs parallax (color = max drift px)")
    fig.colorbar(sc, ax=ax[1], label="max anchor-drift [px]")
    # %clean vs length (does purity survive length?)
    Ls = np.arange(2, int(length.max()) + 1)
    pct = [100 * clean[length >= L].mean() if (length >= L).sum() else np.nan for L in Ls]
    cnt = [(length >= L).sum() for L in Ls]
    ax[2].plot(Ls, pct, color="tab:green", label="% clean")
    ax[2].set(xlabel="length >= L [frames]", ylabel="% clean (drift<3px)",
              title="does purity survive length?", ylim=(0, 100))
    axc = ax[2].twinx(); axc.semilogy(Ls, cnt, color="tab:gray", alpha=0.5)
    axc.set_ylabel("# tracks with length>=L", color="tab:gray")
    ax[2].legend(loc="lower left")
    fig.suptitle(f"{root.parents[0].name} track population (gate={args.gate:g})")
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
