"""Stage 0 (MidAir) — track population: do long, clean, high-parallax tracks exist?

MidAir port of the EuRoC `track_population.py`. The MSCKF accuracy payoff needs the
"long AND pure AND high-parallax" quadrant to be populated in ECHO-LI's front-end.
On EuRoC that quadrant was measured near-empty (money quadrant 0.5–0.7%; ~95% of long
tracks drift >3px). This measures the SAME quantities on MidAir, on exactly the tracks
`midair_e2e_depth_nees.py` feeds Sparse3D — the real Rudolf-V front-end via
`tracker.process(img)` with NO VIO pose prior (that harness feeds no prior to the gate
either, so the track set is identical). GT geometry comes from MidAir's exact per-pixel
range map + exact pose (`midair_drift.MidAir`), so there is no ground-truth error.

Per Rudolf-V track over its whole life:

  length   : frames survived (max track age from track_meta)
  baseline : GT camera translation birth->last [m]
  parallax : subtended viewing-angle of the birth landmark, birth->last [deg]
  drift    : GT anchor-reprojection error [px] — anchor the birth pixel to its GT 3D
             world point (GT range + GT pose at birth), reproject into each later frame's
             GT pose, compare to the tracked pixel. Pure track ~0; drifted / mis-associated
             track grows. Direct "long red trail" purity measure.

Decision output (expectation-setting, NOT a build gate — the MSCKF build proceeds
regardless): of tracks that get LONG, what fraction stay CLEAN and carry usable parallax?
This calibrates how to interpret the navigation result: a non-empty money quadrant means
the active structureless update has raw material; an empty one means long tracks are
inherently contaminated here (the migration-note risk) and any accuracy win must come
from the covariance channel, not track length.

  PY=echo-li-python/venv/bin/python
  S=echo-li-python/tests/diagnostics/track_population_midair.py
  ROOT=~/18TB/datasets/dataset_MidAir/MidAir
  $PY $S --root $ROOT --set Kite_training --cond sunny --trajs 0,2 --frames 600
  $PY $S --root $ROOT --set VO_test       --cond sunny --trajs 0,2 --frames 600
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402
import echo_li  # noqa: E402

DRIFT_PX = 3.0          # a track is "clean" if its max anchor-drift stays below this
LONG_FRAMES = 20        # "long" track threshold (frames); ~0.8 s at 25 Hz MidAir
PARALLAX_DEG = 2.0      # "usable" triangulation parallax threshold


def run_traj(ds, tracker, W, H, f, cx, cy, nframes, start, out_png, dump, label):
    """Characterize every Rudolf-V track on one trajectory. Returns arrays + prints."""
    # Per-track running state keyed by frontend id (scalars only).
    tr = {}   # fid -> dict(birth_world, birth_cam, length, baseline, parallax,
              #             drift_max, drift_final, n_obs)
    tstart = time.time()
    last = min(start + nframes, ds.n)
    for i in range(start, last):
        img = ds.image(i)
        feats, _stats = tracker.process(img)
        if not feats:
            continue
        meta = {int(m["id"]): m for m in tracker.track_meta()}
        T_wb = ds.pose(i)                          # body->world (GT, exact)
        T_wc = T_wb @ md.RT_BC                      # camera->world
        cam_c = T_wc[:3, 3]
        depth_map = ds.depth(i)                     # EUCLIDEAN RANGE map (metres)

        for fd in feats:
            fid = int(fd["id"])
            u, v = float(fd["x"]), float(fd["y"])
            age = int(meta[fid]["age"]) if fid in meta else 1
            rec = tr.get(fid)
            if rec is None:
                # Birth: anchor the GT 3D world point from GT range at this pixel.
                gx, gy = int(round(u)), int(round(v))
                world = None
                if 0 <= gx < W and 0 <= gy < H:
                    d_rng = float(depth_map[gy, gx])
                    if 1.0 < d_rng < md.SKY:
                        world = md.backproject_world((u, v), d_rng, T_wb, f, cx, cy)
                tr[fid] = dict(birth_world=world, birth_cam=cam_c.copy(), length=age,
                               baseline=0.0, parallax=0.0, drift_max=0.0,
                               drift_final=0.0, n_obs=1)
                continue
            rec["length"] = max(rec["length"], age)
            rec["n_obs"] += 1
            rec["baseline"] = float(np.linalg.norm(cam_c - rec["birth_cam"]))
            world = rec["birth_world"]
            if world is not None:
                # anchor-drift: reproject birth world point into current GT pose.
                pred, _rng, z = md.project_world(world, T_wb, f, cx, cy)
                if pred is not None and z > 0.1:
                    drift = float(np.hypot(u - pred[0], v - pred[1]))
                    rec["drift_final"] = drift
                    rec["drift_max"] = max(rec["drift_max"], drift)
                # parallax: angle between the two viewing rays at the landmark.
                r0 = rec["birth_cam"] - world
                r1 = cam_c - world
                n0, n1 = np.linalg.norm(r0), np.linalg.norm(r1)
                if n0 > 1e-9 and n1 > 1e-9:
                    cosang = np.clip(np.dot(r0, r1) / (n0 * n1), -1, 1)
                    rec["parallax"] = max(rec["parallax"],
                                          float(np.degrees(np.arccos(cosang))))
        if (i - start) % 200 == 0:
            fps = (i - start + 1) / max(time.time() - tstart, 1e-9)
            print(f"  [{i - start}/{last - start}] tracks={len(tr)} {fps:.0f}fps")

    # Assemble per-track arrays (only GT-anchored, observed >=2 times get scored).
    scored = [r for r in tr.values() if r["birth_world"] is not None and r["n_obs"] >= 2]
    if not scored:
        print(f"  {label}: no GT-anchored tracks scored!")
        return None
    length = np.array([r["length"] for r in scored], float)
    baseline = np.array([r["baseline"] for r in scored], float)
    parallax = np.array([r["parallax"] for r in scored], float)
    drift_max = np.array([r["drift_max"] for r in scored], float)
    clean = drift_max < DRIFT_PX

    print(f"\n=== track population {label} ({len(scored)} GT-anchored tracks; "
          f"{len(tr)} total) ===")
    print(f"length  frames: median {np.median(length):.0f}  p90 {np.percentile(length,90):.0f}  "
          f"max {length.max():.0f}")
    print(f"baseline  m   : median {np.median(baseline):.2f}  p90 {np.percentile(baseline,90):.2f}")
    print(f"parallax  deg : median {np.median(parallax):.2f}  p90 {np.percentile(parallax,90):.2f}  "
          f"max {parallax.max():.2f}")
    print(f"drift_max px  : median {np.median(drift_max):.2f}  p90 {np.percentile(drift_max,90):.2f}")
    print(f"clean (drift<{DRIFT_PX:g}px) overall: {100*clean.mean():.1f}%\n")

    # The crisp test: as tracks get LONGER, do they stay clean and gain parallax?
    print(f"{'length>=L':>10} {'#tracks':>8} {'%clean':>7} {'medParallax':>11} "
          f"{'%clean&par>=' + str(PARALLAX_DEG):>14}")
    for L in (5, 10, 20, 40, 80):
        sel = length >= L
        if sel.sum() == 0:
            continue
        good = sel & clean & (parallax >= PARALLAX_DEG)
        print(f"{L:>10} {int(sel.sum()):>8} {100*clean[sel].mean():>6.1f}% "
              f"{np.median(parallax[sel]):>11.2f} {100*good.sum()/sel.sum():>13.1f}%")

    n_quad = int((clean & (length >= LONG_FRAMES) & (parallax >= PARALLAX_DEG)).sum())
    print(f"\nMONEY QUADRANT  long(>={LONG_FRAMES}f) & clean(<{DRIFT_PX:g}px) & "
          f"parallax(>={PARALLAX_DEG:g}deg): {n_quad} tracks "
          f"({100*n_quad/len(scored):.1f}% of scored)\n")

    if dump:
        np.savez(dump, length=length, baseline=baseline, parallax=parallax,
                 drift_max=drift_max)

    if out_png:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].hist(length, bins=40, color="tab:blue", alpha=0.8)
        ax[0].axvline(LONG_FRAMES, color="red", ls="--", label=f"long={LONG_FRAMES}")
        ax[0].set(xlabel="track length [frames]", ylabel="# tracks",
                  title="length distribution")
        ax[0].legend()
        sc = ax[1].scatter(length, parallax, c=np.clip(drift_max, 0, 10), s=8,
                           cmap="RdYlGn_r", alpha=0.6, vmin=0, vmax=10)
        ax[1].axhline(PARALLAX_DEG, color="k", ls=":", lw=1)
        ax[1].axvline(LONG_FRAMES, color="k", ls=":", lw=1)
        ax[1].set(xlabel="track length [frames]", ylabel="parallax [deg]",
                  title="length vs parallax (color = max drift px)")
        fig.colorbar(sc, ax=ax[1], label="max anchor-drift [px]")
        Ls = np.arange(2, int(length.max()) + 1)
        pct = [100 * clean[length >= L].mean() if (length >= L).sum() else np.nan for L in Ls]
        cnt = [(length >= L).sum() for L in Ls]
        ax[2].plot(Ls, pct, color="tab:green", label="% clean")
        ax[2].set(xlabel="length >= L [frames]", ylabel="% clean (drift<3px)",
                  title="does purity survive length?", ylim=(0, 100))
        axc = ax[2].twinx(); axc.semilogy(Ls, cnt, color="tab:gray", alpha=0.5)
        axc.set_ylabel("# tracks with length>=L", color="tab:gray")
        ax[2].legend(loc="lower left")
        fig.suptitle(f"{label} track population (MidAir, exact GT)")
        fig.tight_layout(); fig.savefig(out_png, dpi=130)
        print(f"saved {out_png}")

    return dict(length=length, parallax=parallax, drift_max=drift_max,
                n_quad=n_quad, n_scored=len(scored))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--root", required=True, help="path to MidAir root")
    ap.add_argument("--set", dest="subset", default="Kite_training")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=None, help="single trajectory index")
    ap.add_argument("--trajs", default=None,
                    help="comma-separated trajectory indices (e.g. 0,2); overrides --traj")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="image scale factor (0.5 = 512x256; matches e2e harness)")
    ap.add_argument("--config",
                    default=str(repo / "configs" / "diagnostics_midair_e2e_depth_nees.yaml"),
                    help="YAML with the frontend section (same as e2e harness)")
    ap.add_argument("--out-prefix", default="track_population_midair",
                    help="PNG/npz output prefix; per-traj suffix appended")
    ap.add_argument("--no-fig", action="store_true", help="skip PNG figures")
    args = ap.parse_args()

    if args.trajs is not None:
        trajs = [int(x) for x in args.trajs.split(",") if x.strip() != ""]
    elif args.traj is not None:
        trajs = [args.traj]
    else:
        trajs = [0]

    summary = []
    for traj in trajs:
        ds = md.MidAir(args.root, args.subset, args.cond, traj, args.scale)
        im0 = ds.image(args.start)
        H, W = im0.shape
        f, cx, cy = md.intrinsics(W, H)

        # Fresh tracker per trajectory (mirrors the e2e harness set-up: pinhole, no
        # distortion, frontend section read from the same YAML).
        fcfg = echo_li.FrontendConfig.from_yaml(args.config)
        fcfg.set_camera(f, f, cx, cy, W, H, [])
        tracker = echo_li.Frontend(fcfg, W, H)

        label = f"{args.subset}/{args.cond}/t{traj}"
        print(f"\n########## {label}  {W}x{H} f={f:.0f}  "
              f"{min(args.frames, ds.n - args.start)} frames ##########")
        out_png = None if args.no_fig else f"{args.out_prefix}_{args.subset}_t{traj}.png"
        dump = f"{args.out_prefix}_{args.subset}_t{traj}.npz"
        res = run_traj(ds, tracker, W, H, f, cx, cy, args.frames, args.start,
                       out_png, dump, label)
        if res is not None:
            summary.append((label, res))

    if len(summary) > 1:
        print(f"\n{'=' * 60}\nMONEY-QUADRANT SUMMARY (long>={LONG_FRAMES}f & clean<{DRIFT_PX:g}px "
              f"& parallax>={PARALLAX_DEG:g}deg)\n{'=' * 60}")
        print(f"{'trajectory':>28} {'#scored':>8} {'#quad':>7} {'%quad':>7}")
        for label, r in summary:
            print(f"{label:>28} {r['n_scored']:>8} {r['n_quad']:>7} "
                  f"{100*r['n_quad']/r['n_scored']:>6.1f}%")


if __name__ == "__main__":
    main()
