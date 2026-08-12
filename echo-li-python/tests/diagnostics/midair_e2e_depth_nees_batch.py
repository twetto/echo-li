"""Batch runner for midair_e2e_depth_nees.py across all trajectories in a MidAir subset.

Reads per-trajectory IMU noise + scene depth overrides from a JSON file
produced by midair_measure_imu_noise.py, then runs the e2e depth NEES test
on each trajectory and prints an aggregated summary table.

Usage:
  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_e2e_depth_nees_batch.py \
      --root ~/18TB/datasets/dataset_MidAir/MidAir \
      --set Kite_training \
      --config configs/diagnostics_midair_e2e_depth_nees.yaml \
      --noise-json /tmp/kite_imu_noise.json \
      --pose-range-scale 0.003 --min-track 20
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Batch depth NEES across all MidAir trajectories.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="Kite_training")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--config", required=True)
    ap.add_argument("--noise-json", required=True,
                    help="JSON from midair_measure_imu_noise.py")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--pose-range-scale", type=float, default=0.003)
    ap.add_argument("--min-track", type=int, default=20)
    ap.add_argument("--pvv-scale", type=float, default=1.0)
    ap.add_argument("--eqf-selection", choices=("grid", "existing-first"),
                    default="grid")
    ap.add_argument("--out-dir", default="/tmp/midair_e2e_depth_nees_batch",
                    help="directory for per-trajectory .npz results")
    args = ap.parse_args()

    noise = json.load(open(args.noise_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve().parent / "midair_e2e_depth_nees.py")
    python = sys.executable

    n_trajs = len(noise)
    for idx_str, params in sorted(noise.items(), key=lambda x: int(x[0])):
        idx = int(idx_str)
        npz_path = out_dir / f"traj_{idx}.npz"

        print(f"\n{'=' * 70}")
        print(f"  {args.subset} traj {idx} / {n_trajs - 1}  "
              f"vel_acc={params['sig_a']:.4f}  vel_gyr={params['sig_g']:.5f}  "
              f"sceneDepth={params['sceneDepth']:.0f}")
        print(f"{'=' * 70}")

        cmd = [
            python, script,
            "--root", args.root,
            "--set", args.subset,
            "--cond", args.cond,
            "--traj", str(idx),
            "--frames", str(args.frames),
            "--config", args.config,
            "--pose-range-scale", str(args.pose_range_scale),
            "--min-track", str(args.min_track),
            "--pvv-scale", str(args.pvv_scale),
            "--eqf-selection", args.eqf_selection,
            "--vel-acc", str(params["sig_a"]),
            "--vel-gyr", str(params["sig_g"]),
            "--bias-acc", str(params["biasAcc"]),
            "--bias-gyr", str(params["biasGyr"]),
            # sceneDepth: use the config default (200m) — large enough to
            # suppress phantom parallax from sky features at infinity.
            # "--scene-depth", str(params["sceneDepth"]),
            "--save-npz", str(npz_path),
            "--no-progress",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Print last 15 lines of output (the summary)
        lines = (result.stdout + result.stderr).strip().split("\n")
        for line in lines[-15:]:
            print(line)

    # --- Aggregate ---
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"{'tj':>3} {'E':>8} {'zone':>6} {'n':>6} {'med':>7} "
          f"{'%>95':>6} {'%>99':>6} {'err%':>7} {'sig':>6} | "
          f"{'tl20-40':>8} {'tl40-80':>8} {'tl80+':>8}")
    print("-" * 105)

    all_nees = []
    honest_nees = []
    for idx_str in sorted(noise.keys(), key=int):
        idx = int(idx_str)
        npz_path = out_dir / f"traj_{idx}.npz"
        if not npz_path.exists():
            print(f"{idx:3d} MISSING")
            continue
        A = np.load(npz_path, allow_pickle=True)["data"]
        if len(A) == 0:
            print(f"{idx:3d} NO OBS")
            continue

        nees = A[:, 4]
        tl = A[:, 1]
        relerr = A[:, 2] / A[:, 3]
        sig = A[:, 5]
        E = noise[idx_str]["E"]
        zone = "DEG" if E < 0.005 else ("PAR" if E < 0.05 else "SAFE")

        tl_meds = []
        for lo, hi in [(20, 40), (40, 80), (80, 1e9)]:
            m = (tl >= lo) & (tl < hi)
            tl_meds.append(f"{np.median(nees[m]):8.2f}" if m.sum() >= 5 else "     ---")

        p95 = 100 * np.mean(nees > 3.84)
        p99 = 100 * np.mean(nees > 6.63)
        err = 100 * np.median(relerr)
        honest = p95 <= 20 and abs(err) < 50

        print(f"{idx:3d} {E:8.5f} {zone:>6} {len(A):6d} {np.median(nees):7.3f} "
              f"{p95:5.1f}% {p99:5.1f}% {err:+6.1f}% {np.median(sig):6.1f} | "
              f"{tl_meds[0]} {tl_meds[1]} {tl_meds[2]}"
              f"{'  ✓' if honest else '  ✗ VIO'}")
        all_nees.extend(nees.tolist())
        if honest:
            honest_nees.extend(nees.tolist())

    if all_nees:
        all_nees = np.array(all_nees)
        print("-" * 105)
        print(f"{'ALL':>3} {'':>8} {'':>6} {len(all_nees):6d} "
              f"{np.median(all_nees):7.3f} "
              f"{100 * np.mean(all_nees > 3.84):5.1f}% "
              f"{100 * np.mean(all_nees > 6.63):5.1f}%")
    if honest_nees:
        honest_nees = np.array(honest_nees)
        n_honest = sum(1 for idx_str in noise
                       if (out_dir / f"traj_{int(idx_str)}.npz").exists())
        print(f"{'HON':>3} {'':>8} {'':>6} {len(honest_nees):6d} "
              f"{np.median(honest_nees):7.3f} "
              f"{100 * np.mean(honest_nees > 3.84):5.1f}% "
              f"{100 * np.mean(honest_nees > 6.63):5.1f}%")
    print(f"  t(2.6) ref:                        0.609    15.9%     9.5%")
    print(f"  chi2(1) ref:                       0.455     5.0%     1.0%")


if __name__ == "__main__":
    main()
