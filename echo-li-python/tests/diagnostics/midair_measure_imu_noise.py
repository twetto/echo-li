"""Measure per-trajectory IMU noise and scene depth for MidAir.

Computes, for each trajectory in the specified MidAir subset:
  - Accelerometer and gyroscope noise (per-sample std of GT residuals)
  - Bias magnitude (mean of residuals, squared for variance prior)
  - Median scene depth from the first frame's GT depth map
  - Excitation metric E = σ(ω_z) · σ(a_y)

Outputs a JSON file keyed by trajectory index, consumed by
midair_e2e_depth_nees.py via --vel-acc, --vel-gyr, --bias-acc, --bias-gyr,
--scene-depth CLI overrides (or by midair_e2e_depth_nees_batch.py).

Usage:
  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_measure_imu_noise.py \
      --root ~/18TB/datasets/dataset_MidAir/MidAir \
      --set Kite_training --out /tmp/kite_imu_noise.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md

SKY = getattr(md, "SKY", 500.0)


def measure_trajectory(db, traj_key, root, subset, cond, traj_idx, scale):
    """Return dict of measured noise parameters for one trajectory."""
    gt = db[traj_key]["groundtruth"]
    vel = gt["velocity"][:]
    att = gt["attitude"][:]
    acc_meas = db[traj_key]["imu"]["accelerometer"][:]
    gyr_meas = db[traj_key]["imu"]["gyroscope"][:]

    t_ned_to_nwu = np.diag([1.0, -1.0, -1.0])
    dt = 0.01  # 100 Hz
    g_up = np.array([0, 0, 9.81])

    n = min(len(acc_meas), len(att), len(vel))

    # GT body-frame specific force: R^T @ (a_world + g)
    vel_nwu = vel[:n] * np.array([1, -1, -1])  # NED → NWU
    a_world = (vel_nwu[1:] - vel_nwu[:-1]) / dt

    acc_resid = []
    gyr_resid = []
    for j in range(min(n - 1, len(acc_meas))):
        q = att[j]
        R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_nwu = t_ned_to_nwu @ R

        f_expected = R_nwu.T @ (a_world[j] + g_up)
        acc_resid.append(acc_meas[j] - f_expected)

        gyr_gt = gt["angular_velocity"][j]
        gyr_expected = R @ np.array(gyr_gt)  # body → spatial frame
        gyr_resid.append(gyr_meas[j] - gyr_expected)

    acc_resid = np.array(acc_resid)
    gyr_resid = np.array(gyr_resid)

    # Per-sample std (this is what the EqF velocityNoise expects)
    sig_a = float(np.mean([np.std(acc_resid[:, ax]) for ax in range(3)]))
    sig_g = float(np.mean([np.std(gyr_resid[:, ax]) for ax in range(3)]))

    # Bias = mean of residuals → variance prior = bias²
    bias_a = float(np.mean(np.abs(np.mean(acc_resid, axis=0))))
    bias_g = float(np.mean(np.abs(np.mean(gyr_resid, axis=0))))
    biasAcc = max(bias_a ** 2, 1e-6)
    biasGyr = max(bias_g ** 2, 1e-8)

    # Excitation metric E = σ(ω_z) · σ(a_y) over first 5 s
    n_e = min(500, len(gyr_meas))
    E = float(np.std(gyr_meas[:n_e, 2]) * np.std(acc_meas[:n_e, 1]))

    # Scene depth from first frame
    ds = md.MidAir(str(root), subset, cond, traj_idx, scale)
    d0 = ds.depth(0)
    valid = d0[(d0 > 0.5) & (d0 < SKY)]
    sceneDepth = float(np.median(valid)) if len(valid) > 0 else 24.0

    return {
        "E": E,
        "sig_a": sig_a,
        "sig_g": sig_g,
        "biasAcc": biasAcc,
        "biasGyr": biasGyr,
        "sceneDepth": sceneDepth,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Measure per-trajectory IMU noise + scene depth for MidAir.")
    ap.add_argument("--root", required=True, help="path to MidAir root")
    ap.add_argument("--set", dest="subset", default="Kite_training")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    hdf5_path = Path(args.root) / args.subset / args.cond / "sensor_records.hdf5"
    db = h5py.File(hdf5_path, "r")
    trajs = sorted(k for k in db.keys() if k.startswith("trajectory"))

    print(f"Measuring {len(trajs)} trajectories in {args.subset}/{args.cond}")
    print(f"{'traj':>5}  {'E':>8}  {'sig_a':>7}  {'sig_g':>7}  "
          f"{'biasAcc':>10}  {'biasGyr':>12}  {'depth':>6}")
    print("-" * 70)

    results = {}
    for t in trajs:
        idx = int(t.split("_")[1])
        r = measure_trajectory(db, t, args.root, args.subset, args.cond, idx, args.scale)
        results[str(idx)] = r
        zone = "DEG" if r["E"] < 0.005 else ("PAR" if r["E"] < 0.05 else "SAFE")
        print(f"{idx:5d}  {r['E']:8.5f}  {r['sig_a']:7.4f}  {r['sig_g']:7.5f}  "
              f"{r['biasAcc']:10.7f}  {r['biasGyr']:12.10f}  {r['sceneDepth']:6.0f}  {zone}")

    db.close()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
