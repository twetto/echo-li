"""Check MidAir IMU/GT frame conventions against HDF5 metadata.

This script is intentionally independent of EqF. It answers two questions:

1. Does `groundtruth/attitude` behave like body-to-world or world-to-body?
2. Do `imu/gyroscope` and `groundtruth/angular_velocity` match body or spatial
   angular velocity under that attitude convention?

Examples:
  PY=echo-li-python/venv/bin/python
  $PY echo-li-python/tests/diagnostics/midair_imu_frame_check.py \
      --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir --cond sunny --traj all
  $PY echo-li-python/tests/diagnostics/midair_imu_frame_check.py \
      --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir --set all --cond all --traj all
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def quat_wxyz_to_matrix(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def so3_log(R):
    return Rot.from_matrix(R).as_rotvec()


def robust_stats(err):
    n = np.linalg.norm(err, axis=1)
    return float(np.median(n)), float(np.percentile(n, 90))


def attr_text(dset):
    pairs = []
    for k, v in dset.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        pairs.append(f"{k}={v}")
    return ", ".join(pairs) if pairs else "(none)"


def finite_difference_gyro(R_bw, dt):
    body = []
    spatial = []
    for i in range(len(R_bw) - 1):
        Ri = R_bw[i]
        Rj = R_bw[i + 1]
        body.append(so3_log(Ri.T @ Rj) / dt)
        spatial.append(so3_log(Rj @ Ri.T) / dt)
    return np.asarray(body), np.asarray(spatial)


def accel_residuals(R_bw, gt_acc_world, imu_accel, gravity):
    pred = np.einsum("nij,nj->ni", np.transpose(R_bw, (0, 2, 1)), gt_acc_world - gravity)
    return imu_accel - pred


def check_traj(h5, dataset_name, traj_name, max_samples):
    g = h5[traj_name]["groundtruth"]
    imu = h5[traj_name]["imu"]

    n = len(g["attitude"])
    if max_samples > 0:
        n = min(n, max_samples)

    q = g["attitude"][:n]
    R_raw = np.asarray([quat_wxyz_to_matrix(x) for x in q])
    R_as_body_to_world = R_raw
    R_as_world_to_body = np.transpose(R_raw, (0, 2, 1))

    gt_acc = g["acceleration"][:n]
    gt_omega = g["angular_velocity"][:n]
    imu_acc = imu["accelerometer"][:n]
    imu_gyro = imu["gyroscope"][:n]
    bias = np.asarray(imu.attrs.get("init_bias_est", np.zeros(6)))
    gyro_unbiased = imu_gyro - bias[:3]

    gravity_ned = np.array([0.0, 0.0, 9.81])
    acc_bw = accel_residuals(R_as_body_to_world, gt_acc, imu_acc, gravity_ned)
    acc_wb = accel_residuals(R_as_world_to_body, gt_acc, imu_acc, gravity_ned)

    gyro_body_bw, gyro_spatial_bw = finite_difference_gyro(R_as_body_to_world, 0.01)
    gyro_body_wb, gyro_spatial_wb = finite_difference_gyro(R_as_world_to_body, 0.01)

    rows = []
    for source_name, source in [
        ("imu gyro - init_bias", gyro_unbiased[:-1]),
        ("gt angular_velocity", gt_omega[:-1]),
    ]:
        for convention_name, body, spatial in [
            ("q as body->world", gyro_body_bw, gyro_spatial_bw),
            ("q as world->body", gyro_body_wb, gyro_spatial_wb),
        ]:
            for frame_name, pred in [("body fd", body), ("spatial fd", spatial)]:
                med, p90 = robust_stats(source - pred)
                rows.append((source_name, convention_name, frame_name, med, p90))

    best_acc = "body->world" if robust_stats(acc_bw)[0] <= robust_stats(acc_wb)[0] else "world->body"
    best_imu = min(
        rows[:4],
        key=lambda x: x[3],
    )
    best_gt = min(
        rows[4:],
        key=lambda x: x[3],
    )

    print(f"\n{dataset_name}/{traj_name}")
    print(f"  imu/gyroscope attrs: {attr_text(imu['gyroscope'])}")
    print(f"  imu/accelerometer attrs: {attr_text(imu['accelerometer'])}")
    print(f"  gt/angular_velocity attrs: {attr_text(g['angular_velocity'])}")
    print(f"  gt/attitude attrs: {attr_text(g['attitude'])}")
    print(f"  gt/acceleration attrs: {attr_text(g['acceleration'])}")
    med, p90 = robust_stats(acc_bw)
    print(f"  accel residual if q is body->world: median={med:.4g} p90={p90:.4g} m/s^2")
    med, p90 = robust_stats(acc_wb)
    print(f"  accel residual if q is world->body: median={med:.4g} p90={p90:.4g} m/s^2")
    for source_name, convention_name, frame_name, med, p90 in rows:
        print(f"  {source_name:22s} vs {convention_name:16s} {frame_name:10s}: "
              f"median={med:.5g} p90={p90:.5g} rad/s")
    print(f"  best: accel q={best_acc}; imu gyro={best_imu[1]} {best_imu[2]}; "
          f"gt omega={best_gt[1]} {best_gt[2]}")
    return {
        "dataset": dataset_name,
        "traj": traj_name,
        "best_acc": best_acc,
        "best_imu": f"{best_imu[1]} {best_imu[2]}",
        "best_gt": f"{best_gt[1]} {best_gt[2]}",
        "imu_med": best_imu[3],
        "gt_med": best_gt[3],
    }


def h5_paths(root, subset, cond):
    root = Path(root)
    if subset == "all":
        paths = sorted(root.glob("*/**/sensor_records.hdf5"))
    elif cond == "all":
        paths = sorted((root / subset).glob("*/sensor_records.hdf5"))
    else:
        paths = [root / subset / cond / "sensor_records.hdf5"]
    return [p for p in paths if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test", help="dataset split, or 'all'")
    ap.add_argument("--cond", default="sunny", help="condition/season, or 'all'")
    ap.add_argument("--traj", default="all", help="'all' or a trajectory number")
    ap.add_argument("--max-samples", type=int, default=5000)
    args = ap.parse_args()

    summaries = []
    for h5_path in h5_paths(args.root, args.subset, args.cond):
        dataset_name = str(h5_path.relative_to(args.root).parent)
        with h5py.File(h5_path, "r") as h5:
            if args.traj == "all":
                trajs = sorted(k for k in h5.keys() if k.startswith("trajectory_"))
            else:
                trajs = [f"trajectory_{int(args.traj):04d}"]
            for traj in trajs:
                summaries.append(check_traj(h5, dataset_name, traj, args.max_samples))

    if summaries:
        print("\n=== Summary ===")
        print(f"{'dataset':25s} {'traj':15s} {'accel q':12s} {'imu gyro best':30s} "
              f"{'gt omega best':30s} {'imu med':>9s} {'gt med':>9s}")
        for s in summaries:
            print(f"{s['dataset']:25s} {s['traj']:15s} {s['best_acc']:12s} "
                  f"{s['best_imu']:30s} {s['best_gt']:30s} "
                  f"{s['imu_med']:9.4g} {s['gt_med']:9.4g}")
    else:
        print("no sensor_records.hdf5 files matched")


if __name__ == "__main__":
    main()
