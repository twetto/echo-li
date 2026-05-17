#!/usr/bin/env python3
"""EuRoC dataset tracking demo using echo-li's Rudolf-V frontend.

Usage:
    python euroc_tracking.py /path/to/V1_01_easy
    python euroc_tracking.py /path/to/V1_01_easy --config ../../configs/eqvio_euroc_euclid.yaml
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import echo_li


# ---------------------------------------------------------------------------
# EuRoC ASL data loader (pure Python, minimal)
# ---------------------------------------------------------------------------

def load_imu(dataset_root: Path):
    """Yield (stamp_s, gyr[3], acc[3]) from imu0/data.csv."""
    csv_path = dataset_root / "imu0" / "data.csv"
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            t = int(row[0]) * 1e-9
            gyr = [float(row[1]), float(row[2]), float(row[3])]
            acc = [float(row[4]), float(row[5]), float(row[6])]
            yield t, gyr, acc


def load_images(dataset_root: Path, cam_lag: float = 0.0):
    """Yield (stamp_s, image_path) from cam0/data.csv."""
    csv_path = dataset_root / "cam0" / "data.csv"
    data_dir = dataset_root / "cam0" / "data"
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            t = int(row[0]) * 1e-9 + cam_lag
            yield t, data_dir / row[1].strip()


def load_ground_truth(dataset_root: Path):
    """Load ground truth positions and timestamps from state_groundtruth_estimate0/data.csv.

    Returns (positions [N,3], timestamps [N]).
    """
    csv_path = dataset_root / "state_groundtruth_estimate0" / "data.csv"
    if not csv_path.exists():
        return None, None
    positions, times = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            times.append(int(row[0]) * 1e-9)
            positions.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(positions, dtype=np.float32), np.array(times)


def load_camera_config(dataset_root: Path):
    """Read cam0/sensor.yaml and return intrinsics + extrinsics.

    Returns (width, height, fx, fy, cx, cy, dist_model, dist_coeffs, T_BS_4x4_or_None).
    """
    import yaml

    sensor_yaml = dataset_root / "cam0" / "sensor.yaml"
    with open(sensor_yaml) as f:
        cfg = yaml.safe_load(f)

    w, h = cfg["resolution"]
    intr = cfg["intrinsics"]
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    dist_model = cfg.get("distortion_model")
    dist_coeffs = cfg.get("distortion_coefficients", [])

    t_bs = None
    if "T_BS" in cfg and "data" in cfg["T_BS"]:
        t_bs = np.array(cfg["T_BS"]["data"], dtype=np.float64).reshape(4, 4)

    return w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EuRoC tracking demo")
    parser.add_argument("dataset", help="Path to EuRoC dataset (e.g. V1_01_easy)")
    parser.add_argument("--config", help="Optional YAML config file")
    parser.add_argument("--max-features", type=int, default=200)
    parser.add_argument("--cam-lag", type=float, default=0.0)
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if (dataset_root / "mav0").exists():
        dataset_root = dataset_root / "mav0"

    # Load camera config (intrinsics + extrinsics)
    w, h, fx, fy, cx, cy, dist_model, dist_coeffs, t_bs = load_camera_config(dataset_root)
    print(f"Camera: {w}x{h}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    if dist_model:
        print(f"Distortion: {dist_model} {dist_coeffs}")

    # Create frontend
    if args.config:
        config = echo_li.FrontendConfig.from_yaml(args.config)
    else:
        config = echo_li.FrontendConfig(max_features=args.max_features)
    config.set_camera(fx, fy, cx, cy, w, h, dist_coeffs if dist_coeffs else [])
    tracker = echo_li.Frontend(config, w, h)
    print(f"Frontend: {tracker}")

    # Build time-sorted event stream
    print("Loading dataset...")
    imu_events = [(t, "imu", (gyr, acc)) for t, gyr, acc in load_imu(dataset_root)]
    img_events = [(t, "img", path) for t, path in load_images(dataset_root, args.cam_lag)]
    events = sorted(imu_events + img_events, key=lambda e: e[0])
    print(f"  {len(imu_events)} IMU, {len(img_events)} images")

    # Optionally set up VIO filter
    vio = None
    cam = None
    if args.config:
        if dist_model and "radtan" in dist_model.lower():
            cam = echo_li.RadTanCamera(fx, fy, cx, cy, *dist_coeffs[:4])
        else:
            cam = echo_li.PinholeCamera(fx, fy, cx, cy)
        vio = echo_li.VIOFilter(args.config, cam)
        if t_bs is not None:
            vio.set_camera_extrinsics(t_bs)
            print(f"Camera extrinsics (T_BS) loaded")
        print(f"VIO filter loaded from {args.config}")

    # Set up trajectory visualizer when VIO is active
    visualiser = None
    if vio:
        gt_pos, gt_times = load_ground_truth(dataset_root)
        if gt_pos is not None:
            print(f"  {len(gt_pos)} ground truth poses")
        from visualiser import TrajectoryVisualiser
        visualiser = TrajectoryVisualiser(gt_pos, gt_times)

    # Main loop
    show = not args.no_display
    frame_count = 0
    t_start = time.time()

    for stamp, etype, data in events:
        if etype == "imu":
            gyr, acc = data
            if vio:
                vio.process_imu(stamp, gyr, acc)
            continue

        # Image event
        img_path = data
        if not img_path.exists():
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        features, stats = tracker.process(gray)
        frame_count += 1

        # Run VIO vision update
        if vio and vio.is_initialized:
            feature_uvs = {f["id"]: (f["x"], f["y"]) for f in features}
            vio.process_vision(stamp, feature_uvs)

            if visualiser:
                pos, _ = vio.get_pose()
                landmarks = vio.get_landmarks()
                lm_dict = {int(k): np.array(v) for k, v in landmarks.items()}
                visualiser.update(stamp, np.array(pos), lm_dict)

        # Progress
        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed
            msg = f"[{frame_count:4d}] t={stamp:.3f}  tracked={stats['tracked']}  total={stats['total']}  {fps:.1f} fps"
            if vio and vio.is_initialized:
                pos, quat = vio.get_pose()
                msg += f"  pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
            print(msg)

        # Display
        if show:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for f in features:
                x, y = int(f["x"]), int(f["y"])
                cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    str(f["id"] % 1000),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 200, 200),
                    1,
                )
            cv2.putText(
                vis,
                f"frame {frame_count}  tracked {stats['tracked']}  total {stats['total']}  {stats['timing_ms']:.1f}ms",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow("ECHO-LI Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    elapsed = time.time() - t_start
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")

    if show:
        cv2.destroyAllWindows()

    if visualiser:
        visualiser.finish()


if __name__ == "__main__":
    main()
