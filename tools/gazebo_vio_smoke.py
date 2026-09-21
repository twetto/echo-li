#!/usr/bin/env python3
"""Thin ROS 2 smoke adapter for Gazebo -> Rudolf-V / ECHO-LI.

Ground truth is stored only by the evaluator and never enters the frontend or
filter. The VIO mode reports rigidly aligned trajectory error after shutdown.
"""

import argparse
from collections import deque
import json
import statistics
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu

import echo_li


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def align_positions(estimated, truth, with_scale=False):
    """Align estimated positions to truth and return RMSE, scale, and points."""
    if len(estimated) < 3:
        return None, None, None
    a = np.asarray(estimated, dtype=np.float64)
    b = np.asarray(truth, dtype=np.float64)
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    rotated = ac @ r.T
    scale = 1.0
    if with_scale:
        denominator = float(np.sum(rotated * rotated))
        if denominator > 0.0:
            scale = float(np.sum(rotated * bc) / denominator)
    aligned = scale * rotated + b.mean(axis=0)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))
    return rmse, scale, aligned


def quaternion_matrix_xyzw(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1.0e-15:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


class GazeboEchoLi(Node):
    def __init__(self, args):
        super().__init__('echo_li_gazebo_smoke')
        self.args = args
        self.frontend = None
        self.vio = None
        self.camera = None
        self.pending_imu = []
        self.imu_queue = deque()
        self.vision_queue = deque()
        self.depth_by_stamp = {}
        self.right_by_stamp = {}
        self.last_depth_received_stamp = -float('inf')
        self.last_right_received_stamp = -float('inf')
        self.last_imu_received_stamp = -float('inf')
        self.last_imu_processed_stamp = -float('inf')
        self.event_order_violations = 0
        self.max_imu_queue = 0
        self.max_vision_queue = 0
        self.imu_stamps = []
        self.imu_gyro = []
        self.imu_accel = []
        self.truth_stamps = []
        self.truth_positions_all = []
        self.truth_quaternions = []
        self.truth_linear_velocity = []
        self.truth_angular_velocity = []
        self.estimated_stamps = []
        self.estimated_positions = []
        self.estimated_quaternions = []
        self.estimated_velocities = []
        self.estimated_gyro_biases = []
        self.estimated_accel_biases = []
        self.frame_count = 0
        self.imu_count = 0
        self.bad_images = 0
        self.totals = []
        self.tracked = []
        self.timing_ms = []
        self.depth_messages = 0
        self.depth_matches = 0
        self.depth_estimated_ranges = []
        self.depth_ground_truth_ranges = []
        self.depth_track_ages = []
        self.depth_landmark_candidates = 0
        self.depth_landmark_valid = 0
        self.gt_depth_prior_candidates = 0
        self.gt_depth_prior_births = 0
        self.stereo = None
        self.stereo_messages = 0
        self.stereo_matches = 0
        self.stereo_prior_candidates = 0
        self.stereo_prior_births = 0
        self.start_wall = time.monotonic()
        self.first_image_wall = None
        self.last_image_wall = None
        self.last_report = self.start_wall
        self.rerun_aligned_truth = np.empty((0, 3))
        self.rr = None
        if args.visualize:
            import rerun as rr
            import rerun.blueprint as rrb
            self.rr = rr
            rr.init('echo_li_gazebo_vio', spawn=True)
            rr.send_blueprint(rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            name='Tracker', origin='camera',
                            contents=['camera/**'],
                            visual_bounds=rrb.VisualBounds2D(
                                x_range=[0.0, 512.0], y_range=[0.0, 512.0])),
                        rrb.Spatial2DView(
                            name='Depth GT', origin='depth',
                            contents=['depth/**'],
                            visual_bounds=rrb.VisualBounds2D(
                                x_range=[0.0, 512.0], y_range=[0.0, 512.0])),
                        rrb.Spatial2DView(
                            name='Right camera', origin='stereo',
                            contents=['stereo/**'],
                            visual_bounds=rrb.VisualBounds2D(
                                x_range=[0.0, 512.0], y_range=[0.0, 512.0]))),
                    rrb.Spatial3DView(
                        name='World', origin='world', contents=['world/**']),
                    column_shares=[1.0, 1.0]),
                auto_layout=False, auto_views=False, collapse_panels=True))

        self.create_subscription(
            CameraInfo, '/echo_li/camera/camera_info', self.on_info,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, '/echo_li/camera/image', self.on_image,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, '/echo_li/camera/right/image', self.on_right,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, '/echo_li/camera/depth', self.on_depth,
            qos_profile_sensor_data)
        self.create_subscription(
            Imu, '/echo_li/imu', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, '/echo_li/ground_truth', self.on_truth,
            qos_profile_sensor_data)

    def on_info(self, msg):
        if self.frontend is not None:
            return
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.camera = echo_li.PinholeCamera(fx, fy, cx, cy)
        cfg = echo_li.FrontendConfig.from_yaml(str(self.args.config))
        cfg.set_camera(fx, fy, cx, cy, msg.width, msg.height, [])
        self.frontend = echo_li.Frontend(cfg, msg.width, msg.height)
        if self.args.stereo:
            self.stereo = echo_li.Stereo.from_pinhole(
                fx, fy, cx, cy, msg.width, msg.height,
                [-self.args.stereo_baseline, 0.0, 0.0], None,
                str(self.args.config))
        if self.args.vio:
            self.vio = echo_li.VIOFilter(str(self.args.config), self.camera)
            # Gazebo body FLU (x forward, y left, z up) to optical camera
            # (x right, y down, z forward), expressed as T_BS rotation.
            t_bs = np.eye(4, dtype=np.float64)
            t_bs[:3, :3] = np.array([
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ])
            t_bs[:3, 3] = [0.18, 0.06, 0.0]
            self.vio.set_camera_extrinsics(t_bs)
            self.imu_queue.extend(self.pending_imu)
        self.pending_imu.clear()
        mode = 'vio' if self.args.vio else 'frontend'
        self.get_logger().info(
            f'camera ready: {msg.width}x{msg.height}, '
            f'fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, '
            f'mode={mode}, stereo={self.args.stereo}')

    def on_right(self, msg):
        if not self.args.stereo:
            return
        try:
            gray = self.image_gray(msg)
        except ValueError as exc:
            self.bad_images += 1
            self.get_logger().error(str(exc))
            return
        stamp = stamp_seconds(msg.header.stamp)
        key = int(round(stamp * 1.0e9))
        self.right_by_stamp[key] = gray
        self.last_right_received_stamp = max(self.last_right_received_stamp, stamp)
        self.stereo_messages += 1
        if len(self.right_by_stamp) > 128:
            for old_key in sorted(self.right_by_stamp)[:-128]:
                del self.right_by_stamp[old_key]
        self.drain_vio_queue()

    def on_imu(self, msg):
        stamp = stamp_seconds(msg.header.stamp)
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z]
        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y,
                 msg.linear_acceleration.z]
        self.imu_count += 1
        self.imu_stamps.append(stamp)
        self.imu_gyro.append(gyro)
        self.imu_accel.append(accel)
        if stamp <= self.last_imu_received_stamp:
            self.event_order_violations += 1
        self.last_imu_received_stamp = max(self.last_imu_received_stamp, stamp)
        if self.vio is None:
            if len(self.pending_imu) < 1000:
                self.pending_imu.append((stamp, gyro, accel))
            return
        self.imu_queue.append((stamp, gyro, accel))
        self.max_imu_queue = max(self.max_imu_queue, len(self.imu_queue))
        self.drain_vio_queue()

    def on_truth(self, msg):
        self.truth_stamps.append(stamp_seconds(msg.header.stamp))
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        self.truth_positions_all.append([p.x, p.y, p.z])
        self.truth_quaternions.append([q.x, q.y, q.z, q.w])
        self.truth_linear_velocity.append([v.x, v.y, v.z])
        self.truth_angular_velocity.append([w.x, w.y, w.z])

    @staticmethod
    def depth_float32(msg):
        if msg.encoding != '32FC1':
            raise ValueError(
                f'unsupported depth encoding: {msg.encoding}; expected 32FC1')
        byte_rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.step)
        packed = byte_rows[:, :msg.width * 4].copy()
        dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
        return packed.view(dtype).reshape(msg.height, msg.width)

    def on_depth(self, msg):
        try:
            depth = self.depth_float32(msg)
        except ValueError as exc:
            self.bad_images += 1
            self.get_logger().error(str(exc))
            return
        stamp = stamp_seconds(msg.header.stamp)
        self.depth_by_stamp[int(round(stamp * 1.0e9))] = depth
        self.last_depth_received_stamp = max(
            self.last_depth_received_stamp, stamp)
        self.depth_messages += 1
        if len(self.depth_by_stamp) > 128:
            for key in sorted(self.depth_by_stamp)[:-128]:
                del self.depth_by_stamp[key]
        self.drain_vio_queue()

    @staticmethod
    def ground_truth_range(depth, feature):
        u = int(round(float(feature['x'])))
        v = int(round(float(feature['y'])))
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return None
        patch = depth[max(0, v-1):min(depth.shape[0], v+2),
                      max(0, u-1):min(depth.shape[1], u+2)]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 80.0)]
        if valid.size == 0:
            return None
        optical_z = float(np.median(valid))
        nx = (float(feature['x']) - 256.0) / 443.405
        ny = (float(feature['y']) - 256.0) / 443.405
        return optical_z * np.sqrt(nx*nx + ny*ny + 1.0)

    @staticmethod
    def image_gray(msg):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == 'mono8':
            return raw.reshape(msg.height, msg.step)[:, :msg.width].copy()
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'unsupported encoding: {msg.encoding}')
        rgb = raw.reshape(msg.height, msg.step)[:, :msg.width * 3]
        rgb = rgb.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            rgb = rgb[:, :, ::-1]
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] +
                0.114 * rgb[:, :, 2])
        return gray.astype(np.uint8)

    def on_image(self, msg):
        if self.frontend is None:
            return
        try:
            gray = self.image_gray(msg)
        except ValueError as exc:
            self.bad_images += 1
            self.get_logger().error(str(exc))
            return
        features, stats = self.frontend.process(gray)
        track_meta = {int(m['id']): m for m in self.frontend.track_meta()}
        preprocessed = self.frontend.preprocessed_image()
        display_gray = (np.asarray(preprocessed).copy()
                        if preprocessed is not None else gray)
        now = time.monotonic()
        if self.first_image_wall is None:
            self.first_image_wall = now
        self.last_image_wall = now
        self.frame_count += 1
        self.totals.append(int(stats['total']))
        self.tracked.append(int(stats['tracked']))
        self.timing_ms.append(float(stats['timing_ms']))

        if self.vio is not None:
            stamp = stamp_seconds(msg.header.stamp)
            observations = {
                int(f['id']): (float(f['x']), float(f['y'])) for f in features
            }
            self.vision_queue.append(
                (stamp, observations, display_gray, features, track_meta))
            self.max_vision_queue = max(
                self.max_vision_queue, len(self.vision_queue))
            self.drain_vio_queue()

        if now - self.last_report >= 5.0:
            recent = min(125, len(self.totals))
            self.get_logger().info(
                f'frames={self.frame_count} imu={self.imu_count} '
                f'total_med={statistics.median(self.totals[-recent:]):.1f} '
                f'tracked_med={statistics.median(self.tracked[-recent:]):.1f} '
                f'frontend_ms_med={statistics.median(self.timing_ms[-recent:]):.2f}')
            self.last_report = now

    def drain_vio_queue(self):
        if self.vio is None:
            return
        while (self.vision_queue and self.imu_queue and
               self.last_imu_received_stamp >= self.vision_queue[0][0]):
            image_stamp = self.vision_queue[0][0]
            depth_key = int(round(image_stamp * 1.0e9))
            depth = self.depth_by_stamp.get(depth_key)
            if depth is None and self.last_depth_received_stamp < image_stamp:
                break
            right = self.right_by_stamp.get(depth_key)
            if self.args.stereo and right is None:
                if self.last_right_received_stamp < image_stamp:
                    break
                self.get_logger().warning(
                    f'no exact right-image match at {image_stamp:.9f}')
            _, observations, gray, features, track_meta = self.vision_queue.popleft()
            if depth is not None:
                del self.depth_by_stamp[depth_key]
                self.depth_matches += 1
            if right is not None:
                del self.right_by_stamp[depth_key]
                self.stereo_matches += 1
            while self.imu_queue and self.imu_queue[0][0] <= image_stamp:
                stamp, gyro, accel = self.imu_queue.popleft()
                if stamp <= self.last_imu_processed_stamp:
                    self.event_order_violations += 1
                    continue
                self.vio.process_imu(stamp, gyro, accel)
                self.last_imu_processed_stamp = stamp
            if self.args.stereo and right is not None:
                priors = dict(self.stereo.range_priors(right, self.frontend))
                self.stereo_prior_candidates += len(priors)
                landmarks_before = set(self.vio.get_landmarks())
                self.vio.process_vision_with_depth_priors(
                    image_stamp, observations, priors)
                landmarks_after = set(self.vio.get_landmarks())
                self.stereo_prior_births += len(
                    (landmarks_after - landmarks_before) & set(priors))
            elif self.args.gt_depth_priors and depth is not None:
                depth_priors = {}
                for feature in features:
                    gt_range = self.ground_truth_range(depth, feature)
                    if gt_range is not None:
                        depth_priors[int(feature['id'])] = (gt_range, 0.0004)
                self.gt_depth_prior_candidates += len(depth_priors)
                landmarks_before = set(self.vio.get_landmarks())
                self.vio.process_vision_with_depth_priors(
                    image_stamp, observations, depth_priors)
                landmarks_after = set(self.vio.get_landmarks())
                self.gt_depth_prior_births += len(
                    (landmarks_after - landmarks_before) & set(depth_priors))
            else:
                self.vio.process_vision(image_stamp, observations)
            position, quaternion = self.vio.get_pose()
            velocity = self.vio.get_velocity()
            gyro_bias, accel_bias = self.vio.get_biases()
            if not np.all(np.isfinite(position)):
                continue
            self.estimated_stamps.append(image_stamp)
            self.estimated_positions.append(np.asarray(position).copy())
            self.estimated_quaternions.append(np.asarray(quaternion).copy())
            self.estimated_velocities.append(np.asarray(velocity).copy())
            self.estimated_gyro_biases.append(np.asarray(gyro_bias).copy())
            self.estimated_accel_biases.append(np.asarray(accel_bias).copy())
            if self.rr is not None:
                if right is not None:
                    self.rr.log('stereo/image', self.rr.Image(right))
                self.log_rerun(
                    image_stamp, gray, depth, features, track_meta,
                    position, quaternion,
                    velocity, gyro_bias, accel_bias)

    def log_rerun(self, stamp, gray, depth, features, track_meta,
                  position, quaternion,
                  velocity, gyro_bias, accel_bias):
        rr = self.rr
        rr.set_time('sim_time', timestamp=stamp)
        rr.log('camera/image', rr.Image(gray))
        if depth is not None:
            rr.log('depth/image', rr.DepthImage(
                depth, meter=1.0, depth_range=[0.05, 8.0]))

        landmarks = {int(k): np.asarray(v) for k, v in self.vio.get_landmarks().items()}
        feature_by_id = {int(f['id']): f for f in features}
        r_wb = quaternion_matrix_xyzw(quaternion)
        r_bc = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
        p_wc = np.asarray(position) + r_wb @ np.array([0.18, 0.0, 0.0])
        r_wc = r_wb @ r_bc
        ids, points, pixels, ranges, gt_ranges, ages = [], [], [], [], [], []
        for landmark_id, point in landmarks.items():
            feature = feature_by_id.get(landmark_id)
            if feature is None:
                continue
            self.depth_landmark_candidates += 1
            camera_point = r_wc.T @ (point - p_wc)
            estimated_range = float(np.linalg.norm(camera_point))
            if depth is None:
                ids.append(landmark_id)
                points.append(point)
                pixels.append([feature['x'], feature['y']])
                ranges.append(estimated_range)
                gt_ranges.append(float('nan'))
                ages.append(int(track_meta.get(landmark_id, {}).get('age', 0)))
                continue
            gt_range = self.ground_truth_range(depth, feature)
            if gt_range is None:
                continue
            self.depth_landmark_valid += 1
            ids.append(landmark_id)
            points.append(point)
            pixels.append([feature['x'], feature['y']])
            ranges.append(estimated_range)
            gt_ranges.append(gt_range)
            ages.append(int(track_meta.get(landmark_id, {}).get('age', 0)))
            self.depth_estimated_ranges.append(estimated_range)
            self.depth_ground_truth_ranges.append(gt_range)
            self.depth_track_ages.append(ages[-1])
        if points:
            ranges_array = np.asarray(ranges)
            normalized = np.clip(ranges_array / 12.0, 0.0, 1.0)
            colors = np.column_stack([
                (255 * normalized).astype(np.uint8),
                (255 * (1.0 - normalized)).astype(np.uint8),
                np.full(len(points), 64, dtype=np.uint8),
            ])
            labels = [
                (f'{i} e={est:.2f} gt={gt:.2f} err={est-gt:+.2f}m'
                 if np.isfinite(gt) else f'{i} e={est:.2f} gt=missing')
                for i, est, gt in zip(ids, ranges, gt_ranges)]
            rr.log('camera/image/landmark_depths', rr.Points2D(
                pixels, colors=colors, radii=4.0))
            rr.log('world/landmarks', rr.Points3D(
                points, colors=colors, radii=0.04, labels=labels))

        rr.log('world/estimated_body',
               rr.Transform3D(translation=position,
                              quaternion=rr.Quaternion(xyzw=quaternion)),
               rr.TransformAxes3D(0.4))
        if len(self.estimated_positions) > 1:
            rr.log('world/estimated_trajectory', rr.LineStrips3D(
                [self.estimated_positions], colors=[255, 80, 40], radii=0.02))
        if self.frame_count % 5 == 0:
            estimated, truth, _ = self.matched_trajectories()
            if len(estimated) >= 3:
                estimated_center = estimated.mean(axis=0)
                truth_center = truth.mean(axis=0)
                estimated_zero = estimated - estimated_center
                truth_zero = truth - truth_center
                u, _, vt = np.linalg.svd(estimated_zero.T @ truth_zero)
                rotation_est_to_truth = vt.T @ u.T
                if np.linalg.det(rotation_est_to_truth) < 0.0:
                    vt[-1, :] *= -1.0
                    rotation_est_to_truth = vt.T @ u.T
                self.rerun_aligned_truth = (
                    (truth - truth_center) @ rotation_est_to_truth +
                    estimated_center)
        if len(self.rerun_aligned_truth) > 1:
            rr.log('world/ground_truth_trajectory', rr.LineStrips3D(
                [self.rerun_aligned_truth],
                colors=[40, 220, 80], radii=0.015))
        rr.log('plots/speed', rr.Scalars([float(np.linalg.norm(velocity))]))
        rr.log('plots/gyro_bias_norm',
               rr.Scalars([float(np.linalg.norm(gyro_bias))]))
        rr.log('plots/accel_bias_norm',
               rr.Scalars([float(np.linalg.norm(accel_bias))]))
        rr.log('plots/landmark_count', rr.Scalars([len(landmarks)]))
        rr.log('plots/imu_queue', rr.Scalars([len(self.imu_queue)]))
        rr.log('plots/vision_queue', rr.Scalars([len(self.vision_queue)]))

    def matched_trajectories(self):
        if not self.estimated_stamps or not self.truth_stamps:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
        est_stamps = np.asarray(self.estimated_stamps)
        truth_stamps = np.asarray(self.truth_stamps)
        truth_positions = np.asarray(self.truth_positions_all)
        indices = np.searchsorted(truth_stamps, est_stamps)
        indices = np.clip(indices, 1, len(truth_stamps) - 1)
        left = indices - 1
        choose_left = np.abs(truth_stamps[left] - est_stamps) <= \
            np.abs(truth_stamps[indices] - est_stamps)
        nearest = np.where(choose_left, left, indices)
        errors = np.abs(truth_stamps[nearest] - est_stamps)
        valid = errors <= 0.006
        return (np.asarray(self.estimated_positions)[valid],
                truth_positions[nearest[valid]], errors[valid])

    def imu_consistency(self):
        if len(self.truth_stamps) < 3 or not self.imu_stamps:
            return {}
        ts = np.asarray(self.truth_stamps)
        velocity = np.asarray(self.truth_linear_velocity)
        quaternion = np.asarray(self.truth_quaternions)
        truth_gyro = np.asarray(self.truth_angular_velocity)
        imu_stamps = np.asarray(self.imu_stamps)
        imu_gyro = np.asarray(self.imu_gyro)
        imu_accel = np.asarray(self.imu_accel)
        imu_by_ns = {int(round(t * 1.0e9)): i for i, t in enumerate(imu_stamps)}
        gyro_errors = []
        accel_errors = []
        gravity = np.array([0.0, 0.0, -9.80665])
        for i in range(1, len(ts) - 1):
            j = imu_by_ns.get(int(round(ts[i] * 1.0e9)))
            dt = ts[i + 1] - ts[i - 1]
            if j is None or dt <= 0.0:
                continue
            world_accel = (velocity[i + 1] - velocity[i - 1]) / dt
            r_wb = quaternion_matrix_xyzw(quaternion[i])
            predicted_force = r_wb.T @ (world_accel - gravity)
            gyro_errors.append(imu_gyro[j] - truth_gyro[i])
            accel_errors.append(imu_accel[j] - predicted_force)
        if not gyro_errors:
            return {}
        gyro_errors = np.asarray(gyro_errors)
        accel_errors = np.asarray(accel_errors)
        return {
            'matched_samples': len(gyro_errors),
            'gyro_rmse_rad_s': float(np.sqrt(np.mean(gyro_errors ** 2))),
            'specific_force_rmse_m_s2':
                float(np.sqrt(np.mean(accel_errors ** 2))),
            'specific_force_max_error_m_s2':
                float(np.max(np.linalg.norm(accel_errors, axis=1))),
        }

    def summary(self):
        elapsed = time.monotonic() - self.start_wall
        active_elapsed = 0.0
        if self.first_image_wall is not None and self.last_image_wall is not None:
            active_elapsed = self.last_image_wall - self.first_image_wall
        result = {
            'wall_s': elapsed,
            'active_image_wall_s': active_elapsed,
            'frames': self.frame_count,
            'active_camera_hz': ((self.frame_count - 1) / active_elapsed
                                 if active_elapsed > 0.0 else 0.0),
            'imu_messages': self.imu_count,
            'bad_images': self.bad_images,
            'event_order_violations': self.event_order_violations,
            'max_imu_queue': self.max_imu_queue,
            'max_vision_queue': self.max_vision_queue,
            'unprocessed_imu': len(self.imu_queue),
            'unprocessed_images': len(self.vision_queue),
            'gt_depth_priors_enabled': self.args.gt_depth_priors,
            'gt_depth_prior_candidates': self.gt_depth_prior_candidates,
            'gt_depth_prior_births': self.gt_depth_prior_births,
            'stereo_enabled': self.args.stereo,
            'stereo_messages': self.stereo_messages,
            'matched_stereo_frames': self.stereo_matches,
            'stereo_prior_candidates': self.stereo_prior_candidates,
            'stereo_prior_births': self.stereo_prior_births,
            'imu_consistency': self.imu_consistency(),
        }
        if self.totals:
            result.update({
                'total_median': statistics.median(self.totals),
                'tracked_median': statistics.median(self.tracked),
                'frontend_ms_median': statistics.median(self.timing_ms),
                'frontend_ms_p95': float(np.percentile(self.timing_ms, 95)),
            })
        if self.args.vio:
            estimated, truth, match_errors = self.matched_trajectories()
            se3_rmse, _, _ = align_positions(estimated, truth, False)
            sim3_rmse, scale, aligned = align_positions(estimated, truth, True)
            result.update({
                'matched_pose_count': len(estimated),
                'max_pose_match_error_s': (float(np.max(match_errors))
                                           if len(match_errors) else None),
                'se3_ate_rmse_m': se3_rmse,
                'sim3_ate_rmse_m': sim3_rmse,
                'estimated_to_true_scale': scale,
                'estimated_axis_span_m': (np.ptp(estimated, axis=0).tolist()
                                          if len(estimated) else None),
                'truth_axis_span_m': (np.ptp(truth, axis=0).tolist()
                                      if len(truth) else None),
            })
            self.aligned_positions = aligned
            if self.depth_ground_truth_ranges:
                estimated_ranges = np.asarray(self.depth_estimated_ranges)
                gt_ranges = np.asarray(self.depth_ground_truth_ranges)
                ages = np.asarray(self.depth_track_ages)
                absolute_errors = np.abs(estimated_ranges - gt_ranges)
                relative_errors = absolute_errors / gt_ranges
                depth_audit = {
                    'depth_messages': self.depth_messages,
                    'matched_depth_frames': self.depth_matches,
                    'landmark_depth_coverage': (
                        self.depth_landmark_valid /
                        max(1, self.depth_landmark_candidates)),
                    'samples': len(gt_ranges),
                    'absolute_error_median_m': float(np.median(absolute_errors)),
                    'absolute_error_p95_m': float(np.percentile(
                        absolute_errors, 95)),
                    'relative_error_median': float(np.median(relative_errors)),
                }
                for name, mask in (
                        ('birth_age_le_2', ages <= 2),
                        ('mature_age_ge_10', ages >= 10)):
                    if np.any(mask):
                        depth_audit[name] = {
                            'samples': int(np.sum(mask)),
                            'estimated_range_median_m': float(np.median(
                                estimated_ranges[mask])),
                            'ground_truth_range_median_m': float(np.median(
                                gt_ranges[mask])),
                            'absolute_error_median_m': float(np.median(
                                absolute_errors[mask])),
                        }
                result['landmark_depth_audit'] = depth_audit
        return result

    def write_results(self, summary):
        if self.args.output is None:
            return
        output = self.args.output
        output.mkdir(parents=True, exist_ok=True)
        arrays = {
            'imu_stamps': np.asarray(self.imu_stamps),
            'imu_gyro': np.asarray(self.imu_gyro),
            'imu_accel': np.asarray(self.imu_accel),
            'truth_stamps': np.asarray(self.truth_stamps),
            'truth_positions': np.asarray(self.truth_positions_all),
            'truth_quaternions_xyzw': np.asarray(self.truth_quaternions),
            'truth_linear_velocity': np.asarray(self.truth_linear_velocity),
            'truth_angular_velocity': np.asarray(self.truth_angular_velocity),
            'estimated_stamps': np.asarray(self.estimated_stamps),
            'estimated_positions': np.asarray(self.estimated_positions),
            'estimated_quaternions_xyzw': np.asarray(self.estimated_quaternions),
            'estimated_velocities': np.asarray(self.estimated_velocities),
            'estimated_gyro_biases': np.asarray(self.estimated_gyro_biases),
            'estimated_accel_biases': np.asarray(self.estimated_accel_biases),
            'frontend_total': np.asarray(self.totals),
            'frontend_tracked': np.asarray(self.tracked),
            'frontend_timing_ms': np.asarray(self.timing_ms),
        }
        np.savez_compressed(output / 'diagnostic.npz', **arrays)
        (output / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().parents[1] /
        'configs/eqvio_gazebo.yaml')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--vio', action='store_true',
                        help='Run EqVIO after the frontend gate.')
    parser.add_argument('--output', type=Path,
                        help='Directory for diagnostic.npz and summary.json.')
    parser.add_argument('--visualize', action='store_true',
                        help='Spawn a live Rerun landmark/trajectory dashboard.')
    parser.add_argument(
        '--gt-depth-priors', action='store_true',
        help='Oracle diagnostic: initialize new EqF landmarks from GT depth.')
    parser.add_argument(
        '--stereo', action='store_true',
        help='Use synchronized right images for stereo depth priors.')
    parser.add_argument('--stereo-baseline', type=float, default=0.12)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GazeboEchoLi(args)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    summary = node.summary()
    node.write_results(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
