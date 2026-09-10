#!/usr/bin/env python3
"""Real-time ROS 2 adapter for VOXL2 sensors and ECHO-LI."""

from collections import deque
import queue
import statistics
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image, Imu
from tf2_ros import TransformBroadcaster

import echo_li

from echo_li_ros2.dis_depth import from_config as dis_from_config


NSEC_PER_SEC = 1_000_000_000


def _stamp_ns(stamp):
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def _time_msg(stamp_ns):
    return rclpy.time.Time(nanoseconds=int(stamp_ns)).to_msg()


# OpenCV COLORMAP_JET LUT, matching echo-li-cli's color_for_scalar.
_JET_KNOTS = np.array([
    [0.000, 0, 0, 128],
    [0.125, 0, 0, 255],
    [0.375, 0, 255, 255],
    [0.625, 255, 255, 0],
    [0.875, 255, 0, 0],
    [1.000, 128, 0, 0],
], dtype=np.float64)


def _jet_depth_image(depth, valid, vis_min, vis_max):
    """Render a depth array as an RGB image with a flipped JET colormap.

    Near → red, far → blue (same flip as echo-li-cli's ``color_for_depth``).
    Invalid pixels are black.
    """
    flipped = np.where(valid, vis_max + vis_min - depth, 0.0)
    t = np.clip((flipped - vis_min) / max(vis_max - vis_min, 1e-9), 0.0, 1.0)
    ts = _JET_KNOTS[:, 0]
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    for i in range(len(ts) - 1):
        mask = valid & (t >= ts[i]) & ((t <= ts[i + 1]) if i < len(ts) - 2
                                       else (t <= ts[i + 1] + 1e-9))
        if not np.any(mask):
            continue
        local_t = np.where(
            mask,
            (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9),
            0.0)
        for ch in range(3):
            a = _JET_KNOTS[i, 1 + ch]
            b = _JET_KNOTS[i + 1, 1 + ch]
            rgb[:, :, ch] = np.where(
                mask,
                np.round(a + (b - a) * local_t).astype(np.uint8),
                rgb[:, :, ch])
    return rgb


def _quat_to_se3(position, quaternion):
    """Build a 4×4 SE(3) matrix from position + xyzw quaternion."""
    x, y, z, w = quaternion
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[0, 1] = 2.0 * (x * y - z * w)
    m[0, 2] = 2.0 * (x * z + y * w)
    m[1, 0] = 2.0 * (x * y + z * w)
    m[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[1, 2] = 2.0 * (y * z - x * w)
    m[2, 0] = 2.0 * (x * z - y * w)
    m[2, 1] = 2.0 * (y * z + x * w)
    m[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    m[0, 3] = position[0]
    m[1, 3] = position[1]
    m[2, 3] = position[2]
    return m


def _umeyama_se3(src, dst):
    """Compute the SE(3) transform T that minimises ‖T·src − dst‖².

    Parameters
    ----------
    src, dst : (N, 3) arrays of matched 3D positions (N ≥ 3).

    Returns
    -------
    T : (4, 4) SE(3) matrix  (dst ≈ T @ src).
    """
    assert src.shape == dst.shape and src.shape[0] >= 3
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s_centered = src - mu_s
    d_centered = dst - mu_d
    H = s_centered.T @ d_centered  # 3×3 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])  # correct reflection
    R = Vt.T @ S @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _align_to_estimate(ref_stamps, ref_positions, est_stamps, est_positions):
    """Compute the SE(3) transform that maps reference positions into the
    estimate frame:  T @ ref_pos ≈ est_pos.

    Handles different clock epochs (e.g. VOXL monotonic vs wall-clock mocap)
    by matching on elapsed time from each trajectory's start.

    Returns T (4×4) or None if insufficient overlap.
    """
    if len(ref_positions) < 10 or len(est_positions) < 10:
        return None

    ref_t = np.asarray(ref_stamps, dtype=np.float64)
    est_t = np.asarray(est_stamps, dtype=np.float64)
    est_p = np.asarray(est_positions, dtype=np.float64)  # (M, 3)
    ref_p = np.asarray(ref_positions, dtype=np.float64)  # (N, 3)

    # Convert to elapsed time from each trajectory's own start so that
    # trajectories on different clock epochs can still be matched.
    ref_elapsed = ref_t - ref_t[0]
    est_elapsed = est_t - est_t[0]

    # Keep reference samples that fall within the estimate's elapsed range.
    overlap = (ref_elapsed >= est_elapsed[0]) & (ref_elapsed <= est_elapsed[-1])
    if overlap.sum() < 10:
        return None

    ref_e_in = ref_elapsed[overlap]
    ref_p_in = ref_p[overlap]

    # Interpolate estimated positions at reference elapsed times.
    est_interp = np.column_stack([
        np.interp(ref_e_in, est_elapsed, est_p[:, i]) for i in range(3)])

    return _umeyama_se3(ref_p_in, est_interp)


class Voxl2EchoLi(Node):
    def __init__(self):
        super().__init__('echo_li_voxl2')
        share = get_package_share_directory('echo_li_ros2')

        self.declare_parameter(
            'echo_config_path', share + '/config/eqvio_voxl2.yaml')
        self.declare_parameter('imu_topic', '/voxl/raw_imu')
        self.declare_parameter('image_topic', '/tracking_front/decoded')
        self.declare_parameter('odometry_topic', '/echo_li/odometry')
        self.declare_parameter('odom_frame_id', 'echo_li_odom')
        self.declare_parameter('body_frame_id', 'imu_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('camera_time_offset_sec', -0.0264)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 800)
        self.declare_parameter(
            'intrinsics', [462.45900845409261, 462.54026967058945,
                           670.414298091828, 398.44551581087347])
        self.declare_parameter(
            'distortion_coefficients',
            [0.06799263373296542, 0.0022315464821754502,
             0.004627575172719192, -0.0033193411009612033])
        self.declare_parameter(
            't_bs',
            [0.0, 0.0, 1.0, 0.037,
             1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0006,
             0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('initialization_imu_samples', 100)
        self.declare_parameter('imu_qos_depth', 2000)
        self.declare_parameter('image_qos_depth', 5)
        self.declare_parameter('image_qos_reliability', 'best_effort')
        self.declare_parameter('max_imu_queue', 5000)
        self.declare_parameter('max_image_queue', 4)
        self.declare_parameter('statistics_period_sec', 5.0)
        self.declare_parameter('rerun_enabled', False)
        self.declare_parameter('rerun_url', '')
        self.declare_parameter('rerun_world_stride', 3)
        self.declare_parameter('patch_depth_enabled', True)
        self.declare_parameter('occupancy_enabled', True)
        self.declare_parameter('mapping_stride', 1)
        self.declare_parameter('depth_backend', 'dis')  # 'dis' or 'patch'
        self.declare_parameter('mocap_topic', '')  # e.g. /vrpn_mocap/drone_01/pose
        self.declare_parameter('vio_topic', '')    # e.g. /qvio/odom

        self.config_path = self.get_parameter(
            'echo_config_path').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        odometry_topic = self.get_parameter('odometry_topic').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.body_frame = self.get_parameter('body_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        offset_sec = self.get_parameter('camera_time_offset_sec').value
        self.camera_offset_ns = int(round(offset_sec * NSEC_PER_SEC))
        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value
        intrinsics = list(self.get_parameter('intrinsics').value)
        distortion = list(
            self.get_parameter('distortion_coefficients').value)
        t_bs_values = list(self.get_parameter('t_bs').value)
        self.n_init = self.get_parameter('initialization_imu_samples').value
        self.image_qos_reliability = self.get_parameter(
            'image_qos_reliability').value
        self.max_imu_queue = self.get_parameter('max_imu_queue').value
        self.max_image_queue = self.get_parameter('max_image_queue').value
        self.rerun_enabled = self.get_parameter('rerun_enabled').value
        self.rerun_url = self.get_parameter('rerun_url').value
        self.rerun_world_stride = self.get_parameter(
            'rerun_world_stride').value
        self.patch_depth_enabled = self.get_parameter(
            'patch_depth_enabled').value
        self.occupancy_enabled = self.get_parameter(
            'occupancy_enabled').value
        self.mapping_stride = max(
            1, self.get_parameter('mapping_stride').value)
        self.depth_backend = self.get_parameter('depth_backend').value
        self.mocap_topic = self.get_parameter('mocap_topic').value or ''
        self.vio_topic = self.get_parameter('vio_topic').value or ''
        if self.depth_backend not in ('dis', 'patch'):
            raise ValueError(
                f"depth_backend must be 'dis' or 'patch', "
                f"got '{self.depth_backend}'")

        if len(intrinsics) != 4 or len(distortion) != 4:
            raise ValueError(
                'intrinsics and distortion_coefficients need 4 values')
        if len(t_bs_values) != 16:
            raise ValueError('t_bs needs 16 row-major values')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('image dimensions must be positive')
        if self.rerun_world_stride <= 0:
            raise ValueError('rerun_world_stride must be positive')
        if self.image_qos_reliability not in ('best_effort', 'reliable'):
            raise ValueError(
                'image_qos_reliability must be best_effort or reliable')

        fx, fy, cx, cy = intrinsics
        self.camera = echo_li.EquidistantCamera(
            fx, fy, cx, cy, *distortion)
        frontend_config = echo_li.FrontendConfig.from_yaml(self.config_path)
        frontend_config.set_camera(
            fx, fy, cx, cy, self.width, self.height, distortion,
            distortion_model='equidistant')
        self.frontend = echo_li.Frontend(
            frontend_config, self.width, self.height)
        self.vio = echo_li.VIOFilter(
            self.config_path, self.camera, n_init_samples=self.n_init)
        t_bs = np.asarray(t_bs_values, dtype=np.float64).reshape(4, 4)
        self.vio.set_camera_extrinsics(t_bs)
        # get_pose() returns T_wb (body/IMU frame). Camera pose is
        # T_wc = T_wb @ T_bc where T_bc = T_bs (body-to-sensor).
        self.t_bc = t_bs.copy()

        # ── Sparse 3D filter (out-of-state depth pool for patch mapper seeds) ──
        self.sparse_3d = None
        self._sparse_vog_kwargs = {}
        try:
            import yaml as _yaml
            with open(self.config_path) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            sv = _cfg.get('SparseVog')
            if sv and sv.get('parametrization', '').startswith('bearing'):
                kw = {k: v for k, v in sv.items()
                      if k not in ('parametrization', 'enabled')}
                self.sparse_3d = (
                    echo_li.Sparse3DFilter
                    .bearing_invdepth_additive3d(self.camera, **kw))
                self._sparse_vog_kwargs = kw
                self.get_logger().info(
                    f'Sparse3DFilter: bearing_invdepth_additive3d, '
                    f'pool={kw.get("max_pool_size", "default")}')
        except Exception as exc:
            self.get_logger().warn(
                f'Sparse3DFilter not available: {exc}')

        # ── Dense depth mapper ──
        # depth_backend='dis' → DIS optical flow + two-ray triangulation (Python)
        # depth_backend='patch' → photometric 1D GN along epipolar lines (Rust)
        self.depth_mapper = None
        self._depth_mapper_is_rust = False
        self.occupancy_map = None
        if self.patch_depth_enabled:
            try:
                if self.depth_backend == 'dis':
                    self.depth_mapper = dis_from_config(
                        fx, fy, cx, cy, self.width, self.height,
                        distortion, config_path=self.config_path)
                    self.get_logger().info(
                        f'DISDepthMapper: enabled '
                        f'(scale={self.depth_mapper.scale}, '
                        f'{self.depth_mapper.work_w}x'
                        f'{self.depth_mapper.work_h})')
                else:  # 'patch'
                    self.depth_mapper = echo_li.PatchDepthMapper(
                        self.camera, fx, fy, cx, cy,
                        self.width, self.height,
                        config=self.config_path)
                    self._depth_mapper_is_rust = True
                    self.get_logger().info(
                        f'PatchDepthMapper: enabled '
                        f'(seeds={self.depth_mapper.seed_coordinates})')
                if self.occupancy_enabled:
                    self.occupancy_map = echo_li.LocalOccupancyMap(
                        self.camera, fx, fy, cx, cy, self.width, self.height,
                        config=self.config_path,
                        seed_coordinates=(
                            self.depth_mapper.seed_coordinates))
                    self.get_logger().info('LocalOccupancyMap: enabled')
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to create mapping pipeline: {exc}')
                self.depth_mapper = None
                self.occupancy_map = None
        self.patch_depth_times_ms = deque(maxlen=300)
        self.occupancy_times_ms = deque(maxlen=300)
        self.seed_counts = deque(maxlen=300)
        self.patch_depth_frame_count = 0

        self.imu_queue = deque()
        self.image_queue = deque()
        self.latest_imu_ns = None
        self.last_imu_received_ns = None
        self.last_imu_processed_ns = None
        self.last_image_received_ns = None
        self.imu_processed = 0
        self.imu_received = 0
        self.images_received = 0
        self.images_processed = 0
        self.dropped_imu = 0
        self.dropped_images = 0
        self.bad_images = 0
        self.gray_times_ms = deque(maxlen=300)
        self.frontend_times_ms = deque(maxlen=300)
        self.vision_times_ms = deque(maxlen=300)
        self.total_times_ms = deque(maxlen=300)
        self.track_counts = deque(maxlen=300)
        self.estimated_positions = deque(maxlen=20000)
        self.started_at = time.monotonic()
        self._draining = False
        self.rr = None
        self.rerun_queue = None
        self.rerun_stop = None
        self.rerun_thread = None
        self.rerun_frames_submitted = 0
        self.rerun_frames_logged = 0
        self.rerun_frames_replaced = 0
        self.rerun_errors = 0
        self.rerun_log_times_ms = deque(maxlen=300)

        if self.rerun_enabled:
            self.setup_rerun()

        imu_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.get_parameter('imu_qos_depth').value,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.get_parameter('image_qos_depth').value,
            reliability=(
                QoSReliabilityPolicy.RELIABLE
                if self.image_qos_reliability == 'reliable'
                else QoSReliabilityPolicy.BEST_EFFORT),
            durability=QoSDurabilityPolicy.VOLATILE)

        self.odom_pub = self.create_publisher(Odometry, odometry_topic, 10)
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.publish_tf else None)
        self.create_subscription(Imu, self.imu_topic, self.on_imu, imu_qos)
        self.create_subscription(
            Image, self.image_topic, self.on_image, image_qos)

        # ── Reference trajectory subscriptions (mocap / external VIO) ──
        # Accumulated as (stamp_sec, [x, y, z]) for SE(3) alignment + Rerun.
        self.mocap_stamps = []
        self.mocap_positions = []
        self.vio_stamps = []
        self.vio_positions = []
        self.estimated_stamps = []   # parallel to estimated_positions
        self.mocap_align = None      # 4×4 SE(3): mocap → estimate frame
        self.vio_align = None        # 4×4 SE(3): vio → estimate frame
        self._align_counter = 0
        ref_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=200,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        if self.mocap_topic:
            self.create_subscription(
                PoseStamped, self.mocap_topic, self.on_mocap, ref_qos)
            self.get_logger().info(f'Mocap: subscribing to {self.mocap_topic}')
        if self.vio_topic:
            self.create_subscription(
                Odometry, self.vio_topic, self.on_vio, ref_qos)
            self.get_logger().info(f'VIO ref: subscribing to {self.vio_topic}')

        self.create_timer(
            self.get_parameter('statistics_period_sec').value,
            self.report_statistics)

        self.get_logger().info(
            f'ECHO-LI ready: imu={self.imu_topic}, image={self.image_topic}, '
            f'camera={self.width}x{self.height} equidistant, '
            f'camera_time_offset={offset_sec:+.6f}s, '
            f'image_qos={self.image_qos_reliability}, '
            f'rerun={self.rerun_enabled}')

    def setup_rerun(self):
        import rerun as rr
        import rerun.blueprint as rrb

        rr.init('echo_li_voxl2')
        # Image is logged at half resolution for performance.
        half_w = self.width // 2
        half_h = self.height // 2
        camera_view = rrb.Spatial2DView(
            name='Tracking front', origin='camera',
            contents=['camera/**'],
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0.0, float(half_w)],
                y_range=[0.0, float(half_h)]))
        world_view = rrb.Spatial3DView(
            name='Estimated world', origin='world',
            contents=['world/**'])
        if self.depth_mapper is not None:
            depth_view = rrb.Spatial2DView(
                name='Depth', origin='depth',
                contents=['depth/**'])
            left_column = rrb.Vertical(
                camera_view, depth_view,
                row_shares=[1.0, 0.6])
        else:
            left_column = camera_view
        blueprint = rrb.Blueprint(
            rrb.Horizontal(left_column, world_view,
                           column_shares=[1.0, 1.0]),
            auto_layout=False, auto_views=False, collapse_panels=True)
        if self.rerun_url:
            rr.connect_grpc(self.rerun_url)
            self.get_logger().info(
                f'Rerun: streaming to {self.rerun_url}')
        else:
            rr.spawn()
            self.get_logger().info('Rerun: spawned a local viewer')
        rr.send_blueprint(blueprint)
        self.rr = rr
        self.rerun_queue = queue.Queue(maxsize=30)
        self.rerun_stop = threading.Event()
        self.rerun_thread = threading.Thread(
            target=self.rerun_worker,
            name='echo-li-rerun',
            daemon=True)
        self.rerun_thread.start()

    def on_imu(self, msg):
        stamp_ns = _stamp_ns(msg.header.stamp)
        self.imu_received += 1
        if (self.last_imu_received_ns is not None and
                stamp_ns <= self.last_imu_received_ns):
            self.dropped_imu += 1
            return
        self.last_imu_received_ns = stamp_ns
        self.latest_imu_ns = stamp_ns
        gyro = (msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z)
        accel = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                 msg.linear_acceleration.z)
        self.imu_queue.append((stamp_ns, gyro, accel))
        if len(self.imu_queue) > self.max_imu_queue:
            self.imu_queue.popleft()
            self.dropped_imu += 1
        self.drain_queues()

    def on_image(self, msg):
        self.images_received += 1
        stamp_ns = _stamp_ns(msg.header.stamp) + self.camera_offset_ns
        if (self.last_image_received_ns is not None and
                stamp_ns <= self.last_image_received_ns):
            self.dropped_images += 1
            return
        self.last_image_received_ns = stamp_ns
        self.image_queue.append((stamp_ns, msg))
        if len(self.image_queue) > self.max_image_queue:
            self.image_queue.popleft()
            self.dropped_images += 1
        self.drain_queues()

    def on_mocap(self, msg):
        t = _stamp_ns(msg.header.stamp) / NSEC_PER_SEC
        p = msg.pose
        self.mocap_stamps.append(t)
        self.mocap_positions.append([
            p.position.x, p.position.y, p.position.z])

    def on_vio(self, msg):
        t = _stamp_ns(msg.header.stamp) / NSEC_PER_SEC
        p = msg.pose.pose
        self.vio_stamps.append(t)
        self.vio_positions.append([
            p.position.x, p.position.y, p.position.z])

    def drain_queues(self):
        if self._draining:
            return
        self._draining = True
        try:
            while self.image_queue and self.imu_queue:
                image_ns, image_msg = self.image_queue[0]
                if self.latest_imu_ns < image_ns:
                    break
                if (self.last_imu_processed_ns is not None and
                        image_ns <= self.last_imu_processed_ns):
                    self.image_queue.popleft()
                    self.dropped_images += 1
                    continue

                processed_for_frame = 0
                while self.imu_queue and self.imu_queue[0][0] <= image_ns:
                    stamp_ns, gyro, accel = self.imu_queue.popleft()
                    if (self.last_imu_processed_ns is not None and
                            stamp_ns <= self.last_imu_processed_ns):
                        self.dropped_imu += 1
                        continue
                    self.vio.process_imu(
                        stamp_ns / NSEC_PER_SEC, gyro, accel)
                    self.last_imu_processed_ns = stamp_ns
                    self.imu_processed += 1
                    processed_for_frame += 1

                if processed_for_frame == 0:
                    self.image_queue.popleft()
                    self.dropped_images += 1
                    continue
                self.image_queue.popleft()
                self.process_image(image_ns, image_msg)
        finally:
            self._draining = False

    def process_image(self, stamp_ns, msg):
        started = time.monotonic()
        try:
            gray = self.to_gray(msg)
        except ValueError as exc:
            self.bad_images += 1
            self.get_logger().error(str(exc))
            return
        gray_ms = (time.monotonic() - started) * 1000.0

        frontend_started = time.monotonic()
        features, stats = self.frontend.process(gray)
        frontend_ms = (time.monotonic() - frontend_started) * 1000.0
        observations = {
            int(feature['id']): (float(feature['x']), float(feature['y']))
            for feature in features
        }
        vision_started = time.monotonic()
        self.vio.process_vision(stamp_ns / NSEC_PER_SEC, observations)
        vision_ms = (time.monotonic() - vision_started) * 1000.0
        self.images_processed += 1
        self.gray_times_ms.append(gray_ms)
        self.frontend_times_ms.append(frontend_ms)
        self.vision_times_ms.append(vision_ms)
        self.total_times_ms.append((time.monotonic() - started) * 1000.0)
        self.track_counts.append(int(stats['total']))

        if self.imu_processed < self.n_init:
            self.submit_rerun(stamp_ns, gray, features, None)
            return
        position, quaternion = self.vio.get_pose()
        velocity = self.vio.get_velocity()
        position = np.asarray(position)
        quaternion = np.asarray(quaternion)
        velocity = np.asarray(velocity)
        if not (np.all(np.isfinite(position)) and
                np.all(np.isfinite(quaternion)) and
                np.all(np.isfinite(velocity))):
            self.get_logger().error('ECHO-LI produced a non-finite state')
            return
        self.publish_odometry(stamp_ns, position, quaternion, velocity)

        # ── Update sparse 3D filter (out-of-state depth pool) every frame ──
        # get_pose() returns T_wb (body/IMU); camera pose = T_wb @ T_bc.
        t_wc = _quat_to_se3(position, quaternion) @ self.t_bc
        if self.sparse_3d is not None:
            feature_uvs = {
                int(f['id']): [float(f['x']), float(f['y'])]
                for f in features}
            p_vv, p_ww = None, None
            cov = self.vio.get_camera_pose_covariance()
            if cov is not None:
                p_vv = np.asarray(cov[0]).tolist()
                p_ww = np.asarray(cov[1]).tolist()
            self.sparse_3d.update(
                stamp_ns / NSEC_PER_SEC,
                feature_uvs, t_wc.tolist(), p_vv, p_ww)

        # ── Dense mapping: DIS flow depth → occupancy ──
        depth_result = None
        occupancy_cells = None
        if (self.depth_mapper is not None and
                self.images_processed % self.mapping_stride == 0):
            depth_result, occupancy_cells = self.run_mapping(
                stamp_ns, gray, t_wc, features)

        world = None
        if self.rr is not None:
            world = self.make_rerun_world_payload(
                position, quaternion, features)
        self.submit_rerun(
            stamp_ns, gray, features, world,
            depth_result=depth_result,
            occupancy_cells=occupancy_cells)

    def run_mapping(self, stamp_ns, gray, t_wc, features):
        """Run the dense depth mapper and optionally the occupancy map.

        `t_wc` is the 4×4 camera pose (world ← camera), already composing
        T_wb @ T_bc from the EqF body pose and the camera extrinsic.

        Returns (depth_result, occupancy_cells) where depth_result is the dict
        from the active depth backend (or None), and occupancy_cells is an
        (N,3) float32 array of occupied voxel centres (or None).
        """
        t_wc_list = t_wc.tolist()
        cam_pos = t_wc[:3, 3]

        # Build sparse depth priors.  Prefer Sparse3DFilter (tighter
        # variance when converged); fill remaining features from EqF
        # landmarks so the mapper always has seeds even before sparse3d
        # tracks converge.
        priors = []
        sparse3d_fids = set()
        if self.sparse_3d is not None:
            for feat in features:
                fid = int(feat['id'])
                rng, rng_var = self.sparse_3d.query_range(fid)
                if rng < 0.0:
                    continue  # not converged
                sparse3d_fids.add(fid)
                eta = float(np.log(rng))
                # var(η) ≈ var(range)/range² = rng_var/range²
                eta_var = float(rng_var / (rng * rng)) if rng > 0.01 else 1.0
                priors.append((
                    float(feat['x']), float(feat['y']), eta, eta_var))

        # Fill from EqF landmarks for features sparse3d hasn't converged.
        landmarks = self.vio.get_landmarks()
        for feat in features:
            fid = int(feat['id'])
            if fid in sparse3d_fids or fid not in landmarks:
                continue
            lm = np.asarray(landmarks[fid], dtype=np.float64)
            rng = float(np.linalg.norm(lm - cam_pos))
            if rng < 0.1:
                continue
            eta = float(np.log(rng))
            priors.append((
                float(feat['x']), float(feat['y']),
                eta, 0.25))

        self.seed_counts.append(len(priors))
        self.patch_depth_frame_count += 1
        pd_start = time.monotonic()
        # Rust PatchDepthMapper expects t_wc as a nested list;
        # Python DISDepthMapper accepts ndarray directly.
        t_wc_arg = t_wc_list if self._depth_mapper_is_rust else t_wc
        depth_result = self.depth_mapper.update(
            stamp_ns / NSEC_PER_SEC,
            self.patch_depth_frame_count,
            gray, t_wc_arg, priors)
        self.patch_depth_times_ms.append(
            (time.monotonic() - pd_start) * 1000.0)

        occupancy_cells = None
        if depth_result is not None and self.occupancy_map is not None:
            occ_start = time.monotonic()
            self.occupancy_map.update(
                depth_result['eta'],
                depth_result['eta_var'],
                depth_result['status'],
                t_wc_list)
            self.occupancy_times_ms.append(
                (time.monotonic() - occ_start) * 1000.0)
            occupancy_cells = self.occupancy_map.occupied_cells()

        return depth_result, occupancy_cells

    def set_rerun_time(self, stamp_ns):
        self.rr.set_time(
            'sensor_time', timestamp=stamp_ns / NSEC_PER_SEC)

    def submit_rerun(self, stamp_ns, gray, features, world,
                     depth_result=None, occupancy_cells=None):
        if self.rr is None:
            return
        display = self.frontend.preprocessed_image()
        full = np.ascontiguousarray(
            display if display is not None else gray)
        # Downsample 2× for the Rerun viewer — the image is only for visual
        # reference; halving the resolution cuts serialization, gRPC transfer,
        # and GPU texture cost by 4×.
        image = full[::2, ::2].copy()
        # Scale feature pixel coords to match the half-res image.
        pixels = np.asarray(
            [[float(f['x']) * 0.5, float(f['y']) * 0.5] for f in features],
            dtype=np.float32).reshape(-1, 2)
        payload = {
            'stamp_ns': stamp_ns,
            'image': image,
            'pixels': pixels,
            'world': world,
            'depth_result': depth_result,
            'occupancy_cells': occupancy_cells,
        }
        self.rerun_frames_submitted += 1
        try:
            self.rerun_queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self.rerun_queue.get_nowait()
            self.rerun_queue.task_done()
            self.rerun_frames_replaced += 1
        except queue.Empty:
            pass
        try:
            self.rerun_queue.put_nowait(payload)
        except queue.Full:
            self.rerun_frames_replaced += 1

    def make_rerun_world_payload(self, position, quaternion, features):
        landmark_map = {
            int(landmark_id): np.asarray(point, dtype=np.float64)
            for landmark_id, point in self.vio.get_landmarks().items()
        }
        landmark_ids = list(landmark_map)
        landmarks = [landmark_map[landmark_id]
                     for landmark_id in landmark_ids]
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)
        visible_pixels = np.empty((0, 2), dtype=np.float32)
        visible_colors = np.empty((0, 3), dtype=np.uint8)
        if landmarks:
            points = np.asarray(landmarks)
            ranges = np.linalg.norm(points - position, axis=1)
            normalized = np.clip(ranges / 12.0, 0.0, 1.0)
            colors = np.column_stack([
                (255.0 * (1.0 - normalized)).astype(np.uint8),
                np.full(len(points), 48, dtype=np.uint8),
                (255.0 * normalized).astype(np.uint8),
            ])
            feature_by_id = {int(f['id']): f for f in features}
            visible = [
                (index, feature_by_id[landmark_id])
                for index, landmark_id in enumerate(landmark_ids)
                if landmark_id in feature_by_id
            ]
            if visible:
                visible_pixels = np.asarray([
                    [float(feature['x']), float(feature['y'])]
                    for _, feature in visible
                ], dtype=np.float32)
                visible_colors = np.asarray([
                    colors[index] for index, _ in visible
                ], dtype=np.uint8)

        return {
            'position': position.copy(),
            'quaternion': quaternion.copy(),
            'points': points,
            'colors': colors,
            'visible_pixels': visible_pixels,
            'visible_colors': visible_colors,
        }

    def rerun_worker(self):
        while not self.rerun_stop.is_set():
            try:
                payload = self.rerun_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t0 = time.monotonic()
                self.log_rerun_payload(payload)
                self.rerun_log_times_ms.append(
                    (time.monotonic() - t0) * 1000.0)
                self.rerun_frames_logged += 1
            except Exception as exc:
                self.rerun_errors += 1
                if self.rerun_errors <= 3:
                    self.get_logger().error(f'Rerun logging failed: {exc}')
            finally:
                self.rerun_queue.task_done()

    def log_rerun_payload(self, payload):
        rr = self.rr
        self.set_rerun_time(payload['stamp_ns'])
        rr.log('camera/image', rr.Image(payload['image']))
        if payload['pixels'].size:
            rr.log(
                'camera/image/frontend_tracks',
                rr.Points2D(
                    payload['pixels'], colors=[40, 255, 80], radii=2.0))

        world = payload['world']
        if world is None:
            return
        position = world['position']
        stamp_sec = payload['stamp_ns'] / NSEC_PER_SEC
        self.estimated_positions.append(position)
        self.estimated_stamps.append(stamp_sec)
        if world['points'].size:
            rr.log(
                'world/landmarks',
                rr.Points3D(
                    world['points'], colors=world['colors'], radii=0.04))
        if world['visible_pixels'].size:
            # Scale VIO landmark overlay to half-res image coords.
            rr.log(
                'camera/image/vio_landmarks',
                rr.Points2D(
                    world['visible_pixels'] * 0.5,
                    colors=world['visible_colors'], radii=5.0))

        rr.log(
            'world/body',
            rr.Transform3D(
                translation=position,
                quaternion=rr.Quaternion(xyzw=world['quaternion'])),
            rr.TransformAxes3D(0.35))
        if len(self.estimated_positions) > 1:
            rr.log(
                'world/estimated_trajectory',
                rr.LineStrips3D(
                    [list(self.estimated_positions)],
                    colors=[255, 80, 40], radii=0.02))

        # ── Reference trajectories (mocap / VIO), SE(3)-aligned ──
        self._align_counter += 1
        n_est = len(self.estimated_stamps)
        if n_est >= 50 and self._align_counter % 10 == 0:
            est_s = list(self.estimated_stamps)
            est_p = list(self.estimated_positions)
            if self.mocap_stamps:
                self.mocap_align = _align_to_estimate(
                    list(self.mocap_stamps), list(self.mocap_positions),
                    est_s, est_p)
            if self.vio_stamps:
                self.vio_align = _align_to_estimate(
                    list(self.vio_stamps), list(self.vio_positions),
                    est_s, est_p)
        # Elapsed time since estimate start — used to mask reference
        # trajectories to "up to now" even when clocks differ.
        est_elapsed_now = (stamp_sec - self.estimated_stamps[0]
                           if len(self.estimated_stamps) > 1 else 0.0)
        if self.mocap_align is not None and len(self.mocap_positions) > 1:
            mp = np.asarray(self.mocap_positions, dtype=np.float64)
            mt = np.asarray(self.mocap_stamps, dtype=np.float64)
            n = min(len(mp), len(mt))   # snapshot race guard
            mp, mt = mp[:n], mt[:n]
            m_elapsed = mt - mt[0]
            mask = m_elapsed <= est_elapsed_now
            mp = mp[mask]
            if len(mp) > 1:
                step = max(1, len(mp) // 500)
                pts = mp[::step]
                ones = np.ones((len(pts), 1), dtype=np.float64)
                aligned = (self.mocap_align @ np.hstack([pts, ones]).T).T[:, :3]
                rr.log(
                    'world/mocap_trajectory',
                    rr.LineStrips3D(
                        [aligned.tolist()],
                        colors=[255, 255, 255], radii=0.015))
        if self.vio_align is not None and len(self.vio_positions) > 1:
            vp = np.asarray(self.vio_positions, dtype=np.float64)
            vt = np.asarray(self.vio_stamps, dtype=np.float64)
            n = min(len(vp), len(vt))   # snapshot race guard
            vp, vt = vp[:n], vt[:n]
            v_elapsed = vt - vt[0]
            mask = v_elapsed <= est_elapsed_now
            vp = vp[mask]
            if len(vp) > 1:
                step = max(1, len(vp) // 500)
                pts = vp[::step]
                ones = np.ones((len(pts), 1), dtype=np.float64)
                aligned = (self.vio_align @ np.hstack([pts, ones]).T).T[:, :3]
                rr.log(
                    'world/vio_trajectory',
                    rr.LineStrips3D(
                        [aligned.tolist()],
                        colors=[80, 160, 255], radii=0.015))

        # ── Dense depth map (JET colormap, matching echo-li-cli) ──
        depth_result = payload.get('depth_result')
        if depth_result is not None:
            eta = depth_result['eta']
            status = depth_result['status']
            valid = (status >= 1) & np.isfinite(eta)
            depth = np.where(valid, np.exp(eta), 0.0)
            depth_rgb = _jet_depth_image(depth, valid, 0.3, 8.0)
            rr.log('depth/image', rr.Image(depth_rgb))

        # ── Occupied voxels as solid cubes (matching echo-li-cli) ──
        occupancy_cells = payload.get('occupancy_cells')
        if occupancy_cells is not None:
            occupied = occupancy_cells
            if occupied.shape[0] > 0:
                hx = 0.05  # half voxel at 0.10 m resolution
                rr.log(
                    'world/occupied_cells',
                    rr.Boxes3D(
                        centers=occupied,
                        half_sizes=np.full((len(occupied), 3), hx,
                                          dtype=np.float32),
                        fill_mode=rr.components.FillMode.Solid,
                        colors=np.full((len(occupied), 4),
                                       [255, 0, 0, 255],
                                       dtype=np.uint8)))

    def stop_rerun(self):
        if self.rerun_stop is None:
            return
        self.rerun_stop.set()
        if self.rerun_thread is not None:
            self.rerun_thread.join(timeout=2.0)

    @staticmethod
    def _compact_rows(msg, channels):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.height) * int(msg.step)
        if raw.size < expected:
            raise ValueError(
                f'image data is short: got {raw.size}, expected {expected}')
        rows = raw[:expected].reshape(int(msg.height), int(msg.step))
        return rows[:, :int(msg.width) * channels]

    def to_gray(self, msg):
        if int(msg.width) != self.width or int(msg.height) != self.height:
            raise ValueError(
                f'unexpected image size {msg.width}x{msg.height}; '
                f'expected {self.width}x{self.height}')
        encoding = msg.encoding.lower()
        if encoding in ('mono8', '8uc1'):
            return self._compact_rows(msg, 1).copy()
        if encoding in ('rgb8', 'bgr8'):
            image = self._compact_rows(msg, 3).reshape(
                self.height, self.width, 3)
            if encoding == 'bgr8':
                image = image[:, :, ::-1]
        elif encoding in ('rgba8', 'bgra8'):
            image = self._compact_rows(msg, 4).reshape(
                self.height, self.width, 4)[:, :, :3]
            if encoding == 'bgra8':
                image = image[:, :, ::-1]
        else:
            raise ValueError(f'unsupported image encoding: {msg.encoding}')
        gray = (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2])
        return np.ascontiguousarray(gray, dtype=np.uint8)

    def publish_odometry(self, stamp_ns, position, quaternion, velocity):
        stamp = _time_msg(stamp_ns)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.body_frame
        odom.pose.pose.position.x = float(position[0])
        odom.pose.pose.position.y = float(position[1])
        odom.pose.pose.position.z = float(position[2])
        odom.pose.pose.orientation.x = float(quaternion[0])
        odom.pose.pose.orientation.y = float(quaternion[1])
        odom.pose.pose.orientation.z = float(quaternion[2])
        odom.pose.pose.orientation.w = float(quaternion[3])
        odom.twist.twist.linear.x = float(velocity[0])
        odom.twist.twist.linear.y = float(velocity[1])
        odom.twist.twist.linear.z = float(velocity[2])
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.odom_frame
            transform.child_frame_id = self.body_frame
            transform.transform.translation.x = float(position[0])
            transform.transform.translation.y = float(position[1])
            transform.transform.translation.z = float(position[2])
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def report_statistics(self):
        elapsed = max(time.monotonic() - self.started_at, 1.0e-9)
        gray_ms = (statistics.median(self.gray_times_ms)
                   if self.gray_times_ms else 0.0)
        frontend_ms = (statistics.median(self.frontend_times_ms)
                       if self.frontend_times_ms else 0.0)
        vision_ms = (statistics.median(self.vision_times_ms)
                     if self.vision_times_ms else 0.0)
        total_ms = (statistics.median(self.total_times_ms)
                    if self.total_times_ms else 0.0)
        tracks = (statistics.median(self.track_counts)
                  if self.track_counts else 0.0)
        rerun_log_ms = (statistics.median(self.rerun_log_times_ms)
                        if self.rerun_log_times_ms else None)
        rerun_part = (
            f'rerun: submitted={self.rerun_frames_submitted} '
            f'logged={self.rerun_frames_logged} '
            f'replaced={self.rerun_frames_replaced} '
            f'errors={self.rerun_errors}')
        if rerun_log_ms is not None:
            rerun_part += f' log_med={rerun_log_ms:.1f}ms'
        mapping_part = ''
        if self.patch_depth_times_ms:
            pd_ms = statistics.median(self.patch_depth_times_ms)
            seeds_med = (int(statistics.median(self.seed_counts))
                         if self.seed_counts else 0)
            mapping_part += f'; seeds_med={seeds_med} patch_depth_med={pd_ms:.1f}ms'
        if self.occupancy_times_ms:
            occ_ms = statistics.median(self.occupancy_times_ms)
            if self.occupancy_map is not None:
                u, f, o = self.occupancy_map.counts()
                mapping_part += (
                    f' occupancy_med={occ_ms:.1f}ms'
                    f' (unk={u} free={f} occ={o})')
            else:
                mapping_part += f' occupancy_med={occ_ms:.1f}ms'
        self.get_logger().info(
            f'input: imu={self.imu_received / elapsed:.1f}Hz '
            f'image={self.images_received / elapsed:.1f}Hz; '
            f'processed: imu={self.imu_processed} image={self.images_processed}; '
            f'queues: imu={len(self.imu_queue)} image={len(self.image_queue)}; '
            f'dropped: imu={self.dropped_imu} image={self.dropped_images}; '
            f'gray_med={gray_ms:.1f}ms frontend_med={frontend_ms:.1f}ms '
            f'vision_med={vision_ms:.1f}ms total_med={total_ms:.1f}ms '
            f'tracks_med={tracks:.0f}; {rerun_part}{mapping_part}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Voxl2EchoLi()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_rerun()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
