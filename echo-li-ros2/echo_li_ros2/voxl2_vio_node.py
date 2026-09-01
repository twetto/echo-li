#!/usr/bin/env python3
"""Real-time ROS 2 adapter for VOXL2 sensors and ECHO-LI."""

from collections import deque
import statistics
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
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


NSEC_PER_SEC = 1_000_000_000


def _stamp_ns(stamp):
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def _time_msg(stamp_ns):
    return rclpy.time.Time(nanoseconds=int(stamp_ns)).to_msg()


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
        self.declare_parameter('max_imu_queue', 5000)
        self.declare_parameter('max_image_queue', 4)
        self.declare_parameter('statistics_period_sec', 5.0)
        self.declare_parameter('rerun_enabled', False)
        self.declare_parameter('rerun_url', '')

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
        self.max_imu_queue = self.get_parameter('max_imu_queue').value
        self.max_image_queue = self.get_parameter('max_image_queue').value
        self.rerun_enabled = self.get_parameter('rerun_enabled').value
        self.rerun_url = self.get_parameter('rerun_url').value

        if len(intrinsics) != 4 or len(distortion) != 4:
            raise ValueError(
                'intrinsics and distortion_coefficients need 4 values')
        if len(t_bs_values) != 16:
            raise ValueError('t_bs needs 16 row-major values')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('image dimensions must be positive')

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
        self.vio.set_camera_extrinsics(
            np.asarray(t_bs_values, dtype=np.float64).reshape(4, 4))

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
        self.frontend_times_ms = deque(maxlen=300)
        self.total_times_ms = deque(maxlen=300)
        self.track_counts = deque(maxlen=300)
        self.estimated_positions = deque(maxlen=20000)
        self.started_at = time.monotonic()
        self._draining = False
        self.rr = None

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
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)

        self.odom_pub = self.create_publisher(Odometry, odometry_topic, 10)
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.publish_tf else None)
        self.create_subscription(Imu, self.imu_topic, self.on_imu, imu_qos)
        self.create_subscription(
            Image, self.image_topic, self.on_image, image_qos)
        self.create_timer(
            self.get_parameter('statistics_period_sec').value,
            self.report_statistics)

        self.get_logger().info(
            f'ECHO-LI ready: imu={self.imu_topic}, image={self.image_topic}, '
            f'camera={self.width}x{self.height} equidistant, '
            f'camera_time_offset={offset_sec:+.6f}s, '
            f'rerun={self.rerun_enabled}')

    def setup_rerun(self):
        import rerun as rr
        import rerun.blueprint as rrb

        rr.init('echo_li_voxl2')
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name='Tracking front', origin='camera',
                    contents=['camera/**'],
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0.0, float(self.width)],
                        y_range=[0.0, float(self.height)])),
                rrb.Spatial3DView(
                    name='Estimated world', origin='world',
                    contents=['world/**']),
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

        frontend_started = time.monotonic()
        features, stats = self.frontend.process(gray)
        frontend_ms = (time.monotonic() - frontend_started) * 1000.0
        observations = {
            int(feature['id']): (float(feature['x']), float(feature['y']))
            for feature in features
        }
        self.vio.process_vision(stamp_ns / NSEC_PER_SEC, observations)
        self.images_processed += 1
        self.frontend_times_ms.append(frontend_ms)
        self.total_times_ms.append((time.monotonic() - started) * 1000.0)
        self.track_counts.append(int(stats['total']))

        if self.rr is not None:
            self.log_rerun_frontend(stamp_ns, gray, features)

        if self.imu_processed < self.n_init:
            return
        position, quaternion = self.vio.get_pose()
        velocity = self.vio.get_velocity()
        gyro_bias, accel_bias = self.vio.get_biases()
        position = np.asarray(position)
        quaternion = np.asarray(quaternion)
        velocity = np.asarray(velocity)
        if not (np.all(np.isfinite(position)) and
                np.all(np.isfinite(quaternion)) and
                np.all(np.isfinite(velocity))):
            self.get_logger().error('ECHO-LI produced a non-finite state')
            return
        self.publish_odometry(stamp_ns, position, quaternion, velocity)
        if self.rr is not None:
            self.log_rerun_state(
                stamp_ns, position, quaternion, velocity,
                np.asarray(gyro_bias), np.asarray(accel_bias), features)

    def set_rerun_time(self, stamp_ns):
        self.rr.set_time(
            'sensor_time', timestamp=stamp_ns / NSEC_PER_SEC)

    def log_rerun_frontend(self, stamp_ns, gray, features):
        rr = self.rr
        self.set_rerun_time(stamp_ns)
        display = self.frontend.preprocessed_image()
        rr.log('camera/image', rr.Image(
            display if display is not None else gray))
        if features:
            pixels = [[float(f['x']), float(f['y'])] for f in features]
            rr.log(
                'camera/image/frontend_tracks',
                rr.Points2D(pixels, colors=[40, 255, 80], radii=2.0))

    def log_rerun_state(self, stamp_ns, position, quaternion, velocity,
                        gyro_bias, accel_bias, features):
        rr = self.rr
        self.set_rerun_time(stamp_ns)
        self.estimated_positions.append(position.copy())

        landmark_map = {
            int(landmark_id): np.asarray(point, dtype=np.float64)
            for landmark_id, point in self.vio.get_landmarks().items()
        }
        landmark_ids = list(landmark_map)
        landmarks = [landmark_map[landmark_id]
                     for landmark_id in landmark_ids]
        if landmarks:
            points = np.asarray(landmarks)
            ranges = np.linalg.norm(points - position, axis=1)
            normalized = np.clip(ranges / 12.0, 0.0, 1.0)
            colors = np.column_stack([
                (255.0 * (1.0 - normalized)).astype(np.uint8),
                np.full(len(points), 48, dtype=np.uint8),
                (255.0 * normalized).astype(np.uint8),
            ])
            rr.log(
                'world/landmarks',
                rr.Points3D(points, colors=colors, radii=0.04))

            feature_by_id = {int(f['id']): f for f in features}
            visible = [
                (index, feature_by_id[landmark_id])
                for index, landmark_id in enumerate(landmark_ids)
                if landmark_id in feature_by_id
            ]
            if visible:
                pixels = [[float(feature['x']), float(feature['y'])]
                          for _, feature in visible]
                pixel_colors = [colors[index] for index, _ in visible]
                rr.log(
                    'camera/image/vio_landmarks',
                    rr.Points2D(
                        pixels, colors=pixel_colors, radii=5.0))

        rr.log(
            'world/body',
            rr.Transform3D(
                translation=position,
                quaternion=rr.Quaternion(xyzw=quaternion)),
            rr.TransformAxes3D(0.35))
        if len(self.estimated_positions) > 1:
            rr.log(
                'world/estimated_trajectory',
                rr.LineStrips3D(
                    [list(self.estimated_positions)],
                    colors=[255, 80, 40], radii=0.02))

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
        frontend_ms = (statistics.median(self.frontend_times_ms)
                       if self.frontend_times_ms else 0.0)
        total_ms = (statistics.median(self.total_times_ms)
                    if self.total_times_ms else 0.0)
        tracks = (statistics.median(self.track_counts)
                  if self.track_counts else 0.0)
        self.get_logger().info(
            f'input: imu={self.imu_received / elapsed:.1f}Hz '
            f'image={self.images_received / elapsed:.1f}Hz; '
            f'processed: imu={self.imu_processed} image={self.images_processed}; '
            f'queues: imu={len(self.imu_queue)} image={len(self.image_queue)}; '
            f'dropped: imu={self.dropped_imu} image={self.dropped_images}; '
            f'frontend_med={frontend_ms:.1f}ms total_med={total_ms:.1f}ms '
            f'tracks_med={tracks:.0f}')


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
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
