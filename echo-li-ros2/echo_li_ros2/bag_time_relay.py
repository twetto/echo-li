#!/usr/bin/env python3
"""Put decoded bag images and recorded IMU on one test timeline."""

import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image, Imu


NSEC_PER_SEC = 1_000_000_000


def stamp_ns(stamp):
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def set_stamp_ns(stamp, value):
    stamp.sec = int(value // NSEC_PER_SEC)
    stamp.nanosec = int(value % NSEC_PER_SEC)


class BagTimeRelay(Node):
    def __init__(self):
        super().__init__('echo_li_bag_time_relay')
        self.declare_parameter('source_imu_topic', '/voxl/raw_imu')
        self.declare_parameter(
            'source_image_topic', '/tracking_front/decoded')
        self.declare_parameter('output_imu_topic', '/echo_li_test/imu')
        self.declare_parameter('output_image_topic', '/echo_li_test/image')
        self.declare_parameter('odometry_topic', '/echo_li/odometry')

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)

        self.imu_pub = self.create_publisher(
            Imu, self.get_parameter('output_imu_topic').value, sensor_qos)
        self.image_pub = self.create_publisher(
            Image, self.get_parameter('output_image_topic').value, image_qos)
        self.create_subscription(
            Imu, self.get_parameter('source_imu_topic').value,
            self.on_imu, sensor_qos)
        self.create_subscription(
            Image, self.get_parameter('source_image_topic').value,
            self.on_image, image_qos)
        self.create_subscription(
            Odometry, self.get_parameter('odometry_topic').value,
            self.on_odometry, 10)

        self.latest_imu_ns = None
        self.latest_imu_wall_ns = None
        self.imu_count = 0
        self.image_count = 0
        self.image_without_imu = 0
        self.odometry_count = 0
        self.started_at = time.monotonic()
        self.create_timer(5.0, self.report)

    def on_imu(self, msg):
        self.latest_imu_ns = stamp_ns(msg.header.stamp)
        self.latest_imu_wall_ns = time.monotonic_ns()
        self.imu_count += 1
        self.imu_pub.publish(msg)

    def on_image(self, msg):
        if self.latest_imu_ns is None:
            self.image_without_imu += 1
            return
        elapsed_ns = time.monotonic_ns() - self.latest_imu_wall_ns
        set_stamp_ns(msg.header.stamp, self.latest_imu_ns + elapsed_ns)
        self.image_count += 1
        self.image_pub.publish(msg)

    def on_odometry(self, _msg):
        self.odometry_count += 1

    def report(self):
        elapsed = max(time.monotonic() - self.started_at, 1.0e-9)
        self.get_logger().info(
            f'relay: imu={self.imu_count} ({self.imu_count / elapsed:.1f}Hz) '
            f'image={self.image_count} ({self.image_count / elapsed:.1f}Hz) '
            f'image_without_imu={self.image_without_imu} '
            f'odometry={self.odometry_count}')


def main(args=None):
    rclpy.init(args=args)
    node = BagTimeRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
