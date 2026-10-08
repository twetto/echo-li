#!/usr/bin/env python3
"""Tie the live mocap to the drone's live VIO so both show in the replay
rviz2 layout, whose fixed frame is the VIO's `echo_li_odom`.

Publishes
  TF  echo_li_odom -> mocap_map   the planar fit (heading + origin) of the
                                  mocap track onto the VIO track, refreshed
                                  every second from the last `fit_window_s`
                                  of time-paired positions (paired by receipt
                                  time, so the two clocks need not agree);
                                  origin-only until the drone has moved
                                  `min_spread_m`
  TF  mocap_map -> mocap_body     the drone's mocap pose
  TF  echo_li_odom -> imu_link    ONLY while no VIO odometry arrives (drone
                                  off): the mocap pose stands in
  Path /echo_li/mocap_path        the mocap trail, in mocap_map

    mocap_live_tf.py --ros-args -p mocap_topic:=/mocap_drone_01/vision_pose/pose \
        -p odom_topic:=/echo_li/odometry
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


def planar_fit(src, dst):
    """Heading and origin with dst ~= R(yaw) src + t, least squares in the
    horizontal plane; z from the mean offset."""
    s = np.asarray(src, np.float64); d = np.asarray(dst, np.float64)
    ms, md = s.mean(0), d.mean(0)
    a = s[:, :2] - ms[:2]; b = d[:, :2] - md[:2]
    yaw = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    c, sn = math.cos(yaw), math.sin(yaw)
    t = md - np.array([c * ms[0] - sn * ms[1], sn * ms[0] + c * ms[1], ms[2]])
    return yaw, t


class MocapLiveTf(Node):
    def __init__(self):
        super().__init__("mocap_live_tf")
        mocap_topic = self.declare_parameter("mocap_topic", "/mocap_drone_01/vision_pose/pose").value
        odom_topic = self.declare_parameter("odom_topic", "/echo_li/odometry").value
        self.odom_frame = self.declare_parameter("odom_frame", "echo_li_odom").value
        self.mocap_frame = self.declare_parameter("mocap_frame", "mocap_map").value
        self.body_frame = self.declare_parameter("body_frame", "imu_link").value
        path_topic = self.declare_parameter("path_topic", "/echo_li/mocap_path").value
        self.path_len = int(self.declare_parameter("path_len", 3000).value)
        self.fit_window_s = float(self.declare_parameter("fit_window_s", 90.0).value)
        self.min_spread_m = float(self.declare_parameter("min_spread_m", 0.8).value)
        self.pair_tol_s = float(self.declare_parameter("pair_tol_s", 0.06).value)
        self.tf = TransformBroadcaster(self)
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.path = Path(); self.path.header.frame_id = self.mocap_frame
        self.mocap = []   # (recv_s, x, y, z)
        self.vio = []     # (recv_s, x, y, z)
        self.last_odom_s = 0.0
        self.yaw, self.t = 0.0, np.zeros(3); self.fit_pairs = 0; self.fit_spread = 0.0
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, mocap_topic, self.on_mocap, best_effort)
        self.create_subscription(Odometry, odom_topic, self.on_odom, best_effort)
        self.create_timer(1.0, self.refit)
        self.create_timer(0.1, self.publish_fit_tf)
        self.create_timer(0.2, self.publish_path)
        self.get_logger().info("mocap %s + VIO %s -> TF %s->%s, trail on %s" % (mocap_topic, odom_topic, self.odom_frame, self.mocap_frame, path_topic))

    def on_odom(self, msg: Odometry):
        now = time.monotonic(); self.last_odom_s = now
        p = msg.pose.pose.position; self.vio.append((now, p.x, p.y, p.z))
        cut = now - self.fit_window_s
        while self.vio and self.vio[0][0] < cut:
            self.vio.pop(0)

    def on_mocap(self, msg: PoseStamped):
        now = time.monotonic(); p = msg.pose.position
        self.mocap.append((now, p.x, p.y, p.z))
        cut = now - self.fit_window_s
        while self.mocap and self.mocap[0][0] < cut:
            self.mocap.pop(0)
        # the drone's mocap pose as a frame
        t = TransformStamped(); t.header.stamp = msg.header.stamp; t.header.frame_id = self.mocap_frame; t.child_frame_id = "mocap_body"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
        t.transform.rotation = msg.pose.orientation
        tfs = [t]
        if now - self.last_odom_s > 3.0:
            # no VIO: let the mocap stand in for the body frame so the view follows something
            u = TransformStamped(); u.header.stamp = msg.header.stamp; u.header.frame_id = self.odom_frame; u.child_frame_id = self.body_frame
            c, sn = math.cos(self.yaw), math.sin(self.yaw)
            u.transform.translation.x = c * p.x - sn * p.y + self.t[0]; u.transform.translation.y = sn * p.x + c * p.y + self.t[1]; u.transform.translation.z = p.z + self.t[2]
            u.transform.rotation = msg.pose.orientation
            tfs.append(u)
        self.tf.sendTransform(tfs)
        q = PoseStamped(); q.header.stamp = msg.header.stamp; q.header.frame_id = self.mocap_frame; q.pose = msg.pose
        if not self.path.poses or sum((getattr(q.pose.position, k) - getattr(self.path.poses[-1].pose.position, k)) ** 2 for k in "xyz") > 1e-4:
            self.path.poses.append(q)
            if len(self.path.poses) > self.path_len:
                del self.path.poses[: len(self.path.poses) - self.path_len]

    def refit(self):
        if len(self.vio) < 20 or len(self.mocap) < 20:
            return
        vt = np.array([v[0] for v in self.vio]); vp = np.array([v[1:] for v in self.vio])
        mt = np.array([m[0] for m in self.mocap]); mp = np.array([m[1:] for m in self.mocap])
        j = np.searchsorted(mt, vt); j = np.clip(j, 1, len(mt) - 1)
        j = np.where(np.abs(mt[j] - vt) < np.abs(mt[j - 1] - vt), j, j - 1)
        ok = np.abs(mt[j] - vt) < self.pair_tol_s
        if ok.sum() < 20:
            return
        src, dst = mp[j[ok]], vp[ok]
        spread = float(np.ptp(src[:, :2], axis=0).max())
        if spread >= self.min_spread_m:
            self.yaw, self.t = planar_fit(src, dst)
        else:   # not enough motion for a heading: origin only, on the latest pair
            c, sn = math.cos(self.yaw), math.sin(self.yaw)
            s_, d_ = src[-1], dst[-1]
            self.t = d_ - np.array([c * s_[0] - sn * s_[1], sn * s_[0] + c * s_[1], s_[2]])
        self.fit_pairs, self.fit_spread = int(ok.sum()), spread

    def publish_fit_tf(self):
        t = TransformStamped(); t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame; t.child_frame_id = self.mocap_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(self.t[0]), float(self.t[1]), float(self.t[2])
        t.transform.rotation.z = math.sin(self.yaw / 2); t.transform.rotation.w = math.cos(self.yaw / 2)
        self.tf.sendTransform(t)

    def publish_path(self):
        if self.path.poses:
            self.path.header.stamp = self.path.poses[-1].header.stamp
            self.path_pub.publish(self.path)


def main():
    rclpy.init()
    node = MocapLiveTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
