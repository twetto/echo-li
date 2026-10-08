#!/usr/bin/env python3
"""Keep a pose alive on /tf as a slow dynamic transform.

run_voxl2_bag_humble.sh runs this after a replay ends so rviz2's follow view
keeps its target frame. It deliberately publishes on /tf and not on
/tf_static: once rviz2 has received a frame as static it treats that frame
as static for the life of the window, and the next replay's live transform
for the same frame then fights the frozen one, which shows up as the axes
flickering between two poses (seen 2026-09-24).
"""
import argparse

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class PinPose(Node):
    def __init__(self, args):
        super().__init__("echo_li_final_pose")
        self._args = args
        self._broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / args.rate, self._tick)

    def _tick(self):
        a = self._args
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = a.frame_id
        t.child_frame_id = a.child_frame_id
        t.transform.translation.x = a.x
        t.transform.translation.y = a.y
        t.transform.translation.z = a.z
        t.transform.rotation.x = a.qx
        t.transform.rotation.y = a.qy
        t.transform.rotation.z = a.qz
        t.transform.rotation.w = a.qw
        self._broadcaster.sendTransform(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame-id", required=True)
    ap.add_argument("--child-frame-id", required=True)
    for name in ("x", "y", "z", "qx", "qy", "qz"):
        ap.add_argument(f"--{name}", type=float, default=0.0)
    ap.add_argument("--qw", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=10.0, help="Hz")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = PinPose(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
