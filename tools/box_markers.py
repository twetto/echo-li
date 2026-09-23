#!/usr/bin/env python3
"""Draw mocap-tracked rigid bodies as solid cubes in RViz, live from VRPN.

Subscribes to one PoseStamped per body and republishes each as a CUBE marker,
so the real obstacles appear in the 3D view next to whatever else is shown:
the occupancy cloud, the depth cloud, the trajectory.

    box_markers.py --ros-args \
        -p bodies:="[box1,box2]" \
        -p size:="[0.40, 0.90, 0.50]" -p height_axis:=1 -p anchor:=top

`bodies` and `size` are required: which bodies are obstacles and how big they
are belongs to the experiment, not to this script.

Two things a hand-written marker usually gets wrong:

* **Anchor.** A CUBE marker is centred on its pose, but a VRPN rigid body is
  usually defined at the marker cluster, not the centroid. With `anchor: top`
  the pose is taken as the centre of the TOP surface and the cube is pushed down
  half its height along its own up axis, so the box hangs below the markers
  where the real one is.
* **Up axis.** `size` is given along the body's own axes and `height_axis` says
  which of them is vertical. A VRPN stream is often Y-up, in which case the
  height is the second component and `height_axis:=1`.

Markers are stamped zero ("latest transform"), because the VRPN clock and the
estimator's clock are generally not the same clock -- see `_header`.

Markers are published in the frame the poses arrive in. Relating that frame to
the estimator's is the launcher's job (see run_voxl2_bag_humble.sh --boxes):
the VIO node broadcasts `<odom> -> <mocap frame>` from its mocap fit, and the
launcher adds the static axis change between the fitted mocap frame and the
VRPN one. Without that chain the boxes and the map are two clouds in unrelated
coordinates, and any visual agreement is a coincidence.
"""
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

# Best-effort matches a best-effort publisher and still matches a reliable one,
# so one setting reads any VRPN configuration.
BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

PALETTE = [(0.95, 0.45, 0.15), (0.20, 0.60, 0.95), (0.40, 0.80, 0.35),
           (0.85, 0.30, 0.55), (0.95, 0.80, 0.20)]


def rotate(q, v):
    """Rotate vector v by geometry_msgs Quaternion q."""
    t = [2 * (q.y * v[2] - q.z * v[1]),
         2 * (q.z * v[0] - q.x * v[2]),
         2 * (q.x * v[1] - q.y * v[0])]
    cross = (q.y * t[2] - q.z * t[1],
             q.z * t[0] - q.x * t[2],
             q.x * t[1] - q.y * t[0])
    return [v[i] + q.w * t[i] + cross[i] for i in range(3)]


class BoxMarkers(Node):
    def __init__(self):
        super().__init__("box_markers")
        self.bodies = list(self.declare_parameter("bodies", [""]).value)
        self.bodies = [b for b in self.bodies if b]
        self.template = self.declare_parameter(
            "topic_template", "/vrpn_mocap/{}/pose").value
        self.size = list(self.declare_parameter("size", [0.0]).value)
        self.height_axis = int(self.declare_parameter("height_axis", 2).value)
        self.anchor = self.declare_parameter("anchor", "top").value
        self.alpha = float(self.declare_parameter("alpha", 0.55).value)
        self.frame_override = self.declare_parameter("frame_id", "").value
        self.label = bool(self.declare_parameter("label", True).value)
        rate = float(self.declare_parameter("publish_rate", 10.0).value)

        problems = []
        if not self.bodies:
            problems.append("parameter 'bodies' is empty: name the rigid bodies to draw")
        if len(self.size) != 3 or any(s <= 0 for s in self.size):
            problems.append("parameter 'size' must be three positive extents in metres")
        if self.anchor not in ("top", "centre", "center", "bottom"):
            problems.append("parameter 'anchor' must be top, centre or bottom")
        if not 0 <= self.height_axis <= 2:
            problems.append("parameter 'height_axis' must be 0, 1 or 2")
        if problems:
            for p in problems:
                self.get_logger().error(p)
            raise SystemExit(2)

        self.latest = {}
        for name in self.bodies:
            topic = self.template.format(name)
            self.create_subscription(
                PoseStamped, topic,
                (lambda m, k=name: self.latest.__setitem__(k, m)), BEST_EFFORT)
            self.get_logger().info(f"{name}: {topic}")

        self.pub = self.create_publisher(MarkerArray, "~/markers", 1)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"size {self.size} m, height axis {'xyz'[self.height_axis]}, "
            f"anchor {self.anchor}; publishing ~/markers at {rate:g} Hz")

    def tick(self):
        if not self.latest:
            return
        arr = MarkerArray()
        for i, name in enumerate(self.bodies):
            msg = self.latest.get(name)
            if msg is None:
                continue
            arr.markers.append(self.cube(i, name, msg))
            if self.label:
                arr.markers.append(self.text(i, name, msg))
        self.pub.publish(arr)

    def _anchored(self, msg):
        """Cube centre, given where the pose sits on the box."""
        p = msg.pose.position
        if self.anchor in ("centre", "center"):
            return [p.x, p.y, p.z]
        half = self.size[self.height_axis] / 2.0
        local = [0.0, 0.0, 0.0]
        local[self.height_axis] = -half if self.anchor == "top" else half
        d = rotate(msg.pose.orientation, local)
        return [p.x + d[0], p.y + d[1], p.z + d[2]]

    def _header(self, marker, msg):
        # Stamp zero, not the pose's stamp. RViz looks the transform up at the
        # marker's time, and the VRPN clock need not match the clock the rest
        # of the TF tree is on: with a recorded VOXL stream it is weeks apart,
        # and every marker was silently dropped as "extrapolation". A zero
        # stamp means "latest transform", which is the right thing for bodies
        # that do not move.
        marker.header.frame_id = self.frame_override or msg.header.frame_id

    def cube(self, i, name, msg):
        m = Marker()
        self._header(m, msg)
        m.ns, m.id, m.type, m.action = "mocap_boxes", i, Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = self._anchored(msg)
        m.pose.orientation = msg.pose.orientation
        m.scale.x, m.scale.y, m.scale.z = self.size
        r, g, b = PALETTE[i % len(PALETTE)]
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, self.alpha
        return m

    def text(self, i, name, msg):
        m = Marker()
        self._header(m, msg)
        m.ns, m.id = "mocap_box_labels", i
        m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        # Float the label clear of the top surface, along the body's up axis.
        lift = [0.0, 0.0, 0.0]
        lift[self.height_axis] = 0.15
        if self.anchor in ("centre", "center"):
            lift[self.height_axis] += self.size[self.height_axis] / 2.0
        elif self.anchor == "bottom":
            lift[self.height_axis] += self.size[self.height_axis]
        d = rotate(msg.pose.orientation, lift)
        p = msg.pose.position
        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            p.x + d[0], p.y + d[1], p.z + d[2]
        m.pose.orientation.w = 1.0
        m.scale.z = 0.12
        m.color.r = m.color.g = m.color.b = 1.0
        m.color.a = 0.9
        m.text = name
        return m


def main():
    rclpy.init()
    try:
        node = BoxMarkers()
    except SystemExit as e:
        rclpy.shutdown()
        sys.exit(e.code)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # rclpy's own SIGINT handler shuts the context down first, and spin
        # then raises ExternalShutdownException rather than KeyboardInterrupt.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
