#!/usr/bin/env python3
"""Stream a live voxl2_vio_node run to a Rerun server, for live viewing.

Subscribes to the node's throttled live-view feed (run it with
-p viz_rate_hz:=<Hz>): /echo_li/viz/{image,tracks,odometry,landmarks}. Those
arrive at viz_rate_hz rather than the camera rate, which keeps this process
cheap -- rclpy's executor costs a fixed amount per message, whatever is in it.

ROS traffic stays on this machine; only the Rerun stream crosses the network.
Needs rerun-sdk at the exact version the server and viewer run (0.31.3), and a
Python that can import rclpy (source /opt/ros/humble/setup.bash first):

    python3 rerun_bridge.py --connect rerun+http://<server>:9876/proxy

Entity paths follow echo-li-cli's --vis, so the two look the same:
camera/image, camera/image/features, world/trajectory, world/landmarks.
"""

import argparse
import io
import sys
import time

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from nav_msgs.msg import Odometry
from PIL import Image as PilImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

TIMELINE = "ros_time"
# PointField datatype -> numpy dtype (little-endian).
PF_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def stamp_sec(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def cloud_fields(msg: PointCloud2, names: list[str]) -> np.ndarray | None:
    """The named fields of an unordered cloud as an (n, len(names)) array."""
    by_name = {f.name: f for f in msg.fields}
    if msg.width * msg.height == 0 or any(n not in by_name for n in names):
        return None
    dtype = np.dtype(
        {
            "names": names,
            "formats": [PF_DTYPES[by_name[n].datatype] for n in names],
            "offsets": [by_name[n].offset for n in names],
            "itemsize": msg.point_step,
        }
    )
    rec = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    return np.stack([rec[n].astype(np.float64) for n in names], axis=1)


def id_colors(ids: np.ndarray) -> np.ndarray:
    """A stable colour per track id, so a track keeps its colour across frames."""
    h = (ids.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
    rgb = np.stack([(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF], axis=1).astype(np.uint8)
    return rgb // 2 + 127  # keep them bright against a grey image


class Bridge(Node):
    def __init__(self, args):
        super().__init__("echo_li_rerun_bridge")
        self.args = args
        # The trajectory goes out in segments. Re-sending the whole line grows
        # without bound; a closed segment is sent once, as static data, so the
        # server's memory limit drops old camera frames but never the path.
        self.segment: list[np.ndarray] = []
        self.segment_index = 0
        self.trajectory_points = 0
        self.trajectory_logged_at = 0.0
        self.counts = {"image": 0, "tracks": 0, "odom": 0, "landmarks": 0}
        self.image_size = None  # (w, h), from the first image
        self.started = time.monotonic()

        p = args.prefix
        self.create_subscription(Image, f"{p}/image", self.on_image, 5)
        self.create_subscription(PointCloud2, f"{p}/tracks", self.on_tracks, 5)
        self.create_subscription(Odometry, f"{p}/odometry", self.on_odom, 5)
        self.create_subscription(PointCloud2, f"{p}/landmarks", self.on_landmarks, 5)
        if args.stats_sec > 0:
            self.create_timer(args.stats_sec, self.report)

    def on_image(self, msg: Image):
        enc = msg.encoding.lower()
        channels = {"mono8": 1, "8uc1": 1, "rgb8": 3, "bgr8": 3}.get(enc)
        if channels is None:
            self.get_logger().warn(f"unsupported encoding {msg.encoding!r}", once=True)
            return
        a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
        a = a[:, : msg.width * channels]
        if channels == 3:
            a = a.reshape(msg.height, msg.width, 3)
            if enc == "bgr8":
                a = a[:, :, ::-1]
        buf = io.BytesIO()
        PilImage.fromarray(np.ascontiguousarray(a)).save(
            buf, format="JPEG", quality=self.args.jpeg_quality
        )
        if self.image_size != (msg.width, msg.height):
            # Pin the camera view to the image. Left to auto-fit, it re-frames
            # around whatever is drawn, and the picture wobbles.
            self.image_size = (msg.width, msg.height)
            rr.send_blueprint(blueprint(*self.image_size))
        rr.set_time(TIMELINE, timestamp=stamp_sec(msg.header))
        rr.log("camera/image", rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"))
        self.counts["image"] += 1

    def on_tracks(self, msg: PointCloud2):
        rr.set_time(TIMELINE, timestamp=stamp_sec(msg.header))
        pts = cloud_fields(msg, ["x", "y", "id"])
        if pts is not None and self.image_size is not None:
            w, h = self.image_size
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            pts = pts[inside] if inside.any() else None
        if pts is None:
            rr.log("camera/image/features", rr.Clear(recursive=False))
            return
        rr.log(
            "camera/image/features",
            rr.Points2D(pts[:, :2], colors=id_colors(pts[:, 2]), radii=self.args.point_radius),
        )
        self.counts["tracks"] += 1

    def on_odom(self, msg: Odometry):
        t = stamp_sec(msg.header)
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y, p.z])
        if not np.all(np.isfinite(pos)):
            return
        q = msg.pose.pose.orientation
        rr.set_time(TIMELINE, timestamp=t)
        rr.log(
            "world/body",
            rr.Transform3D(translation=pos, quaternion=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])),
        )
        self.counts["odom"] += 1

        if not self.segment or np.linalg.norm(pos - self.segment[-1]) >= self.args.path_step:
            self.segment.append(pos)
            self.trajectory_points += 1
        full = len(self.segment) >= self.args.segment_points
        if full or t - self.trajectory_logged_at >= 1.0 or t < self.trajectory_logged_at:
            self.trajectory_logged_at = t
            if len(self.segment) >= 2:
                rr.log(
                    f"world/trajectory/{self.segment_index:05d}",
                    rr.LineStrips3D([np.array(self.segment)], colors=[[255, 200, 40]]),
                    static=True,
                )
            if full:  # close it; the next one starts where this ended
                self.segment = self.segment[-1:]
                self.segment_index += 1

    def on_landmarks(self, msg: PointCloud2):
        rr.set_time(TIMELINE, timestamp=stamp_sec(msg.header))
        pts = cloud_fields(msg, ["x", "y", "z"])
        if pts is None:
            rr.log("world/landmarks", rr.Clear(recursive=False))
            return
        rr.log("world/landmarks", rr.Points3D(pts, colors=[[80, 180, 255]], radii=0.02))
        self.counts["landmarks"] += 1

    def report(self):
        el = time.monotonic() - self.started
        rates = ", ".join(f"{k} {v / el:.1f}/s" for k, v in self.counts.items())
        self.get_logger().info(f"{rates}; trajectory {self.trajectory_points} pts")


def blueprint(width: int | None = None, height: int | None = None) -> rrb.Blueprint:
    bounds = None
    if width and height:
        bounds = rrb.VisualBounds2D(x_range=[0, width], y_range=[0, height])
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="camera", name="Camera", visual_bounds=bounds),
            rrb.Spatial3DView(origin="world", name="World"),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--connect", default="rerun+http://127.0.0.1:9876/proxy",
                    help="Rerun server gRPC URL; empty to only --save")
    ap.add_argument("--save", default="", metavar="FILE.rrd",
                    help="also write the stream to a recording file")
    ap.add_argument("--prefix", default="/echo_li/viz", help="the node's live-view topics")
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--point-radius", type=float, default=2.0, help="feature marker, px")
    ap.add_argument("--path-step", type=float, default=0.01,
                    help="metres between kept trajectory points")
    ap.add_argument("--segment-points", type=int, default=500,
                    help="trajectory points per segment; only the open one is re-sent")
    ap.add_argument("--recording-id", default=None,
                    help="reuse one id across restarts so the viewer keeps one recording")
    ap.add_argument("--stats-sec", type=float, default=10.0)
    args, ros_args = ap.parse_known_args()

    # Send each log call within 8 ms, not in the default 200 ms batches: a live
    # viewer following the newest data shows only the last frame of each batch,
    # which capped the camera view at about 5 frames per second at any rate.
    rr.RecordingStream(
        "echo-li-rig",
        recording_id=args.recording_id,
        make_default=True,
        batcher_config=rr.ChunkBatcherConfig.LOW_LATENCY(),
    )
    sinks = [rr.GrpcSink(args.connect)] if args.connect else []
    if args.save:
        sinks.append(rr.FileSink(args.save))
    if not sinks:
        sys.exit("nothing to stream to: give --connect, --save or both")
    rr.set_sinks(*sinks, default_blueprint=blueprint())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/body", rr.TransformAxes3D(0.3), static=True)

    rclpy.init(args=[sys.argv[0], *ros_args])
    node = Bridge(args)
    node.get_logger().info(
        f"{args.prefix}/* -> {args.connect or '-'}, saving to {args.save or '-'}"
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rr.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
