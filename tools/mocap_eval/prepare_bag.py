#!/usr/bin/env python3
"""Decode one VOXL2 + mocap ROS 2 bag into an offline replay cache.

Writes
  <out_dir>/frames.npy   uint8 (N, 800, 1280): the luma plane of every
                         /tracking_front_misp_encoded message, which is exactly
                         what voxl_h265_decoder publishes as mono8.
  <out_dir>/sensors.npz  header and bag-receive stamps (int64 ns) for the
                         camera, /voxl/raw_imu and both mocap topics, plus IMU
                         samples and mocap poses.
  <out_dir>/info.json    counts, frame ids and clock statistics.

The db3 is opened immutable, so the bag (and its -wal/-shm) is never touched.
Needs rclpy (ROS 2 Humble sourced) and PyAV, e.g. in the ubuntu-22-04
distrobox with the handheld0908 venv:

    source /opt/ros/humble/setup.bash
    ~/Downloads/RC_V1p5_Fusion_Pipeline_Jetson_Nano/handheld0908/.venv/bin/python \\
        prepare_bag.py ~/Downloads/手持0908/第一筆 ~/.cache/echo-li/eval/hh1
"""
import argparse
import glob
import json
import os
import sqlite3
import time

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CAM_TOPIC = "/tracking_front_misp_encoded"
IMU_TOPIC = "/voxl/raw_imu"
MOCAP_TOPICS = {"vrpn": "/vrpn_mocap/drone_01/pose",
                "vp": "/mocap_drone_01/vision_pose/pose"}
WIDTH, HEIGHT = 1280, 800


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class Bag:
    def __init__(self, bag_dir):
        dbs = glob.glob(os.path.join(bag_dir, "*.db3"))
        if len(dbs) != 1:
            raise SystemExit(f"expected one .db3 in {bag_dir}, found {len(dbs)}")
        self.path = os.path.abspath(dbs[0])
        self.con = sqlite3.connect(f"file:{self.path}?immutable=1", uri=True)
        self.topics = {name: (tid, typ) for tid, name, typ in
                       self.con.execute("select id, name, type from topics")}

    def count(self, topic):
        tid, _ = self.topics[topic]
        return self.con.execute("select count(*) from messages where topic_id=?",
                                (tid,)).fetchone()[0]

    def messages(self, topic):
        tid, typ = self.topics[topic]
        cls = get_message(typ)
        cur = self.con.execute("select timestamp, data from messages where topic_id=? "
                               "order by timestamp", (tid,))
        for bag_ns, data in cur:
            yield bag_ns, deserialize_message(data, cls)


def extract_imu(bag):
    bag_ns, hdr, gyr, acc = [], [], [], []
    for t, m in bag.messages(IMU_TOPIC):
        bag_ns.append(t)
        hdr.append(stamp_ns(m))
        g, a = m.angular_velocity, m.linear_acceleration
        gyr.append((g.x, g.y, g.z))
        acc.append((a.x, a.y, a.z))
    return dict(imu_bag=np.array(bag_ns, np.int64), imu_hdr=np.array(hdr, np.int64),
                imu_gyr=np.array(gyr), imu_acc=np.array(acc))


def extract_pose(bag, key, topic):
    bag_ns, hdr, pos, quat, frame_id = [], [], [], [], None
    for t, m in bag.messages(topic):
        bag_ns.append(t)
        hdr.append(stamp_ns(m))
        p, q = m.pose.position, m.pose.orientation
        pos.append((p.x, p.y, p.z))
        quat.append((q.x, q.y, q.z, q.w))
        frame_id = m.header.frame_id
    arrays = {f"{key}_bag": np.array(bag_ns, np.int64), f"{key}_hdr": np.array(hdr, np.int64),
              f"{key}_pos": np.array(pos), f"{key}_quat": np.array(quat)}
    return arrays, frame_id


def decode_jpeg_camera(bag, frames_path):
    """Same cache format as decode_camera, for bags whose camera is JPEG."""
    import cv2
    n = bag.count(CAM_TOPIC)
    frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8,
                                       shape=(n, HEIGHT, WIDTH))
    cam_bag = np.zeros(n, np.int64)
    cam_hdr = np.zeros(n, np.int64)
    decoded = np.zeros(n, bool)
    for i, (t, m) in enumerate(bag.messages(CAM_TOPIC)):
        cam_bag[i] = t
        cam_hdr[i] = stamp_ns(m)
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (HEIGHT, WIDTH):
            raise SystemExit(f"frame {i}: decode failed or wrong size {None if img is None else img.shape}")
        frames[i] = img
        decoded[i] = True
    frames.flush()
    return dict(cam_bag=cam_bag, cam_hdr=cam_hdr, cam_decoded=decoded)


def decode_camera(bag, frames_path):
    import av  # only needed for the h265 path
    n = bag.count(CAM_TOPIC)
    frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8,
                                       shape=(n, HEIGHT, WIDTH))
    cam_bag = np.zeros(n, np.int64)
    cam_hdr = np.zeros(n, np.int64)
    decoded = np.zeros(n, bool)
    dec = av.CodecContext.create("hevc", "r")
    dec.thread_type = "AUTO"

    def take(fr):
        if (fr.width, fr.height) != (WIDTH, HEIGHT):
            raise SystemExit(f"unexpected frame size {fr.width}x{fr.height}")
        plane = fr.planes[0]
        y = np.frombuffer(plane, np.uint8).reshape(fr.height, plane.line_size)
        frames[fr.pts] = y[:, :WIDTH]
        decoded[fr.pts] = True

    for i, (t, m) in enumerate(bag.messages(CAM_TOPIC)):
        if m.format != "h265":
            raise SystemExit(f"expected h265, got {m.format!r}")
        cam_bag[i] = t
        cam_hdr[i] = stamp_ns(m)
        pkt = av.Packet(bytes(m.data))
        pkt.pts = i
        for fr in dec.decode(pkt):
            take(fr)
    for fr in dec.decode(None):
        take(fr)
    frames.flush()
    return dict(cam_bag=cam_bag, cam_hdr=cam_hdr, cam_decoded=decoded)


def clock_stats(arrays, key, t0_bag):
    b, h = arrays[f"{key}_bag"], arrays[f"{key}_hdr"]
    off = (h - b) * 1e-9
    dh = np.diff(h) * 1e-9
    return dict(n=int(len(b)),
                bag_start_s=float((b[0] - t0_bag) * 1e-9),
                bag_end_s=float((b[-1] - t0_bag) * 1e-9),
                hdr_minus_bag_median_s=float(np.median(off)),
                hdr_minus_bag_p1_s=float(np.percentile(off, 1)),
                hdr_minus_bag_p99_s=float(np.percentile(off, 99)),
                hdr_dt_median_ms=float(np.median(dh) * 1e3),
                hdr_dt_max_ms=float(dh.max() * 1e3),
                hdr_nonmonotonic=int((dh <= 0).sum()),
                rate_hz=float((len(b) - 1) / ((b[-1] - b[0]) * 1e-9)))


def main():
    global IMU_TOPIC, CAM_TOPIC
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--imu-topic", default=IMU_TOPIC)
    ap.add_argument("--camera-topic", default=CAM_TOPIC)
    ap.add_argument("--camera-codec", default="h265", choices=("h265", "jpeg"))
    args = ap.parse_args()
    IMU_TOPIC, CAM_TOPIC = args.imu_topic, args.camera_topic

    bag = Bag(args.bag_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.monotonic()
    arrays = extract_imu(bag)
    frame_ids = {}
    for key, topic in MOCAP_TOPICS.items():
        if topic in bag.topics:
            pose_arrays, frame_ids[key] = extract_pose(bag, key, topic)
            arrays.update(pose_arrays)
    print(f"sensors extracted in {time.monotonic() - started:.1f}s", flush=True)
    decode = decode_jpeg_camera if args.camera_codec == "jpeg" else decode_camera
    arrays.update(decode(bag, os.path.join(args.out_dir, "frames.npy")))
    print(f"camera decoded in {time.monotonic() - started:.1f}s "
          f"({int(arrays['cam_decoded'].sum())}/{len(arrays['cam_decoded'])} frames)",
          flush=True)
    np.savez(os.path.join(args.out_dir, "sensors.npz"), **arrays)

    t0_bag = int(arrays["imu_bag"][0])
    info = dict(bag=bag.path, topics={k: v[1] for k, v in bag.topics.items()},
                mocap_topics={k: t for k, t in MOCAP_TOPICS.items() if t in bag.topics},
                mocap_frame_ids=frame_ids,
                frames_decoded=int(arrays["cam_decoded"].sum()),
                clocks={k: clock_stats(arrays, k, t0_bag)
                        for k in ["imu", "cam"] + [k for k in MOCAP_TOPICS
                                                   if f"{k}_bag" in arrays]})
    for key in MOCAP_TOPICS:
        if f"{key}_pos" in arrays:
            pq = np.hstack([arrays[f"{key}_pos"], arrays[f"{key}_quat"]])
            info["clocks"][key]["repeat_fraction"] = float(np.mean(np.all(pq[1:] == pq[:-1], axis=1)))
    with open(os.path.join(args.out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(json.dumps(info["clocks"], indent=1))


if __name__ == "__main__":
    main()
