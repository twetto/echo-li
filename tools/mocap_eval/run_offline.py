#!/usr/bin/env python3
"""Deterministic offline ECHO-LI replay on a prepare_bag.py cache.

Mirrors voxl2_vio_node's estimator path: Frontend.process, then
VIOFilter.process_imu for every IMU sample up to the image stamp, then
process_vision at camera stamp + camera_time_offset_sec. As in the node,
non-increasing stamps and images with no new IMU sample are dropped.
Differences from a live bag run:
  * it uses the true camera header stamps. bag_time_relay replaces them with
    latest-IMU-stamp + wall-clock decode delay.
  * it skips dense mapping and Rerun, which never feed back into the pose.

The frontend never sees filter state, so its tracks depend only on the
RudolfV settings and the calibration. They are computed once ("frontend") and
replayed into the filter as often as needed ("filter"), which takes seconds
instead of minutes.

  PY=~/.cache/echo-li/ros2-humble-py3.10/venv/bin/python   # in ubuntu-22-04
  $PY run_offline.py frontend CACHE -o CACHE/tracks_id1.npz
  $PY run_offline.py filter CACHE CACHE/tracks_id1.npz -o traj.npz \\
      --set eqf.velocityNoise.acc=0.005 --camera-offset 0.0
"""
import argparse
import contextlib
import json
import os
import tempfile
import time

import numpy as np
import yaml

import echo_li

NSEC_PER_SEC = 1_000_000_000
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "echo-li-ros2", "config")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "eqvio_voxl2.yaml")
DEFAULT_CALIB = os.path.join(CONFIG_DIR, "voxl2_internal_id_1.yaml")
DEFAULT_CAMERA_OFFSET = 0.004   # run_voxl2_ros2.sh CAMERA_OFFSET


def load_calibration(path):
    with open(path) as f:
        params = yaml.safe_load(f)["echo_li_voxl2"]["ros__parameters"]
    return dict(width=int(params["image_width"]), height=int(params["image_height"]),
                intrinsics=[float(v) for v in params["intrinsics"]],
                distortion=[float(v) for v in params["distortion_coefficients"]],
                t_bs=np.asarray(params["t_bs"], dtype=np.float64).reshape(4, 4))


def parse_overrides(items):
    """['eqf.velocityNoise.acc=0.01'] -> {'eqf.velocityNoise.acc': 0.01}.

    Keys must already exist in the base config (catches typos, which serde
    would otherwise ignore silently); prefix a key with '+' to add a new one.
    """
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override needs key=value: {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def apply_overrides(cfg, overrides):
    for key, value in overrides.items():
        allow_new = key.startswith("+")
        parts = key.lstrip("+").split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                if not allow_new:
                    raise KeyError(f"unknown config section {part!r} in {key!r}")
                node[part] = {}
            node = node[part]
        if parts[-1] not in node and not allow_new:
            raise KeyError(f"unknown config key {key!r} (prefix with '+' to add it)")
        node[parts[-1]] = value
    return cfg


@contextlib.contextmanager
def materialized_config(base_path, overrides):
    """Yield a YAML path with overrides applied; a temp file only if needed."""
    if not overrides:
        yield base_path
        return
    with open(base_path) as f:
        cfg = apply_overrides(yaml.safe_load(f), overrides)
    fd, path = tempfile.mkstemp(prefix="echo_li_cfg_", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        yield path
    finally:
        os.unlink(path)


def run_frontend(cache, calib, config_path, progress_every=500,
                 camera_model="equidistant"):
    frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
    sensors = np.load(os.path.join(cache, "sensors.npz"))
    fx, fy, cx, cy = calib["intrinsics"]
    cfg = echo_li.FrontendConfig.from_yaml(config_path)
    cfg.set_camera(fx, fy, cx, cy, calib["width"], calib["height"], calib["distortion"],
                   distortion_model=camera_model)
    frontend = echo_li.Frontend(cfg, calib["width"], calib["height"])

    index = np.flatnonzero(sensors["cam_decoded"])
    counts, totals, ids, xy = [], [], [], []
    started = time.monotonic()
    for n, i in enumerate(index):
        features, stats = frontend.process(np.ascontiguousarray(frames[i]))
        counts.append(len(features))
        totals.append(int(stats["total"]))
        ids.extend(int(f["id"]) for f in features)
        xy.extend((f["x"], f["y"]) for f in features)
        if progress_every and n % progress_every == 0:
            print(f"frontend {n}/{len(index)} {time.monotonic() - started:.0f}s", flush=True)
    return dict(frame_index=index, stamp_ns=sensors["cam_hdr"][index],
                count=np.asarray(counts, np.int32), total=np.asarray(totals, np.int32),
                ids=np.asarray(ids, np.int64), xy=np.asarray(xy, np.float32).reshape(-1, 2))


def run_filter(sensors, tracks, calib, config_path,
               camera_offset_s=DEFAULT_CAMERA_OFFSET, n_init=100,
               camera_model="equidistant"):
    """Replay IMU + cached tracks through a fresh VIOFilter.

    Returns t_ns (IMU clock), p/q (T_wb, q xyzw), v (body-frame velocity),
    bg/ba biases and drop counters.
    """
    fx, fy, cx, cy = calib["intrinsics"]
    camera = (echo_li.RadTanCamera if camera_model == "radtan"
              else echo_li.EquidistantCamera)(fx, fy, cx, cy, *calib["distortion"])
    vio = echo_li.VIOFilter(config_path, camera, n_init_samples=n_init)
    vio.set_camera_extrinsics(calib["t_bs"])


    imu_t = sensors["imu_hdr"]
    gyr = sensors["imu_gyr"].tolist()
    acc = sensors["imu_acc"].tolist()
    offset_ns = int(round(camera_offset_s * NSEC_PER_SEC))
    starts = np.concatenate([[0], np.cumsum(tracks["count"])])
    track_ids = tracks["ids"].tolist()
    track_xy = tracks["xy"].tolist()

    out = {k: [] for k in ("t_ns", "p", "q", "v", "bg", "ba")}
    k = 0
    last_imu = last_image = None
    imu_processed = dropped_images = 0
    diverged = False
    started = time.monotonic()
    for f, stamp in enumerate(tracks["stamp_ns"].tolist()):
        image_ns = stamp + offset_ns
        if last_image is not None and image_ns <= last_image:
            dropped_images += 1
            continue
        last_image = image_ns
        processed = 0
        while k < len(imu_t) and imu_t[k] <= image_ns:
            s = int(imu_t[k])
            if last_imu is None or s > last_imu:
                vio.process_imu(s / NSEC_PER_SEC, gyr[k], acc[k])
                last_imu = s
                imu_processed += 1
                processed += 1
            k += 1
        if processed == 0:
            dropped_images += 1
            continue
        a, b = starts[f], starts[f + 1]
        vio.process_vision(image_ns / NSEC_PER_SEC,
                           {i: xy for i, xy in zip(track_ids[a:b], track_xy[a:b])})
        if imu_processed < n_init:
            continue
        p, q = vio.get_pose()
        v = vio.get_velocity()
        bg, ba = vio.get_biases()
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
            diverged = True
            break
        for key, value in (("t_ns", image_ns), ("p", p), ("q", q), ("v", v), ("bg", bg), ("ba", ba)):
            out[key].append(value)
    result = {k: np.asarray(v) for k, v in out.items()}
    result["t_ns"] = result["t_ns"].astype(np.int64)
    result.update(dropped_images=dropped_images, imu_processed=imu_processed,
                  diverged=diverged, runtime_s=time.monotonic() - started)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frontend", "filter"):
        p = sub.add_parser(name)
        p.add_argument("cache")
        if name == "filter":
            p.add_argument("tracks")
            p.add_argument("--camera-offset", type=float, default=DEFAULT_CAMERA_OFFSET)
            p.add_argument("--n-init", type=int, default=100)
            p.add_argument("--params", default="",
                           help="sweep-style JSON (config keys + camera_offset); "
                                "--set entries override it")
        p.add_argument("--calib", default=DEFAULT_CALIB)
        p.add_argument("--camera-model", default="equidistant",
                       choices=("equidistant", "radtan"))
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    overrides = {}
    if getattr(args, "params", ""):
        with open(args.params) as f:
            overrides = json.load(f)
        args.camera_offset = overrides.pop("camera_offset", args.camera_offset)
    overrides.update(parse_overrides(args.set))
    meta = dict(cache=os.path.abspath(args.cache), calib=os.path.abspath(args.calib),
                camera_model=args.camera_model,
                config=os.path.abspath(args.config), overrides=overrides)
    with materialized_config(args.config, overrides) as config_path:
        if args.cmd == "frontend":
            result = run_frontend(args.cache, calib, config_path,
                                  camera_model=args.camera_model)
        else:
            sensors = np.load(os.path.join(args.cache, "sensors.npz"))
            tracks = np.load(args.tracks)
            result = run_filter(sensors, tracks, calib, config_path,
                                args.camera_offset, args.n_init,
                                camera_model=args.camera_model)
            meta.update(tracks=os.path.abspath(args.tracks), camera_offset=args.camera_offset,
                        **{k: result.pop(k) for k in ("dropped_images", "imu_processed",
                                                      "diverged", "runtime_s")})
    np.savez(args.out, meta=json.dumps(meta), **result)
    print(json.dumps({k: v for k, v in meta.items() if k != "overrides"}, default=str))


if __name__ == "__main__":
    main()
