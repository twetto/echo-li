#!/usr/bin/env python3
"""Compare frontend (RudolfV) variants at fixed filter parameters.

For every variant and bag, the frontend tracks are computed once (~85 s;
cached as <cache>/tracks_fe_<variant>.npz) and then replayed through the
filter with --params (the sweep's best JSON). Absolute metrics are printed,
because the sweep's normalised score is only comparable within one set of
tracks.

  PY=~/.cache/echo-li/ros2-humble-py3.10/venv/bin/python   # in ubuntu-22-04
  $PY frontend_variants.py --bags hh1 hh2 --params es_best.json -o fe.json
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np

import evaluate as ev
import run_offline as ro

EVAL_ROOT = os.path.expanduser("~/.cache/echo-li/eval")

VARIANTS = {
    "base": {},
    # eqvio_voxl2.yaml spells these clahe_tile_size / clahe_clip_limit, but
    # RudolfVConfig is camelCase, so serde ignores them and the frontend runs
    # with its defaults (256 px, 4.0). This variant applies the intended values.
    "clahe_128_10": {"+RudolfV.claheTileSize": 128, "+RudolfV.claheClipLimit": 10.0},
    "no_ransac": {"RudolfV.enableRansac": False},
    "klt_residual": {"+RudolfV.kltResidual": True},
    "fast20": {"RudolfV.fastThreshold": 20},
    "fast60": {"RudolfV.fastThreshold": 60},
    "dist24": {"RudolfV.featureDist": 24.0},
    "feat150": {"RudolfV.maxFeatures": 150},
    "lbp_hard": {"RudolfV.lbpPolicy": "HardReject"},
}


def run_task(task):
    name, bag, args = task
    cache = os.path.join(EVAL_ROOT, bag)
    calib = ro.load_calibration(args["calib"])
    tracks_path = os.path.join(cache, "tracks_id1.npz" if name == "base"
                               else f"tracks_fe_{name}.npz")
    if not os.path.exists(tracks_path):
        with ro.materialized_config(args["config"], VARIANTS[name]) as path:
            tracks = ro.run_frontend(cache, calib, path, progress_every=0)
        np.savez(tracks_path, **tracks)
    tracks = dict(np.load(tracks_path))
    sensors = dict(np.load(os.path.join(cache, "sensors.npz")))
    mocap = ev.load_mocap(sensors, "vrpn")
    sync = ev.cached_clock_offset(cache, sensors, mocap[0], mocap[2], "vrpn")
    params = args["params"]
    overrides = {k: v for k, v in params.items() if k != "camera_offset"}
    with ro.materialized_config(args["config"], overrides) as path:
        traj = ro.run_filter(sensors, tracks, calib, path,
                             params.get("camera_offset", ro.DEFAULT_CAMERA_OFFSET))
    if traj["diverged"]:
        return name, bag, dict(ok=False, reason="diverged")
    result = ev.evaluate(traj, mocap, sync["offset_ns"],
                         vertical_axis=ev.VERTICAL_AXIS["vrpn"])
    return name, bag, result[0] if isinstance(result, tuple) else result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bags", nargs="+", default=["hh1", "hh2"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--params", default="", help="filter params JSON (sweep best)")
    ap.add_argument("--calib", default=ro.DEFAULT_CALIB)
    ap.add_argument("--config", default=ro.DEFAULT_CONFIG)
    ap.add_argument("-j", "--jobs", type=int, default=3)
    ap.add_argument("-o", "--out", default="frontend_variants.json")
    args = ap.parse_args()
    params = {}
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
    shared = dict(calib=args.calib, config=args.config, params=params)
    tasks = [(n, b, shared) for n in args.variants for b in args.bags]
    rows = []
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for name, bag, m in pool.imap_unordered(run_task, tasks):
            rows.append(dict(variant=name, bag=bag, **m))
            if m.get("ok"):
                print(f"{name:14s} {bag}: ate {m['ate_rmse']:.3f}  vel_body {m['vel_body_rmse']:.3f}  "
                      f"hf {m['vel_body_hf_rmse']:.3f}  rot {m['rot_rmse']:.2f}  "
                      f"scale {m['sim3_scale']:.3f}", flush=True)
            else:
                print(f"{name:14s} {bag}: FAILED {m.get('reason')}", flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
