#!/usr/bin/env python3
"""Overlay several runs' body-frame velocity on mocap for one time window.

Body frame = the /voxl/raw_imu frame (x forward, y right, z down). The GT is
rotated into it with the mocap attitude, so yaw drift does not show up here.

  python3 compare_runs.py ~/.cache/echo-li/eval/hh1 \\
      --run "baseline ID2=runs/base_id2.npz" --run "tuned=runs/tuned.npz" \\
      --t0 60 --t1 80 -o compare.png
"""
import argparse
import json
import os

import matplotlib
import numpy as np

import evaluate as ev

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=NPZ")
    ap.add_argument("--mocap", default="vrpn", choices=["vrpn", "vp"])
    ap.add_argument("--t0", type=float, default=60.0, help="window start [s since first IMU]")
    ap.add_argument("--t1", type=float, default=80.0)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    sensors = np.load(os.path.join(args.cache, "sensors.npz"))
    mocap = ev.load_mocap(sensors, args.mocap)
    sync = ev.cached_clock_offset(args.cache, sensors, mocap[0], mocap[2], args.mocap)
    t_imu0 = int(sensors["imu_hdr"][0])

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.8]))
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green", "tab:purple"]
    table = []
    gt_drawn = False
    for n, spec in enumerate(args.run):
        label, _, path = spec.partition("=")
        m, d = ev.evaluate(np.load(path), mocap, sync["offset_ns"],
                           vertical_axis=ev.VERTICAL_AXIS[args.mocap])
        t = (d["t_ns"] - t_imu0) / 1e9
        w = (t >= args.t0) & (t <= args.t1)
        if not gt_drawn:
            for k in range(3):
                axes[k].plot(t[w], d["VBG"][w, k], "k-", lw=1.6, label="mocap")
            gt_drawn = True
        for k in range(3):
            axes[k].plot(t[w], d["VB"][w, k], "-", color=colors[n % len(colors)], lw=1.0,
                         label=label)
        axes[3].plot(t[w], np.linalg.norm(d["VB"][w] - d["VBG"][w], axis=1), "-",
                     color=colors[n % len(colors)], lw=1.0, label=label)
        table.append(f"{label}: ATE {m['ate_rmse']:.3f} m | v_body {m['vel_body_rmse']:.3f} m/s "
                     f"| v_body HF {m['vel_body_hf_rmse']:.3f} | att {m['rot_rmse']:.1f} deg "
                     f"| scale {m['sim3_scale']:.2f}")
    for k, name in enumerate(("forward", "right", "down")):
        axes[k].set_ylabel(f"v_{name} [m/s]")
    axes[3].set_ylabel("|v_body error| [m/s]")
    axes[3].set_xlabel("time since first IMU sample [s]")
    axes[0].legend(loc="upper right", ncol=len(args.run) + 1)
    fig.suptitle("\n".join(table), fontsize=10, family="monospace")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
