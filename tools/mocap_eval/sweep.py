#!/usr/bin/env python3
"""Tune ECHO-LI filter parameters against mocap on cached bags.

A candidate is a set of config overrides plus the camera time offset. It is
replayed on every bag (run_offline.run_filter on cached frontend tracks,
~10 s per bag) and scored with evaluate.evaluate. The objective is the mean
over bags of  ate_rmse / ate_ref + vel_body_rmse / vel_body_ref, where the references
come from the baseline config. The baseline therefore scores 2.0, and pose
and velocity error carry equal weight.

Modes
  oat     one-at-a-time scan of every parameter around the baseline
  search  (1+lambda) evolution strategy in normalised parameter space,
          starting from --start (JSON of overrides) or the baseline
  eval    score one parameter set (--params JSON), e.g. on held-out bags

  PY=~/.cache/echo-li/ros2-humble-py3.10/venv/bin/python   # in ubuntu-22-04
  $PY sweep.py oat    --bags hh1 hh2 -o runs/oat.jsonl
  $PY sweep.py search --bags hh1 hh2 --start runs/oat_best.json -o runs/es.jsonl
  $PY sweep.py eval   --bags hh3 --params runs/es_best.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import yaml

import evaluate as ev
import run_offline as ro

EVAL_ROOT = os.path.expanduser("~/.cache/echo-li/eval")

# name: (kind, lo, hi) for log / lin / int, (kind, options) for choice.
SPACE = {
    "camera_offset": ("lin", -0.04, 0.04),
    "eqf.maxFeatures": ("int", 15, 120),
    "eqf.measurementNoise.feature": ("log", 0.5, 40.0),
    "eqf.measurementNoise.featureOutlierAbs": ("log", 1.0, 40.0),
    "eqf.velocityNoise.acc": ("log", 2e-4, 5e-2),
    "eqf.velocityNoise.gyr": ("log", 3e-5, 5e-3),
    "eqf.velocityNoise.accBias": ("log", 1e-5, 1e-2),
    "eqf.velocityNoise.gyrBias": ("log", 1e-6, 1e-3),
    "eqf.processVariance.velocity": ("log", 1e-6, 1e-1),
    "eqf.processVariance.position": ("log", 1e-9, 1e-3),
    "eqf.processVariance.attitude": ("log", 1e-8, 1e-3),
    "eqf.processVariance.point": ("log", 1e-6, 1e-2),
    "eqf.initialValue.sceneDepth": ("log", 1.0, 20.0),
    "eqf.initialVariance.point": ("log", 0.1, 200.0),
    "eqf.settings.useMedianDepth": ("choice", [True, False]),
    "eqf.settings.coordinateChoice": ("choice", ["Euclidean", "InvDepth", "Normal"]),
}

# Filled in main() before the worker pool forks.
BAGS = {}
BASE_CONFIG = ro.DEFAULT_CONFIG
CALIB = None


def config_value(cfg, key):
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def baseline_params(base_config):
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    params = {k: config_value(cfg, k) for k in SPACE if k != "camera_offset"}
    params["camera_offset"] = ro.DEFAULT_CAMERA_OFFSET
    return params


def load_bag(name, tracks_file, mocap_key):
    cache = os.path.join(EVAL_ROOT, name)
    sensors = dict(np.load(os.path.join(cache, "sensors.npz")))
    tracks = dict(np.load(os.path.join(cache, tracks_file)))
    mocap = ev.load_mocap(sensors, mocap_key)
    sync = ev.cached_clock_offset(cache, sensors, mocap[0], mocap[2], mocap_key)
    return dict(sensors=sensors, tracks=tracks, mocap=mocap, offset_ns=sync["offset_ns"],
                vertical_axis=ev.VERTICAL_AXIS[mocap_key])


def run_one(task):
    params, bag = task
    b = BAGS[bag]
    overrides = {k: v for k, v in params.items() if k != "camera_offset"}
    started = time.monotonic()
    try:
        with ro.materialized_config(BASE_CONFIG, overrides) as path:
            traj = ro.run_filter(b["sensors"], b["tracks"], CALIB, path,
                                 params.get("camera_offset", ro.DEFAULT_CAMERA_OFFSET))
        if traj["diverged"]:
            metrics = dict(ok=False, reason="diverged")
        else:
            result = ev.evaluate(traj, b["mocap"], b["offset_ns"],
                                 vertical_axis=b["vertical_axis"])
            metrics = result[0] if isinstance(result, tuple) else result
    except Exception as exc:  # a bad config must not kill the sweep
        metrics = dict(ok=False, reason=repr(exc))
    metrics["runtime_s"] = time.monotonic() - started
    return metrics


def objective(per_bag, refs):
    if not all(m.get("ok") for m in per_bag.values()):
        return math.inf
    return float(np.mean([m["ate_rmse"] / refs[b]["ate_rmse"]
                          + m["vel_body_rmse"] / refs[b]["vel_body_rmse"]
                          for b, m in per_bag.items()]))


class Runner:
    KEEP = ("ok", "reason", "ate_rmse", "vel_rmse", "vel_hf_rmse", "hf_ratio", "rot_rmse",
            "rpe1", "sim3_scale", "ate_vert_rmse", "ate_horiz_rmse", "vel_vert_rmse",
            "vel_horiz_rmse", "vel_body_rmse", "vel_body_hf_rmse", "runtime_s")

    def __init__(self, pool, bags, log_path):
        self.pool, self.bags, self.log_path = pool, bags, log_path
        self.refs = None

    def run(self, candidates, tag):
        tasks = [(c, b) for c in candidates for b in self.bags]
        flat = self.pool.map(run_one, tasks, chunksize=1)
        per = [{b: flat[i * len(self.bags) + j] for j, b in enumerate(self.bags)}
               for i in range(len(candidates))]
        if self.refs is None:
            self.refs = per[0]
        scores = [objective(p, self.refs) for p in per]
        with open(self.log_path, "a") as f:
            for c, p, s in zip(candidates, per, scores):
                f.write(json.dumps(dict(tag=tag, score=s, params=c, bags={
                    b: {k: m[k] for k in self.KEEP if k in m} for b, m in p.items()}),
                    default=str) + "\n")
        return scores, per


def oat_values(name, base):
    spec = SPACE[name]
    kind = spec[0]
    if kind == "choice":
        return [v for v in spec[1] if v != base]
    lo, hi = spec[1], spec[2]
    if kind == "lin":
        vals = np.linspace(lo, hi, 9)
    elif kind == "int":
        vals = [int(round(base * f)) for f in (0.5, 0.75, 1.5, 2.5)]
    else:
        vals = [base * f for f in (0.1, 0.3, 3.0, 10.0)]
    return sorted({float(np.clip(v, lo, hi)) if kind != "int" else int(np.clip(v, lo, hi))
                   for v in vals} - {base})


def encode(params):
    u = {}
    for name, spec in SPACE.items():
        if spec[0] == "choice":
            continue
        v, lo, hi = params[name], spec[1], spec[2]
        if spec[0] == "log":
            u[name] = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            u[name] = (v - lo) / (hi - lo)
    return u


def decode(u, template):
    params = dict(template)
    for name, x in u.items():
        kind, lo, hi = SPACE[name][:3]
        x = min(max(x, 0.0), 1.0)
        if kind == "log":
            params[name] = 10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo)))
        elif kind == "int":
            params[name] = int(round(lo + x * (hi - lo)))
        else:
            params[name] = lo + x * (hi - lo)
    return params


def main():
    global BAGS, BASE_CONFIG, CALIB
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["oat", "search", "eval"])
    ap.add_argument("--bags", nargs="+", default=["hh1", "hh2"])
    ap.add_argument("--tracks", default="tracks_id1.npz")
    ap.add_argument("--mocap", default="vrpn", choices=["vrpn", "vp"])
    ap.add_argument("--calib", default=ro.DEFAULT_CALIB)
    ap.add_argument("--config", default=ro.DEFAULT_CONFIG)
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("-o", "--out", default="sweep.jsonl")
    ap.add_argument("--start", default="", help="JSON of params to start the search from")
    ap.add_argument("--params", default="", help="JSON of params for eval mode")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--lam", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    BASE_CONFIG = args.config
    CALIB = ro.load_calibration(args.calib)
    BAGS = {b: load_bag(b, args.tracks, args.mocap) for b in args.bags}
    base = baseline_params(args.config)
    best_path = os.path.splitext(args.out)[0] + "_best.json"

    with mp.get_context("fork").Pool(args.jobs) as pool:
        runner = Runner(pool, args.bags, args.out)
        scores, per = runner.run([base], "baseline")
        print(f"baseline score {scores[0]:.3f} " + " ".join(
            f"{b}: ate={m['ate_rmse']:.3f} vel={m['vel_rmse']:.3f}" for b, m in per[0].items()),
            flush=True)

        if args.mode == "eval":
            with open(args.params) as f:
                params = dict(base, **json.load(f))
            scores, per = runner.run([params], "eval")
            print(json.dumps(dict(score=scores[0], bags=per[0]), indent=1, default=str))
            return

        if args.mode == "oat":
            candidates = [dict(base, **{name: v}) for name in SPACE
                          for v in oat_values(name, base[name])]
            scores, _ = runner.run(candidates, "oat")
            best = dict(base)
            for name in SPACE:  # greedy: combine each parameter's best OAT value
                own = [(s, c[name]) for s, c in zip(scores, candidates)
                       if c[name] != base[name] and all(c[k] == base[k] for k in SPACE if k != name)]
                s, v = min(own, default=(math.inf, None))
                print(f"{name:42s} base={base[name]!s:>10.10} best={v!s:>10.10} score={s:.3f}")
                if s < 2.0 - 0.02:
                    best[name] = v
            scores, per = runner.run([best], "oat_combined")
            print(f"combined OAT improvements: score {scores[0]:.3f}")
            with open(best_path, "w") as f:
                json.dump(best, f, indent=1)
            return

        rng = random.Random(args.seed)
        incumbent = dict(base)
        if args.start:
            with open(args.start) as f:
                incumbent.update(json.load(f))
        scores, _ = runner.run([incumbent], "start")
        best_score, sigma = scores[0], args.sigma
        choices = [n for n, s in SPACE.items() if s[0] == "choice"]
        for gen in range(args.generations):
            u0 = encode(incumbent)
            candidates = []
            for _ in range(args.lam):
                dims = [n for n in u0 if rng.random() < 0.35] or [rng.choice(list(u0))]
                u = dict(u0, **{n: u0[n] + rng.gauss(0.0, sigma) for n in dims})
                c = decode(u, incumbent)
                for n in choices:
                    if rng.random() < 0.1:
                        c[n] = rng.choice(SPACE[n][1])
                candidates.append(c)
            scores, _ = runner.run(candidates, f"gen{gen}")
            i = int(np.argmin(scores))
            improved = scores[i] < best_score
            if improved:
                best_score, incumbent = scores[i], candidates[i]
                with open(best_path, "w") as f:
                    json.dump(incumbent, f, indent=1)
            sigma *= 1.25 if improved else 0.8
            print(f"gen {gen}: best {best_score:.3f} (gen min {scores[i]:.3f}) sigma {sigma:.3f}",
                  flush=True)
        print(json.dumps(incumbent, indent=1))


if __name__ == "__main__":
    main()
