"""Provenance sidecar for diagnostic runs.

Writes ``<artifact>.manifest.json`` next to every output (npz / video / figure) so
"which config produced this?" is always answerable — even when the run's stdout was
filtered by a `grep | tail`. This is a MECHANICAL safeguard against config-provenance
loss (a recurring failure: comparing runs across silently-different configs, and
cross-applying one modality's tuning to another). Record, don't rely on memory.

Usage:
    import run_manifest
    run_manifest.save_run_manifest(args.save_npz, args.config,
                                   extra={"stereo": args.stereo, "traj": args.traj})
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

# Distinctive tuning knobs whose value (with full dotted path) explains most
# behaviour differences between runs. Full path disambiguates e.g.
# initialVariance.biasAcc vs processVariance.biasAcc.
_WATCH = {
    "biasAcc", "biasGyr", "acc", "gyr", "accBias", "gyrBias", "sigma_pixel",
    "stereoMeasurement", "rangeGateChi2", "maxFeatures", "point", "sceneDepth",
    "coordinateChoice", "useEquivariantOutput", "useMedianDepth",
}


def _collect(node, path=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, (dict, list)):
                out.update(_collect(v, p))
            elif k in _WATCH:
                out[p] = v
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_collect(v, f"{path}[{i}]"))
    return out


def _git_rev():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return ""


def save_run_manifest(artifact_path, config_path, extra=None):
    """Write <artifact>.manifest.json capturing config path+hash, command, git rev,
    and the key config params. Returns the manifest path. Best-effort: never raises."""
    try:
        artifact = Path(artifact_path)
        cfg = Path(config_path)
        cfg_bytes = cfg.read_bytes() if cfg.exists() else b""
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16] if cfg_bytes else "MISSING"
        try:
            import yaml
            cfg_dict = yaml.safe_load(cfg_bytes) or {}
        except Exception:
            cfg_dict = {}
        key_params = _collect(cfg_dict)
        man = {
            "artifact": str(artifact),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_path": str(cfg.resolve()) if cfg.exists() else str(cfg),
            "config_sha256_16": cfg_hash,
            "command": " ".join(sys.argv),
            "git_rev": _git_rev(),
            "key_params": key_params,
        }
        if extra:
            man["extra"] = extra
        mpath = artifact.with_name(artifact.name + ".manifest.json")
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(man, indent=2, default=str))
        pick = lambda suf: next((v for k, v in key_params.items() if k.endswith(suf)), "?")
        print(f"[manifest] {mpath} cfg={cfg.name}#{cfg_hash} git={man['git_rev']} "
              f"| initVar.biasAcc={key_params.get('EqF.initialVariance.biasAcc', pick('initialVariance.biasAcc'))} "
              f"velNoise.gyr={pick('velocityNoise.gyr')} sigma_pixel={pick('sigma_pixel')} "
              f"stereoMeas={pick('stereoMeasurement')}")
        return mpath
    except Exception as e:  # provenance must never break a run
        print(f"[manifest] WARN could not write manifest: {e}")
        return None
