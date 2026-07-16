#!/usr/bin/env bash
set -euo pipefail

# Sweep Sparse3D's hard Mahalanobis reset threshold on MidAir exact-GT scoring.
#
# Purpose:
#   Find a less aggressive alternative to chi2=0.2, and test whether the Rust
#   R(age) model helps once the gross tail is gated.
#
# Run from the echo-li repo root after rebuilding/installing echo-li-python:
#   bash scripts/run_midair_sparse3d_gate_sweep.sh
#
# Optional overrides:
#   ROOT=/path/to/MidAir FRAMES=300 SCALE=0.5 bash scripts/run_midair_sparse3d_gate_sweep.sh

PY="${PY:-echo-li-python/venv/bin/python}"
ROOT="${ROOT:-/home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir}"
CONFIG="${CONFIG:-configs/diagnostics_midair_sparse3d.yaml}"
OUTDIR="${OUTDIR:-/tmp/midair_sparse3d_gate_sweep}"

SUBSET="${SUBSET:-VO_test}"
COND="${COND:-sunny}"
TRAJ="${TRAJ:-0}"
FRAMES="${FRAMES:-160}"
SCALE="${SCALE:-0.5}"
MIN_TRACK="${MIN_TRACK:-10}"
AGE_RATE="${AGE_RATE:-0.0058}"
GATES="${GATES:-0.2 0.5 1.0 2.0 4.0 6.0 9.21 13.82 0}"

mkdir -p "$OUTDIR"
COMBINED="$OUTDIR/combined.log"
: > "$COMBINED"

common=(
  echo-li-python/tests/diagnostics/midair_sparse3d_nees.py
  --config "$CONFIG"
  --root "$ROOT"
  --set "$SUBSET"
  --cond "$COND"
  --traj "$TRAJ"
  --frames "$FRAMES"
  --scale "$SCALE"
  --min-track "$MIN_TRACK"
  --measurements rudolf
  --a-init 100000000
  --b-init 0.00000001
)

run_case() {
  local name="$1"
  shift
  echo
  echo "===== $name ====="
  {
    echo "===== $name ====="
    "$PY" "${common[@]}" "$@"
  } 2>&1 | tee "$OUTDIR/$name.log" | tee -a "$COMBINED"
}

for gate in $GATES; do
  safe_gate="${gate//./p}"

  run_case "gate_${safe_gate}_no_rage" \
    --flow-age-rate-px-per-frame 0 \
    --mahalanobis-reset-chi2 "$gate"

  run_case "gate_${safe_gate}_rage" \
    --flow-age-rate-px-per-frame "$AGE_RATE" \
    --mahalanobis-reset-chi2 "$gate"
done

echo
echo "Logs written to $OUTDIR"
echo
echo "Quick summary:"
rg -n "=====|depth NEES-1D|3D NEES|3D split|XY marginal|pixel-plane|measurement drift|occluded obs|border-risk|NIS chi2|valid vis/int/lowmid|valid   drift<=3px|valid xy\\|z|valid full3|top    1%" "$COMBINED" || true
