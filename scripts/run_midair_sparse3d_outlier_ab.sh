#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Sparse3D flow-outlier A/B test on MidAir exact GT.
#
# The comparison answers two separate questions:
#   1. Does Sparse3D stay consistent when the measurements are exact?
#   2. Under Rudolf-V measurements, does white-only noise suffice, or do we need
#      explicit outlier handling/gating?
#
# Run from the echo-li repo root:
#   bash scripts/run_midair_sparse3d_outlier_ab.sh
#
# Optional overrides:
#   ROOT=/path/to/MidAir FRAMES=300 SCALE=0.5 bash scripts/run_midair_sparse3d_outlier_ab.sh

PY="${PY:-echo-li-python/venv/bin/python}"
ROOT="${ROOT:-/home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir}"
CONFIG="${CONFIG:-configs/diagnostics_midair_sparse3d.yaml}"
OUTDIR="${OUTDIR:-/tmp/midair_sparse3d_outlier_ab}"

SUBSET="${SUBSET:-VO_test}"
COND="${COND:-sunny}"
TRAJ="${TRAJ:-0}"
FRAMES="${FRAMES:-160}"
SCALE="${SCALE:-0.5}"
MIN_TRACK="${MIN_TRACK:-10}"

mkdir -p "$OUTDIR"

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
)

run_case() {
  local name="$1"
  shift
  echo
  echo "===== $name ====="
  "$PY" "${common[@]}" "$@" 2>&1 | tee "$OUTDIR/$name.log"
}

run_case exact_white_control \
  --measurements exact \
  --a-init 100000000 --b-init 0.00000001 \
  --mahalanobis-reset-chi2 0

run_case rudolf_white_no_gate \
  --measurements rudolf \
  --a-init 100000000 --b-init 0.00000001 \
  --flow-age-rate-px-per-frame 0 \
  --mahalanobis-reset-chi2 0

run_case rudolf_rage_no_gate \
  --measurements rudolf \
  --a-init 100000000 --b-init 0.00000001 \
  --flow-age-rate-px-per-frame 0.0058 \
  --mahalanobis-reset-chi2 0

run_case rudolf_hard_gate \
  --measurements rudolf \
  --a-init 100000000 --b-init 0.00000001 \
  --flow-age-rate-px-per-frame 0 \
  --mahalanobis-reset-chi2 0.2

run_case rudolf_rage_hard_gate \
  --measurements rudolf \
  --a-init 100000000 --b-init 0.00000001 \
  --flow-age-rate-px-per-frame 0.0058 \
  --mahalanobis-reset-chi2 0.2

run_case rudolf_gb_no_gate \
  --measurements rudolf \
  --a-init 10 --b-init 2 --ab-max 200 \
  --flow-age-rate-px-per-frame 0 \
  --mahalanobis-reset-chi2 0

run_case rudolf_gb_hard_gate \
  --measurements rudolf \
  --a-init 10 --b-init 2 --ab-max 200 \
  --flow-age-rate-px-per-frame 0 \
  --mahalanobis-reset-chi2 0.2

echo
echo "Logs written to $OUTDIR"
