#!/usr/bin/env bash
set -euo pipefail

# Reproduce the MidAir structure-tensor vs SSD-landscape diagnostic figures.
#
# The diagnostic checks whether the local SSD basin shape around exact GT
# correspondence agrees with the KLT windowed structure tensor.  This is the
# visual/quantitative support for using tensor eigenvalues/eigenvectors as the
# per-measurement uncertainty-shape cue.
#
# Run from the echo-li repo root after installing echo-li-python:
#   bash scripts/run_midair_landscape_structure_tensor.sh
#
# Useful overrides:
#   ROOT=/path/to/MidAir bash scripts/run_midair_landscape_structure_tensor.sh
#   SLICES="sunny:0:120 foggy:1000:120 sunset:2000:120" bash scripts/run_midair_landscape_structure_tensor.sh
#   SUMMARY_ONLY=1 bash scripts/run_midair_landscape_structure_tensor.sh
#   EXAMPLES=7 GRID_HALF=6 GRID_STEP=0.25 bash scripts/run_midair_landscape_structure_tensor.sh

PY="${PY:-echo-li-python/venv/bin/python}"
ROOT="${ROOT:-/home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir}"
OUTDIR="${OUTDIR:-/tmp/midair_landscape_structure_tensor_$(date +%Y%m%d_%H%M%S)}"

SUBSET="${SUBSET:-VO_test}"
SLICES="${SLICES:-sunny:0:120 foggy:1000:120 sunset:2000:120}"
DT="${DT:-1}"
SCALE="${SCALE:-0.5}"
MAX_FEATURES="${MAX_FEATURES:-700}"
QUALITY="${QUALITY:-0.01}"
BORDER="${BORDER:-24}"
TENSOR_RADIUS="${TENSOR_RADIUS:-7}"
GRID_HALF="${GRID_HALF:-5}"
GRID_STEP="${GRID_STEP:-0.25}"
CONTEXT="${CONTEXT:-72}"
EXAMPLES="${EXAMPLES:-5}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
NO_HISTEQ="${NO_HISTEQ:-0}"

mkdir -p "$OUTDIR"
COMBINED="$OUTDIR/combined.log"
SUMMARY="$OUTDIR/summary.log"
: > "$COMBINED"
: > "$SUMMARY"

echo "Writing landscape/tensor artifacts to $OUTDIR"
echo "slices: $SLICES"
echo "dt: $DT  scale: $SCALE  tensor_radius: $TENSOR_RADIUS  grid: +/-$GRID_HALF step $GRID_STEP"
echo "summary_only: $SUMMARY_ONLY  no_histeq: $NO_HISTEQ"

common=(
  echo-li-python/tests/diagnostics/midair_landscape_structure_tensor.py
  --root "$ROOT"
  --set "$SUBSET"
  --dt "$DT"
  --scale "$SCALE"
  --max-features "$MAX_FEATURES"
  --quality "$QUALITY"
  --border "$BORDER"
  --tensor-radius "$TENSOR_RADIUS"
  --grid-half "$GRID_HALF"
  --grid-step "$GRID_STEP"
  --context "$CONTEXT"
  --examples "$EXAMPLES"
)

if [[ "$SUMMARY_ONLY" != "0" ]]; then
  common+=(--summary-only)
fi
if [[ "$NO_HISTEQ" != "0" ]]; then
  common+=(--no-histeq)
fi

for slice in $SLICES; do
  IFS=: read -r cond traj frame extra <<< "$slice"
  if [[ -z "${cond:-}" || -z "${traj:-}" || -z "${frame:-}" || -n "${extra:-}" ]]; then
    echo "Bad slice '$slice'. Expected condition:trajectory:frame, e.g. sunny:0:120" >&2
    exit 2
  fi

  tag="${SUBSET}_${cond}_traj${traj}_frame${frame}_dt${DT}"
  log="$OUTDIR/${tag}.log"
  png="$OUTDIR/${tag}.png"
  npz="$OUTDIR/${tag}.npz"

  echo
  echo "===== $tag ====="
  cmd=(
    "$PY" "${common[@]}"
    --cond "$cond"
    --traj "$traj"
    --frame "$frame"
  )
  if [[ "$SUMMARY_ONLY" == "0" ]]; then
    cmd+=(--out "$png" --save-npz "$npz")
  fi

  "${cmd[@]}" 2>&1 | tee "$log"
  cat "$log" >> "$COMBINED"
  {
    echo "===== $tag ====="
    rg -n "usable cases|columns:|^p[0-9][0-9]:|median weak-axis|corr log|wrote " "$log" || true
  } >> "$SUMMARY"
done

echo
echo "Artifacts: $OUTDIR"
echo "Combined log: $COMBINED"
echo "Summary: $SUMMARY"
