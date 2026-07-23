#!/usr/bin/env bash
set -euo pipefail

# Test whether the marginalized per-track pixel/bearing bias model transfers
# across MidAir conditions and trajectories.
#
# Run from the echo-li repo root after rebuilding/installing echo-li-python:
#   bash scripts/run_midair_track_bias_suite.sh
#
# Useful overrides:
#   ROOT=/path/to/MidAir FRAMES=300 SCALE=0.5 bash scripts/run_midair_track_bias_suite.sh
#   SLICES="sunny:0 foggy:1000 sunset:2000" bash scripts/run_midair_track_bias_suite.sh
#   RUN_TEMPORAL=0 BIAS_SIGMAS="0 0.5 1.0" bash scripts/run_midair_track_bias_suite.sh
#   RUN_VIDEO=1 VIDEO_ONLY=1 bash scripts/run_midair_track_bias_suite.sh
#   JOBS=3 bash scripts/run_midair_track_bias_suite.sh

PY="${PY:-echo-li-python/venv/bin/python}"
ROOT="${ROOT:-/home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir}"
CONFIG="${CONFIG:-configs/diagnostics_midair_sparse3d.yaml}"
OUTDIR="${OUTDIR:-/tmp/midair_track_bias_suite_$(date +%Y%m%d_%H%M%S)}"

SUBSET="${SUBSET:-VO_test}"
SLICES="${SLICES:-sunny:0 foggy:1000 sunset:2000}"
FRAMES="${FRAMES:-300}"
SCALE="${SCALE:-0.5}"
MIN_TRACK="${MIN_TRACK:-10}"
GATE="${GATE:-0.2}"
SIGMA_PIXEL="${SIGMA_PIXEL:-0.42}"
BIAS_SIGMAS="${BIAS_SIGMAS:-0 0.25 0.5 1.0}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
MAX_LAG="${MAX_LAG:-60}"
CORE_PX="${CORE_PX:-3}"
RUN_VIDEO="${RUN_VIDEO:-0}"
VIDEO_ONLY="${VIDEO_ONLY:-1}"
VIDEO_FPS="${VIDEO_FPS:-15}"
VIDEO_DRAW_MAX="${VIDEO_DRAW_MAX:-160}"
JOBS="${JOBS:-1}"

mkdir -p "$OUTDIR"
COMBINED="$OUTDIR/combined.log"
SUMMARY="$OUTDIR/summary.log"
: > "$COMBINED"
: > "$SUMMARY"

common_sparse=(
  echo-li-python/tests/diagnostics/midair_sparse3d_nees.py
  --config "$CONFIG"
  --root "$ROOT"
  --set "$SUBSET"
  --frames "$FRAMES"
  --scale "$SCALE"
  --min-track "$MIN_TRACK"
  --measurements rudolf
  --a-init 100000000
  --b-init 0.00000001
  --flow-age-rate-px-per-frame 0
  --mahalanobis-reset-chi2 "$GATE"
  --fisher-sigma-px "$SIGMA_PIXEL"
)

common_temporal=(
  echo-li-python/tests/diagnostics/midair_flow_temporal_correlation.py
  --config "$CONFIG"
  --root "$ROOT"
  --set "$SUBSET"
  --frames "$FRAMES"
  --scale "$SCALE"
  --max-lag "$MAX_LAG"
  --core-px "$CORE_PX"
)

common_video=(
  echo-li-python/tests/diagnostics/midair_sparse3d_nees.py
  --config "$CONFIG"
  --root "$ROOT"
  --set "$SUBSET"
  --frames "$FRAMES"
  --scale "$SCALE"
  --min-track "$MIN_TRACK"
  --measurements rudolf
  --a-init 100000000
  --b-init 0.00000001
  --flow-age-rate-px-per-frame 0
  --mahalanobis-reset-chi2 "$GATE"
  --fps "$VIDEO_FPS"
  --draw-max "$VIDEO_DRAW_MAX"
)

CASE_NAMES=()
RUNNING=0
FAILED=0

wait_one() {
  if (( RUNNING <= 0 )); then
    return
  fi
  if ! wait -n; then
    FAILED=1
  fi
  RUNNING=$((RUNNING - 1))
}

wait_all() {
  while (( RUNNING > 0 )); do
    wait_one
  done
}

run_case() {
  local name="$1"
  shift
  local log="$OUTDIR/$name.log"
  local status="$OUTDIR/$name.status"
  CASE_NAMES+=("$name")
  echo
  echo "===== launch $name ====="
  if (( JOBS > 1 )); then
    (
      set +e
      echo "===== $name ====="
      "$@"
      rc=$?
      echo "$rc" > "$status"
      exit "$rc"
    ) > "$log" 2>&1 &
    RUNNING=$((RUNNING + 1))
    while (( RUNNING >= JOBS )); do
      wait_one
    done
  else
    set +e
    (
      set +e
      echo "===== $name ====="
      "$@"
      rc=$?
      echo "$rc" > "$status"
      exit "$rc"
    ) > "$log" 2>&1
    rc=$?
    set -e
    cat "$log"
    cat "$log" >> "$COMBINED"
    if (( rc != 0 )); then
      FAILED=1
    fi
  fi
}

summarize_case() {
  local name="$1"
  local log="$OUTDIR/$name.log"
  {
    echo "===== $name ====="
    rg -n \
      "temporal-correlation input|radial \\|error\\||outlier >|track count usable|series length|tau_int raw|N_eff/N|Sparse3D exact-GT|depth NEES-1D|3D NEES|3D split|diagnostic Fisher|iid Fisher|bias Fisher|measurement drift|valid full3|valid iid3|valid bias3|top    1%" \
      "$log" || true
  } | tee -a "$SUMMARY" >/dev/null
}

echo "Writing logs to $OUTDIR"
echo "slices: $SLICES"
echo "bias sigmas: $BIAS_SIGMAS"
echo "frames: $FRAMES  scale: $SCALE  gate: $GATE  sigma_pixel: $SIGMA_PIXEL"
echo "video: RUN_VIDEO=$RUN_VIDEO VIDEO_ONLY=$VIDEO_ONLY fps=$VIDEO_FPS draw_max=$VIDEO_DRAW_MAX"
echo "jobs: $JOBS"

for slice in $SLICES; do
  if [[ "$slice" != *:* ]]; then
    echo "Bad slice '$slice'. Expected condition:trajectory, e.g. sunny:0" >&2
    exit 2
  fi
  cond="${slice%%:*}"
  traj="${slice#*:}"
  tag="${SUBSET}_${cond}_traj${traj}"

  if [[ "$RUN_VIDEO" != "0" ]]; then
    name="${tag}_video"
    video_args=(--video --video-out "$OUTDIR/${tag}_measurements.mp4")
    if [[ "$VIDEO_ONLY" != "0" ]]; then
      video_args+=(--video-only)
    fi
    run_case "$name" \
      "$PY" "${common_video[@]}" --cond "$cond" --traj "$traj" "${video_args[@]}"
  fi

  if [[ "$RUN_TEMPORAL" != "0" ]]; then
    name="${tag}_temporal"
    run_case "$name" \
      "$PY" "${common_temporal[@]}" --cond "$cond" --traj "$traj"
  fi

  for bias in $BIAS_SIGMAS; do
    safe_bias="${bias//./p}"
    name="${tag}_bias_${safe_bias}"
    run_case "$name" \
      "$PY" "${common_sparse[@]}" --cond "$cond" --traj "$traj" --bias-sigma-px "$bias"
  done
done

wait_all

if (( JOBS > 1 )); then
  : > "$COMBINED"
  for name in "${CASE_NAMES[@]}"; do
    cat "$OUTDIR/$name.log" >> "$COMBINED"
  done
fi

for name in "${CASE_NAMES[@]}"; do
  summarize_case "$name"
done

echo
echo "Logs written to $OUTDIR"
echo "Combined log: $COMBINED"
echo "Summary: $SUMMARY"
if (( FAILED != 0 )); then
  echo "One or more cases failed. Check *.status and per-case logs in $OUTDIR." >&2
fi
echo
echo "Quick summary:"
cat "$SUMMARY"

if (( FAILED != 0 )); then
  exit 1
fi
