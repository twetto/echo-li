#!/usr/bin/env bash
set -euo pipefail

# Reproduce the MidAir baseline / Sparse3D-seeded / stereo-seeded MidAir test.
#
# Defaults run the full sunny VO_test session: every trajectory under
# color_left/trajectory_*, all frames.
#
#   bash scripts/run_midair_sparse3d_prior_ab.sh
#   ROOT=/path/to/MidAir VIDEO=1 SPARSE_POSE=gt bash scripts/run_midair_sparse3d_prior_ab.sh
#   MODES=stereo TRAJ=2 FRAMES=1500 bash scripts/run_midair_sparse3d_prior_ab.sh
#   MODES=baseline,stereo TRAJ=2 FRAMES=1500 bash scripts/run_midair_sparse3d_prior_ab.sh
#   TRAJ=2 FRAMES=800 bash scripts/run_midair_sparse3d_prior_ab.sh

ROOT="${ROOT:-$HOME/Server250/18TB/datasets/dataset_MidAir_stereo_download/MidAir}"
SET="${SET:-VO_test}"
COND="${COND:-sunny}"
TRAJ="${TRAJ:-all}"
START="${START:-0}"
FRAMES="${FRAMES:-0}"
SCALE="${SCALE:-0.5}"
MIDAIR_CONFIG="${MIDAIR_CONFIG:-configs/eqvio_midair.yaml}"
STEREO_CONFIG="${STEREO_CONFIG:-configs/eqvio_midair_stereo.yaml}"
CONFIG="${CONFIG:-}"
SPARSE_POSE="${SPARSE_POSE:-vio}"
MODES="${MODES:-all}"
SPARSE_MAX_FEATURES="${SPARSE_MAX_FEATURES:-300}"
EQF_MAX_OBS="${EQF_MAX_OBS:-40}"
EQF_SELECTION="${EQF_SELECTION:-prior_uncertainty}"
SPARSE_MIN_TRACK="${SPARSE_MIN_TRACK:-5}"
MAX_REL_SIGMA="${MAX_REL_SIGMA:-2.0}"
STEREO_BASELINE_M="${STEREO_BASELINE_M:-1.0}"
STEREO_SIGMA_PIXEL_SCALE="${STEREO_SIGMA_PIXEL_SCALE:-20.0}"
GYRO_FRAME="${GYRO_FRAME:-repaired_gt}"
OUT_DIR="${OUT_DIR:-/tmp/midair_sparse3d_prior_ab_${COND}_whole_session}"
STATE_PLOT_OUT="${STATE_PLOT_OUT:-}"
VIDEO="${VIDEO:-0}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-$OUT_DIR/videos}"
VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
PREFETCH_IMAGES="${PREFETCH_IMAGES:-16}"
NO_PROGRESS="${NO_PROGRESS:-0}"
VIS_MIN_RANGE_M="${VIS_MIN_RANGE_M:-1.0}"
VIS_MAX_RANGE_M="${VIS_MAX_RANGE_M:-150.0}"
VIS_EQF_RADIUS="${VIS_EQF_RADIUS:-2}"
VIS_STEREO_RADIUS="${VIS_STEREO_RADIUS:-5}"
VIS_STEREO_THICKNESS="${VIS_STEREO_THICKNESS:-1}"
PY="${PY:-echo-li-python/venv/bin/python}"

if [[ "$TRAJ" == "all" ]]; then
  mapfile -t traj_dirs < <(find "$ROOT/$SET/$COND/color_left" -maxdepth 1 -type d -name 'trajectory_*' | sort)
  if [[ "${#traj_dirs[@]}" -eq 0 ]]; then
    echo "no trajectories found under $ROOT/$SET/$COND/color_left" >&2
    exit 1
  fi
  trajs=()
  for d in "${traj_dirs[@]}"; do
    name="$(basename "$d")"
    trajs+=("${name#trajectory_}")
  done
else
  trajs=("$TRAJ")
fi

mkdir -p "$OUT_DIR"

case "$MODES" in
  all) run_modes=(baseline sparse3d stereo) ;;
  baseline) run_modes=(baseline) ;;
  sparse3d|sparse3d_seeded) run_modes=(sparse3d) ;;
  stereo|stereo_seeded) run_modes=(stereo) ;;
  *)
    IFS=',' read -ra run_modes <<< "$MODES"
    ;;
esac

config_for_mode() {
  local mode="$1"
  if [[ -n "$CONFIG" ]]; then
    printf '%s\n' "$CONFIG"
  elif [[ "$mode" == "stereo" || "$mode" == "stereo_seeded" ]]; then
    printf '%s\n' "$STEREO_CONFIG"
  else
    printf '%s\n' "$MIDAIR_CONFIG"
  fi
}

for traj in "${trajs[@]}"; do
  traj_num="$((10#$traj))"
  for mode in "${run_modes[@]}"; do
    case "$mode" in
      baseline) mode_out="baseline" ;;
      sparse3d|sparse3d_seeded) mode_out="sparse3d" ;;
      stereo|stereo_seeded) mode_out="stereo" ;;
      sparse_defer) mode_out="sparse_defer" ;;
      *) mode_out="$mode" ;;
    esac
    run_config="$(config_for_mode "$mode_out")"
    out="$OUT_DIR/${SET}_${COND}_trajectory_${traj}_${mode_out}.npz"
    extra=()
    if [[ -n "$STATE_PLOT_OUT" ]]; then
      if [[ "$TRAJ" == "all" ]]; then
        extra+=(--state-plot-out "$STATE_PLOT_OUT/trajectory_${traj}_${mode_out}_state_monitor.png")
      else
        extra+=(--state-plot-out "$STATE_PLOT_OUT")
      fi
    fi
    if [[ "$NO_PROGRESS" != "0" ]]; then
      extra+=(--no-progress)
    fi
    if [[ "$VIDEO" != "0" ]]; then
      extra+=(--video-out-dir "$VIDEO_OUT_DIR/trajectory_${traj}" --video-stride "$VIDEO_STRIDE")
    fi
    echo "=== $SET/$COND/trajectory_${traj} frames=$FRAMES mode=$mode_out config=$run_config sparse_pose=$SPARSE_POSE gyro=$GYRO_FRAME ==="
    "$PY" echo-li-python/tests/diagnostics/midair_vio_sparse3d_prior_ab.py \
      --root "$ROOT" \
      --set "$SET" \
      --cond "$COND" \
      --traj "$traj_num" \
      --start "$START" \
      --frames "$FRAMES" \
      --scale "$SCALE" \
      --config "$run_config" \
      --modes "$mode_out" \
      --sparse-pose "$SPARSE_POSE" \
      --sparse-max-features "$SPARSE_MAX_FEATURES" \
      --eqf-max-obs "$EQF_MAX_OBS" \
      --eqf-selection "$EQF_SELECTION" \
      --sparse-min-track "$SPARSE_MIN_TRACK" \
      --max-rel-sigma "$MAX_REL_SIGMA" \
      --stereo-baseline-m "$STEREO_BASELINE_M" \
      --stereo-sigma-pixel-scale "$STEREO_SIGMA_PIXEL_SCALE" \
      --gyro-frame "$GYRO_FRAME" \
      --prefetch-images "$PREFETCH_IMAGES" \
      --vis-min-range-m "$VIS_MIN_RANGE_M" \
      --vis-max-range-m "$VIS_MAX_RANGE_M" \
      --vis-eqf-radius "$VIS_EQF_RADIUS" \
      --vis-stereo-radius "$VIS_STEREO_RADIUS" \
      --vis-stereo-thickness "$VIS_STEREO_THICKNESS" \
      --save-npz "$out" \
      "${extra[@]}"
  done
done
