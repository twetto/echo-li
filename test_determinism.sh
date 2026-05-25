#!/usr/bin/env bash
# Run echo-li-cli twice with determinism tracing and compare the det lines.
set -euo pipefail

DATASET="${1:-${ECHO_LI_DATASET:-$HOME/Downloads/vicon_room1/vicon_room1/V1_03_difficult/V1_03_difficult}}"
CONFIG="${ECHO_LI_CONFIG:-configs/eqvio_euroc_euclid.yaml}"
OUT_DIR="${ECHO_LI_DET_OUT:-determinism_runs}"
FEATURES="${ECHO_LI_FEATURES:-rerun}"
CARGO_ARGS=()
if [[ -n "${ECHO_LI_RUDOLF_PATH:-}" ]]; then
    CARGO_ARGS+=(
        --config
        "patch.\"https://github.com/twetto/Rudolf-V\".rudolf-v.path=\"$ECHO_LI_RUDOLF_PATH\""
    )
fi

PATCH_DEPTH_ARGS=()
if [[ "${ECHO_LI_NO_PATCH_DEPTH:-0}" == "1" ]]; then
    PATCH_DEPTH_ARGS+=(--no-patch-depth)
fi

VIS_ARGS=()
if [[ "${ECHO_LI_VIS:-0}" == "1" ]]; then
    VIS_ARGS+=(--vis)
fi

mkdir -p "$OUT_DIR"

run_once() {
    local label="$1"
    local log="$OUT_DIR/${label}.log"
    local det="$OUT_DIR/${label}.det"

    echo "=== $label ==="
    echo "log: $log"

    ECHO_LI_TRACE_DETERMINISM=1 \
        cargo "${CARGO_ARGS[@]}" run -p echo-li-cli --release --features "$FEATURES" -- \
        -d "$DATASET" \
        -c "$CONFIG" \
        "${VIS_ARGS[@]}" \
        "${PATCH_DEPTH_ARGS[@]}" \
        >"$log" 2>&1

    grep '^det ' "$log" >"$det"
    echo "det lines: $(wc -l <"$det")"
}

run_once trace1
run_once trace2

if diff -u "$OUT_DIR/trace1.det" "$OUT_DIR/trace2.det" >"$OUT_DIR/diff.det"; then
    echo "PASS: deterministic"
    echo "det trace: $OUT_DIR/trace1.det"
else
    echo "FAIL: first determinism diff:"
    sed -n '1,80p' "$OUT_DIR/diff.det"
    echo
    echo "full diff: $OUT_DIR/diff.det"
    exit 1
fi
