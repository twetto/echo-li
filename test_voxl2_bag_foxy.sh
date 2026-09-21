#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")
INSIDE_FOXY=0

if [[ "${1:-}" == "--inside-foxy" ]]; then
    INSIDE_FOXY=1
    shift
fi

if [[ "$INSIDE_FOXY" -ne 1 && "${ROS_DISTRO:-}" != "foxy" ]]; then
    if ! command -v distrobox >/dev/null 2>&1; then
        echo "Run this script in ROS 2 Foxy, or install Distrobox." >&2
        exit 1
    fi
    exec distrobox enter ros2-foxy -- \
        bash "$SCRIPT_DIR/test_voxl2_bag_foxy.sh" \
        --inside-foxy "${ORIGINAL_ARGS[@]}"
fi

BAG_PATH="/home/twetto/Downloads/voxl_imu_camera_test_raw_imu_track_encoded_2"
INTERNAL_ID=2
DEPTH_BACKEND="dis"
DECODER="avdec_h265"
RATE="1.0"

usage() {
    cat <<'EOF'
Run an end-to-end Foxy ECHO-LI smoke test with a VOXL2 rosbag.

Usage:
  ./test_voxl2_bag_foxy.sh [options]

Options:
  --bag PATH             Bag directory.
  --internal-id {1|2}   VOXL2 factory calibration (default: 2).
  --depth-backend NAME   Depth estimation backend: dis or patch (default: dis).
  --decoder ELEMENT     GStreamer decoder (default: avdec_h265).
  --rate RATE            Bag playback rate (default: 1.0).
  -h, --help             Show this help.
EOF
}

need_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)
            need_value "$@"
            BAG_PATH="$2"
            shift 2
            ;;
        --internal-id)
            need_value "$@"
            INTERNAL_ID="$2"
            shift 2
            ;;
        --depth-backend)
            need_value "$@"
            DEPTH_BACKEND="$2"
            shift 2
            ;;
        --decoder)
            need_value "$@"
            DECODER="$2"
            shift 2
            ;;
        --rate)
            need_value "$@"
            RATE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$INTERNAL_ID" != "1" && "$INTERNAL_ID" != "2" ]]; then
    echo "--internal-id must be 1 or 2" >&2
    exit 2
fi
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--rate must be a positive number" >&2
    exit 2
fi
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Bag directory not found: $BAG_PATH" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
set -u

DECODER_WS="/home/twetto/voxl_h265_decoder_ws"
if [[ ! -d "$DECODER_WS/src/voxl_h265_decoder" ]]; then
    echo "Decoder workspace not found: $DECODER_WS" >&2
    exit 1
fi

echo "Building the Foxy H.265 decoder..."
  colcon --log-base "$DECODER_WS/log_foxy" build \
    --base-paths "$DECODER_WS/src" \
    --build-base "$DECODER_WS/build_foxy" \
    --install-base "$DECODER_WS/install_foxy" \
    --symlink-install \
    --packages-select voxl_h265_decoder \
    --parallel-workers 1

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
RUNTIME_DIR="${CACHE_ROOT}/ros2-foxy-py${PYTHON_VERSION}"
COLCON_INSTALL="${RUNTIME_DIR}/colcon/install"
VENV_SITE="${RUNTIME_DIR}/venv/lib/python${PYTHON_VERSION}/site-packages"
LOG_DIR="${CACHE_ROOT}/bag-tests/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()

stop_pid() {
    local pid="$1"
    if kill -0 "$pid" >/dev/null 2>&1; then
        kill -INT "$pid" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                break
            fi
            sleep 0.2
        done
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    fi
    wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    local pid
    for pid in "${PIDS[@]}"; do
        stop_pid "$pid"
    done
}
trap cleanup EXIT INT TERM

echo "Starting ECHO-LI..."
"$SCRIPT_DIR/run_voxl2_ros2.sh" \
    --internal-id "$INTERNAL_ID" \
    --ros-distro foxy \
    --depth-backend "$DEPTH_BACKEND" \
    --imu-topic /echo_li_test/imu \
    --image-topic /echo_li_test/image \
    >"$LOG_DIR/echo_li.log" 2>&1 &
ECHO_PID=$!
PIDS+=("$ECHO_PID")

for _ in $(seq 1 120); do
    if ! kill -0 "$ECHO_PID" >/dev/null 2>&1; then
        echo "ECHO-LI exited during startup:" >&2
        tail -80 "$LOG_DIR/echo_li.log" >&2
        exit 1
    fi
    if ros2 node list 2>/dev/null | grep -qx '/echo_li_voxl2'; then
        break
    fi
    sleep 1
done
if ! ros2 node list 2>/dev/null | grep -qx '/echo_li_voxl2'; then
    echo "Timed out waiting for /echo_li_voxl2." >&2
    tail -80 "$LOG_DIR/echo_li.log" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "$COLCON_INSTALL/setup.bash"
# shellcheck disable=SC1090
source "$DECODER_WS/install_foxy/setup.bash"
set -u
export PYTHONPATH="${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting ${DECODER} decoder..."
ros2 run voxl_h265_decoder h265_decoder_node \
    --ros-args -p "decoder:=${DECODER}" \
    >"$LOG_DIR/decoder.log" 2>&1 &
DECODER_PID=$!
PIDS+=("$DECODER_PID")

echo "Starting bag-time relay..."
ros2 run echo_li_ros2 bag_time_relay \
    >"$LOG_DIR/relay.log" 2>&1 &
RELAY_PID=$!
PIDS+=("$RELAY_PID")

sleep 2
if ! kill -0 "$DECODER_PID" >/dev/null 2>&1; then
    echo "Decoder exited during startup:" >&2
    tail -80 "$LOG_DIR/decoder.log" >&2
    exit 1
fi
if ! kill -0 "$RELAY_PID" >/dev/null 2>&1; then
    echo "Bag-time relay exited during startup:" >&2
    tail -80 "$LOG_DIR/relay.log" >&2
    exit 1
fi

echo "Playing $BAG_PATH at ${RATE}x..."
ros2 bag play "$BAG_PATH" --rate "$RATE" \
    >"$LOG_DIR/bag.log" 2>&1
sleep 3

stop_pid "$RELAY_PID"
stop_pid "$DECODER_PID"
stop_pid "$ECHO_PID"
PIDS=()

ODOMETRY_COUNT="$(sed -n \
    's/.*odometry=\([0-9][0-9]*\).*/\1/p' \
    "$LOG_DIR/relay.log" | tail -1)"
ODOMETRY_COUNT="${ODOMETRY_COUNT:-0}"

echo
echo "Decoder summary:"
grep 'H.265 totals:' "$LOG_DIR/decoder.log" | tail -1 || true
echo "Relay summary:"
grep 'relay:' "$LOG_DIR/relay.log" | tail -1 || true
echo "ECHO-LI summary:"
grep 'input:' "$LOG_DIR/echo_li.log" | tail -1 || true
echo "Logs: $LOG_DIR"

if [[ "$ODOMETRY_COUNT" -lt 1 ]]; then
    echo "FAIL: ECHO-LI published no odometry." >&2
    exit 1
fi

echo "PASS: ECHO-LI published ${ODOMETRY_COUNT} odometry messages."
