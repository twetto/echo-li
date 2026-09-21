#!/usr/bin/env bash
# Run ECHO-LI on an NFU bag (raw /imu_apps + /tracking_front, no decoder needed).
#
# Usage:
#   ./run_nfu_bag.sh --bag ~/Downloads/nfu_bags/VIO [--rate 1.0] [--internal-id 2]
#
# Run inside the ubuntu-22-04 Distrobox.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH=""
INTERNAL_ID=2
RATE=1.0
RERUN_URL=""

usage() {
    cat <<'EOF'
Run ECHO-LI on an NFU bag (raw /imu_apps + /tracking_front).

Usage:
  ./run_nfu_bag.sh --bag <path> [options]

Required:
  --bag PATH             Bag directory.

Options:
  --internal-id {1|2}    VOXL2 calibration (default: 2).
  --rate RATE             Playback rate (default: 1.0).
  --rerun-url URL         Use existing Rerun viewer instead of spawning one.
  -h, --help              Show this help.
EOF
}

need_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $1" >&2; usage >&2; exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)         need_value "$@"; BAG_PATH="$2"; shift 2 ;;
        --internal-id) need_value "$@"; INTERNAL_ID="$2"; shift 2 ;;
        --rate)        need_value "$@"; RATE="$2"; shift 2 ;;
        --rerun-url)   need_value "$@"; RERUN_URL="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$BAG_PATH" ]]; then
    echo "--bag is required" >&2; usage >&2; exit 2
fi
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Bag directory not found: $BAG_PATH" >&2; exit 1
fi
if [[ ! -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble is unavailable. Enter ubuntu-22-04 first." >&2; exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# --- Launch Rerun on the host if needed ---
if [[ -z "$RERUN_URL" ]]; then
    if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' >/dev/null 2>&1; then
        if ! command -v distrobox-host-exec >/dev/null 2>&1; then
            echo "distrobox-host-exec is required to launch the host Rerun viewer." >&2
            exit 1
        fi
        echo "Launching Rerun on the host..."
        distrobox-host-exec rerun >/dev/null 2>&1 &
        for _ in $(seq 1 100); do
            if timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' >/dev/null 2>&1; then break; fi
            sleep 0.1
        done
        if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' >/dev/null 2>&1; then
            echo "Timed out waiting for Rerun viewer." >&2; exit 1
        fi
    fi
    RERUN_URL='rerun+http://127.0.0.1:9876/proxy'
fi

# --- Process management ---
PIDS=()

stop_group() {
    local pid="$1"
    if kill -0 "$pid" >/dev/null 2>&1; then
        kill -INT -- "-$pid" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" >/dev/null 2>&1 || break
            sleep 0.2
        done
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM -- "-$pid" >/dev/null 2>&1 || true
        fi
    fi
    wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    trap - EXIT INT TERM
    for ((i=${#PIDS[@]} - 1; i >= 0; i--)); do
        stop_group "${PIDS[i]}"
    done
}
trap cleanup EXIT INT TERM

# --- Start ECHO-LI (subscribes directly to /imu_apps + /tracking_front) ---
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
LOG_DIR="$CACHE_ROOT/nfu-bag-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting ECHO-LI (internal ID ${INTERNAL_ID})..."
setsid "$SCRIPT_DIR/run_voxl2_ros2.sh" \
    --internal-id "$INTERNAL_ID" \
    --ros-distro humble \
    --imu-topic /imu_apps \
    --image-topic /tracking_front \
    --visualize \
    --rerun-url "$RERUN_URL" \
    >"$LOG_DIR/echo_li.log" 2>&1 &
ECHO_PID=$!
PIDS+=("$ECHO_PID")

for _ in $(seq 1 60); do
    if ! kill -0 "$ECHO_PID" >/dev/null 2>&1; then
        echo "ECHO-LI exited during startup:" >&2
        tail -40 "$LOG_DIR/echo_li.log" >&2
        exit 1
    fi
    if ros2 node list 2>/dev/null | grep -qx '/echo_li_voxl2'; then break; fi
    sleep 1
done
if ! ros2 node list 2>/dev/null | grep -qx '/echo_li_voxl2'; then
    echo "Timed out waiting for /echo_li_voxl2." >&2
    tail -40 "$LOG_DIR/echo_li.log" >&2
    exit 1
fi
sleep 1

echo "Playing $BAG_PATH at ${RATE}x. Close Rerun or press Ctrl-C to stop."
set +e
ros2 bag play "$BAG_PATH" --rate "$RATE" \
    >"$LOG_DIR/bag.log" 2>&1
BAG_STATUS=$?
set -e
sleep 3

cleanup
PIDS=()

echo
echo "ECHO-LI summary:"
grep 'input:' "$LOG_DIR/echo_li.log" | tail -1 || true
echo "Logs: $LOG_DIR"

if [[ "$BAG_STATUS" -ne 0 && "$BAG_STATUS" -ne 130 ]]; then
    echo "ros2 bag play exited with status $BAG_STATUS" >&2
    exit "$BAG_STATUS"
fi
