#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="/home/twetto/Downloads/voxl_imu_camera_test_raw_imu_track_encoded_2"
INTERNAL_ID=2
RATE=1.0
DEPTH_BACKEND="dis"
DECODER=auto
RERUN_URL=""
MOCAP_TOPIC=""
VIO_TOPIC=""

usage() {
    cat <<'EOF'
Play a VOXL2 bag through ECHO-LI with live Rerun visualization.

Run this script inside the ubuntu-22-04 Distrobox.

Usage:
  ./run_voxl2_bag_humble.sh [options]

Options:
  --bag PATH             Bag directory.
  --internal-id {1|2}   VOXL2 factory calibration (default: 2).
  --rate RATE            Playback rate (default: 1.0).
  --depth-backend NAME   Depth estimation backend: dis or patch (default: dis).
  --decoder ELEMENT      GStreamer decoder (default: auto; Radeon VA-API preferred).
  --rerun-url URL        Use an existing remote viewer instead of spawning one.
  --mocap-topic TOPIC    Mocap PoseStamped topic to overlay (SE(3)-aligned).
  --vio-topic TOPIC      External VIO Odometry topic to overlay (SE(3)-aligned).
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
        --rate)
            need_value "$@"
            RATE="$2"
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
        --rerun-url)
            need_value "$@"
            RERUN_URL="$2"
            shift 2
            ;;
        --mocap-topic)
            need_value "$@"
            MOCAP_TOPIC="$2"
            shift 2
            ;;
        --vio-topic)
            need_value "$@"
            VIO_TOPIC="$2"
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
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v rate="$RATE" 'BEGIN { exit !(rate > 0) }'; then
    echo "--rate must be a positive number" >&2
    exit 2
fi
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Bag directory not found: $BAG_PATH" >&2
    exit 1
fi
if [[ ! -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble is unavailable. Enter ubuntu-22-04 first." >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

DECODER_WS="${VOXL_DECODER_WS:-/home/twetto/voxl_h265_decoder_ws}"
if [[ ! -d "$DECODER_WS/src/voxl_h265_decoder" ]]; then
    echo "Decoder workspace not found: $DECODER_WS" >&2
    exit 1
fi
for command in colcon ros2 setsid; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command" >&2
        exit 1
    fi
done


CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
LOG_DIR="$CACHE_ROOT/humble-bag-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ -z "$RERUN_URL" ]]; then
    if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' \
            >/dev/null 2>&1; then
        if ! command -v distrobox-host-exec >/dev/null 2>&1; then
            echo "distrobox-host-exec is required to launch the host Rerun viewer." >&2
            exit 1
        fi
        echo "Launching Rerun 0.31.3 on the host..."
        distrobox-host-exec rerun >"$LOG_DIR/rerun.log" 2>&1 &
        for _ in $(seq 1 100); do
            if timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' \
                    >/dev/null 2>&1; then
                break
            fi
            sleep 0.1
        done
        if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/9876' \
                >/dev/null 2>&1; then
            echo "Timed out waiting for the host Rerun viewer." >&2
            tail -80 "$LOG_DIR/rerun.log" >&2 || true
            exit 1
        fi
    fi
    RERUN_URL='rerun+http://127.0.0.1:9876/proxy'
fi

PIDS=()

stop_group() {
    local pid="$1"
    if kill -0 "$pid" >/dev/null 2>&1; then
        kill -INT -- "-$pid" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                break
            fi
            sleep 0.2
        done
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM -- "-$pid" >/dev/null 2>&1 || true
        fi
    fi
    wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    local index
    trap - EXIT INT TERM
    for ((index=${#PIDS[@]} - 1; index >= 0; index--)); do
        stop_group "${PIDS[index]}"
    done
}
trap cleanup EXIT INT TERM

# Only rebuild the decoder if the install is missing.
if [[ ! -f "$DECODER_WS/install_humble/setup.bash" ]]; then
    echo "Building the Humble H.265 decoder..."
    colcon --log-base "$DECODER_WS/log_humble" build \
        --base-paths "$DECODER_WS/src" \
        --build-base "$DECODER_WS/build_humble" \
        --install-base "$DECODER_WS/install_humble" \
        --symlink-install \
        --packages-select voxl_h265_decoder \
        --parallel-workers 1
fi

echo "Starting ECHO-LI and Rerun on ROS domain ${ROS_DOMAIN_ID}..."
ECHO_ARGS=(
    --internal-id "$INTERNAL_ID"
    --ros-distro humble
    --depth-backend "$DEPTH_BACKEND"
    --imu-topic /echo_li_test/imu
    --image-topic /echo_li_test/image
    --visualize
)
if [[ -n "$RERUN_URL" ]]; then
    ECHO_ARGS+=(--rerun-url "$RERUN_URL")
fi
# Pass reference trajectory topics as ROS parameters (via the -- separator).
EXTRA_ROS_ARGS=()
if [[ -n "$MOCAP_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "mocap_topic:=${MOCAP_TOPIC}")
fi
if [[ -n "$VIO_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "vio_topic:=${VIO_TOPIC}")
fi
if [[ ${#EXTRA_ROS_ARGS[@]} -gt 0 ]]; then
    ECHO_ARGS+=(-- --ros-args "${EXTRA_ROS_ARGS[@]}")
fi
setsid "$SCRIPT_DIR/run_voxl2_ros2.sh" "${ECHO_ARGS[@]}" \
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
sleep 2
if ! kill -0 "$ECHO_PID" >/dev/null 2>&1; then
    echo "ECHO-LI exited immediately after discovery:" >&2
    tail -80 "$LOG_DIR/echo_li.log" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "$DECODER_WS/install_humble/setup.bash"
# shellcheck disable=SC1090
source "$CACHE_ROOT/ros2-humble-py3.10/colcon/install/setup.bash"
set -u

echo "Starting ${DECODER} and the bag-time relay..."
# Output mono8 — echo-li only needs grayscale; skipping BGR conversion
# saves the decoder's videoconvert and echo-li's to_gray(), cutting image
# bandwidth by 3× and removing two per-frame color conversions.
setsid ros2 run voxl_h265_decoder h265_decoder_node \
    --ros-args \
    -p "decoder:=${DECODER}" \
    -p "output_encoding:=mono8" \
    >"$LOG_DIR/decoder.log" 2>&1 &
DECODER_PID=$!
PIDS+=("$DECODER_PID")

setsid ros2 run echo_li_ros2 bag_time_relay \
    >"$LOG_DIR/relay.log" 2>&1 &
RELAY_PID=$!
PIDS+=("$RELAY_PID")

sleep 2
if ! kill -0 "$DECODER_PID" >/dev/null 2>&1; then
    echo "The decoder exited during startup." >&2
    tail -80 "$LOG_DIR/decoder.log" >&2 || true
    exit 1
fi
if ! kill -0 "$RELAY_PID" >/dev/null 2>&1; then
    echo "The bag-time relay exited during startup." >&2
    tail -80 "$LOG_DIR/relay.log" >&2 || true
    exit 1
fi

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
echo "Decoder summary:"
grep 'H.265 totals:' "$LOG_DIR/decoder.log" | tail -1 || true
echo "Relay summary:"
grep 'relay:' "$LOG_DIR/relay.log" | tail -1 || true
echo "ECHO-LI summary:"
grep 'input:' "$LOG_DIR/echo_li.log" | tail -1 || true
echo "Logs: $LOG_DIR"

if [[ "$BAG_STATUS" -ne 0 && "$BAG_STATUS" -ne 130 ]]; then
    echo "ros2 bag play exited with status $BAG_STATUS" >&2
    exit "$BAG_STATUS"
fi
