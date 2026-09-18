#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH=""
CALIBRATION="$SCRIPT_DIR/echo-li-ros2-rs/config/myrig_calib.yaml"
ECHO_CONFIG="$SCRIPT_DIR/echo-li-ros2/config/eqvio_myrig.yaml"
RATE=1.0
DEPTH_BACKEND="dis"
IMU_TOPIC="/imu"
IMAGE_COMPRESSED_TOPIC="/camera/image/compressed"
IMAGE_QOS_RELIABILITY="reliable"
IMAGE_SCALE="1.0"
GT_TRAJECTORY=""
MOCAP_TOPIC=""
VIO_TOPIC=""
KILL_STALE=0

usage() {
    cat <<'EOF'
Play a bag with compressed images through the native Rust ROS 2 wrapper.

Run this script inside the ubuntu-22-04 Distrobox.

Usage:
  ./run_bag_humble_rs.sh --bag PATH [options]

Options:
  --bag PATH                  Bag directory (required).
  --calibration PATH          Camera calibration YAML (default: myrig).
  --echo-config PATH          EqVIO config YAML (default: eqvio_myrig).
  --rate RATE                 Playback rate (default: 1.0).
  --depth-backend NAME        dis or patch (default: dis).
  --imu-topic TOPIC           IMU topic in the bag (default: /imu).
  --image-topic TOPIC         CompressedImage topic (default: /camera/image/compressed).
  --image-qos RELIABILITY     Image QoS: reliable or best_effort (default: reliable).
  --image-scale SCALE         Downsample factor: 0.5 for half-res (default: 1.0).
  --gt PATH                   Ground-truth trajectory (TUM) for ATE eval.
  --mocap-topic TOPIC         Mocap PoseStamped topic to overlay.
  --vio-topic TOPIC           External VIO Odometry topic to overlay.
  --kill-stale                Stop leftover processes on the same ROS domain.
  -h, --help                  Show this help.
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
        --calibration)
            need_value "$@"
            CALIBRATION="$2"
            shift 2
            ;;
        --echo-config)
            need_value "$@"
            ECHO_CONFIG="$2"
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
        --imu-topic)
            need_value "$@"
            IMU_TOPIC="$2"
            shift 2
            ;;
        --image-topic)
            need_value "$@"
            IMAGE_COMPRESSED_TOPIC="$2"
            shift 2
            ;;
        --image-qos)
            need_value "$@"
            IMAGE_QOS_RELIABILITY="$2"
            shift 2
            ;;
        --image-scale)
            need_value "$@"
            IMAGE_SCALE="$2"
            shift 2
            ;;
        --gt)
            need_value "$@"
            GT_TRAJECTORY="$2"
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
        --kill-stale)
            KILL_STALE=1
            shift
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

if [[ -z "$BAG_PATH" ]]; then
    echo "--bag is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Bag directory not found: $BAG_PATH" >&2
    exit 1
fi
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v rate="$RATE" 'BEGIN { exit !(rate > 0) }'; then
    echo "--rate must be a positive number" >&2
    exit 2
fi
if [[ ! -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble is unavailable. Enter ubuntu-22-04 first." >&2
    exit 1
fi

VIO_BIN="$SCRIPT_DIR/target-humble/release/voxl2_vio_node"
if [[ ! -x "$VIO_BIN" ]]; then
    echo "Rust VIO binary not found: $VIO_BIN" >&2
    echo "Build it with:" >&2
    echo "  BINDGEN_EXTRA_CLANG_ARGS=\"-I/usr/lib/gcc/x86_64-linux-gnu/11/include\" \\" >&2
    echo "  CARGO_TARGET_DIR=target-humble cargo build --release -p echo-li-ros2 --features parallel" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# Kill stale processes from a previous run on the same ROS domain.
STALE_PIDS=()
for pid in $(pgrep -f 'voxl2_vio_node|republish|ros2 bag play' || true); do
    domain="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null |
        sed -n 's/^ROS_DOMAIN_ID=//p')"
    if [[ "$domain" == "$ROS_DOMAIN_ID" ]]; then
        STALE_PIDS+=("$pid")
    fi
done
if [[ ${#STALE_PIDS[@]} -gt 0 ]]; then
    echo "Leftover pipeline processes on ROS domain ${ROS_DOMAIN_ID}:" >&2
    ps -o pid=,etime=,args= -p "$(IFS=,; echo "${STALE_PIDS[*]}")" | cut -c1-160 >&2 || true
    if [[ "$KILL_STALE" -ne 1 ]]; then
        echo "Stop them or pass --kill-stale." >&2
        exit 1
    fi
    echo "Stopping them (--kill-stale)..." >&2
    kill -INT "${STALE_PIDS[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${STALE_PIDS[@]}" 2>/dev/null || true
fi

CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
LOG_DIR="$CACHE_ROOT/humble-bag-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()

group_gone() {
    local pgid="$1" tries="$2"
    for ((; tries > 0; tries--)); do
        kill -0 -- "-$pgid" >/dev/null 2>&1 || return 0
        sleep 0.2
    done
    ! kill -0 -- "-$pgid" >/dev/null 2>&1
}

stop_group() {
    local pid="$1"
    kill -INT -- "-$pid" >/dev/null 2>&1 || true
    if ! group_gone "$pid" 10; then
        kill -TERM -- "-$pid" >/dev/null 2>&1 || true
        if ! group_gone "$pid" 10; then
            kill -KILL -- "-$pid" >/dev/null 2>&1 || true
        fi
    fi
    wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    local index
    trap '' INT TERM
    trap - EXIT
    for ((index=${#PIDS[@]} - 1; index >= 0; index--)); do
        stop_group "${PIDS[index]}"
    done
}
trap cleanup EXIT INT TERM

# Decompress CompressedImage → raw Image.
IMAGE_RAW_TOPIC="/echo_li/image"
echo "Starting image_transport republish: ${IMAGE_COMPRESSED_TOPIC} → ${IMAGE_RAW_TOPIC}"
setsid ros2 run image_transport republish compressed raw \
    --ros-args \
    --remap "in/compressed:=${IMAGE_COMPRESSED_TOPIC}" \
    --remap "out:=${IMAGE_RAW_TOPIC}" \
    >"$LOG_DIR/republish.log" 2>&1 &
REPUB_PID=$!
PIDS+=("$REPUB_PID")
sleep 2

if ! kill -0 "$REPUB_PID" >/dev/null 2>&1; then
    echo "image_transport republish exited during startup:" >&2
    tail -40 "$LOG_DIR/republish.log" >&2 || true
    exit 1
fi

# Start the Rust VIO node.
echo "Starting ECHO-LI (Rust) on ROS domain ${ROS_DOMAIN_ID}..."
EXTRA_ROS_ARGS=()
if [[ -n "$MOCAP_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "mocap_topic:=${MOCAP_TOPIC}")
fi
if [[ -n "$VIO_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "vio_topic:=${VIO_TOPIC}")
fi
setsid env RUST_LOG="${RUST_LOG:-info}" "$VIO_BIN" --ros-args \
    --params-file "$CALIBRATION" \
    -p "echo_config_path:=${ECHO_CONFIG}" \
    -p "imu_topic:=${IMU_TOPIC}" \
    -p "image_topic:=${IMAGE_RAW_TOPIC}" \
    -p "image_qos_reliability:=${IMAGE_QOS_RELIABILITY}" \
    -p "depth_backend:=${DEPTH_BACKEND}" \
    -p "patch_depth_enabled:=false" \
    -p "occupancy_enabled:=false" \
    -p "image_scale:=${IMAGE_SCALE}" \
    -p "trajectory_output:=${LOG_DIR}/trajectory.tum" \
    "${EXTRA_ROS_ARGS[@]}" \
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

# Launch rviz2 with the ECHO-LI config.
RVIZ_CFG="$SCRIPT_DIR/echo-li-ros2-rs/config/echo_li.rviz"
if [[ -f "$RVIZ_CFG" ]] && command -v rviz2 >/dev/null 2>&1; then
    setsid env QT_QPA_PLATFORM=xcb rviz2 -d "$RVIZ_CFG" \
        >"$LOG_DIR/rviz2.log" 2>&1 &
    RVIZ_PID=$!
    PIDS+=("$RVIZ_PID")
fi

# Override bag QoS to reliable for the image topic so image_transport
# republish (which subscribes reliable) can receive.  Leave IMU at its
# native best-effort — the VIO subscriber is best-effort too.
BAG_QOS="$LOG_DIR/bag_qos_override.yaml"
cat > "$BAG_QOS" <<YAML
${IMAGE_COMPRESSED_TOPIC}:
  reliability: reliable
  history: keep_last
  depth: 5
YAML

echo "Playing $BAG_PATH at ${RATE}x. Press Ctrl-C to stop."
echo "  Odometry: /echo_li/odometry (frame: echo_li_odom)"
echo "  Landmarks: /echo_li/landmarks"
echo "  rviz2 -d $SCRIPT_DIR/echo-li-ros2-rs/config/echo_li.rviz"
set +e
ros2 bag play "$BAG_PATH" --rate "$RATE" \
    --qos-profile-overrides-path "$BAG_QOS" \
    >"$LOG_DIR/bag.log" 2>&1
BAG_STATUS=$?
set -e
sleep 3

cleanup
PIDS=()

echo
echo "ECHO-LI summary:"
grep 'input:' "$LOG_DIR/echo_li.log" | tail -1 || true
echo "Trajectory: $LOG_DIR/trajectory.tum"

if [[ -n "$GT_TRAJECTORY" && -f "$LOG_DIR/trajectory.tum" ]]; then
    T_BC=$(python3 -c "
import yaml, sys
with open('$CALIBRATION') as f:
    d = yaml.safe_load(f)
vals = list(d.values())[0]['ros__parameters']['t_bs']
print(' '.join(f'{v}' for v in vals))
" 2>/dev/null)
    T_BC_ARGS=()
    if [[ -n "$T_BC" ]]; then
        T_BC_ARGS=(--t-bc "$T_BC")
    fi
    echo
    TIME_OFFSET=$(grep -oP 'timeshift_cam_imu\s*=\s*\K[0-9.e+-]+' "$GT_TRAJECTORY" 2>/dev/null || true)
    TIME_ARGS=()
    if [[ -n "$TIME_OFFSET" ]]; then
        TIME_ARGS=(--time-offset "$TIME_OFFSET")
    fi
    python3 "$SCRIPT_DIR/tools/eval_ate.py" "$LOG_DIR/trajectory.tum" "$GT_TRAJECTORY" \
        --max-gap 0.1 --align se3 "${T_BC_ARGS[@]}" "${TIME_ARGS[@]}"
fi

echo "Logs: $LOG_DIR"

if [[ "$BAG_STATUS" -ne 0 && "$BAG_STATUS" -ne 130 ]]; then
    echo "ros2 bag play exited with status $BAG_STATUS" >&2
    exit "$BAG_STATUS"
fi
