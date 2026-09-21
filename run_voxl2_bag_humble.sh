#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="/home/twetto/Downloads/voxl_imu_camera_test_raw_imu_track_encoded_2"
INTERNAL_ID=""
RATE=1.0
DECODER=auto
OCCUPANCY=true
IMAGE_STAMP_MODE="auto"
ECHO_CONFIG="$SCRIPT_DIR/echo-li-ros2/config/eqvio_voxl2.yaml"
GT_TRAJECTORY=""
MOCAP_TOPIC=""
VIO_TOPIC=""
USE_RVIZ=1
KILL_STALE=0
WORLD_FRAME=""
RVIZ_CFG=""
RVIZ_VIEW="follow"
KEEP_RVIZ=1
NODE_PARAMS=()
CALIBRATION_OVERRIDE=""
# What the node names the body in its TF; matches body_frame_id in the node.
BODY_FRAME="imu_link"
WORLD_RPY="0 0 0"
IMU_TOPIC="/echo_li_test/imu"
IMAGE_TOPIC="/echo_li_test/image"

usage() {
    cat <<'EOF'
Play a VOXL2 bag (H.265 camera + raw IMU) through the native Rust ROS 2 wrapper.

Pipeline: bag -> voxl_h265_decoder -> bag_time_relay -> voxl2_vio_node (r2r).

Run this script inside the ubuntu-22-04 Distrobox.

Usage:
  ./run_voxl2_bag_humble.sh [options]

Options:
  --bag PATH             Bag directory.
  --internal-id {1|2}    Use the VOXL2 factory calibration for that unit
                         instead of the Kalibr one, which is the default.
  --rate RATE            Playback rate (default: 1.0).
  --decoder ELEMENT      GStreamer decoder (default: auto).
  --image-stamp-mode M   auto, source or imu_anchored (default: auto).
  --imu-topic TOPIC      What the VIO node subscribes to for IMU
                         (default: /echo_li_test/imu, the relay's output;
                         /voxl/raw_imu takes it straight from the bag).
  --image-topic TOPIC    Same for images (default: /echo_li_test/image).
  --echo-config PATH     EqVIO config YAML (default: eqvio_voxl2).
  --gt PATH              Ground-truth trajectory (TUM) for ATE eval.
  --mocap-topic TOPIC    Mocap PoseStamped topic to plot next to the estimate
                         (default: the bag's */vision_pose/pose, else its
                         /vrpn_mocap/*/pose; "none" to skip it). The node fits
                         the mocap track onto the estimate by heading and
                         origin and republishes it on /echo_li/mocap_path.
  --vio-topic TOPIC      External VIO Odometry topic to overlay.
  --no-occupancy         Disable the local occupancy grid (it is on by default
                         and published as /echo_li/occupancy).
  --no-rviz              Don't launch rviz2.
  --rviz-config PATH     rviz2 config (default: echo_li_voxl2.rviz).
  --rviz-view VIEW       Startup 3D view: follow (default, chases imu_link) or
                         world (fixed view, for debugging the trajectory).
                         Both stay in rviz2's Views panel either way.
  --world-frame NAME     Publish a static TF NAME -> echo_li_odom and make
                         NAME the rviz2 fixed frame. echo_li_odom is already
                         gravity-aligned z-up; its yaw is whatever the rig
                         faced at initialisation.
  --world-rpy "R P Y"    Rotation for that TF, radians (default: 0 0 0).
  --close-rviz           Close rviz2 when the run ends. By default it is left
                         open with the last pose pinned, and the next run on
                         this domain reuses it.
  --calibration PATH     Camera/IMU params file. The default,
                         voxl2_kalibr_20260907.yaml, is the Kalibr camera-IMU
                         result: 1.6 deg and 30 ms better than the factory
                         numbers, and it scored 0.31 m against mocap on the
                         yaw-turning flight where those scored 0.48 m.
  --param NAME:=VALUE    Extra ROS parameter for the VIO node, repeatable
                         (e.g. --param camera_time_offset_sec:=-0.02).
  --kill-stale           Stop leftover processes on the same ROS domain.
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
        --bag) need_value "$@"; BAG_PATH="$2"; shift 2 ;;
        --internal-id) need_value "$@"; INTERNAL_ID="$2"; shift 2 ;;
        --rate) need_value "$@"; RATE="$2"; shift 2 ;;
        --decoder) need_value "$@"; DECODER="$2"; shift 2 ;;
        --image-stamp-mode) need_value "$@"; IMAGE_STAMP_MODE="$2"; shift 2 ;;
        --imu-topic) need_value "$@"; IMU_TOPIC="$2"; shift 2 ;;
        --image-topic) need_value "$@"; IMAGE_TOPIC="$2"; shift 2 ;;
        --echo-config) need_value "$@"; ECHO_CONFIG="$2"; shift 2 ;;
        --gt) need_value "$@"; GT_TRAJECTORY="$2"; shift 2 ;;
        --mocap-topic) need_value "$@"; MOCAP_TOPIC="$2"; shift 2 ;;
        --vio-topic) need_value "$@"; VIO_TOPIC="$2"; shift 2 ;;
        --no-occupancy) OCCUPANCY=false; shift ;;
        --no-rviz) USE_RVIZ=0; shift ;;
        --rviz-config) need_value "$@"; RVIZ_CFG="$2"; shift 2 ;;
        --rviz-view) need_value "$@"; RVIZ_VIEW="$2"; shift 2 ;;
        --world-frame) need_value "$@"; WORLD_FRAME="$2"; shift 2 ;;
        --world-rpy) need_value "$@"; WORLD_RPY="$2"; shift 2 ;;
        --calibration) need_value "$@"; CALIBRATION_OVERRIDE="$2"; shift 2 ;;
        --param) need_value "$@"; NODE_PARAMS+=("$2"); shift 2 ;;
        --close-rviz) KEEP_RVIZ=0; shift ;;
        --kill-stale) KILL_STALE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$INTERNAL_ID" && "$INTERNAL_ID" != "1" && "$INTERNAL_ID" != "2" ]]; then
    echo "--internal-id must be 1 or 2" >&2
    exit 2
fi
case "$IMAGE_STAMP_MODE" in
    auto|source|imu_anchored) ;;
    *) echo "--image-stamp-mode must be auto, source or imu_anchored" >&2; exit 2 ;;
esac
if [[ ! "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v rate="$RATE" 'BEGIN { exit !(rate > 0) }'; then
    echo "--rate must be a positive number" >&2
    exit 2
fi
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Bag directory not found: $BAG_PATH" >&2
    exit 1
fi
if [[ -n "$CALIBRATION_OVERRIDE" ]]; then
    CALIBRATION="$CALIBRATION_OVERRIDE"
elif [[ -n "$INTERNAL_ID" ]]; then
    CALIBRATION="$SCRIPT_DIR/echo-li-ros2/config/voxl2_internal_id_${INTERNAL_ID}.yaml"
else
    CALIBRATION="$SCRIPT_DIR/echo-li-ros2/config/voxl2_kalibr_20260907.yaml"
fi
for file in "$CALIBRATION" "$ECHO_CONFIG"; do
    if [[ ! -r "$file" ]]; then
        echo "Config not readable: $file" >&2
        exit 1
    fi
done
if [[ ! -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble is unavailable. Enter ubuntu-22-04 first." >&2
    exit 1
fi

VIO_BIN="$SCRIPT_DIR/target-humble/release/voxl2_vio_node"
RELAY_BIN="$SCRIPT_DIR/target-humble/release/bag_time_relay"
for binary in "$VIO_BIN" "$RELAY_BIN"; do
    if [[ ! -x "$binary" ]]; then
        echo "Rust binary not found: $binary" >&2
        echo "Build both with:" >&2
        echo "  BINDGEN_EXTRA_CLANG_ARGS=\"-I/usr/lib/gcc/x86_64-linux-gnu/11/include\" \\" >&2
        echo "  CARGO_TARGET_DIR=target-humble cargo build --release -p echo-li-ros2 --features parallel" >&2
        exit 1
    fi
done

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

# Stop leftovers from a previous run on this ROS domain.
STALE_PIDS=()
for pid in $(pgrep -f 'voxl2_vio_node|bag_time_relay|h265_decoder_node|ros2 bag play' || true); do
    domain="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null |
        sed -n 's/^ROS_DOMAIN_ID=//p')"
    if [[ "${domain:-42}" == "$ROS_DOMAIN_ID" ]]; then
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

# The TF helpers an earlier run left for its rviz2 would fight this run's.
for pid in $(pgrep -f 'echo_li_world_tf|echo_li_final_pose|echo_li_view_watch' \
    2>/dev/null || true); do
    domain="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null |
        sed -n 's/^ROS_DOMAIN_ID=//p')"
    if [[ "${domain:-42}" == "$ROS_DOMAIN_ID" ]]; then
        kill "$pid" 2>/dev/null || true
    fi
done

CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
LOG_DIR="$CACHE_ROOT/humble-bag-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()
# rviz2 and the static TFs it needs outlive the pipeline, so the result stays
# on screen after the bag ends.
VIEW_PIDS=()
RVIZ_PID=""
RVIZ_OWN_PID=""

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

# Once the pipeline stops, nothing publishes echo_li_odom -> imu_link any
# more and rviz2 drops the frame the follow view chases, taking the view with
# it. Pin the last estimated pose instead, and let the helpers go when the
# window closes.
keep_or_stop_viewer() {
    local last pose
    if [[ "$KEEP_RVIZ" -ne 1 || -z "$RVIZ_PID" ]] || ! kill -0 "$RVIZ_PID" 2>/dev/null; then
        if [[ -n "$RVIZ_OWN_PID" ]]; then
            stop_group "$RVIZ_OWN_PID"
        fi
        local pid
        for pid in ${VIEW_PIDS[@]+"${VIEW_PIDS[@]}"}; do
            stop_group "$pid"
        done
        return 0
    fi
    last="$(grep -v '^#' "$LOG_DIR/trajectory.tum" 2>/dev/null | tail -1 || true)"
    if [[ -n "$last" ]]; then
        read -r _ pose <<<"$last"
        read -r PX PY PZ QX QY QZ QW <<<"$pose"
        setsid ros2 run tf2_ros static_transform_publisher \
            --frame-id echo_li_odom --child-frame-id "$BODY_FRAME" \
            --x "$PX" --y "$PY" --z "$PZ" \
            --qx "$QX" --qy "$QY" --qz "$QZ" --qw "$QW" \
            --ros-args -r __node:=echo_li_final_pose \
            >"$LOG_DIR/final_pose_tf.log" 2>&1 &
        VIEW_PIDS+=("$!")
    fi
    if [[ ${#VIEW_PIDS[@]} -gt 0 ]]; then
        setsid bash -c "while kill -0 ${RVIZ_PID} 2>/dev/null; do sleep 2; done
            for pid in ${VIEW_PIDS[*]}; do kill -TERM -- -\$pid 2>/dev/null; done
            # echo_li_view_watch" \
            >/dev/null 2>&1 &
    fi
    echo "rviz2 (pid ${RVIZ_PID}) left open, showing the run with the last pose pinned."
    echo "  Close the window when you are done; the next run reuses it."
}

cleanup() {
    local index
    trap '' INT TERM
    trap - EXIT
    for ((index=${#PIDS[@]} - 1; index >= 0; index--)); do
        stop_group "${PIDS[index]}"
    done
    keep_or_stop_viewer
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

# The relay always publishes here; --imu-topic / --image-topic pick what the VIO
# node subscribes to (the relay output by default).
RELAY_IMU_TOPIC="/echo_li_test/imu"
RELAY_IMAGE_TOPIC="/echo_li_test/image"

# Mocap: plot it whenever the bag carries it, unless told otherwise.
if [[ "$MOCAP_TOPIC" == "none" ]]; then
    MOCAP_TOPIC=""
elif [[ -z "$MOCAP_TOPIC" && -f "$BAG_PATH/metadata.yaml" ]]; then
    # What the FC was flying on first, the raw VRPN stream as a fallback.
    for pattern in '/[A-Za-z0-9_]+/vision_pose/pose' '/vrpn_mocap/[A-Za-z0-9_]+/pose'; do
        # A bag without mocap is the normal case, so neither the empty match nor
        # the empty result may take the script down with set -e.
        MOCAP_TOPIC="$(grep -oE "$pattern" "$BAG_PATH/metadata.yaml" | head -1 || true)"
        if [[ -n "$MOCAP_TOPIC" ]]; then
            break
        fi
    done
    if [[ -n "$MOCAP_TOPIC" ]]; then
        echo "Mocap: plotting ${MOCAP_TOPIC} from the bag (--mocap-topic none to skip)"
    fi
fi

echo "Starting ECHO-LI (Rust) on ROS domain ${ROS_DOMAIN_ID}..."
echo "  Calibration: $(basename "$CALIBRATION")"
EXTRA_ROS_ARGS=()
if [[ -n "$MOCAP_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "mocap_topic:=${MOCAP_TOPIC}")
fi
if [[ -n "$VIO_TOPIC" ]]; then
    EXTRA_ROS_ARGS+=(-p "vio_topic:=${VIO_TOPIC}")
fi
for param in ${NODE_PARAMS[@]+"${NODE_PARAMS[@]}"}; do
    EXTRA_ROS_ARGS+=(-p "$param")
done
setsid env RUST_LOG="${RUST_LOG:-info}" "$VIO_BIN" --ros-args \
    --params-file "$CALIBRATION" \
    -p "echo_config_path:=${ECHO_CONFIG}" \
    -p "imu_topic:=${IMU_TOPIC}" \
    -p "image_topic:=${IMAGE_TOPIC}" \
    -p "patch_depth_enabled:=true" \
    -p "occupancy_enabled:=${OCCUPANCY}" \
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

set +u
# shellcheck disable=SC1090
source "$DECODER_WS/install_humble/setup.bash"
set -u

echo "Starting ${DECODER} and the bag-time relay..."
# Output mono8 — echo-li only needs grayscale; skipping BGR conversion
# saves the decoder's videoconvert and echo-li's to_gray(), cutting image
# bandwidth by 3x and removing two per-frame color conversions.
setsid ros2 run voxl_h265_decoder h265_decoder_node \
    --ros-args \
    -p "decoder:=${DECODER}" \
    -p "output_encoding:=mono8" \
    >"$LOG_DIR/decoder.log" 2>&1 &
DECODER_PID=$!
PIDS+=("$DECODER_PID")

setsid env RUST_LOG="${RUST_LOG:-info}" "$RELAY_BIN" --ros-args \
    -p "image_stamp_mode:=${IMAGE_STAMP_MODE}" \
    -p "output_imu_topic:=${RELAY_IMU_TOPIC}" \
    -p "output_image_topic:=${RELAY_IMAGE_TOPIC}" \
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

if [[ -n "$WORLD_FRAME" ]]; then
    read -r WR WP WY <<<"$WORLD_RPY"
    echo "Static TF: ${WORLD_FRAME} -> echo_li_odom (rpy ${WR:-0} ${WP:-0} ${WY:-0} rad)"
    setsid ros2 run tf2_ros static_transform_publisher \
        --frame-id "$WORLD_FRAME" --child-frame-id echo_li_odom \
        --x 0 --y 0 --z 0 --roll "${WR:-0}" --pitch "${WP:-0}" --yaw "${WY:-0}" \
        --ros-args -r __node:=echo_li_world_tf \
        >"$LOG_DIR/static_tf.log" 2>&1 &
    VIEW_PIDS+=("$!")
fi

if [[ -z "$RVIZ_CFG" ]]; then
    RVIZ_CFG="$SCRIPT_DIR/echo-li-ros2-rs/config/echo_li_voxl2.rviz"
fi
case "$RVIZ_VIEW" in
    follow|world) ;;
    *) echo "Unknown --rviz-view: $RVIZ_VIEW (expected follow or world)" >&2; exit 1 ;;
esac
if [[ "$USE_RVIZ" -eq 1 && -f "$RVIZ_CFG" && "$RVIZ_VIEW" != "follow" ]]; then
    # The config starts in the follow view; for --rviz-view world, promote the
    # saved debugging view to Current in a copy under the log directory.
    if RVIZ_WORLD_CFG=$(python3 - "$RVIZ_CFG" "$LOG_DIR/rviz_world.rviz" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
views = cfg["Visualization Manager"]["Views"]
wanted = next(v for v in views["Saved"] if v["Name"].startswith("World"))
views["Current"] = dict(wanted)
yaml.safe_dump(cfg, open(dst, "w"), default_flow_style=False)
print(dst)
PY
    ); then
        RVIZ_CFG="$RVIZ_WORLD_CFG"
    else
        echo "Could not build the world view config; starting in the follow view." >&2
    fi
fi
if [[ "$USE_RVIZ" -eq 1 && -f "$RVIZ_CFG" ]] && command -v rviz2 >/dev/null 2>&1; then
    RVIZ_ARGS=(-d "$RVIZ_CFG")
    if [[ -n "$WORLD_FRAME" ]]; then
        RVIZ_ARGS+=(-f "$WORLD_FRAME")
    fi
    for pid in $(pgrep -x rviz2 2>/dev/null || true); do
        domain="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null |
            sed -n 's/^ROS_DOMAIN_ID=//p')"
        if [[ "${domain:-42}" == "$ROS_DOMAIN_ID" ]]; then
            RVIZ_PID="$pid"
            break
        fi
    done
    if [[ -n "$RVIZ_PID" ]]; then
        echo "Reusing the rviz2 on domain ${ROS_DOMAIN_ID} (pid ${RVIZ_PID});" \
            "close it to pick up a new config or view."
    else
        setsid env QT_QPA_PLATFORM=xcb rviz2 "${RVIZ_ARGS[@]}" \
            >"$LOG_DIR/rviz2.log" 2>&1 &
        RVIZ_PID=$!
        RVIZ_OWN_PID=$!
    fi
fi

echo "Playing $BAG_PATH at ${RATE}x. Press Ctrl-C to stop."
echo "  Odometry: /echo_li/odometry (frame: echo_li_odom)"
echo "  Landmarks: /echo_li/landmarks"
echo "  Trajectory: /echo_li/path"
if [[ -n "$MOCAP_TOPIC" ]]; then
    echo "  Mocap: /echo_li/mocap_path (fitted from ${MOCAP_TOPIC})"
fi
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
grep -E 'H\.265 (totals:|\[)' "$LOG_DIR/decoder.log" | tail -1 || true
echo "Relay summary:"
grep 'relay:' "$LOG_DIR/relay.log" | tail -1 || true
echo "ECHO-LI summary:"
grep 'input:' "$LOG_DIR/echo_li.log" | tail -1 || true
echo "Trajectory: $LOG_DIR/trajectory.tum"

if [[ -n "$GT_TRAJECTORY" && -f "$LOG_DIR/trajectory.tum" ]]; then
    T_BC=$(python3 -c "
import yaml
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
    python3 "$SCRIPT_DIR/tools/eval_ate.py" "$LOG_DIR/trajectory.tum" "$GT_TRAJECTORY" \
        --max-gap 0.1 --align se3 "${T_BC_ARGS[@]}"
fi

echo "Logs: $LOG_DIR"

if [[ "$BAG_STATUS" -ne 0 && "$BAG_STATUS" -ne 130 ]]; then
    echo "ros2 bag play exited with status $BAG_STATUS" >&2
    exit "$BAG_STATUS"
fi
