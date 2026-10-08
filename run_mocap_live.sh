#!/usr/bin/env bash
# Show the arena live in rviz2 with the same layout the bag replay uses: the
# tracked pillars as solid boxes, the drone's mocap pose and trail, and the
# drone's own VIO (/echo_li/odometry, /echo_li/path, landmarks from the Orin)
# when it is on.
#
#   ./run_mocap_live.sh                    # domain 0, network, six pillars 1.05 m
#   ./run_mocap_live.sh --box-size "0.40 1.05 0.50" --boxes box1,box2
#
# Frames: the layout's fixed frame is the VIO's echo_li_odom (the Orin's node
# publishes echo_li_odom -> imu_link). tools/mocap_live_tf.py fits the mocap
# track onto the VIO track and publishes echo_li_odom -> mocap_map (heading +
# origin, refreshed every second); mocap_map -> world (roll +90 deg, Y-up to
# Z-up) carries the VRPN boxes. With the drone off, the mocap pose stands in
# for imu_link. Same chain the replay builds from its mocap fit.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOX_SIZE="0.40 1.05 0.50"
BOX_ANCHOR="top"
BOXES="auto"
DRONE_TOPIC="/mocap_drone_01/vision_pose/pose"
ODOM_TOPIC="/echo_li/odometry"
NAMES_TOPIC="/mocap_obstacles/names"
RVIZ_CFG="$SCRIPT_DIR/echo-li-ros2/config/echo_li_voxl2.rviz"
USE_RVIZ=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box-size) BOX_SIZE="$2"; shift 2 ;;
        --box-anchor) BOX_ANCHOR="$2"; shift 2 ;;
        --boxes) BOXES="$2"; shift 2 ;;
        --drone-topic) DRONE_TOPIC="$2"; shift 2 ;;
        --odom-topic) ODOM_TOPIC="$2"; shift 2 ;;
        --rviz-config) RVIZ_CFG="$2"; shift 2 ;;
        --no-rviz) USE_RVIZ=0; shift ;;
        -h|--help) sed -n 2,12p "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# shellcheck disable=SC1091
set +u; source /opt/ros/humble/setup.bash; set -u   # the setup script reads unset variables
PIDS=()
cleanup() {
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true; done
    sleep 1
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

if [[ "$BOXES" == "auto" ]]; then
    BOXES="$(timeout 8 ros2 topic echo --once --field data "$NAMES_TOPIC" 2>/dev/null | head -1 | tr -d ' ' || true)"
    if [[ -z "$BOXES" ]]; then
        echo "no obstacle names on ${NAMES_TOPIC} (domain ${ROS_DOMAIN_ID}); pass --boxes box1,box2,..." >&2
        exit 1
    fi
fi
read -r BX BY BZ <<<"$BOX_SIZE"
echo "Domain ${ROS_DOMAIN_ID}, boxes ${BOXES} as ${BX} x ${BY} x ${BZ} m (Y-up, anchor ${BOX_ANCHOR}), mocap ${DRONE_TOPIC}, VIO ${ODOM_TOPIC}"

# Each helper in its own process group so cleanup reaches the real binaries
# behind the `ros2 run` wrappers.
setsid python3 "$SCRIPT_DIR/tools/box_markers.py" --ros-args \
    -p "bodies:=[${BOXES}]" -p "size:=[${BX}, ${BY}, ${BZ}]" -p height_axis:=1 -p "anchor:=${BOX_ANCHOR}" \
    >/tmp/mocap_live_boxes.log 2>&1 &
PIDS+=("$!")
setsid ros2 run tf2_ros static_transform_publisher --frame-id mocap_map --child-frame-id world \
    --x 0 --y 0 --z 0 --roll 1.5707963 --pitch 0 --yaw 0 --ros-args -r __node:=mocap_live_world_tf >/dev/null 2>&1 &
PIDS+=("$!")
setsid python3 "$SCRIPT_DIR/tools/mocap_live_tf.py" --ros-args -p "mocap_topic:=${DRONE_TOPIC}" -p "odom_topic:=${ODOM_TOPIC}" \
    >/tmp/mocap_live_tf.log 2>&1 &
PIDS+=("$!")
sleep 1
if [[ "$USE_RVIZ" -eq 1 ]]; then
    rviz2 -d "$RVIZ_CFG"
else
    echo "helpers running (no rviz2); Ctrl-C to stop"
    wait
fi
