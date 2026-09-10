#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INTERNAL_ID=""
REQUESTED_ROS_DISTRO=""
IMU_TOPIC="/voxl/raw_imu"
IMAGE_TOPIC="/tracking_front/decoded"
# CAMERA_OFFSET="-0.0264"
CAMERA_OFFSET="0.004"
DEPTH_BACKEND="dis"
FORCE_REBUILD=0
VISUALIZE=0
RERUN_URL=""
ROS_ARGS=()

usage() {
    cat <<'EOF'
Run real-time ECHO-LI on VOXL2 ROS 2 topics.

Usage:
  ./run_voxl2_ros2.sh --internal-id {1|2} [options] [-- ROS_ARGS...]

Required:
  --internal-id ID       Physical VOXL2 internal ID from its sticker.

Options:
  --ros-distro DISTRO    Force jazzy, humble, or foxy.
  --imu-topic TOPIC      IMU topic (default: /voxl/raw_imu).
  --image-topic TOPIC    Decoded image topic
                         (default: /tracking_front/decoded).
  --camera-offset SEC    Camera timestamp offset (default: -0.0264).
  --depth-backend NAME   Depth estimation backend: dis or patch (default: dis).
  --visualize            Stream live diagnostics to Rerun 0.31.3.
  --rerun-url URL        Connect to a remote Rerun viewer; implies --visualize.
                         Example: rerun+http://192.168.1.112:9876/proxy
  --rebuild              Rebuild the Rust binding and ROS package.
  -h, --help             Show this help.

Examples:
  ./run_voxl2_ros2.sh --internal-id 2
  ./run_voxl2_ros2.sh --internal-id 1 --ros-distro humble
  ./run_voxl2_ros2.sh --internal-id 2 --camera-offset -0.0264
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
        --internal-id)
            need_value "$@"
            INTERNAL_ID="$2"
            shift 2
            ;;
        --ros-distro)
            need_value "$@"
            REQUESTED_ROS_DISTRO="$2"
            shift 2
            ;;
        --imu-topic)
            need_value "$@"
            IMU_TOPIC="$2"
            shift 2
            ;;
        --image-topic)
            need_value "$@"
            IMAGE_TOPIC="$2"
            shift 2
            ;;
        --camera-offset)
            need_value "$@"
            CAMERA_OFFSET="$2"
            shift 2
            ;;
        --depth-backend)
            need_value "$@"
            DEPTH_BACKEND="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=1
            shift
            ;;
        --rerun-url)
            need_value "$@"
            RERUN_URL="$2"
            VISUALIZE=1
            shift 2
            ;;
        --rebuild)
            FORCE_REBUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            ROS_ARGS=("$@")
            break
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

if [[ ! "$CAMERA_OFFSET" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]]; then
    echo "--camera-offset must be a signed number of seconds" >&2
    exit 2
fi

if [[ "$DEPTH_BACKEND" != "dis" && "$DEPTH_BACKEND" != "patch" ]]; then
    echo "--depth-backend must be dis or patch" >&2
    exit 2
fi

source_ros() {
    local distro="$1"
    local setup="/opt/ros/${distro}/setup.bash"
    if [[ ! -r "$setup" ]]; then
        return 1
    fi
    set +u
    # shellcheck disable=SC1090
    source "$setup"
    set -u
}

if [[ -n "$REQUESTED_ROS_DISTRO" ]]; then
    case "$REQUESTED_ROS_DISTRO" in
        jazzy|humble|foxy) ;;
        *)
            echo "--ros-distro must be jazzy, humble, or foxy" >&2
            exit 2
            ;;
    esac
    if ! source_ros "$REQUESTED_ROS_DISTRO"; then
        echo "ROS 2 ${REQUESTED_ROS_DISTRO} is not installed under /opt/ros." >&2
        exit 1
    fi
elif [[ -n "${ROS_DISTRO:-}" ]] && command -v ros2 >/dev/null 2>&1; then
    case "$ROS_DISTRO" in
        jazzy|humble|foxy) ;;
        *)
            echo "Active ROS_DISTRO=${ROS_DISTRO} is unsupported." >&2
            exit 1
            ;;
    esac
else
    FOUND_ROS=0
    for distro in jazzy humble foxy; do
        if source_ros "$distro"; then
            FOUND_ROS=1
            break
        fi
    done
    if [[ "$FOUND_ROS" -ne 1 ]]; then
        cat >&2 <<'EOF'
No supported ROS 2 installation was found.

The native Arch host currently has no compatible rclpy installation. Run this
command inside the Ubuntu/Jazzy distrobox, or on the Orin with ROS 2 Humble.
EOF
        exit 1
    fi
fi

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 was sourced, but ros2 is unavailable on PATH." >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
    exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "Cargo is required. Install a current Rust toolchain with rustup." >&2
    exit 1
fi
if ! command -v colcon >/dev/null 2>&1; then
    echo "colcon is required (install python3-colcon-common-extensions)." >&2
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 8))'; then
    echo "Python 3.8 or newer is required; found ${PYTHON_VERSION}." >&2
    exit 1
fi
if [[ "$VISUALIZE" -eq 1 ]] &&
        ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Rerun 0.31.3 requires Python 3.10 or newer; found ${PYTHON_VERSION}." >&2
    echo "Use ROS 2 Humble or Jazzy for visualization." >&2
    exit 1
fi
if ! python3 -c 'import rclpy, sensor_msgs, nav_msgs, tf2_ros'; then
    echo "The ROS ${ROS_DISTRO} Python environment is incomplete." >&2
    exit 1
fi

CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/echo-li"
RUNTIME_KEY="ros2-${ROS_DISTRO}-py${PYTHON_VERSION}"
RUNTIME_DIR="${CACHE_ROOT}/${RUNTIME_KEY}"
VENV_DIR="${RUNTIME_DIR}/venv"
WHEEL_DIR="${RUNTIME_DIR}/wheels"
COLCON_DIR="${RUNTIME_DIR}/colcon"
STAMP_BINDING="${RUNTIME_DIR}/binding.stamp"
STAMP_ROS="${RUNTIME_DIR}/ros-package.stamp"

mkdir -p "$RUNTIME_DIR" "$WHEEL_DIR" "$COLCON_DIR"

if [[ ! -x "${VENV_DIR}/bin/python" ]] ||
        ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Creating ROS-aware Python environment: ${VENV_DIR}"
    if ! python3 -m venv --clear --system-site-packages "$VENV_DIR"; then
        echo "Install python3-venv for ROS ${ROS_DISTRO}, then retry." >&2
        exit 1
    fi
fi

if ! "${VENV_DIR}/bin/python" -c 'import maturin' >/dev/null 2>&1; then
    echo "Installing Maturin into the runtime environment..."
    "${VENV_DIR}/bin/python" -m pip install 'maturin>=1,<2'
fi

if [[ "$VISUALIZE" -eq 1 ]] &&
        ! "${VENV_DIR}/bin/python" -c \
            'import rerun; raise SystemExit(rerun.__version__ != "0.31.3")' \
            >/dev/null 2>&1; then
    echo "Installing Rerun 0.31.3 into the runtime environment..."
    "${VENV_DIR}/bin/python" -m pip install 'rerun-sdk==0.31.3'
fi
if [[ "$VISUALIZE" -eq 1 ]] &&
        ! "${VENV_DIR}/bin/python" -c \
            'import pandas; raise SystemExit(int(pandas.__version__.split(".")[0]) < 2)' \
            >/dev/null 2>&1; then
    echo "Installing Pandas >= 2 for Rerun..."
    "${VENV_DIR}/bin/python" -m pip install 'pandas>=2,<3'
fi

# The system python3-opencv (apt) is compiled against the system NumPy.  pip
# packages above may have pulled a newer NumPy into the venv, making the
# system cv2 ABI-incompatible.  Fall back to a PyPI build when that happens.
if ! "${VENV_DIR}/bin/python" -c 'import cv2' >/dev/null 2>&1; then
    echo "Installing opencv-python-headless (system cv2 incompatible with runtime NumPy)..."
    "${VENV_DIR}/bin/python" -m pip install 'opencv-python-headless>=4.8'
fi

# The PyO3/numpy binding's ABI depends on the NumPy major version at compile
# time (rust-numpy 0.25 supports both 1.x and 2.x, but the wheel is specific).
# Record the major version in the stamp so we rebuild when it changes.
NUMPY_MAJOR="$("${VENV_DIR}/bin/python" -c \
    'import numpy; print(numpy.__version__.split(".")[0])' 2>/dev/null || echo unknown)"
STAMP_NUMPY="${RUNTIME_DIR}/numpy-major.stamp"

BINDING_CHANGED=0
if [[ "$FORCE_REBUILD" -eq 1 || ! -f "$STAMP_BINDING" ]]; then
    BINDING_CHANGED=1
elif [[ ! -f "$STAMP_NUMPY" || "$(cat "$STAMP_NUMPY")" != "$NUMPY_MAJOR" ]]; then
    echo "NumPy major version changed → rebuilding binding..."
    BINDING_CHANGED=1
elif find \
        "$SCRIPT_DIR/echo-li-python" \
        "$SCRIPT_DIR/echo-li-core" \
        "$SCRIPT_DIR/echo-lie" \
        "$SCRIPT_DIR/Cargo.toml" \
        "$SCRIPT_DIR/Cargo.lock" \
        -type f -newer "$STAMP_BINDING" -print -quit | grep -q .; then
    BINDING_CHANGED=1
elif ! "${VENV_DIR}/bin/python" -c 'import echo_li' >/dev/null 2>&1; then
    BINDING_CHANGED=1
fi

if [[ "$BINDING_CHANGED" -eq 1 ]]; then
    echo "Building the ECHO-LI Python binding for Python ${PYTHON_VERSION}..."
    rm -f "${WHEEL_DIR}"/echo_li-*.whl
    CARGO_TARGET_DIR="${RUNTIME_DIR}/cargo-target" \
    "${VENV_DIR}/bin/python" -m maturin build \
        --release \
        --compatibility linux \
        --manifest-path "$SCRIPT_DIR/echo-li-python/Cargo.toml" \
        --interpreter "${VENV_DIR}/bin/python" \
        --out "$WHEEL_DIR"
    WHEEL_PATH="$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'echo_li-*.whl' -print -quit)"
    if [[ -z "$WHEEL_PATH" ]]; then
        echo "Maturin completed without producing an ECHO-LI wheel." >&2
        exit 1
    fi
    "${VENV_DIR}/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL_PATH"
    touch "$STAMP_BINDING"
    echo "$NUMPY_MAJOR" > "$STAMP_NUMPY"
fi

ROS_PACKAGE_CHANGED=0
if [[ "$FORCE_REBUILD" -eq 1 || ! -f "$STAMP_ROS" ]]; then
    ROS_PACKAGE_CHANGED=1
elif find "$SCRIPT_DIR/echo-li-ros2" -type f \
        -newer "$STAMP_ROS" -print -quit | grep -q .; then
    ROS_PACKAGE_CHANGED=1
fi

if [[ "$ROS_PACKAGE_CHANGED" -eq 1 ]]; then
    echo "Building the ECHO-LI ROS 2 package..."
    colcon --log-base "$COLCON_DIR/log" build \
        --base-paths "$SCRIPT_DIR/echo-li-ros2" \
        --build-base "$COLCON_DIR/build" \
        --install-base "$COLCON_DIR/install" \
        --symlink-install \
        --packages-select echo_li_ros2
    touch "$STAMP_ROS"
fi

set +u
# shellcheck disable=SC1090
source "$COLCON_DIR/install/setup.bash"
set -u

VENV_SITE="$(${VENV_DIR}/bin/python -c \
    'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${VENV_SITE}:${VENV_SITE}/rerun_sdk${PYTHONPATH:+:${PYTHONPATH}}"

CALIBRATION="$SCRIPT_DIR/echo-li-ros2/config/voxl2_internal_id_${INTERNAL_ID}.yaml"
ECHO_CONFIG="$SCRIPT_DIR/echo-li-ros2/config/eqvio_voxl2.yaml"

echo "Starting ECHO-LI with ROS ${ROS_DISTRO}, Python ${PYTHON_VERSION}, internal ID ${INTERNAL_ID}."
echo "IMU: ${IMU_TOPIC}"
echo "Image: ${IMAGE_TOPIC}"
echo "Camera offset: ${CAMERA_OFFSET} s"
echo "Depth backend: ${DEPTH_BACKEND}"
if [[ "$VISUALIZE" -eq 1 ]]; then
    if [[ -n "$RERUN_URL" ]]; then
        echo "Rerun: ${RERUN_URL}"
    else
        echo "Rerun: local viewer"
    fi
fi

RERUN_ROS_ARGS=()
if [[ -n "$RERUN_URL" ]]; then
    RERUN_ROS_ARGS=(-p "rerun_url:=${RERUN_URL}")
fi

exec ros2 run echo_li_ros2 voxl2_vio_node \
    --ros-args \
    --params-file "$CALIBRATION" \
    -p "echo_config_path:=${ECHO_CONFIG}" \
    -p "imu_topic:=${IMU_TOPIC}" \
    -p "image_topic:=${IMAGE_TOPIC}" \
    -p "camera_time_offset_sec:=${CAMERA_OFFSET}" \
    -p "depth_backend:=${DEPTH_BACKEND}" \
    -p "rerun_enabled:=$([[ "$VISUALIZE" -eq 1 ]] && echo true || echo false)" \
    "${RERUN_ROS_ARGS[@]}" \
    "${ROS_ARGS[@]}"
