# echo-li-python

Python bindings for ECHO-LI via PyO3/maturin.

## Install

```bash
cd echo-li-python
pip install .
```

For development (faster rebuilds, no wheel packaging):
```bash
pip install maturin
maturin develop --release
```

Runtime dependencies (`numpy`) are pulled automatically. For the EuRoC demo:
```bash
pip install opencv-python pyyaml
```

For the trajectory visualizer (optional):
```bash
pip install pyqtgraph PyQt6 PyOpenGL
```

## Usage

```python
import echo_li
import numpy as np

# Camera models
cam = echo_li.RadTanCamera(fx, fy, cx, cy, k1, k2, p1, p2)
# or
cam = echo_li.PinholeCamera(fx, fy, cx, cy)

# Frontend (feature tracking)
config = echo_li.FrontendConfig(max_features=200)
config.set_camera(fx, fy, cx, cy, width, height, dist_coeffs)
tracker = echo_li.Frontend(config, width, height)

features, stats = tracker.process(gray_image)  # numpy uint8 (H, W)

# FrontendConfig can also load from the YAML used by echo-li-cli:
config = echo_li.FrontendConfig.from_yaml("configs/eqvio_euroc_euclid.yaml")

# VIO filter
vio = echo_li.VIOFilter("configs/eqvio_euroc_euclid.yaml", cam)
vio.set_camera_extrinsics(T_BS_4x4)  # numpy float64 (4, 4) from sensor.yaml

vio.process_imu(stamp, [wx, wy, wz], [ax, ay, az])
vio.process_vision(stamp, {feature_id: (u, v), ...})

pos, quat = vio.get_pose()       # numpy arrays
vel = vio.get_velocity()
landmarks = vio.get_landmarks()   # {id: position[3]}
```

## EuRoC demo

```bash
cd examples
python euroc_tracking.py /path/to/V1_01_easy

# With VIO and trajectory visualization:
python euroc_tracking.py /path/to/V1_01_easy --config ../../configs/eqvio_euroc_euclid.yaml
```

Options:
- `--config` : YAML config file (enables VIO filter + trajectory visualizer)
- `--max-features` : override max features (default: 200, or from YAML if `--config`)
- `--no-display` : disable OpenCV window
