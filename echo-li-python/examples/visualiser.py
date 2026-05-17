"""Level 1 Global Manifold trajectory visualizer.

Shows ground truth (cyan), estimated trajectory (red, Umeyama-aligned),
persistent map points (gray), and active landmarks (yellow).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


def align_umeyama(src: np.ndarray, dst: np.ndarray):
    """SVD-based rigid alignment: find (R, t) minimizing ||R·src + t - dst||².

    Returns (R [3x3], t [3]).
    """
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)

    sigma_sq = np.mean(np.sum((src - mu_s) ** 2, axis=1))
    H = (dst - mu_d).T @ (src - mu_s) / n

    U, S, Vt = np.linalg.svd(H)
    S_sign = np.eye(3)
    if np.linalg.det(H) < 0:
        S_sign[2, 2] = -1.0
        scale_sum = S[0] + S[1] - S[2]
    else:
        scale_sum = S.sum()

    R = U @ S_sign @ Vt
    s = scale_sum / sigma_sq if sigma_sq > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t


class TrajectoryVisualiser:
    def __init__(self, gt_positions: np.ndarray | None = None,
                 gt_times: np.ndarray | None = None,
                 update_interval: int = 5):
        self.app = pg.mkQApp("ECHO-LI Trajectory")
        pg.setConfigOptions(antialias=True)

        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("ECHO-LI: Global Manifold")
        self.win.resize(900, 700)
        layout = QtWidgets.QVBoxLayout()
        self.win.setLayout(layout)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=20, elevation=30, azimuth=-90)
        layout.addWidget(self.view)

        grid = gl.GLGridItem()
        grid.scale(2, 2, 2)
        self.view.addItem(grid)

        self.gt_line = gl.GLLinePlotItem(color=(0, 1, 1, 0.7), width=2.0)
        self.est_line = gl.GLLinePlotItem(color=(1, 0, 0, 1.0), width=2.5)
        self.map_points = gl.GLScatterPlotItem(color=(0.5, 0.5, 0.5, 0.8), size=3.0)
        self.active_points = gl.GLScatterPlotItem(color=(1.0, 1.0, 0.0, 1.0), size=5.0)

        for item in (self.gt_line, self.est_line, self.map_points, self.active_points):
            self.view.addItem(item)

        self.win.show()

        self.gt_positions = gt_positions
        self.gt_times = gt_times

        self.est_positions: list[np.ndarray] = []
        self.est_times: list[float] = []

        self.point_lifetime: dict[int, int] = {}
        self.persistent_points: dict[int, np.ndarray] = {}

        self.update_interval = update_interval
        self._frame_count = 0
        self._R_align = np.eye(3)
        self._t_align = np.zeros(3)

    def _calculate_alignment(self):
        if self.gt_positions is None or len(self.est_positions) < 100:
            return
        est = np.array(self.est_positions)
        t_est = np.array(self.est_times)

        matched_est, matched_gt = [], []
        gt_idx = 0
        for i, t in enumerate(t_est):
            while gt_idx < len(self.gt_times) - 1 and self.gt_times[gt_idx] < t:
                gt_idx += 1
            if gt_idx < len(self.gt_times):
                matched_est.append(est[i])
                matched_gt.append(self.gt_positions[gt_idx])

        if len(matched_est) > 10:
            self._R_align, self._t_align = align_umeyama(
                np.array(matched_est), np.array(matched_gt)
            )

    def update(self, timestamp: float, position: np.ndarray,
               landmarks: dict[int, np.ndarray] | None = None):
        """Call once per vision frame.

        Args:
            timestamp: frame time in seconds
            position: estimated position (3,) from VIOFilter.get_pose()
            landmarks: {id: global_position} from VIOFilter.get_landmarks()
        """
        self.est_positions.append(position.copy())
        self.est_times.append(timestamp)
        self._frame_count += 1

        if landmarks is not None:
            current_ids = set()
            for lid, pos in landmarks.items():
                current_ids.add(lid)
                self.point_lifetime[lid] = self.point_lifetime.get(lid, 0) + 1
                if self.point_lifetime[lid] > 3:
                    self.persistent_points[lid] = pos.copy()
            lost = set(self.point_lifetime) - current_ids
            for lid in lost:
                self.point_lifetime.pop(lid, None)

        if self._frame_count % self.update_interval == 0:
            self._calculate_alignment()
            self._redraw(landmarks)

        self.app.processEvents()

    def _redraw(self, landmarks):
        R, t = self._R_align, self._t_align

        if self.gt_positions is not None and len(self.est_times) > 0:
            mask = self.gt_times <= self.est_times[-1]
            if np.any(mask):
                self.gt_line.setData(pos=self.gt_positions[mask])

        if len(self.est_positions) > 1:
            est = np.array(self.est_positions, dtype=np.float32)
            aligned = (R @ est.T).T + t
            self.est_line.setData(pos=aligned)

        if self.persistent_points:
            pts = np.array(list(self.persistent_points.values()), dtype=np.float32)
            aligned = (R @ pts.T).T + t
            self.map_points.setData(pos=aligned)

        if landmarks:
            pts = np.array(list(landmarks.values()), dtype=np.float32)
            aligned = (R @ pts.T).T + t
            self.active_points.setData(pos=aligned)
        else:
            self.active_points.setData(pos=np.empty((0, 3)))

    def finish(self):
        print("Dataset finished. Close the window to exit.")
        pg.exec()
