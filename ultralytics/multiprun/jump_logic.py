from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

from .config import Config
from .utils_pose import safe_percentile


class JumpRopeState(Enum):
    IDLE = "IDLE"
    JUMPING = "JUMPING"
    STOPPED_ACTIVE = "STOPPED_ACTIVE"
    STOPPED_TRIPPED = "STOPPED_TRIPPED"


class KalmanFilter2D:
    """Constant velocity KF: state=[x,y,vx,vy], measurement=[x,y]."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state = np.zeros(4, dtype=np.float32)  # [x,y,vx,vy]
        self.cov = np.eye(4, dtype=np.float32) * 1000.0

        self.F = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False

    def init(self, x: float, y: float):
        self.state[:] = (x, y, 0.0, 0.0)
        self.cov = np.eye(4, dtype=np.float32) * 1000.0
        self.initialized = True

    def update(self, x: float, y: float) -> np.ndarray:
        if not self.initialized:
            self.init(x, y)
            return self.state[:2].copy()

        # predict
        self.state = self.F @ self.state
        self.cov = self.F @ self.cov @ self.F.T + self.Q

        # correct
        z = np.array([x, y], dtype=np.float32)
        y_res = z - (self.H @ self.state)
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_res
        self.cov = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.cov
        return self.state[:2].copy()


class JumpDetector:
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = float(fps)

        # Hips smoother
        self.kf_lhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_rhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)

        # Ankles more responsive
        self.kf_lank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_rank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)

        # Head medium
        self.kf_nose = KalmanFilter2D(process_noise=0.01, measurement_noise=0.10)
        self.kf_leye = KalmanFilter2D(process_noise=0.01, measurement_noise=0.10)
        self.kf_reye = KalmanFilter2D(process_noise=0.01, measurement_noise=0.10)

        self.hip_y_hist = deque(maxlen=int(self.fps * 2.0))
        self.ankle_y_hist = deque(maxlen=int(self.fps * self.cfg.ground_history_seconds))
        self.ankle_dist_hist = deque(maxlen=10)

        self.prev_head_xy: Optional[np.ndarray] = None
        self.normal_lowest_head_y: Optional[float] = None
        self.jumping_head_y_samples = deque(maxlen=int(self.fps * 2.0))

        self.airborne_frames = 0
        self.ground_frames = 0
        self.is_airborne = False

        self.jump_count = 0
        self.single_count = 0
        self.double_count = 0
        self.last_count_frame = -10_000
        self.jump_times = deque(maxlen=50)

        self._last_ground_y: Optional[float] = None

    def _compute_ground_y(self) -> Optional[float]:
        if len(self.ankle_y_hist) < max(10, int(self.fps * 0.35)):
            return None
        arr = np.array(self.ankle_y_hist, dtype=np.float32)
        p90 = safe_percentile(arr, 90.0)
        p75 = safe_percentile(arr, 75.0)
        ground = 0.7 * p90 + 0.3 * p75
        return float(ground)

    def _smooth_head(self, nose, leye, reye) -> Optional[np.ndarray]:
        pts = []
        if nose is not None:
            pts.append(self.kf_nose.update(*nose))
        if leye is not None:
            pts.append(self.kf_leye.update(*leye))
        if reye is not None:
            pts.append(self.kf_reye.update(*reye))
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        left_hip, right_hip, left_ankle, right_ankle,
        nose, left_eye, right_eye,
        person_h: float
    ) -> Dict[str, Any]:
        lift_th = 0.013 * person_h
        hip_amp_min = 0.012 * person_h
        ankle_dist_th = 18
        head_drop_th = 0.025 * person_h

        out = {
            "hip_y": 0.0,
            "ankle_distance": 0.0,
            "is_airborne": False,
            "is_jumping": False,
            "jump_type": "unknown",
            "jump_count": self.jump_count,
            "single_count": self.single_count,
            "double_count": self.double_count,
            "freq_hz": 0.0,
            "spm": 0.0,
            "head_y": float("nan"),
            "head_move_px": 0.0,
            "head_drop_event": False,
            "ground_y": float("nan"),
            "lift_th": float(lift_th),
        }

        if not (left_hip and right_hip and left_ankle and right_ankle):
            return out

        lh = self.kf_lhip.update(*left_hip)
        rh = self.kf_rhip.update(*right_hip)
        la = self.kf_lank.update(*left_ankle)
        ra = self.kf_rank.update(*right_ankle)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        ankle_y = float((la[1] + ra[1]) / 2.0)

        dx = la[0] - ra[0]
        dy = la[1] - ra[1]
        ankle_dist = float(np.hypot(dx, dy))

        out["hip_y"] = hip_y
        out["ankle_distance"] = ankle_dist

        self.hip_y_hist.append(hip_y)
        self.ankle_y_hist.append(ankle_y)
        self.ankle_dist_hist.append(ankle_dist)

        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            out["jump_type"] = "single" if avg_dist >= ankle_dist_th else "double"

        head_xy = self._smooth_head(nose, left_eye, right_eye)
        if head_xy is not None:
            head_y = float(head_xy[1])
            out["head_y"] = head_y

            is_actively_jumping = False
            if self.jump_count > 0 and len(self.jump_times) > 0:
                is_actively_jumping = (t_sec - self.jump_times[-1]) < 2.0

            if is_actively_jumping:
                self.jumping_head_y_samples.append(head_y)
                if len(self.jumping_head_y_samples) >= int(self.fps * 0.5):
                    arr = np.array(self.jumping_head_y_samples, dtype=np.float32)
                    self.normal_lowest_head_y = safe_percentile(arr, 90.0)
            else:
                if len(self.jump_times) == 0 or (t_sec - self.jump_times[-1]) > 3.0:
                    self.normal_lowest_head_y = None
                    self.jumping_head_y_samples.clear()

            if self.prev_head_xy is not None:
                out["head_move_px"] = float(np.hypot(
                    head_xy[0] - self.prev_head_xy[0],
                    head_xy[1] - self.prev_head_xy[1]
                ))

            if self.normal_lowest_head_y is not None:
                out["head_drop_event"] = head_y > (self.normal_lowest_head_y + head_drop_th)

            self.prev_head_xy = head_xy

        ground_y = self._compute_ground_y()
        if ground_y is None:
            return out

        out["ground_y"] = float(ground_y)
        self._last_ground_y = float(ground_y)

        airborne_now = ankle_y <= (ground_y - lift_th * 1.8)

        if airborne_now:
            self.airborne_frames += 1
            self.ground_frames = 0
        else:
            self.ground_frames += 1
            self.airborne_frames = 0

        if (not self.is_airborne) and (self.airborne_frames >= self.cfg.airborne_confirm_frames):
            self.is_airborne = True

        if self.is_airborne and (self.ground_frames >= self.cfg.ground_confirm_frames):
            hip_amp_ok = False
            if len(self.hip_y_hist) >= int(self.fps * 0.4):
                recent = np.array(list(self.hip_y_hist)[-int(self.fps * 0.4):], dtype=np.float32)
                hip_amp = float(np.max(recent) - np.min(recent))
                hip_amp_ok = hip_amp >= hip_amp_min

            refractory_ok = (frame_idx - self.last_count_frame) >= self.cfg.refractory_frames

            if hip_amp_ok and refractory_ok:
                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(t_sec)

                if out["jump_type"] == "single":
                    self.single_count += 1
                elif out["jump_type"] == "double":
                    self.double_count += 1

            self.is_airborne = False

        out["is_airborne"] = self.is_airborne
        out["is_jumping"] = bool(self.is_airborne)

        if len(self.jump_times) >= 2:
            intervals = np.diff(np.array(self.jump_times, dtype=np.float32))
            if intervals.size > 0:
                avg = float(np.mean(intervals[-5:]))
                if avg > 0:
                    out["freq_hz"] = 1.0 / avg
                    out["spm"] = (1.0 / avg) * 60.0

        out["jump_count"] = self.jump_count
        out["single_count"] = self.single_count
        out["double_count"] = self.double_count
        return out


class JumpRopeStateMachine:
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = float(fps)
        self.state = JumpRopeState.IDLE
        self._last_seen_count = 0
        self.last_count_time: Optional[float] = None
        self.head_event_times = deque(maxlen=int(self.fps * 2.0))

    def _head_event_recent(self, t_sec: float) -> bool:
        t_min = t_sec - self.cfg.head_event_window_sec
        while self.head_event_times and self.head_event_times[0] < t_min:
            self.head_event_times.popleft()
        return len(self.head_event_times) > 0

    def update(self, t_sec: float, jump_count: int, head_drop_event: bool):
        if head_drop_event:
            self.head_event_times.append(t_sec)

        if jump_count > self._last_seen_count:
            self._last_seen_count = jump_count
            self.last_count_time = t_sec
            self.state = JumpRopeState.JUMPING
            return

        time_since_last = (t_sec - self.last_count_time) if self.last_count_time is not None else 999.0

        if self.state == JumpRopeState.IDLE and jump_count > 0:
            self.state = JumpRopeState.JUMPING

        if self.state == JumpRopeState.JUMPING and time_since_last > self.cfg.stop_sudden_sec:
            self.state = JumpRopeState.STOPPED_TRIPPED if self._head_event_recent(t_sec) else JumpRopeState.STOPPED_ACTIVE
