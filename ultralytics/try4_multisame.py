import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from openvino.runtime import Core

# ---------------- CONFIG ----------------
MODEL_PATH = "model_int8.onnx"
VIDEO_SOURCE = "TestVideos/1053.mp4"
IMG_SIZE = 640

DET_THRESH = 0.5
MIN_KPT_CONF = 0.0

# COCO-17 indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Jump logic
GROUND_HISTORY_SECONDS = 1.5
AIRBORNE_CONFIRM_FRAMES = 1
GROUND_CONFIRM_FRAMES = 1
REFRACTORY_FRAMES = 9
STOP_SUDDEN_SEC = 1.2

# ---------------- CYCLE CHECKS (ADDED: vote check) ----------------
# hard dt bounds
MIN_JUMP_INTERVAL_SEC = 0.23
MAX_JUMP_INTERVAL_SEC = 1.60

# cadence tracking (adaptive)
EWMA_ALPHA = 0.22

# amplitude gate (stop/walk suppression)  [kept as a base floor]
AMP_FRAC = 0.032
AMP_MIN_PX = 6.0

# ---------------- DYNAMIC LIFT THRESHOLD ----------------
LIFT_BASE_FRAC = 0.010
AMP_TO_LIFT = 0.35
NOISE_K = 3.0
MIN_LIFT_PX = 4.0
MAX_LIFT_FRAC = 0.050
AMP_EWMA_ALPHA = 0.25

# ---------------- HIP-BASED SETTINGS (FIXED) ----------------
HIP_LIFT_MAX_FRAC = 0.085
HIP_NOISE_BAND_FRAC = 0.18

# ---------------- SHOULDER AMP: DYNAMIC GATE (NEW) ----------------
# The old code used: shoulder_amp < amp_th (amp_th from person_h only).
# Now: shoulder_amp_th adapts using:
#   - base floor from person height (AMP_FRAC * person_h, AMP_MIN_PX)
#   - shoulder amplitude EWMA (tracks typical shoulder jump amplitude)
#   - shoulder ground noise sigma (robust MAD near ground)
SHO_AMP_EWMA_ALPHA = 0.35
SHO_AMP_TO_GATE = 0.50          # fraction of typical shoulder amp used as gate
SHO_AMP_NOISE_K = 3.0           # require amp to exceed noise by this factor
SHO_AMP_MAX_FRAC = 0.060        # cap threshold (fraction of person_h)
SHO_NOISE_BAND_FRAC = 0.25      # same idea as hip, but for shoulders

# Skeleton (drawing only)
SKELETON = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 8), (5, 6),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

# ---------------- STATE ----------------
class JumpRopeState(Enum):
    IDLE = "IDLE"
    JUMPING = "JUMPING"
    STOPPED = "STOPPED"


# ---------------- KALMAN FILTER ----------------
class KalmanFilter2D:
    """Constant velocity KF: state=[x,y,vx,vy], measurement=[x,y]."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.state = np.zeros(4, dtype=np.float32)
        self.cov = np.eye(4, dtype=np.float32) * 1000.0
        self.initialized = False

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

    def update(self, x: float, y: float) -> np.ndarray:
        if not self.initialized:
            self.state[:] = (x, y, 0.0, 0.0)
            self.cov = np.eye(4, dtype=np.float32) * 1000.0
            self.initialized = True
            return self.state[:2].copy()

        self.state = self.F @ self.state
        self.cov = self.F @ self.cov @ self.F.T + self.Q

        z = np.array([x, y], dtype=np.float32)
        y_res = z - (self.H @ self.state)
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_res
        self.cov = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.cov
        return self.state[:2].copy()


# ---------------- UTILS (normalized outputs) ----------------
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img


def decode_bbox_xyxy_norm(det57: np.ndarray) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2, score = det57[:5]
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return float(x1), float(y1), float(x2), float(y2), float(score)


def decode_kpts_17x3(det57: np.ndarray) -> np.ndarray:
    return det57[6:6 + 51].reshape(17, 3).astype(np.float32)


def map_norm_to_frame_xy(xy_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    out = xy_norm.astype(np.float32).copy()
    out[:, 0] *= float(w)
    out[:, 1] *= float(h)
    return out


def bbox_height_px(det57: np.ndarray, w: int, h: int) -> float:
    x1n, y1n, x2n, y2n, _ = decode_bbox_xyxy_norm(det57)
    return float(abs(y2n - y1n) * h)


def bbox_center_px(det57: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    x1n, y1n, x2n, y2n, _ = decode_bbox_xyxy_norm(det57)
    cxn = (x1n + x2n) / 2.0
    cyn = (y1n + y2n) / 2.0
    return float(cxn * w), float(cyn * h)


def bbox_xyxy_px(det57: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n, _ = decode_bbox_xyxy_norm(det57)
    x1 = int(max(0, min(w - 1, x1n * w)))
    y1 = int(max(0, min(h - 1, y1n * h)))
    x2 = int(max(0, min(w - 1, x2n * w)))
    y2 = int(max(0, min(h - 1, y2n * h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_dets(dets: np.ndarray, w: int, h: int, iou_th: float = 0.45) -> np.ndarray:
    """
    NMS on det57 array using bbox IoU. Keeps highest-conf boxes.
    This prevents duplicate detections of the same person from creating ID flips.
    """
    if dets.shape[0] <= 1:
        return dets

    boxes = np.array([bbox_xyxy_px(d, w, h) for d in dets], dtype=np.int32)
    scores = dets[:, 4].astype(np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        bi = tuple(boxes[i].tolist())

        ious = np.empty((rest.size,), dtype=np.float32)
        for k, j in enumerate(rest):
            bj = tuple(boxes[int(j)].tolist())
            ious[k] = iou_xyxy(bi, bj)

        order = rest[ious < iou_th]

    return dets[np.array(keep, dtype=np.int32)]


def get_kpt_xy(det57: np.ndarray, idx: int, w: int, h: int) -> Optional[Tuple[float, float]]:
    kpts = decode_kpts_17x3(det57)
    if float(kpts[idx, 2]) < MIN_KPT_CONF:
        return None
    xy = map_norm_to_frame_xy(kpts[:, :2], w, h)
    return float(xy[idx, 0]), float(xy[idx, 1])


def draw_pose(frame: np.ndarray, det57: np.ndarray, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n, _ = decode_bbox_xyxy_norm(det57)
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    kpts = decode_kpts_17x3(det57)
    xy = map_norm_to_frame_xy(kpts[:, :2], w, h)
    kc = kpts[:, 2]

    pts = [(int(xy[i, 0]), int(xy[i, 1])) for i in range(17)]
    for i in range(17):
        if float(kc[i]) >= MIN_KPT_CONF:
            cv2.circle(frame, pts[i], 3, (0, 0, 255), -1)

    for a, b in SKELETON:
        if float(kc[a]) >= MIN_KPT_CONF and float(kc[b]) >= MIN_KPT_CONF:
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)


# ---------------- JUMP DETECTOR ----------------
class JumpDetector:
    def __init__(self, fps: float):
        self.fps = float(fps)

        self.kf_lhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_rhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_lank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_rank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_lsho = KalmanFilter2D(process_noise=0.008, measurement_noise=0.14)
        self.kf_rsho = KalmanFilter2D(process_noise=0.008, measurement_noise=0.14)

        self.hip_y_hist = deque(maxlen=max(10, int(self.fps * GROUND_HISTORY_SECONDS)))
        self.shoulder_y_hist = deque(maxlen=max(10, int(self.fps * GROUND_HISTORY_SECONDS)))

        self.airborne_frames = 0
        self.ground_frames = 0
        self.is_airborne = False

        self.air_min_y: Optional[float] = None
        self.air_min_shoulder_y: Optional[float] = None

        self.jump_count = 0
        self.last_count_frame = -10_000

        self.jump_times = deque(maxlen=50)
        self.cycle_interval_hist = deque(maxlen=20)
        self.cycle_expected_interval: Optional[float] = None

        self.amp_ewma: Optional[float] = None
        self.shoulder_amp_ewma: Optional[float] = None  # NEW

    def _reset_cadence(self):
        self.jump_times.clear()
        self.cycle_interval_hist.clear()
        self.cycle_expected_interval = None

    @staticmethod
    def _robust_sigma(vals: np.ndarray) -> float:
        if vals.size == 0:
            return 0.0
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return 1.4826 * mad

    def _estimate_ground_noise_sigma(self, y_hist: deque, ground_y: float, band_frac: float) -> float:
        arr = np.array(y_hist, dtype=np.float32)
        if arr.size < 10:
            return 0.0

        span = float(arr.max() - arr.min())
        band = max(6.0, float(band_frac) * span)
        near = arr[arr >= (ground_y - band)]
        if near.size < 6:
            near = arr

        sigma = self._robust_sigma(near)
        return float(max(0.0, sigma))

    def _dynamic_lift_th(self, person_h: float, ground_sigma: float) -> float:
        if self.amp_ewma is None:
            lift_from_amp = LIFT_BASE_FRAC * person_h
        else:
            lift_from_amp = AMP_TO_LIFT * float(self.amp_ewma)

        lift_from_noise = NOISE_K * float(ground_sigma)

        lift_th = max(MIN_LIFT_PX, lift_from_amp, lift_from_noise)
        lift_th = min(lift_th, HIP_LIFT_MAX_FRAC * person_h)
        return float(lift_th)

    def _dynamic_shoulder_amp_th(self, person_h: float, shoulder_sigma: float) -> float:
        # base "stop/walk" floor (height-normalized)
        base = max(AMP_MIN_PX, AMP_FRAC * person_h)

        # adapt to this person's typical shoulder amplitude (if available)
        if self.shoulder_amp_ewma is None:
            from_ewma = 0.0
        else:
            from_ewma = SHO_AMP_TO_GATE * float(self.shoulder_amp_ewma)

        # noise-aware gate (reject tiny bumps)
        from_noise = SHO_AMP_NOISE_K * float(shoulder_sigma)

        th = max(base, from_ewma, from_noise)
        th = min(th, SHO_AMP_MAX_FRAC * person_h)
        return float(th)

    def update(self, frame_idx: int, t_sec: float,
               lhip, rhip, lank, rank, lsho, rsho, person_h: float) -> Dict[str, Any]:

        out = {
            "is_airborne": False,
            "jump_count": self.jump_count,
            "spm": 0.0,
            "ground_y": float("nan"),
            "lift_th": float("nan"),
            "lift_noise_sigma": float("nan"),
            "amp_ewma": float("nan"),
            "hip_y": float("nan"),
            "shoulder_y": float("nan"),
            "shoulder_ground_y": float("nan"),
            "shoulder_amp": float("nan"),
            "shoulder_amp_th": float("nan"),          # NEW
            "shoulder_noise_sigma": float("nan"),     # NEW
            "shoulder_amp_ewma": float("nan"),        # NEW
            "dt": float("nan"),
            "expected_dt": float("nan"),
            "amp": float("nan"),
            "min_air_y": float("nan"),
        }

        if not (lhip and rhip and lank and rank and lsho and rsho) or person_h <= 1.0:
            return out

        lh = self.kf_lhip.update(*lhip)
        rh = self.kf_rhip.update(*rhip)
        _la = self.kf_lank.update(*lank)
        _ra = self.kf_rank.update(*rank)
        ls = self.kf_lsho.update(*lsho)
        rs = self.kf_rsho.update(*rsho)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        shoulder_y = float((ls[1] + rs[1]) / 2.0)

        out["hip_y"] = hip_y
        out["shoulder_y"] = shoulder_y

        self.hip_y_hist.append(hip_y)
        self.shoulder_y_hist.append(shoulder_y)

        if len(self.hip_y_hist) < 10:
            return out

        # --- HIP ground + lift threshold (unchanged) ---
        ground_y = float(np.percentile(np.array(self.hip_y_hist, dtype=np.float32), 90.0))
        out["ground_y"] = ground_y

        hip_sigma = self._estimate_ground_noise_sigma(self.hip_y_hist, ground_y, HIP_NOISE_BAND_FRAC)
        lift_th = self._dynamic_lift_th(person_h, hip_sigma)
        out["lift_noise_sigma"] = float(hip_sigma)
        out["lift_th"] = float(lift_th)
        out["amp_ewma"] = float(self.amp_ewma) if self.amp_ewma is not None else float("nan")

        # --- SHOULDER ground + dynamic shoulder amp threshold (NEW) ---
        if len(self.shoulder_y_hist) >= 10:
            shoulder_ground_y = float(np.percentile(np.array(self.shoulder_y_hist, dtype=np.float32), 90.0))
            sho_sigma = self._estimate_ground_noise_sigma(self.shoulder_y_hist, shoulder_ground_y, SHO_NOISE_BAND_FRAC)
            shoulder_amp_th = self._dynamic_shoulder_amp_th(person_h, sho_sigma)
        else:
            shoulder_ground_y = float("nan")
            sho_sigma = 0.0
            shoulder_amp_th = max(AMP_MIN_PX, AMP_FRAC * person_h)

        out["shoulder_ground_y"] = shoulder_ground_y
        out["shoulder_noise_sigma"] = float(sho_sigma)
        out["shoulder_amp_th"] = float(shoulder_amp_th)
        out["shoulder_amp_ewma"] = float(self.shoulder_amp_ewma) if self.shoulder_amp_ewma is not None else float("nan")

        airborne_now = hip_y <= (ground_y - lift_th)

        if airborne_now:
            self.airborne_frames += 1
            self.ground_frames = 0
        else:
            self.ground_frames += 1
            self.airborne_frames = 0

        if (not self.is_airborne) and (self.airborne_frames >= AIRBORNE_CONFIRM_FRAMES):
            self.is_airborne = True
            self.air_min_y = hip_y
            self.air_min_shoulder_y = shoulder_y

        if self.is_airborne:
            self.air_min_y = hip_y if self.air_min_y is None else float(min(self.air_min_y, hip_y))
            self.air_min_shoulder_y = shoulder_y if self.air_min_shoulder_y is None else float(min(self.air_min_shoulder_y, shoulder_y))

        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

            min_air_y = self.air_min_y if self.air_min_y is not None else hip_y
            hip_amp = float(ground_y - float(min_air_y))
            out["amp"] = hip_amp
            out["min_air_y"] = float(min_air_y)

            min_air_sho = self.air_min_shoulder_y if self.air_min_shoulder_y is not None else shoulder_y
            if not np.isnan(shoulder_ground_y):
                shoulder_amp = float(shoulder_ground_y - float(min_air_sho))
            else:
                shoulder_amp = float("nan")
            out["shoulder_amp"] = shoulder_amp

            self.is_airborne = False
            self.air_min_y = None
            self.air_min_shoulder_y = None

            if refractory_ok:
                dt = None
                if len(self.jump_times) >= 1:
                    dt = float(t_sec - float(self.jump_times[-1]))
                    out["dt"] = dt

                    if dt < MIN_JUMP_INTERVAL_SEC:
                        return out
                    if dt > MAX_JUMP_INTERVAL_SEC:
                        self._reset_cadence()
                        return out

                # ---------------- DYNAMIC SHOULDER AMP GATE (UPDATED) ----------------
                # If shoulder_amp is nan (rare), fall back to hip_amp gate using the base floor.
                if np.isnan(shoulder_amp):
                    base_fallback = max(AMP_MIN_PX, AMP_FRAC * person_h)
                    if hip_amp < base_fallback:
                        return out
                else:
                    if shoulder_amp < shoulder_amp_th:
                        return out

                # count
                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(float(t_sec))

                # update cadence EWMA + store expected_dt
                if dt is not None:
                    self.cycle_interval_hist.append(float(dt))
                    if self.cycle_expected_interval is None:
                        self.cycle_expected_interval = float(dt)
                    else:
                        self.cycle_expected_interval = float(
                            (1.0 - EWMA_ALPHA) * self.cycle_expected_interval + EWMA_ALPHA * dt
                        )
                    out["expected_dt"] = float(self.cycle_expected_interval)

                # update HIP amp EWMA (unchanged)
                if self.amp_ewma is None:
                    self.amp_ewma = float(hip_amp)
                else:
                    self.amp_ewma = float((1.0 - AMP_EWMA_ALPHA) * self.amp_ewma + AMP_EWMA_ALPHA * hip_amp)
                out["amp_ewma"] = float(self.amp_ewma)

                # update SHOULDER amp EWMA (NEW)
                if not np.isnan(shoulder_amp):
                    if self.shoulder_amp_ewma is None:
                        self.shoulder_amp_ewma = float(shoulder_amp)
                    else:
                        self.shoulder_amp_ewma = float(
                            (1.0 - SHO_AMP_EWMA_ALPHA) * self.shoulder_amp_ewma + SHO_AMP_EWMA_ALPHA * float(shoulder_amp)
                        )
                    out["shoulder_amp_ewma"] = float(self.shoulder_amp_ewma)

        out["is_airborne"] = self.is_airborne
        out["jump_count"] = self.jump_count

        if len(self.jump_times) >= 2:
            intervals = np.diff(np.array(self.jump_times, dtype=np.float32))
            if intervals.size:
                avg = float(np.mean(intervals[-5:]))
                if avg > 0:
                    out["spm"] = 60.0 / avg

        return out


# ---------------- STATE MACHINE ----------------
class JumpRopeStateMachine:
    def __init__(self):
        self.state = JumpRopeState.IDLE
        self._last_seen_count = 0
        self.last_count_time: Optional[float] = None
        self._pending_restart_penalty = False

    def update(self, t_sec: float, jump_count: int) -> int:
        if jump_count > self._last_seen_count:
            if self.state == JumpRopeState.STOPPED and self._pending_restart_penalty:
                jump_count = self._last_seen_count
                self._pending_restart_penalty = False

            self._last_seen_count = jump_count
            self.last_count_time = t_sec
            self.state = JumpRopeState.JUMPING
            return jump_count

        if self.last_count_time is None:
            return jump_count

        if self.state == JumpRopeState.JUMPING and (t_sec - self.last_count_time) > STOP_SUDDEN_SEC:
            self.state = JumpRopeState.STOPPED
            self._pending_restart_penalty = True

        return jump_count


# ---------------- MULTI-PERSON TRACKING ----------------
@dataclass
class Track:
    track_id: int
    detector: JumpDetector
    sm: JumpRopeStateMachine
    last_seen_frame: int
    last_bbox: Tuple[int, int, int, int]
    prev_center: Tuple[float, float]
    prev_h: float
    color: Tuple[int, int, int]  # BGR


class MultiPersonTracker:
    """
    ID stability improvements:
    1) NMS removes duplicate detections per frame.
    2) "Fallback reassociation": if a detection is unmatched, attach it to the nearest track
       instead of instantly spawning a new ID (prevents ID switches after brief gate failures).
    3) We never drop tracks (per your request).
    """

    def __init__(
        self,
        fps: float,
        iou_gate: float = 0.10,
        center_gate_frac: float = 0.90,
        height_gate: float = 0.55,   # more tolerant
        fallback_dist_frac: float = 1.20,
    ):
        self.fps = float(fps)
        self.iou_gate = float(iou_gate)
        self.center_gate_frac = float(center_gate_frac)
        self.height_gate = float(height_gate)
        self.fallback_dist_frac = float(fallback_dist_frac)

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _color_for(self, tid: int) -> Tuple[int, int, int]:
        return (int((tid * 73) % 255), int((tid * 151) % 255), int((tid * 211) % 255))

    def _new_track(self, frame_idx: int, det57: np.ndarray, w: int, h: int) -> Track:
        bb = bbox_xyxy_px(det57, w, h)
        cx, cy = bbox_center_px(det57, w, h)
        hh = bbox_height_px(det57, w, h)

        tid = self._next_id
        self._next_id += 1

        return Track(
            track_id=tid,
            detector=JumpDetector(self.fps),
            sm=JumpRopeStateMachine(),
            last_seen_frame=frame_idx,
            last_bbox=bb,
            prev_center=(cx, cy),
            prev_h=hh,
            color=self._color_for(tid),
        )

    def _match_score(self, tr: Track, det57: np.ndarray, w: int, h: int) -> Optional[float]:
        bb = bbox_xyxy_px(det57, w, h)
        cx, cy = bbox_center_px(det57, w, h)
        hh = bbox_height_px(det57, w, h)

        if hh <= 2.0 or tr.prev_h <= 2.0:
            return None

        if not ((1.0 - self.height_gate) * tr.prev_h <= hh <= (1.0 + self.height_gate) * tr.prev_h):
            return None

        dx = float(cx - tr.prev_center[0])
        dy = float(cy - tr.prev_center[1])
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.center_gate_frac * tr.prev_h:
            return None

        iou = iou_xyxy(tr.last_bbox, bb)
        if iou < self.iou_gate:
            return None

        dist_norm = dist / (tr.prev_h + 1e-6)
        return float(iou - 0.20 * dist_norm)

    def associate(self, frame_idx: int, dets: np.ndarray, w: int, h: int) -> Dict[int, int]:
        if dets.shape[0] == 0:
            return {}

        det_count = dets.shape[0]
        track_ids = list(self.tracks.keys())

        candidates: List[Tuple[float, int, int]] = []
        for tid in track_ids:
            tr = self.tracks[tid]
            for j in range(det_count):
                s = self._match_score(tr, dets[j], w, h)
                if s is not None:
                    candidates.append((s, tid, j))

        candidates.sort(reverse=True, key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()
        assignment: Dict[int, int] = {}

        for s, tid, j in candidates:
            if tid in assigned_tracks or j in assigned_dets:
                continue
            assignment[tid] = j
            assigned_tracks.add(tid)
            assigned_dets.add(j)

        unmatched = [j for j in range(det_count) if j not in assigned_dets]
        for j in unmatched:
            cx, cy = bbox_center_px(dets[j], w, h)

            best_tid = None
            best_dist = 1e18

            for tid, tr in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                dx = float(cx - tr.prev_center[0])
                dy = float(cy - tr.prev_center[1])
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                tr = self.tracks[best_tid]
                gate = self.fallback_dist_frac * max(20.0, tr.prev_h)
                if best_dist <= gate:
                    assignment[best_tid] = j
                    assigned_tracks.add(best_tid)
                    assigned_dets.add(j)
                    continue

            tr_new = self._new_track(frame_idx, dets[j], w, h)
            self.tracks[tr_new.track_id] = tr_new
            assignment[tr_new.track_id] = j
            assigned_tracks.add(tr_new.track_id)
            assigned_dets.add(j)

        return assignment


# ---------------- MAIN ----------------
def main():
    os.makedirs("jump_rope_results", exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: cannot open video:", VIDEO_SOURCE)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    core = Core()
    model = core.read_model(MODEL_PATH)
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")
    out_layer = compiled.output(0)

    mp = MultiPersonTracker(
        fps=fps,
        iou_gate=0.10,
        center_gate_frac=0.90,
        height_gate=0.55,
        fallback_dist_frac=1.20,
    )

    out_video_path = f"jump_rope_results/jump_rope_multiperson_stable_{os.path.basename(VIDEO_SOURCE).split('_')[0]}.mp4"
    out_csv_path = f"jump_rope_results/jump_rope_multiperson_stable_{os.path.basename(VIDEO_SOURCE).split('_')[0]}.csv"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_data = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = frame_idx / fps

        t0 = time.perf_counter()
        preds = compiled([preprocess(frame)])[out_layer][0]  # (N,57)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        valid = preds[preds[:, 4] >= DET_THRESH]
        if valid.shape[0] == 0:
            writer.write(frame)
            frame_idx += 1
            continue

        valid = nms_dets(valid, width, height, iou_th=0.6)
        assign = mp.associate(frame_idx, valid, width, height)

        for tid, det_idx in assign.items():
            tr = mp.tracks[tid]
            det57 = valid[det_idx]

            bb = bbox_xyxy_px(det57, width, height)
            cx, cy = bbox_center_px(det57, width, height)
            hh = bbox_height_px(det57, width, height)

            tr.last_bbox = bb
            tr.prev_center = (cx, cy)
            tr.prev_h = hh
            tr.last_seen_frame = frame_idx

            draw_pose(frame, det57, color=tr.color)

            person_h = hh
            lhip = get_kpt_xy(det57, LEFT_HIP, width, height)
            rhip = get_kpt_xy(det57, RIGHT_HIP, width, height)
            lank = get_kpt_xy(det57, LEFT_ANKLE, width, height)
            rank = get_kpt_xy(det57, RIGHT_ANKLE, width, height)
            lsho = get_kpt_xy(det57, LEFT_SHOULDER, width, height)
            rsho = get_kpt_xy(det57, RIGHT_SHOULDER, width, height)

            det = tr.detector.update(frame_idx, t_sec, lhip, rhip, lank, rank, lsho, rsho, person_h)
            det["jump_count"] = tr.sm.update(t_sec, det["jump_count"])
            tr.detector.jump_count = det["jump_count"]

            x1, y1, x2, y2 = bb
            label = f"ID {tid} | Jump {det['jump_count']} | {tr.sm.state.value} | SPM {det['spm']:.1f} "
            cv2.putText(
                frame, label, (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, tr.color, 7
            )

            frame_data.append({
                "frame_idx": frame_idx,
                "timestamp": t_sec,
                "track_id": tid,
                "conf": float(det57[4]),
                "infer_ms": infer_ms,
                "jump_count": det["jump_count"],
                "spm": det["spm"],
                "state": tr.sm.state.value,
                "dt": det.get("dt", float("nan")),
                "expected_dt": det.get("expected_dt", float("nan")),
                "hip_amp": det.get("amp", float("nan")),
                "shoulder_amp": det.get("shoulder_amp", float("nan")),
                "shoulder_amp_th": det.get("shoulder_amp_th", float("nan")),          # NEW
                "shoulder_noise_sigma": det.get("shoulder_noise_sigma", float("nan")),# NEW
                "shoulder_amp_ewma": det.get("shoulder_amp_ewma", float("nan")),      # NEW
                "amp_ewma": det.get("amp_ewma", float("nan")),
                "lift_th": det.get("lift_th", float("nan")),
                "lift_noise_sigma": det.get("lift_noise_sigma", float("nan")),
                "ground_y": det.get("ground_y", float("nan")),
                "hip_y": det.get("hip_y", float("nan")),
                "shoulder_ground_y": det.get("shoulder_ground_y", float("nan")),
                "shoulder_y": det.get("shoulder_y", float("nan")),
                "is_airborne": det.get("is_airborne", False),
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            })

        writer.write(frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(frame_data).to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print("DONE")
    if len(mp.tracks) == 0:
        print("No tracks were created.")
    else:
        all_counts = [(tid, tr.detector.jump_count) for tid, tr in mp.tracks.items()]
        all_counts.sort(key=lambda x: x[1], reverse=True)
        top_k = all_counts[:4]
        print("Top 4 jump counts:")
        for rank, (tid, cnt) in enumerate(top_k, start=1):
            print(f"  #{rank}: ID {tid} -> {cnt}")

    print("Output video:", out_video_path)
    print("Output csv:", out_csv_path)


if __name__ == "__main__":
    main()