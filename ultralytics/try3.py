import os
import time
from collections import deque
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from openvino.runtime import Core

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo26n-pose.static_int8.onnx"
VIDEO_SOURCE = "TestVideos/race3_25fps.mp4"
IMG_SIZE = 640

DET_THRESH = 0.3
MIN_KPT_CONF = 0.25

# COCO-17 indices
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Jump logic
GROUND_HISTORY_SECONDS = 1.5
AIRBORNE_CONFIRM_FRAMES = 1
GROUND_CONFIRM_FRAMES = 1
REFRACTORY_FRAMES = 7
STOP_SUDDEN_SEC = 1.5

# ---------------- CYCLE CHECKS (IMPROVED) ----------------
CYCLE_WINDOW = 4
INTERVAL_TOL = 0.22
ALLOWED_OUTLIERS = 1
MIN_JUMP_INTERVAL_SEC = 0.23
MAX_JUMP_INTERVAL_SEC = 1.70

# cadence tracking (adaptive)
EWMA_ALPHA = 0.22
MAD_K = 3.0
MIN_MAD_SEC = 0.03

# amplitude gate (stop/walk suppression)
AMP_FRAC = 0.020
AMP_MIN_PX = 6.0

# ---------------- DYNAMIC LIFT THRESHOLD ----------------
LIFT_BASE_FRAC = 0.010
AMP_TO_LIFT = 0.35
NOISE_K = 3.0
MIN_LIFT_PX = 4.0
MAX_LIFT_FRAC = 0.050
AMP_EWMA_ALPHA = 0.25

# ---------------- HIP-BASED SETTINGS (FIXED) ----------------
# Hip lift can be larger; don't cap it too low.
HIP_LIFT_MAX_FRAC = 0.080      # was 0.035 (WRONG); allow larger hip-based lift_th
HIP_NOISE_BAND_FRAC = 0.18     # ok to keep

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

# ---------------- UTILS ----------------
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def decode_bbox_xyxy_modelpx(det57: np.ndarray) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2, score = det57[:5]
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return float(x1), float(y1), float(x2), float(y2), float(score)

def decode_kpts_17x3(det57: np.ndarray) -> np.ndarray:
    return det57[6:6 + 51].reshape(17, 3).astype(np.float32)

def map_modelpx_to_frame(xy_model: np.ndarray, w: int, h: int) -> np.ndarray:
    out = xy_model.astype(np.float32).copy()
    out[:, 0] *= (w / float(IMG_SIZE))
    out[:, 1] *= (h / float(IMG_SIZE))
    return out

def bbox_height_px(det57: np.ndarray, w: int, h: int) -> float:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    return float(abs(y2m - y1m) * (h / float(IMG_SIZE)))

def bbox_center_modelpx(det57: np.ndarray) -> Tuple[float, float]:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    return float((x1m + x2m) / 2.0), float((y1m + y2m) / 2.0)

def get_kpt_xy(det57: np.ndarray, idx: int, w: int, h: int) -> Optional[Tuple[float, float]]:
    kpts = decode_kpts_17x3(det57)
    if float(kpts[idx, 2]) < MIN_KPT_CONF:
        return None
    xy = map_modelpx_to_frame(kpts[:, :2], w, h)
    return float(xy[idx, 0]), float(xy[idx, 1])

def draw_pose(frame: np.ndarray, det57: np.ndarray):
    h, w = frame.shape[:2]
    x1m, y1m, x2m, y2m, conf = decode_bbox_xyxy_modelpx(det57)
    sx, sy = w / float(IMG_SIZE), h / float(IMG_SIZE)
    x1, y1, x2, y2 = int(x1m * sx), int(y1m * sy), int(x2m * sx), int(y2m * sy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    kpts = decode_kpts_17x3(det57)
    xy = map_modelpx_to_frame(kpts[:, :2], w, h)
    kc = kpts[:, 2]

    pts = [(int(xy[i, 0]), int(xy[i, 1])) for i in range(17)]
    for i in range(17):
        if float(kc[i]) >= MIN_KPT_CONF:
            cv2.circle(frame, pts[i], 3, (0, 0, 255), -1)

    for a, b in SKELETON:
        if float(kc[a]) >= MIN_KPT_CONF and float(kc[b]) >= MIN_KPT_CONF:
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)

# ---------------- JUMP DETECTOR (HIP-BASED) ----------------
class JumpDetector:
    """
    Hip-based version:
      - ground_y from hip_y
      - airborne_now from hip_y vs ground_y - lift_th
      - amplitude uses hip segment: amp = ground_y - min_hip_y_while_airborne
      - dynamic lift uses amp EWMA + noise sigma
      - cadence voting identical
    """

    def __init__(self, fps: float):
        self.fps = float(fps)

        self.kf_lhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_rhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_lank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_rank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)

        self.hip_y_hist = deque(maxlen=max(10, int(self.fps * GROUND_HISTORY_SECONDS)))
        self.ankle_dist_hist = deque(maxlen=10)

        self.airborne_frames = 0
        self.ground_frames = 0
        self.is_airborne = False

        self.air_min_y: Optional[float] = None

        self.jump_count = 0
        self.single_count = 0
        self.double_count = 0
        self.last_count_frame = -10_000

        self.jump_times = deque(maxlen=50)
        self.interval_hist = deque(maxlen=20)
        self.expected_interval: Optional[float] = None

        self.amp_ewma: Optional[float] = None

    def _reset_cadence(self):
        self.jump_times.clear()
        self.interval_hist.clear()
        self.expected_interval = None

    @staticmethod
    def _robust_sigma(vals: np.ndarray) -> float:
        if vals.size == 0:
            return 0.0
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return 1.4826 * mad

    def _estimate_ground_noise_sigma(self, y_hist: deque, ground_y: float) -> float:
        arr = np.array(y_hist, dtype=np.float32)
        if arr.size < 10:
            return 0.0

        span = float(arr.max() - arr.min())
        band = max(6.0, HIP_NOISE_BAND_FRAC * span)
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

    def _cycle_vote_ok(self, dt: float) -> bool:
        tmp = list(self.interval_hist) + [float(dt)]
        if len(tmp) < CYCLE_WINDOW:
            return True

        window = np.array(tmp[-CYCLE_WINDOW:], dtype=np.float32)
        base = float(self.expected_interval) if self.expected_interval else float(np.median(window))

        if len(self.interval_hist) == 0:
            mad = MIN_MAD_SEC
        else:
            hist = np.array(self.interval_hist, dtype=np.float32)
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med)))
            mad = max(mad, MIN_MAD_SEC)

        base_tol = max(INTERVAL_TOL * base, MAD_K * mad)
        lo = base - base_tol
        hi = base + base_tol

        good = int(np.sum((window >= lo) & (window <= hi)))
        return good >= (CYCLE_WINDOW - ALLOWED_OUTLIERS)

    def update(self, frame_idx: int, t_sec: float,
               lhip, rhip, lank, rank,
               person_h: float) -> Dict[str, Any]:

        out = {
            "is_airborne": False,
            "jump_type": "unknown",
            "jump_count": self.jump_count,
            "single_count": self.single_count,
            "double_count": self.double_count,
            "spm": 0.0,
            "ground_y": float("nan"),
            "lift_th": float("nan"),
            "lift_noise_sigma": float("nan"),
            "amp_ewma": float("nan"),
            "ankle_distance": 0.0,
            "hip_y": float("nan"),
            "cycle_ok": False,
            "dt": float("nan"),
            "expected_dt": float("nan"),
            "amp": float("nan"),
            "min_air_y": float("nan"),
        }

        if not (lhip and rhip and lank and rank) or person_h <= 1.0:
            return out

        ankle_dist_th = 25
        amp_th = max(AMP_MIN_PX, AMP_FRAC * person_h)  # <-- NOT scaled down

        lh = self.kf_lhip.update(*lhip)
        rh = self.kf_rhip.update(*rhip)
        la = self.kf_lank.update(*lank)
        ra = self.kf_rank.update(*rank)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        ankle_dist = float(np.hypot(la[0] - ra[0], la[1] - ra[1]))

        out["hip_y"] = hip_y
        out["ankle_distance"] = ankle_dist

        self.hip_y_hist.append(hip_y)
        self.ankle_dist_hist.append(ankle_dist)

        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            out["jump_type"] = "single" if avg_dist >= ankle_dist_th else "double"

        if len(self.hip_y_hist) < 10:
            return out

        ground_y = float(np.percentile(np.array(self.hip_y_hist, dtype=np.float32), 90.0))
        out["ground_y"] = ground_y

        sigma = self._estimate_ground_noise_sigma(self.hip_y_hist, ground_y)
        lift_th = self._dynamic_lift_th(person_h, sigma)
        out["lift_noise_sigma"] = float(sigma)
        out["lift_th"] = float(lift_th)
        out["amp_ewma"] = float(self.amp_ewma) if self.amp_ewma is not None else float("nan")

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

        if self.is_airborne:
            self.air_min_y = hip_y if self.air_min_y is None else float(min(self.air_min_y, hip_y))

        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

            min_air_y = self.air_min_y if self.air_min_y is not None else hip_y
            amp = float(ground_y - float(min_air_y))
            out["amp"] = amp
            out["min_air_y"] = float(min_air_y)

            self.is_airborne = False
            self.air_min_y = None

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

                if amp < amp_th:
                    return out

                cycle_ok = True
                if dt is not None:
                    cycle_ok = self._cycle_vote_ok(dt)
                out["cycle_ok"] = bool(cycle_ok)
                if not cycle_ok:
                    return out

                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(float(t_sec))

                if dt is not None:
                    self.interval_hist.append(float(dt))
                    if self.expected_interval is None:
                        self.expected_interval = float(dt)
                    else:
                        self.expected_interval = float((1.0 - EWMA_ALPHA) * self.expected_interval + EWMA_ALPHA * dt)
                    out["expected_dt"] = float(self.expected_interval)

                if self.amp_ewma is None:
                    self.amp_ewma = float(amp)
                else:
                    self.amp_ewma = float((1.0 - AMP_EWMA_ALPHA) * self.amp_ewma + AMP_EWMA_ALPHA * amp)
                out["amp_ewma"] = float(self.amp_ewma)

                if out["jump_type"] == "single":
                    self.single_count += 1
                else:
                    self.double_count += 1

        out["is_airborne"] = self.is_airborne
        out["jump_count"] = self.jump_count
        out["single_count"] = self.single_count
        out["double_count"] = self.double_count
        if self.expected_interval is not None:
            out["expected_dt"] = float(self.expected_interval)

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

    def update(self, t_sec: float, jump_count: int):
        if jump_count > self._last_seen_count:
            self._last_seen_count = jump_count
            self.last_count_time = t_sec
            self.state = JumpRopeState.JUMPING
            return
        if self.last_count_time is None:
            return
        if self.state == JumpRopeState.JUMPING and (t_sec - self.last_count_time) > STOP_SUDDEN_SEC:
            self.state = JumpRopeState.STOPPED

def draw_hud(frame: np.ndarray, det: Dict[str, Any], state: JumpRopeState, infer_ms: float):
    c = (160, 160, 160) if state == JumpRopeState.IDLE else (0, 255, 0) if state == JumpRopeState.JUMPING else (0, 165, 255)

    y = 30
    cv2.putText(frame, f"Jump: {det['jump_count']} (S:{det['single_count']} D:{det['double_count']})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    y += 26
    cv2.putText(frame, f"State: {state.value}  Type: {det['jump_type']}  SPM: {det['spm']:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    y += 24
    cv2.putText(frame, f"hip lift={det.get('lift_th', float('nan')):.1f}  sigma={det.get('lift_noise_sigma', float('nan')):.2f}  ampEWMA={det.get('amp_ewma', float('nan')):.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 22
    cv2.putText(frame, f"cycle_ok={bool(det.get('cycle_ok', False))} dt={det.get('dt', float('nan')):.2f} exp={det.get('expected_dt', float('nan')):.2f} amp={det.get('amp', float('nan')):.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 22
    cv2.putText(frame, f"{infer_ms:.1f} ms",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

    detector = JumpDetector(fps)
    sm = JumpRopeStateMachine()

    out_video_path = "jump_rope_results/jump_rope_hip_dynamic_lift.mp4"
    out_csv_path = "jump_rope_results/jump_rope_hip_dynamic_lift.csv"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_data = []
    frame_idx = 0
    prev_center = None
    prev_h = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = frame_idx / fps

        t0 = time.perf_counter()
        preds = compiled([preprocess(frame)])[out_layer][0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        valid = preds[preds[:, 4] >= DET_THRESH]
        if valid.shape[0] == 0:
            frame_idx += 1
            continue

        if prev_center is None:
            best = valid[int(np.argmax(valid[:, 4]))]
        else:
            centers = np.array([bbox_center_modelpx(d) for d in valid], dtype=np.float32)
            dist2 = (centers[:, 0] - float(prev_center[0])) ** 2 + (centers[:, 1] - float(prev_center[1])) ** 2

            heights = np.array([bbox_height_px(d, width, height) for d in valid], dtype=np.float32)
            if prev_h is not None and prev_h > 1.0:
                mask = (heights >= 0.7 * prev_h) & (heights <= 1.3 * prev_h)
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    best = valid[idxs[int(np.argmin(dist2[idxs]))]]
                else:
                    best = valid[int(np.argmin(dist2))]
            else:
                best = valid[int(np.argmin(dist2))]

        prev_center = bbox_center_modelpx(best)
        prev_h = bbox_height_px(best, width, height)
        best_conf = float(best[4])

        draw_pose(frame, best)

        person_h = bbox_height_px(best, width, height)
        lhip = get_kpt_xy(best, LEFT_HIP, width, height)
        rhip = get_kpt_xy(best, RIGHT_HIP, width, height)
        lank = get_kpt_xy(best, LEFT_ANKLE, width, height)
        rank = get_kpt_xy(best, RIGHT_ANKLE, width, height)

        det = detector.update(frame_idx, t_sec, lhip, rhip, lank, rank, person_h)
        sm.update(t_sec, det["jump_count"])

        draw_hud(frame, det, sm.state, infer_ms)
        writer.write(frame)

        frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": t_sec,
            "best_conf": best_conf,
            "infer_ms": infer_ms,
            "jump_count": det["jump_count"],
            "single_count": det["single_count"],
            "double_count": det["double_count"],
            "jump_type": det["jump_type"],
            "spm": det["spm"],
            "state": sm.state.value,
            "cycle_ok": bool(det.get("cycle_ok", False)),
            "dt": det.get("dt", float("nan")),
            "expected_dt": det.get("expected_dt", float("nan")),
            "amp": det.get("amp", float("nan")),
            "amp_ewma": det.get("amp_ewma", float("nan")),
            "lift_th": det.get("lift_th", float("nan")),
            "lift_noise_sigma": det.get("lift_noise_sigma", float("nan")),
            "min_air_y": det.get("min_air_y", float("nan")),
            "ground_y": det.get("ground_y", float("nan")),
            "ankle_distance": det.get("ankle_distance", 0.0),
            "hip_y": det.get("hip_y", float("nan")),
            "is_airborne": det.get("is_airborne", False),
        })

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(frame_data).to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print("DONE")
    print("Total count:", detector.jump_count)
    print("Output video:", out_video_path)
    print("Output csv:", out_csv_path)

if __name__ == "__main__":
    main()
