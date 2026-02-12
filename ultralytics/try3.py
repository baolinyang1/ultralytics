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
VIDEO_SOURCE = "TestVideos/sport3_25fps.mp4"
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

# ---- CYCLE CHECKS (THE IMPORTANT PART) ----
# Jump rope cadence is usually stable; walking/random isn't.
MIN_JUMP_INTERVAL_SEC = 0.25   # faster than this is probably noise
MAX_JUMP_INTERVAL_SEC = 1.50   # slower than this is probably not jump rope
CYCLE_WINDOW = 4               # how many recent intervals to check (3-5 works)
CYCLE_CV_MAX = 0.22            # coefficient of variation threshold (std/mean)

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

# ---------------- JUMP DETECTOR (WITH CYCLE CHECKS) ----------------
class JumpDetector:
    """
    Base idea:
    - detect candidate jumps from airborne->landing
    - BUT accept count only when cadence is consistent (cycle checks)
    """

    def __init__(self, fps: float):
        self.fps = float(fps)

        self.kf_lhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_rhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_lank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_rank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)

        self.hip_y_hist = deque(maxlen=int(self.fps * 0.6))
        self.ankle_y_hist = deque(maxlen=max(10, int(self.fps * GROUND_HISTORY_SECONDS)))
        self.ankle_dist_hist = deque(maxlen=10)

        self.airborne_frames = 0
        self.ground_frames = 0
        self.is_airborne = False

        self.jump_count = 0
        self.single_count = 0
        self.double_count = 0
        self.last_count_frame = -10_000

        # cycle tracking
        self.land_times = deque(maxlen=10)   # timestamps of accepted/provisional landings
        self.jump_times = deque(maxlen=50)   # accepted jump timestamps (for SPM)

    def _cycle_ok(self) -> bool:
        """
        Check if recent landing intervals are consistent:
          - enough samples
          - each interval in [MIN, MAX]
          - coefficient of variation (std/mean) <= CYCLE_CV_MAX
        """
        if len(self.land_times) < (CYCLE_WINDOW + 1):
            return False

        ts = np.array(self.land_times, dtype=np.float32)
        intervals = np.diff(ts)[-CYCLE_WINDOW:]  # last CYCLE_WINDOW intervals

        if np.any(intervals < MIN_JUMP_INTERVAL_SEC) or np.any(intervals > MAX_JUMP_INTERVAL_SEC):
            return False

        mean_i = float(np.mean(intervals))
        std_i = float(np.std(intervals))
        if mean_i <= 1e-6:
            return False

        cv = std_i / mean_i
        return cv <= CYCLE_CV_MAX

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
            "ankle_distance": 0.0,
            "hip_y": 0.0,
            "cycle_ok": False,
        }

        if not (lhip and rhip and lank and rank) or person_h <= 1.0:
            return out

        # thresholds normalized by height
        lift_th = 0.013 * person_h
        hip_amp_min = 0.012 * person_h
        ankle_dist_th = 25

        lh = self.kf_lhip.update(*lhip)
        rh = self.kf_rhip.update(*rhip)
        la = self.kf_lank.update(*lank)
        ra = self.kf_rank.update(*rank)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        ankle_y = float((la[1] + ra[1]) / 2.0)
        ankle_dist = float(np.hypot(la[0] - ra[0], la[1] - ra[1]))

        out["hip_y"] = hip_y
        out["ankle_distance"] = ankle_dist
        out["lift_th"] = float(lift_th)

        self.hip_y_hist.append(hip_y)
        self.ankle_y_hist.append(ankle_y)
        self.ankle_dist_hist.append(ankle_dist)

        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            out["jump_type"] = "single" if avg_dist >= ankle_dist_th else "double"

        if len(self.ankle_y_hist) < 10:
            return out

        ground_y = float(np.percentile(np.array(self.ankle_y_hist, dtype=np.float32), 90.0))
        out["ground_y"] = ground_y

        airborne_now = ankle_y <= (ground_y - lift_th)

        if airborne_now:
            self.airborne_frames += 1
            self.ground_frames = 0
        else:
            self.ground_frames += 1
            self.airborne_frames = 0

        if (not self.is_airborne) and (self.airborne_frames >= AIRBORNE_CONFIRM_FRAMES):
            self.is_airborne = True

        # landing -> candidate event
        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            # hip amplitude gate (still needed to avoid tiny noise)
            hip_amp_ok = False
            if len(self.hip_y_hist) >= max(3, int(self.fps * 0.35)):
                recent = np.array(self.hip_y_hist, dtype=np.float32)
                hip_amp_ok = float(recent.max() - recent.min()) >= hip_amp_min

            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

            if hip_amp_ok and refractory_ok:
                # record landing time (for cadence check)
                self.land_times.append(t_sec)

                cycle_ok = self._cycle_ok()
                out["cycle_ok"] = cycle_ok

                # Warm-up: allow first 2 jumps to seed the cadence,
                # after that require cadence consistency.
                allow = (self.jump_count < 2) or cycle_ok

                if allow:
                    self.jump_count += 1
                    self.last_count_frame = frame_idx
                    self.jump_times.append(t_sec)

                    if out["jump_type"] == "single":
                        self.single_count += 1
                    else:
                        self.double_count += 1

            self.is_airborne = False

        out["is_airborne"] = self.is_airborne
        out["jump_count"] = self.jump_count
        out["single_count"] = self.single_count
        out["double_count"] = self.double_count

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
    y += 26
    cv2.putText(frame, f"cycle_ok: {bool(det.get('cycle_ok', False))}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 26
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

    out_video_path = "jump_rope_results/jump_rope_cycle_checked.mp4"
    out_csv_path = "jump_rope_results/jump_rope_cycle_checked.csv"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_data = []
    frame_idx = 0
    prev_center = None

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

        # track same person by nearest bbox center
        if prev_center is None:
            best = valid[int(np.argmax(valid[:, 4]))]
        else:
            centers = np.array([bbox_center_modelpx(d) for d in valid], dtype=np.float32)
            dx = centers[:, 0] - float(prev_center[0])
            dy = centers[:, 1] - float(prev_center[1])
            best = valid[int(np.argmin(dx * dx + dy * dy))]

        prev_center = bbox_center_modelpx(best)
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
