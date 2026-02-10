from pathlib import Path
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
VIDEO_SOURCE = "TestVideos/1053_25fps.mp4"
IMG_SIZE = 640

DET_THRESH = 0.3

# Only accept keypoints if their own confidence is >= this (important!)
MIN_KPT_CONF = 0.25

# Keypoint indices (COCO-17)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Airborne detection (frame confirmations)
AIRBORNE_CONFIRM_FRAMES = 1
GROUND_CONFIRM_FRAMES = 1
GROUND_HISTORY_SECONDS = 1.5  # slightly longer = more stable ground estimate

# Quality gating
REFRACTORY_FRAMES = 7

# STOP detection
STOP_SUDDEN_SEC = 1.5

# Passive stop (TRIPPED) head event thresholds (normalized by person height)
HEAD_EVENT_WINDOW_SEC = 0.45

# Skeleton connections
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
    STOPPED_ACTIVE = "STOPPED_ACTIVE"
    STOPPED_TRIPPED = "STOPPED_TRIPPED"

# ---------------- KALMAN FILTER ----------------
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

        self.Q = np.eye(4, dtype=np.float32) * process_noise   # process covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise  # measurement covariance
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

# ---------------- UTILS ----------------
def safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))

def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def decode_bbox_xyxy_modelpx(det57: np.ndarray):
    x1, y1, x2, y2, score = det57[:5]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return float(x1), float(y1), float(x2), float(y2), float(score)

def decode_kpts_17x3(det57: np.ndarray) -> np.ndarray:
    kpt_flat = det57[6:6 + 51]
    return kpt_flat.reshape(17, 3).astype(np.float32)

def map_modelpx_to_frame(xy_model: np.ndarray, w: int, h: int) -> np.ndarray:
    sx = w / float(IMG_SIZE)
    sy = h / float(IMG_SIZE)
    out = xy_model.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out

def bbox_height_px(det57: np.ndarray, w: int, h: int) -> float:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    sy = h / float(IMG_SIZE)
    return float(abs((y2m - y1m) * sy))

def bbox_center_modelpx(det57: np.ndarray) -> Tuple[float, float]:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    return float((x1m + x2m) / 2.0), float((y1m + y2m) / 2.0)

def extract_keypoint_xy_from_det(
    det57: np.ndarray, kpt_idx: int, w: int, h: int, min_kpt_conf: float = MIN_KPT_CONF
) -> Optional[Tuple[float, float]]:
    kpts = decode_kpts_17x3(det57)
    if not (0 <= kpt_idx < 17):
        return None
    if float(kpts[kpt_idx, 2]) < float(min_kpt_conf):
        return None
    xy_frame = map_modelpx_to_frame(kpts[:, :2], w, h)
    x, y = xy_frame[kpt_idx]
    return float(x), float(y)

def draw_pose_overlay(frame: np.ndarray, det57: np.ndarray):
    """Draw bbox + 17 kpts + skeleton on frame."""
    h, w = frame.shape[:2]
    x1m, y1m, x2m, y2m, conf = decode_bbox_xyxy_modelpx(det57)

    sx = w / float(IMG_SIZE)
    sy = h / float(IMG_SIZE)

    x1, y1 = int(x1m * sx), int(y1m * sy)
    x2, y2 = int(x2m * sx), int(y2m * sy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{conf:.3f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    kpts = decode_kpts_17x3(det57)
    xy_frame = map_modelpx_to_frame(kpts[:, :2], w, h)
    kconf = kpts[:, 2]

    pts = []
    for i in range(17):
        cx, cy = int(xy_frame[i, 0]), int(xy_frame[i, 1])
        pts.append((cx, cy))

        # draw only if conf >= MIN_KPT_CONF (you can set 0.0 if you want all)
        if float(kconf[i]) >= MIN_KPT_CONF:
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for a, b in SKELETON:
        if float(kconf[a]) >= MIN_KPT_CONF and float(kconf[b]) >= MIN_KPT_CONF:
            ax, ay = pts[a]
            bx, by = pts[b]
            cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

# ---------------- JUMP DETECTOR ----------------
class JumpDetector:
    """
    Patched improvements:
    - per-joint KF tuning (ankles more responsive)
    - thresholds normalized by person height (bbox height)
    - more robust ground estimate (blend percentiles)
    """
    def __init__(self, fps: float):
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
        self.ankle_y_hist = deque(maxlen=int(self.fps * GROUND_HISTORY_SECONDS))
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

        # for debug overlay
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
        if nose is not None: pts.append(self.kf_nose.update(*nose))
        if leye is not None: pts.append(self.kf_leye.update(*leye))
        if reye is not None: pts.append(self.kf_reye.update(*reye))
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    def update(self, frame_idx: int, t_sec: float,
               left_hip, right_hip, left_ankle, right_ankle,
               nose, left_eye, right_eye,
               person_h: float) -> Dict[str, Any]:

        # Normalize thresholds by person height (bbox height in pixels)
        # These ratios are good starting points and work better across videos.
        lift_th = 0.013 * person_h        # airborne lift threshold
        hip_amp_min = 0.012 * person_h      # minimum hip motion
        ankle_dist_th = 18             # single vs double
        head_drop_th = 0.025 * person_h   # head drop event threshold

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

        # Need hips + ankles to run jump logic
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

        # jump type (single/double) using normalized threshold
        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            out["jump_type"] = "single" if avg_dist >= ankle_dist_th else "double"

        # head smoothing + drop event
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
                out["head_move_px"] = float(np.hypot(head_xy[0] - self.prev_head_xy[0],
                                                     head_xy[1] - self.prev_head_xy[1]))

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

        if (not self.is_airborne) and (self.airborne_frames >= AIRBORNE_CONFIRM_FRAMES):
            self.is_airborne = True

        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            hip_amp_ok = False
            if len(self.hip_y_hist) >= int(self.fps * 0.4):
                recent = np.array(list(self.hip_y_hist)[-int(self.fps * 0.4):], dtype=np.float32)
                hip_amp = float(np.max(recent) - np.min(recent))
                hip_amp_ok = hip_amp >= hip_amp_min

            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

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
    def __init__(self, fps: float):
        self.fps = float(fps)
        self.state = JumpRopeState.IDLE
        self._last_seen_count = 0
        self.last_count_time: Optional[float] = None
        self.head_event_times = deque(maxlen=int(self.fps * 2.0))

    def _head_event_recent(self, t_sec: float) -> bool:
        t_min = t_sec - HEAD_EVENT_WINDOW_SEC
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

        if self.state == JumpRopeState.JUMPING and time_since_last > STOP_SUDDEN_SEC:
            self.state = JumpRopeState.STOPPED_TRIPPED if self._head_event_recent(t_sec) else JumpRopeState.STOPPED_ACTIVE

def draw_annotations(frame: np.ndarray, jump_count: int, single_count: int, double_count: int,
                     state: JumpRopeState, jump_type: str, spm: float,
                     ground_y: float, lift_th: float) -> np.ndarray:
    annotated = frame
    colors = {
        JumpRopeState.IDLE: (160, 160, 160),
        JumpRopeState.JUMPING: (0, 255, 0),
        JumpRopeState.STOPPED_ACTIVE: (0, 165, 255),
        JumpRopeState.STOPPED_TRIPPED: (0, 0, 255),
    }
    c = colors.get(state, (255, 255, 255))

    y = 30
    cv2.putText(annotated, f"Jump Count: {jump_count} (S:{single_count} D:{double_count})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 28
    cv2.putText(annotated, f"State: {state.value}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 28
    cv2.putText(annotated, f"Type: {jump_type}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 28
    cv2.putText(annotated, f"SPM: {spm:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    # Debug line for ground estimate (helps tuning)
    if np.isfinite(ground_y):
        gy = int(ground_y)
        cv2.line(annotated, (0, gy), (annotated.shape[1] - 1, gy), (255, 255, 255), 1)
        cv2.putText(annotated, f"ground_y={ground_y:.1f} lift_th={lift_th:.1f}",
                    (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated

# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print("Jump Rope Detection (OpenVINO + static INT8 ONNX) + Draw Pose (PATCHED)")
    print("=" * 60)

    video_path = Path(VIDEO_SOURCE)
    model_path = Path(MODEL_PATH)

    if not video_path.exists():
        print(f"ERROR: video not found: {VIDEO_SOURCE}")
        return
    if not model_path.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {VIDEO_SOURCE}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"Video: {width}x{height}, FPS={fps_src:.2f}, frames={total_frames}")

    core = Core()
    model = core.read_model(str(model_path))
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")
    output_layer = compiled.output(0)

    detector = JumpDetector(fps_src)
    sm = JumpRopeStateMachine(fps_src)

    out_dir = Path("jump_rope_results")
    out_dir.mkdir(exist_ok=True)
    out_video_path = out_dir / f"{video_path.stem}_jump_rope_openvino_draw_patched.mp4"
    out_csv_path = out_dir / f"{video_path.stem}_jump_rope_openvino_draw_patched.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_src, (width, height))

    frame_data = []
    frame_idx = 0
    t_start = time.time()

    # Track same subject across frames (instead of argmax confidence)
    prev_center_model = None  # (cx, cy) in model px

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = frame_idx / fps_src

        # Inference
        t0 = time.perf_counter()
        out = compiled([preprocess(frame)])[output_layer]  # (1, N, 57)
        preds = out[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Choose best detection robustly:
        # - if we have a previous center, choose the nearest detection above DET_THRESH
        # - otherwise pick highest conf
        valid = preds[preds[:, 4] >= DET_THRESH]
        if valid.shape[0] == 0:
            best = preds[int(np.argmax(preds[:, 4]))]
        else:
            if prev_center_model is None:
                best = valid[int(np.argmax(valid[:, 4]))]
            else:
                centers = np.array([bbox_center_modelpx(d) for d in valid], dtype=np.float32)
                dx = centers[:, 0] - float(prev_center_model[0])
                dy = centers[:, 1] - float(prev_center_model[1])
                dist2 = dx * dx + dy * dy
                best = valid[int(np.argmin(dist2))]

        best_conf = float(best[4])
        prev_center_model = bbox_center_modelpx(best)

        # Draw pose overlay if confident
        if best_conf >= DET_THRESH:
            draw_pose_overlay(frame, best)

        # Person height for normalized thresholds
        person_h = bbox_height_px(best, width, height)
        if person_h <= 1.0:
            person_h = max(1.0, 0.5 * height)  # fallback

        # Extract keypoints (with kpt confidence gating)
        lhip = rhip = lank = rank = nose = leye = reye = None
        if best_conf >= DET_THRESH:
            lhip = extract_keypoint_xy_from_det(best, LEFT_HIP, width, height, MIN_KPT_CONF)
            rhip = extract_keypoint_xy_from_det(best, RIGHT_HIP, width, height, MIN_KPT_CONF)
            lank = extract_keypoint_xy_from_det(best, LEFT_ANKLE, width, height, MIN_KPT_CONF)
            rank = extract_keypoint_xy_from_det(best, RIGHT_ANKLE, width, height, MIN_KPT_CONF)
            nose = extract_keypoint_xy_from_det(best, NOSE, width, height, MIN_KPT_CONF)
            leye = extract_keypoint_xy_from_det(best, LEFT_EYE, width, height, MIN_KPT_CONF)
            reye = extract_keypoint_xy_from_det(best, RIGHT_EYE, width, height, MIN_KPT_CONF)

        det = detector.update(frame_idx, t_sec, lhip, rhip, lank, rank, nose, leye, reye, person_h)
        sm.update(t_sec, det["jump_count"], bool(det["head_drop_event"]))

        annotated = draw_annotations(
            frame,
            det["jump_count"],
            det["single_count"],
            det["double_count"],
            sm.state,
            det["jump_type"],
            float(det["spm"]),
            float(det.get("ground_y", float("nan"))),
            float(det.get("lift_th", 0.0)),
        )

        cv2.putText(annotated, f"{infer_ms:.1f} ms", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(annotated)

        frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": t_sec,
            "best_conf": best_conf,
            "infer_ms": infer_ms,
            "hip_y": det["hip_y"],
            "head_y": det["head_y"],
            "head_move_px": det["head_move_px"],
            "head_drop_event": det["head_drop_event"],
            "ankle_distance": det["ankle_distance"],
            "is_airborne": det["is_airborne"],
            "jump_count": det["jump_count"],
            "single_count": det["single_count"],
            "double_count": det["double_count"],
            "jump_type": det["jump_type"],
            "freq_hz": det["freq_hz"],
            "spm": float(det["spm"]),
            "state": sm.state.value,
            "ground_y": det.get("ground_y", float("nan")),
            "person_h": person_h,
        })

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(
                f"Processed {frame_idx}/{total_frames} | "
                f"conf={best_conf:.3f} | total={det['jump_count']} "
                f"(S:{det['single_count']} D:{det['double_count']}) | "
                f"state={sm.state.value} | spm={float(det['spm']):.1f}"
            )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(frame_data)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    dt = time.time() - t_start
    print("\nDONE")
    print(f"Total count:  {detector.jump_count}")
    print(f"Output video: {out_video_path}")
    print(f"Output csv:   {out_csv_path}")
    print(f"Time: {dt:.2f}s")

if __name__ == "__main__":
    main()
