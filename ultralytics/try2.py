"""
Jump Rope Detection - YOLO26n-pose INT8 (OpenVINO)
Features:
- Jump rope counting (robust: airborne -> landing event)
- Single/Double foot jump type (ankle x-distance)
- State detection (IDLE / JUMPING / STOPPED_ACTIVE / STOPPED_TRIPPED)
- Speed estimation (Hz / SPM)
- Kalman filter smoothing for keypoints
"""
# detector uses kalaman, it is the main part, state machine only controls and updates the state

from pathlib import Path
import time
from collections import deque
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "yolo26s-pose.pt"
CALIB_DATA = "coco8-pose.yaml"
VIDEO_SOURCE = "TestVideos/race1_25fps.mp4"
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
# ideal FPS
FPS_FALLBACK = 25.0
# Typical jump rope: ~150-220 spm @ 25fps => ~roughly 7-10 frames per jump, now i dont use it anyways
MIN_JUMP_FRAMES = 6
MAX_JUMP_FRAMES = 15
# Airborne detection, it was 10 before! 
LIFT_THRESHOLD_PX = 2.4            # was 10 before,ankle_y must be this much "higher" (smaller y) than ground baseline
AIRBORNE_CONFIRM_FRAMES = 1       # consecutive frames to confirm airborne, was 2 before!
GROUND_CONFIRM_FRAMES = 1         # consecutive frames to confirm ground (for landing), was 2 before!
GROUND_HISTORY_SECONDS = 1.0      # baseline window length
# Jump type
ANKLE_DISTANCE_THRESHOLD = 20     # px, avg of last few frames, 12 is good enough here!
# Speed / state
SPM_VALID_MIN = 130
SPM_VALID_MAX = 240
STOP_SUDDEN_SEC = 1.5             # sudden stop if we had stable cadence and then no jumps for this long
STOP_GRADUAL_SEC = 3.0           # gradual stop if no jumps for this long
MIN_STABLE_JUMPS_TO_ENTER = 1     # require at least N counted jumps to enter JUMPING
# Cycle quality gating
HIP_AMPLITUDE_MIN_PX = 2.4          # require some hip movement amplitude to accept a jump
REFRACTORY_FRAMES = 2             # minimum frames after counting before counting again

# State
class JumpRopeState(Enum):
    IDLE = "IDLE"
    JUMPING = "JUMPING"
    STOPPED_ACTIVE = "STOPPED_ACTIVE"
    STOPPED_TRIPPED = "STOPPED_TRIPPED"

# Kalman Filter
class KalmanFilter2D:
    """2D constant-velocity KF for smoothing (x,y)."""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state = np.zeros(4, dtype=np.float32)  # [x,y,vx,vy]
        self.cov = np.eye(4, dtype=np.float32) * 1000.0

        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

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

        z = np.array([x, y], dtype=np.float32)
        y_res = z - (self.H @ self.state)
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_res
        self.cov = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.cov

        return self.state[:2].copy()


# Utils Functions

def extract_keypoint_xy(res, kpt_idx: int, conf_th: float = 0.5) -> Optional[Tuple[float, float]]:
    """Extract a single keypoint (x,y) from a YOLO Results object."""
    try:
        if not hasattr(res, "keypoints") or res.keypoints is None:
            return None
        if len(res.keypoints) == 0:
            return None

        kpts = res.keypoints.xy[0].cpu().numpy()  # (K,2)
        if kpt_idx >= len(kpts):
            return None

        if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
            conf = res.keypoints.conf[0].cpu().numpy()
            if conf[kpt_idx] < conf_th:
                return None

        x, y = kpts[kpt_idx]
        return float(x), float(y)
    except Exception:
        return None


def safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))

# Jump Detector!!!
class JumpDetector:
    """
    Robust counting by event:
    - Detect airborne with ankle_y relative to ground baseline
    - Count when airborne -> ground (landing) with valid duration
    - Gate with hip amplitude + duration + refractory
    """

    def __init__(self, fps: float):
        self.fps = float(fps)

        # KF
        self.kf_lhip = KalmanFilter2D(0.01, 0.1)
        self.kf_rhip = KalmanFilter2D(0.01, 0.1)
        self.kf_lank = KalmanFilter2D(0.01, 0.1)
        self.kf_rank = KalmanFilter2D(0.01, 0.1)
        # histories
        self.hip_y_hist = deque(maxlen=int(self.fps * 2.0))  # 2s
        self.ankle_y_hist = deque(maxlen=int(self.fps * GROUND_HISTORY_SECONDS))
        self.ankle_dist_hist = deque(maxlen=10)
        # airborne state
        self.airborne_frames = 0
        self.ground_frames = 0
        self.is_airborne = False
        self.airborne_start_frame: Optional[int] = None
        # counting
        self.jump_count = 0
        self.single_count = 0   
        self.double_count = 0   
        self.last_count_frame = -10_000
        # timestamps (when count happens) for frequency/spm
        self.jump_times = deque(maxlen=50)

    def _compute_ground_y(self) -> Optional[float]:
        if len(self.ankle_y_hist) < max(8, int(self.fps * 0.3)):
            return None
        arr = np.array(self.ankle_y_hist, dtype=np.float32)
        # ground is "low" in image => larger y. Use high percentile to be robust.
        return safe_percentile(arr, 90.0)

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        left_hip: Optional[Tuple[float, float]],
        right_hip: Optional[Tuple[float, float]],
        left_ankle: Optional[Tuple[float, float]],
        right_ankle: Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:

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
        }

        # need all 4 points
        if not (left_hip and right_hip and left_ankle and right_ankle):
            return out

        # smooth
        lh = self.kf_lhip.update(*left_hip)
        rh = self.kf_rhip.update(*right_hip)
        la = self.kf_lank.update(*left_ankle)
        ra = self.kf_rank.update(*right_ankle)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        ankle_y = float((la[1] + ra[1]) / 2.0)
        # This old one is only horizontal distance, not vertical distance
        #ankle_dist = float(abs(la[0] - ra[0]))
        dx = la[0] - ra[0]
        dy = la[1] - ra[1]
        ankle_dist = float(np.hypot(dx, dy))   # sqrt(dx^2 + dy^2)


        out["hip_y"] = hip_y
        out["ankle_distance"] = ankle_dist

        self.hip_y_hist.append(hip_y)
        self.ankle_y_hist.append(ankle_y)
        self.ankle_dist_hist.append(ankle_dist)

        # jump type
        jump_type = "unknown"
        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            jump_type = "single" if avg_dist >= ANKLE_DISTANCE_THRESHOLD else "double"
        out["jump_type"] = jump_type

        # baseline + airborne
        ground_y = self._compute_ground_y()
        if ground_y is None:
            return out

        # in image coords: smaller y => higher (airborne)
        airborne_now = ankle_y < (ground_y - LIFT_THRESHOLD_PX)

        if airborne_now:
            self.airborne_frames += 1
            self.ground_frames = 0
        else:
            self.ground_frames += 1
            self.airborne_frames = 0

        # confirm airborne / ground
        if (not self.is_airborne) and (self.airborne_frames >= AIRBORNE_CONFIRM_FRAMES):
            self.is_airborne = True
            self.airborne_start_frame = frame_idx

        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            # landing event candidate
            landing_frame = frame_idx
            start = self.airborne_start_frame if self.airborne_start_frame is not None else landing_frame
            airborne_len = landing_frame - start

            # hip amplitude check
            hip_amp_ok = False
            if len(self.hip_y_hist) >= int(self.fps * 0.4):
                recent = np.array(list(self.hip_y_hist)[-int(self.fps * 0.4):], dtype=np.float32)
                hip_amp = float(np.max(recent) - np.min(recent))
                hip_amp_ok = hip_amp >= HIP_AMPLITUDE_MIN_PX

            # refractory
            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

            # OPTIONAL: duration gate, disabled for now
            # duration_ok = (MIN_JUMP_FRAMES <= airborne_len <= MAX_JUMP_FRAMES)

            if hip_amp_ok and refractory_ok:
                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(t_sec)

                # split counter by type
                if jump_type == "single":
                    self.single_count += 1
                elif jump_type == "double":
                    self.double_count += 1

            # reset airborne state
            self.is_airborne = False
            self.airborne_start_frame = None

        out["is_airborne"] = self.is_airborne
        out["is_jumping"] = bool(self.is_airborne)

        # frequency from counted jumps
        if len(self.jump_times) >= 2:
            intervals = np.diff(np.array(self.jump_times, dtype=np.float32))
            if intervals.size > 0:
                avg = float(np.mean(intervals[-5:]))  # last few
                if avg > 0:
                    out["freq_hz"] = 1.0 / avg
                    out["spm"] = out["freq_hz"] * 60.0

        out["jump_count"] = self.jump_count
        out["single_count"] = self.single_count   
        out["double_count"] = self.double_count   
        return out


# =========================
# State Machine
# =========================

class JumpRopeStateMachine:
    def __init__(self, fps: float):
        self.fps = float(fps)
        self.state = JumpRopeState.IDLE
        self.last_count_time: Optional[float] = None
        self.count_times = deque(maxlen=50)

    def update(self, t_sec: float, jump_count: int):
        if not hasattr(self, "_last_seen_count"):
            self._last_seen_count = jump_count

        if jump_count > self._last_seen_count:
            self.count_times.append(t_sec)
            self.last_count_time = t_sec
            # fix the bug that it never goes out of STOPPED_ACTIVE or STOPPED_TRIPPED state
            if self.state == JumpRopeState.JUMPING:
                self._last_seen_count = jump_count

        time_since_last = (t_sec - self.last_count_time) if self.last_count_time is not None else 999.0

        stable = False
        if len(self.count_times) >= 3:
            intervals = np.diff(np.array(self.count_times, dtype=np.float32))
            if intervals.size > 0:
                avg = float(np.mean(intervals[-3:]))
                spm = (60.0 / avg) if avg > 0 else 0.0
                stable = (SPM_VALID_MIN <= spm <= SPM_VALID_MAX)

        if self.state == JumpRopeState.IDLE:
            if jump_count >= MIN_STABLE_JUMPS_TO_ENTER:
                self.state = JumpRopeState.JUMPING

        elif self.state == JumpRopeState.JUMPING:
            if time_since_last > STOP_SUDDEN_SEC and stable:
                self.state = JumpRopeState.STOPPED_TRIPPED
            elif time_since_last > STOP_GRADUAL_SEC:
                self.state = JumpRopeState.STOPPED_ACTIVE

        elif self.state in (JumpRopeState.STOPPED_ACTIVE, JumpRopeState.STOPPED_TRIPPED):
            if jump_count > self._last_seen_count:
                self.state = JumpRopeState.JUMPING


# =========================
# Drawing
# =========================

def draw_annotations(frame: np.ndarray, jump_count: int, single_count: int, double_count: int,
                     state: JumpRopeState, jump_type: str, spm: float) -> np.ndarray:
    annotated = frame.copy()

    colors = {
        JumpRopeState.IDLE: (160, 160, 160),
        JumpRopeState.JUMPING: (0, 255, 0),
        JumpRopeState.STOPPED_ACTIVE: (0, 165, 255),
        JumpRopeState.STOPPED_TRIPPED: (0, 0, 255),
    }
    c = colors.get(state, (255, 255, 255))

    y = 30
    cv2.putText(annotated, f"Jump Count: {jump_count} (S:{single_count} D:{double_count})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    y += 30
    cv2.putText(annotated, f"State: {state.value}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    y += 30
    cv2.putText(annotated, f"Type: {jump_type}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    y += 30
    cv2.putText(annotated, f"SPM: {spm:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    return annotated

def main():
    print("=" * 60)
    print("Jump Rope Detection")
    print("=" * 60)

    video_path = Path(VIDEO_SOURCE)
    if not video_path.exists():
        print(f"ERROR: video not found: {VIDEO_SOURCE}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {VIDEO_SOURCE}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps_src = float(fps_src) if fps_src and fps_src > 1e-3 else FPS_FALLBACK
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {width}x{height}, FPS={fps_src:.2f}, frames={total_frames}")

    model_path = Path(MODEL_PATH)
    int8_model_path = model_path.parent / f"{model_path.stem}_int8_openvino_model"
    if not int8_model_path.exists():
        print("Exporting INT8 OpenVINO model...")
        try:
            from ultralytics.yoloVideo_int8 import export_int8_openvino
            int8_model_path = export_int8_openvino(str(model_path), CALIB_DATA)
        except Exception as e:
            print("ERROR: cannot export int8 openvino model. Check your export script/module.")
            print(e)
            return
    else:
        print(f"Using existing INT8 model: {int8_model_path}")

    model = YOLO(str(int8_model_path))

    detector = JumpDetector(fps_src)
    sm = JumpRopeStateMachine(fps_src)

    out_dir = Path("jump_rope_results")
    out_dir.mkdir(exist_ok=True)
    out_video_path = out_dir / f"{video_path.stem}_jump_rope.mp4"
    out_csv_path = out_dir / f"{video_path.stem}_jump_rope.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_src, (width, height))

    frame_data = []
    frame_idx = 0
    t0 = time.time()

    for res in model.track(
        source=str(video_path),
        imgsz=640,
        stream=True,
        verbose=False,
        persist=True,
    ):
        t_sec = frame_idx / fps_src

        lhip = extract_keypoint_xy(res, LEFT_HIP)
        rhip = extract_keypoint_xy(res, RIGHT_HIP)
        lank = extract_keypoint_xy(res, LEFT_ANKLE)
        rank = extract_keypoint_xy(res, RIGHT_ANKLE)

        det = detector.update(frame_idx, t_sec, lhip, rhip, lank, rank)

        sm.update(t_sec, det["jump_count"])

        spm = float(det["spm"])

        annotated = res.plot()
        if annotated.shape[0] != height or annotated.shape[1] != width:
            annotated = cv2.resize(annotated, (width, height))

        annotated = draw_annotations(
            annotated,
            det["jump_count"],
            det["single_count"],
            det["double_count"],
            sm.state,
            det["jump_type"],
            spm,
        )

        writer.write(annotated)

        frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": t_sec,
            "hip_y": det["hip_y"],
            "ankle_distance": det["ankle_distance"],
            "is_airborne": det["is_airborne"],
            "jump_count": det["jump_count"],
            "single_count": det["single_count"],   
            "double_count": det["double_count"],   
            "jump_type": det["jump_type"],
            "freq_hz": det["freq_hz"],
            "spm": spm,
            "state": sm.state.value,
        })

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(
                f"Processed {frame_idx}/{total_frames} | "
                f"total={det['jump_count']} (S:{det['single_count']} D:{det['double_count']}) | "
                f"state={sm.state.value} | spm={spm:.1f}"
            )
        if frame_idx == total_frames:
            print(
                f"Processed {frame_idx}/{total_frames} | "
                f"total={det['jump_count']} (S:{det['single_count']} D:{det['double_count']}) | "
                f"state={sm.state.value} | spm={spm:.1f}"
            )
    writer.release()

    df = pd.DataFrame(frame_data)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    dt = time.time() - t0
    print("\nDONE")
    print(f"Total count:  {detector.jump_count}")
    print(f"Single count: {detector.single_count}")
    print(f"Double count: {detector.double_count}")
    print(f"Output video: {out_video_path}")
    print(f"Output csv:   {out_csv_path}")


if __name__ == "__main__":
    main()
