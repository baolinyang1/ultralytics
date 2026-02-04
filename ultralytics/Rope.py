"""
Jump Rope Detection - YOLO26s-pose INT8
Features:
- Jump rope counting (robust: airborne -> landing event)
- Single/Double foot jump type
- State detection (IDLE / JUMPING / STOPPED_ACTIVE / STOPPED_TRIPPED)
- Speed estimation (SPM)
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

MODEL_PATH = "TestModels/yolo26s-pose.pt"
CALIB_DATA = "coco8-pose.yaml"
VIDEO_SOURCE = "TestVideos/rock2-25.mp4"
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
FPS_FALLBACK = 25.0
# Airborne detection
LIFT_THRESHOLD_PX = 3
AIRBORNE_CONFIRM_FRAMES = 1
GROUND_CONFIRM_FRAMES = 1
GROUND_HISTORY_SECONDS = 1.0
# Jump type
ANKLE_DISTANCE_THRESHOLD = 20
# Quality gating
HIP_AMPLITUDE_MIN_PX = 2.4
REFRACTORY_FRAMES = 1
# STOP detection 
STOP_SUDDEN_SEC = 1.5  # no new counted jumps for this long => STOPPED_*
# Passive stop (TRIPPED) head event thresholds
HEAD_EVENT_WINDOW_SEC = 0.45      # head event must be recent (before/around stop)
HEAD_DROP_BELOW_BASELINE_PX = 5.0 # head drops this many px below normal lowest point during jumping

# State
class JumpRopeState(Enum):
    IDLE = "IDLE"
    JUMPING = "JUMPING"
    STOPPED_ACTIVE = "STOPPED_ACTIVE"   # 主动停止
    STOPPED_TRIPPED = "STOPPED_TRIPPED" # 被动停止

# Kalman Filter
class KalmanFilter2D:
    """2D constant-velocity KF for smoothing (x,y)."""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state = np.zeros(4, dtype=np.float32)  # [x,y,vx,vy]
        self.cov = np.eye(4, dtype=np.float32) * 1000.0

        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

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

# Utils
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

# Jump Detector
class JumpDetector:
    """
    Counting:
    - ankle_y relative to ground baseline => airborne
    - count at landing event (airborne -> ground)
    - gate with hip amplitude + refractory
    - head_drop_event: nose/eyes sudden big downward move， for passive stop
    """
    def __init__(self, fps: float):
        self.fps = float(fps)
        # KF for body
        self.kf_lhip = KalmanFilter2D(0.01, 0.1)
        self.kf_rhip = KalmanFilter2D(0.01, 0.1)
        self.kf_lank = KalmanFilter2D(0.01, 0.1)
        self.kf_rank = KalmanFilter2D(0.01, 0.1)
        # KF for head
        self.kf_nose = KalmanFilter2D(0.01, 0.1)
        self.kf_leye = KalmanFilter2D(0.01, 0.1)
        self.kf_reye = KalmanFilter2D(0.01, 0.1)
        # histories
        self.hip_y_hist = deque(maxlen=int(self.fps * 2.0))
        self.ankle_y_hist = deque(maxlen=int(self.fps * GROUND_HISTORY_SECONDS))
        self.ankle_dist_hist = deque(maxlen=10)
        self.prev_head_xy: Optional[np.ndarray] = None
        self.head_y_hist = deque(maxlen=int(self.fps * 1.0))  # 1s
        
        # normal lowest head Y during jumping (baseline for trip detection)
        self.normal_lowest_head_y: Optional[float] = None
        self.jumping_head_y_samples = deque(maxlen=int(self.fps * 2.0))  # track head Y during jumping

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
        self.jump_times = deque(maxlen=50)

    def _compute_ground_y(self) -> Optional[float]:
        if len(self.ankle_y_hist) < max(8, int(self.fps * 0.3)):
            return None
        arr = np.array(self.ankle_y_hist, dtype=np.float32)
        return safe_percentile(arr, 90.0)

    def _smooth_head(self,
                     nose: Optional[Tuple[float, float]],
                     leye: Optional[Tuple[float, float]],
                     reye: Optional[Tuple[float, float]]) -> Optional[np.ndarray]:
        """Return head point (x,y) as mean of available (nose/eyes) after KF."""
        pts = []
        if nose:
            pts.append(self.kf_nose.update(*nose))
        if leye:
            pts.append(self.kf_leye.update(*leye))
        if reye:
            pts.append(self.kf_reye.update(*reye))
        if not pts:
            return None
        arr = np.stack(pts, axis=0)  # (n,2)
        return np.mean(arr, axis=0)

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        left_hip: Optional[Tuple[float, float]],
        right_hip: Optional[Tuple[float, float]],
        left_ankle: Optional[Tuple[float, float]],
        right_ankle: Optional[Tuple[float, float]],
        nose: Optional[Tuple[float, float]],
        left_eye: Optional[Tuple[float, float]],
        right_eye: Optional[Tuple[float, float]],
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
            "head_y": float("nan"),
            "head_move_px": 0.0,
            "head_drop_event": False,
        }

        # need legs+hips to count
        if not (left_hip and right_hip and left_ankle and right_ankle):
            return out

        # smooth body
        lh = self.kf_lhip.update(*left_hip)
        rh = self.kf_rhip.update(*right_hip)
        la = self.kf_lank.update(*left_ankle)
        ra = self.kf_rank.update(*right_ankle)

        hip_y = float((lh[1] + rh[1]) / 2.0)
        ankle_y = float((la[1] + ra[1]) / 2.0)
        # total ankle distance
        dx = la[0] - ra[0]
        dy = la[1] - ra[1]
        ankle_dist = float(np.hypot(dx, dy))

        out["hip_y"] = hip_y
        out["ankle_distance"] = ankle_dist

        self.hip_y_hist.append(hip_y)
        self.ankle_y_hist.append(ankle_y)
        self.ankle_dist_hist.append(ankle_dist)

        # jump type (single/double)
        jump_type = "unknown"
        if len(self.ankle_dist_hist) >= 5:
            avg_dist = float(np.mean(list(self.ankle_dist_hist)[-5:]))
            jump_type = "single" if avg_dist >= ANKLE_DISTANCE_THRESHOLD else "double"
        out["jump_type"] = jump_type

        # head event compute (passive stop)
        head_xy = self._smooth_head(nose, left_eye, right_eye)
        if head_xy is not None:
            head_y = float(head_xy[1])
            out["head_y"] = head_y
            self.head_y_hist.append(head_y)
            # Track head Y during jumping to establish normal lowest baseline
            # Consider actively jumping if we have jump_count > 0 and recent jumps
            is_actively_jumping = False
            if self.jump_count > 0 and len(self.jump_times) > 0:
                time_since_last_jump = t_sec - self.jump_times[-1]
                # Consider jumping if last jump was within 2 seconds
                is_actively_jumping = time_since_last_jump < 2.0
            
            if is_actively_jumping:
                # Track head Y samples during jumping
                self.jumping_head_y_samples.append(head_y)
                # Update normal lowest head Y using 90th percentile (more robust than max)
                # max Y = lowest point, since Y increases downward
                if len(self.jumping_head_y_samples) >= int(self.fps * 0.5):  # Need at least 0.5s of samples
                    arr = np.array(self.jumping_head_y_samples, dtype=np.float32)
                    self.normal_lowest_head_y = safe_percentile(arr, 90.0)
            else:
                # Not actively jumping - reset baseline if we've been idle too long
                if len(self.jump_times) == 0 or (t_sec - self.jump_times[-1]) > 3.0:
                    self.normal_lowest_head_y = None
                    self.jumping_head_y_samples.clear()

            # Calculate head movement for display
            if self.prev_head_xy is None:
                out["head_move_px"] = 0.0
            else:
                move_px = float(np.hypot(head_xy[0] - self.prev_head_xy[0], head_xy[1] - self.prev_head_xy[1]))
                dy_down = float(head_xy[1] - self.prev_head_xy[1])  # positive => moved down
                out["head_move_px"] = move_px

            # Check if head dropped 5 px below normal lowest point during jumping
            out["head_drop_event"] = False
            if self.normal_lowest_head_y is not None:
                # head_y > normal_lowest_head_y means head moved down (Y increases downward)
                # If head is 5 px below baseline, trigger trip event
                if head_y > (self.normal_lowest_head_y + HEAD_DROP_BELOW_BASELINE_PX):
                    out["head_drop_event"] = True

            self.prev_head_xy = head_xy

        # baseline + airborne
        ground_y = self._compute_ground_y()
        if ground_y is None:
            return out

        airborne_now = ankle_y < (ground_y - LIFT_THRESHOLD_PX)

        if airborne_now:
            self.airborne_frames += 1
            self.ground_frames = 0
        else:
            self.ground_frames += 1
            self.airborne_frames = 0

        if (not self.is_airborne) and (self.airborne_frames >= AIRBORNE_CONFIRM_FRAMES):
            self.is_airborne = True
            self.airborne_start_frame = frame_idx

        if self.is_airborne and (self.ground_frames >= GROUND_CONFIRM_FRAMES):
            # landing event candidate
            # hip amplitude gate
            hip_amp_ok = False
            if len(self.hip_y_hist) >= int(self.fps * 0.4):
                recent = np.array(list(self.hip_y_hist)[-int(self.fps * 0.4):], dtype=np.float32)
                hip_amp = float(np.max(recent) - np.min(recent))
                hip_amp_ok = hip_amp >= HIP_AMPLITUDE_MIN_PX

            refractory_ok = (frame_idx - self.last_count_frame) >= REFRACTORY_FRAMES

            if hip_amp_ok and refractory_ok:
                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(t_sec)

                if jump_type == "single":
                    self.single_count += 1
                elif jump_type == "double":
                    self.double_count += 1

            self.is_airborne = False
            self.airborne_start_frame = None

        out["is_airborne"] = self.is_airborne
        out["is_jumping"] = bool(self.is_airborne)

        # freq / spm from counted jumps
        if len(self.jump_times) >= 2:
            intervals = np.diff(np.array(self.jump_times, dtype=np.float32))
            if intervals.size > 0:
                avg = float(np.mean(intervals[-5:]))
                if avg > 0:
                    out["freq_hz"] = 1.0 / avg
                    out["spm"] = out["freq_hz"] * 60.0

        out["jump_count"] = self.jump_count
        out["single_count"] = self.single_count
        out["double_count"] = self.double_count
        return out

# State Machine 
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

        # new jump counted => JUMPING, optimized
        if jump_count > self._last_seen_count:
            self._last_seen_count = jump_count
            self.last_count_time = t_sec
            self.state = JumpRopeState.JUMPING
            return

        time_since_last = (t_sec - self.last_count_time) if self.last_count_time is not None else 999.0

        # IDLE -> JUMPING once we start counting
        if self.state == JumpRopeState.IDLE:
            if jump_count > 0:
                self.state = JumpRopeState.JUMPING

        # sudden stop => decide ACTIVE vs TRIPPED
        if self.state == JumpRopeState.JUMPING:
            if time_since_last > STOP_SUDDEN_SEC:
                if self._head_event_recent(t_sec):
                    self.state = JumpRopeState.STOPPED_TRIPPED   # 被动停止
                else:
                    self.state = JumpRopeState.STOPPED_ACTIVE    # 主动停止

# Drawing
def draw_annotations(frame: np.ndarray, jump_count: int, single_count: int, double_count: int,
                     state: JumpRopeState, jump_type: str, spm: float,
                     head_move_px: float, head_drop_event: bool) -> np.ndarray:
    annotated = frame.copy()

    colors = {JumpRopeState.IDLE: (160, 160, 160),JumpRopeState.JUMPING: (0, 255, 0),
        JumpRopeState.STOPPED_ACTIVE: (0, 165, 255), JumpRopeState.STOPPED_TRIPPED: (0, 0, 255),
    }
    c = colors.get(state, (255, 255, 255))

    y = 30
    cv2.putText(annotated, f"Jump Count: {jump_count} (S:{single_count} D:{double_count})",(10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 30
    cv2.putText(annotated, f"State: {state.value}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 30
    cv2.putText(annotated, f"Type: {jump_type}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 30
    cv2.putText(annotated, f"SPM: {spm:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    y += 30
    #cv2.putText(annotated, f"DropEvent: {int(head_drop_event)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
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
        nose = extract_keypoint_xy(res, NOSE)
        leye = extract_keypoint_xy(res, LEFT_EYE)
        reye = extract_keypoint_xy(res, RIGHT_EYE)

        det = detector.update(frame_idx, t_sec, lhip, rhip, lank, rank, nose, leye, reye) #update each frame's keypoints

        sm.update(t_sec, det["jump_count"], bool(det["head_drop_event"])) #update each frame's state 

        spm = float(det["spm"])

        annotated = res.plot()
        if annotated.shape[0] != height or annotated.shape[1] != width:
            annotated = cv2.resize(annotated, (width, height))

        annotated = draw_annotations( #draw!!!
            annotated,
            det["jump_count"],
            det["single_count"],
            det["double_count"],
            sm.state,
            det["jump_type"],
            spm,
            float(det["head_move_px"]),
            bool(det["head_drop_event"]),
        )
        writer.write(annotated)

        frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": t_sec,
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
            "spm": spm,
            "state": sm.state.value,
        })

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(
                f"Processed {frame_idx}/{total_frames} | "
                f"total={det['jump_count']} (S:{det['single_count']} D:{det['double_count']}) | "
                f"state={sm.state.value} | spm={spm:.1f} | "
                #f"headDrop={int(det['head_drop_event'])}"
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
    print(f"Time: {dt:.2f}s")
if __name__ == "__main__":
    main()