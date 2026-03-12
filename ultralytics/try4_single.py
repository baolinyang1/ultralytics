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
MODEL_PATH = "model_int8.onnx"
VIDEO_SOURCE = "TestVideos/107.2.mp4"
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
REFRACTORY_FRAMES = 10
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
HIP_LIFT_MAX_FRAC = 0.085
HIP_NOISE_BAND_FRAC = 0.18

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


# ---------------- UTILS (NEW MODEL: normalized outputs) ----------------
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img

def decode_bbox_xyxy_norm(det57: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    New model: bbox is normalized to full frame: x1,y1,x2,y2 in [0..1] (usually).
    """
    x1, y1, x2, y2, score = det57[:5]
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return float(x1), float(y1), float(x2), float(y2), float(score)

def decode_kpts_17x3(det57: np.ndarray) -> np.ndarray:
    """
    det length 57: [0..3]=bbox, [4]=conf, [5]=cls, [6..56]=51 kpt vals
    kpts are normalized to full frame (0..1), conf is at [:,2]
    """
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

def get_kpt_xy(det57: np.ndarray, idx: int, w: int, h: int) -> Optional[Tuple[float, float]]:
    kpts = decode_kpts_17x3(det57)  # normalized
    if float(kpts[idx, 2]) < MIN_KPT_CONF:
        return None
    xy = map_norm_to_frame_xy(kpts[:, :2], w, h)
    return float(xy[idx, 0]), float(xy[idx, 1])

def draw_pose(frame: np.ndarray, det57: np.ndarray):
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n, conf = decode_bbox_xyxy_norm(det57)
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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


# ---------------- JUMP DETECTOR (HIP AIRBORNE, SHOULDER CYCLE CHECK) ----------------
class JumpDetector:
    """
    - Airborne detection & landing event: HIP-based (hip_y vs ground_y - lift_th)
    - Cycle/cadence vote: SHOULDER-driven (dt history + EWMA expected interval)
    - Amplitude gate: SHOULDER amplitude (to suppress walk/stop noise)
    - Dynamic lift threshold: still uses amp_ewma + ground noise sigma from HIP
    """

    def __init__(self, fps: float):
        self.fps = float(fps)

        # hip/ankle filters
        self.kf_lhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_rhip = KalmanFilter2D(process_noise=0.005, measurement_noise=0.12)
        self.kf_lank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)
        self.kf_rank = KalmanFilter2D(process_noise=0.03, measurement_noise=0.08)

        # shoulder filters
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

        # cadence (SHOULDER-based)
        self.shoulder_interval_hist = deque(maxlen=20)
        self.shoulder_expected_interval: Optional[float] = None

        # for dynamic lift
        self.amp_ewma: Optional[float] = None

    def _reset_cadence(self):
        self.jump_times.clear()
        self.shoulder_interval_hist.clear()
        self.shoulder_expected_interval = None

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
        tmp = list(self.shoulder_interval_hist) + [float(dt)]
        if len(tmp) < CYCLE_WINDOW:
            return True

        window = np.array(tmp[-CYCLE_WINDOW:], dtype=np.float32)
        base = float(self.shoulder_expected_interval) if self.shoulder_expected_interval else float(np.median(window))

        if len(self.shoulder_interval_hist) == 0:
            mad = MIN_MAD_SEC
        else:
            hist = np.array(self.shoulder_interval_hist, dtype=np.float32)
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
               lsho, rsho,
               person_h: float) -> Dict[str, Any]:

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
            "cycle_ok": False,
            "dt": float("nan"),
            "expected_dt": float("nan"),
            "amp": float("nan"),
            "min_air_y": float("nan"),
        }

        if not (lhip and rhip and lank and rank and lsho and rsho) or person_h <= 1.0:
            return out

        amp_th = max(AMP_MIN_PX, AMP_FRAC * person_h)

        lh = self.kf_lhip.update(*lhip)
        rh = self.kf_rhip.update(*rhip)
        la = self.kf_lank.update(*lank)
        ra = self.kf_rank.update(*rank)
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

        ground_y = float(np.percentile(np.array(self.hip_y_hist, dtype=np.float32), 90.0))
        out["ground_y"] = ground_y

        sigma = self._estimate_ground_noise_sigma(self.hip_y_hist, ground_y)
        lift_th = self._dynamic_lift_th(person_h, sigma)
        out["lift_noise_sigma"] = float(sigma)
        out["lift_th"] = float(lift_th)
        out["amp_ewma"] = float(self.amp_ewma) if self.amp_ewma is not None else float("nan")

        if len(self.shoulder_y_hist) >= 10:
            shoulder_ground_y = float(np.percentile(np.array(self.shoulder_y_hist, dtype=np.float32), 90.0))
        else:
            shoulder_ground_y = float("nan")
        out["shoulder_ground_y"] = shoulder_ground_y

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

                if np.isnan(shoulder_amp) or shoulder_amp < amp_th:
                    return out

                cycle_ok = True
                if dt is not None:
                    cycle_ok = self._cycle_vote_ok(dt)
                out["cycle_ok"] = bool(cycle_ok)

                self.jump_count += 1
                self.last_count_frame = frame_idx
                self.jump_times.append(float(t_sec))

                if dt is not None:
                    self.shoulder_interval_hist.append(float(dt))
                    if self.shoulder_expected_interval is None:
                        self.shoulder_expected_interval = float(dt)
                    else:
                        self.shoulder_expected_interval = float(
                            (1.0 - EWMA_ALPHA) * self.shoulder_expected_interval + EWMA_ALPHA * dt
                        )
                    out["expected_dt"] = float(self.shoulder_expected_interval)

                if self.amp_ewma is None:
                    self.amp_ewma = float(hip_amp)
                else:
                    self.amp_ewma = float((1.0 - AMP_EWMA_ALPHA) * self.amp_ewma + AMP_EWMA_ALPHA * hip_amp)
                out["amp_ewma"] = float(self.amp_ewma)

        out["is_airborne"] = self.is_airborne
        out["jump_count"] = self.jump_count
        if self.shoulder_expected_interval is not None:
            out["expected_dt"] = float(self.shoulder_expected_interval)

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


def draw_hud(frame: np.ndarray, det: Dict[str, Any], state: JumpRopeState, infer_ms: float):
    c = (160, 160, 160) if state == JumpRopeState.IDLE else (0, 255, 0) if state == JumpRopeState.JUMPING else (0, 165, 255)

    y = 30
    cv2.putText(frame, f"Jump: {det['jump_count']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    y += 26
    cv2.putText(frame, f"State: {state.value}  SPM: {det['spm']:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)


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

    out_video_path = f"jump_rope_results/jump_rope_newest_{os.path.basename(VIDEO_SOURCE).split('_')[0]}.mp4"
    out_csv_path = f"jump_rope_results/jump_rope_newest_{os.path.basename(VIDEO_SOURCE).split('_')[0]}.csv"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_data = []
    frame_idx = 0
    prev_center_px = None
    prev_h_px = None

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
            frame_idx += 1
            continue

        # tracking: pick best match near previous bbox center (NOW in frame pixels)
        if prev_center_px is None:
            best = valid[int(np.argmax(valid[:, 4]))]
        else:
            centers_px = np.array([bbox_center_px(d, width, height) for d in valid], dtype=np.float32)
            dist2 = (centers_px[:, 0] - float(prev_center_px[0])) ** 2 + (centers_px[:, 1] - float(prev_center_px[1])) ** 2

            heights_px = np.array([bbox_height_px(d, width, height) for d in valid], dtype=np.float32)
            if prev_h_px is not None and prev_h_px > 1.0:
                mask = (heights_px >= 0.7 * prev_h_px) & (heights_px <= 1.3 * prev_h_px)
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    best = valid[idxs[int(np.argmin(dist2[idxs]))]]
                else:
                    best = valid[int(np.argmin(dist2))]
            else:
                best = valid[int(np.argmin(dist2))]

        prev_center_px = bbox_center_px(best, width, height)
        prev_h_px = bbox_height_px(best, width, height)
        best_conf = float(best[4])

        draw_pose(frame, best)

        person_h = bbox_height_px(best, width, height)

        lhip = get_kpt_xy(best, LEFT_HIP, width, height)
        rhip = get_kpt_xy(best, RIGHT_HIP, width, height)
        lank = get_kpt_xy(best, LEFT_ANKLE, width, height)
        rank = get_kpt_xy(best, RIGHT_ANKLE, width, height)
        lsho = get_kpt_xy(best, LEFT_SHOULDER, width, height)
        rsho = get_kpt_xy(best, RIGHT_SHOULDER, width, height)

        det = detector.update(frame_idx, t_sec, lhip, rhip, lank, rank, lsho, rsho, person_h)
        det["jump_count"] = sm.update(t_sec, det["jump_count"])
        detector.jump_count = det["jump_count"]

        draw_hud(frame, det, sm.state, infer_ms)
        writer.write(frame)

        frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": t_sec,
            "best_conf": best_conf,
            "infer_ms": infer_ms,
            "jump_count": det["jump_count"],
            "spm": det["spm"],
            "state": sm.state.value,
            "cycle_ok": bool(det.get("cycle_ok", False)),
            "dt": det.get("dt", float("nan")),
            "expected_dt": det.get("expected_dt", float("nan")),
            "hip_amp": det.get("amp", float("nan")),
            "shoulder_amp": det.get("shoulder_amp", float("nan")),
            "amp_ewma": det.get("amp_ewma", float("nan")),
            "lift_th": det.get("lift_th", float("nan")),
            "lift_noise_sigma": det.get("lift_noise_sigma", float("nan")),
            "ground_y": det.get("ground_y", float("nan")),
            "hip_y": det.get("hip_y", float("nan")),
            "shoulder_ground_y": det.get("shoulder_ground_y", float("nan")),
            "shoulder_y": det.get("shoulder_y", float("nan")),
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