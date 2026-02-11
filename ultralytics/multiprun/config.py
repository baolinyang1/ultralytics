from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    # ---------------- CONFIG ----------------
    model_path: str = "yolo26n-pose.static_int8.onnx"
    video_source: str = "TestVideos/1053_25fps.mp4"
    img_size: int = 640

    det_thresh: float = 0.3
    min_kpt_conf: float = 0.25

    # Multi-person tracking params
    max_track_age_sec: float = 1.0     # if a person not seen for this long => drop track
    match_dist_ratio: float = 0.65     # max match distance = ratio * bbox_diag
    max_tracks: int = 20               # safety

    # Save summary CSV of per-person totals
    summary_csv: bool = True

    # Airborne detection (frame confirmations)
    airborne_confirm_frames: int = 1
    ground_confirm_frames: int = 1
    ground_history_seconds: float = 1.5

    # Quality gating
    refractory_frames: int = 7

    # STOP detection
    stop_sudden_sec: float = 1.5

    # Passive stop (TRIPPED) head event thresholds (normalized by person height)
    head_event_window_sec: float = 0.45

    # Output directory
    out_dir: str = "jump_rope_results"


# Keypoint indices (COCO-17)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Skeleton connections (COCO-17)
SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 8), (5, 6),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]
