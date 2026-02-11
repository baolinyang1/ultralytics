from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2

from .config import Config, SKELETON


def safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def decode_bbox_xyxy_modelpx(det57: np.ndarray) -> Tuple[float, float, float, float, float]:
    """YOLO det row: [x1,y1,x2,y2,score,cls?, ... keypoints ...] in *model* pixel coords."""
    x1, y1, x2, y2, score = det57[:5]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return float(x1), float(y1), float(x2), float(y2), float(score)


def decode_kpts_17x3(det57: np.ndarray) -> np.ndarray:
    """Return (17,3) where each row is (x,y,conf) in model pixel coords."""
    kpt_flat = det57[6:6 + 51]
    return kpt_flat.reshape(17, 3).astype(np.float32)


def map_modelpx_to_frame(xy_model: np.ndarray, w: int, h: int, cfg: Config) -> np.ndarray:
    sx = w / float(cfg.img_size)
    sy = h / float(cfg.img_size)
    out = xy_model.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def bbox_height_px(det57: np.ndarray, w: int, h: int, cfg: Config) -> float:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    sy = h / float(cfg.img_size)
    return float(abs((y2m - y1m) * sy))


def bbox_frame_xyxy(det57: np.ndarray, w: int, h: int, cfg: Config) -> Tuple[int, int, int, int]:
    x1m, y1m, x2m, y2m, _ = decode_bbox_xyxy_modelpx(det57)
    sx = w / float(cfg.img_size)
    sy = h / float(cfg.img_size)
    x1, y1 = int(x1m * sx), int(y1m * sy)
    x2, y2 = int(x2m * sx), int(y2m * sy)
    return x1, y1, x2, y2


def bbox_center_frame(det57: np.ndarray, w: int, h: int, cfg: Config) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_frame_xyxy(det57, w, h, cfg)
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def bbox_diag_frame(det57: np.ndarray, w: int, h: int, cfg: Config) -> float:
    x1, y1, x2, y2 = bbox_frame_xyxy(det57, w, h, cfg)
    return float(np.hypot(x2 - x1, y2 - y1))


def extract_keypoint_xy_from_det(
    det57: np.ndarray,
    kpt_idx: int,
    w: int,
    h: int,
    cfg: Config,
    min_kpt_conf: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if min_kpt_conf is None:
        min_kpt_conf = cfg.min_kpt_conf

    kpts = decode_kpts_17x3(det57)
    if not (0 <= kpt_idx < 17):
        return None
    if float(kpts[kpt_idx, 2]) < float(min_kpt_conf):
        return None

    xy_frame = map_modelpx_to_frame(kpts[:, :2], w, h, cfg)
    x, y = xy_frame[kpt_idx]
    return float(x), float(y)


def draw_pose_overlay(frame: np.ndarray, det57: np.ndarray, cfg: Config, color=(0, 255, 0)):
    """Draw bbox + 17 keypoints + skeleton."""
    h, w = frame.shape[:2]
    x1m, y1m, x2m, y2m, conf = decode_bbox_xyxy_modelpx(det57)
    sx = w / float(cfg.img_size)
    sy = h / float(cfg.img_size)
    x1, y1 = int(x1m * sx), int(y1m * sy)
    x2, y2 = int(x2m * sx), int(y2m * sy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{conf:.3f}",
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    kpts = decode_kpts_17x3(det57)
    xy_frame = map_modelpx_to_frame(kpts[:, :2], w, h, cfg)
    kconf = kpts[:, 2]

    pts = []
    for i in range(17):
        cx, cy = int(xy_frame[i, 0]), int(xy_frame[i, 1])
        pts.append((cx, cy))
        if float(kconf[i]) >= cfg.min_kpt_conf:
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    for a, b in SKELETON:
        if float(kconf[a]) >= cfg.min_kpt_conf and float(kconf[b]) >= cfg.min_kpt_conf:
            ax, ay = pts[a]
            bx, by = pts[b]
            cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)
