from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from .config import Config
from .jump_logic import JumpDetector, JumpRopeStateMachine
from .utils_pose import bbox_center_frame, bbox_diag_frame, bbox_frame_xyxy


def color_from_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973)
    c = rng.integers(low=40, high=255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


@dataclass
class PersonTrack:
    id: int
    detector: JumpDetector
    sm: JumpRopeStateMachine
    last_seen_t: float
    last_center: Tuple[float, float]
    last_det: Optional[np.ndarray]
    color: Tuple[int, int, int]
    best_conf_seen: float = 0.0


class MultiPersonTracker:
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = float(fps)
        self.tracks: Dict[int, PersonTrack] = {}
        self.finished: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1

    def _prepare_det_meta(self, det_list: List[np.ndarray], w: int, h: int):
        centers, diags = [], []
        for d in det_list:
            centers.append(bbox_center_frame(d, w, h, self.cfg))
            diags.append(max(1.0, bbox_diag_frame(d, w, h, self.cfg)))
        return centers, diags

    def update(self, det_list: List[np.ndarray], t_sec: float, w: int, h: int) -> None:
        det_centers, det_diags = self._prepare_det_meta(det_list, w, h)

        unmatched_det_idxs = set(range(len(det_list)))
        used_tracks = set()
        track_ids = list(self.tracks.keys())

        # candidates: (dist, track_id, det_idx)
        candidates = []
        for tid in track_ids:
            tcx, tcy = self.tracks[tid].last_center
            for di in unmatched_det_idxs:
                dcx, dcy = det_centers[di]
                dist = float(np.hypot(dcx - tcx, dcy - tcy))
                max_dist = self.cfg.match_dist_ratio * det_diags[di]
                if dist <= max_dist:
                    candidates.append((dist, tid, di))

        candidates.sort(key=lambda x: x[0])
        assignments = []

        for dist, tid, di in candidates:
            if tid in used_tracks or di not in unmatched_det_idxs:
                continue
            used_tracks.add(tid)
            unmatched_det_idxs.remove(di)
            assignments.append((tid, di))

        # new tracks for remaining detections
        for di in list(unmatched_det_idxs):
            if len(self.tracks) >= self.cfg.max_tracks:
                break
            tid = self.next_track_id
            self.next_track_id += 1

            tr = PersonTrack(
                id=tid,
                detector=JumpDetector(self.cfg, self.fps),
                sm=JumpRopeStateMachine(self.cfg, self.fps),
                last_seen_t=t_sec,
                last_center=det_centers[di],
                last_det=det_list[di],
                color=color_from_id(tid),
                best_conf_seen=float(det_list[di][4]),
            )
            self.tracks[tid] = tr
            assignments.append((tid, di))

        # update assigned tracks
        for tid, di in assignments:
            tr = self.tracks[tid]
            tr.last_seen_t = t_sec
            tr.last_center = det_centers[di]
            tr.last_det = det_list[di]
            tr.best_conf_seen = max(tr.best_conf_seen, float(det_list[di][4]))

        # drop old tracks (but store finished totals)
        to_drop = [tid for tid, tr in self.tracks.items() if (t_sec - tr.last_seen_t) > self.cfg.max_track_age_sec]
        for tid in to_drop:
            tr = self.tracks[tid]
            self.finished[tid] = {
                "track_id": tid,
                "final_jump_count": tr.detector.jump_count,
                "final_single_count": tr.detector.single_count,
                "final_double_count": tr.detector.double_count,
                "last_state": tr.sm.state.value,
                "last_seen_t": tr.last_seen_t,
                "best_conf_seen": tr.best_conf_seen,
            }
            del self.tracks[tid]


def draw_person_panel(frame: np.ndarray, det57: np.ndarray, cfg: Config, track: PersonTrack, det_out: Dict[str, Any]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_frame_xyxy(det57, w, h, cfg)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    tx = x1
    ty = max(0, y1 - 10)

    panel_lines = [
        f"ID {track.id}  {track.sm.state.value}",
        f"J:{det_out['jump_count']}  S:{det_out['single_count']} D:{det_out['double_count']}",
        f"Type:{det_out['jump_type']}  SPM:{float(det_out['spm']):.1f}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 2
    pad = 4

    sizes = [cv2.getTextSize(s, font, scale, thick)[0] for s in panel_lines]
    box_w = max(sz[0] for sz in sizes) + pad * 2
    box_h = sum(sz[1] + 6 for sz in sizes) + pad

    bx1 = tx
    by1 = max(0, ty - box_h)
    bx2 = min(w - 1, bx1 + box_w)
    by2 = min(h - 1, by1 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), track.color, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    y = by1 + pad + sizes[0][1]
    for s, sz in zip(panel_lines, sizes):
        cv2.putText(frame, s, (bx1 + pad, y), font, scale, (255, 255, 255), thick)
        y += sz[1] + 6

    ground_y = float(det_out.get("ground_y", float("nan")))
    if np.isfinite(ground_y):
        gy = int(ground_y)
        cv2.line(frame, (x1, gy), (x2, gy), track.color, 1)
