from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Any
import cv2
import numpy as np
import pandas as pd

from .config import (
    Config,
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE
)
from .pose_model import PoseModel
from .tracking import MultiPersonTracker, draw_person_panel
from .utils_pose import (
    draw_pose_overlay,
    bbox_height_px,
    extract_keypoint_xy_from_det,
)


class JumpRopeApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        print("=" * 60)
        print("Jump Rope Detection (Multi-Person) OpenVINO + static INT8 ONNX")
        print("=" * 60)

        video_path = Path(self.cfg.video_source)
        model_path = Path(self.cfg.model_path)

        if not video_path.exists():
            print(f"ERROR: video not found: {self.cfg.video_source}")
            return
        if not model_path.exists():
            print(f"ERROR: model not found: {self.cfg.model_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: cannot open video: {self.cfg.video_source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        print(f"Video: {width}x{height}, FPS={fps_src:.2f}, frames={total_frames}")

        model = PoseModel(self.cfg, device="CPU")
        tracker = MultiPersonTracker(self.cfg, fps=fps_src)

        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(exist_ok=True)

        out_video_path = out_dir / f"{video_path.stem}_jump_rope_multi.mp4"
        out_csv_path = out_dir / f"{video_path.stem}_jump_rope_multi.csv"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_src, (width, height))

        frame_data = []
        frame_idx = 0
        t_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_sec = frame_idx / fps_src

            # --- Inference ---
            det_list, infer_ms = model.infer(frame)

            # --- Update tracker ---
            tracker.update(det_list, t_sec, width, height)

            # --- Run jump logic per track ---
            for tid, tr in tracker.tracks.items():
                if tr.last_det is None:
                    continue

                det57 = tr.last_det
                conf = float(det57[4])

                draw_pose_overlay(frame, det57, self.cfg, color=tr.color)

                person_h = bbox_height_px(det57, width, height, self.cfg)
                if person_h <= 1.0:
                    person_h = max(1.0, 0.5 * height)

                lhip = rhip = lank = rank = nose = leye = reye = None
                if conf >= self.cfg.det_thresh:
                    lhip = extract_keypoint_xy_from_det(det57, LEFT_HIP, width, height, self.cfg)
                    rhip = extract_keypoint_xy_from_det(det57, RIGHT_HIP, width, height, self.cfg)
                    lank = extract_keypoint_xy_from_det(det57, LEFT_ANKLE, width, height, self.cfg)
                    rank = extract_keypoint_xy_from_det(det57, RIGHT_ANKLE, width, height, self.cfg)
                    nose = extract_keypoint_xy_from_det(det57, NOSE, width, height, self.cfg)
                    leye = extract_keypoint_xy_from_det(det57, LEFT_EYE, width, height, self.cfg)
                    reye = extract_keypoint_xy_from_det(det57, RIGHT_EYE, width, height, self.cfg)

                det_out = tr.detector.update(
                    frame_idx, t_sec,
                    lhip, rhip, lank, rank,
                    nose, leye, reye,
                    person_h
                )
                tr.sm.update(t_sec, det_out["jump_count"], bool(det_out["head_drop_event"]))

                draw_person_panel(frame, det57, self.cfg, tr, det_out)

                frame_data.append({
                    "frame_idx": frame_idx,
                    "timestamp": t_sec,
                    "track_id": tid,
                    "conf": conf,
                    "best_conf_seen": tr.best_conf_seen,
                    "infer_ms": infer_ms,
                    "hip_y": det_out["hip_y"],
                    "head_y": det_out["head_y"],
                    "head_move_px": det_out["head_move_px"],
                    "head_drop_event": det_out["head_drop_event"],
                    "ankle_distance": det_out["ankle_distance"],
                    "is_airborne": det_out["is_airborne"],
                    "jump_count": det_out["jump_count"],
                    "single_count": det_out["single_count"],
                    "double_count": det_out["double_count"],
                    "jump_type": det_out["jump_type"],
                    "freq_hz": det_out["freq_hz"],
                    "spm": float(det_out["spm"]),
                    "state": tr.sm.state.value,
                    "ground_y": det_out.get("ground_y", float("nan")),
                    "person_h": person_h,
                })

            # HUD
            cv2.putText(
                frame,
                f"{infer_ms:.1f} ms | tracks={len(tracker.tracks)}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            frame_idx += 1
            if frame_idx % 50 == 0:
                summary = sorted(
                    [(tid, tr.detector.jump_count) for tid, tr in tracker.tracks.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                print(f"Processed {frame_idx}/{total_frames} | active_tracks={len(tracker.tracks)} | top={summary}")

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Save per-frame CSV
        df = pd.DataFrame(frame_data)
        df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

        # Merge active + finished for final reporting
        final_people: Dict[int, Dict[str, Any]] = dict(tracker.finished)

        for tid, tr in tracker.tracks.items():
            final_people[tid] = {
                "track_id": tid,
                "final_jump_count": tr.detector.jump_count,
                "final_single_count": tr.detector.single_count,
                "final_double_count": tr.detector.double_count,
                "last_state": tr.sm.state.value,
                "last_seen_t": tr.last_seen_t,
                "best_conf_seen": tr.best_conf_seen,
            }

        rows = sorted(final_people.values(), key=lambda r: r["final_jump_count"], reverse=True)

        print("\n================ PER-PERSON TOTALS ================")
        for r in rows:
            print(
                f"Person ID {r['track_id']}: total={r['final_jump_count']} "
                f"(S:{r['final_single_count']} D:{r['final_double_count']}) "
                f"best_conf={float(r.get('best_conf_seen', float('nan'))):.3f} "
                f"last_state={r['last_state']}"
            )

        if self.cfg.summary_csv:
            summary_path = Path(self.cfg.out_dir) / f"{video_path.stem}_jump_rope_multi_summary.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"Summary csv:  {summary_path}")

        dt = time.time() - t_start
        print("\nDONE")
        print(f"Output video: {out_video_path}")
        print(f"Output csv:   {out_csv_path}")
        print(f"Time: {dt:.2f}s")


def main():
    cfg = Config()
    JumpRopeApp(cfg).run()


if __name__ == "__main__":
    main()
