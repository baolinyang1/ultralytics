from pathlib import Path
import time
import cv2
import numpy as np
from openvino.runtime import Core

MODEL_PATH = "model_int8.onnx"
VIDEO_SOURCE = "onnx_video_results/still_cleaned_pose.mp4"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5

# 17 keypoints exist. Skeleton is just which points to CONNECT.
SKELETON = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 8), (5, 6),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# model outputs bbox is (x1, y1, x2, y2)
def decode_bbox_xyxy_norm(det: np.ndarray):
    x1, y1, x2, y2, score = det[:5]
    #x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2, float(score)

# keypoints start at index 6 (after bbox(4)+conf(1)+class(1)), det is one detection row from YOLO output: those 57 nums
def decode_kpts_17x3(det: np.ndarray) -> np.ndarray:
    # det length is 57: [0..3]=bbox, [4]=conf, [5]=cls, [6..56]=51 kpt vals
    kpt_flat = det[6:6 + 51]
    # use reshape to get 17x3 array -> (17, 3)
    return kpt_flat.reshape(17, 3).astype(np.float32)

def map_kpts_to_frame_fullframe_norm(kpts: np.ndarray, w: int, h: int) -> np.ndarray:
    # keypoints are normalized to full image (0..1), so scale directly
    xs = kpts[:, 0]
    ys = kpts[:, 1]
    return np.stack([xs * w, ys * h], axis=1)

def draw_pose(frame: np.ndarray, det: np.ndarray, w: int, h: int):
    x1n, y1n, x2n, y2n, score = decode_bbox_xyxy_norm(det)
    if score < DET_THRESH:
        return

    x1i, y1i = int(x1n * w), int(y1n * h)
    x2i, y2i = int(x2n * w), int(y2n * h)

    # bbox
    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.3f}", (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # kpts
    kpts = decode_kpts_17x3(det)
    pts_xy = map_kpts_to_frame_fullframe_norm(kpts, w, h)

    # draw ALL keypoints + index labels
    pts = []
    for i in range(17):
        cx, cy = int(pts_xy[i, 0]), int(pts_xy[i, 1])
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        pts.append((cx, cy))

    # draw skeleton
    for a, b in SKELETON:
        ax, ay = pts[a]
        bx, by = pts[b]
        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

def compute_kpt_stability_dist(kpt_positions: dict[int, list[tuple[float, float]]]) -> float:
    kpt_stds = []
    for k in range(17):
        pos = kpt_positions.get(k, [])
        if len(pos) >= 2:
            arr = np.array(pos, dtype=np.float32)  # (N,2)
            mean = arr.mean(axis=0)                # (2,)
            d = np.linalg.norm(arr - mean, axis=1) # (N,)
            kpt_stds.append(float(np.std(d)))
    return float(np.mean(kpt_stds)) if kpt_stds else 0.0

def compute_kpt_stds_dist(kpt_positions: dict[int, list[tuple[float, float]]]) -> list[float]:
    """
    Return the 17 per-keypoint stds (distance-to-mean). If a keypoint has <2 samples, its std is NaN.
    """
    out = []
    for k in range(17):
        pos = kpt_positions.get(k, [])
        if len(pos) >= 2:
            arr = np.array(pos, dtype=np.float32)  # (N,2)
            mean = arr.mean(axis=0)
            d = np.linalg.norm(arr - mean, axis=1)
            out.append(float(np.std(d)))
        else:
            out.append(float("nan"))
    return out

def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vid_path = Path(VIDEO_SOURCE)
    if not vid_path.exists():
        raise FileNotFoundError(f"Video not found: {vid_path}")

    # OpenVINO init
    core = Core()
    model = core.read_model(str(model_path))
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")
    output_layer = compiled.output(0)

    # Open video
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid_path}")

    # Read video properties (fallbacks included)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    # Output writer
    out_path = OUT_DIR / f"{vid_path.stem}_openvino_int8_pose_fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    print(f"Input:  {vid_path} ({in_w}x{in_h} @ {fps:.2f} fps)")
    print(f"Output: {out_path}")

    # --- stability collection ---
    kpt_positions: dict[int, list[tuple[float, float]]] = {k: [] for k in range(17)}
    stats_frames_used = 0

    # --- average inference time collection (ms/frame) ---
    infer_times_ms: list[float] = []

    frame_idx = 0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]

        # Inference
        t0 = time.perf_counter() # start time
        out = compiled([preprocess(frame)])[output_layer]  # (1, N, 57)
        preds = out[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # collect inference time for average
        infer_times_ms.append(float(infer_ms))

        # Filter detections by score
        preds_f = preds[preds[:, 4] >= DET_THRESH]

        if len(preds_f) > 0:
            # sort high->low score
            preds_f = preds_f[np.argsort(-preds_f[:, 4])]

            # draw top 3, does not matter here coz i have only one person in the video
            for det in preds_f[:3]:
                draw_pose(frame, det, w, h)

            # collect stability from the TOP detection only (highest score)
            best = preds_f[0]
            kpts = decode_kpts_17x3(best)  # (17,3) normalized
            pts_xy = map_kpts_to_frame_fullframe_norm(kpts, w, h)  # (17,2) pixels

            for k in range(17):
                x, y = float(pts_xy[k, 0]), float(pts_xy[k, 1])
                kpt_positions[k].append((x, y))
            stats_frames_used += 1

        else:
            cv2.putText(frame, "No detections above DET_THRESH", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # FPS overlay (actual loop fps)
        now = time.perf_counter()
        loop_fps = 1.0 / max(1e-9, (now - last_t))
        last_t = now

        cv2.putText(frame, f"{infer_ms:.1f} ms  |  {loop_fps:.1f} FPS", (30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Write + show
        writer.write(frame)

        # ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # --- stability results ---
    stability_dist = compute_kpt_stability_dist(kpt_positions)

    # --- per-keypoint stds (distance-to-mean) ---
    kpt_stds = compute_kpt_stds_dist(kpt_positions)
    valid_stds = [(i, s) for i, s in enumerate(kpt_stds) if np.isfinite(s)]

    if valid_stds:
        min_kpt, min_std = min(valid_stds, key=lambda t: t[1])
        max_kpt, max_std = max(valid_stds, key=lambda t: t[1])
    else:
        min_kpt = max_kpt = -1
        min_std = max_std = float("nan")

    # --- average inference time results ---
    avg_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0

    print("\nDone.")
    print("Saved video:", out_path)
    print(f"Stability frames used: {stats_frames_used}")
    print(f"Keypoint stability (dist std avg): {stability_dist:.2f} px")
    print(f"平均推理时间: {avg_infer_ms:.2f} ms/帧")
    print(f"17个关键点标准差最小: kpt[{min_kpt}] = {min_std:.2f} px")
    print(f"17个关键点标准差最大: kpt[{max_kpt}] = {max_std:.2f} px")
    print(f"{valid_stds} valid stds")


if __name__ == "__main__":
    main()
