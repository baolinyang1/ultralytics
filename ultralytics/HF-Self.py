from pathlib import Path
import time
import cv2
import numpy as np
from openvino.runtime import Core

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo26n-pose.static_int8.onnx"
IMAGE_SOURCE = "TestVideos/TestImage2.png"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5

SKELETON = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 8), (5, 6),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

# ---------------- PREPROCESS ----------------
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img

# ---------------- DECODE ----------------
# IMPORTANT: outputs are in MODEL PIXELS (0..~640), not normalized (0..1).
def decode_bbox_xyxy_modelpx(det: np.ndarray):
    x1, y1, x2, y2, score = det[:5]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return float(x1), float(y1), float(x2), float(y2), float(score)

def decode_kpts_17x3(det: np.ndarray) -> np.ndarray:
    kpt_flat = det[6:6 + 51]
    return kpt_flat.reshape(17, 3).astype(np.float32)

def map_xy_modelpx_to_frame(xy_model: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    xy_model: (N,2) in model pixel space (0..IMG_SIZE)
    map to original frame pixel space (0..w/h)
    This assumes preprocess uses direct resize (no letterbox) — which your code does.
    """
    sx = w / float(IMG_SIZE)
    sy = h / float(IMG_SIZE)
    out = xy_model.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out

# ---------------- DRAW ----------------
def draw_pose(frame: np.ndarray, det: np.ndarray, debug: bool = True):
    h, w = frame.shape[:2]

    x1m, y1m, x2m, y2m, score = decode_bbox_xyxy_modelpx(det)
    if score < DET_THRESH:
        return

    sx = w / float(IMG_SIZE)
    sy = h / float(IMG_SIZE)

    # bbox from modelpx -> frame px
    x1i, y1i = int(x1m * sx), int(y1m * sy)
    x2i, y2i = int(x2m * sx), int(y2m * sy)

    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.3f}", (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # keypoints: x,y are modelpx -> framepx
    kpts = decode_kpts_17x3(det)          # (17,3): x,y in modelpx, conf in 0..1
    xy_frame = map_xy_modelpx_to_frame(kpts[:, :2], w, h)

    pts = []
    for i in range(17):
        cx, cy = int(xy_frame[i, 0]), int(xy_frame[i, 1])
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        pts.append((cx, cy))

    for a, b in SKELETON:
        ax, ay = pts[a]
        bx, by = pts[b]
        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

# ---------------- DEBUG PRINT ----------------
def print_det_array(det: np.ndarray, name: str = "det"):
    det = det.astype(np.float32).ravel()
    print(f"\n--- {name} ---")
    print("length:", det.shape[0])
    print("min/max:", float(det.min()), float(det.max()))
    for i, v in enumerate(det):
        print(f"{i:02d}: {v:.6f}")

# ---------------- MAIN ----------------
def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    img_path = Path(IMAGE_SOURCE)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    core = Core()
    model = core.read_model(str(model_path))
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")
    output_layer = compiled.output(0)

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    t0 = time.perf_counter()
    out = compiled([preprocess(frame)])[output_layer]  # (1, N, 57)
    preds = out[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("preds shape:", preds.shape, "dtype:", preds.dtype)

    top_idx = int(np.argmax(preds[:, 4]))
    print("top_idx:", top_idx, "top_conf:", float(preds[top_idx, 4]))
    print_det_array(preds[top_idx], name="top_by_conf")

    preds_f = preds[preds[:, 4] >= DET_THRESH]
    if len(preds_f) > 0:
        preds_f = preds_f[np.argsort(-preds_f[:, 4])]
        for det in preds_f[:3]:
            draw_pose(frame, det, debug=True)
    else:
        cv2.putText(frame, "No detections above DET_THRESH", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"{dt_ms:.1f} ms", (30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out_path = OUT_DIR / f"{img_path.stem}_openvino_pose_FIXED.png"
    cv2.imwrite(str(out_path), frame)

    cv2.imshow("OpenVINO Pose (FIXED)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDone.")
    print("Saved image:", out_path)
    print(f"Inference time: {dt_ms:.2f} ms")


if __name__ == "__main__":
    main()

