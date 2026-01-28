from pathlib import Path
import time
import cv2
import numpy as np
from openvino.runtime import Core

MODEL_PATH = "../yolo26n-pose-ONNX/onnx/model_int8.onnx"
IMAGE_SOURCE = "TestVideos/TestImage2.png"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5

# 17 keypoints exist. Skeleton is just which points to CONNECT.
SKELETON = [
    (0, 1), (0, 2), (1, 2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(5,6),(7,9),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)
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
    x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
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
    xs = kpts[:, 0] #Selects all rows (:) and the first column (0). 
    ys = kpts[:, 1] #Selects all rows (:) and the second column (1).
    return np.stack([xs * w, ys * h], axis=1) 

def draw_pose(frame: np.ndarray, det: np.ndarray, w: int, h: int, debug: bool = True):
    x1n, y1n, x2n, y2n, score = decode_bbox_xyxy_norm(det)
    if score < DET_THRESH:
        return

    x1i, y1i = int(x1n * w), int(y1n * h)
    x2i, y2i = int(x2n * w), int(y2n * h)

    # bbox
    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.3f}", (x1i, max(0, y1i -6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # kpts
    kpts = decode_kpts_17x3(det)
    conf = kpts[:, 2]
    pts_xy = map_kpts_to_frame_fullframe_norm(kpts, w, h)

    # draw ALL keypoints + index labels (always)
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

def print_det_array(det: np.ndarray, name: str = "det"):
    det = det.astype(np.float32).ravel()
    print(f"\n--- {name} ---")
    print("length:", det.shape[0])
    print("min/max:", float(det.min()), float(det.max()))
    for i, v in enumerate(det):
        print(f"{i:02d}: {v:.6f}")


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

    # read image
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    h, w = frame.shape[:2]

    # inference
    t0 = time.perf_counter()
    out = compiled([preprocess(frame)])[output_layer]  # (1, N, 57)
    preds = out[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("preds shape:", preds.shape, "dtype:", preds.dtype)

    # choose best by column 4 (object confidence)
    top_idx = int(np.argmax(preds[:, 4]))
    print("top_idx:", top_idx, "top_conf:", float(preds[top_idx, 4]))
    print_det_array(preds[top_idx], name="top_by_conf")

    # filter + draw top detections
    preds_f = preds[preds[:, 4] >= DET_THRESH]
    if len(preds_f) > 0:
        preds_f = preds_f[np.argsort(-preds_f[:, 4])]
        for det in preds_f[:3]:
            draw_pose(frame, det, w, h, debug=True)
    else:
        cv2.putText(frame, "No detections above DET_THRESH", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"{dt_ms:.1f} ms", (30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # save
    out_path = OUT_DIR / f"{img_path.stem}_openvino_int8_pose_fixed.png"
    cv2.imwrite(str(out_path), frame)

    # show
    cv2.imshow("OpenVINO INT8 Pose (fixed)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDone.")
    print("Saved image:", out_path)
    print(f"Inference time: {dt_ms:.2f} ms")


if __name__ == "__main__":
    main()
