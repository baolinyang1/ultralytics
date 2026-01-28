from pathlib import Path
import time
import cv2
import numpy as np
from openvino.runtime import Core

# ---------------- CONFIG ----------------
MODEL_PATH = "../yolo26n-pose-ONNX/onnx/model_int8.onnx"
VIDEO_SOURCE = "TestVideos/TestImage.png"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5
KPT_THRESH = 0.3

# 17 keypoints exist. Skeleton is just which points to CONNECT.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def decode_bbox_xyxy_norm(det: np.ndarray):
    x1, y1, x2, y2, score = det[:5]
    x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2, float(score)


def score_kpt_block(block: np.ndarray) -> float:
    xs = block[:, 0]
    ys = block[:, 1]
    cs = block[:, 2]

    conf_cnt = float(np.sum(cs >= KPT_THRESH))
    x_spread = float(np.std(xs))
    y_spread = float(np.std(ys))

    collapse_penalty = 0.0
    if x_spread < 0.01:
        collapse_penalty += 5.0
    if y_spread < 0.01:
        collapse_penalty += 2.0

    if not np.isfinite(xs).all() or not np.isfinite(ys).all() or not np.isfinite(cs).all():
        return -1e9

    return conf_cnt * 2.0 + (x_spread + y_spread) * 10.0 - collapse_penalty


def find_best_kpt_block(det: np.ndarray) -> tuple[np.ndarray, int, float]:
    L = det.shape[0]
    best_score = -1e18
    best_kpts = None
    best_start = -1

    for start in range(0, L - 51 + 1):
        block_flat = det[start:start + 51]
        block = block_flat.reshape(17, 3).astype(np.float32)
        sc = score_kpt_block(block)
        if sc > best_score:
            best_score = sc
            best_kpts = block
            best_start = start

    return best_kpts, best_start, float(best_score)


def map_kpts_to_frame(kpts: np.ndarray, bbox_norm: tuple[float, float, float, float], w: int, h: int) -> np.ndarray:
    x1n, y1n, x2n, y2n = bbox_norm
    x1p, y1p, x2p, y2p = x1n * w, y1n * h, x2n * w, y2n * h

    xs = kpts[:, 0]
    ys = kpts[:, 1]

    A = np.stack([xs * w, ys * h], axis=1)
    B = np.stack([x1p + xs * (x2p - x1p), y1p + ys * (y2p - y1p)], axis=1)

    def inside_score(P: np.ndarray) -> int:
        Px, Py = P[:, 0], P[:, 1]
        return int(np.sum((Px >= x1p) & (Px <= x2p) & (Py >= y1p) & (Py <= y2p)))

    if inside_score(B) > inside_score(A):
        return B
    return A


def draw_pose(frame: np.ndarray, det: np.ndarray, w: int, h: int, debug=False):
    x1n, y1n, x2n, y2n, score = decode_bbox_xyxy_norm(det)
    if score < DET_THRESH:
        return

    x1i, y1i = int(x1n * w), int(y1n * h)
    x2i, y2i = int(x2n * w), int(y2n * h)

    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(frame, f"{score:.2f}", (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    kpts, start_idx, sc = find_best_kpt_block(det)
    conf = kpts[:, 2]
    pts_xy = map_kpts_to_frame(kpts, (x1n, y1n, x2n, y2n), w, h)

    if debug:
        cv2.putText(frame, f"kpt_start={start_idx} score={sc:.1f}", (x1i, min(h - 5, y2i + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pts = []
    for i in range(17):
        cx, cy = int(pts_xy[i, 0]), int(pts_xy[i, 1])

        # draw ALL keypoints (ignore threshold)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # still keep "ok" for skeleton lines if you want
        ok = conf[i] >= KPT_THRESH
        pts.append((cx, cy, ok))

    for a, b in SKELETON:
        ax, ay, av = pts[a]
        bx, by, bv = pts[b]
        if av and bv:
            cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)


# ---------------- NEW: ALWAYS PRINT RAW OUTPUT ----------------
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

    img_path = Path(VIDEO_SOURCE)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    core = Core()
    model = core.read_model(str(model_path))
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)
    # ---- read ONE image ----
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    h, w = frame.shape[:2]

    # ---- run ONE inference ----
    t0 = time.perf_counter()
    out = compiled([preprocess(frame)])[output_layer]  # (1,300,57)
    preds = out[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("\n=== RAW MODEL OUTPUT ===")
    print("preds shape:", preds.shape, "dtype:", preds.dtype)

    # Always print a few rows and the top-by-col4 row
    print_det_array(preds[0], name="preds[0]")
    if preds.shape[0] > 1:
        print_det_array(preds[1], name="preds[1]")
    top_idx = int(np.argmax(preds[:, 4])) if preds.shape[1] > 4 else 0
    print_det_array(preds[top_idx], name="top_by_preds_col4")

    # ---- filter + draw ----
    if preds.shape[1] > 4:
        preds_f = preds[preds[:, 4] >= DET_THRESH]
        if len(preds_f) > 0:
            preds_f = preds_f[np.argsort(-preds_f[:, 4])]
            for det in preds_f[:3]:
                draw_pose(frame, det, w, h, debug=True)

    # ---- save ONE image ----
    out_path = OUT_DIR / f"{img_path.stem}_openvino_int8_pose_kpt_autofix.png"
    cv2.imwrite(str(out_path), frame)

    # ---- show until key press ----
    cv2.imshow("OpenVINO INT8 Pose (single image)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDone.")
    print("Saved image:", out_path)
    print(f"Inference time: {dt_ms:.2f} ms")


if __name__ == "__main__":
    main()
