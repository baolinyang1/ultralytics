# finally success to run this model!
# import cv2
# import numpy as np
# from openvino.runtime import Core

# MODEL_PATH = "onnx/model_int8.onnx"
# IMAGE_PATH = "test.png"
# IMG_SIZE = 640

# core = Core()
# model = core.read_model(MODEL_PATH)

# # Compile
# compiled = core.compile_model(model, "CPU")

# # Input/Output objects
# input_layer = compiled.input(0)
# output_layer = compiled.output(0)

# print("Input name:", input_layer.get_any_name())
# print("Input element type:", input_layer.element_type)
# print("Input partial shape:", input_layer.partial_shape)  # ✅ dynamic-safe

# print("\nOutput name:", output_layer.get_any_name())
# print("Output element type:", output_layer.element_type)
# print("Output partial shape:", output_layer.partial_shape)  # ✅ dynamic-safe

# # ---- Preprocess ----
# img0 = cv2.imread(IMAGE_PATH)
# assert img0 is not None, f"Image not found: {IMAGE_PATH}"

# img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# img = img.astype(np.float32) / 255.0
# img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW

# # ---- Inference ----
# out = compiled([img])[output_layer]
# print("\nInference output shape:", out.shape)
from pathlib import Path
import time

import cv2
import numpy as np
from openvino.runtime import Core

# ---------------- CONFIG ----------------
MODEL_PATH = "../yolo26n-pose-ONNX/onnx/model_int8.onnx"   
VIDEO_SOURCE = "TestVideos/still.mp4"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5
KPT_THRESH = 0.3
MAX_FRAMES_FOR_STATS = 500

# COCO-17 skeleton pairs (index-based)
SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16)   # right leg
]


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> model input float32 NCHW (1,3,640,640), RGB, /255."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
    return img


def draw_pose(frame_bgr: np.ndarray, det: np.ndarray, w: int, h: int) -> None:
    """
    det: [x1,y1,x2,y2,score, ...kpts...]
    Some exports include 1 extra value after keypoints -> handle both.
    """
    x1, y1, x2, y2, score = det[:5]
    if score < DET_THRESH:
        return

    # draw bbox (assumes normalized coords)
    x1i, y1i = int(x1 * w), int(y1 * h)
    x2i, y2i = int(x2 * w), int(y2 * h)
    cv2.rectangle(frame_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"{score:.2f}", (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    tail = det[5:]
    tail_len = tail.shape[0]

    # Try: exact 17*3 first
    if tail_len == 51:
        kpts = tail.reshape(17, 3)

    # Your case: 52 -> drop last value (extra field) then reshape
    elif tail_len == 52:
        kpts = tail[:-1].reshape(17, 3)

    # General fallback: infer K from length, allow 1 extra value
    else:
        if (tail_len - 1) % 3 == 0:
            k = (tail_len - 1) // 3
            kpts = tail[:-1].reshape(k, 3)
        elif tail_len % 3 == 0:
            k = tail_len // 3
            kpts = tail.reshape(k, 3)
        else:
            # Can't interpret keypoints -> just skip drawing keypoints
            return

    # draw keypoints + skeleton (if we have at least 17)
    k = kpts.shape[0]
    pts = []
    for i in range(k):
        kx, ky, ks = kpts[i]
        if ks >= KPT_THRESH:
            cx, cy = int(kx * w), int(ky * h)
            pts.append((cx, cy, True))
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 0, 255), -1)
        else:
            pts.append((0, 0, False))

    # only draw COCO skeleton if kpts has 17 points
    if k >= 17:
        for a, b in SKELETON:
            ax, ay, av = pts[a]
            bx, by, bv = pts[b]
            if av and bv:
                cv2.line(frame_bgr, (ax, ay), (bx, by), (255, 0, 0), 2)



def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_path = Path(VIDEO_SOURCE)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ---- Load OpenVINO model ----
    core = Core()
    model = core.read_model(str(model_path))

    # Freeze input shape to avoid dynamic-shape issues
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})

    compiled = core.compile_model(model, "CPU")
    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print("Input name:", input_layer.get_any_name())
    print("Input type:", input_layer.element_type)
    print("Output name:", output_layer.get_any_name())
    print("Output type:", output_layer.element_type)

    # ---- Video IO ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = OUT_DIR / f"{video_path.stem}_yolo26n_pose_onnx_openvino.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))

    # ---- Warmup ----
    ret, frame0 = cap.read()
    if ret:
        inp0 = preprocess(frame0)
        _ = compiled([inp0])[output_layer]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_times_ms = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        inp = preprocess(frame)
        out = compiled([inp])[output_layer]  # expected (1, 300, 57)
        preds = out[0]

        # Filter + sort by score
        preds = preds[preds[:, 4] >= DET_THRESH]
        if len(preds) > 0:
            order = np.argsort(-preds[:, 4])
            preds = preds[order]
            # draw top few people
            for det in preds[:5]:
                draw_pose(frame, det, w, h)

        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        if frame_idx < MAX_FRAMES_FOR_STATS:
            frame_times_ms.append(dt_ms)

        # optional FPS overlay
        cv2.putText(
            frame, f"{dt_ms:.1f} ms", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        writer.write(frame)
        cv2.imshow("yolo26n-pose.onnx (OpenVINO)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frame_times_ms:
        avg_ms = float(np.mean(frame_times_ms))
        std_ms = float(np.std(frame_times_ms))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    else:
        avg_ms = std_ms = fps = 0.0

    print("\nDone.")
    print("Saved video:", out_path)
    print(f"Frames timed: {len(frame_times_ms)}")
    print(f"Avg time: {avg_ms:.2f} ms | Std: {std_ms:.2f} ms | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
