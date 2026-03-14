from pathlib import Path
from collections import deque
import csv

import cv2
import numpy as np
from scipy.signal import savgol_filter
from openvino.runtime import Core

MODEL_PATH = "model_int8.onnx"
VIDEO_SOURCE = "TestVideos/Still2.mp4"
OUT_DIR = Path("onnx_video_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
DET_THRESH = 0.5

# Savitzky-Golay settings
SG_WINDOW_LENGTH = 11   # must be odd
SG_POLYORDER = 2

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
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


def decode_bbox_xyxy_norm(det: np.ndarray):
    x1, y1, x2, y2, score = det[:5]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return float(x1), float(y1), float(x2), float(y2), float(score)


def decode_kpts_17x3(det: np.ndarray) -> np.ndarray:
    return det[6:6 + 51].reshape(17, 3).astype(np.float32)


def map_kpts_to_frame_fullframe_norm(kpts: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.stack([kpts[:, 0] * w, kpts[:, 1] * h], axis=1).astype(np.float32)


def build_csv_columns():
    cols = ["frame_idx", "timestamp", "conf"]
    for i in range(17):
        cols.append(f"kpt{i}_x")
        cols.append(f"kpt{i}_y")
    return cols


def select_best_detection(preds: np.ndarray, det_thresh: float):
    if preds is None or len(preds) == 0:
        return None

    valid = preds[preds[:, 4] >= det_thresh]
    if valid.shape[0] == 0:
        return None

    best_idx = int(np.argmax(valid[:, 4]))
    return valid[best_idx]


def det_to_xy_and_row(
    det,
    frame_idx: int,
    timestamp: float,
    width: int,
    height: int,
    prev_xy: np.ndarray | None = None,
    fill_mode: str = "previous",
):
    if det is None:
        if fill_mode == "previous" and prev_xy is not None:
            xy = prev_xy.copy()
        else:
            xy = np.zeros((17, 2), dtype=np.float32)

        conf = 0.0
    else:
        conf = float(det[4])
        kpts = decode_kpts_17x3(det)
        xy = map_kpts_to_frame_fullframe_norm(kpts, width, height)

    row = [frame_idx, timestamp, conf]
    for i in range(17):
        row.extend([float(xy[i, 0]), float(xy[i, 1])])

    return xy, conf, row


def xy_to_csv_row(frame_idx: int, timestamp: float, conf: float, xy: np.ndarray):
    row = [frame_idx, timestamp, conf]
    for i in range(17):
        row.extend([float(xy[i, 0]), float(xy[i, 1])])
    return row


def draw_pose_from_xy(
    frame: np.ndarray,
    xy: np.ndarray,
    conf: float | None = None,
    label: str | None = None,
):
    pts = []
    h, w = frame.shape[:2]

    for i in range(17):
        cx = int(round(float(xy[i, 0])))
        cy = int(round(float(xy[i, 1])))

        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        pts.append((cx, cy))
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(i),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    for a, b in SKELETON:
        ax, ay = pts[a]
        bx, by = pts[b]
        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

    y_text = 35
    if conf is not None:
        cv2.putText(
            frame,
            f"conf: {conf:.2f}",
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
        )
        y_text += 35

    if label is not None:
        cv2.putText(
            frame,
            label,
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
        )


def get_valid_sg_window_length(n: int, preferred: int, polyorder: int) -> int:
    wl = min(preferred, n)
    if wl % 2 == 0:
        wl -= 1

    if wl < 3:
        return 0

    if wl <= polyorder:
        wl = polyorder + 1
        if wl % 2 == 0:
            wl += 1
        if wl > n:
            wl = n if n % 2 == 1 else n - 1

    if wl < 3 or wl <= polyorder:
        return 0

    return wl


def smooth_xy_sequence(xy_seq: list[np.ndarray], preferred_wl: int, polyorder: int) -> np.ndarray:
    """
    xy_seq: list of length N, each item is (17,2)
    returns: smoothed array of shape (N,17,2)
    """
    arr = np.stack(xy_seq, axis=0).astype(np.float32)   # (N,17,2)
    n = arr.shape[0]

    wl = get_valid_sg_window_length(n, preferred_wl, polyorder)
    if wl == 0:
        return arr.copy()

    flat = arr.reshape(n, 34)  # 17*2
    smooth_flat = savgol_filter(flat, window_length=wl, polyorder=polyorder, axis=0)
    return smooth_flat.reshape(n, 17, 2).astype(np.float32)


def write_csv_and_clean_video_simultaneously():
    core = Core()
    model = core.read_model(MODEL_PATH)
    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
    compiled = core.compile_model(model, "CPU")
    out_layer = compiled.output(0)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {VIDEO_SOURCE}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    base = Path(VIDEO_SOURCE).stem
    raw_csv_path = OUT_DIR / f"{base}_kpts_raw.csv"
    cleaned_csv_path = OUT_DIR / f"{base}_kpts_cleaned.csv"
    raw_video_path = OUT_DIR / f"{base}_raw_pose.mp4"
    cleaned_video_path = OUT_DIR / f"{base}_cleaned_pose.mp4"

    raw_writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cleaned_writer = cv2.VideoWriter(
        str(cleaned_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    columns = build_csv_columns()

    # Rolling buffers for streaming smoothing
    frame_buffer = deque()
    meta_buffer = deque()   # (frame_idx, timestamp, conf)
    xy_buffer = deque()

    prev_xy = None
    frame_idx = 0

    with open(raw_csv_path, "w", newline="", encoding="utf-8-sig") as f_raw, \
         open(cleaned_csv_path, "w", newline="", encoding="utf-8-sig") as f_clean:

        raw_csv_writer = csv.writer(f_raw)
        cleaned_csv_writer = csv.writer(f_clean)
        raw_csv_writer.writerow(columns)
        cleaned_csv_writer.writerow(columns)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_sec = frame_idx / fps
            raw_vis = frame.copy()

            preds = compiled([preprocess(frame)])[out_layer][0]
            det = select_best_detection(preds, DET_THRESH)

            xy, conf, raw_row = det_to_xy_and_row(
                det=det,
                frame_idx=frame_idx,
                timestamp=t_sec,
                width=width,
                height=height,
                prev_xy=prev_xy,
                fill_mode="previous",
            )
            prev_xy = xy.copy()

            # Write raw CSV immediately
            raw_csv_writer.writerow(raw_row)

            # Write raw pose video immediately
            draw_pose_from_xy(
                raw_vis,
                xy,
                conf=conf,
                label=f"RAW frame: {frame_idx}",
            )
            raw_writer.write(raw_vis)
            cv2.imshow("Raw Pose", raw_vis)

            # Push into rolling buffers for cleaned output
            frame_buffer.append(frame.copy())
            meta_buffer.append((frame_idx, t_sec, conf))
            xy_buffer.append(xy.copy())

            # Once we have enough frames, smooth the current window and emit the oldest frame
            if len(xy_buffer) >= SG_WINDOW_LENGTH:
                xy_seq = list(xy_buffer)
                smoothed_seq = smooth_xy_sequence(xy_seq, SG_WINDOW_LENGTH, SG_POLYORDER)

                out_frame = frame_buffer[0].copy()
                out_xy = smoothed_seq[0]
                out_frame_idx, out_t_sec, out_conf = meta_buffer[0]

                draw_pose_from_xy(
                    out_frame,
                    out_xy,
                    conf=out_conf,
                    label=f"CLEANED frame: {out_frame_idx}",
                )
                cleaned_writer.write(out_frame)
                cleaned_csv_writer.writerow(
                    xy_to_csv_row(out_frame_idx, out_t_sec, out_conf, out_xy)
                )
                cv2.imshow("Cleaned Pose", out_frame)

                frame_buffer.popleft()
                meta_buffer.popleft()
                xy_buffer.popleft()

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        # Flush remaining frames at the end using smaller valid windows
        while len(xy_buffer) > 0:
            xy_seq = list(xy_buffer)
            smoothed_seq = smooth_xy_sequence(xy_seq, SG_WINDOW_LENGTH, SG_POLYORDER)

            out_frame = frame_buffer[0].copy()
            out_xy = smoothed_seq[0]
            out_frame_idx, out_t_sec, out_conf = meta_buffer[0]

            draw_pose_from_xy(
                out_frame,
                out_xy,
                conf=out_conf,
                label=f"CLEANED frame: {out_frame_idx}",
            )
            cleaned_writer.write(out_frame)
            cleaned_csv_writer.writerow(
                xy_to_csv_row(out_frame_idx, out_t_sec, out_conf, out_xy)
            )

            frame_buffer.popleft()
            meta_buffer.popleft()
            xy_buffer.popleft()

    cap.release()
    raw_writer.release()
    cleaned_writer.release()
    cv2.destroyAllWindows()

    print("DONE")
    print("Raw CSV       :", raw_csv_path)
    print("Cleaned CSV   :", cleaned_csv_path)
    print("Raw video     :", raw_video_path)
    print("Cleaned video :", cleaned_video_path)
    print(f"SavGol window : {SG_WINDOW_LENGTH}, polyorder: {SG_POLYORDER}")


if __name__ == "__main__":
    write_csv_and_clean_video_simultaneously()





































# A good test!!
#from pathlib import Path
#import cv2
#import numpy as np
#import pandas as pd
#from scipy.signal import savgol_filter
#from openvino.runtime import Core

#MODEL_PATH = "model_int8.onnx"
#VIDEO_SOURCE = "TestVideos/Testa.mp4"
#OUT_DIR = Path("onnx_video_results")
#OUT_DIR.mkdir(parents=True, exist_ok=True)

#IMG_SIZE = 640
#DET_THRESH = 0.5

#SKELETON = [
#    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
#    (3, 5), (4, 6), (5, 7), (6, 8), (5, 6),
#    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
#    (11, 13), (12, 14), (13, 15), (14, 16)
#]


#def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
#    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#    img = img.astype(np.float32) / 255.0
#    img = np.transpose(img, (2, 0, 1))[None, ...]
#    return img


#def decode_bbox_xyxy_norm(det: np.ndarray):
#    x1, y1, x2, y2, score = det[:5]
#    x1, x2 = min(x1, x2), max(x1, x2)
#    y1, y2 = min(y1, y2), max(y1, y2)
#    return float(x1), float(y1), float(x2), float(y2), float(score)


#def decode_kpts_17x3(det: np.ndarray) -> np.ndarray:
#    return det[6:6 + 51].reshape(17, 3).astype(np.float32)


#def map_kpts_to_frame_fullframe_norm(kpts: np.ndarray, w: int, h: int) -> np.ndarray:
#    return np.stack([kpts[:, 0] * w, kpts[:, 1] * h], axis=1).astype(np.float32)


#def build_csv_columns():
#    cols = ["frame_idx", "timestamp", "conf"]
#    for i in range(17):
#        cols.append(f"kpt{i}_x")
#        cols.append(f"kpt{i}_y")
#    return cols


#def select_best_detection(preds: np.ndarray, det_thresh: float) -> np.ndarray | None:
#    if preds is None or len(preds) == 0:
#        return None

#    valid = preds[preds[:, 4] >= det_thresh]
#    if valid.shape[0] == 0:
#        return None

#    best_idx = int(np.argmax(valid[:, 4]))
#    return valid[best_idx]


#def det_to_csv_row(
#    det: np.ndarray | None,
#    frame_idx: int,
#    timestamp: float,
#    width: int,
#    height: int,
#    prev_xy: np.ndarray | None = None,
#    fill_mode: str = "previous",
#) -> tuple[list[float], np.ndarray]:
#    if det is None:
#        if fill_mode == "previous" and prev_xy is not None:
#            xy = prev_xy.copy()
#        else:
#            xy = np.zeros((17, 2), dtype=np.float32)

#        row = [frame_idx, timestamp, 0.0]
#        for i in range(17):
#            row.extend([float(xy[i, 0]), float(xy[i, 1])])
#        return row, xy

#    conf = float(det[4])
#    kpts = decode_kpts_17x3(det)
#    xy = map_kpts_to_frame_fullframe_norm(kpts, width, height)

#    row = [frame_idx, timestamp, conf]
#    for i in range(17):
#        row.extend([float(xy[i, 0]), float(xy[i, 1])])

#    return row, xy


#def draw_pose_from_xy(
#    frame: np.ndarray,
#    xy: np.ndarray,
#    bbox: tuple[int, int, int, int] | None = None,
#    draw_bbox: bool = False,
#    conf: float | None = None,
#):
#    if draw_bbox and bbox is not None:
#        x1, y1, x2, y2 = bbox
#        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#    pts = []
#    for i in range(17):
#        cx = int(round(float(xy[i, 0])))
#        cy = int(round(float(xy[i, 1])))
#        pts.append((cx, cy))
#        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
#        cv2.putText(
#            frame,
#            str(i),
#            (cx + 4, cy - 4),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.45,
#            (0, 255, 255),
#            1,
#        )

#    for a, b in SKELETON:
#        ax, ay = pts[a]
#        bx, by = pts[b]
#        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

#    if conf is not None:
#        cv2.putText(
#            frame,
#            f"conf: {conf:.2f}",
#            (20, 70),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.8,
#            (0, 255, 0),
#            2,
#        )


#def clean_kpts_csv(
#    raw_csv_path: Path,
#    cleaned_csv_path: Path,
#    window_length: int = 11,
#    polyorder: int = 2,
#):
#    df = pd.read_csv(raw_csv_path)

#    coord_cols = []
#    for i in range(17):
#        coord_cols.append(f"kpt{i}_x")
#        coord_cols.append(f"kpt{i}_y")

#    n = len(df)
#    if n < 3:
#        df.to_csv(cleaned_csv_path, index=False)
#        return

#    wl = min(window_length, n if n % 2 == 1 else n - 1)
#    if wl < 3:
#        df.to_csv(cleaned_csv_path, index=False)
#        return

#    if wl <= polyorder:
#        wl = polyorder + 1
#        if wl % 2 == 0:
#            wl += 1
#        if wl > n:
#            wl = n if n % 2 == 1 else n - 1
#        if wl < 3 or wl <= polyorder:
#            df.to_csv(cleaned_csv_path, index=False)
#            return

#    for col in coord_cols:
#        df[col] = savgol_filter(df[col].to_numpy(dtype=np.float32), wl, polyorder)

#    df.to_csv(cleaned_csv_path, index=False)


#def render_video_from_csv(
#    video_path: str,
#    csv_path: Path,
#    out_video_path: Path,
#    show_window: bool = True,
#):
#    df = pd.read_csv(csv_path)

#    cap = cv2.VideoCapture(video_path)
#    if not cap.isOpened():
#        print(f"ERROR: cannot open video for rendering: {video_path}")
#        return

#    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

#    writer = cv2.VideoWriter(
#        str(out_video_path),
#        cv2.VideoWriter_fourcc(*"mp4v"),
#        fps,
#        (width, height),
#    )

#    frame_idx = 0
#    n_rows = len(df)

#    while True:
#        ok, frame = cap.read()
#        if not ok or frame is None:
#            break
#        if frame_idx >= n_rows:
#            break

#        row = df.iloc[frame_idx]

#        xy = np.zeros((17, 2), dtype=np.float32)
#        for i in range(17):
#            xy[i, 0] = float(row[f"kpt{i}_x"])
#            xy[i, 1] = float(row[f"kpt{i}_y"])

#        draw_pose_from_xy(
#            frame=frame,
#            xy=xy,
#            bbox=None,
#            draw_bbox=False,
#            conf=float(row["conf"]),
#        )

#        cv2.putText(
#            frame,
#            f"Frame: {int(row['frame_idx'])}",
#            (20, 40),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            1.0,
#            (0, 255, 255),
#            2,
#        )

#        writer.write(frame)

#        if show_window:
#            cv2.imshow("Cleaned Pose Video", frame)
#            if cv2.waitKey(1) & 0xFF == 27:
#                break

#        frame_idx += 1

#    cap.release()
#    writer.release()
#    cv2.destroyAllWindows()


#def write_all_kpts_to_csv_and_clean_video():
#    core = Core()
#    model = core.read_model(MODEL_PATH)
#    model.reshape({model.inputs[0]: [1, 3, IMG_SIZE, IMG_SIZE]})
#    compiled = core.compile_model(model, "CPU")
#    out_layer = compiled.output(0)

#    cap = cv2.VideoCapture(VIDEO_SOURCE)
#    if not cap.isOpened():
#        print(f"ERROR: cannot open video: {VIDEO_SOURCE}")
#        return

#    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

#    raw_pose_video_path = OUT_DIR / f"{Path(VIDEO_SOURCE).stem}_raw_pose.mp4"
#    raw_csv_path = OUT_DIR / f"{Path(VIDEO_SOURCE).stem}_kpts_raw.csv"
#    cleaned_csv_path = OUT_DIR / f"{Path(VIDEO_SOURCE).stem}_kpts_cleaned.csv"
#    cleaned_video_path = OUT_DIR / f"{Path(VIDEO_SOURCE).stem}_cleaned_pose.mp4"

#    raw_writer = cv2.VideoWriter(
#        str(raw_pose_video_path),
#        cv2.VideoWriter_fourcc(*"mp4v"),
#        fps,
#        (width, height),
#    )

#    rows = []
#    frame_idx = 0
#    prev_xy = None

#    while True:
#        ok, frame = cap.read()
#        if not ok or frame is None:
#            break

#        t_sec = frame_idx / fps
#        vis_frame = frame.copy()

#        preds = compiled([preprocess(frame)])[out_layer][0]
#        det = select_best_detection(preds, DET_THRESH)

#        row, prev_xy = det_to_csv_row(
#            det=det,
#            frame_idx=frame_idx,
#            timestamp=t_sec,
#            width=width,
#            height=height,
#            prev_xy=prev_xy,
#            fill_mode="previous",
#        )
#        rows.append(row)

#        if det is not None:
#            kpts = decode_kpts_17x3(det)
#            xy = map_kpts_to_frame_fullframe_norm(kpts, width, height)
#            _, _, _, _, conf = decode_bbox_xyxy_norm(det)
#            draw_pose_from_xy(vis_frame, xy, conf=conf)

#        cv2.putText(
#            vis_frame,
#            f"Frame: {frame_idx}",
#            (20, 40),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            1.0,
#            (0, 255, 255),
#            2,
#        )

#        raw_writer.write(vis_frame)
#        cv2.imshow("Raw Pose", vis_frame)

#        frame_idx += 1
#        if cv2.waitKey(1) & 0xFF == 27:
#            break

#    cap.release()
#    raw_writer.release()
#    cv2.destroyAllWindows()

#    df = pd.DataFrame(rows, columns=build_csv_columns())
#    df.to_csv(raw_csv_path, index=False)

#    clean_kpts_csv(
#        raw_csv_path=raw_csv_path,
#        cleaned_csv_path=cleaned_csv_path,
#        window_length=11,
#        polyorder=2,
#    )

#    render_video_from_csv(
#        video_path=VIDEO_SOURCE,
#        csv_path=cleaned_csv_path,
#        out_video_path=cleaned_video_path,
#        show_window=True,
#    )

#    print("DONE")
#    print("Raw pose video :", raw_pose_video_path)
#    print("Raw CSV        :", raw_csv_path)
#    print("Cleaned CSV    :", cleaned_csv_path)
#    print("Cleaned video  :", cleaned_video_path)


#if __name__ == "__main__":
#    write_all_kpts_to_csv_and_clean_video()