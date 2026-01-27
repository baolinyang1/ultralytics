from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODELS_TO_TEST = [
    "TestModels/yolov8n-pose.pt",
    "TestModels/yolo26n-pose.pt",
    "TestModels/yolo26s-pose.pt",
]

# 标定 / 验证用数据集
CALIB_DATA = "coco8-pose.yaml" 

VIDEO_SOURCE = "TestVideos/still.mp4"
MAX_FRAMES_FOR_STATS = 500


def export_int8_openvino(model_path: str, data_yaml: str) -> Path:
    """
    将 .pt 模型导出为 INT8 OpenVINO 模型，返回导出的模型路径。
    """
    print(f"\n==> 导出 INT8 OpenVINO 模型: {model_path}")
    base_model = YOLO(model_path)
    export_path = base_model.export(
        format="openvino",
        int8=True,       
        data=data_yaml,     
        imgsz=640,
        #fraction=0.13,
        nms=True,
    )
    export_path = Path(export_path)
    print(f"    导出完成: {export_path}")
    return export_path


def benchmark_video_int8(int8_model_path: Path, model_name: str, video_source: str, out_dir: Path):
    """
    使用 INT8 模型对视频进行推理，统计速度并保存可视化结果视频。
    同时计算关键点位置稳定性（每个关键点全帧位置的标准差,再对关键点std取平均)。
    返回 (avg_dt_ms, std_dt_ms, fps, kpt_stability_px, det_rate, out_video_path)
    """
    print(f"\n==> 使用 INT8 模型处理视频: {model_name}")
    model = YOLO(int8_model_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"    视频属性: {width}x{height}, FPS={fps_src:.2f}")

    # 预热一帧
    ret, warmup_frame = cap.read()
    if ret:
        _ = model.predict(warmup_frame, imgsz=640, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头

    # 输出视频
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{Path(video_source).stem}_{model_name}_INT8.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))

    frame_times = []  # 毫秒
    det_frames = 0
    # 存储关键点位置：{track_id: {kpt_idx: [(x, y), ...]}}
    kpt_positions_by_track = {}

    # 使用 track() 保持 ID 连续
    for r in model.track(source=video_source, imgsz=640, stream=True, verbose=False, persist=True):
        # 统计前 MAX_FRAMES_FOR_STATS 帧的总耗时（预处理 + 推理 + 后处理）
        if det_frames < MAX_FRAMES_FOR_STATS:
            dt = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
            
            frame_times.append(dt)

        # 关键点位置收集：取首个检测框，对每个关键点存位置
        try:
            if hasattr(r, "keypoints") and r.keypoints is not None and len(r.keypoints) > 0:
                track_ids = None
                if hasattr(r, "boxes") and r.boxes is not None and hasattr(r.boxes, "id") and r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)

                kpts_xy = r.keypoints.xy[0].cpu().numpy()  # (K,2)
                num_keypoints = kpts_xy.shape[0]
                track_id = int(track_ids[0]) if track_ids is not None and len(track_ids) > 0 else 0

                if track_id not in kpt_positions_by_track:
                    kpt_positions_by_track[track_id] = {k: [] for k in range(num_keypoints)}

                if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                    vis = r.keypoints.conf[0].cpu().numpy()  # (K,)
                    for kpt_idx in range(num_keypoints):
                        # if vis[kpt_idx] > 0.5:
                        x, y = kpts_xy[kpt_idx]
                        kpt_positions_by_track[track_id][kpt_idx].append((x, y))
                else:
                    for kpt_idx in range(num_keypoints):
                        x, y = kpts_xy[kpt_idx]
                        kpt_positions_by_track[track_id][kpt_idx].append((x, y))

                det_frames += 1
        except Exception:
            pass

        # 写入可视化帧
        im = r.plot()
        if im.shape[1] != width or im.shape[0] != height:
            im = cv2.resize(im, (width, height))
        writer.write(im)

    cap.release()
    writer.release()

    # 速度统计
    if frame_times:
        avg_dt = float(np.mean(frame_times))
        std_dt = float(np.std(frame_times))
        fps = 1000.0 / avg_dt if avg_dt > 0 else 0.0
    else:
        avg_dt = std_dt = fps = 0.0

    # 关键点稳定性：每个关键点位置 std 的平均值
    kpt_stability = 0.0
    if kpt_positions_by_track:
        main_track_id = list(kpt_positions_by_track.keys())[0]
        kpt_stds = []
        for kpt_idx, positions in kpt_positions_by_track[main_track_id].items():
            if len(positions) >= 2:
                arr = np.array(positions)  # (N,2)
                std_x = float(np.std(arr[:, 0]))
                std_y = float(np.std(arr[:, 1]))
                kpt_stds.append((std_x + std_y) / 2.0)
        if kpt_stds:
            kpt_stability = float(np.mean(kpt_stds))

    det_rate = (det_frames / max(1, MAX_FRAMES_FOR_STATS)) * 100.0

    print(f"    视频完成: {out_path}")
    print(f"    统计帧数: {len(frame_times)} | 平均耗时: {avg_dt:.2f} ms | FPS: {fps:.2f} | 关键点稳定性: {kpt_stability:.2f}px")

    return avg_dt, std_dt, fps, kpt_stability, det_rate, out_path


def main():
    video_path = VIDEO_SOURCE
    if not Path(video_path).exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    out_dir = Path("int8_video_results")
    out_dir.mkdir(exist_ok=True)

    all_results = []

    for model_path in MODELS_TO_TEST:
        model_name = Path(model_path).stem

        # export, then run on the video!
        int8_path = export_int8_openvino(model_path, CALIB_DATA)

       
        avg_dt, std_dt, fps, kpt_stability, det_rate, out_video = benchmark_video_int8(
            int8_model_path=int8_path,
            model_name=model_name,
            video_source=video_path,
            out_dir=out_dir,
        )

        all_results.append(
            {
                "模型": model_name + " (INT8)",
                "平均推理时间 (ms/帧)": f"{avg_dt:.2f}",
                "耗时抖动(std, ms)": f"{std_dt:.2f}",
                "实际FPS": f"{fps:.2f}",
                "关键点稳定性(px)": f"{kpt_stability:.2f}",
                "输出视频": str(out_video),
            }
        )

    # summarize results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = out_dir / "int8_video_comparison.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print("INT8 量化后：视频对比")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\n结果已保存到: {csv_path}")
    else:
        print("未得到任何结果，请检查模型路径、权重文件和数据配置。")


if __name__ == "__main__":
    main()

