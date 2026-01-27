from ultralytics import YOLO
import numpy as np
import pandas as pd
import time
from pathlib import Path
import cv2

# --- 配置部分 ---
video_source = "TestVideos/still.mp4"  
models_to_test = [
    ("TestModels/yolov8n-pose.pt", "YOLOv8n"),
    ("TestModels/yolo26n-pose.pt", "YOLO26n"),
    ("TestModels/yolo26s-pose.pt", "YOLO26s"),
]

max_frames_for_stats = 300
results = []

# 创建结果目录
results_dir = Path("video_performance_results")
results_dir.mkdir(exist_ok=True)

print(f"开始视频性能测试并处理视频 (源: {video_source})...")

for model_path, model_name in models_to_test:
    print(f"\n测试并处理模型: {model_name}")
    try:
        model = YOLO(model_path)

        # 获取视频属性（宽、高、FPS），用于保存处理后的视频
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        if fps_src is None or fps_src <= 0:
            fps_src = 30  # 回退到 30 FPS
        cap.release()

        # 预热 (Warmup) - 用首帧做一次推理，避免第一次推理过慢
        cap = cv2.VideoCapture(video_source)
        ret, warmup_frame = cap.read()
        cap.release()
        if ret:
            _ = model.predict(warmup_frame, imgsz=640, verbose=False)

        # 统计：速度稳定性（耗时抖动）仍保留，但“稳定性”主指标改为关键点位置抖动
        frame_times = []  # ms, 每帧总耗时（pre+infer+post）
        det_frames = 0  # 有检测到目标的帧数
        
        # 存储每个关键点在所有帧中的位置: {track_id: {keypoint_idx: [(x1, y1), (x2, y2), ...]}}
        kpt_positions_by_track = {}  # {track_id: {kpt_idx: list of (x, y) tuples}}

        # 准备输出视频
        out_path = results_dir / f"{Path(video_source).stem}_{model_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))

        # 使用 track() 而不是 predict()，确保跟踪同一个目标
        results_gen = model.track(
            source=video_source,
            imgsz=640,
            stream=True,
            verbose=False,
            persist=True,  # 保持跟踪ID在帧之间连续
            # device="0",  # 如需强制指定 GPU，可解除注释并设置为 "0"；默认自动选择设备
        )
        for i, r in enumerate(results_gen):
            # 获取单帧总耗时（预处理 + 推理 + 后处理）
            # Ultralytics 返回的 speed 字典单位为毫秒
            if i < max_frames_for_stats:
                dt = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
                frame_times.append(dt)

            # -------- 关键点“稳定性”指标：每个关键点位置的标准差 --------
            # 方法：对每个关键点，收集它在所有帧中的位置，然后计算标准差
            # 最后对所有关键点的标准差取平均，得到单一稳定性指标
            try:
                if hasattr(r, "keypoints") and r.keypoints is not None and len(r.keypoints) > 0:
                    # 获取跟踪ID（如果有）
                    track_ids = None
                    if hasattr(r, "boxes") and r.boxes is not None:
                        if hasattr(r.boxes, "id") and r.boxes.id is not None:
                            track_ids = r.boxes.id.cpu().numpy().astype(int)
                    
                    # 只处理第一个目标（主目标）
                    if len(r.keypoints) > 0:
                        kpts_xy = r.keypoints.xy[0].cpu().numpy()  # (K, 2) 像素坐标
                        num_keypoints = kpts_xy.shape[0]
                        
                        # 获取track_id（如果有）
                        track_id = int(track_ids[0]) if track_ids is not None and len(track_ids) > 0 else 0
                        
                        # 初始化该track_id的存储结构
                        if track_id not in kpt_positions_by_track:
                            kpt_positions_by_track[track_id] = {kpt_idx: [] for kpt_idx in range(num_keypoints)}
                        
                        # 收集每个关键点的位置（只统计可见的关键点）
                        if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                            vis = r.keypoints.conf[0].cpu().numpy()  # (K,)
                            for kpt_idx in range(num_keypoints):
                                if vis[kpt_idx] > 0.5:  # 只统计可见的关键点
                                    x, y = kpts_xy[kpt_idx]
                                    kpt_positions_by_track[track_id][kpt_idx].append((x, y))
                        else:
                            # 没有可见性信息，使用所有关键点
                            for kpt_idx in range(num_keypoints):
                                x, y = kpts_xy[kpt_idx]
                                kpt_positions_by_track[track_id][kpt_idx].append((x, y))
                        
                        det_frames += 1
            except Exception as e:
                # 如果出错，跳过这一帧
                pass

            # 获取带可视化结果的图像并写入视频
            annotated = r.plot()  # numpy.ndarray, BGR
            # 确保尺寸与 VideoWriter 一致
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)

            if i % 20 == 0:
                print(f"  已处理 {i} 帧...")

        writer.release()

        if frame_times:
            # 统计分析（基于前 max_frames_for_stats 帧）
            avg_dt = np.mean(frame_times)
            std_dt = np.std(frame_times)
            fps = 1000 / avg_dt if avg_dt > 0 else 0
        else:
            avg_dt = std_dt = fps = 0.0

        # 关键点位置稳定性统计：对每个关键点计算位置标准差，然后取平均
        # 方法：每个关键点 -> 所有帧位置 -> std -> 所有关键点的std -> 平均
        kpt_stability = 0.0
        if kpt_positions_by_track:
            # 使用第一个track_id（主目标）
            main_track_id = list(kpt_positions_by_track.keys())[0]
            kpt_stds = []
            
            for kpt_idx, positions in kpt_positions_by_track[main_track_id].items():
                if len(positions) >= 2:  # 至少需要2个点才能计算std
                    positions_array = np.array(positions)  # (N, 2)
                    # 计算x和y坐标的标准差，然后取平均（或者用欧氏距离的std）
                    # 方法1：分别计算x和y的std，然后取平均
                    std_x = float(np.std(positions_array[:, 0]))
                    std_y = float(np.std(positions_array[:, 1]))
                    kpt_std = (std_x + std_y) / 2.0
                    kpt_stds.append(kpt_std)
            
            if kpt_stds:
                kpt_stability = float(np.mean(kpt_stds))  # 所有关键点标准差的平均值
        
        det_rate = (det_frames / max(1, min(max_frames_for_stats, i + 1))) * 100.0  # %

        results.append(
            {
                "模型": model_name,
                "平均推理时间 (ms/帧)": f"{avg_dt:.2f}",
                # 速度稳定性（耗时抖动）保留为参考
                #"耗时抖动(std, ms)": f"{std_dt:.2f}",
                # 关键点位置稳定性（主指标）：每个关键点位置std的平均值
                "关键点稳定性(px)": f"{kpt_stability:.2f}",
                "检测覆盖率(%)": f"{det_rate:.1f}",
                "实时FPS": f"{fps:.2f}",
                "输出视频": str(out_path),
            }
        )

        print(f"  ✓ 输出视频: {out_path}")
        print(f"  ✓ FPS({len(frame_times)}帧): {fps:.2f} | 平均耗时: {avg_dt:.2f}ms | 关键点稳定性: {kpt_stability:.2f}px")

    except Exception as e:
        print(f"  ✗ 出错: {e}")

# --- 输出结果 ---
if results:
    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("视频推理性能比较 (INT8量化前)")
    print(df.to_string(index=False))

    # 保存
    df.to_csv(results_dir / "video_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {results_dir / 'video_comparison.csv'}")
else:
    print("\n未生成有效结果，请检查模型权重路径和视频路径。")











### the version without images save
# from ultralytics import YOLO
# import numpy as np
# import pandas as pd
# import time
# from pathlib import Path
# import shutil

# # 定义要测试的三个模型
# models_to_test = [
#     ("yolov8n-pose.pt", "YOLOv8n"),
#     ("yolo26n-pose.pt", "YOLO26n"),
#     ("yolo26s-pose.pt", "YOLO26s"),
# ]

# # 数据集配置
# # 选项1: coco8-pose.yaml - 小型测试数据集， auto download 
# # 选项2: coco-pose.yaml - 完整COCO数据集 Too large to download
# # 如果使用完整数据集，请确保数据集已下载到 D:datasets/coco-pose/ 目录 20GB？？？
# data_config = "coco8-pose.yaml"  # 默认使用小型数据集，如需完整数据集请改为 "coco-pose.yaml"
# num_runs = 5  # 运行次数，用于计算稳定性（标准差）

# results = []

# # 创建结果目录
# results_dir = Path("model_comparison_results")
# results_dir.mkdir(exist_ok=True)

# print("开始模型性能比较...")
# print("=" * 80)

# for model_path, model_name in models_to_test:
#     print(f"\n正在测试模型: {model_name} ({model_path})")
#     print("-" * 80)
    
#     try:
#         # 加载模型
#         model = YOLO(model_path)
        
#         # 存储多次运行的推理时间
#         inference_times = []
#         map50_values = []
#         map50_95_values = []
        
#         # 运行多次验证以计算稳定性
#         for run in range(num_runs):
#             print(f"  运行 {run + 1}/{num_runs}...", end=" ", flush=True)
            
#             # 执行验证
#             project_name = "model_comparison" 
#             name = model_name.lower()
            
#             metrics = model.val(
#                 data=data_config, 
#                 imgsz=640, 
#                 verbose=False,
#                 project=project_name,
#                 name=name,
#                 save_json=False,  # 不保存JSON以节省空间
#             )
            
#             # 收集指标
#             inference_time = metrics.speed["inference"]  # 毫秒/帧
#             inference_times.append(inference_time)
            
#             # 获取准确度指标
#             if hasattr(metrics, 'pose'):
#                 map50 = metrics.pose.map50
#                 map50_95 = metrics.pose.map
#             else:
#                 map50 = metrics.box.map50 if hasattr(metrics, 'box') else 0.0
#                 map50_95 = metrics.box.map if hasattr(metrics, 'box') else 0.0
            
#             map50_values.append(map50)
#             map50_95_values.append(map50_95)
            
#             print(f"完成 (推理时间: {inference_time:.2f}ms)")
        
#         # 计算统计信息
#         avg_inference_time = np.mean(inference_times)
#         std_inference_time = np.std(inference_times)  # 稳定性用标准差表示，越小越稳定
        
#         avg_map50 = np.mean(map50_values)
#         avg_map50_95 = np.mean(map50_95_values)
        
#         # 计算FPS -> frames per second
#         fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
#         # 存储结果
#         results.append({
#             "模型": model_name,
#             "平均推理时间 (ms/帧)": f"{avg_inference_time:.2f}",
#             "稳定性 (标准差, ms)": f"{std_inference_time:.2f}",  # 标准差越小越稳定
#             "FPS": f"{fps:.2f}",
#             "mAP50": f"{avg_map50:.4f}",
#             "mAP50-95": f"{avg_map50_95:.4f}",
#         })
        
#         print(f"  ✓ 平均推理时间: {avg_inference_time:.2f} ms/帧")
#         print(f"  ✓ 稳定性 (标准差): {std_inference_time:.2f} ms (越小越稳定)")
#         print(f"  ✓ mAP50: {avg_map50:.4f}")
#         print(f"  ✓ mAP50-95: {avg_map50_95:.4f}")
#         print(f"  ✓ FPS: {fps:.2f}")
        
#     except Exception as e:
#         print(f"  ✗ 错误: {str(e)}")
#         results.append({
#             "模型": model_name,
#             "平均推理时间 (ms/帧)": "N/A",
#             "推理时间标准差 (ms)": "N/A",
#             "稳定性 (std)": "N/A",
#             "FPS": "N/A",
#             "mAP50": "N/A",
#             "mAP50-95": "N/A",
#         })

# # 创建比较表格
# print("\n" + "=" * 80)
# print("模型性能比较结果")
# print("=" * 80)

# df = pd.DataFrame(results)

# print(df.to_string(index=False))

# # 保存到CSV文件
# csv_filename = results_dir / "model_comparison_results.csv"
# df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
# print(f"\n结果已保存到: {csv_filename}")

# # 打印总结
# print("\n" + "=" * 80)
# print("性能总结:")
# print("=" * 80)

# # 找出最快的模型
# valid_results = [r for r in results if r["平均推理时间 (ms/帧)"] != "N/A"]
# if valid_results:
#     fastest = min(valid_results, key=lambda x: float(x["平均推理时间 (ms/帧)"]))
#     print(f"最快模型: {fastest['模型']} ({fastest['平均推理时间 (ms/帧)']} ms/帧, {fastest['FPS']} FPS)")
    
#     # 找出最准确的模型
#     most_accurate = max(valid_results, key=lambda x: float(x["mAP50-95"]))
#     print(f"最准确模型: {most_accurate['模型']} (mAP50-95: {most_accurate['mAP50-95']})")
    
#     # 找出最稳定的模型（标准差越小越稳定）
#     most_stable = min(valid_results, key=lambda x: float(x["稳定性 (标准差, ms)"]))
#     print(f"最稳定模型: {most_stable['模型']} (标准差: {most_stable['稳定性 (标准差, ms)']} ms)")

# print("\n完成!")
