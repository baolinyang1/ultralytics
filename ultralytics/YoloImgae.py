from ultralytics import YOLO
import numpy as np
import pandas as pd
import time
from pathlib import Path
import shutil

# 定义要测试的三个模型
models_to_test = [
    ("yolov8n-pose.pt", "YOLOv8n"),
    ("yolo26n-pose.pt", "YOLO26n"),
    ("yolo26s-pose.pt", "YOLO26s"),
]

# 数据集配置
# 选项1: coco8-pose.yaml - 小型测试数据集， auto download 
# 选项2: coco-pose.yaml - 完整COCO数据集 Too large to download
# 如果使用完整数据集，请确保数据集已下载到 D:datasets/coco-pose/ 目录 20GB？？？
data_config = "coco8-pose.yaml"  # 默认使用小型数据集，如需完整数据集请改为 "coco-pose.yaml"
num_runs = 5  # 运行次数，用于计算稳定性（标准差）

results = []

# 创建结果目录
results_dir = Path("model_comparison_results")
results_dir.mkdir(exist_ok=True)

print("开始模型性能比较...")
print("=" * 80)

for model_path, model_name in models_to_test:
    print(f"\n正在测试模型: {model_name} ({model_path})")
    print("-" * 80)
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 存储多次运行的推理时间
        inference_times = []
        map50_values = []
        map50_95_values = []
        
        # 运行多次验证以计算稳定性
        for run in range(num_runs):
            print(f"  运行 {run + 1}/{num_runs}...", end=" ", flush=True)
            
            # 执行验证
            # 最后一次运行启用plots以保存可视化结果，其他运行关闭plots以提高速度
            save_plots = (run == num_runs - 1)  # 只在最后一次运行保存图像
            project_name = "model_comparison" if save_plots else None
            name = model_name.lower() if save_plots else None
            
            metrics = model.val(
                data=data_config, 
                imgsz=640, 
                plots=save_plots, 
                verbose=False,
                project=project_name,
                name=name,
                save_json=False,  # 不保存JSON以节省空间
            )
            
            # 收集指标
            inference_time = metrics.speed["inference"]  # 毫秒/帧
            inference_times.append(inference_time)
            
            # 获取准确度指标
            if hasattr(metrics, 'pose'):
                map50 = metrics.pose.map50
                map50_95 = metrics.pose.map
            else:
                map50 = metrics.box.map50 if hasattr(metrics, 'box') else 0.0
                map50_95 = metrics.box.map if hasattr(metrics, 'box') else 0.0
            
            map50_values.append(map50)
            map50_95_values.append(map50_95)
            
            print(f"完成 (推理时间: {inference_time:.2f}ms)")
            
            # 如果这是最后一次运行且保存了图像，复制图像到结果目录
            if save_plots:
                # 获取验证保存目录 - 从validator获取实际的保存路径
                # 注意：YOLO会在project/name目录下创建runs/pose/val或类似结构
                val_save_dir = Path(f"model_comparison/{model_name.lower()}")
                
                # 如果指定路径不存在，尝试查找实际保存位置
                if not val_save_dir.exists():
                    # 尝试查找runs目录下的验证结果
                    runs_dir = Path("runs")
                    if runs_dir.exists():
                        # 查找最近的pose验证目录
                        pose_dirs = list((runs_dir / "pose").glob("val*")) if (runs_dir / "pose").exists() else []
                        if pose_dirs:
                            val_save_dir = sorted(pose_dirs, key=lambda x: x.stat().st_mtime)[-1]
                
                if val_save_dir.exists():
                    model_results_dir = results_dir / model_name.lower()
                    model_results_dir.mkdir(exist_ok=True)
                    
                    # 复制所有图像文件（.jpg 和 .png）
                    img_files = list(val_save_dir.glob("*.jpg")) + list(val_save_dir.glob("*.png"))
                    for img_file in img_files:
                        shutil.copy2(img_file, model_results_dir)
                    
                    if img_files:
                        print(f"  ✓ 结果图像已保存到: {model_results_dir} ({len(img_files)} 个文件)")
                    else:
                        print(f"  ⚠ 目录存在但未找到图像文件: {val_save_dir}")
                else:
                    print(f"  ⚠ 验证结果目录不存在: {val_save_dir}")
        
        # 计算统计信息
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)  # 稳定性用标准差表示，越小越稳定
        
        avg_map50 = np.mean(map50_values)
        avg_map50_95 = np.mean(map50_95_values)
        
        # 计算FPS -> frames per second
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 存储结果
        results.append({
            "模型": model_name,
            "平均推理时间 (ms/帧)": f"{avg_inference_time:.2f}",
            "稳定性 (标准差, ms)": f"{std_inference_time:.2f}",  # 标准差越小越稳定
            "FPS": f"{fps:.2f}",
            "mAP50": f"{avg_map50:.4f}",
            "mAP50-95": f"{avg_map50_95:.4f}",
        })
        
        print(f"  ✓ 平均推理时间: {avg_inference_time:.2f} ms/帧")
        print(f"  ✓ 稳定性 (标准差): {std_inference_time:.2f} ms (越小越稳定)")
        print(f"  ✓ mAP50: {avg_map50:.4f}")
        print(f"  ✓ mAP50-95: {avg_map50_95:.4f}")
        print(f"  ✓ FPS: {fps:.2f}")
        
    except Exception as e:
        print(f"  ✗ 错误: {str(e)}")
        results.append({
            "模型": model_name,
            "平均推理时间 (ms/帧)": "N/A",
            "推理时间标准差 (ms)": "N/A",
            "稳定性 (std)": "N/A",
            "FPS": "N/A",
            "mAP50": "N/A",
            "mAP50-95": "N/A",
        })

# 创建比较表格
print("\n" + "=" * 80)
print("模型性能比较结果")
print("=" * 80)

df = pd.DataFrame(results)

print(df.to_string(index=False))

# 保存到CSV文件
csv_filename = results_dir / "model_comparison_results.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {csv_filename}")
print(f"所有结果图像保存在: {results_dir}/")

# 打印总结
print("\n" + "=" * 80)
print("性能总结:")
print("=" * 80)

# 找出最快的模型
valid_results = [r for r in results if r["平均推理时间 (ms/帧)"] != "N/A"]
if valid_results:
    fastest = min(valid_results, key=lambda x: float(x["平均推理时间 (ms/帧)"]))
    print(f"最快模型: {fastest['模型']} ({fastest['平均推理时间 (ms/帧)']} ms/帧, {fastest['FPS']} FPS)")
    
    # 找出最准确的模型
    most_accurate = max(valid_results, key=lambda x: float(x["mAP50-95"]))
    print(f"最准确模型: {most_accurate['模型']} (mAP50-95: {most_accurate['mAP50-95']})")
    
    # 找出最稳定的模型（标准差越小越稳定）
    most_stable = min(valid_results, key=lambda x: float(x["稳定性 (标准差, ms)"]))
    print(f"最稳定模型: {most_stable['模型']} (标准差: {most_stable['稳定性 (标准差, ms)']} ms)")

print("\n完成!")
