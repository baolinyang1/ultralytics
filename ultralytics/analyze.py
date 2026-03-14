import pandas as pd

csv_path = "/home/nyk750/ultralytics/ultralytics/onnx_video_results/still_kpts_cleaned.csv"
df = pd.read_csv(csv_path)

total_std = 0
for i in range(17):
    std_x = df[f"kpt{i}_x"].std()
    std_y = df[f"kpt{i}_y"].std()

    kpt_std = (std_x + std_y) / 2
    total_std += kpt_std

final_avg_std = total_std / 17

print("Average STD acorss 17 kpts:", final_avg_std)
