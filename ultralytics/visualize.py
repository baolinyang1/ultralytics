import pandas as pd
import matplotlib.pyplot as plt

csv_path = "jump_rope_results/jump_rope_multiperson_stable_1055.mp4.csv"
df = pd.read_csv(csv_path)

hip_y_index = df.columns.get_loc("shoulder_y")
print("hip_y column index:", hip_y_index)

for tid in [1,2,3]:
    df_track = df[df["track_id"] == tid]
    plt.figure(figsize=(35,6))
    plt.plot(df_track["frame_idx"], df_track["hip_y"])
    plt.xlabel("frame_idx")
    plt.ylabel("hip_y(pxs)")
    plt.savefig(f"dataimages/plot_track_{tid}.png", dpi =300)