import pandas as pd
import matplotlib.pyplot as plt

csv_path = "jump_rope_results/race4_25fps_jump_rope_openvino_draw.csv"
df = pd.read_csv(csv_path)

hip_y_index = df.columns.get_loc("hip_y")
print("hip_y column index:", hip_y_index)

plt.figure()
plt.plot(df["frame_idx"], df["hip_y"])
plt.xlabel("frame_idx")
plt.ylabel("hip_y(pxs)")
plt.title("hip_y over time")
plt.show()