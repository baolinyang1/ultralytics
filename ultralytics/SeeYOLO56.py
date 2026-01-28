from ultralytics import YOLO
import torch

# Load the model
model = YOLO("yolo26n-pose.pt")

# Run inference
results = model("TestVideos/TestImage2.png")
res = results[0]

if res.keypoints is not None and len(res.keypoints) > 0:
    # 1. Get Normalized Box (x1, y1, x2, y2) + Confidence
    # res.boxes.xyxyn returns [x1, y1, x2, y2] in 0-1 range
    box_norm = res.boxes.xyxyn[0] 
    box_conf = res.boxes.conf[0].unsqueeze(0)
    box_part = torch.cat((box_norm, box_conf)) # Total 5 values

    # 2. Get Normalized Keypoints (x, y) + Keypoint Confidence
    # res.keypoints.xyn is [N, 17, 2], res.keypoints.conf is [N, 17]
    kpts_coords = res.keypoints.xyn[0] # [17, 2]
    kpts_conf = res.keypoints.conf[0].unsqueeze(1) # [17, 1]
    
    # Combine x, y, and conf for each of the 17 points
    # This creates a [17, 3] tensor, then we flatten it to 51 values
    kpts_part = torch.cat((kpts_coords, kpts_conf), dim=1).flatten()

    # 3. Final Combine
    combined_56 = torch.cat((box_part, kpts_part))
    
    print(f"Full Normalized 56-value array:")
    print(combined_56)
    print(f"Total count: {len(combined_56)}")
