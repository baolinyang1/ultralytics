import cv2

# Settings
input_path = "TestVideos/1055.mp4"
output_path = "TestVideos/1055_25fps.mp4"
target_fps = 25.0

# Open input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec

# Initialize VideoWriter with the NEW target FPS
out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

print(f"Converting {input_path} to {target_fps} FPS...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Saved as {output_path}")
