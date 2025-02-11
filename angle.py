import os
from ultralytics import YOLO
import cv2
import torch
import torch.serialization

# Add the DetectionModel to safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Set up video paths
VIDEOS_DIR = os.path.join(".", "videos")
video_path = os.path.join(VIDEOS_DIR, "tanmay.mp4")
video_path_out = os.path.join(VIDEOS_DIR, "output.avi")  # Changed to .avi format

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Failed to open video file")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer with MJPG codec (more compatible)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(
    video_path_out,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise ValueError("Failed to create video writer")

# Load YOLO models
try:
    # First attempt: Load directly with YOLO
    water_model = YOLO('runs/detect/train/weights/water.pt')
    feeder_model = YOLO('runs/detect/train2/weights/best.pt')
except Exception as e:
    print(f"First loading attempt failed: {e}")
    try:
        # Second attempt: Load with torch.load and weights_only=False
        water_model = YOLO(torch.load('runs/detect/train/weights/water.pt', 
                                    map_location='cpu', 
                                    weights_only=False))
        feeder_model = YOLO(torch.load('runs/detect/train2/weights/best.pt', 
                                     map_location='cpu', 
                                     weights_only=False))
    except Exception as e:
        print(f"Second loading attempt failed: {e}")
        raise

threshold = 0.5

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        water_results = water_model(frame)[0]
        feeder_results = feeder_model(frame)[0]
        
        # Draw water detections
        for result in water_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 
                            2)
                cv2.putText(frame,
                           f"Water {score:.2f}",
                           (int(x1), int(y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 255, 0),
                           2)
        
        # Draw feeder detections
        for result in feeder_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (255, 0, 0), 
                            2)
                cv2.putText(frame,
                           f"Feeder {score:.2f}",
                           (int(x1), int(y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (255, 0, 0),
                           2)
        
        # Write frame
        out.write(frame)
        
        # Display frame (optional)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete!")
