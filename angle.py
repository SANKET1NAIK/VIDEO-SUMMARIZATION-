import os
from ultralytics import YOLO
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path

# Add DetectionModel to safe globals before loading models
torch.serialization.add_safe_globals([DetectionModel])

VIDEOS_DIR = os.path.join(".", "videos")

video_path = os.path.join(VIDEOS_DIR, "walk.mp4")
video_path_out = f"{video_path}_out.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(
    video_path_out,
    cv2.VideoWriter_fourcc(*"avc1"),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (W, H),
)

# Load both models
water_model_path = os.path.join(".","water.pt")
feeder_model_path = os.path.join(".","feeder.pt")

# Alternative loading method
water_model = YOLO(water_model_path, task='detect')  # Explicitly specify task
feeder_model = YOLO(feeder_model_path, task='detect')  # Explicitly specify task

threshold = 0.5

while ret:
    # Run both models on the same frame
    water_results = water_model(frame)[0]
    feeder_results = feeder_model(frame)[0]

    # Process water detection results
    for result in water_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(
                frame,
                water_results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    # Process feeder detection results
    for result in feeder_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(
                frame,
                feeder_results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
