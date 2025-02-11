import os
from ultralytics import YOLO
import cv2
import torch

VIDEOS_DIR = os.path.join(".", "videos")
video_path = os.path.join(VIDEOS_DIR, "tanmay.mp4")
video_path_out = f"{video_path}_out.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Use hardware-accelerated encoding for Jetson
gst_out = f"appsrc ! video/x-raw, format=BGR ! videoconvert ! video/x-raw, format=I420 ! nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location={video_path_out}"
out = cv2.VideoWriter(
    gst_out,
    cv2.CAP_GSTREAMER,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (W, H)
)

if not out.isOpened():
    # Fallback to software encoding if GStreamer fails
    out = cv2.VideoWriter(
        video_path_out,
        cv2.VideoWriter_fourcc(*'avc1'),  # Try H.264 codec
        int(cap.get(cv2.CAP_PROP_FPS)),
        (W, H)
    )

# Load models with proper error handling
water_model_path = os.path.join(".", "runs", "detect", "train", "weights", "water.pt")
feeder_model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")

try:
    # Load YOLO models with specific device placement
    water_model = YOLO(water_model_path)
    water_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    feeder_model = YOLO(feeder_model_path)
    feeder_model.to('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    # Try alternative loading method
    water_model = torch.load(water_model_path, map_location='cpu', weights_only=True)
    feeder_model = torch.load(feeder_model_path, map_location='cpu', weights_only=True)

threshold = 0.5

try:
    while ret:
        # Run inference with error handling
        try:
            water_results = water_model(frame)[0]
            feeder_results = feeder_model(frame)[0]
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Handle CUDA out of memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Process a resized frame
                small_frame = cv2.resize(frame, (W//2, H//2))
                water_results = water_model(small_frame)[0]
                feeder_results = feeder_model(small_frame)[0]
            else:
                raise e

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

except Exception as e:
    print(f"Error during processing: {str(e)}")
finally:
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
