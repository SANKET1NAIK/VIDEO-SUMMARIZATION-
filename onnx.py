import cv2
import numpy as np
import threading
import time
from queue import Queue
from ultralytics import YOLO
import onnxruntime

# Configuration
nvr_ip = "192.168.1.124"
username = "admin"
password = "Jepl@1234"
port = 554
channels = list(range(1, 5))  # Channels 1 to 10
rtsp_urls = [
    f"rtsp://{username}:{password}@{nvr_ip}:{port}/Streaming/Channels/{ch}02"
    for ch in channels
]

# Model paths - update these
yolo_model_path = "yolov8l.pt"
vitpose_onnx_path = "vitpose-s-ap10k.onnx"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
pose_imgsz = (192, 256)
yolo_imgsz = 224
yolo_step = 15

# Load models once
yolo = YOLO(yolo_model_path, task="detect")
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)
ort_session = onnxruntime.InferenceSession(
    vitpose_onnx_path, sess_options, providers=["CUDAExecutionProvider"]
)


def preprocess_image(img, target_size):
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    img_norm = (img_norm - MEAN) / STD
    img_input = img_norm.transpose(2, 0, 1).astype(np.float32)
    return np.expand_dims(img_input, axis=0)


def postprocess_heatmaps(heatmaps, orig_w, orig_h):
    keypoints = []
    num_keypoints = heatmaps.shape[1]
    for i in range(num_keypoints):
        heatmap = heatmaps[0, i, :, :]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x = x * orig_w / heatmap.shape[1]
        y = y * orig_h / heatmap.shape[0]
        keypoints.append((int(x), int(y)))
    return keypoints


def run_inference_on_frame(frame, last_boxes, frame_idx):
    if frame_idx % yolo_step == 0:
        results = yolo(frame[..., ::-1], imgsz=yolo_imgsz)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
    else:
        boxes = last_boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        crop = frame[y1:y2, x1:x2]
        img_input = preprocess_image(crop, pose_imgsz)
        heatmaps = ort_session.run(None, {ort_session.get_inputs()[0].name: img_input})[
            0
        ]
        keypoints = postprocess_heatmaps(heatmaps, x2 - x1, y2 - y1)

        for x, y in keypoints:
            cv2.circle(frame, (x + x1, y + y1), 3, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, boxes


def camera_worker(rtsp_url, frame_queue, index):
    cap = cv2.VideoCapture(rtsp_url)
    last_boxes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {index+1}: stream ended or failed")
            break

        frame, last_boxes = run_inference_on_frame(frame, last_boxes, frame_idx)

        # Resize frame to fixed size for grid display
        frame_small = cv2.resize(frame, (320, 180))
        frame_queue.put((index, frame_small))

        frame_idx += 1

    cap.release()
    frame_queue.put((index, None))  # Signal end of stream


def display_grid(frame_queue, num_cams=10, cols=5):
    rows = (num_cams + cols - 1) // cols
    blank_image = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    frames = [None] * num_cams

    while True:
        try:
            idx, frame = frame_queue.get(timeout=1)
        except:
            continue

        if frame is None:
            # Camera closed
            frames[idx] = np.zeros((180, 320, 3), dtype=np.uint8)
        else:
            frames[idx] = frame

        # Construct grid
        for i in range(num_cams):
            r = i // cols
            c = i % cols
            y1, y2 = r * 180, (r + 1) * 180
            x1, x2 = c * 320, (c + 1) * 320
            if frames[i] is not None:
                blank_image[y1:y2, x1:x2] = frames[i]
            else:
                blank_image[y1:y2, x1:x2] = np.zeros((180, 320, 3), dtype=np.uint8)

        cv2.imshow("NVR Camera Grid - Pose Estimation", blank_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# Main execution
import queue

frame_queue = queue.Queue(maxsize=20)
threads = []

for i, url in enumerate(rtsp_urls):
    t = threading.Thread(target=camera_worker, args=(url, frame_queue, i), daemon=True)
    t.start()
    threads.append(t)

display_grid(frame_queue, num_cams=len(rtsp_urls), cols=5)
