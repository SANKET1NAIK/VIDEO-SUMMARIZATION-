import cv2
import numpy as np
import tensorrt as trt
import torch
import os
from collections import deque

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU and PyTorch installation.")

class CombinedVisualizer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.activities = deque(maxlen=window_size)
        self.activity_counts = {
            "walking": 0,
            "standing": 0,
            "sitting": 0,
            "pawing": 0,
            "Unknown": 0,
        }
        self.total_frames = 0

    def update(self, activity):
        self.activities.append(activity)
        self.activity_counts[activity] += 1
        self.total_frames += 1

    def create_visualization(self, frame):
        frame_height, frame_width = frame.shape[:2]
        defrag_height = 150
        defrag_width = frame_width
        defrag_image = np.ones((defrag_height, defrag_width, 3), dtype=np.uint8) * 255

        colors = {
            "walking": (0, 0, 255),  # Red
            "standing": (255, 0, 0),  # Blue
            "sitting": (0, 255, 0),  # Green
            "pawing": (255, 165, 0),  # Orange
            "Unknown": (128, 128, 128),  # Gray
        }

        segment_width = defrag_width // self.window_size
        for i, activity in enumerate(self.activities):
            x_start = i * segment_width
            x_end = x_start + segment_width
            color = colors.get(activity, (128, 128, 128))
            cv2.rectangle(defrag_image, (x_start, 0), (x_end, 80), color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        legend_items = [
            ("Walking", (0, 0, 255)),
            ("Standing", (255, 0, 0)),
            ("Sitting", (0, 255, 0)),
            ("Pawing", (255, 165, 0)),
            ("Unknown", (128, 128, 128)),
        ]

        x_offset = 10
        y_offset = 120
        for text, color in legend_items:
            cv2.rectangle(
                defrag_image,
                (x_offset, y_offset - 15),
                (x_offset + 20, y_offset + 5),
                color,
                -1,
            )
            percentage = (
                self.activity_counts.get(text.lower(), 0) / max(1, self.total_frames)
            ) * 100
            cv2.putText(
                defrag_image,
                f"{text}: {percentage:.1f}%",
                (x_offset + 30, y_offset),
                font,
                0.5,
                (0, 0, 0),
                1,
            )
            x_offset += 160

        combined_height = frame_height + defrag_height
        combined_image = np.zeros((combined_height, frame_width, 3), dtype=np.uint8)
        combined_image[:frame_height] = frame
        combined_image[frame_height:] = defrag_image

        return combined_image

class TensorRTInference:
    """TensorRT inference class for ViTPose model, optimized for GPU with TensorRT 10.8.0.43"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)

        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        
        self.input_shape = (1, 3, 256, 192)
        self.output_shape = (1, 17, 64, 48)
        
        self.input_name = "input"
        self.output_name = "output"
        
        self.allocate_buffers()

    def allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []

        num_io_tensors = self.engine.num_io_tensors
        for i in range(num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            if tensor_dtype == np.float32:
                torch_dtype = torch.float32
            elif tensor_dtype == np.float16:
                torch_dtype = torch.float16
            else:
                raise ValueError(f"Unsupported dtype: {tensor_dtype}")

            tensor = torch.zeros(tuple(tensor_shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(tensor.data_ptr())
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)

            self.context.set_tensor_address(tensor_name, tensor.data_ptr())

    def preprocess_image(self, img):
        img = cv2.resize(img, (192, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).to('cuda')

    def postprocess_output(self, output):
        heatmaps = output.reshape(self.output_shape)
        keypoints = {}
        for batch_idx in range(heatmaps.shape[0]):
            person_keypoints = torch.zeros((17, 3), device='cuda')
            for kpt_idx in range(17):
                heatmap = heatmaps[batch_idx, kpt_idx]
                flat_idx = torch.argmax(heatmap)
                y, x = torch.unravel_index(flat_idx, heatmap.shape)
                orig_x = x * (192 / 48)
                orig_y = y * (256 / 64)
                confidence = heatmap[y, x]
                person_keypoints[kpt_idx] = torch.tensor([orig_x, orig_y, confidence], device='cuda')
            keypoints[batch_idx] = person_keypoints.cpu().numpy()
        return keypoints
    
    def infer(self, img):
        preprocessed = self.preprocess_image(img)
        self.inputs[0].copy_(preprocessed)
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        output = self.outputs[0]
        keypoints = self.postprocess_output(output)
        return keypoints

class HorseGaitMonitor:
    def __init__(self, model_path, output_dir="monitoring_output"):
        self.model = TensorRTInference(model_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pose_dirs = {
            "standing": os.path.join(output_dir, "standing"),
            "walking": os.path.join(output_dir, "walking"),
            "sitting": os.path.join(output_dir, "sitting"),
            "pawing": os.path.join(output_dir, "pawing"),
        }
        for dir_path in self.pose_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self.prev_positions = [None, None]
        self.movement_buffer = [[], []]
        self.state_buffer = [[], []]
        self.visualizers = [CombinedVisualizer(), CombinedVisualizer()]
        
        self.front_paw_positions_history = [deque(maxlen=30), deque(maxlen=30)]
        self.front_leg_angles_history = [deque(maxlen=30), deque(maxlen=30)]
        self.pawing_pattern_count = [0, 0]
        self.pawing_cooldown = [0, 0]
        self.pawing_detection_threshold = 3
        self.pawing_angle_threshold = (100, 140)

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def detect_pawing_pattern(self, front_paw_positions, front_leg_angles, camera_idx):
        self.front_paw_positions_history[camera_idx].append(front_paw_positions)
        self.front_leg_angles_history[camera_idx].append(front_leg_angles)
        
        if len(self.front_paw_positions_history[camera_idx]) < 10:
            return False
            
        for leg_idx in range(2):
            y_positions = np.array([pos[leg_idx][1] for pos in self.front_paw_positions_history[camera_idx]])
            y_movements = np.diff(y_positions)
            direction_changes = np.sign(y_movements[1:]) != np.sign(y_movements[:-1])
            num_direction_changes = np.sum(direction_changes)
            
            if num_direction_changes >= 3:
                vertical_range = np.max(y_positions) - np.min(y_positions)
                angles = np.array([angles[leg_idx] for angles in self.front_leg_angles_history[camera_idx]])
                angles_in_range = np.any((angles >= self.pawing_angle_threshold[0]) & 
                                       (angles <= self.pawing_angle_threshold[1]))
                
                if vertical_range > 15 and angles_in_range:
                    return True
        return False

    def detect_state(self, keypoints, camera_idx):
        movement_detected = False

        for person_id, kp_array in keypoints.items():
            leg_points = {
                "L_F_Hip": kp_array[5][:2],
                "L_F_Knee": kp_array[6][:2],
                "L_F_Paw": kp_array[7][:2],
                "R_F_Hip": kp_array[8][:2],
                "R_F_Knee": kp_array[9][:2],
                "R_F_Paw": kp_array[10][:2],
                "L_B_Hip": kp_array[11][:2],
                "L_B_Knee": kp_array[12][:2],
                "L_B_Paw": kp_array[13][:2],
                "R_B_Hip": kp_array[14][:2],
                "R_B_Knee": kp_array[15][:2],
                "R_B_Paw": kp_array[16][:2]
            }

            angles = {
                "left_front": self.calculate_angle(
                    np.array(leg_points["L_F_Hip"]),
                    np.array(leg_points["L_F_Knee"]),
                    np.array(leg_points["L_F_Paw"])
                ),
                "right_front": self.calculate_angle(
                    np.array(leg_points["R_F_Hip"]),
                    np.array(leg_points["R_F_Knee"]),
                    np.array(leg_points["R_F_Paw"])
                ),
                "left_back": self.calculate_angle(
                    np.array(leg_points["L_B_Hip"]),
                    np.array(leg_points["L_B_Knee"]),
                    np.array(leg_points["L_B_Paw"])
                ),
                "right_back": self.calculate_angle(
                    np.array(leg_points["R_B_Hip"]),
                    np.array(leg_points["R_B_Knee"]),
                    np.array(leg_points["R_B_Paw"])
                )
            }
            
            front_paw_positions = [leg_points["L_F_Paw"], leg_points["R_F_Paw"]]
            front_leg_angles = [angles["left_front"], angles["right_front"]]
            
            is_pawing_pattern = self.detect_pawing_pattern(front_paw_positions, front_leg_angles, camera_idx)
            
            if self.pawing_cooldown[camera_idx] > 0:
                self.pawing_cooldown[camera_idx] -= 1
                return "pawing"
                
            if is_pawing_pattern:
                self.pawing_pattern_count[camera_idx] += 1
                if self.pawing_pattern_count[camera_idx] >= self.pawing_detection_threshold:
                    self.pawing_pattern_count[camera_idx] = 0
                    self.pawing_cooldown[camera_idx] = 15
                    return "pawing"
            else:
                self.pawing_pattern_count[camera_idx] = max(0, self.pawing_pattern_count[camera_idx] - 0.5)
            
            if all(angle < 90 for angle in angles.values()):
                return "sitting"

            paw_positions = np.array([
                leg_points["L_F_Paw"],
                leg_points["R_F_Paw"],
                leg_points["L_B_Paw"],
                leg_points["R_B_Paw"]
            ])

            if self.prev_positions[camera_idx] is not None:
                movements = np.linalg.norm(paw_positions - self.prev_positions[camera_idx], axis=1)
                self.movement_buffer[camera_idx].append(np.mean(movements))
                if len(self.movement_buffer[camera_idx]) > 10:
                    self.movement_buffer[camera_idx].pop(0)
                movement_detected = np.mean(self.movement_buffer[camera_idx]) > 5.0

            self.prev_positions[camera_idx] = paw_positions

        current_state = "walking" if movement_detected else "standing"
        self.state_buffer[camera_idx].append(current_state)
        if len(self.state_buffer[camera_idx]) > 15:
            self.state_buffer[camera_idx].pop(0)
        return max(set(self.state_buffer[camera_idx]), key=self.state_buffer[camera_idx].count)

    def save_frame(self, frame, state, frame_count, camera_idx):
        if state in self.pose_dirs:
            filename = os.path.join(self.pose_dirs[state], f"cam{camera_idx + 1}_frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

    def process_nvr_cameras(self, nvr_ip="192.168.1.113", username="admin", password="Jepl@1234", channels=[1, 2], port=554):
        """Process live feeds from two cameras via NVR with debugging."""
        rtsp_urls = [
            f"rtsp://{username}:{password}@{nvr_ip}:{port}/Streaming/Channels/{channels[0]}01",  # Camera 1: 192.168.1.101
            f"rtsp://{username}:{password}@{nvr_ip}:{port}/Streaming/Channels/{channels[1]}01"   # Camera 2: 192.168.1.102
        ]
        print("Attempting to connect to the following RTSP URLs via NVR:")
        for i, url in enumerate(rtsp_urls):
            print(f"Camera {i + 1} (Channel {channels[i]}): {url}")

        # Fallback: Direct camera IPs
        fallback_urls = [
            f"rtsp://{username}:{password}@192.168.1.101:{port}/stream",
            f"rtsp://{username}:{password}@192.168.1.102:{port}/stream"
        ]
        print("Fallback direct camera IPs available:")
        for i, url in enumerate(fallback_urls):
            print(f"Camera {i + 1} (Direct IP): {url}")

        caps = [cv2.VideoCapture(url) for url in rtsp_urls]
        for i, cap in enumerate(caps):
            if not cap.isOpened():
                print(f"Error: Could not open NVR stream for camera {i + 1}. Trying direct IP...")
                caps[i] = cv2.VideoCapture(fallback_urls[i])
                if not caps[i].isOpened():
                    print(f"Error: Could not open direct stream {i + 1}. Verify URL, network, and camera settings.")
                    print(f"Suggestion: Test the URL in VLC: {fallback_urls[i]}")
                    raise ValueError(f"Could not open camera stream {i + 1}")
                else:
                    print(f"Camera {i + 1} direct stream opened successfully.")
            else:
                print(f"Camera {i + 1} NVR stream opened successfully.")

        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(caps[0].get(cv2.CAP_PROP_FPS)) or 30

        combined_height = height + 150
        output_path = os.path.join(self.output_dir, "nvr_dual_camera_analysis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, combined_height))

        frame_count = 0
        cv2.namedWindow("Horse Gait Analysis - NVR Cameras", cv2.WINDOW_NORMAL)

        try:
            while True:
                frames = []
                for i, cap in enumerate(caps):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Failed to grab frame from camera {i + 1}")
                        break
                    frames.append(frame)

                if len(frames) != 2:
                    break

                states = []
                annotated_frames = []
                for i, frame in enumerate(frames):
                    keypoints = self.model.infer(frame)
                    current_state = self.detect_state(keypoints, i)
                    states.append(current_state)
                    
                    annotated_frame = frame.copy()
                    self.save_frame(annotated_frame, current_state, frame_count, i)
                    annotated_frames.append(annotated_frame)
                    self.visualizers[i].update(current_state)

                combined_frame = np.hstack((annotated_frames[0], annotated_frames[1]))
                vis1 = self.visualizers[0].create_visualization(annotated_frames[0])
                vis2 = self.visualizers[1].create_visualization(annotated_frames[1])
                combined_display = np.hstack((vis1[:, :width], vis2[:, :width]))

                out.write(combined_display)
                cv2.imshow("Horse Gait Analysis - NVR Cameras", combined_display)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            for cap in caps:
                cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete! Output saved to: {output_path}")

def main():
    engine_path = "C:/Users/hp/Downloads/vitpose-l-ap10k.engine"
    nvr_ip = "192.168.1.113"      # Your NVR IP address
    username = "admin"            # NVR and camera username
    password = "Jepl@1234"        # NVR and camera password
    channels = [1, 2]             # Channels for 192.168.1.101 and 192.168.1.102
    port = 554                    # Default RTSP port
    
    try:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        monitor = HorseGaitMonitor(engine_path)
        monitor.process_nvr_cameras(nvr_ip, username, password, channels, port)
    except Exception as e:
        print(f"Error processing NVR camera feeds: {str(e)}")
        raise

if __name__ == "__main__":
    main()
