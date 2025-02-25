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
        """Initialize TensorRT engine and allocate buffers on GPU."""
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        # Load engine from file
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)

        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        
        # Define input and output shapes
        self.input_shape = (1, 3, 256, 192)  # ViTPose standard input
        self.output_shape = (1, 17, 64, 48)  # ViTPose standard output (17 keypoints)
        
        # Tensor names (adjust based on your engine)
        self.input_name = "input"
        self.output_name = "output"
        
        # Allocate GPU buffers
        self.allocate_buffers()

    def allocate_buffers(self):
        """Allocate CUDA memory for inputs and outputs."""
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

            # Explicitly allocate on GPU (CUDA)
            tensor = torch.zeros(tuple(tensor_shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(tensor.data_ptr())
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)

            self.context.set_tensor_address(tensor_name, tensor.data_ptr())

    def preprocess_image(self, img):
        """Preprocess image for model input on GPU."""
        # Convert to RGB and resize on CPU first (cv2 doesn't support GPU natively)
        img = cv2.resize(img, (192, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Move to GPU
        return torch.from_numpy(img).to('cuda')

    def postprocess_output(self, output):
        """Convert model output to keypoints on GPU, then move to CPU for further processing."""
        heatmaps = output.reshape(self.output_shape)
        keypoints = {}
        for batch_idx in range(heatmaps.shape[0]):
            person_keypoints = torch.zeros((17, 3), device='cuda')  # Process on GPU
            for kpt_idx in range(17):
                heatmap = heatmaps[batch_idx, kpt_idx]
                flat_idx = torch.argmax(heatmap)
                y, x = torch.unravel_index(flat_idx, heatmap.shape)
                orig_x = x * (192 / 48)  # Scale to input width
                orig_y = y * (256 / 64)  # Scale to input height
                confidence = heatmap[y, x]
                person_keypoints[kpt_idx] = torch.tensor([orig_x, orig_y, confidence], device='cuda')
            keypoints[batch_idx] = person_keypoints.cpu().numpy()  # Move to CPU for compatibility
        return keypoints
    def infer(self, img):
        """Run inference on an image using GPU."""
        preprocessed = self.preprocess_image(img)
        self.inputs[0].copy_(preprocessed)
        
        # Execute inference asynchronously on GPU
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()  # Wait for GPU computation to finish
        
        output = self.outputs[0]  # Already on GPU
        keypoints = self.postprocess_output(output)
        return keypoints

class HorseGaitMonitor:
    def __init__(self, model_path, output_dir="monitoring_output"):
        """Initialize the horse gait monitoring system with TensorRT on GPU."""
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

        self.prev_positions = None
        self.movement_buffer = []
        self.state_buffer = []
        self.visualizer = CombinedVisualizer()

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (CPU-based for simplicity)."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def detect_state(self, keypoints):
        """Detect the horse's current state based on keypoints."""
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

            if any(100 <= angle <= 130 for angle in [angles["left_front"], angles["right_front"]]):
                return "pawing"
            if all(angle < 90 for angle in angles.values()):
                return "sitting"

            paw_positions = np.array([
                leg_points["L_F_Paw"],
                leg_points["R_F_Paw"],
                leg_points["L_B_Paw"],
                leg_points["R_B_Paw"]
            ])

            if self.prev_positions is not None:
                movements = np.linalg.norm(paw_positions - self.prev_positions, axis=1)
                self.movement_buffer.append(np.mean(movements))
                if len(self.movement_buffer) > 10:
                    self.movement_buffer.pop(0)
                movement_detected = np.mean(self.movement_buffer) > 5.0

            self.prev_positions = paw_positions

        current_state = "walking" if movement_detected else "standing"
        self.state_buffer.append(current_state)
        if len(self.state_buffer) > 15:
            self.state_buffer.pop(0)
        return max(set(self.state_buffer), key=self.state_buffer.count)

    def draw_state_annotation(self, frame, state):
        """Draw state annotation on frame."""
        annotated_frame = frame.copy()
        colors = {
            "standing": (255, 0, 0),  # Blue
            "walking": (0, 0, 255),   # Red
            "sitting": (0, 255, 0),   # Green
            "pawing": (255, 165, 0)   # Orange
        }
        cv2.putText(
            annotated_frame,
            f"State: {state.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors.get(state, (128, 128, 128)),
            2
        )
        return annotated_frame

    def save_frame(self, frame, state, frame_count):
        """Save frame to appropriate directory based on state."""
        if state in self.pose_dirs:
            filename = os.path.join(self.pose_dirs[state], f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

    def process_video(self, video_path):
        """Process video file and analyze horse gait on GPU."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        combined_height = height + 150  # Add space for visualizer
        output_path = os.path.join(self.output_dir, "video_with_analysis2.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, combined_height))

        frame_count = 0
        last_announced_state = None

        cv2.namedWindow("Horse Gait Analysis", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints = self.model.infer(frame)
                current_state = self.detect_state(keypoints)
                

                if current_state != last_announced_state:
                    print(f"Frame {frame_count}: Horse is now {current_state}")
                    last_announced_state = current_state

                annotated_frame = self.draw_state_annotation(frame, current_state)
                self.save_frame(annotated_frame, current_state, frame_count)

                self.visualizer.update(current_state)
                combined_display = self.visualizer.create_visualization(annotated_frame)

                cv2.imshow("Horse Gait Analysis", combined_display)
                out.write(combined_display)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing progress: {progress:.1f}%")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete! Output saved to: {output_path}")

def main():
    """Main function to run the horse gait analysis on GPU."""
    engine_path = "C:/Users/hp/Downloads/vitpose-l-ap10k.engine"
    try:
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        monitor = HorseGaitMonitor(engine_path)
        monitor.process_video("E:/vitpose/strach.mp4")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

if __name__ == "__main__":
    main()
