import cv2
import numpy as np
import os
from collections import deque
import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from easy_ViTPose.easy_ViTPose import VitInference

# Add safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([DetectionModel])

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


class HorseGaitMonitor:
    def __init__(self, model_path, yolo_path, output_dir="monitoring_output"):
        print("Initializing HorseGaitMonitor...")
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Initialize YOLO model first
            print("Loading YOLO model...")
            self.yolo_model = YOLO(yolo_path)
            print("YOLO model loaded successfully")
            
            # Initialize ViTPose model
            print("Loading ViTPose model...")
            self.model = VitInference(
                model_path,
                None,  # We'll use our pre-loaded YOLO model
                model_name="s",
                yolo_size=320,
                is_video=True,
                device=self.device,
                weights_only=False
            )
            # Replace the YOLO model in VitInference with our pre-loaded one
            self.model.yolo = self.yolo_model
            print("ViTPose model loaded successfully")
            
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            raise

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
        """Calculate angle between three points."""
        # Convert inputs to tensors if they aren't already
        if not isinstance(p1, torch.Tensor):
            p1 = torch.tensor(p1, device=self.device, dtype=torch.float32)
        if not isinstance(p2, torch.Tensor):
            p2 = torch.tensor(p2, device=self.device, dtype=torch.float32)
        if not isinstance(p3, torch.Tensor):
            p3 = torch.tensor(p3, device=self.device, dtype=torch.float32)

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        angle = torch.degrees(torch.acos(torch.clamp(cos_angle, -1.0, 1.0)))

        return angle.item()

    def detect_state(self, keypoints):
        if not keypoints:
            return "Unknown"

        movement_detected = False

        for person_id, kp_array in keypoints.items():
            # Convert keypoints to tensor
            kp_tensor = torch.tensor(kp_array, device=self.device)
            
            # Extract leg keypoints
            leg_points = {
                "L_F_Knee": kp_tensor[6][:2],
                "L_F_Paw": kp_tensor[7][:2],
                "R_F_Knee": kp_tensor[9][:2],
                "R_F_Paw": kp_tensor[10][:2],
                "L_B_Knee": kp_tensor[12][:2],
                "L_B_Paw": kp_tensor[13][:2],
                "R_B_Knee": kp_tensor[15][:2],
                "R_B_Paw": kp_tensor[16][:2],
                "L_F_Hip": kp_tensor[5][:2],
                "R_F_Hip": kp_tensor[8][:2],
                "L_B_Hip": kp_tensor[11][:2],
                "R_B_Hip": kp_tensor[14][:2],
            }

            # Calculate angles
            left_front_angle = self.calculate_angle(
                leg_points["L_F_Hip"],
                leg_points["L_F_Knee"],
                leg_points["L_F_Paw"]
            )

            right_front_angle = self.calculate_angle(
                leg_points["R_F_Hip"],
                leg_points["R_F_Knee"],
                leg_points["R_F_Paw"]
            )

            left_back_angle = self.calculate_angle(
                leg_points["L_B_Hip"],
                leg_points["L_B_Knee"],
                leg_points["L_B_Paw"]
            )

            right_back_angle = self.calculate_angle(
                leg_points["R_B_Hip"],
                leg_points["R_B_Knee"],
                leg_points["R_B_Paw"]
            )

            # State detection logic
            if 100 <= left_front_angle <= 130 or 100 <= right_front_angle <= 130:
                return "pawing"

            if all(
                angle < 90
                for angle in [
                    left_front_angle,
                    right_front_angle,
                    left_back_angle,
                    right_back_angle,
                ]
            ):
                return "sitting"

            # Movement detection
            paw_positions = torch.stack([
                leg_points["L_F_Paw"],
                leg_points["R_F_Paw"],
                leg_points["L_B_Paw"],
                leg_points["R_B_Paw"]
            ])

            if self.prev_positions is not None:
                prev_positions_tensor = torch.tensor(self.prev_positions, device=self.device)
                movements = torch.norm(paw_positions - prev_positions_tensor, dim=1)
                self.movement_buffer.append(torch.mean(movements).item())
                if len(self.movement_buffer) > 10:
                    self.movement_buffer.pop(0)

                avg_movement = np.mean(self.movement_buffer)
                movement_detected = avg_movement > 5.0

            self.prev_positions = paw_positions.cpu().numpy()

        current_state = "walking" if movement_detected else "standing"
        self.state_buffer.append(current_state)
        if len(self.state_buffer) > 15:
            self.state_buffer.pop(0)

        return max(set(self.state_buffer), key=self.state_buffer.count)

    def draw_state_annotation(self, frame, state):
        if isinstance(frame, np.ndarray):
            annotated_frame = frame.copy()
        else:
            annotated_frame = np.array(frame)

        colors = {
            "standing": (255, 0, 0),  # Blue
            "walking": (0, 0, 255),  # Red
            "sitting": (0, 255, 0),  # Green
            "pawing": (255, 165, 0),  # Orange
            "Unknown": (128, 128, 128),  # Gray
        }

        cv2.putText(
            annotated_frame,
            f"State: {state.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors.get(state, (128, 128, 128)),
            2,
        )

        return annotated_frame

    def save_frame(self, frame, state, frame_count):
        if state in self.pose_dirs:
            filename = os.path.join(self.pose_dirs[state], f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

    def process_video(self, video_path):
        print(f"Processing video: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")

        combined_height = height + 150
        output_path = os.path.join(self.output_dir, "video_with_analysis.mp4")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, combined_height))

        frame_count = 0
        current_state = None
        last_announced_state = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                try:
                    keypoints = self.model.inference(frame)
                    current_state = self.detect_state(keypoints)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    current_state = "Unknown"

                if current_state != last_announced_state:
                    print(f"Frame {frame_count}: Horse is now {current_state}")
                    last_announced_state = current_state

                annotated_frame = self.draw_state_annotation(frame, current_state)
                self.save_frame(annotated_frame, current_state, frame_count)

                self.visualizer.update(current_state)
                combined_display = self.visualizer.create_visualization(annotated_frame)

                out.write(combined_display)
                cv2.imshow("Horse Gait Analysis", combined_display)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing progress: {progress:.1f}%")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            raise

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete! Output saved to: {output_path}")


def main():
    print("Starting Horse Gait Analysis...")
    
    model_path = "vitpose-l-ap10k.onnx"
    yolo_path = "yolov8x.pt"
    
    try:
        monitor = HorseGaitMonitor(model_path, yolo_path)
        video_path = "blackwalk.mp4"
        monitor.process_video(video_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
