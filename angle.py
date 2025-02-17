import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
from collections import deque

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
    def __init__(self, engine_path, yolo_path, output_dir="monitoring_output"):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create directories for saving frames
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

    def load_engine(self,engine_path):
        """Loads a TensorRT engine from the given path with debug info."""
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"❌ Engine file not found: {engine_path}")
    
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
    
        print(f"✅ Found engine file: {engine_path}")
    
        with open(engine_path, "rb") as f:
            engine_data = f.read()
    
        if not engine_data:
            raise ValueError("❌ Engine file is empty or corrupted.")
    
        engine = runtime.deserialize_cuda_engine(engine_data)
    
        if engine is None:
            raise ValueError("❌ Failed to deserialize TensorRT engine. Possible causes:\n"
                             "   - Engine file is incompatible with this TensorRT version\n"
                             "   - Engine file is corrupted\n"
                             "   - Engine file was built on a different GPU architecture")
    
        print("✅ Successfully loaded TensorRT engine!")
        return engine

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def detect_state(self, keypoints):
        current_positions = None
        movement_detected = False

        for person_id, kp_array in keypoints.items():
            # Extract leg keypoints
            leg_points = {
                "L_F_Knee": kp_array[6][:2],  # Left Front Knee
                "L_F_Paw": kp_array[7][:2],  # Left Front Paw
                "R_F_Knee": kp_array[9][:2],  # Right Front Knee
                "R_F_Paw": kp_array[10][:2],  # Right Front Paw
                "L_B_Knee": kp_array[12][:2],  # Left Back Knee
                "L_B_Paw": kp_array[13][:2],  # Left Back Paw
                "R_B_Knee": kp_array[15][:2],  # Right Back Knee
                "R_B_Paw": kp_array[16][:2],  # Right Back Paw
                "L_F_Hip": kp_array[5][:2],  # Left Front Hip
                "R_F_Hip": kp_array[8][:2],  # Right Front Hip
                "L_B_Hip": kp_array[11][:2],  # Left Back Hip
                "R_B_Hip": kp_array[14][:2],  # Right Back Hip
            }

            # Calculate leg angles
            left_front_angle = self.calculate_angle(
                np.array(leg_points["L_F_Hip"]),
                np.array(leg_points["L_F_Knee"]),
                np.array(leg_points["L_F_Paw"]),
            )

            right_front_angle = self.calculate_angle(
                np.array(leg_points["R_F_Hip"]),
                np.array(leg_points["R_F_Knee"]),
                np.array(leg_points["R_F_Paw"]),
            )

            left_back_angle = self.calculate_angle(
                np.array(leg_points["L_B_Hip"]),
                np.array(leg_points["L_B_Knee"]),
                np.array(leg_points["L_B_Paw"]),
            )

            right_back_angle = self.calculate_angle(
                np.array(leg_points["R_B_Hip"]),
                np.array(leg_points["R_B_Knee"]),
                np.array(leg_points["R_B_Paw"]),
            )

            # Check for pawing
            if 100 <= left_front_angle <= 130 or 100 <= right_front_angle <= 130:
                return "pawing"

            # Check for sitting
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

            # Check for walking/standing
            paw_positions = np.array(
                [
                    leg_points["L_F_Paw"],
                    leg_points["R_F_Paw"],
                    leg_points["L_B_Paw"],
                    leg_points["R_B_Paw"],
                ]
            )

            if self.prev_positions is not None:
                movements = np.linalg.norm(paw_positions - self.prev_positions, axis=1)
                self.movement_buffer.append(np.mean(movements))
                if len(self.movement_buffer) > 10:
                    self.movement_buffer.pop(0)

                avg_movement = np.mean(self.movement_buffer)
                movement_detected = avg_movement > 5.0

            self.prev_positions = paw_positions

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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        combined_height = height + 150
        output_path = os.path.join(self.output_dir, "video_with_analysis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, combined_height))

        frame_count = 0
        current_state = None
        last_announced_state = None

        cv2.namedWindow("Horse Gait Analysis", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints = self.run_inference(frame)
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

    def run_inference(self, frame):
        
    # Your inference logic here (ensure it is implemented)
        keypoints = {}  # Replace with actual model inference output
        print("Inference output:", keypoints)  # Debugging line
        return keypoints



def main():
    engine_path = "E:/vitpose.engine"  # TensorRT engine model
    yolo_path = "E:/yolov8n.pt"
    monitor = HorseGaitMonitor(engine_path, yolo_path)

    video_path = "E:/vitpose/walk.mp4"
    try:
        monitor.process_video(video_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise


if __name__ == "__main__":
    main()
