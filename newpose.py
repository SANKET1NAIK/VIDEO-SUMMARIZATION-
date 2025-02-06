import cv2
import numpy as np
from easy_ViTPose.easy_ViTPose import VitInference
from datetime import datetime
import json
import os


class HorseGaitMonitor:
    """A system for monitoring horse walking and standing states."""

    def __init__(self, model_path, yolo_path, output_dir="monitoring_output"):
        self.model = VitInference(
            model_path,
            yolo_path,
            model_name="s",
            yolo_size=320,
            is_video=True,
            device="cpu",  # Explicitly use CPU provider
        )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.prev_positions = None
        self.movement_buffer = []
        self.state_buffer = []

    def detect_state(self, keypoints):
        """
        Detect if the horse is walking or standing.
        """
        current_positions = None
        movement_detected = False

        for person_id, kp_array in keypoints.items():
            # Extract leg keypoints
            leg_points = [
                kp_array[7],  # L_F_Paw
                kp_array[10],  # R_F_Paw
                kp_array[13],  # L_B_Paw
                kp_array[16],  # R_B_Paw
            ]

            current_positions = np.array([point[:2] for point in leg_points])

            if self.prev_positions is not None:
                movements = np.linalg.norm(
                    current_positions - self.prev_positions, axis=1
                )
                self.movement_buffer.append(np.mean(movements))
                if len(self.movement_buffer) > 10:
                    self.movement_buffer.pop(0)

                avg_movement = np.mean(self.movement_buffer)
                movement_detected = avg_movement > 5.0

        self.prev_positions = current_positions

        current_state = "walking" if movement_detected else "standing"
        self.state_buffer.append(current_state)
        if len(self.state_buffer) > 15:
            self.state_buffer.pop(0)

        return max(set(self.state_buffer), key=self.state_buffer.count)

    def draw_state_annotation(self, frame, state):
        """
        Draw state annotation on frame, ensuring proper image format.
        """
        # Ensure frame is in the correct format for OpenCV
        if isinstance(frame, np.ndarray):
            annotated_frame = frame.copy()
        else:
            annotated_frame = np.array(frame)

        # Convert to BGR if in RGB
        if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Draw text
        cv2.putText(
            annotated_frame,
            f"State: {state.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if state == "standing" else (0, 0, 255),
            2,
        )

        return annotated_frame

    def process_video(self, video_path):
        """
        Process video and detect horse states.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(self.output_dir, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        current_state = None
        last_announced_state = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for ViTPose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect keypoints
                keypoints = self.model.inference(frame_rgb)

                # Detect state
                current_state = self.detect_state(keypoints)

                # Announce state changes
                if current_state != last_announced_state:
                    print(f"Frame {frame_count}: Horse is now {current_state}")
                    last_announced_state = current_state

                # Get base visualization
                annotated_frame = self.model.draw(show_yolo=True)

                # Convert to proper format and add state annotation
                annotated_frame = np.array(annotated_frame)
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                # Add state annotation
                annotated_frame = self.draw_state_annotation(
                    annotated_frame, current_state
                )

                # Write frame
                out.write(annotated_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing progress: {progress:.1f}%")

        finally:
            cap.release()
            out.release()
            print(f"\nProcessing complete! Output saved to: {output_path}")


def main():
    """Example usage of the horse gait monitoring system."""
    # Initialize system
    model_path = "vitpose-l-ap10k.onnx"
    yolo_path = "yolov8x.pt"
    monitor = HorseGaitMonitor(model_path, yolo_path)

    # Process video
    video_path = "walking.mp4"  # Replace with your video path
    try:
        monitor.process_video(video_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise  # Re-raise the exception to see the full stack trace


if __name__ == "__main__":
    main()
