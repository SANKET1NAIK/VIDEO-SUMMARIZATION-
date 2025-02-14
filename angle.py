import cv2
import numpy as np
from easy_ViTPose.easy_ViTPose import VitInference
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
            "walking": (0, 0, 255),
            "standing": (255, 0, 0),
            "sitting": (0, 255, 0),
            "pawing": (255, 165, 0),
            "Unknown": (128, 128, 128),
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
        combined_image = cv2.cuda_GpuMat(combined_height, frame_width, cv2.CV_8UC3)
        combined_image.upload(np.zeros((combined_height, frame_width, 3), dtype=np.uint8))
        
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame)
        defrag_gpu = cv2.cuda_GpuMat()
        defrag_gpu.upload(defrag_image)
        
        frame_gpu.copyTo(combined_image[0:frame_height, 0:frame_width])
        defrag_gpu.copyTo(combined_image[frame_height:combined_height, 0:frame_width])
        
        return combined_image.download()


class HorseGaitMonitor:
    def __init__(self, model_path, yolo_path, output_dir="monitoring_output"):
        self.model = VitInference(
            model_path,
            yolo_path,
            model_name="s",
            yolo_size=320,
            is_video=True,
            device="cuda",
        )

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
        
        self.stream = cv2.cuda.Stream()
        self.gpu_resize = cv2.cuda.resize
        self.gpu_cvtColor = cv2.cuda.cvtColor
        
        # Initialize CUDA image filters
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.5)
        self.bilateral_filter = cv2.cuda.createBilateralFilter(cv2.CV_8UC3, 5, 75, 75)

    def preprocess_frame(self, frame_gpu):
        # Apply CUDA-accelerated preprocessing
        filtered_frame = cv2.cuda_GpuMat(frame_gpu.size(), frame_gpu.type())
        self.gaussian_filter.apply(frame_gpu, filtered_frame, stream=self.stream)
        self.bilateral_filter.apply(filtered_frame, frame_gpu, stream=self.stream)
        return frame_gpu

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def detect_state(self, keypoints):
        current_positions = None
        movement_detected = False

        for person_id, kp_array in keypoints.items():
            leg_points = {
                "L_F_Knee": kp_array[6][:2],
                "L_F_Paw": kp_array[7][:2],
                "R_F_Knee": kp_array[9][:2],
                "R_F_Paw": kp_array[10][:2],
                "L_B_Knee": kp_array[12][:2],
                "L_B_Paw": kp_array[13][:2],
                "R_B_Knee": kp_array[15][:2],
                "R_B_Paw": kp_array[16][:2],
                "L_F_Hip": kp_array[5][:2],
                "R_F_Hip": kp_array[8][:2],
                "L_B_Hip": kp_array[11][:2],
                "R_B_Hip": kp_array[14][:2],
            }

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

    def draw_state_annotation(self, frame_gpu, state):
        frame_cpu = frame_gpu.download()
        colors = {
            "standing": (255, 0, 0),
            "walking": (0, 0, 255),
            "sitting": (0, 255, 0),
            "pawing": (255, 165, 0),
        }
        
        cv2.putText(
            frame_cpu,
            f"State: {state.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors.get(state, (128, 128, 128)),
            2,
        )
        
        result_gpu = cv2.cuda_GpuMat()
        result_gpu.upload(frame_cpu)
        return result_gpu

    def save_frame(self, frame_gpu, state, frame_count):
        if state in self.pose_dirs:
            filename = os.path.join(self.pose_dirs[state], f"frame_{frame_count}.jpg")
            frame_cpu = frame_gpu.download()
            cv2.imwrite(filename, frame_cpu)

    def process_video(self, video_path):
        # Try to use CUDA video decoder
        try:
            cap = cv2.cudacodec.createVideoReader(video_path)
            using_cuda_decoder = True
        except:
            cap = cv2.VideoCapture(video_path)
            using_cuda_decoder = False
            
        if not (using_cuda_decoder or cap.isOpened()):
            raise ValueError(f"Could not open video file: {video_path}")

        if using_cuda_decoder:
            width = cap.format().width
            height = cap.format().height
            fps = cap.format().fps
            total_frames = int(cap.format().totalFrames)
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        combined_height = height + 150
        output_path = os.path.join(self.output_dir, "video_with_analysis.mp4")
        
        # Try to use CUDA video encoder
        try:
            out = cv2.cudacodec.createVideoWriter(output_path, 
                                                (width, combined_height),
                                                fps)
            using_cuda_encoder = True
        except:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, combined_height))
            using_cuda_encoder = False

        frame_count = 0
        current_state = None
        last_announced_state = None

        cv2.namedWindow("Horse Gait Analysis", cv2.WINDOW_NORMAL)

        try:
            while True:
                if using_cuda_decoder:
                    ret, frame_gpu = cap.nextFrame()
                    if not ret:
                        break
                    frame = frame_gpu.download()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_gpu = cv2.cuda_GpuMat()
                    frame_gpu.upload(frame)

                # Preprocess frame using CUDA
                frame_gpu = self.preprocess_frame(frame_gpu)

                # VitPose inference (requires CPU tensor)
                keypoints = self.model.inference(frame_gpu.download())
                current_state = self.detect_state(keypoints)

                if current_state != last_announced_state:
                    print(f"Frame {frame_count}: Horse is now {current_state}")
                    last_announced_state = current_state

                annotated_frame_gpu = self.draw_state_annotation(frame_gpu, current_state)
                self.save_frame(annotated_frame_gpu, current_state, frame_count)

                self.visualizer.update(current_state)
                combined_display = self.visualizer.create_visualization(annotated_frame_gpu.download())

                cv2.imshow("Horse Gait Analysis", combined_display)
                
                if using_cuda_encoder:
                    combined_gpu = cv2.cuda_GpuMat()
                    combined_gpu.upload(combined_display)
                    out.write(combined_gpu)
                else:
                    out.write(combined_display)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing progress: {progress:.1f}%")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            if using_cuda_decoder:
                cap.release()
            else:
                cap.release()
            if using_cuda_encoder:
                out.release()
            else:
                out.release()
            cv2.destroyAllWindows()
            cv2.cuda.destroyAllWindows()
            print(f"\nProcessing complete! Output saved to: {output_path}")


def main():
    model_path = "vitpose-l-ap10k.onnx"
    yolo_path = "yolov8x.pt"
    monitor = HorseGaitMonitor(model_path, yolo_path)

    video_path = "blackwalk.mp4"
    try:
        monitor.process_video(video_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise


if __name__ == "__main__":
    main()
