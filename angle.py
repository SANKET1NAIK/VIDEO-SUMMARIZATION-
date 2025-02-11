import cv2
import sys
import argparse

def setup_video(video_path):
    """
    Initialize video playback from file or URL
    Returns video capture object if successful, None if failed
    """
    try:
        # Create video capture object
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return None
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video Properties:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
            
        return cap
        
    except Exception as e:
        print(f"Error initializing video: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Play video from file or URL')
    parser.add_argument('video_path', help='Path to video file or URL')
    args = parser.parse_args()
    
    # Initialize video
    cap = setup_video(args.video_path)
    if cap is None:
        sys.exit(1)
    
    # Create named window
    window_name = "Video Player"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("\nControls:")
    print("Press 'q' to quit")
    print("Press 'space' to pause/resume")
    
    paused = False
    try:
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()
                
                # Check if video ended
                if not ret:
                    print("\nEnd of video")
                    break
                    
                # Display the frame
                cv2.imshow(window_name, frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                break
                
            # Space to pause/resume
            elif key == ord(' '):
                paused = not paused
                status = "Paused" if paused else "Playing"
                print(f"\nVideo {status}")
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
