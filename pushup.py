import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def draw_progress_bar(frame, angle, x, y, width, height):
    """Draw a vertical progress bar for push-up progress"""
    # Map angle to progress (90° = full, 160° = empty)
    progress = np.interp(angle, (90, 160), (height, 0))
    progress = max(0, min(height, progress))  # Clamp values
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    # Draw progress fill
    if progress > 0:
        fill_height = int(height - progress)
        for i in range(fill_height):
            # Color gradient from red to green
            ratio = i / max(1, fill_height)
            color = (0, int(255 * ratio), int(255 * (1 - ratio)))
            cv2.line(frame, (x + 2, y + height - i), (x + width - 2, y + height - i), color, 1)
    
    # Draw angle text
    cv2.putText(frame, f'{int(angle)}°', (x - 20, y + height + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Make sure your camera is connected and not being used by another application")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize variables
    counter = 0
    stage = None
    start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Create window
    cv2.namedWindow('AI Push-Up Tracker', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('AI Push-Up Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("AI Push-Up Tracker Started!")
    print("Position yourself so your left side is visible to the camera")
    print("Press 'q' to quit, 'r' to reset counter, 'f' to toggle fullscreen")
    
    with mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose detection
            results = pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Create semi-transparent header
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f'FPS: {fps:.1f}', (width - 150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates for left arm
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                    
                    # Calculate elbow angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Push-up logic
                    if angle > 160:
                        stage = "up"
                    if angle < 90 and stage == 'up':
                        stage = "down"
                        counter += 1
                        print(f"Push-up #{counter} completed!")
                    
                    # Display information
                    cv2.putText(frame, f'PUSH-UPS: {counter}', (width//2 - 120, 60), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
                    
                    cv2.putText(frame, f'Elbow Angle: {int(angle)}°', (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    cv2.putText(frame, f'Stage: {stage if stage else "Ready"}', (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    # Draw progress bars
                    draw_progress_bar(frame, angle, 50, 150, 40, 300)
                    draw_progress_bar(frame, angle, width - 90, 150, 40, 300)
                    
                    # Highlight elbow joint
                    elbow_point = (int(elbow[0]), int(elbow[1]))
                    cv2.circle(frame, elbow_point, 15, (255, 0, 255), -1)
                    cv2.circle(frame, elbow_point, 20, (255, 255, 255), 3)
                    
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2, circle_radius=4),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 0, 255), thickness=2)
                    )
                    
                else:
                    # No pose detected
                    cv2.putText(frame, 'No pose detected', (width//2 - 100, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, 'Position yourself in front of camera', 
                                (width//2 - 180, height//2 + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                cv2.putText(frame, 'Processing error', (width//2 - 80, height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Instructions
            cv2.putText(frame, 'Press Q to quit | R to reset | F for fullscreen', 
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('AI Push-Up Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter = 0
                stage = None
                print("Counter reset!")
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty('AI Push-Up Tracker', cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty('AI Push-Up Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty('AI Push-Up Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession Summary:")
    print(f"Total Push-ups: {counter}")
    print(f"Great workout! 💪")

if __name__ == "__main__":
    main()
