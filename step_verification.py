import cv2
import mediapipe as mp
import torch
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for inference")

# Load MiDaS_small for Speed Mode
def load_midas_model():
    model = torch.hub.load("isl-org/MiDaS", "MiDaS_small").to(device)
    model.eval()
    return model

midas_model = load_midas_model()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Expected Incision Points
expected_cuts = [(320, 240), (400, 280)]  
optimal_depth = 5  # Optimal depth in mm

# Instrument Dictionary
instrument_dict = {
    "scalpel": "Perform Incision",
    "forceps": "Grasp Tissue",
    "cannula": "Irrigate Eye",
    "phaco probe": "Phacoemulsification"
}

# Frame Preprocessing (Speed Optimized)
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (640, 360))  
    return cv2.GaussianBlur(resized_frame, (3, 3), 0)

# Eye Detection
def detect_eye_region(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch_frame = [rgb_frame]
    results = yolo_model.predict(batch_frame)

    instrument_detected = False
    detected_instrument = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = result.names[int(box.cls[0])]  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Identify instruments for slow-down trigger
            if label in instrument_dict:
                instrument_detected = True
                detected_instrument = label

    return frame, instrument_detected, detected_instrument

# Depth Estimation with Dynamic Scaling
def estimate_tool_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = img.to(device)

    with torch.no_grad():
        depth_map = midas_model(img).squeeze().cpu().numpy()

    avg_depth = np.mean(depth_map)
    color, message = ((0, 255, 0), "Depth Correct") if abs(avg_depth - optimal_depth) < 0.5 else \
                     ((0, 255, 255), "Adjust Depth") if abs(avg_depth - optimal_depth) < 1.5 else \
                     ((0, 0, 255), "Too Deep!")

    cv2.putText(frame, f"{message}: {avg_depth:.2f} mm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame

# Incision Marking
def mark_incision_points(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            if (x, y) in expected_cuts:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Incision Here", (x + 10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# Main Video Processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Ensure fast playback by increasing FPS
    frame_delay = max(1, int(1000 / (fps * 1.5)))  # Speed Boost by 1.5x

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        frame, instrument_detected, detected_instrument = detect_eye_region(frame)
        frame = estimate_tool_depth(frame)
        frame = mark_incision_points(frame)

        # FPS Display & Slow Motion Control
        current_time = time.time()
        fps_text = f"FPS: {1 / (current_time - prev_time):.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Slow Down for Critical Steps
        if instrument_detected:
            frame_delay = int(1000 / (fps * 0.7))  # Slow down only when needed
            instruction_text = f"{instrument_dict[detected_instrument]}"
            cv2.putText(frame, instruction_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            frame_delay = max(1, int(1000 / (fps * 1.5)))

        prev_time = current_time

        cv2.imshow("Fast Mode - Cataract Surgery Guidance", frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example Usage
video_path = r"E:\First Project\mark4\case_2004.mp4"
process_video(video_path)
