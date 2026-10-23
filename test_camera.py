import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time
from collections import deque

# FIX 1: Upgrade from 'lite' to 'full' model for much steadier skeleton tracking!
model_path = "pose_landmarker_full.task"
if not os.path.exists(model_path):
    print("Downloading FULL accuracy AI model (~9MB)... wait 5 seconds!")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7, # Ignore weak/blurry detections
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)

BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# FIX 2: Check if the camera can ACTUALLY see the joints on screen!
def joints_are_visible(landmarks, indices, threshold=0.65):
    for idx in indices:
        # Must be confident AND inside the screen borders (0.02 to 0.98)
        if landmarks[idx].visibility < threshold:
            return False
        if not (0.02 < landmarks[idx].x < 0.98 and 0.02 < landmarks[idx].y < 0.98):
            return False
    return True

# Game & Anti-Bounce Variables
reps = 0
stage = "UP"
current_exercise = "ARM_RAISES"
feedback = "Step back so joints are visible!"

angle_history = deque(maxlen=5)  # FIX 3: Smooths out jitter over last 5 frames
hold_frames = 0                  # Requires holding pose for 3 frames to trigger
REQUIRED_HOLD = 3
last_rep_time = 0                # FIX 4: Cooldown timer (max 1 rep per 0.8 seconds)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms += 33
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]

        # Draw skeleton (Green if visible, Gray if off-screen/guessed)
        for start_idx, end_idx in BODY_CONNECTIONS:
            vis = landmarks[start_idx].visibility > 0.65 and landmarks[end_idx].visibility > 0.65
            color = (0, 255, 0) if vis else (100, 100, 100)
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, color, 3)

        def get_xy(idx):
            return [landmarks[idx].x, landmarks[idx].y]

        now = time.time()

        # ---------------- EXERCISE 1: ARM RAISES ----------------
        if current_exercise == "ARM_RAISES":
            # Require Nose(0), Shoulders(11,12), and Wrists(15,16) to be clearly visible
            if not joints_are_visible(landmarks, [0, 11, 15]):
                feedback = "Make sure upper body & hands are in frame!"
            else:
                nose_y = landmarks[0].y
                shoulder_y = landmarks[11].y
                wrist_y = landmarks[15].y

                if wrist_y > shoulder_y + 0.05: # Hand clearly below shoulder
                    hold_frames = hold_frames + 1 if stage == "UP" else 0
                    if hold_frames >= REQUIRED_HOLD:
                        stage = "DOWN"
                        feedback = "Now punch HIGH above your head!"
                elif wrist_y < nose_y - 0.05:   # Hand clearly above nose
                    hold_frames = hold_frames + 1 if stage == "DOWN" else 0
                    if hold_frames >= REQUIRED_HOLD and (now - last_rep_time > 0.7):
                        stage = "UP"
                        reps += 1
                        last_rep_time = now
                        feedback = "Clean Rep!"

        # ---------------- EXERCISE 2: SQUATS ----------------
        elif current_exercise == "SQUATS":
            # Require Hip(23), Knee(25), Ankle(27) to be on screen!
            if not joints_are_visible(landmarks, [23, 25, 27], threshold=0.6):
                feedback = "Step back! Need to see Hips & Knees."
            else:
                raw_angle = calculate_angle(get_xy(23), get_xy(25), get_xy(27))
                angle_history.append(raw_angle)
                smooth_angle = np.mean(angle_history)

                if smooth_angle > 160:
                    hold_frames = hold_frames + 1 if stage == "DOWN" else 0
                    if hold_frames >= REQUIRED_HOLD and (now - last_rep_time > 1.0):
                        stage = "UP"
                        reps += 1
                        last_rep_time = now
                        feedback = "Great Squat!"
                elif smooth_angle < 105:
                    hold_frames = hold_frames + 1 if stage == "UP" else 0
                    if hold_frames >= REQUIRED_HOLD:
                        stage = "DOWN"
                        feedback = f"Good depth ({int(smooth_angle)} deg)! Stand UP!"

        # ---------------- EXERCISE 3: PUSH-UPS ----------------
        elif current_exercise == "PUSHUPS":
            # Require BOTH arms (Shoulders 11,12, Elbows 13,14, Wrists 15,16) visible
            if not joints_are_visible(landmarks, [11, 12, 13, 14, 15, 16]):
                feedback = "Show BOTH arms & hands clearly on screen!"
            else:
                # Form Check: Wrists must be BELOW shoulders (prevents head-scratching reps)
                hands_below_shoulders = (landmarks[15].y > landmarks[11].y) and (landmarks[16].y > landmarks[12].y)
                
                if not hands_below_shoulders:
                    feedback = "Get in push-up position (Hands below shoulders)!"
                else:
                    # Average both left and right elbow angles
                    left_angle = calculate_angle(get_xy(11), get_xy(13), get_xy(15))
                    right_angle = calculate_angle(get_xy(12), get_xy(14), get_xy(16))
                    angle_history.append((left_angle + right_angle) / 2.0)
                    smooth_angle = np.mean(angle_history)

                    if smooth_angle > 155:
                        hold_frames = hold_frames + 1 if stage == "DOWN" else 0
                        if hold_frames >= REQUIRED_HOLD and (now - last_rep_time > 0.9):
                            stage = "UP"
                            reps += 1
                            last_rep_time = now
                            feedback = "Solid Push-up!"
                    elif smooth_angle < 95:
                        hold_frames = hold_frames + 1 if stage == "UP" else 0
                        if hold_frames >= REQUIRED_HOLD:
                            stage = "DOWN"
                            feedback = "Chest down! Now push UP!"

    # UI Scoreboard
    cv2.rectangle(frame, (0, 0), (640, 85), (30, 30, 30), -1)
    cv2.putText(frame, f"Mode: {current_exercise} | Reps: {reps}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"[{stage}] {feedback}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    cv2.imshow("Step 2 (Fixed): Rock-Solid Rep Counter", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3')]:
        angle_history.clear()
        hold_frames = 0
        reps = 0
        if key == ord('1'):
            current_exercise, stage = "ARM_RAISES", "DOWN"
        elif key == ord('2'):
            current_exercise, stage = "SQUATS", "UP"
        elif key == ord('3'):
            current_exercise, stage = "PUSHUPS", "UP"

cap.release()
cv2.destroyAllWindows()