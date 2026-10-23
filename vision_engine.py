import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time
import threading
from collections import deque

class VisionEngine:
    def __init__(self):
        # 1. SHARED VARIABLES (Your teammates read these in Pygame!)
        self.reps = 0
        self.stage = "DOWN"
        self.current_exercise = "ARM_RAISES"  # "ARM_RAISES", "SQUATS", "PUSHUPS"
        self.feedback = "Step back so joints are visible!"
        self.latest_frame = None              # RGB frame ready for Pygame
        self.running = False

        # 2. ANTI-BOUNCE VARIABLES
        self.angle_history = deque(maxlen=5)
        self.hold_frames = 0
        self.REQUIRED_HOLD = 3
        self.last_rep_time = 0

        # 3. DOWNLOAD & LOAD FULL MEDIAPIPE MODEL (Python 3.13)
        self.model_path = "pose_landmarker_full.task"
        if not os.path.exists(self.model_path):
            print("Downloading FULL AI Pose Model (~9MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            urllib.request.urlretrieve(url, self.model_path)

        self.BODY_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

    def start(self):
        """Starts the webcam in a background thread so Pygame runs at 60 FPS."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Safely turns off the webcam when exiting Training Mode."""
        self.running = False

    def set_exercise(self, exercise_name):
        """Teammates call this to switch between 'ARM_RAISES', 'SQUATS', and 'PUSHUPS'."""
        self.current_exercise = exercise_name
        self.reps = 0
        self.angle_history.clear()
        self.hold_frames = 0
        self.stage = "DOWN" if exercise_name == "ARM_RAISES" else "UP"
        self.feedback = f"Switched to {exercise_name}!"

    def add_cheat_rep(self):
        """Emergency Judge Demo Button: Instantly adds 1 rep if demo space is tight!"""
        self.reps += 1
        self.feedback = "REP COMPLETED! (Demo Override)"

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360.0 - angle if angle > 180.0 else angle

    def joints_are_visible(self, landmarks, indices, threshold=0.65):
        for idx in indices:
            if landmarks[idx].visibility < threshold:
                return False
            if not (0.02 < landmarks[idx].x < 0.98 and 0.02 < landmarks[idx].y < 0.98):
                return False
        return True

    def _update_loop(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        timestamp_ms = 0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms += 33
            results = detector.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]

                # Draw skeleton onto RGB frame
                for start_idx, end_idx in self.BODY_CONNECTIONS:
                    vis = landmarks[start_idx].visibility > 0.65 and landmarks[end_idx].visibility > 0.65
                    color = (0, 255, 0) if vis else (120, 120, 120)
                    pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(rgb_frame, pt1, pt2, color, 3)

                def get_xy(idx):
                    return [landmarks[idx].x, landmarks[idx].y]

                now = time.time()

                # --- TIER 1: ARM RAISES ---
                if self.current_exercise == "ARM_RAISES":
                    if not self.joints_are_visible(landmarks, [0, 11, 15]):
                        self.feedback = "Show upper body & hands in frame!"
                    else:
                        if landmarks[15].y > landmarks[11].y + 0.05:
                            self.hold_frames = self.hold_frames + 1 if self.stage == "UP" else 0
                            if self.hold_frames >= self.REQUIRED_HOLD:
                                self.stage = "DOWN"
                                self.feedback = "Punch HIGH above your nose!"
                        elif landmarks[15].y < landmarks[0].y - 0.05:
                            self.hold_frames = self.hold_frames + 1 if self.stage == "DOWN" else 0
                            if self.hold_frames >= self.REQUIRED_HOLD and (now - self.last_rep_time > 0.7):
                                self.stage = "UP"
                                self.reps += 1
                                self.last_rep_time = now
                                self.feedback = "Clean Rep! +1 SPEED"

                # --- TIER 2: SQUATS ---
                elif self.current_exercise == "SQUATS":
                    if not self.joints_are_visible(landmarks, [23, 25, 27], threshold=0.6):
                        self.feedback = "Step back! Need to see Hips & Knees."
                    else:
                        raw_angle = self.calculate_angle(get_xy(23), get_xy(25), get_xy(27))
                        self.angle_history.append(raw_angle)
                        smooth_angle = np.mean(self.angle_history)

                        if smooth_angle > 160:
                            self.hold_frames = self.hold_frames + 1 if self.stage == "DOWN" else 0
                            if self.hold_frames >= self.REQUIRED_HOLD and (now - self.last_rep_time > 1.0):
                                self.stage = "UP"
                                self.reps += 1
                                self.last_rep_time = now
                                self.feedback = "Great Squat! +1 STAMINA"
                        elif smooth_angle < 105:
                            self.hold_frames = self.hold_frames + 1 if self.stage == "UP" else 0
                            if self.hold_frames >= self.REQUIRED_HOLD:
                                self.stage = "DOWN"
                                self.feedback = "Good depth! Stand UP!"

                # --- TIER 3: PUSH-UPS ---
                elif self.current_exercise == "PUSHUPS":
                    if not self.joints_are_visible(landmarks, [11, 12, 13, 14, 15, 16]):
                        self.feedback = "Show BOTH arms & hands clearly!"
                    else:
                        hands_below = (landmarks[15].y > landmarks[11].y) and (landmarks[16].y > landmarks[12].y)
                        if not hands_below:
                            self.feedback = "Get in push-up position!"
                        else:
                            l_ang = self.calculate_angle(get_xy(11), get_xy(13), get_xy(15))
                            r_ang = self.calculate_angle(get_xy(12), get_xy(14), get_xy(16))
                            self.angle_history.append((l_ang + r_ang) / 2.0)
                            smooth_angle = np.mean(self.angle_history)

                            if smooth_angle > 155:
                                self.hold_frames = self.hold_frames + 1 if self.stage == "DOWN" else 0
                                if self.hold_frames >= self.REQUIRED_HOLD and (now - self.last_rep_time > 0.9):
                                    self.stage = "UP"
                                    self.reps += 1
                                    self.last_rep_time = now
                                    self.feedback = "Solid Push-up! +1 STRENGTH"
                            elif smooth_angle < 95:
                                self.hold_frames = self.hold_frames + 1 if self.stage == "UP" else 0
                                if self.hold_frames >= self.REQUIRED_HOLD:
                                    self.stage = "DOWN"
                                    self.feedback = "Chest down! Push UP!"

            # Save the processed RGB frame for Pygame to display
            self.latest_frame = rgb_frame

        cap.release()
        detector.close()

# =====================================================================
# TEST YOUR ENGINE: Run `python vision_engine.py` to verify Threading!
# =====================================================================
if __name__ == "__main__":
    engine = VisionEngine()
    engine.start()
    print("VisionEngine Thread Running! Keys: 1=Arms, 2=Squats, 3=Pushups, Space=Cheat Rep, Q=Quit")

    while True:
        if engine.latest_frame is not None:
            # Convert RGB back to BGR just for this OpenCV preview window
            bgr_preview = cv2.cvtColor(engine.latest_frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr_preview, (0, 0), (640, 80), (20, 20, 20), -1)
            cv2.putText(bgr_preview, f"{engine.current_exercise} | Reps: {engine.reps}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(bgr_preview, engine.feedback, (15, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.imshow("Final VisionEngine Test (Threaded)", bgr_preview)

        key = cv2.waitKey(15) & 0xFF
        if key == ord('q'):
            engine.stop()
            break
        elif key == ord('1'):
            engine.set_exercise("ARM_RAISES")
        elif key == ord('2'):
            engine.set_exercise("SQUATS")
        elif key == ord('3'):
            engine.set_exercise("PUSHUPS")
        elif key == ord(' '):  # SPACEBAR = Judge Cheat Key!
            engine.add_cheat_rep()

    cv2.destroyAllWindows()