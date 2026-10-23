import os
import time
import threading
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================================================================
# TUNING NUMBERS - change these if counting feels wrong
# ======================================================================
SQUAT_UP = 0.75        # hips-to-knees ratio when standing (lowered from 0.85 so all body types register)
SQUAT_DOWN = 0.50      # ratio when squatting low
PUSHUP_UP = 155        # elbow angle, arms straight
PUSHUP_DOWN = 95       # elbow angle, chest low
ARM_MARGIN = 0.05      # how far above nose / below shoulder hands must go
VISIBLE = 0.65         # how sure the AI must be that a joint is visible

EXERCISES = ["ARM_RAISES", "SQUATS", "PUSHUPS"]


class VisionEngine:
    def __init__(self):
        # ---- SHARED VARIABLES (teammates read these in Pygame) ----
        self.reps = 0
        self.stage = "UP"
        self.current_exercise = "ARM_RAISES"
        self.feedback = "Step back so joints are visible!"
        self.latest_frame = None       # RGB frame ready for Pygame
        self.in_frame = False          # True when a body is detected
        self.debug_value = 0.0         # the number being tracked (for tuning)
        self.running = False
        self.thread = None

        # ---- ANTI-BOUNCE VARIABLES ----
        self.history = deque(maxlen=5)  # smooths jitter over last 5 frames
        self.hold_frames = 0
        self.REQUIRED_HOLD = 3
        self.last_rep_time = 0

        # ---- MODEL (stored next to this file, so it works from any folder) ----
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pose_landmarker_full.task"
        )
        if not os.path.exists(self.model_path):
            print("Downloading FULL AI Pose Model (~9MB)...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )
            urllib.request.urlretrieve(url, self.model_path)

        self.BODY_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
        ]

    # ------------------------------------------------------------------
    # PUBLIC FUNCTIONS (teammates call these)
    # ------------------------------------------------------------------
    def start(self):
        """Starts the webcam in a background thread so Pygame stays smooth."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Turns off the webcam."""
        self.running = False

    def set_exercise(self, name):
        """Switch exercise. NOTE: this resets reps to 0."""
        if name not in EXERCISES:
            raise ValueError(f"Unknown exercise {name}. Use one of {EXERCISES}")
        self.current_exercise = name
        self.reps = 0
        self.history.clear()
        self.hold_frames = 0
        self.stage = "UP"
        self.feedback = f"Switched to {name}!"

    def add_cheat_rep(self):
        """Emergency demo button: adds 1 rep instantly."""
        self.reps += 1
        self.feedback = "REP COMPLETED! (Demo Override)"

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _angle(self, a, b, c):
        """Angle at point b (degrees) given three [x, y] points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        ang = np.abs(rad * 180.0 / np.pi)
        return 360.0 - ang if ang > 180.0 else ang

    def _visible(self, lm, idx, threshold=VISIBLE):
        """Joint is confidently seen AND inside the screen."""
        p = lm[idx]
        return (
            p.visibility >= threshold
            and 0.02 < p.x < 0.98
            and 0.02 < p.y < 0.98
        )

    def _all_visible(self, lm, indices, threshold=VISIBLE):
        return all(self._visible(lm, i, threshold) for i in indices)

    def _count(self, at_top, at_bottom, now, cooldown, msg_bottom, msg_rep):
        """
        One rep = go to the bottom position, then come back to the top.
        You must hold each position for a few frames, so jitter is ignored.
        """
        if at_bottom:
            self.hold_frames = self.hold_frames + 1 if self.stage == "UP" else 0
            if self.hold_frames >= self.REQUIRED_HOLD:
                self.stage = "DOWN"
                self.hold_frames = 0
                self.feedback = msg_bottom
        elif at_top:
            self.hold_frames = self.hold_frames + 1 if self.stage == "DOWN" else 0
            if self.hold_frames >= self.REQUIRED_HOLD and (now - self.last_rep_time > cooldown):
                self.stage = "UP"
                self.hold_frames = 0
                self.reps += 1
                self.last_rep_time = now
                self.feedback = msg_rep
        else:
            self.hold_frames = 0   # in the middle zone: start counting again

    def _draw(self, rgb, lm):
        h, w, _ = rgb.shape
        for s, e in self.BODY_CONNECTIONS:
            seen = lm[s].visibility > VISIBLE and lm[e].visibility > VISIBLE
            color = (0, 255, 0) if seen else (120, 120, 120)
            p1 = (int(lm[s].x * w), int(lm[s].y * h))
            p2 = (int(lm[e].x * w), int(lm[e].y * h))
            cv2.line(rgb, p1, p2, color, 3)

    # ------------------------------------------------------------------
    # EXERCISES
    # ------------------------------------------------------------------
    def _arm_raises(self, lm, now):
        if not self._all_visible(lm, [0, 11, 12]):
            self.feedback = "Show your upper body in the frame!"
            return
        # Either hand works: use the higher visible wrist
        wrists = [lm[i].y for i in (15, 16) if self._visible(lm, i)]
        if not wrists:
            self.feedback = "Show your hands in the frame!"
            return
        top_hand = min(wrists)                     # smaller y = higher on screen
        shoulder_y = (lm[11].y + lm[12].y) / 2
        self.debug_value = shoulder_y - top_hand   # positive = hand above shoulder

        self._count(
            at_top=top_hand < lm[0].y - ARM_MARGIN,         # hand above nose
            at_bottom=top_hand > shoulder_y + ARM_MARGIN,   # hands below shoulders
            now=now, cooldown=0.7,
            msg_bottom="Now punch HIGH above your nose!",
            msg_rep="Clean Rep! +1 SPEED",
        )

    def _squats(self, lm, now):
        # Works from the FRONT: measures how far hips drop toward knees
        if not self._all_visible(lm, [11, 12, 23, 24, 25, 26], threshold=0.6):
            self.feedback = "Step back! Need to see hips & knees."
            return
        shoulder_y = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        knee_y = (lm[25].y + lm[26].y) / 2
        torso = abs(hip_y - shoulder_y) + 1e-6
        ratio = (knee_y - hip_y) / torso           # standing ~0.85-1.0, deep squat ~0.35

        self.history.append(ratio)
        smooth = float(np.mean(self.history))
        self.debug_value = smooth

        self._count(
            at_top=smooth > SQUAT_UP,
            at_bottom=smooth < SQUAT_DOWN,
            now=now, cooldown=1.0,
            msg_bottom="Good depth! Stand UP!",
            msg_rep="Great Squat! +1 STAMINA",
        )

    def _pushups(self, lm, now):
        # Works from the SIDE: uses whichever arm (+ hip) the camera sees better
        left, right = (11, 13, 15, 23), (12, 14, 16, 24)
        left_score = sum(lm[i].visibility for i in left)
        right_score = sum(lm[i].visibility for i in right)
        side = left if left_score >= right_score else right

        if not self._all_visible(lm, side, threshold=0.5):
            self.feedback = "Show your side profile clearly (arm & hip)!"
            return
        shoulder, elbow, wrist, hip = side

        # Form Check 1: Hands must be below shoulders
        # Form Check 2: Body must be in a horizontal plank (wider in X than tall in Y)
        is_plank = abs(lm[shoulder].x - lm[hip].x) > abs(lm[shoulder].y - lm[hip].y) * 0.7
        if lm[wrist].y < lm[shoulder].y - 0.03 or not is_plank:
            self.feedback = "Get in horizontal push-up plank position!"
            return

        ang = self._angle(
            [lm[shoulder].x, lm[shoulder].y],
            [lm[elbow].x, lm[elbow].y],
            [lm[wrist].x, lm[wrist].y]
        )
        self.history.append(ang)
        smooth = float(np.mean(self.history))
        self.debug_value = smooth

        self._count(
            at_top=smooth > PUSHUP_UP,
            at_bottom=smooth < PUSHUP_DOWN,
            now=now, cooldown=0.9,
            msg_bottom="Chest down! Push UP!",
            msg_rep="Solid Push-up! +1 STRENGTH",
        )

    # ------------------------------------------------------------------
    # BACKGROUND THREAD
    # ------------------------------------------------------------------
    def _loop(self):
        detector = None
        cap = None
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            detector = vision.PoseLandmarker.create_from_options(options)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.feedback = "Camera not found!"
                return

            start_time = time.time()
            last_ts = 0

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)       # don't burn the CPU if frames fail
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Real timestamps (must always increase)
                ts = int((time.time() - start_time) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect_for_video(mp_image, ts)

                if results.pose_landmarks:
                    lm = results.pose_landmarks[0]
                    self.in_frame = True
                    self._draw(rgb, lm)
                    now = time.time()
                    if self.current_exercise == "ARM_RAISES":
                        self._arm_raises(lm, now)
                    elif self.current_exercise == "SQUATS":
                        self._squats(lm, now)
                    elif self.current_exercise == "PUSHUPS":
                        self._pushups(lm, now)
                else:
                    self.in_frame = False
                    self.feedback = "No body detected, step into the frame!"

                self.latest_frame = rgb
        finally:
            if cap is not None:
                cap.release()
            if detector is not None:
                detector.close()
            self.running = False


# ======================================================================
# TEST: run `python vision_engine.py`
# Keys: 1=Arms, 2=Squats, 3=Pushups, Space=Cheat Rep, Q=Quit
# ======================================================================
if __name__ == "__main__":
    engine = VisionEngine()
    engine.start()
    print("Keys: 1=Arms, 2=Squats, 3=Pushups, Space=Cheat Rep, Q=Quit")

    while engine.running:
        if engine.latest_frame is not None:
            preview = cv2.cvtColor(engine.latest_frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(preview, (0, 0), (640, 100), (20, 20, 20), -1)
            cv2.putText(preview, f"{engine.current_exercise} | Reps: {engine.reps} | [{engine.stage}]",
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview, engine.feedback, (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(preview, f"value: {engine.debug_value:.2f}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("VisionEngine Test", preview)

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
        elif key == ord(' '):
            engine.add_cheat_rep()

    cv2.destroyAllWindows()
