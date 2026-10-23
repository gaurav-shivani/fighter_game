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
# TUNING NUMBERS
# ======================================================================
SMOOTHING = 0.55       # 0.55 smooths out webcam jitter while staying fast!
SQUAT_UP = 0.70        # hips-to-knees ratio when standing
SQUAT_DOWN = 0.55      # ratio when squatting low
PUSHUP_UP = 145        # elbow angle, arms straight
PUSHUP_DOWN = 105      # elbow angle, chest low
VISIBLE = 0.35         # joint visibility threshold

EXERCISES = [
    # Legacy aliases (so teammate code using old names still works)
    "ARM_RAISES", "SQUATS", "PUSHUPS",

    # 1. YOGA (Counts SECONDS held in self.reps!)
    "YOGA_1_MOUNTAIN", "YOGA_2_OVERHEAD", "YOGA_3_NAMASTE", "YOGA_4_CHAIR",
    "YOGA_5_FOLD", "YOGA_6_WARRIOR2", "YOGA_7_TREE", "YOGA_8_WARRIOR3",
    "YOGA_9_DANCER", "YOGA_10_SUN_SALUTATION",

    # 2. STRENGTH (Muscle Bruiser)
    "STR_1_ARM_RAISES", "STR_2_BICEP_CURLS", "STR_3_SHOULDER_PRESS",
    "STR_4_SQUATS", "STR_5_LUNGES", "STR_6_WALL_SIT", "STR_7_PUSHUPS",
    "STR_8_PLANK", "STR_9_JUMP_SQUATS", "STR_10_BURPEES",

    # 3. CARDIO (Speedster)
    "CAR_1_MARCHING", "CAR_2_HIGH_KNEES", "CAR_3_JUMPING_JACKS",
    "CAR_4_SIDE_STEPS", "CAR_6_SKATER_HOPS", "CAR_5_BUTT_KICKS",
    "CAR_7_JUMP_ROPE", "CAR_8_MOUNTAIN_CLIMBERS", "CAR_9_TUCK_JUMPS",
    "CAR_10_SPRINT",

    # 4. MARTIAL ARTS (Warrior)
    "MA_1_GUARD", "MA_2_PUNCHES", "MA_3_JAB_CROSS", "MA_4_BLOCKS",
    "MA_5_HORSE_STANCE", "MA_6_FRONT_KICKS", "MA_7_KNEE_STRIKES",
    "MA_8_PUNCH_KICK_COMBO", "MA_9_SIDE_KICKS", "MA_10_KATA",
]


class SmoothPoint:
    """Holds smoothed (x, y, z, visibility) for a single body landmark."""
    def __init__(self, x, y, z, v):
        self.x, self.y, self.z, self.visibility = x, y, z, v


class VisionEngine:
    def __init__(self):
        # ---- SHARED VARIABLES (teammates read these in Pygame) ----
        self.reps = 0
        self.stage = "UP"
        self.current_exercise = "MA_2_PUNCHES"
        self.feedback = "Step into frame!"
        self.latest_frame = None       # RGB frame ready for Pygame
        self.in_frame = False          # True when a body is detected
        self.debug_value = 0.0         # live number being tracked
        self.progress = 0.0            # 0.0 to 1.0 progress bar for current rep!
        self.running = False
        self.thread = None

        # ---- SMOOTHING & STATE TRACKERS ----
        self.smooth_lm = None          # 33 smoothed landmarks
        self.prev_wrists = None        # tracks wrist speed for front punches
        self.x_center_hist = deque(maxlen=25)
        self.y_bounce_hist = deque(maxlen=15)
        self.hold_frames = 0
        self.last_rep_time = 0

        self.hold_accum = 0.0          # accumulates seconds for Yoga / Holds
        self.grace_timer = 0.0         # prevents Yoga holds from resetting on 1-frame glitches
        self.last_frame_time = time.time()
        self.last_side = None          # "LEFT" or "RIGHT" for alternating moves
        self.chain_step = 0            # step 0..N for Lv10 boss combos
        self.punch_state = {"L": "NEED_GUARD", "R": "NEED_GUARD"}

        # ---- MODEL ----
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
        if self.thread is not None and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def set_exercise(self, name):
        alias_map = {
            "ARM_RAISES": "STR_1_ARM_RAISES",
            "SQUATS": "STR_4_SQUATS",
            "PUSHUPS": "STR_7_PUSHUPS",
        }
        name = alias_map.get(name, name)
        if name not in EXERCISES:
            raise ValueError(f"Unknown exercise {name}. Choose from EXERCISES list.")

        self.current_exercise = name
        self.reps = 0
        self.hold_frames = 0
        self.hold_accum = 0.0
        self.grace_timer = 0.0
        self.progress = 0.0
        self.last_side = None
        self.chain_step = 0
        self.punch_state = {"L": "NEED_GUARD", "R": "NEED_GUARD"}
        self.stage = "UP"
        self.feedback = f"Ready: {name}!"

    def add_cheat_rep(self):
        self.reps += 1
        self.progress = 1.0
        self.feedback = "COMPLETED! (Demo Override)"

    # ------------------------------------------------------------------
    # LANDMARK SMOOTHER & GEOMETRY HELPERS
    # ------------------------------------------------------------------
    def _smooth_landmarks(self, raw_lm):
        """Exponential Moving Average filter over all 33 joints to kill jitter."""
        if self.smooth_lm is None:
            self.smooth_lm = [SmoothPoint(p.x, p.y, p.z, p.visibility) for p in raw_lm]
        else:
            a = SMOOTHING
            for i, p in enumerate(raw_lm):
                s = self.smooth_lm[i]
                s.x = a * p.x + (1.0 - a) * s.x
                s.y = a * p.y + (1.0 - a) * s.y
                s.z = a * p.z + (1.0 - a) * s.z
                s.visibility = a * p.visibility + (1.0 - a) * s.visibility
        return self.smooth_lm

    def _angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        ang = np.abs(rad * 180.0 / np.pi)
        return 360.0 - ang if ang > 180.0 else ang

    def _xy(self, lm, idx):
        return [lm[idx].x, lm[idx].y]

    def _dist(self, lm, i, j):
        return float(np.hypot(lm[i].x - lm[j].x, lm[i].y - lm[j].y))

    def _best_side(self, lm):
        left = (11, 13, 15, 23, 25, 27)
        right = (12, 14, 16, 24, 26, 28)
        l_score = sum(lm[i].visibility for i in left)
        r_score = sum(lm[i].visibility for i in right)
        return left if l_score >= r_score else right

    # ------------------------------------------------------------------
    # 4 CORE EXERCISE ENGINES (Rep, Hold, Alternate, Chain)
    # ------------------------------------------------------------------
    def _count(self, at_top, at_bottom, now, cooldown, msg_bottom, msg_rep, hold=2):
        if at_bottom:
            self.progress = 0.15
            if self.stage == "UP":
                self.hold_frames += 1
                if self.hold_frames >= hold:
                    self.stage = "DOWN"
                    self.hold_frames = 0
                    self.feedback = msg_bottom
        elif at_top:
            self.progress = 1.0
            if self.stage == "DOWN":
                self.hold_frames += 1
                if self.hold_frames >= hold and (now - self.last_rep_time > cooldown):
                    self.stage = "UP"
                    self.hold_frames = 0
                    self.reps += 1
                    self.last_rep_time = now
                    self.feedback = msg_rep
        else:
            self.hold_frames = 0
            self.progress = 0.5 if self.stage == "DOWN" else 0.0

    def _hold(self, in_pose, dt, pose_name, hint_msg):
        """Yoga/Isometric timer with a 0.35s grace buffer so brief glitches don't interrupt."""
        if in_pose:
            self.grace_timer = 0.35
            self.stage = "HOLDING"
            self.hold_accum += dt
            self.progress = min(1.0, (self.hold_accum % 1.0))
            self.debug_value = self.hold_accum
            new_seconds = int(self.hold_accum)
            if new_seconds > self.reps:
                self.reps = new_seconds
                self.progress = 1.0
            self.feedback = f"Holding {pose_name}... {self.hold_accum:.1f}s!"
        else:
            if self.grace_timer > 0:
                self.grace_timer -= dt
                self.hold_accum += dt
            else:
                self.stage = "ADJUST"
                self.progress = 0.0
                self.feedback = hint_msg

    def _alternate(self, left_active, right_active, now, cooldown, move_name):
        if left_active and not right_active and self.last_side != "LEFT":
            if now - self.last_rep_time > cooldown:
                self.last_side = "LEFT"
                self.stage = "LEFT"
                self.reps += 1
                self.progress = 1.0
                self.last_rep_time = now
                self.feedback = f"Left {move_name}! Now Right!"
        elif right_active and not left_active and self.last_side != "RIGHT":
            if now - self.last_rep_time > cooldown:
                self.last_side = "RIGHT"
                self.stage = "RIGHT"
                self.reps += 1
                self.progress = 1.0
                self.last_rep_time = now
                self.feedback = f"Right {move_name}! Now Left!"
        else:
            self.progress = 0.3

    def _chain(self, conditions, prompts, now):
        step = self.chain_step
        self.progress = step / float(len(conditions))
        self.debug_value = float(step + 1)
        if conditions[step]:
            self.hold_frames += 1
            if self.hold_frames >= 2:
                self.hold_frames = 0
                self.chain_step += 1
                if self.chain_step >= len(conditions):
                    self.chain_step = 0
                    self.reps += 1
                    self.progress = 1.0
                    self.last_rep_time = now
                    self.feedback = "COMBO FINISHED! +1 REP!"
                else:
                    self.feedback = f"Good! Next: {prompts[self.chain_step]}"
        else:
            self.hold_frames = 0
            self.stage = f"STEP {step + 1}/{len(conditions)}"
            self.feedback = f"Step {step + 1}: {prompts[step]}"

    def _draw(self, rgb, lm):
        h, w, _ = rgb.shape
        for s, e in self.BODY_CONNECTIONS:
            seen = lm[s].visibility > VISIBLE and lm[e].visibility > VISIBLE
            color = (0, 255, 0) if seen else (120, 120, 120)
            p1 = (int(lm[s].x * w), int(lm[s].y * h))
            p2 = (int(lm[e].x * w), int(lm[e].y * h))
            cv2.line(rgb, p1, p2, color, 3)

    # ------------------------------------------------------------------
    # REUSABLE POSE DETECTORS
    # ------------------------------------------------------------------
    def _torso_len(self, lm):
        sh_y = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        return abs(hip_y - sh_y) + 1e-6

    def _squat_ratio(self, lm):
        knee_y = (lm[25].y + lm[26].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        return (knee_y - hip_y) / self._torso_len(lm)

    def _pose_mountain(self, lm):
        sh_x, hip_x = (lm[11].x + lm[12].x) / 2, (lm[23].x + lm[24].x) / 2
        upright = abs(sh_x - hip_x) < 0.12 and self._torso_len(lm) > 0.15
        hands_down = lm[15].y > lm[11].y + 0.15 and lm[16].y > lm[12].y + 0.15
        return upright and hands_down and self._squat_ratio(lm) > 0.65

    def _pose_overhead(self, lm):
        return lm[15].y < lm[0].y + 0.02 and lm[16].y < lm[0].y + 0.02

    def _pose_namaste(self, lm):
        sh_y, hip_y = (lm[11].y + lm[12].y) / 2, (lm[23].y + lm[24].y) / 2
        return self._dist(lm, 15, 16) < 0.18 and (sh_y - 0.08 < lm[15].y < hip_y + 0.05)

    def _pose_chair(self, lm):
        arms_up = lm[15].y < lm[11].y and lm[16].y < lm[12].y
        return arms_up and self._squat_ratio(lm) < 0.68

    def _pose_fold(self, lm):
        hip_y, sh_y = (lm[23].y + lm[24].y) / 2, (lm[11].y + lm[12].y) / 2
        return max(lm[15].y, lm[16].y) > hip_y + 0.08 and abs(sh_y - hip_y) < 0.22

    def _pose_plank(self, lm):
        sh, el, wr, hip, kn, ank = self._best_side(lm)
        return abs(lm[sh].x - lm[hip].x) > abs(lm[sh].y - lm[hip].y) * 0.65

    def _arm_punch_metrics(self, lm, side_char, sh_w, torso):
        """
        Calculates whether an arm is in GUARD (retracted) or PUNCHING (extended).
        Requires hand to be above waist so arms hanging at sides NEVER count!
        """
        sh, el, wr = (11, 13, 15) if side_char == "L" else (12, 14, 16)
        hip_y = (lm[23].y + lm[24].y) / 2

        # 1. Strike-zone check: hand must be above waist/belly (not hanging down!)
        hand_up_in_zone = lm[wr].y < hip_y - 0.12 * torso
        if not hand_up_in_zone:
            return False, False, 0.0

        ang = self._angle(self._xy(lm, sh), self._xy(lm, el), self._xy(lm, wr))
        reach = self._dist(lm, sh, wr) / sh_w

        # Front-punch check: when punching straight at camera, elbow rises behind wrist
        # and wrist moves outward/upward from guard position
        elbow_lifted_behind_fist = abs(lm[el].y - lm[wr].y) < 0.16 * torso and lm[wr].y < lm[sh].y + 0.25 * torso

        # Retracted in Guard: fist is near chin/chest/shoulder AND elbow is bent below fist
        in_guard = (ang < 115 and reach < 0.82) and (lm[wr].y < lm[el].y + 0.05)

        # Extended in Punch: either arm straightens/reaches out OR front-camera snap punch
        is_punch = (ang > 132 and reach > 0.85) or (reach > 1.05) or (elbow_lifted_behind_fist and reach > 0.72 and ang > 118)

        score = np.clip((max(ang - 80, reach * 100 - 45)) / 65.0, 0.0, 1.0)
        return in_guard, is_punch, float(score)

    # ------------------------------------------------------------------
    # MASTER ROUTER FOR ALL 40 EXERCISES
    # ------------------------------------------------------------------
    def _process_exercise(self, lm, now, dt):
        ex = self.current_exercise
        sh_y = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        hip_x = (lm[23].x + lm[24].x) / 2
        sh_w = abs(lm[11].x - lm[12].x) + 1e-6
        torso = self._torso_len(lm)

        self.x_center_hist.append(hip_x)
        self.y_bounce_hist.append(sh_y)

        # ==================== 1. YOGA (1-10) ====================
        if ex == "YOGA_1_MOUNTAIN":
            self._hold(self._pose_mountain(lm), dt, "Mountain Pose", "Stand tall with arms relaxed at your sides!")
        elif ex == "YOGA_2_OVERHEAD":
            self._hold(self._pose_overhead(lm), dt, "Arms Overhead", "Raise BOTH hands above your head!")
        elif ex == "YOGA_3_NAMASTE":
            self._hold(self._pose_namaste(lm), dt, "Namaste", "Press palms together at chest height!")
        elif ex == "YOGA_4_CHAIR":
            self._hold(self._pose_chair(lm), dt, "Chair Pose", "Bend knees slightly & raise both arms up!")
        elif ex == "YOGA_5_FOLD":
            self._hold(self._pose_fold(lm), dt, "Forward Fold", "Bend forward & reach hands down below hips!")
        elif ex == "YOGA_6_WARRIOR2":
            arms_out = abs(lm[15].x - lm[16].x) > sh_w * 1.6 and abs(lm[15].y - sh_y) < 0.20 and abs(lm[16].y - sh_y) < 0.20
            wide = abs(lm[25].x - lm[26].x) > sh_w * 1.15 or abs(lm[27].x - lm[28].x) > sh_w * 1.25
            self._hold(arms_out and wide, dt, "Warrior II", "Step feet wide & stretch both arms straight out!")
        elif ex == "YOGA_7_TREE":
            one_up = abs(lm[27].y - lm[28].y) > 0.07 or abs(lm[25].y - lm[26].y) > 0.05
            self._hold(one_up, dt, "Tree Pose", "Balance on one leg, lift other foot to your knee!")
        elif ex == "YOGA_8_WARRIOR3":
            self._hold(abs(sh_y - hip_y) < 0.22, dt, "Warrior III", "Lean torso forward flat like a 'T'!")
        elif ex == "YOGA_9_DANCER":
            lifted = abs(lm[27].y - lm[28].y) > 0.08 or (lm[27].y < lm[25].y + 0.08) or (lm[28].y < lm[26].y + 0.08)
            self._hold(lifted and (lm[15].y < hip_y or lm[16].y < hip_y), dt, "Dancer Pose", "Lift one foot behind you & reach forward!")
        elif ex == "YOGA_10_SUN_SALUTATION":
            self._chain([self._pose_overhead(lm), self._pose_fold(lm), self._pose_chair(lm), self._pose_mountain(lm)],
                        ["1. Arms Overhead", "2. Forward Fold", "3. Chair Pose", "4. Mountain Pose"], now)

        # ==================== 2. STRENGTH (1-10) ====================
        elif ex == "STR_1_ARM_RAISES":
            top_hand = min(lm[15].y, lm[16].y)
            self.debug_value = lm[0].y - top_hand
            self._count(top_hand < lm[0].y, top_hand > sh_y, now, 0.4, "Drop hand to shoulder, then raise above nose!", "Clean Arm Raise!", hold=2)

        elif ex == "STR_2_BICEP_CURLS":
            # Uses wrist height relative to elbow so front-facing curls work 100% of the time!
            ang_l = self._angle(self._xy(lm, 11), self._xy(lm, 13), self._xy(lm, 15))
            ang_r = self._angle(self._xy(lm, 12), self._xy(lm, 14), self._xy(lm, 16))
            curled_up = (lm[15].y < lm[13].y - 0.03 * torso and ang_l < 110) or (lm[16].y < lm[14].y - 0.03 * torso and ang_r < 110)
            extended_down = (lm[15].y > lm[13].y + 0.15 * torso) and (lm[16].y > lm[14].y + 0.15 * torso)
            self.debug_value = min(ang_l, ang_r)
            self._count(curled_up, extended_down, now, 0.45, "Curl hands UP above elbows!", "Nice Bicep Curl!", hold=2)

        elif ex == "STR_3_SHOULDER_PRESS":
            both_up = lm[15].y < lm[0].y - 0.02 and lm[16].y < lm[0].y - 0.02
            at_shoulders = (lm[0].y + 0.02 < lm[15].y < hip_y) and (lm[0].y + 0.02 < lm[16].y < hip_y)
            self._count(both_up, at_shoulders, now, 0.5, "Bring hands to shoulders, then press OVERHEAD!", "Strong Shoulder Press!", hold=2)

        elif ex == "STR_4_SQUATS":
            r = self._squat_ratio(lm)
            self.debug_value = r
            self._count(r > SQUAT_UP, r < SQUAT_DOWN, now, 0.65, "Good depth! Stand UP!", "Great Squat!", hold=2)

        elif ex == "STR_5_LUNGES":
            ang_l = self._angle(self._xy(lm, 23), self._xy(lm, 25), self._xy(lm, 27))
            ang_r = self._angle(self._xy(lm, 24), self._xy(lm, 26), self._xy(lm, 28))
            r = self._squat_ratio(lm)
            self.debug_value = r
            self._count(r > 0.70 and min(ang_l, ang_r) > 145, r < 0.58 or min(ang_l, ang_r) < 125, now, 0.65, "Drive back UP from lunge!", "Solid Lunge!", hold=2)

        elif ex == "STR_6_WALL_SIT":
            r = self._squat_ratio(lm)
            self.debug_value = r
            self._hold(r < 0.62, dt, "Wall Sit", "Drop hips into a squat & HOLD still!")

        elif ex == "STR_7_PUSHUPS":
            sh, el, wr, hip, _, _ = self._best_side(lm)
            ang = self._angle(self._xy(lm, sh), self._xy(lm, el), self._xy(lm, wr))
            self.debug_value = ang
            self._count(ang > PUSHUP_UP, ang < PUSHUP_DOWN, now, 0.6, "Chest down! Push UP!", "Power Push-up!", hold=2)

        elif ex == "STR_8_PLANK":
            self._hold(self._pose_plank(lm), dt, "Plank", "Hold a horizontal plank position!")

        elif ex == "STR_9_JUMP_SQUATS":
            r = self._squat_ratio(lm)
            mean_y = float(np.mean(self.y_bounce_hist))
            self.debug_value = r
            self._count(r > 0.68 and sh_y < mean_y - 0.015, r < 0.56, now, 0.6, "EXPLODE UP into a jump!", "Explosive Jump Squat!", hold=1)

        elif ex == "STR_10_BURPEES":
            standing = self._squat_ratio(lm) > 0.68 and not self._pose_plank(lm)
            squatting = self._squat_ratio(lm) < 0.56
            planking = self._pose_plank(lm)
            self._chain([standing, squatting, planking, standing],
                        ["1. Stand Tall", "2. Squat Down", "3. Plank Position", "4. Stand Up!"], now)

        # ==================== 3. CARDIO (1-10) ====================
        elif ex in ("CAR_1_MARCHING", "CAR_2_HIGH_KNEES", "CAR_10_SPRINT"):
            knee_diff = (lm[26].y - lm[25].y) / torso
            self.debug_value = knee_diff
            thresh = 0.07 if ex == "CAR_1_MARCHING" else 0.11
            cd = 0.28 if ex == "CAR_1_MARCHING" else (0.18 if ex == "CAR_2_HIGH_KNEES" else 0.12)
            self._alternate(knee_diff > thresh, knee_diff < -thresh, now, cd, "Knee Up")

        elif ex == "CAR_3_JUMPING_JACKS":
            hands_up = lm[15].y < lm[0].y + 0.02 and lm[16].y < lm[0].y + 0.02
            hands_down = lm[15].y > lm[11].y + 0.05 and lm[16].y > lm[12].y + 0.05
            self._count(hands_up, hands_down, now, 0.35, "Jump & clap hands overhead!", "Jumping Jack!", hold=1)

        elif ex in ("CAR_4_SIDE_STEPS", "CAR_6_SKATER_HOPS"):
            offset = hip_x - float(np.mean(self.x_center_hist))
            self.debug_value = offset
            thresh = 0.035 if ex == "CAR_4_SIDE_STEPS" else 0.06
            self._alternate(offset < -thresh, offset > thresh, now, 0.30, "Step")

        elif ex == "CAR_5_BUTT_KICKS":
            ank_diff = (lm[28].y - lm[27].y) / torso
            ang_l = self._angle(self._xy(lm, 23), self._xy(lm, 25), self._xy(lm, 27))
            ang_r = self._angle(self._xy(lm, 24), self._xy(lm, 26), self._xy(lm, 28))
            self._alternate(ank_diff > 0.07 or ang_l < 125, ank_diff < -0.07 or ang_r < 125, now, 0.22, "Heel Kick")

        elif ex == "CAR_7_JUMP_ROPE":
            bounce = float(np.mean(self.y_bounce_hist)) - sh_y
            self.debug_value = bounce * 100
            self._count(bounce > 0.008, bounce < -0.005, now, 0.20, "Bounce!", "Jump Rope Hop!", hold=1)

        elif ex == "CAR_8_MOUNTAIN_CLIMBERS":
            diff = self._dist(lm, 26, 12) - self._dist(lm, 25, 11)
            self._alternate(diff > 0.05, diff < -0.05, now, 0.22, "Climber Step")

        elif ex == "CAR_9_TUCK_JUMPS":
            r = self._squat_ratio(lm)
            mean_y = float(np.mean(self.y_bounce_hist))
            self._count(r < 0.56 or sh_y < mean_y - 0.025, r > 0.66, now, 0.5, "JUMP & tuck knees!", "Tuck Jump!", hold=1)

        # ==================== 4. MARTIAL ARTS (1-10) ====================
        elif ex == "MA_1_GUARD":
            l_guard, _, _ = self._arm_punch_metrics(lm, "L", sh_w, torso)
            r_guard, _, _ = self._arm_punch_metrics(lm, "R", sh_w, torso)
            self._hold(l_guard and r_guard, dt, "Guard Stance", "Hold BOTH fists up by your chin (elbows bent)!")

        elif ex in ("MA_2_PUNCHES", "MA_3_JAB_CROSS"):
            l_guard, l_punch, l_score = self._arm_punch_metrics(lm, "L", sh_w, torso)
            r_guard, r_punch, r_score = self._arm_punch_metrics(lm, "R", sh_w, torso)
            self.debug_value = max(l_score, r_score)
            self.progress = max(l_score, r_score)

            # If hands are dropped down by waist, prompt user to raise guard!
            if lm[15].y > hip_y - 0.10 * torso and lm[16].y > hip_y - 0.10 * torso:
                self.feedback = "Bring fists UP to chest/chin level first!"
                self.stage = "GUARD LOW"
                return

            # State machine per arm: MUST retract to Guard before a Punch can score!
            if l_guard:
                self.punch_state["L"] = "READY"
            if r_guard:
                self.punch_state["R"] = "READY"

            if ex == "MA_2_PUNCHES":
                self.stage = f"L:{self.punch_state['L'][0]} R:{self.punch_state['R'][0]}"
                if l_punch and self.punch_state["L"] == "READY" and (now - self.last_rep_time > 0.25):
                    self.punch_state["L"] = "PUNCHED"
                    self.reps += 1
                    self.last_rep_time = now
                    self.feedback = "LEFT PUNCH! Pull back to chin for next!"
                elif r_punch and self.punch_state["R"] == "READY" and (now - self.last_rep_time > 0.25):
                    self.punch_state["R"] = "PUNCHED"
                    self.reps += 1
                    self.last_rep_time = now
                    self.feedback = "RIGHT PUNCH! Pull back to chin for next!"
                elif self.punch_state["L"] != "READY" and self.punch_state["R"] != "READY":
                    self.feedback = "Pull fists back to chin (Guard) to reload!"
                else:
                    self.feedback = "Fists ready! Punch out (slight angle works best)!"
            else:
                # MA_3_JAB_CROSS: Must alternate Left then Right (or Right then Left)
                if l_punch and self.punch_state["L"] == "READY" and self.last_side != "LEFT" and (now - self.last_rep_time > 0.25):
                    self.punch_state["L"] = "PUNCHED"
                    self.last_side = "LEFT"
                    self.reps += 1
                    self.last_rep_time = now
                    self.feedback = "Left Jab! Now Right Cross!"
                elif r_punch and self.punch_state["R"] == "READY" and self.last_side != "RIGHT" and (now - self.last_rep_time > 0.25):
                    self.punch_state["R"] = "PUNCHED"
                    self.last_side = "RIGHT"
                    self.reps += 1
                    self.last_rep_time = now
                    self.feedback = "Right Cross! Now Left Jab!"

        elif ex == "MA_4_BLOCKS":
            high_block = lm[15].y < lm[0].y + 0.05 and lm[16].y < lm[0].y + 0.05
            low_ready = lm[15].y > lm[11].y + 0.04 and lm[16].y > lm[12].y + 0.04
            self._count(high_block, low_ready, now, 0.4, "Raise BOTH forearms to block your face!", "Solid Block!", hold=1)

        elif ex == "MA_5_HORSE_STANCE":
            wide = abs(lm[25].x - lm[26].x) > sh_w * 1.15 or abs(lm[27].x - lm[28].x) > sh_w * 1.25
            self._hold(wide and self._squat_ratio(lm) < 0.68, dt, "Horse Stance", "Spread feet wide, sink low, and HOLD!")

        elif ex == "MA_6_FRONT_KICKS":
            knee_diff = (lm[26].y - lm[25].y) / torso
            ank_diff = (lm[28].y - lm[27].y) / torso
            self._alternate(knee_diff > 0.10 or ank_diff > 0.14, knee_diff < -0.10 or ank_diff < -0.14, now, 0.4, "Front Kick")

        elif ex == "MA_7_KNEE_STRIKES":
            knee_up = abs(lm[25].y - lm[26].y) / torso > 0.10 or self._squat_ratio(lm) < 0.58
            hands_up = lm[15].y < lm[11].y + 0.05 and lm[16].y < lm[12].y + 0.05
            hands_pull = lm[15].y > lm[11].y + 0.05 or lm[16].y > lm[12].y + 0.05
            self._count(knee_up and hands_pull, hands_up and not knee_up, now, 0.45, "Hands UP, then pull DOWN as knee strikes!", "Brutal Knee Strike!", hold=1)

        elif ex == "MA_8_PUNCH_KICK_COMBO":
            _, l_punch, _ = self._arm_punch_metrics(lm, "L", sh_w, torso)
            _, r_punch, _ = self._arm_punch_metrics(lm, "R", sh_w, torso)
            kick = abs(lm[25].y - lm[26].y) / torso > 0.10 or abs(lm[27].y - lm[28].y) / torso > 0.14
            self._chain([l_punch, r_punch, kick], ["1. Left Punch!", "2. Right Punch!", "3. Front Kick!"], now)

        elif ex == "MA_9_SIDE_KICKS":
            l_out = abs(lm[27].x - lm[23].x) > sh_w * 0.65 or abs(lm[25].x - lm[23].x) > sh_w * 0.45
            r_out = abs(lm[28].x - lm[24].x) > sh_w * 0.65 or abs(lm[26].x - lm[24].x) > sh_w * 0.45
            lifted = abs(lm[25].y - lm[26].y) / torso > 0.08 or abs(lm[27].y - lm[28].y) / torso > 0.12
            self._count((l_out or r_out) and lifted, not lifted, now, 0.5, "Kick one leg out wide to the side!", "Sharp Side Kick!", hold=1)

        elif ex == "MA_10_KATA":
            l_guard, l_punch, _ = self._arm_punch_metrics(lm, "L", sh_w, torso)
            r_guard, r_punch, _ = self._arm_punch_metrics(lm, "R", sh_w, torso)
            block = self._pose_overhead(lm)
            kick = abs(lm[25].y - lm[26].y) / torso > 0.10 or abs(lm[27].y - lm[28].y) / torso > 0.14
            self._chain([l_guard and r_guard, l_punch, r_punch, block, kick],
                        ["1. Guard Stance", "2. Left Punch", "3. Right Punch", "4. High Block", "5. Kick!"], now)

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
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = vision.PoseLandmarker.create_from_options(options)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.feedback = "Camera not found!"
                return

            start_time = time.time()
            self.last_frame_time = start_time
            last_ts = 0

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                now = time.time()
                dt = min(now - self.last_frame_time, 0.1)
                self.last_frame_time = now

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ts = int((now - start_time) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect_for_video(mp_image, ts)

                if results.pose_landmarks:
                    # Smooth all 33 joints BEFORE drawing or checking exercises!
                    lm = self._smooth_landmarks(results.pose_landmarks[0])
                    self.in_frame = True
                    self._draw(rgb, lm)
                    self._process_exercise(lm, now, dt)
                else:
                    self.in_frame = False
                    self.smooth_lm = None
                    self.feedback = "No body detected, step into the frame!"

                self.latest_frame = rgb
        finally:
            if cap is not None:
                cap.release()
            if detector is not None:
                detector.close()
            self.running = False


# ======================================================================
# TEST ALL 40 EXERCISES: run `python vision_engine.py`
# Starts right on MA_2_PUNCHES so you can test punches immediately!
# ======================================================================
if __name__ == "__main__":
    engine = VisionEngine()
    engine.start()
    playable = EXERCISES[3:]
    idx = playable.index("MA_2_PUNCHES")  # Starts directly on Punches!
    engine.set_exercise(playable[idx])

    while engine.running:
        if engine.latest_frame is not None:
            preview = cv2.cvtColor(engine.latest_frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(preview, (0, 0), (640, 118), (20, 20, 20), -1)
            cv2.putText(preview, f"({idx+1}/40) {engine.current_exercise} | Score: {engine.reps}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            cv2.putText(preview, engine.feedback, (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
            cv2.putText(preview, f"[{engine.stage}] N/P=Move | 1=Yoga 2=Str 3=Car 4=MA | Space=Cheat",
                        (15, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

            # Live Progress Bar (0% to 100%)
            cv2.rectangle(preview, (15, 96), (625, 110), (60, 60, 60), -1)
            bar_w = int(610 * np.clip(engine.progress, 0.0, 1.0))
            cv2.rectangle(preview, (15, 96), (15 + bar_w, 110), (0, 255, 120), -1)

            cv2.imshow("VisionEngine - Stabilized + Progress Bar", preview)

        key = cv2.waitKey(15) & 0xFF
        if key == ord('q'):
            engine.stop()
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(playable)
            engine.set_exercise(playable[idx])
        elif key == ord('p'):
            idx = (idx - 1) % len(playable)
            engine.set_exercise(playable[idx])
        elif key == ord('1'):
            idx = 0
            engine.set_exercise(playable[idx])
        elif key == ord('2'):
            idx = 10
            engine.set_exercise(playable[idx])
        elif key == ord('3'):
            idx = 20
            engine.set_exercise(playable[idx])
        elif key == ord('4'):
            idx = 30
            engine.set_exercise(playable[idx])
        elif key == ord(' '):
            engine.add_cheat_rep()

    cv2.destroyAllWindows()