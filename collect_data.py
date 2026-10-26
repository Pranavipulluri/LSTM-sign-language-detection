"""
collect_data.py  –  Collect training data for sign language detection
=======================================================================
For each sign, you'll record 30 sequences × 30 frames each.

HOW TO USE:
  1. Run the script.
  2. A window will open showing your webcam feed with landmarks.
  3. For each sign, you get a 2-second pause to position your hands,
     then 30 frames are captured automatically.
  4. This repeats 30 times per sign (sequences 0–29).
  5. Press 'q' at any time to abort.

Controls shown on screen:
  • Countdown phase  — position your hands before recording starts
  • COLLECTING phase — hold the sign steady
"""

import cv2
import numpy as np
import os
import time
import mediapipe as mp

# ─── Settings (must match detect.py / train.py) ────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), 'MP_Data')
actions         = np.array(['cat', 'food', 'help'])
no_sequences    = 30   # number of video sequences per sign
sequence_length = 30   # frames per sequence

# ─── MediaPipe ──────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    face = (np.array([[r.x, r.y, r.z]
                      for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3))
    lh   = (np.array([[r.x, r.y, r.z]
                      for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z]
                      for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, face, lh, rh])


def overlay_text(image, line1, line2='', color=(255, 255, 255)):
    cv2.rectangle(image, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(image, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    if line2:
        cv2.putText(image, line2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (200, 200, 200), 1, cv2.LINE_AA)


# ─── Create folder structure ─────────────────────────────────────────────────
print("[INFO] Creating MP_Data folder structure …")
for action in actions:
    for seq in range(no_sequences):
        folder = os.path.join(DATA_PATH, action, str(seq))
        os.makedirs(folder, exist_ok=True)
print("[INFO] Folders ready.")
print(f"[INFO] Will collect {no_sequences} sequences × {sequence_length} frames "
      f"for each of: {list(actions)}\n")

# ─── Collection loop ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

aborted = False

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    for action in actions:
        for sequence in range(no_sequences):
            if aborted:
                break

            # ── Countdown / get-ready phase ──────────────────────────────
            ready_start = time.time()
            while time.time() - ready_start < 2:
                ret, frame = cap.read()
                if not ret:
                    continue
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                countdown = int(2 - (time.time() - ready_start)) + 1
                overlay_text(
                    image,
                    f"Sign: '{action.upper()}'  |  Sequence: {sequence+1}/{no_sequences}",
                    f"Get ready … starting in {countdown}s",
                    color=(0, 200, 255))

                cv2.imshow('Data Collection  —  press Q to abort', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    aborted = True
                    break

            if aborted:
                break

            # ── Recording phase ──────────────────────────────────────────
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                overlay_text(
                    image,
                    f"Sign: '{action.upper()}'  |  Seq {sequence+1}/{no_sequences}  "
                    f"Frame {frame_num+1}/{sequence_length}",
                    "COLLECTING — hold the sign steady",
                    color=(0, 255, 0))

                cv2.imshow('Data Collection  —  press Q to abort', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    aborted = True
                    break

                # Save keypoints
                keypoints = extract_keypoints(results)
                save_path = os.path.join(
                    DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(save_path, keypoints)

            if not aborted:
                print(f"  ✓  '{action}'  sequence {sequence+1}/{no_sequences} saved")

        if not aborted:
            print(f"\n[INFO] All sequences for '{action}' done!\n")

cap.release()
cv2.destroyAllWindows()

if aborted:
    print("\n[WARN] Collection aborted by user.")
else:
    print("\n[INFO] Data collection complete!")
    print("[INFO] Now run:  python train.py")
