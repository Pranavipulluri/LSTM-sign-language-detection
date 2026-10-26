"""
Realtime Sign Language Detection using LSTM Model
==================================================
Detects: 'cat', 'food', 'help' signs in real-time.
Press 'q' to quit.
"""

import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow as tf
from scipy import stats

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'model.h5')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.h5')

actions         = np.array(['cat', 'food', 'help'])
sequence_length = 30    # frames per prediction window
threshold       = 0.70  # minimum confidence to accept a prediction
vote_window     = 20    # how many recent frames must agree
vote_ratio      = 0.70  # fraction of vote_window that must agree (stability)
idle_timeout    = 3.0   # seconds with no confident pred → auto-clear sentence
colors          = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# ─── MediaPipe setup ─────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Run mediapipe holistic on a BGR frame; returns BGR frame + results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draw coloured landmarks for face, pose, and both hands."""
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
    """
    Flatten all landmark coordinates into one 1662-element array:
      pose  33 × 4  = 132
      face 468 × 3  = 1404
      lh    21 × 3  = 63
      rh    21 × 3  = 63
                      ────
                      1662
    """
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


def prob_viz(res, actions, input_frame, colors):
    """Overlay a probability bar for each action class on the frame."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame,
                      (0, 60 + num * 40), (int(prob * 100), 90 + num * 40),
                      colors[num], -1)
        cv2.putText(output_frame, actions[num],
                    (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


# ─── Load model ───────────────────────────────────────────────────────────────
print("[INFO] Loading model …")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.h5 not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
model.load_weights(WEIGHTS_PATH)
print("[INFO] Model loaded successfully.")
print(f"[INFO] Detecting signs: {list(actions)}")
print("[INFO] Press 'q' to quit.\n")


# ─── Real-time detection loop ─────────────────────────────────────────────────
sequence     = []    # rolling 30-frame window of keypoints
sentence     = []    # confirmed sign words to display
predictions  = []    # recent argmax predictions for stability voting
last_added_t = None  # timestamp of the last confirmed word
res          = None  # latest model output (for prob_viz)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (device 0).")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame; retrying …")
            continue

        # ── MediaPipe inference ──────────────────────────────────────────────
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # ── Build keypoint sequence ──────────────────────────────────────────
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        # ── Predict once we have a full 30-frame window ──────────────────────
        if len(sequence) == sequence_length:
            res = model.predict(
                np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = int(np.argmax(res))
            confidence    = float(res[predicted_idx])
            predictions.append(predicted_idx)

            # Stability vote: require vote_ratio of last vote_window frames to agree
            if len(predictions) >= vote_window:
                recent    = predictions[-vote_window:]
                mode_val  = stats.mode(recent, keepdims=True)[0][0]
                mode_frac = recent.count(int(mode_val)) / vote_window
                confident = (int(mode_val) == predicted_idx
                             and confidence > threshold
                             and mode_frac >= vote_ratio)

                if confident:
                    if len(sentence) == 0 or actions[predicted_idx] != sentence[-1]:
                        sentence.append(actions[predicted_idx])
                        last_added_t = time.time()

        # ── Idle timeout: clear sentence if no confident pred for N seconds ───
        if last_added_t and (time.time() - last_added_t) > idle_timeout:
            sentence     = []
            last_added_t = None

        # Keep sentence to 5 words max
        if len(sentence) > 5:
            sentence = sentence[-5:]

        # ── Probability bars (only after first prediction) ───────────────────
        if res is not None:
            image = prob_viz(res, actions, image, colors)

        # ── Sentence banner ──────────────────────────────────────────────────
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence),
                    (3, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detector  —  press Q to quit', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
