"""
train.py  –  Train the LSTM model on your collected data
=========================================================
Run this AFTER collect_data.py has finished.

What this does:
  1. Loads all .npy keypoint files from MP_Data/
  2. Builds the same LSTM architecture as the original model
  3. Trains with early stopping (saves best weights automatically)
  4. Overwrites model.h5 and model_weights.h5
  5. Prints a confusion matrix + accuracy score at the end
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# ─── Settings (must match collect_data.py / detect.py) ─────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), 'MP_Data')
MODEL_PATH      = os.path.join(os.path.dirname(__file__), 'model.h5')
WEIGHTS_PATH    = os.path.join(os.path.dirname(__file__), 'model_weights.h5')

actions         = np.array(['cat', 'food', 'help'])
no_sequences    = 30   # sequences per sign (must match collect_data.py)
sequence_length = 30   # frames per sequence

# ─── 1. Verify data exists ───────────────────────────────────────────────────
print("[INFO] Checking data …")
missing = []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        missing.append(action)
        continue
    for seq in range(no_sequences):
        for frame in range(sequence_length):
            p = os.path.join(action_path, str(seq), f"{frame}.npy")
            if not os.path.exists(p):
                missing.append(f"{action}/seq{seq}/frame{frame}")

if missing:
    print("[ERROR] Missing data files. Run collect_data.py first.")
    print("  Missing:", missing[:5], "…" if len(missing) > 5 else "")
    exit(1)

print(f"[INFO] Data OK — {len(actions)} signs × {no_sequences} sequences "
      f"× {sequence_length} frames")

# ─── 2. Load data ────────────────────────────────────────────────────────────
print("[INFO] Loading keypoint arrays …")
label_map  = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    seq_dirs = sorted(
        [int(d) for d in os.listdir(os.path.join(DATA_PATH, action))
         if d.isdigit()])
    for seq in seq_dirs:
        window = []
        for frame_num in range(sequence_length):
            p = os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy")
            window.append(np.load(p))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)                    # (N, 30, 1662)
y = to_categorical(labels).astype(int)     # (N, num_classes)

print(f"[INFO] X shape: {X.shape}   y shape: {y.shape}")

# ─── 3. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y.argmax(axis=1))

print(f"[INFO] Train: {X_train.shape[0]} samples   Test: {X_test.shape[0]} samples")

# ─── 4. Build model ──────────────────────────────────────────────────────────
print("[INFO] Building LSTM model …")
model = Sequential([
    LSTM(64,  return_sequences=True,  activation='relu',
         input_shape=(sequence_length, X.shape[2])),
    LSTM(128, return_sequences=True,  activation='relu'),
    LSTM(64,  return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax'),
])

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()

# ─── 5. Train ────────────────────────────────────────────────────────────────
print("\n[INFO] Training … (this may take a few minutes)")

callbacks = [
    EarlyStopping(monitor='val_categorical_accuracy',
                  patience=100,
                  restore_best_weights=True,
                  verbose=1),
    ModelCheckpoint(MODEL_PATH,
                    monitor='val_categorical_accuracy',
                    save_best_only=True,
                    verbose=0),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=2000,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)

# ─── 6. Save ─────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
model.save_weights(WEIGHTS_PATH)
print(f"\n[INFO] Model saved  → {MODEL_PATH}")
print(f"[INFO] Weights saved → {WEIGHTS_PATH}")

# ─── 7. Evaluate ─────────────────────────────────────────────────────────────
print("\n[INFO] Evaluating on test set …")
yhat_prob = model.predict(X_test, verbose=0)
ytrue = np.argmax(y_test,    axis=1).tolist()
yhat  = np.argmax(yhat_prob, axis=1).tolist()

cm  = multilabel_confusion_matrix(ytrue, yhat)
acc = accuracy_score(ytrue, yhat)

print("\nConfusion matrix (one per class):")
for i, action in enumerate(actions):
    print(f"  {action}: TN={cm[i,0,0]}  FP={cm[i,0,1]}  "
          f"FN={cm[i,1,0]}  TP={cm[i,1,1]}")

print(f"\n✅ Test accuracy: {acc * 100:.1f}%")
print("\n[INFO] Done! Now run detect.py to test in real-time.")
