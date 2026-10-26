"""
evaluate.py  –  Evaluate the trained model with full metrics + visual confusion matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score,
                              classification_report)
import tensorflow as tf

# ─── Settings ────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), 'MP_Data')
MODEL_PATH      = os.path.join(os.path.dirname(__file__), 'model.h5')
WEIGHTS_PATH    = os.path.join(os.path.dirname(__file__), 'model_weights.h5')

actions         = np.array(['cat', 'food', 'help'])
no_sequences    = 30
sequence_length = 30

# ─── Load data ────────────────────────────────────────────────────────────────
print("[INFO] Loading data …")
from tensorflow.keras.utils import to_categorical

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    seq_dirs = sorted([int(d) for d in os.listdir(action_path) if d.isdigit()])
    for seq in seq_dirs:
        window = []
        for frame_num in range(sequence_length):
            p = os.path.join(action_path, str(seq), f"{frame_num}.npy")
            window.append(np.load(p))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ─── Load model ───────────────────────────────────────────────────────────────
print("[INFO] Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
model.load_weights(WEIGHTS_PATH)

# ─── Predict on all data ──────────────────────────────────────────────────────
print("[INFO] Running predictions on full dataset …")
yhat_prob = model.predict(X, verbose=0)
ytrue = np.argmax(y,        axis=1)
yhat  = np.argmax(yhat_prob, axis=1)

# ─── Metrics ──────────────────────────────────────────────────────────────────
acc  = accuracy_score(ytrue, yhat)
prec = precision_score(ytrue, yhat, average='macro', zero_division=0)
rec  = recall_score(ytrue, yhat, average='macro', zero_division=0)
f1   = f1_score(ytrue, yhat, average='macro', zero_division=0)

print("\n" + "="*50)
print("       OVERALL METRICS  (full dataset)")
print("="*50)
print(f"  Accuracy  : {acc*100:6.1f}%")
print(f"  Precision : {prec*100:6.1f}%")
print(f"  Recall    : {rec*100:6.1f}%")
print(f"  F1 Score  : {f1*100:6.1f}%")
print("="*50)

print("\n  Per-class report:")
print(classification_report(ytrue, yhat, target_names=actions))

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(ytrue, yhat)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sign Language Model — Evaluation", fontsize=14, fontweight='bold')

# Left: raw counts
ax = axes[0]
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('Confusion Matrix (counts)', fontsize=12)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_xticks(range(len(actions)))
ax.set_yticks(range(len(actions)))
ax.set_xticklabels(actions, fontsize=11)
ax.set_yticklabels(actions, fontsize=11)
plt.colorbar(im, ax=ax)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=14, fontweight='bold')

# Right: normalised (%)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
ax2 = axes[1]
im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=100)
ax2.set_title('Confusion Matrix (% per true class)', fontsize=12)
ax2.set_xlabel('Predicted label')
ax2.set_ylabel('True label')
ax2.set_xticks(range(len(actions)))
ax2.set_yticks(range(len(actions)))
ax2.set_xticklabels(actions, fontsize=11)
ax2.set_yticklabels(actions, fontsize=11)
plt.colorbar(im2, ax=ax2)

thresh2 = 50
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax2.text(j, i, f"{cm_norm[i, j]:.0f}%",
                 ha='center', va='center',
                 color='white' if cm_norm[i, j] > thresh2 else 'black',
                 fontsize=13, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'evaluation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[INFO] Confusion matrix saved → {out_path}")
plt.show()
print("[INFO] Done.")
