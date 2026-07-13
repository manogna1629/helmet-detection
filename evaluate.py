"""
evaluate.py
-----------
Shared evaluation utilities: collecting predictions from a tf.data test set,
computing the standard metric bundle (accuracy/precision/recall/F1/ROC-AUC/
confusion matrix/classification report), ROC curve points, and a simple
inference-latency benchmark.
"""

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)


def collect_predictions(model, test_ds):
    """
    Run model over every batch of test_ds and collect true labels,
    predicted probabilities, and the raw images (needed later for Grad-CAM).

    Returns: y_true (N,), y_prob (N,), images (N, H, W, 3)
    """
    y_true_list, y_prob_list, image_list = [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy().reshape(-1))
        image_list.append(images.numpy())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    images = np.concatenate(image_list, axis=0)
    return y_true, y_prob, images


def compute_metrics(y_true, y_prob, class_names, threshold=0.5):
    """Compute the full metric bundle at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def roc_points(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def benchmark_inference_time(model, images, n_warmup=5, n_runs=50):
    """
    Mean single-image inference time in milliseconds, plus total trainable
    + non-trainable parameter count. Uses a subset of `images` (already in
    whatever scale/dtype the model expects).
    """
    n_runs = min(n_runs, len(images))
    sample = images[:n_runs]

    for i in range(min(n_warmup, len(sample))):
        model.predict(sample[i:i + 1], verbose=0)

    start = time.perf_counter()
    for i in range(n_runs):
        model.predict(sample[i:i + 1], verbose=0)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / n_runs) * 1000
    n_params = model.count_params()

    return {"mean_ms": mean_ms, "n_params": n_params}
