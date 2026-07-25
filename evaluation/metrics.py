"""Evaluation metrics for spatial spike sorting (Akhoundi 2025 Eq. 10)."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_spikes_to_gt(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    tolerance_samples: int = 30,
) -> tuple:
    """Greedy time-based matching of detected spikes to ground truth.

    Each GT spike is matched to at most one predicted spike within tolerance.

    Returns:
        matched_pred: (n_matched,) indices into pred_times.
        matched_gt: (n_matched,) indices into gt_times.
    """
    if len(pred_times) == 0 or len(gt_times) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Sort both
    pred_order = np.argsort(pred_times)
    gt_order = np.argsort(gt_times)
    pred_sorted = pred_times[pred_order]
    gt_sorted = gt_times[gt_order]

    matched_pred = []
    matched_gt = []
    pred_used = np.zeros(len(pred_sorted), dtype=bool)

    j_start = 0
    for gi in range(len(gt_sorted)):
        gt_t = gt_sorted[gi]
        # Advance j_start to first pred within range
        while j_start < len(pred_sorted) and pred_sorted[j_start] < gt_t - tolerance_samples:
            j_start += 1

        # Find closest unused pred within tolerance
        best_j = -1
        best_dist = tolerance_samples + 1
        for j in range(j_start, len(pred_sorted)):
            if pred_sorted[j] > gt_t + tolerance_samples:
                break
            if pred_used[j]:
                continue
            dist = abs(pred_sorted[j] - gt_t)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            pred_used[best_j] = True
            matched_pred.append(pred_order[best_j])
            matched_gt.append(gt_order[gi])

    return np.array(matched_pred, dtype=np.int64), np.array(matched_gt, dtype=np.int64)


def clustering_accuracy(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    tolerance_samples: int = 30,
) -> dict:
    """Compute clustering accuracy per Akhoundi 2025 Eq. 10.

    1. Match predicted spikes to GT by time.
    2. For matched spikes, find best label mapping (Hungarian algorithm).
    3. Accuracy = N_correctly_clustered / N_GT.

    Returns:
        dict with keys: accuracy, n_matched, n_gt, n_pred, label_mapping
    """
    matched_pred, matched_gt = match_spikes_to_gt(pred_times, gt_times, tolerance_samples)

    n_gt = len(gt_times)
    n_pred = len(pred_times)
    n_matched = len(matched_pred)

    if n_matched == 0:
        return {
            "accuracy": 0.0,
            "n_matched": 0,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "label_mapping": {},
        }

    pred_matched_labels = pred_labels[matched_pred]
    gt_matched_labels = gt_labels[matched_gt]

    # Build cost matrix for optimal label assignment
    pred_unique = np.unique(pred_matched_labels)
    gt_unique = np.unique(gt_matched_labels)

    n_pred_classes = len(pred_unique)
    n_gt_classes = len(gt_unique)
    cost_matrix = np.zeros((n_pred_classes, n_gt_classes), dtype=np.int64)

    for i, pl in enumerate(pred_unique):
        for j, gl in enumerate(gt_unique):
            cost_matrix[i, j] = np.sum(
                (pred_matched_labels == pl) & (gt_matched_labels == gl)
            )

    # Hungarian algorithm (maximize matches = minimize negative)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    n_correct = cost_matrix[row_ind, col_ind].sum()

    label_mapping = {int(pred_unique[r]): int(gt_unique[c]) for r, c in zip(row_ind, col_ind)}

    accuracy = n_correct / n_gt if n_gt > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "n_matched": int(n_matched),
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "n_correct": int(n_correct),
        "label_mapping": label_mapping,
    }


def detection_metrics(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    tolerance_samples: int = 30,
) -> dict:
    """Detection-level precision, recall, F1."""
    matched_pred, matched_gt = match_spikes_to_gt(pred_times, gt_times, tolerance_samples)
    tp = len(matched_pred)
    fp = len(pred_times) - tp
    fn = len(gt_times) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
