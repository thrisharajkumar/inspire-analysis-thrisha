"""
Core evaluation metrics — AUPRC as the primary metric (not AUROC alone, which can look
deceptively good at this class rarity), plus best-F1 threshold tuning instead of a
fixed, meaningless-at-this-prevalence 0.5. Matches the reporting convention already
established (see docs/current §7.3 for the confirmed baseline numbers this should be
directly comparable against).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, f1_score,
)


def compute_core_metrics(y_true, y_prob):
    return {
        "auprc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def best_f1_threshold(y_true, y_prob):
    """Picks the threshold maximizing F1 from the precision-recall curve, rather than
    a fixed 0.5 — meaningless at ~4% test-set prevalence."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx]


def evaluate_at_best_threshold(y_true, y_prob):
    threshold, precision, recall = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y_true, y_pred),
        "n_flagged": int(y_pred.sum()),
        "n_true_positives": int(((y_pred == 1) & (y_true == 1)).sum()),
        "n_real_positives": int(y_true.sum()),
    }
