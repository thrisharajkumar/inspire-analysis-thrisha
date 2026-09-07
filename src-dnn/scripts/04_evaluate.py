#!/usr/bin/env python3
"""
Stage 4: evaluate a trained model against the established metrics — AUPRC-primary,
best-F1 threshold, directly comparable to the confirmed baseline numbers in
docs/current/PACO_Net_Latest_Work_and_Results.md §7.3 (AUPRC 0.658, AUROC 0.967 on the
10,942-patient downsampled cohort).

Usage:
    python scripts/04_evaluate.py --config configs/default.yaml --checkpoint data/checkpoints/latest.pt
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inspire.eval.metrics import compute_core_metrics, evaluate_at_best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)  # noqa: F841 (kept for parity with other scripts / future use)

    # TODO: load the trained model from args.checkpoint and run inference on the held-out
    # test set once the DataLoader (flagged in scripts/03_train_model.py) is wired up.
    # This script's metric-computation and reporting shape is ready to use as-is.
    y_true = None
    y_prob = None
    if y_true is None:
        print("Model inference not yet wired up — see the TODO above.")
        return

    core = compute_core_metrics(y_true, y_prob)
    threshold_eval = evaluate_at_best_threshold(y_true, y_prob)

    print(f"AUPRC: {core['auprc']:.3f}  (baseline: 0.658)")
    print(f"AUROC: {core['auroc']:.3f}  (baseline: 0.967)")
    print(f"Brier: {core['brier']:.3f}  (baseline: 0.097)")
    print(f"At best-F1 threshold ({threshold_eval['threshold']:.3f}): "
          f"{threshold_eval['n_true_positives']}/{threshold_eval['n_real_positives']} "
          f"deaths caught, precision={threshold_eval['precision']:.2f}")


if __name__ == "__main__":
    main()
