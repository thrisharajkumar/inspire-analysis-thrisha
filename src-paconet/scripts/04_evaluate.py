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

    # TODO for a REAL run: load your actual held-out test set. For this dry run, uses
    # a held-out slice of the synthetic dry-run cohort.
    import pandas as pd
    from torch.utils.data import DataLoader
    from inspire.data.dataset import INSPIREDataset, collate_batch
    from inspire.training.checkpoint import load_checkpoint
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from importlib import import_module
    train_module = import_module("03_train_model")

    long_df = pd.read_parquet("data/processed/phase_tagged_imputed.parquet")
    static_df = pd.read_parquet("data/processed/static_features.parquet")
    test_static_df = static_df.sample(frac=0.25, random_state=1)  # simple holdout for the dry run
    dataset = INSPIREDataset(long_df, test_static_df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_module.PACONetV2(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())  # only needed to satisfy load_checkpoint's signature
    load_checkpoint(args.checkpoint, model, optimizer, device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            hazard_logits, _attn, _contrib = model(batch)
            risk_logit = hazard_logits.mean(dim=1)
            all_probs.append(torch.sigmoid(risk_logit).numpy())
            all_labels.append(batch["label"].numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    print(f"Held-out dry-run set: {len(y_true)} patients, {int(y_true.sum())} 'died' "
          f"(synthetic — not a real clinical result).")

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
