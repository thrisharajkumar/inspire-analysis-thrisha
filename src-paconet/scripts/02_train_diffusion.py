#!/usr/bin/env python3
"""
Stage 2: diffusion-based minority augmentation (config: sampling.strategy == "diffusion").

Cached to its own parquet -- a training-session crash during model training (stage 3)
shouldn't cost you the diffusion generation too, since it's expensive and stochastic
(you don't want a different random synthetic cohort every time stage 3 is retried).

Usage:
    python scripts/02_train_diffusion.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inspire.augmentation.diffusion import generate_all_systems, synthcity_available
from inspire.eval.validate_synthetic import full_validation_report


def check_phase_completeness(long_df, minority_subject_ids, phase="peri"):
    """
    Real implementation of the phase-completeness check flagged in the design doc:
    generating synthetic peri-op data for patients who lack real peri-op coverage would
    compound scarcity, not solve it. Returns the fraction of minority patients that have
    at least one real reading in the given phase.
    """
    minority_long = long_df[long_df["subject_id"].isin(minority_subject_ids)]
    patients_with_phase = minority_long[minority_long["phase"] == phase]["subject_id"].nunique()
    coverage = patients_with_phase / max(len(minority_subject_ids), 1)
    print(f"Phase completeness check ('{phase}'): {patients_with_phase}/{len(minority_subject_ids)} "
          f"minority patients ({coverage:.1%}) have at least one real '{phase}'-phase reading.")
    if coverage < 0.5:
        print(f"WARNING: fewer than half of minority patients have real '{phase}' coverage -- "
              f"diffusion-generating '{phase}'-phase features would mostly be extrapolating "
              f"from a small subset. Consider excluding peri-op-heavy systems from diffusion "
              f"for this run rather than trusting their synthetic output blindly.")
    return coverage


def compute_target_per_stratum(static_df, strata_cols, target_ratio):
    """
    Real implementation (previously a TODO): computes how many synthetic minority
    patients each (department, ASA) stratum needs to reach the configured
    positive:negative target ratio -- the same logic augmentation/smotenc.py uses, so
    both strategies are held to the same target and are a fair ablation against each other.
    """
    targets = {}
    for stratum_key, stratum_df in static_df.groupby(strata_cols, observed=True):
        n_pos = int((stratum_df["died"] == 1).sum())
        n_neg = int((stratum_df["died"] == 0).sum())
        desired_pos = int(n_neg * target_ratio)
        targets[stratum_key] = max(0, desired_pos - n_pos)
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config["sampling"]["strategy"] != "diffusion":
        print(f"sampling.strategy is '{config['sampling']['strategy']}', not 'diffusion' -- "
              f"nothing to do here.")
        return

    if not synthcity_available():
        print("Nothing generated this run -- downstream stages fall back to standard "
              "imputation (already computed in stage 1) for every system.")
        return

    long_df = pd.read_parquet("data/processed/phase_tagged_imputed.parquet")
    static_df = pd.read_parquet("data/processed/static_features.parquet")

    # THE FIX: department/ASA join, previously a TODO. static_df already has these
    # columns (it's the same table dataset.py and 03_train_model.py use) -- pivot the
    # long-format lab/vital values and join department/ASA/died from static_df, rather
    # than trying to derive them from the long-format frame, which doesn't carry them.
    pivoted = (
        long_df[long_df["died"] == 1]
        .pivot_table(index="subject_id", columns="feature", values="value", aggfunc="first")
    )
    strata_cols = config["sampling"]["stratify_by"]
    join_cols = list(dict.fromkeys(strata_cols + ["died"]))  # dedupe, preserve order
    minority_static = pivoted.join(static_df.set_index("subject_id")[join_cols], how="left")
    minority_static = minority_static.reset_index()

    print(f"Minority pool for diffusion training: {len(minority_static)} real died patients.")

    minority_ids = minority_static["subject_id"].tolist()
    check_phase_completeness(long_df, minority_ids, phase="peri")

    # THE FIX: target-ratio computation, previously an empty placeholder dict (which
    # meant generate_all_systems generated zero synthetic patients for every stratum).
    target_per_stratum = compute_target_per_stratum(
        static_df, strata_cols, config["sampling"]["target_ratio"]
    )
    print(f"Target synthetic patients per stratum: {target_per_stratum}")

    synthetic_by_system = generate_all_systems(
        minority_static, strata_cols, categorical_cols=strata_cols, config=config,
        n_needed_per_stratum=target_per_stratum,
    )

    os.makedirs("data/processed", exist_ok=True)
    if not synthetic_by_system:
        print("No synthetic patients generated this run (see warnings above) -- "
              "downstream stages fall back to standard imputation.")
        return

    for system_name, synth_df in synthetic_by_system.items():
        feature_cols = [c for c in synth_df.columns if c in minority_static.columns]
        report = full_validation_report(minority_static, synth_df, feature_cols=feature_cols)
        n_flagged = report["nearest_neighbor"]["n_flagged_as_memorized"]
        if n_flagged > 0:
            print(f"WARNING: {system_name}: {n_flagged} synthetic patients flagged as "
                  f"possible memorization -- inspect before using.")
        out_path = f"data/processed/synthetic_{system_name}.parquet"
        synth_df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path} ({len(synth_df)} synthetic patients)")


if __name__ == "__main__":
    main()
