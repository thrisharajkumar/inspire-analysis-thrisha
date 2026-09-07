#!/usr/bin/env python3
"""
Stage 2: diffusion-based minority augmentation (config: sampling.strategy == "diffusion").

Cached to its own parquet — a training-session crash during model training (stage 3)
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
from inspire.augmentation.diffusion import generate_all_systems
from inspire.eval.validate_synthetic import full_validation_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config["sampling"]["strategy"] != "diffusion":
        print(f"sampling.strategy is '{config['sampling']['strategy']}', not 'diffusion' — "
              f"nothing to do here. Use scripts/02b_train_smotenc.py for the SMOTENC path "
              f"(same ablation, different config).")
        return

    long_df = pd.read_parquet("data/processed/phase_tagged_imputed.parquet")
    static_df = (
        long_df[long_df["died"] == 1]
        .pivot_table(index="subject_id", columns="feature", values="value", aggfunc="first")
    )
    # TODO: join department/ASA columns onto static_df from your static-feature table —
    # not present in the long-format phase-tagged frame itself; wire this from
    # whatever produces your existing static feature set (dnn_mortality_data_real.py).
    strata = config["sampling"]["stratify_by"]

    print(f"Minority pool for diffusion training: {len(static_df)} real died patients.")
    print("Checking phase-completeness before generating peri-op features "
          "(generating synthetic peri-op data for patients who lack real peri-op "
          "coverage would compound scarcity, not solve it) — see docs/current.")
    # TODO: implement the phase-completeness check flagged in the design doc, before
    # calling generate_all_systems for systems that rely on peri-op features.

    n_majority = (long_df.groupby("subject_id")["died"].first() == 0).sum()
    target_per_stratum = {}  # TODO: compute target_minority_count per stratum from
    # config["sampling"]["target_ratio"], same logic as augmentation/smotenc.py's
    # target_minority_count, so both strategies are held to the same target ratio.

    synthetic_by_system = generate_all_systems(
        static_df, strata, categorical_cols=strata, config=config,
        n_needed_per_stratum=target_per_stratum,
    )

    os.makedirs("data/processed", exist_ok=True)
    for system_name, synth_df in synthetic_by_system.items():
        report = full_validation_report(static_df, synth_df, feature_cols=list(synth_df.columns))
        n_flagged = report["nearest_neighbor"]["n_flagged_as_memorized"]
        if n_flagged > 0:
            print(f"WARNING: {system_name}: {n_flagged} synthetic patients flagged as "
                  f"possible memorization — inspect before using.")
        out_path = f"data/processed/synthetic_{system_name}.parquet"
        synth_df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path} ({len(synth_df)} synthetic patients)")


if __name__ == "__main__":
    main()
