#!/usr/bin/env python3
"""
Stage 1: raw subject JSON -> cached, phase-tagged, imputed parquet.

This is the ~15-20 minute, memory-heavy step (ward_vitals alone flattens to an
estimated 250M+ rows at full scale). Runs once; every later stage reads the cached
output instead of re-parsing — this is what makes a free-tier session crash mid-training
NOT cost you the parse step too.

Usage:
    python scripts/01_preprocess.py --config configs/default.yaml
    python scripts/01_preprocess.py --config configs/default.yaml --max-subjects-per-class 500  # smoke test
"""

import argparse
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inspire.data.loader import load_subjects, build_phase_tagged_frame
from inspire.data.preprocess import compute_population_stats, impute_patient_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-subjects-per-class", type=int, default=None,
                         help="Override config for a quick smoke test before the full run.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    max_per_class = args.max_subjects_per_class or config["data"]["max_subjects_per_class"]
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading subjects from {config['data']['subjects_dir']} "
          f"(max_per_class={max_per_class})...")
    subjects = load_subjects(config["data"]["subjects_dir"], max_per_class, config["data"]["seed"])
    print(f"Loaded {len(subjects)} subjects.")

    post_op_hours = int(config["phases"]["post_op_window"].rstrip("h")) \
        if config["phases"]["post_op_window"].endswith("h") else 72

    all_frames = []
    n_skipped = 0
    for i, (sid, sub) in enumerate(subjects.items()):
        df = build_phase_tagged_frame(sub, post_op_window_hours=post_op_hours)
        if df is None:
            n_skipped += 1
            continue
        df["subject_id"] = sid
        df["died"] = sub.died()
        all_frames.append(df)
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1}/{len(subjects)} subjects processed")

    print(f"Skipped {n_skipped} subjects (no parseable operation timestamps).")
    long_df = pd.concat(all_frames, ignore_index=True)
    print(f"Phase-tagged long-format frame: {len(long_df):,} rows.")

    if config["imputation"]["attach_missingness_mask"]:
        pop_stats = compute_population_stats(long_df)
        long_df = pd.concat(
            [impute_patient_frame(g, pop_stats) for _, g in long_df.groupby("subject_id")],
            ignore_index=True,
        )

    out_path = os.path.join(out_dir, "phase_tagged_imputed.parquet")
    long_df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    n_died = long_df.groupby("subject_id")["died"].first().sum()
    n_total = long_df["subject_id"].nunique()
    print(f"Cohort: {n_total} patients, {n_died} died — "
          f"expect ~469 at full scale, per confirmed EDA.")


if __name__ == "__main__":
    main()
