#!/usr/bin/env python3
"""
DRY-RUN ONLY — generates SYNTHETIC data matching the exact schema that
data/loader.py's build_phase_tagged_frame() produces, so the rest of the pipeline
(stages 2-4) can be run and verified end-to-end without your real INSPIRE files.

This is NOT real patient data and the "results" it produces are NOT clinically
meaningful — random noise won't predict anything. Its only purpose is to prove the
pipeline's plumbing (Dataset/DataLoader, model wiring, checkpointing, training loop,
evaluation) actually runs correctly, so integration bugs get caught here instead of
during your first real run on Kaggle/Colab.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inspire.features.organ_systems import FEATURE_MAP, ENCODER_SYSTEMS, Phase


def generate_synthetic_cohort(n_patients=400, death_rate=0.08, seed=42):
    """Builds a long-format frame with the same columns as build_phase_tagged_frame(),
    plus department/asa static columns, for n_patients synthetic patients."""
    rng = np.random.RandomState(seed)
    departments = ["CTS", "General", "Orthopaedics"]
    all_rows = []
    static_rows = []

    n_died = int(n_patients * death_rate)
    died_flags = np.array([1] * n_died + [0] * (n_patients - n_died))
    rng.shuffle(died_flags)

    for i in range(n_patients):
        subject_id = f"synthetic_{i:05d}"
        died = int(died_flags[i])
        department = rng.choice(departments)
        asa = rng.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        # sicker patients (higher ASA, died=1) get systematically worse synthetic vitals —
        # a deliberately easy synthetic signal, just so we can confirm AUPRC/AUROC move
        # off pure-random when the pipeline actually learns something.
        severity = asa * 0.3 + died * 1.5 + rng.randn() * 0.5

        static_rows.append({"subject_id": subject_id, "died": died,
                             "department": department, "asa": asa, "severity": severity})

        for system in ENCODER_SYSTEMS:
            system_features = [f for f, (s, *_r) in FEATURE_MAP.items() if s == system]
            n_readings = rng.randint(3, 15)
            for _ in range(n_readings):
                feature = rng.choice(system_features)
                _, _, phases, role = FEATURE_MAP[feature]
                phase = rng.choice([p.value for p in phases])
                value = rng.randn() + severity * 0.3  # sicker patients -> shifted values
                chart_time = int(rng.randint(0, 100000))
                all_rows.append({
                    "feature": feature, "value": value, "chart_time": chart_time,
                    "phase": phase, "system": system, "role": role.value,
                    "source_table": "synthetic", "subject_id": subject_id, "died": died,
                    "observed": 1,
                })

    long_df = pd.DataFrame(all_rows)
    static_df = pd.DataFrame(static_rows)
    return long_df, static_df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    long_df, static_df = generate_synthetic_cohort(n_patients=400, death_rate=0.08)
    long_df.to_parquet("data/processed/phase_tagged_imputed.parquet", index=False)
    static_df.to_parquet("data/processed/static_features.parquet", index=False)
    print(f"Synthetic cohort generated: {len(static_df)} patients, "
          f"{static_df['died'].sum()} died ({static_df['died'].mean()*100:.1f}%)")
    print(f"Long-format rows: {len(long_df):,}")
    print("Saved: data/processed/phase_tagged_imputed.parquet, "
          "data/processed/static_features.parquet")
