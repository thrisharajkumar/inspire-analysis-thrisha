"""
Missingness masking and imputation.

Treats missingness as MAR (depends on observed phase — e.g. urine output is 0% pre-op
because catheters aren't in yet) with a plausible added MNAR layer (a clinician not
ordering a discretionary test can itself be a low-risk signal). Every imputation method
below is paired with an explicit missingness mask feature, per that reasoning —
see docs/current/PACO_Net_Latest_Work_and_Results.md and the archived
Data_Imbalance_and_Imputation_Reference.md for the full method comparison.
"""

import numpy as np
import pandas as pd


def add_missingness_mask(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Adds a boolean 'observed' column: 1 = real reading, 0 = will be imputed."""
    df = df.copy()
    df["observed"] = df[value_col].notna().astype(int)
    return df


def impute_forward_fill_then_median(patient_series: pd.Series, population_median: float) -> pd.Series:
    """
    Default imputation decision tree (config: imputation.strategy ==
    "forward_fill_then_median"): forward-fill within-patient if a prior value exists,
    else use the population median. Always paired with the missingness mask above.
    """
    filled = patient_series.ffill()
    filled = filled.fillna(population_median)
    return filled


def compute_population_stats(long_df: pd.DataFrame, value_col: str = "value") -> dict:
    """
    Per-feature population median, computed on the TRAINING set only (call this after
    the train/test split, on train rows, and reuse the same stats at inference — never
    recompute on test data, which would leak information).
    """
    return long_df.groupby("feature", observed=True)[value_col].median().to_dict()


def impute_patient_frame(patient_df: pd.DataFrame, population_stats: dict,
                          value_col: str = "value") -> pd.DataFrame:
    """
    Applies add_missingness_mask + impute_forward_fill_then_median per feature, for one
    patient's long-format frame (as produced by data.loader.build_phase_tagged_frame).
    """
    patient_df = add_missingness_mask(patient_df, value_col)
    out_frames = []
    for feature, group in patient_df.groupby("feature", observed=True):
        group = group.sort_values("chart_time").copy()
        median = population_stats.get(feature, np.nan)
        group[value_col] = impute_forward_fill_then_median(group[value_col], median)
        out_frames.append(group)
    if not out_frames:
        return patient_df
    return pd.concat(out_frames, ignore_index=True)
