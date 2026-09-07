"""
Grouped SMOTENC — kept as an available strategy (config: sampling.strategy == "smotenc")
even though diffusion is now the default, so the ablation comparison in
docs/current/PACO_Net_Latest_Work_and_Results.md is a straight config-flip, not a
separate code path per method.

Includes the fix for the real bug found during the 10,942-patient run: imbalanced-learn
0.14.x's SMOTENC expects `categorical_features` as INTEGER INDICES, not a boolean mask.
Passing a boolean mask silently fails on every stratum and falls back to plain
class-weighting with no error — this function passes indices explicitly to avoid that.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks


def grouped_smotenc(df, feature_cols, categorical_cols, stratify_cols,
                     target_ratio=0.10, tomek_cleanup=True, random_state=42):
    """
    Runs SMOTENC within each (department, ASA) stratum only — never blends clinically
    dissimilar patients across strata. Returns the resampled DataFrame.

    df: patient-level static feature table, one row per patient, with a 'died' column.
    categorical_cols: subset of feature_cols that are categorical (department, ASA, etc.)

    Real gotcha handled here: TomekLinks (unlike SMOTENC) requires purely numeric
    input — it errors on string categoricals ("could not convert string to float").
    SMOTENC runs on the original values (it needs real categories to interpolate
    correctly); categoricals are ordinal-encoded ONLY for the Tomek cleanup step,
    then decoded back immediately after.
    """
    # THE FIX (SMOTENC bug from the 10,942-patient run): integer indices, not a boolean mask.
    categorical_indices = [feature_cols.index(c) for c in categorical_cols]

    resampled_frames = []
    for stratum_key, stratum_df in df.groupby(stratify_cols, observed=True):
        X = stratum_df[feature_cols].values
        y = stratum_df["died"].values
        n_minority = int(y.sum())

        if n_minority < 2:
            # Too few real minority examples in this stratum for SMOTENC's k-neighbors —
            # keep the stratum as-is rather than crashing or silently skipping.
            resampled_frames.append(stratum_df)
            continue

        n_majority = int((y == 0).sum())
        target_minority_count = max(n_minority, int(n_majority * target_ratio))
        k_neighbors = max(1, min(5, n_minority - 1))

        try:
            sampler = SMOTENC(
                categorical_features=categorical_indices,  # <-- the fix
                sampling_strategy={1: target_minority_count},
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
            X_res, y_res = sampler.fit_resample(X, y)
        except ValueError as e:
            print(f"WARNING: SMOTENC failed on stratum {stratum_key} ({e}); "
                  f"keeping stratum unresampled rather than silently falling back.")
            resampled_frames.append(stratum_df)
            continue

        if tomek_cleanup and categorical_indices:
            # Encode categoricals to numeric ONLY for TomekLinks (see docstring), then
            # decode immediately after — SMOTENC output above already used real values.
            encoder = OrdinalEncoder()
            X_encoded = X_res.copy()
            X_encoded[:, categorical_indices] = encoder.fit_transform(X_res[:, categorical_indices])
            X_encoded = X_encoded.astype(float)
            X_clean_encoded, y_res = TomekLinks().fit_resample(X_encoded, y_res)
            # X_clean_encoded is float64 (all-numeric, for TomekLinks) — cast to object
            # BEFORE writing decoded strings/ints back in, or the assignment below tries
            # to cast e.g. 'CTS' into a float64 array and fails.
            X_res = X_clean_encoded.astype(object)
            X_res[:, categorical_indices] = encoder.inverse_transform(
                X_clean_encoded[:, categorical_indices]
            )
        elif tomek_cleanup:
            X_res, y_res = TomekLinks().fit_resample(X_res, y_res)

        resampled_df = pd.DataFrame(X_res, columns=feature_cols)
        resampled_df["died"] = y_res
        resampled_frames.append(resampled_df)

    return pd.concat(resampled_frames, ignore_index=True)
