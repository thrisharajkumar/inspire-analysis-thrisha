"""
Stage 1 of two-stage synthetic patient generation: decide WHICH features a synthetic
patient has observed, before generating their VALUES (diffusion.py does stage 2).

Without this, a diffusion model generates a full panel of labs for every synthetic
patient — but a real ASA-2 elective patient would plausibly be missing troponin, so a
"complete" synthetic patient is itself a plausibility red flag, not a good sign.
Simple conditional-Bernoulli-per-feature is enough here; this does not need to be a
diffusion model itself.
"""

import numpy as np
import pandas as pd


def fit_missingness_rates(long_df: pd.DataFrame, stratify_cols=("department", "asa")) -> pd.DataFrame:
    """
    Computes each feature's real observation rate per stratum, from the training set.
    Returns a DataFrame indexed by (stratify_cols..., feature) with column 'observed_rate'.
    """
    rates = (
        long_df.groupby(list(stratify_cols) + ["feature"], observed=True)["observed"]
        .mean()
        .rename("observed_rate")
        .reset_index()
    )
    return rates


def sample_missingness_mask(rates: pd.DataFrame, stratum_key: dict, feature_list: list,
                             n_synthetic: int, random_state=42) -> pd.DataFrame:
    """
    Samples a boolean observed/missing mask for n_synthetic patients in one stratum,
    one independent Bernoulli draw per feature at that feature's real observed_rate.

    Returns: DataFrame, shape (n_synthetic, len(feature_list)), boolean.
    """
    rng = np.random.RandomState(random_state)
    mask = {}
    stratum_rates = rates
    for col, val in stratum_key.items():
        stratum_rates = stratum_rates[stratum_rates[col] == val]

    for feature in feature_list:
        row = stratum_rates[stratum_rates["feature"] == feature]
        rate = float(row["observed_rate"].iloc[0]) if len(row) else 0.0
        mask[feature] = rng.binomial(1, rate, size=n_synthetic).astype(bool)

    return pd.DataFrame(mask)
