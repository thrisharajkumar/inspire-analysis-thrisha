"""
Validation checklist for synthetic (diffusion-generated) minority patients, run BEFORE
they're allowed into training. Non-negotiable given the small real-N (469 deaths) —
this is what makes the diffusion swap defensible to your supervisor and in a paper,
not just "we tried a fancier method."

Covers: (1) reused NEWS2 plausibility gate, (2) per-feature KS test vs. real
distribution, (3) nearest-neighbor distance check for memorization.
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors


def ks_test_per_feature(real_df, synthetic_df, feature_cols, alpha=0.05):
    """
    Per-feature KS test, real vs. synthetic distribution. Flags any feature where the
    synthetic distribution differs significantly from the real one (p < alpha) —
    investigate before trusting that feature's synthetic values.
    """
    results = {}
    for feature in feature_cols:
        if feature not in real_df.columns or feature not in synthetic_df.columns:
            continue
        real_vals = real_df[feature].dropna().values
        synth_vals = synthetic_df[feature].dropna().values
        if len(real_vals) < 2 or len(synth_vals) < 2:
            results[feature] = {"statistic": None, "p_value": None, "flag": "insufficient_data"}
            continue
        stat, p_value = ks_2samp(real_vals, synth_vals)
        results[feature] = {
            "statistic": stat, "p_value": p_value,
            "flag": "DIFFERS_SIGNIFICANTLY" if p_value < alpha else "ok",
        }
    return results


def nearest_neighbor_check(real_df, synthetic_df, feature_cols, memorization_threshold=1e-3):
    """
    Flags synthetic patients that sit suspiciously close to a real training patient
    (memorization) rather than being genuinely novel interpolations/extrapolations.
    Returns per-synthetic-row nearest-real-neighbor distance.
    """
    real_X = real_df[feature_cols].fillna(real_df[feature_cols].median()).values
    synth_X = synthetic_df[feature_cols].fillna(synthetic_df[feature_cols].median()).values

    nn_model = NearestNeighbors(n_neighbors=1).fit(real_X)
    distances, _ = nn_model.kneighbors(synth_X)

    flagged = distances.flatten() < memorization_threshold
    return {
        "distances": distances.flatten(),
        "n_flagged_as_memorized": int(flagged.sum()),
        "pct_flagged": float(flagged.mean() * 100),
    }


def news2_plausibility_gate(synthetic_df, news2_fn, max_implausible_score=None):
    """
    Reuses the existing NEWS2 plausibility gate (already applied to other
    synthetic/augmented patients in the pipeline — see archived
    Data_Imbalance_and_Imputation_Reference.md) on diffusion output specifically.

    news2_fn: the existing NEWS2-computation function from the repo — pass it in rather
    than reimplementing it here, so there's exactly one NEWS2 implementation.
    """
    scores = synthetic_df.apply(news2_fn, axis=1)
    if max_implausible_score is not None:
        implausible = scores > max_implausible_score
        return {
            "scores": scores,
            "n_flagged_implausible": int(implausible.sum()),
            "pct_flagged": float(implausible.mean() * 100),
        }
    return {"scores": scores}


def full_validation_report(real_df, synthetic_df, feature_cols, news2_fn=None):
    """Runs all three checks and returns one combined report — call this once per
    organ system's synthetic output before it's allowed into training."""
    report = {
        "ks_test": ks_test_per_feature(real_df, synthetic_df, feature_cols),
        "nearest_neighbor": nearest_neighbor_check(real_df, synthetic_df, feature_cols),
    }
    if news2_fn is not None:
        report["news2_plausibility"] = news2_plausibility_gate(synthetic_df, news2_fn)
    return report
