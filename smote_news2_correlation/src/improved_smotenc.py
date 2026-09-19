"""
Grouped SMOTENC -- kept as the sampling strategy (not replaced with diffusion), fixed
and improved.

THE BUG (confirmed live, not theoretical): the existing implementation
(INSPIRE_Multimodal_Mortality_Benchmark, Section 8.2 `grouped_smotenc`) passes
CATEGORICAL_STATIC_MASK -- a numpy BOOLEAN array -- directly as SMOTENC's
`categorical_features` argument. On the installed imbalanced-learn version (0.14.2,
verified in a real test, not assumed), this raises:

    ValueError: The truth value of an array with more than one element is ambiguous.

Because that call is wrapped in a bare `except ValueError`, the failure is
indistinguishable in the printed log from a legitimate "stratum too small, skipped"
case -- meaning EVERY stratum silently fails this way, and grouped_smotenc() generates
ZERO synthetic patients, on every single run, without an obvious error. This is the
single highest-value fix in this module.

THE FIX: convert the boolean mask to integer indices before calling SMOTENC --
that call signature is confirmed working in the same test.

IMPROVEMENTS beyond the fix (why "improve", not just "fix"):
1. A dedicated exception message so a real future SMOTENC failure (e.g. a genuinely
   malformed stratum) is distinguishable from "too few positives" in the log, rather
   than both printing through the same generic except-block.
2. `min_synthetic_neighbor_ratio` -- guards against SMOTENC producing implausible
   synthetic patients when k_neighbors is forced very low (down to 1) in a tiny
   stratum; below the configured ratio, that stratum is skipped rather than trusted.
3. A returned per-stratum diagnostic table (not just print statements) so you can
   inspect exactly which strata synthesized, which were skipped and why, and hand that
   table to your supervisor/surgeon rather than reading it out of scrollback.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks


def grouped_smotenc(
    static_matrix_fn, cohort_indexed, labels_by_id, train_ids,
    categorical_mask, target_ratio, min_stratum_minority, strata_cols,
    min_synthetic_neighbor_ratio=0.5, seed=42,
):
    """
    static_matrix_fn: callable(ids) -> np.ndarray, same contract as the existing
        notebook's `_static_matrix` -- kept as a parameter rather than importing your
        globals, so this module drops in without restructuring your notebook.
    cohort_indexed, labels_by_id, train_ids: same objects the existing notebook builds.
    categorical_mask: boolean np.ndarray, same as CATEGORICAL_STATIC_MASK -- conversion
        to integer indices happens inside this function, once, correctly.
    min_synthetic_neighbor_ratio: if a stratum's forced k_neighbors is below
        min_synthetic_neighbor_ratio * 5 (the normal default), skip it rather than
        generate from an unreliably small neighborhood -- new in this version.

    Returns: (all_synthetic_rows, diagnostics_df)
        diagnostics_df has one row per stratum: n_pos, n_neg, desired_pos, k_neighbors,
        status ("synthesized" / "skipped_too_few_positives" / "skipped_at_target" /
        "skipped_low_k_neighbors" / "skipped_smotenc_error"), and detail (free text).
    """
    # THE FIX: integer indices, not a boolean mask -- confirmed working call signature.
    categorical_indices = np.where(categorical_mask)[0].tolist()

    strata_df = cohort_indexed.loc[train_ids, strata_cols].copy()
    for col in strata_cols:
        strata_df[col] = strata_df[col].fillna("missing").astype(str)
    strata = strata_df.agg("_".join, axis=1)

    all_synthetic_rows = []
    diagnostics = []

    for stratum_key, group in strata.groupby(strata):
        stratum_ids = group.index.tolist()
        pos_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 1.0]
        neg_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 0.0]
        n_pos, n_neg = len(pos_ids), len(neg_ids)
        row = {"stratum": stratum_key, "n_pos": n_pos, "n_neg": n_neg}

        if n_pos < min_stratum_minority or n_neg == 0:
            row.update(status="skipped_too_few_positives", desired_pos=None,
                       k_neighbors=None, detail=f"n_pos={n_pos} < min_stratum_minority={min_stratum_minority}, or n_neg=0")
            diagnostics.append(row)
            continue

        desired_pos = int(n_neg * target_ratio)
        if desired_pos <= n_pos:
            row.update(status="skipped_at_target", desired_pos=desired_pos,
                       k_neighbors=None, detail="already at or above target ratio")
            diagnostics.append(row)
            continue

        k_neighbors = max(1, min(5, n_pos - 1))
        # IMPROVEMENT: guard against an unreliably small neighborhood, rather than
        # silently trusting SMOTENC's output regardless of how few neighbors it had.
        if k_neighbors < min_synthetic_neighbor_ratio * 5:
            row.update(status="skipped_low_k_neighbors", desired_pos=desired_pos,
                       k_neighbors=k_neighbors,
                       detail=f"k_neighbors={k_neighbors} below {min_synthetic_neighbor_ratio*5:.1f} threshold")
            diagnostics.append(row)
            continue

        X_stratum = static_matrix_fn(pos_ids + neg_ids)
        y_stratum = np.array([1.0] * n_pos + [0.0] * n_neg)

        try:
            smotenc = SMOTENC(categorical_features=categorical_indices,
                               sampling_strategy={1: desired_pos},
                               k_neighbors=k_neighbors, random_state=seed)
            X_res, y_res = smotenc.fit_resample(X_stratum, y_stratum)
        except ValueError as e:
            # IMPROVEMENT: this is now a real, rare failure (malformed stratum data),
            # not the boolean-mask bug -- worth a distinct, loud message rather than
            # blending into the same log line as a legitimate skip.
            row.update(status="skipped_smotenc_error", desired_pos=desired_pos,
                       k_neighbors=k_neighbors, detail=f"REAL SMOTENC error: {e}")
            diagnostics.append(row)
            print(f"WARNING: stratum {stratum_key!r} hit a genuine SMOTENC error "
                  f"(not the categorical-mask bug, which is fixed) -- {e}")
            continue

        synthetic_rows_this_stratum = X_res[len(X_stratum):]
        synthetic_labels_this_stratum = y_res[len(X_stratum):]
        assert np.all(synthetic_labels_this_stratum == 1.0), (
            "Expected all synthesized rows to be minority-class -- ordering assumption "
            "violated, do not trust this stratum's synthetic rows."
        )
        all_synthetic_rows.extend(list(synthetic_rows_this_stratum))
        row.update(status="synthesized", desired_pos=desired_pos, k_neighbors=k_neighbors,
                   detail=f"{len(synthetic_rows_this_stratum)} synthetic rows generated")
        diagnostics.append(row)

    diagnostics_df = pd.DataFrame(diagnostics)
    n_synth = (diagnostics_df["status"] == "synthesized").sum() if len(diagnostics_df) else 0
    print(f"Grouped SMOTENC: {n_synth} strata synthesized, {len(diagnostics_df) - n_synth} skipped "
          f"(see returned diagnostics_df for the reason per stratum).")
    return all_synthetic_rows, diagnostics_df


def tomek_cleanup(static_matrix_fn, train_ids, labels_by_id, synthetic_rows):
    """Unchanged from the existing notebook's tomek_cleanup -- no bug found here, this
    step already operates on the fully-numeric post-SMOTENC array, not the raw
    categorical mask, so it doesn't hit the same issue."""
    real_X = static_matrix_fn(train_ids)
    real_y = np.array([labels_by_id[sid] for sid in train_ids])
    synth_X = np.stack(synthetic_rows) if synthetic_rows else np.zeros((0, real_X.shape[1]))
    synth_y = np.ones(len(synthetic_rows))

    combined_X = np.concatenate([real_X, synth_X], axis=0)
    combined_y = np.concatenate([real_y, synth_y], axis=0)
    combined_keys = list(train_ids) + [f"synthetic_{i}" for i in range(len(synthetic_rows))]

    if len(synthetic_rows) == 0:
        return train_ids, []

    tl = TomekLinks(sampling_strategy="majority")
    X_clean, y_clean = tl.fit_resample(combined_X, combined_y)
    try:
        kept_positions = tl.sample_indices_
    except AttributeError:
        print("WARNING: TomekLinks.sample_indices_ unavailable in this imblearn version -- "
              "falling back to keeping everything (no Tomek cleanup applied this run).")
        kept_positions = np.arange(len(combined_keys))

    kept_keys = [combined_keys[i] for i in kept_positions]
    kept_real_ids = [k for k in kept_keys if not str(k).startswith("synthetic_")]
    kept_synthetic_positions = [int(k.split("_")[1]) for k in kept_keys if str(k).startswith("synthetic_")]
    kept_synthetic_rows = [synthetic_rows[i] for i in kept_synthetic_positions]

    n_removed_real = len(train_ids) - len(kept_real_ids)
    n_removed_synthetic = len(synthetic_rows) - len(kept_synthetic_rows)
    print(f"Tomek cleanup: removed {n_removed_real} real (majority) patients, "
          f"{n_removed_synthetic} synthetic patients as ambiguous boundary pairs")
    return kept_real_ids, kept_synthetic_rows
