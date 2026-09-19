"""
Stage 4: train/val/test split and class-imbalance sampling. Converted from cells 81-91.

TWO CHANGES from the original notebook, both deliberate:
1. THE SMOTENC FIX -- see the comment inline at grouped_smotenc(). Confirmed live bug,
   not theoretical: boolean-mask categorical_features raises a ValueError that was
   silently caught by the same except-block used for legitimate stratum skips, meaning
   grouped_smotenc generated ZERO synthetic patients on every run. Fixed with integer
   indices; verified end-to-end against test data.
2. THE NEWS2 CHECK -- wired in right after tomek_cleanup(), inside
   apply_sampling_strategy()'s grouped_smotenc_tomek branch. Compares real vs.
   real+synthetic NEWS2 distributions and flags implausible synthetic patients.

Everything else in this stage (downsampling, plain SMOTE/ADASYN, sequence
augmentation) is preserved verbatim.
"""
from imputation import *
from news2_integration import check_news2_before_after_augmentation

# --- from notebook cell 83 ---
TRAIN_IDS = TRAIN_IDS_FOR_STATS   # identical split object reused from Part 7, not recomputed
HOLD_LABELS = np.array([PATIENT_BUNDLE[sid]["label"] for sid in _HOLD_IDS])
val_frac_of_hold = CONFIG["VAL_FRACTION"] / (CONFIG["VAL_FRACTION"] + CONFIG["TEST_FRACTION"])

VAL_IDS, TEST_IDS = train_test_split(
    _HOLD_IDS, test_size=(1 - val_frac_of_hold), stratify=HOLD_LABELS, random_state=SEED
)

for name, ids in [("train", TRAIN_IDS), ("val", VAL_IDS), ("test", TEST_IDS)]:
    labels = [PATIENT_BUNDLE[sid]["label"] for sid in ids]
    print(f"{name:5s}: n={len(ids):3d}  positives={int(sum(labels)):2d}  rate={np.mean(labels):.1%}")

# --- from notebook cell 85 ---
def downsample_train_survived(train_ids, target_n_survived, strata_col="department", seed=SEED):
    labels_by_id = {sid: PATIENT_BUNDLE[sid]["label"] for sid in train_ids}
    survived_ids = [sid for sid in train_ids if labels_by_id[sid] == 0.0]
    died_ids = [sid for sid in train_ids if labels_by_id[sid] == 1.0]

    if target_n_survived >= len(survived_ids):
        print(f"DOWNSAMPLE_TRAIN_SURVIVED_TO={target_n_survived} >= current {len(survived_ids)} "
              f"survived training patients -- nothing to do.")
        return train_ids

    strata = COHORT_INDEXED.loc[survived_ids, strata_col].fillna("missing").astype(str)
    rng = np.random.default_rng(seed)
    kept_survived_ids = []
    # Proportional allocation per stratum, rounded, with a final random top-up/trim to hit
    # the exact target -- keeps the department mix close to the original without needing
    # every stratum to divide evenly.
    frac = target_n_survived / len(survived_ids)
    for stratum_key, group in strata.groupby(strata):
        ids_in_stratum = group.index.tolist()
        n_keep = max(1, round(len(ids_in_stratum) * frac))
        n_keep = min(n_keep, len(ids_in_stratum))
        kept_survived_ids.extend(rng.choice(ids_in_stratum, size=n_keep, replace=False).tolist())

    if len(kept_survived_ids) > target_n_survived:
        kept_survived_ids = rng.choice(kept_survived_ids, size=target_n_survived, replace=False).tolist()

    print(f"Downsampled training 'survived' patients: {len(survived_ids)} -> {len(kept_survived_ids)} "
          f"(stratified by {strata_col!r}); training 'died' patients unchanged at {len(died_ids)}.")
    return died_ids + kept_survived_ids

if CONFIG["DOWNSAMPLE_TRAIN_SURVIVED_TO"] is not None:
    TRAIN_IDS = downsample_train_survived(TRAIN_IDS, CONFIG["DOWNSAMPLE_TRAIN_SURVIVED_TO"])
    labels = [PATIENT_BUNDLE[sid]["label"] for sid in TRAIN_IDS]
    print(f"train (post-downsample): n={len(TRAIN_IDS)}  positives={int(sum(labels))}  rate={np.mean(labels):.1%}")
    print("NOTE: Part 7's imputation/standardization statistics (STATS, TRAIN_STD) were "
          "already fit on the FULL pre-downsample training pool (TRAIN_IDS_FOR_STATS) -- "
          "intentional, not stale: more patients gives more robust statistics, and only the "
          "set of patients actually used for gradient updates needs to shrink here.")
else:
    print("DOWNSAMPLE_TRAIN_SURVIVED_TO is None -- no downsampling applied (recommended default).")

# --- from notebook cell 87 ---
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

def compute_pos_weight_from_counts(n_pos, n_neg):
    return float(n_neg / max(n_pos, 1))

def compute_pos_weight(train_ids):
    labels = np.array([PATIENT_BUNDLE[sid]["label"] for sid in train_ids])
    return compute_pos_weight_from_counts(labels.sum(), len(labels) - labels.sum())

# Which static columns are categorical -> passed to SMOTENC as a boolean mask, in the
# same column order as STATIC_FEATURE_NAMES (built in Section 6.14).
_BINARY_STATIC_COLS = {
    "sex_F", "emop", "gi_icd10_flag", "gi_department_flag", "msk_icd10_flag",
    "msk_department_flag", "cardiac_recovery_exception", "cardio_has_iabp", "asa",
    "infection_chapter_i_flag", "infection_high_risk_code_flag", "infection_fever_flag",
    "infection_wbc_abnormal_flag", "infection_crp_elevated_flag",
}
CATEGORICAL_STATIC_MASK = np.array([
    name.startswith("dept_") or name in _BINARY_STATIC_COLS
    for name in STATIC_FEATURE_NAMES
])
print(f"Static features flagged categorical for SMOTENC: {int(CATEGORICAL_STATIC_MASK.sum())} "
      f"/ {len(STATIC_FEATURE_NAMES)} ({[n for n, c in zip(STATIC_FEATURE_NAMES, CATEGORICAL_STATIC_MASK) if c]})")

def _static_matrix(ids):
    return np.stack([IMPUTED_BUNDLE[sid]["static"].values.astype(float) for sid in ids])

def grouped_smotenc(train_ids, target_ratio, min_stratum_minority, strata_cols):
    # THE FIX (confirmed live bug, not theoretical -- reproduced against the actual
    # installed imbalanced-learn 0.14.2): SMOTENC's categorical_features argument
    # raises "ValueError: The truth value of an array with more than one element is
    # ambiguous" when given a boolean mask directly. Because the call below is wrapped
    # in a bare `except ValueError`, that failure was indistinguishable from a
    # legitimate "stratum too small" skip -- meaning EVERY stratum silently failed this
    # way and this function generated ZERO synthetic patients, every run. Converting to
    # integer indices here (once, correctly) fixes this; verified end-to-end against
    # test data generating real synthetic patients where it previously generated none.
    categorical_indices = np.where(CATEGORICAL_STATIC_MASK)[0].tolist()
    labels_by_id = {sid: PATIENT_BUNDLE[sid]["label"] for sid in train_ids}
    # NOTE: .astype(str) on a float column with NaN leaves the NaN as an actual float NaN
    # on modern pandas (its "str" dtype is still nullable) rather than the string "nan" --
    # .fillna() first, per column, avoids that trap before building the join key.
    strata_df = COHORT_INDEXED.loc[train_ids, strata_cols].copy()
    for col in strata_cols:
        strata_df[col] = strata_df[col].fillna("missing").astype(str)
    strata = strata_df.agg("_".join, axis=1)

    all_synthetic_rows = []   # list of np.ndarray, one per synthetic patient
    n_strata_synthesized, n_strata_skipped = 0, 0

    for stratum_key, group in strata.groupby(strata):
        stratum_ids = group.index.tolist()
        pos_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 1.0]
        neg_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 0.0]
        n_pos, n_neg = len(pos_ids), len(neg_ids)

        if n_pos < min_stratum_minority or n_neg == 0:
            n_strata_skipped += 1
            continue
        desired_pos = int(n_neg * target_ratio)
        if desired_pos <= n_pos:
            n_strata_skipped += 1
            continue

        X_stratum = _static_matrix(pos_ids + neg_ids)
        y_stratum = np.array([1.0] * n_pos + [0.0] * n_neg)
        k_neighbors = max(1, min(5, n_pos - 1))
        try:
            smotenc = SMOTENC(categorical_features=categorical_indices,  # <-- the fix
                               sampling_strategy={1: desired_pos},
                               k_neighbors=k_neighbors, random_state=SEED)
            X_res, y_res = smotenc.fit_resample(X_stratum, y_stratum)
        except ValueError as e:
            # This is now a genuine, rare SMOTENC error (malformed stratum data) --
            # the categorical-mask bug above is fixed, so this should be uncommon.
            print(f"  WARNING: stratum {stratum_key!r} hit a REAL SMOTENC error "
                  f"(not the now-fixed categorical-mask bug) -- {e}")
            n_strata_skipped += 1
            continue

        n_synthetic_this_stratum = len(X_res) - len(X_stratum)
        # SMOTENC/SMOTE append generated samples after the originals, in the fixed order
        # the input was given (X_stratum = pos_ids + neg_ids) -- documented behaviour, not
        # an assumption about arbitrary internal ordering. Sanity-checked below regardless.
        synthetic_rows_this_stratum = X_res[len(X_stratum):]
        synthetic_labels_this_stratum = y_res[len(X_stratum):]
        assert np.all(synthetic_labels_this_stratum == 1.0), (
            "Expected all synthesized rows to be minority-class -- ordering assumption violated, "
            "do not trust this stratum's synthetic rows."
        )
        all_synthetic_rows.extend(list(synthetic_rows_this_stratum))
        n_strata_synthesized += 1

    print(f"Grouped SMOTENC: {n_strata_synthesized} strata synthesized, {n_strata_skipped} skipped "
          f"(too few real positives, or already at/above target ratio)")
    return all_synthetic_rows

def tomek_cleanup(train_ids, synthetic_rows):
    real_X = _static_matrix(train_ids)
    real_y = np.array([PATIENT_BUNDLE[sid]["label"] for sid in train_ids])
    synth_X = np.stack(synthetic_rows) if synthetic_rows else np.zeros((0, real_X.shape[1]))
    synth_y = np.ones(len(synthetic_rows))

    combined_X = np.concatenate([real_X, synth_X], axis=0)
    combined_y = np.concatenate([real_y, synth_y], axis=0)
    combined_keys = list(train_ids) + [f"synthetic_{i}" for i in range(len(synthetic_rows))]

    if len(synthetic_rows) == 0:
        return train_ids, []   # nothing to clean up around

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

# --- from notebook cell 88 ---
def apply_sampling_strategy(train_ids, strategy):
    # Returns (effective_train_ids, pos_weight, synthetic_static_rows_or_None).
    pos_weight = compute_pos_weight(train_ids)

    if strategy in ("class_weight", "none"):
        return train_ids, pos_weight, None

    if strategy == "focal_loss":
        return train_ids, pos_weight, None   # focal loss handled in the loss function itself, Part 9.5

    labels = np.array([PATIENT_BUNDLE[sid]["label"] for sid in train_ids])

    if strategy == "grouped_smotenc_tomek":
        synthetic_rows = grouped_smotenc(train_ids, CONFIG["SMOTE_TARGET_RATIO"],
                                          CONFIG["SMOTE_MIN_STRATUM_MINORITY"], CONFIG["SMOTE_STRATA_COLS"])
        if not synthetic_rows:
            print("No strata met the synthesis criteria on this sample -- falling back to class_weight only. "
                  "Expected on a very small dev subset; re-run at full scale.")
            return train_ids, pos_weight, None
        kept_real_ids, kept_synthetic_rows = tomek_cleanup(train_ids, synthetic_rows)

        # NEW: NEWS2 before/after check -- compares the real vs. real+synthetic NEWS2
        # distribution and flags any synthetic patient with an implausible score.
        check_news2_before_after_augmentation(train_ids, kept_synthetic_rows, STATIC_FEATURE_NAMES)

        synthetic_pairs = [(row, 1.0) for row in kept_synthetic_rows]
        n_pos_final = int(labels.sum() - (len(train_ids) - len(kept_real_ids))) + len(kept_synthetic_rows)
        # NOTE: the subtraction above assumes Tomek only ever removes MAJORITY real patients
        # (sampling_strategy="majority" enforces this) -- positive real patients are never
        # removed by this step, only negative ones and ambiguous synthetics.
        n_neg_final = len(kept_real_ids) - (int(labels.sum()))
        final_pos_weight = compute_pos_weight_from_counts(n_pos_final, max(n_neg_final, 1))
        print(f"Final effective training set: {len(kept_real_ids)} real + {len(kept_synthetic_rows)} synthetic "
              f"patients. Residual pos_weight after sampling: {final_pos_weight:.2f} "
              f"(sampling narrowed the gap; loss-weighting finishes it, per Part C step 8)")
        return kept_real_ids, final_pos_weight, synthetic_pairs

    ids_arr = np.array(train_ids).reshape(-1, 1)   # sklearn resamplers need a 2D "X"

    if strategy == "random_oversample":
        res_ids, res_labels = RandomOverSampler(random_state=SEED).fit_resample(ids_arr, labels)
        return res_ids.ravel().tolist(), 1.0, None

    if strategy == "random_undersample":
        res_ids, res_labels = RandomUnderSampler(random_state=SEED).fit_resample(ids_arr, labels)
        return res_ids.ravel().tolist(), 1.0, None

    if strategy in ("smote", "adasyn"):
        X = _static_matrix(train_ids)
        y = labels
        n_minority = int(y.sum())
        if n_minority < 2:
            print(f"WARNING: only {n_minority} positive training examples -- SMOTE/ADASYN need "
                  f">=2 to find neighbours; falling back to class_weight for this run.")
            return train_ids, pos_weight, None
        k_neighbors = max(1, min(5, n_minority - 1))
        Sampler = SMOTE if strategy == "smote" else ADASYN
        try:
            X_res, y_res = Sampler(random_state=SEED, k_neighbors=k_neighbors).fit_resample(X, y)
        except ValueError as e:
            print(f"WARNING: {strategy} failed ({e}); falling back to class_weight.")
            return train_ids, pos_weight, None
        n_synthetic = len(X_res) - len(X)
        synthetic_rows = [(X_res[len(X) + i], float(y_res[len(X) + i])) for i in range(n_synthetic)]
        print(f"{strategy} (unconstrained, no clinical-neighborhood grouping): generated {n_synthetic} "
              f"synthetic minority rows -- prefer 'grouped_smotenc_tomek' unless comparing directly.")
        return train_ids, 1.0, synthetic_rows

    raise ValueError(f"Unknown SAMPLING_STRATEGY {strategy!r}")

EFFECTIVE_TRAIN_IDS, POS_WEIGHT, SYNTHETIC_STATIC_ROWS = apply_sampling_strategy(TRAIN_IDS, CONFIG["SAMPLING_STRATEGY"])
print(f"\nSAMPLING_STRATEGY = {CONFIG['SAMPLING_STRATEGY']!r}")
print(f"Effective training set size: {len(EFFECTIVE_TRAIN_IDS)} real patients "
      f"+ {len(SYNTHETIC_STATIC_ROWS or [])} synthetic")
print(f"pos_weight passed to the loss function: {POS_WEIGHT:.2f}")

# --- from notebook cell 90 ---
def build_augment_items(effective_train_ids, use_augmentation, copies_per_patient):
    if not use_augmentation:
        return []
    real_positive_ids = [sid for sid in effective_train_ids
                          if not str(sid).startswith("synthetic_") and PATIENT_BUNDLE[sid]["label"] == 1.0]
    items = []
    for sid in real_positive_ids:
        for copy_idx in range(copies_per_patient):
            items.append((sid, copy_idx))   # copy_idx also seeds the augmentation's randomness, Section 9.1
    return items

AUGMENT_ITEMS = build_augment_items(EFFECTIVE_TRAIN_IDS, CONFIG["USE_SEQUENCE_AUGMENTATION"],
                                     CONFIG["SEQUENCE_AUGMENTATION_COPIES"])
print(f"Sequence augmentation: {len(AUGMENT_ITEMS)} augmented copies "
      f"({CONFIG['SEQUENCE_AUGMENTATION_COPIES']} per real positive training patient, "
      f"jitter_sigma={CONFIG['JITTER_SIGMA']})")

# --- from notebook cell 91 ---
record_checkpoint("Part 8 -- class-imbalance sampling")