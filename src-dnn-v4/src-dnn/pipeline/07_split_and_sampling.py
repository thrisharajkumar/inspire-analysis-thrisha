# %% [markdown]
# # Part 8 -- Train/val/test split and class-imbalance sampling
#
# This section implements the combined pipeline agreed on after reviewing external advice
# against this notebook's actual multi-branch architecture (full writeup:
# `Data_Imbalance_and_Imputation_Reference.md`, Part C). In order:
#
# - **8.1** stratified split (non-negotiable, unchanged from the original design)
# - **8.2** the new default strategy -- **grouped SMOTENC + Tomek-link cleanup**, applied
#   only to the static branch, only within clinical strata (department x ASA), targeting a
#   1:10 positive:negative ratio rather than full 1:1 balance
# - **8.3** sequence-branch augmentation (jitter + time-masking) for real minority training
#   patients -- the sequence branch never receives SMOTE-style synthetic data (Part C's own
#   reasoning: interpolating between two raw physiological trajectories in different
#   patients' time coordinates isn't a well-defined operation)
# - **8.4** the older, simpler strategies (plain SMOTE/ADASYN, random over/undersample,
#   class-weight-only) are kept available via `CONFIG['SAMPLING_STRATEGY']` for direct
#   comparison, but are no longer the default

# %% [markdown]
# ## 8.1 Stratified split
#
# Re-derives the same split as Part 7's `TRAIN_IDS_FOR_STATS` / `_HOLD_IDS` (same `SEED`,
# same call) and further splits the holdout into validation and test. With only 30 patients
# in this dev subset, don't over-read exact percentages here -- the code is written to hold
# correctly once the full cohort is loaded, where `VAL_FRACTION`/`TEST_FRACTION` will
# produce far more stable folds.

# %%
TRAIN_IDS = TRAIN_IDS_FOR_STATS   # identical split object reused from Part 7, not recomputed
HOLD_LABELS = np.array([PATIENT_BUNDLE[sid]["label"] for sid in _HOLD_IDS])
val_frac_of_hold = CONFIG["VAL_FRACTION"] / (CONFIG["VAL_FRACTION"] + CONFIG["TEST_FRACTION"])

VAL_IDS, TEST_IDS = train_test_split(
    _HOLD_IDS, test_size=(1 - val_frac_of_hold), stratify=HOLD_LABELS, random_state=SEED
)

for name, ids in [("train", TRAIN_IDS), ("val", VAL_IDS), ("test", TEST_IDS)]:
    labels = [PATIENT_BUNDLE[sid]["label"] for sid in ids]
    print(f"{name:5s}: n={len(ids):3d}  positives={int(sum(labels)):2d}  rate={np.mean(labels):.1%}")

# %% [markdown]
# ## 8.1b Optional: department-stratified downsample of TRAINING survived patients only
#
# **Off by default** (`CONFIG['DOWNSAMPLE_TRAIN_SURVIVED_TO'] = None`). This is a
# memory/compute lever, not a statistical recommendation — see Part 3's "note on memory
# strategy" for the full reasoning on why this should be the last resort, not the first
# move, and why it only ever touches training-split `survived` patients:
#
# - **Never touches `died` patients** — they're already the scarce class; downsampling them
#   further would be actively counterproductive.
# - **Never touches `VAL_IDS`/`TEST_IDS`** — validation and test must keep the true
#   prevalence for AUPRC/calibration to mean anything (Part C step 9).
# - **Stratified by department** (not "organ-system ICD" — see Part 3's note on why) so the
#   downsampled training pool's department mix still resembles the real cohort's, rather
#   than randomly dropping an entire department's worth of survived patients by chance.

# %%
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

# %% [markdown]
# ## 8.2 Grouped SMOTENC + Tomek-link cleanup (new default)
#
# **Why grouped:** plain SMOTE/SMOTENC would happily blend a cardiothoracic ASA-5 patient
# with an outpatient orthopedic ASA-1 patient purely because both happened to die -- that's
# not a clinically meaningful neighborhood. Splitting the minority cohort into strata by
# `(department, asa)` and running SMOTENC **within** each stratum keeps every synthetic
# patient's interpolation partners clinically comparable to each other.
#
# **Why SMOTENC, not plain SMOTE:** the static vector mixes continuous features (age, the
# 36 aggregated lab/vital statistics from Section 6.15) with one-hot department columns and
# binary flags (`sex_F`, `cardiac_recovery_exception`, the GI/MSK ICD-10 flags, ...). Plain
# SMOTE would linearly interpolate those binary/one-hot columns into meaningless fractional
# values (e.g. `dept_GS = 0.6`). SMOTENC is told exactly which columns are categorical and
# handles them by majority vote among neighbors instead of interpolation.
#
# **Why a 1:10 target, not full 1:1 balance:** at the full cohort's ~469 real deaths,
# synthesizing up to 46,900+ to match ~99,417 survivors would force >99x amplification of
# the same ~469 real points -- pure noise-filling well past where SMOTE's own literature
# says new information stops being added. 1:10 (~4,700 synthetic positives against the full
# survivor pool) is the conservative end of the two external reviews' suggested 1:10-1:4
# range.
#
# **The Tomek-link cleanup step, and how synthetic vs. real rows are tracked correctly:**
# after SMOTENC runs per stratum, a global Tomek-link pass identifies and removes majority
# (survived) points that are each a synthetic minority point's nearest opposite-class
# neighbor -- this widens the decision margin around the newly-added synthetic points. Row
# identity through this step is tracked via each sampler's own `sample_indices_` attribute
# (the array of retained input-row positions that `imbalanced-learn` documents and
# guarantees), **not** by assuming synthetic rows land at a particular position in the
# output array -- the earlier draft of this pipeline (reviewed in
# `Data_Imbalance_and_Imputation_Reference.md` Part A.1) relied on exactly that unstable
# assumption; this version doesn't.

# %%
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
    "sex_F", "emop", "gi_department_flag",
    "msk_department_flag", "cardiac_recovery_exception", "cardio_has_iabp", "asa",
    "infection_chapter_i_flag", "infection_high_risk_code_flag", "infection_fever_flag",
    "infection_wbc_abnormal_flag", "infection_crp_elevated_flag",
    "news2_high_flag", "news2_red_flag", "news2_available",   # v2 -- binary NEWS2 flags
}
CATEGORICAL_STATIC_MASK = np.array([
    name.startswith("dept_") or name in _BINARY_STATIC_COLS
    or (name.startswith("sysdx_") and name.endswith("_flag"))   # v3: per-organ-system diagnosis flags
    or name.startswith("antype_")                                # v4: anaesthesia type one-hot
    or (name.startswith("proc_") and name.endswith("_last_op"))  # v4: is the operation on this system?
    for name in STATIC_FEATURE_NAMES
])
# v2 FIX: imbalanced-learn 0.14.x rejects a boolean mask for `categorical_features`
# ("The truth value of an array ... is ambiguous"). Integer column indices are accepted by
# every imblearn version, so they are what gets passed to SMOTENC below.
CATEGORICAL_STATIC_IDX = np.where(CATEGORICAL_STATIC_MASK)[0].tolist()
print(f"Static features flagged categorical for SMOTENC: {int(CATEGORICAL_STATIC_MASK.sum())} "
      f"/ {len(STATIC_FEATURE_NAMES)} ({[n for n, c in zip(STATIC_FEATURE_NAMES, CATEGORICAL_STATIC_MASK) if c]})")

def _static_matrix(ids):
    return np.stack([IMPUTED_BUNDLE[sid]["static"].values.astype(float) for sid in ids])

def grouped_smotenc(train_ids, target_ratio, min_stratum_minority, strata_cols):
    labels_by_id = {sid: PATIENT_BUNDLE[sid]["label"] for sid in train_ids}
    # NOTE: .astype(str) on a float column with NaN leaves the NaN as an actual float NaN
    # on modern pandas (its "str" dtype is still nullable) rather than the string "nan" --
    # .fillna() first, per column, avoids that trap before building the join key.
    strata_df = COHORT_INDEXED.loc[train_ids, strata_cols].copy()
    for col in strata_cols:
        strata_df[col] = strata_df[col].fillna("missing").astype(str)
    strata = strata_df.agg("_".join, axis=1)

    all_synthetic_rows = []   # list of np.ndarray, one per synthetic patient
    all_synthetic_strata = [] # v2: which stratum each synthetic row came from (for §8.2c donors)
    n_strata_synthesized, n_strata_skipped, n_strata_errored = 0, 0, 0

    for stratum_key, group in strata.groupby(strata):
        stratum_ids = group.index.tolist()
        pos_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 1.0]
        neg_ids = [sid for sid in stratum_ids if labels_by_id[sid] == 0.0]
        n_pos, n_neg = len(pos_ids), len(neg_ids)

        if n_pos < min_stratum_minority or n_neg == 0:
            n_strata_skipped += 1
            continue
        desired_pos = int(n_neg * target_ratio)
        if CONFIG["SMOTE_MAX_AMPLIFICATION"] is not None:   # v3: cap per-stratum amplification
            desired_pos = min(desired_pos, n_pos * (1 + CONFIG["SMOTE_MAX_AMPLIFICATION"]))
        if desired_pos <= n_pos:
            n_strata_skipped += 1
            continue

        X_stratum = _static_matrix(pos_ids + neg_ids)
        y_stratum = np.array([1.0] * n_pos + [0.0] * n_neg)
        k_neighbors = max(1, min(5, n_pos - 1))
        try:
            smotenc = SMOTENC(categorical_features=CATEGORICAL_STATIC_IDX,   # v2: indices, not the mask
                               sampling_strategy={1: desired_pos},
                               k_neighbors=k_neighbors, random_state=SEED)
            X_res, y_res = smotenc.fit_resample(X_stratum, y_stratum)
        except ValueError as e:
            # v2: an ERROR is not the same as "too few positives" (handled above) -- count it
            # separately so a library/version problem can never again pass as a quiet skip.
            print(f"  stratum {stratum_key!r}: SMOTENC ERROR ({e})")
            n_strata_errored += 1
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
        all_synthetic_strata.extend([stratum_key] * len(synthetic_rows_this_stratum))
        n_strata_synthesized += 1

    print(f"Grouped SMOTENC: {n_strata_synthesized} strata synthesized "
          f"({len(all_synthetic_rows)} synthetic patients), {n_strata_skipped} skipped "
          f"(too few real positives, or already at/above target ratio), "
          f"{n_strata_errored} ERRORED")
    if n_strata_errored:
        msg = (f"{n_strata_errored} strata raised an error inside SMOTENC -- this is a code/library "
               f"problem, not a data-size skip. Fix it before trusting this run's results.")
        if CONFIG["STRICT_SAMPLING"]:
            raise RuntimeError(msg + " (set CONFIG['STRICT_SAMPLING']=False to continue anyway)")
        print("WARNING: " + msg)
    GROUPED_SMOTENC_SUMMARY.update(synthesized_strata=n_strata_synthesized, skipped_strata=n_strata_skipped,
                                   errored_strata=n_strata_errored, n_synthetic=len(all_synthetic_rows))
    GROUPED_SMOTENC_STRATA.clear()
    GROUPED_SMOTENC_STRATA.extend(all_synthetic_strata)
    return all_synthetic_rows

GROUPED_SMOTENC_STRATA = []   # filled by grouped_smotenc(), aligned with its returned rows
GROUPED_SMOTENC_SUMMARY = {}  # counts, read by the Part 12 verification summary

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
    TOMEK_KEPT_SYNTHETIC_POSITIONS[:] = kept_synthetic_positions
    return kept_real_ids, kept_synthetic_rows

TOMEK_KEPT_SYNTHETIC_POSITIONS = []   # which of the SMOTENC rows survived Tomek (v2, for donor/stratum bookkeeping)

# %%
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
        synthetic_pairs = [(row, 1.0) for row in kept_synthetic_rows]
        # v2 FIX: the v1 line subtracted the number of Tomek-removed MAJORITY patients from the
        # POSITIVE count (never reached in v1 because SMOTENC produced nothing). Tomek with
        # sampling_strategy="majority" never removes real positives, so:
        n_real_pos = int(labels.sum())
        n_pos_final = n_real_pos + len(kept_synthetic_rows)
        n_neg_final = len(kept_real_ids) - n_real_pos
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

# %% [markdown]
# ### 8.2b NEWS2 before vs. after SMOTENC (v2)
#
# SMOTENC interpolates NEWS2 like any continuous feature, so the question is whether the
# synthetic "died" patients still carry clinically possible NEWS2 values that look like
# the real "died" patients they were made from. **Before** = real died patients in the
# training split; **after** = the synthetic ones that survived Tomek cleanup. Hard fail
# if any value falls outside 0-20; REVIEW if the distribution shifts materially (KS > 0.2).
# Interpolated scores are non-integer by construction — expected for a continuous
# feature, reported so it isn't a surprise.

# %%
NEWS2_AUGMENTATION_CHECKS = []
if not USE_NEWS2:
    print("NEWS2 not in the static vector this run -- nothing to check.")
elif not SYNTHETIC_STATIC_ROWS:
    print("No synthetic patients this run -- nothing to compare against.")
else:
    _idx = {n: STATIC_FEATURE_NAMES.index(n) for n in ("news2_max", "news2_mean", "news2_available")}
    def _raw_static(rows):   # back to clinical units (inverse of Part 7.8's StandardScaler)
        return static_scaler.inverse_transform(np.stack(rows))
    _real_pos_ids = [sid for sid in EFFECTIVE_TRAIN_IDS if PATIENT_BUNDLE[sid]["label"] == 1.0]
    _real_neg_ids = [sid for sid in EFFECTIVE_TRAIN_IDS if PATIENT_BUNDLE[sid]["label"] == 0.0]
    _before = _raw_static([IMPUTED_BUNDLE[s]["static"].values.astype(float) for s in _real_pos_ids])
    _after = _raw_static([row for row, _ in SYNTHETIC_STATIC_ROWS])
    _neg = _raw_static([IMPUTED_BUNDLE[s]["static"].values.astype(float) for s in _real_neg_ids])
    # compare only patients who genuinely had a NEWS2 (available flag ~1 before scaling)
    _b_ok = _before[:, _idx["news2_available"]] > 0.5
    _a_ok = _after[:, _idx["news2_available"]] > 0.5
    for feat in ("news2_max", "news2_mean"):
        res = compare_before_after(_before[_b_ok, _idx[feat]], _after[_a_ok, _idx[feat]], name=f"{feat} (SMOTENC)")
        res["pct_non_integer_after"] = float(np.mean(np.abs(_after[_a_ok, _idx[feat]] - np.round(_after[_a_ok, _idx[feat]])) > 1e-6)) \
            if _a_ok.any() else np.nan
        NEWS2_AUGMENTATION_CHECKS.append(res)
    print(pd.DataFrame(NEWS2_AUGMENTATION_CHECKS).set_index("feature").T)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _bins = np.arange(0, 21, 1)
    ax.hist(_neg[_neg[:, _idx["news2_available"]] > 0.5, _idx["news2_max"]], bins=_bins, alpha=0.35, density=True, label="real survived", color="#2980b9")
    ax.hist(_before[_b_ok, _idx["news2_max"]], bins=_bins, alpha=0.6, density=True, label="real died (before)", color="#c0392b")
    ax.hist(_after[_a_ok, _idx["news2_max"]], bins=_bins, alpha=0.5, density=True, label="synthetic died (after)", color="#f39c12", histtype="step", linewidth=2)
    ax.set_xlabel("worst pre-op NEWS2"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("NEWS2 before vs. after SMOTENC")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### 8.2c Time series for synthetic patients (v2)
#
# SMOTENC only creates a synthetic **static** vector. In v1 every synthetic patient got a
# fully empty, fully masked time series — and every synthetic patient is labelled "died",
# so "no observations anywhere" became a perfect death signal inside training data that
# does not exist in real test patients. With `SYNTHETIC_TS_MODE='nearest_real_donor'` each
# synthetic patient instead borrows the time series of the nearest real died patient from
# the **same department × ASA stratum** (distance on continuous static features), with
# fresh jitter every epoch (Part 9.1) so it is never an exact copy.

# %%
SYNTHETIC_TS_DONORS = []
if SYNTHETIC_STATIC_ROWS and CONFIG["SYNTHETIC_TS_MODE"] == "nearest_real_donor":
    _cont_idx = [i for i, c in enumerate(CATEGORICAL_STATIC_MASK) if not c]
    _strata_df = COHORT_INDEXED.loc[EFFECTIVE_TRAIN_IDS, CONFIG["SMOTE_STRATA_COLS"]].copy()
    for _c in CONFIG["SMOTE_STRATA_COLS"]:
        _strata_df[_c] = _strata_df[_c].fillna("missing").astype(str)
    _stratum_of = _strata_df.agg("_".join, axis=1).to_dict()
    _pos_by_stratum = defaultdict(list)
    for _sid in EFFECTIVE_TRAIN_IDS:
        if PATIENT_BUNDLE[_sid]["label"] == 1.0:
            _pos_by_stratum[_stratum_of[_sid]].append(_sid)
    _all_pos = [s for v in _pos_by_stratum.values() for s in v]
    _kept_strata = [GROUPED_SMOTENC_STRATA[p] for p in TOMEK_KEPT_SYNTHETIC_POSITIONS] \
        if len(TOMEK_KEPT_SYNTHETIC_POSITIONS) == len(SYNTHETIC_STATIC_ROWS) else [None] * len(SYNTHETIC_STATIC_ROWS)
    for (row, _), stratum in zip(SYNTHETIC_STATIC_ROWS, _kept_strata):
        candidates = _pos_by_stratum.get(stratum) or _all_pos
        cand_X = _static_matrix(candidates)[:, _cont_idx]
        d = np.linalg.norm(cand_X - np.asarray(row)[_cont_idx], axis=1)
        SYNTHETIC_TS_DONORS.append(candidates[int(np.argmin(d))])
    _uses = Counter(SYNTHETIC_TS_DONORS)
    print(f"Assigned time-series donors to {len(SYNTHETIC_TS_DONORS)} synthetic patients "
          f"from {len(_uses)} distinct real died patients (max reuse of one donor: {max(_uses.values())}).")
elif SYNTHETIC_STATIC_ROWS:
    print("SYNTHETIC_TS_MODE='empty' -- synthetic patients get an all-unobserved time series (v1 behaviour).")

# %% [markdown]
# ## 8.3 Sequence-branch augmentation for real minority patients (jitter + time-mask)
#
# Applies **only to real positive training patients** (never to survived patients, never to
# synthetic static rows, which have no real sequence to begin with -- Section 9.1 already
# fills those with zeros on the standardized scale). Two augmentations, chosen to match
# what's actually buildable on this notebook's fixed-length resampled time grid (Section
# 6.3's `TARGET_SEQ_LEN` points per system) rather than raw irregular timestamps:
#
# - **Magnitude jitter**: adds small Gaussian noise (`CONFIG['JITTER_SIGMA']`, on the
#   already-standardized z-score scale) to observed values only -- imputed/masked positions
#   are left alone so jitter doesn't compound with the imputation strategy's own guesswork.
# - **Time-masking**: randomly zeroes out a short contiguous span of timesteps (and their
#   mask), forcing the encoder to tolerate a missing stretch it hasn't seen at exactly that
#   position before -- a practical, honestly-labelled stand-in for the "window slicing /
#   cropping" augmentation discussed in the external review, adapted for a fixed-length grid
#   rather than literal random-interval cropping of raw timestamps.
#
# **DTW Barycentric Averaging and embedding-space SMOTE are deliberately not implemented
# here** -- both were flagged in `Data_Imbalance_and_Imputation_Reference.md` Part A.2 as
# higher-effort, later-pass items (DBA needs a proper aligned-averaging implementation;
# embedding-space SMOTE is only valid after the Section 9.4 pre-training phase completes,
# which hasn't happened yet at this point in the notebook).

# %%
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

# %%
record_checkpoint("Part 8 -- class-imbalance sampling")
