# %% [markdown]
# ## 6.14 Assembling the final per-patient feature bundle
#
# Pulls everything from §6.1–§6.13 into one `PATIENT_BUNDLE` dict keyed by `subject_id`,
# containing:
#
# - `ts[system]` → `{"raw": [T,F] with NaN gaps, "mask": [T,F], "feature_names": [...]}`
#   for each of the six time-series systems (renal/cardiovascular/respiratory/
#   metabolic_hepatic/haematology/neurological) — still **unimputed**, exactly as built in
#   §6.3, ready for §7.
# - `static` → one flat vector: demographics/ASA/department (§6.12), GI/MSK/HFRS (§6.6–6.8),
#   operation-count features (§6.10), the cardiac-recovery exception flag (§6.11), the
#   cardiovascular summary + renal-cardiac interaction (§6.9), and medication aggregates
#   (§6.13).
# - `label` → the 30-day mortality target from §4.2.
#
# This is the single object every later section (§7 imputation, §8 split/sampling, §9 model)
# reads from.

# %%
def build_static_vector(sid):
    parts = {}
    parts.update(static_df.loc[sid].to_dict())
    parts.update(icd10_features_df.loc[sid].to_dict())
    parts.update(infection_features_df.loc[sid].to_dict())   # §6.6, roadmap §4.1a -- new this revision
    parts.update(op_count_df.loc[sid].to_dict())
    parts["cardiac_recovery_exception"] = CARDIAC_EXCEPTION_FLAG[sid]
    parts.update(cardiac_summary_df.loc[sid].to_dict())
    # v3: the hand-crafted creatinine x |MAP deviation| interaction (RENAL_CARDIAC_INTERACTION,
    # still computed above for reference) is NOT a model input: inter-system relationships
    # are learned by the coupling layer, never defined by hand.
    parts.update(med_features_df.loc[sid].to_dict())
    if len(organ_content_df.columns):  # v4: procedure body-system + medicine-family features per organ system
        parts.update(organ_content_df.loc[sid].to_dict())
    parts.update(AGG_FEATURES[sid])   # §6.15, Part C §4 of the imbalance/imputation reference
    if USE_NEWS2:                      # §6.16 (v2) -- only if the self-test passed
        parts.update(news2_features_df.loc[sid].to_dict())
    # v2 FIX: return the features in SORTED name order. STATIC_FEATURE_NAMES (below) is sorted,
    # and every later step that picks static columns BY NAME (SMOTENC's categorical columns,
    # the GI/MSK branches, NEWS2 checks, donor matching) indexes into that sorted list. In v1
    # the values stayed in build order, so names and values were silently misaligned (e.g. the
    # columns taken as 'gi_*' actually held white-cell-count statistics).
    return {k: parts[k] for k in sorted(parts)}

PATIENT_BUNDLE = {}
for sid in cohort_df["subject_id"]:
    ts = {system: RAW_SYSTEM_TENSORS[(sid, system)] for system in TIME_SERIES_SYSTEMS}
    PATIENT_BUNDLE[sid] = {
        "ts": ts,
        "static": build_static_vector(sid),
        "label": float(COHORT_INDEXED.loc[sid, "died_30day_from_last_op"]),
    }

STATIC_FEATURE_NAMES = sorted(next(iter(PATIENT_BUNDLE.values()))["static"].keys())
print(f"Bundle built for {len(PATIENT_BUNDLE)} patients.")
print(f"Static feature vector length: {len(STATIC_FEATURE_NAMES)}")
print(f"Static features: {STATIC_FEATURE_NAMES}")
for system in TIME_SERIES_SYSTEMS:
    fnames = PATIENT_BUNDLE[cohort_df['subject_id'].iloc[0]]['ts'][system]['feature_names']
    print(f"  {system:18s} time-series features ({len(fnames)}): {fnames}")

# %%
# Free most raw long-format tables now that PATIENT_BUNDLE holds everything §7-§12 need --
# medications_df and diagnoses_df specifically, since nothing downstream reads them again.
# labs_df/vitals_df/ward_vitals_df are DELIBERATELY KEPT ALIVE (unlike an earlier revision
# of this cell, which freed all five): §11.8's risk-trajectory analysis needs to re-query
# these at custom time cutoffs after the model is trained, which isn't possible once
# they're gone. This keeps a modest amount of memory alive at full scale in exchange for
# that analysis working -- worth it now that Part 3's chunked parsing (not these already
# fairly compact flattened tables) is the fix that solved the real memory bottleneck.
import gc

_freed_mb = sum(df.memory_usage(deep=True).sum() for df in [medications_df, diagnoses_df]) / 1e6
del medications_df, diagnoses_df
RAW_SYSTEM_TENSORS.clear()   # PATIENT_BUNDLE['ts'][system] already holds these dicts directly -- see §6.3's note
gc.collect()
print(f"Freed medications_df/diagnoses_df (~{_freed_mb:.1f} MB on this dev subset). "
      f"labs_df/vitals_df/ward_vitals_df are kept alive -- needed by §11.8's risk-trajectory "
      f"analysis, which re-queries them at custom time cutoffs after training.")

# %%
# Quick label sanity check on the assembled bundle before moving to §7.
labels = np.array([PATIENT_BUNDLE[sid]["label"] for sid in cohort_df["subject_id"]])
print(f"Positive rate in bundle: {labels.mean():.1%}  ({int(labels.sum())} / {len(labels)})")
print(f"For reference, the full cohort's rate is ~0.5-0.9% (§1.4). A run that keeps every death "
      f"but caps survivors (MAX_SUBJECTS_PER_CLASS={CONFIG['MAX_SUBJECTS_PER_CLASS']}) has a higher rate "
      f"than deployment -- AUPRC and calibration here are not deployment numbers.")
print(f"NEWS2 in the static vector: {USE_NEWS2}  |  renal_cardiac_interaction (manual coupling feature) "
      f"included: {'renal_cardiac_interaction' in STATIC_FEATURE_NAMES}")

# %%
record_checkpoint("Parts 4-6 -- labelling + organ-system feature engineering")
