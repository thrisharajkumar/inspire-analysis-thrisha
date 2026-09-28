# %% [markdown]
# # Part 6 — Organ-system feature engineering
#
# This is the section that turns the theory in §1.1–§1.6 into actual feature tables. Order
# of build, each tied back to a Part 1 section:
#
# - §6.1 the organ-system → raw-signal grouping table (the routing map)
# - §6.2 pre-op / peri-op time-window extraction per patient (§1.6.3)
# - §6.3–§6.5 the four "genuine time series" systems: Renal, Respiratory, Metabolic/hepatic, Haematology
# - §6.9 Cardiovascular, built *jointly* with Renal because of the coupling in §1.6.2
# - §6.6 ICD-10 features: chapter flags/counts + HFRS (§1.2)
# - §6.7–§6.8 GI and MSK (diagnosis/department-driven, §1.2.3)
# - §6.10 operation-count-in-same-area features (§1.6.4)
# - §6.11 the 6-month post-cardiac-surgery exception flag (§1.6.4, rule-based on purpose)
# - §6.12 static features (age, sex, ASA, emop, department, weight, height) (§1.5.1, §1.2.2)
# - §6.13 medications → ATC-level aggregate features (§1.5.3)
# - §6.14 assembling everything into the final per-patient, per-system feature bundle

# %% [markdown]
# ## 6.1 The organ-system → raw-signal routing map
#
# This extends the six-system grouping already sketched in the source repo's own
# `docs/roadmap_and_architecture.md` §4.1 with the two new systems from the meeting notes,
# and is filtered to the item-names actually observed in §4.1's inventory (so nothing below
# references a feature this dataset doesn't have).

# %%
# --- Time-series systems: item_name -> system, split by source table ---
# Kept as an explicit, editable dict (not auto-derived) so the organ-system assignment is
# a reviewable, documented decision -- exactly the kind of thing you said you want to be
# able to interrogate and change yourself.

SYSTEM_LABS = {
    "renal":          ["bun", "calcium", "chloride", "creatinine", "ica", "phosphorus", "potassium", "sodium"],
    "cardiovascular":  ["ck", "ckmb", "troponin_i"],   # troponin_t not present in this subset's inventory (§4.1)
    "respiratory":    ["be", "hco3", "paco2", "pao2", "ph", "sao2"],
    "metabolic_hepatic": ["albumin", "alp", "alt", "ast", "glucose", "hba1c", "lacate", "total_bilirubin", "total_protein"],
    "haematology":    ["aptt", "crp", "fibrinogen", "hb", "hct", "lymphocyte", "platelet", "ptinr", "seg", "wbc"],
    "neurological":   [],   # no dedicated neuro lab in this dataset
}

SYSTEM_WARD_VITALS = {
    "renal":          ["crrt", "uo"],
    "cardiovascular":  ["hr", "nibp_sbp", "nibp_dbp", "nibp_mbp", "iabp"],
    "respiratory":    ["fio2", "rr", "spo2", "vent", "ecmo"],
    "metabolic_hepatic": ["bt"],
    "haematology":    [],
    "neurological":   ["gcs_e", "gcs_m", "gcs_v"],
}

SYSTEM_INTRAOP_VITALS = {   # only used when CONFIG['TIME_WINDOW'] == 'peri_op'
    "renal":          ["uo"],
    "cardiovascular":  ["hr", "art_sbp", "art_dbp", "art_mbp", "ci", "cvp", "svi",
                        "nibp_sbp", "nibp_dbp", "nibp_mbp", "pap_sbp", "pap_dbp", "pap_mbp"],
    "respiratory":    ["etco2", "fio2", "spo2", "peep", "pip", "pplat", "rr", "minvol", "vt"],
    "metabolic_hepatic": ["bt"],
    "haematology":    ["ebl", "rbc", "ffp"],
    "neurological":   ["bis"],
}

TIME_SERIES_SYSTEMS = ["renal", "cardiovascular", "respiratory", "metabolic_hepatic", "haematology", "neurological"]

# Sanity-check every listed item_name actually exists in this dataset's inventory (§4.1).
for sysdict, inv_key in [(SYSTEM_LABS, "labs"), (SYSTEM_WARD_VITALS, "ward_vitals"), (SYSTEM_INTRAOP_VITALS, "vitals (intra-op)")]:
    known = set(INVENTORY[inv_key])
    for system, items in sysdict.items():
        unknown = [i for i in items if i not in known]
        if unknown:
            print(f"WARNING: {inv_key}/{system} references unseen item_names: {unknown}")
print("Routing map validated against §4.1 inventory (no warnings above = every referenced item_name exists).")

# %% [markdown]
# ## 6.2 Pre-op / peri-op window extraction
#
# Implements §1.6.3's `CONFIG['TIME_WINDOW']` toggle. `'pre_op'` keeps only records in
# `[orin_time - PRE_OP_DAYS*24*60, orin_time]` for the **last** operation (matching the label
# definition in §4.2 — Path C, "from last operation"; §11.7 revisits first-operation and
# exclude-multi-op as a sensitivity check). `'peri_op'` extends the upper bound to
# `orout_time` and additionally pulls in intra-op `vitals`.

# %%
def get_window(subject_id, cohort_row):
    orin = cohort_row["orin_time_last"]
    orout = cohort_row["orout_time_last"]
    lo = orin - CONFIG["PRE_OP_DAYS"] * 24 * 60
    if CONFIG["TIME_WINDOW"] == "pre_op":
        hi = orin
    elif CONFIG["TIME_WINDOW"] == "peri_op":
        hi = orout if pd.notna(orout) else orin
    else:
        raise ValueError(f"Unknown TIME_WINDOW {CONFIG['TIME_WINDOW']!r}")
    return lo, hi

COHORT_INDEXED = cohort_df.set_index("subject_id")

# --- Fast per-patient/feature lookup (replaces a full-table scan per call) -----------
# The naive version of windowed_series() -- filter the whole DataFrame by subject_id AND
# item_name AND time range, on every single call -- is fine on the 30-patient dev subset
# but does NOT scale: this notebook calls it roughly 60-70 times per patient across
# Part 6, so at the full ~99,886-patient cohort that's several million calls, each an
# O(table_size) scan over tables with tens of millions of rows. Realistically that's days,
# not hours -- it would exceed any Colab/Kaggle session limit long before the GPU or RAM
# did. The fix below groups each table ONCE into a plain Python dict keyed by
# (subject_id, item_name) -> (chart_time array, value array); every lookup after that is a
# dict hash lookup plus a small in-memory numpy filter, both essentially O(1) with respect
# to the table's total size. (An earlier version of this fix used a sorted pandas
# MultiIndex + .loc[] instead -- measured at only ~20-90x faster than the naive scan on
# benchmark tables, versus >4,000x for the plain-dict version below: pandas' .loc[] call
# overhead dominates for small per-key results, which is exactly this notebook's access
# pattern. Benchmarked below rather than assumed.) Output is identical either way -- this
# only changes speed, verified by confirming every printed count/statistic in this
# notebook is unchanged after the switch.
_GROUPED_ARRAYS_CACHE = {}

def _get_grouped_arrays(df):
    key = id(df)
    if key not in _GROUPED_ARRAYS_CACHE:
        grouped = {}
        if len(df) and "item_name" in df.columns:
            for gkey, g in df.groupby(["subject_id", "item_name"], observed=True, sort=False):
                grouped[gkey] = (g["chart_time"].to_numpy(), g["value"].to_numpy())
        _GROUPED_ARRAYS_CACHE[key] = grouped
    return _GROUPED_ARRAYS_CACHE[key]

def windowed_series(df, subject_id, item_name, lo, hi, subj_col="subject_id"):
    # Returns a sorted (chart_time -> value) dict for one patient/feature within [lo, hi].
    if len(df) == 0 or "item_name" not in df.columns:
        return {}
    grouped = _get_grouped_arrays(df)
    arrs = grouped.get((subject_id, item_name))
    if arrs is None:
        return {}
    times, vals = arrs
    mask = (times >= lo) & (times <= hi) & ~np.isnan(vals.astype(float))
    if not mask.any():
        return {}
    t_sel, v_sel = times[mask], vals[mask]
    order = np.argsort(t_sel)
    return dict(zip(t_sel[order], v_sel[order]))

print(f"TIME_WINDOW = {CONFIG['TIME_WINDOW']!r}  (see §1.6.3 for what each option represents)")

# %% [markdown]
# ### A note on why this matters more than it looks
#
# The grouped-lookup rewrite above is the single highest-impact change in this notebook for
# runnability at full scale — more than any of the memory optimizations. A self-contained
# timing comparison on synthetic data of a similar shape to the full cohort (run once,
# below, to make this concrete rather than asserted) shows the real speedup, split into the
# one-time index-build cost (paid once per table, however large the cohort) and the
# marginal per-lookup cost (paid millions of times, so this is the number that actually
# determines whether a full run finishes in minutes/hours or days).

# %%
import time

def _make_synthetic_table(n_subjects, avg_rows_per_subject, n_item_names=40):
    rng = np.random.default_rng(0)
    n_rows = n_subjects * avg_rows_per_subject
    subject_ids = rng.integers(0, n_subjects, size=n_rows).astype(str)
    item_names = rng.choice([f"item_{i}" for i in range(n_item_names)], size=n_rows)
    chart_times = rng.integers(-10000, 10000, size=n_rows)
    values = rng.normal(size=n_rows).astype(np.float32)
    df = pd.DataFrame({"subject_id": subject_ids, "item_name": item_names,
                        "chart_time": chart_times, "value": values})
    df["subject_id"] = df["subject_id"].astype("category")
    df["item_name"] = df["item_name"].astype("category")
    return df

_SYN_N_SUBJECTS = 2000
_synthetic_df = _make_synthetic_table(_SYN_N_SUBJECTS, avg_rows_per_subject=200)
print(f"Synthetic benchmark table: {len(_synthetic_df):,} rows, {_SYN_N_SUBJECTS:,} subjects "
      f"(a small slice of the full cohort's scale, kept small here so the OLD method's demo "
      f"below finishes quickly -- the gap only widens at the real ~99,886-patient size)")

def _old_windowed_series(df, subject_id, item_name, lo, hi):
    sub = df[(df["subject_id"] == subject_id) & (df["item_name"] == item_name) &
             (df["chart_time"] >= lo) & (df["chart_time"] <= hi)]
    return dict(zip(sub["chart_time"], sub["value"]))

_sample_subject_ids = np.random.default_rng(1).choice(_synthetic_df["subject_id"].cat.categories, size=150, replace=True)
_sample_items = [f"item_{i}" for i in range(6)]
n_calls = len(_sample_subject_ids) * len(_sample_items)

_t0 = time.time()
for sid in _sample_subject_ids:
    for item in _sample_items:
        _old_windowed_series(_synthetic_df, sid, item, -10000, 10000)
_old_elapsed = time.time() - _t0

_GROUPED_ARRAYS_CACHE.clear()
_t0 = time.time()
_ = windowed_series(_synthetic_df, _sample_subject_ids[0], _sample_items[0], -10000, 10000)   # triggers the one-time group build
_build_elapsed = time.time() - _t0

_t0 = time.time()
for sid in _sample_subject_ids:
    for item in _sample_items:
        windowed_series(_synthetic_df, sid, item, -10000, 10000)
_new_lookups_elapsed = time.time() - _t0

old_ms_per_call = _old_elapsed / n_calls * 1000
new_ms_per_call = _new_lookups_elapsed / n_calls * 1000
print(f"\n{n_calls:,} lookups against a {len(_synthetic_df):,}-row table:")
print(f"  old (full-scan every call):    {_old_elapsed:.2f}s total  ({old_ms_per_call:.3f} ms/call)")
print(f"  new -- one-time group build:   {_build_elapsed:.2f}s (paid ONCE per table for the whole notebook run)")
print(f"  new -- {n_calls:,} grouped lookups: {_new_lookups_elapsed:.4f}s  ({new_ms_per_call:.5f} ms/call)")
print(f"  marginal per-call speedup:     {old_ms_per_call/max(new_ms_per_call,1e-9):.0f}x  "
      f"(this is the number that matters at full scale, where millions of calls share one build)")
print(f"\nThe one-time build cost scales with table size (more rows/groups to organise), so at "
      f"the real full-cohort scale expect the build itself to take real minutes, not seconds -- "
      f"that is an acceptable, one-time cost given it replaces what would otherwise be a "
      f"multi-day total runtime under the old per-call scan.")

del _synthetic_df
_GROUPED_ARRAYS_CACHE.clear()   # drop the benchmark's groups; the real tables get grouped fresh on first real use

# %% [markdown]
# ## 6.3 Per-system time-series extraction, resampling, and the missingness mask
#
# For each patient and each of the six time-series systems (§6.1), build a
# `[TARGET_SEQ_LEN, n_features_in_system]` array plus a same-shaped **mask** array
# (1 = observed/interpolated-from-real-data, 0 = no data at all for that patient/feature —
# see §1.3's MNAR discussion for why the mask is kept as its own feature rather than
# discarded once a value is filled in). The actual *value*-filling strategy is deliberately
# left as a placeholder here (`_TODO_impute`) and implemented properly in §7, once all the
# methods have been introduced together — this section only handles resampling onto a
# common time grid, which is method-independent.

# %%
def resample_to_grid(chart_time2value, lo, hi, n_points):
    # Evenly-spaced grid of n_points between lo and hi; each grid point takes the NEAREST
    # observation, but only if it is within one grid step -- otherwise NaN, left for Part 7's
    # imputation. v2: vectorised with numpy (identical output to the v1 per-point loop, which
    # was one of the slowest steps at full-cohort scale). Ties go to the later observation,
    # exactly as before.
    grid = np.linspace(lo, hi, n_points)
    out = np.full(n_points, np.nan)
    if len(chart_time2value) == 0:
        return grid, out
    keys = sorted(chart_time2value.keys())
    times = np.array(keys, dtype=float)
    values = np.array([chart_time2value[t] for t in keys], dtype=float)
    idx = np.searchsorted(times, grid)
    right = np.clip(idx, 0, len(times) - 1)
    left = np.clip(idx - 1, 0, len(times) - 1)
    d_right = np.where(idx < len(times), np.abs(times[right] - grid), np.inf)
    d_left = np.where(idx > 0, np.abs(times[left] - grid), np.inf)
    use_right = d_right <= d_left
    best_d = np.where(use_right, d_right, d_left)
    best_v = np.where(use_right, values[right], values[left])
    step = (hi - lo) / max(n_points - 1, 1)
    ok = best_d <= max(step, 1e-6)
    out[ok] = best_v[ok]
    return grid, out

def extract_system_tensor(subject_id, system, lo, hi):
    # Returns (raw_values [T,F] with NaN for gaps, mask [T,F], feature_names) for one
    # patient/system, pooling labs + ward_vitals (+ intra-op vitals if peri_op).
    feature_names = []
    columns = []
    sources = [(labs_df, SYSTEM_LABS[system]), (ward_vitals_df, SYSTEM_WARD_VITALS[system])]
    if CONFIG["TIME_WINDOW"] == "peri_op":
        sources.append((vitals_df, SYSTEM_INTRAOP_VITALS[system]))
    for df, items in sources:
        for item in items:
            series = windowed_series(df, subject_id, item, lo, hi)
            grid, raw = resample_to_grid(series, lo, hi, CONFIG["TARGET_SEQ_LEN"])
            feature_names.append(item)
            columns.append(raw)
    if not columns:
        return (np.zeros((CONFIG["TARGET_SEQ_LEN"], 0)),
                np.zeros((CONFIG["TARGET_SEQ_LEN"], 0)),
                [])
    raw_values = np.stack(columns, axis=1)          # [T, F]
    mask = (~np.isnan(raw_values)).astype(np.float32)
    return raw_values, mask, feature_names

# Build the raw (pre-imputation) tensors for every patient x system, kept in a dict so §7
# can impute in place without re-doing the windowing/resampling work.
RAW_SYSTEM_TENSORS = {}   # (subject_id, system) -> dict(raw=[T,F], mask=[T,F], feature_names=[...])
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    for system in TIME_SERIES_SYSTEMS:
        raw, mask, fnames = extract_system_tensor(sid, system, lo, hi)
        RAW_SYSTEM_TENSORS[(sid, system)] = {"raw": raw, "mask": mask, "feature_names": fnames}

# Quick coverage summary: for each system, what fraction of (patient, feature) cells have
# at least one real observation in the window?
for system in TIME_SERIES_SYSTEMS:
    fnames = RAW_SYSTEM_TENSORS[(cohort_df["subject_id"].iloc[0], system)]["feature_names"]
    if not fnames:
        print(f"{system:18s}: no features assigned (see §6.1)")
        continue
    total, observed = 0, 0
    for sid in cohort_df["subject_id"]:
        m = RAW_SYSTEM_TENSORS[(sid, system)]["mask"]
        total += m.shape[0] * m.shape[1]
        observed += m.sum()
    print(f"{system:18s}: {len(fnames):2d} features, {observed/total:5.1%} of (patient,timepoint,feature) cells observed in-window")

# %% [markdown]
# ## 6.9 Cardiovascular ↔ Renal coupling (§1.6.2)
#
# Two concrete implementations of the meeting note "renal directly proportionate to
# cardiovascular," matching the two mechanisms described in §1.6.2:
#
# 1. **Architectural**: a compact cardiovascular summary (mean heart rate, mean arterial
#    pressure deviation from normal, presence of an IABP flag) is computed here and appended
#    to the *renal* branch's static side-input in §9 — so the renal encoder sees
#    cardiovascular context directly, not just renal labs.
# 2. **Hand-crafted interaction feature**: `renal_cardiac_interaction`, a creatinine ×
#    blood-pressure-deviation product, added to the static feature table (§6.14) as a
#    cheap, sample-efficient prior (§1.6.2 explains why this is added *alongside* the
#    learned coupling, not instead of it).
#
# `CONFIG['SYMMETRIC_CARDIORENAL_COUPLING']` toggles whether the *cardiovascular* branch
# symmetrically receives a renal summary back — off by default, per the §1.6.2 reasoning.

# %%
NORMAL_MAP_MMHG = 93.0   # a commonly used "normal" mean arterial pressure reference point

def cardiac_summary_for_patient(sid, lo, hi):
    # A small, fixed-size cardiovascular summary vector -- fed into the renal branch (§9)
    # as the architectural half of the cardiorenal coupling, and used below to build the
    # hand-crafted interaction feature.
    hr_series = windowed_series(ward_vitals_df, sid, "hr", lo, hi)
    map_series = windowed_series(ward_vitals_df, sid, "nibp_mbp", lo, hi)
    iabp_series = windowed_series(ward_vitals_df, sid, "iabp", lo, hi)

    mean_hr = float(np.mean(list(hr_series.values()))) if hr_series else np.nan
    mean_map = float(np.mean(list(map_series.values()))) if map_series else np.nan
    map_deviation = (mean_map - NORMAL_MAP_MMHG) if not np.isnan(mean_map) else np.nan
    has_iabp = float(len(iabp_series) > 0 and any(v > 0 for v in iabp_series.values()))

    return {"cardio_mean_hr": mean_hr, "cardio_map_deviation": map_deviation, "cardio_has_iabp": has_iabp}

CARDIAC_SUMMARY = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    CARDIAC_SUMMARY[sid] = cardiac_summary_for_patient(sid, lo, hi)

cardiac_summary_df = pd.DataFrame(CARDIAC_SUMMARY).T
cardiac_summary_df.index.name = "subject_id"
print(f"Cardiovascular summary built for {len(cardiac_summary_df)} patients "
      f"(NaNs below reflect real missingness -- filled properly in §7, not here).")
cardiac_summary_df.describe()

# %%
def renal_cardiac_interaction_for_patient(sid, lo, hi):
    # Hand-crafted §1.6.2 interaction feature: creatinine x |MAP deviation|.
    # NaN-safe -- returns NaN if either side is unavailable, filled like any other
    # static feature in §7.
    creat_series = windowed_series(labs_df, sid, "creatinine", lo, hi)
    mean_creat = float(np.mean(list(creat_series.values()))) if creat_series else np.nan
    map_dev = CARDIAC_SUMMARY[sid]["cardio_map_deviation"]
    if np.isnan(mean_creat) or (map_dev is None) or np.isnan(map_dev):
        return np.nan
    return mean_creat * abs(map_dev)

RENAL_CARDIAC_INTERACTION = {
    sid: renal_cardiac_interaction_for_patient(sid, *get_window(sid, COHORT_INDEXED.loc[sid]))
    for sid in cohort_df["subject_id"]
}
pd.Series(RENAL_CARDIAC_INTERACTION, name="renal_cardiac_interaction").describe()

# %% [markdown]
# ## 6.6 / 6.7 / 6.8 — ICD-10 chapter features, HFRS, GI and MSK systems, infection flag
#
# All built together because they share the same underlying computation (diagnosis chapter
# membership within a lookback window) — GI and MSK (§1.2.3, §1.6.1) are simply the two
# chapters the meeting notes called out by name, exposed as their own system entries rather
# than folded into a generic "diagnosis count" feature.
#
# - **GI** = ICD-10 chapter **XI** (`K00-K93`) flag/count + `department == 'GS'` (general
#   surgery, the closest department proxy for GI surgical burden in this dataset).
# - **MSK** = ICD-10 chapter **XIII** (`M00-M99`) flag/count + `department == 'OS'`
#   (orthopaedic surgery).
# - **HFRS** (Hospital Frailty Risk Score, Gilbert et al. 2018) — a validated, externally
#   weighted score over 109 specific ICD-10 codes, restricted here to the published lookback
#   window (`CONFIG['HFRS_LOOKBACK_YEARS']`, default 2, and only meaningfully applied for
#   patients 75+ per the original paper — see §1.2.1). **This revision uses the full 109-code
#   weights table**, extracted directly from the source repo's `frailty_hfrs.py` (an earlier
#   revision of this notebook shipped a ~30-code representative subset — that limitation,
#   flagged in `Multimodal_Notebook_Summary.md` §7, is now closed).
# - **Infection/inflammation flag** — the cross-cutting signal `roadmap_and_architecture.md`
#   §4.1a designed but had not yet implemented (until this revision). Per SOFA (Vincent et
#   al. 1996) and Sepsis-3 (Singer et al. 2016), infection is modelled as a **modifier
#   layered across systems**, not a competing 7th/9th organ system — implemented here as a
#   small set of static features (chapter-I diagnosis flag, the six specific high-mortality
#   codes already flagged in the source repo's EDA, fever, abnormal WBC, and a data-driven
#   "elevated CRP" flag — see the code cell for why CRP uses a relative, not absolute,
#   threshold) plus a 0–5 composite count, fed into the static branch alongside GI/MSK/HFRS.

# %%
# v2 SPEED FIX: group diagnoses and medications by patient ONCE. The v1 code filtered the
# whole table for every patient (df[df.subject_id == sid]), which is fine at 10k patients
# but grows roughly with (patients x rows) -- hours at the full ~99,886-patient cohort.
# Same rows, same order within a patient, so every feature below is unchanged.
def _group_by_subject(df):
    if len(df) == 0:
        return {}
    return {k: g for k, g in df.groupby("subject_id", observed=True, sort=False)}

_DX_BY_SUBJECT = _group_by_subject(diagnoses_df)
_MEDS_BY_SUBJECT = _group_by_subject(medications_df)
_EMPTY_DX = diagnoses_df.iloc[0:0]
_EMPTY_MEDS = medications_df.iloc[0:0]

def dx_in_window(sid, lo, hi):
    g = _DX_BY_SUBJECT.get(sid, _EMPTY_DX)
    return g[(g["chart_time"] >= lo) & (g["chart_time"] <= hi)]

def meds_in_window(sid, lo, hi):
    g = _MEDS_BY_SUBJECT.get(sid, _EMPTY_MEDS)
    return g[(g["chart_time"] >= lo) & (g["chart_time"] <= hi)]

# Full Hospital Frailty Risk Score weights table (Gilbert et al. 2018, Table A2, 109
# ICD-10-CM clusters), extracted directly from the source repo's frailty_hfrs.py so this
# notebook stays self-contained/Kaggle-portable without importing that module.
HFRS_WEIGHTS = {
    'A04': 1.1, 'A09': 1.1, 'A41': 1.6, 'B95': 1.7, 'B96': 2.9, 'D64': 0.4,
    'E05': 0.9, 'E16': 1.4, 'E53': 1.9, 'E55': 1.0, 'E83': 0.4, 'E86': 2.3,
    'E87': 2.3, 'F00': 7.1, 'F01': 2.0, 'F03': 2.1, 'F05': 3.2, 'F10': 0.7,
    'F32': 0.5, 'G20': 1.8, 'G30': 4.0, 'G31': 1.2, 'G40': 1.5, 'G45': 1.2,
    'G81': 4.4, 'H54': 1.9, 'H91': 0.9, 'I63': 0.8, 'I67': 2.6, 'I69': 3.7,
    'I95': 1.6, 'J18': 1.1, 'J22': 0.7, 'J69': 1.0, 'J96': 1.5, 'K26': 1.6,
    'K52': 0.3, 'K59': 1.8, 'K92': 0.8, 'L03': 2.0, 'L08': 0.4, 'L89': 1.7,
    'L97': 1.6, 'M15': 0.4, 'M19': 1.5, 'M25': 2.3, 'M41': 0.9, 'M48': 0.5,
    'M79': 1.1, 'M80': 0.8, 'M81': 1.4, 'N17': 1.8, 'N18': 1.4, 'N19': 1.6,
    'N20': 0.7, 'N28': 1.3, 'N39': 3.2, 'R00': 0.7, 'R02': 1.0, 'R11': 0.3,
    'R13': 0.8, 'R26': 2.6, 'R29': 3.6, 'R31': 3.0, 'R32': 1.2, 'R33': 1.3,
    'R40': 2.5, 'R41': 2.7, 'R44': 1.6, 'R45': 1.2, 'R47': 1.0, 'R50': 0.1,
    'R54': 2.2, 'R55': 1.8, 'R56': 2.6, 'R63': 0.9, 'R69': 1.3, 'R79': 0.6,
    'R94': 1.4, 'S00': 3.2, 'S01': 1.1, 'S06': 2.4, 'S09': 1.2, 'S22': 1.8,
    'S32': 1.4, 'S42': 2.3, 'S51': 0.5, 'S72': 1.4, 'S80': 2.0, 'T83': 2.4,
    'U80': 0.8, 'W01': 0.9, 'W06': 1.1, 'W10': 0.9, 'W18': 2.1, 'W19': 3.2,
    'X59': 1.5, 'Y84': 0.7, 'Y95': 1.2, 'Z22': 1.7, 'Z50': 2.1, 'Z60': 1.8,
    'Z73': 0.6, 'Z74': 1.1, 'Z75': 2.0, 'Z87': 1.5, 'Z91': 0.5, 'Z93': 1.0,
    'Z99': 0.8,
}
assert len(HFRS_WEIGHTS) == 109, f"Expected 109 HFRS codes (Gilbert et al. Table A2), got {len(HFRS_WEIGHTS)}"

def compute_hfrs(sid, lo, hi, age):
    # Sum of HFRS weights for matching diagnosis codes within the lookback window,
    # restricted to patients aged 75+ (per Gilbert et al.'s validated population -- Sec 1.2.1).
    # Returns 0.0 (not NaN) for younger patients: HFRS is defined as not-applicable, not
    # missing, below the validated age range.
    if pd.isna(age) or age < 75:
        return 0.0
    lookback_minutes = CONFIG["HFRS_LOOKBACK_YEARS"] * 365 * 24 * 60
    dx = dx_in_window(sid, lo - lookback_minutes, hi)
    score = 0.0
    for code in dx["icd10_cm"].dropna():
        code3 = str(code)[:3].upper()
        if code3 in HFRS_WEIGHTS:
            score += HFRS_WEIGHTS[code3]
    return score

# v3: every organ system gets diagnosis-code features from its own ICD-10 chapter, so all
# eight systems are built the same way (GI and MSK are no longer a special case -- they are
# simply the two systems with no time-series measurements in this dataset).
ORGAN_SYSTEM_ICD10_CHAPTERS = {
    "renal": ["XIV"],              # genitourinary
    "cardiovascular": ["IX"],      # circulatory
    "respiratory": ["X"],          # respiratory
    "metabolic_hepatic": ["IV"],   # endocrine, nutritional, metabolic
    "haematology": ["III"],        # blood and immune
    "neurological": ["VI"],        # nervous system
    "gi": ["XI"],                  # digestive
    "msk": ["XIII"],               # musculoskeletal
}
ORGAN_SYSTEMS = list(ORGAN_SYSTEM_ICD10_CHAPTERS)   # the eight organ systems, one order everywhere

def organ_system_dx_features(sid, lo, hi, department):
    dx = dx_in_window(sid, lo, hi)
    chapters = dx["icd10_cm"].dropna().apply(icd10_chapter)
    n_chapters = chapters.value_counts()
    feats = {}
    for system, chs in ORGAN_SYSTEM_ICD10_CHAPTERS.items():
        n = int(sum(n_chapters.get(c, 0) for c in chs))
        feats[f"sysdx_{system}_count"] = n
        feats[f"sysdx_{system}_flag"] = float(n > 0)
    # department flags kept from v1 (surgical specialty treating the system)
    feats["gi_department_flag"] = float(department == "GS")
    feats["msk_department_flag"] = float(department == "OS")
    feats["n_diagnoses_in_window"] = int(len(dx))
    feats["n_distinct_chapters_in_window"] = int(chapters.nunique())
    return feats

ICD10_FEATURES = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    feats = organ_system_dx_features(sid, lo, hi, row["department"])
    if CONFIG["INCLUDE_HFRS"]:
        feats["hfrs"] = compute_hfrs(sid, lo, hi, row["age"])
    ICD10_FEATURES[sid] = feats

icd10_features_df = pd.DataFrame(ICD10_FEATURES).T
icd10_features_df.index.name = "subject_id"
print("Patients with >=1 diagnosis in each organ system's ICD-10 chapter (pre-op window):")
for _s in ORGAN_SYSTEMS:
    print(f"  {_s:18s} {int(icd10_features_df[f'sysdx_{_s}_flag'].sum()):6d} / {len(icd10_features_df)}")
if CONFIG["INCLUDE_HFRS"]:
    n_scored = int((icd10_features_df["hfrs"] > 0).sum())
    print(f"Patients with HFRS > 0 (i.e. aged 75+ with a matching code): {n_scored} / {len(icd10_features_df)}")
icd10_features_df.describe()

# %% [markdown]
# ### Infection / inflammation cross-cutting flag (`roadmap_and_architecture.md` §4.1a)
#
# Six components, each a real, cheap-to-compute signal rather than a learned sub-model —
# consistent with the design note that this is a *modifier*, not a competing organ system.
# The CRP threshold is deliberately **data-driven (this cohort's own 75th percentile)**
# rather than an absolute clinical cutoff (e.g. "CRP > 100 mg/L") -- this notebook never
# loaded `parameters.csv`, so CRP's exact reporting unit for this INSPIRE export isn't
# independently confirmed here, and a wrong absolute threshold is worse than an honestly
# relative one. Fever and abnormal-WBC thresholds use standard, unit-unambiguous clinical
# cutoffs (°C and count/volume respectively), so those stay absolute.

# %%
HIGH_RISK_INFECTION_CODES = {"D65", "I46", "R57", "J80", "K72", "A41"}   # from the source repo's own EDA (docs/eda_findings.md)
FEVER_THRESHOLD_C = 38.0
WBC_NORMAL_RANGE = (4.0, 11.0)   # x10^9/L, standard adult reference range

# Data-driven CRP threshold: computed once from the training-eligible population (here,
# the whole cohort at bundle-build time, since this runs before the train/val/test split
# in Part 8 -- acceptable for a threshold that's a description of the cohort's own
# distribution, not a fitted statistic that could leak label information).
_all_crp_values = labs_df.loc[labs_df["item_name"] == "crp", "value"].dropna()
CRP_ELEVATED_THRESHOLD = float(_all_crp_values.quantile(0.75)) if len(_all_crp_values) else None
print(f"CRP 'elevated' threshold (75th percentile of all observed CRP in this cohort): "
      f"{CRP_ELEVATED_THRESHOLD}")

def infection_inflammation_features(sid, lo, hi):
    dx = dx_in_window(sid, lo, hi)
    codes = set(dx["icd10_cm"].dropna().astype(str))
    chapter_i_flag = float(any(icd10_chapter(c) == "I" for c in codes))
    high_risk_flag = float(len(codes & HIGH_RISK_INFECTION_CODES) > 0)

    bt_series = windowed_series(ward_vitals_df, sid, "bt", lo, hi)
    fever_flag = float(len(bt_series) > 0 and max(bt_series.values()) >= FEVER_THRESHOLD_C)

    wbc_series = windowed_series(labs_df, sid, "wbc", lo, hi)
    wbc_abnormal_flag = 0.0
    if wbc_series:
        wbc_vals = list(wbc_series.values())
        wbc_abnormal_flag = float(any(v < WBC_NORMAL_RANGE[0] or v > WBC_NORMAL_RANGE[1] for v in wbc_vals))

    crp_series = windowed_series(labs_df, sid, "crp", lo, hi)
    crp_elevated_flag = 0.0
    if crp_series and CRP_ELEVATED_THRESHOLD is not None:
        crp_elevated_flag = float(max(crp_series.values()) >= CRP_ELEVATED_THRESHOLD)

    composite = chapter_i_flag + high_risk_flag + fever_flag + wbc_abnormal_flag + crp_elevated_flag

    return {
        "infection_chapter_i_flag": chapter_i_flag,
        "infection_high_risk_code_flag": high_risk_flag,
        "infection_fever_flag": fever_flag,
        "infection_wbc_abnormal_flag": wbc_abnormal_flag,
        "infection_crp_elevated_flag": crp_elevated_flag,
        "infection_composite_score": composite,   # 0-5, a simple count of the above
    }

INFECTION_FEATURES = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    INFECTION_FEATURES[sid] = infection_inflammation_features(sid, lo, hi)

infection_features_df = pd.DataFrame(INFECTION_FEATURES).T
infection_features_df.index.name = "subject_id"
print(f"\nPatients with infection_composite_score > 0: "
      f"{(infection_features_df['infection_composite_score'] > 0).sum()} / {len(infection_features_df)}")
infection_features_df.describe()

# %% [markdown]
# ## 6.10 Operation-count-in-same-area features (§1.6.4)
#
# For the **last** operation (the one the label is computed relative to, §4.2), count how
# many *prior* operations that patient had in the **same department** and compute the time
# gap in days since the most recent one. This is offered to the model as a feature, on
# purpose — unlike the exception in §6.11 below, whether repeat-same-area surgery predicts
# higher or lower risk is exactly the kind of question this notebook's own framing (§1.6.4)
# says should be *learned*, not assumed.

# %%
def operation_count_features(sid):
    history = OPS_HISTORY[sid]   # sorted by orin_time, built in §4.3
    if len(history) <= 1:
        return {"n_prior_ops_same_dept": 0, "days_since_last_op_same_dept": np.nan,
                "n_prior_ops_any_dept": 0, "days_since_last_op_any_dept": np.nan}
    last_op = history[-1]
    prior_ops = history[:-1]
    same_dept = [op for op in prior_ops if op.get("department") == last_op.get("department")]

    def days_since(ops):
        if not ops:
            return np.nan
        most_recent = max(op["orin_time"] for op in ops if op.get("orin_time") is not None)
        return (last_op["orin_time"] - most_recent) / (24 * 60)

    return {
        "n_prior_ops_same_dept": len(same_dept),
        "days_since_last_op_same_dept": days_since(same_dept),
        "n_prior_ops_any_dept": len(prior_ops),
        "days_since_last_op_any_dept": days_since(prior_ops),
    }

OP_COUNT_FEATURES = {sid: operation_count_features(sid) for sid in cohort_df["subject_id"]}
op_count_df = pd.DataFrame(OP_COUNT_FEATURES).T
op_count_df.index.name = "subject_id"
print(f"Patients with >=1 prior operation in the same department as their last: "
      f"{(op_count_df['n_prior_ops_same_dept'] > 0).sum()} / {len(op_count_df)}")
op_count_df.describe()

# %% [markdown]
# ## 6.11 The 6-month post-cardiovascular-surgery exception (§1.6.4)
#
# Implemented as a **rule**, deliberately not a learned feature, for the reasons argued in
# §1.6.4 (sample size + actionability). `CTS` (cardiothoracic surgery) is used as this
# dataset's department proxy for "cardiovascular surgery" — check this mapping against your
# site's actual department coding before trusting the flag on a new cohort.

# %%
CARDIAC_DEPARTMENTS = {"CTS"}       # adjust if your site's department coding differs
MIN_RECOVERY_DAYS_AFTER_CARDIAC_SURGERY = 180   # ~6 months, the clinical protocol window

def cardiac_recovery_exception_flag(sid):
    # 1.0 if the patient's last operation occurred less than 180 days after a PRIOR
    # cardiovascular-department operation -- i.e. the protocol window was violated (or the
    # two operations are causally linked, e.g. a staged/urgent re-intervention). This is a
    # flag for the model and for manual review, not an automatic exclusion -- see §11.6 for
    # the follow-up qualitative check this notebook recommends running once the model exists.
    history = OPS_HISTORY[sid]
    if len(history) <= 1:
        return 0.0
    last_op = history[-1]
    prior_cardiac_ops = [op for op in history[:-1] if op.get("department") in CARDIAC_DEPARTMENTS]
    if not prior_cardiac_ops:
        return 0.0
    most_recent_cardiac = max(op["orin_time"] for op in prior_cardiac_ops if op.get("orin_time") is not None)
    gap_days = (last_op["orin_time"] - most_recent_cardiac) / (24 * 60)
    return float(gap_days < MIN_RECOVERY_DAYS_AFTER_CARDIAC_SURGERY)

CARDIAC_EXCEPTION_FLAG = {sid: cardiac_recovery_exception_flag(sid) for sid in cohort_df["subject_id"]}
n_flagged = sum(CARDIAC_EXCEPTION_FLAG.values())
print(f"Patients flagged (last op within 6 months of a prior CTS op): {int(n_flagged)} / {len(CARDIAC_EXCEPTION_FLAG)}")
print("(Low or zero counts are expected on this 30-patient dev subset -- re-run on the full "
      "cohort, where cardiothoracic re-intervention is far more likely to appear.)")

# %% [markdown]
# ## 6.12 Static features (§1.5.1, §1.2.2)
#
# Age, sex, ASA, `emop` (emergency-operation flag), department (one-hot), weight, height —
# 100% coverage, cheap, and (per §1.2.2/§1.6.1) currently unused by the DNN in the source
# repo despite ASA being one of the strongest established predictors in the literature.
# Included here by default via `CONFIG['INCLUDE_STATIC_ASA']`.

# %%
def static_features_row(row):
    return {
        "age": row["age"],
        "sex_F": float(row["sex"] == "F"),
        "asa": row["asa"] if CONFIG["INCLUDE_STATIC_ASA"] else np.nan,
        "emop": row["emop"],
        "weight": row["weight"],
        "height": row["height"],
        "n_operations_total": row["n_operations"],
    }

static_df = cohort_df.set_index("subject_id").apply(static_features_row, axis=1, result_type="expand")
dept_onehot = pd.get_dummies(cohort_df.set_index("subject_id")["department"], prefix="dept").astype(float)
static_df = pd.concat([static_df, dept_onehot], axis=1)
static_df.describe()

# %% [markdown]
# ## 6.13 Medications → ATC-level aggregate features (§1.5.3)
#
# Per §1.5.3's decision, this notebook aggregates at the **WHO ATC level-2** (e.g. the first
# 3 characters of the ATC code, roughly the "therapeutic subgroup" level) rather than
# per-drug — far fewer categories, each with enough examples in a small cohort to be a
# usable count feature rather than a near-empty one-hot. A learned per-class embedding
# (rather than a raw count) is a natural upgrade once the full cohort is available — flagged
# as an extension, not built by default here, per §1.5.3's reasoning about label-starved
# high-cardinality embeddings.

# %%
def atc_level2(code):
    if not isinstance(code, str) or len(code) < 3:
        return None
    return code[:3].upper()

def medication_features(sid, lo, hi):
    meds = meds_in_window(sid, lo, hi)
    atc2 = meds["atc_code"].dropna().apply(atc_level2)
    return {
        "n_medication_administrations": int(len(meds)),
        "n_distinct_atc2_classes": int(atc2.nunique()),
        "n_distinct_drug_names": int(meds["drug_name"].nunique()) if "drug_name" in meds else 0,
    }

MED_FEATURES = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    MED_FEATURES[sid] = medication_features(sid, lo, hi)

med_features_df = pd.DataFrame(MED_FEATURES).T
med_features_df.index.name = "subject_id"
med_features_df.describe()

# %% [markdown]
# ## 6.15 Aggregated time-series-derived static features (Part C §4 of the imbalance/imputation reference)
#
# The second external review's most concretely useful suggestion: give the static branch
# richer clinical summaries of the time series, rather than only demographics/ICD-10/
# operation-history. Two benefits at once — it enriches what the DNN's static branch has to
# work with, **and** it gives §8's grouped SMOTENC more informative features to interpolate
# over. For each patient, within the same window as §6.2 (`get_window`):
#
# - **Mean / min / max / std** of six clinically load-bearing signals: `creatinine`,
#   `potassium`, `glucose`, `wbc`, `lactate`, `hb` (labs) and `hr`, `nibp_mbp`, `spo2`
#   (ward/intra-op vitals) — 9 signals × 4 statistics = 36 features.
# - **Severe-hypotension reading count** — number of mean-arterial-pressure readings below
#   65 mmHg in the window. Stated precisely as a **reading count, not a duration**: INSPIRE's
#   vitals are irregularly sampled, so "minutes spent hypotensive" would need an assumption
#   about how long each reading represents, which this notebook avoids asserting.
# - **Vasopressor administration count** and **high-alert medication administration count**
#   — INSPIRE's `medications` table has **no dose field** (checked directly against the
#   schema in §3), so the reviewed suggestion's "cumulative mcg dose" isn't computable as
#   asked. This substitutes an **administration count** for a fixed drug-name/ATC keyword
#   list — a real, honestly-weaker proxy (it can't distinguish one large dose from several
#   small ones, or detect a true *escalation* in dose over time), stated here rather than
#   silently implied to be the same thing.

# %%
AGG_LABS = ["creatinine", "potassium", "glucose", "wbc", "lacate", "hb"]     # 'lacate' matches this dataset's item_name spelling (see §4.1)
AGG_WARD_VITALS = ["hr", "nibp_mbp", "spo2"]
SEVERE_HYPOTENSION_MAP_THRESHOLD = 65.0   # mmHg, a standard clinical cutoff for organ-perfusion risk

# Fixed keyword lists rather than a formal drug ontology lookup -- simple, auditable, and
# easy to extend; matched case-insensitively against drug_name (route/ATC not required).
VASOPRESSOR_KEYWORDS = ["norepinephrine", "noradrenaline", "epinephrine", "adrenaline",
                         "dopamine", "dobutamine", "vasopressin", "phenylephrine"]
HIGH_ALERT_KEYWORDS = VASOPRESSOR_KEYWORDS + ["fentanyl", "propofol", "midazolam", "heparin", "insulin"]

def _series_stats(chart_time2value, prefix):
    if not chart_time2value:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan, f"{prefix}_std": np.nan}
    vals = np.array(list(chart_time2value.values()), dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
        f"{prefix}_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
    }

def aggregate_features_for_patient(sid, lo, hi):
    feats = {}
    for lab in AGG_LABS:
        series = windowed_series(labs_df, sid, lab, lo, hi)
        feats.update(_series_stats(series, f"agg_{lab}"))
    for vital in AGG_WARD_VITALS:
        series = windowed_series(ward_vitals_df, sid, vital, lo, hi)
        feats.update(_series_stats(series, f"agg_{vital}"))
        if CONFIG["TIME_WINDOW"] == "peri_op" and vital in SYSTEM_INTRAOP_VITALS.get("cardiovascular", []) + SYSTEM_INTRAOP_VITALS.get("respiratory", []):
            iv_series = windowed_series(vitals_df, sid, vital, lo, hi)
            for t, v in iv_series.items():
                series.setdefault(t, v)

    map_series = windowed_series(ward_vitals_df, sid, "nibp_mbp", lo, hi)
    feats["severe_hypotension_reading_count"] = int(sum(1 for v in map_series.values() if v < SEVERE_HYPOTENSION_MAP_THRESHOLD))

    meds = meds_in_window(sid, lo, hi)
    if len(meds) == 0:
        feats["vasopressor_administration_count"] = 0
        feats["high_alert_med_administration_count"] = 0
    else:
        # .astype(object) (not .astype(str)) deliberately -- on some pandas versions,
        # .astype(str) on an EMPTY series produces the newer pandas "str" extension dtype,
        # whose .sum() concatenates strings ('') instead of summing booleans numerically.
        # object dtype sidesteps that entirely and behaves the same either way.
        drug_names_lower = meds["drug_name"].astype(object).str.lower()
        feats["vasopressor_administration_count"] = int(drug_names_lower.apply(lambda n: any(k in n for k in VASOPRESSOR_KEYWORDS)).astype(bool).sum())
        feats["high_alert_med_administration_count"] = int(drug_names_lower.apply(lambda n: any(k in n for k in HIGH_ALERT_KEYWORDS)).astype(bool).sum())

    return feats

AGG_FEATURES = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    AGG_FEATURES[sid] = aggregate_features_for_patient(sid, lo, hi)

agg_features_df = pd.DataFrame(AGG_FEATURES).T
agg_features_df.index.name = "subject_id"
print(f"Aggregated features built: {agg_features_df.shape[1]} new static features "
      f"(36 mean/min/max/std + 3 count-based)")
print(f"Patients with >=1 severe-hypotension reading: "
      f"{(agg_features_df['severe_hypotension_reading_count'] > 0).sum()} / {len(agg_features_df)}")
print(f"Patients with >=1 vasopressor administration in-window: "
      f"{(agg_features_df['vasopressor_administration_count'] > 0).sum()} / {len(agg_features_df)}")
agg_features_df.describe().T[["mean", "std", "min", "max"]].head(10)
