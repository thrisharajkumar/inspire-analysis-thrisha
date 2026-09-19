"""
Stage 2: organ-system feature engineering -- the routing map, pre-op/peri-op window
extraction, per-system time-series extraction and resampling, the existing
cardiovascular<->renal coupling (kept as-is, see system_correlation_layer.py for the
new general-purpose addition), ICD-10/HFRS/GI/MSK features, operation-count features,
static features, medication ATC aggregates, and the final per-patient PATIENT_BUNDLE.
Converted from cells 28-58, preserved verbatim (no logic changes in this stage).
"""
from data_loading import *

# --- from notebook cell 30 ---
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

# --- from notebook cell 32 ---
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

# --- from notebook cell 34 ---
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

# --- from notebook cell 36 ---
def resample_to_grid(chart_time2value, lo, hi, n_points):
    # Evenly-spaced target grid of n_points between lo and hi; nearest-observation lookup
    # per grid point, returned alongside a raw (unimputed) array with NaN for empty points.
    # Actual interpolation/imputation happens in §7 -- this function only regularises timing.
    grid = np.linspace(lo, hi, n_points)
    if len(chart_time2value) == 0:
        return grid, np.full(n_points, np.nan)
    times = np.array(sorted(chart_time2value.keys()), dtype=float)
    values = np.array([chart_time2value[t] for t in sorted(chart_time2value.keys())], dtype=float)
    out = np.full(n_points, np.nan)
    for i, g in enumerate(grid):
        idx = np.searchsorted(times, g)
        # nearest of the two neighbouring observed points, used later as the "observed value"
        # a real imputation strategy (§7) then fills the true gaps.
        candidates = []
        if idx < len(times):
            candidates.append((abs(times[idx] - g), values[idx]))
        if idx > 0:
            candidates.append((abs(times[idx - 1] - g), values[idx - 1]))
        if candidates:
            # only accept "nearest observed" as a real observation if it's within one grid step,
            # otherwise leave it NaN so §7's imputation (not this raw resample) fills the gap
            step = (hi - lo) / max(n_points - 1, 1)
            best = min(candidates, key=lambda c: c[0])
            if best[0] <= max(step, 1e-6):
                out[i] = best[1]
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

# --- from notebook cell 38 ---
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

# --- from notebook cell 39 ---
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

# --- from notebook cell 41 ---
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
    dx = diagnoses_df[(diagnoses_df["subject_id"] == sid) &
                       (diagnoses_df["chart_time"] >= lo - lookback_minutes) &
                       (diagnoses_df["chart_time"] <= hi)]
    score = 0.0
    for code in dx["icd10_cm"].dropna():
        code3 = str(code)[:3].upper()
        if code3 in HFRS_WEIGHTS:
            score += HFRS_WEIGHTS[code3]
    return score

def gi_msk_icd10_features(sid, lo, hi, department):
    dx = diagnoses_df[(diagnoses_df["subject_id"] == sid) &
                       (diagnoses_df["chart_time"] >= lo) & (diagnoses_df["chart_time"] <= hi)]
    chapters = dx["icd10_cm"].dropna().apply(icd10_chapter)
    n_chapters = chapters.value_counts()
    return {
        "gi_icd10_count":  int(n_chapters.get("XI", 0)),
        "gi_icd10_flag":   float(n_chapters.get("XI", 0) > 0),
        "gi_department_flag": float(department == "GS"),
        "msk_icd10_count": int(n_chapters.get("XIII", 0)),
        "msk_icd10_flag":  float(n_chapters.get("XIII", 0) > 0),
        "msk_department_flag": float(department == "OS"),
        "n_diagnoses_in_window": int(len(dx)),
        "n_distinct_chapters_in_window": int(chapters.nunique()),
    }

ICD10_FEATURES = {}
for sid in cohort_df["subject_id"]:
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    feats = gi_msk_icd10_features(sid, lo, hi, row["department"])
    if CONFIG["INCLUDE_HFRS"]:
        feats["hfrs"] = compute_hfrs(sid, lo, hi, row["age"])
    ICD10_FEATURES[sid] = feats

icd10_features_df = pd.DataFrame(ICD10_FEATURES).T
icd10_features_df.index.name = "subject_id"
print(f"GI flag positive: {int(icd10_features_df['gi_icd10_flag'].sum())} / {len(icd10_features_df)} patients")
print(f"MSK flag positive: {int(icd10_features_df['msk_icd10_flag'].sum())} / {len(icd10_features_df)} patients")
if CONFIG["INCLUDE_HFRS"]:
    n_scored = int((icd10_features_df["hfrs"] > 0).sum())
    print(f"Patients with HFRS > 0 (i.e. aged 75+ with a matching code): {n_scored} / {len(icd10_features_df)}")
icd10_features_df.describe()

# --- from notebook cell 43 ---
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
    dx = diagnoses_df[(diagnoses_df["subject_id"] == sid) &
                       (diagnoses_df["chart_time"] >= lo) & (diagnoses_df["chart_time"] <= hi)]
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

# --- from notebook cell 45 ---
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

# --- from notebook cell 47 ---
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

# --- from notebook cell 49 ---
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

# --- from notebook cell 51 ---
def atc_level2(code):
    if not isinstance(code, str) or len(code) < 3:
        return None
    return code[:3].upper()

def medication_features(sid, lo, hi):
    meds = medications_df[(medications_df["subject_id"] == sid) &
                           (medications_df["chart_time"] >= lo) & (medications_df["chart_time"] <= hi)]
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

# --- from notebook cell 53 ---
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

    meds = medications_df[(medications_df["subject_id"] == sid) &
                           (medications_df["chart_time"] >= lo) & (medications_df["chart_time"] <= hi)]
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

# --- from notebook cell 55 ---
def build_static_vector(sid):
    parts = {}
    parts.update(static_df.loc[sid].to_dict())
    parts.update(icd10_features_df.loc[sid].to_dict())
    parts.update(infection_features_df.loc[sid].to_dict())   # §6.6, roadmap §4.1a -- new this revision
    parts.update(op_count_df.loc[sid].to_dict())
    parts["cardiac_recovery_exception"] = CARDIAC_EXCEPTION_FLAG[sid]
    parts.update(cardiac_summary_df.loc[sid].to_dict())
    parts["renal_cardiac_interaction"] = RENAL_CARDIAC_INTERACTION[sid]
    parts.update(med_features_df.loc[sid].to_dict())
    parts.update(AGG_FEATURES[sid])   # §6.15, Part C §4 of the imbalance/imputation reference
    return parts

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

# --- from notebook cell 56 ---
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

# --- from notebook cell 57 ---
# Quick label sanity check on the assembled bundle before moving to §7.
labels = np.array([PATIENT_BUNDLE[sid]["label"] for sid in cohort_df["subject_id"]])
print(f"Positive rate in bundle: {labels.mean():.1%}  ({int(labels.sum())} / {len(labels)})")
print("This dev-subset rate (~23%) is NOT representative of the full-cohort deployment rate "
      "(~0.5-0.9%, per §1.4) -- keep this in mind when reading any metric later in the "
      "notebook as a sanity check, not a final result.")

# --- from notebook cell 58 ---
record_checkpoint("Parts 4-6 -- labelling + organ-system feature engineering")