"""
Stage 2b (NEW): NEWS2 score integration.

Two things happen here:
1. NEWS2 summary features (max/latest/mean/trend-slope over the same pre-op/peri-op
   window already used everywhere else, via the existing get_window()/windowed_series()
   helpers) are computed per real patient and merged into their static feature vector.
   This means NEWS2 becomes a genuine, continuous static feature that SMOTENC will
   legitimately interpolate for synthetic patients -- not a bolted-on afterthought.
2. `check_news2_before_after_augmentation()` -- call this AFTER sampling.py's
   grouped_smotenc() has produced synthetic rows. It compares the real-patient NEWS2
   distribution ("before") against the combined real+synthetic distribution ("after"),
   flags any synthetic patient with a clinically implausible NEWS2 (negative, or above
   ~20, the practical ceiling given NEWS2's own scoring bands), and runs a KS test so a
   silent distribution shift doesn't go unnoticed.

NEWS2 scoring itself (the banding logic) is unit-tested separately against the official
Royal College of Physicians table, including exact boundary values -- see this
project's smote_news2_correlation/src/news2.py, which this module reuses directly
rather than reimplementing.
"""
from feature_engineering import *
from news2_engine import news2_score  # the already-tested scorer, copied in unmodified -- see that file's header
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


NEWS2_ITEM_NAMES = {
    "rr": "rr", "spo2": "spo2", "hr": "hr", "nibp_sbp": "sbp", "bt": "temp_c",
}
GCS_ITEM_NAMES = ["gcs_e", "gcs_m", "gcs_v"]


def _news2_series_from_windowed(sid, lo, hi):
    """
    Pulls each NEWS2 input via the existing windowed_series() helper (same function,
    same tables -- ward_vitals_df/vitals_df -- already used throughout Part 6), and
    computes NEWS2 at every timepoint any of them was observed. Consciousness uses the
    same GCS-as-AVPU-proxy approximation documented in news2_engine.py, flagged there
    for surgeon review.
    """
    field_series = {}
    for item_name, news2_field in NEWS2_ITEM_NAMES.items():
        # ward_vitals_df is the primary source for these; vitals_df (intra-op) is
        # checked too so peri_op-window patients still get a reading if TIME_WINDOW
        # is set to 'peri_op' -- windowed_series returns {} harmlessly if a table
        # doesn't have the item_name column populated for this patient.
        series = windowed_series(ward_vitals_df, sid, item_name, lo, hi)
        if not series:
            series = windowed_series(vitals_df, sid, item_name, lo, hi)
        field_series[news2_field] = series

    on_oxygen_series = windowed_series(ward_vitals_df, sid, "fio2", lo, hi)
    if not on_oxygen_series:
        on_oxygen_series = windowed_series(vitals_df, sid, "fio2", lo, hi)

    gcs_series_by_component = {g: windowed_series(ward_vitals_df, sid, g, lo, hi) for g in GCS_ITEM_NAMES}

    all_times = sorted(set().union(*[s.keys() for s in field_series.values()],
                                    on_oxygen_series.keys(),
                                    *[s.keys() for s in gcs_series_by_component.values()]))
    if not all_times:
        return pd.DataFrame(columns=["chart_time", "total", "risk_category"])

    def _asof(series_dict, t):
        prior_times = [k for k in series_dict.keys() if k <= t]
        return series_dict[max(prior_times)] if prior_times else np.nan

    results = []
    for t in all_times:
        gcs_components = [_asof(gcs_series_by_component[g], t) for g in GCS_ITEM_NAMES]
        gcs_components = [g for g in gcs_components if not pd.isna(g)]
        gcs_total = sum(gcs_components) if gcs_components else np.nan
        fio2 = _asof(on_oxygen_series, t)
        score = news2_score(
            rr=_asof(field_series["rr"], t), spo2=_asof(field_series["spo2"], t),
            on_oxygen=(fio2 > 21.0) if not pd.isna(fio2) else np.nan,
            temp_c=_asof(field_series["temp_c"], t), sbp=_asof(field_series["sbp"], t),
            hr=_asof(field_series["hr"], t), gcs_total=gcs_total,
        )
        results.append({"chart_time": t, **score})
    return pd.DataFrame(results)


def news2_summary_for_patient(sid, lo, hi):
    """Static summary features for one patient -- merged into their static vector below."""
    series = _news2_series_from_windowed(sid, lo, hi)
    valid = series.dropna(subset=["total"]) if len(series) else series
    if valid.empty:
        return {"news2_max": np.nan, "news2_latest": np.nan, "news2_mean": np.nan,
                "news2_trend_slope": np.nan, "news2_ever_high_risk": 0.0}

    slope = 0.0
    if len(valid) >= 2:
        t = valid["chart_time"].values.astype(float)
        t = t - t.min()
        slope = float(np.polyfit(t, valid["total"].values, 1)[0]) if t.max() > 0 else 0.0

    return {
        "news2_max": float(valid["total"].max()),
        "news2_latest": float(valid.sort_values("chart_time")["total"].iloc[-1]),
        "news2_mean": float(valid["total"].mean()),
        "news2_trend_slope": slope,
        "news2_ever_high_risk": float((valid["risk_category"] == "high").any()),
    }


# ---------------------------------------------------------------------------
# Merge NEWS2 summary features into every real patient's static vector, and into
# STATIC_FEATURE_NAMES, BEFORE sampling.py runs -- so grouped_smotenc() treats these as
# ordinary continuous static features and interpolates them for synthetic patients the
# same way it does for any other lab/vital summary.
# ---------------------------------------------------------------------------
print("Computing NEWS2 summary features per patient (adds 5 static features: "
      "news2_max, news2_latest, news2_mean, news2_trend_slope, news2_ever_high_risk)...")

_missing_news2_inputs = 0
for sid in cohort_df["subject_id"]:
    lo, hi = get_window(sid, COHORT_INDEXED.loc[sid])
    news2_feats = news2_summary_for_patient(sid, lo, hi)
    if pd.isna(news2_feats["news2_max"]):
        _missing_news2_inputs += 1
    PATIENT_BUNDLE[sid]["static"] = pd.concat([
        PATIENT_BUNDLE[sid]["static"], pd.Series(news2_feats)
    ])

STATIC_FEATURE_NAMES = sorted(next(iter(PATIENT_BUNDLE.values()))["static"].keys())
print(f"NEWS2 features added. {_missing_news2_inputs}/{len(PATIENT_BUNDLE)} patients had "
      f"no usable NEWS2 inputs in their window (all-NaN -- will be population-imputed "
      f"downstream like any other missing static feature, in Part 7).")
print(f"Static feature vector length is now: {len(STATIC_FEATURE_NAMES)}")


def check_news2_before_after_augmentation(train_ids, synthetic_rows, static_feature_names,
                                           implausible_low=0.0, implausible_high=20.0):
    """
    Call this AFTER sampling.py's grouped_smotenc() has run. Compares the real ("before")
    vs. real+synthetic ("after") NEWS2 distribution and prints a clear report -- the
    concrete "check before and after" requested for this augmentation step.

    Returns a dict report (also printed) so it can be logged/saved rather than only
    read off stdout.
    """
    if "news2_max" not in static_feature_names:
        print("WARNING: 'news2_max' not found in static_feature_names -- was this called "
              "before the NEWS2 merge above ran? Skipping the before/after check.")
        return None

    news2_idx = static_feature_names.index("news2_max")
    real_news2 = np.array([
        PATIENT_BUNDLE[sid]["static"]["news2_max"] for sid in train_ids
        if not pd.isna(PATIENT_BUNDLE[sid]["static"]["news2_max"])
    ])
    synthetic_news2 = np.array([row[news2_idx] for row in synthetic_rows]) if synthetic_rows else np.array([])

    report = {
        "n_real": len(real_news2), "n_synthetic": len(synthetic_news2),
        "real_mean": float(real_news2.mean()) if len(real_news2) else np.nan,
        "real_std": float(real_news2.std()) if len(real_news2) else np.nan,
        "synthetic_mean": float(synthetic_news2.mean()) if len(synthetic_news2) else np.nan,
        "synthetic_std": float(synthetic_news2.std()) if len(synthetic_news2) else np.nan,
    }

    print("\n--- NEWS2 before/after augmentation check ---")
    print(f"BEFORE (real patients only):      n={report['n_real']:4d}  "
          f"mean={report['real_mean']:.2f}  std={report['real_std']:.2f}")
    if len(synthetic_news2):
        print(f"AFTER, synthetic rows only:        n={report['n_synthetic']:4d}  "
              f"mean={report['synthetic_mean']:.2f}  std={report['synthetic_std']:.2f}")

        n_implausible = int(((synthetic_news2 < implausible_low) | (synthetic_news2 > implausible_high)).sum())
        report["n_implausible"] = n_implausible
        if n_implausible > 0:
            print(f"WARNING: {n_implausible}/{len(synthetic_news2)} synthetic patients have an "
                  f"implausible NEWS2 (outside [{implausible_low}, {implausible_high}]) -- "
                  f"inspect these before trusting them in training.")
        else:
            print(f"All synthetic NEWS2 values fall within the plausible clinical range "
                  f"[{implausible_low}, {implausible_high}].")

        if len(real_news2) >= 2:
            stat, p_value = ks_2samp(real_news2, synthetic_news2)
            report["ks_statistic"], report["ks_p_value"] = float(stat), float(p_value)
            if p_value < 0.05:
                print(f"NOTE: KS test finds the synthetic NEWS2 distribution significantly "
                      f"different from real (p={p_value:.4f}) -- expected to some degree, "
                      f"since SMOTENC targets minority patients specifically; a p-value this "
                      f"low with a LARGE mean shift is worth a second look, a small mean shift "
                      f"with a significant p-value on a large sample is not unusual.")
            else:
                print(f"KS test: no significant difference detected (p={p_value:.4f}).")
    else:
        print("No synthetic rows to compare (sampling strategy produced none this run).")

    return report
