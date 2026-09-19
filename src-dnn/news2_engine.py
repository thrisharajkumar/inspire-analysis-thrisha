"""
NEWS2 (National Early Warning Score 2) -- Royal College of Physicians, 2017 revision.
Standard UK clinical deterioration score, computed here from parameters already in the
organ-system mapping (respiratory rate, SpO2, supplemental O2 use, temperature,
systolic BP, heart rate, level of consciousness), so it slots directly onto existing
ward_vitals/vitals data without needing anything not already collected.

Two outputs, both useful for different purposes:
  - `news2_score()`: the score at a single timepoint -- usable as a time-varying feature.
  - `news2_summary_features()`: max/latest/trend over a patient's stay -- usable as a
    static feature, same treatment as your other aggregated summaries (§6.15 pattern).

Reference: Royal College of Physicians. National Early Warning Score (NEWS) 2:
Standardising the assessment of acute-illness severity in the NHS. London: RCP, 2017.
"""

import numpy as np
import pandas as pd


def _band_respiratory_rate(rr):
    if pd.isna(rr):
        return np.nan
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def _band_spo2_scale1(spo2):
    """Scale 1 -- most patients, no target range for chronic hypercapnic respiratory failure."""
    if pd.isna(spo2):
        return np.nan
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def _band_supplemental_o2(on_oxygen):
    if pd.isna(on_oxygen):
        return np.nan
    return 2 if bool(on_oxygen) else 0


def _band_temperature(temp_c):
    if pd.isna(temp_c):
        return np.nan
    if temp_c <= 35.0:
        return 3
    if temp_c <= 36.0:
        return 1
    if temp_c <= 38.0:
        return 0
    if temp_c <= 39.0:
        return 1
    return 2


def _band_systolic_bp(sbp):
    if pd.isna(sbp):
        return np.nan
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def _band_heart_rate(hr):
    if pd.isna(hr):
        return np.nan
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def _band_consciousness(gcs_total):
    """NEWS2 uses AVPU, not GCS directly -- GCS 15 (fully alert) maps to AVPU 'Alert'
    (band 0); anything below 15 is treated as a reduced level of consciousness (band 3),
    which is a coarser mapping than true AVPU but the only one derivable from the GCS
    components already collected here (gcs_e/gcs_m/gcs_v). Flagged as an approximation,
    not the literal NEWS2 AVPU assessment -- worth confirming with your surgeon whether
    this substitution is acceptable or whether a real AVPU field should be sourced instead."""
    if pd.isna(gcs_total):
        return np.nan
    return 0 if gcs_total >= 15 else 3


def news2_score(rr=None, spo2=None, on_oxygen=None, temp_c=None, sbp=None, hr=None, gcs_total=None):
    """
    Computes NEWS2 at a single timepoint from the given parameters (any subset may be
    missing -- np.nan for a missing sub-score is excluded from the total, and the total
    is np.nan only if ALL sub-scores are missing).

    Returns: dict with 'total', each sub-score, and 'risk_category'
    ('low'/'low-medium'/'medium'/'high', per RCP banding on the total plus the
    any-single-parameter-scoring-3 rule).
    """
    sub_scores = {
        "resp_rate": _band_respiratory_rate(rr),
        "spo2": _band_spo2_scale1(spo2),
        "supplemental_o2": _band_supplemental_o2(on_oxygen),
        "temperature": _band_temperature(temp_c),
        "systolic_bp": _band_systolic_bp(sbp),
        "heart_rate": _band_heart_rate(hr),
        "consciousness": _band_consciousness(gcs_total),
    }
    valid_scores = [v for v in sub_scores.values() if not pd.isna(v)]
    if not valid_scores:
        total = np.nan
        risk_category = "unknown"
    else:
        total = float(np.sum(valid_scores))
        any_single_3 = any(v == 3 for v in valid_scores)
        # RCP banding: total >=7 -> high; total 5-6 OR any single parameter scoring 3 ->
        # medium; total 1-4 -> low-medium; total 0 -> low.
        if total >= 7:
            risk_category = "high"
        elif total >= 5 or any_single_3:
            risk_category = "medium"
        elif total >= 1:
            risk_category = "low-medium"
        else:
            risk_category = "low"

    return {"total": total, "risk_category": risk_category, **sub_scores}


def news2_series_for_patient(patient_long_df, phase_filter=None):
    """
    Computes NEWS2 at every timepoint where enough readings coincide, from a patient's
    long-format frame (columns: feature, value, chart_time, phase -- matching
    data/loader.py's build_phase_tagged_frame output). Readings are matched to the
    nearest prior reading of each other required feature (forward-fill within a
    tolerance window), since NEWS2's inputs are rarely all charted at the exact same
    second in real data.

    phase_filter: optional, e.g. "pre" to restrict to pre-op NEWS2 only.
    Returns: DataFrame with columns [chart_time, total, risk_category, <sub-scores>]
    """
    NEWS2_FEATURE_MAP = {
        "rr": "rr", "spo2": "spo2", "fio2": "on_oxygen",  # fio2 > 21% treated as "on oxygen"
        "bt": "temp_c", "nibp_sbp": "sbp", "hr": "hr",
    }
    df = patient_long_df[patient_long_df["feature"].isin(NEWS2_FEATURE_MAP.keys())].copy()
    if phase_filter is not None:
        df = df[df["phase"] == phase_filter]
    if df.empty:
        return pd.DataFrame(columns=["chart_time", "total", "risk_category"])

    df["news2_field"] = df["feature"].map(NEWS2_FEATURE_MAP)
    wide = df.pivot_table(index="chart_time", columns="news2_field", values="value", aggfunc="first")
    wide = wide.sort_index().ffill()  # carry the last known reading forward to each timepoint

    if "on_oxygen" in wide.columns:
        wide["on_oxygen"] = wide["on_oxygen"] > 21.0  # FiO2 > room air (21%) => on supplemental O2

    gcs = patient_long_df[
        (patient_long_df["feature"].isin(["gcs_e", "gcs_m", "gcs_v"]))
        & (phase_filter is None or patient_long_df["phase"] == phase_filter)
    ]
    gcs_total_by_time = (
        gcs.pivot_table(index="chart_time", columns="feature", values="value", aggfunc="first")
        .sum(axis=1, min_count=1)
        if not gcs.empty else pd.Series(dtype=float)
    )

    results = []
    for chart_time, row in wide.iterrows():
        gcs_total = gcs_total_by_time.asof(chart_time) if len(gcs_total_by_time) else np.nan
        score = news2_score(
            rr=row.get("rr"), spo2=row.get("spo2"), on_oxygen=row.get("on_oxygen"),
            temp_c=row.get("temp_c"), sbp=row.get("sbp"), hr=row.get("hr"), gcs_total=gcs_total,
        )
        results.append({"chart_time": chart_time, **score})

    return pd.DataFrame(results)


def news2_summary_features(patient_long_df, phase_filter=None):
    """
    Static summary features for one patient, for use in the static feature table
    (same treatment as your other aggregated time-series summaries, §6.15 pattern).
    Returns a dict: news2_max, news2_latest, news2_mean, news2_trend_slope,
    news2_ever_high_risk.
    """
    series = news2_series_for_patient(patient_long_df, phase_filter)
    valid = series.dropna(subset=["total"])
    if valid.empty:
        return {
            "news2_max": np.nan, "news2_latest": np.nan, "news2_mean": np.nan,
            "news2_trend_slope": np.nan, "news2_ever_high_risk": 0.0,
        }

    slope = np.nan
    if len(valid) >= 2:
        t = valid["chart_time"].values.astype(float)
        t = (t - t.min())
        slope = float(np.polyfit(t, valid["total"].values, 1)[0]) if t.max() > 0 else 0.0

    return {
        "news2_max": float(valid["total"].max()),
        "news2_latest": float(valid.sort_values("chart_time")["total"].iloc[-1]),
        "news2_mean": float(valid["total"].mean()),
        "news2_trend_slope": slope,
        "news2_ever_high_risk": float((valid["risk_category"] == "high").any()),
    }
