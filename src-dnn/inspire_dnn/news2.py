# NEWS2 -- National Early Warning Score 2 (Royal College of Physicians, 2017).
#
# Vectorised numpy implementation: every band function takes a scalar OR an array and
# returns float scores with NaN wherever the input is missing, so one code path serves
# single readings, whole observation sets, and the resampled model grids used by the
# augmentation check.
#
# Scoring table implemented (RCP NEWS2 chart, SpO2 Scale 1):
#   Respiration rate   <=8:3 | 9-11:1 | 12-20:0 | 21-24:2 | >=25:3
#   SpO2 (Scale 1)     <=91:3 | 92-93:2 | 94-95:1 | >=96:0
#   Air or oxygen      oxygen:2 | air:0
#   Systolic BP        <=90:3 | 91-100:2 | 101-110:1 | 111-219:0 | >=220:3
#   Pulse              <=40:3 | 41-50:1 | 51-90:0 | 91-110:1 | 111-130:2 | >=131:3
#   Consciousness      Alert:0 | new confusion / V / P / U:3
#   Temperature        <=35.0:3 | 35.1-36.0:1 | 36.1-38.0:0 | 38.1-39.0:1 | >=39.1:2
# Aggregate 0-20. Clinical response bands: 0-4 low, any single parameter scoring 3
# "low-medium" (urgent ward review), 5-6 medium, >=7 high.
#
# Documented approximations for THIS dataset (flagged for clinical sign-off):
#   * Scale 1 is used for everyone -- Scale 2 needs a clinician-set hypercapnic target
#     range that the dataset does not record.
#   * Consciousness: AVPU is not recorded, so GCS is used -- "Alert" only if every
#     observed GCS component is at its maximum (E4, M6, V5); anything lower scores 3.
#   * Oxygen: "on oxygen" = FiO2 above room air (>21%, either % or fraction units) or a
#     ventilator flag. Missing FiO2 is treated as room air only if
#     assume_air_if_missing=True (pre-op ward patients are usually on air) -- the
#     alternative is to leave that component unscored.
import numpy as np

NEWS2_PARAMS = ("rr", "spo2", "on_oxygen", "sbp", "hr", "alert", "temp_c")
NEWS2_CORE_VITALS = ("rr", "spo2", "sbp", "hr", "temp_c")   # the five measured vitals


def _arr(x):
    return np.asarray(x, dtype=float)


def _banded(x, conditions_scores):
    # conditions_scores: list of (boolean_array, score); first match wins; NaN stays NaN.
    x = _arr(x)
    out = np.full(x.shape, np.nan)
    assigned = np.isnan(x)
    for cond, score in conditions_scores:
        take = cond & ~assigned
        out[take] = score
        assigned |= take
    return out


def score_respiration_rate(rr):
    rr = _arr(rr)
    return _banded(rr, [(rr <= 8, 3), (rr < 12, 1), (rr <= 20, 0), (rr <= 24, 2), (rr > 24, 3)])


def score_spo2_scale1(spo2):
    s = _arr(spo2)
    return _banded(s, [(s <= 91, 3), (s < 94, 2), (s < 96, 1), (s >= 96, 0)])


def score_air_or_oxygen(on_oxygen):
    o = _arr(on_oxygen)
    return _banded(o, [(o > 0.5, 2), (o <= 0.5, 0)])


def score_systolic_bp(sbp):
    b = _arr(sbp)
    return _banded(b, [(b <= 90, 3), (b <= 100, 2), (b <= 110, 1), (b < 220, 0), (b >= 220, 3)])


def score_pulse(hr):
    h = _arr(hr)
    return _banded(h, [(h <= 40, 3), (h <= 50, 1), (h <= 90, 0), (h <= 110, 1), (h <= 130, 2), (h > 130, 3)])


def score_consciousness(alert):
    a = _arr(alert)
    return _banded(a, [(a > 0.5, 0), (a <= 0.5, 3)])


def score_temperature(temp_c):
    t = _arr(temp_c)
    return _banded(t, [(t <= 35.0, 3), (t <= 36.0, 1), (t <= 38.0, 0), (t <= 39.0, 1), (t > 39.0, 2)])


SCORERS = {
    "rr": score_respiration_rate, "spo2": score_spo2_scale1, "on_oxygen": score_air_or_oxygen,
    "sbp": score_systolic_bp, "hr": score_pulse, "alert": score_consciousness, "temp_c": score_temperature,
}


def on_supplemental_oxygen(fio2=np.nan, vent=np.nan, assume_air_if_missing=True):
    # FiO2 may be recorded as a percentage (21-100) or a fraction (0.21-1.0): anything
    # <= 1.0 is read as a fraction. "Above room air" uses a small tolerance (21.5%) so a
    # charted 21% is air, not oxygen.
    f = _arr(fio2)
    v = _arr(vent)
    f_pct = np.where(f <= 1.0, f * 100.0, f)
    on = np.where(np.isnan(f_pct), np.nan, (f_pct > 21.5).astype(float))
    on = np.where((~np.isnan(v)) & (v > 0), 1.0, on)
    if assume_air_if_missing:
        on = np.where(np.isnan(on), 0.0, on)
    return on


def alert_from_gcs(gcs_e=np.nan, gcs_m=np.nan, gcs_v=np.nan):
    # 1.0 = Alert, 0.0 = not alert, NaN = no GCS component observed.
    e, m, v = _arr(gcs_e), _arr(gcs_m), _arr(gcs_v)
    below_max = ((~np.isnan(e)) & (e < 4)) | ((~np.isnan(m)) & (m < 6)) | ((~np.isnan(v)) & (v < 5))
    any_obs = ~(np.isnan(e) & np.isnan(m) & np.isnan(v))
    return np.where(~any_obs, np.nan, np.where(below_max, 0.0, 1.0))


def news2_components(rr=np.nan, spo2=np.nan, on_oxygen=np.nan, sbp=np.nan, hr=np.nan,
                     alert=np.nan, temp_c=np.nan):
    vals = dict(rr=rr, spo2=spo2, on_oxygen=on_oxygen, sbp=sbp, hr=hr, alert=alert, temp_c=temp_c)
    return {p: SCORERS[p](vals[p]) for p in NEWS2_PARAMS}


def news2_total(components, assume_alert_if_missing=True, min_core_vitals=3):
    # Returns (total, n_core_vitals_observed, any_single_3). A missing component
    # contributes 0 -- which is why a total is only reported (not NaN) when at least
    # `min_core_vitals` of the five measured vitals were actually observed.
    stacked = np.stack([_arr(components[p]) for p in NEWS2_PARAMS])
    if assume_alert_if_missing:
        i = NEWS2_PARAMS.index("alert")
        stacked[i] = np.where(np.isnan(stacked[i]), 0.0, stacked[i])
    total = np.nansum(stacked, axis=0)
    n_core = np.sum(~np.isnan(np.stack([_arr(components[p]) for p in NEWS2_CORE_VITALS])), axis=0)
    any3 = np.any(stacked == 3, axis=0)
    total = np.where(n_core >= min_core_vitals, total, np.nan)
    return total, n_core, any3


def news2_risk_band(total, any_single_3):
    # 0 = low (0-4), 1 = low-medium (single parameter 3, total <5), 2 = medium (5-6),
    # 3 = high (>=7); NaN where total is NaN.
    t = _arr(total)
    a = np.asarray(any_single_3, dtype=bool)
    band = np.where(t >= 7, 3, np.where(t >= 5, 2, np.where(a, 1, 0))).astype(float)
    return np.where(np.isnan(t), np.nan, band)


def news2_observation_sets(series_by_param, bucket_minutes=60, carry_forward_minutes=240,
                           assume_air_if_missing=True, assume_alert_if_missing=True,
                           min_core_vitals=3):
    # series_by_param: {raw_item_name: {chart_time: value}} using the dataset's ward-vital
    # names: rr, spo2, fio2, vent, bt, nibp_sbp, hr, gcs_e, gcs_m, gcs_v.
    # Ward vitals are charted as a "set" at roughly the same time; each set is
    # reconstructed by bucketing chart_times into `bucket_minutes` bins, and a parameter
    # missing from a bin is carried forward from its last reading only if that reading is
    # within `carry_forward_minutes` (bounded LOCF -- never across a long gap).
    # Returns a dict of equal-length arrays, one entry per bucket that had any reading.
    all_times = sorted({t for s in series_by_param.values() for t in s})
    if not all_times:
        return {"time": np.array([]), "total": np.array([]), "n_core": np.array([]),
                "any3": np.array([], dtype=bool), "band": np.array([])}
    t0 = all_times[0]
    bucket_of = lambda t: int((t - t0) // bucket_minutes)
    buckets = sorted({bucket_of(t) for t in all_times})
    bucket_end = np.array([t0 + (b + 1) * bucket_minutes for b in buckets], dtype=float)

    def value_at_buckets(item):
        s = series_by_param.get(item) or {}
        out = np.full(len(buckets), np.nan)
        if not s:
            return out
        times = np.array(sorted(s), dtype=float)
        vals = np.array([s[t] for t in sorted(s)], dtype=float)
        for i, end in enumerate(bucket_end):
            j = np.searchsorted(times, end, side="left") - 1   # last reading before bucket end
            if j >= 0 and (end - times[j]) <= carry_forward_minutes + bucket_minutes:
                out[i] = vals[j]
        return out

    rr, spo2, sbp, hr, bt = (value_at_buckets(k) for k in ("rr", "spo2", "nibp_sbp", "hr", "bt"))
    on_o2 = on_supplemental_oxygen(value_at_buckets("fio2"), value_at_buckets("vent"),
                                   assume_air_if_missing=assume_air_if_missing)
    alert = alert_from_gcs(value_at_buckets("gcs_e"), value_at_buckets("gcs_m"), value_at_buckets("gcs_v"))
    comps = news2_components(rr=rr, spo2=spo2, on_oxygen=on_o2, sbp=sbp, hr=hr, alert=alert, temp_c=bt)
    total, n_core, any3 = news2_total(comps, assume_alert_if_missing, min_core_vitals)
    return {"time": bucket_end, "total": total, "n_core": n_core, "any3": any3,
            "band": news2_risk_band(total, any3)}


def news2_summary_features(sets):
    # Static per-patient summary of the observation sets. NaN where no valid set exists
    # (filled by the normal static-feature imputation), with news2_available recording
    # whether a real score existed -- the same observed/imputed discipline as the masks.
    valid = ~np.isnan(sets["total"]) if len(sets["total"]) else np.array([], dtype=bool)
    if not valid.any():
        return {"news2_max": np.nan, "news2_last": np.nan, "news2_mean": np.nan,
                "news2_n_sets": 0.0, "news2_high_flag": 0.0, "news2_red_flag": 0.0,
                "news2_available": 0.0}
    tot = sets["total"][valid]
    return {
        "news2_max": float(np.max(tot)),
        "news2_last": float(tot[-1]),
        "news2_mean": float(np.mean(tot)),
        "news2_n_sets": float(len(tot)),
        "news2_high_flag": float(np.any(tot >= 7)),                             # RCP 'high' band ever reached
        "news2_red_flag": float(np.any(sets["any3"][valid]) or np.any(tot >= 5)),  # medium-or-worse ever
        "news2_available": 1.0,
    }


NEWS2_FEATURE_NAMES = ("news2_max", "news2_last", "news2_mean", "news2_n_sets",
                       "news2_high_flag", "news2_red_flag", "news2_available")
NEWS2_BINARY_FEATURES = ("news2_high_flag", "news2_red_flag", "news2_available")


def news2_self_test():
    # Checks every band edge of the official RCP table plus two worked totals. Returns
    # (passed: bool, failures: list[str]). The pipeline only includes NEWS2 as a model
    # feature if this passes -- "only include it if it is right", made mechanical.
    cases = [
        (score_respiration_rate, [(8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3), (5, 3), (40, 3)]),
        (score_spo2_scale1, [(91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0), (85, 3)]),
        (score_air_or_oxygen, [(1, 2), (0, 0)]),
        (score_systolic_bp, [(90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3), (70, 3)]),
        (score_pulse, [(40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (110, 1), (111, 2), (130, 2), (131, 3)]),
        (score_consciousness, [(1, 0), (0, 3)]),
        (score_temperature, [(35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0), (38.1, 1), (39.0, 1), (39.1, 2), (34.0, 3)]),
    ]
    failures = []
    for fn, pairs in cases:
        for x, want in pairs:
            got = float(fn(x))
            if got != want:
                failures.append(f"{fn.__name__}({x}) = {got}, expected {want}")
    if not np.isnan(float(score_pulse(np.nan))):
        failures.append("missing input must stay NaN")
    # Worked example 1 -- everything normal on air, alert: total 0, low band.
    c = news2_components(rr=16, spo2=98, on_oxygen=0, sbp=125, hr=72, alert=1, temp_c=36.8)
    t, n, a = news2_total(c)
    if float(t) != 0 or float(news2_risk_band(t, a)) != 0:
        failures.append(f"normal patient total {float(t)} (expected 0, band low)")
    # Worked example 2 -- RR 22 (2), SpO2 93 (2), O2 (2), SBP 105 (1), HR 115 (2), alert (0), 38.5 (1) = 10, high.
    c = news2_components(rr=22, spo2=93, on_oxygen=1, sbp=105, hr=115, alert=1, temp_c=38.5)
    t, n, a = news2_total(c)
    if float(t) != 10 or float(news2_risk_band(t, a)) != 3:
        failures.append(f"sick patient total {float(t)} (expected 10, band high)")
    # Single red score with a low total -> low-medium band.
    c = news2_components(rr=16, spo2=98, on_oxygen=0, sbp=125, hr=72, alert=0, temp_c=36.8)
    t, n, a = news2_total(c)
    if float(t) != 3 or float(news2_risk_band(t, a)) != 1:
        failures.append(f"single-red-score patient total {float(t)} band {float(news2_risk_band(t, a))} (expected 3, low-medium)")
    # Unit handling and GCS mapping.
    if float(on_supplemental_oxygen(0.21)) != 0 or float(on_supplemental_oxygen(40)) != 1 or float(on_supplemental_oxygen(np.nan, 1)) != 1:
        failures.append("FiO2/vent -> oxygen mapping")
    if float(alert_from_gcs(4, 6, 5)) != 1 or float(alert_from_gcs(4, 6, 4)) != 0 or not np.isnan(float(alert_from_gcs())):
        failures.append("GCS -> alert mapping")
    # Too few observed vitals -> no score rather than a falsely reassuring low one.
    c = news2_components(hr=72)
    t, n, a = news2_total(c, min_core_vitals=3)
    if not np.isnan(float(t)):
        failures.append("a set with only 1 core vital must not produce a score")
    return (len(failures) == 0), failures


def compare_before_after(before, after, name="news2", plausible_range=(0.0, 20.0), min_n=20):
    # Distribution check between a "before" sample (real patients) and an "after" sample
    # (synthetic or augmented patients). Returns a plain dict for printing/logging.
    b = _arr(before); b = b[~np.isnan(b)]
    a = _arr(after); a = a[~np.isnan(a)]
    res = {"feature": name, "n_before": int(len(b)), "n_after": int(len(a))}
    if len(b) == 0 or len(a) == 0:
        res["verdict"] = "not enough data"
        return res
    lo, hi = plausible_range
    res.update({
        "median_before": float(np.median(b)), "median_after": float(np.median(a)),
        "mean_before": float(np.mean(b)), "mean_after": float(np.mean(a)),
        "pct_high_before": float(np.mean(b >= 7)), "pct_high_after": float(np.mean(a >= 7)),
        "n_outside_range_after": int(np.sum((a < lo) | (a > hi))),
    })
    try:
        from scipy.stats import ks_2samp
        ks = ks_2samp(b, a)
        res["ks_stat"], res["ks_p"] = float(ks.statistic), float(ks.pvalue)
    except Exception:
        res["ks_stat"], res["ks_p"] = np.nan, np.nan
    # Verdict: implausible values are a hard fail; otherwise judge the SIZE of the shift
    # (KS statistic), not just its p-value -- with hundreds of synthetic rows a clinically
    # trivial shift is "significant".
    if res["n_outside_range_after"] > 0:
        res["verdict"] = "FAIL: values outside the possible NEWS2 range"
    elif len(a) < min_n:
        res["verdict"] = f"TOO FEW to judge the shape (n_after={len(a)} < {min_n}); range check passed"
    elif not np.isnan(res["ks_stat"]) and res["ks_stat"] > 0.2:
        res["verdict"] = "REVIEW: distribution shifted (KS > 0.2)"
    else:
        res["verdict"] = "PASS"
    return res
