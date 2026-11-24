"""Pre-operative POSSUM / P-POSSUM scores from INSPIRE records.

POSSUM (Copeland et al., 1991) has a 12-item physiological score (PS, range 12-88) and a 6-item
operative score (OS). P-POSSUM (Prytherch et al., 1998) converts them to mortality risk:
    logit(R) = -9.065 + 0.1692 * PS + 0.1550 * OS

Only information available BEFORE the index (last) operation is used:
  * physiology: last value in the same 5-day pre-operative window as the DNN;
  * cardiac / respiratory history and ECG: diagnosis codes and medications charted before the
    operation (INSPIRE has no ECG or chest X-ray; codes are proxies);
  * operative score: mode of surgery (emop) and malignancy (ICD-10 C codes). Operative severity,
    multiple procedures, blood loss and peritoneal soiling are intra-/post-operative or not
    recorded, so they take the minimum score (1), as is conventional for missing POSSUM items.
Missing physiological values also score 1 (normal). This is therefore a pre-operative POSSUM
approximation, not the full published score.
"""
import numpy as np
import pandas as pd

PRE_OP_DAYS = 5
LAB_ITEMS = ["bun", "hb", "wbc", "sodium", "potassium"]
VITAL_ITEMS = ["hr", "nibp_sbp", "gcs_e", "gcs_m", "gcs_v"]
BUN_MGDL_TO_UREA_MMOL = 0.357


def _band(x, rules, default=1):
    """rules: list of (predicate, points); first match wins; NaN -> default."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    for pred, pts in rules:
        if pred(x):
            return pts
    return default


def score_age(a):
    return _band(a, [(lambda v: v <= 60, 1), (lambda v: v <= 70, 2), (lambda v: True, 4)])


def score_sbp(v):
    return _band(v, [(lambda x: x <= 89, 8), (lambda x: x <= 99, 4), (lambda x: x <= 109, 2),
                     (lambda x: x <= 130, 1), (lambda x: x <= 170, 2), (lambda x: True, 4)])


def score_pulse(v):
    return _band(v, [(lambda x: x <= 39, 8), (lambda x: x <= 49, 2), (lambda x: x <= 80, 1),
                     (lambda x: x <= 100, 2), (lambda x: x <= 120, 4), (lambda x: True, 8)])


def score_gcs(v):
    return _band(v, [(lambda x: x >= 15, 1), (lambda x: x >= 12, 2), (lambda x: x >= 9, 4), (lambda x: True, 8)])


def score_hb(v):  # g/dL
    return _band(v, [(lambda x: x <= 9.9, 8), (lambda x: x <= 11.4, 4), (lambda x: x <= 12.9, 2),
                     (lambda x: x <= 16.0, 1), (lambda x: x <= 17.0, 2), (lambda x: x <= 18.0, 4), (lambda x: True, 8)])


def score_wbc(v):  # 10^9/L
    return _band(v, [(lambda x: x <= 3.0, 4), (lambda x: x <= 4.0, 2), (lambda x: x <= 10.0, 1),
                     (lambda x: x <= 20.0, 2), (lambda x: True, 4)])


def score_urea(v):  # mmol/L
    return _band(v, [(lambda x: x <= 7.5, 1), (lambda x: x <= 10.0, 2), (lambda x: x <= 15.0, 4), (lambda x: True, 8)])


def score_na(v):
    return _band(v, [(lambda x: x >= 136, 1), (lambda x: x >= 131, 2), (lambda x: x >= 126, 4), (lambda x: True, 8)])


def score_k(v):
    return _band(v, [(lambda x: x <= 2.8, 8), (lambda x: x <= 3.1, 4), (lambda x: x <= 3.4, 2), (lambda x: x <= 5.0, 1),
                     (lambda x: x <= 5.3, 2), (lambda x: x <= 5.9, 4), (lambda x: True, 8)])


def _has(codes, prefixes):
    return any(c.startswith(p) for c in codes for p in prefixes)


def score_cardiac(codes, atcs):
    if _has(codes, ["I50", "I517"]):                      # heart failure / cardiomegaly
        return 8
    if _has(atcs, ["B01AA"]) or _has(codes, ["R60"]):      # warfarin / oedema
        return 4
    if _has(atcs, ["C01", "C02", "C03", "C07", "C08", "C09"]):  # cardiac / antihypertensive / diuretic drugs
        return 2
    return 1


def score_resp(codes):
    if _has(codes, ["J84", "J96"]):                       # fibrosis / respiratory failure
        return 8
    if _has(codes, ["J44"]):                              # COPD
        return 4
    if _has(codes, ["J40", "J41", "J42", "J43", "J45", "J46", "J47"]):
        return 2
    return 1


def score_ecg(codes):
    if _has(codes, ["I44", "I45", "I47", "I49"]):         # conduction disorders / other arrhythmias
        return 8
    if _has(codes, ["I48"]):                              # atrial fibrillation
        return 4
    return 1


def score_malignancy(codes):
    if _has(codes, ["C78", "C79"]):                       # distant metastases
        return 8
    if _has(codes, ["C77"]):                              # nodal metastases
        return 4
    if _has(codes, ["C"]):
        return 2
    return 1


def _in_window(df, items, win):
    """Rows for the given items with chart_time in [orin - 5 d, orin]."""
    df = df[df["item_name"].isin(items) & df["subject_id"].isin(win.index)]
    hi = df["subject_id"].map(win)
    return df[(df["chart_time"] >= hi - PRE_OP_DAYS * 1440) & (df["chart_time"] <= hi) & df["value"].notna()]


def _last_value(df):
    """Latest value per (subject, item), by chart_time."""
    return df.sort_values("chart_time").groupby(["subject_id", "item_name"])["value"].last().unstack()


def possum_for_cohort(dataset_dir, cohort, ward_vitals_chunks=5_000_000):
    win = cohort.set_index("subject_id")["orin_last"]
    labs = pd.read_csv(f"{dataset_dir}/labs.csv")
    lab_last = _last_value(_in_window(labs, LAB_ITEMS, win))
    parts = [_in_window(chunk, VITAL_ITEMS, win)
             for chunk in pd.read_csv(f"{dataset_dir}/ward_vitals.csv", chunksize=ward_vitals_chunks,
                                      usecols=["subject_id", "chart_time", "item_name", "value"])]
    vit_last = _last_value(pd.concat(parts))   # combine chunks first, then pick the latest by time
    dx = pd.read_csv(f"{dataset_dir}/diagnosis.csv")
    dx = dx[dx["subject_id"].isin(win.index)]
    dx = dx[dx["chart_time"] < dx["subject_id"].map(win)]
    dx_codes = dx.groupby("subject_id")["icd10_cm"].apply(lambda s: set(s.astype(str)))
    med = pd.read_csv(f"{dataset_dir}/medications.csv", usecols=["subject_id", "chart_time", "atc_code"])
    med = med[med["subject_id"].isin(win.index)]
    med = med[med["chart_time"] < med["subject_id"].map(win)]
    med_atc = med.groupby("subject_id")["atc_code"].apply(lambda s: set(s.dropna().astype(str)))

    rows = []
    for r in cohort.itertuples(index=False):
        sid = r.subject_id
        L = lab_last.loc[sid] if sid in lab_last.index else {}
        V = vit_last.loc[sid] if sid in vit_last.index else {}
        get = lambda d, k: float(d[k]) if k in getattr(d, "index", d) and pd.notna(d[k]) else np.nan
        gcs_parts = [get(V, k) for k in ("gcs_e", "gcs_m", "gcs_v")]
        gcs = sum(gcs_parts) if all(not np.isnan(x) for x in gcs_parts) else np.nan
        urea = get(L, "bun") * BUN_MGDL_TO_UREA_MMOL
        codes = dx_codes.get(sid, set())
        atcs = med_atc.get(sid, set())
        items = {
            "age": score_age(r.age), "cardiac": score_cardiac(codes, atcs), "resp": score_resp(codes),
            "sbp": score_sbp(get(V, "nibp_sbp")), "pulse": score_pulse(get(V, "hr")), "gcs": score_gcs(gcs),
            "hb": score_hb(get(L, "hb")), "wbc": score_wbc(get(L, "wbc")), "urea": score_urea(urea),
            "na": score_na(get(L, "sodium")), "k": score_k(get(L, "potassium")), "ecg": score_ecg(codes),
        }
        ps = sum(items.values())
        mode = 4 if r.emop == 1 else 1
        os_ = 1 + 1 + 1 + 1 + score_malignancy(codes) + mode   # severity, multiple, blood loss, soiling = 1
        observed = sum(not np.isnan(x) for x in [get(V, "nibp_sbp"), get(V, "hr"), gcs, get(L, "hb"),
                                                  get(L, "wbc"), urea, get(L, "sodium"), get(L, "potassium")])
        rows.append({"subject_id": sid, "possum_ps": ps, "possum_os": os_,
                     "p_possum_logit": -9.065 + 0.1692 * ps + 0.1550 * os_,
                     "possum_physiology_observed": observed})
    return pd.DataFrame(rows)
