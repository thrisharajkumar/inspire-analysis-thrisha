# %% [markdown]
# ## 6.16 NEWS2 early-warning score (v2) — computed, verified, then included
#
# NEWS2 (Royal College of Physicians, 2017) is built here from the **pre-op ward vitals
# already loaded** (RR, SpO2, FiO2/ventilator for "air or oxygen", SBP, HR, temperature,
# GCS for consciousness) — same window as every other feature, so no leakage from surgery
# or after it. It is added to the static vector **before** SMOTENC, so synthetic patients
# get an interpolated NEWS2 like any other continuous feature; Part 8.2b and Part 9.1b then
# check NEWS2 **before vs. after** each augmentation step.
#
# "Only include it if it is right" is enforced mechanically: the scoring function is
# self-tested against every band edge of the RCP chart below, and if any check fails
# NEWS2 is **left out of the model** (with the failures printed) rather than included.
#
# Documented approximations (for clinical sign-off): SpO2 Scale 1 for everyone; AVPU is
# not recorded so GCS stands in (all observed components at maximum = Alert); missing
# FiO2/ventilator = room air; a set needs ≥ `NEWS2_MIN_CORE_VITALS` of the five measured
# vitals or it gets no score (never a falsely reassuring low one).

# %%
# INLINE: inspire_dnn/news2.py
from inspire_dnn.news2 import *

# %%
NEWS2_VALIDATED, _news2_failures = news2_self_test()
print(f"NEWS2 scoring self-test against the RCP 2017 chart: {'PASSED' if NEWS2_VALIDATED else 'FAILED'}")
for _f in _news2_failures:
    print("   ", _f)
USE_NEWS2 = bool(CONFIG["INCLUDE_NEWS2"] and NEWS2_VALIDATED)
if CONFIG["INCLUDE_NEWS2"] and not NEWS2_VALIDATED:
    print("NEWS2 will NOT be added to the model this run -- fix the failures above first.")
elif not CONFIG["INCLUDE_NEWS2"]:
    print("CONFIG['INCLUDE_NEWS2']=False -- NEWS2 computed for reference only, not used as a feature.")

# %%
NEWS2_INPUT_ITEMS = ["rr", "spo2", "fio2", "vent", "bt", "nibp_sbp", "hr", "gcs_e", "gcs_m", "gcs_v"]
_missing_items = [i for i in NEWS2_INPUT_ITEMS if i not in set(INVENTORY["ward_vitals"])]
if _missing_items:
    print(f"Ward-vital items not present in this cohort (scored as missing): {_missing_items}")

NEWS2_SETS, NEWS2_FEATURES = {}, {}
_t0 = time.time()
for sid in cohort_df["subject_id"]:
    lo, hi = get_window(sid, COHORT_INDEXED.loc[sid])
    series = {item: windowed_series(ward_vitals_df, sid, item, lo, hi) for item in NEWS2_INPUT_ITEMS}
    sets = news2_observation_sets(
        series, bucket_minutes=CONFIG["NEWS2_BUCKET_MINUTES"],
        carry_forward_minutes=CONFIG["NEWS2_CARRY_FORWARD_MINUTES"],
        assume_air_if_missing=CONFIG["NEWS2_ASSUME_AIR_IF_MISSING"],
        assume_alert_if_missing=CONFIG["NEWS2_ASSUME_ALERT_IF_MISSING"],
        min_core_vitals=CONFIG["NEWS2_MIN_CORE_VITALS"])
    NEWS2_SETS[sid] = sets
    NEWS2_FEATURES[sid] = news2_summary_features(sets)
news2_features_df = pd.DataFrame(NEWS2_FEATURES).T
news2_features_df.index.name = "subject_id"
print(f"NEWS2 computed for {len(news2_features_df)} patients in {time.time() - _t0:.1f}s")
print(f"  patients with >=1 scorable observation set: {int(news2_features_df['news2_available'].sum())} "
      f"({news2_features_df['news2_available'].mean():.1%})")
_all_n_core = np.concatenate([s["n_core"] for s in NEWS2_SETS.values() if len(s["n_core"])]) \
    if any(len(s["n_core"]) for s in NEWS2_SETS.values()) else np.array([])
if len(_all_n_core):
    print(f"  core vitals observed per set (of 5): mean {_all_n_core.mean():.2f}; "
          f"sets with all 5: {np.mean(_all_n_core == 5):.1%} -- the rest had a component carried "
          f"forward or unscored (see the approximation note above)")
news2_features_df.describe().T[["mean", "std", "min", "max"]]

# %%
# Construct-validity check (descriptive, like §5.1's ASA check): if NEWS2 is being computed
# correctly, 30-day mortality should rise with the worst pre-op NEWS2 band.
_v = news2_features_df.join(COHORT_INDEXED[["died_30day_from_last_op"]])
_v = _v[_v["news2_available"] == 1]
NEWS2_MORTALITY_MONOTONE = None
if len(_v):
    _v["worst_band"] = pd.cut(_v["news2_max"], bins=[-0.1, 4, 6, 20], labels=["low (0-4)", "medium (5-6)", "high (7+)"])
    news2_validity = _v.groupby("worst_band", observed=False)["died_30day_from_last_op"].agg(["mean", "count"])
    news2_validity.columns = ["mortality_rate", "n_patients"]
    print(news2_validity)
    _rates = news2_validity["mortality_rate"].dropna().values
    NEWS2_MORTALITY_MONOTONE = bool(len(_rates) >= 2 and np.all(np.diff(_rates) >= 0))
    print(f"\nMortality rises across NEWS2 bands: {'YES' if NEWS2_MORTALITY_MONOTONE else 'NO -- check before trusting the feature'}"
          f" (a sanity check on the implementation, not a model result)")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    news2_validity["mortality_rate"].plot(kind="bar", ax=ax, color="#c0392b")
    ax.set_title("30-day mortality by worst pre-op NEWS2 band"); ax.set_ylabel("mortality rate"); ax.set_xlabel("")
    plt.tight_layout(); plt.show()
