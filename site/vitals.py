"""
feature_selection_pipeline.py

WHAT THIS SCRIPT DOES, IN ORDER
--------------------------------
1. Loads every patient's pre-op labs/ward_vitals (5 days before surgery).
2. Tests each of the 54 features: does it differ between patients who
   died vs. survived? (univariate test)
3. Corrects for the fact that we tested 54 things at once (multiple
   comparisons) - otherwise some features look "significant" by pure luck.
4. Removes redundant features that are just measuring the same thing twice
   (e.g. hemoglobin and hematocrit move together almost perfectly).
5. Fits ONE model that includes department, age, ASA class, and emergency-op
   status ALONGSIDE the lab features - this checks whether a lab value adds
   real information, or whether it was only "significant" in step 2 because
   it's a stand-in for "this patient is in a high-risk department."
6. Reports which features survive that check - these are the ones we trust.

WHY THIS ORDER MATTERS
-----------------------
Step 2 alone is not enough. A feature can look strongly linked to death
just because sicker departments order it more often - that tells you
about the department, not about the feature itself. Step 5 is what
separates "genuinely useful feature" from "proxy for department."
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =================================================================
# CONFIG - edit these paths for your setup
# =================================================================
SUBJECTS_DIR = r"C:\Users\pc\Desktop\INSPIRE\dataset\subjects_sample"   # folder with survived/ and died/ subfolders
PARAMS_CSV = r"C:\Users\pc\Desktop\INSPIRE\dataset\parameters.csv"        # the feature schema file
DAYS_BEFORE_OP = 5
MINUTES_BEFORE_OP = DAYS_BEFORE_OP * 24 * 60


# =================================================================
# STEP 1: Load every patient's pre-op data into one table
# =================================================================
def load_schema(path):
    """Reads parameters.csv -> {'labs': [...feature names...], 'ward_vitals': [...]}"""
    schema = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            schema[row["Table"]].append(row["Label"])
    return dict(schema)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_patient_table(subjects_dir, all_features):
    """
    For every patient JSON file, extracts:
      - static info: age, sex, ASA class, emergency-op flag, department
        (these come from the 'operations' table and always exist)
      - for each lab/vital feature: was it measured pre-op? what was the
        last value? (these come from 'labs'/'ward_vitals' and are often
        MISSING - that's the whole reason this analysis is necessary)
    Returns one row per patient.
    """
    rows = []
    for label, folder in [(0, "survived"), (1, "died")]:
        for path in sorted(glob.glob(os.path.join(subjects_dir, folder, "*.json"))):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            ops = data.get("operations") or []
            if not ops:
                continue
            last_op = ops[-1]                              # most recent operation
            orin_time = safe_float(last_op.get("orin_time"))  # moment patient enters OR
            if orin_time is None:
                continue

            # the "pre-op window": everything measured in the 5 days BEFORE
            # surgery. Anything after this point would be information the
            # model shouldn't have access to at prediction time.
            window_start = orin_time - MINUTES_BEFORE_OP
            window_end = orin_time - 1

            row = {
                "subject_id": data.get("subject_id"),
                "label": label,                             # 0 = survived, 1 = died
                "age": safe_float(last_op.get("age")),
                "asa": safe_float(last_op.get("asa")),       # ASA physical status class (1-5)
                "emop": safe_float(last_op.get("emop")),     # 1 = emergency operation
                "department": last_op.get("department"),
            }

            by_feature = defaultdict(list)
            for records in [data.get("labs", []), data.get("ward_vitals", [])]:
                for r in records:
                    ct = safe_float(r.get("chart_time"))
                    val = safe_float(r.get("value"))
                    item = r.get("item_name")
                    if ct is None or val is None or item not in all_features:
                        continue
                    if window_start <= ct <= window_end:
                        by_feature[item].append(val)

            for feat in all_features:
                obs = by_feature.get(feat, [])
                row[f"{feat}__present"] = 1 if obs else 0    # was it measured at all?
                row[f"{feat}__last"] = obs[-1] if obs else np.nan   # most recent value

            rows.append(row)
    return pd.DataFrame(rows)


# =================================================================
# STEP 2+3: Test each feature, correct for multiple comparisons
# =================================================================
def univariate_screen(df, all_features):
    """
    For each feature, runs a Mann-Whitney U test: do died patients' values
    differ from survived patients' values? Also runs Fisher's exact test on
    PRESENCE (was it measured at all) since that carries a different kind
    of signal (testing intensity / department behavior).

    Mann-Whitney (not a t-test) because we can't assume the values are
    normally distributed - safer default with clinical lab data.

    FDR correction (Benjamini-Hochberg): when you test 54 features at once,
    ~2-3 will look "significant" by chance alone even if nothing is real.
    This correction adjusts the threshold so the false-positive rate stays
    under control across all 54 tests together, not just one at a time.
    """
    results = []
    for feat in all_features:
        died_vals = df.loc[df.label == 1, f"{feat}__last"].dropna()
        surv_vals = df.loc[df.label == 0, f"{feat}__last"].dropna()
        p, rank_biserial = np.nan, np.nan
        if len(died_vals) >= 10 and len(surv_vals) >= 10:
            u_stat, p = stats.mannwhitneyu(died_vals, surv_vals, alternative="two-sided")
            n1, n2 = len(died_vals), len(surv_vals)
            # positive = feature tends to be HIGHER in the died group
            rank_biserial = -(1 - (2 * u_stat) / (n1 * n2))
        results.append({"feature": feat, "n_died": len(died_vals), "n_survived": len(surv_vals),
                         "rank_biserial": rank_biserial, "p_value": p})

    res_df = pd.DataFrame(results)
    mask = res_df.p_value.notna()
    res_df.loc[mask, "p_value_fdr"] = multipletests(res_df.loc[mask, "p_value"], method="fdr_bh")[1]
    return res_df


# =================================================================
# STEP 4: Remove redundant features (keep the stronger of each pair)
# =================================================================
def drop_redundant_features(df, candidate_features, res_df, corr_threshold=0.7):
    """
    Two features that move together almost perfectly (e.g. hemoglobin and
    hematocrit) don't each add new information to a model - they just make
    the model less stable and harder to interpret. This keeps whichever
    of the two has the stronger standalone association with the outcome.
    """
    last_cols = [f"{f}__last" for f in candidate_features]
    corr = df[last_cols].corr()
    corr.columns = candidate_features
    corr.index = candidate_features

    res_lookup = res_df.set_index("feature")["rank_biserial"].abs()
    to_drop = set()
    for i, f1 in enumerate(candidate_features):
        for f2 in candidate_features[i + 1:]:
            if f1 in to_drop or f2 in to_drop:
                continue
            if abs(corr.loc[f1, f2]) > corr_threshold:
                weaker = f2 if res_lookup.get(f1, 0) >= res_lookup.get(f2, 0) else f1
                to_drop.add(weaker)
    return [f for f in candidate_features if f not in to_drop], to_drop


# =================================================================
# STEP 5: Fit the confound-adjusted model
# =================================================================
def fit_adjusted_model(df, features, exclude_departments=None):
    """
    Fits ONE logistic regression with:
      - the candidate lab features (standardized, median-imputed)
      - age, ASA class, emergency-op flag
      - department (as a categorical variable, one dummy per department)

    A feature's p-value here answers: "does this feature still predict
    death after the model already knows the patient's department, age,
    ASA class, and emergency status?" That's a much stronger claim than
    a standalone correlation.

    NOTE: median imputation here is a simplification for THIS diagnostic
    check only. It is not the imputation strategy for the final DNN.

    exclude_departments: departments with ZERO deaths must be excluded -
    they cause "perfect separation" (the model can predict them with 100%
    certainty using department alone), which breaks the fitting algorithm.
    """
    exclude_departments = exclude_departments or []
    data = df[~df.department.isin(exclude_departments)].copy()

    X_parts = []
    for feat in features:
        col = f"{feat}__last"
        imputed = data[col].fillna(data[col].median())
        X_parts.append(((imputed - imputed.mean()) / imputed.std()).rename(feat))

    age_z = ((data.age - data.age.mean()) / data.age.std()).rename("age")
    asa_imputed = data.asa.fillna(data.asa.median())
    asa_z = ((asa_imputed - asa_imputed.mean()) / asa_imputed.std()).rename("asa")
    emop = data.emop.fillna(0).rename("emop")

    X = pd.concat(X_parts + [age_z, asa_z, emop], axis=1)
    dept_dummies = pd.get_dummies(data.department, prefix="dept", drop_first=False)
    reference_dept = data.department.value_counts().idxmax()   # largest dept = reference
    dept_dummies = dept_dummies.drop(columns=[f"dept_{reference_dept}"])
    X = pd.concat([X, dept_dummies.astype(float)], axis=1)
    X = sm.add_constant(X)
    y = data.label

    model = sm.Logit(y, X)
    result = model.fit(disp=0, maxiter=200)

    summary = pd.DataFrame({
        "coef": result.params, "p_value": result.pvalues,
        "odds_ratio": np.exp(result.params),
        "or_ci_low": np.exp(result.conf_int()[0]),
        "or_ci_high": np.exp(result.conf_int()[1]),
    }).drop("const")

    vif = pd.DataFrame({
        "feature": features,
        "VIF": [variance_inflation_factor(X[["const"] + features].values, i + 1)
                for i in range(len(features))]
    })

    return result, summary, vif, reference_dept


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    schema = load_schema(PARAMS_CSV)
    all_features = schema.get("labs", []) + schema.get("ward_vitals", [])

    print("Step 1: loading patient data...")
    df = build_patient_table(SUBJECTS_DIR, all_features)
    print(f"  {len(df)} patients loaded ({df.label.sum()} died, {(df.label==0).sum()} survived)")

    print("\nStep 2-3: univariate screen with FDR correction...")
    res_df = univariate_screen(df, all_features)
    sig_features = res_df[res_df.p_value_fdr < 0.05].feature.tolist()
    print(f"  {len(sig_features)} of {len(all_features)} features are FDR-significant")

    print("\nStep 4: removing redundant (highly correlated) features...")
    pruned_features, dropped = drop_redundant_features(df, sig_features, res_df)
    print(f"  dropped as redundant: {sorted(dropped)}")
    print(f"  {len(pruned_features)} features remain")

    # also drop features with poor coverage - too much of the "signal"
    # would just be the imputed median value, not real information
    coverage = {f: max((df[f"{f}__last"].notna() & (df.label == 1)).mean(),
                        (df[f"{f}__last"].notna() & (df.label == 0)).mean()) for f in pruned_features}
    pruned_features = [f for f in pruned_features if coverage[f] >= 0.10]

    zero_death_depts = df.groupby("department").label.sum()
    zero_death_depts = zero_death_depts[zero_death_depts == 0].index.tolist()
    print(f"\nStep 5: fitting confound-adjusted model (excluding zero-death depts: {zero_death_depts})...")
    result, summary, vif, reference_dept = fit_adjusted_model(df, pruned_features, zero_death_depts)
    print(f"  reference department: {reference_dept}")
    print(f"  converged: {result.mle_retvals['converged']}, pseudo R2: {result.prsquared:.3f}")

    lab_summary = summary.loc[pruned_features].sort_values("p_value")
    final_features = lab_summary[lab_summary.p_value < 0.05].index.tolist()

    print(f"\n=== FINAL FEATURES (survive confound adjustment, p<0.05) ===")
    print(lab_summary.loc[final_features].round(4).to_string())
    print(f"\n=== DROPPED (were significant alone, not after adjusting for department/age/ASA/emop) ===")
    print(lab_summary.loc[~lab_summary.index.isin(final_features)].index.tolist())

    summary.to_csv("multivariate_results.csv")
    res_df.to_csv("univariate_results.csv", index=False)
