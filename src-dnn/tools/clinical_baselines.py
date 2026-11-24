"""Clinical baselines on the SAME cohort and test split as the paper's DNN.

Reproduces the pipeline's cohort selection (01_data_loading.stream_parse_subjects),
label (02_labels_and_eda.compute_labels: in-hospital death < last-op orout + 30 days)
and split (06_imputation / 07_split_and_sampling: stratified 60/20/20, random_state=42),
then scores simple clinical comparators:

  * ASA alone (ordinal score)                       -- no fitting
  * worst pre-op NEWS2 alone (5-day window)          -- no fitting, pipeline's own scorer
  * emergency flag alone                             -- no fitting
  * LR: ASA + age + sex + emergency                  -- fitted on the training split
  * LR: ASA + age + sex + emergency + NEWS2 max/mean -- fitted on the training split
  * POSSUM physiological score and pre-operative P-POSSUM (see possum.py) -- no fitting

Bootstrap: 1,000 patient resamples of the fixed test set (same as the DNN headline CIs).

Usage (from src-dnn/):
  python tools/clinical_baselines.py --dataset-dir <csv folder> [--subjects-dir <died/survived folder>] \
      --cohort subset --cohort-cache cohort_subset.csv --out baselines_subset.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inspire_dnn.news2 import news2_observation_sets, news2_summary_features  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possum import possum_for_cohort  # noqa: E402

SEED = 42
PRE_OP_DAYS = 5
NEWS2_ITEMS = ["rr", "spo2", "fio2", "vent", "bt", "nibp_sbp", "hr", "gcs_e", "gcs_m", "gcs_v"]


def select_subjects(subjects_dir, max_per_class, seed=SEED):
    # Mirrors stream_parse_subjects(): sorted paths, random.Random(seed).sample when capped.
    rng = random.Random(seed)
    ids = []
    for folder in ["died", "survived"]:
        fps = sorted(glob.glob(os.path.join(subjects_dir, folder, "*.json")))
        if max_per_class is not None and len(fps) > max_per_class:
            fps = rng.sample(fps, max_per_class)
            fps.sort()
        ids.extend(int(os.path.splitext(os.path.basename(fp))[0]) for fp in fps)
    return ids


def build_cohort(ops, ids):
    ops = ops[ops["subject_id"].isin(set(ids))]
    rows = []
    for sid, g in ops.groupby("subject_id"):          # groupby sorts subject_id, as the pipeline
        g = g.sort_values("orin_time")
        first, last = g.iloc[0], g.iloc[-1]
        death = first["inhosp_death_time"]
        y = bool(pd.notna(death) and pd.notna(last["orout_time"])
                 and death < last["orout_time"] + 30 * 24 * 60)
        rows.append({"subject_id": sid, "label": int(y), "asa": last["asa"], "age": last["age"],
                     "male": float(last["sex"] == "M"), "emop": float(last["emop"]),
                     "orin_last": last["orin_time"], "n_ops": len(g)})
    return pd.DataFrame(rows)


def news2_for_cohort(ward_vitals_csv, cohort):
    win = cohort.set_index("subject_id")["orin_last"]
    keep = set(win.index)
    parts = []
    for chunk in pd.read_csv(ward_vitals_csv, chunksize=5_000_000,
                             usecols=["subject_id", "chart_time", "item_name", "value"]):
        chunk = chunk[chunk["item_name"].isin(NEWS2_ITEMS) & chunk["subject_id"].isin(keep)]
        if len(chunk):
            hi = chunk["subject_id"].map(win)
            chunk = chunk[(chunk["chart_time"] >= hi - PRE_OP_DAYS * 1440) & (chunk["chart_time"] <= hi)
                          & chunk["value"].notna()]
            parts.append(chunk)
    wv = pd.concat(parts) if parts else pd.DataFrame(columns=["subject_id", "chart_time", "item_name", "value"])
    feats = {}
    grouped = {sid: g for sid, g in wv.groupby("subject_id")}
    for sid in cohort["subject_id"]:
        g = grouped.get(sid)
        series = {}
        if g is not None:
            for item, gi in g.groupby("item_name"):
                gi = gi.sort_values("chart_time")
                series[item] = dict(zip(gi["chart_time"].astype(float), gi["value"].astype(float)))
        sets = news2_observation_sets(series, bucket_minutes=60, carry_forward_minutes=240,
                                      assume_air_if_missing=True, assume_alert_if_missing=True,
                                      min_core_vitals=3)
        feats[sid] = news2_summary_features(sets)
    return pd.DataFrame(feats).T.rename_axis("subject_id").reset_index()


def boot_ci(y, s, n=1000, seed=2026):
    rng = np.random.default_rng(seed)
    au, ap = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        au.append(roc_auc_score(y[b], s[b]))
        ap.append(average_precision_score(y[b], s[b]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(au), q(ap)


def evaluate(name, y, s):
    au_ci, ap_ci = boot_ci(y, s)
    r = {"model": name, "auroc": float(roc_auc_score(y, s)), "auroc_ci": au_ci,
         "auprc": float(average_precision_score(y, s)), "auprc_ci": ap_ci}
    print(f"  {name:<42} AUROC {r['auroc']:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})   "
          f"AUPRC {r['auprc']:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})")
    return r


def paired_vs_dnn(y, s_dnn, s_base, n=1000, seed=2026):
    # Paired patient bootstrap of (DNN - baseline) on identical test patients.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    dau, dap = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        dau.append(roc_auc_score(y[b], s_dnn[b]) - roc_auc_score(y[b], s_base[b]))
        dap.append(average_precision_score(y[b], s_dnn[b]) - average_precision_score(y[b], s_base[b]))
    dau, dap = np.array(dau), np.array(dap)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"d_auroc": float(roc_auc_score(y, s_dnn) - roc_auc_score(y, s_base)), "d_auroc_ci": q(dau),
            "d_auroc_p_le0": float(np.mean(dau <= 0)),
            "d_auprc": float(average_precision_score(y, s_dnn) - average_precision_score(y, s_base)), "d_auprc_ci": q(dap),
            "d_auprc_p_le0": float(np.mean(dap <= 0))}


def run(cohort, exclude_asa6=False, dnn=None):
    df = cohort.copy()
    all_ids = df["subject_id"].tolist()
    labels = df["label"].values
    tr, hold = train_test_split(all_ids, test_size=0.4, stratify=labels, random_state=SEED)
    lab = df.set_index("subject_id")["label"]
    va, te = train_test_split(hold, test_size=0.5, stratify=lab.loc[hold].values, random_state=SEED)
    split = {"train": len(tr), "val": len(va), "test": len(te),
             "train_pos": int(lab.loc[tr].sum()), "val_pos": int(lab.loc[va].sum()),
             "test_pos": int(lab.loc[te].sum())}
    d = df.set_index("subject_id")
    if exclude_asa6:   # applied after the split so the test set is the paper's test set minus ASA VI
        d = d[d["asa"] != 6]
        tr = [i for i in tr if i in d.index]
        te = [i for i in te if i in d.index]
        split.update({"test_after_asa6_exclusion": len(te), "test_pos_after_asa6_exclusion": int(d.loc[te, "label"].sum())})
    med = d.loc[tr].median(numeric_only=True)
    X = d.fillna(med)
    y_te = X.loc[te, "label"].values.astype(int)
    print(f"split: {split}")
    scores = {"ASA class alone": X.loc[te, "asa"].values.astype(float),
              "Emergency flag alone": X.loc[te, "emop"].values.astype(float)}
    if "news2_max" in X:
        scores["Worst pre-op NEWS2 alone"] = X.loc[te, "news2_max"].values.astype(float)
    if "possum_ps" in X:
        scores["POSSUM physiological score"] = X.loc[te, "possum_ps"].values.astype(float)
        scores["P-POSSUM (pre-op items)"] = X.loc[te, "p_possum_logit"].values.astype(float)
    for name, cols in [("LR: ASA + age + sex + emergency", ["asa", "age", "male", "emop"]),
                       ("LR: ASA + age + sex + emergency + NEWS2", ["asa", "age", "male", "emop", "news2_max", "news2_mean", "news2_available"])]:
        cols = [c for c in cols if c in X]
        Xtr, Xte = X.loc[tr, cols].values.astype(float), X.loc[te, cols].values.astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((Xtr - mu) / sd, X.loc[tr, "label"].values)
        scores[name] = lr.predict_proba((Xte - mu) / sd)[:, 1]
    res = [evaluate(k, y_te, v) for k, v in scores.items()]
    if dnn is not None:
        missing = [i for i in te if i not in dnn]
        split["test_ids_missing_from_dnn"] = len(missing)
        if missing:
            print(f"WARNING: {len(missing)} test patients have no DNN prediction -- split mismatch; skipping paired tests")
        else:
            s_dnn = np.array([dnn[i] for i in te])
            res.append(evaluate("Organ-system DNN (v4, this split)", y_te, s_dnn))
            for r in res[:-1]:
                r["paired_vs_dnn"] = paired_vs_dnn(y_te, s_dnn, scores[r["model"]])
                pv = r["paired_vs_dnn"]
                print(f"    DNN - {r['model']:<40} dAUROC {pv['d_auroc']:+.3f} ({pv['d_auroc_ci'][0]:+.3f},{pv['d_auroc_ci'][1]:+.3f})"
                      f"  dAUPRC {pv['d_auprc']:+.3f} ({pv['d_auprc_ci'][0]:+.3f},{pv['d_auprc_ci'][1]:+.3f})")
    res.append({"model": "Chance reference", "auroc": 0.5, "auprc": float(y_te.mean())})
    return {"split": split, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="folder with INSPIRE CSVs (operations, labs, ward_vitals, diagnosis, medications)")
    ap.add_argument("--subjects-dir", help="folder with died/ and survived/ JSONs (default: <dataset-dir>/subjects)")
    ap.add_argument("--cohort", choices=["subset", "full"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-news2", action="store_true")
    ap.add_argument("--no-possum", action="store_true")
    ap.add_argument("--cohort-cache", help="CSV to save/load the cohort table (with NEWS2) to skip re-parsing ward vitals")
    ap.add_argument("--dnn-predictions", help="predictions.jsonl from a v4 run on the SAME split; enables paired tests")
    a = ap.parse_args()
    t0 = time.time()
    if a.cohort_cache and os.path.exists(a.cohort_cache):
        cohort = pd.read_csv(a.cohort_cache)
        print(f"loaded cohort cache {a.cohort_cache}")
    else:
        ids = select_subjects(a.subjects_dir or os.path.join(a.dataset_dir, "subjects"),
                              10000 if a.cohort == "subset" else None)
        ops = pd.read_csv(os.path.join(a.dataset_dir, "operations.csv"))
        cohort = build_cohort(ops, ids)
    print(f"{a.cohort}: {len(cohort)} patients, {cohort['label'].sum()} positives, "
          f"ASA VI patients {int((cohort['asa'] == 6).sum())} ({int(cohort.loc[cohort['asa'] == 6, 'label'].sum())} deaths)")
    if not a.no_news2 and "news2_max" not in cohort:
        n2 = news2_for_cohort(os.path.join(a.dataset_dir, "ward_vitals.csv"), cohort)
        cohort = cohort.merge(n2, on="subject_id", how="left")
        print(f"NEWS2 available for {int(cohort['news2_available'].sum())} patients ({time.time() - t0:.0f}s)")
        if a.cohort_cache:
            cohort.to_csv(a.cohort_cache, index=False)
    if not a.no_possum and "possum_ps" not in cohort:
        ps = possum_for_cohort(a.dataset_dir, cohort)
        cohort = cohort.merge(ps, on="subject_id", how="left")
        print(f"POSSUM computed; median physiology items observed {cohort['possum_physiology_observed'].median():.0f}/8 "
              f"({time.time() - t0:.0f}s)")
        if a.cohort_cache:
            cohort.to_csv(a.cohort_cache, index=False)
    dnn = None
    if a.dnn_predictions:
        dnn = {}
        with open(a.dnn_predictions) as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == "test":
                    dnn[int(r["patient_id"])] = float(r["probability"])
        print(f"loaded {len(dnn)} DNN test predictions")
    out = {"cohort": a.cohort, "n_patients": len(cohort), "n_positive": int(cohort["label"].sum()),
           "n_asa6": int((cohort["asa"] == 6).sum()), "n_asa6_deaths": int(cohort.loc[cohort["asa"] == 6, "label"].sum())}
    print("== all patients ==")
    out["all"] = run(cohort, dnn=dnn)
    print("== ASA VI excluded ==")
    out["excl_asa6"] = run(cohort, exclude_asa6=True, dnn=dnn)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
