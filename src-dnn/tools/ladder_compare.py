"""Paired comparison of the A/B/C ladder with each other and with clinical baselines.

    python tools/ladder_compare.py --ladder-dir <run_ladder --out> --cohort-cache <cohort_subset.csv>

The cohort cache is written by clinical_baselines.py (--cohort subset --cohort-cache ...) and
holds ASA, age, sex, emergency, NEWS2 and POSSUM per patient. All comparisons use the same test
patients; differences are paired patient-bootstrap (2,000 resamples).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
NAMES = {"A_isolated_organs": "A: isolated organs", "B_plus_cross_learning": "B: + cross-learning",
         "C_full_model": "C: full model"}


def ci(y, s, fn, n, rng):
    idx = np.arange(len(y))
    v = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() != y[b].max():
            v.append(fn(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def paired(y, a, b, n, rng):
    idx = np.arange(len(y))
    out = {}
    for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
        d = []
        for _ in range(n):
            s = rng.choice(idx, len(idx), replace=True)
            if y[s].min() != y[s].max():
                d.append(fn(y[s], a[s]) - fn(y[s], b[s]))
        d = np.array(d)
        out[name] = (fn(y, a) - fn(y, b), *np.percentile(d, [2.5, 97.5]))
    return out


def fmt(d):
    sig = "*" if (d[1] > 0 or d[2] < 0) else ""
    return f"{d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f}){sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-dir", required=True)
    ap.add_argument("--cohort-cache", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(2026)

    preds = {}
    for k in NAMES:
        p = os.path.join(a.ladder_dir, f"pred_{k}.csv")
        if os.path.exists(p):
            preds[k] = pd.read_csv(p).set_index("subject_id")
    ids = list(next(iter(preds.values())).index)
    y = next(iter(preds.values())).loc[ids, "label"].values.astype(int)

    # Clinical baselines on the same split (reconstructed exactly as clinical_baselines.py).
    co = pd.read_csv(a.cohort_cache).set_index("subject_id")
    all_ids = co.index.tolist()
    tr, hold = train_test_split(all_ids, test_size=0.4, stratify=co["label"].values, random_state=SEED)
    va, te = train_test_split(hold, test_size=0.5, stratify=co.loc[hold, "label"].values, random_state=SEED)
    assert set(te) == set(ids), "ladder test set differs from the baseline split"
    assert (co.loc[ids, "label"].values == y).all(), "labels differ"
    X = co.fillna(co.loc[tr].median(numeric_only=True))
    cols = ["asa", "age", "male", "emop", "news2_max", "news2_mean", "news2_available"]
    mu, sd = X.loc[tr, cols].mean(), X.loc[tr, cols].std() + 1e-9
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((X.loc[tr, cols] - mu) / sd, X.loc[tr, "label"])
    base = {"P-POSSUM (pre-op items)": X.loc[ids, "p_possum_logit"].values,
            "7-variable LR (ASA, age, sex, emergency, NEWS2)": lr.predict_proba((X.loc[ids, cols] - mu) / sd)[:, 1]}

    scores = {NAMES[k]: v.loc[ids, "probability"].values for k, v in preds.items()}
    scores.update(base)
    lines = [f"# Cross-learning ladder (subset test set: {len(y):,} patients, {int(y.sum())} deaths)", "",
             "## Discrimination (95% CI, patient bootstrap)", "", "| Model | AUROC | AUPRC |", "|---|---|---|"]
    res = {"n_test": len(y), "deaths": int(y.sum()), "models": {}, "paired": {}}
    for n_, s in scores.items():
        au, ap_ = roc_auc_score(y, s), average_precision_score(y, s)
        c1, c2 = ci(y, s, roc_auc_score, a.n_boot, rng), ci(y, s, average_precision_score, a.n_boot, rng)
        res["models"][n_] = {"auroc": au, "auroc_ci": list(c1), "auprc": ap_, "auprc_ci": list(c2)}
        lines.append(f"| {n_} | {au:.3f} ({c1[0]:.3f}-{c1[1]:.3f}) | {ap_:.3f} ({c2[0]:.3f}-{c2[1]:.3f}) |")

    pairs = []
    have = [NAMES[k] for k in NAMES if k in preds]
    if len(have) >= 2:
        pairs += [(have[1], have[0], "effect of cross-learning (B - A)")]
    if len(have) == 3:
        pairs += [(have[2], have[1], "effect of whole-patient + arbitration (C - B)"),
                  (have[2], have[0], "all mixing vs isolated organs (C - A)")]
    for m in have:
        for b in base:
            pairs.append((m, b, f"{m.split(':')[0]} vs {b.split(' (')[0]}"))
    lines += ["", "## Paired differences (* = 95% CI excludes 0)", "",
              "| Comparison | dAUROC (95% CI) | dAUPRC (95% CI) |", "|---|---|---|"]
    for m1, m2, label in pairs:
        p = paired(y, scores[m1], scores[m2], a.n_boot, rng)
        res["paired"][label] = {k: list(map(float, v)) for k, v in p.items()}
        lines.append(f"| {label} | {fmt(p['auroc'])} | {fmt(p['auprc'])} |")

    md = "\n".join(lines)
    open(os.path.join(a.ladder_dir, "ladder_comparison.md"), "w").write(md)
    json.dump(res, open(os.path.join(a.ladder_dir, "ladder_comparison.json"), "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
