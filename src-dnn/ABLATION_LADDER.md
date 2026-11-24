# Cross-learning ablation (A/B/C) and POSSUM baseline: preprint subset

**Goal.** Two additions to the preprint:

1. **Does cross-learning between organ systems help?** Train three versions of the model on the
   same data, split and seed, with the components switched on one at a time:

   | Run | Coupling (cross-learning) | Whole-patient layer + arbitration | Meaning |
   |---|---|---|---|
   | **A** isolated organs | off | off | each organ predicts only from its own data; risk = sum of organ terms |
   | **B** + cross-learning | on | off | organs can exchange information |
   | **C** full model | on | on | the model reported in the preprint |

   B vs A is the effect of cross-learning; C vs B is the effect of the whole-patient layer.

2. **POSSUM comparison.** Pre-operative POSSUM physiology score and P-POSSUM risk, on the same
   test patients, alongside ASA, NEWS2 and a 7-variable logistic regression.

**Data scope.** The preprint subset only: all 942 "died" records + 10,000 survivors sampled
with seed 42 (10,942 patients, 469 deaths; test set 2,189 patients, 94 deaths). This matches
`EXPERIMENT1.txt`. It is **not** a full-cohort run.

## Requirements

Python with `torch`, `imbalanced-learn`, `pyarrow`, `scikit-learn`, `pandas`, `matplotlib`
(the same environment used for the preprint run). Run from the `src-dnn/` folder.

You need two folders:
- `SUBJECTS`: the per-patient JSON folder with `died/` and `survived/` (as used for the preprint run).
- `CSVS`: the INSPIRE CSV folder with `operations.csv`, `labs.csv`, `ward_vitals.csv`,
  `diagnosis.csv`, `medications.csv`.

## Steps

```bash
cd src-dnn
OUT=ablation_subset            # any new folder
mkdir -p $OUT

# 1. Clinical baselines + POSSUM (about 2 min; writes the per-patient cohort table used in step 3)
python tools/clinical_baselines.py --dataset-dir "$CSVS" --subjects-dir "$SUBJECTS" \
    --cohort subset --cohort-cache $OUT/cohort_subset.csv --out $OUT/baselines_subset.json

# 2. A/B/C ladder (about 40 min on CPU: data parsing + pipeline stages 00-09 + three trainings)
python tools/run_ladder.py --subjects-dir "$SUBJECTS" --out $OUT

# 3. Paired comparison table
python tools/ladder_compare.py --ladder-dir $OUT --cohort-cache $OUT/cohort_subset.csv
```

## Outputs to send back

From `$OUT/`:
- `ladder_comparison.md`: the results table (AUROC/AUPRC with 95% CIs, paired differences)
- `ladder_meta.json`: run settings and per-variant scores
- `baselines_subset.json`: clinical baselines incl. POSSUM

Per-patient prediction files (`pred_*.csv`, `cohort_subset.csv`) contain patient IDs: keep them
local, do not share outside the project.

## Checks built in

- Step 1 reconstructs the preprint split; it should print
  `split: {'train': 6565, 'val': 2188, 'test': 2189, ... 'test_pos': 94}`.
- Step 3 stops if the ladder's test patients or labels differ from the baseline split.
- A, B and C are trained by the same function (`train_full_model`, seed 42), so differences
  come from the switched components, not from the recipe. C is retrained here, so its score
  can differ slightly from the preprint's 0.621 (CPU/platform variation).

## Files

| File | Purpose |
|---|---|
| `tools/run_ladder.py` | runs stages 00-09 with the preprint subset settings, trains A/B/C, saves test predictions |
| `tools/ladder_compare.py` | paired bootstrap comparisons: B-A, C-B, C-A, and each model vs P-POSSUM and the 7-variable LR |
| `tools/clinical_baselines.py` | rebuilds the exact cohort/split; ASA, NEWS2, emergency, LR baselines and POSSUM |
| `tools/possum.py` | pre-operative POSSUM / P-POSSUM scoring (items and limitations documented in the file) |

## POSSUM limitations (for the paper)

Operative severity, multiple procedures, blood loss and peritoneal soiling are not known before
surgery and score the minimum (1). Cardiac and respiratory history and ECG are approximated from
pre-operative diagnosis codes and medications. Missing physiology also scores 1. In the subset,
the median patient has 2 of 8 physiological measurements in the 5-day window.
