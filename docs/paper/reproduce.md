# Reproduce

## 1. Data

INSPIRE is distributed by PhysioNet under a credentialed data-use agreement (Lim et al.,
*Scientific Data* 11:655, 2024). After access is granted you need:

- `SUBJECTS`: per-patient JSON folders `died/` and `survived/`.
- `CSVS`: `operations.csv`, `labs.csv`, `ward_vitals.csv`, `diagnosis.csv`, `medications.csv`.

No patient data is stored in this repository.

## 2. Environment

Python 3.10+ with `torch`, `scikit-learn`, `imbalanced-learn`, `pandas`, `pyarrow`,
`matplotlib`. CPU is sufficient for the subset (about 40 minutes per full run).

```bash
cd src-dnn
pip install -r requirements.txt
```

## 3. Main result (preprint subset)

```bash
python run_pipeline.py --subjects-dir "$SUBJECTS" \
  --config '{"MAX_SUBJECTS_PER_CLASS": 10000, "BATCH_SIZE": 256,
             "SKIP_IMPUTATION_ACCURACY_BENCHMARKS": true, "SKIP_FUSION_ABLATION": true}'
```

This loads all `died/` records plus 10,000 survivors (seed 42), and runs all stages: labels,
features, NEWS2, imputation, split and sampling, training, evaluation and diagnostics. The
reference log is [`src-dnn/EXPERIMENT1.txt`][log]. Check the cohort lines: 10,942 patients,
469 deaths, test set 2,189 / 94.

## 4. Cross-learning ablation and POSSUM

```bash
OUT=ablation_subset; mkdir -p $OUT
# clinical baselines incl. POSSUM (~2 min)
python tools/clinical_baselines.py --dataset-dir "$CSVS" --subjects-dir "$SUBJECTS" \
  --cohort subset --cohort-cache $OUT/cohort_subset.csv --out $OUT/baselines_subset.json
# A / B / C models (~40 min)
python tools/run_ladder.py --subjects-dir "$SUBJECTS" --out $OUT
# paired comparison table -> $OUT/ladder_comparison.md
python tools/ladder_compare.py --ladder-dir $OUT --cohort-cache $OUT/cohort_subset.csv
```

Details and built-in checks: [`src-dnn/ABLATION_LADDER.md`][ladder].

## Code map

| Path | Content |
|---|---|
| `src-dnn/run_pipeline.py` | Runs the numbered stages in one namespace |
| `src-dnn/pipeline/00–12_*.py` | Setup, data loading, labels, features, NEWS2, imputation, split/sampling, model, training, evaluation, ablations, wrap-up |
| `src-dnn/inspire_dnn/layers.py` | All model components (encoder, whole-patient, coupling, arbitration, fusion) |
| `src-dnn/inspire_dnn/news2.py` | NEWS2 scoring with RCP self-tests |
| `src-dnn/tools/` | Clinical baselines, POSSUM, ablation ladder and comparison |
| `src-dnn/tests/` | Unit tests (`python -m pytest -q tests`, no data needed) |

[log]: https://github.com/thrisharajkumar/inspire-analysis-thrisha/blob/main/src-dnn/EXPERIMENT1.txt
[ladder]: https://github.com/thrisharajkumar/inspire-analysis-thrisha/blob/main/src-dnn/ABLATION_LADDER.md
