# src-dnn — INSPIRE organ-system mortality DNN (v3)

The code behind the two notebooks in `notebooks/` — `INSPIRE_DNN_v3_full_cohort.ipynb` (whole cohort)
and `INSPIRE_DNN_v3_subset_analysis.ipynb` (sample analysis) — reworked from
`src/INSPIRE_Multimodal_Mortality_Benchmark (1).ipynb`. **The stage files in `pipeline/`
are the single source of truth; the notebook is generated from them**, so the two cannot drift.

```
src-dnn/
├── pipeline/                 # one file per notebook Part, "percent" cell format (# %%)
│   ├── 00_setup.py           # imports, CONFIG (every v2 switch lives here)
│   ├── 01_data_loading.py    # chunked JSON -> Parquet parse + cache
│   ├── 02_labels_and_eda.py  # 30-day label recompute, EDA
│   ├── 03_organ_system_features.py
│   ├── 04_news2.py           # NEW: NEWS2 + self-test gate + validity check
│   ├── 05_patient_bundle.py
│   ├── 06_imputation.py
│   ├── 07_split_and_sampling.py   # SMOTENC fix, NEWS2 before/after SMOTENC, synthetic donors
│   ├── 08_dataset_and_model.py    # dataset, NEWS2 before/after jitter, model assembly
│   ├── 09_training.py
│   ├── 10_evaluation.py      # threshold on val, recalibration, LR baseline, coupling plots
│   ├── 11_ablations_and_sensitivity.py  # architecture ablation, coupling stability, trajectories
│   └── 12_wrapup.py          # timing + one-table verification summary
├── inspire_dnn/              # pure, data-free code (inlined into the notebook at build time)
│   ├── news2.py              # vectorised NEWS2 + RCP self-test + before/after comparison
│   └── layers.py             # encoders, SystemCouplingLayer, SystemArbitrationLayer, NAM, model
├── tests/                    # pytest: NEWS2 bands, layer wiring, SMOTENC bug pin
├── tools/make_synthetic_subjects.py   # fake INSPIRE-format cohort for crash-testing
├── run_pipeline.py           # run all stages in one namespace (= Run All in the notebook)
├── build_notebook.py         # pipeline/ -> notebook   (--check to detect drift)
├── CHANGES.md                # what changed vs. the Benchmark notebook, and why
└── RUN_REPORT_subjects_sample.md   # results of every check on the 4,097-patient sample
```

## Running

```bash
pip install -r requirements.txt
python -m pytest -q tests                                  # ~20 s, no data needed
python run_pipeline.py --subjects-dir /path/to/subjects    # folder with died/ and survived/
python run_pipeline.py --subjects-dir ... --config '{"COUPLING_MODE": "none", "USE_ARBITRATION_LAYER": false}'
python build_notebook.py                                   # rebuilds BOTH notebooks after any edit
python build_notebook.py --check                           # fails if a notebook drifted from pipeline/
```

On Kaggle/Colab just open the notebook and Run All; it needs nothing from this repo.
Quick variants without editing CONFIG: set the environment variable
`INSPIRE_CONFIG_OVERRIDES='{"ENCODER_SIZE": "small"}'` before running.

## Editing workflow

1. Change a stage in `pipeline/` (or a module in `inspire_dnn/`).
2. `python -m pytest -q tests` and, for anything touching the data path, a dry run on fake
   data: `python tools/make_synthetic_subjects.py --out /tmp/fake && python run_pipeline.py --subjects-dir /tmp/fake --config '{"EPOCHS_PRETRAIN": 2, "EPOCHS_FINETUNE": 3}'`.
3. `python build_notebook.py`, commit both.

The earlier conversion in this folder (`config.py`, `model.py`, `news2_engine.py`,
`system_correlation_layer.py`, …) is superseded; see CHANGES.md §"Replaced files".
