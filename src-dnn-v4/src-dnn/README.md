# src-dnn — INSPIRE organ-system mortality DNN (v4)

**New here? Start with [`INSPIRE_DNN_v4_explained_simply.md`](INSPIRE_DNN_v4_explained_simply.md)** —
the whole pipeline, every step and every term, in plain words.

The code behind the three notebooks in `notebooks/` — `INSPIRE_DNN_v4_subset_analysis.ipynb` (sample
analysis), `INSPIRE_DNN_v4_full_cohort.ipynb` (whole cohort, Kaggle/Colab) and
`INSPIRE_DNN_v4_isambard.ipynb` (whole cohort, big GPU) — reworked from
`src/INSPIRE_Multimodal_Mortality_Benchmark (1).ipynb`. **The stage files in `pipeline/`
are the single source of truth; the notebook is generated from them**, so the two cannot drift.

```
src-dnn/
├── pipeline/                 # one file per notebook Part, "percent" cell format (# %%)
│   ├── 00_setup.py           # imports, CONFIG (every switch), profiles subset/full/isambard, figure saving
│   ├── 01_data_loading.py    # chunked JSON -> Parquet parse + cache
│   ├── 02_labels_and_eda.py  # 30-day label recompute, EDA
│   ├── 03_organ_system_features.py   # + v4: procedure codes & medicine families per organ system, anaesthesia type
│   ├── 04_news2.py           # NEW: NEWS2 + self-test gate + validity check
│   ├── 05_patient_bundle.py
│   ├── 06_imputation.py
│   ├── 07_split_and_sampling.py   # SMOTENC fix, NEWS2 before/after SMOTENC, synthetic donors
│   ├── 08_dataset_and_model.py    # dataset, NEWS2 before/after jitter, model assembly, layer-size table
│   ├── 09_training.py        # + v4: per-epoch diary by emergency/scheduled, embedding snapshots, model save + reload check
│   ├── 10_evaluation.py      # threshold on val, recalibration, LR baseline, coupling plots
│   ├── 11_ablations_and_sensitivity.py  # architecture ablation (+ v4 rows), coupling stability, trajectories
│   ├── 12_representation_and_explainability.py  # v4 Part 13: curves, embedding maps, clusters, similarity,
│   │                                            #   emergency/scheduled tables, NEWS2 checks, SHAP, size sweep, reuse test
│   └── 13_wrapup.py          # Part 14: timing + one-table verification summary
├── inspire_dnn/              # pure, data-free code (inlined into the notebook at build time)
│   ├── news2.py              # vectorised NEWS2 + RCP self-test + before/after comparison
│   ├── layers.py             # encoders, coupling, arbitration, NAM, surgical-context term, model
│   ├── explain.py            # v4: expected gradients (SHAP-style) inside one risk term
│   └── analysis.py           # v4: groups, metrics by group, 2-D maps, enrichment, clusters, plots
├── tests/                    # pytest: NEWS2 bands, layer wiring, SMOTENC bug pin, v4 explain + analysis
├── isambard/                 # v4: Slurm script + how to run on Isambard-AI
├── tools/make_synthetic_subjects.py   # fake INSPIRE-format cohort for crash-testing
├── run_pipeline.py           # run all stages in one namespace (= Run All in the notebook)
├── build_notebook.py         # pipeline/ -> notebook   (--check to detect drift)
├── CHANGES.md                # what changed in every version, and why
└── INSPIRE_DNN_v4_explained_simply.md   # plain-language guide to everything
```

## Running

```bash
pip install -r requirements.txt
python -m pytest -q tests                                  # ~20 s, no data needed
python run_pipeline.py --subjects-dir /path/to/subjects    # folder with died/ and survived/
python run_pipeline.py --subjects-dir ... --config '{"COUPLING_MODE": "none", "USE_ARBITRATION_LAYER": false}'
python run_pipeline.py --profile isambard --subjects-dir ...   # full cohort on a big GPU (see isambard/)
python build_notebook.py                                   # rebuilds ALL THREE notebooks after any edit
python build_notebook.py --check                           # fails if a notebook drifted from pipeline/
```

On Kaggle/Colab just open the notebook and Run All; it needs nothing from this repo.
Everything a run produces lands in `inspire_outputs/` (on Kaggle: `/kaggle/working/inspire_outputs`):
`figures/` (every plot as PNG), `models/`, `snapshots/`, `tables/` (every results table as CSV).
Quick variants without editing CONFIG: set the environment variable
`INSPIRE_CONFIG_OVERRIDES='{"ENCODER_SIZE": "small"}'` before running.

## Editing workflow

1. Change a stage in `pipeline/` (or a module in `inspire_dnn/`).
2. `python -m pytest -q tests` and, for anything touching the data path, a dry run on fake
   data: `python tools/make_synthetic_subjects.py --out /tmp/fake && python run_pipeline.py --subjects-dir /tmp/fake --config '{"EPOCHS_PRETRAIN": 2, "EPOCHS_FINETUNE": 3}'`.
3. `python build_notebook.py`, commit both.

The earlier conversion in this folder (`config.py`, `model.py`, `news2_engine.py`,
`system_correlation_layer.py`, …) is superseded; see CHANGES.md §"Replaced files".
