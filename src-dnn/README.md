# smote_pipeline

`INSPIRE_Multimodal_Mortality_Benchmark.ipynb`, converted into scripts. Run in order:
`config.py` -> `data_loading.py` -> `feature_engineering.py` -> `news2_integration.py`
-> `imputation.py` -> `sampling.py` -> `model.py` -> `train.py` -> `evaluate.py`.

## How the conversion was done, and what that means for trust

Each stage does `from <previous_stage> import *` at the top -- the same shared-global-
state execution the notebook already relied on (dozens of cells reference `PATIENT_BUNDLE`,
`CONFIG`, `COHORT_INDEXED`, etc. built by earlier cells), just split into files you can
read, diff, and run individually rather than one 141-cell notebook. **Stages not listed
below as changed are preserved as close to verbatim as possible** -- the goal was
converting the format, not rewriting logic I can't test against your real data.

Two notebook sections were deliberately NOT converted yet -- they're genuine appendices
(ablation comparisons, sensitivity analyses, risk-trajectory plots, the experiment log),
not core pipeline, and converting them risked introducing bugs in code I have no way to
verify without your real data. They remain notebook-only for now; say if you want them
next.

## What changed, and what was actually verified (not just written)

### 1. The SMOTENC fix (`sampling.py`) -- confirmed live bug, not theoretical

Reproduced against your actual installed `imbalanced-learn` (0.14.2): `SMOTENC`'s
`categorical_features` argument raises `ValueError: The truth value of an array with
more than one element is ambiguous` when given a boolean mask directly -- which is
exactly what `grouped_smotenc()` was doing. Because that error was caught by the same
bare `except ValueError` used for legitimate "stratum too small" skips, **every stratum
was silently failing this way, and grouped_smotenc generated zero synthetic patients,
every run.** Fixed with integer indices (`np.where(mask)[0].tolist()`).

### 2. NEWS2 integration (`news2_integration.py`, new file) -- augmentation check, before and after

- Computes NEWS2 (Royal College of Physicians, 2017) per patient from data already
  collected (respiratory rate, SpO2, oxygen use, temperature, systolic BP, heart rate,
  consciousness), using the exact same `get_window()`/`windowed_series()` helpers Part 6
  already uses elsewhere -- no new data plumbing needed.
- Merges 5 NEWS2 summary features (max/latest/mean/trend-slope/ever-high-risk) into
  every patient's static vector, **before** sampling runs -- so SMOTENC treats NEWS2 as
  an ordinary continuous feature and legitimately interpolates it for synthetic
  patients, rather than NEWS2 being computed only after the fact.
- `check_news2_before_after_augmentation()` -- wired into `sampling.py` right after
  `tomek_cleanup()` -- compares the real ("before") vs. real+synthetic ("after") NEWS2
  distribution, flags any synthetic patient with an implausible score (outside [0, 20]),
  and runs a KS test. This is the concrete "check before and after" for the
  augmentation step.
- NEWS2's scoring bands were unit-tested separately against the official RCP table,
  including exact boundary values (SBP 90 vs. 91, HR 130 vs. 131, temp 39.0 vs. 39.1) --
  all pass. **One documented approximation** worth your surgeon's input: NEWS2 normally
  uses AVPU for consciousness, substituted here with GCS=15 for "Alert."

### 3. System correlation layer (`system_correlation_layer.py` + wired into `model.py`)

Your existing architecture hand-codes exactly one cross-system link (cardiovascular
summary stats fed into the renal branch, encoding cardiorenal syndrome). This adds a
general learned-attention layer across ALL organ systems on top of that -- **it does
not remove the existing hand-coded link**, it adds a mechanism that can discover other
relationships too (e.g. respiratory<->cardiovascular), and reports exactly which
systems it found move together (`model_output["correlation_report"]["top_correlations"]`),
so it can be sanity-checked against known physiology.

**Verified with a real forward pass** (`test_model_wiring.py` -- run it, no real data
needed): correct output shapes, `correlation_report` populated with sensible-looking
top pairs, and — critically — the NAM fusion's per-system interpretability output is
still present and decomposable, so this addition doesn't quietly break the
architecture's main interpretability claim. Also confirmed: setting
`CONFIG["USE_SYSTEM_CORRELATION_LAYER"] = False` runs the original architecture
unchanged, for a clean ablation between the two.

## Running it

```bash
python config.py            # or just start importing from data_loading.py onward --
python data_loading.py      # each stage's header comment says what it needs
python feature_engineering.py
python news2_integration.py
python imputation.py
python sampling.py
python model.py
python train.py
python evaluate.py

# Before a real run, a fast sanity check on the model changes specifically:
python test_model_wiring.py
```

Since each file does `from <previous> import *`, running any single stage (e.g. just
`python train.py`) transitively runs everything before it in order -- same as running
notebook cells 2 through 111 top to bottom.
