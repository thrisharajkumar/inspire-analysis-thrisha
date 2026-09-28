# What changed

## v3 (current) — architecture and anti-memorisation

Two notebooks, one codebase (`python build_notebook.py` writes both):
`INSPIRE_DNN_v3_full_cohort.ipynb` (`RUN_PROFILE='full'`: all ~99,886 patients, every analysis on,
no run-time or memory budget) and `INSPIRE_DNN_v3_subset_analysis.ipynb` (`RUN_PROFILE='subset'`).

| Change | CONFIG switch | Why |
|---|---|---|
| Eight equal organ systems; every system = its measurements over time (if any) + its own ICD-10 chapter features (`sysdx_*`) + its related summary features. GI/MSK are organ systems with no time series, not a special case | — (always) | consistency; the heart system now also knows about circulatory diagnoses, etc. |
| One shared organ-system network with a per-system tag and per-system input layer | `SHARE_ENCODER` | ~8x fewer encoder parameters; small systems borrow from data-rich ones — less memorisation |
| Whole-patient layer: learns from all features, guides each system network through a narrow bottleneck, feeds arbitration, has its own risk term | `USE_WHOLE_PATIENT_LAYER` | your "learns from everything, supports each system" layer |
| Inter-system (coupling) layer pre-trained by **masked-system prediction** on every patient, then fine-tuned | `MASKED_SYSTEM_WEIGHT` | learns physiology from ~99,900 patients instead of ~290 deaths — the fix for unstable coupling maps |
| Gentle fine-tuning (pre-trained parts at 0.1x LR), AdamW | `FINETUNE_ENCODER_LR_MULT`, `WEIGHT_DECAY` | memorisation happens in the supervised step; keep it small |
| Organ-system dropout during fine-tuning | `SYSTEM_DROPOUT` | no over-reliance on one system; robust to missing systems |
| SMOTENC amplification capped per department x ASA stratum | `SMOTE_MAX_AMPLIFICATION` (5) | uncapped, full-cohort simulation gave 85x from 3 real deaths in one stratum |
| Balanced batches (~10% deaths per batch) | `BALANCED_SAMPLER`, `SAMPLER_TARGET_POSITIVE_RATE` | at 1:200, ~29% of batches had no deaths |
| Part 6 speed fix: diagnoses/medications grouped by patient once | — | scans grew ~80x at full scale; outputs unchanged |
| Diagnostics: bootstrap CIs, gradient boosting baseline, N-seed ensemble, coupling stability across ensemble, learning curve, train/val gap, SMOTE-donor memorisation check, leakage (own-data share) check, share-of-risk report | `N_ENSEMBLE`, `RUN_GBM_BASELINE`, `RUN_LEARNING_CURVE`, `N_BOOTSTRAP` | shows whether the model learns or memorises |
| Ablation rebuilt: organ systems only → + coupling → + arbitration → full; separate encoders; no masked pre-training; concat | `SKIP_FUSION_ABLATION` | each piece must earn its place |

The manual cardio→renal mode was removed in v3 (`COUPLING_MODE` is `'learned'` or `'none'`).
The static→organ-system mapping is printed in Part 9.6 — **needs surgeon sign-off**
(placements of lactate, temperature, CRP and the aggregate features are judgement calls).

# v1 → v2

Base: `src/INSPIRE_Multimodal_Mortality_Benchmark (1).ipynb` (the 10,942-patient run:
AUPRC 0.658, AUROC 0.967, 66,472 parameters, SMOTENC 0 synthetic). Every change is a
`CONFIG` switch, so each can be ablated one at a time. The verification summary at the
end of the notebook (Part 12.2) reports each check's result for the run.

## Bugs fixed

| # | What was wrong | Effect in v1 | Fix | Where |
|---|---|---|---|---|
| 1 | SMOTENC got a **boolean mask** for `categorical_features`; imbalanced-learn 0.14.x raises "truth value … ambiguous", swallowed by a bare `except ValueError` | 0 synthetic patients on every run | integer indices; errors counted separately from size skips; `STRICT_SAMPLING` raises on any error | 8.2 |
| 1b | Static feature **values** stored in build order while `STATIC_FEATURE_NAMES` is **sorted**; every by-name column pick was misaligned | v1: SMOTENC's categorical columns were the wrong ones (latent — SMOTENC never ran). v2 first Colab run: the GI/MSK branches read WBC stats, NEWS2 SMOTENC check read wrong columns | values returned in sorted order + an assertion in Part 7 | 6.14, 7.4 |
| 2 | `pos_weight` after Tomek subtracted removed **survivors** from the **death** count | latent (never ran because of #1) — would have mis-weighted the loss the moment #1 was fixed | `n_pos = real deaths + kept synthetic` | 8.2 |
| 3 | Synthetic patients got a completely empty, fully masked time series, and all are labelled "died" | shortcut "no data anywhere ⇒ died" available in training | synthetic patients borrow the jittered series of the nearest real death in the same dept × ASA stratum (`SYNTHETIC_TS_MODE`) | 8.2c, 9.1 |
| 4 | Manual cardio→renal side-input was `cardio_x.mean(dim=1)[:, :3]` = **CK, CK-MB, troponin**, not HR / MAP / IABP as documented | the documented cardiorenal coupling was never what the model received | replaced by the learned coupling layer; the `manual` ablation now feeds the three named features | 9.6 |
| 5 | Best-F1 threshold chosen **on the test set** | the "76% caught / 60% precision" figure is slightly optimistic | threshold chosen on validation, applied to test; test-optimal shown for reference | 11.1b |
| 6 | Jitter applied to discrete items (GCS, CRRT/IABP/vent/ECMO flags) | impossible values (GCS 3.98); caught by the new NEWS2 before/after check | `JITTER_EXCLUDE_ITEMS` | 9.1 |
| 7 | Augmentation seeds used Python `hash()` (randomised per process) | augmentation not reproducible between sessions | `zlib.crc32` seed | 9.1 |
| 8 | Unconditional `drive.mount()` | crashes on Kaggle/local | Colab-only | Part 2 |
| 9 | Attention audit skipped the pre-norm of a `norm_first` layer | audit weights slightly off | applies `norm1` first | 11.3 |
| 10 | Printed notes still described the 30-patient dev subset | misleading text on real runs | printed from the actual run | several |

## Additions

- **Encoder capacity** (`ENCODER_SIZE`): v1 was 66k parameters in total, and only ~17.6k of
  that was the six system encoders. `medium` (default) ≈ 161k in the encoders (≈ 278k total);
  `small` reproduces v1; `large` ≈ 617k. Bigger is not automatically better with a few
  hundred deaths — Part 11.4 compares variants on the same split.
- **NEWS2** (`INCLUDE_NEWS2`): computed from pre-op ward vitals in observation sets
  (1-hour buckets, bounded 4-hour carry-forward, ≥ 3 of the 5 measured vitals needed).
  Included **only if** the RCP band-edge self-test passes. Checks: mortality by worst band
  (construct validity), before/after SMOTENC, before/after jitter + time-mask.
  Approximations for sign-off: Scale 1 for all; GCS stands in for AVPU; no FiO2 = air.
- **Learned inter-system coupling** (`COUPLING_MODE='learned'`): `SystemCouplingLayer` —
  each system reads the *other* systems by attention (own token masked). **No link is
  defined by hand**: the hand-wired renal side-input and the hand-crafted
  `renal_cardiac_interaction` feature are both removed in this mode, and
  `COUPLING_PRIOR_LINKS=[]`. A per-system gate (starts at 0.1) controls how much each system
  uses what it hears.
- **Surgeon validation of the learned coupling** (11.2b): "who listens to whom" heatmap
  in plain body-system names; network diagrams for died vs. survived; how much each
  system's answer changed because of coupling; a CSV validation sheet listing each link in
  plain words with a bootstrap CI and blank "plausible? / comment" columns.
  **11.4b** retrains extra seeds and reports whether the same links re-appear — the gate
  for whether individual links are worth showing a surgeon at all.
- **Arbitration layer** (`USE_ARBITRATION_LAYER`): each system gives its answer f_s; the
  layer gives it a say w_s per patient (average 1, starts equal), read from the patient
  context by default. Logit = bias + f_context + Σ w_s·f_s — still an exact additive
  breakdown. 11.2c shows the say per department.
- **GI and MSK as their own branches** (`SEPARATE_GI_MSK_BRANCHES`): in v1 they lived
  inside the single static term.
- **Evaluation**: model selection on validation AUPRC; Platt recalibration on validation;
  logistic-regression baseline on the same features and split; architecture ablation
  (no coupling / manual / learned / learned + arbitration / concat).

## Replaced files (earlier conversion)

`config.py, data_loading.py, feature_engineering.py, imputation.py, sampling.py, model.py,
train.py, evaluate.py, news2_engine.py, news2_integration.py, system_correlation_layer.py,
test_model_wiring.py` → replaced by `pipeline/`, `inspire_dnn/`, `tests/`. Two problems in
the old versions besides format: the `from previous_stage import *` chaining drops every
`_underscore` global (e.g. `_HOLD_IDS`) and misses later re-assignments; and
`SystemCorrelationLayer` included self-attention and reported its matrix with sender and
receiver swapped.
