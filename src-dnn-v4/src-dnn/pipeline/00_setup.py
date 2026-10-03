# %% [markdown]
# # INSPIRE Perioperative Mortality — Organ-System DNN (v3)
#
# PROFILE_BANNER
#
# **What v3 is:** eight equal organ systems (kidneys, heart & circulation, lungs, metabolism
# & liver, blood, brain & nerves, digestive, bones & joints) — each built from its own
# measurements (where the dataset has any) and its own ICD-10 diagnosis chapter — read by
# one shared organ-system network; a **whole-patient layer** that learns from every feature
# and guides each system network; a **learned inter-system layer** pre-trained on every
# patient by predicting a hidden organ system from the others; an **arbitration layer**
# setting each system's say; and an additive output, so every prediction splits exactly
# into per-system contributions. Anti-memorisation: label-free pre-training, gentle
# fine-tuning, AdamW, organ-system dropout, capped SMOTENC, balanced batches, a seed
# ensemble — plus diagnostics that show whether it is learning or memorising.
# Every change is a `CONFIG` switch; the full list is in `src-dnn/CHANGES.md`.
# Generated from `src-dnn/pipeline/` — edit there, then run `python build_notebook.py`.
#
# Predicts 30-day post-surgical mortality using per-organ-system encoders (renal,
# cardiovascular, respiratory, metabolic/hepatic, haematology, neurological, GI, MSK),
# fused through a Neural Additive Model so every prediction stays explainable
# system-by-system. **For the full reasoning behind every design choice below, see the
# project's docs site** (`Multimodal_Notebook_Summary.md`, `roadmap_and_architecture.md`) —
# this notebook is deliberately code-first.
#
# **Tuned for a real, single Colab/Kaggle session:** default config runs on a
# ~10,000-patient sample (all real deaths kept, survivors capped — see Part 3), targets
# under 20 minutes total, checkpoints training every few epochs, and caches the parsed data
# so a session that dies mid-run costs you minutes, not a restart from zero.
#
# **Quick start:** run every cell top to bottom. Part 2 auto-detects Colab vs. Kaggle and
# mounts/finds your data. Part 12 prints exactly how long your run actually took, stage by
# stage.

# %% [markdown]
# # Part 2 — Setup
#
# Installs (Kaggle usually has most of these; the cell is safe to run either way), imports,
# and the `CONFIG` dict that every flag from §1.7 lives in. **Change values here, then
# Run All** — nothing below this cell should need editing to run an experiment.

# %%
# Kaggle already ships torch, sklearn, pandas, numpy, matplotlib.
# imbalanced-learn is usually NOT preinstalled -> install if missing.
# Which notebook is this? build_notebook.py sets this line: 'subset' (your analysis runs on
# the sample) or 'full' (the whole ~99,886-patient cohort, every analysis switched on).
RUN_PROFILE = "subset"
import importlib, subprocess, sys, os
# v4: a headless run (run_pipeline.py, e.g. on Isambard-AI) can pick the profile without
# editing this file: INSPIRE_RUN_PROFILE=isambard python run_pipeline.py ...
RUN_PROFILE = os.environ.get("INSPIRE_RUN_PROFILE", RUN_PROFILE)

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("imbalanced-learn", "imblearn")
_ensure("pyarrow")   # needed for Part 3's Parquet caching of parsed subject data
_ensure("umap-learn", "umap")   # v4, optional: UMAP embedding maps in Part 13 (skipped, and said so, if this fails)
print("Setup check complete.")

# %%
import os, glob, json, math, random, warnings, time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (must precede IterativeImputer import)
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              roc_curve, precision_recall_curve, confusion_matrix)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("(Even the 'large' encoder preset is under ~1M parameters -- GPU memory is very unlikely "
          "to be the constraint. System RAM, from the raw data tables in Part 3, is the one "
          "actually worth watching -- see that Part's memory printout.)")

# %%
# Mount Google Drive on Colab only. (The original cell mounted unconditionally, which
# crashes immediately on Kaggle or a local run; the cell below needs Drive on Colab.)
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")

# %%
import os, sys, zipfile, time

IN_COLAB = "google.colab" in sys.modules
COLAB_SUBJECTS_DIR = None

if IN_COLAB:
    # v2: first of these that exists on your Drive is used -- put your zip in MyDrive root,
    # or set DRIVE_ZIP_PATH yourself.
    _ZIP_CANDIDATES = (["/content/drive/MyDrive/subjects.zip"] if RUN_PROFILE == "full"
                       else ["/content/drive/MyDrive/subjects_sample.zip", "/content/drive/MyDrive/subjects.zip"])
    DRIVE_ZIP_PATH = next((p for p in _ZIP_CANDIDATES if os.path.isfile(p)), _ZIP_CANDIDATES[0])
    print(f"Using data zip: {DRIVE_ZIP_PATH}")
    COLAB_SUBJECTS_DIR = "/content/inspire_subjects_data_" + os.path.splitext(os.path.basename(DRIVE_ZIP_PATH))[0]   # local disk -- much faster than Drive

    assert os.path.isfile(DRIVE_ZIP_PATH), f"Can't find {DRIVE_ZIP_PATH} -- check the exact filename/path on your Drive."

    if not os.path.isdir(COLAB_SUBJECTS_DIR):
        print(f"Extracting {DRIVE_ZIP_PATH} -> {COLAB_SUBJECTS_DIR} ...")
        _t0 = time.time()
        with zipfile.ZipFile(DRIVE_ZIP_PATH, "r") as z:
            z.extractall(COLAB_SUBJECTS_DIR)
        print(f"Done in {time.time() - _t0:.1f}s.")
    else:
        print(f"{COLAB_SUBJECTS_DIR} already exists -- skipping. Delete it first if you want to re-extract.")

    # Handle a zip that has one extra nested folder inside it
    _contents = os.listdir(COLAB_SUBJECTS_DIR)
    if len(_contents) == 1 and os.path.isdir(os.path.join(COLAB_SUBJECTS_DIR, _contents[0])):
        COLAB_SUBJECTS_DIR = os.path.join(COLAB_SUBJECTS_DIR, _contents[0])

    n_files = sum(len(files) for _, _, files in os.walk(COLAB_SUBJECTS_DIR))
    print(f"\n{n_files} files found under {COLAB_SUBJECTS_DIR}")
    print("Top-level contents:", os.listdir(COLAB_SUBJECTS_DIR))

    assert "died" in os.listdir(COLAB_SUBJECTS_DIR) and "survived" in os.listdir(COLAB_SUBJECTS_DIR), (
        f"{COLAB_SUBJECTS_DIR} doesn't contain 'died' and 'survived' subfolders -- check the zip's structure."
    )
    print("\nLooks good.")
else:
    print("Not running on Colab -- skipping the Drive-zip extraction step. On Kaggle, "
          "attach your dataset and Part 2's auto-detection will find it under /kaggle/input.")

# %%
# ---------------------------------------------------------------------------
# CONFIG — every flag documented in Part 1 §1.7. Edit here, not below.
# ---------------------------------------------------------------------------
CONFIG = {
    # --- data location (auto-detected below; override if detection is wrong) ---
    "SUBJECTS_DIR": COLAB_SUBJECTS_DIR,   # folder containing died/ and survived/ subfolders of per-patient JSON
    "CODES_DIR": None,         # optional: folder with icd10.json.gz / WHO_ATC-DDD_*.csv (finer descriptions only)

    # --- full-scale readiness ---
    # Caps patients PER FOLDER (died/survived) -- NOT a percentage sample, and it never
    # removes real death cases: the full cohort has only ~469 real deaths, always fewer
    # than this cap, so every real death is always kept; only the (much larger) survived
    # pool gets capped -- this cap IS the downsampling mechanism, and it's deliberately
    # class-aware (never touches the rare class) rather than a blind random sample of
    # everyone. Default here is 10,000 -- a ~10,469-patient run, comfortably inside a
    # 20-minute budget given the batch-size/early-stopping settings below. Set to None
    # for the true full run once you have the time budget for it (expect data loading
    # alone to take several minutes at that scale, per Part 3's own measured timing).
    "MAX_SUBJECTS_PER_CLASS": 10000,

    # --- memory safety for Part 3's parse (new) -- patients are parsed and flushed to
    # disk in chunks of this size, rather than holding every row of every table for the
    # WHOLE cohort as Python dict objects in memory at once (which is roughly 6x more
    # memory-hungry per row than the eventual optimized DataFrame -- easily enough to
    # exceed a Kaggle/Colab session's RAM at the full ~99,886-patient scale). Lower this
    # if you still see an out-of-memory restart during Part 3; raise it for speed if you
    # have RAM to spare.
    "PARSE_CHUNK_SIZE": 2000,   # v2: 5000 held a whole 4k-patient sample in RAM at once

    # --- memory/compute lever, off by default -- see Part 3's "note on memory strategy" ---
    # If set, caps the number of REAL SURVIVED patients in the TRAINING split only (never
    # died patients, never val/test) to this count, via a department-stratified sample so
    # the department mix of the training pool is preserved. Try TIME_WINDOW='pre_op' and a
    # MAX_SUBJECTS_PER_CLASS smoke test first -- this is a last resort, not a first move.
    "DOWNSAMPLE_TRAIN_SURVIVED_TO": None,

    # --- §1.6.3 time window ---
    "TIME_WINDOW": "pre_op",              # 'pre_op' | 'peri_op'
    "PRE_OP_DAYS": 5,                     # matches the source repo's existing window

    # --- §1.3 missing data ---
    "IMPUTATION_STRATEGY": "decision_tree",  # 'decision_tree' | 'median' | 'interpolate' | 'knn'
    "KNN_NEIGHBORS": 5,

    # --- §1.4 / Part C class imbalance (updated to the agreed combined pipeline) ---
    "SAMPLING_STRATEGY": "grouped_smotenc_tomek",
    # 'grouped_smotenc_tomek' (new default, Part C §6) | 'class_weight' | 'smote' | 'adasyn'
    # | 'random_oversample' | 'random_undersample' | 'none'
    "SMOTE_TARGET_RATIO": 0.10,     # positive:negative target, e.g. 0.10 = 1:10 (Part C's conservative pick)
    "SMOTE_MIN_STRATUM_MINORITY": 3,   # skip a department x ASA stratum with fewer real positives than this
    # v3: cap synthetic deaths per stratum at this multiple of its REAL deaths. Uncapped, a
    # 1:10 target made ~85 synthetic copies from 3 real deaths in low-risk strata at full scale.
    "SMOTE_MAX_AMPLIFICATION": 5,      # None = uncapped (v2 behaviour)
    # v3: balanced batches -- at ~1:200, ~29% of 256-patient batches contain no death at all.
    "BALANCED_SAMPLER": True,
    "SAMPLER_TARGET_POSITIVE_RATE": 0.10,
    # Part C §6's clinical-neighbourhood grouping; v4 adds emop, so a synthetic death is only
    # ever blended from patients with the same department, ASA AND emergency/scheduled status.
    "SMOTE_STRATA_COLS": ["department", "asa", "emop"],
    "USE_SEQUENCE_AUGMENTATION": True,      # Part C §7: jitter + time-mask for REAL minority training patients
    "SEQUENCE_AUGMENTATION_COPIES": 2,      # extra augmented copies per real positive training patient
    "JITTER_SIGMA": 0.05,                   # on the standardized (z-scored) scale, per Part C §7
    # v2: never jitter discrete/ordinal/binary items -- GCS 4 -> 3.98 or a ventilator flag of
    # 0.97 is not a real reading (caught by the NEWS2 before/after check, Part 9.1b).
    "JITTER_EXCLUDE_ITEMS": ["gcs_e", "gcs_m", "gcs_v", "crrt", "iabp", "vent", "ecmo"],
    "USE_FOCAL_LOSS": False,
    "FOCAL_GAMMA": 2.0,
    # v2: raise loudly if SMOTENC ERRORS on a stratum (vs. a legitimate "too few positives"
    # skip). The original notebook caught both with one bare `except ValueError`, which is
    # how SMOTENC silently produced zero synthetic patients on every run.
    "STRICT_SAMPLING": True,
    # v2: what time series a SMOTENC synthetic patient gets. SMOTENC only synthesises the
    # STATIC vector. 'empty' (the original behaviour) gives every synthetic -- all labelled
    # "died" -- a completely unobserved time series, which teaches the model the shortcut
    # "no data at all => died". 'nearest_real_donor' borrows the jittered series of the
    # nearest real died patient in the same training set instead.
    "SYNTHETIC_TS_MODE": "nearest_real_donor",   # 'nearest_real_donor' | 'empty'

    # --- v2: NEWS2 (Royal College of Physicians 2017), computed from pre-op ward vitals ---
    # Added to the static vector ONLY if the scoring self-test passes (Part 6.16).
    "INCLUDE_NEWS2": True,
    "NEWS2_BUCKET_MINUTES": 60,          # vitals charted within the same hour = one observation set
    "NEWS2_CARRY_FORWARD_MINUTES": 240,  # bounded LOCF for a parameter missing from a set
    "NEWS2_MIN_CORE_VITALS": 3,          # of RR/SpO2/SBP/HR/temp -- fewer => no score (not a falsely low one)
    "NEWS2_ASSUME_AIR_IF_MISSING": True, # no FiO2/vent charted => room air (flagged approximation)
    "NEWS2_ASSUME_ALERT_IF_MISSING": True,

    # --- §1.5 architecture ---
    # (a flat-vs-system-split ablation flag was considered here and dropped rather than
    # shipped unwired -- see §1.1's note. The organ-system split is not optional in this
    # notebook's architecture.)
    "FUSION_STRATEGY": "nam",              # 'nam' | 'concat'  ('gated' is superseded by the
                                           # arbitration layer below, which keeps NAM additivity)
    # v2: encoder capacity. The v1 model was 66k parameters in TOTAL, and only ~17k of that
    # was the six system encoders (~3k each) -- most of it was the reconstruction heads
    # and the static MLP. 'medium' is ~9x the encoder capacity; 'small' reproduces v1.
    "ENCODER_SIZE": "medium",              # 'small' (v1) | 'medium' | 'large' -- sets the 4 keys below
    "EMBED_DIM": None, "TRANSFORMER_HEADS": None, "TRANSFORMER_LAYERS": None, "FF_MULTIPLIER": None,
    "DROPOUT": 0.1,

    # --- v3: the eight organ systems (renal, cardiovascular, respiratory, metabolic/hepatic,
    # haematology, neurological, GI, MSK) are all built the same way and run through ONE
    # shared encoder body (per-system input adapters + a system tag). False = a separate
    # body per system (the v2 behaviour, ~8x more encoder parameters) -- for ablation.
    "SHARE_ENCODER": True,
    "SYSTEM_DROPOUT": 0.1,            # fine-tuning: hide a whole system for 10% of patients per step

    # --- v3: whole-patient layer -- reads ALL features, gently re-scales each system's
    # summary (scale-only bottleneck, 0.5x-1.5x) and adds its own whole-patient term.
    "USE_WHOLE_PATIENT_LAYER": True,

    # --- inter-system coupling layer -- learned, nothing hand-defined ---
    "COUPLING_MODE": "learned",       # 'learned' | 'none' (ablation baseline)
    "COUPLING_HEADS": 2,
    "COUPLING_PRIOR_LINKS": [],       # left empty by design: every link must be learned from data
    "COUPLING_PRIOR_STRENGTH": 1.0,
    "COUPLING_GATE_INIT": 0.1,
    # v3: masked-system pre-training -- hide one system per patient and predict it from the
    # other seven through the coupling layer. Label-free, so it learns from EVERY patient.
    "MASKED_SYSTEM_WEIGHT": 1.0,      # 0 = v2 behaviour (coupling learns from deaths only)

    # --- arbitration layer -- how much "say" each organ system's answer gets per patient ---
    "USE_ARBITRATION_LAYER": True,
    "ARBITRATION_REG": 1e-3,

    # --- v3: learn, don't memorise ---
    "WEIGHT_DECAY": 1e-2,             # AdamW
    "FINETUNE_ENCODER_LR_MULT": 0.1,  # fine-tuning moves the pre-trained encoder 10x more gently
    "N_ENSEMBLE": 3,                  # Part 11.4b: models trained from different seeds, averaged
    "RUN_LEARNING_CURVE": True,       # Part 11.4c: retrain on 25% / 50% / 100% of training patients
    "RUN_GBM_BASELINE": True,         # Part 11.1d: gradient boosting next to logistic regression
    "N_BOOTSTRAP": 1000,              # confidence intervals for AUROC / AUPRC
    "TRAIN_METRIC_SUBSAMPLE": 5000,   # real training patients scored each epoch (train vs val gap)

    # --- §1.2 ICD-10 ---
    "USE_ICD10_EMBEDDING": False,
    "INCLUDE_HFRS": True,
    "HFRS_LOOKBACK_YEARS": 2,              # published Gilbert et al. window, age 75+
    "INCLUDE_STATIC_ASA": True,

    # --- training ---
    "BATCH_SIZE": 256,        # raised again from 128 -- at 10,469 patients, 128 still means
                              # ~4,400 training iterations total, risking most of a 20-minute
                              # budget on training alone. 256 roughly halves that with no
                              # accuracy cost for a model this small (see the run-specs doc for
                              # the measured basis). Lower toward 8-16 only for the tiny
                              # 30-patient dev subset.
    "EPOCHS_PRETRAIN": 30,     # upper CAP -- early stopping (below) will normally cut this short
    "EPOCHS_FINETUNE": 60,     # upper CAP -- early stopping (below) will normally cut this short
    "EARLY_STOPPING_PATIENCE": 8,   # stop a phase if its tracked metric hasn't improved for this
                                     # many epochs -- protects both the time budget (don't burn the
                                     # full epoch cap when it's not helping) and quality (still lets
                                     # training run as long as it's genuinely improving)
    "LR": 1e-3,
    # v2: select the best fine-tuning epoch on validation AUPRC (the primary metric at
    # ~0.5-4% prevalence) rather than AUROC.
    "MODEL_SELECTION_METRIC": "auprc",   # 'auprc' | 'auroc'
    "VAL_FRACTION": 0.2,
    "TEST_FRACTION": 0.2,
    "TARGET_SEQ_LEN": 24,      # number of time points each system's series is resampled/padded to

    # --- checkpointing (matters most on Colab -- free-tier sessions can disconnect for
    # reasons unrelated to memory: idle timeout, daily usage caps) ---
    "CHECKPOINT_DIR": None,          # auto-set below: Drive if on Colab+mounted, else local
    "CHECKPOINT_EVERY_N_EPOCHS": 5,

    # --- data-parsing cache (Part 3) -- avoids re-parsing tens of thousands of JSON
    # files on every session. First run parses once and saves Parquet here; every
    # later run with the same SUBJECTS_DIR/MAX_SUBJECTS_PER_CLASS/TIME_WINDOW loads
    # from cache instead, in seconds rather than minutes. ---
    "CACHE_DIR": None,               # auto-set below: Drive if on Colab+mounted, else local
    "USE_CACHE": True,

    # --- run-time budget controls (new) -- specifically for hitting a reliable 20-30
    # minute total runtime. Both of these sections are genuinely useful analysis, not
    # dead weight -- but they're also the two most expensive OPTIONAL steps in the
    # notebook (the regression/MICE imputation benchmark in particular), so they're
    # switchable independently of the core pipeline. See the "Run Specifications" doc
    # for measured per-section timing this decision is based on.
    "SKIP_IMPUTATION_ACCURACY_BENCHMARKS": True,   # Part 7.6/7.7 -- skips the held-out masking benchmarks (rule-based AND regression/MICE)
    "SKIP_FUSION_ABLATION": True,                  # Part 11.4 -- skips retraining a second (concat-fusion) model just for comparison
    # v2 evaluation additions (cheap -- no retraining of the DNN)
    "RECALIBRATE_ON_VAL": True,     # Platt-scale on the validation set (sampling + pos_weight inflate raw probabilities)
    "RUN_SIMPLE_BASELINE": True,    # logistic regression on the same features, same split

    # ======================= v4 additions (see CHANGES.md "v4") ======================= #
    # --- 1. saving: model, figures, per-epoch snapshots ---
    "OUTPUT_DIR": None,               # auto-set below: Drive on Colab, ./inspire_outputs elsewhere
    "SAVE_FIGURES": True,             # every plt.show() also writes a PNG to OUTPUT_DIR/figures
    "SAVE_FINAL_MODEL": True,         # final model + everything needed to reload it -> OUTPUT_DIR/models
    "SNAPSHOT_N_PATIENTS": 3000,      # validation patients whose embeddings are snapshotted (every val death always kept)
    "SNAPSHOT_FINETUNE_EPOCHS": [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80],   # + always: before training, after pre-training, last, best

    # --- 2. pictures of the embeddings ---
    "EMBEDDING_METHODS": ["pca", "tsne", "umap"],   # umap is skipped (and said so) if umap-learn is missing
    "TSNE_MAX_POINTS": 3000,          # t-SNE slows down quadratically; deaths are always kept
    "KNN_ENRICHMENT_K": 15,           # "what fraction of a patient's 15 nearest neighbours died?"

    # --- 4. SHAP-style attribution inside each organ system (expected gradients) ---
    "RUN_SHAP": True,
    "SHAP_N_PATIENTS": 600,           # test patients explained (every test death always kept)
    "SHAP_N_BACKGROUND": 200,         # "typical patients" the explanation is measured against
    "SHAP_N_SAMPLES": 32,             # Monte-Carlo samples per patient (more = smoother, slower)

    # --- 5. feature upgrades ---
    "USE_PROCEDURE_CODES": True,      # ICD-10-PCS body-system of the operation -> each organ system (GI and MSK get real content)
    "USE_ATC_SYSTEM_GROUPS": True,    # medication families (ATC) -> each organ system (A=gut, M=bones/muscles, ...)
    "USE_ANTYPE": True,               # anaesthesia type (one-hot), part of the surgical context
    "USE_SURGICAL_CONTEXT_TERM": True,      # emergency/scheduled, ASA, department, anaesthesia -> their OWN risk term
    "SURGICAL_CONTEXT_EXCLUSIVE": True,     # those columns are read ONLY by that term (clean "how much did surgery add")
    "RUN_SIZE_SWEEP": False,          # Part 13.9: retrain at several embedding sizes (the "score vs hidden size" figure)
    "SIZE_SWEEP": ["small", "medium", "large", "xlarge"],
    "SIZE_SWEEP_SEEDS": 2,

    # --- compute (big-data / Isambard-AI) ---
    "NUM_WORKERS": 0,                 # DataLoader worker processes (0 = main process; 8+ on a big node)
    "MIXED_PRECISION": False,         # bf16 autocast during training on GPUs that support it (H100/GH200/A100)
}

# Run profile presets (see RUN_PROFILE at the top). 'full' switches every analysis on and
# ignores run-time budgets -- intended for a machine with plenty of RAM and time.
PROFILE_PRESETS = {
    "subset": {},
    "full": {
        "MAX_SUBJECTS_PER_CLASS": None,          # the whole cohort
        "PARSE_CHUNK_SIZE": 5000,
        "SKIP_IMPUTATION_ACCURACY_BENCHMARKS": False,
        "SKIP_FUSION_ABLATION": False,
        "EPOCHS_PRETRAIN": 40, "EPOCHS_FINETUNE": 80, "EARLY_STOPPING_PATIENCE": 10,
        "N_ENSEMBLE": 5,
        "ENCODER_SIZE": "large",                 # v4: 64-number embeddings for the full cohort
        "TSNE_MAX_POINTS": 5000, "SNAPSHOT_N_PATIENTS": 5000, "SHAP_N_PATIENTS": 1000,
    },
    # v4: the whole cohort on a big GPU node (Isambard-AI: 4x NVIDIA GH200 per node, aarch64).
    # The model is small (well under 10M parameters even at 'xxlarge'), so ONE GH200 per run is
    # plenty; the node's value is running the full cohort, the full ensemble and the
    # embedding-size sweep in hours instead of days.
    "isambard": {
        "MAX_SUBJECTS_PER_CLASS": None,
        "PARSE_CHUNK_SIZE": 10000,
        "SKIP_IMPUTATION_ACCURACY_BENCHMARKS": False,
        "SKIP_FUSION_ABLATION": False,
        "EPOCHS_PRETRAIN": 60, "EPOCHS_FINETUNE": 100, "EARLY_STOPPING_PATIENCE": 12,
        "N_ENSEMBLE": 5, "N_BOOTSTRAP": 2000,
        "ENCODER_SIZE": "xlarge",                # 128-number embeddings, 8 heads, 3 layers
        "BATCH_SIZE": 1024, "NUM_WORKERS": 8, "MIXED_PRECISION": True,
        "TSNE_MAX_POINTS": 10000, "SNAPSHOT_N_PATIENTS": 10000, "SHAP_N_PATIENTS": 2000, "SHAP_N_BACKGROUND": 500,
        "SNAPSHOT_FINETUNE_EPOCHS": [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100],
        "RUN_SIZE_SWEEP": True, "SIZE_SWEEP": ["small", "medium", "large", "xlarge", "xxlarge"], "SIZE_SWEEP_SEEDS": 3,
    },
}
CONFIG.update(PROFILE_PRESETS[RUN_PROFILE])

# Optional overrides without editing this cell -- used by src-dnn/run_pipeline.py --config,
# and handy on Kaggle for quick variants: set INSPIRE_CONFIG_OVERRIDES='{"ENCODER_SIZE": "small"}'.
_overrides = json.loads(os.environ.get("INSPIRE_CONFIG_OVERRIDES", "{}") or "{}")
_unknown = sorted(set(_overrides) - set(CONFIG))
assert not _unknown, f"Unknown CONFIG keys in INSPIRE_CONFIG_OVERRIDES: {_unknown}"
CONFIG.update(_overrides)
if _overrides:
    print(f"CONFIG overrides applied: {_overrides}")

# Resolve the encoder-size preset (explicit values win over the preset).
_ENCODER_PRESETS = {"small": (16, 2, 1, 2), "medium": (32, 4, 2, 4), "large": (64, 4, 2, 4),
                    "xlarge": (128, 8, 3, 4), "xxlarge": (256, 8, 4, 4)}   # (embed, heads, layers, ff) -- same as layers.py
for _key, _val in zip(["EMBED_DIM", "TRANSFORMER_HEADS", "TRANSFORMER_LAYERS", "FF_MULTIPLIER"],
                      _ENCODER_PRESETS[CONFIG["ENCODER_SIZE"]]):
    if CONFIG[_key] is None:
        CONFIG[_key] = _val
assert CONFIG["COUPLING_MODE"] in ("learned", "none")
assert CONFIG["FUSION_STRATEGY"] in ("nam", "concat")
print(f"Architecture: ENCODER_SIZE={CONFIG['ENCODER_SIZE']!r} (embed={CONFIG['EMBED_DIM']}, "
      f"heads={CONFIG['TRANSFORMER_HEADS']}, layers={CONFIG['TRANSFORMER_LAYERS']}), "
      f"COUPLING_MODE={CONFIG['COUPLING_MODE']!r}, ARBITRATION={CONFIG['USE_ARBITRATION_LAYER']}, "
      f"WHOLE_PATIENT={CONFIG['USE_WHOLE_PATIENT_LAYER']}, SHARED_ENCODER={CONFIG['SHARE_ENCODER']}, "
      f"NEWS2={CONFIG['INCLUDE_NEWS2']}, profile={RUN_PROFILE!r}, N_ENSEMBLE={CONFIG['N_ENSEMBLE']}")

# ---------------------------------------------------------------------------
# Environment detection: Kaggle vs. Colab vs. local. Matters for both data-path
# auto-detection and where checkpoints should live (Colab's local disk is ephemeral --
# a checkpoint saved there is lost on disconnect just like everything else).
# ---------------------------------------------------------------------------
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.isdir("/kaggle/input")

if IN_COLAB:
    try:
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            print("Colab detected -- mounting Google Drive (needed so your data/checkpoints "
                  "survive a session disconnect; a browser auth prompt may appear)...")
            drive.mount("/content/drive")
        else:
            print("Colab detected -- Google Drive already mounted.")
    except Exception as e:
        print(f"Colab detected but Drive mount failed or was skipped ({e}). "
              f"Data auto-detection and checkpointing will fall back to local (ephemeral) storage.")

# NOTE: checks IN_COLAB first -- the "google.colab" import check is a definitive signal,
# whereas an unrelated leftover "/kaggle/input" directory (e.g. from a prior session state)
# is not, and previously caused this to misreport "Kaggle" on an actual Colab run.
print(f"Environment: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'local/other'}")

# ---------------------------------------------------------------------------
# Auto-detect data location: Kaggle input dirs, Colab Drive paths, then local upload paths.
# ---------------------------------------------------------------------------
def _find_subjects_dir():
    candidates = []
    # Fast, exact-path check first -- your data is at this known Drive location, so this
    # skips the (slower, walk-based) broader search below entirely when it matches.
    _known_exact_paths = [
        "/content/drive/MyDrive/subjects",
    ]
    for p in _known_exact_paths:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "died")) and os.path.isdir(os.path.join(p, "survived")):
            return p
    if IN_KAGGLE:
        # Depth-capped AND stops at the first match -- without both of these, this can hang
        # for a very long time on a dataset with many files: os.walk with no depth limit
        # will also descend INTO a matched died/survived folder to enumerate its (possibly
        # thousands of) files, and Kaggle's dataset mount can have real per-file latency on
        # first access. Neither is needed just to confirm the folder exists.
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root[len("/kaggle/input"):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            if "died" in dirs and "survived" in dirs:
                candidates.append(root)
                dirs[:] = []   # don't descend into died/survived's own contents
                break          # found it -- stop searching entirely
    if IN_COLAB:
        # Common places people drop an uploaded/extracted dataset on Drive -- searched
        # shallowly (max depth ~4) since walking all of MyDrive can be slow if it's large.
        drive_roots = ["/content/drive/MyDrive", "/content/drive/MyDrive/INSPIRE",
                       "/content/drive/MyDrive/inspire", "/content/drive/MyDrive/data",
                       "/content/drive/MyDrive/subjects", "/content"]
        for base in drive_roots:
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                depth = root[len(base):].count(os.sep)
                if depth > 4:
                    dirs[:] = []   # don't descend further from here
                    continue
                if "died" in dirs and "survived" in dirs:
                    candidates.append(root)
    for p in ["/mnt/user-data/uploads/inspire_subjects_small",
              "/home/claude/work/inspire_subjects_small/inspire_subjects_small",
              "./inspire_subjects_small", "./subjects"]:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "died")):
            candidates.append(p)
    return candidates[0] if candidates else None

def _find_codes_dir():
    if IN_KAGGLE:
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root[len("/kaggle/input"):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            if "icd10.json.gz" in files or any(f.startswith("WHO_ATC") for f in files):
                return root
    if IN_COLAB and os.path.isdir("/content/drive/MyDrive"):
        for root, dirs, files in os.walk("/content/drive/MyDrive"):
            depth = root[len("/content/drive/MyDrive"):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            if "icd10.json.gz" in files or any(f.startswith("WHO_ATC") for f in files):
                return root
    for p in ["/home/claude/work/inspire-analysis/inspire-analysis-thrisha-main/codes", "./codes"]:
        if os.path.isdir(p):
            return p
    return None

if CONFIG["SUBJECTS_DIR"] is None:
    CONFIG["SUBJECTS_DIR"] = _find_subjects_dir()
if CONFIG["CODES_DIR"] is None:
    CONFIG["CODES_DIR"] = _find_codes_dir()
if CONFIG["CHECKPOINT_DIR"] is None:
    if IN_COLAB and os.path.isdir("/content/drive/MyDrive"):
        CONFIG["CHECKPOINT_DIR"] = "/content/drive/MyDrive/inspire_checkpoints"
    else:
        CONFIG["CHECKPOINT_DIR"] = "./inspire_checkpoints"   # Kaggle/local: ephemeral, but Kaggle sessions are less disconnect-prone
os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
if CONFIG["CACHE_DIR"] is None:
    if IN_COLAB and os.path.isdir("/content/drive/MyDrive"):
        CONFIG["CACHE_DIR"] = "/content/drive/MyDrive/inspire_parsed_cache"
    else:
        CONFIG["CACHE_DIR"] = "./inspire_parsed_cache"
os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
if CONFIG["OUTPUT_DIR"] is None:
    if IN_COLAB and os.path.isdir("/content/drive/MyDrive"):
        CONFIG["OUTPUT_DIR"] = "/content/drive/MyDrive/inspire_outputs"
    else:
        CONFIG["OUTPUT_DIR"] = "./inspire_outputs"   # on Kaggle this is /kaggle/working -> downloadable
FIGURE_DIR = os.environ.get("INSPIRE_FIG_DIR") or os.path.join(CONFIG["OUTPUT_DIR"], "figures")
MODEL_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "models")
SNAPSHOT_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "snapshots")
TABLE_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "tables")
for _d in (FIGURE_DIR, MODEL_DIR, SNAPSHOT_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)

print("SUBJECTS_DIR   ->", CONFIG["SUBJECTS_DIR"])
print("CODES_DIR      ->", CONFIG["CODES_DIR"], "(optional — used only for richer ICD-10/ATC descriptions)")
print("CHECKPOINT_DIR ->", CONFIG["CHECKPOINT_DIR"],
      "(on Drive -- survives a Colab disconnect)" if CONFIG["CHECKPOINT_DIR"].startswith("/content/drive") else
      "(local/ephemeral -- will NOT survive a Colab disconnect; mount Drive if you're on Colab)")
print("OUTPUT_DIR     ->", CONFIG["OUTPUT_DIR"], "(figures/, models/, snapshots/, tables/)")
print("CACHE_DIR      ->", CONFIG["CACHE_DIR"],
      "(on Drive -- parsed data survives a Colab disconnect, only parsed once ever)" if CONFIG["CACHE_DIR"].startswith("/content/drive") else
      "(local/ephemeral -- re-parses every fresh session)")
assert CONFIG["SUBJECTS_DIR"] is not None, (
    "Could not auto-find a subjects folder (must contain died/ and survived/ subfolders of JSON). "
    "Set CONFIG['SUBJECTS_DIR'] manually -- e.g. on Colab, '/content/drive/MyDrive/<wherever you put it>'."
)

# %% [markdown]
# ### Run-time tracking
#
# Every major Part below records how long it actually took, on *your* hardware and *your*
# config -- not an estimate. Part 12 prints the full breakdown at the end, so every run
# tells you directly whether it hit the 20-30 minute target, and exactly which Part to
# adjust first if it didn't.

# %%
RUN_TIMER = {"checkpoints": [("notebook start", time.time())]}

def record_checkpoint(label):
    RUN_TIMER["checkpoints"].append((label, time.time()))

def print_timing_summary():
    cps = RUN_TIMER["checkpoints"]
    print(f"{'Part':35s} {'elapsed':>10s}")
    print("-" * 47)
    for i in range(1, len(cps)):
        label, t = cps[i]
        prev_t = cps[i - 1][1]
        print(f"{label:35s} {t - prev_t:8.1f}s")
    total = cps[-1][1] - cps[0][1]
    print("-" * 47)
    print(f"{'TOTAL':35s} {total:8.1f}s  ({total/60:.1f} min)")

print("Run-time tracking initialised.")

# %% [markdown]
# ### Saving every figure (v4)
#
# Every `plt.show()` below also writes the figure as a PNG into `OUTPUT_DIR/figures/`,
# numbered in order and named after its title — so a full run leaves a folder of every
# plot, ready for the paper or a supervisor meeting. Headless runs (`run_pipeline.py`)
# save and close instead of displaying.

# %%
import re as _re
HEADLESS = bool(os.environ.get("INSPIRE_HEADLESS"))
SAVED_FIGURES = []

def _figure_title(fig):
    if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
        return fig._suptitle.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return "figure"

if (CONFIG["SAVE_FIGURES"] or HEADLESS) and not getattr(plt.show, "_inspire_saver", False):
    _ORIGINAL_SHOW = plt.show

    def _show_and_save(*args, **kwargs):
        for num in (plt.get_fignums() if CONFIG["SAVE_FIGURES"] else []):
            fig = plt.figure(num)
            slug = _re.sub(r"[^a-z0-9]+", "_", _figure_title(fig).lower()).strip("_")[:60] or "figure"
            path = os.path.join(FIGURE_DIR, f"fig_{len(SAVED_FIGURES) + 1:03d}_{slug}.png")
            try:
                fig.savefig(path, dpi=150, bbox_inches="tight")
                SAVED_FIGURES.append(path)
            except Exception as e:   # a figure that cannot be saved must never stop the run
                print(f"(could not save figure {slug}: {e})")
        if HEADLESS:
            plt.close("all")
        else:
            _ORIGINAL_SHOW(*args, **kwargs)
    _show_and_save._inspire_saver = True
    plt.show = _show_and_save
    print(f"Figures will be saved to {FIGURE_DIR}")

# Mixed precision (v4): bf16 autocast for training on GPUs that support it. Off on CPU.
USE_AMP = bool(CONFIG["MIXED_PRECISION"] and DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
print(f"Mixed precision (bf16) during training: {USE_AMP}")

