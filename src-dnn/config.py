"""
Stage 0: setup, environment detection, and CONFIG.

Converted from INSPIRE_Multimodal_Mortality_Benchmark.ipynb (cells 2-6) into a script.
Content is preserved as closely as possible to the working notebook -- the one fix
applied here is removing a redundant, unguarded `drive.mount()` call (the notebook's
own environment-detection block later in this same file already does this correctly,
gated on IN_COLAB; the removed block would crash immediately on Kaggle or a local run).

Every later stage in this pipeline does `from config import *` -- same effect as
notebook cells sharing one namespace, just split into files you can read, run, and
diff individually.
"""

# --- from notebook cell 2 ---
# Kaggle already ships torch, sklearn, pandas, numpy, matplotlib.
# imbalanced-learn is usually NOT preinstalled -> install if missing.
import importlib, subprocess, sys

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("imbalanced-learn", "imblearn")
_ensure("pyarrow")   # needed for Part 3's Parquet caching of parsed subject data
print("Setup check complete.")

# --- from notebook cell 3 ---
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
    print("(This notebook's model is small -- tens of thousands of parameters, small batches -- "
          "GPU memory is very unlikely to be the constraint. System RAM, from the raw data "
          "tables in Part 3, is the one actually worth watching -- see that Part's memory printout.)")

# NOTE: cell 4's original unconditional 'from google.colab import drive; drive.mount(...)'
# was removed here -- it would raise ImportError immediately outside Colab (Kaggle/local).
# The environment-detection block below already does this correctly, guarded by IN_COLAB.

# --- from notebook cell 5 ---
import os, sys, zipfile, time

IN_COLAB = "google.colab" in sys.modules
COLAB_SUBJECTS_DIR = None

if IN_COLAB:
    DRIVE_ZIP_PATH = "/content/drive/MyDrive/subjects.zip"
    COLAB_SUBJECTS_DIR = "/content/inspire_subjects_data"   # local disk -- much faster than Drive

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

# --- from notebook cell 6 ---
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
    "PARSE_CHUNK_SIZE": 5000,

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
    "SMOTE_STRATA_COLS": ["department", "asa"],   # Part C §6's clinical-neighborhood grouping
    "USE_SEQUENCE_AUGMENTATION": True,      # Part C §7: jitter + time-mask for REAL minority training patients
    "SEQUENCE_AUGMENTATION_COPIES": 2,      # extra augmented copies per real positive training patient
    "JITTER_SIGMA": 0.05,                   # on the standardized (z-scored) scale, per Part C §7
    "USE_FOCAL_LOSS": False,
    "FOCAL_GAMMA": 2.0,

    # --- §1.5 architecture ---
    # (a flat-vs-system-split ablation flag was considered here and dropped rather than
    # shipped unwired -- see §1.1's note. The organ-system split is not optional in this
    # notebook's architecture.)
    "FUSION_STRATEGY": "nam",              # 'nam' | 'concat' | 'gated'
    "EMBED_DIM": 16,                       # per-system embedding size
    "TRANSFORMER_HEADS": 2,
    "TRANSFORMER_LAYERS": 1,

    # --- §1.6.2 cardio-renal coupling ---
    "SYMMETRIC_CARDIORENAL_COUPLING": False,

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
}

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

print("SUBJECTS_DIR   ->", CONFIG["SUBJECTS_DIR"])
print("CODES_DIR      ->", CONFIG["CODES_DIR"], "(optional — used only for richer ICD-10/ATC descriptions)")
print("CHECKPOINT_DIR ->", CONFIG["CHECKPOINT_DIR"],
      "(on Drive -- survives a Colab disconnect)" if CONFIG["CHECKPOINT_DIR"].startswith("/content/drive") else
      "(local/ephemeral -- will NOT survive a Colab disconnect; mount Drive if you're on Colab)")
print("CACHE_DIR      ->", CONFIG["CACHE_DIR"],
      "(on Drive -- parsed data survives a Colab disconnect, only parsed once ever)" if CONFIG["CACHE_DIR"].startswith("/content/drive") else
      "(local/ephemeral -- re-parses every fresh session)")
assert CONFIG["SUBJECTS_DIR"] is not None, (
    "Could not auto-find a subjects folder (must contain died/ and survived/ subfolders of JSON). "
    "Set CONFIG['SUBJECTS_DIR'] manually -- e.g. on Colab, '/content/drive/MyDrive/<wherever you put it>'."
)