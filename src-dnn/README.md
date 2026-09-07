# src-dnn — PACO-Net v2, as scripts and modules instead of notebooks

This folder restructures the PACO-Net v2 design (phase-aware timeline, learned
inter-organ coupling, diffusion-based augmentation — see
`../docs/current/PACO_Net_Latest_Work_and_Results.md`) into a proper package: staged,
resumable scripts instead of one long notebook, so a free-tier Colab/Kaggle disconnect
doesn't cost you re-running everything from scratch.

**This does not replace `../src/`** — the original `subject.py`, `read_subjects()`, and
the working baseline pipeline are reused, not duplicated. `src-dnn` wraps that existing,
tested code and adds the new pieces on top: phase tagging, the full 117-parameter organ
mapping, learned coupling, diffusion augmentation, and a resumable trainer.

## Structure

```
src-dnn/
├── configs/default.yaml       # every tunable value, in one place
├── data/
│   ├── processed/             # cached parquet — stage 1/2 output, checked in .gitignore
│   └── checkpoints/           # model checkpoints — same
├── src/inspire/
│   ├── data/                  # loader.py (wraps ../../../src/subject.py), preprocess.py
│   ├── features/              # organ_systems.py — the single source of truth mapping
│   ├── augmentation/          # smotenc.py (bug-fixed), diffusion.py, missingness_model.py
│   ├── models/                # encoders.py, coupling.py, fusion.py, head.py
│   ├── training/              # checkpoint.py, train.py (resumable Trainer)
│   └── eval/                  # metrics.py, validate_synthetic.py
├── scripts/                   # 01_preprocess -> 02_train_diffusion -> 03_train_model -> 04_evaluate
└── notebooks/                 # thin Colab/Kaggle launchers — logic lives in scripts/, not here
```

## Status — what's real vs. what's a wiring TODO

**Fully implemented and internally verified** (real smoke tests run, not just syntax checks):
- `features/organ_systems.py` — all 117 unique INSPIRE parameters mapped to organ
  system + phase + role (physiology vs. SOFA-style support). Includes a self-check
  (`python src/inspire/features/organ_systems.py`) that caught one real gap (`aft`,
  alfentanil) during development, since fixed — run it again if you edit the mapping.
- `data/loader.py` — phase-tagging logic, field names verified against the real
  `subject.py` record format (`item_name`/`chart_time`/`value` as used in
  `get_lab()`/`convert_vitals_to_dictionary()`).
- `augmentation/smotenc.py` — the actual SMOTENC bug fix from the 10,942-patient run
  (integer indices, not a boolean mask) is applied here. A SECOND, separate bug was
  found and fixed during this session's testing: `TomekLinks` cannot handle string
  categoricals directly (unlike SMOTENC) — categoricals are now ordinal-encoded only
  for the Tomek step and decoded back immediately after. Verified end-to-end on
  synthetic data.
- `augmentation/missingness_model.py` — verified end-to-end on synthetic data.
- `models/coupling.py`, `models/fusion.py`, `models/head.py`, `models/encoders.py` —
  each passes a real forward-pass test with correct tensor shapes.
- `training/checkpoint.py` — save/load verified end-to-end. One real bug found and
  fixed here too: PyTorch 2.6+ changed `torch.load`'s default to `weights_only=True`,
  which rejects the RNG-state dict this module saves — fixed with an explicit
  `weights_only=False` (safe for our own checkpoints, not arbitrary files).
- `eval/metrics.py` — verified against synthetic labeled data.

**Genuine open TODOs, marked inline with `# TODO`, not silently guessed at:**
- `models/encoders.py` per-system `feature_dim` — depends on your final per-system
  feature count once you decide the peri-op resampling bin size (config:
  `phases.peri_op_bin_minutes`).
- `training/train.py` `_forward_step()` and `scripts/03_train_model.py`'s
  `PACONetV2.forward()` — need your real batch/DataLoader structure to wire the four
  model pieces together end-to-end. The pieces themselves are ready; the glue isn't,
  because it depends on decisions (post-op window, bin size) flagged as open in the
  design doc.
- `scripts/02_train_diffusion.py` — department/ASA join onto the static minority table,
  the phase-completeness check before generating peri-op features, and target-ratio
  computation are flagged as TODOs — same open items as the design doc §5.
- `models/fusion.py`'s cardiac washout bias magnitude/sign — placeholder value, needs
  your clinical confirmation of which direction it should move risk.

None of these were skipped by oversight — each is genuinely gated on one of the four
open decisions in the design doc (post-op window length, peri-op bin size, fluid
placement, diffusion phase-completeness). Resolve those and the TODOs become
mechanical.

## Running it

See `notebooks/colab_entry.ipynb` / `notebooks/kaggle_entry.ipynb` for the free-tier-safe
launch sequence, or run the four scripts directly:

```bash
python scripts/01_preprocess.py --config configs/default.yaml
python scripts/02_train_diffusion.py --config configs/default.yaml
python scripts/03_train_model.py --config configs/default.yaml --resume
python scripts/04_evaluate.py --config configs/default.yaml --checkpoint data/checkpoints/latest.pt
```

Add `--max-subjects-per-class 500` to `01_preprocess.py` to smoke-test the full pipeline
on a small sample before committing a long run to the real ~99,886-patient cohort.
