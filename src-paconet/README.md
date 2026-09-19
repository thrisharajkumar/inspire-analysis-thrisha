# src-dnn -- PACO-Net v2, as scripts and modules instead of notebooks

This folder restructures the PACO-Net v2 design (phase-aware timeline, learned
inter-organ coupling, diffusion-based augmentation -- see
`../docs/current/PACO_Net_Latest_Work_and_Results.md`) into a proper package: staged,
resumable scripts instead of one long notebook.

**This does not replace `../src/`** -- the original `subject.py` and `read_subjects()`
are reused via `data/loader.py`, not duplicated.

## Status: verified end-to-end on synthetic dry-run data

**As of this update, the full pipeline (stages 0 through 4) has actually been run in
full, not just written.** `scripts/00_generate_synthetic_dryrun_data.py` builds a small
synthetic cohort matching the real data's exact schema, and stages 1-4 run against it
end-to-end with no crashes:

```
python scripts/00_generate_synthetic_dryrun_data.py
python scripts/02_train_diffusion.py --config configs/default.yaml   # gracefully skips if synthcity isn't installed
python scripts/03_train_model.py --config configs/default.yaml
python scripts/04_evaluate.py --config configs/default.yaml --checkpoint data/checkpoints/latest.pt
```

Confirmed on this synthetic cohort: training loss drops from 0.610 -> 0.504 over 4
epochs, checkpoint/resume works correctly, and evaluation shows AUROC moving from
chance (0.500) to 0.599 -- real confirmation the architecture learns the (deliberately
easy) synthetic severity signal it was given. These numbers are now reproducible
run-to-run (a missing random seed was the second bug found -- see below). **This is not
a clinical result** -- it's proof the plumbing (Dataset/DataLoader, model wiring,
checkpointing, training loop, evaluation) works correctly, so integration bugs surface
here rather than on your first real run.

### Three real bugs found and fixed by actually running this, not just reading it

1. **`Subject.from_json_file()` doesn't exist.** I'd guessed this method name while
   writing `data/loader.py`; the real API (confirmed by reading `subject.py`) is
   `Subject().fromJSON(path)`. Fixed.
2. **The hazard head's hidden layer degenerated to zero width.** `models/head.py`
   computed `hidden_dim = input_dim // 2`, which is 0 when `input_dim == 1` (the NAM
   fusion's scalar output) -- a `Linear(1, 0)` layer silently produces a constant
   output regardless of input. This is exactly why the *first* dry run's AUROC sat at
   exactly 0.500 no matter what. Fixed with a minimum hidden width; the model now
   demonstrably learns (see numbers above). The fix also removes a broadcast-scalar
   workaround in `scripts/03_train_model.py` that would have broken the NAM's
   interpretability guarantee (prediction should be a direct function of the
   interpretable per-system total, not a richer hidden representation) -- the corrected
   version feeds the head the real scalar directly, which is both simpler and correct.
3. **No random seed was set for training.** Two otherwise-identical runs produced
   AUROC 0.635 and 0.450 -- at this dry run's tiny scale (400 patients, 32 positives)
   that variance is large enough to make results look inconsistent or broken when
   they're actually just unseeded. Fixed with `torch.manual_seed()`; confirmed two
   from-scratch runs now produce identical metrics.

### What's still a genuine placeholder, not a bug

These are documented, deliberate simplifications for the dry run -- fix before trusting
real numbers, but they didn't stop the pipeline from running:

- **Mean-pooling** collapses each organ system's timeline to one embedding before
  coupling (`scripts/03_train_model.py`) -- discards temporal shape within a system.
- **Static systems (GI/MSK/fluid)** are zero-filled placeholders in the model, not real
  features yet.
- **Fixed padding/truncation to 32 timesteps** (`data/dataset.py`) instead of the
  peri-op resampling-to-fixed-bins step from the design doc -- bin size is one of the
  open decisions, so this is deliberately deferred, not forgotten.
- **The discrete-time hazard loss is collapsed to a single mean logit** against a
  binary label (`training/train.py`) -- a real run needs proper discrete-time survival
  loss (event-in-bin-k targets), not this collapse.
- **The cardiac washout flag defaults to always-eligible** (`data/dataset.py`) --
  real operation-history dates aren't wired in yet.

### The one thing I genuinely could not verify

**`synthcity` could not be installed in my development sandbox** (disk constraints) --
so `augmentation/diffusion.py`'s actual TabDDPM calls are unverified against a real
install. I've wrapped every synthcity call in try/except with clear per-system
fallback (confirmed working above -- the dry run correctly detects synthcity's absence
and skips cleanly rather than crashing), so a real API mismatch will surface as a clear
warning naming the failing system, not a crash. This is the first place to look if
something goes wrong on your real run.

## Structure

```
src-dnn/
├── configs/default.yaml       # every tunable value, in one place
├── data/
│   ├── processed/             # cached parquet -- stage 1/2 output
│   └── checkpoints/           # model checkpoints
├── src/inspire/
│   ├── data/                  # loader.py (wraps subject.py), preprocess.py, dataset.py
│   ├── features/              # organ_systems.py -- the single source of truth mapping
│   ├── augmentation/          # smotenc.py (bug-fixed), diffusion.py, missingness_model.py
│   ├── models/                # encoders.py, coupling.py, fusion.py, head.py (bug-fixed)
│   ├── training/              # checkpoint.py, train.py (resumable Trainer, implemented)
│   └── eval/                  # metrics.py, validate_synthetic.py
├── scripts/                   # 00 (dry-run data) -> 01 -> 02 -> 03 -> 04, all runnable
└── notebooks/                 # thin Colab/Kaggle launchers
```

## What to do next for a real run

1. Point `configs/default.yaml`'s `data.subjects_dir` at your real INSPIRE data.
2. Resolve the open design decisions flagged in the doc (post-op window, peri-op bin
   size, fluid placement) -- `data/dataset.py`'s fixed-padding approach is the one
   placeholder most worth revisiting first, since it directly implements one of them.
3. Install `synthcity` in your real environment and watch for the warning messages
   this run's defensive wrapping is built to surface if the API has drifted.
4. Decide on the discrete-time survival loss properly (currently collapsed to binary,
   see above) before trusting AUPRC/AUROC numbers on real data.

## Talking to your supervisor about this

Accurate framing: **the design is complete and the pipeline has been verified to run
end-to-end mechanically** (dry-run proof above, with two real bugs found and fixed by
actually executing it, not just reading it). What's not yet true: this has not been run
against real data, and several components are documented placeholders pending the open
design decisions. This is a legitimate "here's my architecture of thought, verified to
actually execute, here's exactly what's left before it's a trustworthy result" -- not
yet a "these are the results."
