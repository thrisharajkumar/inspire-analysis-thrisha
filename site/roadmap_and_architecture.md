# Multimodal Organ-System DNN — Notebook Summary

> **What this file is.** A short, current-status companion to `Research_Aim.md` and
> `roadmap_and_architecture.md`, written to accompany
> `src/INSPIRE_Multimodal_Mortality_Kaggle_Notebook.ipynb` — the runnable Kaggle notebook
> that implements the system-separated architecture those two docs describe, extended with
> the decisions from the most recent working session (GI/MSK systems, cardiorenal coupling,
> operation-count features, the 6-month post-cardiac-surgery exception). If this file and
> the notebook ever disagree, the notebook is the source of truth — this file is a map of
> it, not a spec for it.

## 1. What the notebook is, in one paragraph

A complete, theory-then-code Kaggle notebook (12 parts, ~100 cells) that builds a
**multimodal, organ-system-separated deep neural network** for 30-day peri-operative
mortality prediction on INSPIRE. Six physiological systems (Renal, Cardiovascular,
Respiratory, Metabolic/hepatic, Haematology, Neurological — the existing repo's grouping)
each get their own transformer time-series encoder; two new systems (**Gastrointestinal**,
**Musculoskeletal**) are added as diagnosis/department-driven branches, since neither has a
dedicated lab/vital panel in this dataset (confirmed programmatically in the notebook, not
assumed). An explicit **cardiovascular → renal coupling** is built into the renal encoder's
input (plus a hand-crafted interaction feature), implementing the cardiorenal-syndrome
relationship raised in the working session. All six time-series embeddings plus a static
branch (age/sex/ASA/department/GI/MSK/HFRS/operation-history/medications) fuse into one
mortality prediction via a **Neural Additive Model** by default, so the prediction can be
read back off as a per-system decomposition, not just a single opaque number.

Every data-cleaning, imputation, sampling, and architecture choice is implemented as an
interchangeable, documented `CONFIG` flag rather than a silent default — the notebook's
Part 1 is a from-scratch theory explanation of every option (with citations) *before* any
code runs, specifically so the reasoning behind each choice is reviewable and experimentable
with, not just a comment in code.

## 2. Where it lives, and how it relates to the existing pipeline

| File | Relationship |
|---|---|
| `src/INSPIRE_Multimodal_Mortality_Kaggle_Notebook.ipynb` | **New.** Self-contained (no imports from this repo's `src/*.py` — see §4), Kaggle-portable, implements the system-separated + GI/MSK + cardiorenal architecture end to end. |
| `src/dnn_mortality_pipeline.py` / `dnn_mortality_pipeline_real.py` | The existing **flat, single-encoder** transformer pipeline (7 pre-op features). Still the reference implementation for the two-phase pretrain→finetune pattern the notebook reuses per-system. Not superseded — a useful flat baseline to compare against (see §5). |
| `docs/roadmap_and_architecture.md` §4 | The **target architecture** (six-system grouping, NAM fusion, Mermaid diagram) this notebook is a direct, runnable implementation of. Section 4.1's table is extended in the notebook with the GI/MSK rows. |
| `docs/Research_Aim.md` §2.5, §2.9, §2.10 | The fuller reasoning behind the system-split, fusion strategy, and categorical-embedding decisions — the notebook's Part 1 theory section restates the parts directly relevant to running the pipeline, but this doc has the longer discussion (trade-offs, alternative architectures considered). |
| `src/diagnosis.py` | Its ICD-10 chapter table is reimplemented **inline** in the notebook (self-contained, Kaggle-portable) rather than imported — the chapter ranges are copied verbatim; if this file's `chapters` list is ever edited, mirror the change in the notebook's §5.3/§6.6 cell. |
| `src/frailty_hfrs.py` | Its full 109-code HFRS weights table is **not** imported by default — the notebook ships a representative ~30-code subset for demonstration speed. Swap in the full table (copy the `get_hfrs_weights()` dict into the notebook's §6.6 cell) before trusting any HFRS number from the notebook for publication. |
| `src/nela.py`, `src/score_models.py` (NEWS2) | Not wired into the notebook. Flagged in the notebook's §12.2/§12.3 as a natural next cell — both are already-working, ready to import. |

## 3. What's new relative to the existing six-system architecture

| Meeting-note item | Where it's implemented in the notebook | Key design decision |
|---|---|---|
| Add Gastrointestinal system | §6.7 (folded into §6.6's ICD-10 cell) | ICD-10 chapter XI (`K00-K93`) flag/count + `department=='GS'`, fed into the **static** branch, not a time-series encoder — this dataset has no GI-specific lab/vital panel (confirmed in §4.1) |
| Add Musculoskeletal system | §6.8 (folded into §6.6's ICD-10 cell) | ICD-10 chapter XIII (`M00-M99`) flag/count + `department=='OS'`, same reasoning as GI |
| Cardiovascular parameters added to renal; renal proportional to cardiovascular | §6.9, §9.2, §9.6 | Two mechanisms, not one: (1) the renal `SystemEncoder` receives a pooled cardiovascular summary (mean HR, MAP deviation, IABP flag) as an `extra_context` side-input at every timestep; (2) a hand-crafted `renal_cardiac_interaction` feature (creatinine × \|MAP deviation\|) in the static vector. Deliberately **asymmetric** by default (`CONFIG['SYMMETRIC_CARDIORENAL_COUPLING']=False`) — cardiology does not receive a renal summary back unless this flag is flipped |
| Features learned per system, then jointly learned/fused | §9 (whole part), §1.5.4 | Two senses of "joint," both implemented: (1) end-to-end joint gradient through all encoders + fusion during fine-tuning; (2) unsupervised, label-free autoencoder pre-training per system on the full cohort *before* fine-tuning, so splitting into systems doesn't also split the scarce mortality labels six ways before the network has learned anything |
| Number of operations in the same area matters | §6.10 | `n_prior_ops_same_dept` / `days_since_last_op_same_dept`, computed per patient, offered as a **learned** feature (direction of effect intentionally not assumed) |
| Exception: 6-month wait after cardiovascular surgery | §6.11 | Implemented as a **rule**, not a learned feature, on purpose (sample-size and actionability reasoning in the notebook's §1.6.4) — `cardiac_recovery_exception` flag, `CARDIAC_DEPARTMENTS = {'CTS'}` (verify against your site's department coding), 180-day threshold |
| Pre-op / intra-op / post-op all matter | §1.6.3, `CONFIG['TIME_WINDOW']` | Defaults to `pre_op` (5 days pre-`orin_time`, matching the existing pipeline); `peri_op` extends to `orout_time` and pulls in intra-op `vitals`. A genuine sensitivity comparison, not assumed — §11.5 is the template for running both and comparing |
| ICD-10 code usage | §1.2, §6.6 | Three uses, kept explicitly separate: chapter-membership flags/counts (GI/MSK), a curated externally-weighted score (HFRS), and a documented-but-not-built option for learned per-code embeddings (needs the full cohort to be sample-efficient — see §1.5.3) |

## 4. Design note: why the notebook doesn't import this repo's `src/*.py`

The notebook re-implements JSON loading, the ICD-10 chapter lookup, and the 30-day label
logic **inline**, rather than `import subject`, `import diagnosis`, etc. This is a
deliberate portability choice, not a duplication oversight: Kaggle notebooks run in an
isolated environment where uploading and path-wiring this entire repo as a second dataset
just to get a handful of small utility functions is more friction than it's worth. If this
repo's `src/` modules change in ways that matter (e.g. the ICD-10 chapter ranges, or the
30-day label formula in `subject.py`), the corresponding notebook cell should be
updated by hand — there's a comment at each duplicated piece of logic pointing back to its
source-repo origin for exactly this reason.

## 5. Known limitations of the current notebook run (dev subset, n=30)

- All metrics in the notebook as delivered are from the **30-patient development subset**
  (10 died / 20 survived) — directional only. The notebook is written to scale unchanged to
  the full ~99,886-patient cohort by changing one config path
  (`CONFIG['SUBJECTS_DIR']`); nothing else needs to change.
- SMOTE/ADASYN and MICE (`IterativeImputer`) are implemented but not exercised as the
  default at this sample size — both need more positive examples / more rows than 30
  patients give them to work as intended (documented in the notebook itself, §1.4.2/§12.2).
- The GI/MSK department proxies (`GS`, `OS`) and the cardiac-department proxy (`CTS`) were
  inferred from this dataset's observed department codes, not from an authoritative
  INSPIRE data dictionary — worth a second confirmation pass before publication.
- HFRS uses a ~30-code representative subset of the full Gilbert et al. 109-code table
  (§2 above) — swap in `frailty_hfrs.py`'s full table for a publication-grade HFRS number.

## 6. Suggested immediate next steps (mirrors the notebook's own §12.3, restated here for repo-level tracking)

1. Point `CONFIG['SUBJECTS_DIR']` at the full cohort and re-run.
2. Run the `TIME_WINDOW` (pre-op vs. peri-op) and multi-operation label (last-op vs.
   first-op) sensitivity sweeps for real, not as the template the notebook currently ships.
3. Wire NELA/NEWS2 in as baseline comparison points (`src/nela.py`, `src/score_models.py`
   are both ready to import).
4. Re-run the NAM-vs-concat fusion ablation with more epochs/seeds once the full cohort
   makes the comparison statistically meaningful.
5. Swap in the full HFRS weights table and fix the 2-year/age-75+ window exactly as
   `roadmap_and_architecture.md`'s frailty checklist item already specifies.
