# Research Notes & Questions — Living Log

> **How to use this page:** this is the one file that keeps growing. When a new research
> question, angle, or justification comes up — in a meeting, a paper, or while looking at
> results — add it to the log in Section 3, dated, in a couple of sentences. Nothing here
> needs to be polished. The master checklist in Section 2 is the standing structure that
> Section 3's entries get filed under; tick a box when a question is genuinely settled
> (with a link to where the answer lives), not when it's merely been discussed.
>
> **How this differs from the other research pages**, so nothing gets duplicated:
>
> | Page | Purpose |
> |---|---|
> | `Research_Aim.md` | The deep-dive — full reasoning, literature context, and justification for each question, written once things are worked through |
> | `roadmap_and_architecture.md` | The prioritised task list and the target model architecture |
> | **This page** | The scratchpad — where a question lives *before* it's been worked through, plus the running literature-review tracker and baseline/explainability/evaluation checklists |
>
> When an entry in Section 3 gets fully reasoned out, migrate the conclusion into
> `Research_Aim.md` (or `roadmap_and_architecture.md` if it's a task-list item) and leave a
> one-line pointer here instead of deleting it — the log should show the trail, not just
> the current state.

---

## 1. Where this project is in the research process

Initial phase (dataset understanding, EDA, reviewing the existing DNN pipeline) is done —
see `INSPIRE_Project_Notes.md`, `eda_findings.md`, `feature_audit_findings.md`. The project
is now in the **research and experimentation phase**: establishing a clinically valid
prediction framework, designing the target architecture, and systematically evaluating
approaches against baselines.

**Objectives for this phase:**

- Establish a clinically correct mortality definition
- Review the relevant literature (INSPIRE, NELA, POSSUM, ACS NSQIP, and recent
  perioperative-mortality-prediction work)
- Develop the clinically interpretable deep learning architecture
- Design meaningful feature representations (static, temporal, embedded)
- Compare multiple baseline models
- Develop explainability techniques
- Produce publication-quality experiments and documentation

---

## 2. Master checklist

Organised into the six research phases. Each item links to where it's already answered,
in progress, or open — check items off as they're genuinely settled, not just discussed.

### Phase 1 — Clinical research

**Mortality definition**

- [x] Identify that folder labels use `died()` not `died_30day()` — see `index.md` §4,
      `Research_Aim.md` §1
- [x] Decide to use `died_30day()` going forward — see `Research_Aim.md` §2.1
- [ ] Confirm with James whether the full-dataset folders have the same issue
- [ ] Literature review: how do INSPIRE's own publications, NELA, POSSUM, and ACS NSQIP
      each define their mortality endpoint? *(NELA and POSSUM are pre-op/peri-op scoring
      tools rather than outcome-definition papers — worth checking what endpoint each was
      validated against, not just the tool's inputs.)*
- [ ] Is mortality calculated from the first surgery, the last surgery, per-surgery, or
      per-patient? — see Path C in `roadmap_and_architecture.md` §3, and the open code
      comment in `subject.py`

**Multiple surgeries**

- [ ] Literature review: how do published perioperative-mortality studies handle repeat/
      staged procedures? (exclude, treat independently, merge into one episode)
- [ ] Decide the INSPIRE-specific approach, backed by the literature answer above and the
      EDA already run (`eda_findings.md` §12, chart `16_multi_operation_mortality.png`)

**Prediction window**

- [ ] Why has 30-day mortality become the standard endpoint in perioperative research,
      specifically (vs. 7-day or 90-day)? Find the primary source for this convention
      rather than assuming it — it's cited constantly but rarely justified in one place.
- [ ] Consider whether 7-day and 90-day are worth reporting alongside 30-day as
      secondary endpoints, not replacements

**Clinical variables**

- [ ] Full literature-backed list of established mortality predictors: ASA grade,
      frailty, age, sex, emergency status, labs, medications, vitals, prior admissions,
      comorbidities — cross-check against the organ-system feature table already in
      `roadmap_and_architecture.md` §4.1

### Phase 2 — Feature engineering research

- [ ] Static features: demographics, ASA, procedure type, specialty, diagnosis/ICD
      codes, BMI, Charlson Comorbidity Index, frailty score (HFRS already implemented,
      Charlson not yet — worth deciding if both are needed or if they're redundant)
- [ ] Temporal features: compare raw sequences (current approach) vs. summary
      statistics vs. learned temporal embeddings vs. irregular-time-series-aware encoding
      (e.g. time-aware LSTM/GRU variants, continuous-time transformers)
- [ ] Feature embeddings: one-hot vs. learned embeddings vs. clinical-ontology embeddings
      (e.g. ICD-10 hierarchy, ATC hierarchy) vs. graph embeddings — builds on the ATC-level
      starting point already scoped in `Research_Aim.md` §2.10

### Phase 3 — Model architecture

Target architecture already drafted in `roadmap_and_architecture.md` §4 (system-separated
encoders + Neural Additive Model fusion). Open sub-questions:

- [ ] Multi-modal fusion strategy — early concatenation vs. gated/mixture-of-experts vs.
      hierarchical/additive (`Research_Aim.md` §2.9) — pick one to prototype first
- [ ] Static encoder design — currently unspecified beyond "a small MLP"
- [ ] How much should system-specific encoders share weights vs. specialise
      (`Research_Aim.md` §2.13, item 12)

### Phase 4 — Baseline models

Which comparisons already exist vs. still need implementing:

| Category | Model | Status |
|---|---|---|
| Statistical | Logistic regression | ⬜ Not implemented |
| Tree-based | Random forest | ⬜ Not implemented |
| Gradient boosting | XGBoost | ✅ `gbm_mortality.py` |
| Gradient boosting | LightGBM | ⬜ Not implemented |
| Gradient boosting | CatBoost | ⬜ Not implemented |
| Clinical score | NELA | ✅ `nela.py` |
| Clinical score | POSSUM / P-POSSUM | ⬜ Not implemented — see `Research_Aim.md` §2.4 |
| Clinical score | NEWS2 | ✅ Implemented in `score_models.py`, not yet compared anywhere |
| Deep learning | Existing two-phase transformer | ✅ `dnn_mortality_pipeline.py` / `_real.py` |
| Deep learning | GRU | ⬜ Not implemented |
| Deep learning | TCN | ⬜ Not implemented |
| Healthcare transformer | RETAIN | ⬜ Not implemented — cited as related work, `Research_Aim.md` §2.11 |
| Healthcare transformer | BEHRT | ⬜ Not implemented — new addition, needs a literature read first |
| Healthcare transformer | Med-BERT | ⬜ Not implemented — new addition, needs a literature read first |

### Phase 5 — Explainability

- [ ] Global: SHAP (with the copyright/reproduction caveats already noted for
      TimeSHAP in `Research_Aim.md` §2.11 — SHAP on tabular/static features is more
      standard and lower-risk to start with than TimeSHAP on the full time series)
- [ ] Local: LIME — not yet discussed elsewhere in the docs, worth a short comparison
      against SHAP for why one might be preferred over the other on this data
- [ ] Attention maps — already planned as "attention auditing" in
      `roadmap_and_architecture.md` §4.4
- [ ] Integrated gradients — used by Shickel et al. 2023, the main benchmark paper; worth
      implementing partly *because* it enables a direct comparison
- [ ] Counterfactual explanations — new addition, not yet discussed elsewhere; needs
      scoping (what would "this patient would have survived if X" mean clinically and
      is it defensible to generate)
- [ ] Concept Bottleneck Models, sparse autoencoders, prototype retrieval — already
      planned in `roadmap_and_architecture.md` §4.4

### Phase 6 — Evaluation strategy

**Performance metrics**

- [x] AUROC — in use
- [x] AUPRC — in use
- [ ] Precision, recall, F1, sensitivity, specificity at a clinically chosen threshold
      (not just the F1-optimal threshold currently reported)
- [ ] Calibration curves + Brier score — flagged as a gap in `Research_Aim.md` §3D

**Clinical evaluation**

- [ ] Decision curve analysis — new addition, not yet discussed elsewhere; worth a short
      literature read on how it's normally reported for a perioperative-mortality model
- [ ] External validation — depends on getting a second dataset or a held-out INSPIRE
      split that behaves like one (e.g. a later admission-year cohort)
- [ ] Subgroup analysis — by department, ASA class, emergency vs. scheduled, age band
- [ ] Fairness analysis — by sex, age, department; needs a decision on which fairness
      metric is clinically meaningful here before running it
- [ ] Robustness testing — degrade/remove inputs and check the model doesn't fail badly
      (already partly scoped as "robustness" in `Research_Aim.md` §2.13, item 12)

---

## 3. Open questions log

Add new entries at the top, newest first. Keep each entry short — question, the current
angle/hypothesis, and what would justify or settle it. Status moves
`open → in progress → resolved (link)`.

<!--
Template for a new entry:

### YYYY-MM-DD — Short question title

**Question:** ...
**Angle / hypothesis:** ...
**Justification needed:** what evidence, reading, or experiment would settle this
**Status:** open
-->

### 2026-07-29 — Where should "infection/sepsis" live in the organ-system grouping?

**Question:** The current 6-system grouping (renal, cardiovascular, respiratory,
metabolic, haematology, neurological) has no clean home for infection/sepsis-driven
deterioration, even though the most mortality-linked ICD-10 codes found in EDA (D65, I46,
R57, J80, K72, A41) cluster there.

**Angle / hypothesis:** either fold sepsis-adjacent codes into haematology/coagulation
(where D65 already sits) and cardiovascular (I46, R57), or give infection its own system
box and accept some ICD-10 overlap between systems.

**Justification needed:** check whether treating sepsis as its own system changes
anything in the multi-operation / department cross-tab analysis, or whether it's purely a
labelling convenience.

**Status:** open — see `roadmap_and_architecture.md` §4.1 footnote.

---

*(This is the seed entry, migrated from the architecture discussion so the log format is
established. Everything above this line in Section 3 should be added going forward — don't
delete this entry, just keep adding above/below it as new questions come up.)*

---

## 4. Literature tracker

Running list of papers to read, currently reading, or already incorporated. Keep the
one-line note honest about how deeply it's actually been read (skimmed abstract vs. fully
read vs. implemented from).

### Already cited and incorporated

| Paper | Note |
|---|---|
| Shickel et al. 2023, *Scientific Reports* | Main benchmark — 56,242 patients, AUROC 0.92, MC Dropout, integrated gradients |
| Gilbert et al. 2018 (HFRS), *The Lancet* | Fully implemented in `frailty_hfrs.py` |
| Koh et al. 2020 (Concept Bottleneck Models), ICML | Read for the CBM architecture idea |
| Choi et al. 2016 (RETAIN), NeurIPS | Read for the attention-over-time idea; not yet implemented as a baseline |
| Bento et al. 2021 (TimeSHAP), KDD | Read for the time-series-SHAP idea |
| Lee et al. 2018/2019 (DeepHit / Dynamic-DeepHit) | Read for the time-to-event reframing |
| DySurv, 2024–2025 | Read — conditional-VAE dynamic survival model, closest recent comparator |

### To review (new, from this phase's literature pass)

| Topic | Why | Status |
|---|---|---|
| POSSUM / P-POSSUM original papers (Copeland 1991, Prytherch 1998) | Needed before implementing as a baseline | ⬜ not started |
| ACS NSQIP | Named in the uploaded roadmap as a literature source, not yet reviewed here | ⬜ not started |
| ASA governance documentation | Primary source for the classification itself | ⬜ not started |
| BEHRT | Healthcare transformer baseline candidate | ⬜ not started |
| Med-BERT | Healthcare transformer baseline candidate | ⬜ not started |
| Why 30-day specifically became the standard perioperative endpoint | Needed for Phase 1, prediction-window question | ⬜ not started |
| Decision curve analysis, standard reporting conventions | Needed for Phase 6 | ⬜ not started |
| INSPIRE dataset paper (Lee et al.) — exact citation | Flagged repeatedly across `index.md` and `Research_Aim.md` as unconfirmed | ⬜ not started |

*As with every other citation list in this repo — treat "reviewed" as a floor, not a
guarantee of exact volume/page/DOI accuracy. Confirm before anything goes into a
manuscript.*

---

## 5. Proposed future split

The docs are compact enough for now that everything research-related lives in three
files (`Research_Aim.md`, `roadmap_and_architecture.md`, this page). If this page grows
past a comfortable single-page length, the natural split — by phase — would be:

```
docs/research/
├── 01_literature_review.md
├── 02_clinical_questions.md
├── 03_mortality_definition.md
├── 04_feature_engineering.md
├── 05_model_architecture.md
├── 06_baseline_models.md
├── 07_explainability.md
├── 08_evaluation.md
├── 09_experiment_log.md
└── 10_future_work.md
```

Not worth doing yet — splitting now would mean navigating ten near-empty files instead of
one. Revisit this once Section 3's log is long enough that scrolling past it to reach
Section 2 becomes annoying.
