# Novelty & Research Ledger

> **What this page is.** A running record of every research-backed idea explored for this
> project — what it is, *why* it's a genuine methodological choice (not just a feature
> added), what it's grounded in, and its real status. This page exists so that "what's
> novel here" has one answer, not a scattered one across many conversations. Update it
> whenever a new idea is researched, built, or tested — don't let findings live only in
> chat history.

**Status legend:** 🔵 Researched only · 🟡 Partially built · 🟢 Built & tested · ⚪ Idea, not yet researched

---

## 1. Architecture novelty (PACO-Net core claims)

Full detail and every supporting paper: `PACO_Net_Architecture.md`. Summarised here for
one-place review.

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| System-level (not feature-level) additive decomposition | Every published additive/interpretable clinical model found decomposes into single scalar features. Ours decomposes into whole *organ systems*, each term the output of a full temporal network over dozens of raw features | Benchmarked against a 59,985-patient postoperative-mortality GAM-NN (feature-level, explicitly avoids interactions) | 🟢 Built & tested (NAM fusion, current baseline) |
| Learned inter-organ coupling (graph-attention) | Replaces one hand-picked renal↔cardiac link with a layer that discovers which systems move together from the data itself | IOC-MT (Kong et al. 2025) — closest precedent, but general ICU organ-dysfunction, not perioperative mortality, and uses an adjustment mechanism, not a strict additive sum | 🔵 Researched only |
| Mixed-modality organ set | Six systems with real time series + two (GI, MSK) with only diagnosis/department proxies, unified in one framework | No precedent found — IOC-MT's six organs are all time-series-derived | 🟢 Built & tested (GI/MSK from ICD-10 chapters) |
| Explicit hard-rule-vs-learned split | 6-month cardiac washout stays a fixed rule beside a learned coupling layer, for a stated reason (sample size + actionability) | Not discussed as a deliberate stance anywhere found — papers are either fully learned or fully rule-based | 🟢 Built & tested |
| Phase-aware continuous timeline (segment-embedding, not tripled encoders) | Cheaper, more data-efficient way to test pre/peri/post than physically tripling every organ encoder | My own proposed refinement, reasoning from BERT-style segment embeddings — not lifted from one paper | 🔵 Researched only |
| SurvTRACE-style discrete-time hazard output | Predicts risk *per future time bin*, not one flat probability — answers "when," not just "if" | Wang & Sun 2022; clinically validated on real cardiovascular patients, Shinohara et al. 2024 | 🔵 Researched only |

---

## 2. Data & feature novelty

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| ICD-10-PCS procedure features | `icd10_pcs` has been in `operations_df` since Part 3 but never turned into a dedicated feature — same treatment gap `icd10_cm` had before GI/MSK were built | Direct parallel to the existing, already-validated ICD-10 diagnosis-chapter routing | 🔵 Researched only, real gap identified |
| NEWS2 as a computed feature | 6 of 7 official NEWS2 parameters already exist in `ward_vitals` (`rr`, `spo2`, `nibp_sbp`, `hr`, `bt`, `fio2` as an oxygen-use proxy); consciousness needs an approximate GCS→ACVPU mapping, stated explicitly as an approximation | Royal College of Physicians NEWS2 (2017); ≥5 is the established, externally-validated urgent-response threshold | 🔵 Researched only |
| NEWS2 as a third SMOTENC stratification axis | Adds *current acute physiological state* to the existing (department, ASA) grouping — two patients in the same department/ASA but very different NEWS2 arguably shouldn't be blended | Extension of the already-built grouped-SMOTENC clinical-neighbourhood principle | ⚪ Idea |
| Regression (MICE) imputation, benchmarked not assumed | Tested head-to-head against rule-based methods on real held-out masked values — won 1/6 systems on the 10,942-patient run, not a universal upgrade | Confirms published findings that MICE-style methods are dataset-dependent, not universally beaten by deep imputation at this missingness level | 🟢 Built & tested |
| SAITS as the next imputation upgrade | Self-attention imputation, better-matched to irregular, sparse, multivariate clinical time series than classic MICE | Du, Cote & Liu 2023; 2026 Nature Sci Rep benchmark found deep methods (SAITS, BRITS) generally ahead of MICE for multivariate time series | 🔵 Researched only |
| PIP/PEEP absence treated as "not applicable," not missing | PIP (peak inspiratory pressure) and PEEP (positive end-expiratory pressure) are ventilator-*setting* values — they only exist for patients actually on mechanical ventilation. Most patients aren't ventilated, so absence here is the same "not applicable" logic already used for HFRS (age<75), not a data-quality gap to explain away | Direct extension of the existing HFRS not-applicable design pattern | 🔵 Researched only — likely already correctly handled by the existing observed/imputed mask machinery, but not explicitly verified for these two features specifically; worth a direct check before claiming it's "built" |
| Per-patient regression-to-the-mean imputation | Current population-average fallback fills a patient's total gap toward the *whole cohort's* mean. Refinement: when a patient has *some* real readings of a feature, fill their remaining gaps toward *that patient's own* mean, not the population's — more personalised, likely more accurate for features with high inter-patient but low intra-patient variation (e.g. baseline creatinine) | Standard "regression to the mean" statistical concept, applied per-patient rather than per-population; a genuine refinement of the existing fade-to-mean interpolation | ⚪ Idea — directly benchmarkable against existing methods using the already-built held-out masking test (§7.6/7.7) once implemented |
| NELA as an added external comparator | National Emergency Laparotomy Audit — a real, established UK risk-prediction benchmark, more directly comparable to this cohort's general/emergency surgery population than ASA/POSSUM alone | Widely-cited real clinical audit tool, same category as the existing POSSUM/ASA comparisons | ⚪ Idea |
| Manual-feature-extraction baseline (logistic regression / XGBoost on hand-picked features) | Tests whether the organ-system deep architecture actually earns its complexity over a much simpler, standard approach — a necessary, honest comparison for any paper claiming the architecture adds real value | Standard ML methodology practice — any deep model claiming an advantage should be shown beating a well-tuned simple baseline | ⚪ Idea |

---

## 3. Interpretability & validation novelty

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| Held-out masking benchmark for imputation accuracy | Real, measured MAE per system per method — not a visual impression | Standard held-out validation principle, applied per-system (not commonly done at this granularity in clinical imputation literature) | 🟢 Built & tested |
| SHAP as an independent cross-check on NAM | NAM's decomposition is architectural (exact by construction); SHAP is post-hoc. If SHAP on a simpler baseline model roughly agrees with NAM's per-system story, that's real external evidence the interpretability isn't an architectural artifact | Lundberg & Lee 2017 (SHAP); real caveat found — SHAP on deep/sequence models (`DeepExplainer`) is markedly less stable than on tree models (`TreeExplainer`) per a recent comparison study, so the baseline model for this check matters | ⚪ Idea |
| Broaden beyond SHAP: LIME, Integrated Gradients, counterfactual explanations | Explicit goal — don't lean on one XAI technique alone. Each has different assumptions and failure modes: LIME (local surrogate models, can be unstable run-to-run), Integrated Gradients (gradient-based, well-suited to the existing transformer encoders since they're differentiable), counterfactual explanations ("what would need to change for this prediction to flip" — often more clinically intuitive to a doctor than a feature-importance number) | Standard XAI literature; Concept Bottleneck Models already referenced in `PACO_Net_Architecture.md`'s paper list as a related principle | ⚪ Idea — a small comparative section (NAM vs. SHAP vs. Integrated Gradients vs. one counterfactual example, showing broad agreement) would be a genuinely strong interpretability-validation section for a paper |
| Attention-audit check (observed vs. imputed) | Sanity check on whether the model's attention concentrates on genuinely observed values, not filled-in guesses | Already built as a diagnostic in §11.3 | 🟢 Built & tested |

---

## 4. Anomaly detection & data-quality novelty

Full design and both interview-question answers: see the anomaly-detection/longitudinal
discussion in project chat history — worth its own page if this becomes a real build.

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| Reuse autoencoder reconstruction error as an anomaly score | Phase 1 pre-training already trains each organ system to reconstruct normal patient data and already computes a masked reconstruction loss — a patient with unusually high reconstruction error *is* an anomaly signal, already sitting in the pipeline, unused for this purpose | Standard reconstruction-based anomaly detection principle (dominant modern approach for multivariate time series), applied to an asset this project already has rather than a new model | ⚪ Idea — cheapest anomaly-detection win available, since it needs no new training |
| Two-layer hybrid: deterministic plausibility rules + learned reconstruction-error score | Rules catch the cheap, obvious, fully-explainable cases fast; the learned layer catches subtle multivariate patterns rules structurally can't. Neither layer disrupts clean data — only flagged records pay extra latency | Statistical-process-control methods (CUSUM, control charts) documented to degrade under the non-stationarity/irregular sampling of real clinical data — motivates *not* relying on Layer 1 alone | ⚪ Idea |
| NEWS2 before/after consistency check on **synthetic and augmented** records | Compute NEWS2 on a record *before* and *after* it's generated — whether via SMOTENC blending two similar real patients, or via sequence jitter/time-masking of one real patient — and check the score stays clinically consistent. A wildly different NEWS2 after generation is a concrete, testable signal that the synthetic/augmented record has drifted into physiological implausibility, not just "close in raw feature space." Concretely: pick one parameter, compare its value (or its NEWS2 contribution) before and after generation — if unchanged/consistent, that's evidence the generated record is more reliable | Direct extension of the reconstruction-error anomaly reasoning above, applied specifically to this project's own two synthesis mechanisms (grouped SMOTENC and sequence augmentation) rather than to incoming raw data | ⚪ Idea — clarified and broadened from an earlier draft of this idea that only covered augmented sequences; now explicitly covers SMOTENC-blended patients too |

---

## 5. Longitudinal analysis novelty

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| Mixed-effects models for individual vs. cohort trends | Fixed effects = population-average trend (cohort-level); random effects = per-patient deviation from it (individual-level) — one coherent framework instead of two separate ad hoc analyses | A real, directly relevant finding: mixed-effects models detected genuine apathy progression in Parkinson's disease over 7 years where plain t-tests and linear regression did not — precise evidence for "how do you know a trend is real, not noise" | ⚪ Idea |
| Informative-observation-process bias awareness | Sicker patients get measured more often — a naive trend line can mistake "measured more frequently" for "getting worse." This is the *same* MNAR reasoning already central to the project's missing-data theory, extended into the time dimension | Established longitudinal-EHR statistics literature on informative visiting processes | 🔵 Researched only — strong connecting thread between existing missingness work and any future longitudinal work |
| Extend risk-over-time work into a formal mixed-effects model | The existing §11.8/§11.9/§11.10 risk-trajectory plots are an informal, per-patient version of this — a real mixed-effects fit would make the trend claims statistically defensible, not just visually suggestive | Same grounding as above | 🟡 Partially built (the visual/mechanism exists; the statistical rigor doesn't yet) |

---

## 6. Department & procedure-granularity modeling

New this round — raised in a clinician review conversation, not yet in the codebase.

| Idea | Reasoning | Grounded in | Status |
|---|---|---|---|
| Department-specific model vs. the pooled/generalised model — comparative study | Tests a real clinician-raised concern directly: pooling every department into one model risks one department's patterns diluting or "poisoning" another's — general surgery (GS, the largest single department in the cohort — 3,160 patients in the dev-scale run) is the natural first candidate, since it's the only one with enough volume to train a standalone model with real statistical power | Standard model-specialisation-vs-pooling ablation methodology; the clinician's own domain concern about cross-department contamination | ⚪ Idea |
| ICD-10-PCS as a finer-grained grouping than department | Department tells you *where* (which surgical service treated the patient); PCS tells you *what* (which specific operation was actually done). Two genuinely different levels of granularity — a department-level model captures broad specialty patterns, while PCS-level stratification could capture heterogeneity *within* a department that department alone misses (two very different operations both coded under the same department) | Direct extension of the ICD-10-PCS feature gap already identified in §2 | ⚪ Idea |
| Candidate cross-system coupling hypotheses to validate the (still unbuilt) learned coupling layer against | Clinician-suggested chains, offered as concrete things the graph-attention coupling layer (§1) *should* discover if it's working correctly: abnormal blood results → ventilatory need → respiratory system involvement; renal function decline → immediate ICU escalation. This gives that still-unbuilt component a real validation target beyond "did the metric go up" — does it actually recover known clinical chains | Clinical domain input; ties directly to the graph-attention coupling item already in §1 | ⚪ Idea — the most useful contribution of this note: a concrete way to *validate* the coupling layer once it exists, not just train it |

**A note on scope, not a new row:** several other points from this same review were confirmations of designs already built, not new ideas — worth naming so they aren't mistakenly re-proposed: the two-phase unsupervised-pretrain-then-supervised-finetune design (🟢 already built, §Architecture), and the constraint that generated/augmented data must stay within clinically plausible noise bounds (🟢 already built via `JITTER_SIGMA`, tuned deliberately small).

**One phrase from this review I want to flag rather than guess at:** *"not incubated then they are fit enough"* — my best-effort reading is "not intubated, therefore fit enough" (ventilation status as a proxy for baseline fitness/readiness), which would connect to the PIP/PEEP not-applicable point in §2 — but this is a genuine guess, not a confident interpretation. Worth confirming before it's treated as a real ledger item.

---

## 7. How to use this page

- **Before claiming novelty anywhere** (paper, interview, meeting) — check the row's
  grounding and status here first. A 🔵 or ⚪ item is a real, defensible *direction*, not
  yet a *result* — say so plainly if asked.
- **When something moves status** (researched → built → tested) — update its row, don't
  leave it stale.
- **The standing caveat that applies to every 🔵/⚪ row**: grounding here comes from
  targeted searches across this project's research sessions, not a systematic literature
  review. Treat each as "a strong reason to formally investigate," not "confirmed novel,"
  until independently verified.
