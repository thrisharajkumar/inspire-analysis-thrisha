# INSPIRE Project — Research Questions, Explorations & Roadmap

## 0. How this fits with the rest of the repo

The repo already has four documentation files, and they are not redundant with each
other or with this one — each answers a different question:

| File | Answers | Read this if you want... |
|---|---|---|
| `README.md` | What are the five raw INSPIRE CSV tables? | The original data dictionary, as published with the dataset |
| `docs/index.md` (site "Home") | What is the current architecture, current config, current results, and the full prioritised roadmap? | **The single source of truth.** Everything technical and current. |
| `docs/INSPIRE_Project_Notes.md` | How does the pipeline work, explained simply, end to end? | A plain-English walkthrough of the transformer, attention, pre-training, class imbalance, and the three comparison models (NELA / GBM / DNN) |
| `docs/eda_findings.md` | What does the full 99,886-patient cohort actually look like? | Cohort-level findings — who dies, which departments, which ICD-10 codes, feature coverage, correlations, frailty |
| `docs/feature_audit_findings.md` | What data does each *individual* patient actually have? | Per-patient completeness on the 30-patient development subset |
| **This file** | What were the open research questions, and what do they lead to? | The connective layer: your questions → what's already answered → what's still open → new ideas |

If you're starting a new conversation with no other context loaded, `docs/index.md` is
still the one file to paste in first — it's the project's memory. This file is the
research-direction layer on top of it.

---

## 1. Where the project stands right now — one-paragraph version

**Goal:** predict 30-day post-surgical mortality from the INSPIRE Korean perioperative
dataset (~99,886 patients, 2011–2020), using a deep learning model that reads clinical
time series rather than a single snapshot — and, critically, that can *explain* its
predictions in a way a surgeon could act on (which organ system is driving the risk, not
just a number). **Current pipeline:** a two-phase transformer (`dnn_mortality_pipeline.py`)
— autoencoder pre-training on unlabelled time series, then a classifier fine-tuned
end-to-end — running on a 29–30 patient development subset, with AUROC fluctuating
between 0.67 and 0.86 depending on the run (not yet meaningful; the test set is too small
— one wrong prediction moves AUROC by ~0.11). **Comparison baselines:** NELA (fixed
clinical equation, implemented, `nela.py`), GBM (Saranya's XGBoost model with/without
frailty, `gbm_mortality.py`), and the DNN. **Benchmark to beat:** Shickel et al. 2023
(*Scientific Reports*), AUROC 0.92 on 56,242 patients from a similar (though not
identical) problem. **The single highest-priority open issue**, found during EDA and not
yet fixed in the pipeline: the `survived/`/`died/` folder labels were built from "died at
any point in this record," not "died within 30 days of the last operation" — 473 of the
942 all-cause deaths in the full cohort are likely **not** true 30-day deaths, and this
has not yet been corrected in the loader. Everything downstream — every AUROC number, the
`pos_weight` calculation, the whole class-imbalance strategy — is provisional until this
is fixed.

---

## 2. The research questions, worked through

This section takes the questions and ideas from the working session, in the rough order
they came up, and works through each one against what's already in the repo.

### 2.1 "30-day perioperative mortality prediction" — what exactly is being predicted?

The target is clear in principle: `inhosp_death_30day`, defined in the codebase as

```python
label = 1 if inhosp_death_time < orout_time + (30 × 24 × 60) else 0
# died within 30 days of leaving the operating room
```

But two things about this definition are still open, and both matter more than they look:

1. **The label bug above (Section 1).** The folder structure everyone has been loading
   data from encodes "died ever," not "died within 30 days." This needs fixing before any
   number from this project should be trusted or published.
2. **"Perioperative" vs. "pre-operative."** The project is *described* as peri-operative,
   but the pipeline as implemented only reads data from the 5 days *before* surgery — it
   never looks at what happens during or after the operation. This is a real, undecided
   scope question (see Section 2.6 below), not a technical limitation — the peri-operative
   window (intraoperative vitals) is fully available in the data, just not wired in.

**Recommendation:** treat definition #1 as a blocking bug fix (see the roadmap in Section
3), and definition #2 as a genuine research question to answer empirically — build both a
pre-op-only and a peri-operative version and report both, rather than picking one by
assumption.

---

### 2.2 Multiple surgeries — last surgery, first surgery, or exclude?

This came up as an open argument: should patients with more than one recorded operation
be excluded entirely (as an experiment), or should the label be computed relative to
their first operation or their last?

There's already a code-level flag about exactly this in `subject.py`:

```python
# Should it not be 30 days from the last operation not the first operation?
#  Could be that subsequent operations are seen as causal consequence of initial operation?
#  This still seems wrong, many subjects have multiple operations over 10 years.
```

Laying out the three options properly:

| Option | What it means | Argument for | Argument against |
|---|---|---|---|
| **From last operation** (current implementation) | 30-day window starts at the most recent surgery before the outcome | Matches "did this specific operation, the one we're predicting risk for, kill the patient" | If a patient had 4 operations over 6 years, the earlier ones tell you nothing about *this* prediction task even though they're in the record |
| **From first operation** | 30-day window starts at the patient's earliest recorded surgery | Captures "this patient's surgical journey started here" | Loses the fact that most recent surgery is usually what's clinically relevant to current risk; a patient who had a minor procedure in 2012 and a major one in 2019 gets the wrong reference point |
| **Exclude multi-operation patients** | Only model patients with exactly one recorded operation | Cleanest possible label — no ambiguity about which surgery caused what | **Selection bias risk**: patients who need multiple operations are very plausibly the *sicker* patients (repeat interventions, complications, staged procedures for cancer, etc.) — dropping them could make the remaining cohort look artificially healthy and the model's risk estimates artificially optimistic. This needs checking, not assuming. |

**What the repo already tells us about this question:** `docs/eda_findings.md` §12 has
already run "mortality rate vs. number of operations" as an open EDA cell — that chart
(`16_multi_operation_mortality.png`) is the direct empirical answer to whether multi-op
patients are higher risk, which is exactly the evidence needed before deciding whether
excluding them would introduce selection bias.

**Recommendation:** don't pick one definition — treat this as a **sensitivity analysis**
that is itself a small, reportable piece of research: run the pipeline under all three
definitions and see whether the model's conclusions (which features matter, which AUROC
range) are stable across them. If they are, that's a robustness finding worth stating in
the paper. If they aren't, that tells you the operation-selection question genuinely
changes what "risk" means in this dataset — also worth reporting, and it would settle the
argument with real evidence.

---

### 2.3 Understanding ASA

The **ASA Physical Status Classification System** (American Society of Anesthesiologists)
is the oldest and simplest of the three clinical scores in this project. Some
context that's useful for the comparison in 2.4:

- It's a **subjective clinical judgement**, not a calculation — an anaesthesiologist looks
  at the patient and assigns a class from **I to VI**, on the day of surgery.
- **I** = healthy patient; **II** = mild systemic disease, no functional limitation; **III**
  = severe systemic disease; **IV** = severe systemic disease that is a constant threat to
  life; **V** = moribund, not expected to survive without the operation; **VI** = declared
  brain-dead organ donor.
- An **"E"** suffix (e.g. "III E") denotes an emergency — defined as a situation where
  delaying treatment would significantly increase the threat to life or limb.
- It has been in continuous use since **1961/1963**, and is explicitly *not* recommended by
  the ASA itself as a standalone risk-prediction tool — the ASA House of Delegates has
  stated it should be combined with other factors (surgery type, frailty, deconditioning),
  not used in isolation.
- Its main known weakness is **inter-rater reliability** — different anaesthesiologists can
  and do assign different classes to the same patient, because it's a gestalt judgement,
  not a formula.

In INSPIRE, the `asa` field lives in the `operations` table (values 1–5 observed in this
dataset) and is already validated in the repo's own EDA: `docs/eda_findings.md` §2 shows
mortality rate climbing steadily and monotonically from ASA 1 through ASA 6, under both
the all-cause and 30-day definitions — exactly what you'd expect if the data and labels
are trustworthy. **It is currently not used as an input feature anywhere in the DNN
pipeline** (only NELA and the GBM's clinical baseline reference it) — this is a concrete,
low-effort thing to add, given it has 100% coverage (every patient has exactly one
operation record) and is one of the strongest, most established predictors in the
literature.

---

### 2.4 ASA vs. POSSUM vs. NELA — laid out clearly

All three are pre-existing clinical tools this project compares itself against. None of
them are machine-learned — they're either a judgement call (ASA) or a fixed equation
fitted once, long ago, on a specific cohort, and then frozen. That's the point of
including them: they're the thing any new model has to beat to be worth using clinically.

| | **ASA** | **POSSUM / P-POSSUM** | **NELA** |
|---|---|---|---|
| **Full name** | American Society of Anesthesiologists Physical Status | Physiological and Operative Severity Score for the enUmeration of Mortality and morbidity | National Emergency Laparotomy Audit |
| **What it is** | A clinician's subjective judgement, class I–VI | A statistical equation: 12 physiological factors + 6 operative factors, each scored 1/2/4/8 (exponential severity), summed and fed through logistic regression | A statistical equation: 25 variables (age, ASA class, albumin, urea, blood pressure, urgency, etc.), fed through logistic regression |
| **Developed for** | General pre-anaesthesia assessment, any surgery | Auditing/comparing surgical unit performance across *any* general surgery (originally a quality-improvement tool, not meant for individual-patient prediction) | Specifically emergency (unplanned) laparotomy, as a UK national audit standard |
| **Equation (mortality)** | None — ordinal category only | `ln(R/(1-R)) = -7.04 + 0.13×PhysiologyScore + 0.16×OperativeScore` (original); Portsmouth revision uses different, better-calibrated coefficients because the original consistently **overestimates** mortality in low-risk patients | Published logistic-regression equation with 25 coefficients, e.g. `logit = -3.04678 + 0.06660×age + 1.13007×(ASA=3) + ... - 0.04323×albumin + ...` — implemented directly in `nela.py` |
| **Needs data from** | Clinical gestalt only | Pre-op labs/vitals + intra-operative findings (blood loss, contamination, malignancy) | Pre-op labs/vitals + urgency + operative severity, specific to laparotomy |
| **Known bias** | Inter-rater variability; explicitly *not* validated as a standalone predictor by its own governing body | Systematically **overestimates** mortality, especially in low-risk patients — this is *why* P-POSSUM exists as a correction | Built and validated on a large, specific UK population (emergency laparotomy only) — applying it to INSPIRE's broader, non-UK, mixed elective/emergency population is a genuine external-validation question, not a given |
| **Status in this project** | Not currently a model input | Not implemented | Implemented (`nela.py`), used as the fixed clinical baseline |

**Why this table matters for the project, beyond background:** all three tools share one
property that makes them a fair and important baseline — they are **fixed, published, and
frozen**. They don't learn from INSPIRE data at all (NELA in particular was built on a UK
population, not INSPIRE). Their performance on INSPIRE tells you two different things at
once: (1) how good the *idea* of pre-operative risk scoring is in general, and (2) how well
a UK/general-surgery-derived tool transfers to a Korean, mixed-department population — a
distribution-shift question worth stating explicitly in any write-up, since a
lower-than-published NELA AUROC on INSPIRE could mean the tool travels poorly, not that the
tool is bad. **POSSUM is the one of the three not yet implemented in this codebase** — it
would be a natural fourth baseline alongside NELA, since its variable list overlaps heavily
with what's already extracted for NELA and the GBM.

---

### 2.5 Feature selection → model per system/function → feature embeddings → autoencoder

This is the architecture from the diagram you brought to this conversation, and it turns
out to match — almost exactly — the target architecture already sketched in
`docs/index.md` §14 (written before this conversation, from earlier discussions with your
supervisor). Worth stating plainly: **this is not a new idea to introduce, it's the
existing target design, and this project already has the two building blocks it needs
(organ-system feature groupings, and a working autoencoder-pretraining pipeline) — they
just haven't been connected yet.**

Reading your diagram against the current codebase:

```
Meds ~500, Ward Vitals ~120, Labs ~75   ← raw inputs (medications.csv, ward_vitals.csv, labs.csv)
        ↓ (each arrow = feature routed to its physiological system)
Respiratory / Nervous / Cardiovascular / Neurological   ← "clinically meaningful function/system"
        ↓
Mr, Mn, Mc, Mn (orange)   ← one MODEL (encoder) per system
        ↓
Fr, Fn, Fc, Fn (green)    ← one FEATURE/EMBEDDING per system, output of that system's model
        ↓ (all converge)
Mr (large orange, right)  ← a final fusion/meta-model combining all system embeddings
        ↓
D̈ (green circle)          ← the final prediction (death / mortality risk)
```

This is precisely the "system-separated embedding architecture" already described (with
slightly different notation) in `docs/index.md` §14:

```
Current (black box):        7 features → 1 transformer → 1 embedding → mortality
Target (explainable):
  Renal features          → Encoder 1 → renal_embedding
  Cardiovascular features → Encoder 2 → cardio_embedding
  Respiratory features    → Encoder 3 → resp_embedding
  Metabolic features      → Encoder 4 → metabolic_embedding
                                   ↓ concatenate
                          NAM classifier (one shape function per system)
                                   ↓
     "Renal: CRITICAL | Respiratory: CONCERN | Cardio: OK | Metabolic: OK" → mortality = 0.84
```

**Concretely, how to build it, using what already exists in this repo:**

1. **Feature selection** — already scoped in `docs/index.md` §15 item 4 (the 6-system
   grouping of all 126 parameters into Renal / Cardiovascular / Respiratory /
   Metabolic-hepatic / Haematology-coagulation / Neurological) and *already implemented and
   running* as statistical feature selection in `feature_selection_pipeline.py` (see
   Section 2.13 below — this script already does univariate testing, multiple-comparison
   correction, redundancy removal, and department-adjusted significance checking; it just
   hasn't been surfaced as "done" in earlier docs).
2. **Model per system** — replace the single `TimeSeriesTransformer` in
   `dnn_mortality_pipeline.py` with **N separate instances** of it, one per organ system,
   each taking only that system's features as input. The existing two-phase training
   pattern (autoencode → freeze/unfreeze → classify) applies to each system encoder
   independently.
3. **Feature embeddings per system** — each system's encoder, after mean-pooling over time
   (exactly as the current single encoder does), produces one embedding vector per system
   per patient. This is the `Fr` / `Fn` / `Fc` / `Fn` layer in your diagram.
4. **Autoencoder** — unchanged in principle, just run per-system rather than once — each
   system's encoder is still pre-trained unsupervised on reconstruction, which matters even
   more here because splitting into systems also splits the already-scarce mortality
   labels' effective sample size per branch.
5. **Fusion / final model** — this is the one genuinely new design decision, and it's the
   one that determines how interpretable the result actually is (see Section 2.14C). A
   plain concatenation + `Linear` layer works but is not meaningfully more interpretable
   than the current single model — a **Neural Additive Model** (each system embedding
   contributes its own separately-plottable, additive shape function to the final logit)
   is the version that actually delivers "Renal: CRITICAL, Respiratory: CONCERN" as a real
   decomposition rather than a post-hoc story.

---

### 2.6 Pre-operative, peri-operative, and post-operative — what actually helps decide mortality?

The three windows map onto real fields already in every subject JSON:

```
[admission_time=0] ... [orin_time] —— surgery —— [orout_time] ... [discharge_time]
        ↑ pre-op window                              ↑ post-op window
                        ↑ intra-op window (orin_time to orout_time)
```

- **Pre-operative** — the only window the current pipeline reads (5 days before `orin_time`).
  Captures the patient's *baseline* state: chronic disease, nutritional status, kidney/liver
  function, how sick they already were walking in. This is what ASA, POSSUM, and NELA are
  all fundamentally trying to summarise in a single number.
- **Intra-operative** — not used by the current pipeline at all (72-parameter `vitals` table,
  entirely unused — see Section 2.7). Captures the *acute physiological stress response* to
  the surgery and anaesthesia itself: how much blood was lost, how unstable the blood
  pressure was, how long anaesthesia ran, whether oxygenation held up. Two patients with
  identical pre-op labs can have very different intra-operative courses.
- **Post-operative** — captures whether the patient is *recovering or deteriorating*: ICU
  length of stay, whether CRRT/ECMO/IABP got started, trending labs after surgery. This is
  the window where the time-to-event/deterioration-timing question in Section 2.8 lives.

**Which one "helps decide"?** All three carry different, complementary information, and
the honest answer is that this is an empirical question this project is well-positioned to
answer directly, because — unusually — it has all three windows available in the same
dataset:

- Train a **pre-op-only** model (what exists today).
- Train a **pre-op + intra-op** model (a true peri-operative model).
- Train a **pre-op + intra-op + early post-op** model, and see where the AUROC gain plateaus.

This directly answers a clinically meaningful question with three distinct use cases: a
pre-op-only model is a **decision-support tool** (should we operate, should ICU be booked
in advance); a peri-operative model is a **real-time monitoring tool** (should the surgical
team escalate mid-case); a model that includes early post-op data is an **early-warning /
rescue tool** (should this patient be flagged now, before a formal deterioration event).
These are three different products, not three versions of the same one — worth framing that
way in any write-up, rather than treating "more data = better AUROC" as the only finding.

---

### 2.7 "Vitals are more accurate during surgery"

This intuition is directly confirmed by the repo's own EDA, not just plausible reasoning.
Two concrete findings from `docs/eda_findings.md` §6–7 explain *why*:

- **`spo2` coverage drops from 93% "anywhere in the record" to 40% "in the pre-op
  window"** — the explanation given in the doc is that monitors get physically attached to
  a patient once they're admitted or in theatre, not beforehand. The same pattern holds for
  `uo` (urine output): 63% "anywhere," **0% pre-op**.
- The intra-operative `vitals` table is populated automatically by anaesthesia machines and
  monitors, at regular (often minute-level) intervals, for the entire duration of surgery —
  a fundamentally different data-generating process from ward vitals, which are manually
  charted by nursing staff at irregular intervals and are much sparser (the worked patient
  example in `docs/index.md` §6 has 7,567 ward-vital rows vs. 457 intra-op vital rows over a
  much shorter window — intra-op vitals are denser *per minute of coverage*, even though
  fewer total rows exist because surgery itself is short).

**Why this matters for interpreting any future AUROC gain from adding intra-op data:** if
adding intra-op vitals improves the model, part of that improvement is genuinely new
physiological information (Section 2.6), and part of it may simply be *better measurement
quality* — less missingness, less linear-interpolation guessing (`align_time_series()`
currently fills gaps by drawing straight lines between sparse points; there's much less
gap-filling to do when a monitor samples every minute). A fair comparison should report
the missingness/coverage rate for each window alongside the AUROC, so a reviewer can see how
much of the gain is "more information" vs. "less noisy information."

---

### 2.8 From a single risk number to a risk trajectory — time-to-event / deterioration timing

The idea raised in the session — using a **risk score regression or probability model** so
that it's possible to analyse **when** a patient started to decline, rather than just
whether they eventually died — is a well-established shift in framing called **survival
analysis** or **time-to-event modelling**, and it fits this dataset unusually well because
every measurement already carries a precise `chart_time` in minutes.

**The core reframe:**

```
Current:  P(died within 30 days) = 0.73          ← one number, computed once, pre-op
Target:   h(t) for t = 0 ... 30 days              ← a hazard/risk curve, updated as new
                                                     data arrives, showing WHEN risk rose
```

A short, honest tour of the relevant model family, roughly in order of how far each moves
away from classical statistics:

| Model | Idea | Relevant limitation |
|---|---|---|
| **Cox Proportional Hazards** (classical) | Linear model of log-hazard; assumes each covariate's effect on risk is constant over time (the "proportional hazards" assumption) | Too rigid for a physiology that changes hour to hour |
| **DeepSurv** | Replaces Cox's linear risk function with a neural network, but keeps the proportional-hazards assumption and the Cox partial-likelihood loss | On purely tabular clinical data, gains over plain Cox are often modest — it doesn't use the time series *within* a patient, only a static snapshot |
| **DeepHit** | Fully discrete-time, distribution-free — learns the probability mass function of "time to event" directly, with a ranking loss on top, and natively supports **competing risks** (more than one possible cause of the event) | Static — one prediction per patient, made once |
| **Dynamic-DeepHit** | Extends DeepHit to **longitudinal** data — re-estimates the risk distribution every time new measurements arrive, using an RNN to summarise the patient's history so far | This is the closest existing published architecture to "when did this patient start to decline" |
| **DySurv** (2024–2025, conditional-VAE based) | Combines static admission data *and* time-series data in a single multimodal dynamic model; evaluated specifically on ICU data (MIMIC-IV, eICU) and shown to **outperform APACHE and SOFA scores** — the ICU-equivalent of NELA/POSSUM for this kind of problem | Newest of the family; worth reading as the most directly comparable prior work to what this project is trying to build |

One finding from the DySurv paper is worth quoting because it is exactly the pattern this
project is asking about: in ICU-style data, **predicted survival curves show a visible drop
starting a few days before death**, rather than dropping suddenly at the moment of death —
i.e. the deterioration is detectable in the trajectory before the event, which is the
entire premise behind wanting a time-to-event model instead of a single pre-op number.

**What this looks like concretely for INSPIRE:** instead of one label
(`inhosp_death_30day`), define a *discrete-time hazard target* — did the patient die in
[day 0–1], [day 1–3], [day 3–7], [day 7–14], [day 14–30], not-yet-observed? — and train the
existing transformer encoder to output a hazard at each interval, updated every time a new
lab/vital arrives, instead of a single sigmoid at the end. **Censoring and competing risks
need explicit handling**: patients discharged alive before 30 days are *censored*, not
labelled "survived" with certainty about what happens after discharge; and the ICD-10
cause-of-instability codes already surfaced in EDA (D65 disseminated intravascular
coagulation, I46 cardiac arrest, R57 shock, J80 ARDS, K72 hepatic failure, A41 sepsis —
`docs/eda_findings.md` §4) are a natural, already-available set of **competing-risk labels**
(died-of-bleeding vs. died-of-cardiac-cause vs. died-of-sepsis), rather than treating "death"
as one undifferentiated event.

---

### 2.9 A multimodal modelling pipeline, like the figure

Putting Sections 2.5 and 2.8 together, here is a fully spelled-out version of the pipeline
in your diagram, using the actual INSPIRE modalities:

**Modality-specific encoders** (each handles a genuinely different data type correctly,
rather than forcing everything into one shape):

| Modality | Data type | How to encode it |
|---|---|---|
| Labs, ward vitals, intra-op vitals | Continuous time series, irregular sampling | The existing `TimeSeriesTransformer`, one instance per organ system (Section 2.5) |
| Medications | Categorical sequence (1,238 distinct drugs × route, timestamped) | Learned embedding table per drug/ATC code (Section 2.10), fed through a sequence model or simple time-windowed aggregation |
| Diagnoses (ICD-10) | Categorical, mostly static-ish (diagnosed at a point in time, relevant thereafter) | Learned embedding per code, or — as a validated starting point — reuse the existing HFRS-weighted sum (`frailty_hfrs.py`) as a hand-crafted feature until a learned embedding is validated against it |
| Static operative/demographic (age, sex, ASA, `emop`, department) | Fixed-size vector, one row per patient, 100% coverage | A small MLP — cheap, currently entirely unused by the DNN (Section 2.3) |

**Fusion strategies**, from simplest to most interpretable:

1. **Early concatenation** — stack all system/modality embeddings into one vector, one
   final classifier. Simplest to implement, but no more interpretable than what exists now.
2. **Gated / mixture-of-experts fusion** — a small gating network learns *how much weight*
   to give each modality's embedding per patient, and that gate is itself inspectable
   ("for this patient, 70% of the decision weight came from the renal branch") — a genuine
   step up in interpretability over (1) for very little extra complexity.
3. **Hierarchical / additive fusion** — each system produces its own interpretable
   sub-score first (closer to a differentiable POSSUM/NELA sub-component than a raw
   embedding), and a **Neural Additive Model** combines those sub-scores into the final
   risk, with each system's contribution individually plottable as its own shape function.
   This is the version that actually earns the word "clinically interpretable" rather than
   "clinically inspired" (expanded in Section 2.14C).

---

### 2.10 Learning embeddings for categorical entities — medications, diagnoses, and beyond

There are two genuinely different embedding problems hiding under one word here, and it's
worth being explicit about the difference:

1. **Embedding a continuous time series** (glucose over time, heart rate over time) — this
   is what the transformer encoder already does; the "embedding" is the pooled output of
   the sequence model.
2. **Embedding a categorical entity** (a specific drug, a specific ICD-10 code, a specific
   ATC therapeutic class) — this needs a **learned lookup table**: every distinct entity
   gets its own trainable dense vector, the same idea as word embeddings in NLP, adapted to
   clinical codes. This is genuinely new work relative to what the pipeline does today.

Two ways to train such a table, both established in the clinical-ML literature:

- **Supervised, end-to-end**: the embedding table is just another set of parameters,
  trained jointly with the mortality objective. Simple, but with only 469–942 positive
  labels in the whole dataset and potentially 1,000+ distinct drug codes, this table would
  be badly under-constrained — most codes would see too few examples to learn anything
  reliable (the same sparsity problem already flagged for the main model's `pos_weight`).
- **Unsupervised pre-training on co-occurrence**, before mortality labels are touched at
  all — e.g. two drugs that are frequently prescribed together, or a diagnosis and a drug
  that typically co-occur, end up with similar vectors, learned purely from patterns across
  all ~100k patients regardless of outcome. This is the approach taken by established
  EHR-embedding methods in the literature (e.g. the *Med2Vec*-style approach from Choi et
  al., and *RETAIN* — already cited in `docs/index.md` §16 for its two-level attention over
  clinical time series) — it makes full use of the large unlabelled cohort exactly the way
  the current autoencoder pre-training phase already does for lab time series, just applied
  to categorical codes instead of continuous ones.

**A concrete, low-risk starting point using what's already in the repo:** rather than
learning one embedding per raw drug name (1,238 distinct types, most very rare — the raw
data was already filtered to exclude drugs given to fewer than 100 patients, so the tail is
still long), start one level up the hierarchy using the **WHO ATC classification** already
included in the repo (`codes/WHO_ATC-DDD_2024-07-31.csv`) — embed at the ATC level-2 or
level-3 (therapeutic subgroup, e.g. "beta-blocking agents" rather than a specific brand),
which shrinks the vocabulary enormously and gives each embedding far more training
examples to learn from, before considering whether finer-grained drug-level embeddings are
worth the added sparsity risk.

---

### 2.11 Research papers for ICD-10 / ASA / POSSUM-style categorisation

What the repo already has and cites:

- **HFRS** (Hospital Frailty Risk Score) — Gilbert et al. 2018, *The Lancet* — already
  fully implemented in `frailty_hfrs.py`, with all 109 weighted ICD-10 codes. This is
  itself a validated example of exactly the kind of "categorise ICD-10 codes into a
  clinically meaningful score" task being asked about — worth treating as the working
  template rather than starting from nothing.
- **Shickel et al. 2023**, *Scientific Reports* (doi:10.1038/s41598-023-27418-5) — the
  closest published comparison for the overall project (56,242 patients, AUROC 0.92,
  includes MC Dropout and integrated gradients).
- **Koh et al. 2020** (ICML) — original Concept Bottleneck Models paper — already flagged
  in `docs/index.md` §14/§16.
- **Choi et al. 2016** (NeurIPS) — RETAIN, two-level attention for clinical time series —
  already cited.

New references worth adding, found in support of the sections above:

- **POSSUM**: Copeland et al. 1991 — original 12-physiology + 6-operative-factor equation;
  **Portsmouth-POSSUM (P-POSSUM)**: Prytherch et al. 1998, *British Journal of Surgery* —
  the correction for POSSUM's known tendency to overestimate mortality in low-risk
  patients.
- **ASA governance**: the current (December 2020-amended) ASA Physical Status
  Classification, as maintained by the ASA House of Delegates — useful as the primary
  source if the paper needs to cite the classification itself rather than a secondary
  description of it.
- **DeepHit**: Lee et al. 2018 — discrete-time, competing-risks survival model.
- **Dynamic-DeepHit**: Lee et al. 2019 — longitudinal extension of DeepHit.
- **DySurv**: 2024–2025 (JAMIA / arXiv) — conditional-VAE dynamic survival model,
  multimodal static + time-series, validated on MIMIC-IV and eICU against APACHE/SOFA —
  the most directly comparable recent architecture to the time-to-event reframing in
  Section 2.8.
- **Concept Bottleneck Models, recent extensions** (2024–2025): clinical-knowledge-guided
  CBMs, label-free CBMs (Oikarinen et al. 2023), and CBMs learned from mechanistic
  explanations — relevant to Section 2.14C.

**On categorising ICD-10 itself:** rather than inventing a new grouping from scratch, two
options already sit inside this project and are worth trying first, before reaching for a
novel scheme: (1) the **official ICD-10 chapter structure** (21 chapters, e.g. "Diseases of
the circulatory system," "Endocrine, nutritional and metabolic diseases") as a coarse,
already-standardised category system, and (2) the **HFRS's own curated 109-code list**
(Gilbert et al.) as a validated "high-frailty-relevance" subset, which the project already
has full weights for and can check for overlap against the mortality-linked codes already
surfaced in EDA (D65, I46, R57, J80, K72, A41 — `docs/eda_findings.md` §4).

*(Note, matching the caution already present in `docs/index.md` — the exact NELA original
citation and the exact INSPIRE dataset paper citation are both flagged there as
"needs confirming before publication." That caution should extend to the new references
listed here too — treat all of the above as strong starting points for a literature
search, not as final, verified citations to paste into a manuscript.)*

---

### 2.12 A more concrete picture of the architecture — what do the "categories" actually look like?

The diagram is easiest to reason about with a real patient plugged in. The repo already
has a fully worked example — **Subject 100033460** (`docs/index.md` §6) — an 80-year-old
woman who died 15 days after an emergency bowel operation, with chronic + acute kidney
failure, peritonitis, and colon cancer. Here's what her "system boxes" would literally
contain, if the target architecture in Section 2.5 existed today:

| System box | What would feed into it, for this patient | What the system encoder would plausibly need to output |
|---|---|---|
| **Renal** | Creatinine trajectory (up to 5.55 mg/dL pre-op, severely elevated), sodium (128 → 132, hyponatraemia), potassium (5.3, elevated), diagnosis codes N18 (chronic kidney disease, ×6) and N17 (acute kidney failure), `crrt` ward-vital flag (792 dialysis-active readings) | A `renal_embedding` that a downstream classifier would very plausibly flag as **CRITICAL** |
| **Metabolic** | Glucose spike (78 → 264 → 84 mg/dL — a classic stress-hyperglycaemia pattern around the time of surgery) | Likely **CONCERN**, transient rather than chronic |
| **Cardiovascular** | Heart rate, non-invasive blood pressure from ward vitals (908–941 readings) | Depends on values not detailed here — this is exactly the kind of gap a real system-level audit would fill in |
| **Infection/inflammation-adjacent** (not yet a named system in the current 6-group scheme) | Diagnosis K65 (peritonitis — the reason for the emergency operation) | Currently has nowhere clean to go — worth deciding whether "infection/sepsis" deserves its own system box, since D65/I46/R57/J80/K72/A41 (the most mortality-linked ICD-10 codes found in EDA) cluster around acute infective/inflammatory deterioration, not neatly under Renal/Cardiac/Respiratory alone |

This worked-through example does two useful things: it makes the abstract diagram
concrete, and it gives a ready-made **qualitative validation case** — once the
system-separated architecture exists, this patient is a natural first check: does the
renal branch actually light up as critical for her, the way the clinical narrative says it
should?

---

### 2.13 What further analysis can be done with what's already here — no new engineering required

Several of the things asked about in this session are **already partly built** in this
repo and just haven't been run at full scale or surfaced clearly in the docs yet. Worth
listing explicitly, because some of these are genuinely just "run this":

1. **`feature_selection_pipeline.py`** already does almost exactly the "feature
   selection" step requested: it (a) tests each of the 54 pre-op features for a
   died-vs-survived difference, (b) corrects for testing 54 things at once (multiple
   comparisons), (c) drops features that are redundant with each other (e.g. haemoglobin
   vs. haematocrit), and (d) fits a model that includes department, age, ASA, and
   emergency-status *alongside* each lab feature specifically to check whether a lab looks
   predictive only because sicker departments happen to order it more — i.e. it already
   distinguishes a genuinely useful feature from a department proxy. This has apparently
   only been run on a small config so far; running it at full scale is close to zero new
   engineering.
2. **`audit_features.py` and `audit_static_categorical.py`**, currently run only on the
   30-patient subset (`docs/feature_audit_findings.md`) — re-running both against the full
   99,886-patient cohort would upgrade every coverage-tier number in that document from
   "small-sample estimate" to "real."
3. **Resolving the label-definition bug** (Section 1/2.1) — the single highest-value,
   lowest-effort fix available, and it's a strict prerequisite for trusting any of the
   above.
4. **Cross-tabulating department × ICD-10 × mortality jointly** — flagged as the direct
   next step in `docs/eda_findings.md` §3, and it's exactly the analysis the ICD-10/department
   interpretability focus (Section 2 of `docs/index.md`) needs.
5. **The HFRS 2-year time-window fix** — the current `compute_hfrs()` counts a patient's
   entire diagnosis history with no limit, rather than the published Gilbert et al. 2-year
   window for patients 75+; flagged repeatedly in the repo as a known correctness gap.
6. **`score_models.py` contains a complete, working NEWS2 implementation**
   (National Early Warning Score 2) that isn't mentioned anywhere in the existing docs or
   compared against anything. NEWS2 is a fourth, genuinely independent clinical scoring
   system (vitals-based, ward-deterioration-focused, rather than pre-op-risk-focused like
   ASA/POSSUM/NELA) — worth a deliberate decision on whether to fold it into the ASA vs.
   POSSUM vs. NELA comparison table in Section 2.4, since the code to do so already exists.

---

## 3. New research ideas — additions beyond what's already documented

Organised by theme, each one chosen to be a natural extension of a question raised in this
session rather than an unrelated idea.

### A. Data and labelling

- **Competing-risks relabelling using the project's own 30-day/all-cause gap.** The 473
  patients who died *after* 30 days but are currently folder-labelled "died" are not just a
  bug to fix — they're a genuinely interesting comparison group. A model that can
  distinguish "died acutely, likely surgery-related" from "died later, likely
  surgery-unrelated" is answering a more clinically specific and more defensible question
  than a single undifferentiated "died" label.
- **Explicit distribution-shift framing for the NELA/POSSUM comparison.** NELA was built
  and validated on a UK national emergency-laparotomy population; INSPIRE is a single
  Korean centre, mixed elective/emergency, all departments. A lower NELA AUROC on INSPIRE
  is at least as likely to be a population-mismatch finding as a "NELA is a weak tool"
  finding — worth stating and, ideally, testing directly (e.g. does NELA perform closer to
  its published numbers specifically on INSPIRE's emergency-laparotomy-equivalent subset?).

### B. Modelling

- **Treat missingness as a signal, not just noise to interpolate away.** The current
  pipeline fills every gap with linear interpolation and a mask flag. A complementary
  hypothesis worth testing directly against the repo's own missingness heatmap
  (`docs/eda_findings.md` §7): a clinician's *decision not to order a test* may itself carry
  information (e.g. a test skipped because the patient looked well enough not to need it) —
  this is a missing-not-at-random hypothesis that's directly testable with the data already
  audited.
- **Move pre-training beyond plain reconstruction.** The known "0.47 for everyone" failure
  mode (`docs/index.md` §13, now partly fixed by unfreezing the encoder) exists because
  reconstruction and mortality-discrimination are different objectives. Two established
  alternatives, both compatible with the existing transformer: a masked-value objective
  (mask a lab value, predict it from surrounding context — the BERT-style approach) and a
  contrastive objective (pull same-outcome patients' embeddings closer together, push
  different-outcome patients apart) — both are known to transfer better to a downstream
  discriminative task than plain autoencoding.
- **Fully unsupervised foundation-model-style pre-training on the complete 99,886-patient
  cohort**, before any label is touched. With only 469–942 positive examples total, the
  unlabelled pool is by far the largest source of information available — worth treating
  pre-training as the primary event, not a preliminary step before the "real" supervised
  phase.

### C. Interpretability — directly addressing the stated main aim

The project's stated aim is a genuinely interpretable model, explicitly *not* one that
relies on SHAP-style post-hoc explanation. That distinction is worth taking seriously:
SHAP and similar methods (LIME, Grad-CAM-style saliency) approximate a trained black box
*after the fact* — they don't reveal what the network actually computed internally, they
fit a separate, simpler explanation model around it, and for correlated or
temporally-dependent features (exactly what this dataset has — labs that move together,
time steps that depend on each other) those approximations are known to be unstable and
sometimes misleading. The alternative is to build interpretability **into the
architecture**, so the explanation *is* the mechanism, not a story told about it
afterwards. Four complementary layers, all buildable on top of what already exists here:

1. **Concept Bottleneck Models (already flagged in `docs/index.md` §14, worth expanding).**
   Force the network to predict clinically defined, named intermediate concepts *before*
   it predicts mortality — AKI stage (0–3) from creatinine via KDIGO criteria, a
   haemodynamic-instability score from blood pressure and vasopressor use, respiratory
   failure staging from PaO2/FiO2 via Berlin criteria. The mortality prediction is then
   *read off* those named concepts, not off an opaque embedding — a genuine mechanism, not
   a rationalisation. Recent work (2024–2025) extends this specifically with
   clinically-guided and label-free variants, worth reading before implementing.
2. **Prototype / case-based reasoning.** Rather than "this patient has a 0.84 risk because
   feature X contributed +0.3," a prototype network explains a prediction as "this
   patient's renal trajectory closely resembles these 3 prior patients' trajectories, 2 of
   whom died" — an explanation format doctors already use natively (case-based clinical
   reasoning), originally developed for image classification ("This Looks Like That," Chen
   et al.) and increasingly adapted to clinical time series and medical imaging.
3. **Sparse autoencoders on the learned embeddings — a genuine mechanistic-interpretability
   step, and a natural fit here specifically because the pipeline already does autoencoder
   pre-training.** Rather than trusting the raw 8–14 dimensional embedding vector as-is,
   train a second, sparse, overcomplete autoencoder on top of it (the approach behind
   recent mechanistic-interpretability work on decomposing neural representations into
   sparse, individually-nameable "directions" rather than dense, tangled ones) — the aim is
   to end up with a larger set of *sparse* features, each of which reliably tracks one
   recognisable clinical pattern, instead of one dense vector where every dimension mixes
   several things together (the "polysemanticity" problem). This reuses machinery the
   project already has working.
4. **Attention auditing, not just attention as a mechanism.** The transformer already has
   8 attention heads — currently used only to produce predictions, never inspected. Logging
   and visualising which time steps and features each head attends to, specifically for
   correctly vs. incorrectly classified patients, and checking whether attention reliably
   concentrates around the acute deterioration diagnoses already found in EDA (D65, I46,
   R57, J80, K72, A41) is a direct, checkable test of whether the network is attending to
   clinically sensible moments — genuine evidence about internal behaviour, not a
   post-hoc plot.

**How these fit together, and why this is a stronger design than SHAP-on-top:** Concept
Bottleneck answers *what* clinical concept is driving a prediction; the sparse-autoencoder
and attention audit together answer *how* the network's internal computation actually
produces that; prototype retrieval answers *who else* looked like this before, grounding
the prediction in real precedent. Together they form three different, complementary views
into the same model, built as part of training it — not one explanation method wrapped
around a finished black box after the fact.

### D. Evaluation

- **Calibration, not just discrimination.** AUROC and AUPRC measure whether the model
  *ranks* patients correctly relative to each other — neither tells you whether a predicted
  "0.3" genuinely corresponds to a 30% chance of death. Given how rare deaths are here
  (0.3–1% depending on the label definition), calibration curves and the Brier score are
  essential additions before any clinical framing of the output as an actual probability —
  a "high AUROC, badly calibrated" model is a real and common failure mode in exactly this
  kind of imbalanced clinical setting.
- **Connect uncertainty quantification (already flagged — MC Dropout, conformal
  prediction) to the time-to-event reframing in Section 2.8**, so the output becomes not
  just "mortality ∈ [0.58, 0.84] with 90% guaranteed coverage" but a *trajectory* with
  uncertainty bands that widen or narrow as more peri-operative data arrives — directly
  useful for the "when did this patient start to decline" question, and directly gives a
  clinician a sense of how much to trust an early prediction versus a later one.

---

## 4. Consolidated, reprioritised roadmap

This merges the existing roadmap in `docs/index.md` §15 with everything raised in this
session. Numbering restarts here deliberately — this is a fresh prioritisation, not a
renumbering of the old list, though every existing item is still present (cross-referenced
in the right-hand column).

| # | Item | Why it's at this priority | Where else it's discussed |
|---|---|---|---|
| 1 | **Fix the 30-day vs. all-cause label definition** | Blocks every other number in the project — `pos_weight`, AUROC, everything | §1, §2.1; `docs/index.md` §4, §13 |
| 2 | **Run `feature_selection_pipeline.py` and both audit scripts at full 99,886-patient scale** | Already built, close to zero new engineering, upgrades every "small-sample" caveat in the existing docs | §2.13; `docs/feature_audit_findings.md` |
| 3 | **Decide the pre-op-only vs. peri-operative scope question** | Determines whether 72 intra-op parameters are even in scope for everything downstream | §2.6; `docs/index.md` §4, §15 item 5 |
| 4 | **Build the organ-system feature grouping + system-separated encoder architecture** (matches the session's diagram) | The project's core stated contribution — everything else feeds into this | §2.5, §2.12; `docs/index.md` §14, §15 item 4 |
| 5 | **HFRS 2-year time-window fix, then re-run frailty comparisons** | Known correctness gap, flagged repeatedly, currently invalidates the frailty numbers | §2.13; `docs/eda_findings.md` §10 |
| 6 | **Joint department × ICD-10 × mortality analysis** | Directly the current stated interpretability focus | §2.13; `docs/eda_findings.md` §3 |
| 7 | **Multi-operation sensitivity analysis** (last-op vs. first-op vs. exclude) | Settles an open argument with evidence instead of assumption; EDA chart already exists | §2.2; `docs/eda_findings.md` §12 |
| 8 | **Implement POSSUM (and consider NEWS2) alongside the existing ASA/NELA comparison** | Completes the clinical-baseline comparison table; NEWS2 code already exists and is unused | §2.4, §2.13 |
| 9 | **Time-to-event reframing** (Dynamic-DeepHit / DySurv-style dynamic hazard model), as a parallel track alongside the binary classifier | Directly answers the "when did decline start" question; genuinely different output type, not a replacement | §2.8 |
| 10 | **Interpretability layer**: Concept Bottleneck head + sparse autoencoder on embeddings + attention audit + prototype retrieval | The project's stated main aim; builds on machinery already in place | §2.14C |
| 11 | **Categorical embeddings for medications and diagnoses** (starting at ATC level-2/3, not raw drug name) | Feeds directly into the multimodal architecture in item 4 | §2.9, §2.10 |
| 12 | **Full-scale run and formal benchmark against Shickel et al. 2023 (0.92 AUROC)**, once items 1–4 are stable | The eventual headline comparison for the paper | `docs/index.md` §14, §16 |

Items 1–3 are genuine prerequisites for everything below them — the rest can and should
proceed in parallel, exactly as the existing roadmap in `docs/index.md` §15 already notes.

---

## 5. References

**Already cited in this repo** (`docs/index.md` §16) — kept here for completeness:

- Shickel et al. 2023, *Scientific Reports*, doi:10.1038/s41598-023-27418-5 — closest
  published benchmark for the overall task.
- Koh et al. 2020 (ICML) — Concept Bottleneck Models.
- Choi et al. 2016 (NeurIPS) — RETAIN, two-level attention for clinical time series.
- Bento et al. 2021 (KDD) — TimeSHAP.
- Gilbert et al. 2018 — Hospital Frailty Risk Score, implemented in `frailty_hfrs.py`.
- INSPIRE dataset paper (Lee et al.) — exact citation still needs confirming per the
  existing note in `docs/index.md`.

**New, added by this document:**

- Copeland et al. 1991 — original POSSUM equation (12 physiological + 6 operative factors).
- Prytherch et al. 1998, *British Journal of Surgery* — Portsmouth-POSSUM (P-POSSUM)
  correction for POSSUM's overestimation of mortality in low-risk patients.
- ASA Physical Status Classification System, as maintained by the ASA House of Delegates
  (current version amended December 2020) — primary source for the classification itself.
- Lee et al. 2018 — DeepHit: discrete-time, competing-risks deep survival model.
- Lee et al. 2019 — Dynamic-DeepHit: longitudinal extension of DeepHit.
- DySurv, 2024–2025 (*JAMIA*; also on arXiv) — conditional-VAE dynamic, multimodal
  (static + time-series) survival model, validated against APACHE/SOFA on MIMIC-IV and
  eICU — closest recent comparator for the time-to-event reframing in Section 2.8.
- Recent (2024–2025) Concept Bottleneck Model extensions: label-free CBMs (Oikarinen et
  al. 2023), clinically-guided CBMs, and CBMs derived from mechanistic explanations —
  relevant to Section 2.14C.
- Bricken et al. 2023 — sparse autoencoders for decomposing neural network
  representations into sparse, individually interpretable features (the mechanistic-
  interpretability approach referenced in Section 2.14C).
- Chen et al. — "This Looks Like That": prototype-based, case-based-reasoning
  interpretable networks, the origin of the prototype-network family referenced in
  Section 2.14C.

*As with the existing citations in this repo, the new references above are strong starting
points for a literature search and have not all been independently verified for exact
volume/page/DOI — confirm each before it goes into a manuscript, consistent with the
caution already applied to the NELA and INSPIRE citations elsewhere in this project.*