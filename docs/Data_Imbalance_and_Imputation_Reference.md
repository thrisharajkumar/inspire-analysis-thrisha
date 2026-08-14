# Data imbalance and missing-data reference — INSPIRE multimodal notebook

> Consolidates every technique discussed across this conversation (mine) and the three
> external AI responses you pasted in, with pros/cons and a concrete verdict for *this*
> architecture specifically (six per-system transformer encoders on raw sequences + one
> static/tabular branch). Confirmed numbers used throughout: **469 true 30-day deaths /
> 99,886 patients (~212:1)** — not the folder label's incorrect 942/99,886 (~105:1).

---

## Part A — Class imbalance / sampling techniques

### A.1 Techniques that touch the STATIC branch only (tabular: age, ASA, department, ICD-10 flags, etc.)

| Technique | Source | What it does | Pros | Cons | Fit for INSPIRE | Verdict |
|---|---|---|---|---|---|---|
| Random oversampling | Me | Duplicates real minority rows | Trivial, zero risk of inventing bad data | No new information; encourages memorization | Works but weak at 469 real cases | Skip — SMOTE variants below dominate it |
| Random undersampling | Me | Drops majority rows to balance | Simple; sometimes helps very large majority classes train faster | Throws away real survived-patient diversity, which the autoencoder pre-training phase needs | Bad fit — we specifically want the full survived pool for unsupervised pre-training | Skip |
| Plain SMOTE | Me / Doc 1 | Linear interpolation between two minority neighbors | Well-established, cheap | Assumes all-continuous features; corrupts one-hot/binary columns (e.g. `dept_GS = 0.6`) | Our static vector has one-hot department + binary ICD-10/exception flags → breaks | Don't use as-is |
| Borderline-SMOTE | Me | Only interpolates near the decision boundary | Focuses effort on ambiguous cases, not "obvious" deaths | Same continuous-only assumption as plain SMOTE | Same categorical-corruption problem | Skip unless combined with SMOTENC's categorical handling |
| ADASYN | Me | Like SMOTE, generates more synthetic points for *hard* minority cases | Adaptive to difficulty | Same categorical-corruption problem; less predictable than SMOTE | Same issue | Skip for now |
| **SMOTENC** | Doc 1 | SMOTE variant that explicitly declares which columns are categorical, uses mode/median instead of interpolation for them | Fixes the exact problem above | Needs you to pass exact column indices; still needs enough minority rows to find k neighbors | **Correct SMOTE variant for our static branch** (one-hot dept + binary flags) | **Adopt** |
| SMOTEN | Doc 1 | SMOTENC's all-categorical sibling | Right tool if every feature were categorical | Our static vector also has continuous fields (age, HFRS, op-recency days) — not a full match | Not quite our shape | Skip (SMOTENC is the right one, not SMOTEN) |
| SMOTE-Tomek | Doc 1 | SMOTE, then deletes majority-class patient from any "Tomek link" pair (opposite-class nearest neighbors) | Widens the decision margin, cheap cleanup step | Removes real survived patients — small effect at low synthesis rates, larger at high ones | Good paired with SMOTENC | **Adopt**, as a cleanup step after SMOTENC |
| SMOTE-ENN | Doc 1 | SMOTE, then deletes any point whose 3 nearest neighbors mismatch its class | More aggressive cleanup than Tomek | Can delete a lot of data; more aggressive than needed for our modest target ratio | Not necessary if Tomek is enough | Optional / skip initially |
| **Grouped (stratified) SMOTENC + Tomek** | Doc 3, code included | Split minority cohort into clinical strata (department × ASA), run SMOTENC *within* each stratum only, then a global Tomek cleanup | This is the "clinically constrained" version — never blends a cardiac patient with an orthopedic one | Many strata will have <3 minority cases and get skipped entirely → actual achieved ratio likely falls short of the target, needs checking empirically. The reference code's `X_res.iloc[len(X_g):]` trick for finding synthetic rows relies on unstable ordering — track synthetic rows by an explicit tag instead | **This is the best-fit version for our data** | **Adopt, with the indexing fix** |
| Class weighting (`pos_weight` / `scale_pos_weight`) | Me / Doc 1 | Leave data untouched; scale the loss so the rare class counts more | No synthetic data, no data loss, already implemented in our notebook | Doesn't help if the model never sees *enough distinct* examples, only under-weighted ones | Always safe, always worth using | **Adopt — combine with sampling, don't replace it** |
| Focal loss | Me | Down-weights already-easy examples, focuses gradient on hard ones | Complements class weighting | One more hyperparameter (γ) to tune | Optional refinement | Optional |
| Threshold tuning off the PR curve | Doc 1 | Pick the operating probability cutoff (not 0.5) from the precision-recall trade-off | Free, no retraining needed, directly clinically actionable | Needs a clinician-agreed cost trade-off (missed death vs. false alarm) to pick the actual number | Directly usable once the model is trained | **Adopt** |

### A.2 Techniques for the SEQUENCE branch (the raw per-timestep organ-system time series)

| Technique | Source | What it does | Pros | Cons | Fit for INSPIRE | Verdict |
|---|---|---|---|---|---|---|
| **Window slicing / cropping** | Doc 3 | Extract random sub-intervals from a real minority patient's trajectory | Zero invented values — it's real data, just re-windowed; trivial to implement | Only works if the trajectory is long enough to sub-sample meaningfully | Intra-op vitals (many points) — good fit; sparse pre-op labs — poor fit (too few points to slice) | **Adopt for intra-op/peri-op windows** |
| **Magnitude jittering** (Gaussian noise, σ ≈ 0.05×std) | Doc 3 | Adds small random noise to real values | Simple, well-established (originates in wearable-sensor augmentation literature, Um et al. 2017) | Can't fix sparsity — a patient with 2 real points still only has 2 (noisy) points | Reasonable regularizer regardless of sparsity | **Adopt** |
| **DTW Barycentric Averaging (DBA)** | Doc 3 | Averages two real minority trajectories in "aligned time" space rather than raw time — the sequence-level equivalent of SMOTE | Doesn't require fixed-interval sampling in principle; respects trajectory *shape* | Real algorithmic complexity to implement correctly on irregularly-sampled, already-imputed sequences; needs the underlying `tslearn`-style DBA implementation, not something to hand-roll quickly | Worth it, but the highest-effort item on this list | **Adopt later**, after A.1 is validated — not a first pass |
| **Embedding-space SMOTE** (SMOTE after passing sequences through a *pre-trained* encoder) | Doc 3 | Run SMOTE on the pooled embeddings our autoencoder pre-training (Part 9.4) already produces, not raw values | Uses the model's own learned representation, where interpolation is more likely to stay meaningful | Only valid **after** pre-training, not before (an untrained encoder's embeddings are random and meaningless to interpolate) — Doc 3's own text is ambiguous on this and should be read as "self-supervised, i.e. post-pretraining" only | Directly compatible with our two-phase pretrain→finetune design | **Adopt as a later experiment**, gated on pre-training being done first |
| TimeGAN | Me / Doc 2 / Doc 3 | GAN trained specifically to generate realistic multi-channel time series | Most expressive option for trajectories specifically; Doc 3 gives a concrete recipe (train only on first 120 min of intra-op vitals for deceased patients) | Doc 3's own threshold: needs 1,500-2,000+ real minority examples to avoid mode collapse — we have 469. **Below the usable threshold right now** | Not yet viable at our sample size | **Skip until the death count grows** (unlikely without a bigger cohort — flag as long-term, not this project phase) |
| Conditional VAE | Me | Generative model trained on real minority examples, samples new plausible ones | More expressive than SMOTE in principle | Same sub-1,000-example instability Doc 2 flags for TimeGAN applies here too | Same sample-size problem as TimeGAN | Skip for now |
| medGAN / EHR-GAN | Me | GAN built for mixed continuous+categorical EHR rows | State-of-the-art for tabular EHR specifically | Same instability at small N; heavier infrastructure to validate | Same sample-size problem | Skip for now |
| TabDDPM / tabular diffusion | Me / Doc 2 | Diffusion-based tabular data synthesis | Newest, strong benchmark results on tabular data | Doc 2 correctly notes: only sensible on the *engineered static matrix*, not raw relational tables; least mature tooling for our specific shape | Same instability concerns as GANs/VAEs at 469 examples | Skip for now |

**On the "how much to synthesize" question, both external sources converge:** Doc 2 mentions no synthetic-to-real ratio directly; Doc 3 gives a concrete range — **1:10 to 1:4** (positive:negative), i.e. 4,700-25,000 synthetic positives from 469 real ones, but its own "information saturation beyond ~20x" argument actually argues against its own 1:4 upper bound (that's ~53x). **Recommended target: 1:10** (≈4,700 synthetic positives) — the conservative end of their range, consistent with everything discussed in this thread.

---

## Part B — Missing data / imputation techniques (already in the notebook's theory, restated here since you asked for "filling values" too)

| Technique | What it does | Pros | Cons | Fit for INSPIRE | Verdict |
|---|---|---|---|---|---|
| Median/mode fill | Replace every gap with the training-set median/mode | Fast, safe baseline | Ignores time — a 2am value and a 2pm value get the same fill | Fine for static features (age missing → median age) | **Adopt for static features** |
| Population mean/stats fill | Same idea, used when a patient has *zero* observations of a feature at all | Only option when there's nothing to interpolate from | Same as above | Needed as the fallback for entirely-missing time series per patient | **Adopt as fallback** |
| Linear interpolation, fading to population mean outside the observed range | Straight line between real observed points; fades toward the average the further from any real data | Good for values that plausibly change smoothly (heart rate, blood pressure) | Poor for values that jump (a single abnormal lab spike gets smoothed away) | Matches the source repo's existing `smooth_fade_to_mean_interpolator` approach | **Adopt for fast-changing vitals** |
| Forward-fill (LOCF) | Repeat the last real value forward | Simple, defensible for slow-changing labs | Poor for genuinely fast-changing values | Good for creatinine/albumin-type labs | **Adopt for slow-changing labs** |
| KNN imputation | Fill a gap using similar *other patients'* values for that feature | Useful when a patient has almost no data of their own | Computationally heavier; needs a sensible distance metric across mixed feature types | Reasonable alternative, not the default | Optional, available as a switchable strategy |
| MICE / `IterativeImputer` | Iteratively predicts each missing feature from all the others | The statistically principled choice under "missing at random" | Needs enough rows to fit stable per-feature models; expensive at small N | Same small-N caution as SMOTE — not reliable yet at 469 positive examples, more so at full cohort | Available, not default yet |
| **Missingness mask feature** (always applied, regardless of fill method) | A second 0/1 feature: was this value real or filled in? | Lets the model recover signal from *why* something is missing (e.g. a clinician chose not to order a test because the patient looked fine) — this is a real, not hypothetical, MNAR mechanism in EHR data | Doubles the feature count | Already implemented throughout the notebook | **Always keep this — never drop it, regardless of which fill method above is chosen** |

**Order of operations that avoids leaking test data into training statistics (this matters for every method above, and for sampling):**
```
1. Split into train / val / test (stratified) FIRST
2. Compute imputation statistics (medians, MICE model, KNN neighbors) from TRAIN ONLY
3. Impute train, val, test using those train-only statistics
4. Standardize (z-score) using TRAIN-ONLY mean/std
5. THEN apply sampling (SMOTENC + Tomek) — to the TRAIN split only, never val/test
```
Steps 1-4 are already correct in the current notebook (Parts 7-8). Sampling is the next
piece to update to match this reference.

---

## Part C — The recommended combined pipeline for INSPIRE, in order

1. **Impute** using the notebook's existing decision-tree strategy (interpolate-and-fade
   for fast-changing vitals, forward-fill for slow-changing labs, population mean
   fallback for zero-observation features) — fit on train only.
2. **Keep the missingness mask** for every value, no exceptions.
3. **Standardize** (z-score, train-fit).
4. **Enrich the static branch** with the aggregated-time-series features flagged in the
   second document — mean/min/max/std per key lab and vital, time spent in severe
   hypotension (MAP < 65), cumulative high-alert drug doses, count of vasopressor
   escalations. This also gives SMOTENC more to work with.
5. **Split** (stratified, train/val/test) — before any sampling.
6. **On the static branch only**: grouped SMOTENC (stratified by department × ASA) +
   Tomek-link cleanup, targeting **1:10** (≈4,700 synthetic positives from 469 real, at
   full scale) — with the synthetic-row-tracking fix from Part A.1.
7. **On the sequence branch, for real minority patients only** (no change to survived
   patients): window-slicing/cropping for intra-op vitals, magnitude jittering
   everywhere. DBA and embedding-space SMOTE deferred to a later pass.
8. **Loss function**: `pos_weight`/`scale_pos_weight` combined with the above — sampling
   takes the edge off severe imbalance, loss-weighting finishes the job. Don't rely on
   either alone.
9. **Evaluate primarily on PR-AUC**, report ROC-AUC alongside for reference, and pick the
   decision threshold from the precision-recall curve against an agreed clinical cost
   trade-off — not the default 0.5.
10. **Deferred, not now**: TimeGAN, conditional VAE, medGAN, TabDDPM — all correctly
    flagged by the external sources as needing more real minority examples (~1,500-2,000+)
    than the confirmed 469 we currently have. Revisit only if the cohort or label
    definition changes to give more positive examples.
