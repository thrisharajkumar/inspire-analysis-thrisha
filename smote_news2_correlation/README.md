# smote_news2_correlation

Three additions to the existing SMOTE-based notebook (`INSPIRE_Multimodal_Mortality_Benchmark`),
built and tested as standalone modules so they drop in without restructuring your notebook.

## 1. `src/improved_smotenc.py` -- SMOTE kept, a real bug fixed, real improvements added

**The bug (confirmed live, not theoretical):** your notebook's `grouped_smotenc`
(Section 8.2) passes a boolean numpy mask directly as SMOTENC's `categorical_features`
argument. Tested against your actual installed `imbalanced-learn` (0.14.2): this raises
`ValueError: The truth value of an array with more than one element is ambiguous`. Because
that error is caught by a bare `except ValueError` (the same one used for legitimate
"stratum too small" skips), **every stratum silently fails this way and your
grouped_smotenc generates zero synthetic patients, every run** -- indistinguishable in
the printed log from a genuine skip.

**The fix:** convert the boolean mask to integer indices before calling SMOTENC.
Verified working end-to-end on synthetic test data (33 real synthetic patients
generated from a 500-patient test cohort, vs. zero before the fix).

**The improvements, beyond just the fix:**
- A `min_synthetic_neighbor_ratio` guard -- skips a stratum rather than trusting
  SMOTENC's output when it was forced to use an unreliably small neighborhood.
- A returned `diagnostics_df` -- one row per stratum, with exactly why it synthesized
  or skipped, so you can inspect and share this rather than reading it out of
  scrollback.
- A genuine SMOTENC error (rare, real data issue) now prints a distinctly different
  message than "too few positives," so a real future problem doesn't hide in the
  same log line as an expected skip.

## 2. `src/news2.py` -- NEWS2 score

Standard UK clinical deterioration score (Royal College of Physicians, 2017), computed
from parameters your pipeline already collects (respiratory rate, SpO2, oxygen use,
temperature, systolic BP, heart rate, consciousness level) -- no new data needed.

Every scoring band tested against the official RCP table, including exact boundary
values (SBP 90 vs. 91, HR 130 vs. 131, temp 39.0 vs. 39.1, etc.) -- all pass. Two usage
modes:
- `news2_series_for_patient()` -- NEWS2 at every timepoint, for a time-varying feature.
- `news2_summary_features()` -- max/latest/mean/trend-slope, for your static feature
  table (same treatment as your existing §6.15 aggregated summaries).

**One documented approximation** (flagged for your surgeon, see the review doc): NEWS2
normally uses AVPU for consciousness, not GCS -- this substitutes GCS=15 for "Alert"
and anything below for the most severe band. Worth clinical confirmation.

## 3. `src/system_correlation.py` -- learned system correlation layer

Your existing architecture hand-codes exactly one cross-system link (cardiovascular
summary stats fed into the renal branch, encoding cardiorenal syndrome). This module
generalizes that into a learned correlation across ALL organ systems via multi-head
attention -- it doesn't remove your existing hand-coded link, it adds a mechanism that
can discover others too (e.g. respiratory<->cardiovascular), and reports exactly which
systems it found move together, so you can sanity-check that against known physiology.

Tested with a real forward pass (correct output shapes, `[B, n_systems, embed_dim]` in
and out) and the surgeon-facing reporting function (`top_correlations`, a plain list of
`{from, to, weight}` pairs -- the thing to actually check against clinical expectation
once run on real data).

## 4. `docs/Surgeon_Feature_Review.md`

Every feature currently used, organized by organ system, in plain clinical language --
ready to send as-is. Includes two explicit open questions (the fluid-volume placement,
and the NEWS2 AVPU-vs-GCS substitution) and one thing specifically worth a surgeon's
eye once run on real data (whether the correlation layer's top findings look clinically
sane).

## How to use these with your existing notebook

None of these three modules require restructuring `INSPIRE_Multimodal_Mortality_Benchmark`:

```python
# Replace the existing grouped_smotenc/tomek_cleanup calls with:
from improved_smotenc import grouped_smotenc, tomek_cleanup
synthetic_rows, diagnostics_df = grouped_smotenc(
    _static_matrix, COHORT_INDEXED, labels_by_id, train_ids,
    CATEGORICAL_STATIC_MASK, target_ratio=..., min_stratum_minority=..., strata_cols=...,
)

# Add NEWS2 as a static feature, per patient, during your existing static feature build:
from news2 import news2_summary_features
news2_feats = news2_summary_features(patient_long_df)  # merge into STATIC_FEATURE_NAMES

# Add the correlation layer in MortalityModel.forward(), after your per-system
# SystemEncoder outputs are stacked into [B, n_systems, embed_dim]:
from system_correlation import SystemCorrelationLayer
self.correlation_layer = SystemCorrelationLayer(embed_dim=..., n_systems=len(TIME_SERIES_SYSTEMS))
# ...
correlated, report = self.correlation_layer(system_embeddings, system_names=TIME_SERIES_SYSTEMS)
# feed `correlated` into your existing fusion step in place of the raw stacked embeddings
```

Every function above has been run against real test data in this session -- not just
written and assumed correct.
