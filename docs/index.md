# INSPIRE DNN Mortality Pipeline

Predicting 30-day post-surgical mortality from clinical time series with a two-phase
transformer, on the INSPIRE Korean perioperative dataset — structured so a prediction
decomposes into per-organ-system contributions rather than one opaque number.

## Start here

- **[Latest Work & Results](current/PACO_Net_Latest_Work_and_Results.md)** — current
  architecture (phase-aware timeline, learned inter-organ coupling, diffusion-based
  augmentation), the full feature mapping, and results confirmed to date. This is the
  one doc to read first.
- **[Surgeon-Facing Summary](current/Surgeon_Facing_Summary.md)** — plain-language
  summary for clinical review, no ML background assumed.

## Everything else

Planning history, EDA detail, and past meeting/results notes have moved to the archive,
so the docs stay focused on current work:

- **Archive – Planning & Reference** — research aims, roadmap, clinician Q&A, feature
  audits, imputation reference
- **Archive – Results & History** — EDA findings, baseline pipeline docs, past meeting
  summaries, the original (superseded) PACO-Net design

Use the tabs above to navigate either section.
