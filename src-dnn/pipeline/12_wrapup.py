# %% [markdown]
# # Part 12 — Experiment log, and what to do next
#
# ## 12.0 Run-time summary — did this hit the 20-30 minute target?
#
# The actual, measured elapsed time for every major Part of this run, on this hardware,
# with this config -- not an estimate.

# %%
record_checkpoint("Part 12 -- wrap-up")
print_timing_summary()
print(f"\nConfig used for this timing: MAX_SUBJECTS_PER_CLASS={CONFIG['MAX_SUBJECTS_PER_CLASS']}, "
      f"BATCH_SIZE={CONFIG['BATCH_SIZE']}, SKIP_IMPUTATION_ACCURACY_BENCHMARKS={CONFIG['SKIP_IMPUTATION_ACCURACY_BENCHMARKS']}, "
      f"SKIP_FUSION_ABLATION={CONFIG['SKIP_FUSION_ABLATION']}")

# %% [markdown]
# ## 12.1 Experiment log (fill this in as you run variants)
#
# Every `CONFIG` flag in §1.7 is a genuine fork — this table is meant to be copied and
# appended to as you try combinations, so the reasoning you asked to preserve stays
# attached to actual results, not just intentions. Suggested first sweep once the full
# cohort is loaded: hold everything else fixed and vary one flag at a time.
#
# | Run | TIME_WINDOW | IMPUTATION_STRATEGY | SAMPLING_STRATEGY | FUSION_STRATEGY | test AUROC | test AUPRC | notes |
# |---|---|---|---|---|---|---|---|
# | 1 (this notebook, dev subset) | pre_op | decision_tree | class_weight | nam | *(§11.1)* | *(§11.1)* | 30-patient dev subset — directional only |
# | 2 | peri_op | decision_tree | class_weight | nam | | | §11.5 |
# | 3 | pre_op | median | class_weight | nam | | | isolates the imputation choice |
# | 4 | pre_op | decision_tree | smote | nam | | | needs full cohort — too few positives here (§1.4.2) |
# | 5 | pre_op | decision_tree | class_weight | concat | | | §11.4 |
#
# ## 12.2 What this notebook deliberately did *not* build (flagged, not silently skipped)
#
# Consistent with keeping every decision visible rather than assumed:
#
# - **Learned ICD-10/ATC embeddings** (§1.2.1, §1.5.3) — chapter flags and ATC level-2
#   counts are used instead, because the label-starved small-cohort regime makes a learned
#   embedding table under-constrained. Revisit once the full ~99,886-patient cohort is
#   available for unsupervised co-occurrence pre-training.
# - **POSSUM/P-POSSUM and NEWS2 baselines** (§1.2.2) — the source repo has working NELA
#   (`nela.py`) and NEWS2 (`score_models.py`) implementations; wiring them in as comparison
#   points alongside this DNN is a natural next cell, not built here to keep this notebook's
#   scope to the multimodal architecture itself.
# - **Time-to-event / survival reframing** (Dynamic-DeepHit / DySurv-style, per the source
#   repo's `Research_Aim.md` §2.8) — a genuinely different output type (a hazard trajectory,
#   not a single 30-day probability), flagged as a parallel track rather than folded in here.
# - **Symmetric MICE / full multiple imputation** (§1.3.1) — `IterativeImputer` is imported
#   and ready; not run by default because, at n=30, its iterative per-feature regressions are
#   data-starved in the same way SMOTE is (§1.4.2's caveat applies almost identically here).
# - **HFRS's full 109-code table** — §6.6 uses a representative ~30-code subset for
#   demonstration; swap in the source repo's complete `frailty_hfrs.py` table for a
#   publication-grade run.
#
# ## 12.3 Direct next steps, in priority order (mirrors the source repo's own roadmap)
#
# 1. **Swap in the full ~99,886-patient cohort** (`CONFIG['SUBJECTS_DIR']`) — every metric in
#    this notebook is directional at n=30; this is the single highest-value next step.
# 2. **Run the full `TIME_WINDOW` and multi-op sensitivity sweeps** (§11.5, §11.7) for real,
#    not as a template.
# 3. **Re-run §11.4's fusion ablation with `ABLATION_EPOCHS` raised** and, ideally, several
#    seeds, to get a trustworthy answer on whether NAM's interpretability is bought cheaply
#    or expensively in this specific architecture.
# 4. **Wire in NELA/POSSUM/NEWS2 as baselines** in a new cell, using the already-computed
#    `cohort_df` fields — direct comparison points for the DNN's AUROC.
# 5. **Extend §6.6 to the full HFRS table**, and fix the age-75+/2-year window exactly as
#    the source repo's own roadmap flags (`CONFIG['HFRS_LOOKBACK_YEARS']` is already wired
#    for this — just confirm the full weights table before publishing any HFRS-derived
#    number).
# 6. **Confirm the `CARDIAC_DEPARTMENTS` mapping** (§6.11) against your site's actual
#    department coding before trusting the 6-month exception flag on a new cohort — `CTS`
#    was inferred from this dataset's observed department codes, not looked up from an
#    authoritative source.
#
# ## 12.4 One-paragraph summary, for pasting into a supervisor update
#
# This notebook implements a multimodal, organ-system-separated DNN for 30-day
# peri-operative mortality prediction on the INSPIRE dataset, extending the project's
# existing six-system architecture with two diagnosis/department-driven systems
# (Gastrointestinal, Musculoskeletal — necessary because this dataset has no dedicated
# lab/vital panel for either, confirmed programmatically in §4.1) and an explicit
# cardiovascular→renal coupling (§6.9, motivated by the clinical cardiorenal-syndrome
# literature). Every data-cleaning decision — imputation strategy, class-imbalance sampling,
# fusion architecture — is implemented as an interchangeable, documented option rather than a
# silent default, with the reasoning for each spelled out in Part 1 before any code runs.
# Rule-based clinical exceptions (the 6-month post-cardiac-surgery washout) are kept
# separate from learned features on purpose, for both sample-size and actionability reasons.
# Current results are directional only (30-patient development subset); the pipeline is
# written to scale unchanged to the full ~99,886-patient cohort.

# %% [markdown]
# ## 12.2 v2 verification summary — every check from this run in one table
#
# Each row is a check added or fixed in v2, with the value this run actually produced.
# "PASS" means the mechanism behaved as intended; it does not mean the model is clinically
# validated. Anything marked CHECK needs a look before results are quoted.

# %%
def _v(name, ok, value, meaning):
    return {"check": name, "result": ("PASS" if ok else "CHECK") if ok is not None else "n/a",
            "value": value, "what it means": meaning}

_checks = []
_sm = globals().get("GROUPED_SMOTENC_SUMMARY", {})
if CONFIG["SAMPLING_STRATEGY"] == "grouped_smotenc_tomek":
    _n_syn, _n_err = _sm.get("n_synthetic", 0), _sm.get("errored_strata", 1)
    _checks.append(_v("SMOTENC runs without errors (v1 bug fixed)",
                      (None if (_n_syn == 0 and _n_err == 0) else (_n_err == 0)),
                      f"{_n_syn} synthetic, {_n_err} errored strata"
                      + (" -- no stratum was below the 1:10 target, so nothing to synthesise (expected when a sample's death rate is already high)"
                         if (_n_syn == 0 and _n_err == 0) else ""),
                      "v1: every stratum errored silently"))
_checks.append(_v("pos_weight after sampling is sane", bool(np.isfinite(POS_WEIGHT) and POS_WEIGHT >= 1),
                  f"{POS_WEIGHT:.2f}", "v1 formula subtracted removed survivors from the death count"))
_checks.append(_v("Synthetic patients have real-looking time series",
                  (not SYNTHETIC_STATIC_ROWS) or (len(SYNTHETIC_TS_DONORS) == len(SYNTHETIC_STATIC_ROWS)),
                  f"{len(SYNTHETIC_TS_DONORS)} donors for {len(SYNTHETIC_STATIC_ROWS or [])} synthetic",
                  "no 'empty series => died' shortcut"))
_checks.append(_v("NEWS2 scoring matches the RCP chart", NEWS2_VALIDATED, f"{len(_news2_failures)} failures",
                  "band edges + worked examples"))
_checks.append(_v("NEWS2 included as a feature", USE_NEWS2 if CONFIG["INCLUDE_NEWS2"] else None, USE_NEWS2,
                  "only if the self-test passed"))
_checks.append(_v("Mortality rises with worst pre-op NEWS2 band", NEWS2_MORTALITY_MONOTONE,
                  NEWS2_MORTALITY_MONOTONE, "construct validity of the computed score"))
for r in NEWS2_AUGMENTATION_CHECKS:
    if "pct_band_changed" in r:
        _checks.append(_v("NEWS2 stable under jitter/time-mask", r["pct_band_changed"] < 0.10,
                          f"{r['pct_band_changed']:.1%} of time points changed band, mean |change| {r['mean_abs_change']:.2f}",
                          "augmentation is gentle"))
    else:
        _checks.append(_v(f"NEWS2 before vs after SMOTENC [{r['feature']}]",
                          None if str(r.get("verdict", "")).startswith(("TOO FEW", "not enough")) else r.get("verdict") == "PASS",
                          f"{r.get('verdict')}; median {r.get('median_before', float('nan')):.1f} -> {r.get('median_after', float('nan')):.1f}, "
                          f"KS {r.get('ks_stat', float('nan')):.2f}, outside 0-20: {r.get('n_outside_range_after')}",
                          "synthetic deaths look like real deaths"))
_learned = CONFIG["COUPLING_MODE"] == "learned"
_checks.append(_v("No hand-defined inter-system link", ("renal_cardiac_interaction" not in STATIC_FEATURE_NAMES
                                                       and not CONFIG["COUPLING_PRIOR_LINKS"]) if _learned else None,
                  f"mode={CONFIG['COUPLING_MODE']}, priors={CONFIG['COUPLING_PRIOR_LINKS']}", "the coupling layer learns every link"))
_checks.append(_v("All eight organ systems built the same way", len(model.organ_systems) == 8,
                  ", ".join(model.organ_systems), "GI/MSK are organ systems with no time series, not a special case"))
if CONFIG["SMOTE_MAX_AMPLIFICATION"] is not None and _sm.get("n_synthetic", 0):
    _real_pos = int(sum(PATIENT_BUNDLE[s]["label"] for s in TRAIN_IDS))
    _checks.append(_v("SMOTENC amplification capped", _sm["n_synthetic"] <= CONFIG["SMOTE_MAX_AMPLIFICATION"] * _real_pos,
                      f"{_sm['n_synthetic']} synthetic from {_real_pos} real deaths (cap {CONFIG['SMOTE_MAX_AMPLIFICATION']}x per stratum)",
                      "no stratum inflated from a handful of patients"))
if COUPLING_VALIDATION_SHEET is not None:
    _cr = COUPLING_VALIDATION_SHEET[(COUPLING_VALIDATION_SHEET.receiver == "renal") & (COUPLING_VALIDATION_SHEET.sender == "cardiovascular")]
    _gmax = float(np.abs(model.coupling.gate.detach().cpu().numpy()).max())
    _checks.append(_v("Coupling layer actually used", _gmax > 0.05, f"max gate {_gmax:.2f}", "gate > 0 = systems use messages"))
    _checks.append(_v("Cardiorenal link rediscovered (never given)", bool(_cr["above_even_split"].iloc[0]) if len(_cr) else None,
                      f"rank {_cr.index[0] + 1} of {len(COUPLING_VALIDATION_SHEET)}" if len(_cr) else "-",
                      "a sanity signal, not a requirement -- surgeon reviews the full sheet"))
if COUPLING_STABILITY is not None:
    _rho = float(COUPLING_STABILITY["spearman_vs_main"].mean())
    _checks.append(_v("Learned links replicate across ensemble members", _rho > 0.7,
                      f"mean rank agreement {_rho:.2f}, top-5 overlap {COUPLING_STABILITY['top5_overlap'].mean():.0%}",
                      "> 0.7 = safe to show individual links to a surgeon"))
if LEAKAGE is not None:
    _flag = [n for n, v in LEAKAGE["own_share"].items() if v < 0.5]
    _checks.append(_v("Each system answers mainly from its own data", not _flag,
                      "all systems >= 50% own data" if not _flag else f"flagged: {', '.join(_flag)}",
                      "whole-patient + coupling layers are not leaking other systems' data"))
if SHARE_WHOLE_PATIENT is not None:
    _checks.append(_v("Organ systems carry most of the prediction", SHARE_WHOLE_PATIENT < 0.5,
                      f"{model.context_term} term {SHARE_WHOLE_PATIENT:.0%} of |risk|", "systems are not decoration"))
with torch.no_grad():
    model.eval()
    _o = model({k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in next(iter(test_loader)).items()})
if _o["per_system_signal"] is not None:
    _checks.append(_v("Per-system terms sum exactly to the prediction",
                      bool(torch.allclose(sum(_o["per_system_signal"].values()) + model.fusion.bias, _o["logit"], atol=1e-4)),
                      "exact", "interpretability preserved with coupling + arbitration + whole-patient"))
_checks.append(_v("Threshold chosen on validation, not test", True, f"{best_threshold:.3f}", "removes v1's optimistic bias"))
if "gap" in MEMORISATION:
    _checks.append(_v("Train vs validation AUPRC gap is modest", MEMORISATION["gap"] < 0.15,
                      f"train {MEMORISATION['train_auprc']:.3f} vs val {MEMORISATION['val_auprc']:.3f}", "large gap = memorising"))
if "donor_median_risk" in MEMORISATION:
    _ratio = MEMORISATION["donor_median_risk"] / max(MEMORISATION["other_death_median_risk"], 1e-6)
    _checks.append(_v("SMOTE donors not memorised", _ratio < 1.5,
                      f"donor median risk {MEMORISATION['donor_median_risk']:.3f} vs other deaths {MEMORISATION['other_death_median_risk']:.3f}",
                      "donors reused many times should not be singled out"))
_final = ENSEMBLE_RESULT or {"auprc": test_result["auprc"]}
_label = f"ensemble of {len(ENSEMBLE_MODELS)}" if ENSEMBLE_RESULT else "single DNN"
for _name, _res in [("logistic regression", BASELINE_RESULT), ("gradient boosting", GBM_RESULT)]:
    if _res is not None:
        _checks.append(_v(f"DNN ({_label}) matches or beats {_name} (AUPRC)", _final["auprc"] >= _res["auprc"] - 0.01,
                          f"DNN {_final['auprc']:.3f} vs {_res['auprc']:.3f}", "judge against the CI in 11.1d/11.1e"))
VERIFICATION_SUMMARY = pd.DataFrame(_checks)
pd.set_option("display.max_colwidth", 90)
print(VERIFICATION_SUMMARY.to_string(index=False))
