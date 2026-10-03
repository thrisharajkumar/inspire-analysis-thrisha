# %% [markdown]
# ## 11.4 Ablation: does each piece earn its place? (v3)
#
# Same data, same split, same sampling — one architecture switch changes per row, so any
# difference can be attributed. The full v3 row reuses the main model (no retraining).
# Read differences against the ensemble spread from 11.1e: a gap smaller than the
# seed-to-seed spread is not a real difference.

# %%
def _test_scores(m):
    r = evaluate(m, test_loader)
    return {"test_auprc": r["auprc"], "test_auroc": r["auroc"], "n_params": sum(p.numel() for p in m.parameters())}

ABLATION_VARIANTS = [
    ("organ systems only (no coupling / arbitration / whole-patient)",
     dict(coupling_mode="none", use_arbitration=False, use_whole_patient=False), None),
    ("+ learned coupling", dict(coupling_mode="learned", use_arbitration=False, use_whole_patient=False), None),
    ("+ arbitration", dict(coupling_mode="learned", use_arbitration=True, use_whole_patient=False), None),
    ("full v3, separate encoder per system", dict(share_encoder=False), None),
    ("full v3, no masked-system pre-training", dict(), {"MASKED_SYSTEM_WEIGHT": 0.0}),
    ("concat fusion (no per-system breakdown)", dict(fusion_strategy="concat", use_arbitration=False), None),
    # v4 -- the two model-changing upgrades, each tested by taking it away:
    ("v4 without the surgical-context term", dict(use_surgical_context=False), None),
    ("v4 surgical context NOT exclusive (whole-patient also reads it)", dict(), {"SURGICAL_CONTEXT_EXCLUSIVE": False}),
    ("6 organ systems (GI and MSK removed -> their features go to whole-patient)", dict(drop_systems=("gi", "msk")), None),
]
if CONFIG["SKIP_FUSION_ABLATION"]:
    print(f"CONFIG['SKIP_FUSION_ABLATION']=True -- skipping the ablation ({len(ABLATION_VARIANTS)} retrainings).")
    ablation_df = pd.DataFrame(columns=["test_auprc", "test_auroc", "n_params"])
else:
    rows = [{"variant": "full v3 (this notebook's model)", **_test_scores(model), "minutes": 0.0}]
    for vname, kw, cfg_patch in ABLATION_VARIANTS:
        _saved = {k: CONFIG[k] for k in (cfg_patch or {})}
        CONFIG.update(cfg_patch or {})
        _t0 = time.time()
        try:
            m_v, _ = train_full_model(seed=SEED, **kw)
            rows.append({"variant": vname, **_test_scores(m_v), "minutes": (time.time() - _t0) / 60})
            del m_v
        finally:
            CONFIG.update(_saved)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    ablation_df = pd.DataFrame(rows).set_index("variant")
    print("\n" + ablation_df.round(3).to_string())
ablation_df

# %% [markdown]
# ### 11.4b Do the learned links replicate? (coupling stability across the ensemble)
#
# Each ensemble member learned its own "who listens to whom" matrix. If the pattern comes
# from the data, the members agree; if it comes from the random start, they don't. Rank
# correlation over all off-diagonal links, plus top-5 overlap, versus the main model.
# > 0.7: individual links are safe to show a surgeon; 0.4-0.7: top links only; < 0.4: not yet.

# %%
@torch.no_grad()
def mean_coupling_matrix(m, ids):
    out = collect_system_outputs(m, ids)
    if out["weights"] is None:
        return None
    return out["weights"].mean(axis=0) * np.abs(m.coupling.gate.detach().cpu().numpy())[:, None]

COUPLING_STABILITY = None
if CONFIG["COUPLING_MODE"] == "learned" and len(ENSEMBLE_MODELS) > 1:
    from scipy.stats import spearmanr
    _off = ~np.eye(len(model.organ_systems), dtype=bool)
    mats = [mean_coupling_matrix(m, _eval_ids) for m in ENSEMBLE_MODELS]
    _top = lambda M: set(np.argsort(-M[_off])[:5])
    COUPLING_STABILITY = pd.DataFrame([{
        "member": k, "spearman_vs_main": spearmanr(mats[0][_off], mats[k][_off]).correlation,
        "top5_overlap": len(_top(mats[0]) & _top(mats[k])) / 5} for k in range(1, len(mats))])
    print(COUPLING_STABILITY.round(2).to_string(index=False))
    _rho = COUPLING_STABILITY["spearman_vs_main"].mean()
    print(f"\nMean rank agreement with the main model: {_rho:.2f}")
    ENSEMBLE_COUPLING = np.mean(mats, axis=0)   # the averaged map: the most stable picture to show
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(mats[0].max(), ENSEMBLE_COUPLING.max())
    for ax, M, t in [(axes[0], mats[0], "main model"), (axes[1], ENSEMBLE_COUPLING, f"average of {len(mats)} models")]:
        im = ax.imshow(np.where(_off, M, np.nan), cmap="Purples", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(_names))); ax.set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(_names))); ax.set_yticklabels(_names, fontsize=8); ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("this system's reading..."); axes[1].set_xlabel("...takes information FROM")
    plt.suptitle("Effective inter-system coupling (attention x gate)", fontsize=11)
    plt.tight_layout(); plt.show()
else:
    print("Coupling stability needs a learned coupling layer and N_ENSEMBLE >= 2.")

# %% [markdown]
# ### 11.4c Learning curve — is the model starved of data, or memorising? (v3)
#
# Retrains on 25% and 50% of the real training patients (department-stratified, same
# validation and test sets) and compares with 100%. Validation AUPRC still climbing steeply
# at 100% means more data would help (expected — this is the case for the full cohort);
# a flat curve means the architecture, not the data, is the limit. The subsets use real
# patients + their jittered copies only (no SMOTENC), so they sit slightly below a
# like-for-like line; the shape is what matters.

# %%
LEARNING_CURVE = None
if CONFIG["RUN_LEARNING_CURVE"]:
    _real = [s for s in EFFECTIVE_TRAIN_IDS]
    _lab = np.array([PATIENT_BUNDLE[s]["label"] for s in _real])
    rows = []
    for frac in (0.25, 0.5):
        sub, _ = train_test_split(_real, train_size=frac, stratify=_lab, random_state=SEED)
        ds_sub = INSPIREDataset(sub, None, build_augment_items(sub, CONFIG["USE_SEQUENCE_AUGMENTATION"],
                                                               CONFIG["SEQUENCE_AUGMENTATION_COPIES"]))
        _t0 = time.time()
        m_f, best_val = train_full_model(seed=SEED, train_ds=ds_sub)
        rows.append({"train_fraction": frac, "n_train": len(sub), "train_deaths": int(sum(PATIENT_BUNDLE[s]["label"] for s in sub)),
                     "val_auprc": best_val, "test_auprc": evaluate(m_f, test_loader)["auprc"], "minutes": (time.time() - _t0) / 60})
        del m_f
    rows.append({"train_fraction": 1.0, "n_train": len(_real), "train_deaths": int(_lab.sum()),
                 "val_auprc": best_val_auroc, "test_auprc": test_result["auprc"], "minutes": np.nan})
    LEARNING_CURVE = pd.DataFrame(rows)
    print(LEARNING_CURVE.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(LEARNING_CURVE.n_train, LEARNING_CURVE.val_auprc, "o-", label="validation AUPRC", color="#8e44ad")
    ax.plot(LEARNING_CURVE.n_train, LEARNING_CURVE.test_auprc, "s--", label="test AUPRC", color="#0a7d6e")
    ax.set_xlabel("training patients"); ax.set_ylabel("AUPRC"); ax.legend(); ax.set_title("Learning curve")
    plt.tight_layout(); plt.show()
else:
    print("RUN_LEARNING_CURVE=False -- skipped.")

# %% [markdown]
# ### 11.4d Memorisation checks (v3)
#
# Two direct tests. (1) **Train vs. validation gap** at the chosen epoch: a large gap means
# the model fits training patients far better than new ones. (2) **Donor check**: real deaths
# whose time series were borrowed by synthetic patients (8.2c) are seen many more times in
# training. If the model scores them much higher than other real training deaths, it has
# memorised those individuals.

# %%
MEMORISATION = {}
_best_epoch = int(np.nanargmax(history["val_auprc"])) if history["val_auprc"] else None
if _best_epoch is not None and history.get("train_auprc"):
    MEMORISATION["train_auprc"] = float(history["train_auprc"][_best_epoch])
    MEMORISATION["val_auprc"] = float(history["val_auprc"][_best_epoch])
    MEMORISATION["gap"] = MEMORISATION["train_auprc"] - MEMORISATION["val_auprc"]
    print(f"At the selected epoch ({_best_epoch}): train AUPRC {MEMORISATION['train_auprc']:.3f} vs "
          f"validation {MEMORISATION['val_auprc']:.3f} -> gap {MEMORISATION['gap']:+.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history["train_auprc"], label="train (real patients)", color="#c0392b")
    ax.plot(history["val_auprc"], label="validation", color="#2980b9")
    ax.axvline(_best_epoch, color="gray", ls=":", label="selected epoch")
    ax.set_xlabel("fine-tuning epoch"); ax.set_ylabel("AUPRC"); ax.legend(); ax.set_title("Train vs. validation")
    plt.tight_layout(); plt.show()
_donors = sorted(set(SYNTHETIC_TS_DONORS))
_train_deaths = [s for s in EFFECTIVE_TRAIN_IDS if PATIENT_BUNDLE[s]["label"] == 1.0]
_others = [s for s in _train_deaths if s not in set(_donors)]
if _donors and _others:
    _pd = evaluate(model, DataLoader(INSPIREDataset(_donors), batch_size=256, collate_fn=collate_bundle))["probs"]
    _po = evaluate(model, DataLoader(INSPIREDataset(_others), batch_size=256, collate_fn=collate_bundle))["probs"]
    MEMORISATION.update(donor_median_risk=float(np.median(_pd)), other_death_median_risk=float(np.median(_po)),
                        n_donors=len(_donors))
    print(f"Donor check: median predicted risk {np.median(_pd):.3f} for {len(_donors)} donor deaths vs "
          f"{np.median(_po):.3f} for {len(_others)} other real training deaths.")
else:
    print("Donor check skipped (no synthetic donors this run).")

# %% [markdown]
# ## 11.5 Sensitivity: pre-op-only vs. peri-operative (§1.6.3, Path B)
#
# The full re-run this needs (re-window every patient with `TIME_WINDOW='peri_op'`, rebuild
# §6-§10 end to end) is expensive to repeat inline for every cell execution, so this section
# is a **template you run deliberately** rather than something that fires automatically:
# change `CONFIG['TIME_WINDOW']` to `'peri_op'` in Part 2, then **Run All** again, and record
# the resulting test AUROC/AUPRC/coverage numbers in the comparison table below by hand (or
# extend the automation from §11.4 to also loop over `TIME_WINDOW`, which is the natural next
# step once this notebook runs on a machine fast enough to afford two full passes).

# %% [markdown]
# | Run | TIME_WINDOW | test AUROC | test AUPRC | mean cardiovascular coverage (§5.2-style) | notes |
# |---|---|---|---|---|---|
# | 1 | `pre_op` | *(fill in from §11.1)* | | | Decision-support framing (§1.6.3) |
# | 2 | `peri_op` | *(re-run and fill in)* | | | Real-time monitoring framing (§1.6.3) |
#
# Per §1.6.3's own reasoning: a higher `peri_op` AUROC is expected partly from genuinely new
# physiological information and partly from better measurement *density* (intra-op vitals are
# less sparse than ward vitals per §2.7 of the source repo's `Research_Aim.md`) — report the
# coverage numbers from §5.2 alongside AUROC for either run so a reader can tell the two
# effects apart, exactly as that section recommends.

# %% [markdown]
# ## 11.6 Cardiac-recovery-exception qualitative check (§1.6.4)
#
# Does the model's own cardiovascular-system contribution (from the NAM breakdown, §11.2)
# shift for patients flagged by the rule-based §6.11 exception, versus similar unflagged
# patients? A genuine qualitative validation in the spirit of the source repo's own
# worked-patient sanity check (Subject 100033460 in its docs) — small-N here, but the
# comparison is structured to scale directly once more flagged patients exist in the full
# cohort.

# %%
@torch.no_grad()
def cardiac_contribution_by_exception_flag(model, all_ids):
    rows = []
    ds = INSPIREDataset(all_ids)
    loader_all = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_bundle)
    model.eval()
    for batch in loader_all:
        batch_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_dev)
        if out["per_system_signal"] is None:
            return None
        for i, sid in enumerate(batch["subject_id"]):
            rows.append({
                "subject_id": sid,
                "cardiac_recovery_exception": CARDIAC_EXCEPTION_FLAG[sid],
                "cardiovascular_contribution": float(out["per_system_signal"]["cardiovascular"][i]),
                "true_label": float(batch["label"][i]),
            })
    return pd.DataFrame(rows)

exc_df = cardiac_contribution_by_exception_flag(model, ALL_IDS)
if exc_df is not None:
    summary = exc_df.groupby("cardiac_recovery_exception")["cardiovascular_contribution"].agg(["mean", "std", "count"])
    print(summary)
    if (exc_df["cardiac_recovery_exception"] == 1).sum() == 0:
        print("\nNo patients in this dev subset are flagged by the §6.11 rule -- this comparison "
              "is a no-op here. Re-run once the full cohort (with more CTS re-interventions "
              "within the 6-month window) is loaded, where this check becomes informative.")

# %% [markdown]
# ## 11.7 Multi-operation label sensitivity (§1.6.4, Path C — last-op vs. first-op)
#
# §4.2 already computed `died_30day_from_first_op` alongside the training label
# (`died_30day_from_last_op`). A full re-run under the first-op definition needs the same
# "change config, Run All" treatment as §11.5 for a genuine model comparison — but the
# **label-agreement** check itself doesn't need retraining and is worth doing right here.

# %%
disagreement = cohort_df[cohort_df["died_30day_from_last_op"] != cohort_df["died_30day_from_first_op"]]
print(f"Patients where last-op and first-op 30-day labels disagree: {len(disagreement)} / {len(cohort_df)}")
if len(disagreement):
    print(disagreement[["subject_id", "n_operations", "died_30day_from_last_op", "died_30day_from_first_op"]])
print("\nA non-trivial disagreement rate here is exactly the evidence §1.6.4/Path C says to "
      "collect before picking one definition -- if the disagreement rate is high and "
      "concentrated among multi-operation patients specifically, that's a strong argument "
      "for running (not assuming) the full three-way sensitivity analysis the source repo's "
      "own roadmap already calls for.")

# %% [markdown]
# ## 11.8 Risk trajectory over time -- watching risk evolve as more data arrives
#
# Everything in Part 11 so far scores a patient using their **complete** pre-op window. This
# section instead asks the trained model the same question at **several earlier cutoff
# points** for one patient -- re-extracting and re-imputing only the time-series inputs up to
# each cutoff (the static features -- demographics, ICD-10, operation history -- are held at
# their full-window values for this demo, a simplification worth knowing about if this is
# presented further: a fully rigorous version would also restrict static features like
# "diagnoses so far" to each cutoff). This is Option 1 from the three timeline-architecture
# options discussed earlier in this project ("repeated snapshot") -- the cheapest of the
# three to build, since it reuses the already-trained model rather than requiring a new
# hazard-curve architecture (Option 2) or a separate deterioration-trend detector (Option 3).
#
# **What this demonstrates:** the model's risk estimate for a patient as a function of how
# much of their pre-op window has been observed -- a rising curve as a real deterioration
# enters the data is the qualitative pattern worth showing; the exact numbers, like
# everywhere else in this notebook at this sample size, are illustrative rather than
# validated.

# %%
def predict_risk_at_cutoff(sid, cutoff_time, model, device=DEVICE):
    # Re-extracts and re-imputes only the time-series inputs for one patient, using
    # [lo, cutoff_time] instead of the patient's full pre-op window -- static features are
    # held at their full-window values (see the markdown note above). Reuses the exact
    # imputation/standardisation statistics already fit on the training split, so this is
    # a fair comparison against the model's normal predictions, not a different pipeline.
    row = COHORT_INDEXED.loc[sid]
    full_lo, full_hi = get_window(sid, row)
    lo = full_lo
    hi = min(cutoff_time, full_hi)   # never look beyond what the model was normally given

    batch = {}
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[sid]["ts"][system]["feature_names"]
        if not fnames:
            raw = np.zeros((CONFIG["TARGET_SEQ_LEN"], 0))
            mask = np.zeros((CONFIG["TARGET_SEQ_LEN"], 0))
        else:
            raw, mask, _ = extract_system_tensor(sid, system, lo, hi)
            grid = np.linspace(lo, hi, CONFIG["TARGET_SEQ_LEN"])
            fast_idx = [j for j, f in enumerate(fnames) if f in FAST_CHANGING_ITEMS]
            slow_idx = [j for j, f in enumerate(fnames) if f not in FAST_CHANGING_ITEMS]
            imputed = raw.copy()
            if fast_idx:
                imputed[:, fast_idx] = impute_interpolate(raw[:, fast_idx], [fnames[j] for j in fast_idx], system, STATS, grid)
            if slow_idx:
                imputed[:, slow_idx] = impute_forward_fill(raw[:, slow_idx], [fnames[j] for j in slow_idx], system, STATS)
            for j, fname in enumerate(fnames):
                mean, std = STATS[(system, fname)]["mean"], TRAIN_STD[(system, fname)]
                imputed[:, j] = (imputed[:, j] - mean) / std
            raw, mask = imputed, mask
        batch[f"{system}_x"] = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(device)
        batch[f"{system}_mask"] = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

    static_vec = IMPUTED_BUNDLE[sid]["static"].values.astype(float)   # held at full-window values, see note above
    batch["static"] = torch.tensor(static_vec, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(batch)
    prob = float(torch.sigmoid(out["logit"])[0])
    per_system = {k: float(v[0]) for k, v in out["per_system_signal"].items()} if out["per_system_signal"] is not None else None
    return prob, per_system

def plot_risk_trajectory(sid, n_cutoffs=6):
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    cutoffs = np.linspace(lo + (hi - lo) * 0.15, hi, n_cutoffs)   # skip the very first sliver -- too little data to mean anything
    probs = []
    for c in cutoffs:
        prob, _ = predict_risk_at_cutoff(sid, c, model)
        probs.append(prob)

    true_label = "died" if PATIENT_BUNDLE[sid]["label"] == 1.0 else "survived"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours_before_surgery = (cutoffs - hi) / 60.0
    ax.plot(hours_before_surgery, probs, "o-", color="#c0392b" if true_label == "died" else "#2980b9")
    ax.set_xlabel("hours before surgery (0 = end of pre-op window)")
    ax.set_ylabel("predicted 30-day mortality risk")
    ax.set_title(f"Patient {sid} (true outcome: {true_label}) -- risk as more pre-op data arrives")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return cutoffs, probs

# Pick one real patient who died, to show the pattern this is meant to illustrate.
_died_ids = [sid for sid in ALL_IDS if PATIENT_BUNDLE[sid]["label"] == 1.0]
_example_sid = _died_ids[0] if _died_ids else ALL_IDS[0]
print(f"Plotting risk trajectory for patient {_example_sid}")
_cutoffs, _probs = plot_risk_trajectory(_example_sid)
print(f"Risk at each cutoff: {[f'{p:.2f}' for p in _probs]}")
print(f"\nNOTE: the model behind this trajectory was trained on {len(TRAIN_IDS)} real patients "
      f"({int(sum(PATIENT_BUNDLE[s]['label'] for s in TRAIN_IDS))} deaths) -- it shows what the mechanism "
      f"produces; treat it as illustrative until repeated on the full cohort.")

# %% [markdown]
# ### Reading this plot for the meeting
#
# This is the mechanism you told James you'd present -- a working demonstration that risk can
# be read out **as a function of how much data has arrived**, not just as one end-of-window
# number. Two honest framings for presenting it:
#
# - **What's real and presentable:** the pipeline can genuinely produce a risk-over-time
#   curve from partial data, using the same trained model, with no architecture change
#   needed. That's the actual engineering result.
# - **What isn't validated yet:** the specific shape of the curve above, and whether it
#   reliably rises before a real deterioration, needs the full cohort to test properly --
#   at 18 training patients the network hasn't seen enough real trajectories to have learned
#   a trustworthy notion of "getting worse over time" yet.

# %% [markdown]
# ## 11.9 The clinical story -- risk trajectory alongside the measurements driving it
#
# Section 11.8 showed risk rising over time. This section adds the "why" underneath it: the
# actual raw measurements (not standardized, not imputed -- the real observed values) for
# whichever organ systems the model's own per-system breakdown (§11.2) says contributed most
# to this patient's prediction. The two contributing systems shown are chosen dynamically,
# per patient, from the model's own NAM contributions -- this isn't a hand-picked example,
# it's read directly from what the model itself flagged as driving the prediction.

# %%
HEADLINE_FEATURE_PER_SYSTEM = {
    "renal": "creatinine", "cardiovascular": "hr", "respiratory": "spo2",
    "metabolic_hepatic": "glucose", "haematology": "hb", "neurological": "gcs_e",
}

def plot_patient_clinical_story(sid, n_cutoffs=6):
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    cutoffs = np.linspace(lo + (hi - lo) * 0.15, hi, n_cutoffs)

    probs, per_system_at_end = [], None
    for i, c in enumerate(cutoffs):
        prob, per_sys = predict_risk_at_cutoff(sid, c, model)
        probs.append(prob)
        if i == len(cutoffs) - 1:
            per_system_at_end = per_sys

    sys_contribs = {k: v for k, v in (per_system_at_end or {}).items() if k in TIME_SERIES_SYSTEMS}   # v2: gi/msk/static have no time series to plot
    top_systems = sorted(sys_contribs, key=lambda k: -abs(sys_contribs[k]))[:2]
    top_systems = [s for s in top_systems if PATIENT_BUNDLE[sid]["ts"][s]["feature_names"]]   # only systems with real features to plot

    true_label = "died" if PATIENT_BUNDLE[sid]["label"] == 1.0 else "survived"
    n_panels = 1 + len(top_systems)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    hours_before = (cutoffs - hi) / 60.0

    axes[0].plot(hours_before, probs, "o-", color="#c0392b" if true_label == "died" else "#2980b9", linewidth=2)
    axes[0].set_ylabel("predicted\nmortality risk")
    axes[0].set_title(f"Patient {sid} (true outcome: {true_label}) -- risk and the measurements driving it")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)

    for ax_idx, system in enumerate(top_systems, start=1):
        fnames = PATIENT_BUNDLE[sid]["ts"][system]["feature_names"]
        feature = HEADLINE_FEATURE_PER_SYSTEM.get(system)
        if feature not in fnames:
            feature = fnames[0]   # fall back to whatever this system actually has for this patient
        s1 = windowed_series(labs_df, sid, feature, lo, hi)
        s2 = windowed_series(ward_vitals_df, sid, feature, lo, hi)
        series = {**s1, **s2}
        ax = axes[ax_idx]
        contrib = sys_contribs.get(system, 0.0)
        if series:
            times = sorted(series.keys())
            vals = [series[t] for t in times]
            hrs = [(t - hi) / 60.0 for t in times]
            ax.plot(hrs, vals, "o-", color="#8e44ad")
        else:
            ax.text(0.5, 0.5, "no observed values in this window", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_ylabel(f"{system}\n({feature})")
        ax.set_title(f"NAM contribution: {contrib:+.2f} ({'pushes toward died' if contrib > 0 else 'pushes toward survived'})", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("hours before surgery (0 = end of pre-op window)")
    plt.tight_layout()
    plt.show()
    return top_systems, sys_contribs

print(f"Clinical story for patient {_example_sid} (reusing §11.8's example patient):")
_top_sys, _contribs = plot_patient_clinical_story(_example_sid)
print(f"Top contributing systems for this patient: {_top_sys}")
print(f"Full per-system contribution at the final cutoff: {_contribs}")

# %% [markdown]
# ### Reading this for the meeting
#
# Top panel: the same risk curve as §11.8. Panels below it: the *actual* clinical
# measurements for the one or two organ systems the model's own fusion layer says drove that
# prediction, over the same time axis -- so a rise in predicted risk can be pointed at
# directly against a real, observed change in a real measurement, not just asserted. This is
# the version of "these were the measurements and this was where risk was rising" your
# notebook can now produce directly from the trained model, for any patient, not just this
# one example -- call `plot_patient_clinical_story(<subject_id>)` for any patient in
# `ALL_IDS` to generate the same figure for someone else.

# %% [markdown]
# ## 11.10 One unified plot: real readings and the risk line, on the same axis
#
# Section 11.9 put the risk curve and the measurements in separate stacked panels. This is
# the design actually requested: **one chart**, with a black risk line drawn *through* the
# patient's real readings over time -- so you can see directly whether the risk line moves
# together with a real clinical change, not just alongside it in a different panel.
#
# **The one real design problem this has to solve, stated plainly:** heart rate lives in the
# 60-110 range, creatinine lives in the 0.5-1.5 range, and risk is a probability from 0 to 1
# -- plotting them on the same raw axis would make the smaller-scale ones invisible. The fix:
# every measurement trace is **rescaled to 0-1 using the population's own observed range**
# for that feature (not this one patient's range, which would make different patients'
# plots incomparable to each other) -- so "near the top of its line" consistently means
# "near the highest values seen across the training population," for every trace, on every
# patient's plot. Risk is already a 0-1 probability, so it needs no rescaling at all -- it's
# the one line already honestly on the shared axis by construction, which is part of why it
# reads as the "anchor" line the others are compared against.
#
# **Honest scope note:** this currently spans the pre-op window only (or pre-op + surgery
# if `CONFIG['TIME_WINDOW']='peri_op'`) -- extending it into true post-operative recovery
# data is not built yet (see the Option B / post-op discussion) and would need the data
# window itself extended past `orout_time`, not just a change to this plotting function.

# %%
def compute_population_range(system, feature_idx, patient_ids):
    vals = []
    for sid in patient_ids:
        raw = PATIENT_BUNDLE[sid]["ts"][system]["raw"]
        mask = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
        obs = raw[:, feature_idx][mask[:, feature_idx] == 1]
        vals.extend(obs.tolist())
    if not vals:
        return None, None
    return float(np.min(vals)), float(np.max(vals))

def plot_unified_risk_overlay(sid, n_cutoffs=6, max_traces=3):
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    cutoffs = np.linspace(lo + (hi - lo) * 0.15, hi, n_cutoffs)

    probs, per_system_at_end = [], None
    for i, c in enumerate(cutoffs):
        prob, per_sys = predict_risk_at_cutoff(sid, c, model)
        probs.append(prob)
        if i == len(cutoffs) - 1:
            per_system_at_end = per_sys

    sys_contribs = {k: v for k, v in (per_system_at_end or {}).items() if k in TIME_SERIES_SYSTEMS}   # v2: gi/msk/static have no time series to plot
    top_systems = sorted(sys_contribs, key=lambda k: -abs(sys_contribs[k]))[:max_traces]
    top_systems = [s for s in top_systems if PATIENT_BUNDLE[sid]["ts"][s]["feature_names"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    hours_before = (cutoffs - hi) / 60.0
    ax.plot(hours_before, probs, "-", color="black", linewidth=3, zorder=10, label="predicted mortality risk (0-1)")
    ax.scatter(hours_before, probs, color="black", zorder=11, s=45)

    trace_colors = ["#c0392b", "#2980b9", "#27ae60"]
    for color, system in zip(trace_colors, top_systems):
        fnames = PATIENT_BUNDLE[sid]["ts"][system]["feature_names"]
        feature = HEADLINE_FEATURE_PER_SYSTEM.get(system)
        if feature not in fnames:
            feature = fnames[0]
        f_idx = fnames.index(feature)
        vmin, vmax = compute_population_range(system, f_idx, TRAIN_IDS_FOR_STATS)

        s1 = windowed_series(labs_df, sid, feature, lo, hi)
        s2 = windowed_series(ward_vitals_df, sid, feature, lo, hi)
        series = {**s1, **s2}
        if series and vmin is not None and vmax != vmin:
            times = sorted(series.keys())
            raw_vals = [series[t] for t in times]
            norm_vals = [float(np.clip((v - vmin) / (vmax - vmin), 0, 1)) for v in raw_vals]
            hrs = [(t - hi) / 60.0 for t in times]
            contrib = sys_contribs.get(system, 0.0)
            ax.plot(hrs, norm_vals, "o--", color=color, alpha=0.85,
                     label=f"{system} ({feature}) -- contribution {contrib:+.2f}")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("hours before surgery (0 = end of pre-op window)")
    ax.set_ylabel("risk (black, solid) / normalised measurement (coloured, dashed)")
    true_label = "died" if PATIENT_BUNDLE[sid]["label"] == 1.0 else "survived"
    ax.set_title(f"Patient {sid} (true outcome: {true_label}) -- risk overlaid on its own driving measurements")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_unified_risk_overlay(_example_sid)

# %%
record_checkpoint("Part 11 -- evaluation, interpretability, ablations")
