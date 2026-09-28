# %% [markdown]
# # Part 11 — Evaluation, calibration, and interpretability
#
# - §11.1 Held-out test metrics (AUROC, AUPRC, Brier score, confusion matrix)
# - §11.2 Per-system NAM contribution breakdown — the actual "Renal: CRITICAL, Cardio: OK"
#   decomposition promised in §1.1 and §1.5.2
# - §11.3 Attention audit — do heads attend to sensible time steps?
# - §11.4 Ablation: NAM vs. concat fusion — what does the additive structure cost/buy?
# - §11.5 Sensitivity: pre-op-only vs. peri-operative (§1.6.3, Path B)
# - §11.6 Cardiac-recovery-exception qualitative check (§1.6.4)
# - §11.7 Multi-operation sensitivity: last-op vs. first-op label (§1.6.4, Path C)
#
# All caveats about small-N noise from Parts 8–10 apply throughout — read directions and
# relative comparisons, not exact numbers, until this runs on the full cohort.

# %% [markdown]
# ## 11.1 Held-out test metrics
#
# **AUPRC (PR-AUC) is the metric to trust first, not AUROC** — per Part C step 9 of the
# imbalance/imputation reference. At the true full-cohort prevalence (~0.47%, 469/99,886),
# AUROC can look deceptively good while precision stays poor, because AUROC's false-positive
# rate is measured against an enormous majority class — the same failure mode both external
# reviews independently flagged. AUROC is still reported below for reference and comparison
# to published benchmarks (e.g. Shickel et al.'s 0.92), but AUPRC is the number to actually
# judge this model by.

# %%
def bootstrap_ci(labels, probs, metric, n=None, seed=SEED):
    # 95% percentile interval, resampling test patients with replacement (v3). With ~94
    # test deaths the interval is wide -- that width IS the honest uncertainty.
    n = n or CONFIG["N_BOOTSTRAP"]
    rng = np.random.default_rng(seed); vals = []
    labels, probs = np.asarray(labels), np.asarray(probs)
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) > 1:
            vals.append(metric(labels[idx], probs[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

test_result = evaluate(model, test_loader)
TEST_CI = {}
if len(np.unique(test_result["labels"])) > 1:
    TEST_CI = {"auprc": bootstrap_ci(test_result["labels"], test_result["probs"], average_precision_score),
               "auroc": bootstrap_ci(test_result["labels"], test_result["probs"], roc_auc_score)}
print(f"TEST  AUPRC={test_result['auprc']:.3f} (95% CI {TEST_CI.get('auprc', (np.nan,)*2)[0]:.3f}-{TEST_CI.get('auprc', (np.nan,)*2)[1]:.3f})  "
      f"AUROC={test_result['auroc']:.3f} (95% CI {TEST_CI.get('auroc', (np.nan,)*2)[0]:.3f}-{TEST_CI.get('auroc', (np.nan,)*2)[1]:.3f})  "
      f"(n={len(test_result['labels'])}, deaths={int(test_result['labels'].sum())}; chance AUPRC = {test_result['labels'].mean():.3f})")

if len(np.unique(test_result["labels"])) > 1:
    brier = brier_score_loss(test_result["labels"], test_result["probs"])
    print(f"Brier score: {brier:.3f}  (0 = perfect calibration, 0.25 = a coin flip's worth of miscalibration)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    fpr, tpr, _ = roc_curve(test_result["labels"], test_result["probs"])
    axes[0].plot(fpr, tpr, color="#2980b9"); axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"ROC (AUROC={test_result['auroc']:.3f})"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

    prec, rec, _ = precision_recall_curve(test_result["labels"], test_result["probs"])
    axes[1].plot(rec, prec, color="#c0392b"); axes[1].set_title(f"PR (AUPRC={test_result['auprc']:.3f})")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")

    # Calibration curve (§1 Part D: "calibration, not just discrimination")
    bins = np.linspace(0, 1, 6)
    bin_ids = np.digitize(test_result["probs"], bins) - 1
    bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
    obs_rate, pred_rate = [], []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.sum() > 0:
            obs_rate.append(test_result["labels"][mask].mean())
            pred_rate.append(test_result["probs"][mask].mean())
    axes[2].plot(pred_rate, obs_rate, "o-", color="#8e44ad", label="model")
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axes[2].set_title("Calibration"); axes[2].set_xlabel("predicted probability"); axes[2].set_ylabel("observed rate"); axes[2].legend()
    plt.tight_layout(); plt.show()
else:
    print("Only one class present in the test split at this sample size -- AUROC/calibration "
          "undefined this run. Re-run with a different SEED or, better, the full cohort.")

cm_default = confusion_matrix(test_result["labels"], (test_result["probs"] >= 0.5).astype(int))
print("\nConfusion matrix @ threshold 0.5 (rows=true, cols=pred) [[TN, FP], [FN, TP]]:")
print(cm_default)
print("0.5 is an arbitrary default threshold -- see §11.1b below for a threshold actually "
      "chosen from the PR curve, per Part C step 9.")

# %% [markdown]
# ## 11.1b Choosing an operating threshold — on validation, applied to test (v2)
#
# 0.5 is meaningless as a cutoff at this prevalence. The threshold that maximises F1 is a
# reasonable default, **but it has to be chosen on the validation set and then applied to
# the test set.** v1 chose it on the test set itself, which makes the reported
# "76% of deaths caught at 60% precision" slightly optimistic (the threshold was fitted to
# the very patients it was scored on). The test-optimal threshold is still printed, for
# reference only, so the size of that optimism is visible.

# %%
def best_f1_threshold(labels, probs):
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    i = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return (float(thr[i]) if len(thr) else 0.5), prec, rec, thr, i

def metrics_at(labels, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {"threshold": threshold, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "deaths_caught": f"{tp}/{tp + fn}", "false_alarms": fp}

val_result = evaluate(model, val_loader)
if len(np.unique(test_result["labels"])) > 1 and len(np.unique(val_result["labels"])) > 1:
    best_threshold, *_ = best_f1_threshold(val_result["labels"], val_result["probs"])
    test_opt_threshold, prec_arr, rec_arr, thresh_arr, best_idx = best_f1_threshold(test_result["labels"], test_result["probs"])
    threshold_table = pd.DataFrame([
        {"chosen on": "validation (honest)", **metrics_at(test_result["labels"], test_result["probs"], best_threshold)},
        {"chosen on": "test itself (v1 method, optimistic)", **metrics_at(test_result["labels"], test_result["probs"], test_opt_threshold)},
    ]).set_index("chosen on")
    print("Test-set performance at the best-F1 threshold:")
    print(threshold_table)
    cm_tuned = confusion_matrix(test_result["labels"], (test_result["probs"] >= best_threshold).astype(int))
    print(f"\nConfusion matrix on TEST @ validation-chosen threshold {best_threshold:.3f} [[TN, FP], [FN, TP]]:")
    print(cm_tuned)
    print("\nNearby thresholds on the test PR curve, for choosing a clinical trade-off by hand "
          "(then fix it on validation data before quoting it):")
    for i in range(max(0, best_idx - 3), min(len(thresh_arr), best_idx + 4)):
        print(f"  threshold={thresh_arr[i]:.3f}  precision={prec_arr[i]:.2f}  recall={rec_arr[i]:.2f}")
else:
    best_threshold = 0.5
    print("Only one class present in the validation or test split at this sample size -- skipping threshold tuning.")

# %% [markdown]
# ## 11.1c Recalibration on the validation set (v2)
#
# SMOTENC, sequence augmentation and `pos_weight` all push the model's raw probabilities
# upward on purpose (they make deaths "louder" during training), which is why the raw
# Brier score is poor even when ranking (AUROC/AUPRC) is good. Platt scaling — a one-feature
# logistic regression on the model's logit, fitted on the **validation** set — maps the
# scores back to real-world probabilities. It is monotonic, so AUROC and AUPRC are
# unchanged; only calibration (Brier, the calibration curve) moves.

# %%
from sklearn.linear_model import LogisticRegression

def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

CALIBRATOR = None
if CONFIG["RECALIBRATE_ON_VAL"] and len(np.unique(val_result["labels"])) > 1 and len(np.unique(test_result["labels"])) > 1:
    CALIBRATOR = LogisticRegression(C=1e6).fit(_logit(val_result["probs"]).reshape(-1, 1), val_result["labels"])
    test_probs_calibrated = CALIBRATOR.predict_proba(_logit(test_result["probs"]).reshape(-1, 1))[:, 1]
    _b_raw = brier_score_loss(test_result["labels"], test_result["probs"])
    _b_cal = brier_score_loss(test_result["labels"], test_probs_calibrated)
    print(f"Test Brier score: raw {_b_raw:.4f} -> recalibrated {_b_cal:.4f}  "
          f"(test prevalence {test_result['labels'].mean():.2%}; mean predicted risk raw "
          f"{test_result['probs'].mean():.2%} -> recalibrated {test_probs_calibrated.mean():.2%})")
    fig, ax = plt.subplots(figsize=(5, 4))
    for probs, name, color in [(test_result["probs"], "raw", "#8e44ad"), (test_probs_calibrated, "recalibrated", "#27ae60")]:
        q = np.unique(np.quantile(probs, np.linspace(0, 1, 11)))
        ids = np.clip(np.digitize(probs, q[1:-1]), 0, len(q) - 2)
        pts = [(probs[ids == b].mean(), test_result["labels"][ids == b].mean()) for b in range(len(q) - 1) if (ids == b).any()]
        ax.plot(*zip(*pts), "o-", color=color, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("predicted risk (decile bins)"); ax.set_ylabel("observed death rate"); ax.legend(); ax.set_title("Calibration, test set")
    plt.tight_layout(); plt.show()
else:
    print("Recalibration skipped (flag off, or a split has only one class).")

# %% [markdown]
# ## 11.1d A simple baseline on the same data (v2)
#
# Listed as an open limitation in the surgeon-facing summary: the DNN had not been
# compared with a simple model. This fits a class-weighted logistic regression on the same
# standardised static vector plus each time-series feature's mean and "ever observed"
# flag — same training patients (real only), same test patients. If the DNN does not beat
# this clearly, the extra complexity has not earned its place yet; if it does, that is the
# number to quote alongside it.

# %%
def _flat_features(ids):
    rows = []
    for sid in ids:
        parts = [IMPUTED_BUNDLE[sid]["static"].values.astype(float)]
        for system in TIME_SERIES_SYSTEMS:
            x = IMPUTED_BUNDLE[sid][system]
            if x.shape[1]:
                parts += [x.mean(axis=0), PATIENT_BUNDLE[sid]["ts"][system]["mask"].max(axis=0)]
        rows.append(np.concatenate(parts))
    return np.stack(rows)

BASELINE_RESULT, GBM_RESULT = None, None
if CONFIG["RUN_SIMPLE_BASELINE"] and len(np.unique(test_result["labels"])) > 1:
    _train_real = [s for s in TRAIN_IDS]
    _ytr = np.array([PATIENT_BUNDLE[s]["label"] for s in _train_real])
    _lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(_flat_features(_train_real), _ytr)
    _p = _lr.predict_proba(_flat_features(TEST_IDS))[:, 1]
    _yte = np.array([PATIENT_BUNDLE[s]["label"] for s in TEST_IDS])
    BASELINE_RESULT = {"auprc": average_precision_score(_yte, _p), "auroc": roc_auc_score(_yte, _p)}
    _rows = {"logistic regression": BASELINE_RESULT}
    GBM_RESULT = None
    if CONFIG["RUN_GBM_BASELINE"]:
        # v3: gradient boosting -- often the strongest model on tabular clinical data, so it is
        # the bar the DNN has to meet (matching it WITH interpretability is itself a result).
        from sklearn.ensemble import HistGradientBoostingClassifier
        _gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced",
                                              early_stopping=True, random_state=SEED)
        _gbm.fit(_flat_features(_train_real), _ytr)
        _pg = _gbm.predict_proba(_flat_features(TEST_IDS))[:, 1]
        GBM_RESULT = {"auprc": average_precision_score(_yte, _pg), "auroc": roc_auc_score(_yte, _pg)}
        _rows["gradient boosting"] = GBM_RESULT
    _rows["organ-system DNN (this run)"] = {"auprc": test_result["auprc"], "auroc": test_result["auroc"]}
    _rows["chance"] = {"auprc": float(_yte.mean()), "auroc": 0.5}
    comparison = pd.DataFrame(_rows).T
    comparison["auprc_95ci"] = [f"{a:.3f}-{b:.3f}" for a, b in [bootstrap_ci(_yte, p, average_precision_score, n=300)
                                for p in ([_p] + ([_pg] if GBM_RESULT else []) + [test_result["probs"]])]] + ["-"]
    print(comparison.round(3))
    print(f"\nTest set: {len(_yte)} patients, {int(_yte.sum())} deaths. With this many deaths a gap "
          f"of a few AUPRC points can be noise -- repeat over seeds (Part 12) before concluding.")
else:
    print("Baseline skipped (flag off, or the test split has only one class).")

# %% [markdown]
# ## 11.1e Ensemble of models trained from different random starts (v3)
#
# Trains `N_ENSEMBLE - 1` more copies with the exact same recipe and data, then averages
# their predicted risks. Averaging several models reduces the variance of any single model
# (less memorisation of one random start), usually improves calibration, and gives each
# patient a spread that shows how much the models disagree. The same members are reused in
# 11.4b to test whether the learned inter-system links replicate. This is the number to
# quote for the final model; the single model above is kept for the interpretability plots.

# %%
ENSEMBLE_MODELS, ENSEMBLE_RESULT = [model], None
if CONFIG["N_ENSEMBLE"] > 1:
    print(f"Training {CONFIG['N_ENSEMBLE'] - 1} more ensemble members (same recipe, different seeds)...")
    for k in range(1, CONFIG["N_ENSEMBLE"]):
        _t0 = time.time()
        m_k, _ = train_full_model(seed=SEED + 101 * k, verbose=True)
        ENSEMBLE_MODELS.append(m_k)
        print(f"    ({(time.time() - _t0) / 60:.1f} min)")
    _member_test = [evaluate(m, test_loader) for m in ENSEMBLE_MODELS]
    _ens_probs = np.mean([r["probs"] for r in _member_test], axis=0)
    _yt = test_result["labels"]
    if len(np.unique(_yt)) > 1:
        ENSEMBLE_RESULT = {"auprc": average_precision_score(_yt, _ens_probs), "auroc": roc_auc_score(_yt, _ens_probs),
                           "auprc_ci": bootstrap_ci(_yt, _ens_probs, average_precision_score),
                           "auroc_ci": bootstrap_ci(_yt, _ens_probs, roc_auc_score), "probs": _ens_probs}
        members = pd.DataFrame([{"member": i, "test_auprc": r["auprc"], "test_auroc": r["auroc"]} for i, r in enumerate(_member_test)])
        print(members.round(3).to_string(index=False))
        print(f"\nSpread across members: AUPRC {members.test_auprc.min():.3f}-{members.test_auprc.max():.3f} "
              f"(this spread is seed noise -- differences smaller than it are not real)")
        print(f"ENSEMBLE of {len(ENSEMBLE_MODELS)}: AUPRC {ENSEMBLE_RESULT['auprc']:.3f} "
              f"(95% CI {ENSEMBLE_RESULT['auprc_ci'][0]:.3f}-{ENSEMBLE_RESULT['auprc_ci'][1]:.3f}), "
              f"AUROC {ENSEMBLE_RESULT['auroc']:.3f} (95% CI {ENSEMBLE_RESULT['auroc_ci'][0]:.3f}-{ENSEMBLE_RESULT['auroc_ci'][1]:.3f})")
        _sd = np.std([r["probs"] for r in _member_test], axis=0)
        print(f"Per-patient disagreement between members: median SD of predicted risk {np.median(_sd):.3f}, "
              f"90th percentile {np.percentile(_sd, 90):.3f}")
else:
    print("N_ENSEMBLE=1 -- single model only.")

# %% [markdown]
# ## 11.2 Per-system contribution breakdown — the interpretability payoff
#
# If `FUSION_STRATEGY == 'nam'`, `per_system_signal` from the model's forward pass is the
# literal additive decomposition promised in §1.1/§1.5.2 — each system's contribution to the
# final logit, individually plottable per patient. This is the concrete version of "Renal:
# CRITICAL | Cardio: OK" from the architecture diagram in your meeting notes.

# %%
@torch.no_grad()
def per_system_breakdown(model, loader, n_patients=6):
    model.eval()
    rows = []
    count = 0
    for batch in loader:
        batch_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_dev)
        if out["per_system_signal"] is None:
            print("FUSION_STRATEGY is not 'nam' -- no per-system decomposition available. "
                  "Set CONFIG['FUSION_STRATEGY']='nam' and re-run Part 9-10 to see this.")
            return None
        for i, sid in enumerate(batch["subject_id"]):
            row = {"subject_id": sid, "true_label": float(batch["label"][i]),
                   "predicted_prob": float(torch.sigmoid(out["logit"][i]))}
            for system, contrib in out["per_system_signal"].items():
                row[system] = float(contrib[i])
            rows.append(row)
            count += 1
            if count >= n_patients:
                break
        if count >= n_patients:
            break
    return pd.DataFrame(rows)

breakdown_df = per_system_breakdown(model, test_loader, n_patients=min(8, len(test_dataset)))
if breakdown_df is not None:
    display_cols = [c for c in breakdown_df.columns if c not in ("subject_id", "true_label", "predicted_prob")]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(breakdown_df))))
    breakdown_df.set_index("subject_id")[display_cols].plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Per-system additive contribution to the mortality logit (NAM fusion, §9.7)")
    ax.set_xlabel("contribution to logit (positive = pushes toward 'died')")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.show()
    breakdown_df

# %% [markdown]
# ## 11.2b What the coupling layer learned — for surgeon validation (v2)
#
# No inter-system relationship was defined by hand; the layer learned them. This section
# turns what it learned into three things a surgeon can check against physiology:
#
# 1. **"Who listens to whom" heatmap** — row = the system whose reading is being updated,
#    column = the system it takes information from. Each row sums to 100%.
# 2. **Network diagram** — the strongest learned links as arrows, thickness = strength,
#    shown separately for patients who died and who survived.
# 3. **How much it mattered** — for each system, how far its answer moved (in risk-logit
#    units) because of what it heard from the others. A strong-looking link that changes
#    nothing is not worth a surgeon's time; this separates the two.
#
# Plus a **validation sheet** (CSV) listing each link in plain words with a bootstrap
# confidence interval and blank columns for the surgeon's verdict.
#
# How to read it honestly: attention shows **what the model uses**, not what causes what in
# the body. A surgeon's "yes, that's physiological" is evidence the model learned something
# sensible; a "no" is a flag to investigate (data artefact, confounding by department).
# Links only matter if the gate (how much each system actually uses its message) is
# non-trivial — both are shown.

# %%
PLAIN_NAMES = {
    "renal": "Kidneys", "cardiovascular": "Heart & circulation", "respiratory": "Lungs & breathing",
    "metabolic_hepatic": "Metabolism & liver", "haematology": "Blood", "neurological": "Brain & nerves",
    "gi": "Digestive", "msk": "Bones & joints", "static": "Patient context", "whole_patient": "Whole patient",
}

@torch.no_grad()
def collect_system_outputs(model, ids):
    ds = INSPIREDataset(ids)
    loader = DataLoader(ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, collate_fn=collate_bundle)
    model.eval()
    W, effect, say, answers, contrib, labels, sids = [], [], [], [], [], [], []
    terms = model.term_names
    for batch in loader:
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        if out["coupling_report"] is not None:
            W.append(out["coupling_report"]["weights"].cpu().numpy())
            eff = coupling_effect(model, out)
            effect.append(np.stack([eff[s].cpu().numpy() for s in model.organ_systems], axis=1))
        if out["system_say"] is not None:
            say.append(np.stack([out["system_say"][s].cpu().numpy() for s in model.organ_systems], axis=1))
        if out["system_answers"] is not None:
            answers.append(np.stack([out["system_answers"][s].cpu().numpy() for s in model.organ_systems], axis=1))
            contrib.append(np.stack([out["per_system_signal"][s].cpu().numpy() for s in terms], axis=1))
        labels.append(batch["label"].numpy()); sids += list(batch["subject_id"])
    cat = lambda xs: np.concatenate(xs) if xs else None
    return {"weights": cat(W), "effect": cat(effect), "say": cat(say), "answers": cat(answers),
            "contrib": cat(contrib), "labels": cat(labels), "subject_id": sids}

# Use val + test patients (never trained on) so the picture is not a memorised one.
_eval_ids = list(VAL_IDS) + list(TEST_IDS)
SYSTEM_OUTPUTS = collect_system_outputs(model, _eval_ids)
_S = model.organ_systems
_names = [PLAIN_NAMES.get(s, s) for s in _S]

if SYSTEM_OUTPUTS["weights"] is None:
    print(f"COUPLING_MODE={CONFIG['COUPLING_MODE']!r} -- no learned coupling layer in this model, nothing to plot.")
    COUPLING_VALIDATION_SHEET = None
else:
    Wt = SYSTEM_OUTPUTS["weights"]                 # [N, receiver, sender]
    y = SYSTEM_OUTPUTS["labels"]
    gate = model.coupling.gate.detach().cpu().numpy()
    mean_W = Wt.mean(axis=0)

    # --- 1. heatmap -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    shown = np.where(np.eye(len(_S), dtype=bool), np.nan, mean_W * 100)
    im = ax.imshow(shown, cmap="Purples", vmin=0, vmax=np.nanmax(shown))
    for i in range(len(_S)):
        for j in range(len(_S)):
            if i != j:
                ax.text(j, i, f"{shown[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if shown[i, j] > 0.6 * np.nanmax(shown) else "black")
    ax.set_xticks(range(len(_S))); ax.set_xticklabels(_names, rotation=40, ha="right")
    ax.set_yticks(range(len(_S)))
    ax.set_yticklabels([f"{n}  (uses msgs: {abs(g):.2f})" for n, g in zip(_names, gate)])
    ax.set_xlabel("...takes information FROM this system"); ax.set_ylabel("This system's reading...")
    ax.set_title("Learned inter-system coupling (val+test patients)\nrow sums to 100%; diagonal blank = a system never reads itself")
    plt.colorbar(im, ax=ax, fraction=0.04, label="% of attention")
    plt.tight_layout(); plt.show()

    # --- 2. network diagram, died vs survived ------------------------------------------
    def _network(ax, M, title, top_k=8):
        n = len(_S)
        ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
        pos = np.c_[np.cos(ang), np.sin(ang)]
        eff = M * np.abs(gate)[:, None]                        # attention x how much the receiver uses it
        np.fill_diagonal(eff, 0)
        idx = np.dstack(np.unravel_index(np.argsort(-eff, axis=None), eff.shape))[0][:top_k]
        vmax = eff.max() if eff.max() > 0 else 1
        for r, snd in idx:
            ax.annotate("", xy=pos[r] * 0.86, xytext=pos[snd] * 0.86,
                        arrowprops=dict(arrowstyle="-|>", color="#8e44ad", alpha=0.85,
                                        lw=0.5 + 5 * eff[r, snd] / vmax, shrinkA=14, shrinkB=14,
                                        connectionstyle="arc3,rad=0.12"))
        for k in range(n):
            ax.scatter(*pos[k], s=900, color="#0a7d6e", zorder=3)   # v3: all eight organ systems drawn alike
            ax.text(*(pos[k] * 1.22), _names[k], ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5); ax.axis("off"); ax.set_title(title, fontsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, sel, t in [(axes[0], y == 1, "Patients who died"), (axes[1], y == 0, "Patients who survived")]:
        if sel.any():
            _network(ax, Wt[sel].mean(axis=0), f"{t} (n={int(sel.sum())})\narrow = information flow, thickness = strength x use")
    plt.suptitle("Strongest learned links between body systems (top 8)", fontsize=11)
    plt.tight_layout(); plt.show()

    # --- 3. how much the coupling changed each system's answer --------------------------
    E = SYSTEM_OUTPUTS["effect"]                   # [N, S] change in logit units
    fig, ax = plt.subplots(figsize=(9, 4))
    xk = np.arange(len(_S))
    for off, sel, col, lab in [(-0.2, y == 1, "#c0392b", "died"), (0.2, y == 0, "#2980b9", "survived")]:
        if sel.any():
            ax.bar(xk + off, np.abs(E[sel]).mean(axis=0), width=0.4, color=col, label=lab)
    ax.set_xticks(xk); ax.set_xticklabels(_names, rotation=30, ha="right")
    ax.set_ylabel("mean |change in answer|\n(risk-logit units)")
    ax.set_title("How much each system's answer changed because of what it heard from other systems")
    ax.legend(); plt.tight_layout(); plt.show()

    # --- validation sheet ---------------------------------------------------------------
    rng_b = np.random.default_rng(SEED)
    boot = np.stack([Wt[rng_b.integers(0, len(Wt), len(Wt))].mean(axis=0) for _ in range(200)])
    rows = []
    for r in range(len(_S)):
        for snd in range(len(_S)):
            if r == snd:
                continue
            rows.append({
                "receiver": _S[r], "sender": _S[snd],
                "in plain words": f"The model's reading of {_names[r]} draws on {_names[snd]}",
                "attention_%": 100 * mean_W[r, snd],
                "95%_CI": f"{100*np.percentile(boot[:, r, snd], 2.5):.0f}-{100*np.percentile(boot[:, r, snd], 97.5):.0f}%",
                "died_%": 100 * Wt[y == 1][:, r, snd].mean() if (y == 1).any() else np.nan,
                "survived_%": 100 * Wt[y == 0][:, r, snd].mean() if (y == 0).any() else np.nan,
                "receiver_uses_messages": abs(float(gate[r])),
                "effective_strength": mean_W[r, snd] * abs(float(gate[r])),
                "above_even_split": bool(np.percentile(boot[:, r, snd], 2.5) > 1 / (len(_S) - 1)),
                "Surgeon: physiologically plausible? (Y/N/unsure)": "",
                "Surgeon: comment": "",
            })
    COUPLING_VALIDATION_SHEET = pd.DataFrame(rows).sort_values("effective_strength", ascending=False).reset_index(drop=True)
    _csv = os.path.join(CONFIG["CHECKPOINT_DIR"], "coupling_validation_sheet.csv")
    COUPLING_VALIDATION_SHEET.to_csv(_csv, index=False)
    print(f"Even split for a single link = {100/(len(_S)-1):.1f}% (attention spread evenly over the other {len(_S)-1} systems).")
    print("The 95% CI resamples PATIENTS for this one trained model -- it shows how consistent the pattern is "
          "across patients, not whether it would re-appear if the model were retrained. Part 11.4b tests that.")
    if np.abs(gate).max() < 0.05:
        print("\nWARNING: every coupling gate is < 0.05 -- the model is barely using what systems hear from each "
              "other, so the links above have almost no effect on predictions. Not worth a surgeon's review "
              "until a run where the gates are larger (longer training / full cohort).")
    print(COUPLING_VALIDATION_SHEET.head(10)[["in plain words", "attention_%", "95%_CI", "died_%",
                                             "survived_%", "receiver_uses_messages", "above_even_split"]].round(2).to_string())
    _cr = COUPLING_VALIDATION_SHEET[(COUPLING_VALIDATION_SHEET.receiver == "renal") &
                                    (COUPLING_VALIDATION_SHEET.sender == "cardiovascular")]
    if len(_cr):
        print(f"\nCardiorenal check (never given to the model): Kidneys <- Heart link ranks "
              f"#{_cr.index[0] + 1} of {len(COUPLING_VALIDATION_SHEET)} by effective strength, "
              f"{'above' if _cr['above_even_split'].iloc[0] else 'NOT above'} an even split.")
    print(f"\nFull sheet for the surgeon (with blank verdict columns): {_csv}")

# %% [markdown]
# ## 11.2c How much say each system got (arbitration layer, v2)
#
# Every system gives its own answer; the arbitration layer decides how much that answer
# counts for each patient (1.0 = an equal share). Shown by department, because the layer
# reads the patient's context — a sensible model should, for example, give the heart more
# say in cardiac-surgery patients. That is the clinical check here.

# %%
if SYSTEM_OUTPUTS["say"] is None:
    print("USE_ARBITRATION_LAYER=False (or concat fusion) -- every system has equal say.")
else:
    say_df = pd.DataFrame(SYSTEM_OUTPUTS["say"], columns=_names)
    say_df["department"] = COHORT_INDEXED.loc[SYSTEM_OUTPUTS["subject_id"], "department"].values
    say_by_dept = say_df.groupby("department").mean()
    say_by_dept.insert(0, "n", say_df.groupby("department").size())
    say_by_dept = say_by_dept[say_by_dept["n"] >= 10]
    print("Mean say per system, by department (1.0 = equal share):")
    print(say_by_dept.round(2).to_string())
    if len(say_by_dept):
        fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(say_by_dept))))
        im = ax.imshow(say_by_dept[_names].values, cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(_names))); ax.set_xticklabels(_names, rotation=30, ha="right")
        ax.set_yticks(range(len(say_by_dept))); ax.set_yticklabels([f"{d} (n={n})" for d, n in zip(say_by_dept.index, say_by_dept["n"])])
        for i in range(len(say_by_dept)):
            for j, nm in enumerate(_names):
                ax.text(j, i, f"{say_by_dept[nm].iloc[i]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="say (1 = equal)"); ax.set_title("How much each system's answer counts, by department")
        plt.tight_layout(); plt.show()
    print(f"\nSpread of say across all patients: min {SYSTEM_OUTPUTS['say'].min():.2f}, max {SYSTEM_OUTPUTS['say'].max():.2f}. "
          f"If everything is ~1.0 the layer found no reason to re-weight systems -- a valid result, not a bug.")

# %% [markdown]
# ## 11.2d Does each organ system answer from its own data? (v3 leakage check + share of risk)
#
# Two checks that protect the "per-system" story once the whole-patient and coupling layers
# exist:
#
# - **Own-data share** — shuffle one system's own inputs across patients and see how much
#   its answer moves, then shuffle everything else instead. Share near 1 = the kidney answer
#   really is about the kidneys; below 0.5 = it mostly echoes other systems (flagged).
# - **Share of risk** — how much of each patient's risk comes from the whole-patient term
#   versus the eight organ systems. If the whole-patient term takes nearly all of it, the
#   organ systems have become decoration.

# %%
LEAKAGE = None
if SYSTEM_OUTPUTS["answers"] is not None:
    _big = DataLoader(INSPIREDataset(list(TEST_IDS)), batch_size=min(1024, len(TEST_IDS)), shuffle=True,
                      collate_fn=collate_bundle, generator=torch.Generator().manual_seed(SEED))
    LEAKAGE = pd.DataFrame(own_data_share(model, _to_device(next(iter(_big))))).T
    LEAKAGE.index = [PLAIN_NAMES.get(s, s) for s in LEAKAGE.index]
    LEAKAGE["verdict"] = np.where(LEAKAGE["own_share"] >= 0.5, "own data", "FLAG: mostly other systems")
    print(LEAKAGE.round(3).to_string())
    _C = np.abs(SYSTEM_OUTPUTS["contrib"])
    _ctx = model.term_names.index(model.context_term)
    SHARE_WHOLE_PATIENT = float(np.mean(_C[:, _ctx] / np.clip(_C.sum(axis=1), 1e-9, None)))
    print(f"\nShare of risk (mean over val+test patients): {model.context_term} term {SHARE_WHOLE_PATIENT:.0%}, "
          f"eight organ systems {1 - SHARE_WHOLE_PATIENT:.0%}.")
    _share_sys = pd.Series(_C[:, :_ctx].mean(axis=0) / _C.sum(axis=1).mean(), index=_names)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(_names + [PLAIN_NAMES.get(model.context_term)], list(_share_sys.values) + [SHARE_WHOLE_PATIENT],
           color=["#0a7d6e"] * len(_names) + ["#7f8c8d"])
    ax.set_ylabel("average share of |risk|"); ax.set_title("Where each prediction comes from")
    ax.tick_params(axis="x", rotation=30); plt.tight_layout(); plt.show()
else:
    SHARE_WHOLE_PATIENT = None
    print("Concat fusion -- no per-system answers to check.")

# %% [markdown]
# ## 11.3 Attention audit (§1.1, §3C item 4 of the source repo's roadmap)
#
# Do the transformer heads concentrate on clinically sensible time steps? A lightweight
# check: for one system and one patient, plot the mean attention weight the encoder's first
# layer places on each time step, alongside the observed/imputed mask — a real audit for a
# full run would extend this per-head and cross-reference against the acute-deterioration
# ICD-10 codes already flagged in the source repo's EDA (D65, I46, R57, J80, K72, A41).

# %%
@torch.no_grad()
def get_attention_weights(model, batch_item, system):
    # First-layer attention of the shared organ-system encoder for one patient/system.
    # Token 0 = the system's summary token (its static features), tokens 1..T = time points.
    x = batch_item[f"{system}_x"].unsqueeze(0).to(DEVICE) if f"{system}_x" in batch_item else None
    m = batch_item[f"{system}_mask"].unsqueeze(0).to(DEVICE) if f"{system}_mask" in batch_item else None
    st = batch_item["static"].unsqueeze(0).to(DEVICE)
    idx = model.system_static_idx[system]
    if x is None:
        return None
    return model.encoder.attention_map(system, x, m, st[:, idx] if idx else None).squeeze(0).cpu().numpy()

example_item = test_dataset[0]
example_system = "renal"
attn = get_attention_weights(model, example_item, example_system)
if attn is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(attn, cmap="viridis"); axes[0].set_title(f"{example_system} attention (layer 0, head-avg)")
    axes[0].set_xlabel("attended-to token (0 = summary)"); axes[0].set_ylabel("query token")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    mask = example_item[f"{example_system}_mask"].numpy()
    axes[1].imshow(mask.T, aspect="auto", cmap="Greys"); axes[1].set_title("observed(black)/imputed(white) mask")
    axes[1].set_xlabel("timestep"); axes[1].set_ylabel("feature index")
    plt.tight_layout(); plt.show()
    # Does attention favour genuinely observed time points? (v3: computed, not just suggested)
    _obs = mask.max(axis=1) > 0
    _col = attn[:, 1:].mean(axis=0)
    if _obs.any() and (~_obs).any():
        print(f"Mean attention on time points with a real observation: {_col[_obs].mean():.4f} vs imputed-only: "
              f"{_col[~_obs].mean():.4f} (ratio {_col[_obs].mean() / max(_col[~_obs].mean(), 1e-9):.2f}; >1 = favours real data)")
