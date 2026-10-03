# %% [markdown]
# # Part 13 — Looking inside the model (v4)
#
# Everything James asked for in the 28-09-2026 meeting, built on the "diary" the training
# loop kept in Part 10 (per-epoch metrics + embedding snapshots):
#
# | Section | Question it answers | Like the reference paper's… |
# |---|---|---|
# | 13.1 | Is everything saved — model, ensemble, predictions, figures? | "save the model / read the model / save the images" |
# | 13.2 | How good was the model before training, after pre-training, after each epoch — for emergency and scheduled patients? | training curves |
# | 13.3 | What do the patients look like in the model's mind, before vs after training? | Fig. 5 t-SNE, epoch 1 vs epoch 32 |
# | 13.4 | Are there natural groups of patients nobody labelled? | "useful features clustering" |
# | 13.5 | How alike does the model think patients are, early vs late? | Fig. 4 cosine-similarity heatmaps |
# | 13.6 | Does the model work as well for emergency as for scheduled surgery? | results table |
# | 13.7 | Does filling missing data, or making synthetic patients, distort NEWS2? | — |
# | 13.8 | Inside each organ system, which measurements drove its risk points? (SHAP) | — |
# | 13.9 | Does a bigger embedding help? | Fig. 6 score vs hidden size |
# | 13.10 | Are the learned embeddings useful for OTHER questions? | "using the same feature vector for other tasks" |
#
# **Plain words.** An *embedding* is the model's short private summary of a patient — a
# list of numbers (32 per organ system with the `medium` preset). Patients the model sees as
# alike get similar lists. We cannot look at 32 dimensions, so PCA / t-SNE / UMAP squash
# them onto a flat map where each dot is a patient. If the dots for patients who died drift
# together as training goes on, the model has learned something real about risk.
#
# Every section is run by `run_section(...)`: if one fails, it says so loudly (with the
# error) and the rest of Part 13 still runs. Nothing fails silently — the status of every
# section is listed at the end (13.11) and in Part 14's verification table.

# %%
# INLINE: inspire_dnn/analysis.py
from inspire_dnn.analysis import *

# %%
# INLINE: inspire_dnn/explain.py
from inspire_dnn.explain import *

# %%
import traceback

PART13 = {}            # every result this Part produces, by name
SECTION_STATUS = {}    # section -> "ok" | "skipped: why" | "FAILED: why"

def run_section(name, fn):
    _t0 = time.time()
    try:
        r = fn()
        SECTION_STATUS[name] = r if isinstance(r, str) and r.startswith("skipped") else "ok"
    except Exception as e:
        SECTION_STATUS[name] = f"FAILED: {type(e).__name__}: {e}"
        print(f"\n!! {name} FAILED -- {type(e).__name__}: {e}\n   (the rest of Part 13 continues; full error below)")
        traceback.print_exc()
    print(f"[{name}: {SECTION_STATUS[name]} -- {time.time() - _t0:.1f}s]")

def _plain(t):
    return PLAIN_NAMES.get(t, t)

def _cohort_value(ids, col, default=np.nan):
    return np.array([COHORT_INDEXED.loc[s, col] if col in COHORT_INDEXED.columns else default for s in ids], dtype=object)

SNAP_EMOP = _emop_of(SNAPSHOT_IDS)
SNAP_DIED = np.array([PATIENT_BUNDLE[s]["label"] for s in SNAPSHOT_IDS], dtype=float)
SNAP_GROUPS = group_labels(SNAP_EMOP, SNAP_DIED)

def patient_embedding(snap, which="systems"):
    # One vector per patient: every organ system's embedding side by side + the whole-patient summary.
    parts = [snap[which].reshape(len(snap[which]), -1)]
    if snap.get("whole_patient") is not None:
        parts.append(snap["whole_patient"])
    return np.concatenate(parts, axis=1).astype(np.float32)

def _snap(tag_prefix):
    for sn in SNAPSHOTS:
        if sn["tag"].startswith(tag_prefix):
            return sn
    return None

def key_snapshots():
    # before training, after pre-training, one fine-tuning epoch in the middle, chosen model
    ft = [sn for sn in SNAPSHOTS if sn["phase"] == "fine-tuning"]
    mid = ft[len(ft) // 2] if ft else None
    out = [_snap("before_training"), _snap("after_pretraining"), mid, _snap("chosen_model")]
    seen, keep = set(), []
    for sn in out:
        if sn is not None and sn["tag"] not in seen:
            keep.append(sn); seen.add(sn["tag"])
    return keep

def _snapshot_title(sn):
    return {"before training": "Before any training", "after pre-training": "After pre-training (no labels yet)",
            "chosen model": f"Chosen model (epoch {sn['epoch']})"}.get(sn["phase"], f"Fine-tuning epoch {sn['epoch']}")

def _subsample_keep_deaths(n_max, seed=SEED):
    idx = np.arange(len(SNAPSHOT_IDS))
    if len(idx) <= n_max:
        return idx
    dead = idx[SNAP_DIED == 1]; alive = idx[SNAP_DIED == 0]
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([dead, rng.choice(alive, size=max(0, n_max - len(dead)), replace=False)]))

print(f"Part 13 inputs: {len(SNAPSHOTS)} snapshots of {len(SNAPSHOT_IDS)} validation patients "
      f"({int(SNAP_DIED.sum())} deaths; groups: {pd.Series(SNAP_GROUPS).value_counts().to_dict()})")

# %% [markdown]
# ## 13.1 Save everything (model, ensemble, predictions)
#
# The chosen model was saved and reload-checked in Part 10.3. Here: every ensemble member,
# and one CSV with every test patient's prediction and per-term risk points — so any later
# analysis can start from files instead of re-training.

# %%
def section_save_everything():
    saved = []
    if CONFIG["SAVE_FINAL_MODEL"] and len(ENSEMBLE_MODELS) > 1:
        for i, m in enumerate(ENSEMBLE_MODELS[1:], start=1):
            saved.append(save_final_model(m, name=f"ensemble_member_{i}"))
    out = collect_system_outputs(model, list(TEST_IDS))
    df = pd.DataFrame({"subject_id": out["subject_id"], "died_30d": out["labels"],
                       "predicted_risk_raw": test_result["probs"]})
    if globals().get("CALIBRATOR") is not None:
        df["predicted_risk_recalibrated"] = test_probs_calibrated
    if ENSEMBLE_RESULT is not None:
        df["predicted_risk_ensemble"] = ENSEMBLE_RESULT["probs"]
    df["emergency"] = _emop_of(df["subject_id"])
    for col in ("department", "asa"):
        df[col] = _cohort_value(df["subject_id"], col)
    if out["contrib"] is not None:
        for j, t in enumerate(model.term_names):
            df[f"points_{t}"] = out["contrib"][:, j]
    path = os.path.join(TABLE_DIR, "test_predictions.csv")
    df.to_csv(path, index=False)
    PART13["test_predictions"] = df
    print(f"Test predictions + per-term risk points -> {path} ({len(df)} patients)")
    print(f"Models in {MODEL_DIR}: {sorted(os.listdir(MODEL_DIR))}")
    print(f"Figures saved so far: {len(SAVED_FIGURES)} in {FIGURE_DIR}")

run_section("13.1 save everything", section_save_everything)

# %% [markdown]
# ## 13.2 Training curves — before training, after pre-training, and every epoch
#
# The left panel is AUPRC (how well deaths are found without false alarms), the right is
# AUROC. Each colour is a patient group: all, emergency, scheduled. The two points on the
# far left are **before any training** (random dials) and **after pre-training** (the body
# knowledge is learned, but the model has not yet been told who died). Dotted horizontal
# lines show what random guessing would score for each group (its death rate, for AUPRC).
# The grey line marks the epoch that was kept. Emergency and scheduled curves are computed
# on far fewer deaths than "all" — read their trend, not single wiggles.

# %%
def section_training_curves():
    df = pd.DataFrame(EPOCH_LOG)
    if df.empty or "val_auprc_all" not in df:
        return "skipped: no epoch log"
    ft = df[df.phase == "fine-tuning"]
    if ft.empty:
        return "skipped: no fine-tuning epochs logged (resumed from a finished checkpoint?)"
    pre = df[df.phase.isin(["before training", "after pre-training"])]
    xs_pre = {"before training": -3.5, "after pre-training": -1.8}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {"all": "#2c3e50", "emergency": "#d94801", "scheduled": "#08519c"}
    yv = np.array([PATIENT_BUNDLE[s]["label"] for s in VAL_IDS], dtype=float)
    for ax, metric in zip(axes, ("auprc", "auroc")):
        for grp, c in colors.items():
            col = f"val_{metric}_{grp}"
            ax.plot(ft["epoch"], ft[col], "-", color=c, label=f"{grp} (val)")
            for _, r in pre.iterrows():
                ax.plot(xs_pre[r["phase"]], r[col], "o", color=c, ms=7)
            if metric == "auprc":
                sel = {"all": np.ones(len(yv), bool), "emergency": VAL_EMOP > 0.5, "scheduled": VAL_EMOP <= 0.5}[grp]
                if sel.any():
                    ax.axhline(yv[sel].mean(), color=c, ls=":", lw=1)
        if metric == "auprc" and "train_auprc_all" in ft:
            ax.plot(ft["epoch"], ft["train_auprc_all"], "--", color="#7f8c8d", label="all (training patients)")
        if metric == "auroc":
            ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.axvline(BEST_EPOCH, color="gray", lw=1)
        ax.set_xticks([-3.5, -1.8] + list(ax.get_xticks()[ax.get_xticks() >= 0]))
        ax.set_xticklabels(["before\ntraining", "after\npre-train"] + [str(int(t)) for t in ax.get_xticks()[2:]])
        ax.set_xlim(-4.3, max(ft["epoch"].max(), 1) + 1)
        ax.set_xlabel("fine-tuning epoch"); ax.set_ylabel(metric.upper()); ax.legend(fontsize=8)
    axes[0].set_title("How well deaths are found (AUPRC), by group, epoch by epoch")
    axes[1].set_title("How well died vs survived are ranked (AUROC), by group")
    plt.tight_layout(); plt.show()
    rows = []
    for label, sel in (("before training", df.phase == "before training"), ("after pre-training", df.phase == "after pre-training"),
                       ("fine-tuning epoch 0", (df.phase == "fine-tuning") & (df.epoch == 0)),
                       (f"chosen epoch {BEST_EPOCH}", (df.phase == "fine-tuning") & (df.epoch == BEST_EPOCH)),
                       ("last epoch", (df.phase == "fine-tuning") & (df.epoch == ft.epoch.max()))):
        if sel.any():
            r = df[sel].iloc[0]
            rows.append({"moment": label, **{f"{m}_{g}": r[f"val_{m}_{g}"] for m in ("auprc", "auroc")
                                              for g in ("all", "emergency", "scheduled")}})
    tab = pd.DataFrame(rows).set_index("moment")
    PART13["training_moments"] = tab
    tab.to_csv(os.path.join(TABLE_DIR, "training_moments_by_group.csv"))
    print(tab.round(3).to_string())
    n = df.iloc[-1]
    print(f"\nValidation deaths behind these curves: all {int(n.get('val_n_deaths_all', 0))}, emergency "
          f"{int(n.get('val_n_deaths_emergency', 0))}, scheduled {int(n.get('val_n_deaths_scheduled', 0))}.")

run_section("13.2 training curves by group", section_training_curves)

# %% [markdown]
# ## 13.3 Embedding maps — the patients in the model's mind, before vs after training
#
# One row per method (PCA = a straight-line squash; t-SNE and UMAP = keep near neighbours
# near). One column per moment in training. Each dot is a validation patient, coloured by
# the four groups James asked for: **emergency / scheduled × died / survived**. Deaths are
# drawn larger and on top, because there are so few.
#
# **How to read it.** Before training the colours are mixed — the dials are random. If after
# pre-training (no labels used!) deaths already drift towards one region, the body knowledge
# itself separates sick from well. After fine-tuning, deaths should gather further. Distance
# and axis numbers on t-SNE/UMAP maps mean nothing; only "who is near whom" does — which is
# why 13.3b also reports a number computed in the full embedding, not on the flat map.

# %%
def section_embedding_maps():
    snaps = key_snapshots()
    if not snaps:
        return "skipped: no snapshots"
    idx = _subsample_keep_deaths(CONFIG["TSNE_MAX_POINTS"])
    coords = {}
    methods = [m for m in CONFIG["EMBEDDING_METHODS"]]
    for method in methods:
        fig, axes = plt.subplots(1, len(snaps), figsize=(4.6 * len(snaps), 4.6), squeeze=False)
        any_ok = False
        for ax, sn in zip(axes[0], snaps):
            Z = project_2d(patient_embedding(sn)[idx], method=method, seed=SEED)
            if Z is None:
                ax.set_axis_off(); ax.set_title(f"{method}: umap-learn not installed", fontsize=9); continue
            any_ok = True
            coords[(method, sn["tag"])] = Z
            plot_embedding_map(ax, Z, SNAP_GROUPS[idx], _snapshot_title(sn))
        if any_ok:
            h, l = axes[0][-1].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
            fig.suptitle(f"{method.upper()} map of patient embeddings: emergency/scheduled x died/survived", y=1.02)
            plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()
        else:
            plt.close(fig)
            print(f"{method}: not available in this environment -- skipped (install umap-learn to get it).")
    PART13["map_coords"] = coords
    PART13["map_idx"] = idx

run_section("13.3 embedding maps (PCA / t-SNE / UMAP)", section_embedding_maps)

# %% [markdown]
# ### 13.3b A number for "do deaths sit near deaths?" — at every snapshot
#
# For each patient, look at its 15 nearest neighbours in the FULL embedding (not the flat
# map). **Enrichment** = how much more often a died patient's neighbours died than the death
# rate overall. 1.0 = deaths scattered at random; 3.0 = a died patient's neighbourhood has
# three times the usual death rate. The same is shown for emergency status (how strongly the
# embedding encodes "this was an emergency"), and for each organ system's own embedding.

# %%
def section_enrichment():
    rows = []
    for sn in SNAPSHOTS:
        X = patient_embedding(sn)
        d = knn_death_enrichment(X, SNAP_DIED, CONFIG["KNN_ENRICHMENT_K"])
        e = knn_death_enrichment(X, SNAP_EMOP, CONFIG["KNN_ENRICHMENT_K"])
        rows.append({"snapshot": sn["tag"], "phase": sn["phase"], "epoch": sn["epoch"],
                     "death_enrichment": d["enrichment_ratio"], "died_neighbour_rate_for_died": d["died_neighbour_rate_for_died"],
                     "death_rate": d["base_rate"], "emergency_enrichment": e["enrichment_ratio"]})
    tab = pd.DataFrame(rows)
    PART13["enrichment_by_snapshot"] = tab
    tab.to_csv(os.path.join(TABLE_DIR, "embedding_enrichment_by_snapshot.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].plot(range(len(tab)), tab["death_enrichment"], "o-", color="#c0392b", label="deaths")
    axes[0].plot(range(len(tab)), tab["emergency_enrichment"], "s--", color="#d35400", label="emergency operations")
    axes[0].axhline(1, color="gray", ls=":"); axes[0].set_xticks(range(len(tab)))
    axes[0].set_xticklabels([t.replace("finetune_epoch_", "ep ").replace("_", " ") for t in tab["snapshot"]], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel(f"enrichment among {CONFIG['KNN_ENRICHMENT_K']} nearest neighbours")
    axes[0].set_title("Do similar patients share the outcome? (1 = random)"); axes[0].legend()
    sn = _snap("chosen_model") or SNAPSHOTS[-1]
    per_sys = {}
    for j, s in enumerate(sn["organ_systems"]):
        per_sys[_plain(s)] = knn_death_enrichment(sn["systems"][:, j, :], SNAP_DIED, CONFIG["KNN_ENRICHMENT_K"])["enrichment_ratio"]
    per_sys = pd.Series(per_sys).sort_values()
    axes[1].barh(per_sys.index, per_sys.values, color="#0a7d6e"); axes[1].axvline(1, color="gray", ls=":")
    axes[1].set_title("Chosen model: which organ system's own notes group the deaths best?")
    plt.tight_layout(); plt.show()
    PART13["enrichment_by_system"] = per_sys

run_section("13.3b neighbourhood enrichment", section_enrichment)

# %% [markdown]
# ### 13.3c The chosen model's map, coloured four more ways
#
# Same dots, same positions (the chosen model's t-SNE, or PCA if t-SNE is off): coloured by
# department, ASA grade, worst pre-op NEWS2 band, and which risk term pushed that patient's
# risk up the most. Then one map per organ system (that system's own notes only).

# %%
def section_map_colourings():
    sn = _snap("chosen_model") or SNAPSHOTS[-1]
    coords, idx = PART13.get("map_coords", {}), PART13.get("map_idx")
    method = next((m for m in ("tsne", "umap", "pca") if (m, sn["tag"]) in coords), None)
    if method is None:
        idx = _subsample_keep_deaths(CONFIG["TSNE_MAX_POINTS"]); method = "pca"
        Z = project_2d(patient_embedding(sn)[idx], "pca", SEED)
    else:
        Z = coords[(method, sn["tag"])]
    ids = [SNAPSHOT_IDS[i] for i in idx]
    news2 = []
    for s in ids:
        v = PATIENT_BUNDLE[s]["static"].get("news2_max", np.nan)
        news2.append("no score" if (v is None or not np.isfinite(v)) else ("high (7+)" if v >= 7 else "medium (5-6)" if v >= 5 else "low (0-4)"))
    top_term = np.array(["-"] * len(idx), dtype=object)
    if sn.get("contrib") is not None:
        top_term = np.array([_plain(sn["term_names"][j]) for j in np.argmax(sn["contrib"][idx], axis=1)], dtype=object)
    colourings = [("Department", _cohort_value(ids, "department")), ("ASA grade", _cohort_value(ids, "asa")),
                  ("Worst pre-op NEWS2 band", np.array(news2, dtype=object)), ("Term adding the most risk", top_term)]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (title, vals) in zip(axes, colourings):
        plot_categorical_map(ax, Z, vals, f"{title} ({method.upper()}, chosen model)")
        ax.legend(fontsize=6, markerscale=2, loc="best")
    plt.tight_layout(); plt.show()
    systems = sn["organ_systems"]
    ncol = 4; nrow = int(np.ceil(len(systems) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow), squeeze=False)
    for k, s in enumerate(systems):
        ax = axes[k // ncol][k % ncol]
        Zs = project_2d(sn["systems"][idx, k, :], "tsne" if "tsne" in CONFIG["EMBEDDING_METHODS"] else "pca", SEED)
        plot_embedding_map(ax, Zs, SNAP_GROUPS[idx], f"{_plain(s)} -- its own notes")
    for k in range(len(systems), nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Each organ system's embedding, chosen model (emergency/scheduled x died/survived)", y=1.01)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.show()

run_section("13.3c map colourings + per-system maps", section_map_colourings)

# %% [markdown]
# ## 13.4 Natural patient groups nobody labelled (clustering)
#
# K-means groups the chosen model's embeddings into k clusters (k from 3 to 8, chosen by
# the silhouette score: how cleanly separated the groups are, −1 to 1). Each cluster is then
# described in plain terms: size, death rate (and how many times the overall rate), share of
# emergency operations, typical department and ASA, and which term usually adds the most
# risk. A cluster with a high death rate that a surgeon recognises ("emergency general
# surgery, ASA 4, kidneys driving the risk") is exactly the qualitative evidence the paper
# needs. A cluster nobody recognises is a question, not a finding.

# %%
def section_clustering():
    sn = _snap("chosen_model") or SNAPSHOTS[-1]
    X = patient_embedding(sn)
    labels, sil = cluster_embeddings(X, seed=SEED)
    print("Silhouette by k (higher = cleaner groups):"); print(sil.round(3).to_string(index=False))
    top_term = (np.array([_plain(sn["term_names"][j]) for j in np.argmax(sn["contrib"], axis=1)], dtype=object)
                if sn.get("contrib") is not None else None)
    extra = {"department": _cohort_value(SNAPSHOT_IDS, "department"),
             "asa": pd.to_numeric(pd.Series(_cohort_value(SNAPSHOT_IDS, "asa")), errors="coerce").values,
             "predicted_risk": 1 / (1 + np.exp(-sn["logit"].astype(float)))}
    if top_term is not None:
        extra["top_risk_term"] = top_term
    prof = cluster_profile(labels, SNAP_DIED, SNAP_EMOP, extra)
    PART13["cluster_profile"] = prof
    prof.to_csv(os.path.join(TABLE_DIR, "embedding_clusters.csv"))
    print("\nCluster profiles (sorted by death rate):"); print(prof.round(3).to_string())
    Z = PART13.get("map_coords", {}).get(("tsne", sn["tag"]))
    if Z is not None:
        idx = PART13["map_idx"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        plot_categorical_map(ax, Z, np.array([f"cluster {c}" for c in labels[idx]], dtype=object), "Clusters on the chosen model's t-SNE map")
        dead = SNAP_DIED[idx] == 1
        ax.scatter(Z[dead, 0], Z[dead, 1], s=26, facecolors="none", edgecolors="black", linewidths=0.8, label="died")
        ax.legend(fontsize=7, markerscale=2); plt.tight_layout(); plt.show()

run_section("13.4 clustering", section_clustering)

# %% [markdown]
# ## 13.5 How alike does the model think patients are? (cosine similarity, early vs late)
#
# Up to 30 patients from each of the four groups, in group order. Each square compares two
# patients' embeddings: 1 (dark) = the model sees them as the same kind of patient, 0 = no
# relation, negative = opposite. With random dials everything looks alike (a uniformly dark
# square). As the model learns, blocks appear — patients within a group resemble each other
# more than patients across groups. The numbers printed under the plot say how much.

# %%
def section_similarity():
    rng = np.random.default_rng(SEED)
    pick = []
    for g in GROUP_ORDER:
        members = np.where(SNAP_GROUPS == g)[0]
        if len(members):
            pick += list(rng.choice(members, size=min(30, len(members)), replace=False))
    pick = np.array(pick)
    snaps = [sn for sn in (_snap("before_training"), _snap("after_pretraining"), _snap("chosen_model")) if sn is not None]
    fig, axes = plt.subplots(1, len(snaps), figsize=(5.3 * len(snaps), 4.8), squeeze=False)
    rows = []
    for ax, sn in zip(axes[0], snaps):
        X = patient_embedding(sn)[pick]
        X = X - X.mean(axis=0, keepdims=True)   # centred, so "everyone shares the same offset" does not read as similarity
        C = cosine_similarity_matrix(X)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        bounds = np.cumsum([int((SNAP_GROUPS[pick] == g).sum()) for g in GROUP_ORDER])[:-1]
        for b in bounds:
            ax.axhline(b - 0.5, color="black", lw=0.6); ax.axvline(b - 0.5, color="black", lw=0.6)
        ax.set_title(_snapshot_title(sn), fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        same = SNAP_GROUPS[pick][:, None] == SNAP_GROUPS[pick][None, :]
        off = ~np.eye(len(pick), dtype=bool)
        rows.append({"snapshot": sn["tag"], "mean_similarity_same_group": float(C[same & off].mean()),
                     "mean_similarity_different_group": float(C[~same].mean())})
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="cosine similarity")
    fig.suptitle("Cosine similarity of patient embeddings (blocks: " + " | ".join(GROUP_ORDER) + ")", fontsize=10)
    plt.show()
    tab = pd.DataFrame(rows); tab["gap"] = tab["mean_similarity_same_group"] - tab["mean_similarity_different_group"]
    PART13["similarity"] = tab
    print(tab.round(3).to_string(index=False))
    print("A gap that grows from 'before' to 'chosen' = the model increasingly treats patients in the same group as alike.")

run_section("13.5 cosine similarity heatmaps", section_similarity)

# %% [markdown]
# ## 13.6 Emergency vs scheduled surgery — results, risk points, and missing data
#
# Test patients only. **Table 1**: for each group, the number of patients and deaths, AUROC
# and AUPRC with 95% confidence intervals, the Brier score (lower = better calibrated), the
# average predicted risk next to the real death rate, and — at the threshold chosen on
# validation in 11.1b — how many deaths are caught (recall) and how often an alarm is right
# (precision). Computed for the single model, the recalibrated model and the ensemble where
# available. **Table 2**: the average risk points each term adds for each group — this is
# where the surgical-context term shows, in one number, how much "emergency" itself adds.
# **Table 3**: how much real (not filled-in) data each organ system had, by group — emergency
# patients usually arrive with much less pre-op data, and the model can learn that.

# %%
def section_emergency_scheduled():
    ids = list(TEST_IDS)
    y = test_result["labels"]; em = _emop_of(ids)
    grp = np.where(em > 0.5, "Emergency", "Scheduled")
    variants = [("single model (raw)", test_result["probs"])]
    if globals().get("CALIBRATOR") is not None:
        variants.append(("single model (recalibrated)", test_probs_calibrated))
    if ENSEMBLE_RESULT is not None:
        variants.append((f"ensemble of {len(ENSEMBLE_MODELS)}", ENSEMBLE_RESULT["probs"]))
    tabs = []
    for name, p in variants:
        t = metrics_by_group(y, p, grp, n_boot=min(CONFIG["N_BOOTSTRAP"], 500), seed=SEED,
                             threshold=best_threshold if name == "single model (raw)" else None)
        t.insert(0, "model", name); tabs.append(t)
    tab = pd.concat(tabs)
    PART13["emergency_scheduled_metrics"] = tab
    tab.to_csv(os.path.join(TABLE_DIR, "emergency_vs_scheduled_metrics.csv"))
    pd.set_option("display.width", 200)
    print("Table 1 -- performance by group (test set)"); print(tab.round(3).to_string())
    four = pd.Series(group_labels(em, y)).value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    print("\nTest patients in the four groups:", four.to_dict())

    out = collect_system_outputs(model, ids)
    if out["contrib"] is not None:
        C = pd.DataFrame(out["contrib"], columns=[_plain(t) for t in model.term_names])
        C["group"] = group_labels(em, y)
        mean_pts = C.groupby("group").mean().reindex([g for g in GROUP_ORDER if g in set(C["group"])])
        PART13["points_by_group"] = mean_pts
        mean_pts.to_csv(os.path.join(TABLE_DIR, "risk_points_by_group.csv"))
        print("\nTable 2 -- average risk points per term (positive = pushes towards 'died')")
        print(mean_pts.round(3).T.to_string())
        fig, ax = plt.subplots(figsize=(12, 4))
        mean_pts.T.plot(kind="bar", ax=ax, color=[GROUP_COLORS[g] for g in mean_pts.index])
        ax.axhline(0, color="gray", lw=0.8); ax.set_ylabel("average risk points (logit)")
        ax.set_title("Where the risk comes from: emergency/scheduled x died/survived (test set)")
        ax.tick_params(axis="x", rotation=30); plt.tight_layout(); plt.show()

    obs = {}
    for s in TIME_SERIES_SYSTEMS:
        frac = np.array([PATIENT_BUNDLE[i]["ts"][s]["mask"].mean() if PATIENT_BUNDLE[i]["ts"][s]["mask"].size else np.nan for i in ids])
        obs[_plain(s)] = {"Emergency": np.nanmean(frac[em > 0.5]) if (em > 0.5).any() else np.nan,
                          "Scheduled": np.nanmean(frac[em <= 0.5]) if (em <= 0.5).any() else np.nan}
    obs = pd.DataFrame(obs).T
    PART13["observed_share_by_group"] = obs
    obs.to_csv(os.path.join(TABLE_DIR, "observed_data_share_by_group.csv"))
    print("\nTable 3 -- share of the 24 time points that hold a REAL reading (rest are filled in)")
    print(obs.round(3).to_string())

run_section("13.6 emergency vs scheduled", section_emergency_scheduled)

# %% [markdown]
# ## 13.7 NEWS2 after filling missing data, and synthetic-patient consistency
#
# **(a) After filling.** NEWS2 is scored at every one of the 24 time points twice: once using
# only real readings (as every earlier check did), once using the filled-in timeline the model
# actually sees. If filling in often moves a time point into a different NEWS2 band, the
# filling method is inventing deterioration — or false reassurance. Reported: at time points
# that already had a real score, how often the band changes; at time points that had NO real
# score, what the filled-in scores look like next to real ones; and per patient, the worst
# score from real readings vs from the filled timeline.
#
# **(b) Synthetic patients.** A SMOTENC patient's NEWS2 number (in its facts card) is BLENDED
# from two real deaths, but its timeline is BORROWED from one real donor. Here the blended
# worst-NEWS2 is compared with the worst NEWS2 recomputed from the borrowed timeline. Real
# deaths go through the same comparison (their facts-card NEWS2 vs their own timeline) as the
# reference: synthetic patients should disagree no more than real ones do.

# %%
def _band(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.isnan(v), np.nan, np.where(v >= 7, 2, np.where(v >= 5, 1, 0)))

def section_news2_checks():
    if not NEWS2_VALIDATED:
        return "skipped: NEWS2 self-test failed"
    ids = (list(VAL_IDS) + list(TEST_IDS))[:3000]
    pt_real, pt_fill, pt_fill_only, pmax_real, pmax_fill = [], [], [], [], []
    for sid in ids:
        xs = {s: IMPUTED_BUNDLE[sid][s] for s in TIME_SERIES_SYSTEMS}
        ms = {s: PATIENT_BUNDLE[sid]["ts"][s]["mask"] for s in TIME_SERIES_SYSTEMS}
        ones = {s: np.ones_like(ms[s]) for s in TIME_SERIES_SYSTEMS}
        t_r, _ = _grid_news2(sid, xs, ms)
        t_f, _ = _grid_news2(sid, xs, ones)
        both = ~np.isnan(t_r) & ~np.isnan(t_f)
        pt_real += list(t_r[both]); pt_fill += list(t_f[both])
        pt_fill_only += list(t_f[np.isnan(t_r) & ~np.isnan(t_f)])
        pmax_real.append(np.nanmax(t_r) if (~np.isnan(t_r)).any() else np.nan)
        pmax_fill.append(np.nanmax(t_f) if (~np.isnan(t_f)).any() else np.nan)
    pt_real, pt_fill, pt_fill_only = map(np.array, (pt_real, pt_fill, pt_fill_only))
    pmax_real, pmax_fill = np.array(pmax_real), np.array(pmax_fill)
    res = {"time_points_with_real_score": int(len(pt_real)),
           "pct_band_changed_where_real_score_existed": float(np.mean(_band(pt_real) != _band(pt_fill))) if len(pt_real) else np.nan,
           "mean_abs_change_where_real_score_existed": float(np.mean(np.abs(pt_fill - pt_real))) if len(pt_real) else np.nan,
           "time_points_scored_only_after_filling": int(len(pt_fill_only)),
           "pct_high_band_real_points": float(np.mean(pt_real >= 7)) if len(pt_real) else np.nan,
           "pct_high_band_filled_only_points": float(np.mean(pt_fill_only >= 7)) if len(pt_fill_only) else np.nan}
    okp = ~np.isnan(pmax_real) & ~np.isnan(pmax_fill)
    res["patients_compared"] = int(okp.sum())
    res["pct_patients_worst_band_changed"] = float(np.mean(_band(pmax_real[okp]) != _band(pmax_fill[okp]))) if okp.any() else np.nan
    PART13["news2_after_filling"] = res
    print("(a) NEWS2 on real readings vs on the filled-in timeline"); print(pd.Series(res).round(4).to_string())
    print("Reading this: a few % band changes where a real score existed is expected (filling supplies the missing "
          "components of a partial set). The filled-only points should look like real ones -- many more 'high' "
          "scores there would mean filling is inventing deterioration.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(-0.5, 21.5, 1)
    if len(pt_real):
        axes[0].hist(pt_real, bins=bins, density=True, alpha=0.6, label="real readings", color="#2980b9")
    if len(pt_fill_only):
        axes[0].hist(pt_fill_only, bins=bins, density=True, alpha=0.6, label="filled-in only", color="#f39c12")
    axes[0].set_xlabel("NEWS2 at a time point"); axes[0].legend(); axes[0].set_title("NEWS2: real vs filled-in time points")
    axes[1].scatter(pmax_real[okp] + np.random.default_rng(0).uniform(-.2, .2, okp.sum()), pmax_fill[okp], s=6, alpha=0.4)
    axes[1].plot([0, 20], [0, 20], "--", color="gray"); axes[1].set_xlabel("worst NEWS2, real readings")
    axes[1].set_ylabel("worst NEWS2, filled timeline"); axes[1].set_title("Per patient: worst score before vs after filling")
    plt.tight_layout(); plt.show()

    if not (USE_NEWS2 and SYNTHETIC_STATIC_ROWS and SYNTHETIC_TS_DONORS):
        print("\n(b) synthetic-patient consistency: skipped -- needs NEWS2 in the static vector and synthetic patients this run.")
        return
    j = STATIC_FEATURE_NAMES.index("news2_max")
    ja = STATIC_FEATURE_NAMES.index("news2_available")
    def grid_max(sid):
        xs = {s: IMPUTED_BUNDLE[sid][s] for s in TIME_SERIES_SYSTEMS}
        ms = {s: PATIENT_BUNDLE[sid]["ts"][s]["mask"] for s in TIME_SERIES_SYSTEMS}
        t, _ = _grid_news2(sid, xs, ms)
        return np.nanmax(t) if (~np.isnan(t)).any() else np.nan
    syn_card, syn_grid = [], []
    for (vec, _lab), donor in zip(SYNTHETIC_STATIC_ROWS, SYNTHETIC_TS_DONORS):
        raw = static_scaler.inverse_transform(np.asarray(vec, dtype=float).reshape(1, -1))[0]
        if raw[ja] > 0.5:
            syn_card.append(raw[j]); syn_grid.append(grid_max(donor))
    real_card, real_grid = [], []
    for sid in TRAIN_IDS:
        if PATIENT_BUNDLE[sid]["label"] == 1 and PATIENT_BUNDLE[sid]["static"].get("news2_available", 0) > 0.5:
            real_card.append(PATIENT_BUNDLE[sid]["static"]["news2_max"]); real_grid.append(grid_max(sid))
    def summ(card, grid, who):
        card, grid = np.array(card, float), np.array(grid, float)
        ok = ~np.isnan(card) & ~np.isnan(grid)
        return {"who": who, "n": int(ok.sum()),
                "mean_abs_difference": float(np.mean(np.abs(card[ok] - grid[ok]))) if ok.any() else np.nan,
                "pct_band_disagree": float(np.mean(_band(card[ok]) != _band(grid[ok]))) if ok.any() else np.nan,
                "correlation": float(np.corrcoef(card[ok], grid[ok])[0, 1]) if ok.sum() > 2 else np.nan}, card[ok], grid[ok]
    s_syn, c1, g1 = summ(syn_card, syn_grid, "synthetic deaths (blended card vs borrowed timeline)")
    s_real, c2, g2 = summ(real_card, real_grid, "real training deaths (card vs own timeline) -- reference")
    tab = pd.DataFrame([s_syn, s_real]).set_index("who")
    PART13["news2_synthetic_consistency"] = tab
    print("\n(b) Synthetic-patient NEWS2 consistency"); print(tab.round(3).to_string())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(c2, g2, s=12, alpha=0.6, label="real deaths (reference)", color="#2980b9")
    ax.scatter(c1, g1, s=12, alpha=0.6, label="synthetic deaths", color="#f39c12")
    ax.plot([0, 20], [0, 20], "--", color="gray"); ax.set_xlabel("worst NEWS2 on the facts card")
    ax.set_ylabel("worst NEWS2 from the timeline"); ax.legend(); ax.set_title("Do synthetic patients' NEWS2 stories agree?")
    plt.tight_layout(); plt.show()

run_section("13.7 NEWS2 after filling + synthetic consistency", section_news2_checks)

# %% [markdown]
# ## 13.8 SHAP inside each organ system — which measurements drove its risk points?
#
# The model already reports how many risk points each organ system added. This splits each
# system's points across **that system's own inputs** (its measurements over time — value
# and whether it was measured — and its own facts: diagnoses, procedure, medicines, summary
# stats). Method: expected gradients, the estimator behind SHAP's `GradientExplainer`
# (`inspire_dnn/explain.py`), measured against a pool of typical training patients.
#
# **Reading the plots.** One row per input, most important at the top. One dot per test
# patient. Dots to the right pushed that system's risk UP; to the left, DOWN. Colour is the
# input's own value: red = high, blue = low. "creatinine: red dots on the right" = high
# creatinine raises the kidney points — what a clinician would expect. The whole-patient and
# surgical-context terms are explained the same way over their facts-card inputs.
#
# This is a check INSIDE each system — the project's main explanation is still the exact
# per-system breakdown. `completeness gap` near 0 means the attributions add up to the
# system's points (relative to a typical patient), as they should.

# %%
def section_shap():
    if not CONFIG["RUN_SHAP"]:
        return "skipped: CONFIG['RUN_SHAP'] is False"
    if model.fusion_strategy != "nam":
        return "skipped: needs NAM fusion"
    rng = np.random.default_rng(SEED + 13)
    t_dead = [s for s in TEST_IDS if PATIENT_BUNDLE[s]["label"] == 1]
    t_alive = [s for s in TEST_IDS if PATIENT_BUNDLE[s]["label"] == 0]
    n_alive = max(0, min(len(t_alive), CONFIG["SHAP_N_PATIENTS"] - len(t_dead)))
    explain_ids = t_dead + (list(rng.choice(t_alive, size=n_alive, replace=False)) if n_alive else [])
    bg_ids = list(rng.choice(list(TRAIN_IDS), size=min(CONFIG["SHAP_N_BACKGROUND"], len(TRAIN_IDS)), replace=False))
    bg = _to_device(collate_bundle([INSPIREDataset(bg_ids)[i] for i in range(len(bg_ids))]))
    died = np.array([PATIENT_BUNDLE[s]["label"] for s in explain_ids])
    chunk = 128
    batches = [_to_device(collate_bundle([INSPIREDataset(explain_ids[i:i + chunk])[k] for k in range(len(explain_ids[i:i + chunk]))]))
               for i in range(0, len(explain_ids), chunk)]
    ts_names = {s: PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][s]["feature_names"] for s in TIME_SERIES_SYSTEMS}
    results, long_rows, gaps = {}, [], {}
    for term in model.term_names:
        if term in model.organ_systems:
            ts_sys = [term] if term in model.ts_systems else []
            cols = list(model.system_static_idx[term])
        elif term == "surgical_context":
            ts_sys, cols = [], list(model.surgical_context_idx)
        else:   # whole-patient term: its facts-card inputs
            ts_sys = []
            keep = model.whole_patient.static_keep_idx if model.whole_patient is not None else model.context_idx
            cols = list(range(N_STATIC_FEATURES)) if keep is None else list(keep)
        if not ts_sys and not cols:
            continue
        A_parts, V_parts, gap_list = [], [], []
        for b in batches:
            r = expected_gradients(model, b, bg, term, ts_systems=ts_sys, static_cols=cols,
                                   n_samples=CONFIG["SHAP_N_SAMPLES"], seed=SEED)
            a_cols, v_cols = [], []
            for s in ts_sys:
                a_cols.append(r["ts"][s].cpu().numpy())
                x, m = b[f"{s}_x"].cpu().numpy(), b[f"{s}_mask"].cpu().numpy()
                v_cols.append(np.where(m.sum(1) > 0, (x * m).sum(1) / np.clip(m.sum(1), 1, None), np.nan))
            if cols:
                a_cols.append(r["static"].cpu().numpy()); v_cols.append(b["static"][:, cols].cpu().numpy())
            A_parts.append(np.concatenate(a_cols, axis=1)); V_parts.append(np.concatenate(v_cols, axis=1))
            gap_list.append(r.get("completeness_gap", np.nan))
        names = [f"{f} (over time)" for s in ts_sys for f in ts_names[s]] + [STATIC_FEATURE_NAMES[c] for c in cols]
        A, V = np.concatenate(A_parts), np.concatenate(V_parts)
        results[term] = (A, V, names)
        gaps[_plain(term)] = float(np.nanmean(gap_list))
        imp = mean_abs_importance(A, names)
        for rank, (f, v) in enumerate(imp.items(), start=1):
            jx = names.index(f)
            long_rows.append({"term": _plain(term), "feature": f, "rank": rank, "mean_abs_attribution": v,
                              "mean_attribution_died": float(A[died == 1, jx].mean()) if (died == 1).any() else np.nan,
                              "mean_attribution_survived": float(A[died == 0, jx].mean()) if (died == 0).any() else np.nan})
    PART13["shap"] = results
    tab = pd.DataFrame(long_rows)
    PART13["shap_table"] = tab
    tab.to_csv(os.path.join(TABLE_DIR, "shap_by_system.csv"), index=False)
    PART13["shap_completeness_gap"] = gaps
    print(f"Explained {len(explain_ids)} test patients ({int(died.sum())} deaths) against {len(bg_ids)} background patients.")
    print("Completeness gap per term (0 = attributions add up exactly; < 0.1 is good for a Monte-Carlo estimate):")
    print(pd.Series(gaps).round(3).to_string())
    terms = list(results)
    ncol = 2; nrow = int(np.ceil(len(terms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow), squeeze=False)
    for k, term in enumerate(terms):
        A, V, names = results[term]
        beeswarm(axes[k // ncol][k % ncol], A, V, names, top=10, title=f"{_plain(term)} -- top inputs")
    for k in range(len(terms), nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle("SHAP inside each risk term (dot = patient; right = raises that term's risk; red = high value)", y=1.0)
    plt.tight_layout(); plt.show()
    top3 = tab[tab["rank"] <= 3].groupby("term")["feature"].apply(lambda s: ", ".join(s))
    print("\nTop 3 inputs per term:"); print(top3.to_string())

run_section("13.8 SHAP inside each system", section_shap)

# %% [markdown]
# ## 13.9 Does a bigger embedding help? (the "score vs hidden size" figure)
#
# Retrains the full recipe at several embedding sizes (`CONFIG['SIZE_SWEEP']`, e.g. 16, 32,
# 64, 128 numbers per organ system), with `SIZE_SWEEP_SEEDS` random starts each, and plots
# test AUPRC against size with a band showing the spread across seeds — like Fig. 6 of the
# reference paper. Pre-training learns from every patient, so a bigger encoder has plenty to
# learn from; but the mortality part learns from only a few hundred deaths, so bigger is not
# automatically better. This figure is how that question gets answered with data. Off by
# default on Kaggle (it is many retrainings); on in the `isambard` profile.

# %%
def section_size_sweep():
    if not CONFIG["RUN_SIZE_SWEEP"]:
        return "skipped: CONFIG['RUN_SIZE_SWEEP'] is False (on in the isambard profile)"
    rows = []
    for size in CONFIG["SIZE_SWEEP"]:
        for k in range(CONFIG["SIZE_SWEEP_SEEDS"]):
            _t0 = time.time()
            m, best_val = train_full_model(seed=SEED + 1000 * (k + 1), encoder_size=size)
            r = evaluate(m, test_loader)
            rows.append({"encoder_size": size, "embed_dim": _ENCODER_PRESETS[size][0], "seed": k,
                         "n_params": sum(p.numel() for p in m.parameters()), "val_best": best_val,
                         "test_auprc": r["auprc"], "test_auroc": r["auroc"], "minutes": (time.time() - _t0) / 60})
            print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items()})
            del m
    df = pd.DataFrame(rows)
    PART13["size_sweep"] = df
    df.to_csv(os.path.join(TABLE_DIR, "embedding_size_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("test_auprc", "test_auroc")):
        g = df.groupby("embed_dim")[metric]
        mu, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(mu.index, mu.values, "o-", color="#08519c")
        ax.fill_between(mu.index, lo.values, hi.values, alpha=0.25, color="#08519c")
        ax.set_xscale("log", base=2); ax.set_xticks(mu.index); ax.set_xticklabels([str(int(v)) for v in mu.index])
        ax.set_xlabel("embedding size per organ system (numbers)"); ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"{metric.replace('test_', '').upper()} vs embedding size (band = spread over seeds)")
    plt.tight_layout(); plt.show()
    print(df.groupby("encoder_size")[["embed_dim", "n_params", "test_auprc", "test_auroc", "minutes"]].mean().round(3).to_string())

run_section("13.9 embedding-size sweep", section_size_sweep)

# %% [markdown]
# ## 13.10 Reuse test — are the embeddings useful for OTHER questions?
#
# The embeddings are frozen (no retraining) and a deliberately simple model (logistic
# regression, 5-fold cross-validation) is asked to predict, from them alone: 30-day death,
# **ICU admission after the operation** and **a long hospital stay** (more than 14 days
# after the operation, among survivors). The same test is run on the snapshots from before
# training and after pre-training. If the after-pre-training embeddings already predict ICU
# admission and long stay well — though they were never shown any outcome — the label-free
# pre-training learned real physiology. That is the meeting's "using the same feature
# vector for other tasks" point, measured.

# %%
def section_reuse_test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids = SNAPSHOT_IDS
    last_ops = [OPS_HISTORY[s][-1] for s in ids]
    icu = np.array([float(op.get("icuin_time") is not None and pd.notna(op.get("icuin_time"))
                          and op["icuin_time"] >= op["orin_time"]) for op in last_ops])
    dis = pd.to_numeric(pd.Series(_cohort_value(ids, "discharge_time_last")), errors="coerce").values
    orout = pd.to_numeric(pd.Series(_cohort_value(ids, "orout_time_last")), errors="coerce").values
    los_days = (dis - orout) / (24 * 60)
    long_stay = np.where(np.isnan(los_days) | (SNAP_DIED == 1), np.nan, (los_days > 14).astype(float))
    targets = {"30-day death": SNAP_DIED, "ICU admission after the operation": icu, "hospital stay > 14 days (survivors)": long_stay}
    rows = []
    for sn in [x for x in (_snap("before_training"), _snap("after_pretraining"), _snap("chosen_model")) if x is not None]:
        X = patient_embedding(sn)
        for tname, y in targets.items():
            ok = ~np.isnan(y)
            yy = y[ok]
            if ok.sum() < 50 or len(np.unique(yy)) < 2 or min(yy.sum(), (1 - yy).sum()) < 5:
                rows.append({"snapshot": _snapshot_title(sn), "target": tname, "auroc": np.nan, "auprc": np.nan,
                             "n": int(ok.sum()), "positives": int(np.nansum(yy))}); continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            p = cross_val_predict(clf, X[ok], yy, cv=cv, method="predict_proba")[:, 1]
            rows.append({"snapshot": _snapshot_title(sn), "target": tname, "auroc": roc_auc_score(yy, p),
                         "auprc": average_precision_score(yy, p), "rate": float(yy.mean()),
                         "n": int(ok.sum()), "positives": int(yy.sum())})
    tab = pd.DataFrame(rows)
    PART13["reuse_test"] = tab
    tab.to_csv(os.path.join(TABLE_DIR, "embedding_reuse_test.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    piv = tab.pivot(index="target", columns="snapshot", values="auroc")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    piv.plot(kind="bar", ax=ax); ax.axhline(0.5, color="gray", ls=":"); ax.set_ylabel("AUROC (5-fold, frozen embeddings)")
    ax.set_title("Reuse test: what else can a simple model read from the frozen embeddings?")
    ax.tick_params(axis="x", rotation=0); plt.tight_layout(); plt.show()

run_section("13.10 reuse test (other outcomes)", section_reuse_test)

# %% [markdown]
# ## 13.11 Part 13 status and saved files

# %%
PART13_STATUS = pd.Series(SECTION_STATUS, name="status")
print(PART13_STATUS.to_string())
print(f"\nSaved figures: {len(SAVED_FIGURES)} in {FIGURE_DIR}")
print(f"Tables: {sorted(os.listdir(TABLE_DIR))}")
print(f"Snapshots: {len(os.listdir(SNAPSHOT_DIR))} files in {SNAPSHOT_DIR}")
record_checkpoint("Part 13 -- looking inside the model")
