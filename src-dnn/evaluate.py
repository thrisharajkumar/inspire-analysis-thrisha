"""
Stage 7: evaluation -- held-out test metrics (AUROC, AUPRC as the metric to trust
first at this class rarity), PR-curve-based threshold selection, per-system NAM
contribution breakdown, and the attention audit. Converted from cells 112-120,
preserved verbatim. The per-system contribution breakdown (§11.2) now reflects
correlation-informed embeddings, not just the raw pooled ones, since the correlation
layer runs before fusion -- still fully interpretable, just a richer signal feeding
each system's term.
"""
from train import *

# --- from notebook cell 114 ---
test_result = evaluate(model, test_loader)
print(f"TEST  AUPRC={test_result['auprc']:.3f} (primary metric)  AUROC={test_result['auroc']:.3f} (reference)  "
      f"(n={len(test_result['labels'])}, positives={int(test_result['labels'].sum())})")

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

# --- from notebook cell 116 ---
if len(np.unique(test_result["labels"])) > 1:
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(test_result["labels"], test_result["probs"])
    f1_arr = np.where((prec_arr + rec_arr) > 0, 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12), 0.0)
    best_idx = int(np.argmax(f1_arr[:-1])) if len(f1_arr) > 1 else 0   # last point has no matching threshold
    best_threshold = float(thresh_arr[best_idx]) if len(thresh_arr) > 0 else 0.5

    print(f"Best-F1 threshold on this test set: {best_threshold:.3f} "
          f"(precision={prec_arr[best_idx]:.2f}, recall={rec_arr[best_idx]:.2f}, F1={f1_arr[best_idx]:.2f})")

    cm_tuned = confusion_matrix(test_result["labels"], (test_result["probs"] >= best_threshold).astype(int))
    print(f"\nConfusion matrix @ tuned threshold {best_threshold:.3f} [[TN, FP], [FN, TP]]:")
    print(cm_tuned)

    print("\nNearby thresholds, for picking a different clinical trade-off by hand:")
    for i in range(max(0, best_idx - 3), min(len(thresh_arr), best_idx + 4)):
        marker = " <- best F1" if i == best_idx else ""
        print(f"  threshold={thresh_arr[i]:.3f}  precision={prec_arr[i]:.2f}  recall={rec_arr[i]:.2f}{marker}")
else:
    print("Only one class present in the test split at this sample size -- skipping threshold tuning.")

# --- from notebook cell 118 ---
@torch.no_grad()
def per_system_breakdown(model, loader, n_patients=6):
    model.eval()
    rows = []
    count = 0
    for batch in loader:
        batch_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_dev)
        if out["per_system_signal"] is None:
            print("FUSION_STRATEGY is not 'nam' (or 'gated') -- no per-system decomposition available. "
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

# --- from notebook cell 120 ---
@torch.no_grad()
def get_attention_weights(model, batch_item, system):
    # Runs one patient through just the named system's transformer and extracts the first
    # layer's self-attention weights (averaged over heads) -- a minimal, dependency-light
    # audit hook rather than a full attention-rollout implementation.
    encoder = model.encoders[system].transformer.layers[0].self_attn
    x = batch_item[f"{system}_x"].unsqueeze(0).to(DEVICE)
    mask = batch_item[f"{system}_mask"].unsqueeze(0).to(DEVICE)
    if x.shape[-1] == 0:
        return None
    combined = torch.cat([x, mask], dim=-1)
    h = model.encoders[system].input_proj(combined) + model.encoders[system].pos_embedding
    _, attn_weights = encoder(h, h, h, need_weights=True, average_attn_weights=True)
    return attn_weights.squeeze(0).cpu().numpy()   # [T, T]

example_item = test_dataset[0]
example_system = "renal"
attn = get_attention_weights(model, example_item, example_system)
if attn is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(attn, cmap="viridis"); axes[0].set_title(f"{example_system} attention (layer 0, head-avg)")
    axes[0].set_xlabel("attended-to timestep"); axes[0].set_ylabel("query timestep")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    mask = example_item[f"{example_system}_mask"].numpy()
    axes[1].imshow(mask.T, aspect="auto", cmap="Greys"); axes[1].set_title("observed(black)/imputed(white) mask")
    axes[1].set_xlabel("timestep"); axes[1].set_ylabel("feature index")
    plt.tight_layout(); plt.show()
    print("A genuinely informative audit would check whether attention concentrates on OBSERVED "
          "(not imputed) timesteps more than chance -- worth computing directly once run at full scale.")