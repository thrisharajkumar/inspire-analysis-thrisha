# %% [markdown]
# # Part 10 — Training: label-free pre-training, then gentle fine-tuning (v3)
#
# **Phase 1 — pre-training, no labels, real training patients only.** Two tasks at once:
# every organ system rebuilds its own data from its own summary (as before), and the
# **masked-system task** hides one system per patient and rebuilds it from the other seven
# through the coupling layer. That is how the inter-system layer learns real relationships
# from every patient instead of from a few hundred deaths.
#
# **Phase 2 — fine-tuning for 30-day mortality.** AdamW with weight decay; the pre-trained
# encoder moves 10x more gently than the new parts (`FINETUNE_ENCODER_LR_MULT`); balanced
# batches; organ-system dropout; best epoch chosen on validation AUPRC. Training AUPRC on a
# fixed set of real training patients is tracked alongside validation — a widening gap is
# the clearest sign of memorising.
#
# Both phases checkpoint every few epochs and resume automatically if the cell is re-run.
# `train_full_model()` (bottom) repeats the exact same recipe for the ensemble, ablations
# and learning curve in Part 11.
#
# **v4 — the model's diary.** While it trains, the notebook now also writes down:
# - every epoch: validation AUROC/AUPRC for ALL patients, and separately for EMERGENCY and
#   SCHEDULED patients (`EPOCH_LOG`), starting from *before any training*;
# - at chosen epochs: a **snapshot** of every snapshot patient's embeddings (the 32-number
#   notes per organ system), their risk points and their predicted risk (`SNAPSHOTS`, also
#   saved to `OUTPUT_DIR/snapshots/`) — Part 13 turns these into the before/after maps;
# - at the end: the **final model**, saved with everything needed to reload it, and a
#   reload check proving the saved file gives identical predictions.

# %%
def save_checkpoint(path, **state):
    torch.save(state, path)

def load_checkpoint_if_exists(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=False)
        except Exception as e:
            print(f"Found a checkpoint at {path} but failed to load it ({e}) -- starting fresh.")
    return None

def _to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

import contextlib

def _autocast():
    # v4: bf16 mixed precision for TRAINING only (evaluation stays in full precision).
    return torch.autocast("cuda", dtype=torch.bfloat16) if USE_AMP else contextlib.nullcontext()

def run_pretrain_epoch(model, loader, optimizer):
    model.train()
    tot, own, ms, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = _to_device(batch)
        optimizer.zero_grad()
        with _autocast():
            loss, parts = model.pretrain_loss(batch, CONFIG["MASKED_SYSTEM_WEIGHT"])
        loss.backward()
        optimizer.step()
        tot += float(loss); own += parts["own"]; ms += parts["masked_system"]; n += 1
    return tot / max(n, 1), own / max(n, 1), ms / max(n, 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        out = model(_to_device(batch))
        all_logits.append(out["logit"].cpu().numpy())
        all_labels.append(batch["label"].numpy())
    logits, labels = np.concatenate(all_logits), np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    result = {"probs": probs, "labels": labels, "logits": logits}
    both = len(np.unique(labels)) > 1
    result["auroc"] = roc_auc_score(labels, probs) if both else float("nan")
    result["auprc"] = average_precision_score(labels, probs) if both else float("nan")
    return result

def run_finetune_epoch(model, loader, optimizer, pos_weight, use_focal, gamma):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch)
        optimizer.zero_grad()
        with _autocast():
            out = model(batch)
            loss = mortality_loss(out["logit"].float(), batch["label"], pos_weight, use_focal, gamma)
            loss = loss + CONFIG["ARBITRATION_REG"] * out["aux_loss"].float()
        loss.backward()
        optimizer.step()
        total += float(loss); n += 1
    return total / max(n, 1)

def make_pretrain_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

def make_finetune_optimizer(model):
    enc_ids = {id(p) for p in model.encoder_parameters()}
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    base = CONFIG["LR"] * 0.5
    return torch.optim.AdamW([{"params": enc, "lr": base * CONFIG["FINETUNE_ENCODER_LR_MULT"]},
                              {"params": rest, "lr": base}], weight_decay=CONFIG["WEIGHT_DECAY"])

_rng_tm = np.random.default_rng(SEED)
_real_train = [s for s in EFFECTIVE_TRAIN_IDS]
TRAIN_METRIC_IDS = list(_rng_tm.choice(_real_train, size=min(CONFIG["TRAIN_METRIC_SUBSAMPLE"], len(_real_train)), replace=False))
train_metric_loader = DataLoader(INSPIREDataset(TRAIN_METRIC_IDS), batch_size=CONFIG["BATCH_SIZE"],
                                 shuffle=False, collate_fn=collate_bundle, **LOADER_KW)
print(f"Training-set AUPRC will be tracked on {len(TRAIN_METRIC_IDS)} real training patients "
      f"({int(sum(PATIENT_BUNDLE[s]['label'] for s in TRAIN_METRIC_IDS))} deaths).")

# ---- v4: the model's diary -- per-epoch metrics by emergency/scheduled, and snapshots ----
def _emop_of(ids):
    return np.array([float(COHORT_INDEXED.loc[s, "emop"]) if pd.notna(COHORT_INDEXED.loc[s, "emop"]) else 0.0 for s in ids])

VAL_EMOP = _emop_of(VAL_IDS)
EPOCH_LOG = []   # one dict per logged step: phase, epoch, val metrics (all / emergency / scheduled), losses

def _group_metrics(labels, probs, emop, prefix="val"):
    out = {}
    for name, sel in (("all", np.ones(len(labels), bool)), ("emergency", emop > 0.5), ("scheduled", emop <= 0.5)):
        y, p = labels[sel], probs[sel]
        ok = sel.any() and len(np.unique(y)) > 1
        out[f"{prefix}_auroc_{name}"] = float(roc_auc_score(y, p)) if ok else np.nan
        out[f"{prefix}_auprc_{name}"] = float(average_precision_score(y, p)) if ok else np.nan
        out[f"{prefix}_n_deaths_{name}"] = int(y.sum()) if sel.any() else 0
    return out

def log_epoch(phase, epoch, val_res=None, **extra):
    row = {"phase": phase, "epoch": epoch, **extra}
    if val_res is not None:
        row.update(_group_metrics(val_res["labels"], val_res["probs"], VAL_EMOP))
    EPOCH_LOG.append(row)
    return row

# Snapshot patients: every validation death + a random sample of validation survivors.
_rng_snap = np.random.default_rng(SEED + 7)
_val_dead = [s for s in VAL_IDS if PATIENT_BUNDLE[s]["label"] == 1]
_val_alive = [s for s in VAL_IDS if PATIENT_BUNDLE[s]["label"] == 0]
_n_alive = max(0, min(len(_val_alive), CONFIG["SNAPSHOT_N_PATIENTS"] - len(_val_dead)))
SNAPSHOT_IDS = _val_dead + list(_rng_snap.choice(_val_alive, size=_n_alive, replace=False)) if _n_alive else list(_val_dead)
snapshot_loader = DataLoader(INSPIREDataset(SNAPSHOT_IDS), batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                             collate_fn=collate_bundle, **LOADER_KW)
SNAPSHOTS = []   # list of dicts (tag, phase, epoch, arrays) -- also written to OUTPUT_DIR/snapshots/*.npz

@torch.no_grad()
def take_snapshot(m, tag, phase, epoch):
    # Records, for every snapshot patient: each organ system's embedding BEFORE and AFTER the
    # whole-patient + coupling layers, the whole-patient summary, the surgical-context summary,
    # every term's risk points, and the predicted risk. float16 on disk to keep files small.
    try:
        m.eval()
        own, final, g, sc, contrib, logit = [], [], [], [], [], []
        for batch in snapshot_loader:
            out = m(_to_device(batch))
            own.append(torch.stack([out["embeddings_own"][s] for s in m.organ_systems], 1).float().cpu().numpy())
            final.append(torch.stack([out["embeddings"][s] for s in m.organ_systems], 1).float().cpu().numpy())
            g.append(out["static_embedding"].float().cpu().numpy())
            if out.get("surgical_context_embedding") is not None:
                sc.append(out["surgical_context_embedding"].float().cpu().numpy())
            if out["per_system_signal"] is not None:
                contrib.append(torch.stack([out["per_system_signal"][t] for t in m.term_names], 1).float().cpu().numpy())
            logit.append(out["logit"].float().cpu().numpy())
        cat = lambda xs: np.concatenate(xs) if xs else None
        snap = {"tag": tag, "phase": phase, "epoch": epoch, "systems_own": cat(own), "systems": cat(final),
                "whole_patient": cat(g), "surgical_context": cat(sc), "contrib": cat(contrib), "logit": cat(logit),
                "term_names": list(m.term_names), "organ_systems": list(m.organ_systems)}
        SNAPSHOTS.append(snap)
        np.savez_compressed(os.path.join(SNAPSHOT_DIR, f"snapshot_{len(SNAPSHOTS):02d}_{tag}.npz"),
                            **{k: (v.astype(np.float16) if isinstance(v, np.ndarray) else np.array(v))
                               for k, v in snap.items() if v is not None},
                            subject_id=np.array(SNAPSHOT_IDS))
        return snap
    except Exception as e:   # a diary entry must never stop training -- but it is reported loudly
        print(f"!! SNAPSHOT '{tag}' FAILED ({type(e).__name__}: {e}) -- training continues; Part 13 will have one fewer picture.")
        return None

print(f"Snapshots will follow {len(SNAPSHOT_IDS)} validation patients ({len(_val_dead)} deaths, "
      f"{int(_emop_of(_val_dead).sum()) if _val_dead else 0} of them emergency).")

PRETRAIN_CKPT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], f"pretrain_checkpoint_{ARCHITECTURE_FINGERPRINT}.pt")
FINETUNE_CKPT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], f"finetune_checkpoint_{ARCHITECTURE_FINGERPRINT}.pt")

# %%
pretrain_optimizer = make_pretrain_optimizer(model)
pretrain_history = {"total": [], "own": [], "masked_system": []}
start_epoch_pretrain = 0
_ckpt = load_checkpoint_if_exists(PRETRAIN_CKPT_PATH)
if _ckpt is not None and _ckpt.get("phase") == "pretrain":
    try:
        model.load_state_dict(_ckpt["model_state"]); pretrain_optimizer.load_state_dict(_ckpt["optimizer_state"])
        pretrain_history = _ckpt["history"]; start_epoch_pretrain = _ckpt["epoch"] + 1
        print(f"Resuming pre-training at epoch {start_epoch_pretrain}")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"Checkpoint does not match this model ({e}) -- starting fresh.")

if start_epoch_pretrain == 0:
    # v4: "before any training" -- random dials. Everything later is compared with this.
    _r0 = evaluate(model, val_loader)
    log_epoch("before training", -1, _r0)
    take_snapshot(model, "before_training", "before training", -1)
    print(f"Before any training: val AUPRC {_r0['auprc']:.3f} (a random model scores about the death rate, "
          f"{_r0['labels'].mean():.3f})")

print("Phase 1: label-free pre-training (own-data reconstruction + masked-system prediction)")
_best, _stale = float("inf"), 0
for epoch in range(start_epoch_pretrain, CONFIG["EPOCHS_PRETRAIN"]):
    tot, own, ms = run_pretrain_epoch(model, pretrain_loader, pretrain_optimizer)
    for k, v in zip(("total", "own", "masked_system"), (tot, own, ms)):
        pretrain_history[k].append(v)
    log_epoch("pre-training", epoch, None, pretrain_own=own, pretrain_masked_system=ms)
    if epoch % max(CONFIG["EPOCHS_PRETRAIN"] // 6, 1) == 0 or epoch == CONFIG["EPOCHS_PRETRAIN"] - 1:
        print(f"  epoch {epoch:3d}  own-data={own:.4f}  masked-system={ms:.4f}")
    if (epoch + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == CONFIG["EPOCHS_PRETRAIN"] - 1:
        save_checkpoint(PRETRAIN_CKPT_PATH, phase="pretrain", epoch=epoch, model_state=model.state_dict(),
                        optimizer_state=pretrain_optimizer.state_dict(), history=pretrain_history)
    if tot < _best - 1e-5:
        _best, _stale = tot, 0
    else:
        _stale += 1
    if _stale >= CONFIG["EARLY_STOPPING_PATIENCE"]:
        print(f"  early stopping at epoch {epoch}")
        break

_r_pt = evaluate(model, val_loader)
log_epoch("after pre-training", -1, _r_pt)
take_snapshot(model, "after_pretraining", "after pre-training", -1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(pretrain_history["own"], color="#0a7d6e", label="own-data reconstruction")
ax.plot(pretrain_history["masked_system"], color="#8e44ad", label="masked-system prediction")
ax.set_title("Phase 1: label-free pre-training"); ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE"); ax.legend()
plt.tight_layout(); plt.show()
print("The masked-system curve falling means the coupling layer is learning to predict a hidden organ "
      "system from the others -- i.e. it is finding real inter-system relationships before it ever sees a label.")

# %%
finetune_optimizer = make_finetune_optimizer(model)
history = {"train_loss": [], "val_auroc": [], "val_auprc": [], "train_auprc": []}
_SEL = CONFIG["MODEL_SELECTION_METRIC"]
best_val_auroc, best_state, start_epoch_finetune = -1.0, None, 0   # name kept for checkpoint compatibility
_ckpt = load_checkpoint_if_exists(FINETUNE_CKPT_PATH)
if _ckpt is not None and _ckpt.get("phase") == "finetune":
    try:
        model.load_state_dict(_ckpt["model_state"]); finetune_optimizer.load_state_dict(_ckpt["optimizer_state"])
        history, best_val_auroc, best_state = _ckpt["history"], _ckpt["best_val_auroc"], _ckpt["best_state"]
        start_epoch_finetune = _ckpt["epoch"] + 1
        print(f"Resuming fine-tuning at epoch {start_epoch_finetune} (best val {_SEL.upper()} so far {best_val_auroc:.3f})")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"Checkpoint does not match this model ({e}) -- starting fresh.")

print("\nPhase 2: fine-tuning on 30-day mortality")
_stale = 0
for epoch in range(start_epoch_finetune, CONFIG["EPOCHS_FINETUNE"]):
    train_dataset.epoch_seed = 1000 + epoch
    train_loss = run_finetune_epoch(model, train_loader, finetune_optimizer, POS_WEIGHT, CONFIG["USE_FOCAL_LOSS"], CONFIG["FOCAL_GAMMA"])
    val_result = evaluate(model, val_loader)
    tr_result = evaluate(model, train_metric_loader)
    history["train_loss"].append(train_loss); history["val_auroc"].append(val_result["auroc"])
    history["val_auprc"].append(val_result["auprc"]); history["train_auprc"].append(tr_result["auprc"])
    log_epoch("fine-tuning", epoch, val_result, train_loss=train_loss, train_auprc_all=tr_result["auprc"])
    if epoch in CONFIG["SNAPSHOT_FINETUNE_EPOCHS"]:
        take_snapshot(model, f"finetune_epoch_{epoch:03d}", "fine-tuning", epoch)
    if not math.isnan(val_result[_SEL]) and val_result[_SEL] > best_val_auroc:
        best_val_auroc, best_state, _stale = val_result[_SEL], {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        _stale += 1
    if epoch % max(CONFIG["EPOCHS_FINETUNE"] // 8, 1) == 0 or epoch == CONFIG["EPOCHS_FINETUNE"] - 1:
        print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  train_auprc={tr_result['auprc']:.3f}  "
              f"val_auprc={val_result['auprc']:.3f}  val_auroc={val_result['auroc']:.3f}")
    if (epoch + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == CONFIG["EPOCHS_FINETUNE"] - 1:
        save_checkpoint(FINETUNE_CKPT_PATH, phase="finetune", epoch=epoch, model_state=model.state_dict(),
                        optimizer_state=finetune_optimizer.state_dict(), history=history,
                        best_val_auroc=best_val_auroc, best_state=best_state)
    if _stale >= CONFIG["EARLY_STOPPING_PATIENCE"]:
        print(f"  early stopping at epoch {epoch} -- val {_SEL.upper()} hasn't improved in {CONFIG['EARLY_STOPPING_PATIENCE']} epochs")
        break

LAST_EPOCH = len(history["val_auprc"]) - 1
if LAST_EPOCH >= 0 and LAST_EPOCH not in CONFIG["SNAPSHOT_FINETUNE_EPOCHS"]:
    take_snapshot(model, f"finetune_epoch_{LAST_EPOCH:03d}_last", "fine-tuning", LAST_EPOCH)

if best_state is not None:
    model.load_state_dict(best_state)
    BEST_EPOCH = int(np.nanargmax(history[f"val_{_SEL}"]))
    print(f"\nRestored best epoch {BEST_EPOCH} (val {_SEL.upper()}={best_val_auroc:.3f}).")
else:
    BEST_EPOCH = len(history["val_auprc"]) - 1
    print("\nNo validation improvement recorded -- using final-epoch weights.")

take_snapshot(model, "chosen_model", "chosen model", BEST_EPOCH)
EPOCH_LOG_DF = pd.DataFrame(EPOCH_LOG)
EPOCH_LOG_DF.to_csv(os.path.join(TABLE_DIR, "epoch_log.csv"), index=False)
print(f"Diary: {len(EPOCH_LOG_DF)} logged steps, {len(SNAPSHOTS)} embedding snapshots "
      f"({', '.join(sn['tag'] for sn in SNAPSHOTS)})")

record_checkpoint("Parts 9-10 -- model build + two-phase training")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], color="#c0392b")
axes[0].set_title("Phase 2: fine-tuning loss"); axes[0].set_xlabel("epoch")
axes[1].plot(history["train_auprc"], label="training AUPRC (real patients)", color="#e67e22")
axes[1].plot(history["val_auprc"], label="validation AUPRC", color="#8e44ad")
axes[1].axvline(BEST_EPOCH, color="gray", linestyle=":", label="chosen epoch")
axes[1].set_title("Learning or memorising? training vs. validation"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()
TRAIN_VAL_GAP = float(history["train_auprc"][BEST_EPOCH] - history["val_auprc"][BEST_EPOCH])
print(f"At the chosen epoch: training AUPRC {history['train_auprc'][BEST_EPOCH]:.3f} vs validation "
      f"{history['val_auprc'][BEST_EPOCH]:.3f} (gap {TRAIN_VAL_GAP:+.3f}). A gap that keeps widening after the "
      f"chosen epoch is memorisation; early stopping is what protects the chosen model from it.")
_n_val_pos = int(sum(PATIENT_BUNDLE[s]["label"] for s in VAL_IDS))
print(f"Validation set: {len(VAL_IDS)} patients, {_n_val_pos} deaths -- read the trend, not single epochs.")

# %% [markdown]
# ## 10.3 Saving the final model, and proving the saved file works (v4)
#
# The chosen model is saved with everything needed to use it again without re-running the
# notebook: the weights, the CONFIG, the feature names and which organ system owns each, the
# training-set scaling numbers (so a new patient is put on the same ruler), and the training
# diary. Then it is loaded back into a brand-new model and must give **identical**
# predictions on the validation patients — otherwise the file is not trustworthy.

# %%
def _model_payload(m):
    return {
        "model_state": {k: v.detach().cpu() for k, v in m.state_dict().items()},
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT, "version": "v4",
        "config": dict(CONFIG), "run_profile": RUN_PROFILE,
        "static_feature_names": list(STATIC_FEATURE_NAMES),
        "ts_feature_names": {s: PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][s]["feature_names"] for s in TIME_SERIES_SYSTEMS},
        "ts_feature_counts": dict(TS_FEATURE_COUNTS), "system_static_idx": dict(SYSTEM_STATIC_IDX),
        "context_idx": list(CONTEXT_IDX), "surgical_context_idx": list(SURGICAL_CONTEXT_IDX),
        "static_scaler_mean": static_scaler.mean_, "static_scaler_scale": static_scaler.scale_,
        "ts_means": {f"{k[0]}|{k[1]}": float(v["mean"]) for k, v in STATS.items()},
        "ts_stds": {f"{k[0]}|{k[1]}": float(v) for k, v in TRAIN_STD.items()},
        "best_epoch": BEST_EPOCH, "history": history, "epoch_log": EPOCH_LOG,
        "torch_version": torch.__version__,
    }

def save_final_model(m, name="final_model"):
    path = os.path.join(MODEL_DIR, f"{name}_{ARCHITECTURE_FINGERPRINT}.pt")
    torch.save(_model_payload(m), path)
    return path

def load_inspire_model(path, **build_kwargs):
    # Rebuilds the model with THIS notebook's data layout and loads the saved weights.
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    assert payload["static_feature_names"] == list(STATIC_FEATURE_NAMES), "static features differ from this run"
    m = build_model(**build_kwargs)
    m.load_state_dict(payload["model_state"])
    m.eval()
    return m, payload

FINAL_MODEL_PATH, MODEL_RELOAD_OK = None, None
if CONFIG["SAVE_FINAL_MODEL"]:
    FINAL_MODEL_PATH = save_final_model(model)
    _reloaded, _ = load_inspire_model(FINAL_MODEL_PATH)
    _a = evaluate(model, val_loader)["logits"]; _b = evaluate(_reloaded, val_loader)["logits"]
    MODEL_RELOAD_OK = bool(np.allclose(_a, _b, atol=1e-5))
    print(f"Saved the chosen model to {FINAL_MODEL_PATH} ({os.path.getsize(FINAL_MODEL_PATH) / 1e6:.1f} MB)")
    print(f"Reload check -- the saved file gives identical validation predictions: {MODEL_RELOAD_OK}")
    del _reloaded
else:
    print("CONFIG['SAVE_FINAL_MODEL']=False -- model not saved.")

# %%
def train_full_model(seed=SEED, train_ds=None, verbose=False, **model_kwargs):
    # The exact recipe above (pre-train -> gentle fine-tune -> best val epoch), without
    # checkpoint files. Used by the ensemble, the ablations and the learning curve.
    m = build_model(seed=seed, **model_kwargs)
    torch.manual_seed(seed)
    ds = train_ds or train_dataset
    real_ids = ds.subject_ids
    pl = DataLoader(INSPIREDataset(real_ids), batch_size=min(CONFIG["BATCH_SIZE"], len(real_ids)), shuffle=True,
                    collate_fn=collate_bundle, generator=torch.Generator().manual_seed(seed), **LOADER_KW)
    opt = make_pretrain_optimizer(m); best, stale = float("inf"), 0
    for ep in range(CONFIG["EPOCHS_PRETRAIN"]):
        tot, _, _ = run_pretrain_epoch(m, pl, opt)
        best, stale = (tot, 0) if tot < best - 1e-5 else (best, stale + 1)
        if stale >= CONFIG["EARLY_STOPPING_PATIENCE"]:
            break
    tl = make_train_loader(ds, seed=seed)
    opt = make_finetune_optimizer(m); best, best_state, stale = -1.0, None, 0
    for ep in range(CONFIG["EPOCHS_FINETUNE"]):
        ds.epoch_seed = 1000 + ep
        run_finetune_epoch(m, tl, opt, POS_WEIGHT, CONFIG["USE_FOCAL_LOSS"], CONFIG["FOCAL_GAMMA"])
        v = evaluate(m, val_loader)[_SEL]
        if not math.isnan(v) and v > best:
            best, best_state, stale = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= CONFIG["EARLY_STOPPING_PATIENCE"]:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    if verbose:
        print(f"  seed {seed}: best val {_SEL.upper()} {best:.3f}")
    return m, best
