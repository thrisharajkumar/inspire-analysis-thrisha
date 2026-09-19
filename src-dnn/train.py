"""
Stage 6: training -- two-phase pattern (autoencoder pre-training, then supervised
fine-tuning), with checkpoint save/resume keyed to an architecture fingerprint (so a
checkpoint from a differently-shaped model, e.g. a smoke test that happened to miss a
rare department, is never silently loaded into a mismatched model). Converted from
cells 107-111, preserved verbatim -- unaffected by the correlation-layer addition,
since it only touches out["logit"], not the internals the correlation layer changed.

If your session disconnects mid-run: just re-run this file after re-running
config.py through model.py -- checkpoint resume picks up where it left off.
"""
from model import *

# --- from notebook cell 108 ---
def save_checkpoint(path, **state):
    torch.save(state, path)

def load_checkpoint_if_exists(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=DEVICE)
        except Exception as e:
            print(f"Found a checkpoint at {path} but failed to load it ({e}) -- starting fresh.")
    return None

PRETRAIN_CKPT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], f"pretrain_checkpoint_{ARCHITECTURE_FINGERPRINT}.pt")
FINETUNE_CKPT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], f"finetune_checkpoint_{ARCHITECTURE_FINGERPRINT}.pt")

def run_pretrain_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch)
        loss = 0.0
        for system in TIME_SERIES_SYSTEMS:
            loss = loss + masked_reconstruction_loss(out["recon_preds"][system], batch[f"{system}_x"], batch[f"{system}_mask"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        n_batches += 1
    return total_loss / max(n_batches, 1)

pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
pretrain_losses = []
start_epoch_pretrain = 0

_ckpt = load_checkpoint_if_exists(PRETRAIN_CKPT_PATH)
if _ckpt is not None and _ckpt.get("phase") == "pretrain":
    try:
        model.load_state_dict(_ckpt["model_state"])
        pretrain_optimizer.load_state_dict(_ckpt["optimizer_state"])
        pretrain_losses = _ckpt["pretrain_losses"]
        start_epoch_pretrain = _ckpt["epoch"] + 1
        print(f"Resuming pre-training from checkpoint at epoch {start_epoch_pretrain} "
              f"(found at {PRETRAIN_CKPT_PATH})")
    except RuntimeError as e:
        print(f"WARNING: checkpoint at {PRETRAIN_CKPT_PATH} does not match this model's shape "
              f"({e}) -- ignoring it and starting fresh. (The architecture-fingerprint filename "
              f"should prevent this; seeing it anyway means something else changed too.)")

print("Phase 1: autoencoder pre-training (label-free)")
_best_pretrain_loss, _epochs_no_improve = float("inf"), 0
for epoch in range(start_epoch_pretrain, CONFIG["EPOCHS_PRETRAIN"]):
    train_dataset.epoch_seed = epoch   # fresh jitter/time-mask draw each epoch
    loss = run_pretrain_epoch(model, train_loader, pretrain_optimizer)
    pretrain_losses.append(loss)
    if epoch % max(CONFIG["EPOCHS_PRETRAIN"] // 6, 1) == 0 or epoch == CONFIG["EPOCHS_PRETRAIN"] - 1:
        print(f"  epoch {epoch:3d}  reconstruction_loss={loss:.4f}")
    if (epoch + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == CONFIG["EPOCHS_PRETRAIN"] - 1:
        save_checkpoint(PRETRAIN_CKPT_PATH, phase="pretrain", epoch=epoch,
                         model_state=model.state_dict(), optimizer_state=pretrain_optimizer.state_dict(),
                         pretrain_losses=pretrain_losses)

    if loss < _best_pretrain_loss - 1e-5:
        _best_pretrain_loss, _epochs_no_improve = loss, 0
    else:
        _epochs_no_improve += 1
    if _epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
        print(f"  early stopping at epoch {epoch} -- reconstruction loss hasn't improved in "
              f"{CONFIG['EARLY_STOPPING_PATIENCE']} epochs (saves time without sacrificing quality: "
              f"the epochs being skipped weren't helping anyway)")
        break

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(pretrain_losses, color="#0a7d6e")
ax.set_title("Phase 1: pre-training reconstruction loss"); ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE")
plt.tight_layout(); plt.show()

# --- from notebook cell 109 ---
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        all_logits.append(out["logit"].cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    result = {"probs": probs, "labels": labels}
    if len(np.unique(labels)) > 1:   # AUROC/AUPRC undefined with only one class present
        result["auroc"] = roc_auc_score(labels, probs)
        result["auprc"] = average_precision_score(labels, probs)
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result

def run_finetune_epoch(model, loader, optimizer, pos_weight, use_focal, gamma):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch)
        loss = mortality_loss(out["logit"], batch["label"], pos_weight, use_focal, gamma)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        n_batches += 1
    return total_loss / max(n_batches, 1)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"] * 0.5)   # smaller LR: don't destroy the pre-trained encoders
history = {"train_loss": [], "val_auroc": [], "val_auprc": []}
best_val_auroc, best_state = -1.0, None
start_epoch_finetune = 0

_ckpt = load_checkpoint_if_exists(FINETUNE_CKPT_PATH)
if _ckpt is not None and _ckpt.get("phase") == "finetune":
    try:
        model.load_state_dict(_ckpt["model_state"])
        finetune_optimizer.load_state_dict(_ckpt["optimizer_state"])
        history = _ckpt["history"]
        best_val_auroc = _ckpt["best_val_auroc"]
        best_state = _ckpt["best_state"]
        start_epoch_finetune = _ckpt["epoch"] + 1
        print(f"Resuming fine-tuning from checkpoint at epoch {start_epoch_finetune} "
              f"(found at {FINETUNE_CKPT_PATH}, best val AUROC so far={best_val_auroc:.3f})")
    except RuntimeError as e:
        print(f"WARNING: checkpoint at {FINETUNE_CKPT_PATH} does not match this model's shape "
              f"({e}) -- ignoring it and starting fresh.")

print("\nPhase 2: supervised fine-tuning on the mortality objective")
_epochs_no_improve_ft = 0
for epoch in range(start_epoch_finetune, CONFIG["EPOCHS_FINETUNE"]):
    train_dataset.epoch_seed = 1000 + epoch   # offset from pretrain's seeds, still deterministic per run
    train_loss = run_finetune_epoch(model, train_loader, finetune_optimizer, POS_WEIGHT, CONFIG["USE_FOCAL_LOSS"], CONFIG["FOCAL_GAMMA"])
    val_result = evaluate(model, val_loader)
    history["train_loss"].append(train_loss)
    history["val_auroc"].append(val_result["auroc"])
    history["val_auprc"].append(val_result["auprc"])

    if not math.isnan(val_result["auroc"]) and val_result["auroc"] > best_val_auroc:
        best_val_auroc = val_result["auroc"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        _epochs_no_improve_ft = 0
    else:
        _epochs_no_improve_ft += 1

    if epoch % max(CONFIG["EPOCHS_FINETUNE"] // 8, 1) == 0 or epoch == CONFIG["EPOCHS_FINETUNE"] - 1:
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_auroc={val_result['auroc']:.3f}  val_auprc={val_result['auprc']:.3f}")

    if (epoch + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == CONFIG["EPOCHS_FINETUNE"] - 1:
        save_checkpoint(FINETUNE_CKPT_PATH, phase="finetune", epoch=epoch,
                         model_state=model.state_dict(), optimizer_state=finetune_optimizer.state_dict(),
                         history=history, best_val_auroc=best_val_auroc, best_state=best_state)

    if _epochs_no_improve_ft >= CONFIG["EARLY_STOPPING_PATIENCE"]:
        print(f"  early stopping at epoch {epoch} -- val AUROC hasn't improved in "
              f"{CONFIG['EARLY_STOPPING_PATIENCE']} epochs")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint (val AUROC={best_val_auroc:.3f}).")
else:
    print("\nNo val-AUROC improvement was recorded (can happen with very few val positives) -- using final-epoch weights.")

record_checkpoint("Parts 9-10 -- model build + two-phase training")

# --- from notebook cell 111 ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], color="#c0392b")
axes[0].set_title("Phase 2: fine-tuning loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")

axes[1].plot(history["val_auroc"], label="val AUROC", color="#2980b9")
axes[1].plot(history["val_auprc"], label="val AUPRC", color="#8e44ad")
axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUROC)")
axes[1].set_title("Validation metrics per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()

print("Read this chart directionally, not literally: with ~6 validation patients on the "
      "dev subset (Sec 1.4), a single flipped prediction swings AUROC by roughly 1/(n_pos*n_neg) "
      "-- exactly the caution the source repo's own docs give for its 29-30 patient runs. "
      "Re-run on the full cohort for a trustworthy curve.")