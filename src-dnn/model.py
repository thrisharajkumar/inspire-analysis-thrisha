"""
Stage 5: the multimodal organ-system DNN -- INSPIREDataset, SystemEncoder,
StaticEncoder, ReconstructionHead, loss functions, NAM/Concat/Gated fusion, and
MortalityModel. Converted from cells 92-106.

ONE CHANGE from the original notebook: MortalityModel now includes the learned
SystemCorrelationLayer (system_correlation_layer.py), wired in right after
encode_all_systems() produces the per-system pooled embeddings, before fusion. This
does NOT replace the existing hand-coded cardiovascular->renal coupling (kept, as a
documented clinical prior) -- it adds a general mechanism on top that can discover
other cross-system relationships too, and reports which systems it found move
together (model_output["correlation_report"]["top_correlations"]) for a surgeon's
sanity check against known physiology.

Toggle via CONFIG["USE_SYSTEM_CORRELATION_LAYER"] (default True) -- set False to run
the original architecture unchanged, for a clean ablation between the two.
"""
from sampling import *
from system_correlation_layer import SystemCorrelationLayer

# New config flags for the correlation layer -- added here rather than editing
# config.py directly, so this stage's addition is self-contained and visible in one
# place; CONFIG.get(...) with a default in model_def_fixed.py means these are optional.
CONFIG.setdefault("USE_SYSTEM_CORRELATION_LAYER", True)
CONFIG.setdefault("CORRELATION_LAYER_HEADS", 4)
CONFIG.setdefault("CORRELATION_LAYER_SPARSITY_TOPK", 3)

# --- from notebook cell 94 ---
def _jitter_and_mask_sequence(raw, mask, sigma, rng):
    # raw, mask: [T, F] numpy arrays, already imputed + standardized (Part 7).
    # Jitter only OBSERVED positions (mask==1) so noise doesn't compound with imputation.
    # Time-mask: zero out a short contiguous span of timesteps (all features at once) --
    # the fixed-length-grid stand-in for window slicing/cropping, see Part 8.3.
    if raw.shape[1] == 0:
        return raw.copy(), mask.copy()
    out_raw = raw.copy()
    out_mask = mask.copy()

    noise = rng.normal(loc=0.0, scale=sigma, size=out_raw.shape)
    out_raw = out_raw + noise * out_mask   # only perturb where mask==1 (real observations)

    T = out_raw.shape[0]
    if T >= 4:
        span = max(1, T // 6)
        start = rng.integers(0, max(T - span, 1))
        out_raw[start:start + span, :] = 0.0
        out_mask[start:start + span, :] = 0.0

    return out_raw, out_mask

class INSPIREDataset(Dataset):
    def __init__(self, subject_ids, synthetic_static_rows=None, augment_items=None, epoch_seed=0):
        self.subject_ids = list(subject_ids)
        self.synthetic_rows = synthetic_static_rows or []
        self.augment_items = augment_items or []   # list of (sid, copy_idx)
        self.n_real = len(self.subject_ids)
        self.n_synthetic = len(self.synthetic_rows)
        self.epoch_seed = epoch_seed   # bump this between epochs for fresh augmentation noise

    def __len__(self):
        return self.n_real + self.n_synthetic + len(self.augment_items)

    def __getitem__(self, idx):
        if idx < self.n_real:
            sid = self.subject_ids[idx]
            item = {"is_synthetic": False, "is_augmented": False, "subject_id": sid}
            for system in TIME_SERIES_SYSTEMS:
                raw = IMPUTED_BUNDLE[sid][system]
                mask = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
                item[f"{system}_x"] = torch.tensor(raw, dtype=torch.float32)
                item[f"{system}_mask"] = torch.tensor(mask, dtype=torch.float32)
            item["static"] = torch.tensor(IMPUTED_BUNDLE[sid]["static"].values.astype(float), dtype=torch.float32)
            item["label"] = torch.tensor(PATIENT_BUNDLE[sid]["label"], dtype=torch.float32)
            return item

        idx -= self.n_real
        if idx < self.n_synthetic:
            static_vec, label = self.synthetic_rows[idx]
            item = {"is_synthetic": True, "is_augmented": False, "subject_id": f"synthetic_{idx}"}
            for system in TIME_SERIES_SYSTEMS:
                fnames = PATIENT_BUNDLE[self.subject_ids[0]]["ts"][system]["feature_names"]
                # NOTE: IMPUTED_BUNDLE is standardised (§7.6, z-score, mean 0) by the time this
                # runs, so "population mean" for a synthetic patient is simply 0 in this space
                # -- not the raw STATS mean, which is still in original clinical units.
                filled = np.zeros((CONFIG["TARGET_SEQ_LEN"], len(fnames))) if fnames else np.zeros((CONFIG["TARGET_SEQ_LEN"], 0))
                item[f"{system}_x"] = torch.tensor(filled, dtype=torch.float32)
                item[f"{system}_mask"] = torch.zeros(filled.shape, dtype=torch.float32)   # honestly: nothing was observed
            item["static"] = torch.tensor(static_vec, dtype=torch.float32)
            item["label"] = torch.tensor(label, dtype=torch.float32)
            return item

        idx -= self.n_synthetic
        sid, copy_idx = self.augment_items[idx]
        rng = np.random.default_rng(hash((sid, copy_idx, self.epoch_seed)) % (2**32))
        item = {"is_synthetic": False, "is_augmented": True, "subject_id": f"{sid}_aug{copy_idx}"}
        for system in TIME_SERIES_SYSTEMS:
            raw = IMPUTED_BUNDLE[sid][system]
            mask = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
            aug_raw, aug_mask = _jitter_and_mask_sequence(raw, mask, CONFIG["JITTER_SIGMA"], rng)
            item[f"{system}_x"] = torch.tensor(aug_raw, dtype=torch.float32)
            item[f"{system}_mask"] = torch.tensor(aug_mask, dtype=torch.float32)
        item["static"] = torch.tensor(IMPUTED_BUNDLE[sid]["static"].values.astype(float), dtype=torch.float32)
        item["label"] = torch.tensor(PATIENT_BUNDLE[sid]["label"], dtype=torch.float32)
        return item

def collate_bundle(batch):
    out = {}
    for key in batch[0]:
        if key in ("is_synthetic", "is_augmented", "subject_id"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out

train_dataset = INSPIREDataset(EFFECTIVE_TRAIN_IDS, SYNTHETIC_STATIC_ROWS, AUGMENT_ITEMS)
val_dataset   = INSPIREDataset(VAL_IDS)
test_dataset  = INSPIREDataset(TEST_IDS)

train_loader = DataLoader(train_dataset, batch_size=min(CONFIG["BATCH_SIZE"], len(train_dataset)),
                           shuffle=True, collate_fn=collate_bundle)
val_loader   = DataLoader(val_dataset, batch_size=min(CONFIG["BATCH_SIZE"], max(len(val_dataset), 1)),
                           shuffle=False, collate_fn=collate_bundle)
test_loader  = DataLoader(test_dataset, batch_size=min(CONFIG["BATCH_SIZE"], max(len(test_dataset), 1)),
                           shuffle=False, collate_fn=collate_bundle)

print(f"train_dataset: {len(train_dataset)} items "
      f"({train_dataset.n_real} real, {train_dataset.n_synthetic} synthetic, {len(AUGMENT_ITEMS)} augmented)")
print(f"val_dataset:   {len(val_dataset)} items")
print(f"test_dataset:  {len(test_dataset)} items")

_batch = next(iter(train_loader))
print("\nExample batch shapes:")
for k, v in _batch.items():
    if torch.is_tensor(v):
        print(f"  {k:20s} {tuple(v.shape)}")

# --- from notebook cell 96 ---
class SystemEncoder(nn.Module):
    def __init__(self, n_features, embed_dim, n_heads, n_layers, target_seq_len,
                 extra_context_dim=0, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.has_context = extra_context_dim > 0
        if n_features == 0:
            # System with no assigned raw features (e.g. neurological has few/no labs in
            # some configs) -- degrade gracefully to a learned constant embedding rather
            # than crash, so the architecture still runs on datasets where a system is
            # genuinely empty.
            self.empty_embedding = nn.Parameter(torch.zeros(embed_dim))
            return
        input_dim = n_features * 2   # value + mask, concatenated per feature (see docstring above)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_seq_len, embed_dim) * 0.02)
        if self.has_context:
            self.context_proj = nn.Linear(extra_context_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask, extra_context=None):
        # x, mask: [B, T, n_features]
        if self.n_features == 0:
            batch_size = x.shape[0]
            return self.empty_embedding.unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([x, mask], dim=-1)          # [B, T, 2*n_features]
        h = self.input_proj(combined) + self.pos_embedding
        if self.has_context and extra_context is not None:
            h = h + self.context_proj(extra_context).unsqueeze(1)   # broadcast over time
        h = self.transformer(h)                           # [B, T, embed_dim]
        h = self.out_norm(h)
        embedding = h.mean(dim=1)                          # mean-pool over time -> [B, embed_dim]
        return embedding, h   # also return per-timestep h for the attention-audit hook in §11.3

print("SystemEncoder defined.")

# --- from notebook cell 98 ---
class StaticEncoder(nn.Module):
    def __init__(self, n_static_features, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static_features, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

print("StaticEncoder defined.")

# --- from notebook cell 100 ---
class ReconstructionHead(nn.Module):
    # Decodes a pooled system embedding back into a full [T, n_features] reconstruction --
    # the autoencoder half of §1.5.4's two-phase pattern.
    def __init__(self, embed_dim, n_features, target_seq_len):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.n_features = n_features
        if n_features == 0:
            return
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, target_seq_len * n_features),
        )

    def forward(self, embedding):
        if self.n_features == 0:
            return None
        out = self.decoder(embedding)
        return out.view(-1, self.target_seq_len, self.n_features)

def masked_reconstruction_loss(pred, target, mask):
    # MSE only over positions that were genuinely OBSERVED (mask==1) -- reconstructing
    # imputed filler values would just teach the network to reproduce the imputation
    # strategy from §7 instead of the real underlying signal.
    if pred is None:
        return torch.tensor(0.0, device=target.device)
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom

print("ReconstructionHead + masked_reconstruction_loss defined.")

# --- from notebook cell 102 ---
def focal_loss_with_logits(logits, targets, pos_weight, gamma=2.0):
    # Lin et al. 2017 focal loss, generalised with a pos_weight term so it still respects
    # class imbalance (gamma alone down-weights EASY examples of either class; pos_weight
    # additionally up-weights the rarer class specifically -- combining both per §1.4.1).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    weight = torch.where(targets == 1, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
    return (weight * focal_term * bce).mean()

def mortality_loss(logits, targets, pos_weight, use_focal, gamma):
    if use_focal:
        return focal_loss_with_logits(logits, targets, pos_weight, gamma)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(pos_weight, device=logits.device))

print(f"Loss configured: {'focal loss' if CONFIG['USE_FOCAL_LOSS'] else 'pos_weight BCE'}, pos_weight={POS_WEIGHT:.2f}")

# --- from notebook cell 104 ---
class NAMFusion(nn.Module):
    def __init__(self, system_names, embed_dim, static_included=True):
        super().__init__()
        self.system_names = list(system_names) + (["static"] if static_included else [])
        self.shape_functions = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
            for name in self.system_names
        })
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, embeddings_dict):
        contributions = {name: self.shape_functions[name](embeddings_dict[name]).squeeze(-1)
                          for name in self.system_names}
        logit = sum(contributions.values()) + self.bias
        return logit, contributions   # contributions: name -> [B] per-system additive term


class ConcatFusion(nn.Module):
    def __init__(self, system_names, embed_dim, static_included=True):
        super().__init__()
        n_inputs = len(system_names) + (1 if static_included else 0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_inputs, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )
        self.system_names = list(system_names) + (["static"] if static_included else [])

    def forward(self, embeddings_dict):
        stacked = torch.cat([embeddings_dict[name] for name in self.system_names], dim=-1)
        logit = self.classifier(stacked).squeeze(-1)
        return logit, None   # no per-system decomposition available -- the point of §11.4's ablation


class GatedFusion(nn.Module):
    def __init__(self, system_names, embed_dim, static_included=True):
        super().__init__()
        self.system_names = list(system_names) + (["static"] if static_included else [])
        n = len(self.system_names)
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * n, n), )
        self.classifier = nn.Sequential(nn.Linear(embed_dim * n, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, embeddings_dict):
        stacked = torch.cat([embeddings_dict[name] for name in self.system_names], dim=-1)
        gates = torch.softmax(self.gate_net(stacked), dim=-1)   # [B, n_systems] -- inspectable per-patient weights
        weighted = torch.cat([
            embeddings_dict[name] * gates[:, i:i+1] for i, name in enumerate(self.system_names)
        ], dim=-1)
        logit = self.classifier(weighted).squeeze(-1)
        gate_dict = {name: gates[:, i] for i, name in enumerate(self.system_names)}
        return logit, gate_dict   # gate_dict doubles as an interpretability signal, like NAM's contributions

FUSION_CLASSES = {"nam": NAMFusion, "concat": ConcatFusion, "gated": GatedFusion}
print(f"Fusion strategy configured: {CONFIG['FUSION_STRATEGY']!r} -> {FUSION_CLASSES[CONFIG['FUSION_STRATEGY']].__name__}")

# --- from notebook cell 106 ---
CARDIAC_CONTEXT_DIM = 3   # cardio_mean_hr, cardio_map_deviation, cardio_has_iabp -- from §6.9

class MortalityModel(nn.Module):
    def __init__(self, system_feature_counts, n_static_features, embed_dim, n_heads, n_layers,
                 target_seq_len, fusion_strategy, symmetric_coupling):
        super().__init__()
        self.system_names = TIME_SERIES_SYSTEMS
        self.symmetric_coupling = symmetric_coupling

        self.encoders = nn.ModuleDict()
        for system in self.system_names:
            extra_ctx = 0
            if system == "renal":
                extra_ctx = CARDIAC_CONTEXT_DIM                                  # §1.6.2 point 1
            elif system == "cardiovascular" and symmetric_coupling:
                extra_ctx = 2                                                    # renal summary dim, §1.6.2's ablation
            self.encoders[system] = SystemEncoder(
                n_features=system_feature_counts[system], embed_dim=embed_dim,
                n_heads=n_heads, n_layers=n_layers, target_seq_len=target_seq_len,
                extra_context_dim=extra_ctx,
            )

        self.recon_heads = nn.ModuleDict({
            system: ReconstructionHead(embed_dim, system_feature_counts[system], target_seq_len)
            for system in self.system_names
        })

        self.static_encoder = StaticEncoder(n_static_features, embed_dim)
        self.fusion = FUSION_CLASSES[fusion_strategy](self.system_names, embed_dim, static_included=True)

        # NEW: system correlation layer. Does NOT replace the existing hand-coded
        # cardiovascular->renal coupling above (that stays, as a documented clinical
        # prior -- cardiorenal syndrome is real and well-established). This ADDS a
        # general mechanism that can also discover OTHER cross-system relationships
        # your architecture doesn't currently encode by hand (e.g.
        # respiratory<->cardiovascular), and reports which systems it found move
        # together, so it can be sanity-checked against known physiology. Runs on the
        # pooled per-system embeddings, AFTER each SystemEncoder, BEFORE fusion -- so
        # NAM's per-system interpretability output still decomposes the prediction,
        # just now over correlation-informed embeddings rather than raw pooled ones.
        self.use_correlation_layer = CONFIG.get("USE_SYSTEM_CORRELATION_LAYER", True)
        if self.use_correlation_layer:
            self.correlation_layer = SystemCorrelationLayer(
                embed_dim=embed_dim, n_systems=len(self.system_names),
                n_heads=CONFIG.get("CORRELATION_LAYER_HEADS", 4),
                sparsity_topk=CONFIG.get("CORRELATION_LAYER_SPARSITY_TOPK", 3),
            )

    def encode_all_systems(self, batch, cardio_context, renal_context=None):
        embeddings, recon_preds, per_timestep = {}, {}, {}
        for system in self.system_names:
            x, mask = batch[f"{system}_x"], batch[f"{system}_mask"]
            ctx = cardio_context if system == "renal" else (renal_context if (system == "cardiovascular" and self.symmetric_coupling) else None)
            emb, h = self.encoders[system](x, mask, extra_context=ctx)
            embeddings[system] = emb
            per_timestep[system] = h
            recon_preds[system] = self.recon_heads[system](emb)
        return embeddings, recon_preds, per_timestep

    def forward(self, batch):
        # cardio_context: the §6.9 side-input for the renal branch. Built here from the raw
        # cardiovascular time series' first few pooled stats (cheap, always available -- a
        # richer version could reuse the cardiovascular SystemEncoder's own embedding, but
        # that would create a forward-pass ordering dependency; a direct small summary is
        # simpler and keeps the two branches independently trainable).
        cardio_x = batch["cardiovascular_x"]     # [B, T, F_cardio]
        cardio_context = cardio_x.mean(dim=1)[:, :CARDIAC_CONTEXT_DIM] if cardio_x.shape[-1] >= CARDIAC_CONTEXT_DIM else \
            F.pad(cardio_x.mean(dim=1), (0, CARDIAC_CONTEXT_DIM - cardio_x.shape[-1]))

        renal_context = None
        if self.symmetric_coupling:
            renal_x = batch["renal_x"]
            renal_context = renal_x.mean(dim=1)[:, :2] if renal_x.shape[-1] >= 2 else F.pad(renal_x.mean(dim=1), (0, 2 - renal_x.shape[-1]))

        embeddings, recon_preds, per_timestep = self.encode_all_systems(batch, cardio_context, renal_context)

        # NEW: learned system correlation, on top of (not instead of) the existing
        # hard-coded cardio->renal link above.
        correlation_report = None
        if self.use_correlation_layer:
            stacked = torch.stack([embeddings[s] for s in self.system_names], dim=1)  # [B, n_systems, D]
            correlated, correlation_report = self.correlation_layer(stacked, system_names=self.system_names)
            embeddings = {s: correlated[:, i, :] for i, s in enumerate(self.system_names)}

        static_emb = self.static_encoder(batch["static"])
        embeddings_for_fusion = dict(embeddings)
        embeddings_for_fusion["static"] = static_emb

        logit, per_system_signal = self.fusion(embeddings_for_fusion)
        return {
            "logit": logit,
            "embeddings": embeddings,
            "static_embedding": static_emb,
            "recon_preds": recon_preds,
            "per_system_signal": per_system_signal,   # NAM contributions or gate weights or None (concat)
            "per_timestep": per_timestep,              # for the §11.3 attention audit
            "correlation_report": correlation_report,  # NEW -- top_correlations for surgeon review
        }

SYSTEM_FEATURE_COUNTS = {system: len(PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][system]["feature_names"]) for system in TIME_SERIES_SYSTEMS}
N_STATIC_FEATURES = len(STATIC_FEATURE_NAMES)

model = MortalityModel(
    system_feature_counts=SYSTEM_FEATURE_COUNTS,
    n_static_features=N_STATIC_FEATURES,
    embed_dim=CONFIG["EMBED_DIM"], n_heads=CONFIG["TRANSFORMER_HEADS"], n_layers=CONFIG["TRANSFORMER_LAYERS"],
    target_seq_len=CONFIG["TARGET_SEQ_LEN"], fusion_strategy=CONFIG["FUSION_STRATEGY"],
    symmetric_coupling=CONFIG["SYMMETRIC_CARDIORENAL_COUPLING"],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"MortalityModel built: {n_params:,} parameters")
print(f"System feature counts: {SYSTEM_FEATURE_COUNTS}")

# Architecture fingerprint -- used to name checkpoints (Part 10) so a checkpoint from a
# DIFFERENT run (e.g. a MAX_SUBJECTS_PER_CLASS smoke test, where a rare department might be
# absent and shrink the static feature count by a column or two) can never be silently
# loaded into a model it doesn't actually match. Derived from the things that determine the
# model's actual shape, not from config values that merely correlate with it.
import hashlib
_fingerprint_source = repr((
    SYSTEM_FEATURE_COUNTS, N_STATIC_FEATURES, CONFIG["EMBED_DIM"], CONFIG["TRANSFORMER_HEADS"],
    CONFIG["TRANSFORMER_LAYERS"], CONFIG["TARGET_SEQ_LEN"], CONFIG["FUSION_STRATEGY"],
    CONFIG["SYMMETRIC_CARDIORENAL_COUPLING"],
))
ARCHITECTURE_FINGERPRINT = hashlib.sha256(_fingerprint_source.encode()).hexdigest()[:10]
print(f"Architecture fingerprint: {ARCHITECTURE_FINGERPRINT} "
      f"(checkpoints are tagged with this -- a checkpoint from a run with a different "
      f"static/time-series feature count, e.g. a smoke test that happened to miss a rare "
      f"department, will be ignored rather than loaded into a mismatched model)")

# Smoke test: one forward pass on a real batch.
_batch_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in _batch.items()}
with torch.no_grad():
    _out = model(_batch_dev)
print(f"\nSmoke test forward pass OK. logit shape: {tuple(_out['logit'].shape)}")
if _out["per_system_signal"] is not None:
    print("Per-system signal keys:", list(_out["per_system_signal"].keys()))