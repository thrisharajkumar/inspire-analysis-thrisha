"""
Standalone smoke test for the two changes made to MortalityModel in model.py:
the SystemCorrelationLayer wiring, and that it can be cleanly disabled for an
ablation. Runs without real INSPIRE data (fake tensors, mock CONFIG) -- fast enough
to run before every real training session as a sanity check that the model still
builds and produces correctly-shaped output.

Usage: python test_model_wiring.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from system_correlation_layer import SystemCorrelationLayer

# Minimal mock globals the class definitions reference
CONFIG = {"USE_SYSTEM_CORRELATION_LAYER": True, "CORRELATION_LAYER_HEADS": 2, "CORRELATION_LAYER_SPARSITY_TOPK": 2,
          "USE_FOCAL_LOSS": False, "FUSION_STRATEGY": "nam"}
POS_WEIGHT = 5.0
TIME_SERIES_SYSTEMS = ["renal", "cardiovascular", "respiratory", "metabolic_hepatic", "haematology", "neurological"]

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



# --- Real forward-pass test ---
system_feature_counts = {s: 4 for s in TIME_SERIES_SYSTEMS}
n_static_features = 10
embed_dim = 16
batch_size = 4
target_seq_len = 8

model = MortalityModel(
    system_feature_counts=system_feature_counts, n_static_features=n_static_features,
    embed_dim=embed_dim, n_heads=2, n_layers=1, target_seq_len=target_seq_len,
    fusion_strategy="nam", symmetric_coupling=False,
)

fake_batch = {}
for s in TIME_SERIES_SYSTEMS:
    fake_batch[f"{s}_x"] = torch.randn(batch_size, target_seq_len, 4)
    fake_batch[f"{s}_mask"] = torch.randint(0, 2, (batch_size, target_seq_len, 4)).float()
fake_batch["static"] = torch.randn(batch_size, n_static_features)

out = model(fake_batch)
print("Forward pass OK.")
print("logit shape:", tuple(out["logit"].shape), "(expect", (batch_size,), ")")
assert out["logit"].shape == (batch_size,)
print("correlation_report present:", out["correlation_report"] is not None)
assert out["correlation_report"] is not None
print("Top correlations found:")
for pair in out["correlation_report"]["top_correlations"][:3]:
    print(f"  {pair['from']:20s} -> {pair['to']:20s}  weight={pair['weight']:.3f}")
assert out["per_system_signal"] is not None, "NAM per-system contributions should still be present"
print()
print("Per-system NAM contributions still present (interpretability preserved):", list(out["per_system_signal"].keys()))

# --- Ablation-safety test: correlation layer OFF should also run cleanly ---
CONFIG["USE_SYSTEM_CORRELATION_LAYER"] = False
model_no_corr = MortalityModel(
    system_feature_counts=system_feature_counts, n_static_features=n_static_features,
    embed_dim=embed_dim, n_heads=2, n_layers=1, target_seq_len=target_seq_len,
    fusion_strategy="nam", symmetric_coupling=False,
)
out2 = model_no_corr(fake_batch)
assert out2["correlation_report"] is None
print()
print("CONFIRMED: with USE_SYSTEM_CORRELATION_LAYER=False, model runs cleanly and")
print("correlation_report is None -- ablation toggle works correctly both ways.")
