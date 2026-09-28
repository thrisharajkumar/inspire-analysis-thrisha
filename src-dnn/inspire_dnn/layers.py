# Model building blocks for the INSPIRE organ-system DNN (v3). Pure PyTorch, no dataset
# globals: everything is passed in explicitly, so every piece is unit-tested with random
# tensors (tests/test_layers.py) and inlined into the notebooks unchanged.
#
#   8 organ systems (renal, cardiovascular, respiratory, metabolic/hepatic, haematology,
#   neurological, GI, MSK) -- all built the SAME way:
#       measurements over time (empty for GI/MSK in this dataset) + own static features
#       (own ICD-10 chapter, own summary stats)
#            |  OrganSystemEncoder: ONE shared transformer body for all 8, a small
#            |  per-system input adapter, and a "which system am I" tag
#            v
#   WholePatientLayer: reads ALL features; gently re-scales each system's summary (a narrow,
#            |  scale-only bottleneck) and gives its own whole-patient term
#            v
#   SystemCouplingLayer: each system reads the OTHER systems (learned attention; nothing
#            |  hand-defined). Pre-trained label-free by predicting a hidden system from the
#            |  other seven (masked-system task), then fine-tuned for mortality.
#            v
#   NAM answers f_s  x  SystemArbitrationLayer say w_s
#   logit = bias + f_whole_patient + sum_s w_s * f_s          (exact additive breakdown)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_SIZE_PRESETS = {
    # name: (embed_dim, n_heads, n_layers, ff_multiplier)
    "small":  (16, 2, 1, 2),
    "medium": (32, 4, 2, 4),
    "large":  (64, 4, 2, 4),
}


class OrganSystemEncoder(nn.Module):
    # One encoder for all eight organ systems. Each system has its own small input adapters
    # (its measurements and its static features differ in number), then everything runs
    # through ONE shared transformer body. Sharing means ~8x fewer body parameters and lets
    # small systems (GI, MSK) benefit from what the data-rich systems teach the body.
    #
    # Token layout per system: [summary token = own static features] + [T time tokens].
    # A system with no measurements (GI/MSK here) is just the summary token.
    def __init__(self, system_specs, embed_dim, n_heads, n_layers, ff_multiplier, target_seq_len,
                 dropout=0.1, share_body=True):
        super().__init__()
        self.systems = list(system_specs)
        self.specs = {s: tuple(v) for s, v in system_specs.items()}   # s -> (n_ts_features, n_static_features)
        self.T = target_seq_len
        self.ts_in = nn.ModuleDict({s: nn.Linear(2 * f, embed_dim) for s, (f, _) in self.specs.items() if f > 0})
        self.static_in = nn.ModuleDict({s: nn.Sequential(nn.Linear(k, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
                                        for s, (_, k) in self.specs.items() if k > 0})
        self.empty_static = nn.Parameter(torch.zeros(embed_dim))
        self.system_tag = nn.Parameter(torch.randn(len(self.systems), embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_seq_len + 1, embed_dim) * 0.02)

        def make_body():
            layer = nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim * ff_multiplier, dropout,
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.share_body = share_body
        if share_body:
            self.body = make_body()
        else:
            self.bodies = nn.ModuleDict({s: make_body() for s in self.systems})
        self.out_norm = nn.LayerNorm(embed_dim)

    def _body(self, s):
        return self.body if self.share_body else self.bodies[s]

    def tokens(self, s, x, mask, static_s, batch_size):
        tag = self.system_tag[self.systems.index(s)]
        if s in self.static_in:
            summary = self.static_in[s](static_s)
        else:
            summary = self.empty_static.unsqueeze(0).expand(batch_size, -1)
        seq = (summary + tag).unsqueeze(1)
        if s in self.ts_in:
            seq = torch.cat([seq, self.ts_in[s](torch.cat([x, mask], dim=-1)) + tag], dim=1)
        return seq + self.pos_embedding[:, : seq.shape[1]]

    def forward(self, s, x, mask, static_s, batch_size):
        h = self.out_norm(self._body(s)(self.tokens(s, x, mask, static_s, batch_size)))
        return h.mean(dim=1), (h[:, 1:] if s in self.ts_in else None)

    @torch.no_grad()
    def attention_map(self, s, x, mask, static_s):
        # First-layer self-attention for one system (averaged over heads), pre-norm applied --
        # the audit hook for Part 11.3. Token 0 = summary token, 1..T = time points.
        seq = self.tokens(s, x, mask, static_s, x.shape[0] if x is not None else static_s.shape[0])
        layer0 = self._body(s).layers[0]
        h = layer0.norm1(seq)
        _, w = layer0.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        return w


class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim, n_out, target_seq_len=None):
        super().__init__()
        self.n_out, self.T = n_out, target_seq_len
        if n_out == 0:
            return
        size = n_out * (target_seq_len or 1)
        self.decoder = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, size))

    def forward(self, e):
        if self.n_out == 0:
            return None
        out = self.decoder(e)
        return out.view(-1, self.T, self.n_out) if self.T else out


def masked_reconstruction_loss(pred, target, mask=None):
    if pred is None:
        return None
    if mask is None:
        return ((pred - target) ** 2).mean()
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


class WholePatientLayer(nn.Module):
    # Reads EVERY feature (full static vector + a per-feature summary of every time series)
    # and returns (g, scales). g is the whole-patient summary (its own NAM term, and the input
    # to the arbitration layer). scales re-weight each organ system's summary channel by
    # channel within [0.5, 1.5] -- a deliberately narrow, scale-only bottleneck: it can change
    # how strongly a system's own evidence counts, but cannot add another system's values.
    # The scale projection is zero-initialised, so it starts as "no adjustment".
    def __init__(self, n_static, ts_feature_counts, system_names, embed_dim, dropout=0.1):
        super().__init__()
        self.ts_systems = [s for s, f in ts_feature_counts.items() if f > 0]
        in_dim = n_static + sum(2 * ts_feature_counts[s] for s in self.ts_systems)
        self.system_names = list(system_names)
        self.net = nn.Sequential(nn.Linear(in_dim, embed_dim * 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dim * 2, embed_dim))
        self.film = nn.Linear(embed_dim, len(self.system_names) * embed_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.embed_dim = embed_dim

    def forward(self, batch):
        parts = [batch["static"]]
        for s in self.ts_systems:
            parts += [batch[f"{s}_x"].mean(dim=1), batch[f"{s}_mask"].mean(dim=1)]
        g = self.net(torch.cat(parts, dim=-1))
        scales = 1.0 + 0.5 * torch.tanh(self.film(g)).view(-1, len(self.system_names), self.embed_dim)
        return g, scales


class SystemCouplingLayer(nn.Module):
    # Learned inter-system coupling. Each system (query) attends over the OTHER systems
    # (its own token is masked out), so the message it receives is information from elsewhere
    # in the body.   e_s' = e_s + g_s * message_s   (g_s a learnable gate, starts at 0.1).
    # forward() also returns the raw messages, which masked-system pre-training uses to
    # predict a hidden system from the other seven.
    def __init__(self, system_names, embed_dim, n_heads=2, prior_links=(), prior_strength=1.0,
                 dropout=0.1, gate_init=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by coupling heads"
        self.system_names = list(system_names)
        S = len(self.system_names)
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        self.identity = nn.Parameter(torch.randn(S, embed_dim) * 0.02)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full((S,), float(gate_init)))
        prior = torch.zeros(S, S)
        for sender, receiver in prior_links:
            if sender in self.system_names and receiver in self.system_names:
                prior[self.system_names.index(receiver), self.system_names.index(sender)] = prior_strength
        self.prior_bias = nn.Parameter(prior)
        self.register_buffer("self_mask", torch.eye(S, dtype=torch.bool))

    def forward(self, E, return_messages=False):
        B, S, D = E.shape
        tagged = E + self.identity.unsqueeze(0)
        q = self.q(tagged).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(tagged).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(E).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + self.prior_bias
        attn = torch.softmax(scores.masked_fill(self.self_mask, float("-inf")), dim=-1)
        msg = self.o((self.dropout(attn) @ v).transpose(1, 2).reshape(B, S, D))
        out = E + self.gate.view(1, S, 1) * msg
        report = {"weights": attn.mean(dim=1), "gate": self.gate.detach()}   # weights: [B, receiver, sender]
        return (out, report, msg) if return_messages else (out, report)


class SystemArbitrationLayer(nn.Module):
    # How much "say" each organ system's answer gets for THIS patient:
    #   w = n_systems * softmax(a(x))  -> average say is exactly 1, starts equal (zero-init).
    def __init__(self, system_names, input_dim, hidden_dim=32):
        super().__init__()
        self.system_names = list(system_names)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, len(self.system_names)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        w = len(self.system_names) * torch.softmax(self.net(x), dim=-1)
        return w, (torch.log(w.clamp_min(1e-6)) ** 2).mean()


class NAMFusion(nn.Module):
    def __init__(self, term_names, embed_dim):
        super().__init__()
        self.system_names = list(term_names)
        hidden = max(embed_dim // 2, 4)
        self.shape_functions = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for n in self.system_names})
        self.bias = nn.Parameter(torch.zeros(1))

    def answers(self, emb):
        return {n: self.shape_functions[n](emb[n]).squeeze(-1) for n in self.system_names}

    def forward(self, emb, say=None):
        answers = self.answers(emb)
        contributions = {n: (answers[n] * say[n] if (say is not None and n in say) else answers[n])
                         for n in self.system_names}
        return sum(contributions.values()) + self.bias, contributions, answers


class ConcatFusion(nn.Module):
    def __init__(self, term_names, embed_dim):
        super().__init__()
        self.system_names = list(term_names)
        self.classifier = nn.Sequential(nn.Linear(embed_dim * len(self.system_names), embed_dim),
                                        nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, emb, say=None):
        return self.classifier(torch.cat([emb[n] for n in self.system_names], dim=-1)).squeeze(-1), None, None


FUSION_CLASSES = {"nam": NAMFusion, "concat": ConcatFusion}


class MortalityModel(nn.Module):
    # ts_feature_counts: {organ_system: n_time_series_features} for ALL 8 systems (0 = none).
    # system_static_idx:  {organ_system: [columns of the static vector owned by that system]}
    # context_idx:        columns of the static vector that belong to no single system
    #                     (age, ASA, department, NEWS2, frailty, ...).
    def __init__(self, ts_feature_counts, system_static_idx, context_idx, n_static,
                 embed_dim=32, n_heads=4, n_layers=2, ff_multiplier=4, target_seq_len=24,
                 fusion_strategy="nam", share_encoder=True,
                 coupling_mode="learned", coupling_heads=2, coupling_prior_links=(),
                 coupling_prior_strength=1.0, coupling_gate_init=0.1,
                 use_arbitration=True, use_whole_patient=True, system_dropout=0.0, dropout=0.1):
        super().__init__()
        self.organ_systems = list(ts_feature_counts)
        self.ts_systems = [s for s in self.organ_systems if ts_feature_counts[s] > 0]
        self.system_names = self.ts_systems   # compatibility with older analysis cells
        self.system_static_idx = {s: list(system_static_idx.get(s, [])) for s in self.organ_systems}
        self.context_idx = list(context_idx)
        self.coupling_mode = coupling_mode
        self.use_manual = False                # v3: no hand-defined inter-system input exists any more
        self.system_dropout = system_dropout

        specs = {s: (ts_feature_counts[s], len(self.system_static_idx[s])) for s in self.organ_systems}
        self.encoder = OrganSystemEncoder(specs, embed_dim, n_heads, n_layers, ff_multiplier,
                                          target_seq_len, dropout, share_body=share_encoder)
        self.recon_ts = nn.ModuleDict({s: ReconstructionHead(embed_dim, ts_feature_counts[s], target_seq_len)
                                       for s in self.ts_systems})
        self.recon_static = nn.ModuleDict({s: ReconstructionHead(embed_dim, len(self.system_static_idx[s]))
                                           for s in self.organ_systems if self.system_static_idx[s]})
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))   # stands in for a hidden / dropped system

        self.use_whole_patient = use_whole_patient
        if use_whole_patient:
            self.whole_patient = WholePatientLayer(n_static, ts_feature_counts, self.organ_systems, embed_dim, dropout)
            self.context_term = "whole_patient"
        else:
            self.whole_patient = None
            self.context_encoder = nn.Sequential(nn.Linear(max(len(self.context_idx), 1), embed_dim * 2), nn.ReLU(),
                                                 nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim))
            self.context_term = "static"
        self.coupling = None
        if coupling_mode == "learned":
            self.coupling = SystemCouplingLayer(self.organ_systems, embed_dim, coupling_heads,
                                                coupling_prior_links, coupling_prior_strength, dropout,
                                                gate_init=coupling_gate_init)
        self.term_names = self.organ_systems + [self.context_term]
        self.fusion_strategy = fusion_strategy
        self.fusion = FUSION_CLASSES[fusion_strategy](self.term_names, embed_dim)
        self.arbitration = SystemArbitrationLayer(self.organ_systems, embed_dim) \
            if (use_arbitration and fusion_strategy == "nam") else None

    # ---- pieces -------------------------------------------------------------------------
    def encoder_parameters(self):
        return list(self.encoder.parameters()) + list(self.recon_ts.parameters()) + list(self.recon_static.parameters())

    def _static_s(self, static, s):
        idx = self.system_static_idx[s]
        return static[:, idx] if idx else None

    def encode_systems(self, batch):
        static = batch["static"]
        B = static.shape[0]
        emb, per_t = {}, {}
        for s in self.organ_systems:
            x = batch.get(f"{s}_x"); m = batch.get(f"{s}_mask")
            emb[s], per_t[s] = self.encoder(s, x, m, self._static_s(static, s), B)
        return emb, per_t

    # ---- phase 1: label-free pre-training ----------------------------------------------
    def pretrain_loss(self, batch, masked_system_weight=1.0):
        # (a) each system reconstructs its OWN data from its own summary (autoencoder);
        # (b) masked-system task: one system per patient is hidden and must be predicted
        #     from the OTHER systems through the coupling layer -- this is how the coupling
        #     layer learns real inter-system relationships from every patient, no labels.
        emb, _ = self.encode_systems(batch)
        static = batch["static"]
        losses = []
        for s in self.ts_systems:
            l = masked_reconstruction_loss(self.recon_ts[s](emb[s]), batch[f"{s}_x"], batch[f"{s}_mask"])
            losses.append(l)
        for s in self.recon_static:
            losses.append(masked_reconstruction_loss(self.recon_static[s](emb[s]), self._static_s(static, s)))
        own = torch.stack(losses).mean()
        if self.coupling is None or masked_system_weight == 0:
            return own, {"own": float(own), "masked_system": 0.0}
        B, S = static.shape[0], len(self.organ_systems)
        E = torch.stack([emb[s] for s in self.organ_systems], dim=1)
        hidden = torch.randint(0, S, (B,), device=E.device)
        onehot = F.one_hot(hidden, S).bool().unsqueeze(-1)
        E_masked = torch.where(onehot, self.mask_token.view(1, 1, -1).expand_as(E), E)
        _, _, msg = self.coupling(E_masked, return_messages=True)
        m_losses = []
        for i, s in enumerate(self.organ_systems):
            sel = hidden == i
            if not sel.any():
                continue
            pred_in = self.mask_token + msg[sel, i]
            if s in self.recon_ts:
                m_losses.append(masked_reconstruction_loss(self.recon_ts[s](pred_in), batch[f"{s}_x"][sel], batch[f"{s}_mask"][sel]))
            if s in self.recon_static:
                m_losses.append(masked_reconstruction_loss(self.recon_static[s](pred_in), self._static_s(static, s)[sel]))
        ms = torch.stack(m_losses).mean() if m_losses else torch.zeros((), device=E.device)
        return own + masked_system_weight * ms, {"own": float(own), "masked_system": float(ms)}

    # ---- full forward (phase 2 + inference) ------------------------------------------------
    def forward(self, batch):
        static = batch["static"]
        emb, per_t = self.encode_systems(batch)
        own_emb = dict(emb)
        if self.whole_patient is not None:
            g, scales = self.whole_patient(batch)
            for i, s in enumerate(self.organ_systems):
                emb[s] = emb[s] * scales[:, i]
        else:
            g = self.context_encoder(static[:, self.context_idx] if self.context_idx else torch.zeros_like(static[:, :1]))
            scales = None
        if self.training and self.system_dropout > 0:
            for s in self.organ_systems:   # organ-system dropout: the model can't lean on one system
                drop = (torch.rand(static.shape[0], 1, device=static.device) < self.system_dropout)
                emb[s] = torch.where(drop, self.mask_token.expand_as(emb[s]), emb[s])
        pre_coupling = {s: emb[s] for s in self.organ_systems}
        coupling_report = None
        if self.coupling is not None:
            E2, coupling_report = self.coupling(torch.stack([emb[s] for s in self.organ_systems], dim=1))
            for i, s in enumerate(self.organ_systems):
                emb[s] = E2[:, i]
        emb[self.context_term] = g
        say, aux = None, torch.zeros((), device=static.device)
        if self.arbitration is not None:
            w, aux = self.arbitration(g)
            say = {s: w[:, i] for i, s in enumerate(self.organ_systems)}
        logit, contributions, answers = self.fusion(emb, say)
        return {
            "logit": logit,
            "embeddings": {s: emb[s] for s in self.organ_systems},
            "embeddings_own": own_emb,
            "embeddings_pre_coupling": pre_coupling,
            "static_embedding": g,
            "whole_patient_scales": scales,
            "per_system_signal": contributions,
            "system_answers": answers,
            "system_say": say,
            "coupling_report": coupling_report,
            "aux_loss": aux,
            "per_timestep": per_t,
        }


def coupling_effect(model, out):
    # Per patient: f_s(after coupling) - f_s(before coupling), in logit units.
    if model.coupling is None or out["system_answers"] is None:
        return None
    with torch.no_grad():
        before = model.fusion.answers({**out["embeddings_pre_coupling"], model.context_term: out["static_embedding"]})
    return {s: out["system_answers"][s] - before[s] for s in model.organ_systems}


@torch.no_grad()
def own_data_share(model, batch, n_repeats=3, seed=0):
    # Leakage check. For each organ system s: shuffle s's OWN inputs across patients and
    # measure how much s's answer moves; then shuffle EVERYTHING ELSE and measure again.
    # own_share = own / (own + other). Near 1 = the system's answer is driven by its own
    # data (what a surgeon would assume); low = it is mostly echoing other systems.
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = model(batch)["system_answers"]
    if base is None:
        return None
    B = batch["static"].shape[0]
    res = {}
    for s in model.organ_systems:
        own_cols = model.system_static_idx[s]
        d_own, d_other = 0.0, 0.0
        for _ in range(n_repeats):
            perm = torch.randperm(B, generator=gen).to(batch["static"].device)
            b1 = dict(batch); st = batch["static"].clone()
            if own_cols:
                st[:, own_cols] = batch["static"][perm][:, own_cols]
            b1["static"] = st
            if s in model.ts_systems:
                b1[f"{s}_x"], b1[f"{s}_mask"] = batch[f"{s}_x"][perm], batch[f"{s}_mask"][perm]
            d_own += float((model(b1)["system_answers"][s] - base[s]).abs().mean())
            b2 = {k: v for k, v in batch.items()}
            st2 = batch["static"][perm].clone()
            if own_cols:
                st2[:, own_cols] = batch["static"][:, own_cols]
            b2["static"] = st2
            for t in model.ts_systems:
                if t != s:
                    b2[f"{t}_x"], b2[f"{t}_mask"] = batch[f"{t}_x"][perm], batch[f"{t}_mask"][perm]
            d_other += float((model(b2)["system_answers"][s] - base[s]).abs().mean())
        res[s] = {"change_own": d_own / n_repeats, "change_other": d_other / n_repeats,
                  "own_share": d_own / max(d_own + d_other, 1e-12)}
    return res
