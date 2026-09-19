"""
NAM-style additive fusion — one term per organ SYSTEM (not per single feature, which is
what every published additive/interpretable clinical model does instead — this is the
architecture's most defensible novelty claim, already built and tested in the baseline).

This module also applies:
  - The static/aggregate features (GI, MSK, fluid_resuscitation) that don't go through
    a timeline encoder at all — see features/organ_systems.py STATIC_SYSTEMS.
  - The 6-month cardiac washout rule as a hard-coded eligibility flag — kept OUTSIDE the
    learned coupling layer on purpose (see coupling.py docstring).
"""

import torch
import torch.nn as nn


class NAMFusion(nn.Module):
    def __init__(self, embed_dim, n_encoder_systems, n_static_systems, n_static_scalar_features):
        """
        n_encoder_systems: organ systems with a timeline encoder (renal, cardiovascular,
            respiratory, metabolic_hepatic, haematology, neurological) — post-coupling.
        n_static_systems: GI, MSK, fluid_resuscitation — fed directly, no timeline encoder.
        n_static_scalar_features: age, ASA, HFRS, operation history, etc.
        """
        super().__init__()
        # One small network per encoder-system term — this is what makes each term's
        # contribution separately inspectable (per-system risk breakdown).
        self.encoder_system_terms = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
            for _ in range(n_encoder_systems)
        ])
        self.static_system_terms = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
            for _ in range(n_static_systems)
        ])
        self.static_scalar_term = nn.Sequential(
            nn.Linear(n_static_scalar_features, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, coupled_encoder_embeddings, static_system_embeddings, static_scalar_features,
                cardiac_washout_eligible):
        """
        coupled_encoder_embeddings: (batch, n_encoder_systems, embed_dim) — from coupling.py
        static_system_embeddings:   (batch, n_static_systems, embed_dim) — GI/MSK/fluid
        static_scalar_features:     (batch, n_static_scalar_features) — age, ASA, HFRS, etc.
        cardiac_washout_eligible:   (batch,) bool — the hard rule, applied here, not learned

        Returns: (risk_logit, per_system_contributions) — the latter is your
        interpretability payoff: exactly which system pushed the prediction up or down.
        """
        contributions = {}
        total = 0.0

        for i, term in enumerate(self.encoder_system_terms):
            contrib = term(coupled_encoder_embeddings[:, i, :]).squeeze(-1)
            contributions[f"encoder_system_{i}"] = contrib
            total = total + contrib

        for i, term in enumerate(self.static_system_terms):
            contrib = term(static_system_embeddings[:, i, :]).squeeze(-1)
            contributions[f"static_system_{i}"] = contrib
            total = total + contrib

        scalar_contrib = self.static_scalar_term(static_scalar_features).squeeze(-1)
        contributions["static_scalars"] = scalar_contrib
        total = total + scalar_contrib

        # Hard rule: cardiac washout period is NOT learned. Applied as a fixed additive
        # bias rather than fed through any network, so it always fires the same way
        # regardless of training — the explicit hard-rule-vs-learned split.
        # NOTE: modelling choice to confirm — this treats being OUTSIDE the washout
        # window as a fixed positive risk contribution (washout_bias), and inside the
        # window as neutral (0). Swap the value/sign to match how the rule should
        # actually move the prediction; the point is that it's a constant, never learned.
        washout_bias = 1.0  # placeholder magnitude — tune/replace based on clinical guidance
        washout_term = torch.where(
            cardiac_washout_eligible, torch.zeros_like(total), torch.full_like(total, washout_bias)
        )
        contributions["cardiac_washout_flag"] = washout_term
        total = total + washout_term

        return total, contributions
