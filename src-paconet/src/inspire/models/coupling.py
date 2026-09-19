"""
Learned inter-organ coupling via graph-attention — replaces the single hand-picked
renal<->cardiac link in the original baseline. Closest precedent: IOC-MT (Kong et al.
2025) for general ICU organ dysfunction (not perioperative mortality specifically —
see novelty ledger). Attention weights are the interpretability payoff: report, per
patient, which systems the model decided move together, rather than assuming it's
always renal<->cardiac.

The 6-month cardiac washout rule is DELIBERATELY NOT part of this module — it stays a
static, hard-coded eligibility flag applied at the fusion stage (models/fusion.py),
not something this learned layer should be able to override. See the explicit
hard-rule-vs-learned split documented in the novelty ledger.
"""

import torch
import torch.nn as nn


class LearnedOrganCoupling(nn.Module):
    def __init__(self, embed_dim, n_systems, n_heads=4, sparsity_topk=None):
        """
        sparsity_topk: if set, keep only the top-k attention edges per system when
        reporting weights (config: model.coupling_sparsity_topk). Keeps the learned
        coupling graph reportable in a meeting rather than a dense n_systems x n_systems
        blob. Note: this only affects the RETURNED attention weights for
        interpretability reporting — it does not mask the actual forward pass, so the
        model still uses full attention. Add masking here if you want sparsity to
        affect training itself, not just reporting.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.sparsity_topk = sparsity_topk

    def forward(self, system_embeddings):
        """
        system_embeddings: (batch, n_systems, embed_dim) — one pooled embedding per
        organ system (renal, cardiovascular, respiratory, metabolic_hepatic,
        haematology, neurological, gi, msk — see features/organ_systems.py).

        Returns: (coupled_embeddings, attn_weights)
          coupled_embeddings: (batch, n_systems, embed_dim) — feed into fusion.py
          attn_weights:       (batch, n_systems, n_systems) — for interpretability reporting
        """
        coupled, attn_weights = self.attn(system_embeddings, system_embeddings, system_embeddings)
        out = self.norm(system_embeddings + coupled)

        if self.sparsity_topk is not None:
            attn_weights = self._sparsify_for_reporting(attn_weights, self.sparsity_topk)

        return out, attn_weights

    @staticmethod
    def _sparsify_for_reporting(attn_weights, k):
        """Zeroes all but the top-k attention weights per system, for interpretable
        reporting only (see docstring above — does not affect the forward pass)."""
        topk_vals, topk_idx = torch.topk(attn_weights, k=min(k, attn_weights.shape[-1]), dim=-1)
        sparse = torch.zeros_like(attn_weights)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse
