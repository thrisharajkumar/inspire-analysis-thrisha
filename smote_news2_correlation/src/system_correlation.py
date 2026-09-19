"""
Learned system correlation layer -- generalizes the existing notebook's hard-coded
Cardiovascular -> Renal coupling (Section 6.9/9.2: "architectural" cardio summary
concatenated into the renal branch only) into a learned correlation across ALL organ
systems, via multi-head self-attention.

WHY THIS, NOT A REPLACEMENT: your existing cardio->renal link stays as a documented,
clinically-motivated prior (a real, well-established relationship -- cardiorenal
syndrome). This module doesn't remove it; it adds a general mechanism that can ALSO
discover other couplings your architecture doesn't currently encode (e.g.
respiratory<->cardiovascular, haematology<->renal), while reporting exactly which
systems it found move together -- so you and your surgeon can check whether what it
learns matches known physiology or flags something worth a closer look.

Drop-in usage with the existing MortalityModel (Section 9.2): after each SystemEncoder
produces its per-system pooled embedding (the same [B, embed_dim] output already
computed for fusion), stack all systems' embeddings into [B, n_systems, embed_dim] and
pass through this layer before whatever fusion step follows.
"""

import torch
import torch.nn as nn


class SystemCorrelationLayer(nn.Module):
    def __init__(self, embed_dim, n_systems, n_heads=4, sparsity_topk=None, dropout=0.1):
        """
        n_systems: total systems going in -- e.g. your 6 TIME_SERIES_SYSTEMS, optionally
            plus a static-branch embedding if you want GI/MSK/fluid represented too.
        sparsity_topk: if set, the RETURNED attention weights (for reporting/plotting)
            keep only the top-k correlated systems per system -- keeps a printed or
            plotted correlation matrix readable rather than a dense blob. This does NOT
            affect the forward pass itself, only what's reported.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.sparsity_topk = sparsity_topk
        self.n_systems = n_systems

    def forward(self, system_embeddings, system_names=None):
        """
        system_embeddings: [B, n_systems, embed_dim] -- one pooled embedding per organ
            system, stacked in a fixed, known order (pass system_names to make the
            returned correlation matrix labeled rather than just indexed).

        Returns:
            correlated: [B, n_systems, embed_dim] -- feed into your existing fusion step
                in place of the raw per-system embeddings.
            correlation_report: dict with 'attn_weights' [B, n_systems, n_systems] and,
                if system_names was given, 'labeled_matrix' -- a per-sample DataFrame-
                ready structure pairing (system_i, system_j, weight) for the strongest
                learned correlations, meant to be handed to a clinician for a sanity check.
        """
        correlated, attn_weights = self.attn(system_embeddings, system_embeddings, system_embeddings)
        correlated = self.norm(system_embeddings + correlated)  # residual, same pattern as coupling.py

        reported_weights = attn_weights
        if self.sparsity_topk is not None:
            reported_weights = self._sparsify_for_reporting(attn_weights, self.sparsity_topk)

        report = {"attn_weights": reported_weights}
        if system_names is not None:
            report["top_correlations"] = self._top_correlations_summary(reported_weights, system_names)

        return correlated, report

    @staticmethod
    def _sparsify_for_reporting(attn_weights, k):
        topk_vals, topk_idx = torch.topk(attn_weights, k=min(k, attn_weights.shape[-1]), dim=-1)
        sparse = torch.zeros_like(attn_weights)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    @staticmethod
    def _top_correlations_summary(attn_weights, system_names, top_n=5):
        """
        Averages attention weights across the batch and returns the top_n strongest
        (system_i -> system_j) pairs as a plain list of dicts -- meant to be printed or
        handed to a surgeon directly, e.g.:
          [{"from": "cardiovascular", "to": "renal", "weight": 0.34}, ...]
        This is the artifact to check against known physiology (cardiorenal syndrome
        should show up here on real data if the model is learning something sensible).
        """
        mean_weights = attn_weights.mean(dim=0).detach()  # [n_systems, n_systems] -- detached,
        # this is a reporting artifact, not part of the computation graph
        n = mean_weights.shape[0]
        pairs = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pairs.append({
                    "from": system_names[i], "to": system_names[j],
                    "weight": float(mean_weights[i, j]),
                })
        pairs.sort(key=lambda p: p["weight"], reverse=True)
        return pairs[:top_n]
