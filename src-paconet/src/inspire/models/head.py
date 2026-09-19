"""
SurvTRACE-style discrete-time hazard head (Wang & Sun 2022) — predicts P(dies in bin k)
for each of a fixed number of future time bins, rather than a single flat probability.
Answers "when," not just "if." Unchanged by the phase-aware/coupling/diffusion revisions
in this folder — sits downstream of fusion.py's output either way.
"""

import torch.nn as nn


class DiscreteTimeHazardHead(nn.Module):
    def __init__(self, input_dim, n_time_bins):
        super().__init__()
        # THE FIX: input_dim // 2 degenerates to 0 when input_dim == 1 (the scalar NAM
        # fusion output) -- a Linear(1, 0) layer silently produces a constant output
        # regardless of input, which is exactly why an earlier dry run's AUROC sat at
        # exactly 0.500 (chance) no matter how training progressed. A fixed minimum
        # hidden width fixes this while keeping the head small for a scalar input.
        hidden_dim = max(4, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_time_bins),
        )

    def forward(self, fused_representation):
        """
        fused_representation: (batch, input_dim) — from fusion.py's `total` output
        Returns: (batch, n_time_bins) hazard logits — apply sigmoid per-bin at
        inference/loss time (standard discrete-time survival formulation), not softmax,
        since bins are not mutually exclusive in the hazard formulation.
        """
        return self.net(fused_representation)
