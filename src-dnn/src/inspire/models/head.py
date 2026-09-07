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
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_time_bins),
        )

    def forward(self, fused_representation):
        """
        fused_representation: (batch, input_dim) — from fusion.py's `total` output
        Returns: (batch, n_time_bins) hazard logits — apply sigmoid per-bin at
        inference/loss time (standard discrete-time survival formulation), not softmax,
        since bins are not mutually exclusive in the hazard formulation.
        """
        return self.net(fused_representation)
