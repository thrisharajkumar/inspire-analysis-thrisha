"""
Per-organ-system encoders reading one continuous, phase-tagged timeline — replacing the
pre-op-only fixed window. Uses BERT-style segment embeddings for phase identity (own
adaptation, no clinical precedent found doing this specifically — see novelty ledger in
docs/current/PACO_Net_Latest_Work_and_Results.md §6) plus continuous time-delta encoding
so position isn't reset at phase boundaries (peri-op is far denser than pre/post — see
§ "Handling the density mismatch" in the same doc; resample peri-op to fixed bins
BEFORE this encoder sees it, in data/preprocess.py, not here).
"""

import math
import torch
import torch.nn as nn


class ContinuousTimeEmbedding(nn.Module):
    """Time2Vec-style continuous time embedding — avoids resetting position at phase
    boundaries, unlike a standard learned positional embedding indexed by sequence position."""

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, embed_dim - 1)

    def forward(self, timestamps):
        # timestamps: (batch, seq_len, 1), real elapsed time (seconds, normalized upstream)
        linear_part = self.linear(timestamps)
        periodic_part = torch.sin(self.periodic(timestamps))
        return torch.cat([linear_part, periodic_part], dim=-1)


class PhaseAwareTimelineEncoder(nn.Module):
    """
    One continuous timeline per organ system, phase-tagged. Feeds a standard transformer
    encoder — this module only handles the input embedding (value + phase + time); the
    transformer body itself is unchanged from the existing per-system encoders in
    dnn_mortality_pipeline_real.py and can be reused directly on top of this output.
    """

    def __init__(self, feature_dim, embed_dim, n_phases=3):
        super().__init__()
        self.value_proj = nn.Linear(feature_dim, embed_dim)
        self.phase_embed = nn.Embedding(n_phases, embed_dim)  # 0=pre, 1=peri, 2=post
        self.time_encode = ContinuousTimeEmbedding(embed_dim)
        self.missingness_proj = nn.Linear(feature_dim, embed_dim)  # carries the MNAR mask signal

    def forward(self, values, missingness_mask, phase_ids, timestamps):
        """
        values:            (batch, seq_len, feature_dim)
        missingness_mask:  (batch, seq_len, feature_dim) — 1=observed, 0=imputed
        phase_ids:         (batch, seq_len) int64, 0/1/2
        timestamps:        (batch, seq_len, 1) — real elapsed time, normalized

        Returns: (batch, seq_len, embed_dim) — ready for the existing transformer body.
        """
        x = self.value_proj(values)
        x = x + self.phase_embed(phase_ids)
        x = x + self.time_encode(timestamps)
        x = x + self.missingness_proj(missingness_mask)  # per-timestep MNAR signal
        return x
