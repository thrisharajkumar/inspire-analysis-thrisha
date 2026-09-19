#!/usr/bin/env python3
"""
Stage 3: main PACO-Net v2 training — phase-aware encoders + learned coupling + NAM
fusion + hazard head. Resumable via --resume, checkpoints on a timer (config:
training.save_every_minutes) so a Colab/Kaggle disconnect never costs more than that
interval's worth of progress.

Usage:
    python scripts/03_train_model.py --config configs/default.yaml
    python scripts/03_train_model.py --config configs/default.yaml --resume
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inspire.models.encoders import PhaseAwareTimelineEncoder
from inspire.models.coupling import LearnedOrganCoupling
from inspire.models.fusion import NAMFusion
from inspire.models.head import DiscreteTimeHazardHead
from inspire.training.train import Trainer
from inspire.features.organ_systems import ENCODER_SYSTEMS, STATIC_SYSTEMS


class PACONetV2(torch.nn.Module):
    """Wires the four model pieces together. See docs/current/PACO_Net_Latest_Work_and_Results.md
    §4 for the full architecture diagram this implements."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config["model"]["embed_dim"]
        self.encoders = torch.nn.ModuleDict({
            system: PhaseAwareTimelineEncoder(feature_dim=1, embed_dim=embed_dim)  # TODO: feature_dim per system
            for system in ENCODER_SYSTEMS
        })
        self.coupling = LearnedOrganCoupling(
            embed_dim=embed_dim, n_systems=len(ENCODER_SYSTEMS) + len(STATIC_SYSTEMS),
            n_heads=config["model"]["n_heads"],
            sparsity_topk=config["model"]["coupling_sparsity_topk"],
        )
        self.fusion = NAMFusion(
            embed_dim=embed_dim, n_encoder_systems=len(ENCODER_SYSTEMS),
            n_static_systems=len(STATIC_SYSTEMS), n_static_scalar_features=4,  # TODO: real count (age, ASA, HFRS, op history)
        )
        self.head = DiscreteTimeHazardHead(input_dim=1, n_time_bins=config["model"]["hazard_time_bins"])

    def forward(self, batch):
        """
        Wires encoders -> mean-pool -> coupling -> fusion -> head, using the batch
        structure produced by data/dataset.py's collate_batch().

        NOTE (dry-run simplification): mean-pools each organ system's timeline into a
        single embedding before coupling, rather than a learned pooling/attention
        pool — reasonable for verifying the pipeline runs, but worth revisiting for a
        real run since mean-pooling discards temporal shape within each system.
        """
        system_embeddings = []
        for system in ENCODER_SYSTEMS:
            values, mask, phase_ids, timestamps = batch["per_system"][system]
            encoded = self.encoders[system](values, mask, phase_ids, timestamps)  # (B, T, D)
            pooled = encoded.mean(dim=1)  # (B, D) — see NOTE above
            system_embeddings.append(pooled)
        encoder_stack = torch.stack(system_embeddings, dim=1)  # (B, n_encoder_systems, D)

        # Static systems (GI/MSK/fluid) aren't wired to real features in this dry run —
        # zero-filled placeholders, matching the count fusion.py expects.
        batch_size = encoder_stack.shape[0]
        static_system_stack = torch.zeros(batch_size, len(STATIC_SYSTEMS), encoder_stack.shape[-1])

        coupled, attn_weights = self.coupling(
            torch.cat([encoder_stack, static_system_stack], dim=1)
        )
        coupled_encoder_part = coupled[:, :len(ENCODER_SYSTEMS), :]
        coupled_static_part = coupled[:, len(ENCODER_SYSTEMS):, :]

        total, contributions = self.fusion(
            coupled_encoder_part, coupled_static_part,
            batch["static_scalar"], batch["cardiac_washout_eligible"],
        )
        # FIXED: previously broadcast the scalar `total` to embed_dim width as a
        # workaround for the head's now-fixed zero-width bug (see models/head.py). Feeding
        # the real scalar directly, rather than a richer embedding, is also the CORRECT
        # choice for this architecture, not just simpler: the NAM fusion's whole
        # interpretability guarantee is that the prediction is a direct function of the
        # interpretable per-system additive total. Feeding the head a richer
        # pre-collapse representation instead would let the model's actual prediction
        # depend on information the per-system breakdown doesn't show -- breaking the
        # "the total explains the prediction" property this architecture exists for.
        hazard_logits = self.head(total.unsqueeze(-1))
        return hazard_logits, attn_weights, contributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["data"]["seed"])
    # THE FIX: no seed was set before this -- meaning weight init and DataLoader
    # shuffling varied every run, and at this dry run's tiny scale (400 patients, 32
    # positives) that variance was large enough to swing AUROC from 0.635 to 0.450
    # between two otherwise-identical runs. Fixed so results are at least reproducible;
    # doesn't remove genuine small-N variance, which is worth keeping in mind at real
    # full-cohort scale too, just less severe there.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PACONetV2(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    pos_weight = config["training"]["pos_weight"]
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight) if pos_weight else None
    )

    trainer = Trainer(
        model, optimizer, loss_fn, device,
        checkpoint_dir=config["training"]["checkpoint_dir"],
        save_every_minutes=config["training"]["save_every_minutes"],
        mixed_precision=config["training"]["mixed_precision"],
    )

    # TODO for a REAL run: build this from data/processed/phase_tagged_imputed.parquet
    # (+ synthetic_*.parquet if sampling.strategy == "diffusion"). For this dry run,
    # uses the synthetic dry-run data generated by scripts/00_generate_synthetic_dryrun_data.py.
    import pandas as pd
    from torch.utils.data import DataLoader
    from inspire.data.dataset import INSPIREDataset, collate_batch

    long_df = pd.read_parquet("data/processed/phase_tagged_imputed.parquet")
    static_df = pd.read_parquet("data/processed/static_features.parquet")
    dataset = INSPIREDataset(long_df, static_df)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                             shuffle=True, collate_fn=collate_batch)

    trainer.fit(dataloader, n_epochs=config["training"]["n_epochs"], resume=args.resume)


if __name__ == "__main__":
    main()
