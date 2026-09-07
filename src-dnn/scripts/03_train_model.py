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
        raise NotImplementedError(
            "Wire the four modules together against your real batch structure: "
            "run each organ system's timeline through its encoder, pool to one "
            "embedding per system, pass all system embeddings through self.coupling, "
            "then self.fusion, then self.head. Each module's docstring specifies its "
            "expected input/output shapes."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    # TODO: build the real DataLoader from data/processed/phase_tagged_imputed.parquet
    # (+ synthetic_*.parquet if sampling.strategy == "diffusion") once the batch
    # structure referenced in Trainer._forward_step and PACONetV2.forward is finalized.
    dataloader = None
    if dataloader is None:
        print("DataLoader not yet wired up — see the TODO above. This script's structure "
              "(config loading, model assembly, resumable Trainer) is ready; only the "
              "dataset/collate step needs finishing once the batch format is decided.")
        return

    trainer.fit(dataloader, n_epochs=config["training"]["n_epochs"], resume=args.resume)


if __name__ == "__main__":
    main()
