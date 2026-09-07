"""
Resumable trainer. Time-based checkpointing (not just per-epoch) matters specifically
for free-tier GPU sessions — Colab/Kaggle disconnects don't wait for an epoch boundary,
and a full-cohort epoch could easily outlast a save interval.
"""

import time
import torch
from torch.cuda.amp import autocast, GradScaler

from inspire.training.checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint_path


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, checkpoint_dir,
                 save_every_minutes=15, mixed_precision=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_every_seconds = save_every_minutes * 60
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.mixed_precision)

    def maybe_resume(self):
        ckpt_path = latest_checkpoint_path(self.checkpoint_dir)
        if ckpt_path is None:
            return 0, 0
        epoch, step, extra = load_checkpoint(ckpt_path, self.model, self.optimizer, self.device)
        print(f"Resumed from epoch {epoch}, step {step}")
        return epoch, step

    def fit(self, dataloader, n_epochs, resume=False):
        start_epoch, start_step = self.maybe_resume() if resume else (0, 0)
        last_save = time.time()

        for epoch in range(start_epoch, n_epochs):
            for step, batch in enumerate(dataloader):
                if epoch == start_epoch and step < start_step:
                    continue  # skip already-completed steps in a resumed epoch

                self.optimizer.zero_grad()
                with autocast(enabled=self.mixed_precision):
                    loss = self._forward_step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if time.time() - last_save > self.save_every_seconds:
                    save_checkpoint(f"{self.checkpoint_dir}/latest.pt", self.model,
                                     self.optimizer, epoch, step)
                    last_save = time.time()
                    print(f"Checkpoint saved at epoch {epoch}, step {step}")

            # Always checkpoint at epoch end too, regardless of the time interval.
            save_checkpoint(f"{self.checkpoint_dir}/latest.pt", self.model,
                             self.optimizer, epoch + 1, 0)
            print(f"Epoch {epoch} complete, loss={loss.item():.4f}")

    def _forward_step(self, batch):
        """
        Wires a batch through model -> loss. Left as a hook to fill in against your
        specific batch structure (organ-system tensors + phase_ids + timestamps +
        static features + labels) once the dataset/collate function is finalized —
        the encoder/coupling/fusion/head modules define the forward pass shape, but
        the exact batch dict keys depend on your DataLoader implementation.
        """
        raise NotImplementedError(
            "Wire this to your batch structure: unpack organ-system tensors, phase_ids, "
            "timestamps, static features, and labels from `batch`, run them through "
            "encoders -> coupling -> fusion -> head, and return self.loss_fn(pred, label)."
        )
