"""
Checkpoint save/load with full RNG state — the core piece that makes free-tier GPU runs
(Colab/Kaggle, unpredictable session lengths) safe. See docs/current for the
Colab-vs-Kaggle persistence trade-off discussion.
"""

import os
import random
import torch
import numpy as np


def save_checkpoint(path, model, optimizer, epoch, step, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer, device):
    # weights_only=False: this checkpoint is our own (RNG state, optimizer state), not
    # an untrusted third-party file — safe here, but only load checkpoints you trust,
    # since this restores PyTorch 2.6+'s default weights_only=True safety restriction.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    torch.set_rng_state(ckpt["rng_state"]["torch"])
    if ckpt["rng_state"]["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_state"]["torch_cuda"])
    np.random.set_state(ckpt["rng_state"]["numpy"])
    random.setstate(ckpt["rng_state"]["python"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("extra", {})


def latest_checkpoint_path(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "latest.pt")
    return path if os.path.exists(path) else None
