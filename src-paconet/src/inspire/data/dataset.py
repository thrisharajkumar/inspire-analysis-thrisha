"""
PyTorch Dataset/DataLoader — finishes the batch-structure TODO flagged in
training/train.py and scripts/03_train_model.py. Pools each patient's per-system
readings into a fixed-length padded tensor per organ system, plus phase_ids,
timestamps, and a missingness mask, matching what models/encoders.py expects.

NOTE: this uses simple padding/truncation to a fixed MAX_SEQ_LEN, not the peri-op
resampling-to-fixed-bins step described in the design doc — that's still a genuine
TODO for the real run, since bin size is one of the open decisions. This is the
minimal version needed to make the pipeline runnable end-to-end.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

from inspire.features.organ_systems import ENCODER_SYSTEMS

PHASE_TO_ID = {"pre": 0, "peri": 1, "post": 2}
MAX_SEQ_LEN = 32


class INSPIREDataset(Dataset):
    def __init__(self, long_df, static_df, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.static_df = static_df.reset_index(drop=True)
        self.subject_ids = self.static_df["subject_id"].tolist()

        # Pre-group once, rather than filtering the full long_df per __getitem__ call.
        self.long_by_subject = {sid: g for sid, g in long_df.groupby("subject_id")}

        # Static feature encoding — minimal, matches what generate_synthetic_cohort produces.
        self.dept_to_id = {d: i for i, d in enumerate(sorted(static_df["department"].unique()))}
        self.asa_values = sorted(static_df["asa"].astype(int).unique())

    def __len__(self):
        return len(self.subject_ids)

    def _system_tensor(self, patient_long_df, system):
        rows = patient_long_df[patient_long_df["system"] == system]
        n = min(len(rows), self.max_seq_len)

        values = torch.zeros(self.max_seq_len, 1)
        mask = torch.zeros(self.max_seq_len, 1)
        phase_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        timestamps = torch.zeros(self.max_seq_len, 1)

        if n > 0:
            sample = rows.iloc[:n]
            values[:n, 0] = torch.tensor(sample["value"].values, dtype=torch.float32)
            mask[:n, 0] = torch.tensor(sample["observed"].values, dtype=torch.float32)
            phase_ids[:n] = torch.tensor(
                [PHASE_TO_ID[p] for p in sample["phase"]], dtype=torch.long
            )
            t = sample["chart_time"].values.astype(float)
            t = (t - t.min()) / (t.max() - t.min() + 1e-6)  # normalize to [0, 1]
            timestamps[:n, 0] = torch.tensor(t, dtype=torch.float32)

        return values, mask, phase_ids, timestamps

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        row = self.static_df.iloc[idx]
        if subject_id in self.long_by_subject:
            patient_long_df = self.long_by_subject[subject_id]
        else:
            import pandas as pd
            patient_long_df = pd.DataFrame(columns=["system", "value", "observed", "phase", "chart_time"])

        per_system = {}
        for system in ENCODER_SYSTEMS:
            per_system[system] = self._system_tensor(patient_long_df, system)

        static_scalar = torch.tensor(
            [float(row["asa"]), float(row.get("severity", 0.0)), 0.0, 0.0], dtype=torch.float32
        )  # TODO: replace the two zero placeholders with real HFRS + operation-history
        # features once those are wired into the static feature table for a real run.

        return {
            "per_system": per_system,
            "static_scalar": static_scalar,
            "cardiac_washout_eligible": torch.tensor(True),  # TODO: wire the real 6-month
            # washout rule once operation-history dates are available — defaulted to
            # eligible (no penalty) for this dry run so it doesn't distort the synthetic loss.
            "label": torch.tensor(float(row["died"])),
        }


def collate_batch(batch):
    """Stacks a list of __getitem__ dicts into batched tensors."""
    out = {"per_system": {}, "static_scalar": [], "cardiac_washout_eligible": [], "label": []}
    for system in ENCODER_SYSTEMS:
        values = torch.stack([b["per_system"][system][0] for b in batch])
        mask = torch.stack([b["per_system"][system][1] for b in batch])
        phase_ids = torch.stack([b["per_system"][system][2] for b in batch])
        timestamps = torch.stack([b["per_system"][system][3] for b in batch])
        out["per_system"][system] = (values, mask, phase_ids, timestamps)
    out["static_scalar"] = torch.stack([b["static_scalar"] for b in batch])
    out["cardiac_washout_eligible"] = torch.stack([b["cardiac_washout_eligible"] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch])
    return out
