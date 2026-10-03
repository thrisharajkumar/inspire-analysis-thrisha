# Running INSPIRE DNN v4 on Isambard-AI

## What Isambard-AI is (in one paragraph)

Isambard-AI is the UK's national AI supercomputer in Bristol. Each node has **4 NVIDIA GH200
"Grace Hopper" superchips** (one GPU + one ARM CPU each). Two things matter for us:

1. The CPUs are **ARM (aarch64)**, not the usual Intel/AMD (x86). Software must be the ARM
   version. The simplest way is NVIDIA's ready-made **NGC PyTorch container**, which
   Isambard's documentation recommends.
2. Jobs are submitted with **Slurm** (`sbatch`), not run interactively like Kaggle.

## Why one GPU is enough

Even the biggest preset (`xxlarge`, 256-number embeddings) is a small model by GPU
standards. What Isambard gives us is **time and memory**: the full cohort, the 5-model
ensemble, all ablations and the embedding-size sweep in one job, instead of several
12-hour Kaggle sessions. If you want more speed, run several jobs at once (one GPU each),
e.g. one per embedding size — not one job across four GPUs.

## Steps

1. **Get the code and data onto Isambard** (`git clone` the repo; copy `subjects/` — the
   folder with `died/` and `survived/` — to your project storage).
2. **Get a PyTorch container (once)**:
   `apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:<tag>-py3`
   Pick a `<tag>` from Isambard's "containers on ARM" documentation page (it must support
   aarch64). Point `SIF` in the script at the file.
3. **Edit the three `CHANGE-ME` lines** in `run_inspire_isambard.sbatch`.
4. **Submit**: `sbatch run_inspire_isambard.sbatch`. The job first runs the unit tests (a
   couple of minutes); if any fail it stops instead of wasting hours.
5. **Watch**: `squeue --me`, and `tail -f inspire_v4_<jobid>.out`.
6. **Results** land in `$OUT/outputs/`: `figures/` (every plot, PNG), `models/` (final model
   + ensemble members), `snapshots/` (embeddings per epoch), `tables/` (every results table
   as CSV, including `verification_summary.csv`).

## What the `isambard` profile changes (all in `pipeline/00_setup.py`)

| Setting | Kaggle subset | Kaggle full | Isambard |
|---|---|---|---|
| Patients | ~10,469 | ~99,886 | ~99,886 |
| Embedding size per organ system | 32 (`medium`) | 64 (`large`) | 128 (`xlarge`) |
| Transformer heads / layers | 4 / 2 | 4 / 2 | 8 / 3 |
| Batch size | 256 | 256 | 1024 |
| Mixed precision (bf16) | off | off | on |
| Data-loading workers | 0 | 0 | 8 |
| Max epochs (pre-train / fine-tune) | 30 / 60 | 40 / 80 | 60 / 100 |
| Ensemble members | 3 | 5 | 5 |
| Embedding-size sweep (16 → 256) | off | off | on, 3 seeds each |

Any setting can be overridden without editing code, e.g. add
`"ENCODER_SIZE": "xxlarge"` to `CONFIG_JSON` in the script.

## Honest caveats

- I could not test on Isambard itself. The pipeline is plain PyTorch and runs headless
  through `run_pipeline.py`, which is the same path used for local dry-runs, but the
  container tag, account line and storage paths are yours to fill in.
- `umap-learn` may not install inside some containers; Part 13 then skips UMAP maps and
  says so (PCA and t-SNE still run).
- A bigger embedding is not automatically better: there are only ~469 deaths. The size
  sweep (Part 13.9) is there to decide this with data.
