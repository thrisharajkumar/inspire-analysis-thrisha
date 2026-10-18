#!/bin/bash
# INSPIRE DNN v4 -- full ~99,886-patient cohort on ONE Isambard-AI GH200 GPU (headless).
# Fill in the three CHANGE-ME lines, then:   POOLING=mean TRAINING_SEED=101 sbatch run_inspire_isambard.sbatch
#SBATCH --job-name=inspire-v4
#SBATCH --gpus=1                      # the model is small: one GH200 is plenty
#SBATCH --time=24:00:00               # CHANGE-ME if your project allows less/more
#SBATCH --output=inspire_v4_%j.out
##SBATCH --account=CHANGE-ME          # your Isambard project, if your setup needs it

set -euo pipefail
: "${TRAINING_SEED:?Set TRAINING_SEED explicitly; use different values for independent repeats}"
: "${POOLING:?Set POOLING=mean or POOLING=attention explicitly before submitting}"
case "$POOLING" in
  mean|attention) ;;
  *) echo "POOLING must be mean or attention" >&2; exit 2 ;;
esac
REPO=${REPO:-$HOME/inspire-analysis-thrisha/src-dnn}            # CHANGE-ME: where src-dnn lives
DATA=${DATA:-$PROJECTDIR/inspire/subjects}                      # CHANGE-ME: folder with died/ and survived/
OUT=${OUT:-$PROJECTDIR/inspire/runs/v4_${SLURM_JOB_ID}}
SIF=${SIF:-$PROJECTDIR/containers/pytorch.sif}                  # an aarch64 NGC PyTorch image (see README)
mkdir -p "$(dirname "$OUT")"

# Where everything is written: outputs (figures/models/snapshots/tables), checkpoints, parse cache.
CONFIG_JSON=$(printf '{"CACHE_DIR": "%s", "N_ENSEMBLE": 1}' "$PROJECTDIR/inspire/parse_cache")

srun apptainer exec --nv "$SIF" bash -c "
  pip install --user -q imbalanced-learn pyarrow umap-learn || true
  cd '$REPO'
  python -m pytest -q tests/ || { echo 'UNIT TESTS FAILED -- not starting the long run'; exit 1; }
  python run_pipeline.py --pooling '$POOLING' --training-seed '$TRAINING_SEED' --split-seed 42 --run-dir '$OUT' --profile isambard --subjects-dir '$DATA' \
      --config '$CONFIG_JSON'
"
echo "Done. Everything is in $OUT"
