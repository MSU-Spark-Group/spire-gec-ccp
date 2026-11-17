#!/bin/bash --login
#SBATCH --job-name=spire-train
#SBATCH --account=<ACCOUNT>                    # <-- EDIT
#SBATCH --partition=<PARTITION>                # <-- EDIT
#SBATCH --time=01:00:00                        # the full training setting takes about 0.35 h on one GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=1                               # <-- EDIT per your cluster's GPU directive
#SBATCH --output=slurm-%x-%j.out

# Training of the SPIRE reconstruction model on one GPU.
# Edit the marked lines; the resources above are those of the training reported in the paper.
set -euo pipefail

REPO_ROOT="/path/to/spire-gec-ccp"             # <-- EDIT
VENV="${REPO_ROOT}/.venv"                      # <-- EDIT
DATA_DIR="/path/to/case/files"                 # <-- EDIT: folder with train_*.npz and val_*.npz
OUT_DIR="${REPO_ROOT}/runs"                    # <-- EDIT
# module load <PYTHON_MODULE> <CUDA_MODULE> <CUDNN_MODULE>   # <-- EDIT per your cluster
source "${VENV}/bin/activate"
export TF_USE_LEGACY_KERAS=1
cd "${REPO_ROOT}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
nvidia-smi
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

# Optional arguments of train.py:
#   --epochs N           number of epochs (the default is the full training setting)
#   --causal-encoder     train the causal form of the encoder
python train.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}"

echo "finished $(date); the run directory is under ${OUT_DIR}"
