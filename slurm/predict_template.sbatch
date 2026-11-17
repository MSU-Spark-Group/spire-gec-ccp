#!/bin/bash --login
#SBATCH --job-name=spire-predict
#SBATCH --account=<ACCOUNT>                    # <-- EDIT
#SBATCH --partition=<PARTITION>                # <-- EDIT
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=<MEMORY>                         # <-- EDIT
#SBATCH --gpus=1                               # <-- EDIT per your cluster's GPU directive
#SBATCH --output=slurm-%x-%j.out

# Reconstruction of one case with a checkpoint written by train.py:
# prints the RMSE of log10(n_e) and writes a picture of the reconstructed field,
# the fluid-model field and their difference at one sampled time.
set -euo pipefail

REPO_ROOT="/path/to/spire-gec-ccp"             # <-- EDIT
VENV="${REPO_ROOT}/.venv"                      # <-- EDIT
CHECKPOINT="/path/to/run/model-<epoch>-<loss>-<rmse>.h5"   # <-- EDIT: a checkpoint of the run directory
CASE="/path/to/case.npz"                       # <-- EDIT: one case file
FRAME=0                                        # which of the 20 sampled times to draw (0 to 19)
OUT="${REPO_ROOT}/spire_prediction.png"        # <-- EDIT
# module load <PYTHON_MODULE> <CUDA_MODULE> <CUDNN_MODULE>   # <-- EDIT per your cluster
source "${VENV}/bin/activate"
export TF_USE_LEGACY_KERAS=1
cd "${REPO_ROOT}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"

# Add --causal-encoder when the checkpoint was trained with it.
python predict.py \
  --checkpoint "${CHECKPOINT}" \
  --case "${CASE}" \
  --frame "${FRAME}" \
  --out "${OUT}"
