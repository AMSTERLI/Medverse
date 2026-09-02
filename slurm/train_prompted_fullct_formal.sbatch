#!/bin/bash
#SBATCH --job-name=pfc_train_full
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/no_icl_prompted_fullct
RESUME=${RESUME:-0}

cd "${PROJECT_DIR}"
test -f "${ROOT}/PREPARED"
mkdir -p work/slurm "${ROOT}/logs" "${ROOT}/checkpoints"

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${ROOT}/checkpoints
export nnUNet_compile=False

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "${ROOT}/gpu_${SLURM_JOB_ID}.txt"
python scripts/install_nnunet_prompted_fullct.py

TRAIN_ARGS=(
  502 3d_fullres 0
  -tr nnUNetTrainerPromptedFullCT_100epochs
  -p nnUNetPlans_PromptedFullCT
  --npz
)
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--c)
fi

nnUNetv2_train "${TRAIN_ARGS[@]}"
touch "${ROOT}/TRAINING_COMPLETED"
