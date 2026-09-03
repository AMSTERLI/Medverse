#!/bin/bash
#SBATCH --job-name=pfc_val_best
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/no_icl_prompted_fullct

cd "${PROJECT_DIR}"
test -f "${ROOT}/PREPARED"
test -f "${ROOT}/checkpoints/Dataset502_PromptedFullCT/nnUNetTrainerPromptedFullCT_100epochs__nnUNetPlans_PromptedFullCT__3d_fullres/fold_0/checkpoint_best.pth"
mkdir -p work/slurm "${ROOT}/logs"

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${ROOT}/checkpoints
export nnUNet_compile=False

GT_DIR=${nnUNet_preprocessed}/Dataset502_PromptedFullCT/gt_segmentations
if [[ ! -e "${GT_DIR}" ]]; then
  ln -s "${nnUNet_raw}/Dataset502_PromptedFullCT/labelsTr" "${GT_DIR}"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "${ROOT}/gpu_validation_${SLURM_JOB_ID}.txt"
python scripts/install_nnunet_prompted_fullct.py

nnUNetv2_train \
  502 3d_fullres 0 \
  -tr nnUNetTrainerPromptedFullCT_100epochs \
  -p nnUNetPlans_PromptedFullCT \
  --val \
  --val_best

touch "${ROOT}/VALIDATION_BEST_COMPLETED"
