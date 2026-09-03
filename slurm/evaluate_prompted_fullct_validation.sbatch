#!/bin/bash
#SBATCH --job-name=pfc_eval_val
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/no_icl_prompted_fullct
DATASET=Dataset502_PromptedFullCT
MODEL_DIR=${ROOT}/checkpoints/${DATASET}/nnUNetTrainerPromptedFullCT_100epochs__nnUNetPlans_PromptedFullCT__3d_fullres
VAL_DIR=${MODEL_DIR}/fold_0/validation
GT_DIR=${NNUNET_STORAGE}/nnUNet_preprocessed/${DATASET}/gt_segmentations
RAW_LABELS=${NNUNET_STORAGE}/nnUNet_raw/${DATASET}/labelsTr

cd "${PROJECT_DIR}"
test -d "${RAW_LABELS}"
test -d "${VAL_DIR}"
test -f "${MODEL_DIR}/plans.json"
test -f "${MODEL_DIR}/dataset.json"

if [[ ! -e "${GT_DIR}" ]]; then
  ln -s "${RAW_LABELS}" "${GT_DIR}"
fi

nnUNetv2_evaluate_folder \
  "${GT_DIR}" \
  "${VAL_DIR}" \
  -djfile "${MODEL_DIR}/dataset.json" \
  -pfile "${MODEL_DIR}/plans.json" \
  -o "${VAL_DIR}/summary.json" \
  -np "${SLURM_CPUS_PER_TASK}" \
  --chill

touch "${ROOT}/VALIDATION_BEST_COMPLETED"
