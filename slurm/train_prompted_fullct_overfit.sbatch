#!/bin/bash
#SBATCH --job-name=pfc_overfit
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/overfit_prompted_fullct
cd "${PROJECT_DIR}"
export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${ROOT}/checkpoints
export nnUNet_compile=False
python scripts/install_nnunet_prompted_fullct.py
nnUNetv2_train 503 3d_fullres 0 \
  -tr nnUNetTrainerPromptedFullCT_50epochs \
  -p nnUNetPlans_PromptedFullCT --npz
nnUNetv2_predict -i "${ROOT}/input" -o "${ROOT}/predictions" \
  -d 503 -c 3d_fullres -f 0 \
  -tr nnUNetTrainerPromptedFullCT_50epochs \
  -p nnUNetPlans_PromptedFullCT -chk checkpoint_final.pth
python scripts/build_nnunet_fullct_prediction_manifest.py \
  --manifest "${ROOT}/manifest.jsonl" --split train \
  --prediction-dir "${ROOT}/predictions" \
  --output "${ROOT}/predictions.jsonl"
python scripts/evaluate_liver_kidney_original_space.py \
  --manifest "${ROOT}/manifest.jsonl" --split train \
  --predictions "${ROOT}/predictions.jsonl" \
  --prediction-label-value 2 --output-dir "${ROOT}/metrics"
python scripts/check_overfit_gate.py \
  --metrics-csv "${ROOT}/metrics/per_case_metrics.csv" \
  --gate "${PROJECT_DIR}/work/experiments/gates/no_icl_prompted_fullct_overfit.pass" \
  --expected-cases 4
