#!/bin/bash
#SBATCH --job-name=pfc_prepare
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${MEDVERSE_PYTHON:-python}
FIXED_MANIFEST=${FIXED_MANIFEST:-${PROJECT_DIR}/work/manifests/kits23_lits_mswal_liver_kidney_fixed.jsonl}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/overfit_prompted_fullct
SOURCE_MANIFEST=${ROOT}/source_manifest.jsonl
EXPORTED_MANIFEST=${ROOT}/manifest.jsonl
DATASET_ID=503
DATASET_NAME=PromptedFullCTOverfit
DATASET=Dataset503_PromptedFullCTOverfit

cd "${PROJECT_DIR}"
mkdir -p work/slurm "${ROOT}"
"${PYTHON}" scripts/prepare_overfit_manifest.py \
  --manifest "${FIXED_MANIFEST}" --output "${SOURCE_MANIFEST}" \
  --cases-per-task 2 --context-k 1

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${ROOT}/checkpoints
"${PYTHON}" scripts/export_prompted_fullct_nnunet_dataset.py \
  --manifest "${SOURCE_MANIFEST}" --nnunet-raw "${nnUNet_raw}" \
  --nnunet-preprocessed "${nnUNet_preprocessed}" --output-manifest "${EXPORTED_MANIFEST}" \
  --dataset-id "${DATASET_ID}" --dataset-name "${DATASET_NAME}" --workers 4
"${PYTHON}" scripts/configure_prompted_nnunet_overfit_split.py \
  --manifest "${EXPORTED_MANIFEST}" \
  --splits-file "${nnUNet_preprocessed}/${DATASET}/splits_final.json" \
  --prediction-input "${ROOT}/input" --fold 0
"${PYTHON}" scripts/install_nnunet_prompted_fullct.py
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 3d_fullres -np 4 \
  --verify_dataset_integrity \
  -preprocessor_name PromptedFullCTPreprocessor \
  -overwrite_plans_name nnUNetPlans_PromptedFullCT

"${PYTHON}" scripts/snapshot_nnunetv2_plan.py \
  --plans "${nnUNet_preprocessed}/${DATASET}/nnUNetPlans_PromptedFullCT.json" \
  --fingerprint "${nnUNet_preprocessed}/${DATASET}/dataset_fingerprint.json" \
  --configuration 3d_fullres --output "${ROOT}/plan_snapshot.json"
echo "prepared_manifest=${EXPORTED_MANIFEST}"
