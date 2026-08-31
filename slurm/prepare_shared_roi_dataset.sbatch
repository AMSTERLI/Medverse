#!/bin/bash
#SBATCH --job-name=pan_ca_roi
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${DATA_PYTHON:-python}
NNUNET_BIN=${NNUNET_BIN:-nnUNetv2_plan_and_preprocess}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${PROJECT_DIR}/work/manifests/paot2_manifest.jsonl}
DERIVED_ROOT=${DERIVED_ROOT:-${PROJECT_DIR}/work/data/derived_liver_kidney_v1}
FIXED_MANIFEST=${DERIVED_ROOT}/manifests/fixed_split.jsonl
ROI_ROOT=${DERIVED_ROOT}
ROI_MANIFEST=${DERIVED_ROOT}/manifests/roi_manifest.jsonl
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"${PYTHON}" scripts/prepare_main_experiment_manifest.py \
  --input "${SOURCE_MANIFEST}" \
  --output "${FIXED_MANIFEST}" \
  --totalseg-root "${DERIVED_ROOT}" \
  --inspect-labels \
  --context-k 3 \
  --seed 20260831

"${PYTHON}" scripts/materialize_totalseg_roi_dataset.py \
  --manifest "${FIXED_MANIFEST}" \
  --output-root "${ROI_ROOT}" \
  --output-manifest "${ROI_MANIFEST}" \
  --margin-mm 30 \
  --workers "${SLURM_CPUS_PER_TASK}"

"${PYTHON}" scripts/export_nnunetv2_dataset.py \
  --manifest "${ROI_MANIFEST}" \
  --nnunet-raw "${NNUNET_STORAGE}/nnUNet_raw" \
  --nnunet-preprocessed "${NNUNET_STORAGE}/nnUNet_preprocessed" \
  --dataset-id 501 \
  --dataset-name LiverKidneyTotalSegROI

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${NNUNET_STORAGE}/nnUNet_results
"${NNUNET_BIN}" -d 501 --verify_dataset_integrity

DATASET_DIR=${nnUNet_preprocessed}/Dataset501_LiverKidneyTotalSegROI
"${PYTHON}" scripts/snapshot_nnunetv2_plan.py \
  --plans "${DATASET_DIR}/nnUNetPlans.json" \
  --fingerprint "${DATASET_DIR}/dataset_fingerprint.json" \
  --output work/experiments/nnunet_plan_snapshot.json
