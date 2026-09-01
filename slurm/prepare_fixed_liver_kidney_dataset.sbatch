#!/bin/bash
#SBATCH --job-name=lk_prepare
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${MEDVERSE_PYTHON:-python}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${PROJECT_DIR}/data/manifests/kits23_lits_mswal_tumor_v1.jsonl}
DERIVED_ROOT=${DERIVED_ROOT:-${PROJECT_DIR}/work/data/kits23_lits_mswal_liver_kidney_v1}
FIXED_MANIFEST=${FIXED_MANIFEST:-${PROJECT_DIR}/work/manifests/kits23_lits_mswal_liver_kidney_fixed.jsonl}
ROI_MANIFEST=${ROI_MANIFEST:-${DERIVED_ROOT}/manifests/roi_manifest.jsonl}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
DATASET_ID=${NNUNET_DATASET_ID:-501}
DATASET_NAME=${NNUNET_DATASET_NAME:-PanCancerTotalSegROI}

cd "${PROJECT_DIR}"
mkdir -p work/slurm "$(dirname "${FIXED_MANIFEST}")" "${DERIVED_ROOT}/manifests"

"${PYTHON}" scripts/prepare_main_experiment_manifest.py \
  --input "${SOURCE_MANIFEST}" \
  --output "${FIXED_MANIFEST}" \
  --totalseg-root "${DERIVED_ROOT}" \
  --context-k 3 \
  --preserve-existing-split

"${PYTHON}" scripts/materialize_totalseg_roi_dataset.py \
  --manifest "${FIXED_MANIFEST}" \
  --output-root "${DERIVED_ROOT}" \
  --output-manifest "${ROI_MANIFEST}" \
  --margin-mm 30 \
  --workers 8 \
  --qc-samples-per-stratum 3 \
  --fail-on-geometric-qc

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${PROJECT_DIR}/work/experiments/no_icl_nnunet/checkpoints
"${PYTHON}" scripts/export_nnunetv2_dataset.py \
  --manifest "${ROI_MANIFEST}" \
  --nnunet-raw "${nnUNet_raw}" \
  --nnunet-preprocessed "${nnUNet_preprocessed}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-name "${DATASET_NAME}"

"${PYTHON}" scripts/install_nnunet_mixed_preprocessor.py
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 3d_fullres -np 8 \
  --verify_dataset_integrity \
  -preprocessor_name MixedChannelPreprocessor \
  -overwrite_plans_name nnUNetPlans_MixedChannels

DATASET_DIR=${nnUNet_preprocessed}/Dataset$(printf '%03d' "${DATASET_ID}")_${DATASET_NAME}
"${PYTHON}" scripts/snapshot_nnunetv2_plan.py \
  --plans "${DATASET_DIR}/nnUNetPlans_MixedChannels.json" \
  --fingerprint "${DATASET_DIR}/dataset_fingerprint.json" \
  --configuration 3d_fullres \
  --output "${PROJECT_DIR}/work/experiments/nnunet_plan_snapshot.json"

echo "prepared_roi_manifest=${ROI_MANIFEST}"
echo "nnunet_preprocessed=${DATASET_DIR}"
