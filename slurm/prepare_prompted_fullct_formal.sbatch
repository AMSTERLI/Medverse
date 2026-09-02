#!/bin/bash
#SBATCH --job-name=pfc_prepare_full
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${MEDVERSE_PYTHON:-python}
FIXED_MANIFEST=${FIXED_MANIFEST:-${PROJECT_DIR}/work/manifests/kits23_lits_mswal_liver_kidney_fixed.jsonl}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/no_icl_prompted_fullct
EXPORTED_MANIFEST=${ROOT}/manifest.jsonl
REUSE_EXPORT=${REUSE_EXPORT:-0}
DATASET_ID=502
DATASET_NAME=PromptedFullCT
DATASET=Dataset502_PromptedFullCT

cd "${PROJECT_DIR}"
mkdir -p work/slurm "${ROOT}/logs" "${ROOT}/checkpoints"
test -f "${PROJECT_DIR}/work/experiments/gates/no_icl_prompted_fullct_overfit.pass"
test -f "${FIXED_MANIFEST}"

cp configs/main_experiment/no_icl_prompted_fullct.yaml "${ROOT}/config.yaml"
git rev-parse HEAD > "${ROOT}/git_commit.txt"
"${PYTHON}" --version > "${ROOT}/python_version.txt" 2>&1

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${ROOT}/checkpoints

if [[ "${REUSE_EXPORT}" == "1" ]]; then
  test -f "${EXPORTED_MANIFEST}"
  test -f "${nnUNet_raw}/${DATASET}/dataset.json"
  test -f "${nnUNet_preprocessed}/${DATASET}/splits_final.json"
else
  "${PYTHON}" scripts/export_prompted_fullct_nnunet_dataset.py \
    --manifest "${FIXED_MANIFEST}" \
    --nnunet-raw "${nnUNet_raw}" \
    --nnunet-preprocessed "${nnUNet_preprocessed}" \
    --output-manifest "${EXPORTED_MANIFEST}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --workers 4
fi

"${PYTHON}" scripts/check_prompted_fullct_formal_export.py \
  --manifest "${EXPORTED_MANIFEST}" \
  --splits-file "${nnUNet_preprocessed}/${DATASET}/splits_final.json" \
  --dataset-json "${nnUNet_raw}/${DATASET}/dataset.json" \
  --expected-train 611 --expected-val 87 --expected-test 176

"${PYTHON}" scripts/install_nnunet_prompted_fullct.py
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 3d_fullres -np 4 \
  --verify_dataset_integrity \
  -preprocessor_name PromptedFullCTPreprocessor \
  -overwrite_plans_name nnUNetPlans_PromptedFullCT

"${PYTHON}" scripts/snapshot_nnunetv2_plan.py \
  --plans "${nnUNet_preprocessed}/${DATASET}/nnUNetPlans_PromptedFullCT.json" \
  --fingerprint "${nnUNet_preprocessed}/${DATASET}/dataset_fingerprint.json" \
  --configuration 3d_fullres \
  --output "${ROOT}/plan_snapshot.json"

touch "${ROOT}/PREPARED"
echo "prepared_manifest=${EXPORTED_MANIFEST}"
