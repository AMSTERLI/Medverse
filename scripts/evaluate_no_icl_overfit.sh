#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${MEDVERSE_PYTHON:-python}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
ROOT=${PROJECT_DIR}/work/experiments/overfit
MANIFEST=${ROOT}/manifest.jsonl
GATES=${PROJECT_DIR}/work/experiments/gates
RESULTS=${ROOT}/no_icl/checkpoints
INPUT=${ROOT}/no_icl/input
ROI_PREDICTIONS=${ROOT}/no_icl/roi_predictions
RESTORED=${ROOT}/no_icl/restored
METRICS=${ROOT}/no_icl/metrics
CHECKPOINT=${RESULTS}/Dataset501_PanCancerTotalSegROI/nnUNetTrainer_20epochs__nnUNetPlans_MixedChannels__3d_fullres/fold_1/checkpoint_best.pth

for path in "${MANIFEST}" "${CHECKPOINT}" "${INPUT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required input does not exist: ${path}" >&2
    exit 2
  fi
done

cd "${PROJECT_DIR}"
mkdir -p "${ROI_PREDICTIONS}" "${RESTORED}" "${METRICS}" "${GATES}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${RESULTS}
export nnUNet_compile=${NNUNET_COMPILE:-False}

nnUNetv2_predict -i "${INPUT}" -o "${ROI_PREDICTIONS}" \
  -d 501 -c 3d_fullres -f 1 -tr nnUNetTrainer_20epochs \
  -p nnUNetPlans_MixedChannels -chk checkpoint_best.pth
"${PYTHON}" scripts/restore_nnunet_predictions.py --manifest "${MANIFEST}" --split train \
  --roi-predictions "${ROI_PREDICTIONS}" --output-dir "${RESTORED}"
"${PYTHON}" scripts/evaluate_liver_kidney_original_space.py --manifest "${MANIFEST}" --split train \
  --predictions "${RESTORED}/predictions.jsonl" --output-dir "${METRICS}"
"${PYTHON}" scripts/check_overfit_gate.py --metrics-csv "${METRICS}/per_case_metrics.csv" \
  --gate "${GATES}/no_icl_overfit.pass" --expected-cases 8
