#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${MEDVERSE_PYTHON:-python}
RUN_TARGET=${1:-both}
ROI_MANIFEST=${MAIN_ROI_MANIFEST:-${PROJECT_DIR}/work/data/kits23_lits_mswal_liver_kidney_v1/manifests/roi_manifest.jsonl}
PLAN_SNAPSHOT=${NNUNET_PLAN_SNAPSHOT:-${PROJECT_DIR}/work/experiments/nnunet_plan_snapshot.json}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
PRETRAINED=${MEDVERSE_PRETRAINED:-${PROJECT_DIR}/work/pretrained/Medverse.ckpt}
ROOT=${PROJECT_DIR}/work/experiments/overfit
MANIFEST=${ROOT}/manifest.jsonl
GATES=${PROJECT_DIR}/work/experiments/gates
DATASET=Dataset501_PanCancerTotalSegROI

case "${RUN_TARGET}" in
  both|no-icl|icl) ;;
  *) echo "usage: $0 [both|no-icl|icl]" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
mkdir -p "${ROOT}" "${GATES}"
"${PYTHON}" scripts/prepare_overfit_manifest.py --manifest "${ROI_MANIFEST}" \
  --output "${MANIFEST}" --cases-per-task 4 --context-k 3
"${PYTHON}" scripts/configure_nnunet_overfit_split.py --manifest "${MANIFEST}" \
  --splits-file "${NNUNET_STORAGE}/nnUNet_preprocessed/${DATASET}/splits_final.json" \
  --prediction-input "${ROOT}/no_icl/input" --fold 1
SPACING=$("${PYTHON}" scripts/snapshot_nnunetv2_plan.py \
  --plans "${NNUNET_STORAGE}/nnUNet_preprocessed/${DATASET}/nnUNetPlans_MixedChannels.json" \
  --fingerprint "${NNUNET_STORAGE}/nnUNet_preprocessed/${DATASET}/dataset_fingerprint.json" \
  --output "${PLAN_SNAPSHOT}" --print-spacing)

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed

run_no_icl() {
  export CUDA_VISIBLE_DEVICES=0
  export nnUNet_results=${ROOT}/no_icl/checkpoints
  mkdir -p "${nnUNet_results}" "${ROOT}/no_icl/roi_predictions" "${ROOT}/no_icl/restored"
  nnUNetv2_train 501 3d_fullres 1 -tr nnUNetTrainer_20epochs -p nnUNetPlans_MixedChannels --npz
  nnUNetv2_predict -i "${ROOT}/no_icl/input" -o "${ROOT}/no_icl/roi_predictions" \
    -d 501 -c 3d_fullres -f 1 -tr nnUNetTrainer_20epochs -p nnUNetPlans_MixedChannels -chk checkpoint_best.pth
  "${PYTHON}" scripts/restore_nnunet_predictions.py --manifest "${MANIFEST}" --split train \
    --roi-predictions "${ROOT}/no_icl/roi_predictions" --output-dir "${ROOT}/no_icl/restored"
  "${PYTHON}" scripts/evaluate_liver_kidney_original_space.py --manifest "${MANIFEST}" --split train \
    --predictions "${ROOT}/no_icl/restored/predictions.jsonl" --output-dir "${ROOT}/no_icl/metrics"
  "${PYTHON}" scripts/check_overfit_gate.py --metrics-csv "${ROOT}/no_icl/metrics/per_case_metrics.csv" \
    --gate "${GATES}/no_icl_overfit.pass" --expected-cases 8
}

run_icl() {
  local gpu=1
  [[ "${RUN_TARGET}" == "icl" ]] && gpu=0
  export CUDA_VISIBLE_DEVICES=${gpu}
  local out=${ROOT}/icl_bam_k3/checkpoints
  mkdir -p "${out}" "${ROOT}/icl_bam_k3/predictions"
  "${PYTHON}" scripts/train_pan_cancer_icl.py --manifest "${MANIFEST}" \
    --train-split train --val-split train --output-dir "${out}" --epochs 50 \
    --image-size 128 --target-spacing "${SPACING}" --organ-channel \
    --positive-patch-probability 1 --positive-weight 8 --loss-mode smoothl3_dice \
    --prediction-mode regression --val-positive-patches --val-cases-per-task 4 \
    --channels 32,64,128,256,512 --conv-layers 2 --num-context 3 --batch-size 1 \
    --gradient-accumulation 2 --workers 2 --unfreeze-scope all --lr 1e-4 \
    --unfreeze-lr 1e-4 --weight-decay 0 --disable-ccti --seed 20260831 \
    --device cuda --pretrained "${PRETRAINED}" --log-every 4
  "${PYTHON}" scripts/infer_pan_cancer_icl.py --manifest "${MANIFEST}" --split train \
    --checkpoint "${out}/best.pt" --output-dir "${ROOT}/icl_bam_k3/predictions" \
    --target-spacing "${SPACING}" --device cuda
  "${PYTHON}" scripts/evaluate_liver_kidney_original_space.py --manifest "${MANIFEST}" --split train \
    --predictions "${ROOT}/icl_bam_k3/predictions/predictions.jsonl" \
    --output-dir "${ROOT}/icl_bam_k3/metrics"
  "${PYTHON}" scripts/check_overfit_gate.py --metrics-csv "${ROOT}/icl_bam_k3/metrics/per_case_metrics.csv" \
    --gate "${GATES}/icl_bam_k3_overfit.pass" --expected-cases 8
}

declare -a PIDS=()
if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "no-icl" ]]; then
  run_no_icl >"${ROOT}/no_icl.log" 2>&1 &
  PID0=$!
  PIDS+=("${PID0}")
  echo "no_icl_pid=${PID0} gpu=0"
fi
if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "icl" ]]; then
  ICL_GPU=1
  [[ "${RUN_TARGET}" == "icl" ]] && ICL_GPU=0
  run_icl >"${ROOT}/icl_bam_k3.log" 2>&1 &
  PID1=$!
  PIDS+=("${PID1}")
  echo "icl_bam_k3_pid=${PID1} gpu=${ICL_GPU}"
fi
STATUS=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || STATUS=$?
done
exit "${STATUS}"
