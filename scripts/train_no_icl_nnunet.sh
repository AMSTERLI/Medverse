#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-start}
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-/home_data/home/wangyb12023/Medverse-PAOT2}
EXPERIMENT_DIR=${NO_ICL_EXPERIMENT_DIR:-${PROJECT_DIR}/work/experiments/no_icl_nnunet}
NNUNET_STORAGE=${NNUNET_STORAGE:-${PROJECT_DIR}/work/experiments/nnunet_storage}
DATASET_ID=${NNUNET_DATASET_ID:-501}
TRAINER=${NNUNET_TRAINER:-nnUNetTrainer_100epochs}

export nnUNet_raw=${NNUNET_STORAGE}/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_STORAGE}/nnUNet_preprocessed
export nnUNet_results=${EXPERIMENT_DIR}/checkpoints
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

command -v nnUNetv2_train >/dev/null 2>&1 || {
  echo "nnUNetv2_train is not installed in the active environment" >&2
  exit 127
}

COMMAND=(nnUNetv2_train "${DATASET_ID}" 3d_fullres 0 -tr "${TRAINER}" --npz)
if [[ "${MODE}" == "resume" ]]; then
  COMMAND+=(--c)
elif [[ "${MODE}" != "start" ]]; then
  echo "mode must be start or resume" >&2
  exit 2
fi

echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "nnunet_dataset=${DATASET_ID} configuration=3d_fullres fold=0 trainer=${TRAINER} mode=${MODE}"
exec "${COMMAND[@]}"
