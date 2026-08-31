#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-start}
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-/home_data/home/wangyb12023/Medverse-PAOT2}
PYTHON=${MEDVERSE_PYTHON:-/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python}
EXPERIMENT_DIR=${ICL_EXPERIMENT_DIR:-${PROJECT_DIR}/work/experiments/icl_medverse_bam_k3}
MANIFEST=${MAIN_ROI_MANIFEST:-${PROJECT_DIR}/work/experiments/shared_roi_manifest.jsonl}
PRETRAINED=${MEDVERSE_PRETRAINED:-${PROJECT_DIR}/work/pretrained/Medverse.ckpt}
CHECKPOINT_DIR=${EXPERIMENT_DIR}/checkpoints
mkdir -p "${CHECKPOINT_DIR}"

COMMAND=(
  "${PYTHON}" scripts/train_pan_cancer_icl.py
  --manifest "${MANIFEST}"
  --output-dir "${CHECKPOINT_DIR}"
  --image-size 128
  --target-spacing 1.5,1.5,2.0
  --positive-patch-probability 0.67
  --positive-weight 8
  --loss-mode smoothl3_dice
  --prediction-mode regression
  --val-overlap 0.5
  --val-cases-per-task 100000
  --channels 32,64,128,256,512
  --conv-layers 2
  --num-context 3
  --batch-size 1
  --gradient-accumulation 2
  --workers 4
  --epochs 20
  --freeze-backbone-epochs 1
  --unfreeze-scope decoder
  --lr 1e-4
  --unfreeze-lr 3e-5
  --weight-decay 0
  --disable-ccti
  --seed 20260831
  --device cuda
)

if [[ "${MODE}" == "resume" ]]; then
  COMMAND+=(--resume "${CHECKPOINT_DIR}/last.pt")
elif [[ "${MODE}" == "start" ]]; then
  COMMAND+=(--pretrained "${PRETRAINED}")
else
  echo "mode must be start or resume" >&2
  exit 2
fi

echo "gpu=${CUDA_VISIBLE_DEVICES:-unset} context_k=3 bam=original ccti=disabled mode=${MODE}"
cd "${PROJECT_DIR}"
exec "${COMMAND[@]}"
