#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-start}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON=${MEDVERSE_PYTHON:-python}
EXPERIMENT_DIR=${ICL_EXPERIMENT_DIR:-${PROJECT_DIR}/work/experiments/icl_medverse_bam_k3}
MANIFEST=${MAIN_ROI_MANIFEST:-${PROJECT_DIR}/work/data/derived_liver_kidney_v1/manifests/roi_manifest.jsonl}
PRETRAINED=${MEDVERSE_PRETRAINED:-${PROJECT_DIR}/work/pretrained/Medverse.ckpt}
PLAN_SNAPSHOT=${NNUNET_PLAN_SNAPSHOT:-${PROJECT_DIR}/work/experiments/nnunet_plan_snapshot.json}
CHECKPOINT_DIR=${EXPERIMENT_DIR}/checkpoints
mkdir -p "${CHECKPOINT_DIR}"
[[ -f "${PLAN_SNAPSHOT}" ]] || {
  echo "missing nnU-Net planner snapshot: ${PLAN_SNAPSHOT}" >&2
  exit 4
}
TARGET_SPACING=$("${PYTHON}" -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["spacing_xyz_mm"])))' "${PLAN_SNAPSHOT}")

COMMAND=(
  "${PYTHON}" scripts/train_pan_cancer_icl.py
  --manifest "${MANIFEST}"
  --output-dir "${CHECKPOINT_DIR}"
  --image-size 128
  --target-spacing "${TARGET_SPACING}"
  --organ-channel
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

echo "gpu=${CUDA_VISIBLE_DEVICES:-unset} context_k=3 bam=original ccti=disabled organ_channel=enabled spacing=${TARGET_SPACING} mode=${MODE}"
cd "${PROJECT_DIR}"
exec "${COMMAND[@]}"
