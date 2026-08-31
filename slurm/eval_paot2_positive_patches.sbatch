#!/bin/bash
#SBATCH --job-name=paot2_pos_eval
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

MEDVERSE_PROJECT_DIR=/home_data/home/wangyb12023/Medverse-PAOT2
MEDVERSE_PYTHON=/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python
MEDVERSE_MANIFEST=${MEDVERSE_PROJECT_DIR}/work/paot2_icl_inspected.jsonl

cd "${MEDVERSE_PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start_time=$(date --iso-8601=seconds)"

COMMON_ARGS=(
  --manifest "${MEDVERSE_MANIFEST}"
  --image-size 128
  --target-spacing 2.0,2.0,3.0
  --positive-patch-probability 1.0
  --positive-weight 8
  --loss-mode smoothl3_dice
  --prediction-mode regression
  --val-cases-per-task 5
  --val-positive-patches
  --channels 32,64,128,256,512
  --conv-layers 2
  --num-context 2
  --batch-size 1
  --workers 0
  --eval-only
  --seed 17
  --device cuda
)

echo "evaluation=ccti"
"${MEDVERSE_PYTHON}" scripts/train_pan_cancer_icl.py \
  "${COMMON_ARGS[@]}" \
  --output-dir work/eval/positive_patch_ccti \
  --checkpoint work/runs/pretrained_ccti_seed17/best.pt \
  --ccti-mode learned \
  --ccti-ratio 0.25

echo "evaluation=no_ccti"
"${MEDVERSE_PYTHON}" scripts/train_pan_cancer_icl.py \
  "${COMMON_ARGS[@]}" \
  --output-dir work/eval/positive_patch_no_ccti \
  --checkpoint work/runs/pretrained_no_ccti_seed17/best.pt \
  --disable-ccti

echo "end_time=$(date --iso-8601=seconds)"
