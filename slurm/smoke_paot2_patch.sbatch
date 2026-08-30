#!/bin/bash
#SBATCH --job-name=medverse_smoke
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

cd "${MEDVERSE_PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
nvidia-smi

"${MEDVERSE_PYTHON}" scripts/train_pan_cancer_icl.py \
  --manifest work/paot2_icl_inspected.jsonl \
  --output-dir work/runs/smoke_patch128 \
  --image-size 128 \
  --target-spacing 2.0,2.0,3.0 \
  --channels 8,16,32,64,128 \
  --conv-layers 2 \
  --num-context 1 \
  --batch-size 1 \
  --workers 2 \
  --epochs 1 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --positive-patch-probability 0.7 \
  --positive-weight 8 \
  --val-overlap 0.5 \
  --ccti-mode learned \
  --ccti-ratio 0.25 \
  --seed 17 \
  --device cuda
