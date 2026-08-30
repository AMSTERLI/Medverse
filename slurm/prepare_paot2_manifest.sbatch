#!/bin/bash
#SBATCH --job-name=paot2_manifest
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

MEDVERSE_PROJECT_DIR=/home_data/home/wangyb12023/Medverse-PAOT2
MEDVERSE_PYTHON=/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python

cd "${MEDVERSE_PROJECT_DIR}"
export PYTHONUNBUFFERED=1
"${MEDVERSE_PYTHON}" scripts/prepare_paot2_manifest.py \
  --train-pairs data_lists/PAOT2_train.txt \
  --val-pairs data_lists/PAOT2_val.txt \
  --data-root /public_bme2/bme-dgshen/RunqiMeng/project \
  --inspect-labels \
  --keep-empty \
  --progress-every 50 \
  --output work/paot2_icl_inspected.jsonl

"${MEDVERSE_PYTHON}" scripts/validate_pan_cancer_manifest.py work/paot2_icl_inspected.jsonl
