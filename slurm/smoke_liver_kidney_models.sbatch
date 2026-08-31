#!/bin/bash
#SBATCH --job-name=lk_model_smoke
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${MEDVERSE_PYTHON:-python}
cd "${PROJECT_DIR}"
mkdir -p work/slurm work/experiments/smoke
"${PYTHON}" scripts/smoke_test_liver_kidney_models.py \
  --device cuda \
  --output work/experiments/smoke/model_smoke.json
