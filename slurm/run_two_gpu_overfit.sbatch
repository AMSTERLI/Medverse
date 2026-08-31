#!/bin/bash
#SBATCH --job-name=lk_overfit2
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
cd "${PROJECT_DIR}"
export MEDVERSE_PROJECT_DIR=${PROJECT_DIR}
echo "job_id=${SLURM_JOB_ID} host=$(hostname) visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash scripts/run_two_gpu_overfit.sh
