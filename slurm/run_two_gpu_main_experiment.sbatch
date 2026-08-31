#!/bin/bash
#SBATCH --job-name=pan_ca_main2
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:2
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

MEDVERSE_PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
cd "${MEDVERSE_PROJECT_DIR}"
export MEDVERSE_PROJECT_DIR
export OMP_NUM_THREADS=8

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash scripts/run_two_gpu_main_experiment.sh both
