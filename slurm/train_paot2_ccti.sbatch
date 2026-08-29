#!/bin/bash
#SBATCH --job-name=medverse_ccti
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

MEDVERSE_PROJECT_DIR=/home_data/home/wangyb12023/Medverse-PAOT2
MEDVERSE_PYTHON=/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python
MEDVERSE_MANIFEST=${MEDVERSE_PROJECT_DIR}/work/paot2_icl_uninspected.jsonl
MEDVERSE_RUN_DIR=${MEDVERSE_PROJECT_DIR}/work/runs/learned_seed17_64

cd "${MEDVERSE_PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start_time=$(date --iso-8601=seconds)"
nvidia-smi
"${MEDVERSE_PYTHON}" -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0)})"

"${MEDVERSE_PYTHON}" scripts/train_pan_cancer_icl.py \
  --manifest "${MEDVERSE_MANIFEST}" \
  --output-dir "${MEDVERSE_RUN_DIR}" \
  --image-size 64 \
  --channels 8,16,32,64,128 \
  --conv-layers 2 \
  --num-context 1 \
  --batch-size 1 \
  --workers 4 \
  --epochs 20 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --ccti-mode learned \
  --ccti-ratio 0.25 \
  --seed 17 \
  --allow-uninspected-context \
  --device cuda

echo "end_time=$(date --iso-8601=seconds)"
