#!/bin/bash
#SBATCH --job-name=medverse_ccti
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

MEDVERSE_PROJECT_DIR=/home_data/home/wangyb12023/Medverse-PAOT2
MEDVERSE_PYTHON=/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python
MEDVERSE_MANIFEST=${MEDVERSE_PROJECT_DIR}/work/paot2_icl_inspected.jsonl
MEDVERSE_PRETRAINED=${MEDVERSE_PROJECT_DIR}/work/pretrained/Medverse.ckpt
MEDVERSE_RUN_DIR=${MEDVERSE_PROJECT_DIR}/work/runs/pretrained_ccti_seed17

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
  --image-size 128 \
  --target-spacing 2.0,2.0,3.0 \
  --positive-patch-probability 1.0 \
  --positive-weight 8 \
  --loss-mode smoothl3_dice \
  --prediction-mode regression \
  --val-overlap 0.5 \
  --val-cases-per-task 20 \
  --channels 32,64,128,256,512 \
  --conv-layers 2 \
  --num-context 2 \
  --batch-size 1 \
  --workers 4 \
  --epochs 5 \
  --pretrained "${MEDVERSE_PRETRAINED}" \
  --freeze-backbone-epochs 1 \
  --unfreeze-scope decoder \
  --lr 1e-4 \
  --unfreeze-lr 3e-5 \
  --weight-decay 0 \
  --ccti-mode learned \
  --ccti-ratio 0.25 \
  --seed 17 \
  --device cuda

echo "end_time=$(date --iso-8601=seconds)"
