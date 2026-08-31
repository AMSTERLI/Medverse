#!/bin/bash
#SBATCH --job-name=pan_ca_roi
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/home_data/home/wangyb12023/Medverse-PAOT2
PYTHON=/home_data/home/wangyb12023/anaconda3/envs/renal/bin/python
FIXED_MANIFEST=${PROJECT_DIR}/work/experiments/main_fixed_split_manifest.jsonl
ROI_ROOT=${PROJECT_DIR}/work/experiments/shared_roi
ROI_MANIFEST=${PROJECT_DIR}/work/experiments/shared_roi_manifest.jsonl
NNUNET_STORAGE=${PROJECT_DIR}/work/experiments/nnunet_storage

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"${PYTHON}" scripts/materialize_totalseg_roi_dataset.py \
  --manifest "${FIXED_MANIFEST}" \
  --output-root "${ROI_ROOT}" \
  --output-manifest "${ROI_MANIFEST}" \
  --spacing 1.5,1.5,2.0 \
  --margin-mm 30 \
  --hu-window=-1000,1000 \
  --workers "${SLURM_CPUS_PER_TASK}"

"${PYTHON}" scripts/export_nnunetv2_dataset.py \
  --manifest "${ROI_MANIFEST}" \
  --nnunet-raw "${NNUNET_STORAGE}/nnUNet_raw" \
  --nnunet-preprocessed "${NNUNET_STORAGE}/nnUNet_preprocessed" \
  --dataset-id 501 \
  --dataset-name PanCancerTotalSegROI
