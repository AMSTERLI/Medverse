#!/bin/bash
#SBATCH --job-name=lk_totalseg
#SBATCH --partition=bme_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${TOTALSEG_PYTHON:-python}
TOTALSEG_CLI=${TOTALSEG_CLI:-TotalSegmentator}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${PROJECT_DIR}/work/manifests/paot2_manifest.jsonl}
DERIVED_ROOT=${DERIVED_ROOT:-${PROJECT_DIR}/work/data/derived_liver_kidney_v1}
cd "${PROJECT_DIR}"
mkdir -p work/slurm

ARGS=(
  --manifest "${SOURCE_MANIFEST}"
  --output-root "${DERIVED_ROOT}"
  --cli "${TOTALSEG_CLI}"
  --device gpu
)
if [[ -n "${MAX_CASES:-}" ]]; then ARGS+=(--max-cases "${MAX_CASES}"); fi
if [[ -n "${MAX_CASES_PER_STRATUM:-}" ]]; then
  ARGS+=(--max-cases-per-stratum "${MAX_CASES_PER_STRATUM}")
fi
if [[ -n "${SHARD_COUNT:-}" ]]; then ARGS+=(--shard-count "${SHARD_COUNT}"); fi
if [[ -n "${SHARD_INDEX:-}" ]]; then ARGS+=(--shard-index "${SHARD_INDEX}"); fi

echo "job_id=${SLURM_JOB_ID} host=$(hostname) visible_gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"${PYTHON}" scripts/run_totalsegmentator_organs.py "${ARGS[@]}"
