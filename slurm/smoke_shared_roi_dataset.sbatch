#!/bin/bash
#SBATCH --job-name=pan_roi_smoke
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=work/slurm/%x-%j.out
#SBATCH --error=work/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}
PYTHON=${DATA_PYTHON:-python}
SMOKE_MANIFEST=${SMOKE_MANIFEST:-work/experiments/liver_kidney_smoke_manifest.jsonl}
cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/materialize_totalseg_roi_dataset.py \
  --manifest "${SMOKE_MANIFEST}" \
  --output-root work/experiments/shared_roi_smoke \
  --output-manifest work/experiments/shared_roi_smoke_manifest.jsonl \
  --margin-mm 30 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --max-cases 16 \
  --qc-samples-per-stratum 2
