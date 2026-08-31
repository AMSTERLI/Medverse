#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${MEDVERSE_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
RUN_TARGET=${1:-both}
NO_ICL_MODE=${NO_ICL_MODE:-start}
ICL_MODE=${ICL_MODE:-start}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GATE_DIR=${PROJECT_DIR}/work/experiments/gates

case "${RUN_TARGET}" in
  both|no-icl|icl) ;;
  *) echo "usage: $0 [both|no-icl|icl]" >&2; exit 2 ;;
esac

if [[ "${ALLOW_UNPASSED_GATES:-0}" != "1" ]]; then
  if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "no-icl" ]]; then
    [[ -f "${GATE_DIR}/no_icl_overfit.pass" ]] || {
      echo "No-ICL overfit gate has not passed" >&2
      exit 5
    }
  fi
  if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "icl" ]]; then
    [[ -f "${GATE_DIR}/icl_bam_k3_overfit.pass" ]] || {
      echo "ICL+BAM K=3 overfit gate has not passed" >&2
      exit 5
    }
  fi
fi

mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
REQUIRED=2
[[ "${RUN_TARGET}" == "no-icl" ]] && REQUIRED=1
if (( ${#GPU_LINES[@]} < REQUIRED )); then
  echo "need ${REQUIRED} visible GPUs, found ${#GPU_LINES[@]}" >&2
  exit 3
fi
printf 'detected_gpu=%s\n' "${GPU_LINES[@]}"

NO_ICL_DIR=${PROJECT_DIR}/work/experiments/no_icl_nnunet
ICL_DIR=${PROJECT_DIR}/work/experiments/icl_medverse_bam_k3
mkdir -p "${NO_ICL_DIR}/logs" "${NO_ICL_DIR}/checkpoints" "${ICL_DIR}/logs" "${ICL_DIR}/checkpoints"
cp "${PROJECT_DIR}/configs/main_experiment/no_icl_nnunet.yaml" "${NO_ICL_DIR}/config.yaml"
cp "${PROJECT_DIR}/configs/main_experiment/icl_medverse_bam_k3.yaml" "${ICL_DIR}/config.yaml"
cp "${PROJECT_DIR}/configs/main_experiment/shared_data.yaml" "${NO_ICL_DIR}/shared_data.yaml"
cp "${PROJECT_DIR}/configs/main_experiment/shared_data.yaml" "${ICL_DIR}/shared_data.yaml"

declare -a PIDS=()
declare -a NAMES=()

start_arm() {
  local name=$1 gpu=$2 mode=$3 script=$4 directory=$5
  local pid_file=${directory}/${name}.pid
  if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
    echo "${name} already running with PID $(<"${pid_file}")" >&2
    return 4
  fi
  local log=${directory}/logs/${name}_${TIMESTAMP}.log
  CUDA_VISIBLE_DEVICES="${gpu}" bash "${PROJECT_DIR}/${script}" "${mode}" >"${log}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  echo "started name=${name} gpu=${gpu} mode=${mode} pid=${pid} log=${log}"
  PIDS+=("${pid}")
  NAMES+=("${name}")
}

if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "no-icl" ]]; then
  start_arm no_icl 0 "${NO_ICL_MODE}" scripts/train_no_icl_nnunet.sh "${NO_ICL_DIR}"
fi
if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "icl" ]]; then
  start_arm icl_bam_k3 1 "${ICL_MODE}" scripts/train_icl_medverse_bam_k3.sh "${ICL_DIR}"
fi

STATUS=0
set +e
for index in "${!PIDS[@]}"; do
  wait "${PIDS[index]}"
  code=$?
  echo "finished name=${NAMES[index]} exit_code=${code}"
  if (( code != 0 )); then STATUS=${code}; fi
done
set -e
exit "${STATUS}"
