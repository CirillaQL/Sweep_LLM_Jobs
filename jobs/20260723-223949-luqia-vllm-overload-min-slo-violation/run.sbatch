#!/usr/bin/env bash
#SBATCH --job-name=vllm-overload-min-slo
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-223949-luqia-vllm-overload-min-slo-violation/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260723-223949-luqia-vllm-overload-min-slo-violation/slurm-%j.err

set -uo pipefail

JOB_NAME="20260723-223949-luqia-vllm-overload-min-slo-violation"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
SOURCE_JOB="${BROKER_ROOT}/jobs/20260722-161842-luqia-vllm-scheduler-saturation-gate"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
OUT_DIR="${JOB_DIR}/overload_fallback_results"
WORKLOAD_FILE="${JOB_DIR}/workloads.csv"
PYTHON_BIN="/data/users/chjing/miniforge3/envs/cuda-env/bin/python"
SCHEDULER="${SOURCE_JOB}/scheduler.py"
MODEL_BUNDLE="${SOURCE_JOB}/model_bundle.json"
SATURATION_BUNDLE="${SOURCE_JOB}/saturation_bundle.json"

mkdir -p "$OUT_DIR"
echo "placement,workload_id,status,num_candidates,num_safe,prefill_gpu,prefill_freq_mhz,decode_gpu,decode_freq_mhz,predicted_overload_violation_probability,predicted_joint_latency_violation_probability,predicted_joint_saturation_probability,predicted_max_slo_excess_ratio,is_safe" > "${OUT_DIR}/summary.csv"

FINAL_RC=0
COUNT=0
for placement in l40s-prefill-l4-decode auto; do
  while IFS=, read -r workload_id il ol rate description; do
    if [ "$workload_id" = workload_id ] || [ -z "$workload_id" ]; then
      continue
    fi
    COUNT=$((COUNT + 1))
    decision="${OUT_DIR}/${placement}_${workload_id}.json"
    "$PYTHON_BIN" "$SCHEDULER" \
      --bundle "$MODEL_BUNDLE" \
      --saturation-bundle "$SATURATION_BUNDLE" \
      --policy latency_plus_saturation \
      --placement "$placement" \
      --overload-action min-slo-violation \
      --il "$il" --ol "$ol" --rate "$rate" \
      --slo-ttft 500 --slo-tpot 200 --tp 1 \
      --output "$decision" > "${OUT_DIR}/${placement}_${workload_id}.stdout.json"
    scheduler_rc=$?
    echo "decision placement=${placement} workload=${workload_id} scheduler_rc=${scheduler_rc}"
    if [ "$scheduler_rc" -ne 0 ]; then
      FINAL_RC=20
      continue
    fi

    "$PYTHON_BIN" - "$decision" "$placement" "$workload_id" >> "${OUT_DIR}/summary.csv" <<'PY'
import json
import sys

path, placement, workload_id = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
if result.get("status") != "OVERLOAD_FALLBACK":
    raise SystemExit(f"expected OVERLOAD_FALLBACK, got {result.get('status')}")
if result.get("num_safe") != 0:
    raise SystemExit(f"expected num_safe=0, got {result.get('num_safe')}")
recommended = result["recommended"]
if recommended.get("is_safe") is not False:
    raise SystemExit("overload recommendation must retain is_safe=false")
selected = recommended["predicted_overload_violation_probability"]
if any(
    selected > alternative["predicted_overload_violation_probability"]
    for alternative in result.get("alternatives", [])
):
    raise SystemExit("recommended candidate does not minimize overload violation")
prefill = recommended["prefill"]
decode = recommended["decode"]
print(
    ",".join(map(str, [
        placement,
        workload_id,
        result["status"],
        result["num_candidates"],
        result["num_safe"],
        prefill["gpu_type"],
        prefill["freq_mhz"],
        decode["gpu_type"],
        decode["freq_mhz"],
        recommended["predicted_overload_violation_probability"],
        recommended["predicted_joint_latency_violation_probability"],
        recommended["predicted_joint_saturation_probability"],
        recommended["predicted_max_slo_excess_ratio"],
        str(recommended["is_safe"]).lower(),
    ]))
)
PY
    validation_rc=$?
    echo "validation placement=${placement} workload=${workload_id} rc=${validation_rc}"
    if [ "$validation_rc" -ne 0 ]; then
      FINAL_RC=21
    fi
  done < "$WORKLOAD_FILE"
done

echo "decision_count=${COUNT}"
echo "gpu_tres_requested=false"
echo "gpu_commands_executed=false"
echo "vllm_started=false"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
