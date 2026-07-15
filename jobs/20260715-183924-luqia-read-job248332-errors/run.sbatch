#!/usr/bin/env bash
#SBATCH --job-name=read-248332-errors
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

TARGET_JOB_ID=248332
TARGET_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260715-181257-luqia-vllm-f990-ganymede"
RESULT_DIR="${TARGET_DIR}/vllm_f990_results"

section() {
  echo
  echo "===== $1 ====="
}

show_file() {
  local path="$1"
  local lines="${2:-300}"
  section "file: ${path}"
  if [ ! -e "$path" ]; then
    echo "exists=false"
    return
  fi
  echo "exists=true"
  ls -lh "$path" || true
  echo "--- last ${lines} lines ---"
  tail -n "$lines" "$path" 2>&1 || true
}

show_csv_edges() {
  local path="$1"
  section "csv edges: ${path}"
  if [ ! -e "$path" ]; then
    echo "exists=false"
    return
  fi
  echo "exists=true"
  ls -lh "$path" || true
  echo "--- first 8 lines ---"
  head -n 8 "$path" 2>&1 || true
  echo "--- last 20 lines ---"
  tail -n 20 "$path" 2>&1 || true
}

section "inspector_identity"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "inspector_slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "target_slurm_job_id=${TARGET_JOB_ID}"
echo "target_dir=${TARGET_DIR}"

section "target_accounting"
sacct -X -j "$TARGET_JOB_ID" \
  -o JobID,JobName,User,Partition,State,ExitCode,Elapsed,Start,End,NodeList%20,Reason%40 \
  2>&1 || true

section "target_directory_listing"
if [ -d "$TARGET_DIR" ]; then
  find "$TARGET_DIR" -maxdepth 2 -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %M %u:%g %s %p\n' 2>&1 | sort || true
else
  echo "target_directory_exists=false"
  ls -ld "$TARGET_DIR" 2>&1 || true
fi

show_file "${TARGET_DIR}/status.json" 200
show_file "${TARGET_DIR}/status.log" 200
show_file "${TARGET_DIR}/slurm-${TARGET_JOB_ID}.out" 500
show_file "${TARGET_DIR}/slurm-${TARGET_JOB_ID}.err" 500
show_file "${RESULT_DIR}/vllm_server.log" 300
show_file "${RESULT_DIR}/bench_iter1.txt" 200
show_csv_edges "${RESULT_DIR}/nvidia_smi_monitor.csv"
show_csv_edges "${RESULT_DIR}/jing_gpu_monitor.csv"

section "error_search"
for path in \
  "${TARGET_DIR}/slurm-${TARGET_JOB_ID}.out" \
  "${TARGET_DIR}/slurm-${TARGET_JOB_ID}.err" \
  "${RESULT_DIR}/vllm_server.log"; do
  if [ -r "$path" ]; then
    echo "--- ${path} ---"
    grep -Ein 'error|exception|traceback|failed|failure|fatal|killed|denied|cuda|nccl|out of memory|oom|server_died|reset|clock' "$path" \
      | tail -n 200 || true
  fi
done

