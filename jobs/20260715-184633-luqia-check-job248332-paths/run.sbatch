#!/usr/bin/env bash
#SBATCH --job-name=check-248332-paths
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

TARGET_JOB_ID=248332
TARGET_NAME="20260715-181257-luqia-vllm-f990-ganymede"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
MATCH_FILE="/tmp/job248332-matches-${SLURM_JOB_ID:-$$}.txt"
trap 'rm -f "$MATCH_FILE"' EXIT

section() {
  echo
  echo "===== $1 ====="
}

section "worker_submit_context"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "whoami=$(whoami)"
echo "pwd=$PWD"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "resolved_submit_dir=${SUBMIT_DIR}"
echo "inspector_job_id=${SLURM_JOB_ID:-unset}"

section "chjing_top_level"
ls -lah /data/users/chjing 2>&1 || true

ROOTS=(
  "$SUBMIT_DIR"
  "/data/users/chjing/Sweep_LLM_Jobs_broker"
  "/data/users/chjing/Sweep_LLM_Jobs_broker/jobs"
  "/data/users/chjing/Sweep_LLM_Jobs_broker/.broker"
  "/data/users/chjing/.broker"
  "/data/users/chjing/gsync"
)

: > "$MATCH_FILE"
section "candidate_roots"
for root in "${ROOTS[@]}"; do
  echo "--- ${root} ---"
  if [ ! -d "$root" ]; then
    echo "directory_missing=true"
    continue
  fi
  ls -ld "$root" 2>&1 || true
  find "$root" -maxdepth 2 \
    \( \
      -type f -name "slurm-${TARGET_JOB_ID}.*" -o \
      -type f -name 'vllm_server.log' -o \
      -type f -name 'nvidia_smi_monitor.csv' -o \
      -type d -name 'vllm_f990_results' \
    \) -print 2>/dev/null >> "$MATCH_FILE" || true
done

sort -u "$MATCH_FILE" -o "$MATCH_FILE"
section "matches"
echo "match_count=$(wc -l < "$MATCH_FILE")"
cat "$MATCH_FILE"

section "matched_contents"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  echo
  echo "----- ${path} -----"
  ls -lah "$path" 2>&1 || true
  if [ -f "$path" ]; then
    case "$path" in
      *.csv)
        head -n 8 "$path" 2>&1 || true
        echo "--- tail ---"
        tail -n 30 "$path" 2>&1 || true
        ;;
      *)
        tail -n 500 "$path" 2>&1 || true
        ;;
    esac
  fi
done < "$MATCH_FILE"

section "exact_target_directory"
TARGET_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${TARGET_NAME}"
ls -lah "$TARGET_DIR" 2>&1 || true
ls -lah "${TARGET_DIR}/vllm_f990_results" 2>&1 || true

