#!/usr/bin/env bash
#SBATCH --job-name=chjing-inventory
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

BASE="/data/users/chjing"

run_shell() {
  local title="$1"
  local command="$2"
  echo
  echo "===== ${title} ====="
  echo "command=${command}"
  bash --noprofile --norc -lc "$command"
  local rc=$?
  echo "===== ${title} rc=${rc} ====="
}

check_path() {
  local path="$1"
  echo
  echo "===== check_path ${path} ====="
  if [ -e "$path" ]; then
    echo "exists=yes"
  else
    echo "exists=no"
  fi
  if [ -r "$path" ]; then
    echo "readable=yes"
  else
    echo "readable=no"
  fi
  if [ -x "$path" ]; then
    echo "executable/searchable=yes"
  else
    echo "executable/searchable=no"
  fi
  stat "$path" || true
  ls -ld "$path" || true
}

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "USER=${USER:-unset}"
echo "LOGNAME=${LOGNAME:-unset}"
echo "HOME=${HOME:-unset}"
echo "BASE=${BASE}"

echo
echo "===== slurm_job_record ====="
scontrol show job "${SLURM_JOB_ID:-0}" || true

check_path "$BASE"

run_shell "top_level_listing_non_recursive" \
  "find '$BASE' -maxdepth 1 -mindepth 1 -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM %p\n' | sort"

run_shell "top_level_directory_names" \
  "find '$BASE' -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort"

run_shell "candidate_shared_paths" \
  "for p in '$BASE/shared' '$BASE/Sweep_LLM_shared' '$BASE/model' '$BASE/models' '$BASE/containers' '$BASE/Sweep_LLM_Jobs_broker' '$BASE/Sweep_LLM_Jobs'; do if [ -e \"\$p\" ]; then ls -ld \"\$p\"; else echo \"missing \$p\"; fi; done"

run_shell "broker_clone_top_level" \
  "if [ -d '$BASE/Sweep_LLM_Jobs_broker' ]; then find '$BASE/Sweep_LLM_Jobs_broker' -maxdepth 1 -mindepth 1 -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM %p\n' | sort | head -n 120; else echo 'broker clone not found'; fi"

run_shell "visible_sif_files_top_two_levels" \
  "find '$BASE' -maxdepth 2 -path '$BASE/.ssh' -prune -o -path '$BASE/.gnupg' -prune -o -type f -name '*.sif' -printf '%M %u %g %s %p\n' 2>/dev/null | sort | head -n 80"

run_shell "visible_model_like_dirs_top_two_levels" \
  "find '$BASE' -maxdepth 2 -path '$BASE/.ssh' -prune -o -path '$BASE/.gnupg' -prune -o -type d \\( -iname '*model*' -o -iname '*mistral*' -o -iname '*llm*' -o -iname '*vllm*' \\) -printf '%M %u %g %p\n' 2>/dev/null | sort | head -n 80"

echo
echo "===== note ====="
echo "This job only lists paths and metadata. It does not read file contents."
