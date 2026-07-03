#!/usr/bin/env bash
#SBATCH --job-name=luqia-perms
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

CONTAINER="/data/users/luqia/vllm_quick.sif"
MODEL_DIR="/data/users/luqia/model/Mistral-7B-v0.1"

run() {
  local title="$1"
  shift
  echo
  echo "===== ${title} ====="
  echo "command=$*"
  "$@"
  local rc=$?
  echo "===== ${title} rc=${rc} ====="
}

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
  if [ -d "$path" ]; then
    echo "type=directory"
  elif [ -f "$path" ]; then
    echo "type=file"
  else
    echo "type=other_or_missing"
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
echo "CONTAINER=${CONTAINER}"
echo "MODEL_DIR=${MODEL_DIR}"

run "slurm_job_record" scontrol show job "${SLURM_JOB_ID:-0}"

echo
echo "===== parent_path_permissions ====="
for path in /data /data/users /data/users/luqia /data/users/luqia/model "$MODEL_DIR"; do
  check_path "$path"
done

check_path "$CONTAINER"

run_shell "container_read_first_4_bytes" "head -c 4 '$CONTAINER' | od -An -tx1"
run_shell "container_file_size" "du -h '$CONTAINER'"

echo
echo "===== model_directory_listing ====="
run_shell "model_top_level_listing" "find '$MODEL_DIR' -maxdepth 1 -mindepth 1 -printf '%M %u %g %s %p\n' | sort | head -n 80"
run_shell "model_file_count" "find '$MODEL_DIR' -type f | wc -l"
run_shell "model_total_size" "du -sh '$MODEL_DIR'"

echo
echo "===== common_model_files ====="
for file in \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/tokenizer.json" \
  "$MODEL_DIR/tokenizer.model" \
  "$MODEL_DIR/tokenizer_config.json" \
  "$MODEL_DIR/generation_config.json"
do
  check_path "$file"
  run_shell "read_test ${file}" "test -r '$file' && head -c 64 '$file' >/dev/null"
done

echo
echo "===== model_weight_read_tests ====="
run_shell "list_weight_candidates" "find '$MODEL_DIR' -maxdepth 1 -type f \\( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \\) -printf '%M %u %g %s %p\n' | sort | head -n 20"
run_shell "read_first_weight_byte" "first=\$(find '$MODEL_DIR' -maxdepth 1 -type f \\( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \\) | head -n 1); if [ -n \"\$first\" ]; then echo \"first_weight=\$first\"; head -c 1 \"\$first\" >/dev/null; else echo 'no_weight_file_found'; fi"

echo
echo "===== container_runtime_tests ====="
if command -v apptainer >/dev/null 2>&1; then
  echo "runtime=apptainer"
  run "apptainer_version" apptainer --version
  run "apptainer_sif_list" apptainer sif list "$CONTAINER"
  run "apptainer_inspect" apptainer inspect "$CONTAINER"
  run "apptainer_exec_true" apptainer exec "$CONTAINER" true
  run "apptainer_exec_python_version" apptainer exec "$CONTAINER" python --version
elif command -v singularity >/dev/null 2>&1; then
  echo "runtime=singularity"
  run "singularity_version" singularity --version
  run "singularity_sif_list" singularity sif list "$CONTAINER"
  run "singularity_inspect" singularity inspect "$CONTAINER"
  run "singularity_exec_true" singularity exec "$CONTAINER" true
  run "singularity_exec_python_version" singularity exec "$CONTAINER" python --version
else
  echo "container_runtime=not_found"
fi

echo
echo "===== summary_hints ====="
echo "If all path checks are readable/searchable and container exec rc=0, chjing jobs can use these luqia assets."
echo "If any parent directory is not searchable, chjing jobs cannot reach files below it."
