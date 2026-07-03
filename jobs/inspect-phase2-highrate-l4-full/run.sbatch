#!/usr/bin/env bash
#SBATCH --job-name=inspect-phase2-l4
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

SCRIPT="/data/users/chjing/vLLM_test/phase2_highrate_l4.sh"

section() {
  echo
  echo "===== $1 ====="
}

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "script=${SCRIPT}"

section "script_stat"
ls -l "$SCRIPT" || true
stat "$SCRIPT" || true

section "function_and_section_index"
grep -nE '^(# =+|# [0-9]+\.|[[:alpha:]_][[:alnum:]_]*\(\) \{)' "$SCRIPT" || true

section "frequency_related_lines"
grep -nE 'nvidia-smi|SUPPORTED_MEM_FREQ|ALL_GPU_FREQS|GPU_TYPE|STEP|RUNS|TP|HR_|set_gpu_freq|reset|cleanup|monitor|vllm|benchmark|MODEL|OUTPUT_DIR|MASTER_CSV' "$SCRIPT" || true

section "full_script_numbered"
nl -ba "$SCRIPT" || true
