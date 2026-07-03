#!/usr/bin/env bash
#SBATCH --job-name=vllm-gpufreq-inspect
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

ROOT="/data/users/chjing/vLLM_test"
TERMS='nvidia-smi|applications-clocks|lock-gpu-clocks|reset-gpu-clocks|gpu[-_ ]?freq|FREQ|frequency|clock|nvml|NVML|set.*clock|reset.*clock'

section() {
  echo
  echo "===== $1 ====="
}

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

section "root"
stat "$ROOT" || true
ls -ld "$ROOT" || true

section "top_level_scripts"
find "$ROOT" -maxdepth 2 -type f \( -name '*.sh' -o -name '*.py' \) \
  -printf '%M %u %g %s %TY-%Tm-%Td %TH:%TM %p\n' | sort

section "frequency_keyword_hits_scripts_only"
grep -RInE "$TERMS" "$ROOT" \
  --include='*.sh' --include='*.py' \
  --exclude-dir='results' --exclude-dir='Phase2_Results_L4' --exclude-dir='Phase2_Results_L40S' \
  --exclude-dir='synthetic_traces' --exclude-dir='__pycache__' 2>/dev/null || true

section "frequency_keyword_context"
for file in $(grep -RIlE "$TERMS" "$ROOT" \
  --include='*.sh' --include='*.py' \
  --exclude-dir='results' --exclude-dir='Phase2_Results_L4' --exclude-dir='Phase2_Results_L40S' \
  --exclude-dir='synthetic_traces' --exclude-dir='__pycache__' 2>/dev/null | sort); do
  echo
  echo "----- ${file} -----"
  grep -nE -C 4 "$TERMS" "$file" | head -n 220
done

section "selected_script_heads"
for file in \
  "$ROOT/submit_experiment_A_highfreq_probe.sh" \
  "$ROOT/submit_experiment_CD_cross_gpu_phase_only.sh" \
  "$ROOT/gpu_monitor.py" \
  "$ROOT/vllm_calibration.sh" \
  "$ROOT/run_disagg_benchmark.sh" \
  "$ROOT/validate_obs2_l4.sh"; do
  if [ -f "$file" ]; then
    echo
    echo "----- head ${file} -----"
    sed -n '1,220p' "$file"
  fi
done
