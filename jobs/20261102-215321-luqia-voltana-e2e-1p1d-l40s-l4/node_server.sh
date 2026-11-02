#!/usr/bin/env bash
# One SGLang P/D server of the VoltanaLLM end-to-end job, run as an srun step on its node:
#   node_server.sh prefill|decode
# Resolves the allocated GPU (NVML index), refuses a GPU that already runs compute processes,
# starts the NVML sampler, then execs SGLang with NIXL, EcoFreq (starts in dummy mode: the driver
# switches it per configuration) and the server settings the EcoPred tables were profiled with.
# Everything else comes from run.sbatch's environment (srun exports it).
set -uo pipefail
ROLE="$1"
fail() { echo "NODE_SERVER_FAILED role=${ROLE} stage=$1"; exit 1; }

GPU_INDEX="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
if [ "$(nvidia-smi -L | wc -l)" -eq 1 ]; then GPU_INDEX=0; fi   # constrained view: only ours is visible
[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || fail "unexpected_gpu_allocation=${GPU_INDEX}"
GPU_NAME=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name --format=csv,noheader)
GPU_UUID=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
case "${ROLE}:${GPU_NAME}" in
  prefill:*L40S*) ;;
  decode:*L40S*) fail "wrong_gpu=${GPU_NAME}" ;;
  decode:*L4*) ;;
  *) fail "wrong_gpu=${GPU_NAME}" ;;
esac
{ echo "host=$(hostname) gpu_index=${GPU_INDEX} uuid=${GPU_UUID} name=${GPU_NAME}"
  echo "slurm_step_gpus=${SLURM_STEP_GPUS:-} slurm_job_gpus=${SLURM_JOB_GPUS:-} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi -i "${GPU_INDEX}" --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv
} > "${WORK}/${ROLE}_gpu.txt"
echo "${GPU_INDEX}" > "${WORK}/${ROLE}_gpu_index"

# exclusivity: nobody else may compute on this GPU (no --exclusive; energy is per GPU)
OTHERS=$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid,process_name --format=csv,noheader)
echo "compute_apps_before_start=${OTHERS:-none}" >> "${WORK}/${ROLE}_gpu.txt"
[ -z "${OTHERS}" ] || fail "gpu_busy (${OTHERS})"

# clock locking must work (EcoFreq's setter uses the same sudo -n nvidia-smi)
LOCK=$(sudo -n nvidia-smi -i "${GPU_INDEX}" -lgc 1005,1005 2>&1)
echo "${LOCK}" | grep -q "All done" || fail "clock_lock_not_permitted (${LOCK})"
sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc > /dev/null 2>&1

"${PY}" "${JOB_DIR}/gpu_sampler.py" --index "${GPU_INDEX}" --out "${WORK}/${ROLE}_nvml.csv" --interval 0.2 &

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}" VOLTANA_NVML_INDEX="${GPU_INDEX}" VOLTANA_FREQ_SETTER=nvidia-smi
export TMPDIR="${WORK}/tmp-${ROLE}" SGLANG_ENABLE_TORCH_INFERENCE_MODE=1
mkdir -p "${TMPDIR}"
unset UCX_TLS UCX_NET_DEVICES   # the profiling scripts pin UCX to lo; across nodes UCX picks TCP itself

if [ "${ROLE}" = prefill ]; then
  # as 20261026-141000-luqia-voltana-prof-prefill-l40s (p_server.sh, CONTEXT_LEN 16384)
  PORT="${P_PORT}" TABLE="${P_TABLE}" SLO="${P_SLO_MS}" FLIST="${P_FLIST}" MAX_BS=1024
  role_args=(--mem-fraction-static 0.85 --max-running-requests 1024 --max-prefill-tokens 16384
    --context-length 16384 --chunked-prefill-size 16384 --attention-backend flashinfer
    --disable-overlap-schedule --force-cuda-graph-prefill
    --disaggregation-bootstrap-port "${BOOT_PORT}")
else
  # as 20261026-141030-luqia-voltana-prof-decode-l4 (d_server.sh, CONTEXT_LEN 8192)
  PORT="${D_PORT}" TABLE="${D_TABLE}" SLO="${D_SLO_MS}" FLIST="${D_FLIST}" MAX_BS=512
  role_args=(--mem-fraction-static 0.88 --max-running-requests 512 --max-prefill-tokens 8192
    --context-length 8192 --chunked-prefill-size 8192 --attention-backend triton
    --num-reserved-decode-tokens 8)
fi
cuda_graph_bs=()
for bs in 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024; do
  (( bs <= MAX_BS )) && cuda_graph_bs+=("${bs}")
done
read -r -a flist <<< "${FLIST}"

args=(
  --model-path "${MODEL_PATH}" --trust-remote-code --random-seed 0 --watchdog-timeout 86400
  --host 0.0.0.0 --port "${PORT}" --device cuda --base-gpu-id 0
  --schedule-policy fcfs --page-size 1 --sampling-backend flashinfer --grammar-backend xgrammar
  --cuda-graph-bs "${cuda_graph_bs[@]}" --disable-radix-cache
  "${role_args[@]}"
  --disaggregation-mode "${ROLE}" --disaggregation-transfer-backend nixl
  --enable-energy-monitor --energy-monitor-interval 0.01
  --enable-freq-manager --freq-manager-dummy-run --freq-manager-interval 0
  --freq-manager-latency-lookup-table-path "${TABLE}"
  --freq-manager-slo "${SLO}" --freq-manager-slo-margin 0
  --freq-manager-f-list "${flist[@]}"
  # TPS / RPS overrides off, as in the demo scripts
  --freq-manager-tps-threshold 100000000 --freq-manager-tps-window 5
  --freq-manager-rps-threshold 1000000 --freq-manager-rps-window 5
  --monitor-iteration-metrics
)
echo "launch ${ROLE}: ${PY} -m sglang.launch_server ${args[*]}"
exec "${PY}" -m sglang.launch_server "${args[@]}"
