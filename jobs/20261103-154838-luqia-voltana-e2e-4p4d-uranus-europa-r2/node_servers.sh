#!/usr/bin/env bash
# The four SGLang P or D servers of one node in the VoltanaLLM 4P4D job, run as one srun step:
#   node_servers.sh prefill|decode
# Resolves the node's four GPUs (NVML indices), refuses a GPU that already runs compute processes
# or whose clocks cannot be locked, starts one NVML sampler for the node, then starts server i on
# GPU i with NIXL, EcoFreq (dummy mode at start: the driver switches it per configuration) and the
# server settings the EcoPred tables were profiled with. Waits for all four; TERM stops them.
# Everything else comes from run.sbatch's environment (srun exports it).
set -uo pipefail
ROLE="$1"
fail() { echo "NODE_SERVERS_FAILED role=${ROLE} stage=$1"; exit 1; }

GPU_LIST="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
if [ -z "${GPU_LIST}" ] && [ "$(nvidia-smi -L | wc -l)" -eq 4 ]; then GPU_LIST=0,1,2,3; fi
[[ "${GPU_LIST}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail "unexpected_gpu_allocation=${GPU_LIST}"
IFS=, read -r -a GPUS <<< "${GPU_LIST}"
: > "${WORK}/${ROLE}_gpus.txt"
for i in 0 1 2 3; do
  idx="${GPUS[$i]}"
  name=$(nvidia-smi -i "${idx}" --query-gpu=name --format=csv,noheader)
  case "${ROLE}:${name}" in
    prefill:*L40S*) ;;
    decode:*L40S*) fail "wrong_gpu=${idx}:${name}" ;;
    decode:*L4*) ;;
    *) fail "wrong_gpu=${idx}:${name}" ;;
  esac
  others=$(nvidia-smi -i "${idx}" --query-compute-apps=pid,process_name --format=csv,noheader)
  { echo "slot=${i} host=$(hostname) gpu_index=${idx} name=${name}" \
      "uuid=$(nvidia-smi -i "${idx}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')" \
      "compute_apps_before_start=${others:-none}"
    nvidia-smi -i "${idx}" --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader
  } >> "${WORK}/${ROLE}_gpus.txt"
  [ -z "${others}" ] || fail "gpu_busy ${idx} (${others})"
  # clock locking must work (EcoFreq's setter uses the same sudo -n nvidia-smi)
  lock=$(sudo -n nvidia-smi -i "${idx}" -lgc 1005,1005 2>&1)
  echo "${lock}" | grep -q "All done" || fail "clock_lock_not_permitted ${idx} (${lock})"
  sudo -n nvidia-smi -i "${idx}" -rgc > /dev/null 2>&1
done
echo "${GPU_LIST}" > "${WORK}/${ROLE}_gpu_indices"

PIDS=()
stop() { kill "${PIDS[@]}" 2>/dev/null; sleep 10; kill -9 "${PIDS[@]}" 2>/dev/null; exit 143; }
trap stop TERM INT

"${PY}" "${JOB_DIR}/gpu_sampler.py" --indices "${GPU_LIST}" --out "${WORK}/${ROLE}_nvml.csv" --interval 0.2 &
PIDS+=($!)

export SGLANG_ENABLE_TORCH_INFERENCE_MODE=1
unset UCX_TLS UCX_NET_DEVICES   # the profiling scripts pin UCX to lo; across nodes UCX picks TCP itself
if [ "${ROLE}" = prefill ]; then
  # as 20261026-141000-luqia-voltana-prof-prefill-l40s (p_server.sh, CONTEXT_LEN 16384)
  TABLE="${P_TABLE}" SLO="${P_SLO_MS}" FLIST="${P_FLIST}" MAX_BS=1024
  role_args=(--mem-fraction-static 0.85 --max-running-requests 1024 --max-prefill-tokens 16384
    --context-length 16384 --chunked-prefill-size 16384 --attention-backend flashinfer
    --disable-overlap-schedule --force-cuda-graph-prefill)
else
  # as 20261026-141030-luqia-voltana-prof-decode-l4 (d_server.sh, CONTEXT_LEN 8192)
  TABLE="${D_TABLE}" SLO="${D_SLO_MS}" FLIST="${D_FLIST}" MAX_BS=512
  role_args=(--mem-fraction-static 0.88 --max-running-requests 512 --max-prefill-tokens 8192
    --context-length 8192 --chunked-prefill-size 8192 --attention-backend triton
    --num-reserved-decode-tokens 8)
fi
cuda_graph_bs=()
for bs in 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024; do
  (( bs <= MAX_BS )) && cuda_graph_bs+=("${bs}")
done
read -r -a flist <<< "${FLIST}"
read -r -a p_ports <<< "${P_PORTS}"; read -r -a d_ports <<< "${D_PORTS}"; read -r -a boot_ports <<< "${BOOT_PORTS}"

for i in 0 1 2 3; do
  idx="${GPUS[$i]}"
  if [ "${ROLE}" = prefill ]; then
    port="${p_ports[$i]}"; extra=(--disaggregation-bootstrap-port "${boot_ports[$i]}")
  else
    port="${d_ports[$i]}"; extra=()
  fi
  args=(
    --model-path "${MODEL_PATH}" --trust-remote-code --random-seed 0 --watchdog-timeout 86400
    --host 0.0.0.0 --port "${port}" --device cuda --base-gpu-id 0
    --schedule-policy fcfs --page-size 1 --sampling-backend flashinfer --grammar-backend xgrammar
    --cuda-graph-bs "${cuda_graph_bs[@]}" --disable-radix-cache
    "${role_args[@]}" ${extra[@]+"${extra[@]}"}
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
  # one GPU per server; NVML (EcoFreq, energy monitor) uses the physical index; own TMPDIR for
  # the setter's IPC sockets
  mkdir -p "${WORK}/tmp-${ROLE}${i}"
  echo "launch ${ROLE}${i} gpu=${idx}: ${PY} -m sglang.launch_server ${args[*]}" > "${WORK}/logs/${ROLE}${i}.log"
  CUDA_VISIBLE_DEVICES="${idx}" VOLTANA_NVML_INDEX="${idx}" VOLTANA_FREQ_SETTER=nvidia-smi \
    TMPDIR="${WORK}/tmp-${ROLE}${i}" \
    "${PY}" -m sglang.launch_server "${args[@]}" >> "${WORK}/logs/${ROLE}${i}.log" 2>&1 &
  PIDS+=($!)
  echo "started ${ROLE}${i} pid=$! gpu=${idx} port=${port}"
done

# a server that exits ends the step, so the driver sees it
while true; do
  for i in 0 1 2 3; do
    kill -0 "${PIDS[$(( i + 1 ))]}" 2>/dev/null || { echo "NODE_SERVERS_FAILED role=${ROLE} stage=${ROLE}${i}_exited"; stop; }
  done
  sleep 5 & wait $!   # wait: TERM reaches the trap at once
done
