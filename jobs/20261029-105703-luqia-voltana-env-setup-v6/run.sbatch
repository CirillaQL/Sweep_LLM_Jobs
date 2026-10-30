#!/usr/bin/env bash
#SBATCH --job-name=voltana-env
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=ganymede
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=48G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261029-105703-luqia-voltana-env-setup-v6/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261029-105703-luqia-voltana-env-setup-v6/slurm-%j.err

# One-time setup for the VoltanaLLM v0.1 (SGLang 0.4.7.post1) EcoPred latency profiling jobs
# (20261026-1410*-luqia-voltana-prof-*). Builds a separate conda env (the shared cuda-env is
# vLLM 0.15.1 and stays untouched), clones upstream VoltanaLLM at 27417323 (v0.1 release) and applies
# voltana-llm-scripts.patch (env-configurable profiling scripts + fit_latency.py; NIXL
# backend fixes: drop the never-shipped FakeNixl* imports that break `import ...nixl`, and
# the is_sorted argument NIXL 0.9.0 removed), installs NIXL 0.9.0 (bundled UCX), links
# Mistral-7B-v0.1 and ShareGPT from their existing locations, then smoke-tests on one L4:
# clock-lock permission, a 60 s prefill profile at the max clock, a short decode profile
# (bs 8/9) and the headless fit. The smoke run also warms the flashinfer/triton JIT caches
# (L4 and L40S are both sm_89) that the profiling jobs reuse.
# v5: the patch also fixes process_batch_result_decode passing extra_batch_info positionally into
# stream_output's skip_req slot, so the decode server never streamed extra_batch_info (279503's
# decode smoke wrote empty per-request lists and the fit got 0 points; diagnosed by 279524). The
# smoke now fails unless the fit has rows.
# v6: v5 (279535) got full decode data, but fit_latency.py raised RANSAC "could not find a valid
# consensus set" on the bs 1-4 bucket, which held only 3 warm-up iterations (diagnosed by 279553).
# The fit now skips buckets with < --min-points (20) points or no RANSAC consensus, and cleanup
# also copies fit-*.log so a failed fit is visible in results.
# Re-running is safe: an existing env is reused (set FORCE_REBUILD=1 to rebuild) and the
# checkout is reset to 27417323 before the patch is applied again.
# No path on any python command line contains "vllm": the shared-account P/D launcher pkills
# 'python.*vllm|vllm\.entrypoints' node-wide, so everything lives under voltana_work/.
set -uo pipefail
START_EPOCH=$(date +%s)
JOB_NAME="20261029-105703-luqia-voltana-env-setup-v6"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="${JOB_DIR}/results/${JOB_ID}"
VOLTANA_ROOT="/data/users/chjing/voltana_work"
WORK="${VOLTANA_ROOT}/runs/${JOB_ID}"
REPO_URL="https://github.com/Supercomputing-System-AI-Lab/VoltanaLLM.git"
# 27417323 = "Release VoltanaLLM v0.1" (origin/voltanallm, SGLang 0.4.7.post1). Not main's merge
# f7789acf: that merge into SGLang ~0.4.9 is broken (job 279404: schedule_batch imports
# TokenToKVPoolAllocator from memory_pool, io_struct imports the removed openai_api.protocol, ...).
REPO_COMMIT="274173238ba0021ab40d98e5abfec3f47d24bed5"
REPO_DIR="${VOLTANA_ROOT}/VoltanaLLM"
PRED_DIR="${REPO_DIR}/voltana-llm-scripts/predictions"
MINIFORGE="/data/users/chjing/miniforge3"
ENV_DIR="${MINIFORGE}/envs/voltanallm"
PY="${ENV_DIR}/bin/python"
MODEL_SNAPSHOT_ROOT="/data/users/chjing/vllm_job_work/255143/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots"
MODEL_LINK="${VOLTANA_ROOT}/models/Mistral-7B-v0.1"
DATASET="/data/users/chjing/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
PORT=$(( 30000 + ${JOB_ID//[^0-9]/0} % 10000 ))

mkdir -p "${OUT}" "${WORK}" "${VOLTANA_ROOT}/models" "${VOLTANA_ROOT}/cache"
exec > >(tee -a "${OUT}/job_console.log") 2>&1
echo "job_start=$(date -Iseconds) host=$(hostname) job=${JOB_ID}"
fail() { echo "RESULT=failed stage=$1"; exit 1; }

# ---- caches and HOME inside WORK; JIT caches persistent and shared with the profiling jobs ---
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" TMP="${WORK}/tmp" TEMP="${WORK}/tmp"
export XDG_CACHE_HOME="${WORK}/xdg-cache" XDG_CONFIG_HOME="${WORK}/xdg-config"
export PIP_CACHE_DIR="${WORK}/pip-cache" CONDA_PKGS_DIRS="${WORK}/conda-pkgs"
export HF_HOME="${WORK}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${WORK}/torch-cache" TORCH_EXTENSIONS_DIR="${WORK}/torch-extensions"
export CUDA_CACHE_PATH="${WORK}/cuda-cache" MPLCONFIGDIR="${WORK}/matplotlib"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache"
export FLASHINFER_WORKSPACE_BASE="${VOLTANA_ROOT}/cache/flashinfer"
export TRITON_CACHE_DIR="${VOLTANA_ROOT}/cache/triton"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" "${HF_HOME}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" \
  "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}" "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}"

# ---- 1. conda env (python 3.12 + CUDA 12.8 toolkit for flashinfer JIT, as in VoltanaLLM README)
CONDA="${MINIFORGE}/bin/conda"
[ -x "${CONDA}" ] || fail "conda_not_found ${CONDA}"
# v4 (279503) rebuilt the env from requirements-lock.txt; v5/v6 reuse it (FORCE_REBUILD=1 rebuilds).
FORCE_REBUILD=${FORCE_REBUILD-}
if [ -x "${PY}" ] && [ -z "${FORCE_REBUILD:-}" ]; then
  echo "env exists, reusing ${ENV_DIR}"
else
  [ -z "${FORCE_REBUILD:-}" ] || rm -rf "${ENV_DIR}"
  timeout 2400 "${CONDA}" create -y -p "${ENV_DIR}" python=3.12 "nvidia/label/cuda-12.8.1::cuda-toolkit" \
    2>&1 | tail -n 40 || fail conda_create
fi
[ -x "${PY}" ] || fail env_python_missing
export PATH="${ENV_DIR}/bin:${PATH}" CUDA_HOME="${ENV_DIR}"
"${PY}" --version; "${ENV_DIR}/bin/nvcc" --version | tail -n 2 || fail nvcc_missing

# ---- 2. VoltanaLLM checkout at the pinned commit + our profiling-script patch ----------------
if [ ! -d "${REPO_DIR}/.git" ]; then
  timeout 600 git clone --quiet "${REPO_URL}" "${REPO_DIR}" || fail clone
fi
timeout 300 git -C "${REPO_DIR}" fetch --quiet origin || fail fetch
git -C "${REPO_DIR}" checkout --quiet --force "${REPO_COMMIT}" || fail checkout
git -C "${REPO_DIR}" clean -fdq -- voltana-llm-scripts .gitignore
git -C "${REPO_DIR}" apply --whitespace=nowarn "${JOB_DIR}/voltana-llm-scripts.patch" || fail patch_apply
{ echo "commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"; git -C "${REPO_DIR}" status --short; } | tee "${OUT}/repo_state.txt"

# ---- 3. python packages -------------------------------------------------------------------
# [srt] instead of [all]: the profiling path needs neither the litellm/anthropic clients nor
# torch_memory_saver (which would compile a CUDA extension).
# All third-party packages come from requirements-lock.txt (uv pip compile --exclude-newer 2025-06-20,
# i.e. the SGLang 0.4.7.post1 era). Unpinned deps resolved to 2026 releases otherwise: job 279465
# got compressed-tensors 0.15, which needs transformers.masking_utils (transformers >= 4.53).
# --exclude-newer also applies to build envs, so the sdist-only flashinfer-python 0.2.6.post1 is
# built against torch 2.7.1 (its setup.py pins torch to the build env's version; job 279367).
export UV_CACHE_DIR="${WORK}/uv-cache" UV_LINK_MODE=copy
timeout 600 "${PY}" -m pip install uv 2>&1 | tail -n 2
[ "${PIPESTATUS[0]}" -eq 0 ] || fail pip_uv
UV=("${PY}" -m uv pip)
timeout 3600 "${UV[@]}" install --python "${PY}" --exclude-newer 2025-06-20 -r "${JOB_DIR}/requirements-lock.txt" 2>&1 | tail -n 30
[ "${PIPESTATUS[0]}" -eq 0 ] || fail uv_lock_install
timeout 600 "${UV[@]}" install --python "${PY}" --no-deps -e "${REPO_DIR}/python" 2>&1 | tail -n 5
[ "${PIPESTATUS[0]}" -eq 0 ] || fail uv_sglang_editable
"${UV[@]}" show --python "${PY}" flashinfer-python 2>&1 | grep -i '^requires' || true
# NIXL 0.9.0 (same version as the cluster's vLLM jobs; the wheel bundles UCX). The shared
# /data/users/chjing/pylibs/nixl-0.9.0 is built for cuda-env's python 3.10, so install a cp312
# copy here. --no-deps: nixl-cu12 requires an unpinned torch, which would replace SGLang's 2.7.1.
TORCH_BEFORE=$("${PY}" -c 'import torch; print(torch.__version__)')
timeout 600 "${PY}" -m pip install --no-deps "nixl==0.9.0" "nixl-cu12==0.9.0" 2>&1 | tail -n 5
[ "${PIPESTATUS[0]}" -eq 0 ] || fail pip_nixl
TORCH_AFTER=$("${PY}" -c 'import torch; print(torch.__version__)')
echo "torch before=${TORCH_BEFORE} after=${TORCH_AFTER}" | tee "${OUT}/torch_version.txt"
[ "${TORCH_BEFORE}" = "${TORCH_AFTER}" ] || fail torch_changed_by_nixl
"${PY}" -m pip freeze > "${OUT}/pip_freeze.txt"
"${PY}" -m pip check > "${OUT}/pip_check.txt" 2>&1; echo "pip check rc=$?"

# ---- 4. model and dataset ---------------------------------------------------------------
MODEL=""
while IFS= read -r snap; do
  if [ -s "${snap}/config.json" ] && [ -s "${snap}/tokenizer_config.json" ] && find "${snap}" -maxdepth 1 -name '*.safetensors' -size +0c -print -quit | grep -q .; then MODEL="${snap}"; break; fi
done < <(find "${MODEL_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
[ -n "${MODEL}" ] || fail "mistral_snapshot_not_found ${MODEL_SNAPSHOT_ROOT}"
ln -sfn "${MODEL}" "${MODEL_LINK}"
[ -s "${DATASET}" ] || fail "sharegpt_missing ${DATASET}"
echo "model=${MODEL_LINK} -> ${MODEL}" | tee "${OUT}/paths.txt"
echo "dataset=${DATASET}" | tee -a "${OUT}/paths.txt"

# ---- 5. GPU, imports, clock permission ---------------------------------------------------
GPU_INDEX="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [ "$(nvidia-smi -L | wc -l)" -eq 1 ]; then GPU_INDEX=0; fi
[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || fail "unexpected_gpu_allocation=${GPU_INDEX}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}" NVML_GPU_IDX="${GPU_INDEX}"
{ echo "gpu_index=${GPU_INDEX}"; nvidia-smi -i "${GPU_INDEX}" --query-gpu=index,uuid,name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv; } | tee "${OUT}/gpu_allocation.txt"
nvidia-smi -i "${GPU_INDEX}" --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -un | tr '\n' ' ' > "${OUT}/supported_clocks_L4.txt"

"${PY}" - <<'PYEOF' 2>&1 | tee "${OUT}/imports.txt"
import importlib
for name in ("torch", "sglang", "sgl_kernel", "flashinfer", "transformers", "sklearn", "pandas", "orjson", "pynvml", "datasets",
             "nixl", "sglang.srt.disaggregation.fake", "sglang.srt.disaggregation.nixl",
             "sglang.srt.entrypoints.http_server", "sglang.srt.managers.scheduler", "sglang.bench_serving_voltanallm"):
    try:
        module = importlib.import_module(name)
        print(f"ok   {name} {getattr(module, '__version__', '?')}")
    except Exception as error:
        print(f"FAIL {name}: {error!r}")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PYEOF
grep -q '^FAIL' "${OUT}/imports.txt" && fail imports

# One NIXL agent with the UCX backend. Judged by the printed line, not the exit status: the
# cluster's NIXL smoke job 274777 created its agent fine and then segfaulted at interpreter exit.
timeout 120 "${PY}" - <<'PYEOF' > "${OUT}/nixl_check.txt" 2>&1
import nixl
from nixl._api import nixl_agent, nixl_agent_config
agent = nixl_agent("voltana-setup-check", nixl_agent_config(backends=["UCX"]))
print("nixl", nixl.__file__, "plugins", agent.get_plugin_list())
print("NIXL_UCX_AGENT_OK")
PYEOF
echo "nixl_check_rc=$?" >> "${OUT}/nixl_check.txt"
cat "${OUT}/nixl_check.txt"
grep -q NIXL_UCX_AGENT_OK "${OUT}/nixl_check.txt" || fail nixl_agent

SERVER_PID=""
cleanup() {
  set +e
  [ -z "${SERVER_PID}" ] || { kill "${SERVER_PID}" 2>/dev/null; sleep 5; kill -9 "${SERVER_PID}" 2>/dev/null; }
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null
  sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc > "${OUT}/clock_reset.txt" 2>&1
  for log in "${WORK}"/server-*.log "${WORK}"/profile-*.log "${WORK}"/fit-*.log; do
    [ ! -s "${log}" ] || tail -n 200 "${log}" > "${OUT}/$(basename "${log%.log}")_tail.log"
  done
  printf '%s\n' "${WORK}" > "${OUT}/RAW_CACHE_LOCATION.txt"
  echo "end=$(date -Iseconds) elapsed_s=$(( $(date +%s) - START_EPOCH ))"
}
trap cleanup EXIT
trap 'exit 143' TERM

# Judge the lock by nvidia-smi's "All done" (job 267572 exited 9 right after a good lock).
sudo -n nvidia-smi -i "${GPU_INDEX}" -lgc 1500,1500 > "${OUT}/clock_permission.txt" 2>&1
echo "lock_rc=$?" >> "${OUT}/clock_permission.txt"
grep -q "All done" "${OUT}/clock_permission.txt" || fail clock_lock_not_permitted
sudo -n nvidia-smi -i "${GPU_INDEX}" -rgc >> "${OUT}/clock_permission.txt" 2>&1

# ---- 6. smoke test on this L4 -------------------------------------------------------------
export MODEL_PATH="${MODEL_LINK}" DATASET_PATH="${DATASET}" HOST=127.0.0.1 PORT GPU_TAG=L4-24G-smoke
export CONTEXT_LEN=8192 MAX_RUNNING=256 CUDA_GRAPH_MAX_BS=256
F_MAX=$(tr ' ' '\n' < "${OUT}/supported_clocks_L4.txt" | sort -n | tail -n 1)

wait_ready() {  # $1 = server pid
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

run_smoke() {  # $1 = prefill|decode
  local phase=$1 server
  [ "${phase}" = prefill ] && server="${PRED_DIR}/prefill/p_server.sh" || server="${PRED_DIR}/decode/d_server.sh"
  [ "${phase}" = prefill ] && export MEM_FRACTION=0.85 || export MEM_FRACTION=0.88
  mkdir -p "${WORK}/smoke-${phase}"
  cd "${WORK}/smoke-${phase}" || return 1
  bash "${server}" 0 > "${WORK}/server-${phase}.log" 2>&1 &
  SERVER_PID=$!
  local t0; t0=$(date +%s)
  wait_ready "${SERVER_PID}" || { echo "${phase}_server_not_ready"; return 1; }
  echo "${phase}_server_ready_after_s=$(( $(date +%s) - t0 ))" | tee -a "${OUT}/smoke.txt"
  { echo "pkill_pattern_matches_for_this_job:"; pgrep -a -f 'python.*vllm|vllm\.entrypoints' | grep -v pgrep || echo none; } >> "${OUT}/smoke.txt"
  FREQ_LIST="${F_MAX}" RPS_LIST=5 BS_LIST=8,9 "${PY}" "${PRED_DIR}/${phase}/run_profile.py" \
    > "${WORK}/profile-${phase}.log" 2>&1 &
  wait $! || { echo "${phase}_profile_failed"; return 1; }
  "${PY}" "${PRED_DIR}/fit_latency.py" --phase "${phase}" --data-dir . --out-dir "${WORK}/smoke-${phase}/fit" \
    --tokens-min 0 > "${WORK}/fit-${phase}.log" 2>&1 || { echo "${phase}_fit_failed"; return 1; }
  cp "${WORK}/smoke-${phase}/fit/latency-${phase}.csv" "${OUT}/smoke-latency-${phase}.csv"
  local rows=$(( $(wc -l < "${OUT}/smoke-latency-${phase}.csv") - 1 ))
  echo "${phase}_extra_files=$(find . -name '*-extra.json' | wc -l) fit_rows=${rows} $(grep 'points' "${WORK}/fit-${phase}.log" | tail -n 1)" | tee -a "${OUT}/smoke.txt"
  [ "${rows}" -gt 0 ] || { echo "${phase}_fit_has_no_rows"; return 1; }
  kill "${SERVER_PID}"; wait "${SERVER_PID}" 2>/dev/null; SERVER_PID=""
  sleep 10
}

run_smoke prefill || fail smoke_prefill
run_smoke decode || fail smoke_decode
echo "${REPO_COMMIT} $(date -Iseconds) ${JOB_NAME}" > "${VOLTANA_ROOT}/ENV_READY"
echo "RESULT=ok"
