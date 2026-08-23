#!/usr/bin/env bash

# =============================================================================
# Slurm 资源声明
# 两个节点各申请 4 张 GPU。TP 布局 1+1+2 恰好使用每个节点的 4 张 GPU；
# exclusive 避免其他作业共享 GPU、CPU 或网卡而污染延迟和功耗结果。
# =============================================================================
#SBATCH --job-name=nixl-3p3d-predict-dvfs
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260819-210313-luqia-vllm-nixl-3p3d-predictive-dvfs/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260819-210313-luqia-vllm-nixl-3p3d-predictive-dvfs/slurm-%j.err

set -uo pipefail

# =============================================================================
# 任务和公共组件路径
# 本 Job 直接管理完整生命周期，只复用 script/ 下与 Job 无关的通用组件。
# =============================================================================
export JOB_NAME_OVERRIDE="20260819-210313-luqia-vllm-nixl-3p3d-predictive-dvfs"
BROKER_ROOT="/data/users/chjing/Sweep_LLM_Jobs_broker"
JOB_DIR="${BROKER_ROOT}/jobs/${JOB_NAME_OVERRIDE}"
SCRIPT_ROOT="${BROKER_ROOT}/script"
VALIDATOR="${JOB_DIR}/validate_job_config.py"
ROUTE_CHECKER="${JOB_DIR}/check_random_routes.py"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
export PYTHON_BIN="${ENV_DIR}/bin/python"
export VLLM_BIN="${ENV_DIR}/bin/vllm"

# =============================================================================
# 3P3D 拓扑和路由
# 每侧创建三个实例，TP=[1,1,2]。允许非对称 TP 后会形成完整的 3x3 路由；
# 固定随机种子使重复任务能够获得同样的路由序列，便于 A/B 对比。
# =============================================================================
export PREFILL_REPLICAS_OVERRIDE=3
export PREFILL_TP_SIZES_OVERRIDE=1,1,2
export DECODE_REPLICAS_OVERRIDE=3
export DECODE_TP_SIZES_OVERRIDE=1,1,2
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE=true
export CUSTOM_PD_RANDOM_SEED=260819210313

# =============================================================================
# NIXL/UCX KV Cache 传输
# 公共单实例启动器会把连接器配置为 NixlConnector，并分别赋予 producer 和
# consumer 角色。UCX 绑定两台机器各自的高速数据网卡；all 保留可用后端探测。
# 4.3063 Gbit/s 是 Job 252766 的保守实测有效带宽，供延迟模型估算 KV 时间。
# =============================================================================
export PD_KV_CONNECTOR=NixlConnector
export PD_KV_LOAD_FAILURE_POLICY=fail
export UCX_TLS=all
export PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE=4.3063
export PD_DVFS_DISPATCH_MS_OVERRIDE=0

# =============================================================================
# 模型预测调频策略
# 调度器根据请求形状、到达率、GPU/TP 类型枚举频率组合；先过滤 TTFT、TPOT
# 和饱和度不安全的候选，再选择预测功耗最低的一组。若没有安全组合则拒绝请求。
# 时钟代理设频后等待 20 秒，并检查实测频率是否落在目标 ±30 MHz 内。
# =============================================================================
export PD_ENABLE_PREDICTIVE_DVFS_OVERRIDE=true
export PD_DVFS_SLO_TTFT_MS_OVERRIDE=500
export PD_DVFS_SLO_TPOT_MS_OVERRIDE=200
export PD_DVFS_OVERLOAD_ACTION_OVERRIDE=reject
export PD_DVFS_CLOCK_TOLERANCE_MHZ_OVERRIDE=30
export PD_DVFS_SETTLE_SECONDS_OVERRIDE=20
export PD_DVFS_CLOCK_TIMEOUT_SECONDS_OVERRIDE=60

# =============================================================================
# 全路由预热
# 在正式测量前依次跑完 9 条 P-D 路由，初始化模型、NIXL 连接和 KV 数据路径。
# warmup 必须与正式请求使用同样的输出长度；OL=1 超出模型校准形状，会导致
# 所有候选频率被拒绝（历史 Job 252879 已验证该失败模式）。
# =============================================================================
export WARMUP_ALL_ROUTES_OVERRIDE=true
export WARMUP_OUTPUT_TOKENS_OVERRIDE=128
export WARMUP_TIMEOUT_SECONDS_OVERRIDE=300

# =============================================================================
# 正式测量负载
# 36 个固定形状请求按 2 RPS 到达，最多 16 个并发。本脚本会保存详细结果，
# 并验证成功数、路由、调频 ACK、遥测和精确 forwarding TTFT。
# =============================================================================
export MODEL_OVERRIDE=mistralai/Mistral-7B-v0.1
export NUM_PROMPTS_OVERRIDE=36
export REQUEST_RATE_OVERRIDE=2
export INPUT_LEN_OVERRIDE=512
export OUTPUT_LEN_OVERRIDE=128
export MAX_CONCURRENCY_OVERRIDE=16
export BENCH_TIMEOUT_SECONDS_OVERRIDE=1800
export STARTUP_TIMEOUT_SECONDS_OVERRIDE=1500

# =============================================================================
# 把实验参数解析为启动器需要的实际配置
# 这里吸收了旧 Job 的参数展开逻辑，但只保留本任务会用到的 3P3D/NIXL 分支。
# =============================================================================
export MODEL="$MODEL_OVERRIDE"
export PREFILL_NODE=uranus PREFILL_IP=10.1.0.5
export PREFILL_NET_IFACE=enp160s0f0np0 PREFILL_GPU_MODEL=L40S
export PREFILL_REPLICAS="$PREFILL_REPLICAS_OVERRIDE"
export PREFILL_TP_SIZE=1 PREFILL_TP_SIZES="$PREFILL_TP_SIZES_OVERRIDE"
export DECODE_NODE=ganymede DECODE_IP=10.1.0.3
export DECODE_NET_IFACE=eno12399np0 DECODE_GPU_MODEL=L4
export DECODE_REPLICAS="$DECODE_REPLICAS_OVERRIDE"
export DECODE_TP_SIZE=1 DECODE_TP_SIZES="$DECODE_TP_SIZES_OVERRIDE"
export PROXY_NODE=ganymede PROXY_IP=10.1.0.3
export CUSTOM_PD_DEFAULT_ROUTE=random
export CUSTOM_PD_ALLOW_ASYMMETRIC_TP="$CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE"

export PD_ENABLE_PREDICTIVE_DVFS="$PD_ENABLE_PREDICTIVE_DVFS_OVERRIDE"
export PD_DVFS_SLO_TTFT_MS="$PD_DVFS_SLO_TTFT_MS_OVERRIDE"
export PD_DVFS_SLO_TPOT_MS="$PD_DVFS_SLO_TPOT_MS_OVERRIDE"
export PD_DVFS_EXPECTED_REQUEST_RATE="$REQUEST_RATE_OVERRIDE"
export PD_DVFS_INPUT_TOKENS_OVERRIDE="$INPUT_LEN_OVERRIDE"
export PD_DVFS_OVERLOAD_ACTION="$PD_DVFS_OVERLOAD_ACTION_OVERRIDE"
export PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS="$PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE"
export PD_DVFS_DISPATCH_MS="$PD_DVFS_DISPATCH_MS_OVERRIDE"
export PD_DVFS_CLOCK_TOLERANCE_MHZ="$PD_DVFS_CLOCK_TOLERANCE_MHZ_OVERRIDE"
export PD_DVFS_SETTLE_SECONDS="$PD_DVFS_SETTLE_SECONDS_OVERRIDE"
export PD_DVFS_CLOCK_TIMEOUT_SECONDS="$PD_DVFS_CLOCK_TIMEOUT_SECONDS_OVERRIDE"

export MAX_MODEL_LEN=8192 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=32
export GPU_MEMORY_UTILIZATION=0.82 DTYPE=float16 ENFORCE_EAGER=true
export CPUS_PER_GPU=8 VLLM_STEP_MEM=32G
export STARTUP_TIMEOUT_SECONDS="$STARTUP_TIMEOUT_SECONDS_OVERRIDE"

# Job ID 决定一组独立端口，避免同时运行的任务互相占用。
PORT_OFFSET=$(( (${SLURM_JOB_ID:-0} % 60) * 16 ))
export PROXY_HTTP_PORT=$((30000 + PORT_OFFSET))
export PROXY_REGISTER_PORT=$((31000 + PORT_OFFSET))
export PREFILL_HTTP_PORT_BASE=$((32000 + PORT_OFFSET))
export DECODE_HTTP_PORT_BASE=$((33000 + PORT_OFFSET))
export PREFILL_KV_PORT_BASE=$((34000 + PORT_OFFSET))
export DECODE_KV_PORT_BASE=$((34000 + PORT_OFFSET))

OUT_DIR="${JOB_DIR}/results/${SLURM_JOB_ID:-manual}"
export PD_OUT_DIR="${OUT_DIR}/pd_runtime"
export PD_WORK_DIR="/data/users/chjing/vllm_job_work/${SLURM_JOB_ID:-manual}"
export HOME="${PD_WORK_DIR}/home"
export HF_HOME="${PD_WORK_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${PD_WORK_DIR}/xdg-cache"
export TMPDIR="${PD_WORK_DIR}/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PATH="${ENV_DIR}/bin:${PATH}"

LAUNCHER="${SCRIPT_ROOT}/start_pd_vllm.sh"
export PROXY_SCRIPT="${SCRIPT_ROOT}/scheduler_custom_policy.py"
export INSTANCE_SCRIPT="${SCRIPT_ROOT}/start_pd_vllm_instance.sh"
WARMUP_SCRIPT="${SCRIPT_ROOT}/warmup_pd_routes.py"
DVFS_CHECKER="${SCRIPT_ROOT}/check_request_dvfs.py"
TTFT_CORRECTOR="${SCRIPT_ROOT}/correct_vllm_bench_dvfs_ttft.py"
TTFT_SUMMARIZER="${SCRIPT_ROOT}/summarize_forwarding_ttft.py"

BENCH_LOG="${OUT_DIR}/vllm_bench.txt"
BENCH_RESULT="${OUT_DIR}/vllm_bench_detailed.json"
REGISTRY_FILE="${OUT_DIR}/registry.json"
WARMUP_RESULT="${OUT_DIR}/route_warmup.json"
WARMUP_DECISIONS="${OUT_DIR}/route_warmup_dvfs_decisions.jsonl"
MEASUREMENT_PROXY_LOG="${OUT_DIR}/proxy_measurement.log"
ROUTE_CHECK_FILE="${OUT_DIR}/random_route_check.json"
DVFS_CHECK_FILE="${OUT_DIR}/request_dvfs_check.json"
CORRECTED_RESULT="${OUT_DIR}/vllm_bench_detailed_excluding_dvfs_wait.json"
CORRECTED_LOG="${OUT_DIR}/vllm_bench_ttft_excluding_dvfs_wait.txt"
FORWARDING_SUMMARY="${OUT_DIR}/forwarding_ttft_summary.json"
FORWARDING_REQUESTS="${OUT_DIR}/forwarding_ttft_requests.csv"

# =============================================================================
# 安全清理与集群就绪检查
# 任务退出时终止后台进程，并且只允许删除与当前 Slurm Job ID 完全匹配的目录。
# =============================================================================
PD_LAUNCHER_PID=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$PD_LAUNCHER_PID" ] && kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
    kill "$PD_LAUNCHER_PID" 2>/dev/null || true
    wait "$PD_LAUNCHER_PID" 2>/dev/null || true
  fi
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] &&
    [ "$PD_WORK_DIR" = "/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}" ]; then
    rm -rf -- "$PD_WORK_DIR" || rc=3
  else
    echo "cache_cleanup_skipped unsafe_path=${PD_WORK_DIR}" >&2
    rc=3
  fi
  echo "job_cleanup_rc=${rc}"
  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_registry() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
      echo "pd_launcher_exited_before_ready=true" >&2
      return 1
    fi
    if curl -fsS --connect-timeout 2 --max-time 5 "http://${PROXY_IP}:${PROXY_HTTP_PORT}/registry" > "$REGISTRY_FILE" &&
      "$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = (
    d.get("prefill_count") == 3
    and d.get("decode_count") == 3
    and d.get("prefill_tp_sizes") == [1, 1, 2]
    and d.get("decode_tp_sizes") == [1, 1, 2]
    and len(d.get("supported_routes", [])) == 9
    and d.get("allow_asymmetric_tp") is True
    and d.get("default_route") == "random"
    and d.get("kv_connector") == "NixlConnector"
)
raise SystemExit(0 if ok else 1)
' "$REGISTRY_FILE"; then
      echo "pd_registry_ready=true"
      return 0
    fi
    sleep 2
  done
  echo "pd_registry_ready=false timeout_s=${STARTUP_TIMEOUT_SECONDS}" >&2
  return 1
}

# =============================================================================
# 依赖检查、配置校验和集群启动
# 启动代理、3 个 Prefill、3 个 Decode，以及每个实例对应的时钟代理。
# =============================================================================
mkdir -p "$OUT_DIR" "$PD_OUT_DIR" "$HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
ulimit -n 65536

REQUIRED_FILES=(
  "$PYTHON_BIN" "$VLLM_BIN" "$LAUNCHER" "$PROXY_SCRIPT" "$INSTANCE_SCRIPT"
  "$VALIDATOR" "$ROUTE_CHECKER" "$WARMUP_SCRIPT" "$DVFS_CHECKER"
  "$TTFT_CORRECTOR" "$TTFT_SUMMARIZER"
)
for required in "${REQUIRED_FILES[@]}"; do
  [ -e "$required" ] || {
    echo "required_artifact_missing=${required}" >&2
    exit 2
  }
done
for command_name in curl timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "required_command_missing=${command_name}" >&2
    exit 2
  }
done

"$PYTHON_BIN" "$VALIDATOR" || exit 3

echo "phase=start_pd_cluster"
"$LAUNCHER" &
PD_LAUNCHER_PID=$!
wait_for_registry || exit 5
"$PYTHON_BIN" -m json.tool "$REGISTRY_FILE"

# =============================================================================
# 全部 9 条 P-D 路由预热
# 初始化模型和 NIXL 数据路径；预热决策单独保存，不混入正式统计。
# =============================================================================
echo "phase=warmup_all_pd_routes"
WARMUP_ARGS=(
  --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}"
  --model "$MODEL"
  --prefill-tp-sizes "$PREFILL_TP_SIZES"
  --decode-tp-sizes "$DECODE_TP_SIZES"
  --allow-asymmetric-tp
  --input-tokens "$INPUT_LEN_OVERRIDE"
  --output-tokens "$WARMUP_OUTPUT_TOKENS_OVERRIDE"
  --timeout-seconds "$WARMUP_TIMEOUT_SECONDS_OVERRIDE"
  --output "$WARMUP_RESULT"
)
"$PYTHON_BIN" "$WARMUP_SCRIPT" "${WARMUP_ARGS[@]}" || exit 6

MEASUREMENT_REQUEST_START="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_count"])' "$WARMUP_RESULT")"
cp "${PD_OUT_DIR}/request_dvfs_decisions.jsonl" "$WARMUP_DECISIONS" || exit 7
: > "${PD_OUT_DIR}/request_dvfs_decisions.jsonl"
sleep 1
PROXY_MEASUREMENT_START_LINE=$(( $(wc -l < "${PD_OUT_DIR}/proxy.log") + 1 ))
echo "route_warmup_complete=true routes=${MEASUREMENT_REQUEST_START}"

# =============================================================================
# 正式压测
# 运行 36 个固定形状请求，并保存 benchmark 原始日志和逐请求数据。
# =============================================================================
echo "phase=vllm_bench"
BENCH_ARGS=(
  bench serve
  --backend openai
  --base-url "http://${PROXY_IP}:${PROXY_HTTP_PORT}"
  --model "$MODEL"
  --dataset-name random
  --random-input-len "$INPUT_LEN_OVERRIDE"
  --random-output-len "$OUTPUT_LEN_OVERRIDE"
  --num-prompts "$NUM_PROMPTS_OVERRIDE"
  --request-rate "$REQUEST_RATE_OVERRIDE"
  --max-concurrency "$MAX_CONCURRENCY_OVERRIDE"
  --num-warmups 0
  --temperature 0
  --save-result --save-detailed
  --result-dir "$OUT_DIR"
  --result-filename "$(basename "$BENCH_RESULT")"
)
timeout --signal=INT --kill-after=20s "${BENCH_TIMEOUT_SECONDS_OVERRIDE}s" "$VLLM_BIN" "${BENCH_ARGS[@]}" </dev/null > "$BENCH_LOG" 2>&1
BENCH_RC=$?
cat "$BENCH_LOG"

read -r SUCCESSFUL_REQUESTS FAILED_REQUESTS <<< "$("$PYTHON_BIN" -c '
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ok = re.search(r"Successful requests:\s+(\d+)", text)
bad = re.search(r"Failed requests:\s+(\d+)", text)
print(ok.group(1) if ok else -1, bad.group(1) if bad else -1)
' "$BENCH_LOG")"
tail -n "+${PROXY_MEASUREMENT_START_LINE}" "${PD_OUT_DIR}/proxy.log" > "$MEASUREMENT_PROXY_LOG"

# =============================================================================
# 路由、调频和 TTFT 校验
# 把各项检查的返回码汇总为 Job 返回码，避免“压测结束但结果无效”仍显示成功。
# =============================================================================
TTFT_RC=0
TTFT_ARGS=(--input "$BENCH_RESULT" --wait-seconds "$PD_DVFS_SETTLE_SECONDS" --output "$CORRECTED_RESULT")
"$PYTHON_BIN" "$TTFT_CORRECTOR" "${TTFT_ARGS[@]}" > "$CORRECTED_LOG" 2>&1 || TTFT_RC=$?

ROUTE_RC=0
ROUTE_ARGS=(
  --proxy-log "$MEASUREMENT_PROXY_LOG"
  --expected-requests "$NUM_PROMPTS_OVERRIDE"
  --expected-count-start "$MEASUREMENT_REQUEST_START"
  --prefill-tp-sizes "$PREFILL_TP_SIZES"
  --decode-tp-sizes "$DECODE_TP_SIZES"
  --output "$ROUTE_CHECK_FILE"
)
"$PYTHON_BIN" "$ROUTE_CHECKER" "${ROUTE_ARGS[@]}" || ROUTE_RC=$?

DVFS_RC=0
DVFS_ARGS=(
  --decisions "${PD_OUT_DIR}/request_dvfs_decisions.jsonl"
  --telemetry-dir "$PD_OUT_DIR"
  --expected-requests "$NUM_PROMPTS_OVERRIDE"
  --clock-tolerance-mhz "$PD_DVFS_CLOCK_TOLERANCE_MHZ"
  --settle-seconds "$PD_DVFS_SETTLE_SECONDS"
  --output "$DVFS_CHECK_FILE"
)
"$PYTHON_BIN" "$DVFS_CHECKER" "${DVFS_ARGS[@]}" || DVFS_RC=$?

FORWARDING_RC=0
FORWARDING_ARGS=(
  --decisions "${PD_OUT_DIR}/request_dvfs_decisions.jsonl"
  --expected-requests "$NUM_PROMPTS_OVERRIDE"
  --slo-ttft-ms "$PD_DVFS_SLO_TTFT_MS"
  --output "$FORWARDING_SUMMARY"
  --csv-output "$FORWARDING_REQUESTS"
)
"$PYTHON_BIN" "$TTFT_SUMMARIZER" "${FORWARDING_ARGS[@]}" || FORWARDING_RC=$?

FINAL_RC=0
if [ "$BENCH_RC" -ne 0 ] ||
  [ "$SUCCESSFUL_REQUESTS" -ne "$NUM_PROMPTS_OVERRIDE" ] ||
  [ "$FAILED_REQUESTS" -ne 0 ]; then
  FINAL_RC=10
elif [ "$ROUTE_RC" -ne 0 ]; then
  FINAL_RC=11
elif [ "$DVFS_RC" -ne 0 ]; then
  FINAL_RC=13
elif [ "$TTFT_RC" -ne 0 ]; then
  FINAL_RC=14
elif [ "$FORWARDING_RC" -ne 0 ]; then
  FINAL_RC=15
elif ! kill -0 "$PD_LAUNCHER_PID" 2>/dev/null; then
  FINAL_RC=12
fi

echo "job_result bench_rc=${BENCH_RC} successful=${SUCCESSFUL_REQUESTS} failed=${FAILED_REQUESTS} route_rc=${ROUTE_RC} dvfs_rc=${DVFS_RC} ttft_rc=${TTFT_RC} forwarding_rc=${FORWARDING_RC}"
echo "job_exit_rc=${FINAL_RC}"
exit "$FINAL_RC"
