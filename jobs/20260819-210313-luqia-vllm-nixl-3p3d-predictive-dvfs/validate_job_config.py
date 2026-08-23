#!/usr/bin/env python3
"""在占用 GPU 前验证 NIXL 3P3D 预测调频任务的关键不变量。"""

from __future__ import annotations

import json
import os
from typing import Any


# =============================================================================
# 环境变量读取工具
# 所有错误都在这里转成清晰的配置错误，避免任务启动数分钟后才因拼写失败。
# =============================================================================
def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"缺少必需环境变量：{name}")
    return value


def positive_int(name: str) -> int:
    value = int(required(name))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值为 {value}")
    return value


def positive_float(name: str) -> float:
    value = float(required(name))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值为 {value}")
    return value


def tp_sizes(name: str) -> list[int]:
    values = [int(item.strip()) for item in required(name).split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} 必须是逗号分隔的正整数")
    return values


def enabled(name: str) -> bool:
    return required(name).lower() in {"1", "true", "yes", "on"}


# =============================================================================
# 3P3D 和 GPU 资源校验
# 三个实例的 TP 总和必须是 4，才能恰好对应每节点申请的四张 GPU。
# =============================================================================
def validate_topology() -> dict[str, Any]:
    prefill_replicas = positive_int("PREFILL_REPLICAS_OVERRIDE")
    decode_replicas = positive_int("DECODE_REPLICAS_OVERRIDE")
    prefill_tp = tp_sizes("PREFILL_TP_SIZES_OVERRIDE")
    decode_tp = tp_sizes("DECODE_TP_SIZES_OVERRIDE")

    if prefill_replicas != 3 or decode_replicas != 3:
        raise ValueError("本任务必须是 3 Prefill + 3 Decode")
    if len(prefill_tp) != prefill_replicas or len(decode_tp) != decode_replicas:
        raise ValueError("TP 列表长度必须与对应的实例数量一致")
    if sum(prefill_tp) != 4 or sum(decode_tp) != 4:
        raise ValueError("每侧 TP 总和必须为 4，匹配每节点四张 GPU")
    if not enabled("CUSTOM_PD_ALLOW_ASYMMETRIC_TP_OVERRIDE"):
        raise ValueError("必须允许非对称 TP，才能使用完整 3x3 P-D 路由")

    return {"prefill_tp": prefill_tp, "decode_tp": decode_tp, "routes": 9}


# =============================================================================
# NIXL 和预测调频校验
# 明确禁止连接器静默退化，并验证预测器需要的带宽、SLO、稳频和超时参数。
# =============================================================================
def validate_nixl_and_dvfs() -> dict[str, Any]:
    if required("PD_KV_CONNECTOR") != "NixlConnector":
        raise ValueError("PD_KV_CONNECTOR 必须为 NixlConnector")
    if required("PD_KV_LOAD_FAILURE_POLICY") != "fail":
        raise ValueError("KV 加载失败策略必须为 fail，禁止静默隐藏传输错误")
    if not enabled("PD_ENABLE_PREDICTIVE_DVFS_OVERRIDE"):
        raise ValueError("必须启用模型预测调频")

    bandwidth = positive_float("PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS_OVERRIDE")
    ttft = positive_int("PD_DVFS_SLO_TTFT_MS_OVERRIDE")
    tpot = positive_int("PD_DVFS_SLO_TPOT_MS_OVERRIDE")
    settle = positive_float("PD_DVFS_SETTLE_SECONDS_OVERRIDE")
    timeout = positive_float("PD_DVFS_CLOCK_TIMEOUT_SECONDS_OVERRIDE")
    if timeout <= settle:
        raise ValueError("调频超时必须大于稳频等待时间")

    return {
        "connector": "NixlConnector",
        "ucx_tls": required("UCX_TLS"),
        "kv_bandwidth_gbps": bandwidth,
        "slo_ttft_ms": ttft,
        "slo_tpot_ms": tpot,
        "settle_seconds": settle,
    }


# =============================================================================
# 预热和测量形状校验
# 预热输出长度必须等于测量输出长度，保证预测模型处在已校准的工作负载形状内。
# =============================================================================
def validate_workload() -> dict[str, Any]:
    if not enabled("WARMUP_ALL_ROUTES_OVERRIDE"):
        raise ValueError("必须启用全部 9 条 P-D 路由预热")

    input_len = positive_int("INPUT_LEN_OVERRIDE")
    output_len = positive_int("OUTPUT_LEN_OVERRIDE")
    warmup_output = positive_int("WARMUP_OUTPUT_TOKENS_OVERRIDE")
    if warmup_output != output_len:
        raise ValueError("warmup 输出长度必须与正式测量输出长度一致")

    return {
        "model": required("MODEL_OVERRIDE"),
        "input_tokens": input_len,
        "output_tokens": output_len,
        "num_prompts": positive_int("NUM_PROMPTS_OVERRIDE"),
        "request_rate": positive_float("REQUEST_RATE_OVERRIDE"),
        "max_concurrency": positive_int("MAX_CONCURRENCY_OVERRIDE"),
    }


# =============================================================================
# 汇总输出
# 输出一行机器可读 JSON，既是提交日志，也是后续确认实际配置的审计记录。
# =============================================================================
def main() -> int:
    summary = {
        "ok": True,
        "job": required("JOB_NAME_OVERRIDE"),
        "topology": validate_topology(),
        "nixl_and_dvfs": validate_nixl_and_dvfs(),
        "workload": validate_workload(),
    }
    print("job_config_validation=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
