# vLLM 无 saturation gate 边界负载实测

- 日期：2026-07-23（Europe/Stockholm）
- 集群：Minerva
- 成功任务：Slurm Job 250005
- 状态：COMPLETED，ExitCode `0:0`
- 用时：557 秒
- 节点：Neptune（1×L40S）+ Ganymede（1×L4）
- 推理：Mistral-7B，vLLM prefill/decode 分离，TP=1
- 策略：SWEEP latency-only scheduler，不使用 saturation gate

## 目的

从 `Dual_Sweep_LLM/calibration_data` 选择接近或超过容量边界的 workload，
验证没有 saturation gate 时，延迟调度器是否会放行实际吞吐无法承载的配置。

测量饱和定义与 calibration 项目一致：

```text
measured_saturated = achieved_request_throughput / configured_request_rate < 0.95
```

## 数据选择

原始数据：

- `Phase2_Results_L40S_master_results.csv`
- `Phase2_Results_L4_master_results.csv`
- `calibration_out/feasibility_by_config.csv`
- `calibration_out/max_sustainable_rate.csv`

四个 workload 都会被 latency-only 调度器放行，并保持 L40S prefill、L4
decode。它们仍在现有 `max-model-len=4096` 范围内，避免把容量实验混成
上下文长度 OOM 实验。

| Workload | Input/output | 配置速率 | 选择理由 |
|---|---:|---:|---|
| `threshold_long_prompt` | 1024/128 | 1 rps | Saturation gate 对推荐 decode 配置预测约 0.344，刚超过 0.30 阈值。 |
| `long_decode` | 512/1024 | 1 rps | 长输出；gate 对 prefill/decode 的预测饱和概率约为 0.926/0.991。 |
| `moderate_burst` | 128/128 | 5 rps | calibration 的 PD 结果在 5 rps 输入下仅达到约 1.71–1.72 rps。 |
| `extreme_rps` | 2/64 | 50 rps | calibration 的 L4 TP=1 结果在 50 rps 下仅达到约 10.45–10.97 rps。 |

## 实测结果

| Workload | 成功/失败 | 实际 rps | 吞吐比 | 饱和 | P99 TTFT | TTFT≤500 | P99 TPOT | TPOT≤200 |
|---|---:|---:|---:|---|---:|---|---:|---|
| `threshold_long_prompt` | 12/0 | 0.60 | 60.0% | 是 | 353.34 ms | 是 | 63.79 ms | 是 |
| `long_decode` | 12/0 | 0.15 | 15.0% | 是 | 321.52 ms | 是 | 67.58 ms | 是 |
| `moderate_burst` | 50/0 | 1.52 | 30.4% | 是 | 386.66 ms | 是 | 63.18 ms | 是 |
| `extreme_rps` | 500/0 | 3.58 | 7.16% | 是 | 626.38 ms | 否 | 63.85 ms | 是 |

前三组是直接的“延迟 SLO 看起来安全、实际吞吐饱和”反例。第四组负载更高，
不仅吞吐严重不足，TTFT 也因排队升至 626.38 ms。

所有请求最终成功并不代表系统可持续承载输入速率：benchmark 会等待积压请求
排空，因此完整运行时长和实际 request throughput 才能反映容量。

## GPU 频率行为

| Workload | L40S 目标/实测均值 | L4 目标/实测均值 |
|---|---:|---:|
| `threshold_long_prompt` | 1245/1245 MHz | 2040/1217.5 MHz |
| `long_decode` | 735/735 MHz | 2040/1200 MHz |
| `moderate_burst` | 480/480 MHz | 1200/1177.5 MHz |
| `extreme_rps` | 480/480 MHz | 780/780 MHz |

L40S 可以准确维持四个目标。L4 接受 2040 MHz 锁频命令，但主动 CUDA 负载下
只能维持约 1.20–1.22 GHz；1200 和 780 MHz 目标基本可维持。因此调度器除了
预测 saturation，还应把“频率是否能在当前功率/热约束下持续”纳入配置可行性。

首次任务 250004 因 2040 MHz 确认偏差而在 benchmark 前安全退出；两个节点都
完成了双重 `-rgc`。Job 250005 保留 2040 MHz 推荐和实际频率记录，只放宽了
继续 benchmark 的确认容差。

## Slurm 与清理

作业只通过普通 Slurm 指令申请两个节点、每节点一张 GPU，并只在自己的
allocation 内启动 `srun`。没有执行 `sbatch`、`scancel`、`scontrol update`、
drain/resume、MIG、持久化模式、功率上限或 power-gating 操作。

结束时先停止 vLLM server step，再启动 fresh two-node reset step。Ganymede GPU
5 和 Neptune GPU 0 都成功执行双重 `nvidia-smi -rgc`，输出包含：

```text
reset_gpu_clock_verified=true
parent_reset_verified=true
job_exit_rc=0
```

## 结论

1. 四个 latency-only 推荐全部发生实际吞吐饱和。
2. 其中三组仍满足 TTFT/TPOT SLO，证明延迟 gate 不能替代容量/saturation gate。
3. 极端 50 rps 负载的吞吐比仅为 7.16%，并进一步导致 TTFT 违约。
4. L4 的不可持续高频是独立的配置可行性问题，不能把 `nvidia-smi -lgc`
   返回成功等同于目标频率在推理负载下可持续。
5. 下一步调度器应在当前配置饱和时继续搜索路由、可持续频率、活跃 GPU 数和
   TP bundle；只有完整资源可行候选集为空时才进入过载处理。
