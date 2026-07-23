# 无可行 operating point 时的最小 SLO violation fallback

- 日期：2026-07-23（Europe/Stockholm）
- 集群验证：Minerva Slurm Job 250010
- 状态：COMPLETED，ExitCode `0:0`
- 用时：2 秒
- 测试：4 个边界 workload × 固定/自动路由，共 8 次决策

## 调度规则

正常路径保持不变：

1. 枚举路由和 DVFS operating point；
2. 只保留同时通过 latency 与 saturation gate 的安全候选；
3. 在安全候选中选择预测集群功耗最低的配置。

新增的过载路径只在 `num_safe=0` 时触发。调度器返回独立状态
`OVERLOAD_FALLBACK`，并保持 `recommended.is_safe=false`。候选依次按以下指标排序：

1. prefill/decode latency-SLO violation 与两端 saturation 概率的预测并集；
2. TTFT/TPOT 最大归一化预测超限；
3. 联合 latency-SLO violation 概率；
4. 预测集群功耗。

概率并集使用独立性近似，仅作为 best-effort fallback 的排序分数，不应解释为
已校准的端到端 SLO violation 概率。

此外，正常安全集合现在直接要求预测 P99 TTFT/TPOT 不超过请求 SLO。这样可以
阻止此前 `extreme_rps` 中“预测 TTFT/TPOT 已超标但 classifier guard 仍标记
safe”的内部矛盾。

纯 admission-control 实验仍可通过 `--overload-action reject` 保留原来的
`NO_SAFE_CONFIG` 行为。

## 集群决策结果

| Workload | 路由搜索 | 安全候选 | 选择的 prefill | 选择的 decode | 预测 overload violation | 最大预测 SLO 超限 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| `threshold_long_prompt` | 固定/自动 | 0 | L40S 1755 MHz | L4 2040 MHz | 0.406762 | 0 | `OVERLOAD_FALLBACK` |
| `long_decode` | 固定/自动 | 0 | L40S 2520 MHz | L4 2040 MHz | 0.999086 | 0 | `OVERLOAD_FALLBACK` |
| `moderate_burst` | 固定/自动 | 0 | L40S 1755 MHz | L4 2040 MHz | 0.816150 | 0 | `OVERLOAD_FALLBACK` |
| `extreme_rps` | 固定/自动 | 0 | L40S 2265 MHz | L4 2040 MHz | 0.998297 | 0.1018 | `OVERLOAD_FALLBACK` |

自动路由枚举 200 个候选，固定路由枚举 100 个候选；本组数据中，自动路由最终
仍选择 L40S prefill、L4 decode，因此没有找到更低违约的角色交换配置。

八个推荐项都满足：

```text
status=OVERLOAD_FALLBACK
num_safe=0
recommended.is_safe=false
```

标准库单元测试同时覆盖：

- 存在安全候选时继续使用 `OK / safe_min_power`；
- 无安全候选时使用 `OVERLOAD_FALLBACK`；
- gate-only 的 `reject` 兼容模式仍返回 `NO_SAFE_CONFIG`。

## Slurm 与 GPU 风险

Job 250010 只执行调度决策验证，没有请求 GPU TRES，没有启动 vLLM，也没有运行
任何 `nvidia-smi`、DVFS、power-gating 或 Slurm 管理操作。stdout 记录：

```text
decision_count=8
gpu_tres_requested=false
gpu_commands_executed=false
vllm_started=false
job_exit_rc=0
```

stderr 为空。这次验证没有实际执行过载配置，因此不会给其他 GPU workload 带来
资源或频率状态风险。

## 运行时限制

四个 fallback 都选择 L4 2040 MHz，因为模型认为高频能降低违约风险。但此前
Job 250005 已证明，L4 即使接受 2040 MHz 锁频命令，在主动 CUDA 负载下也只能
维持约 1.20–1.22 GHz。因此：

- 2040 MHz 目前只是名义 DVFS target，不是已验证可持续的 operating point；
- 真正执行 fallback 前应加入 runtime-sustainable frequency constraint；
- 若频率确认失败，应继续尝试下一低违约候选，而不是直接结束调度。

当前候选空间仍固定 TP=1、一个 prefill GPU 和一个 decode GPU。只有把额外 GPU
副本、TP bundle 和实际可持续频率加入搜索后，才能把该 fallback 称为完整联合
调度器中的最低 SLO violation 配置。

## 结论

1. “无可行配置时选择最低 SLO violation”已实现为明确的 overload fallback。
2. best-effort 推荐不会被错误标记为安全配置。
3. 正常安全路径新增了预测 P99 对 SLO 的直接检查，修复原实验暴露的明显漏洞。
4. 同一组四个边界 workload 的固定与自动路由决策均通过集群验证。
5. 下一步不是直接执行 L4 2040 MHz，而是先把可持续频率和更完整的 GPU/TP
   operating point 纳入候选约束。
