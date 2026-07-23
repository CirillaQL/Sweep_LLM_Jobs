# vLLM saturation gate 边界负载测试

- 日期：2026-07-23（Europe/Stockholm）
- 集群：Minerva
- Slurm Job：250009
- 状态：COMPLETED，ExitCode `0:0`
- 用时：2 秒
- 测试数据：与无 gate Job 250005 完全相同
- Gate：`mean(p_saturated) < 0.30`
- 搜索：固定 L40S-prefill/L4-decode，以及自动路由

## 目的

复用无 gate 实测中已经确认饱和的四组 workload，检查 saturation gate 是否会：

1. 排除预计饱和的当前配置；
2. 继续搜索其他路由和 DVFS 频率候选；
3. 在候选空间为空时拒绝请求，而不是启动不可承载的 vLLM 配置。

## Gate 决策

| Workload | Input/output | 请求速率 | 固定路由：延迟安全/总候选 | 自动路由：延迟安全/总候选 | Saturation-safe | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `threshold_long_prompt` | 1024/128 | 1 rps | 6/100 | 6/200 | 0 | `NO_SAFE_CONFIG` |
| `long_decode` | 512/1024 | 1 rps | 8/100 | 8/200 | 0 | `NO_SAFE_CONFIG` |
| `moderate_burst` | 128/128 | 5 rps | 45/100 | 45/200 | 0 | `NO_SAFE_CONFIG` |
| `extreme_rps` | 2/64 | 50 rps | 63/100 | 127/200 | 0 | `NO_SAFE_CONFIG` |

固定路由每组枚举 100 个频率候选；自动路由每组枚举 200 个路由/频率候选。
虽然每组都有满足 TTFT/TPOT 预测的候选，但没有任何候选同时通过 saturation
gate。因此两个搜索空间都得到：

```text
workload_count=4
gate_admitted=0 gate_rejected=4
decision_only=true
job_exit_rc=0
```

## 与无 gate 实测对照

| Workload | 无 gate 实际吞吐比 | 无 gate 实际饱和 | Gate 结果 |
|---|---:|---|---|
| `threshold_long_prompt` | 60.0% | 是 | 拒绝 |
| `long_decode` | 15.0% | 是 | 拒绝 |
| `moderate_burst` | 30.4% | 是 | 拒绝 |
| `extreme_rps` | 7.16% | 是 | 拒绝 |

在这四个样本上，gate 的拒绝结果与无 gate 的实际运行结果完全一致：四个被拒绝
的 workload 在 Job 250005 中都低于 95% 可持续吞吐阈值。尤其前三组在无 gate
时仍满足延迟 SLO，因此本次结果验证了 saturation gate 能阻止“延迟安全但容量
不安全”的放行。

## Slurm 与 GPU 风险

本次包装作业没有请求 GPU TRES，只在 Ganymede 上使用 1 个普通 task、2 个 CPU
和 4 GiB 内存执行调度决策。四个请求全部被 gate 拒绝后，runner 在以下操作前
直接成功退出：

- vLLM/proxy 启动；
- `srun` GPU step；
- `nvidia-smi -lgc` 或 `-rgc`；
- GPU telemetry；
- power gating、MIG、功率上限或 Slurm 管理操作。

Slurm stdout 明确记录：

```text
gpu_tres_requested=false
gpu_allocated=false
vllm_started=false
clock_control_requested=false
parent_reset_verified=not_required
```

stderr 为空。因此这次测试没有改变任何 GPU 状态，也没有对其他 Slurm 作业或
节点配置执行管理操作。

## 能力边界

本次实际验证了：

- 改变路由：固定路由与自动 L40S/L4 角色路由；
- 提高频率：候选集中枚举现有 DVFS 频率；
- 所有当前候选不安全时拒绝请求。

当前 scheduler 仍固定 TP=1，并固定为一个 prefill GPU 加一个 decode GPU；它
没有枚举额外活跃 GPU 数、TP bundle，也没有实现跨稳定窗口的 emergency
override。因此这里的 `NO_SAFE_CONFIG` 只表示“现有路由 × DVFS × TP=1 候选为空”，
还不能视为用户定义的完整多级调度策略已经穷尽所有资源候选。

## 结论

1. 相同四组边界负载在固定与自动路由下全部被 gate 正确阻断。
2. 自动路由将候选数从 100 扩展到 200，但 saturation-safe 候选仍为 0。
3. 结果与无 gate 实测的四次实际饱和完全吻合，没有出现错误放行。
4. 这次 Slurm 测试没有申请或控制 GPU，不存在锁频残留、power-gating 或调度管理风险。
5. 下一阶段应把 GPU 副本数、TP bundle 和 emergency override 纳入候选生成后，再
   用同一数据检查“真正过载”是否只在完整候选空间为空时触发。
