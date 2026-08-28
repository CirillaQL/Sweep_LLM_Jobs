# XPYD Phase 3D-B r2：活跃路由 DVFS 与逐决策验证窗口

本任务针对 Slurm Job 255288 暴露的两个证据缺口进行最小修正：旧控制器要求四个
端点 telemetry 全部新鲜，单路由运行后会因未使用端点过期而全局 fallback；旧序列
最后还有一次没有后续负载的 `post_recovery` 决策，而且进入 moderate 窗口的降频
端点 P1 并不在所选 `P0→D1` 路由中。

新任务继续使用 Uranus 的 P0/P1（L40S）和 Ganymede 的 D0/D1（L4），加载本地
Mistral-7B-v0.1，以 Job 255287 的有效 Phase 3D-A 审计为前置证据。它仍是
feedback-only 控制，不使用预测模型、历史组合、oracle、RL 或 bandit。

## 控制修改

- SLO 改为 TTFT 500 ms、TPOT 200 ms；路由安全边界仍使用 0.9 safety fraction。
- freshness 由“所有端点必须新鲜”改为候选 P/D pair 局部判断；未使用端点过期不会
  否定仍有新鲜证据的安全路由。
- measured-safe 路由选出后，只对该路由上的活跃 P/D 端点计算并执行 DVFS；缺少
  安全候选时仍恢复全部经过验证的路由和 HIGH 频率。
- 删除无后续负载的最终控制决策。控制记录与 workload window 一一对应，因此每次
  路由/DVFS 决策之后必有真实请求窗口。
- 每次非 HOLD 动作继续通过 `sudo nvidia-smi -lgc` 执行，成功硬件读回后才开始
  dwell；结束时恢复安全 HIGH，launcher 随后恢复默认 graphics/memory clocks。

## 丰富后的请求结构

每种请求形状包含一个 observation window 和一个同形状 verification window：

| 形状 | Input/Output | 请求数 | Rate | 目的 |
|---|---:|---:|---:|---|
| short | 128/64 | 4+4 | 0.5 RPS | 短输入短输出 |
| prefill-heavy | 2048/64 | 4+4 | 1.0 RPS | 强化 TTFT/P-side 压力 |
| decode-heavy | 128/256 | 4+4 | 1.0 RPS | 强化 TPOT/D-side 压力 |
| balanced | 512/128 | 4+4 | 1.0 RPS | 中等混合形状 |

总计 8 个窗口、32 个请求，connector 最大并发仍为已验证的 1。`*_observe` 记录
形状切换后的反馈，随后控制决策直接作用于同形状的 `*_verify`；不同形状之间的首个
窗口仍保留为真实 transition evidence。

## 增强验收

除原生 `closed_loop_audit.json` 外，独立审计还必须证明：

- 8 次控制决策与 8 个实际窗口按 ID、顺序一一对应，不存在 dangling decision；
- 每个窗口的实际请求路由与决策文件完全一致；
- 每个 selected endpoint 都有真实完成的请求路由，并且其目标 graphics/memory
  clock 在对应服务窗口内被 0.2 秒独立 NVML 监控观察到；监控同时记录正利用率
  样本作为增强证据，但不以容易漏采的瞬时利用率作为唯一硬门槛；
- 每个控制器非 HOLD 动作都发生在下一窗口使用的 active route 上，并在下一窗口
  负载期间保持目标频率；
- 至少一个 `*_verify` 窗口包含 active-route 真实 DVFS；
- 所有 `*_verify` 窗口逐请求检查 TTFT≤500 ms、TPOT≤200 ms，并记录均值、最大值
  和违规请求数；
- 四个端点 identity 正确、无监控错误、最大采样间隔不超过 1 秒；
- 命令读回、请求/token/路由归因、能量窗口和最终恢复全部有效。

主要结果位于 `results/<job_id>/phase3d_closed_loop_smoke/<run-id>/`，增强审计为
`results/<job_id>/phase3d_b_live_clock_audit.json`。

该实验仍只验证 feedback-only 闭环的物理行为和 SLO 安全性，不与 Static/Oracle
比较，也不声称节能最优性。

本目录包含最后创建的 `READY`，由 jobs broker 自动提交。
