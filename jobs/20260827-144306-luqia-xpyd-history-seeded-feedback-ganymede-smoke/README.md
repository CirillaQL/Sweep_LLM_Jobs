# XPYD 历史引导 + 在线反馈闭环 smoke

该 Job 在 Neptune 2×L40S Prefill 与 Ganymede 2×L4 Decode 的真实 2P2D 底座上，验证
“从历史记录选择初始路由/频率组合，再用实时反馈修正”的控制链路。任务使用成功
Phase 3C Job 255143 缓存的本地 Mistral-7B-v0.1，不使用模型预测，也不会下载模型。

## 决策逻辑

1. 从最新有效 Phase 4A `phase4a_summary.json` 读取已审计的配置聚合记录；也可通过
   `XPYD_HISTORY_SUMMARY_OVERRIDE` 指定固定文件。
2. 按 input length、output length、request rate 的对数距离匹配历史 workload。
3. 只保留 TTFT/TPOT “均值 + 95% CI” 位于 90% SLO 安全线内的组合，再按
   `历史距离 → 能耗均值+95% CI → config ID` 排序。
4. 每个 workload 的第一个窗口使用历史组合；后续两个窗口复用现有 3D-B
   feedback scheduler，根据真实 TTFT/TPOT、queue、KV 和能耗逐档调整路由与频率。
5. 所有调频动作使用 `sudo nvidia-smi -lgc`，命令后读取实际 graphics clock；
   任务结束恢复四卡安全 HIGH。

当前 smoke 顺序为：

```text
small_light → prefill_heavy → small_light → decode_heavy → both_heavy → decode_heavy
```

每个状态运行 3 个窗口、每窗 2 个请求。相同 workload 再次出现时会重新查询只读
历史表，因此本实验验证历史初始化与窗口内在线修正，不声称跨 workload 泛化。

## 记录内容

- `history_decisions.csv`：候选集合、拒绝原因、历史距离、预期能耗/延迟、最终路由
  和四卡请求频率；在线反馈轮次同时记录 route evaluation 与 DVFS recommendation。
- `history_actual_outcomes.csv`：每个决策对应的真实请求窗口，包含能耗、TTFT、
  TPOT、SLO、实际路由、请求频率、实际平均频率和时钟匹配率。
- `raw/actuator_actions.csv`：每次 sudo 调频前后频率、命令状态和硬件读回。
- `raw/endpoint_telemetry.csv`：反馈控制器实际读取的端点运行数据。
- `live_clock_monitor/{neptune,ganymede}.jsonl`：独立于控制器、每 0.2 秒采样的实际频率。
- `history_live_validation.json`：把决策、运行窗口、Phase 3C 时钟审计和独立实时
  监控交叉验证后的最终结果。

## 前置证据与输出位置

Job 会在 chjing 的 broker jobs 目录中自动寻找最新有效 Phase 3D-A actuator audit。
Phase 4A summary 优先使用项目文档记录的
`/data/users/chjing/vLLM_test/results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json`，
不存在时再搜索 broker jobs；两类证据找不到都会 fail closed。运行缓存位于
`/data/users/chjing/vllm_job_work/<job_id>/`，结果位于本 Job 的
`results/<job_id>/`。

本目录包含最后创建的 `READY`，由 jobs broker 自动提交。
