# 最新 12 窗口在线 trace：saturation gate 性能与能耗 ablation

日期：2026-07-24

## 结论

在相同 12 窗口 trace、相同 L40S-prefill/L4-decode 放置、TP=1、相同
40 个频率候选和相同 `min-slo-violation` 过载策略下，saturation gate
只改变了 3/12 个窗口的配置选择。

- 开 gate 的窗口期 GPU 能耗为 51.050 kJ；关 gate 为 49.880 kJ。
  关 gate 少 2.29%，即开 gate 相对多消耗 2.35%。
- 在真正由 gate 改变配置的 3 个窗口中，开 gate 相对关 gate 多消耗
  6.79% 能量，同时降低了 `decode_shift` 和 `decode_peak` 的尾延迟；
  `mixed_pressure` 的延迟几乎不变。
- 开 gate 的双 SLO 通过数为 11/12，关 gate 为 10/12。不过多出的失败
  是 `long_prefill`，该窗口两边选择完全相同的 1755/570 MHz，说明这次
  通过数差异包含运行间波动，不能归因于 gate。
- 只看 gate 确实改变配置的三个窗口，两边都是 3/3 通过 SLO。因此本次
  trace 没有证明 gate 提升了 SLO 通过率；它证明的是 gate 在预测饱和
  风险出现时会切到更激进的频率配置，以能耗换取部分尾延迟改善。
- `prefill_peak` 在两边都没有可行 operating point，都执行了最小预测
  SLO violation 配置，且实际 TTFT 均违反 500 ms SLO。这符合“所有候选
  都不可行时选择 violation 最低配置”的约定。

## 对照定义与可比性

| 项目 | Gate on | Gate off |
|---|---:|---:|
| Slurm job | 250014 | 250025 |
| active policy | `latency_plus_saturation` | `latency_only` |
| trace | 同一份 12 窗口 CSV | 同一份 12 窗口 CSV |
| 放置 | L40S prefill / L4 decode | L40S prefill / L4 decode |
| TP | 1/1 | 1/1 |
| 频率上限 | L40S 2520 / L4 780 MHz | L40S 2520 / L4 780 MHz |
| 每窗口候选数 | 40 | 40 |
| overload action | `min-slo-violation` | `min-slo-violation` |
| 请求种子/生成方式 | vLLM finite-rate Poisson, seed 0 | 同左 |
| benchmark failures | 0 | 0 |
| GPU reset verified | 是 | 是 |
| Slurm elapsed | 686 s | 692 s |

核心 `scheduler.py`、模型 bundle 和 saturation bundle 在两次运行之间
没有变化。更强的复现检查是：gate-off 作业同时生成了两种策略的纯诊断
决策；其 12 份 `decision_gate` 和 12 份 `decision_latency` 分别与
gate-on 作业对应文件逐字节一致。因此候选空间和调度器输出完全复现，
只有哪一种决策被用于控制实际频率不同。

应保留一个实验设计限制：用户要求不再重跑 gate-on，所以 gate-on 使用
此前完整作业 250014；新 gate-off 作业 250025 使用 `--exclusive` 独占
两个节点，而 250014 没有显式独占声明。两次运行也不在同一时段。故这是
强控制的单因素对比，但不是同一批次、相邻串行运行的最高等级严格配对。

## 汇总结果

| 指标 | Gate on | Gate off | Gate off 相对变化 |
|---|---:|---:|---:|
| 成功请求 / 失败请求 | 332 / 0 | 332 / 0 | 相同 |
| 双 SLO 通过窗口 | 11/12 | 10/12 | -1 |
| TTFT SLO 通过窗口 | 11/12 | 10/12 | -1 |
| TPOT SLO 通过窗口 | 12/12 | 12/12 | 相同 |
| overload fallback | 6 | 3 | -3 |
| 平均窗口 p99 TTFT | 373.96 ms | 424.70 ms | +13.57% |
| 中位窗口 p99 TTFT | 384.26 ms | 424.09 ms | +10.36% |
| 最大窗口 p99 TTFT | 612.33 ms | 744.48 ms | +21.58% |
| 平均窗口 p99 TPOT | 66.00 ms | 66.98 ms | +1.48% |
| benchmark 总时长 | 211.50 s | 214.52 s | +1.43% |
| 窗口期加权平均 GPU 功率 | 108.29 W | 104.71 W | -3.30% |
| 窗口期 GPU 能量 | 51.050 kJ | 49.880 kJ | -2.29% |
| L40S 窗口期能量 | 31.888 kJ | 30.607 kJ | -4.02% |
| L4 窗口期能量 | 19.162 kJ | 19.273 kJ | +0.58% |
| 窗口期能量 / 成功请求 | 153.77 J | 150.24 J | -2.29% |
| 全遥测区间 GPU 能量（不含 reset probe） | 69.913 kJ | 67.654 kJ | -3.23% |
| 全记录 GPU 能量（含 reset probe） | 71.736 kJ | 69.496 kJ | -3.12% |

窗口期能量是从每个 `workload_start` 到 `workload_end` 的两张 GPU 板级
功率积分，包含该窗口的频率确认和 benchmark。全遥测能量还包含模型启动、
窗口间调度和清理阶段。它们都不包含 CPU、内存、NIC 和风扇。

## 逐窗口结果

频率列格式为 `L40S prefill / L4 decode`。

| # | Workload | Gate on 状态与 MHz | Gate off 状态与 MHz | p99 TTFT on/off (ms) | p99 TPOT on/off (ms) | 双 SLO on/off | 能量 on/off (J) |
|---:|---|---|---|---:|---:|---|---:|
| 1 | steady_start | OK, 480/360 | OK, 480/360 | 280.99 / 279.99 | 64.80 / 64.73 | 是 / 是 | 3078.20 / 2849.76 |
| 2 | short_ramp | OK, 210/360 | OK, 210/360 | 412.03 / 434.02 | 69.31 / 69.33 | 是 / 是 | 3463.21 / 3423.63 |
| 3 | prefill_shift | OK, 480/570 | OK, 480/570 | 433.72 / 472.62 | 65.08 / 65.18 | 是 / 是 | 3201.04 / 3293.83 |
| 4 | decode_shift | fallback, 2265/780 | OK, 210/360 | 292.17 / 361.67 | 64.39 / 73.39 | 是 / 是 | 5157.15 / 4766.99 |
| 5 | sudden_burst | fallback, 1755/780 | fallback, 1755/780 | 405.65 / 474.85 | 66.63 / 66.82 | 是 / 是 | 3446.40 / 3451.91 |
| 6 | short_recovery | OK, 210/360 | OK, 210/360 | 362.87 / 451.49 | 66.30 / 66.25 | 是 / 是 | 3321.68 / 3298.96 |
| 7 | long_prefill | fallback, 1755/570 | fallback, 1755/570 | 485.07 / 621.74 | 68.33 / 68.23 | 是 / 否 | 3359.31 / 3551.41 |
| 8 | mixed_pressure | fallback, 1755/780 | OK, 480/780 | 413.70 / 414.15 | 66.51 / 66.47 | 是 / 是 | 5189.10 / 5023.29 |
| 9 | quiet_window | OK, 480/360 | OK, 480/360 | 262.33 / 273.52 | 64.75 / 64.79 | 是 / 是 | 3058.66 / 3053.82 |
| 10 | prefill_peak | fallback, 2520/780 | fallback, 2520/780 | 612.33 / 744.48 | 68.73 / 68.79 | 否 / 否 | 3898.28 / 3980.11 |
| 11 | decode_peak | fallback, 2265/780 | OK, 210/570 | 248.92 / 280.57 | 64.63 / 67.24 | 是 / 是 | 10671.67 / 9891.77 |
| 12 | final_recovery | OK, 210/570 | OK, 210/570 | 277.75 / 287.32 | 62.53 / 62.48 | 是 / 是 | 3205.63 / 3294.73 |

## Gate 真正改变决策的三个窗口

| Workload | Gate 动作 | Gate on 相对 gate off 的效果 |
|---|---|---|
| decode_shift | 210/360 → 2265/780 | TTFT -19.22%，TPOT -12.26%，能量 +8.18%，平均功率 +12.44% |
| mixed_pressure | 480/780 → 1755/780 | TTFT -0.11%，TPOT +0.06%，能量 +3.30%，平均功率 +3.83% |
| decode_peak | 210/570 → 2265/780 | TTFT -11.28%，TPOT -3.88%，能量 +7.88%，平均功率 +9.51% |

这三个窗口在 gate 规则下都没有 `num_safe > 0` 的候选，因此 gate-on
进入的是 overload fallback，而不是找到一个同时满足 saturation gate
的新 operating point。当前固定放置、TP=1 的候选空间只允许调频；本次
ablation 没有覆盖改路由、增加活跃 GPU 或调整 TP。

## 频率与测量注意事项

- 24 个 GPU-窗口频率确认中，23 个准确命中目标。
- `prefill_peak` 的 L40S 目标为 2520 MHz，但 gate-off 的主动负载探针
  平均只观测到约 2246 MHz，记录为 `clock_target_not_sustained` 后继续。
  gate-on 的窗口遥测平均约 2516.5 MHz。该硬件差异会影响该窗口的绝对
  TTFT，不能将其完全归因于 gate。
- vLLM 0.15.1 不支持本 trace 中的 burstiness 参数；两边实际都使用同一
  finite-rate Poisson 请求生成方式和 seed 0。
- 短 finite-Poisson 窗口的 `successful_requests / request_rate / duration`
  不能可靠判定饱和。旧 gate-on 汇总将其标为 saturated 是此前已确认的
  统计口径缺陷；本报告只用实际 TTFT/TPOT SLO、请求成功数及功率能耗下
  结论，不使用该饱和标签。

## 最终判断

这组 trace 上，saturation gate 的主要作用不是提高 SLO 通过窗口数，而是
在三个预测饱和风险窗口中选择更高频率的 overload fallback。它带来可测
的 TTFT 改善，但增加约 6.79% 的相关窗口能耗；全 trace 的窗口期能耗增加
约 2.35%。在 `mixed_pressure` 上，额外频率几乎没有带来延迟收益，说明
当前 fallback 排序仍有优化空间。

若论文需要最高等级的因果证据，下一步应在不复用旧结果的前提下，让
gate-on 与 gate-off 都使用 `--exclusive`，由依赖关系显式串行提交，并
重复多个随机种子。按照用户本轮指示，本次没有再执行 gate-on。
