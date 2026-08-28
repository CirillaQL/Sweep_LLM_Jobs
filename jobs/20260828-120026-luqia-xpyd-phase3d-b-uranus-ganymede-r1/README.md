# XPYD Phase 3D-B：feedback-only 路由与 DVFS 真实闭环

本任务在 Slurm Job 255287 已通过的 Phase 3D-A 执行器之上，运行项目原生
`Phase3DClosedLoopHarness`。P0/P1 位于 Uranus 的 L40S GPU 0/1，D0/D1
位于 Ganymede 的 L4 GPU 0/1。任务不使用历史模型、性能预测器、离线 oracle、
在线学习或模型预测启动器；所有控制决策只读取刚完成窗口的真实反馈。

## 前置证据

`accepted_phase3d_a_job255287.json` 是 Job 255287 的原生
`actuator_audit.json`，SHA-256 为
`6256a6629507249c8491aecc7a237f4b017efc20725f52943306f27534c669fd`。
预检要求该文件 `valid=true`、所有 hard gate 通过且摘要完全匹配，否则禁止启动
3D-B。

## 闭环链路

1. 使用本地 Mistral-7B-v0.1 snapshot，以离线模式启动持久 2P2D vLLM 0.15.1。
2. 启动时 telemetry 为空，控制器只开放四条已验证路由并保持所有端点 HIGH。
3. 依次运行 `light_before -> moderate -> light_after`，最大 connector 并发固定为 1。
4. 每个窗口结束后读取真实 queue/KV、EWMA TTFT/TPOT、功耗、能量、健康、频率和
   freshness，筛选安全兼容路由；安全候选按实测每请求能耗排序。
5. 对每个端点独立给出 HOLD/STEP_DOWN/STEP_UP/FALLBACK_MAX，并通过
   `sudo nvidia-smi -lgc` 执行。只有成功硬件读回才开始 dwell。
6. 原子 route-control 文件驱动下一窗口的真实请求路由；缺失、过期或不兼容状态
   均 fail closed。
7. 结束时四个端点恢复安全 HIGH，launcher 随后恢复默认 graphics/memory clocks。

控制阈值为 TTFT SLO 1000 ms、TPOT SLO 80 ms；低于 50% 且无 queue/KV 压力时
降一级，高于 80% 或出现压力时升一级，严重压力时恢复最高频率。

## 独立硬件验证

Uranus 与 Ganymede 各有一个独立 NVML 监控器，以 0.2 秒周期记录 GPU identity、
graphics/memory clock、利用率和时间戳。监控器通过 `srun` stdout 回传首轮成功
标记，避免跨节点新文件缓存造成错误超时。

任务结束后的增强审计要求：

- 原生 `closed_loop_audit.json` 和 Job 255287 前置审计均有效；
- 至少一轮 fresh feedback 选择单条 P/D 路由；
- 控制器至少执行一次非最终恢复 DVFS 动作；
- 每次动作命令成功、原生读回匹配，且独立实时轨迹在动作生效区间观察到目标
  graphics clock 与固定 memory clock；
- 四个端点 identity 正确、无采样错误、最大采样间隔不超过 1 秒；
- 四轮控制记录和 light/moderate/light 三个真实窗口全部完成。

## 主要输出

```text
results/<slurm_job_id>/
├── preflight.json
├── launcher/
├── live_clock_monitor/{uranus,ganymede}.jsonl
├── phase3d_closed_loop_smoke/<run-id>/
│   ├── control_iterations.csv
│   ├── requests.csv
│   ├── routes.csv
│   ├── endpoint_telemetry.csv
│   ├── dvfs_actions.csv
│   ├── energy_summary.csv
│   ├── closed_loop_audit.json
│   ├── closed_loop_summary.md
│   └── windows/
└── phase3d_b_live_clock_audit.json
```

本实验只验证 feedback-only 联合路由/DVFS 闭环在真实硬件上可以安全运行，不比较
Static/Oracle，也不声称节能最优性；正式策略对照属于 Phase 4B。

目录包含 `READY`，由 jobs broker 自动提交。
