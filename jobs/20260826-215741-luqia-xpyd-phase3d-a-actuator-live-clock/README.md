# XPYD Phase 3D-A：逐端点调频与实时硬件频率验证

该任务在已经通过的 Phase 3C 真实 2P2D 底座上验证调频执行器。它只运行
Phase 3D-A，不运行反馈路由、动态闭环或模型预测代码。

## 物理范围

- Neptune：P0/P1，L40S GPU 0/1，TP=1；
- Io：D0/D1，L4 GPU 0/1，TP=1；
- vLLM 0.15.1 与 `P2pNcclConnector`；
- 缓存隔离在 `/data/users/chjing/vllm_job_work/<slurm_job_id>/`。

## 调频与验证

1. 外层 XPYD launcher 使用 `sudo nvidia-smi -lgc` 和 `sudo nvidia-smi -lmc`
   把四张卡设置到已经通过 Phase 3C 的安全 HIGH 状态。
2. Phase 3D-A 从硬件支持频率中选择 LOW/MID/HIGH，对 P0、P1、D0、D1
   分别执行 MID、LOW、HIGH 的独立向下/向上切换。
3. 每次动作都核对 GPU index、UUID、PCI bus ID，并轮询实际
   `clocks.current.graphics`，不能只依赖调频命令的返回码；同时读取所有 peer
   GPU，确认没有误调其他显卡。
4. 原始 Phase 3D-A 只包含 P0 MID serving window。本任务的 job-local patch
   增加 P0、P1、D0、D1 各自独立的 MID serving window，其他三张卡保持 HIGH，
   每个窗口复用 Phase 3C 的 SSE、token、路由、NVML 能量和 95% 时钟匹配审计。
5. 两个独立后台监控器在整个 Stage A 期间每 0.2 秒读取四张物理 GPU 的实际
   graphics/memory clocks、利用率、UUID 和 PCI bus ID。最终审计要求实时轨迹
   中观察到每个端点的 LOW/MID/HIGH，且每个端点在真实负载下观察到 MID。

显存频率在本实验中保持 Phase 3C 验证过的固定值：L40S 9001 MHz、L4
6251 MHz；LOW/MID/HIGH 切换针对 graphics clock。

## 主要输出

```text
results/<slurm_job_id>/
├── preflight.json
├── launcher/fixed_clock_evidence.log
├── live_clock_monitor/{neptune,io}.jsonl
├── phase3d_actuator_validation/<run-id>/
│   ├── capabilities.json
│   ├── actuator_actions.csv
│   ├── actuator_readback.csv
│   ├── actuator_audit.json
│   ├── actuator_summary.md
│   └── workload_windows/
└── phase3d_a_live_clock_audit.json
```

最终成功需要 XPYD 原生 `actuator_audit.json` 和独立实时频率审计同时通过。
任务结束时先恢复安全 HIGH，外层 launcher 随后恢复默认 graphics/memory clocks。

## 代码来源与边界

`source/` 基于 `/Users/lukeqian/code/vLLM_test` commit
`0eb8926f965cfd550f5bcee0095b563b1bb4e41e`。除输出路径配置和上述四端点 MID
serving-window 补强外，运行代码保持原样。

本实验只验证真实调频执行能力，不比较能耗策略，也不声称动态控制或最优性。

当前目录故意不包含 `READY`，不会自动提交。
