# XPYD Phase 3C minimal physical reproduction

这个任务直接封装 `vLLM_test` 当前 XPYD 代码，而不是调用
`Sweep_LLM_Jobs/script/` 下的旧预测调频启动器。

## 复现范围

任务复现 XPYD Phase 3C 的真实多端点基础设施验证：

- Neptune：P0、P1，分别使用一张 L40S，TP=1；
- Io：D0、D1，分别使用一张 L4，TP=1；
- vLLM 0.15.1 `P2pNcclConnector`；
- 四条明确允许的 `P0/P1 -> D0/D1` 路由；
- `small=(IL128, OL128)` 与 `prefill_heavy=(IL2048, OL128)`；
- 每条路由承载两种请求形状；
- 真实 SSE、请求/路由审计、Prometheus 指标、逐 GPU NVML 能量和固定频率证据。

这是最小物理复现，不运行 Phase 4A oracle、Phase 4B 策略比较或动态反馈实验。
Phase 3C 通过后，后续 job 才应在它的审计结果上继续运行 Phase 3D/4A/4B。

## 代码来源

`source/` 是从 `/Users/lukeqian/code/vLLM_test` 提取的自包含运行快照：

- source commit：`0eb8926f965cfd550f5bcee0095b563b1bb4e41e`
- `source/run_disagg_benchmark.sh`：XPYD 原生 vLLM/P2P 生命周期；
- `source/gpu_monitor.py`：原始 GPU 监控入口；
- `source/paper/scripts/xpyd/`：完整 XPYD Python 包；
- `source/paper/scripts/replay_synthetic_trace.py`：原始流式请求回放器；
- `source/paper/configs/xpyd_phase3c_2p2d_l40s_l4.json`：Phase 3C 配置，
  仅把输出根目录改为 job 注入的 `$XPYD_PHASE3C_OUTPUT_ROOT`。

保留完整 `xpyd/` 包是为了避免手工裁剪产生隐式 import 或版本漂移；实际 Phase 3C
入口仍然是 `xpyd.phase3c_substrate`。

## 文件

- `run.sbatch`：参考过往 jobs 的 Slurm 外壳，但调用 vendored XPYD launcher；
- `validate_job.py`：占用 GPU 后、启动 vLLM 前检查代码、配置、Python 依赖和版本；
- `source/`：本次复现使用的 XPYD 源码快照。

## 结果与验收

结果写入：

```text
jobs/<job-name>/results/<slurm_job_id>/
```

其中：

- `launcher/`：原始 launcher、server、proxy 与固定频率日志；
- `xpyd_phase3c_substrate/<run-id>/`：`summary.md/json`、`requests.csv`、
  `routes.csv`、`endpoint_summary.csv`、`energy_summary.csv`、`audit.json` 及 raw 证据；
- `preflight.json`：提交环境和源码边界检查。

最终成功以 XPYD Phase 3C 自身 audit 为准：四端点、四路由、真实 SSE、精确 token、
固定频率、四板能量窗口、无 GPU 重叠、无 thermal/hardware slowdown 均须通过。

## 提交门

当前目录故意不包含 `READY`。因此 broker 不会自动提交任务。创建 `READY`、Git
commit/push 或手工运行 `sbatch` 都是独立的后续授权。

