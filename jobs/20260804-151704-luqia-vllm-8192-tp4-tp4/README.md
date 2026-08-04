# Fixed TP=4/TP=4 8192-token latency test

这个任务在压测开始前固定启动两个 vLLM 服务：

- Neptune：一个 Prefill 服务，使用 4 张 L40S，`tensor-parallel-size=4`；
- Ganymede：一个 Decode 服务，使用 4 张 L4，`tensor-parallel-size=4`。

服务使用 16384-token 上下文，四个测试的输入长度均为 8192，覆盖
128-token 输出下的 1/2/4 RPS，以及 512-token 输出下的 1 RPS。两个服务
使用 GPU 自动 DVFS；本任务不执行实例级自动扩缩容。

默认模型改回 `mistralai/Mistral-7B-v0.1`。Prefill 和 Decode 都使用
TP=4，因此 KV cache 的分片布局一致，不再触发异构 TP 的 KV-head 限制。

提交命令：

```bash
sbatch jobs/20260804-151704-luqia-vllm-8192-tp4-tp4/run.sbatch
```

结果写入 `fixed_tp4_tp4_8192_results/`。除原有 `live_summary.json/csv`、GPU
遥测和能耗文件外，任务结束时会运行 `check_latency_metrics.py` 并生成：

- `latency_metrics.md`：便于直接阅读的 TTFT/TPOT/ITL/SLO 表格；
- `latency_metrics.json`：完整机器可读检查结果；
- `latency_metrics.csv`：每个 workload 一行的延迟和吞吐指标。

检查器会把 workload 缺失、非 8192 输入、请求失败、指标缺失、或者不是
Prefill TP=4 / Decode TP=4 拓扑视为任务失败。TTFT 500 ms / TPOT 200 ms
的 SLO 违反会记录在报告中，但默认不会让实验任务失败，因为 SLO 违反本身
也可能是有效测量结果。
