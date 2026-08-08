# 2P2D TP=1 round-robin vLLM bench

这个待提交任务直接使用仓库 `script/` 下的 PD 启动器、单实例启动脚本和
`scheduler_custom_policy.py`，验证以下固定拓扑：

- Uranus：`Prefill_0`、`Prefill_1`，每个实例使用一张 L40S，TP=1；
- Ganymede：`Decode_0`、`Decode_1`，每个实例使用一张 L4，TP=1；
- Proxy：Ganymede；
- 请求状态轮询：`P0-D0 -> P0-D1 -> P1-D0 -> P1-D1`，然后重复。

默认执行 32 个 random-dataset 请求，输入长度 512、输出长度 128、请求率
4 req/s、最大并发 16。可在提交时通过 `--export` 覆盖这些参数。
默认模型为 `mistralai/Mistral-7B-v0.1`。运行期脚本以及 Hugging Face、
vLLM、Torch、Triton、CUDA 缓存统一写入
`/data/users/chjing/vllm_job_work/<job_id>`，不会再读取 `/root/.cache`。

```bash
sbatch \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
```

例如覆盖请求数和请求率：

```bash
sbatch --export=ALL,NUM_PROMPTS_OVERRIDE=64,REQUEST_RATE_OVERRIDE=8 \
  /data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260808-104810-luqia-vllm-pd-2p2d-round-robin-bench/run.sbatch
```

结果写入 `results/<slurm_job_id>/`：

- `vllm_bench.txt`：vLLM bench 控制台结果；
- `vllm_bench_detailed.json`：请求级详细结果；
- `registry.json`：两个 Prefill 和两个 Decode 的注册信息；
- `pd_runtime/proxy.log`：每个请求实际选择的 P/D 地址；
- `round_robin_check.json`：轮询顺序、请求数和四条路线计数的自动检查结果；
- `pd_runtime/Prefill_0.log` 等：四个 vLLM 实例日志。

任务会在 benchmark 后检查成功/失败请求数，并逐条验证 Proxy 的 route count
是否严格符合四状态循环。任一请求失败、请求数不符或路由顺序错误都会让任务
以非零状态退出。该脚本只创建完成，尚未提交到 Slurm。
