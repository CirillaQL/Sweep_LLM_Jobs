# 2P2D TP=1 random-route vLLM bench

这个待提交任务直接使用仓库 `script/` 下的 PD 启动器、单实例启动脚本和
`scheduler_custom_policy.py`，验证以下固定拓扑：

- Uranus：`Prefill_0`、`Prefill_1`，每个实例使用一张 L40S，TP=1；
- Ganymede：`Decode_0`、`Decode_1`，每个实例使用一张 L4，TP=1；
- Proxy：Ganymede；
- KV Connector：`NixlConnector`，通过 UCX 和每实例独立 side-channel 传输；
- 请求默认随机选择链路：分别生成 Prefill 和 Decode 的 `0/1` 随机位，组合为
  `P0/P1 -> D0/D1`；仍可通过请求参数显式指定固定链路。

Proxy 在 Prefill 完成后读取响应中的 `kv_transfer_params`，并将其传给选中的
Decode。启动器会在实例通过 `/v1/models` 健康检查后将 HTTP、side-channel 和
TP 信息注册到 Proxy。

默认执行 32 个 random-dataset 请求，输入长度 512、输出长度 128、请求率
4 req/s、最大并发 16。可在提交时通过 `--export` 覆盖这些参数。
默认模型为 `mistralai/Mistral-7B-v0.1`。运行期脚本以及 Hugging Face、
vLLM、Torch、Triton、CUDA 缓存统一写入
`/data/users/chjing/vllm_job_work/<job_id>`，包括 Hugging Face、XDG、
FlashInfer、vLLM、Torch/TorchInductor、Triton、CUDA、Numba、Ray 和临时文件，
不会再读取 `/root/.cache`。任务无论成功、失败或收到终止信号，都会停止运行实例
并删除对应的 `/data/users/chjing/vllm_job_work/<job_id>` 目录。

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
- `random_route_check.json`：随机链路合法性、请求数和四条路线计数；
- `pd_runtime/Prefill_0.log` 等：四个 vLLM 实例日志。

任务会在 benchmark 后检查成功/失败请求数，并逐条验证 Proxy 的 route count
是否连续且所有随机链路均属于四个合法组合。任一请求失败、请求数不符或出现
非法链路都会让任务以非零状态退出。
