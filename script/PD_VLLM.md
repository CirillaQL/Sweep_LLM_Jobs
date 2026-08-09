# vLLM PD 分离启动脚本

这套脚本把历史 job 中的 PD 启动逻辑整理为可复用入口，默认拓扑为：

- Uranus：两个独立的 L40S、TP=1 Prefill 实例，即 `Prefill_0`、`Prefill_1`；
- Ganymede：两个独立的 L4、TP=1 Decode 实例，即 `Decode_0`、`Decode_1`；
- Ganymede：一个自定义 PD Proxy；
- 状态轮询顺序：P0-D0、P0-D1、P1-D0、P1-D1。

## 文件职责

- `pd_vllm_job.sbatch`：申请节点和 GPU，并进入总启动器；
- `start_pd_vllm.sh`：分配端口，并用互斥 Slurm GPU step 分别启动四个 vLLM；
- `start_pd_vllm_instance.sh`：校验 GPU 型号、构造 NixlConnector 配置并启动单个 vLLM；
- `scheduler_custom_policy.py`：执行请求级 P/D 实例对调度；
- `pd_vllm.env.example`：集中管理节点、TP、模型和 vLLM 参数。

## 启动

将 `script/` 同步到集群的
`/data/users/chjing/Sweep_LLM_Jobs_broker/script/` 后执行：

```bash
sbatch /data/users/chjing/Sweep_LLM_Jobs_broker/script/pd_vllm_job.sbatch
```

使用另一份配置时：

```bash
PD_CONFIG_FILE=/data/users/chjing/my_pd.env \
  sbatch /data/users/chjing/Sweep_LLM_Jobs_broker/script/pd_vllm_job.sbatch
```

当前启动器固定要求 `PREFILL_REPLICAS=2`、`DECODE_REPLICAS=2` 且两侧
`TP_SIZE=1`。Slurm 在 Uranus 和 Ganymede 上各申请两张 GPU；四个实例使用
互斥 GPU step，因此不会共享同一张卡。实例日志分别为
`Prefill_0.log`、`Prefill_1.log`、`Decode_0.log`、`Decode_1.log`。
启动 Proxy 和 vLLM 前，Slurm 主日志会输出四条 `pd_instance_map`，明确记录
实例名、调度别名、节点名、节点 IP、HTTP/KV 端口、GPU 型号和 TP。例如：

```text
pd_instance_map instance=Prefill_0 alias=P0 role=prefill node=uranus node_ip=10.1.0.5 http_endpoint=10.1.0.5:PORT kv_endpoint=10.1.0.5:PORT gpu=L40S tp=1
```

## 指定请求路由

通过请求头指定：

```bash
curl -H 'X-PD-Route: P1->D0' \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistralai/Mistral-7B-v0.1","prompt":"hello","max_tokens":32}' \
  http://10.1.0.3:PORT/v1/completions
```

也可使用查询参数 `?pd_route=P0-D1`，或在 JSON 中加入
`"pd_route":"P0-D1"`。JSON 中的调度字段会在转发给 vLLM 前删除。

运行日志保存在 `PD_OUT_DIR`。运行期脚本复制到
`/data/users/chjing/vllm_job_work/<job_id>`；Hugging Face、vLLM、Torch、
Triton 和 CUDA 缓存也统一放在该目录下，所有进程退出后自动删除。
