# NIXL 3P3D 模型预测调频任务

这个任务在两台节点上运行一个 3 Prefill + 3 Decode 的 vLLM PD 集群，
使用 NIXL/UCX 传输 KV Cache，并在每个请求转发前调用性能模型选择满足
TTFT/TPOT SLO 的最低功耗 Prefill/Decode GPU 频率组合。

## 整体数据流

```text
客户端请求
   |
   v
PD 调度代理
   |-- 1. 从 3 个 Prefill 和 3 个 Decode 中选择 P-D 路由
   |-- 2. 根据输入/输出长度、请求率、GPU 类型和 TP 计算候选频率
   |-- 3. 选择满足 TTFT/TPOT 与饱和度约束的最低功耗频率组合
   |-- 4. 向两个实例的时钟代理下发频率，并等待校验 ACK
   |-- 5. Prefill 计算并通过 NIXL/UCX 传输 KV Cache
   `-- 6. Decode 继续生成并把流式结果返回客户端
```

## 文件与功能

- `run.sbatch`：独立任务入口。直接完成资源声明、3P3D/NIXL/DVFS 配置、
  集群启动、注册等待、全路由预热、压测、结果校验和安全清理。
- `validate_job_config.py`：提交前的配置守卫。验证任务确实是 3P3D、使用
  NIXL、开启模型预测调频，并检查 TP、SLO、带宽和 warmup 参数。
- `check_random_routes.py`：检查正式请求编号是否连续、路由是否属于合法的
  3x3 P-D 路由集合，并输出机器可读报告。
- `READY`：供任务代理识别该目录已经准备好。
- 公共 `script/start_pd_vllm.sh`：在两台节点上分配 GPU 并启动 6 个实例、
  调度代理和时钟代理。
- 公共 `script/start_pd_vllm_instance.sh`：构造 `NixlConnector` 配置并启动
  单个 Prefill 或 Decode vLLM 实例。
- 公共 `script/scheduler_custom_policy.py`：实现 P-D 路由、模型预测、调频握手、
  NIXL 两阶段转发及逐请求决策日志。
- 公共 `script/request_dvfs_predictor.py`：枚举 P/D 频率组合，预测 TTFT、TPOT、
  饱和风险和功耗，选择满足 SLO 的最低功耗组合。
- 公共 `script/pd_clock_agent.py`：执行 `nvidia-smi -lgc`、等待稳定、校验实测
  频率并持续记录 GPU 遥测。

## 核心配置

| 项目 | 配置 |
|---|---|
| 模型 | `mistralai/Mistral-7B-v0.1` |
| Prefill | `uranus`，3 个实例，L40S，TP=`[1,1,2]` |
| Decode | `ganymede`，3 个实例，L4，TP=`[1,1,2]` |
| P-D 路由 | 允许非对称 TP，共 9 条路由，固定种子随机选择 |
| KV 连接器 | `NixlConnector`，失败策略 `fail` |
| NIXL 后端 | UCX，`UCX_TLS=all`，绑定节点的 100GbE 数据网卡 |
| KV 有效带宽模型 | `4.3063 Gbit/s`，来自 Job 252766 的实测值 |
| 调频策略 | 每请求预测，满足 SLO 后最小化预测功耗 |
| SLO | TTFT 500 ms，TPOT 200 ms |
| 稳频 | 20 秒，频率容差 ±30 MHz，超时 60 秒 |
| Slurm 时间上限 | 30 分钟 |
| 测量负载 | 36 请求，2 RPS，输入 512，输出 128，最大并发 16 |
| 预热 | 对全部 9 条 P-D 路由各发送一个同形状请求 |

## 运行与输出

在 broker 根目录提交：

```bash
sbatch jobs/20260819-210313-luqia-vllm-nixl-3p3d-predictive-dvfs/run.sbatch
```

结果写入 `results/<slurm_job_id>/`。重点查看：

- `vllm_bench.txt`：压测汇总；
- `forwarding_ttft_summary.json`：排除调频等待后的精确 TTFT；
- `request_dvfs_check.json`：预测决策和实测频率校验；
- `random_route_check.json`：3P3D 路由覆盖与合法性；
- `pd_runtime/request_dvfs_decisions.jsonl`：逐请求预测与调频明细；
- `pd_runtime/gpu_telemetry_*.csv`：每个实例的频率、功耗和利用率。
