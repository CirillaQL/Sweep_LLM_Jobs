# 2P2D TP=1 random P-D benchmark retry r3

该任务验证以下修改：

- `nccl_num_channels` 以字符串传给 vLLM 0.15.1，避免 P2P NCCL 线程因
  `TypeError: str expected, not int` 退出；
- 每个请求分别生成 Prefill 和 Decode 的 `0/1` 随机位，并组合成
  `P0/P1 -> D0/D1` 链路；
- Proxy 日志记录每个请求实际选择的 P-D 链路；
- 集群端校验 32 个请求全部出现、编号连续且链路属于四种合法组合，并输出
  四种链路及 P/D 节点的实际计数。

拓扑保持为 Uranus 上两个 TP=1 L40S Prefill 实例，以及 Ganymede 上两个
TP=1 L4 Decode 实例。默认模型为 `mistralai/Mistral-7B-v0.1`。

该目录包含 `READY`，推送后由 broker worker 提交。
