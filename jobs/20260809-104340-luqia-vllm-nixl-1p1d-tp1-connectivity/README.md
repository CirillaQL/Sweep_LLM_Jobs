# NixlConnector 1P1D TP=1 connectivity test

该任务用最小拓扑验证跨节点 NIXL KV 传输：

- Uranus：一个 L40S `P0(TP=1)`；
- Ganymede：一个 L4 `D0(TP=1)`；
- KV Connector：`NixlConnector`；
- 唯一合法链路：`P0-D0`。

任务使用默认模型 `mistralai/Mistral-7B-v0.1`，发送 8 个请求（输入 512
tokens、输出 64 tokens）。通过条件包括：两个实例完成 HTTP 注册、Registry
报告 `NixlConnector`、8 个请求全部成功、Proxy 日志只出现 `P0-D0`。

所有 Hugging Face、vLLM、Torch、Triton、CUDA、NIXL/UCX 相关运行缓存和临时
文件均位于 `/data/users/chjing/vllm_job_work/<job_id>`，任务退出时删除。

该目录包含 `READY`，推送后由 broker worker 自动提交。
