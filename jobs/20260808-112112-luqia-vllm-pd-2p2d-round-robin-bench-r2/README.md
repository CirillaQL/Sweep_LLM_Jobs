# 2P2D TP=1 round-robin vLLM bench retry r2

这是失败任务 `252388` 的完整缓存路径修复重试。拓扑、benchmark 参数和轮询
顺序保持不变：`P0-D0 -> P0-D1 -> P1-D0 -> P1-D1`。

本次修复参考 2026-08-07 的 jobs：

- 默认模型仍为 `mistralai/Mistral-7B-v0.1`；
- Hugging Face、XDG、FlashInfer、vLLM、Torch/TorchInductor、Triton、CUDA、
  Numba、Ray 及临时文件全部放到
  `/data/users/chjing/vllm_job_work/<job_id>`；
- 任务成功、失败或收到终止信号时均停止运行实例，并安全删除对应的 job 工作目录；
- 结果和持久日志保留在本目录的 `results/<job_id>/` 下。

该目录包含 `READY`，推送后由 broker worker 提交。
