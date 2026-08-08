# 2P2D TP=1 round-robin vLLM bench retry r1

这是失败任务 `252387` 的缓存路径修复重试。拓扑、benchmark 参数和轮询顺序
保持不变：`P0-D0 -> P0-D1 -> P1-D0 -> P1-D1`。

修复内容：

- 默认模型为 `mistralai/Mistral-7B-v0.1`；
- 运行期脚本放在 `/data/users/chjing/vllm_job_work/<job_id>`；
- Hugging Face、vLLM、Torch、Triton 和 CUDA 缓存均放在同一 job 目录；
- `HF_TOKEN_PATH` 不再指向无权限的 `/root/.cache/huggingface/token`。

该目录通过 wrapper 复用原任务的 benchmark 和轮询验证逻辑，结果写入本重试
目录下的 `results/<job_id>/`。
