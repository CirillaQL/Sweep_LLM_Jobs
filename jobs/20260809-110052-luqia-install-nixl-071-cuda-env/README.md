# Install NIXL 0.7.1 into cuda-env

该维护任务直接修改共享环境：

`/data/users/chjing/miniforge3/envs/cuda-env`

根据 vLLM 0.15.1 的 `requirements/kv_connectors.txt`，NIXL 最低版本为
0.7.1。现有环境使用 Python 3.10 和 CUDA 12.8，因此任务安装固定版本
`nixl[cu12]==0.7.1`，避免自动升级到 NIXL 1.x。

安装完成后，任务会在 Ganymede 和 Uranus 上分别检查：

- `nixl` 与 `nixl-cu12` 包版本；
- `nixl._api.nixl_agent` 和 `nixl_agent_config`；
- `nixl._bindings.nixlXferTelemetry`。

pip 下载缓存和临时文件位于
`/data/users/chjing/vllm_job_work/<job_id>`，任务退出时删除；安装日志保存在
`results/<job_id>/install_nixl.log`。

该目录包含 `READY`，推送后由 broker worker 自动提交。
