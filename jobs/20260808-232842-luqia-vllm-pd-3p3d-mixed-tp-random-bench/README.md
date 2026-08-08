# 3P3D mixed-TP random routing benchmark

该任务在两个节点各使用四张 GPU，验证混合 TP 实例之间的随机 P-D 调度：

- Uranus / L40S：`P0(TP=1)`、`P1(TP=1)`、`P2(TP=2)`；
- Ganymede / L4：`D0(TP=1)`、`D1(TP=1)`、`D2(TP=2)`；
- 每个请求独立地从 `P0/P1/P2` 和 `D0/D1/D2` 中随机选择，形成九种候选链路；
- KV 端口按实例前序 TP 数累加，使 TP=2 实例为两个 rank 保留连续端口；
- Registry 必须报告 Prefill TP 列表 `[1,1,2]` 和 Decode TP 列表 `[1,1,2]`。

默认使用 `mistralai/Mistral-7B-v0.1`，执行 128 个请求。集群端检查所有请求
成功、route count 连续、没有非法链路，并要求九种 P-D 组合全部至少出现一次。
运行缓存仍统一位于 `/data/users/chjing/vllm_job_work/<job_id>`，任务结束后删除。

该目录包含 `READY`，推送后由 broker worker 自动提交。
