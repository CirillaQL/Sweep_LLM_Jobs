# 3P3D compatible-TP random routing benchmark retry r2

该任务修复混合 TP 拓扑中的非对称 TP 链路：

- Uranus / L40S：`P0(TP=1)`、`P1(TP=1)`、`P2(TP=2)`；
- Ganymede / L4：`D0(TP=1)`、`D1(TP=1)`、`D2(TP=2)`；
- 每个请求先从 `P0/P1/P2` 中随机选择 Prefill；
- 选到 `P0` 或 `P1` 后，仅在 `D0/D1` 中随机选择；
- 选到 `P2` 后，只能选择 `D2`。

合法链路只有五条：`P0-D0`、`P0-D1`、`P1-D0`、`P1-D1`、`P2-D2`。
Proxy 的 Registry 也只发布这五条 `supported_routes`，显式请求非对称 TP
链路会被拒绝。

默认使用 `mistralai/Mistral-7B-v0.1`，执行 128 个请求，并要求五条合法链路
全部出现。运行缓存统一位于
`/data/users/chjing/vllm_job_work/<job_id>`，任务结束时删除。

该目录包含 `READY`，推送后由 broker worker 自动提交。
