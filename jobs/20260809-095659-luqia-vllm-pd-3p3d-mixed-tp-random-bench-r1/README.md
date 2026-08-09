# 3P3D mixed-TP random routing benchmark retry r1

这是失败任务 `252544` 的注册元数据修复重试。

- 实际实例保持为 `P0/P1=[TP1]`、`P2=TP2` 和 `D0/D1=[TP1]`、`D2=TP2`；
- Proxy 根据 HTTP 端口偏移和 `TP_SIZES=1,1,2` 校正注册的 TP 信息；
- Registry 必须报告 Prefill 和 Decode TP 列表均为 `[1,1,2]`；
- 执行 128 个请求，在 3×3 的九种 P-D 链路间随机调度，并要求九种组合全部出现。

该目录包含 `READY`，推送后由 broker worker 自动提交。
