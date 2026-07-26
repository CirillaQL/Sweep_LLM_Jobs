# Automatic DVFS strict control for the uncapped scheduler pair

This is the strict automatic-hardware-DVFS control for scheduler Jobs 250155
and 250156. It replays the same byte-identical 12-window request trace and
keeps allocation, exclusive nodes, placement, model, SLOs, benchmark settings,
telemetry, energy integration, clock-mismatch handling, and the current shared
runner unchanged.

The scheduler is disabled and both GPUs use their native automatic clock
control. Placement remains fixed to L40S prefill on Neptune and L4 decode on
Ganymede. No per-window frequency recommendation or `nvidia-smi -lgc` command
is issued.

The modeled KV transfer-time term remains disabled, matching Jobs 250155 and
250156. Runtime vLLM PD inference still performs the real cross-node KV-cache
transfer, so measured TTFT and energy include that path.
