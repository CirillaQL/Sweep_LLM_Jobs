# Exact forwarding TTFT scheduler rerun

This job repeats the 3P3D NIXL predictive-DVFS workload with Prefill and
Decode TP sizes `[1,1,2]`, random routing across all nine P-D pairs, and a
20-second clock stabilization period.

The primary TTFT metric no longer subtracts 20 seconds from client-visible
TTFT. The proxy starts an independent monotonic timer only after both Prefill
and Decode clock agents have applied `nvidia-smi -lgc`, completed the full
settle interval, verified the observed clocks, and published their ACKs. It
stops that timer at the first Decode response chunk.

Primary outputs:

- `forwarding_ttft_summary.json`: exact aggregate, route, and topology metrics
- `forwarding_ttft_requests.csv`: exact per-request timings and predictions
- `request_dvfs_check.json`: includes forwarding-TTFT SLO pass counts
- `pd_runtime/request_dvfs_decisions.jsonl`: raw timestamps and stage timings

The legacy fixed-subtraction result remains for comparison only. All runtime
state and caches use `/data/users/chjing/vllm_job_work/<job_id>` and are
removed at job exit.
