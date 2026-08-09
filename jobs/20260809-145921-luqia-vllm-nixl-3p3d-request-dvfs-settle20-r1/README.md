# NIXL request-level DVFS with a 20-second settle period

This rerun keeps the previous random asymmetric-TP topology and adds a
20-second stabilization period after every `nvidia-smi -lgc` operation.
GPU clock and power telemetry continues throughout the wait. The request is
forwarded only after both Prefill and Decode agents publish their final,
verified clock acknowledgements.

The original vLLM benchmark result keeps client-visible TTFT, including the
wait. `vllm_bench_detailed_excluding_dvfs_wait.json` contains a second TTFT
series with exactly 20 seconds removed from each successful request. Per-request
proxy records similarly retain both raw and wait-excluded TTFT values.

All runtime state and caches use
`/data/users/chjing/vllm_job_work/<job_id>` and are removed after GPU clocks
are reset at job exit.
