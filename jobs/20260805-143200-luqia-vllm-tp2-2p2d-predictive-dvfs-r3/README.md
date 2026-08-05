# TP2 2P/2D predictive DVFS validation, retry 3

This retries Job 251808 after the second Prefill replica registered before its
HTTP server was ready and then hung during vLLM EngineCore initialization.

The retry keeps the same experiment topology and policy:

- Neptune: two Prefill instances, each TP=2 on two L40S GPUs.
- Ganymede: two Decode instances, each TP=2 on two L4 GPUs.
- Model: `mistralai/Mistral-7B-v0.1`.
- Scheduler: minimum-power frequency predicted to satisfy P99 TTFT <= 500 ms
  and P99 TPOT <= 200 ms.
- Manual workload clock control with `sudo nvidia-smi -lgc`.

Startup is hardened in three ways. Additional replicas start sequentially with
Decode ready before Prefill. Model initialization runs at each GPU's maximum
graphics clock and switches to the scheduler-selected clock before workloads.
Per-instance telemetry starts only after that instance's HTTP API is ready, to
avoid extra NVML pressure during initialization. The runner now requires both
registry membership and a successful `/v1/models` response from every one of
the four instances before sending the smoke request or benchmark traffic.
