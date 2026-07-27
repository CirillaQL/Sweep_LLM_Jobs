# KV-aware no-gate saturation stress trace

This job runs the online `latency_only` scheduler without the saturation gate
on seven workloads selected to expose capacity saturation and increasingly
heavy prompt pressure.

The first four workloads come from the completed no-gate boundary calibration
Job 250005. In that run, latency-only scheduling admitted every workload, but
measured achieved/configured throughput ratios were 0.600, 0.150, 0.304, and
0.0716, respectively. The first three still met both latency SLOs, making them
direct examples of latency-safe but capacity-unsafe admission.

Three additional stress points increase prompt length and request rate:

- 1024 input / 128 output at 5 requests/s;
- 2048 input / 128 output at 1 request/s;
- 2048 input / 128 output at 2 requests/s.

The scheduler uses fixed L40S-prefill/L4-decode placement, TP=1, TTFT/TPOT
SLOs of 500/200 ms, and minimum-SLO-violation overload fallback. No manual L4
candidate-frequency cap is applied; the L40S cap remains 2520 MHz.

Predicted TTFT includes the same cross-node KV-cache transfer model used by
Jobs 250159 and 250160: 131072 KV bytes per token, measured effective
bandwidth 24.076 Gbit/s, and zero dispatch time.

The 50 requests/s workload sends 500 prompts, matching the original saturation
calibration instead of being truncated by the newer 100-prompt default.
