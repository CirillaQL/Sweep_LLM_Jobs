# Latency-only KV-aware trace with uncapped L4 frequency search

This job is the no-saturation-gate counterpart to Minerva Slurm Job 250159.
It replays the same byte-identical 12-window finite-rate Poisson trace while
using the online `latency_only` scheduler policy.

No manual L4 candidate-frequency cap is added. The wrapper explicitly clears
`MAX_L4_FREQ_OVERRIDE`, allowing the scheduler to search the complete modeled
L4 frequency list. The L40S candidate cap remains 2520 MHz, matching Jobs
250048 and 250159.

All other experimental settings are retained:

- fixed L40S-prefill/L4-decode placement with TP=1;
- TTFT SLO 500 ms and TPOT SLO 200 ms;
- minimum-SLO-violation overload fallback;
- exclusive Neptune and Ganymede allocation;
- identical benchmark bounds, GPU telemetry, and energy integration.

Predicted TTFT includes the same modeled cross-node KV-cache transfer term:

```text
predicted TTFT
  = predicted P99(queue + prefill)
  + input_tokens * KV_bytes_per_token * 8 / effective_bandwidth
  + dispatch
```

For Mistral-7B-v0.1, the job uses 131072 KV bytes per token and the measured
effective Neptune-Ganymede bandwidth of 24.076 Gbit/s. Dispatch is zero.
