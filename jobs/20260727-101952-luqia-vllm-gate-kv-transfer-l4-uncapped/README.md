# Saturation-gated KV-aware trace with uncapped L4 frequency search

This job reruns Minerva Slurm Job 250048 with the same byte-identical
12-window finite-rate Poisson trace and the same online
`latency_plus_saturation` scheduler policy.

The only intended scheduling change is removal of the manual
`MAX_L4_FREQ_OVERRIDE=780` candidate cap. No replacement L4 cap is added, so
the scheduler can search the complete modeled L4 frequency list. The L40S
candidate cap remains 2520 MHz, matching Job 250048.

All other experimental settings are retained:

- fixed L40S-prefill/L4-decode placement with TP=1;
- TTFT SLO 500 ms and TPOT SLO 200 ms;
- saturation gate threshold 0.30;
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
