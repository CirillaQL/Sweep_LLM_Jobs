# KV-aware vLLM scheduler smoke test

This is a bounded live test using six representative windows selected from
`experiment/input_samples.csv`. It requests one L40S on Neptune and one L4 on
Ganymede, starts the existing two-node vLLM prefill/decode service, and runs
the online `latency_plus_saturation` scheduler before every window.

The placement is fixed to L40S prefill and L4 decode so that online decisions
change GPU frequencies without reloading the model. Safe candidates are ranked
by predicted cluster board power. If no safe candidate exists, the scheduler
uses the existing minimum-SLO-violation fallback.

## KV transfer term

The scheduler adds cross-node KV-cache transfer time to predicted TTFT:

```text
predicted TTFT
  = predicted P99(queue + prefill)
  + input_tokens * KV_bytes_per_token * 8 / effective_bandwidth
  + dispatch
```

For Mistral-7B-v0.1, the job uses 32 layers, 8 KV heads, head dimension 128,
FP16 elements, and the existing measured effective Neptune-Ganymede bandwidth
of 24.076 Gbit/s. Dispatch is zero.

## Monitoring and outputs

The inherited runner records:

- latency-only, saturation-gated, and active scheduler JSON for every window;
- the active placement and requested/observed GPU clocks;
- 0.5-second L4 and L40S power, utilization, clock, temperature, memory, and
  100 GbE telemetry;
- timestamped workload start/end events;
- per-window benchmark output, integrated GPU energy, and joined live summaries.

Results are written under `kv_scheduler_smoke_results/`. The Slurm allocation
is exclusive and has a hard 28-minute time limit.
