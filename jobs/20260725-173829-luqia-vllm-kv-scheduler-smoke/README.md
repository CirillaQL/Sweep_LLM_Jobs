# KV-aware vLLM scheduler smoke test

This is a saturation-oriented live test using fourteen workload windows. It
reserves four L40S GPUs on Neptune and four L4 GPUs on Ganymede, starts one
prefill and one decode instance, and runs the online
`latency_plus_saturation` scheduler before every window.

The placement remains fixed to L40S prefill and L4 decode. When the gate marks
a phase as saturated, the runner increases the planned instance count and
reruns the scheduler with the request rate divided across active instances.
It repeats this until the phase is predicted safe or reaches four instances,
then starts the required vLLM instances on the reserved GPUs and waits for
proxy registration. The proxy round-robins requests across the registered
prefill/decode pools.

Before each later window, the runner also probes whether one prefill and one
decode instance are safe for that window. When they are, it stops every extra
vLLM process, waits for the proxy registry to return to exactly one instance
per role, and only then starts the next workload. It does not migrate in-flight
requests; scale-in occurs between completed benchmark windows.

Safe candidates are ranked by predicted cluster board power. If the configured
four-instance limit is still insufficient, the scheduler retains the existing
minimum-SLO-violation fallback and records that scale-out capacity is exhausted.

## Workload progression

The trace is ordered to exercise separate scale-out transitions rather than
letting one extreme workload immediately consume every reserved GPU:

- short 32/32 requests at 8, 16, and 20 RPS scale decode 1 -> 2 -> 3 -> 4;
- 32/32 at 32 RPS scales prefill to 2 and exercises decode capacity exhaustion;
- balanced 128/128 requests at 16 and 24 RPS scale prefill 2 -> 3 -> 4;
- decode-heavy, long-prompt, balanced-extreme, extreme-RPS, and long-context
  windows exercise saturation while both pools are at their four-instance cap;
- a final 32/32 at 1 RPS verifies scale-in from 4/4 back to 1/1.

The client permits up to 500 prompts and 128 concurrent requests so the high
RPS windows generate sustained pressure instead of being truncated to a small
smoke-test sample.

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

- latency-only, saturation-gated, and active scheduler JSON for every window,
  including active instance counts and effective per-instance request rates;
- the active placement and requested/observed GPU clocks;
- allocation-wide 0.5-second board-power samples for all four L40S and all
  four L4 GPUs, including idle GPUs before scale-out and after scale-in;
- per-instance utilization, clock, temperature, memory, and 100 GbE telemetry;
- timestamped workload start/end events;
- per-window benchmark output, per-GPU/per-host/combined integrated energy, the
  number of covered GPU streams, and joined live summaries.

`monitor_gpu_power.py` creates one allocation-level CSV per host and records
each GPU by UUID. `energy_summary.py` discovers every UUID stream, integrates
each stream independently, and writes `energy_summary.json` plus
`energy_by_workload.csv`. The instance telemetry remains available for
utilization analysis but is not added again when allocation-level power files
exist.

Results are written under `kv_scheduler_smoke_results/`. The Slurm allocation
is exclusive and has a hard 30-minute time limit.
