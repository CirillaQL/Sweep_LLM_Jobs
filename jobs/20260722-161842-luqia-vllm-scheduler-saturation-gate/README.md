# vLLM Saturation-Gated Scheduler Energy Experiment

This job connects the offline saturation work in `Dual_Sweep_LLM/shared_model`
to the live two-node vLLM prefill/decode path.

## Runtime policy

- Allocation: one L40S on Neptune and one L4 on Ganymede.
- Placement is held fixed (Neptune prefill, Ganymede decode) so the experiment
  isolates admission/frequency decisions from model-reload and role-swap cost.
- Latency/power candidates come from the existing portable SWEEP model.
- Saturation probability is the mean of the five strict-config-split,
  per-GPU `local_retrained_75` classifiers from Dual_Sweep_LLM (seeds
  3, 7, 11, 19, and 23).
- A candidate is admitted only when the existing TTFT/TPOT guards pass and both
  role GPUs have `mean(p_saturated) < 0.30`.
- Safe candidates are ranked by predicted cluster board power.

The model target follows the source project exactly:

```text
measured_saturated = request_throughput_rps / request_rate < 0.95
```

The original sklearn classifiers are exported to `saturation_bundle.json`.
The live scheduler is standard-library only. `validate_saturation_export.py`
compares portable and sklearn predictions; the pre-submission check covers
1,000 predictions across both GPUs and all five seeds.

## Workloads and expected gate behavior

The workload file is identical to jobs 249820 and 249822:

| Workload | Input/output | Rate | Expected action |
|---|---:|---:|---|
| `short_r1` | 32/32 | 1 req/s | admit |
| `prefill_r1` | 512/128 | 1 req/s | admit |
| `burst_r10` | 128/32 | 10 req/s | no-admit |

The high-rate workload is intentionally retained. The prior no-gate run
selected 735/1200 MHz but achieved only 6.32 req/s at a requested 10 req/s,
which satisfies the source project's measured-saturation definition. This job
tests whether the new gate blocks that false-safe recommendation.

## Power measurement and comparison

Both nodes sample GPU board power every 0.5 seconds. Explicit start/end events
bracket each admitted `vllm bench serve` command. `energy_summary.py` performs
piecewise-linear trapezoidal integration. `compare_gate_results.py` joins the
live measurements with:

- Job 249822: default-DVFS, no-scheduler baseline;
- Job 249820: latency-only scheduler without the saturation gate.

These are GPU board-power measurements only; CPU, memory, NIC, fans, and PSU
losses are not included. The historical controls are separate runs, so the
result is a functional/initial optimization check rather than a confidence
interval.

## Frequency safety

Each admitted workload applies the scheduler's `rec_freq_mhz` with `-lgc`.
Cleanup stops the server step first, starts a fresh two-node reset step, runs
`sudo nvidia-smi -i 0 -rgc`, performs an active CUDA clock probe, and executes a
second `-rgc` as the final GPU-control operation. Signal and normal-exit paths
share the same cleanup handler.
