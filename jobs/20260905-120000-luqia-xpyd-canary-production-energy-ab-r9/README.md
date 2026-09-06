# Continuous Canary + Production energy observation, r9

Prepared locally; not submitted. No READY marker. The existing r9 directory
name is retained, but this protocol has no A/B or fixed-duration windows.

## Run behavior

Production continuously serves the same seven workload shapes in interleaved
order, concurrency one, with the existing 5-second inter-request gap. Before a
class has a Table entry it uses default high frequency (P=2520, D=1500 MHz).
The first missing entry queues a clone on the independent P0-D0 Canary worker.
As each class's exploration completes and writes its Table entry, subsequent
Production requests of that class immediately use the learned frequency.
Other classes continue using their own Table entry or high fallback.

Stop when every class has completed at least 30 actual Table-hit requests.
Earlier classes continue being served while later classes catch up. High
sample counts and elapsed time are outcomes of the process, not prescribed
windows. A 10-hour safety timeout and 12-hour Slurm limit prevent an endless
failed exploration; neither determines the normal experiment duration.

There is only one initial Production route warmup, recorded separately.
Canary's own warmup, final confirmations, retries, clock changes and waits
remain part of exploration cost. No forced high holdouts, fixed 120-second
stages, idle baseline stages or repeated A/B switchbacks are executed.

## Unchanged experimental inputs

Uranus P0/P1 L40S, Ganymede D0/D1 L4, Mistral-7B, vLLM 0.15.1 and
P2pNcclConnector. Canary=P0-D0, Production=P1-D1. Empty Table, no historical
frequency/energy/SLO/Oracle priors. Hardware-discovered P17/D15; binary SLO
boundary then bounded neighbor energy refinement, 3 repeats per candidate,
3 final confirmations. TTFT P95 <500 ms and TPOT P95 <=200 ms guide Canary.
Production SLO deviations are recorded without failing the energy experiment,
as requested. Table values are measured search winners, not proven global optima.

| Class | Input tokens | Output tokens |
|---|---:|---:|
| small_light | 128 | 64 |
| prefill_medium | 1024 | 64 |
| prefill_heavy | 2048 | 64 |
| decode_medium | 128 | 128 |
| decode_heavy | 128 | 256 |
| balanced_medium | 512 | 128 |
| both_heavy | 2048 | 256 |

## What is recorded

Production: logical ID, class, dispatch sequence, start/end/time, actual
dispatch frequency and Table revision/source, clock change and settle delay,
completed output tokens, service TTFT/TPOT, control-inclusive TTFT, P1/D1
individual and combined energy and mean power, and concurrent Canary class.
Full request energy starts before controller handling and ends at full SSE
completion. Inference energy starts at the core's request_received timestamp.
This is an in-process service harness: control-inclusive latency includes
clock handling but excludes an external HTTP client's transport overhead.

Canary: every probe's class, attempt, search stage (binary/refinement/
confirmation/warmup), axis, grid index, P/D pair, repeat index, candidate ID,
start/end/duration, TTFT/TPOT, P0/D0 energy, mean W and clock evidence.
Inference and control-inclusive probe energy are separate. Candidate
aggregation and the search decision events remain linked by ID. Retry IDs
are unique. Started/completed/failed events preserve attempted combinations
even if a probe fails before valid energy can be measured.

Canary cost: whole exploration gross P0+D0 J and mean W; per-class/per-attempt
time, J and W; whole-phase gaps and retry backoff energy. This is the extra
GPU budget allocated to discovery, not an idle-subtracted marginal-energy
estimate. No counterfactual idle subtraction is made in this continuous
protocol. Model loading, CPU/DRAM, NIC and network equipment are excluded.

## Measurement and interpretation

Two persistent node-local NVML samplers record cumulative energy, sensor W
and SM clocks every 100 ms. Requests do not spawn Slurm steps for energy
reads. Node time versus collector arrival discrepancy must be <=250 ms;
energy intervals need bracketing samples, monotonic counters and gaps <=0.5s.
Inference clock samples must match targets >=95%. Complete SSE DONE, exact
output token usage and request diagnostics are required. Telemetry/stream
failures stop or trigger the original bounded Canary retries; no fake zeros.

Per-class safe_high versus table summaries report counts, time-weighted W,
J/request, durations, TTFT/TPOT P95, observed frequency pairs, and percentage
differences. These are natural observational groups: different sample sizes,
temperature/time drift and concurrent Canary activity can affect differences.
No randomized confidence interval, causal energy claim or payback projection
is inferred. Lower W alone need not mean lower J/request.

Raw 100-ms NVML samples and diagnostics remain in the cluster cache. Compact
1-second per-GPU sensor averages are exported with timestamps for plotting.

## Artifacts

- production_requests.jsonl / .csv: every real Production request.
- canary_probes.jsonl / .csv: individual measured probes.
- canary_probe_events.jsonl: every started/completed/failed probe.
- canary_candidates.jsonl / .csv: repeated-probe aggregation.
- canary_attempts.jsonl: each whole-class exploration attempt.
- canary_overhead.json: exploration cost totals.
- feedback_events.jsonl, service_dispatch.jsonl, frequency_table.json,
  hardware_grids.json: decisions and actual selected clocks.
- power_trace_1s.csv: four-GPU power traces.
- summary.json / .md and audit.json: observational comparison and completeness.

Source checksum manifest and local unit/integration tests are provided.
Real Slurm/GPU sampler alignment, streamed inference and power results require
the cluster run. Submission is a separate READY + commit/push operation.
