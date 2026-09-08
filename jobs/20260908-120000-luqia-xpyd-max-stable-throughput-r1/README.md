# Per-workload maximum stable throughput characterization, r1

Prepared locally; not submitted. No `READY` marker.

## Goal

Measure the maximum stable open-loop offered rate for each of the seven fixed
request classes on the real P1-D1 Production pair at safe-high clocks. This is
a capacity characterization job, not a Canary or frequency-search job.

The result is intended to define normalized Production load bands such as
20%, 50%, and 80% of each class's measured capacity. It does not import any
historical Job result, Oracle, capacity value, or learned frequency Table.

## Hardware and serving path

- Two allocated nodes: Uranus and Ganymede, exclusive allocation.
- P0/P1 are L40S endpoints; D0/D1 are L4 endpoints.
- The measured pair is P1-D1, TP=1 on both sides.
- vLLM 0.15.1 with `P2pNcclConnector` and the existing cross-node KV path.
- All four GPUs are held at their discovered safe-high application clocks;
  the expected P1-D1 pair is 2520/1500 MHz.
- P0-D0 remains outside the offered traffic and is not used as a hidden prior.

## Request classes

| Class | Input tokens | Output tokens |
|---|---:|---:|
| small_light | 128 | 64 |
| prefill_medium | 1024 | 64 |
| prefill_heavy | 2048 | 64 |
| decode_medium | 128 | 128 |
| decode_heavy | 128 | 256 |
| balanced_medium | 512 | 128 |
| both_heavy | 2048 | 256 |

## Search protocol

Each request is scheduled from an absolute monotonic deadline. New arrivals do
not wait for earlier requests to finish, so this is a true open-loop load test.
The client permits up to 64 simultaneous in-flight requests.

For each class:

1. Send three unmeasured warm-up requests to establish the P-D connection and
   warm model/runtime state.
2. Start at 0.10 RPS without consulting earlier measurements.
3. Double the offered rate until an unstable point is found, or halve it until
   a stable point is found. Bounds are 0.0125--2.0 RPS.
4. Refine the stable/unstable bracket for at most five midpoint iterations or
   until the relative bracket width is at most 5%.
5. Require two new stable confirmation windows at the selected rate. If a
   confirmation fails, fall back to a lower independently stable search point.

Each candidate schedules at least 30 requests and targets at least 90 seconds
of arrivals, capped at 120 requests. A candidate may drain for up to 600
seconds. An internal 11-hour bound and 12-hour Slurm allocation are safety
timeouts, not fixed per-class test durations.

## Definition of stable

A candidate is stable only when all gates pass:

- at least 99% request success;
- achieved throughput is at least 95% of offered throughput, including drain
  time in the denominator;
- TTFT P95 is strictly below 500 ms;
- TPOT P95 is at most 200 ms;
- late-window median TTFT does not exceed both the 25% relative-growth and
  50 ms absolute-growth allowance;
- all planned arrivals are issued, the 64-request in-flight guard is not hit,
  and draining does not time out.

The reported maximum is therefore the highest rate that passes the search
window and two independent confirmation windows. If the upper 2.0-RPS bound
remains stable, the result is explicitly marked as a lower bound rather than
an exact maximum.

## Measurements and artifacts

Persistent node-local NVML samplers record all four GPUs every 100 ms. For each
candidate window the job records offered and achieved RPS, request counts,
success ratio, backlog at arrival stop, early/late latency, TTFT/TPOT P95,
P1+D1 energy, mean power, and joules per completed request.

Expected archived outputs include:

- `summary.json` and `summary.md`;
- `capacity_by_workload.jsonl`;
- `capacity_candidates.jsonl` and `.csv`;
- request-level `requests.jsonl` and `.csv`;
- `power_trace_1s.csv`, hardware grids, diagnostics, and audit evidence.

## Interpretation boundary

This job estimates the capacity of one P1-D1 pair, for one model and connector
path, at safe-high clocks. It does not establish fleet-wide capacity, mixed-
workload capacity, or the capacity of a low-frequency Canary Table. GPU energy
is measured, while CPU, DRAM, NIC, switch, and model-loading energy remain out
of scope.

Submission remains a separate `READY`, commit, push, and `sbatch` step.
