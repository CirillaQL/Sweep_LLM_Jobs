# Request-active P/D energy search, r2

Prepared locally; not submitted. There is deliberately no `READY` marker.

## Question

Does the energy-minimizing P/D frequency change with offered load for a fixed
request shape when energy is measured only while one or more requests are in
flight?

This corrects Job 259389/r1. That job integrated continuously from the start of
a candidate window through full drain, so at 0.1 RPS it charged the long gaps
between requests to the requests. The r1 values remain useful as always-on
system cost, but they are not request-active energy.

## Workloads and loads

The experiment remains cold-start and history-free. It uses only P0-D0 Canary:

| Class | Input tokens | Output tokens |
|---|---:|---:|
| small_light | 128 | 64 |
| prefill_medium | 1024 | 64 |

Each class is independently tested at
`0.1, 0.3, 0.5, ..., 2.9, 3.0 RPS`. Requests follow absolute monotonic
open-loop deadlines and may overlap. Each frequency candidate has at least 12
requests, targets 20 seconds of arrivals, and is capped at 60 requests.

## Exact request timeline

Every request records nanosecond-resolution client wall and monotonic stamps
for dispatch, first client-visible non-empty SSE token, final chunk, and stream
completion. It also copies both wall and monotonic proxy stamps for:

- request received and route selected;
- Prefill start and completion;
- synchronous KV handoff completion;
- Decode request start and response headers;
- first Decode chunk received and forwarded;
- last Decode chunk and response completion.

These are software-observation boundaries, not GPU-kernel timestamps. Wall
timestamps align request intervals with node-local NVML samples; monotonic
timestamps provide duration checks without wall-clock steps.

## Primary energy boundary

For successful request `i`, the primary interval is
`[client_send, client_complete]`. The job merges all overlapping request
intervals and integrates the P0+D0 cumulative NVML counters only over their
union. Any interval with zero requests in flight is excluded.

For concurrency, every elementary energy segment is divided equally among the
requests active in that segment. The sum of attributed per-request energy is
required to equal the union energy within numerical tolerance, preventing
overlap from being double-counted.

The job additionally records:

- gross first-send-through-drain energy as a secondary always-on cost metric;
- P0 energy attributed over Prefill-stage intervals;
- D0 energy attributed over Decode-stage intervals;
- active-union duration, excluded idle duration, mean active concurrency, and
  attribution conservation error.

Candidate ranking and binary-search decisions use only active-union P0+D0
joules/completed request. Gross-window energy never selects a frequency.

## Sampling robustness

The persistent node-local NVML sampler defaults to 50 ms. Each sample records
the cumulative board-energy counter, power, SM clock, node-local timestamp,
read duration, controller receipt timestamp, and observed transport delay.
Counter values are linearly interpolated at every request-union boundary.

A delayed stdout delivery is diagnostic evidence, not an automatic fatal
error: it does not change the node-local timestamp or cumulative counter. The
candidate still fails if its boundaries lack samples, a counter resets, a
sample gap exceeds 500 ms, the requested clock matches less than 95% of active
samples, or the per-request attribution fails energy conservation. Sampler
health is written separately.

## Frequency search and TTFT

The runtime-discovered grids must be P17 (900--2520 MHz) and D15
(450--1500 MHz). For each workload/load cell, independently:

1. hold D high and use adjacent-slope discrete binary minimization on P;
2. fix selected P and minimize D the same way;
3. run two fresh valid confirmation windows, with at most three attempts.

This implements the requested `P optimum + D optimum` approximation and assumes
each one-dimensional energy curve is approximately unimodal. It is not a full
17x15 Oracle. TTFT is recorded but is descriptive only; it does not gate
selection.

## Runtime and outputs

Slurm and internal ceilings remain 24 hours and 23 hours. Request and drain
timeouts are 900 seconds. Expected outputs include the active-energy frequency
table, candidate and request CSV/JSONL records, complete proxy diagnostics,
power trace, hardware grids, sampler health, summary, and audit. Raw 50 ms
samples remain under the recorded raw-cache path rather than being committed.

Submission is a separate `READY`, commit, push, and `sbatch` action.
