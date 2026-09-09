# Load-conditioned P/D energy search, r1

Prepared locally; not submitted. No `READY` marker.

## Question

Does the energy-minimizing P/D frequency selected by Canary change as offered
load increases, even when the request shape is unchanged?

This is an exploration-only job. It runs on the dedicated P0-D0 Canary pair
and does not start a Production trace or write a Production control table.
It imports no historical Job result, capacity estimate, Oracle, or learned
frequency Table.

## Workloads and loads

Only the two requested fixed request classes are tested:

| Class | Input tokens | Output tokens |
|---|---:|---:|
| small_light | 128 | 64 |
| prefill_medium | 1024 | 64 |

Each class is independently tested at these 16 offered rates:

`0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0 RPS`.

Requests use absolute monotonic arrival deadlines and may overlap. A frequency
candidate schedules at least 12 requests and otherwise targets 20 seconds of
arrivals, capped at 60 requests. The complete measurement window runs from the
first planned arrival through full drain, so queueing and drain energy are not
silently omitted. Up to 64 requests may be in flight.

## Frequency search

The job discovers the hardware-supported grids at runtime and requires exactly:

- 17 P frequencies from 900 through 2520 MHz;
- 15 D frequencies from 450 through 1500 MHz.

For every workload/load cell, with no warm start from another load:

1. Hold D at its discovered maximum and minimize joint P0+D0 joules/completed
   request over the P axis.
2. Hold P at that selected value and minimize the same joint energy objective
   over the D axis.
3. Require two valid new confirmation windows at the selected P/D pair, with
   at most three attempts so one transient measurement failure does not discard
   all earlier load cells.

Each axis uses adjacent-slope binary minimization: compare the energy at `mid`
and `mid+1`, retain the lower-energy half, then measure the terminal point and
its immediate neighbors. Both grid endpoints are always measured. Every exact
candidate is cached within its axis search, so a binary comparison never
silently reuses a historical run.

This algorithm assumes that each discrete one-dimensional energy curve is
approximately unimodal. It implements the requested assumption
`P-axis optimum + D-axis optimum = global optimum`, but it does **not** prove a
full 17x15 global Oracle. The output explicitly records every comparison,
visited frequency, energy value, terminal point, and selected point so that
non-unimodality and noisy decisions remain auditable.

## TTFT and selection policy

TTFT is recorded per request and summarized as mean, P50, P95, and maximum for
every candidate and confirmation. The previous 500 ms value is retained only
as a descriptive reference column.

Neither TTFT nor TPOT is a frequency-selection gate in this job. TPOT is not
reported. A candidate is excluded only if the measurement itself is invalid:
not all planned requests were issued and completed successfully, the in-flight
guard was hit, draining timed out, or verified-frequency energy integration
failed. Achieved/offered throughput and backlog at arrival stop are recorded to
show when a low-energy result comes from an overloaded regime.

## Energy boundary

A persistent 100 ms NVML sampler records all four allocated GPUs. Candidate
ranking uses P0+D0 board energy integrated from the start of open-loop arrivals
through complete drain, divided by completed requests. Frequency changes and
the 0.5 second settle period occur before the energy window.

The metric includes P/D GPU idle gaps, batching, queueing, inference, and drain
inside the window. It excludes CPU, DRAM, NIC, switch, and cooling energy.

## Runtime and outputs

The Slurm allocation and internal experiment limits are 24 hours and 23 hours.
These are safety ceilings, not fixed experiment durations. Each request and
drain may wait up to 900 seconds so that deliberately overloaded high-RPS/low-
frequency cells fail explicitly instead of being truncated silently.

Expected archived outputs:

- `summary.json` and `summary.md`;
- `load_frequency_table.json`;
- full `load_frequency_candidates.jsonl` and `.csv`;
- request-level `requests.jsonl` and `.csv`;
- `diagnostics.jsonl`, `power_trace_1s.csv`, hardware grids, and audit evidence.

Submission remains a separate `READY`, commit, push, and `sbatch` step.
