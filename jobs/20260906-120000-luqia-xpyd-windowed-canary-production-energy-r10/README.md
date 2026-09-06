# Sequential workload-window Canary + Production energy observation, r10

Prepared locally; not submitted. No READY marker.

## Purpose

This job removes the interleaved round-robin Production order used by Job
257828. The seven request classes run as seven contiguous windows in this
fixed order:

1. small_light
2. prefill_medium
3. prefill_heavy
4. decode_medium
5. decode_heavy
6. balanced_medium
7. both_heavy

Each window is an independent cold-knowledge feedback episode. Production
serves only that window's request shape. Its first missing-Table request queues
P0-D0 Canary exploration while P1-D1 Production continues at safe high
(P=2520, D=1500 MHz). Once Canary writes the class's Table entry, the next
Production request applies the learned P/D pair. Production then serves at
least 30 consecutive Table-hit requests without changing workload type before
advancing to the next window.

This design normally requires one Production change from the preceding
window's frequency to safe high, and one change from safe high to the learned
frequency. Inside the learned Table phase, the audit permits at most one
frequency change: the first Table request. It rejects any later per-request
frequency churn.

## Experimental constants

- Canary=P0-D0; Production=P1-D1.
- Empty Table, no historical SLO/frequency/energy/Oracle prior.
- Hardware-discovered P17 and D15 grids.
- Binary SLO-boundary search plus bounded neighbor energy refinement.
- Three probes per candidate and three final confirmations.
- Canary SLO: TTFT P95 <500 ms and TPOT P95 <=200 ms.
- Production SLO deviations are reported but do not fail this energy experiment.
- Production request interval remains 5 seconds; concurrency remains one.
- Ten-hour internal timeout and twelve-hour Slurm limit are safety bounds, not
  fixed experiment windows.

| Class | Input tokens | Output tokens |
|---|---:|---:|
| small_light | 128 | 64 |
| prefill_medium | 1024 | 64 |
| prefill_heavy | 2048 | 64 |
| decode_medium | 128 | 128 |
| decode_heavy | 128 | 256 |
| balanced_medium | 512 | 128 |
| both_heavy | 2048 | 256 |

## Measurement

Persistent node-local NVML samplers measure all four GPUs every 100 ms.
Production records each request's full control-inclusive duration/energy and
its inference-only duration/energy, actual P/D clocks, Table revision, TTFT,
TPOT, output tokens, window index and within-window request index. Canary keeps
the full per-probe and per-candidate evidence used by r9.

`workload_windows.jsonl` records each contiguous window's boundaries, safe-high
request count, Table request count, first Table sequence, and frequency-change
count. The final audit reconstructs workload blocks from every Production row,
requires exactly seven blocks in configured order, requires at least 30 Table
requests per block, and rejects more than one frequency change in a Table phase.

Canary overhead is summed over seven disjoint class exploration intervals—from
that class's first attempt start through its final attempt end. This includes
warmup, probes, confirmations, retries and same-class retry gaps, but excludes
P0-D0 idle energy while Production finishes the 30-request learned tail of an
earlier window. CPU, DRAM, NIC, network equipment and model loading remain out
of scope.

## Interpretation boundary

The within-class comparison is sequential before/after observation, not a
randomized causal estimate. Windowing removes per-request workload switching
as the explanation for steady Table-phase energy, but time, temperature and
the unequal number of safe-high and Table requests can still confound the
comparison.

Submission remains a separate READY, commit and push step.
