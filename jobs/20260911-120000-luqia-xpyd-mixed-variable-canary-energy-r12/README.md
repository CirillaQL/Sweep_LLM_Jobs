# Mixed variable-shape Canary + Production energy validation (r12)

This is a cold-start 2P2D experiment. It does not import an Oracle, a previous
frequency table, or prior Job results. P0/D0 runs Canary exploration; P1/D1
runs Production.

## What changed

- Production is a seeded shuffled mixture of all seven request classes. There
  is no one-class-per-window phase.
- Every class contains three nearby input/output shapes. For example,
  `small_light` contains 128/48, 160/64 and 200/80.
- Canary evaluates every frequency candidate against the same low/center/high
  shape panel (three requests), rather than letting random shape variation
  confound candidate comparisons.
- The primary request energy boundary is client send through completed SSE
  output. Time between requests, clock settling and queue-free idle power are
  excluded.
- A secondary control-inclusive boundary starts immediately before clock
  actuation. It reports whether DVFS command and settle overhead erase savings.
- Production is serial and waits a seeded random 0.75--1.25 seconds after each
  completed request. The 150-ms frequency settle happens before the next
  request begins, so requests are not burst-filled during clock transitions.

## Protocol

1. Start with no Table. While P0/D0 learns all seven classes, P1/D1 serves a
   shuffled variable-shape stream at safe-high clocks.
2. Canary searches a hardware-validated 17-level P grid and 15-level D grid,
   using binary SLO-boundary search plus bounded neighboring energy refinement.
   The SLO gate remains TTFT P95 < 500 ms and TPOT P95 <= 200 ms.
3. Once all Table entries exist, run 12 matched pairs per class (84 pairs, 168
   Production requests). Each pair uses exactly the same input/output lengths
   once at safe-high and once at the learned Table clocks. Pair order is
   randomized AB/BA and request classes are globally shuffled.
4. Restore all clocks to safe-high even on failure.

The matched comparison is stronger than a simple before/after mean because
each energy delta controls for exact token shape and randomizes whether high or
Table runs first. It still tests serial mixed traffic, not concurrent queueing
or a target RPS.

## Request classes

| Class | Shape panel (input/output tokens) |
|---|---|
| small_light | 128/48, 160/64, 200/80 |
| prefill_medium | 768/48, 1024/64, 1280/80 |
| prefill_heavy | 1792/48, 2048/64, 2304/80 |
| decode_medium | 96/96, 128/128, 200/160 |
| decode_heavy | 96/224, 128/256, 200/320 |
| balanced_medium | 384/96, 512/128, 640/160 |
| both_heavy | 1792/224, 2048/256, 2304/320 |

## Main outputs

- `summary.json` / `summary.md`: paired per-class high-versus-Table results.
- `production_requests.jsonl` / `.csv`: one row per Production request, exact
  shape, TTFT/TPOT, clocks, active energy, control-inclusive energy and client/
  proxy timestamps.
- `canary_overhead.json`: Canary wall time, gross attempt energy, request-active
  probe energy, control-inclusive probe energy, and per-class breakdown.
- `canary_probes.jsonl` / `.csv`: every tested frequency/shape combination.
- `canary_candidates.jsonl` / `.csv`: candidate aggregates and search metadata.
- `matched_pair_plan.json`: reproducible shuffled A/B plan.
- `power_trace_1s.csv` plus raw 50-ms P/D energy streams in scratch storage.
- `audit.json`: completion and measurement-contract gates.

NVML application-clock validation accepts the target within max(15 MHz, 1%)
for at least 95% of active samples. This tolerates normal readback jitter while
still rejecting an actually wrong clock.

## Submission

The `READY` marker requests submission through the repository broker. The
equivalent direct Slurm command is:

```bash
sbatch jobs/20260911-120000-luqia-xpyd-mixed-variable-canary-energy-r12/run.sbatch
```
