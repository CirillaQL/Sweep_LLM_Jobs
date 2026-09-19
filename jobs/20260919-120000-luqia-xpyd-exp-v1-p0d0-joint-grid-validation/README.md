# exp-v1 P0-D0 joint-grid validation

This job validates the prior Canary assumption that a coordinate-wise P/D
search reaches the same minimum-energy configuration as a full joint search.
It allocates only one physical pair:

- P0: GPU 0 on `neptune` (L40S), Prefill.
- D0: GPU 0 on `ganymede` (L4), Decode.

For each of the five request classes that produced a learned low-load setting
in the preceding experiment (`small_light`, `prefill_medium`,
`decode_medium`, `decode_heavy`, and `balanced_medium`), the job uses the
hardware-discovered 17-level P grid and 15-level D grid. It measures every
joint P/D pair exactly three times: 255 candidates and 765 requests per
class, 1,275 candidates and 3,825 requests total. Each repetition shuffles
the 255 pairs deterministically, preventing a fixed frequency order from
being mistaken for an energy effect.

Every request stores timestamp, workload, repetition, requested P/D clocks,
P energy, D energy, total active request energy, request duration, TTFT, and
TPOT. Candidate rows store the three-sample mean energy, standard deviation,
and TTFT/TPOT P95 under the existing 500 ms TTFT / 200 ms TPOT SLO boundary.

## Strategy validation

For each class, the result compares:

1. The SLO-safe global joint optimum: the lowest mean total energy among all
   255 P/D pairs.
2. A reconstruction of the earlier Canary coordinate policy: minimize total
   energy over P while D is high, then minimize total energy over D at that
   selected P.
3. An independent component check: choose the minimum mean P energy and
   minimum mean D energy seen across the full grid, then inspect their joint
   pair.

The JSON report records pair equality with the global optimum and the energy
gap in joules and percent. Failed requests are recorded and the rest of the
grid continues; `audit.json` marks the result invalid if the intended three
successful samples for every candidate were not collected.

Outputs copied into `results/<slurm-job-id>/` are:

- `joint_grid_request_measurements.csv` — all individual requests.
- `joint_grid_candidate_measurements.csv` — 1,275 three-sample aggregates.
- `joint_grid_strategy_validation.json` — global optimum and both strategy
  reconstructions for all five workloads.
- `audit.json` and `preflight.json`.

The `READY` file is intentionally added only after static validation, before
the job is submitted through the repository Git broker.
