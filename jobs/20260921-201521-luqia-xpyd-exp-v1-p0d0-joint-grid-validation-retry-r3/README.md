# exp-v1 P0-D0 joint-grid validation retry r3

This retry addresses the invalid Job 264777 result and r2 startup failure.
All 3,825 attempted
measurements failed before Prefill completed, but the old driver collapsed the
upstream cause to `ProxyUpstreamError: prefill request failed` and continued
through the full grid. Before recording any formal measurement, r2 now makes
one P0→D0 high-frequency streaming request. A failure writes the complete
exception chain and any proxy diagnostics, exits without starting the grid,
and copies both vLLM server logs to the results directory. r3 also explicitly
isolates FlashInfer, vLLM, Torch, XDG, temporary, and dependency caches under
`/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`.

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
gap in joules and percent. The single smoke request is excluded from formal
candidate measurements. Once it passes, failed formal requests are recorded
and the rest of the grid continues; `audit.json` marks the result invalid if
the intended three successful samples for every candidate were not collected.

Outputs copied into `results/<slurm-job-id>/` are:

- `joint_grid_request_measurements.csv` — all individual requests.
- `joint_grid_candidate_measurements.csv` — 1,275 three-sample aggregates.
- `joint_grid_strategy_validation.json` — global optimum and both strategy
  reconstructions for all five workloads.
- `audit.json`, `preflight.json`, `smoke_success.json` (or
  `smoke_failure.json`), `proxy_diagnostics.jsonl`, and the Prefill/Decode
  server logs.

The `READY` file is intentionally added only after static validation, before
the job is submitted through the repository Git broker.
