# XPYD P0-D0 exhaustive per-axis power exploration

This Job performs exploration only. It sends no formal Production requests
to P1-D1 and runs every measured request on P0-D0:

- P0: GPU 0 on `uranus` (L40S), TP=1.
- D0: GPU 0 on `ganymede` (L4), TP=1.
- The P/D path uses `P2pNcclConnector`.

The seven request classes are the established token shapes from
`small_light` through `both_heavy`. For every class, the Job selects 17 P
frequencies and 15 D frequencies by evenly sampling the frequencies actually
reported as supported by each GPU within the configured safe range. The
runtime fails closed if it cannot construct exactly those grids or include
the safe-high endpoint.

## Search and SLO contract

The P sweep holds D at 1500 MHz and tests all 17 P candidates. The P winner is
the SLO-safe candidate with the lowest mean P0 power. The D sweep then holds P
at that winning frequency and tests all 15 D candidates. The D winner is the
SLO-safe candidate with the lowest mean D0 power. This implements the stated
factorization:

`minimum P0 power + minimum D0 power = selected global power configuration`

Because the D sweep uses the selected P frequency, the final P+D combination
is itself measured three times and must meet TTFT p95 < 500 ms and TPOT p95
<= 200 ms. Every candidate uses exactly three requests; there is no adaptive
early stop and no extra confirmation request. Therefore each workload has 32
candidate experiments and 96 measured requests, for 224 candidate rows and
672 measured requests across all seven workloads. One unmeasured P0-D0
full-path warmup runs before the first candidate.

NVML board-energy counters are read independently for P0 and D0 around every
request. The Job records per-endpoint power, per-endpoint energy, joint power,
joint energy, TTFT, TPOT, clock actuation, and clock readback for every sample.

## Completion and outputs

The exploration driver queues all seven classes and waits until the persistent
frequency table contains all seven SLO-safe winners. Its timeout is 25,200
seconds (seven hours), and the Slurm wall time is eight hours. An exploration
failure is detected immediately; the exit trap still copies all intermediate
events and table state.

Compact outputs include:

- `candidate_measurements.csv`: all 224 aggregated candidate measurements.
- `best_power_configurations.{json,csv}`: the seven selected P/D settings.
- `frequency_table.json`: persistent selected configurations.
- `feedback_events.jsonl`: all 672 raw probes, aggregates, selections, and
  frequency readbacks.
- `audit.json`, `summary.json`, and `summary.md`: fail-closed result audit.

Raw data stays below
`/data/users/chjing/vllm_job_work/<job_id>/p0d0_exhaustive_axis_power/`.

The `READY` marker is included as the final file and authorizes broker
submission after this Job directory is committed and pushed.
