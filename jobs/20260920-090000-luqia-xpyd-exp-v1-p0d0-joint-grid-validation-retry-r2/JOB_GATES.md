# P0-D0 joint-grid validation gate

Goal: validate the prior Canary coordinate-wise P/D frequency strategy against
an exhaustive 17×15 joint grid. The job uses only P0 (neptune L40S GPU 0) and
D0 (ganymede L4 GPU 0), five low-load workload classes, and exactly three
formal measurements per pair.

Red lines: do not modify prior Jobs or results; do not allocate P1--P3 or
D1--D3; do not change workload shapes, the 17×15 frequency grids, the SLO,
or the energy/latency measurement boundary. The Job may not claim that the
Canary strategy was validated unless every formal candidate is complete.

Acceptance: static preflight passes; a completed high-frequency P0→D0
streaming smoke request is required before the formal grid; smoke failure
records the complete exception cause chain, proxy diagnostics, and P/D server
logs, then exits nonzero without starting the grid. On smoke success, the job
records 1,275 candidates and 3,825 formal requests, and compares each
workload's SLO-safe global joint-energy optimum with the coordinate and
component reconstructions.

Structural decision: Local Fix. Execution decision: authorized repair and
resubmission. Baseline: Job 264777 completed but was invalid because every
formal request failed with a generic Prefill upstream error. Expected changed
scope: this new r2 job directory only. Rollback: the failed prior Jobs remain
unchanged; r2 can be diagnosed from its copied failure artifacts.
