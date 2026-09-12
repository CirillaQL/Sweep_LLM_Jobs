# r16 bounded gate
Primary skill: code-quality-workflow; library route code-quality-workflow,
ST-A0 opening gates confirmed; snapshot2026-08-18, no domain candidates adopted.

Goal: a new observational Job; SLO/clock mismatch and exhausted feasible search
are records/fallbacks rather than reasons to abort Production.
Red lines: preserve r15/260444 history, request timing, energy conservation,
burst schedule, cold-start search and dependencies; no submission/READY yet.
Acceptance: full mocked noncompliant Production completes; invalid Canary
clock candidates fall back; real execution/data errors remain visible.

Evidence: 260444 audit clock mismatch aborted a successful request stream.
Structural decision: Local Fix. Execution: Local Fix Only by behavior path:
meter clock enforcement/observation; Canary typed-error fallback; runtime
readback-only behavior and driver counters. No architecture rewrite.
Authorization: user explicitly requested a new Job with nonfatal metric outcomes.
Scope: new r16 only; expected meter,driver,controller,table,config,launcher,
validator,two tests,docs,generated manifest (12files), <=650changedlines vs r15.
Baseline:37CPUtests passed. After change:40CPUtests pass.
Regression gates include real-Meter subsample plateau, clock observations,
full mocked concurrent run with all Production SLO violations, readback-only
nonfatal service behavior, exhausted Canary clock fallback, real command failure,
SLO fallback, malformed data/code errors and preserved accounting.
Rollback: discard r16; r15 remains unchanged.
Stop on energy conservation regression, unrelated dependencies or scope expansion.
Real GPU/Slurm gate pending explicit submission. Audit completion is not evidence
of SLO compliance, exact requested clocks or optimal concurrent-load frequencies.
