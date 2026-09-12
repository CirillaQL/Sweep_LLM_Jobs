# r14 bounded implementation gate

Primary skill: code-quality-workflow. Route receipt: code-quality-workflow,
ST-A0 opening gates confirmed, snapshot 2026-08-18. No domain candidates adopted.

Goal: create a new Job with homogeneous multi-batch concurrent Production arrivals.
Red lines: do not modify r13/old jobs, historical Table priors, dependencies,
or submit/create READY without a separate user instruction.
Acceptance: executable concurrent driver, conserved request/burst energy,
same search/fallback/latency logging, CPU tests and source inventory validation.

Evidence: r13 awaited complete SSE output before issuing each subsequent request;
per-request board windows would double count shared energy under concurrency.
Structural decision: Local Fix. Execution decision: Local Fix Only, three
bounded feature paths: burst schedule/clock ownership, energy attribution,
matched concurrent evaluation cohorts. No architectural rewrite.
User authorization: create new Job and change Production arrival behavior.

Expected files: driver, config, launcher, validator, tests, README, this gate,
generated source manifest (8 files). Budget relative to copied r13: 8 files,
700 changed lines. Imported 35-source-file snapshot is not a rewrite.
Forbidden: all unrelated changes, dependency updates, old jobs, git submission.
Rollback: discard only r14, preserving r13 unchanged.
Pre-change baseline: 31 r13 CPU tests passed.
Verification gate: inherited tests plus energy overlap/idle/cohort/concurrency
regressions, SHA256 source inventory, Python/JSON and shell syntax validation.
Stop conditions: failed energy conservation, mixed-clock in-flight epochs,
unrelated required modules, or scope/budget expansion.
Real GPU acceptance pending submission: synthetic bursts do not establish
arbitrary concurrent-load optimality or actual production energy savings.

Verification result: 34 CPU tests passed, including full mocked driver with
>=5 in-flight requests, energy conservation and identical high/Table cohorts.
35-file source SHA256/AST/JSON inventory and both shell syntax checks passed.
Final isolated-snapshot scope: 8 changed files, under 700 changed lines, no warnings.
