# r13 bounded implementation gate

Evidence: r12 Job260328 exhausted SLO retries for two classes, left null Table
entries, then ran 4005 further safe-high requests while waiting for all entries.

Structural decision: Local Fix. Execution: Local Fix Only, separated by path:
terminal worker hook; explicit fallback schema/policy; online dispatch/random
schedule; tests and launch configuration. No architecture rewrite or new dependency.

Authorized contract: publish highest-clock fallback after three SLO failures;
apply each published category immediately; independent category draws per online
request; globally shuffle individual comparison requests. Preserve cold start,
shape panels, search, energy boundaries, clock validation, serialization and gaps.

Scope: new r13 directory only. Imported source snapshot is unchanged except the
three policy modules and config/launcher; tests/docs/manifest are updated. Delta
budget relative to r12: <=10 files and <=700 changed lines. Old jobs, repository
history, unrelated dirty changes, dependencies, submission and READY are forbidden.
Rollback: discard r13 only; r12 remains unchanged.

Pre-change baseline: 24 CPU tests passed. Verification: 31 CPU tests passed,
including full driver with five learned/two fallback classes, immediate early
Production application, 168 globally shuffled paired requests, third-failure
termination, null/non-SLO fallback, and infrastructure-error separation.
35-source-file SHA256 inventory, Python/JSON/shell syntax and real config loader
are checked. Full inherited inventory is not a 23k-line rewrite: scope gate uses
an isolated r12 snapshot with r13 overlaid, including all resulting changes.

Library receipt: route code-quality-workflow; ST-A0; snapshot 2026-08-18;
no domain candidates. Goal/authority and red lines are user-approved above.

Pending gate: actual Slurm/GPU run after explicit submission. Offline evidence
does not prove energy savings or real-device SLO compliance.
