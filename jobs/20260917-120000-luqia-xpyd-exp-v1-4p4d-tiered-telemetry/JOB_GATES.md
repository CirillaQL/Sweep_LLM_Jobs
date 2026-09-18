# exp-v1 implementation gate

Goal: create an unsubmitted low-load transfer experiment with P0/D0 Canary,
the isolated P1/D1 Production group, and a complete auditable comparison of
Canary-learned P/D frequencies against Production before and after migration.

Red lines: do not modify r17 or earlier Jobs/results; do not add dependencies,
alter external services beyond the explicitly authorized broker submission, or
claim calibrated capacity/causal energy savings. ρ values must remain null
without an explicit capacity calibration.

Acceptance: exactly P0--P3 and D0--D3 on GPU IDs 0--3; Slurm and persistent
samplers request four GPUs/node; P1/D1 begins high; the seven workload classes
run strictly one at a time; each class emits 12 high-frequency Production
baseline attempts, one copied-request Canary binary-plus-greedy P-then-D Table
search, and 48 migrated Production attempts
at five-second spacing; missing Table data, metric gaps, request failures and
SLO/clock deviations are recorded without discarding later classes; CPU tests
and static job validation pass.

Structural decision: Local Fix. Execution decision: authorized feature stage.
Baseline: r17's 43 CPU tests and static validator pass. Expected changed scope:
the new job directory only. Stop on mismatched endpoint-to-meter attribution,
mixed clocks within an active burst, any fabricated rho/metric value, or a
needed change outside this new job. Rollback: remove only this unsubmitted
exp-v1 directory. Real 8-GPU startup, metrics availability, clock actuation and
human acceptance remain separate gates.
