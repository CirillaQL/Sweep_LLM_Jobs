# XPYD Phase 4B-r2: actual feedback energy comparison

This job runs the stationary Phase 4B policy matrix on Uranus (2x L40S
prefill) and Ganymede (2x L4 decode), using the local Mistral-7B snapshot.

It compares `STATIC`, `FEEDBACK_ROUTING_ONLY`, `FEEDBACK_DVFS_ONLY`, and
`FULL_FEEDBACK` across four request shapes, five independent repeats, and four
requests per measured window. TTFT/TPOT SLOs are 500/200 ms.

Energy savings are calculated from gross energy of all four GPU boards against
the `STATIC` windows in this same job. The old Neptune/IO Phase 4A oracle is
explicitly not used because its nodes and SLO differ.

The primary audit trail is `phase4b_feedback_decision_trace.csv`. Each row joins
the feedback snapshot and route candidates to the selected route, requested
frequencies, actual request route, per-window observed clock residency,
latency, and measured energy. A separate 0.2-second NVML monitor on each node
independently verifies every controller DVFS transition.

The stationary stage is intentionally the first gate. Dynamic workload
transitions should only run after this job produces a valid physical audit.
