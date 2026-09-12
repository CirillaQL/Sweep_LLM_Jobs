# r17: Three-pair fixed-tier routing flow validation
Submission authorized; READY marker added for broker scheduling.

## Hardware and evidence
Neptune has4L40S GPUs; Ganymede has8L4 GPUs, confirmed by user inventory and
Nowledge memory crystal_4b831032a220 (Minerva DVFS/gsync operational boundaries).
The historical4P4D source uses Neptune GPU2/P2, HTTP8102 and KV14581.
Its D2 uses Io, NOT Ganymede; D2 is mapped to Ganymede GPU2 in this new Job,
using the supplied Ganymede8GPU inventory. This is not a claim that an old
Ganymede P2D2 job was already executed.
- P0D0 Canary: Neptune GPU0 / Ganymede GPU0.
- P1D1 high tier: Neptune GPU1 / Ganymede GPU1, fixedP2520/D1500.
- P2D2 medium tier: Neptune GPU2 / Ganymede GPU2, HTTP8102/8202, KV14581.
Request3GPUs/node; start6TP1 services; sampleGPU0/1/2 on each node.
All three pairs initially use maximum clocks. Only Canary searches.

## Goal / non-goal
Verify startup -> Canary exploration -> robust common medium selection ->
P2D2 clock application -> class-aware routing -> completed response/energy log.
This is an experiment, not a formal persistent service, scheduler-capacity
benchmark, proof of global energy optimality or production-load SLO guarantee.

## Frequency selection without historical priors
Start an empty workload Table. Preserve P17/D15 binary SLO-boundary search plus
bounded energy refinement, three low/center/high shape samples per candidate.
SLO=TTFT P95<500ms, TPOT P95<=200ms.

For each successfully learned class and axis:
1. Keep this-run candidates with10%latency headroom (TTFT<450,TPOT<=180).
2. Regard energy within1%of the best admissible candidate as near-optimal.
3. Choose the lowest tested hardware clock in that near-optimal set, rather
   than the raw minimum or arithmetic average of900/1110/1200MHz.
4. Admit representatives only if both clocks are <=80%of maximum clocks.
   Medium means a power-saving tier, not the arithmetic midpoint of the grid.

Take the componentwise maximum of admitted class representatives to form ONE
shared P/D configuration for P2D2. This is a conservative common proposal,
not P-min+D-min=proven global-minimum energy.
Jointly recheck this common pair on P0D0 for each proposed class:3rounds times
3shape requests=9requests/class. Require all9latencies to have headroom and
mean energy <=1.01times this run's learned final-class energy.
Clock-quality failures or failed confirmation exclude that class, not the job.
The1%tolerance is a heuristic noise band, not a formal statistical confidence
interval. Confirmation reduces single-run sensitivity without proving optimality.

## Routing / clock ownership
P1D1 remains maximum throughout. P2D2 starts maximum, then changes once to the
common medium proposal while it has no in-flight traffic.
Publish tier_routing_table only after joint confirmations finish.
Confirmed eligible types route to P2D2; unknown, failed, fallback or ineligible
types route to P1D1. Until publication all online traffic routes to P1D1.
No per-request/per-class Production DVFS. Each pool keeps its fixed target.
If no class qualifies, retain high routing and record medium_route_exercised=false;
do not invent eligibility to make the flow audit pass.

## Traffic / energy / outcomes
Keep seven variable-shape classes and homogeneous burst epochs fromr16:
1–3batches of1/2/3/5concurrent requests; exponential1smean batch arrival gaps;
drain before the next category. Exact client-send/completed-stream timing,
NVML nominal50ms cumulative sampling, no zero-inflight energy in primary metric,
equal-share attribution and conserved burst totals.
Allocation uses the ACTUALLY selected pair, including P2D2 (not hardcodedP1D1).
Record requested clocks, observed clock quality, latency, SLO and selected route.
SLO/Production clock violations only record. Canary exhausted SLO/clock failures
fall back high. Real command failures, corrupt data or code/service errors still fail.
Finite safety limits remain Slurm16h/driver14h/exploration12h.

Retain12matched pairs/class with identical four-request cohorts. High arm uses
P1D1; routed arm uses P2D2 for eligible classes, otherwise P1D1. This now compares
different GPU boards, so hardware variation is a confounder. This Job validates
flow, not a precise frequency-only causal energy saving.
Each class must complete>=12online requests after routing publication.

## Records
- frequency_table.json: original class Canary evidence/fallbacks.
- tier_routing_table.json: common frequencies, representatives, confirmations,
  eligibility and rejection reasons.
- production_requests.jsonl: routed_pair, routing_reason, applied target,
  allocated energy, latency and compliance.
- production_bursts.jsonl: selected pair and conserved energy.
- canary_attempts/probes/events: includes extra common-tier joint-confirmation cost.
- audit.json: both-route coverage and fixed-target evidence are separate from
  execution completion; valid does not imply all SLOs/clocks passed.

## Offline verification / next gate
43CPUtests pass, including full mocked three-pair execution exercising both routes,
constant high/medium targets, subsample energy conservation, near-tie selection,
fallback/no-candidate high routing and observational violations.
Actual6GPU startup, KV transfer, clock actuation and routing remain unverified
until explicit submission and execution. Old Jobs/results preserved.
