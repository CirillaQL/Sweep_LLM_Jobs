# r15: Homogeneous concurrent burst Production, fixed metering rerun
Submitted through the broker using the READY marker.

Job260443 failed with nonpositive energy during initial concurrent accounting.
This fresh r15 Job submits the tested metering fix while preserving Job260443.
Tiny attribution segments allow zero counter deltas and do not independently
require interior clock samples. Complete active-union positive energy, frequency,
bracketing and counter/sample quality gates remain strict. Segment totals must
conserve the validated active-union energy. GPU rerun remains pending.

Based on r13 source, with an empty Table and no imported historical frequencies.

## Arrival model
Each online epoch independently draws one of seven categories (uniform 1/7).
It schedules 1–3 batches, each containing 1, 2, 3 or 5 requests drawn uniformly.
A batch launches its requests concurrently. Between batch arrivals, the scheduler
waits an exponential random interval with mean 1 second, independent of completion.
Later batches can overlap earlier batches. Maximum epoch size/concurrency is 15.
Within-class low/center/high shape panels are unchanged.

Each epoch reads the current category Table immediately before actuation and
holds those clocks fixed until all epoch requests drain. Other categories never
block Table use. A newly published entry takes effect at the next epoch for that
category, not mid-flight. Category changes drain in-flight requests to avoid
changing shared GPU clocks during another category's inference.
This is synthetic homogeneous burst traffic, NOT a measured real arrival trace,
fixed RPS, or unrestricted overlapping mixed-category traffic.

## Preserved experiment settings
- Neptune P0/P1; Ganymede D0/D1. P0D0 Canary; P1D1 Production.
- Cold-start P17/D15 binary SLO boundary plus bounded neighbor energy search.
- Three shape samples per candidate; TTFT P95 <500ms, TPOT P95 <=200ms.
- Three genuine SLO-failed attempts publish explicit high fallback P2520/D1500;
  infrastructure failures fail fast, not disguised as SLO fallback.
- Production reports latency even if concurrency makes a low-load Table fail SLO.
- 12 applied-policy online requests minimum per category; same long timeouts:
  Slurm 16h, driver 14h, exploration safety 12h; per request 900s.
- Persistent cumulative NVML sampling, nominal 50ms; hardware readback validation.
- Changed clocks settle 150ms; actual actuation time is separately recorded.
- Canary search stays serial/low-load: this measures transfer to concurrent
  Production, NOT a claim that its Table is concurrency-optimal.

## Energy accounting under overlap
Client actual-send and completed-stream timestamps are retained per request.
Partition time by all send/complete boundaries; divide P1+D1 GPU energy in each
segment equally among requests currently in flight. Skip zero-in-flight segments.
Canonical energy_j sums to the burst epoch active GPU energy, with no overlap
double counting. Allocation is an estimate of ownership, not an independently
measured marginal energy per request. Allocated mean power has the same limitation.
Unallocated overlapping board windows are clearly labeled shared_board_window_*.
Control-inclusive energy adds equal shares of the epoch's non-active overhead
(actuation, settling, and any within-epoch zero-inflight intervals), so it also
conserves epoch total energy. Request-between-epoch idle is excluded.
CPU and NIC energy remain excluded.

production_bursts.jsonl records active/control epoch energy and in-flight segments.
production_requests.jsonl records allocated energy, exact boundaries, batch ID,
arrival time, observed clocks, latency and Table revision.
Canary probes, attempts, candidates, gross exploration energy and wall time are
recorded as before.

## Comparison
Retain 12 exact-shape pairs/category (168 requests).
Group each category's pairs into concurrent cohorts of four requests. Both arms
use identical co-running shapes and pair IDs. Shuffle whole high/Table cohorts,
not individual requests; one cohort cannot contain different policies or classes.
This controlled supplemental comparison is separate from online burst traffic.
High arms can incur switching costs: it is NOT a no-switch always-high baseline.

## Validation / next gate
Offline tests include full mocked five-learned/two-fallback execution, overlapping
requests, conservation of active/control energy, idle-gap exclusion, immutable
epoch clocks and matched co-running cohorts. Actual GPU concurrency, queueing,
SLO and energy savings remain unverified until explicit submission and execution.
