# r16: Observational Canary + concurrent Production
Submitted through the broker using the READY marker.
Based on the fixed r15 source; starts with an empty Table, no historical priors.

## Nonfatal experimental outcomes
Production never aborts for TTFT/TPOT SLO violation or GPU clock mismatch.
Every completed request records slo_met, requested P/D MHz, clock_compliant,
actual clock match fractions and observed min/max clocks.
A successful clock command with invalid readback is logged and Production
continues. This does NOT mean the requested clock was successfully achieved.
Active-time clock compliance is recorded again at request and burst-union levels.

Canary still searches SLO-feasible energy candidates; candidates with clock
quality failures are invalidated rather than inserted as a learned optimum.
After three failed exploration attempts consisting of SLO and/or clock-quality
failures, publish a highest-clock fallback with null measurements, zero samples,
slo_met=false and fallback_reason. The source distinguishes SLO exhaustion
from measurement exhaustion. Production uses this entry and continues.
Other categories' entries are applied without waiting for the whole Table.

## Actual failures / safety limits
Do not hide code errors, malformed configuration, missing diagnostics, failed
clock commands, broken HTTP/SSE streams, service failures, missing/reset/nonfinite
energy counters or excessive sampling gaps. These remain explicit failures;
valid energy cannot be invented from corrupted data.
Slurm 16h, driver 14h and exploration safety 12h are retained. These finite
safety limits are not SLO/clock-conformance termination gates.
Completion remains all seven learned/fallback entries plus >=12 online applied
requests/category, followed by 12 exact-shape matched pairs/category.

## Preserved experiment
- Neptune P0/P1, Ganymede D0/D1; P0D0 Canary, P1D1 Production.
- Canary P17/D15 binary boundary plus bounded neighboring energy refinement,
  3 samples/candidate; TTFT P95<500ms, TPOT P95<=200ms.
- Seven unchanged classes and low/center/high input/output shape panels.
- Random category per homogeneous epoch; 1–3 batches of 1/2/3/5 concurrent
  requests, exponential one-second mean batch gaps independent of completion.
  Up to 15 in flight; drain before category/clock change.
- Snapshot the current category Table once per epoch, hold clocks for its
  in-flight requests. Newly learned values apply at its next epoch.
- Nominal50ms NVML cumulative energy sampling; exact client-send/completed-stream
  boundaries; primary excludes zero-in-flight gaps, actuation and settling.
- Equal energy shares per in-flight segment; tiny segments may have zero delta.
  Segment sums must conserve complete active-union energy.
- Control-inclusive energy shares the non-active epoch overhead separately.
- CPU/NIC excluded. Per-request energy is ownership allocation, NOT independent
  marginal request energy. Raw overlapping board windows are labeled separately.
- Supplemental high/Table evaluation randomizes identical cohorts of four;
  12 matched pairs/category,168requests, retaining exact co-running shapes.
  High arms can incur switching: not a no-switch always-high baseline.

## Records / interpretation
production_requests.jsonl: times, allocated energy, latency, slo_met,
clock_compliant, requested clocks, raw shared-board windows and Table revision.
production_bursts.jsonl: conserved active/control energy, in-flight segments
and validated_active_union including actual endpoint clock quality.
canary_probe_events.jsonl: failed-candidate causes; canary_attempts.jsonl:
attempt costs; frequency_table.json: learned or explicitly labeled fallback.
summary.json counts Production SLO and clock-noncompliant requests.
Audit valid means the experiment completed and accounting passed, NOT that
all SLOs/clocks complied or all classes saved energy.
A low-load Canary Table is not proof of concurrent-load optimality.

## Verification
40 CPU tests pass, including a full mocked run where Production SLO/clock
compliance is false, yet all online and168evaluation requests complete.
Canary clock exhaustion publishes explicit fallback; real command failure and
code/data errors remain failures. Actual GPU run remains pending submission.
