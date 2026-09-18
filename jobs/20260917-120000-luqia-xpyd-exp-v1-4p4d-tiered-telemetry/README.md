# exp-v1: low-load Canary-to-Production transfer experiment

New, unsubmitted experiment derived from r17. It starts with an empty
frequency Table and imports no historical frequency or routing decisions.

## Hardware and NodeGroups

- Neptune: four L40S Prefill endpoints, P0--P3.
- Ganymede: four L4 Decode endpoints, D0--D3.
- P0/D0 is the only Canary pair and performs the existing P17/D15 SLO-boundary
  and bounded-energy search.
- At startup every production endpoint is high: P1--P3 are 2520 MHz and
  D1--D3 are 1500 MHz. This makes the initial state deterministic and avoids
  silently inheriting a previous burst's lower clock.
- The logical high/middle/low defaults are P1/D1, P2/D2 and P3/D3. Their
  endpoint targets are respectively 2520/1500, 1710/975 and 900/450 MHz.

This first experiment deliberately uses the isolated P1/D1 Production group.
P0/D0 is the Canary. Each of the seven request classes is a contiguous,
low-load window: Production first sends 12 high-frequency baseline requests;
Canary copies one request, performs its P-axis binary-plus-greedy energy search
followed by its D-axis search, confirms the joint result, and writes one Table
entry. It does not repeat the experiment for that class once the entry exists.
This treats the independently selected P- and D-energy optima as the joint
optimum under the stated separable-energy assumption. Finally Production sends
48 requests at that entry's learned P/D frequency. Both Canary
and Production wait five seconds between requests and concurrency is one.

If Canary cannot write a usable learned entry, the Production transfer window
continues safely at high frequency and records the reason. A Canary issue,
Production request failure, or observed SLO/clock deviation is evidence rather
than an immediate experiment-wide exit. `workload_transfer_windows.jsonl` and
`low_load_transfer_summary.json` compare Canary's learned measurement with
Production's high-frequency baseline and migrated-frequency window. This is a
low-load transferability observation, not a capacity result or causal proof.

## Per-request telemetry contract

`nodegroup_telemetry.jsonl` contains one row for every completed Canary or
Production request. Required fields are `timestamp`, `source`, `input_tokens`,
`max_output_tokens`, `actual_output_tokens` (and compatibility alias
`output_tokens`), `batch_size`, `arrival_rate`, `P_gpu`, `D_gpu`, `P_rho`,
`D_rho`, `P_queue`, `D_queue`, `D_KV_usage`, `P_freq`, `D_freq`, `TTFT`,
`TPOT`, `P_energy`, `D_energy`, `total_energy`, and `SLO_met`.

`P_freq`/`D_freq` retain both requested MHz and observed NVML min/max MHz.
`P_energy` and `D_energy` are the active client-send to completed-stream energy
for the selected boards; `total_energy` is their sum. `TTFT` and `TPOT` are ms.
`batch_size` is max(P-running,D-running), while queues and Decode KV use
read-only vLLM Prometheus gauges. Arrival rate is the completed driver's rolling
60-second send-window rate.

ρ is computed as current vLLM window token rate divided by a separately
calibrated capacity. No such calibration is supplied here, so P_rho/D_rho are
intentionally `null` until the two capacity fields in `telemetry_contract` are
populated. Unavailable Prometheus values are also `null`, with explicit
`telemetry_errors` and missing-metric fields; the job never invents a value.

## Evidence boundary

The primary energy metric excludes zero-in-flight gaps, actuation and settling;
the separate control-inclusive metric is retained in the original request log.
Concurrent burst energy is equal-share attribution over in-flight segments, not
independent marginal energy. Canary candidates remain strict; Production SLO or
clock noncompliance is recorded observationally. The `READY` marker authorizes
broker submission after this job directory is committed and pushed.
