# r13: Random mixed Production + immediate Table feedback + terminal fallback

Submitted through the broker using the READY marker. Based on r12 code only;
the experiment starts with an empty Table and imports no previous frequencies.

## Requested behavior fixes

1. After THREE SLO-failed attempts for a category, publish an explicit
   `safe_high_after_slo_exhausted` Table item at the hardware-grid maximum
   (P2520/D1500 on this platform). This is a fallback policy, NOT an SLO-safe
   measured optimum: `slo_met=false`, sample count zero, measurements null,
   failure reason retained. Successful categories keep measured Table values.
2. Every online Production dispatch immediately reads its category's current
   entry. A missing entry uses high clocks; a learned entry uses its clocks;
   a terminal fallback uses high clocks. Other categories never block it.
3. Every online request independently draws a category with probability 1/7
   and a shape with probability 1/3. Seed 20260912 makes this reproducible.
   There is no balanced seven-request batch and no category window.

Only genuine SLO search failures qualify for fallback. Exhausted clock,
measurement, or transport failures cause an explicit experiment failure rather
than being mislabeled as a safe-high SLO result. Unexpected worker termination
is also detected instead of silently waiting until the long timeout.

## Preserved settings

- P0/P1 on Neptune (L40S), D0/D1 on Ganymede (L4).
  P0/D0 Canary, P1/D1 Production; discovered hardware P17/D15 grids.
- Binary SLO boundary plus bounded neighboring request-energy refinement;
  three low/center/high requests per candidate, TTFT P95<500ms, TPOT P95<=200ms.
- GPU energy primary: client-send to complete SSE stream, excluding idle gaps
  and frequency settling. Secondary: before actuation through completion,
  including switching costs. 50ms cumulative-NVML sampling with interpolation.
- Clock readbacks within max(15MHz,1%) for >=95% of active samples.
- Serial Production; 150ms settle after a changed clock; random 0.75-1.25s
  post-completion gaps. This tests mixed request traffic, not high concurrency.
- Slurm limit 16h; driver total limit 14h. Terminal handling prevents the r12
  failure path from running until the 12h exploration safety limit.

## Shape panels (input/output tokens)

| Category | Three exact shape combinations |
|---|---|
| small_light | 128/48, 160/64, 200/80 |
| prefill_medium | 768/48, 1024/64, 1280/80 |
| prefill_heavy | 1792/48, 2048/64, 2304/80 |
| decode_medium | 96/96, 128/128, 200/160 |
| decode_heavy | 96/224, 128/256, 200/320 |
| balanced_medium | 384/96, 512/128, 640/160 |
| both_heavy | 1792/224, 2048/256, 2304/320 |

## Completion and comparison

The natural online phase continues until all categories have a measured or
fallback policy AND each category has served at least 12 requests with its
published policy. Classes are still sampled independently; there is no class
quota per batch. This guarantees actual Production use of every published item.

Then run a supplemental 84 exact-shape matched pairs (168 requests). Both arms
of every pair are flattened and globally shuffled as individual requests, not
adjacent same-class blocks. One arm forces high clocks; the other uses the
published Table policy. This controlled evaluation is separate from natural
online observations. Fixed pair quotas here support per-class comparison;
they do not constrain the preceding independent online arrivals.

Fallback categories are explicitly marked in the paired report and must not be
reported as learned-configuration savings or as proof of SLO compliance.

## Outputs

`production_requests.jsonl/.csv`: exact shape, phase, applied policy, Table
revision, target clocks, latency, active and control-inclusive energy, client
and proxy timestamps. `canary_attempts/probes/candidates` and
`canary_overhead.json`: every trial and per-class exploration duration/energy.
Canary wall time ends at the final attempt, NOT after online tail requests.
`frequency_table.json`: learned and explicit fallback entries.
`matched_pair_plan.json`, `evaluation_request_plan.json`: pair IDs and globally
randomized dispatch plan. `summary.json/.md`, `audit.json`: policy applications,
fallback classes, paired comparisons, sampler health, and completion gates.

Actual Slurm/GPU validation remains pending explicit submission.
