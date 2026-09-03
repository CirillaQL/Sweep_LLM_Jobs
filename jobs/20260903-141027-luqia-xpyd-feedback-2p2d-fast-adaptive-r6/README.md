# XPYD 2P2D fast adaptive minimum-energy feedback r6

This Job starts from the corrected r5 implementation and keeps its complete
P/D TTFT boundary, minimum measured request-energy objective, serial workload
windows, atomic persistent Table, and dedicated Canary/Production routes.

- Canary P0-D0: GPU 0 on `uranus` (L40S) and GPU 0 on `ganymede` (L4).
- Production P1-D1: GPU 1 on `uranus` and GPU 1 on `ganymede`.
- Both routes use TP=1 and `P2pNcclConnector`.
- A Table miss serves the original immediately on safe-high P1-D1 while one
  deduplicated clone is explored by the single P0-D0 worker.
- A Table hit applies the stored frequencies to P1-D1.

## Time changes justified by Job 255414

Job 255414 ran for 8,986 seconds and stopped before completing `both_heavy`.
Its 336-request Production trace alone spanned about 8,355 seconds. The
controller also inserted 20 seconds before every one of 87 aggregated
candidates (about 1,740 seconds of fixed idle time) and executed 437 Canary
probes.

R6 removes that fixed candidate delay. A clock command is issued only when a
target changes; the existing node-local `nvidia-smi` poll must read the target
frequency back successfully, after which a 0.5-second guard settle is used.
Unchanged targets have no settle delay. Every changed-frequency command and
readback is appended to `feedback_events.jsonl`.

The r5 measurements showed no energy winner at P=1170/1440 or D=1230 MHz.
R6 therefore retains five P candidates `[900,1710,1980,2250,2520]` and four D
candidates `[450,720,975,1500]`, including the safe-high endpoints and the
observed optimum/boundary region. Each candidate takes at least three samples
and at most five. It stops at three only when energy/TTFT/TPOT CV are all at
most 5% and latency has 10% SLO headroom, or when all three samples are already
outside SLO. The selected joint configuration always receives five fresh
confirmation samples before the Table write.

Each pair now performs one discarded full-path warmup instead of two. The 48
formal requests per workload are retained for steady-state evidence, while
their serial-window pacing changes from 25 seconds to 5 seconds (0.20 RPS).
The expected end-to-end runtime is about 55-75 minutes including server
startup, with a two-hour Slurm limit.

## Metrics, SLO, and persistence

Inference TTFT starts at proxy `request_received` after any DVFS settle and
ends at the first real Decode token. TPOT uses the remaining Decode stream.
Both service and exploration thresholds remain TTFT P90 500 ms and TPOT P90
200 ms. Candidate feasibility gates Table writes. Individual Production
violations are retained in `summary.json`, `audit.json`, `requests.csv`, and
the steady-state reports, but are record-only and do not fail the Job.

All raw caches, server logs, frequency readbacks, Production dispatches,
Canary events, latency, energy, Table state, and summaries remain below
`/data/users/chjing/vllm_job_work/<job_id>/feedback_frequency_table_2p2d/`.
Compact artifacts are copied to `results/<job_id>/` for broker pull, including
`RAW_CACHE_LOCATION.txt`.

The `READY` marker is added only after static checks pass and is the explicit
broker submission trigger.
