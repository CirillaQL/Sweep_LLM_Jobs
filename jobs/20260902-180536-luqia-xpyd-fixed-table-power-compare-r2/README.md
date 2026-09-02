# XPYD fixed-table power comparison r2

This Job performs two fresh, sequential 2P2D launches on `uranus` (2 L40S)
and `ganymede` (2 L4). Formal traffic always uses P1-D1 with
`P2pNcclConnector`; P0-D0 remains unused and is excluded from reported power.

The two modes replay the same six serial workload windows, each with 48
requests at 0.04 RPS (one request every 25 seconds):

1. Full-frequency baseline: P=2520 MHz, D=1500 MHz for every workload.
2. Fixed optimal table: the six frequency pairs measured by Job 255414.

`both_heavy` is intentionally excluded. The table is preloaded before the
first formal request, so no exploration or probe traffic is allowed.

Energy is calculated separately for P1, D1, and P1+D1. Each workload window
starts at its first proxy `route_selected` timestamp and ends at its final
`response_completed` timestamp. Frequency switching, the 10-second settle
wait, and startup warmup therefore do not enter the energy or inference
latency windows.

Raw data and all caches stay below
`/data/users/chjing/vllm_job_work/<job_id>/`. Compact results copied back to
the Job include both run audits, per-workload energy CSVs, and the final
`power_comparison.{json,csv,md}` report.

R2 fixes the Job 256655 audit failure by including the two configured service
warmups in the exact vLLM endpoint counter expectations. Warmups remain absent
from the 288 formal request rows and from route-bounded latency/energy windows.

The `NOT_READY` marker is deliberate: this corrected Job has not been submitted.
