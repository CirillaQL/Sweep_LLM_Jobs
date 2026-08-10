# NIXL 3P3D Random predictive-DVFS: concurrency 16 with route warmup

This job extends Job 252772 while keeping its model, topology, NIXL connector,
random asymmetric-TP routing, predictive DVFS policy, and exact forwarding-TTFT
timer.

Changes in this run:

- `max_concurrency=16` for the measured 36-request benchmark.
- Before measurement, one exact 512-token request is sent sequentially through
  every supported P-D route (`P0-D0` through `P2-D2`).
- All nine warmup results are written to `route_warmup.json` and must succeed.
- Warmup DVFS decisions are saved separately in
  `route_warmup_dvfs_decisions.jsonl`.
- Only post-warmup random requests are included in `random_route_check.json`,
  `request_dvfs_check.json`, and the exact forwarding-TTFT summaries.

All caches and transient runtime state remain under
`/data/users/chjing/vllm_job_work/<job_id>` and are deleted on job exit.
