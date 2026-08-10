# NIXL 3P3D Random predictive-DVFS: concurrency 16 with route warmup, r2

This is the corrected successor to failed Job 252879.

- Measured workload: input 512, output 128, 36 requests, 2 RPS,
  `max_concurrency=16`.
- Prefill and Decode TP sizes are both `[1,1,2]`; all nine asymmetric-capable
  P-D routes use `NixlConnector`.
- Before measurement, every P-D route receives one explicit 512-input,
  128-output request.
- Warmup uses the same output length as measurement so predictive DVFS remains
  inside the calibrated workload shape. Job 252879 incorrectly used output 1,
  for which the model rejected all latency candidates.
- Warmup results and DVFS decisions remain separate from measured request
  statistics.

All caches and transient runtime state use
`/data/users/chjing/vllm_job_work/<job_id>` and are deleted on job exit.
