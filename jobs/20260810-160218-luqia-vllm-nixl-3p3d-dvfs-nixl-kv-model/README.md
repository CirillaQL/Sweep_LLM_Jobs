# NIXL 3P3D predictive-DVFS with measured KV bandwidth

This job is a single-variable successor to Job 252880. It keeps the same
NIXL 3P3D topology, route warmup, workload, concurrency, SLOs, and random seed,
but replaces the old NCCL/TCP KV bandwidth proxy with measured NIXL bandwidth.

- Measured workload: input 512, output 128, 36 requests, 2 RPS,
  `max_concurrency=16`.
- Prefill and Decode TP sizes are both `[1,1,2]`; all nine asymmetric-capable
  P-D routes use `NixlConnector`.
- Before measurement, every P-D route receives one explicit 512-input,
  128-output request.
- `PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS=4.3063`, corresponding to
  538.29 MB/s from NIXL length-profile Job 252766. This is the conservative
  end of its 538.29--576.47 MB/s long-transfer measurements.
- For Mistral-7B and input length 512, the 64 MiB KV payload is therefore
  estimated at 124.671 ms instead of 22.299 ms.
- The random seed remains identical to Job 252880 for an A/B comparison.

All caches and transient runtime state use
`/data/users/chjing/vllm_job_work/<job_id>` and are deleted on job exit.
