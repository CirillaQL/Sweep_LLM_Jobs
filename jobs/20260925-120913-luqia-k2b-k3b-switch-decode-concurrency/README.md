# K2b + K3b: clock-switch latency breakdown and D-side decode at production concurrency (one L4, ganymede)

Why: CanTuning design v2 rests on two assumptions that K2/K3 could not settle.

1. **F11, switch latency ≈ 0.5 s.** K2/K3 timed `sudo -n nvidia-smi -lgc` end-to-end
   (L4: 496–503 ms in both directions, suspiciously constant). Published systems report
   <3 ms (NVML `nvmlDeviceSetGpuLockedClocks`, VoltanaLLM) to tens of ms. If the hardware
   change is fast, the 0.5 s is process overhead and a per-group fast "go to H" rule
   becomes possible.
2. **D single tier at 1050.** K3 covered only ≤32 concurrent requests. One group at
   4 req/s with 256 output tokens keeps ≈ 4 × 256 × 0.07 ≈ 72 sequences decoding.

Phases (`config.json`, order = priority):

* `switch_probe`: records `sudo -n -l` (which nvidia-smi arguments are allowed; sudo is
  never run with arguments the rules do not show), times unprivileged
  `nvidia-smi --query-gpu`, tries direct NVML locking as this user (expected
  NoPermission, recorded). Then 6 × (1050→1500, 1500→1050) switches while a 2 ms NVML
  poller records when the SM clock actually arrives: `cmd_ms` (process wall time),
  `reach_ms` (start → clock reached), `reach_minus_exit_ms`. Done idle and again under a
  16-request decode load; loaded requests record `max_gap_ms` (decode stalls).
* `closed_loop`: for D = 1050 then 1500, keep N ∈ {32, 48, 64, 96, 128} requests in
  flight (input 512, output 256, ignore_eos), 15 s warm-up + 30 s window per level.
  Per window: energy, tokens, J/token, tokens/s, KV usage, running/waiting, preemptions
  (`closed_loop_windows.csv`); per request TTFT/TPOT/max gap (`closed_loop.csv`).
  Same server flags as the P/D decode instance (`--gpu-memory-utilization 0.82`,
  max-model-len 4096, no prefix cache, no chunked prefill); KV holds ≈ 33k tokens, so
  levels ≥ 64 are expected to queue — that is the D capacity limit being measured.
* `idle_and_switch`: idle power locked at 750 and 450 MHz (candidates for D park).

Short partition, 1 GPU (physical index from `SLURM_JOB_GPUS`), 8 CPU, 32 GB;
`--dependency=afterany:267652` (K4a uses ganymede GPU 0); clocks reset on exit; caches and
`HOME` under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`; results in
`results/${SLURM_JOB_ID}/`. Expected runtime ≈ 15 min after start.
