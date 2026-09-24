# K2 (+ K1 control): plain-vLLM prefill characterization on one L40S

Short-partition (≤30 min) single-node job on neptune, 1 GPU (whichever Slurm
allocates; its physical index is read from `SLURM_JOB_GPUS`), 8 CPU, 32 GB.
Plain vLLM 0.15.1 (no KV connector), same flags as the P/D jobs
(`--max-model-len 4096 --no-enable-prefix-caching --no-enable-chunked-prefill`).
Requests use exact-length token-id prompts and `max_tokens=1`, so TTFT ≈ prefill.

| Phase | Design |
|---|---|
| `bursts_k1` | 2520 MHz; input 128/1024/2048 × burst 1/2/4/8 × 5 reps (control for K1) |
| `idle_and_switch` | idle power unlocked / 2520 / 1305; 1305↔2520 switch latency × 3 |
| `bursts_freq` | 2115/1815/1305 MHz; same grid × 3 reps |
| `poisson` | 2520 and 1305 MHz; 4/8/12 req/s × 30 s; input 128/512/1024/2048; records in-flight requests/tokens at each arrival |

Clocks are locked with `sudo -n nvidia-smi -lgc/-lmc` and always reset on exit.
Caches and `HOME` live under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`.
Outputs: `bursts.csv`, `poisson.csv`, `gpu_events.csv`, `power_trace.csv`,
`summary.json`, `gpu_allocation.txt`, `vllm_server_tail.log`.
