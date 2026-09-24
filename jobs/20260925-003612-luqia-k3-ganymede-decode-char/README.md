# K3: plain-vLLM decode characterization on one L4

Short-partition (≤30 min) single-node job on ganymede, 1 GPU (physical index from
`SLURM_JOB_GPUS`), 8 CPU, 32 GB. Plain vLLM 0.15.1 with the decode-side flags of
the P/D jobs (`--gpu-memory-utilization 0.82`, `expandable_segments`).

| Phase | Design |
|---|---|
| `decode` | D clock 1500 → 1050 → 750 → 1275 MHz; input 128 × concurrency 1/2/4/8/16/32 and input 1024 × concurrency 1/8/24; 256 output tokens; TPOT per request, batch energy, preemption count |
| `idle_and_switch` | idle power unlocked / 1500 / 1050; 1050↔1500 switch latency × 3 |

Answers: safe concurrency `N_max(f_D)`, whether batching makes TPOT sensitive to
the D clock (decides the B_D tier), loaded decode energy plateau, switch latency.
Outputs: `decode.csv`, `gpu_events.csv`, `power_trace.csv`, `summary.json`,
`gpu_allocation.txt`, `vllm_server_tail.log`.
