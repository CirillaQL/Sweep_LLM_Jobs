# Fixed TP=4/TP=4 8192-token latency test (retry 1)

This retries Slurm job 251725 after fixing the P2P proxy's missing-`tp_size`
fallback to the experiment's fixed TP size of 4.

- Prefill: one Mistral service on Neptune, 4 L40S GPUs, TP=4.
- Decode: one Mistral service on Ganymede, 4 L4 GPUs, TP=4.
- Workloads: four 8192-token input tests.
- Reports: TTFT, TPOT, ITL, throughput, request success, and topology checks.

The request trace, placement, and metric checker are reused from the original
job. Results are written to `fixed_tp4_tp4_8192_results/` in this directory.
