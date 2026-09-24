# K1: P0-D0 burst prefill overhead and decode-busy first-token delay

Short-partition (≤30 min) job on neptune GPU0 (L40S, P0) + ganymede GPU0 (L4, D0).
Reuses the r3 launcher, vLLM 0.15.1 patches, `P2pNcclConnector` (`send_type=PUT`)
and proxy core unchanged; only the experiment driver is new
(`source/paper/scripts/xpyd/pd_burst_overhead.py`).

## Questions
Offline analysis of r15–r17 showed that co-arriving requests are prefilled as one
batch and TTFT grows with the *number* of concurrent prefill requests (≈40–60 ms
each), and that a busy D adds ≈110 ms to the first token.

| Phase | Clocks P/D | Design |
|---|---|---|
| `bursts_reference` | 2520/1500 | input 128/1024 × burst size 1/2/4/8 × 5 reps, 16 output tokens, D idle between bursts |
| `decode_busy_1500`, `decode_busy_1050` | 2520/1500, 2520/1050 | 0/1/4/8/16 background decodes (400 tokens), then 6 single probes (128/1024 input) |
| `bursts_p1305`, `bursts_p1815` | 1305/1500, 1815/1500 | input 1024 × burst 1/4/8 × 3 reps |

The plain-vLLM control (same bursts without KV transfer) runs in the sibling K2 job
on neptune; comparing both isolates the connector's per-request cost.

## Safety and resources
* `--partition=short --time=00:30:00`, 1 GPU + 12 CPU + 48 GB per node (no `--mem=0`).
* GPU guard: the inherited launcher addresses physical GPU 0; the job exits before
  any clock change unless Slurm allocated GPU 0 on both nodes.
* All caches and `HOME` are under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`.
* The driver stops starting new work 150 s before the wall limit; rows are appended
  to `pd_requests.csv` as they complete.

## Outputs (`results/<slurm_id>/`)
`pd_requests.csv` (per-request proxy timestamps relative to client send),
`summary.json` (grouped p50/p95 of TTFT, prefill and D-first-token time),
`proxy_diagnostics.jsonl.gz`, `frequency_actuations.jsonl`, server log tails,
`gpu_allocation.txt`, `preflight.json`.
