# K1b: P0(uranus L40S) → D0(ganymede L4) with `send_type=PUT_ASYNC`

K1 (job 267570) showed that on the r3 substrate a burst's prefill finishes as one
batch whose latency grows with the KV bytes of the whole batch (8×1024 tokens:
plain-vLLM prefill 460 ms vs P/D prefill phase 1578 ms), consistent with the
producer sending KV synchronously (`PUT`) over TCP inside the forward pass.
K1b repeats the burst grid with the producer's `send_type=PUT_ASYNC` and adds an
open-loop Poisson phase through the real P/D path.

| Phase | Clocks P/D | Design |
|---|---|---|
| `bursts_reference` | 2520/1500 | identical to K1: input 128/1024 × burst 1/2/4/8 × 5 reps, 16 output tokens |
| `poisson_2520` | 2520/1500 | 2/4/6/8 req/s × 45 s, input 128/512/1024/2048, 32 output tokens; records requests awaiting their first token (count, tokens) and requests decoding at each arrival |

Differences from K1: P0 runs on uranus (neptune was busy); the launcher now reads
the producer send type from `XPYD_P_SEND_TYPE` (default `PUT`, line 1533 of
`source/run_disagg_benchmark.sh`); per-request timeout 60 s (the r3 notes say an
async send can be lost and leave the consumer waiting); smoke gate unchanged.

Scheduling: job name `cantuning-ganymede` + `--dependency=singleton`, shared with
the K3 retry, so the two never run on ganymede at the same time (this launcher
needs physical GPU 0 there; the GPU guard exits before any clock change otherwise).
Caches and `HOME` under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`; the
script's console is mirrored to `results/<id>/job_console.log`.
