# K4a: P-tier comparison under Poisson load through the real P/D path

Core frequency-scheduling experiment for the CanTuning design (v2 §2.2, §4.3).
P0 = uranus GPU0 (L40S), D0 = ganymede GPU0 (L4), vLLM 0.15.1, `P2pNcclConnector`
with `send_type=PUT_ASYNC` (adopted after K1b, job 267625). Same launcher,
patches and driver as K1b; only new driver phases and config.

## Questions
* At a given load, which P clock (L = 1305, H = 1815, or 2520) gives the lowest
  energy per request, and at what cost in SLO violations? → `τ_up`, `τ_down`.
* Per-pair SLO capacity at each tier → `C_L`, `C_H`.
* Idle (park) power of both GPUs at candidate park clocks → `f_park`.

## Design
| Phase | Design |
|---|---|
| `tier_poisson` | rates 2/4/6 req/s; for each rate one Poisson arrival sequence (60 s) with input lengths drawn from 128/512/1024/2048 and 32 output tokens is generated once and **replayed identically** at P 1305, 1815 and 2520 (D fixed at 1050), so every tier sees exactly the same requests; the P-clock order is rotated per rate. Each request logs the in-flight state at arrival (awaiting-first-token count/tokens, decoding count); each window logs P0/D0 energy. |
| `park_idle` | idle P0/D0 power with the servers loaded at P/D = 900/450, 1305/1050, 1815/1050 (15 s each). |

## Scheduling and safety
* `--partition=short --time=00:30:00`, 1 GPU + 12 CPU + 48 GB per node.
* `--dependency=afterany:267649`: starts only after the K3 retry has left ganymede
  (this launcher needs physical GPU 0 there; the GPU guard exits otherwise).
* Caches and `HOME` under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`; the
  script console is mirrored to `results/<id>/job_console.log`.
* Driver stops starting new windows 150 s before the wall limit; rows are appended
  to `pd_requests.csv` as they complete; per-request timeout 60 s.
* Known caveat: the inherited launcher still runs a node-wide
  `pkill -f 'python.*vllm'` at start/stop (see v2 §9); the K3 retry is immune to it.
