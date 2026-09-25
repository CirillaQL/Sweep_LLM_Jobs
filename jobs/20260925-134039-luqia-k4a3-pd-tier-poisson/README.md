# K4a second retry: P-tier comparison under Poisson load through the real P/D path

**Second retry of K4a.** 267652: the prefill vLLM on uranus (a remote `srun
--overlap` step) wrote no vLLM output for 600 s, and the launcher still ran the
smoke request. 267662: ganymede GPU 0 was in use, Slurm allocated GPU 1, and the
GPU-0-only guard exited (nothing ran, no clock touched).

Changes against K4a:

* **Any allocated GPU.** `run.sbatch` reads the index and UUID Slurm allocated on
  each node from a job step. It exports them as `L40S_GPU_IDS` / `L4_GPU_IDS` and
  `XPYD_EXPECTED_GPU_UUID_<node>`, and writes a runtime copy of the config with
  those `gpu_ids`, which the driver uses for clock actuation and NVML energy.
* **UUID checks.** The launcher checks that the target GPU has the expected UUID
  before every clock lock and reset, and aborts with exit code 97 on a mismatch.
  After each vLLM server is healthy, it confirms that a process of this account
  holds more than 1 GiB on that UUID.
* **Stuck starts.** A vLLM log with no vLLM line after 180 s triggers
  `startup_diagnostics.txt`; the stuck step is killed and the start retried
  once. A server that still fails stops the phase, so no smoke request is run.
* **Park clocks.** Park pairs vary only the P clock (900/1305/1815 with D 1050),
  because K3b measured D idle power at 21.8/22.2/22.4 W for 450/750/1050.
* **No job dependency.**

Everything else (driver, config, measurement design) is unchanged.

Core frequency-scheduling experiment for the CanTuning design (v2 §2.2, §4.3).
P0 = uranus (L40S), D0 = ganymede (L4), on the Slurm-allocated GPU, vLLM 0.15.1, `P2pNcclConnector`
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
| `park_idle` | idle P0/D0 power with the servers loaded at P/D = 900/1050, 1305/1050, 1815/1050 (15 s each). |

## Scheduling and safety
* `--partition=short --time=00:30:00`, 1 GPU + 12 CPU + 48 GB per node.
* No dependency; runs on whichever GPU Slurm allocates on each node (see above).
* Caches and `HOME` under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`; the
  script console is mirrored to `results/<id>/job_console.log`.
* Driver stops starting new windows 150 s before the wall limit; rows are appended
  to `pd_requests.csv` as they complete; per-request timeout 60 s.
* Known caveat: the inherited launcher still runs a node-wide
  `pkill -f 'python.*vllm'` at start/stop (see v2 §9); the K3 retry is immune to it.
