# K3 second retry: plain-vLLM decode characterization on one L4 (ganymede)

Earlier attempts 267572 and 267626 locked the clock successfully and were then
SIGKILLed (console: `Killed ... python -c "import vllm"`). Cause: they started on
ganymede at the same moment as a P/D job using the r3 launcher, which runs
`pkill -KILL -f 'python.*vllm'` / `'vllm.entrypoints'` node-wide; all broker jobs
run as the same account (chjing), so the pkill hit these jobs too.

Fix: no process of this job has a command line matching those patterns.
* `serve.py` starts `vllm.entrypoints.openai.api_server` in-process via `runpy`,
  with the model path and server flags passed through `SERVE_MODEL`/`SERVE_ARGS`;
* the driver receives its output directory and model path via
  `GPU_CHAR_OUTPUT`/`GPU_CHAR_MODEL` (those paths contain `vllm_job_work`);
* the separate `python -c "import vllm"` check is gone (serve.py logs the version);
* `gpu_allocation.txt` lists any processes on the node matching the pkill patterns.

Unchanged design: D clock 1500 → 1050 → 750 → 1275 MHz; input 128 × concurrency
1/2/4/8/16/32 and input 1024 × concurrency 1/8/24; 256 output tokens; idle power
(unlocked/1500/1050) and 1050↔1500 switch latency. Short partition, 1 GPU (physical
index from `SLURM_JOB_GPUS`), 8 CPU, 32 GB; clocks always reset on exit; caches and
`HOME` under `/data/users/chjing/vllm_job_work/${SLURM_JOB_ID}`.
