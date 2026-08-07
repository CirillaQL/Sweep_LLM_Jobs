# Ganymede TP=4 L4 single-pool predictive DVFS retry r1

This retry runs the same five-window, non-PD experiment as the failed job
`252182`. It allocates four L4 GPUs on Ganymede and starts one Mistral-7B-v0.1
vLLM server with TP=4.

The retry uses the corrected runtime-cache policy. All FlashInfer, Torch,
Triton, CUDA, vLLM, Numba, XDG, and temporary caches live under
`/data/users/chjing/vllm_job_work/$SLURM_JOB_ID`. The exit trap stops child
processes, restores GPU clocks with `nvidia-smi -rgc`, validates the exact
numeric job-owned path, and deletes that directory.

Prediction, per-window `-lgc`, telemetry, benchmark, and result-checking logic
is shared with the original job so the retry differs only in the cache fix.
