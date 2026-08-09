# NixlConnector 1P1D TP=1 connectivity test, retry 1

This job reruns the minimal cross-node NIXL KV-transfer test after NIXL was
installed in `/data/users/chjing/miniforge3/envs/cuda-env`:

- Uranus: one L40S `P0(TP=1)`;
- Ganymede: one L4 `D0(TP=1)`;
- KV connector: `NixlConnector`;
- only valid route: `P0-D0`.

The default model is `mistralai/Mistral-7B-v0.1`. The benchmark sends eight
requests with 512 input tokens and 64 output tokens. Success requires both
instances to register, the registry to report `NixlConnector`, all requests to
succeed, and the proxy log to contain only the `P0-D0` route.

All Hugging Face, vLLM, Torch, Triton, CUDA, NIXL/UCX runtime caches and
temporary files are placed under `/data/users/chjing/vllm_job_work/<job_id>`
and removed when the job exits.

The `READY` marker causes the broker worker to submit this job after the commit
is pushed.
