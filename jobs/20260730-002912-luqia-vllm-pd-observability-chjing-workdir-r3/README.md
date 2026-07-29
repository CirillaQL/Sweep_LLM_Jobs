# vLLM PD observability with chjing-owned runtime workspace

This 30-minute Slurm Job is the third repair run after job `250326`.

- Prefill: one Neptune L40S
- Decode: one Ganymede L4
- Model: `mistralai/Mistral-7B-v0.1`
- KV transport: `P2pNcclConnector`
- Attention backend: explicitly fixed to `FLASH_ATTN`
- GPU frequency policy: NVIDIA driver/hardware automatic DVFS

Job `250326` confirmed that FlashAttention was selected successfully. However,
vLLM 0.15.1's P2P NCCL connector imports MLA metadata, which imports
FlashInfer even when FlashInfer is not the selected attention backend. That
transitive import tried to create `/root/.cache` and failed.

This run places all mutable runtime state under:

```text
/data/users/chjing/vllm_job_work/<slurm-job-id>/<hostname>/
```

It explicitly configures:

- FlashInfer workspace
- XDG cache and configuration
- CUDA, Triton, TorchInductor, Torch, vLLM, and Numba caches
- the process temporary directory

The batch process and both node processes also change their actual current
working directory to their respective paths under this root.

`HOME` is not modified. The shared Hugging Face model cache remains at
`/data/users/chjing/.cache/huggingface`.

Preflight now imports the actual `P2pNcclConnector` module, resolves
FlashInfer's effective workspace, verifies it remains under the node's chjing
work directory, and checks that it is writable. This catches the prior failure
before model weights are loaded.

All request IDs, OTLP tracing, KV-cache metrics, scheduler iteration details,
GPU/network telemetry, workload windows, automatic DVFS settings, and
fail-fast server readiness checks remain unchanged.
