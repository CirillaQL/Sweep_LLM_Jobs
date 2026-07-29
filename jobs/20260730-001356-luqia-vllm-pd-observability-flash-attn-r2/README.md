# vLLM PD observability with fixed FlashAttention

This 30-minute Slurm Job is the second repair run after job `250304`.

- Prefill: one Neptune L40S
- Decode: one Ganymede L4
- Model: `mistralai/Mistral-7B-v0.1`
- KV transport: `P2pNcclConnector`
- Attention backend: explicitly fixed to `FLASH_ATTN` on both nodes
- GPU frequency policy: NVIDIA driver/hardware automatic DVFS

The previous run passed OpenTelemetry preflight but vLLM automatic attention
backend discovery imported FlashInfer. FlashInfer then attempted to create
`/root/.cache` and failed with `PermissionError`. This run bypasses FlashInfer
discovery with:

```text
--attention-config.backend FLASH_ATTN
```

Preflight imports the exact vLLM FlashAttention backend before starting either
server. Job and environment CSVs record `attention_backend=FLASH_ATTN`.

The server-readiness loop also watches the two-node `srun` step. If either vLLM
process exits before readiness, the Job immediately prints the server log tails
and fails instead of waiting for the full ten-minute readiness timeout.

All earlier collection remains enabled:

- KV-cache and prefix-cache metrics
- scheduler iteration details
- detailed OTLP traces
- request IDs and W3C trace context
- request/token lifecycle CSVs
- queue state, GPU/network telemetry, and drain measurements

No project scheduler prediction, manual frequency target, or GPU clock-setting
operation is present.
