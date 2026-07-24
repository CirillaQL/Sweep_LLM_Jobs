# Saturation-gated online trace with modeled KV-cache transfer time

This job replays the same 12-window finite-rate Poisson trace as Job 250014
using the `latency_plus_saturation` policy. It keeps the same fixed placement,
TP=1, SLOs, candidate frequency caps, overload fallback, vLLM settings, and GPU
telemetry, while adding a measured-bandwidth KV-cache transfer term to the
predicted TTFT used for candidate admission and ranking.

The model is:

```text
predicted TTFT
  = predicted P99(queue + prefill)
  + input_tokens * KV_bytes_per_token * 8 / effective_bandwidth
  + dispatch
```

For Mistral-7B-v0.1:

```text
KV_bytes_per_token = 2 * 32 layers * 8 KV heads * 128 head_dim * 2 bytes
                   = 131072 bytes
```

`effective_bandwidth=24.076 Gbit/s` comes from Minerva Slurm Job 249400, the
largest-message measurement on the same Neptune-Ganymede NCCL Socket/TCP path.
It is an empirical proxy, not the physical 100 Gbit/s link rate. Dispatch is
set to zero for this experiment.

The job requests both nodes exclusively so no other Slurm job can share their
CPU, network, or GPU resources during the trace.
