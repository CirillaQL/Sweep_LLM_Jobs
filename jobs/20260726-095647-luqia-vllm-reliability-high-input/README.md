# High-input reliability validation

This job validates two runner reliability fixes on the six high-input windows
that Job 250141 did not complete.

## Clock control

Per-window frequency changes no longer start a separate PyTorch CUDA process.
The controller acknowledges a successful `nvidia-smi -lgc` operation, while
the existing 0.5-second workload telemetry records the sustained SM clocks.
This avoids the L4 context-allocation OOM seen at window 11 of Job 250141.

The cleanup path is unchanged: it stops both vLLM servers, launches a fresh
two-node reset step, performs `-rgc`, runs the post-reset active probe after
vLLM has released memory, and makes a second successful `-rgc` the final GPU
control operation.

## Burstiness

The runner now captures `vllm bench serve --help` before checking for
`--burstiness`. This avoids the previous `pipefail`/`grep -q` SIGPIPE false
negative. The two `burst` windows must report both
`vllm_burstiness_supported=true` and their configured value `0.25`.

## Validation trace

The trace covers 3584, 4032, and 4096 input tokens at low and higher/bursty
rates. Every window sends at least 32 prompts. It retains KV-aware online
scheduling, exclusive L40S-prefill/L4-decode allocation, power telemetry,
energy integration, live summaries, and the hard 28-minute limit.
