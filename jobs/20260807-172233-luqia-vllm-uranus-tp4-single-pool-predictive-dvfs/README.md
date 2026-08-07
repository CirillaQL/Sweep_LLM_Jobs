# Uranus TP=4 single-pool predictive DVFS validation

This job is the non-PD counterpart to the TP=4 Uranus/Ganymede experiment.
It allocates four L40S GPUs on Uranus and starts one ordinary Mistral-7B vLLM
server with tensor parallel size four. The request trace and 500/200 ms P99
TTFT/TPOT SLOs are identical to the PD-separated run.

For every workload, the portable L40S model evaluates TTFT and TPOT at every
supported frequency using the same single-pool TP=4 configuration. The lowest
frequency whose point predictions and fitted phase feasibility guards meet
both SLOs is applied to all four GPUs with `sudo nvidia-smi -lgc`. Per-GPU
clocks, utilization, power, benchmark
metrics, prediction errors, and cleanup clock reset output are recorded under
`tp4_single_pool_results/`.

The selection intentionally adds no KV-transfer term: Prefill and Decode share
one vLLM engine and one KV cache on the same four GPUs.
