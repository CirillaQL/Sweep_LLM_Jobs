# Ganymede TP=4 L4 single-pool predictive DVFS validation

This job allocates four L4 GPUs on Ganymede and starts one ordinary
Mistral-7B-v0.1 vLLM server with tensor parallel size four. It is the L4
single-pool counterpart to the Uranus L40S job and uses the same five workload
windows and P99 TTFT/TPOT SLOs of 500/200 ms.

For each workload the local L4 model evaluates both TTFT and TPOT over the L4
frequency grid using one shared TP=4 configuration. The latency-only policy
selects the lowest frequency that passes its point predictions and fitted
feasibility guards, then applies that clock to all four Slurm-allocated GPUs
using `sudo nvidia-smi -lgc`.

The output records decisions, per-GPU clock/power telemetry, benchmark TTFT and
TPOT, prediction errors, and the final `-rgc` cleanup. There is no proxy, second
server, cross-node transfer, or PD KV-transfer term.
