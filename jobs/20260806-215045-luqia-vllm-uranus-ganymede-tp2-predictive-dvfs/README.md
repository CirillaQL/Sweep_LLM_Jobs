# Uranus/Ganymede TP2 predictive DVFS validation

This job runs one Mistral Prefill replica with TP=2 on two Uranus L40S GPUs
and one Decode replica with TP=2 on two Ganymede L4 GPUs.

For every request window, the scheduler predicts P99 TTFT, P99 TPOT, power,
and a target frequency for each role. The shared runner applies each target to
both GPUs in the corresponding TP group with `nvidia-smi -lgc`, records active
clocks and GPU power, runs `vllm bench serve`, and restores default graphics
clocks with `nvidia-smi -rgc` during cleanup.

The result checker writes `prediction_error.json`, `prediction_error.csv`, and
`prediction_error.md`, including predicted versus actual TTFT/TPOT, clock lock
verification for all four GPUs, power error, and SLO classification agreement.
