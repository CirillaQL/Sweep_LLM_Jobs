# Uranus/Ganymede TP4 predictive DVFS validation

This job repeats the TP2 predictive-DVFS experiment with one TP=4 Prefill
replica on four Uranus L40S GPUs and one TP=4 Decode replica on four Ganymede
L4 GPUs. The workload trace and latency-only scheduling policy are held fixed
for a direct TP2-versus-TP4 comparison.

For each request window, the scheduler predicts P99 TTFT, P99 TPOT, power, and
the target frequency for each role. The shared runner applies each frequency
to all eight GPUs with `nvidia-smi -lgc`, records per-GPU clocks and power, and
restores default clocks with `nvidia-smi -rgc` during cleanup.

The checker records predicted versus actual TTFT/TPOT and power, verifies the
TP4/TP4 registry, and validates the clock of every allocated GPU using the
per-window median so an idle TP rank cannot be mistaken for a failed lock.
