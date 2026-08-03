# Latency-only SLO autoscaling test

This focused test removes the saturation gate from the active scheduling and
autoscaling path. The runner executes only the `latency_only` decision. A
prefill instance is added only when its predicted TTFT safety or point estimate
violates the 500 ms SLO; a decode instance is added only when its predicted
TPOT safety or point estimate violates the 200 ms SLO.

The three-window trace provides a clean causal sequence:

1. establish a safe 1P/1D baseline;
2. submit 2048-input/128-output requests at 2 RPS, which the latency model
   predicts will violate both SLOs at 1P/1D, and scale until the model predicts
   both SLOs are safe;
3. return to a light workload and verify scale-in to 1P/1D.

Every window waits 30 seconds after scaling and registration. Client-side
warmup requests are disabled because that explicit settling period replaces
them. The allocation retains all eight L4 GPUs on Ganymede, all four L40S GPUs
on Neptune, and twelve-stream power monitoring.

Results are written to `latency_only_autoscale_results/`.
