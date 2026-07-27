# KV-aware saturation-gated stress trace

This job is the saturation-gated counterpart to no-gate Slurm Job 250161. It
replays the same byte-identical seven-window trace with the online
`latency_plus_saturation` scheduler policy.

All other settings remain paired:

- fixed L40S-prefill/L4-decode placement with TP=1;
- TTFT/TPOT SLOs of 500/200 ms;
- minimum-SLO-violation overload fallback;
- no manual L4 candidate-frequency cap;
- L40S candidate cap of 2520 MHz;
- exclusive Neptune and Ganymede allocation;
- identical benchmark bounds, 500-prompt maximum, telemetry, and energy
  integration.

Predicted TTFT includes the same cross-node KV-cache transfer model as Job
250161: 131072 KV bytes per token, measured effective bandwidth 24.076 Gbit/s,
and zero dispatch time.

The trace contains four workloads previously proven to saturate after
latency-only admission, plus three higher prompt/rate stress points. Diagnostic
latency-only decisions are retained, but only the saturation-gated decisions
control GPU frequencies.
