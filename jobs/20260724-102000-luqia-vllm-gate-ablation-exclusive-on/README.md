# Strict saturation-gate ablation: gate on, exclusive nodes

This is the authoritative gate-on half of the paired Slurm ablation. It replays
the same 12-window finite-rate Poisson trace as the exclusive gate-off job with
identical placement, candidate frequencies, SLOs, power ordering, overload
fallback, vLLM settings, telemetry, and energy integration.

The active scheduler policy is `latency_plus_saturation`. Diagnostic outputs
for both policies are retained, but only the active policy controls placement
and clocks. `--exclusive` prevents the two halves from sharing either node, so
Slurm serializes them.
