# Strict saturation-gate ablation: gate on

This is one half of a paired Slurm ablation. It replays the same 12-window
finite-rate Poisson trace as the gate-off job with identical placement,
candidate frequencies, SLOs, power ordering, overload fallback, vLLM settings,
telemetry, and energy integration.

The active scheduler policy is `latency_plus_saturation`. Diagnostic outputs
for both policies are retained, but only the active policy controls placement
and clocks.
