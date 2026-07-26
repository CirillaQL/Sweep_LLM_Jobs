# Scheduler with saturation gate, uncapped L4 frequency search

This is the strict paired gate-on run for uncapped gate-off Job 250155. It
replays the same byte-identical 12-window request trace and keeps allocation,
placement, models, SLOs, overload handling, benchmark settings, telemetry,
energy integration, and clock-mismatch behavior unchanged.

The only active-policy difference from Job 250155 is
`latency_plus_saturation` instead of `latency_only`. The L4 candidate cap is
unset in both jobs, allowing the scheduler to search the full modeled L4
frequency list through 2040 MHz.

The modeled KV transfer-time term remains disabled to match the original
gate-on/gate-off experiments. Runtime vLLM PD inference still performs the real
cross-node KV-cache transfer, so measured TTFT and energy include that path.
