# Scheduler without saturation gate, uncapped L4 frequency search

This reruns the authoritative gate-off Job 250025 with the same 12-window
request trace and runtime settings. The only experimental change is removal of
the `MAX_L4_FREQ_OVERRIDE=780` candidate cap, allowing the scheduler to search
the full modeled L4 frequency list through 2040 MHz.

The active policy remains `latency_only`; saturation-aware decisions are still
emitted diagnostically but do not control GPU clocks. Placement remains fixed
to L40S prefill on Neptune and L4 decode on Ganymede, both nodes are exclusive,
and overload handling remains `min-slo-violation`.

The modeled KV transfer-time term is left disabled to match the original
gate-off experiment. Runtime vLLM PD inference still performs the real
cross-node KV-cache transfer, so measured TTFT and energy include that path.

Because requested L4 targets above its sustainable active clock may not be
maintained by the hardware, active-clock mismatches are recorded and the
window continues, matching the original experiment's mismatch policy.
