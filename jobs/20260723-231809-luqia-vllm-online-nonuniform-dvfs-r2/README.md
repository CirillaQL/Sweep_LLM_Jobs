# Online non-uniform request trace with dynamic DVFS, revision 2

This reruns the same live Slurm request trace after the first run exposed an
L40S active-clock ceiling during the 2520 MHz prefill peak.

Scheduling remains online: each control window invokes the saturation-aware
scheduler only when that window arrives, then applies the chosen clocks before
issuing finite-rate Poisson requests through the live PD vLLM service.

An active-clock mismatch is now recorded as
`clock_target_not_sustained` and the request window continues. Command failures,
acknowledgement timeouts, server failures, and benchmark failures remain fatal.
This preserves the real hardware limitation in telemetry without allowing the
monitor itself to truncate the remaining workload.

The run uses only its two Slurm-assigned GPUs. Cleanup stops the servers and
performs the existing fresh-step double GPU clock reset. It does not change
power limits, power gating, MIG, persistence mode, or Slurm node/job state.
