# Fixed L40S-prefill/L4-decode automatic-DVFS baseline

This Slurm job replays the exact 12-window request trace from Job 250014.
It fixes Neptune L40S as prefill and Ganymede L4 as decode, bypasses the
scheduler, and issues no manual GPU clock lock or reset commands.

The GPUs remain under their default hardware/driver DVFS policy. Runtime
telemetry records the clocks selected by DVFS, board power, utilization,
temperature, memory, P-state, and network counters every 0.5 seconds.

After the run, per-window board energy is integrated with the same script and
event boundaries used by Job 250014. `energy_comparison.csv/json` compares the
automatic-DVFS baseline against the saturation-aware scheduled run.
