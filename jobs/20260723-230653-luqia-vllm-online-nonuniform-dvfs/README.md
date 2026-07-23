# Online non-uniform request trace with dynamic DVFS

This is a live, trace-driven scheduler experiment. It does not precompute or
locally validate the selected operating points.

Inside the Slurm allocation, the first trace window is scheduled to establish
the fixed L40S-prefill/L4-decode server placement. For every later control
window, the running job:

1. reads the current input length, output length, offered request rate, window
   duration, and burstiness from `request_trace.csv`;
2. invokes the saturation-aware scheduler at that moment;
3. chooses a safe minimum-power point, or `OVERLOAD_FALLBACK` with minimum
   modeled violation when no safe point exists;
4. applies the selected L40S and L4 clocks and verifies them under active CUDA
   load;
5. sends the finite-rate request window through the live PD vLLM service;
6. records performance, power, utilization, clocks, temperature, memory,
   P-state, network traffic, and integrated energy.

The trace deliberately changes request rate, input/output token shape, control
window duration, and arrival burstiness. If the installed vLLM exposes
`--burstiness`, the configured gamma arrival variation is used; otherwise the
finite request rate still produces random Poisson arrivals.

The L4 search is constrained to at most 780 MHz because prior active-load
measurements showed that the nominal 2040 MHz target is not sustainable on
Ganymede. This is a runtime execution constraint, not a local preselection of
workload decisions.

The job uses only its Slurm-assigned GPUs. Cleanup stops the server step before
a fresh two-node double `nvidia-smi -rgc` reset. It does not change power
limits, power gating, MIG, persistence mode, or Slurm node/job state.
