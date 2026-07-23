# No-saturation-gate boundary calibration run, r1

This revision reruns the four calibration-derived workloads from
`20260723-161516-luqia-vllm-nogate-boundary-calibration`.

Job 250004 completed preflight, loaded both vLLM servers, registered the PD
pair, and passed its smoke request. It stopped before the first benchmark
because the L4 accepted a requested 2040 MHz lock but sustained only
1155–1245 MHz under the active CUDA acknowledgement probe. The parent then
stopped both servers and successfully completed the fresh two-node double
`-rgc` reset.

That behavior is relevant to the no-gate experiment: the scheduler may select
a nominal high-frequency configuration that the physical L4 cannot sustain
under load. This revision preserves the selected 2040 MHz target and records
the observed active clock, but widens only the acknowledgement tolerance to
900 MHz so the workload can proceed. It does not change the scheduler,
frequency recommendation, vLLM configuration, or saturation policy.

The job still uses one Slurm-assigned GPU per node and the same reset-safe
cleanup. It performs no Slurm administration, power-gating, MIG, persistence,
or power-limit operation.
