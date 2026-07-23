# Saturation gate boundary calibration

This job reuses the four workloads from the completed no-gate boundary run
`20260723-162136-luqia-vllm-nogate-boundary-calibration-r1`.

It evaluates both scheduler search spaces:

- fixed L40S-prefill/L4-decode placement;
- automatic role routing across L40S and L4.

Every workload is expected to return `NO_SAFE_CONFIG` because no candidate has
predicted saturation probability below 0.30 on both roles. Zero admission is a
successful result for this test.

The Slurm wrapper deliberately requests no GPU TRES. If every request is
rejected, the source runner exits before allocation inspection, vLLM startup,
DVFS control, telemetry, or GPU reset. This makes the test an actual
cluster-side gate execution without changing GPU state.
