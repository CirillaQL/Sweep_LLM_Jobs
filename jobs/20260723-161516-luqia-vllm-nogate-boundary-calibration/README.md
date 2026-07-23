# No-saturation-gate boundary calibration run

This job measures the live two-node vLLM prefill/decode behavior of the
latency-only SWEEP scheduler on calibration-derived boundary workloads.
It intentionally does not load or invoke the saturation classifier.

The runtime reuses the completed, reset-safe scheduler job 249820:

- one L40S GPU on Neptune for prefill;
- one L4 GPU on Ganymede for decode;
- TP=1 and P2pNcclConnector over the verified 100GbE TCP path;
- per-workload latency-only frequency recommendations;
- active clock probes before each benchmark;
- parent-owned server shutdown and a fresh two-node, double-`-rgc` reset.

## Selected workloads

The workload shapes and rates come from:

`/Users/lukeqian/Obsidian/Obsidian/Dual_Sweep_LLM/calibration_data`

| ID | Input | Output | Rate | Why selected |
|---|---:|---:|---:|---|
| `threshold_long_prompt` | 1024 | 128 | 1 | Latency-only admits it; decode saturation probability is about 0.344, just above the 0.30 gate. |
| `long_decode` | 512 | 1024 | 1 | Long generation case; latency-only admits it while both phase saturation probabilities are very high. |
| `moderate_burst` | 128 | 128 | 5 | Calibration disaggregated runs delivered only about 1.71–1.72 req/s at a configured 5 req/s. |
| `extreme_rps` | 2 | 64 | 50 | Calibration L4 TP=1 results delivered about 10.45–10.97 req/s at 50 req/s. |

All requests remain within the existing vLLM `max-model-len=4096`. This first
run therefore isolates queueing/saturation behavior from context-length OOM
behavior.

The benchmark injects ten seconds of nominal traffic. It keeps at least 12
requests and allows up to 500 requests, so the 50 req/s workload is tested
with 500 requests instead of the historical live job's 120-request cap.
Maximum request concurrency remains 16 to preserve the deployed server and
proxy configuration.

## Intended measurements

For every workload the output records:

- configured and achieved request throughput;
- throughput ratio (`achieved_rps / configured_rps`);
- TTFT and TPOT percentiles;
- successful and failed requests;
- L40S and L4 clocks, utilization, power, and network telemetry.

A workload is measured as saturated when its throughput ratio is below 0.95.
No overload rejection or saturation gate is applied in this job.

## Slurm safety

The job only requests resources through normal Slurm directives and launches
steps inside its own allocation. It does not call `sbatch`, `scancel`,
`scontrol update`, node drain/resume, MIG, persistence mode, power limits, or
power-gating controls.

The only privileged GPU operations are `nvidia-smi -lgc` and `-rgc` on the two
GPUs assigned by Slurm. TERM is delivered 180 seconds before the hard limit.
Normal and trapped exits stop vLLM first, then run the established fresh
two-node reset step; each node executes a final second `-rgc`.
