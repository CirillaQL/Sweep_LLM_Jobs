# Multi-workload live-frequency vLLM PD test

This `r2` revision retains the stable in-place acknowledgement fix from r1.
It replaces the third case whose 2040 MHz L4 recommendation could not be
physically sustained with another measured high-rate case recommending 1200
MHz. Reset verification no longer assumes that default DVFS must reach the
advertised maximum clock: L4 remained near 1.2 GHz under the reset probe even
after `-rgc` returned success. Instead, both nodes execute `-rgc`, record an
informational CUDA probe, then execute a second successful `-rgc` as their
final GPU control operation before exit.

This job tests three reduced workloads taken directly from the input-length,
output-length, and request-rate grid in both Dual_Sweep_LLM Phase 2 L4 and
L40S datasets. It allocates one GPU on Neptune and one GPU on Ganymede.

For every workload, the portable scheduler predicts the safe minimum-power
placement and `rec_freq_mhz`. The test requires all three decisions to retain
Neptune as prefill and Ganymede as decode, so the two vLLM servers are loaded
only once. Between benchmarks, a node-local controller applies the new
frequency with `nvidia-smi -lgc` and runs a two-second CUDA clock probe. The
benchmark starts only after both nodes acknowledge the target and observed
active clock.

The reduced sweep is:

| ID | Input | Output | Rate |
|---|---:|---:|---:|
| `short_r1` | 32 | 32 | 1 req/s |
| `prefill_r1` | 512 | 128 | 1 req/s |
| `burst_r10` | 128 | 32 | 10 req/s |

The Slurm time limit is 28 minutes and Slurm signals the parent 90 seconds
before the limit. On every normal or trapped exit, the parent first stops the
server step, then starts a fresh two-node reset step. Each node executes
`sudo nvidia-smi -i "$GPU_ID" -rgc` and verifies the restored default clock
range under CUDA load. Node-local cleanup also attempts `-rgc` as a fallback,
but the fresh parent-owned reset step is authoritative.

Expected local scheduler decisions for TTFT SLO 500 ms and TPOT SLO 200 ms:

| ID | Neptune prefill | Ganymede decode |
|---|---:|---:|
| `short_r1` | 480 MHz | 360 MHz |
| `prefill_r1` | 480 MHz | 780 MHz |
| `burst_r10` | 735 MHz | 1200 MHz |

Runtime results are written to `scheduler_multi_results/`, including scheduler
JSON, benchmark output, node telemetry, per-change CUDA clock probes, proxy and
server logs, and reset verification.
