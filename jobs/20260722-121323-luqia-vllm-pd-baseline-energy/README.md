# Default-DVFS vLLM PD energy baseline

This is the control group for scheduler job 249820. It runs the identical
three-workload dataset, request counts, request rates, warmup count, model, PD
roles, network path, and concurrency settings, but does not load or invoke the
scheduler and never calls `nvidia-smi -lgc`.

- Neptune / L40S is fixed as prefill.
- Ganymede / L4 is fixed as decode.
- Both GPUs execute `-rgc` before the servers start to establish default DVFS.
- Both GPUs execute an initial and final `-rgc` in a fresh parent-owned reset
  step after the servers stop. The second `-rgc` is the final GPU control
  operation on each node.
- Slurm has a 20-minute hard limit and signals the parent 90 seconds early.

GPU board power is sampled every 0.5 seconds. Parent-clock event timestamps
bracket each complete `vllm bench` command, including CLI startup, one warmup,
and the main request run. `energy_summary.py` uses piecewise-linear trapezoidal
integration to report energy by node, workload, and request. The result covers
the two allocated GPU boards; it does not include CPU, RAM, NIC, or fan power.

After the baseline completes, `compare_results.py` compares it with scheduler
job 249820 and writes:

- `baseline_results/energy_summary.json`
- `baseline_results/energy_by_workload.csv`
- `baseline_results/comparison.json`
- `baseline_results/comparison.csv`

The three workloads are copied unchanged from the completed scheduler r2 job:

| ID | Input | Output | Rate | Requests |
|---|---:|---:|---:|---:|
| `short_r1` | 32 | 32 | 1 req/s | 12 |
| `prefill_r1` | 512 | 128 | 1 req/s | 12 |
| `burst_r10` | 128 | 32 | 10 req/s | 100 |
