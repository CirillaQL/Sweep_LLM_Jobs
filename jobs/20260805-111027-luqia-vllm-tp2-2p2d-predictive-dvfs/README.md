# TP2 2P/2D predictive DVFS validation

This job runs four Mistral vLLM services:

- Neptune: two Prefill instances, each using two L40S GPUs with TP=2.
- Ganymede: two Decode instances, each using two L4 GPUs with TP=2.

For every workload, the portable latency model first selects the lowest-power
L40S/L4 frequency pair predicted to satisfy the 500 ms P99 TTFT and 200 ms P99
TPOT SLOs. Each instance then applies that frequency to both of its GPUs with
`sudo nvidia-smi -lgc`. The job records all eight GPUs' board power, per-GPU
clock/utilization telemetry, TTFT, TPOT, ITL, and request throughput.

After the sweep, `check_prediction_error.py` creates:

- `prediction_error.md`: predicted-versus-actual latency, power, and clocks.
- `prediction_error.csv`: one row per workload.
- `prediction_error.json`: complete checks and aggregate error statistics.

SLO prediction mismatches are experimental results and do not fail the job.
Missing metrics, failed requests, an incorrect 2P/2D TP=2 registry, or clocks
that do not match the requested lock frequency do fail the integrity check.

This directory intentionally has no `READY` marker; add one only when the
experiment is ready to submit through the broker.
