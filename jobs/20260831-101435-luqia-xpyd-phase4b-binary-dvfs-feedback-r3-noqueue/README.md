# XPYD Phase 4B fine-grained binary-feedback DVFS r3 (no queue)

This job validates whether online feedback can find lower SLO-safe frequencies
without an offline model or oracle.

- Topology: Uranus `P0/P1` (L40S) + Ganymede `D0/D1` (L4), persistent 2P2D.
- Routing: frozen deterministic round-robin over the disjoint routes
  `P0 -> D0` and `P1 -> D1`. Both routes must appear in every window and their
  request counts may differ by at most one. There is no adaptive scheduler.
- Concurrency: requests are created in closed-loop batches of exactly two.
  Both requests run concurrently on the two disjoint routes; the next pair is
  created only after both finish. The audit requires peak concurrency exactly
  two and client queue delay no greater than 20 ms.
- SLO: TTFT <= 500 ms is measured inside the proxy from request receipt to the
  first real Decode chunk. Client TTFT is retained as a diagnostic. TPOT <=
  200 ms remains based on the streamed output-token interval.
- Frequency grids: 17 hardware-supported L40S points selected from 900--2520
  MHz and 15 hardware-supported L4 points selected from 450--1500 MHz.
- Search: all-HIGH baseline, binary search the P pool with the D pool at HIGH,
  binary search the D pool with selected P, then five joint confirmation
  windows. P0/P1 always share one target and D0/D1 always share one target.
  Each binary decision uses two completed physical concurrent windows.
- Infeasible workload handling: if valid all-HIGH windows still exceed the
  SLO, the job records the maximum-frequency energy and SLO violations as
  `MAX_FREQ_INFEASIBLE`, skips the lower-frequency search, and continues with
  the remaining workloads.
- Recovery: a failed joint confirmation raises the violating axis one grid
  point and repeats; all four GPUs are restored to safe HIGH at exit.
- Physical proof: every `sudo nvidia-smi -lgc` action has immediate readback,
  every request window audits target-clock match, and independent 0.2-second
  NVML monitors verify both actions and under-load clocks.

The base plan is 105 physical windows and 1,680 logical requests. If every
workload needs all three confirmation corrections plus a maximum-frequency
fallback, the upper plan is 185 windows and 2,960 requests.

Raw server logs, Prometheus snapshots, per-window artifacts, actuator logs, and
live-clock JSONL remain in `/data/users/chjing/vllm_job_work/<job_id>`. The Git
job receives only compact final summaries, decision/window statistics, clock
audits, configuration, and a SHA-256 raw-artifact manifest under
`results/<job_id>/`.

This r3 removes load-generator queueing from the latency experiment, records
the queue delay as an explicit hard gate, and uses processing-side TTFT for
frequency decisions. The job includes a `READY` marker for broker submission.
