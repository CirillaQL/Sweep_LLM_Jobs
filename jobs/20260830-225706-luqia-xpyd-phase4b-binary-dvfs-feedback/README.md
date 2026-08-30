# XPYD Phase 4B fine-grained binary-feedback DVFS

This job validates whether online feedback can find lower SLO-safe frequencies
without an offline model or oracle.

- Topology: Uranus `P0/P1` (L40S) + Ganymede `D0/D1` (L4), persistent 2P2D.
- Routing: frozen deterministic round-robin over the disjoint routes
  `P0 -> D0` and `P1 -> D1`. Both routes must appear in every window and their
  request counts may differ by at most one. There is no adaptive scheduler.
- Concurrency: client maximum four, sixteen requests per formal window, and
  the audit requires an actually observed overlap of at least two requests.
- SLO: per-request TTFT <= 500 ms and TPOT <= 200 ms; window p99 must also pass.
- Frequency grids: 17 hardware-supported L40S points selected from 900--2520
  MHz and 15 hardware-supported L4 points selected from 450--1500 MHz.
- Search: all-HIGH baseline, binary search the P pool with the D pool at HIGH,
  binary search the D pool with selected P, then five joint confirmation
  windows. P0/P1 always share one target and D0/D1 always share one target.
  Each binary decision uses two completed physical concurrent windows.
- Recovery: a failed joint confirmation raises the violating axis one grid
  point and repeats; all four GPUs are restored to safe HIGH at exit.
- Physical proof: every `sudo nvidia-smi -lgc` action has immediate readback,
  every request window audits target-clock match, and independent 0.2-second
  NVML monitors verify both actions and under-load clocks.

The base plan is 105 physical windows and 1,680 logical requests. If every
workload needs all three confirmation corrections, the fail-closed upper plan
is 165 windows and 2,640 requests.

Raw server logs, Prometheus snapshots, per-window artifacts, actuator logs, and
live-clock JSONL remain in `/data/users/chjing/vllm_job_work/<job_id>`. The Git
job receives only compact final summaries, decision/window statistics, clock
audits, configuration, and a SHA-256 raw-artifact manifest under
`results/<job_id>/`.

The job is prepared but intentionally has no `READY` marker yet.
