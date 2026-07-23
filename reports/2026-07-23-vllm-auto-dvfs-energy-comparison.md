# Scheduled DVFS vs fixed automatic-DVFS baseline

Date: 2026-07-23  
Scheduled run: Slurm Job 250014  
Fixed automatic-DVFS run: Slurm Job 250017 (`COMPLETED`, `ExitCode=0:0`)

## Experimental control

Both runs used the same:

- Neptune L40S prefill and Ganymede L4 decode placement;
- cross-node PD vLLM 0.15.1 service and 100 GbE TCP transport;
- 12-window request trace, byte-for-byte identical input CSV;
- input/output token lengths, configured request rates, window durations,
  prompt-count limits, random seed, Poisson arrival mode, model, TP=1, and
  serving parameters;
- 0.5-second GPU board-power telemetry, event boundaries, and trapezoidal
  energy integration.

Job 250017 bypassed the scheduler. It issued no `nvidia-smi -lgc` or `-rgc`
commands, did not launch the dynamic clock controller, and left both GPUs
under their default hardware/driver DVFS policy for the entire run.

Both runs completed 332 requests with zero failed requests. The SLO was p99
TTFT <= 500 ms and p99 TPOT <= 200 ms.

## Aggregate comparison

| Metric | Scheduled frequencies | Automatic DVFS | Difference |
|---|---:|---:|---:|
| 12-window GPU energy | 51.05 kJ | 61.22 kJ | **10.17 kJ lower (16.61%)** |
| L40S prefill energy | 31.89 kJ | 36.84 kJ | **13.44% lower** |
| L4 decode energy | 19.16 kJ | 24.38 kJ | **21.39% lower** |
| SLO-pass windows | 11/12 | 11/12 | equal |
| Successful requests | 332 | 332 | equal |
| Failed requests | 0 | 0 | equal |

The only SLO failure in both runs was the 1024-input-token prefill peak.
Scheduled frequency control produced p99 TTFT 612.33 ms; automatic DVFS
produced 639.61 ms. Both runs met TPOT SLO in every window.

## Per-window energy

Positive saving means the scheduled-frequency run consumed less energy.

| # | Workload | Scheduled J | Auto-DVFS J | Scheduled saving |
|---:|---|---:|---:|---:|
| 1 | steady_start | 3,078.20 | 3,889.79 | 20.86% |
| 2 | short_ramp | 3,463.21 | 4,105.25 | 15.64% |
| 3 | prefill_shift | 3,201.04 | 4,434.77 | 27.82% |
| 4 | decode_shift | 5,157.15 | 5,596.90 | 7.86% |
| 5 | sudden_burst | 3,446.40 | 4,255.06 | 19.00% |
| 6 | short_recovery | 3,321.68 | 4,134.89 | 19.67% |
| 7 | long_prefill | 3,359.31 | 4,671.23 | 28.08% |
| 8 | mixed_pressure | 5,189.10 | 5,790.37 | 10.38% |
| 9 | quiet_window | 3,058.66 | 4,233.76 | 27.76% |
| 10 | prefill_peak | 3,898.28 | 3,831.05 | -1.75% |
| 11 | decode_peak | 10,671.67 | 12,403.92 | 13.97% |
| 12 | final_recovery | 3,205.63 | 3,870.39 | 17.18% |

Scheduled control consumed less energy in 11 of 12 windows. The largest
relative reductions occurred in long prefill, input-shift, and quiet windows.
At the prefill peak, both policies operated near maximum L40S frequency and
the scheduled run lasted slightly longer, so its integrated energy was 1.75%
higher despite lower average board power. This isolated reversal should not be
interpreted as an automatic-DVFS efficiency advantage without repeated runs.

## Observed automatic-DVFS behavior

Automatic DVFS kept the L40S at 2520 MHz in almost every telemetry sample.
Its per-window average was 2508–2520 MHz, including low-rate and quiet
windows. The L4 varied more, but still averaged approximately 1681–1916 MHz
and reached 2040 MHz in every window.

Consequently, automatic-DVFS combined average board power was approximately
126.37–140.88 W, compared with 95.23–121.42 W under scheduled frequency
control. Automatic DVFS generally reduced TPOT by several milliseconds, but
the extra latency margin did not improve the number of SLO-pass windows.

The evidence therefore supports the intended energy argument: for this fixed
placement and identical trace, workload-aware frequency selection reduced GPU
energy by 16.61% without reducing SLO compliance. Hardware DVFS alone favored
high clocks and performance margin rather than energy minimization.

## Scope and limitations

Energy covers the two allocated GPU boards only; CPU, RAM, NIC, fans, and
cooling are excluded. Request arrivals are finite-rate Poisson because the
installed vLLM does not expose its configurable burstiness option. The trace
and random seed were identical, but system noise can still change benchmark
duration and energy. Multiple repetitions are needed for confidence intervals,
especially for the single prefill-peak reversal.

