# Maximum stable throughput at safe-high clocks

P1-D1 frequency: 2520/1500 MHz. Arrival process: open loop.

| Workload | Confirmed max stable RPS | Safe lower bound | Unstable upper bound | Confirmed |
|---|---:|---:|---:|---:|
| small_light | 1.850000 | 1.850000 | 1.900000 | True |
| prefill_medium | 1.700000 | 1.700000 | 1.750000 | True |
| prefill_heavy | 0.262500 | 0.262500 | 0.275000 | True |
| decode_medium | 0.350000 | 0.350000 | 0.362500 | True |
| decode_heavy | 0.168750 | 0.168750 | 0.175000 | True |
| balanced_medium | 0.350000 | 0.350000 | 0.362500 | True |
| both_heavy | 0.065625 | 0.065625 | 0.068750 | True |
