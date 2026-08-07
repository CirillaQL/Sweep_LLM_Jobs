# TP4 uranus/ganymede predictive DVFS validation

Integrity: **PASS**; SLO prediction mismatches: **2**; P2P transport clean: **YES**.

| workload | target P/D MHz | actual P/D MHz | predicted/actual P99 TTFT | predicted/actual P99 TPOT | predicted/actual W | SLO match |
|---|---:|---:|---:|---:|---:|---|
| short_low | 210/360 | 210.0/360.0 | 149.1/181.0 ms | 56.7/28.7 ms | 256.2/349.6 | YES |
| prefill_128 | 210/570 | 210.0/570.0 | 86.9/334.7 ms | 40.2/38.7 ms | 365.1/369.1 | YES |
| mixed_512 | 210/570 | 210.0/570.0 | 172.5/669.6 ms | 99.1/31.8 ms | 445.1/376.4 | NO |
| prefill_1024 | 210/570 | 210.0/570.0 | 222.0/857.0 ms | 102.3/34.0 ms | 409.1/372.9 | NO |
| decode_512 | 480/360 | 480.0/360.0 | 137.2/194.2 ms | 33.7/36.6 ms | 548.5/376.1 | YES |

TTFT MAE: 293.76 ms; TPOT MAE: 33.60 ms; power MAE: 74.95 W.

P2P tensor rejection lines: 0; affected requests: 0.
