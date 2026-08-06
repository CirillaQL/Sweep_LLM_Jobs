# TP2 uranus/ganymede predictive DVFS validation

Integrity: **FAIL**; SLO prediction mismatches: **2**; P2P transport clean: **YES**.

| workload | target P/D MHz | actual P/D MHz | predicted/actual P99 TTFT | predicted/actual P99 TPOT | predicted/actual W | SLO match |
|---|---:|---:|---:|---:|---:|---|
| short_low | 210/360 | 210.0/360.0 | 123.5/194.9 ms | 104.2/36.8 ms | 199.0/181.9 | YES |
| prefill_128 | 210/570 | 210.0/570.0 | 97.1/339.3 ms | 69.3/35.9 ms | 256.1/203.1 | YES |
| mixed_512 | 210/1830 | 210.0/1761.2 | 245.3/802.4 ms | 90.4/34.7 ms | 328.6/235.2 | NO |
| prefill_1024 | 1755/1830 | 1755.0/1780.0 | 235.8/290.2 ms | 122.1/32.7 ms | 379.0/238.1 | NO |
| decode_512 | 210/570 | 210.0/570.0 | 170.1/235.7 ms | 44.0/45.5 ms | 313.2/207.1 | YES |

TTFT MAE: 198.16 ms; TPOT MAE: 49.48 ms; power MAE: 82.08 W.

P2P tensor rejection lines: 0; affected requests: 0.
